"""Spin-integrated EOM-CCSDT: singlet and triplet states, right and left.

`SpinEOMCCSDT` acts with the closed-shell CCSDT Jacobian J on M_S = 0
vectors of one spatial irrep and one spin parity (+1: singlets, -1:
triplets), and with its transpose: the left eigenvectors that transition
and spin-orbit matrix elements need. The terms (`_spin_ccsdt_terms`) are the
T1-dressed spin-orbital CCSDT residual of pdaggerq, differentiated along the
direction and integrated over spin; each has one direction factor. Per term,
a binary contraction tree is chosen whose ground-state subtrees are formed
once (an H-bar-like cache); the left action runs the same trees backwards,
so it is the exact transpose of the right one at the same cost. The
contractions run on the symmetry-blocked executor (`_cpu_symmetry_blocks`).

Vectors use the packed spin blocks a, aa, ab, aaa, aab (alpha particles and
holes first; a < b and i < j within equal-spin groups; the b, bb, abb, bbb
blocks follow from the parity), with sqrt(2) on blocks that stand for an
alpha and a beta copy, so the dot product is the determinant-basis one.
Singlet roots equal those of `EOMSinglet`; triplets are the M_S = 0
components.
"""
from itertools import combinations, permutations
from functools import lru_cache
from time import perf_counter

import numpy as np

from . import _runtime
from ._spin_ccsdt_terms import (DIRECTIONS, OCCUPIED_LETTERS, SAME_SPIN, VIRTUAL_LETTERS, jacobian_terms,
                                parity)


# -- contraction trees ---------------------------------------------------------

class _Node:
    __slots__ = ("leaves", "letters", "left", "right", "ground")

    def __init__(self, leaves, letters, left=None, right=None, ground=True):
        self.leaves, self.letters, self.left, self.right, self.ground = leaves, letters, left, right, ground


def _spaces(term):
    spaces = {}
    for f in term.factors:
        spaces.update(zip(f.letters, f.space))
    return spaces


def _size(letters, spaces, no, nv):
    out = 1
    for c in letters:
        out *= no if spaces[c] == "o" else nv
    return out


def _trees(term):
    factors, n = term.factors, len(term.factors)
    out_letters = set(term.letters)

    def survivors(leaves):
        inside = set("".join(factors[i].letters for i in leaves))
        outside = set("".join(factors[i].letters for i in range(n) if i not in leaves)) | out_letters
        order = []
        for i in sorted(leaves):
            for c in factors[i].letters:
                if c in inside & outside and c not in order:
                    order.append(c)
        return "".join(order)

    @lru_cache(maxsize=None)
    def build(leaves):
        if len(leaves) == 1:
            (i,) = leaves
            return [_Node(frozenset(leaves), factors[i].letters, ground=factors[i].tensor not in DIRECTIONS)]
        result, first, rest = [], leaves[0], leaves[1:]
        for mask in range(1 << len(rest)):
            left = (first,) + tuple(x for b, x in enumerate(rest) if mask >> b & 1)
            right = tuple(x for b, x in enumerate(rest) if not mask >> b & 1)
            if right:
                for lt in build(left):
                    for rt in build(right):
                        result.append(_Node(frozenset(leaves), survivors(frozenset(leaves)), lt, rt,
                                            lt.ground and rt.ground))
        return result

    return build(tuple(range(n)))


def _tree_cost(node, spaces, no, nv, cache_limit):
    if node.left is None:
        return 0, True
    if node.ground:
        return 0, _size(node.letters, spaces, no, nv) <= cache_limit
    lc, lok = _tree_cost(node.left, spaces, no, nv, cache_limit)
    rc, rok = _tree_cost(node.right, spaces, no, nv, cache_limit)
    union = "".join(set(node.left.letters) | set(node.right.letters))
    return lc + rc + _size(union, spaces, no, nv), lok and rok


def _best_tree(term, no, nv, cache_limit):
    spaces = _spaces(term)
    best = None
    for node in _trees(term):
        cost, ok = _tree_cost(node, spaces, no, nv, cache_limit)
        if ok and (best is None or cost < best[0]):
            best = (cost, node)
    return best


def _ground_key(node, term):
    """Key of a ground subtree equal up to renaming, and the renaming."""
    factors = [term.factors[i] for i in sorted(node.leaves)]
    order = sorted(range(len(factors)), key=lambda k: (factors[k].tensor, factors[k].block,
                                                       factors[k].space, factors[k].flipped))
    rename = {}
    for k in order:
        for c in factors[k].letters:
            rename.setdefault(c, chr(ord("A") + len(rename)))
    shape = tuple(sorted((factors[k].tensor, factors[k].block, factors[k].space,
                          "".join(rename[c] for c in factors[k].letters)) for k in order))
    return (shape, "".join(sorted(rename[c] for c in node.letters))), rename


# -- execution -----------------------------------------------------------------

def _leaf_groups(factor):
    """Letter groups in which a canonical block is antisymmetric."""
    return [tuple(factor.letters[i] for i in group) for group in SAME_SPIN[factor.block]]


def _symmetry(out, out_groups, a, b):
    """Ordering of one binary product a x b -> out, each operand given as
    (letters, antisymmetric letter groups): the output positions within an
    equal-spin group of `out` in which the product is antisymmetric (letters
    of one antisymmetric group of one operand, absent from the other), the
    summed letters in which both operands are antisymmetric (their product is
    symmetric: the sum over x < y is half the full one), the executor's
    `ordered` constraints and the factor that restores the full sum."""
    held, constraints, factor = [], [], 1.0
    for (letters, groups), (other, _) in ((a, b), (b, a)):
        for group in groups:
            for positions in out_groups:
                inside = [q for q in positions if out[q] in group and out[q] not in other]
                if len(inside) >= 2:
                    held.append(tuple(inside))
                    constraints += [(out[x], out[y], True) for x, y in zip(inside, inside[1:])]
    summed = (set(a[0]) | set(b[0])) - set(out)
    for ga in a[1]:
        for gb in b[1]:
            pair = sorted(set(ga) & set(gb) & summed)
            if len(pair) >= 2:
                constraints += [(x, y, True) for x, y in zip(pair, pair[1:])]
                for k in range(2, len(pair) + 1):
                    factor *= k
    return tuple(sorted(held)), tuple(constraints), factor


class _Executor:
    """Planned right (J r) and left (J^T l) actions on canonical blocks.

    With `ordered`, a product whose output is antisymmetric in letters of one
    operand's antisymmetric group writes only their ordered entries, into an
    accumulator of its own (keyed by those output positions, read back
    through the coset representatives of `_Block.coset_terms`), and a sum over
    letters in which both operands are antisymmetric runs over ordered tuples.
    """

    def __init__(self, terms, ground, no, nv, labels, irrep, parity_, *, cache_limit, threads, ordered=True,
                 cache=None):
        from ._cpu_symmetry_blocks import CpuSymmetryBlocks
        from . import _cpu_quadruples

        self.no, self.nv, self.irrep, self.parity = no, nv, int(irrep), parity_
        self.o, self.v = slice(no), slice(no, no + nv)
        self.ground = ground
        self.ordered = ordered
        labels = np.asarray(labels, dtype=np.int64)
        table = {c: labels[no:] for c in VIRTUAL_LETTERS}
        table.update({c: labels[:no] for c in OCCUPIED_LETTERS})
        nirrep = 1 << int(labels.max()).bit_length() if labels.max() else 1
        self.engine = CpuSymmetryBlocks(table, "", nirrep, fallback=_cpu_quadruples.einsum, threads=threads)
        # Ground-only products (independent of the irrep and the parity): shareable.
        self.cache = {} if cache is None else cache
        self.plans = [self._plan(t, _best_tree(t, no, nv, cache_limit)[1]) for t in terms]

    def _leaf(self, f):
        x = self.ground[f.tensor][f.block]
        if f.tensor in ("f", "g"):
            x = x[tuple(self.o if c == "o" else self.v for c in f.space)]
        return x

    def _shape(self, letters):
        return tuple(self.no if c in OCCUPIED_LETTERS else self.nv for c in letters)

    def _ground_node(self, node, term):
        if node.left is None:
            (i,) = node.leaves
            return self._leaf(term.factors[i]), term.factors[i].letters
        key, rename = _ground_key(node, term)
        inverse = {v: k for k, v in rename.items()}
        letters = "".join(inverse[c] for c in key[1])
        hit = self.cache.get(key)
        if hit is None:
            a, la = self._ground_node(node.left, term)
            b, lb = self._ground_node(node.right, term)
            hit = self.cache[key] = np.empty(self._shape(letters))
            self.engine.einsum(f"{la},{lb}->{letters}", a, b, hit, 1.0, 0.0, irreps=(0, 0))
        return hit, letters

    def _plan(self, term, root):
        chain, node = [], root
        while node.left is not None:
            child = node.left if not node.left.ground else node.right
            chain.append((node, child, node.right if child is node.left else node.left))
            node = child
        (k,) = node.leaves
        direction = term.factors[k]
        out_block = term.output.split(":")[1]
        out_groups = SAME_SPIN[out_block]
        last = len(chain) - 1
        steps = []
        for index, (parent, child, sibling) in enumerate(reversed(chain)):
            array, sl = self._ground_node(sibling, term)
            cl = direction.letters if child.left is None else child.letters
            pl = term.letters if parent is root else parent.letters
            sib_groups = _leaf_groups(term.factors[next(iter(sibling.leaves))]) \
                if (self.ordered and sibling.left is None) else []
            cur_groups = _leaf_groups(direction) if (self.ordered and index == 0) else []
            right = _symmetry(pl, out_groups if (self.ordered and index == last) else (),
                              (cl, cur_groups), (sl, sib_groups))
            grad_groups = ([tuple(term.letters[q] for q in g) for g in out_groups]
                           if (self.ordered and index == last) else [])
            left = _symmetry(cl, SAME_SPIN[direction.block] if (self.ordered and index == 0) else (),
                             (sl, sib_groups), (pl, grad_groups))
            steps.append((array, sl, cl, pl, right, left))
        scale = term.coefficient * (self.parity if direction.flipped else 1)
        return term, direction, steps, scale

    def cached_bytes(self):
        return sum(x.nbytes for x in self.cache.values())

    def _accumulator(self, store, key, shape):
        x = store.get(key)
        if x is None:
            x = store[key] = np.zeros(shape)
        return x

    def right(self, r, out, shapes):
        """Accumulate the unsymmetrized outputs of J r into `out`, keyed by
        (block, ordered positions)."""
        g = self.irrep
        for term, direction, steps, scale in self.plans:
            cur, letters = r[direction.tensor][direction.block], direction.letters
            for i, (sib, sl, cl, pl, (held, constraints, factor), _) in enumerate(steps):
                spec = f"{letters},{sl}->{pl}"
                if i == len(steps) - 1:
                    target = self._accumulator(out, (term.output, held), shapes[term.output])
                    self.engine.einsum(spec, cur, sib, target, scale * factor, 1.0, irreps=(g, 0),
                                       ordered=constraints)
                else:
                    nxt = np.empty(self._shape(pl))
                    self.engine.einsum(spec, cur, sib, nxt, factor, 0.0, irreps=(g, 0), ordered=constraints)
                    cur, letters = nxt, pl
        return out

    def left(self, l, grads, shapes):
        """Accumulate J^T l into `grads` (keyed by (block, ordered positions))
        from antisymmetrized output blocks `l`."""
        g = self.irrep
        for term, direction, steps, scale in self.plans:
            grad, letters = l[term.output], term.letters
            for i, (sib, sl, cl, pl, _, (held, constraints, factor)) in enumerate(reversed(steps)):
                spec = f"{sl},{letters}->{cl}"
                if i == len(steps) - 1:
                    key = (direction.tensor, direction.block)
                    target = self._accumulator(grads, (key, held), shapes[key])
                    self.engine.einsum(spec, sib, grad, target, scale * factor, 1.0, irreps=(0, g),
                                       ordered=constraints)
                else:
                    nxt = np.empty(self._shape(cl))
                    self.engine.einsum(spec, sib, grad, nxt, factor, 0.0, irreps=(0, g), ordered=constraints)
                    grad, letters = nxt, cl
        return grads


# -- ground blocks ---------------------------------------------------------------

def dense_triples(solver, t3):
    """Spatial T3[a,b,c,i,j,k] from `RestrictedCC`'s packed slots."""
    no, nv = solver.layout.no, solver.layout.nv
    dense = np.zeros((nv, nv, nv, no, no, no))
    for occupied, tile in zip(solver.triples_indices, t3):
        for p in permutations(range(3)):
            dense[(slice(None),) * 3 + tuple(occupied[q] for q in p)] = tile.transpose(p)
    return dense


def amplitude_blocks(solver, amplitudes):
    """Canonical spin blocks a, aa, ab, aaa, aab of restricted singlet
    amplitudes (T1[i,a], T2[i,j,a,b], packed T3, as `RestrictedCC` and
    `SingletSpace.unpack` give them): ground amplitudes or singlet EOM vectors."""
    x1, x2, x3 = amplitudes
    ab = np.ascontiguousarray(np.asarray(x2).transpose(2, 3, 0, 1))
    dense = dense_triples(solver, x3)
    aab = dense - dense.transpose(1, 0, 2, 3, 4, 5)
    aaa = np.zeros_like(dense)
    for p in permutations(range(3)):
        aaa += parity(p) * dense.transpose(*p, 3, 4, 5)
    del dense
    return {"1": {"a": np.ascontiguousarray(np.asarray(x1).T)},
            "2": {"aa": ab - ab.transpose(1, 0, 2, 3), "ab": ab},
            "3": {"aaa": aaa, "aab": aab}}


def ground_blocks(solver, amplitudes):
    """Canonical spin blocks of the T1-dressed f and g, and of T2 and T3."""
    f, g = solver._dressed(np.asarray(amplitudes[0]))
    g = np.ascontiguousarray(g)
    blocks = amplitude_blocks(solver, amplitudes)
    return {"f": {"a": np.ascontiguousarray(f)},
            "g": {"aaaa": g - g.transpose(0, 1, 3, 2), "abab": g},
            "t2": blocks["2"], "t3": blocks["3"]}


# -- packed vectors --------------------------------------------------------------

_BLOCKS = (("1:a", "r1", "a", 1, 1), ("2:aa", "r2", "aa", 2, 2), ("2:ab", "r2", "ab", 2, 0),
           ("3:aaa", "r3", "aaa", 3, 3), ("3:aab", "r3", "aab", 3, 2))


@lru_cache(maxsize=None)
def _local_permutations(n, equivalent):
    order = list(permutations(range(n))) if equivalent == n else [tuple(range(n)), (1, 0, 2)]
    return tuple((p, parity(p)) for p in order)


class _Block:
    """One packed spin block: canonical coordinates in row-major order.

    Every map between packed entries and the dense block is a signed sum over
    permuted coordinate sets (the equal-spin permutations; for ab the
    parity's swap is folded into `SpinSpace.project`): gathers run on the
    compiled `PermutedGather`, scatters on the compiled sector scatter of
    disjoint flat positions."""

    def __init__(self, no, nv, labels, irrep, rank, equivalent):
        self.shape = (nv,) * rank + (no,) * rank
        self.rank, self.equivalent = rank, equivalent

        def orbitals(n):
            if not equivalent:
                return np.indices((n, n), dtype=np.int32).reshape(2, -1).T
            canonical = np.asarray(list(combinations(range(n), equivalent)), dtype=np.int32).reshape(-1, equivalent)
            if equivalent == rank:
                return canonical
            return np.column_stack((np.repeat(canonical, n, axis=0),
                                    np.tile(np.arange(n, dtype=np.int32), len(canonical))))
        a, i = orbitals(nv), orbitals(no)
        a_ir = np.bitwise_xor.reduce(labels[no:][a], axis=1)
        i_ir = np.bitwise_xor.reduce(labels[:no][i], axis=1)
        rows, cols = np.nonzero((a_ir[:, None] ^ i_ir[None, :]) == irrep)
        self.coordinates = np.ascontiguousarray(np.column_stack((a[rows], i[cols])).T, dtype=np.int32)
        self.size = len(rows)
        self.scale = np.sqrt(2.) if equivalent else 1.0
        r = rank
        if equivalent:
            local = _local_permutations(r, equivalent)
            self.orders = tuple((tuple(pa) + tuple(r + q for q in pi), sa * si)
                                for pa, sa in local for pi, si in local)
        else:
            self.orders = ((tuple(range(2 * r)), 1),)
        self._gatherer = None
        self._positions = None

    def _gather(self, array, weight):
        """sum over orders of sign * weight * array[permuted coordinates]."""
        from ._tensor_scatter import PermutedGather
        out = np.zeros(self.size)
        if self._gatherer is None:
            self._gatherer = PermutedGather(np.arange(self.size), self.coordinates)
        source = np.ascontiguousarray(array)
        terms = [(axes, sign * weight) for axes, sign in self.orders]
        if not self._gatherer.add(source, out, terms):
            for axes, sign in self.orders:
                out += sign * weight * source[tuple(self.coordinates[a] for a in axes)]
        return out

    def _scatter(self, vector, weight):
        """The dense block with sign * weight * vector at every permuted coordinate set."""
        from . import _cpu_kernels
        from ._runtime import cpu_threads
        if self._positions is None:
            strides = [int(np.prod(self.shape[d + 1:])) for d in range(len(self.shape))]
            self._positions = np.ascontiguousarray(np.concatenate([
                sum(self.coordinates[a].astype(np.int64) * strides[d] for d, a in enumerate(axes))
                for axes, _ in self.orders]))
            self._signs = np.repeat(np.asarray([s for _, s in self.orders], dtype=np.float64), self.size)
        values = np.ascontiguousarray(np.tile(np.asarray(vector, dtype=np.float64) * weight, len(self.orders))
                                      * self._signs)
        kernel = _cpu_kernels.library()
        if kernel is None:
            array = np.zeros(self.shape)
            array.reshape(-1)[self._positions] = values
            return array
        array = np.empty(self.shape)
        kernel.sector_unpack(array.size, values.size, self._positions.ctypes.data, values.ctypes.data,
                             array.ctypes.data, cpu_threads(None))
        return array

    def expand(self, projected):
        """The dense block of packed entries (after `SpinSpace.project`)."""
        return self._scatter(projected, 1.0 / self.scale)

    def coset_terms(self, held, weight):
        """(axes, weight) of the equal-spin permutations whose permuted
        canonical coordinates are ordered within each group of positions
        `held`: one per coset of the permutations of those groups, which an
        array antisymmetric there (and stored only where ordered) needs,
        each weighted by the coset size."""
        size = 1
        for group in held:
            for k in range(2, len(group) + 1):
                size *= k
        return [(axes, sign * weight * size) for axes, sign in self.orders
                if all(axes[x] < axes[y] for group in held for x, y in zip(group, group[1:]))]

    def gather_terms(self, array, terms):
        from ._tensor_scatter import PermutedGather
        out = np.zeros(self.size)
        if self._gatherer is None:
            self._gatherer = PermutedGather(np.arange(self.size), self.coordinates)
        source = np.ascontiguousarray(array)
        if not self._gatherer.add(source, out, terms):
            for axes, weight in terms:
                out += weight * source[tuple(self.coordinates[a] for a in axes)]
        return out

    def expand_adjoint(self, array, held=()):
        return self.gather_terms(array, self.coset_terms(held, 1.0 / self.scale))

    def gather_antisymmetrized(self, array, held=()):
        """The packed entries of the equal-spin antisymmetrization of `array`
        (stored only where ordered within the groups `held`)."""
        return self.gather_terms(array, self.coset_terms(held, self.scale / len(self.orders)))

    def scatter_antisymmetrized(self, vector):
        """The adjoint of `gather_antisymmetrized` (an antisymmetric array)."""
        return self._scatter(vector, self.scale / len(self.orders))


class SpinSpace:
    """Packed M_S = 0 coordinates of one irrep and spin parity."""

    def __init__(self, no, nv, labels, irrep, parity_):
        self.parity, self.irrep = parity_, int(irrep)
        self.blocks = {key: _Block(no, nv, labels, irrep, rank, eq) for key, _, _, rank, eq in _BLOCKS}
        self.offsets = np.cumsum([0] + [b.size for b in self.blocks.values()])
        self.size = int(self.offsets[-1])
        self.slices = tuple(slice(int(a), int(b)) for a, b in zip(self.offsets, self.offsets[1:]))
        a, b, i, j = (c.astype(np.int64) for c in self.blocks["2:ab"].coordinates)
        # Packed position of the (b, a, j, i) partner of each ab entry: the
        # coordinates are row-major over (a, b) pairs, then (i, j) pairs.
        key = ((a * nv + b) * no + i) * no + j
        partner = ((b * nv + a) * no + j) * no + i
        order = np.argsort(key)
        self._ab_partner = order[np.searchsorted(key, partner, sorter=order)]

    def _sections(self):
        for (key, tensor, name, _, _), part in zip(_BLOCKS, self.slices):
            yield key, tensor, name, self.blocks[key], part

    def project(self, vector):
        """Make the ab block consistent with the parity (a symmetric projector)."""
        out = np.array(vector, dtype=np.float64, copy=True)
        part = self.slices[2]
        ab = out[part].copy()
        out[part] = 0.5 * (ab + self.parity * ab[self._ab_partner])
        return out

    def unpack(self, projected):
        r = {"r1": {}, "r2": {}, "r3": {}}
        for _, tensor, name, block, part in self._sections():
            r[tensor][name] = block.expand(projected[part])
        return r

    def unpack_adjoint(self, grads):
        """J^T-side gather of gradient accumulators {((tensor, name), held): array}."""
        out = np.zeros(self.size)
        for _, tensor, name, block, part in self._sections():
            for (key, held), array in grads.items():
                if key == (tensor, name):
                    out[part] += block.expand_adjoint(array, held)
        return self.project(out)

    def pack_antisymmetrized(self, raw):
        """Packed antisymmetrized outputs of accumulators {(block key, held): array}."""
        out = np.zeros(self.size)
        for key, _, _, block, part in self._sections():
            for (name, held), array in raw.items():
                if name == key:
                    out[part] += block.gather_antisymmetrized(array, held)
        return out

    def scatter_antisymmetrized(self, vector):
        return {key: block.scatter_antisymmetrized(vector[part]) for key, _, _, block, part in self._sections()}

    def pack(self, blocks):
        """Packed entries of antisymmetric canonical blocks {'1': {'a'}, '2': {...}, '3': {...}}."""
        return np.concatenate([np.asarray(blocks[key[0]][name])[tuple(block.coordinates)] * block.scale
                               for key, _, name, block, _ in self._sections()])

    def diagonal(self, fock_diagonal, no):
        eo, ev = fock_diagonal[:no], fock_diagonal[no:]
        out = []
        for _, _, _, block, _ in self._sections():
            c, r = block.coordinates, block.rank
            out.append(sum(ev[c[k]] for k in range(r)) - sum(eo[c[r + k]] for k in range(r)))
        return np.concatenate(out)


# -- engine ----------------------------------------------------------------------

class SpinEOMCCSDT:
    """Right and left EOM-CCSDT actions on singlet or triplet M_S = 0 vectors.

    `solver` is a `RestrictedCC` (CCSDT) and `amplitudes` its converged
    (T1, T2, packed T3). With `orbital_irreps` the integrals are projected
    onto the point group (forbidden entries within 2e-9) and the products run
    symmetry-blocked; `irrep` selects the target irrep and `spin` 'singlet'
    or 'triplet'. Ground-state intermediates up to `cache_elements` entries
    each are formed once; `share`, another engine of the same ground state and
    labels, lends its ground blocks and intermediates (other irreps and spins
    need no new ones). CPU only.
    """

    method = "ccsdt"

    def __init__(self, solver, amplitudes, *, orbital_irreps=None, irrep=0, spin="triplet",
                 cache_elements=200_000_000, cpu_threads=None, ordered=True, share=None):
        from .restricted import RestrictedCC

        if spin not in ("singlet", "triplet"):
            raise ValueError("spin must be 'singlet' or 'triplet'")
        self.cpu_threads = _runtime.thread_option(
            cpu_threads if cpu_threads is not None else getattr(solver, "cpu_threads", None))
        self.parity = 1 if spin == "singlet" else -1
        self.spin = spin
        no, nv = solver.layout.no, solver.layout.nv
        self.nocc, self.nvir = no, nv
        n = no + nv
        labels = np.zeros(n, dtype=np.int64) if orbital_irreps is None else np.asarray(orbital_irreps, dtype=np.int64)
        if labels.shape != (n,):
            raise ValueError("Expected one irrep label per orbital")
        self._labels = labels
        if share is not None and (share.nocc != no or share.nvir != nv or not np.array_equal(share._labels, labels)):
            raise ValueError("A shared engine must have the same orbitals and labels")
        h, eri = solver.h, solver.g.transpose(0, 2, 1, 3)
        if np.any(labels):
            from .spatial_symmetry import project_integrals
            h, eri, _ = project_integrals(h, eri, labels, tolerance=2e-9)
        constant = solver.reference_energy - float(np.trace((solver.h + solver.fock)[solver.o, solver.o]))
        with _runtime.thread_scope(self.cpu_threads):
            if share is None:
                primal = RestrictedCC(h, eri, no, constant, cpu_threads=self.cpu_threads)
                amplitudes = primal._validated(tuple(np.asarray(a) for a in amplitudes), triples=True)
                self.ground_energy = primal.reference_energy + primal._energy(*amplitudes[:2])
                self.ground = ground_blocks(primal, amplitudes)
            else:
                self.ground_energy, self.ground = share.ground_energy, share.ground
            self.space = SpinSpace(no, nv, labels, irrep, self.parity)
            self.diagonal = self.space.diagonal(np.diag(self.ground["f"]["a"]), no)
            start = perf_counter()
            self._executor = _Executor(jacobian_terms(), self.ground, no, nv, labels, irrep, self.parity,
                                       cache_limit=int(cache_elements), threads=self.cpu_threads,
                                       ordered=ordered, cache=None if share is None else share._executor.cache)
            self.setup_seconds = perf_counter() - start
        s3 = (nv,) * 3 + (no,) * 3
        self._out_shapes = {"1:a": (nv, no), "2:aa": (nv, nv, no, no), "2:ab": (nv, nv, no, no),
                            "3:aaa": s3, "3:aab": s3}
        self._in_shapes = {("r1", "a"): (nv, no), ("r2", "aa"): (nv, nv, no, no),
                           ("r2", "ab"): (nv, nv, no, no), ("r3", "aaa"): s3, ("r3", "aab"): s3}
        self.matvec_count, self.matvec_seconds = 0, 0.0

    @property
    def cached_bytes(self):
        return self._executor.cached_bytes()

    def _right(self, vector):
        out = self._executor.right(self.space.unpack(self.space.project(vector)), {}, self._out_shapes)
        return self.space.project(self.space.pack_antisymmetrized(out))

    def _left(self, vector):
        grads = self._executor.left(self.space.scatter_antisymmetrized(self.space.project(vector)), {},
                                    self._in_shapes)
        return self.space.unpack_adjoint(grads)

    def _action(self, vector, left):
        vector = np.asarray(vector)
        if np.iscomplexobj(vector):
            value = self._action(vector.real, left)
            return value + 1j * self._action(vector.imag, left) if np.any(vector.imag) else value
        start = perf_counter()
        with _runtime.thread_scope(self.cpu_threads):
            result = self._left(vector) if left else self._right(vector)
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite EOM-CCSDT action")
        self.matvec_count += 1
        self.matvec_seconds += perf_counter() - start
        return result

    def sigma(self, vector):
        """J r in packed coordinates."""
        return self._action(vector, False)

    def sigma_left(self, vector):
        """J^T l in packed coordinates (the exact transpose of `sigma`)."""
        return self._action(vector, True)

    def full_residual_norm(self, vector, energy, *, left=False):
        vector = self.space.project(np.asarray(vector).real)
        residual = self._action(vector, left) - energy * vector
        return float(np.linalg.norm(residual))

    def lambda_source(self):
        """dE/dT in packed singlet A1 coordinates (the ground-state left source)."""
        if self.parity != 1 or self.space.irrep != 0:
            raise ValueError("The ground-state source lives in the singlet totally symmetric space")
        o, v = slice(self.nocc), slice(self.nocc, None)
        f, g = self.ground["f"]["a"], self.ground["g"]
        grads = {key: np.zeros(shape) for key, shape in self._in_shapes.items()}
        grads[("r1", "a")] = f[o, v].T.copy()
        grads[("r2", "aa")] = 0.25 * g["aaaa"][o, o, v, v].transpose(2, 3, 0, 1)
        grads[("r2", "ab")] = g["abab"][o, o, v, v].transpose(2, 3, 0, 1).copy()
        # E = sum_ai f_ia t_ai + 1/4 sum <ij||ab> t_abij in spin orbitals: the
        # alpha and beta copies of each block.
        grads[("r1", "a")] *= 2
        grads[("r2", "aa")] *= 2
        return self.space.unpack_adjoint({(key, ()): array for key, array in grads.items()})

    def solve(self, nroots=1, *, left=False, residual_tol=1e-8, max_iter=100, max_space=20, guesses=None,
              callback=None, checkpoint_callback=None, thick_restart=True):
        """Restarted real non-Hermitian Davidson (`krylov.davidson`) on the
        right (or, with `left`, the transposed) Jacobian."""
        from .eom import EOMResult
        from .krylov import davidson

        action = self.sigma_left if left else self.sigma
        outcome = davidson(action, self.diagonal, self.space.project, size=self.space.size, nroots=nroots, xp=np,
                           residual_tol=residual_tol, max_iter=max_iter, max_space=max_space, guesses=guesses,
                           callback=callback, checkpoint_callback=checkpoint_callback,
                           certify=lambda vector, energy: self.full_residual_norm(vector, energy, left=left),
                           matvecs=lambda: self.matvec_count, retain_previous=thick_restart)
        values = outcome.values
        return EOMResult(outcome.converged, values, self.ground_energy + values, outcome.vectors,
                         outcome.residual_norms, outcome.history, self.method, self.space.irrep,
                         self.matvec_count, self.matvec_seconds)
