"""Occupied-tile execution of the spin-orbital rank-four diagrams."""
from collections import defaultdict
from ._intermediate_cache import IntermediateCache  # compatibility import
from functools import lru_cache
from itertools import product
from math import factorial, prod

import numpy as np

from ._spin_quadruples_diagrams import DIAGRAMS
from ._tensor_symmetry import signed_permutations
from .bounded_einsum import einsum, einsum_into


@lru_cache(maxsize=None)
def contraction_schedule(spec, no, nv, fixed=()):
    """Binary contraction tree of one occupied tile (`fixed`: its unit letters).

    An intermediate that does not read some fixed letters is the same for
    every tile that differs only in them, and the intermediate caches share
    it: its multiply-adds count once per no**(letters it does not read).
    Among the trees whose largest intermediate stays within max(nv**4, the
    smallest possible peak), the tree with the fewest such amortized
    multiply-adds wins, then the smaller peak, then fewer multiply-adds. For
    goovv t2 t4 (mnef,aeij,bcdfklmn) this contracts g with T4 first: an
    intermediate per (k, l) instead of a packed-T4 product per tile.
    """
    inputs, output = spec.split("->")
    inputs = tuple(inputs.split(","))
    count = len(inputs)
    lengths = {c: (1 if c in fixed else nv if c in "abcdef" else no)
               for c in set("".join(inputs))}
    whole = (1 << count) - 1

    @lru_cache(maxsize=None)
    def trees(mask):
        """Every tree of the inputs in `mask`: ((amortized, peak, work), letters, tree)."""
        if mask.bit_count() == 1:
            pos = mask.bit_length() - 1
            return (((0, 0, 0), inputs[pos], pos),)
        inside = set().union(*(set(term) for i, term in enumerate(inputs) if mask >> i & 1))
        outside = set(output).union(*(set(term) for i, term in enumerate(inputs) if not mask >> i & 1))
        letters = "".join(sorted(inside & outside))
        size = prod(lengths[c] for c in letters)
        shared = 1 if mask == whole else no ** sum(c not in inside for c in fixed)
        found = []
        for left in range(1, mask):
            if left & mask != left or not left & (mask & -mask):
                continue
            for ca, a, pa in trees(left):
                for cb, b, pb in trees(mask ^ left):
                    work = prod(lengths[c] for c in set(a + b))
                    cost = (ca[0] + cb[0] + work / shared, max(ca[1], cb[1], size), ca[2] + cb[2] + work)
                    found.append((cost, letters, (a + "," + b + "->" + letters, pa, pb)))
        return tuple(found)

    candidates = trees(whole)
    limit = max(nv**4, min(cost[1] for cost, _, _ in candidates))
    return min((c for c in candidates if c[0][1] <= limit), key=lambda c: c[0])[2]


def _expression_key(tree, keys):
    if isinstance(tree, int):
        return keys[tree]
    spec, left, right = tree
    return spec, _expression_key(left, keys), _expression_key(right, keys)


def _tree_irrep(tree, irreps):
    if isinstance(tree, int):
        return irreps[tree]
    return _tree_irrep(tree[1], irreps) ^ _tree_irrep(tree[2], irreps)


def _evaluate(xp, tree, operands, keys=None, cache=None, blocked=None):
    if isinstance(tree, int):
        return operands[tree]()
    spec, left, right = tree
    key = _expression_key(tree, keys) if cache is not None else None
    if cache is not None:
        found = cache.get(key)
        if found is not None:
            return found
    a = _evaluate(xp, left, operands, keys, cache, blocked)
    b = _evaluate(xp, right, operands, keys, cache, blocked)
    if blocked is None:
        result = einsum(xp, spec, a, b)
    else:
        engine, labels, irreps = blocked
        lhs, output = spec.split("->")
        sa, sb = lhs.split(",")
        extents = dict(zip(sa, a.shape)) | dict(zip(sb, b.shape))
        result = xp.empty(tuple(extents[c] for c in output), dtype=np.float64)
        engine.einsum(spec, a, b, result, labels=labels,
                      irreps=(_tree_irrep(left, irreps), _tree_irrep(right, irreps)))
    if cache is not None:
        cache.put(key, result)
    return result


def tile_product(xp, spec, operands, target, coefficient, no, nv, fixed=(), *, keys=None, cache=None,
                 blocked=None, ordered=()):
    """Evaluate only the requested occupied tile and accumulate into its output."""
    _, output = spec.split("->")
    if len(operands) == 1:
        target += coefficient * xp.einsum(spec, operands[0]())
        return
    tree = contraction_schedule(spec, no, nv, fixed)
    binary, left, right = tree
    axes = binary.split("->")[1]
    view = target.transpose(tuple(output.index(c) for c in axes))
    a = _evaluate(xp, left, operands, keys, cache, blocked)
    b = _evaluate(xp, right, operands, keys, cache, blocked)
    if blocked is None:
        einsum_into(xp, view, coefficient, binary, a, b)
    else:
        engine, labels, irreps = blocked
        engine.einsum(binary, a, b, view, alpha=coefficient, beta=1.0, labels=labels,
                      irreps=(_tree_irrep(left, irreps), _tree_irrep(right, irreps)), ordered=ordered)


def spin_blocks(tensor):
    """Views of interleaved spin tensors indexed by their alpha/beta signatures."""
    if tensor is None:
        return None
    return {signature: tensor[tuple(slice(s, None, 2) for s in signature)]
            for signature in product((0, 1), repeat=tensor.ndim)}


@lru_cache(maxsize=None)
def routes(spec, target, derivative):
    """Spin-conserving primal legs and a single alpha-particle/beta-hole tangent."""
    inputs, output = spec.split("->")
    inputs = inputs.split(",")
    internal = sorted(set("".join(inputs)) - set(output))
    fixed = dict(zip(output, target))
    selected = []
    for spins in product((0, 1), repeat=len(internal)):
        assigned = fixed | dict(zip(internal, spins))
        signatures = tuple(tuple(assigned[c] for c in term) for term in inputs)
        if all(sum(s[len(s) // 2:]) - sum(s[:len(s) // 2]) == (i == derivative)
               for i, s in enumerate(signatures)):
            selected.append(signatures)
    return tuple(selected)


@lru_cache(maxsize=None)
def _permutation_pairs(rank, groups):
    valid = []
    for subsets in groups:
        valid.append(tuple((axes, sign) for axes, sign in signed_permutations(rank)
                           if all(all(axes[i] < axes[j] for i, j in zip(subset, subset[1:]))
                                  for subset in subsets)))
    return tuple(product(*valid))


@lru_cache(maxsize=None)
def external_routes(rank, signature, equivalent=((), ())):
    """Group antisymmetrizer terms by raw virtual and occupied spin pattern."""
    groups = defaultdict(list)
    for (p, sp), (h, sh) in _permutation_pairs(rank, equivalent):
        axes = p + tuple(rank + i for i in h)
        raw = tuple(signature[i] for i in axes)
        groups[raw].append((p, h, sp * sh))
    return tuple((raw, tuple(rows)) for raw, rows in groups.items())


@lru_cache(maxsize=None)
def projection_diagrams(rank):
    """Group diagrams whose intrinsic operand antisymmetry removes the same routes.

    An external label occurs only once in each generated diagram. Swapping
    two external axes in one antisymmetric half of an operand changes the
    contraction sign and the external permutation sign together.
    """
    groups = defaultdict(list)
    for diagram in DIAGRAMS[rank]:
        spec = diagram[1]
        lhs, rhs = spec.split("->")
        inputs = lhs.split(",")
        halves = [half for s in inputs for half in (s[:len(s)//2], s[len(s)//2:])]
        partitions = []
        for side in (rhs[:rank], rhs[rank:]):
            subsets = [tuple(i for i, c in enumerate(side) if c in half and lhs.count(c) == 1)
                       for half in halves]
            partitions.append(tuple(sorted(s for s in subsets if len(s) > 1)))
        groups[tuple(partitions)].append(diagram)
    return tuple((group, prod(factorial(len(s)) for side in group for s in side), tuple(diagrams))
                 for group, diagrams in groups.items())


@lru_cache(maxsize=None)
def ordered_letters(equivalent, raw, output):
    """Ordering constraints (letter, letter, strict) on the particle letters of
    a raw tile that its projection reads. A projection group takes only the
    coset permutations that keep each antisymmetric subset of external
    positions (letters from one operand half) increasing (`_permutation_pairs`):
    raw letters x before y of a subset hold spin orbitals 2x + s_x < 2y + s_y,
    so x < y, or x <= y when s_x < s_y. A tile product forms only those entries
    (its other entries are never read): half of an antisymmetric pair, five
    sixths of a triple."""
    rank = len(output) // 2
    found = []
    for subset in equivalent[0]:
        for p, q in zip(subset, subset[1:]):
            found.append((output[p], output[q], not raw[p] < raw[q]))
    return tuple(found)


_PACKED_TILE_BYTES = 256 * 2**20
_ALL = (None, None)
_TILE_LABELS = {}


def _tile_labels(occupied, selected):
    """Occupied irrep labels of one tile, one dict object per tile: the blocked
    engine keys its plans by the dict's identity instead of its contents."""
    key = id(occupied), tuple((c, s.start, s.stop) for c, s in selected.items())
    found = _TILE_LABELS.get(key)
    if found is not None and found[0] is occupied:
        return found[1]
    labels = {c: occupied[s] for c, s in selected.items()}
    if len(_TILE_LABELS) > 1 << 16:
        _TILE_LABELS.clear()
    _TILE_LABELS[key] = occupied, labels
    return labels


@lru_cache(maxsize=4096)
def _packed_cuts(holes, free, no, nv, budget, occupied_slab):
    """Tile both free and summed occupied axes before expanding packed T4.
    Summed axes are at most `occupied_slab` wide (None: as wide as the byte
    budget allows, so a product sums over as many slots as one call can)."""
    bounds = {c: (start, stop) for c, start, stop in free}
    regions = [bounds.get(c, (0, no)) for c in holes]
    widths = [stop - start if c in bounds or occupied_slab is None else min(occupied_slab, stop - start)
              for c, (start, stop) in zip(holes, regions)]
    max_slots = max(1, budget // (8 * nv**4))
    while prod(widths) > max_slots:
        index = max(range(len(widths)), key=widths.__getitem__)
        widths[index] = (widths[index] + 1) // 2
    return tuple(tuple(zip(holes, ranges)) for ranges in product(
        *(tuple(slice(i, min(i + width, stop)) for i in range(start, stop, width))
          for (start, stop), width in zip(regions, widths))))


def contract_tile(xp, rank, raw, fixed, primal, tangent, no, nv, *, irrep=None,
                  labels=None, engine=None, occupied_slab=None, diagrams=None, intermediates=None,
                  compiled_products=False, out=None, equivalent=None):
    """Raw virtual tile for one external spin pattern and occupied tuple
    (added into `out` when given: a zeroed array of the tile's shape). With
    the projection group's `equivalent` subsets, the blocked engine forms only
    the entries the projection reads (`ordered_letters`)."""
    output_shape = (nv,) * rank + tuple(value.stop - value.start for value in fixed)
    if out is not None and out.shape != output_shape:
        raise ValueError("The output tile has the wrong shape")
    result = xp.zeros(output_shape) if out is None else out
    # T4 enters each diagram at most once. Expanding a given spin slab
    # repeatedly across diagrams is more expensive than the contractions.
    cache = {}
    cache_bytes = 0
    max_cache_bytes = _PACKED_TILE_BYTES
    for coefficient, spec, names in DIAGRAMS[rank] if diagrams is None else diagrams:
        lhs, output = spec.split("->")
        inputs = lhs.split(",")
        cuts = dict(zip(output[rank:], fixed))
        t4 = names.index("t4") if "t4" in names else None
        # The CPU blocked engine reads packed T4 rows in place: the dense
        # slab is never written (it was most of an action's memory traffic).
        packed = (t4 is not None and engine is not None and getattr(engine, "packed_operands", False)
                  and (len(names) == 2 or compiled_products))
        if t4 is None or (packed and occupied_slab is None):
            # Read in place, T4 needs no slab: its summed occupied letters
            # stay whole, one product (and one intermediate per occupied
            # tuple, shared between tiles) instead of one per slab. At CuF
            # size the byte budget below cut them into 28 slabs.
            slabs = ((),)
        else:
            # Caching multi-slot expansions across diagrams is only useful
            # when the free axes are tiled as well as the contracted ones.
            # A wide slab is one deep product instead of several shallow ones
            # that each read and write the whole output tile; a quarter of the
            # cache keeps several expansions reusable across diagrams.
            slabs = _packed_cuts(inputs[t4][4:], tuple((c, s.start, s.stop) for c, s in cuts.items()),
                                 no, nv, max_cache_bytes if occupied_slab else max_cache_bytes // 4,
                                 occupied_slab)
        for derivative in range(len(names)):
            if tangent[names[derivative]] is None:
                continue
            for signatures in routes(spec, raw, derivative):
                for pieces in slabs:
                    selected = cuts | dict(pieces)
                    out_cuts = tuple(slice(selected[c].start - s.start, selected[c].stop - s.start)
                                     for c, s in cuts.items())
                    target = result[(slice(None),) * rank + out_cuts]

                    def fetch(i):
                        nonlocal cache_bytes
                        block = (tangent if i == derivative else primal)[names[i]][signatures[i]]
                        selector = tuple(selected.get(c, slice(None)) for c in inputs[i])
                        if not callable(block):
                            return block[selector]
                        # Packed rows only reach the blocked engine; dense tiles anything.
                        key = packed, i == derivative, signatures[i], tuple((s.start, s.stop) for s in selector)
                        if key in cache:
                            return cache[key]
                        tile = block(selector, packed=True) if packed else block(selector)
                        if tile.nbytes <= max_cache_bytes:
                            while cache_bytes + tile.nbytes > max_cache_bytes:
                                oldest = next(iter(cache))
                                cache_bytes -= cache.pop(oldest).nbytes
                            cache[key] = tile
                            cache_bytes += tile.nbytes
                        return tile

                    operands = tuple(lambda i=i: fetch(i) for i in range(len(names)))
                    ordered = (() if equivalent is None or engine is None else
                               ordered_letters(equivalent, raw, output))
                    if engine is not None and len(names) == 2:
                        engine.einsum(spec, *[op() for op in operands], target, alpha=coefficient,
                                      beta=1.0, labels=_tile_labels(labels["o"], selected),
                                      irreps=tuple(irrep if i == derivative else 0 for i in range(2)),
                                      ordered=ordered)
                    else:
                        unit = tuple(c for c in cuts if selected[c].stop - selected[c].start == 1)
                        keys = None
                        if intermediates is not None:
                            bounds = {c: (s.start, s.stop) for c, s in selected.items()}
                            keys = tuple((name, i == derivative, signatures[i],
                                          tuple(bounds.get(c, _ALL) for c in inputs[i]))
                                         for i, name in enumerate(names))
                        tile_product(xp, spec, operands, target, coefficient, no, nv, unit,
                                     keys=keys, cache=intermediates,
                                     blocked=((engine, _tile_labels(labels["o"], selected),
                                               tuple(irrep if i == derivative else 0 for i in range(len(names))))
                                              if compiled_products and engine is not None else None),
                                     ordered=ordered if compiled_products else ())
    return result
