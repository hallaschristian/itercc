"""Packed T4 blocks and box-level plans for the CPU T4 passes.

The T4 passes of the CCSDTQ residual contract whole blocks of T4 (all values
of one or two occupied indices, the others fixed). Dense blocks are three
quarters symmetry zeros; here each block row is stored as its allowed virtual
quadruples only, in the slot box layout of the row's irrep (`PackedRows`).

Several contractions split a row's virtual quadruple between operands (one
virtual is summed against W, say), so a row's addresses are not a sum of a
row part and a column part over composite irreps. With the irreps of every
index fixed, though, every address in a dense array or a packed row is affine
in the local indices. `build_plan` therefore enumerates the irreps of all
indices (each operand totally symmetric up to its own irrep) and emits one
block per irrep assignment with explicit offset segments. Blocks that differ
only in the irreps of summed indices write the same output entries; they run
as the segments of one fused GEMM (`itercc_fused_contract`: their sums laid
end to end, the output updated once per 256 of them), and the output-disjoint
fused blocks all run in one parallel call. Plans depend only on the
spectators' irreps and are cached.

With an `OutputSector` on the layout (a one-sided singlet EOM-CCSDTQ action)
the ring plans form only the R4 entries of that sector: each block splits into
the row and column classes that reach it, its sum whole (`build_plan`'s
`sector`).
"""
from functools import lru_cache, reduce
from itertools import product
from math import prod
from operator import xor
import threading

import numpy as np

from .symmetry_blocks import _parse
from . import _runtime


class OutputSector:
    """The residual entries of one symmetry sector of a finer group than the
    layout's: the target sector of a one-sided singlet EOM-CCSDTQ action on a
    solver with `spatial_symmetry.sector_labels` (the sector part of the
    residual is all it packs). `occupied` and `virtual` are the finer irrep
    labels of the orbitals in the layout's order; an entry belongs to the
    sector when the labels of its indices XOR to `irrep`. While a layout
    carries one (`layout.output_sector`), the R4 tile programs and the ring
    terms form only those entries, each from the same products in the same
    order (the others are left unformed). The oooo terms and the r2 and r3
    terms of T4 form every entry: split by sector their GEMMs ran no faster
    (13o/63v B1 on one socket: oooo 54 -> 57 s, r2/r3 within noise)."""

    def __init__(self, occupied, virtual, irrep):
        self.occupied = np.ascontiguousarray(occupied, dtype=np.int64)
        self.virtual = np.ascontiguousarray(virtual, dtype=np.int64)
        self.irrep = int(irrep)
        self.key = (self.occupied.tobytes(), self.virtual.tobytes(), self.irrep)

    def target(self, *occupied):
        """The XOR the other indices' labels need with these occupied ones fixed."""
        return reduce(xor, (int(self.occupied[i]) for i in occupied), self.irrep)


def output_sector(layout):
    """The layout's `OutputSector`, or None (every allowed entry)."""
    return getattr(layout, "output_sector", None)


class PackedRows:
    """Rows of a packed T4 block. Free occupied row letters have irrep arrays
    `labels` over their positions; `irrep` is that of the fixed occupied
    indices. Rows of one irrep combination of the row letters form a grid
    (C order over the letters' positions within their irreps), each row the
    packed slot of irrep `irrep ^ combination` (`layout._slot_sizes()`)."""

    def __init__(self, layout, labels, irrep):
        sizes = layout._slot_sizes()
        self.nirrep = layout.nirrep
        self.labels = [np.asarray(label, dtype=np.int64) for label in labels]
        shape = tuple(len(label) for label in self.labels)
        self.offset = np.zeros(shape, dtype=np.int64)
        self.irrep_of = np.zeros(shape, dtype=np.int64)
        self.base, self.length, self.counts = {}, {}, {}
        total = 0
        for combo in product(range(self.nirrep), repeat=len(self.labels)):
            members = [np.nonzero(label == g)[0] for label, g in zip(self.labels, combo)]
            if any(not len(m) for m in members):
                continue
            h = reduce(xor, combo, int(irrep))
            length, counts = int(sizes[h]), [len(m) for m in members]
            n = int(np.prod(counts))
            grid = np.ix_(*members)
            self.offset[grid] = total + (np.arange(n, dtype=np.int64) * length).reshape(counts)
            self.irrep_of[grid] = h
            self.base[combo], self.length[combo], self.counts[combo] = total, length, counts
            total += n * length
        self.size = total

    def cached_strides(self, combo):
        cache = self.__dict__.setdefault("_strides", {})
        if combo not in cache:
            cache[combo] = self.strides(combo)
        return cache[combo]

    def strides(self, combo):
        """Offset steps of the row letters within the grid of `combo`."""
        counts, steps, later = self.counts[combo], [], self.length[combo]
        for count in reversed(counts):
            steps.append(later)
            later *= count
        return steps[::-1]


class Dense:
    """A dense operand (a view): offsets from its strides."""

    def __init__(self, letters, array, irrep=0):
        self.letters, self.irrep = letters, int(irrep)
        self.strides = {c: s // array.itemsize for c, s in zip(letters, array.strides)}

    def place(self, irreps, members):
        return 0, {c: members[c] * self.strides[c] for c in self.letters}


class Packed:
    """Packed block rows: `row_letters` index the rows (`rows`), `virtual` are
    the four box positions, other letters are fixed occupied indices."""

    def __init__(self, letters, rows, row_letters, virtual, layout, irrep=0):
        self.letters, self.rows, self.irrep = letters, rows, int(irrep)
        self.row_letters, self.virtual = tuple(row_letters), tuple(virtual)
        self.vsize = [int(x) for x in layout._box_tables[0]]
        self.boff = [int(x) for x in layout._box_tables[2]]
        self.nirrep = layout.nirrep

    def place(self, irreps, members):
        combo = tuple(irreps[c] for c in self.row_letters)
        g0, g1, g2, g3 = (irreps[c] for c in self.virtual)
        h = g0 ^ g1 ^ g2 ^ g3
        if self.rows.irrep_of[tuple(members[c][0] for c in self.row_letters)] != h:
            raise ValueError("Packed row irrep and box irreps disagree")
        n1, n2, n3 = self.vsize[g1], self.vsize[g2], self.vsize[g3]
        box = (n1 * n2 * n3, n2 * n3, n3, 1)
        N = self.nirrep
        const = self.rows.base[combo] + self.boff[((h * N + g0) * N + g1) * N + g2]
        offsets = {c: _steps(len(members[c]), 0) for c in self.letters}
        for k, c in enumerate(self.virtual):
            offsets[c] = _steps(len(members[c]), box[k])
        for c, step in zip(self.row_letters, self.rows.cached_strides(combo)):
            offsets[c] = _steps(len(members[c]), step)
        return const, offsets


class PairBlocks:
    """Irrep-sorted virtual pairs (the last two axes of a T3-like array, nv^2
    entries per row) stored block by block: the pairs of each irrep pair
    (g, h) contiguous, C order inside, the blocks in lexicographic irrep order.
    `position[v * nv + w]` is where pair (v, w) goes, `index` the inverse (the
    gather table of `permute`).

    In a product the address of a T3 entry is an occupied part plus
    v1 nv^2 + v2 nv + v3. With the pairs blocked it is the occupied part plus
    v1 nv^2 + position[v2 nv + v3], still a sum of a free and a summed part
    whenever v2 and v3 are both free or both summed (`remap`), and a block of
    free pairs runs contiguously over n_g n_h entries (`n_g` virtuals of
    irrep g) instead of single rows of n_h."""

    def __init__(self, virtual_irreps, nirrep):
        vir = np.asarray(virtual_irreps, dtype=np.int64)
        if np.any(np.diff(vir) < 0):
            raise ValueError("Blocked pairs need irrep-sorted virtual orbitals")
        nv = len(vir)
        self.nv, self.nirrep = nv, int(nirrep)
        self.first = np.searchsorted(vir, np.arange(nirrep)).astype(np.int64)
        self.size = np.bincount(vir, minlength=nirrep).astype(np.int64)
        base = np.zeros((nirrep, nirrep), dtype=np.int64)
        total = 0
        for g, h in product(range(nirrep), repeat=2):
            base[g, h] = total
            total += int(self.size[g] * self.size[h])
        self.base = base
        local = np.arange(nv) - self.first[vir]
        position = (base[vir[:, None], vir[None, :]] + local[:, None] * self.size[vir][None, :]
                    + local[None, :])
        self.position = np.ascontiguousarray(position.reshape(-1), dtype=np.int64)
        self.index = np.empty(nv**2, dtype=np.int64)
        self.index[self.position] = np.arange(nv**2)

    def permute(self, array, *, single=False, threads=None):
        """A copy of `array` (C order, last two axes virtual) with every row of
        pairs blocked; float32 with `single`."""
        from ._runtime import cpu_threads
        from . import _cpu_kernels

        array = np.ascontiguousarray(array, dtype=np.float64)
        ncols = self.nv**2
        out = np.empty(array.shape, dtype=np.float32 if single else np.float64)
        library = _cpu_kernels.library()
        kernel = library.permute_columns_f32 if single else library.permute_columns
        kernel(array.ctypes.data, array.size // ncols, ncols, self.index.ctypes.data, out.ctypes.data,
               cpu_threads(threads))
        return out

    def remap(self, table):
        """An offset table of the dense array as one of the blocked copy: the
        pair part (offset mod nv^2) through `position`. Valid for the tables of
        an operand whose last two letters lie in one of them (the other tables
        then hold multiples of nv^2)."""
        table = np.asarray(table, dtype=np.int64)
        pair = table % self.nv**2
        return np.ascontiguousarray(table - pair + self.position[pair], dtype=np.int64)


class Plan:
    """Block tables with their offset segments. Blocks of different irreps of
    the summed indices can write the same output entries (the waves: sets of
    output-disjoint blocks); they run as the segments of one fused block each
    (`itercc_fused_contract`, all output-disjoint fused blocks in one call)."""

    def __init__(self, waves, segments, flags):
        self.waves = [np.ascontiguousarray(np.asarray(t, dtype=np.int64).reshape(-1, 9)) for t in waves]
        (self.a_rows, self.rpos), (self.b_cols, self.cpos), (self.a_sum, self.b_sum) = segments
        self.flags = flags
        self._fused = None

    def _fuse(self):
        """Fused blocks (one per set of output entries, its blocks in wave order)
        and their segments, as core structures."""
        from . import _core

        groups, order = {}, []
        for table in self.waves:
            for nr, nc, nk, r0, c0, k0, ca, cb, co in table.tolist():
                key = (co, self.rpos[r0:r0 + nr].tobytes(), self.cpos[c0:c0 + nc].tobytes())
                if key not in groups:
                    groups[key] = []
                    order.append(key)
                groups[key].append((nr, nc, nk, r0, c0, k0, ca, cb, co))
        count = sum(len(groups[key]) for key in order)
        blocks = (_core.FusedBlock * max(1, len(order)))()
        segments = (_core.FusedSegment * max(1, count))()
        item = lambda array, start: array.ctypes.data + 8 * start
        flags = int(self.flags & 1) | int(self.flags & 2)
        position = 0
        for block, key in zip(blocks, order):
            members = groups[key]
            nr, nc, _, r0, c0, _, _, _, co = members[0]
            block.rows, block.cols, block.o_base = nr, nc, co
            block.rpos, block.cpos = item(self.rpos, r0), item(self.cpos, c0)
            block.seg_start, block.seg_count, block.tag = position, len(members), 0
            for nr, nc, nk, r0, c0, k0, ca, cb, _ in members:
                segment = segments[position]
                segment.x_base, segment.y_base = 0, 1
                segment.x_coef[0], segment.y_coef[0] = ca, cb
                segment.x_rows, segment.x_sum = item(self.a_rows, r0), item(self.a_sum, k0)
                segment.y_sum, segment.y_cols = item(self.b_sum, k0), item(self.b_cols, c0)
                segment.depth, segment.alpha, segment.flags = nk, 1.0, flags
                position += 1
        self._fused = (len(order), blocks, segments)

    def run(self, library, a, b, out, alpha, threads, *, b_f32=False):
        """out += alpha * the product of operands a and b (data pointers); with
        b_f32, b holds floats (widened to double as the GEMMs pack them)."""
        import ctypes

        if self._fused is None:
            self._fuse()
        count, blocks, segments = self._fused
        if count:
            bases = (ctypes.c_void_p * 2)(a, b)
            narrow = (ctypes.c_uint8 * 2)(0, 1) if b_f32 else None
            library.fused_contract(count, ctypes.cast(blocks, ctypes.c_void_p),
                                   ctypes.cast(segments, ctypes.c_void_p),
                                   ctypes.cast(bases, ctypes.c_void_p),
                                   None if narrow is None else ctypes.cast(narrow, ctypes.c_void_p),
                                   out, float(alpha), threads)


_STEPS = {}


def _steps(length, step):
    """arange(length) * step as int64, shared (read-only): plans repeat a few
    hundred (length, step) pairs tens of thousands of times."""
    key = (length, step)
    hit = _STEPS.get(key)
    if hit is None:
        hit = np.arange(length, dtype=np.int64) * step
        hit.flags.writeable = False
        _STEPS[key] = hit
    return hit


_ONE = np.zeros(1, dtype=np.int64)
_ONE.flags.writeable = False


def _grid(letters, offsets):
    """Offsets of the C-order grid over `letters` (one entry without letters;
    read-only, possibly an operand's own offsets)."""
    if not letters:
        return _ONE
    total = offsets[letters[0]]
    for c in letters[1:]:
        total = np.add.outer(total, offsets[c]).reshape(-1)
    return total


def _assignments(letters, options, operands):
    """Irrep assignments of `letters` (from `options`) under which every operand
    has its irrep (XOR over its letters). An operand's last unassigned letter is
    determined by the others, so only consistent branches are visited."""
    allowed = [set(o) for o in options]
    position = {c: i for i, c in enumerate(letters)}
    # An operand's irrep fixes its last letter (in assignment order).
    forcing = [[] for _ in letters]
    for op in operands:
        places = sorted(position[c] for c in set(op.letters))
        if places:
            forcing[places[-1]].append((places[:-1], op.irrep))
    values = [0] * len(letters)

    def visit(i):
        if i == len(letters):
            yield dict(zip(letters, values))
            return
        forced = None
        for others, irrep in forcing[i]:
            need = irrep
            for j in others:
                need ^= values[j]
            if forced is not None and forced != need:
                return
            forced = need
        for value in ([forced] if forced is not None else options[i]):
            if value in allowed[i]:
                values[i] = value
                yield from visit(i + 1)
    yield from visit(0)


def build_plan(spec, labels, a, b, out, nirrep, sector=None):
    """Box-level plan of out += alpha * einsum(spec, a, b) for Dense or Packed
    operand descriptors; `labels` maps every letter to the irreps of its
    positions (the same index range in every operand). With `sector` =
    (fine, irrep), `fine` mapping the output letters to finer irrep labels of
    the same positions, only the output entries whose letters' fine labels
    XOR to `irrep` are formed: a block's rows and columns take the classes of
    their letters, and each block splits into the class pairs that reach the
    sector, its sum whole (those entries get the same products in the same
    order; the others are not written)."""
    if len(_STEPS) > 1 << 16:          # not during a build: its grids key on these arrays
        _STEPS.clear()
    sa, sb, so = _parse(spec)
    rows = [c for c in so if c in sa and c not in sb]
    cols = [c for c in so if c in sb and c not in sa]
    sums = [c for c in sa if c in sb and c not in so]
    letters = list(dict.fromkeys(sa + sb + so))
    if set(letters) != set(rows + cols + sums):
        raise ValueError(f"Unsupported contraction {spec}")
    members = {c: [np.nonzero(np.asarray(labels[c]) == g)[0] for g in range(nirrep)] for c in letters}
    options = [[g for g in range(nirrep) if len(members[c][g])] for c in letters]
    operands = (a, b, out)
    table, steps, seen = [], {}, {}
    segments = {"r": ([], []), "c": ([], []), "k": ([], [])}
    starts = {"r": {}, "c": {}, "k": {}}
    sizes = {"r": 0, "c": 0, "k": 0}

    known = {}

    def segment(kind, first, second):
        # Segments with equal offsets are stored once; equal grid keys mean equal offsets.
        token = (kind, first[0], second[0])
        hit = known.get(token)
        if hit is not None:
            return hit
        first, second = first[1], second[1]
        key = (first.tobytes(), second.tobytes())
        if key not in starts[kind]:
            starts[kind][key] = sizes[kind]
            segments[kind][0].append(first)
            segments[kind][1].append(second)
            sizes[kind] += len(first)
        known[token] = starts[kind][key]
        return known[token]

    placed, grids, targets = {}, {}, {}

    def place(index, g):
        # An operand's offsets depend only on the irreps of its own letters.
        op = operands[index]
        key = (index, tuple(g[c] for c in op.letters))
        if key not in placed:
            placed[key] = op.place(g, {c: members[c][g[c]] for c in op.letters})
        return key, placed[key]

    def grid(group, offsets):
        """(key, offsets) of the grid over `group`: one per sequence of offset
        arrays (the placements keep theirs alive, `_steps` shares equal ones)."""
        index = tuple(id(offsets[c]) for c in group)
        hit = grids.get(index)
        if hit is None:
            hit = grids[index] = (index, _grid(group, offsets))
        return hit

    classes, picks = {}, {}

    def classes_of(group, g):
        """(key, fine class of every entry of the grid over `group`)."""
        key = (group, tuple(g[c] for c in group))
        hit = classes.get(key)
        if hit is None:
            total = np.zeros(1, dtype=np.int64)
            for c in group:
                total = np.bitwise_xor.outer(total, fine[c][members[c][g[c]]]).reshape(-1)
            hit = classes[key] = (key, total)
        return hit

    def pick(entries, cls, value):
        """(key, offsets) of the grid entries of one class."""
        key = (entries[0], cls[0], value)
        hit = picks.get(key)
        if hit is None:
            hit = picks[key] = (key, np.ascontiguousarray(entries[1][cls[1] == value]))
        return hit

    if sector is not None:
        fine, wanted = sector
        fine = {c: np.asarray(fine[c], dtype=np.int64) for c in rows + cols}
    rows_, cols_, sums_ = tuple(rows), tuple(cols), tuple(sums)
    for g in _assignments(letters, options, operands):
        (ka, (ca, oa)), (kb, (cb, ob)), (ko, (co, oo)) = (place(i, g) for i in range(3))
        if not steps:
            for name, offsets in (("a", oa), ("b", ob), ("o", oo)):
                for c, values in offsets.items():
                    steps[name, c] = int(values[1] - values[0]) if len(values) > 1 else 1 << 60
        pieces = [(grid(rows_, oa), grid(rows_, oo), grid(cols_, ob), grid(cols_, oo))]
        if sector is not None:
            # The rows of one class with the columns that complete the sector.
            a_rows, out_rows, b_cols, out_cols = pieces[0]
            r_class, c_class = classes_of(rows_, g), classes_of(cols_, g)
            pieces = [(pick(a_rows, r_class, value), pick(out_rows, r_class, value),
                       pick(b_cols, c_class, value ^ wanted), pick(out_cols, c_class, value ^ wanted))
                      for value in np.unique(r_class[1]).tolist()
                      if np.any(c_class[1] == value ^ wanted)]
        k0 = segment("k", grid(sums_, oa), grid(sums_, ob))
        nk = prod(len(members[c][g[c]]) for c in sums)
        for a_rows, out_rows, b_cols, out_cols in pieces:
            r0 = segment("r", a_rows, out_rows)
            c0 = segment("c", b_cols, out_cols)
            nr, nc = len(out_rows[1]), len(out_cols[1])
            # Blocks with the same output entries go to successive waves.
            token = (co, out_rows[0], out_cols[0])
            target = targets.get(token)
            if target is None:
                target = targets[token] = (co, out_rows[1].tobytes(), out_cols[1].tobytes())
            wave = seen.get(target, 0)
            seen[target] = wave + 1
            while len(table) <= wave:
                table.append([])
            table[wave].append((nr, nc, nk, r0, c0, k0, ca, cb, co))

    def fastest(name, group):
        return min((steps.get((name, c), 1 << 60) for c in group), default=1 << 60)
    flags = (int(fastest("a", rows) < fastest("a", sums)) | 2 * int(fastest("b", sums) < fastest("b", cols))
             | 4 * int(fastest("o", rows) < fastest("o", cols)))
    joined = tuple(tuple(np.ascontiguousarray(np.concatenate(part) if part else np.zeros(0, np.int64),
                                              dtype=np.int64) for part in segments[kind])
                   for kind in ("r", "c", "k"))
    return Plan(table, joined, flags)


_SHARED = {}


def _shared_cache(layout, name):
    """Plans of one kind for every solver whose layout has the same orbital symmetry."""
    key = (name, layout.nocc, layout.nvir, np.asarray(layout.occupied_irreps).tobytes(),
           np.asarray(layout.virtual_irreps).tobytes())
    if len(_SHARED) > 64:
        _SHARED.clear()
    return _SHARED.setdefault(key, {})


def _row_slots(layout, rows_of):
    """(slot, order) of block rows given as occupied 4-tuples in row order."""
    slots, orders = [], []
    for actual in rows_of:
        slots.append(layout._slot[tuple(sorted(actual))])
        orders.append(sorted(range(4), key=actual.__getitem__))
    return (np.ascontiguousarray(slots, dtype=np.int64), np.ascontiguousarray(orders, dtype=np.int32))


def _terms(kind):
    from ._quadruples_cuda import _RING_TERMS, _gather_terms

    if kind == "ring":
        return np.ascontiguousarray(_RING_TERMS, dtype=np.int64), np.ones(3), 1
    terms = _gather_terms(kind)
    return (np.ascontiguousarray([t for t, _ in terms], dtype=np.int64),
            np.ascontiguousarray([w for _, w in terms], dtype=np.float64), 0)


def _gather_permute(layout, t4, target, rows, slots, orders, kind, threads):
    """The rows (slots, orders) of packed `t4` gathered and spin- or ring-summed
    into `target` in one pass (`_permute` of the gathered rows)."""
    perms, weights, ring = _terms(kind)
    offsets = np.ascontiguousarray(rows.offset.reshape(-1), dtype=np.int64)
    gather = (layout._kernels.quad_gather_permute_f32 if target.dtype == np.float32
              else layout._kernels.quad_gather_permute)
    gather(layout._layout, t4.ctypes.data, len(slots), slots.ctypes.data, orders.ctypes.data,
           offsets.ctypes.data, len(weights), perms.ctypes.data, weights.ctypes.data, ring,
           target.ctypes.data, threads)


def _permute(layout, source, target, rows, kind, threads):
    perms, weights, ring = _terms(kind)
    offsets = np.ascontiguousarray(rows.offset.reshape(-1), dtype=np.int64)
    irreps = np.ascontiguousarray(rows.irrep_of.reshape(-1), dtype=np.int32)
    layout._kernels.quad_rows_permute(layout._layout, source.ctypes.data, target.ctypes.data,
                                      len(offsets), offsets.ctypes.data, irreps.ctypes.data,
                                      len(weights), perms.ctypes.data, weights.ctypes.data, ring, threads)


_RING_SPECS = (("mx,mabcd->xabcd", 0, "plain", -1.0), ("mbxe,maecd->xabcd", 1, "plain", -1.0),
               ("mcxe,mabed->xabcd", 1, "plain", -1.0), ("mdxe,mabce->xabcd", 1, "plain", -1.0),
               ("maxe,mbecd->xabcd", 1, "ring", -0.5), ("amxe,mebcd->xabcd", 2, "spin", 0.5))


def add_rings(layout, r4, t4, F_oo, W_ovov, W_voov, threads):
    """`_quadruples_equations.r4_add_rings_triples_` on packed blocks: for each
    sorted spectator triple (p, q, r) the rows T4[m, p, q, r] are gathered
    without symmetry zeros, the ring and spin sums are formed on the packed
    rows, the six contractions write packed rows A[x], and A is added to the
    slots of every target position. With an `OutputSector` the contractions
    form only A's entries in the sector (A[x] goes to slots of x, p, q, r)."""
    nocc, N = layout.nocc, layout.nirrep
    occ = np.asarray(layout.occupied_irreps, dtype=np.int64)
    vir = np.asarray(layout.virtual_irreps, dtype=np.int64)
    labels = dict(m=occ, x=occ, a=vir, b=vir, c=vir, d=vir, e=vir)
    weights = [np.asarray(F_oo), np.asarray(W_ovov), np.asarray(W_voov)]
    signature = tuple((w.shape, w.strides) for w in weights)
    cache = _shared_cache(layout, ("packed rings", signature))
    t4 = np.ascontiguousarray(t4, dtype=np.float64)
    library = layout._kernels
    local = threading.local()
    sector = output_sector(layout)
    if sector is not None:
        fine = dict(x=sector.occupied, a=sector.virtual, b=sector.virtual, c=sector.virtual,
                    d=sector.virtual)

    def run_triple(p, q, r, threads):
        G = int(occ[p] ^ occ[q] ^ occ[r])
        key = G if sector is None else (G, sector.key, sector.target(p, q, r))
        if key not in cache:
            rows = PackedRows(layout, [occ], G)
            within = None if sector is None else (fine, sector.target(p, q, r))
            plans = []
            for spec, index, _, _ in _RING_SPECS:
                sa, sb, so = _parse(spec)
                plans.append(build_plan(spec, labels, Dense(sa, weights[index]),
                                        Packed(sb, rows, sb[:1], sb[1:], layout, G),
                                        Packed(so, rows, so[:1], so[1:], layout, G), N, within))
            cache[key] = (rows, plans)
        rows, plans = cache[key]
        buffers = getattr(local, "buffers", None)
        if buffers is None or len(buffers[0]) < rows.size:
            buffers = local.buffers = [np.zeros(rows.size) for _ in range(4)]
        plain, ring, spin, out = buffers
        slots, orders = _row_slots(layout, [(m, p, q, r) for m in range(nocc)])
        offsets = np.ascontiguousarray(rows.offset, dtype=np.int64)
        library.quad_gather_rows(layout._layout, t4.ctypes.data, nocc, slots.ctypes.data,
                                 orders.ctypes.data, offsets.ctypes.data, plain.ctypes.data,
                                 threads)
        _permute(layout, plain, ring, rows, "ring", threads)
        _permute(layout, plain, spin, rows, "P4_201", threads)
        library.scale(out.ctypes.data, rows.size, 0.0, threads)
        sources = {"plain": plain, "ring": ring, "spin": spin}
        for plan, (spec, index, source, alpha) in zip(plans, _RING_SPECS):
            plan.run(library, weights[index].ctypes.data, sources[source].ctypes.data,
                     out.ctypes.data, alpha, threads)
        for entries in layout.triple_entries((p, q, r)):
            entries = np.ascontiguousarray(entries, dtype=np.int32)
            library.quad_row_add_packed(layout._layout, out.ctypes.data, offsets.ctypes.data,
                                        entries.ctypes.data, len(entries), r4.ctypes.data, threads)

    # Triples sharing at most one index add to disjoint R4 slots (a slot holds
    # {x} + triple), so they run at once on thread teams (ITERCC_LOWER_TEAMS);
    # triples sharing an index pair keep the fixed order: the same results for
    # any team count.
    _run_ordered(_triple_rounds(nocc), _triple_keys, run_triple, lower_teams(threads), threads)


# t4_into_lower_ranks_: (spec, A, out, alpha, row letters, fixed letters). The
# r2 term reads the P4_442_r2 spin sum of T4[m, n, p, q]; the others read the
# P4_201 one, where the t1 term also carries the Fock term (see below).
_LOWER_R2 = ("mnef,mnijefab->ijab", "oovv", "r2", 0.25, "mn", "ij")
_LOWER_201 = (("amef,mijkfebc->ijkabc", "vovv", "r3", 0.5, "mi", "jk"),
              ("mjen,mijkeabc->inkabc", "vo", "r3q", 1.0, "mi", "jk"),
              ("mnef,nijkfabe->abmijk", "oovv_n", "W4", 0.5, "ni", "jk"),
              ("inef,nijkfabc->abcejk", "oovv_n", "W6", -0.5, "ni", "jk"))


def _swap_last(letters):
    """B letters reading the P4_201 rows of (p, q) as those of (q, p)."""
    return letters[:2] + letters[3] + letters[2] + letters[4:6] + letters[7] + letters[6]


@lru_cache(maxsize=None)
def _pair_rounds(n):
    """The pairs p <= q of range(n), each once, in rounds whose pairs share no
    index (round-robin with a bye for odd n; the pair (p, p) takes p's bye, or
    the diagonal pairs form one more round for even n)."""
    players = list(range(n)) + ([None] if n % 2 else [])
    m = len(players)
    rounds = []
    for _ in range(m - 1):
        round_ = []
        for i in range(m // 2):
            a, b = players[i], players[m - 1 - i]
            if a is None or b is None:
                x = b if a is None else a
                round_.append((x, x))
            else:
                round_.append((min(a, b), max(a, b)))
        rounds.append(sorted(round_))
        players = [players[0], players[-1], *players[1:-1]]
    if n % 2 == 0:
        rounds.append([(p, p) for p in range(n)])
    return rounds


@lru_cache(maxsize=None)
def _triple_rounds(n):
    """The sorted triples p <= q <= r of range(n) in rounds (first fit, in
    order) whose triples share at most one index (counted with multiplicity):
    two triples write a common R4 slot only when they share two."""
    from collections import Counter

    rounds = []
    for triple in ((p, q, r) for p in range(n) for q in range(p, n) for r in range(q, n)):
        mine = Counter(triple)
        for round_ in rounds:
            if all(sum((mine & other).values()) <= 1 for other in round_[1]):
                round_[0].append(triple)
                round_[1].append(mine)
                break
        else:
            rounds.append(([triple], [mine]))
    return [triples for triples, _ in rounds]


def _pair_keys(pair):
    """Outputs of a lower-ranks pair task: its indices (r3 slabs [:, :, j])."""
    return set(pair)


def _triple_keys(triple):
    """Outputs of a ring triple task: its index pairs (with multiplicity); two
    triples reach a common R4 slot only when they share one."""
    p, q, r = triple
    return {(p, q), (p, r), (q, r)}


def _run_ordered(rounds, keys, run, teams, threads):
    """run(*task, team_threads) for the tasks of `rounds` (in that order) on
    `teams` thread teams. A task starts once every earlier task sharing one of
    its output keys has finished, so every output receives its contributions
    in the fixed order (the same results for any team count and as one team),
    and a team takes the earliest ready task as soon as it is free instead of
    waiting for a whole round (with 7 teams, rounds of 8-10 pairs at 16-19
    occupied left about half the teams idle). ITERCC_TEAM_SCHEDULE=rounds runs
    the rounds one after another (A/B switch)."""
    tasks = [task for round_ in rounds for task in round_]
    if teams <= 1:
        for task in tasks:
            run(*task, threads)
        return
    executor = _team_executor(teams)
    team_threads = max(1, threads // teams)
    if _runtime.switch("ITERCC_TEAM_SCHEDULE") == "rounds":
        for round_ in rounds:
            _finish([executor.submit(run, *task, team_threads) for task in round_])
        return
    import heapq
    from concurrent.futures import FIRST_COMPLETED, wait

    last, waiting, dependents = {}, [], [[] for _ in tasks]
    for k, task in enumerate(tasks):
        earlier = {last[key] for key in keys(task) if key in last}
        for j in earlier:
            dependents[j].append(k)
        waiting.append(len(earlier))
        last.update((key, k) for key in keys(task))
    ready = [k for k, count in enumerate(waiting) if not count]
    heapq.heapify(ready)
    running, error = {}, None
    while ready or running:
        while ready and len(running) < teams and error is None:
            k = heapq.heappop(ready)
            running[executor.submit(run, *tasks[k], team_threads)] = k
        if not running:
            break
        done, _ = wait(running, return_when=FIRST_COMPLETED)
        for future in done:
            k = running.pop(future)
            if future.exception() is not None:
                error = error or future.exception()
                continue
            for j in dependents[k]:
                waiting[j] -= 1
                if not waiting[j]:
                    heapq.heappush(ready, j)
    if error is not None:
        raise error
    if any(waiting):
        raise RuntimeError("Unscheduled team tasks")


def lower_teams(threads):
    """Pairs of `t4_into_lower_ranks` run at once: ITERCC_LOWER_TEAMS, else one
    team of about 8 threads per pair (at most 7), or 1 below 16 threads."""
    value = _runtime.integer("ITERCC_LOWER_TEAMS")
    if value:
        return max(1, min(value, threads))
    return 1 if threads < 16 else min(7, threads // 8)


_EXECUTORS = {}


def _finish(futures):
    """Wait for every future of a round, then raise the first failure (no
    worker may still write the outputs when the caller sees the error)."""
    from concurrent.futures import wait

    wait(futures)
    for future in futures:
        future.result()


def _team_executor(teams):
    """A persistent pool of `teams` Python threads: each keeps its OpenMP team
    between calls (a new calling thread would start a new libgomp pool)."""
    from concurrent.futures import ThreadPoolExecutor

    if teams not in _EXECUTORS:
        _EXECUTORS[teams] = ThreadPoolExecutor(teams, thread_name_prefix="itercc-lower")
    return _EXECUTORS[teams]


def t4_into_lower_ranks(layout, r2, r3, t4, eris_oovv, eris_vovv, fock_ov, t1_vo, W_vvoooo, W_vvvvoo,
                        chunk, threads, *, single=False):
    """`_quadruples_equations.t4_into_lower_ranks_` on packed blocks: for each
    pair p <= q and chunk of m (all n), the rows T4[m, n, p, q] are gathered
    without symmetry zeros and spin-summed on the packed rows, and the
    contractions add into r2, r3, W_vvoooo and W_vvvvoo. `t1_vo` is
    t1_eris[o, o, v, o]. (Blocks of a few MB in (m, n), to keep the rows in
    cache between the gather and the contractions, took the pass at 13o/47v
    on a Sapphire Rapids socket from 26-31 to 58-62 s: ten times the gathers
    and GEMMs.)

    Two identities cut the passes over the summed rows (the gathers and every
    contraction stream them from memory):
    - P4_201 = 2 - (01) - (02) - (03) on the virtual positions commutes with
      every permutation of positions 1, 2, 3, and T4 is unchanged by a joint
      permutation of its occupied and virtual positions; so the P4_201 rows of
      (q, p) are those of (p, q) with virtual positions 2 and 3 exchanged, and
      one gather serves both orders (read through swapped letters).
    - The Fock term (1/6) f[m, e] S[m, i, j, k, e, a, b, c] -> r3[i, j, k] is
      the n = j row of the t1 term t1[m, j, e, n] S[...] -> r3[i, n, k]; so
      one contraction with A[m, e, n] = -t1[m, j, e, n] / 2 + delta(n, j)
      f[m, e] / 6 adds both.
    The r2 spin sum P4_442_r2 has neither symmetry and is gathered per order.
    With `single`, the summed rows are kept as floats (formed in double, then
    rounded; the GEMMs widen them as they pack): half the bytes for the gather
    to write and every contraction to read.
    """
    nocc, N = layout.nocc, layout.nirrep
    occ = np.asarray(layout.occupied_irreps, dtype=np.int64)
    vir = np.asarray(layout.virtual_irreps, dtype=np.int64)
    t4 = np.ascontiguousarray(t4, dtype=np.float64)
    library = layout._kernels
    cache = _shared_cache(layout, "packed lower")

    def plan_of(spec, A, out, row_letters, fixed, chunk_, p, q, swap):
        m0, m1 = chunk_
        sa, sb, so = _parse(spec)
        # The fixed letters add no offsets; a plan depends on them only through
        # each operand's XOR of their irreps, and on the chunk only through the
        # irreps of its m (chunks alike in them share plans).
        value = {fixed[0]: int(occ[p]), fixed[1]: int(occ[q])}
        parity = tuple(reduce(xor, (value.get(c, 0) for c in letters), 0) for letters in (sa, sb, so))
        key = (spec, swap, parity, occ[m0:m1].tobytes(), A.strides, out.strides, A.shape, out.shape)
        if key not in cache:
            labels = {}
            for c in dict.fromkeys(sa + sb + so):
                labels[c] = (vir if c in "abcdef" else occ[m0:m1] if c == row_letters[0]
                             else occ[p:p + 1] if c == fixed[0] else occ[q:q + 1] if c == fixed[1]
                             else occ)
            rows = cache[("rows", int(occ[p] ^ occ[q]), occ[m0:m1].tobytes())]
            letters = _swap_last(sb) if swap else sb
            cache[key] = build_plan(spec, labels, Dense(sa, A),
                                    Packed(sb, rows, row_letters, letters[4:], layout), Dense(so, out), N)
        return cache[key]

    chunks = [(m0, min(nocc, m0 + chunk)) for m0 in range(0, nocc, chunk)]
    local = threading.local()

    def run_pair(p, q, threads):
        """All contractions of the pair p <= q (both orders) on `threads` threads.
        They write r2[i, j], r3 slabs [:, :, j] and W[..., i, j] for (i, j) in
        {(p, q), (q, p)} only, so pairs without a common index can run at once."""
        orders = [(p, q, False)] + ([(q, p, True)] if q != p else [])
        G = int(occ[p] ^ occ[q])
        for m0, m1 in chunks:
            rows_key = ("rows", G, occ[m0:m1].tobytes())
            if rows_key not in cache:
                cache[rows_key] = PackedRows(layout, [occ[m0:m1], occ], G)
            rows = cache[rows_key]
            buffer = getattr(local, "summed", None)
            if buffer is None or len(buffer) < rows.size:
                buffer = local.summed = np.zeros(rows.size, dtype=np.float32 if single else np.float64)
            summed = buffer
            # r2[p, q] and r2[q, p]: their own P4_442_r2 rows.
            spec, a_name, out_name, alpha, row_letters, fixed = _LOWER_R2
            A = eris_oovv[m0:m1]
            for i, j, _ in orders:
                out = r2[i:i + 1, j:j + 1]
                plan = plan_of(spec, A, out, row_letters, fixed, (m0, m1), i, j, False)
                slots, order = _row_slots(layout, [(m, n, i, j) for m in range(m0, m1)
                                                   for n in range(nocc)])
                _gather_permute(layout, t4, summed, rows, slots, order, "P4_442_r2", threads)
                plan.run(library, A.ctypes.data, summed.ctypes.data, out.ctypes.data, alpha, threads,
                         b_f32=single)
            # The P4_201 rows of (p, q), read for both orders while they are in cache.
            slots, order = _row_slots(layout, [(m, n, p, q) for m in range(m0, m1)
                                               for n in range(nocc)])
            _gather_permute(layout, t4, summed, rows, slots, order, "P4_201", threads)
            for i, j, swap in orders:
                vo = -0.5 * t1_vo[m0:m1, i:i + 1]
                vo[:, 0, :, i] += fock_ov[m0:m1] / 6.0
                operands = dict(vovv=eris_vovv[:, m0:m1], vo=vo, oovv_n=eris_oovv[:, m0:m1])
                outputs = dict(r3=r3[:, i:i + 1, j:j + 1], r3q=r3[:, :, j:j + 1],
                               W4=W_vvoooo[..., i:i + 1, j:j + 1], W6=W_vvvvoo[..., i:i + 1, j:j + 1])
                for spec, a_name, out_name, alpha, row_letters, fixed in _LOWER_201:
                    A, out = operands[a_name], outputs[out_name]
                    plan = plan_of(spec, A, out, row_letters, fixed, (m0, m1), i, j, swap)
                    plan.run(library, A.ctypes.data, summed.ctypes.data, out.ctypes.data, alpha, threads,
                             b_f32=single)

    # Pairs without a common index run at once on teams of threads
    # (ITERCC_LOWER_TEAMS; 1 runs them one after another); pairs sharing an
    # index (an r3 slab) keep the fixed order of the rounds, so the results do
    # not depend on the team count.
    _run_ordered(_pair_rounds(nocc), _pair_keys, run_pair, lower_teams(threads), threads)
