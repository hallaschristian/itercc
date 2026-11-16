"""Compiled CPU kernels for symmetry-packed T4 and R4 (CCSDTQ on the CPU).

`CompiledQuadruplesLayout` keeps the storage of `NumpyQuadruplesLayout` (the
packed slots of `QuadruplesLayout`) and replaces its NumPy index maps by the
OpenMP kernels of `core/src/cpu/quadruples.cpp`, which follow the CUDA
kernels of `_quadruples_cuda`. Tile gathers read the packed slots box by box
(plain, ring and spin-summed tiles in one pass); scatters, row additions,
expansion and compression copy the contiguous d-runs of each (a, b, c);
projection, purification and the Jacobi step act on dense slot batches in
place. `spin_summation` is the compiled spin summation P(T) over the trailing
virtual axes, and `einsum` runs two-operand products on the compiled
symmetry-block executor (with one irrep for dense operands). This module is
the CPU kernel module of `_quadruples_equations` when the compiled core is
available; `_quadruples_cpu` (NumPy) stays the reference
(`ITERCC_CPU_KERNELS=0` or no compiler).
"""
import ctypes
from functools import lru_cache

import numpy as np

from . import _core, _cpu_kernels
from . import _quadruples_cpu as _numpy
from ._cpu_symmetry_blocks import CpuSymmetryBlocks
from ._quadruples_cpu import NumpyQuadruplesLayout
from ._quadruples_cuda import _PERMS3, _PERMS4, _SPIN, _SPLITS, _gather_terms, _slot_batch, prange
from .symmetry_blocks import _parse


def available():
    return _cpu_kernels.library() is not None


def _threads(threads):
    from ._cpu_triples import cpu_threads

    return cpu_threads(threads)


def _host(array):
    """A host copy of an index table built for the device by the parent class."""
    return array if isinstance(array, np.ndarray) else array.get()


def _pointer(array, dtype):
    if not (isinstance(array, np.ndarray) and array.flags.c_contiguous and array.dtype == dtype):
        raise ValueError(f"Expected a C-contiguous {np.dtype(dtype).name} array")
    return array.ctypes.data


@lru_cache(maxsize=None)
def _spin_terms(pattern):
    """(perms int64 (terms, rank), weights, divisor) of the nonzero terms of `pattern`."""
    perms, weights, divisor = _SPIN[pattern]
    chosen = [(perm, weight) for perm, weight in zip(perms, weights) if weight]
    return (np.ascontiguousarray([perm for perm, _ in chosen], dtype=np.int64),
            np.ascontiguousarray([float(weight) for _, weight in chosen]), float(divisor))


def spin_summation(source, target, pattern, alpha=1.0, beta=0.0, *, threads=None):
    """target = alpha * P(source) + beta * target over the trailing virtual axes
    (rank 3 or 4 by `pattern`). Overlapping arrays are handled with a copy."""
    kernels = _cpu_kernels.library()
    perms, weights, divisor = _spin_terms(pattern)
    rank = perms.shape[1]
    nv = source.shape[-1]
    if source.shape != target.shape:
        raise ValueError("Spin summation needs arrays of one shape")
    if kernels is None or not (source.flags.c_contiguous and target.flags.c_contiguous):
        from ._quadruples_cpu import _spin_sum

        _spin_sum(source, target, pattern, alpha, beta)
        return target
    if np.may_share_memory(source, target) and not (
            source.ctypes.data == target.ctypes.data and source.size == target.size):
        source = source.copy()       # partial overlap; the kernel copies an identical array itself
    kernels.quad_spin(_pointer(source, np.float64), _pointer(target, np.float64),
                      source.size // nv**rank, nv, rank, len(weights), perms.ctypes.data,
                      weights.ctypes.data, float(alpha), divisor, float(beta), _threads(threads))
    return target


class _DenseProducts:
    """Two-operand products into `out` on the compiled executor with one irrep:
    the operands are read in place through offset tables and the products
    added into `out` with alpha, without the transposed copies and
    temporaries of `np.einsum` followed by an update."""

    def __init__(self, threads=None):
        self.engine = CpuSymmetryBlocks({}, "", 1, fallback=_numpy.einsum, threads=threads)
        self.labels = {}
        self._zeros = {}            # one label array per extent, shared

    def __call__(self, spec, a, b, out, alpha, beta):
        key = (spec, a.shape, b.shape, out.shape)
        labels = self.labels.get(key)
        if labels is None:
            extent = {}
            for letters, array in zip(_parse(spec), (a, b, out)):
                extent.update(zip(letters, array.shape))
            labels = self.labels[key] = {c: self._zeros.setdefault(n, np.zeros(n, dtype=np.int64))
                                         for c, n in extent.items()}
        return self.engine.einsum(spec, a, b, out, alpha, beta, labels=labels)


_DENSE = []


def einsum(spec, *operands, out=None, alpha=1.0, beta=0.0):
    """out = beta * out + alpha * einsum(spec, *operands) for NumPy arrays; two-operand
    products into a separate `out` run on the compiled executor."""
    if (out is not None and len(operands) == 2 and available()
            and not any(np.may_share_memory(out, x) for x in operands)):
        if not _DENSE:
            _DENSE.append(_DenseProducts())
        return _DENSE[0](spec, *operands, out, alpha, beta)
    return _numpy.einsum(spec, *operands, out=out, alpha=alpha, beta=beta)


def blocked_einsum(labels, combinations, nirrep):
    """A `CpuSymmetryBlocks` whose unsupported products fall back to `einsum` above."""
    return CpuSymmetryBlocks(labels, combinations, nirrep, fallback=einsum)


# Whole-array operations on all threads (core/src/cpu/array_ops.cpp), each with
# the arithmetic of the NumPy expression it replaces; NumPy when not applicable.

def _compiled(*arrays):
    return available() and all(isinstance(x, np.ndarray) and x.dtype == np.float64 for x in arrays)


def _elements(array):
    return np.asarray([stride // array.itemsize for stride in array.strides], dtype=np.int64)


def permuted_add(out, a, b, wa=1.0, wb=1.0):
    """out = wa * a + wb * b for a view `b` of out's shape (a transpose, say).
    `a` may be `out` itself, or None for out = wb * b; b must not overlap out."""
    usable = (_compiled(out, b) and out.flags.c_contiguous and b.shape == out.shape
              and 1 <= out.ndim <= 8 and all(stride % 8 == 0 for stride in b.strides)
              and (a is None or (_compiled(a) and a.flags.c_contiguous and a.shape == out.shape)))
    if not usable:
        out[...] = wb * b if a is None else wa * a + wb * b
        return out
    shape = np.asarray(out.shape, dtype=np.int64)
    strides = _elements(b)
    _cpu_kernels.library().permuted_add(out.ctypes.data, None if a is None else a.ctypes.data,
                                        b.ctypes.data, out.ndim, shape.ctypes.data,
                                        strides.ctypes.data, float(wa), float(wb), _threads(None))
    return out


def add_transpose_(x, perm):
    """x += x.transpose(perm) in place (pairwise, without a temporary, when perm is
    its own inverse)."""
    perm = tuple(int(p) for p in perm)
    if (_compiled(x) and x.flags.c_contiguous and 1 <= x.ndim <= 8
            and all(perm[p] == d for d, p in enumerate(perm))):
        shape = np.asarray(x.shape, dtype=np.int64)
        partner = _elements(x.transpose(perm))
        _cpu_kernels.library().symmetrize_pairs(x.ctypes.data, x.ndim, shape.ctypes.data,
                                                partner.ctypes.data, _threads(None))
    else:
        x += x.transpose(perm)
    return x


def add_transposed(a, perm):
    """a + a.transpose(perm) as a new array."""
    return permuted_add(np.empty(a.shape), a, a.transpose(perm))


def relayout(array, order):
    """The same array (same index order) stored with its axes `order` outermost
    first in memory."""
    stored = np.empty(tuple(array.shape[axis] for axis in order))
    permuted_add(stored, None, array.transpose(order), 0.0, 1.0)
    return stored.transpose(np.argsort(order))


def copy(x):
    """A new C-contiguous float64 copy of `x` (any strides, a transposed view
    say), read and written (and first touched) on all threads in one pass."""
    if not isinstance(x, np.ndarray) or x.dtype != np.float64:
        x = np.asarray(x, dtype=np.float64)
    return permuted_add(np.empty(x.shape), None, x, 0.0, 1.0)


def to_float32(x):
    """A C-contiguous float32 copy of float64 `x` (rounded on all threads)."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    out = np.empty(x.shape, dtype=np.float32)
    if _compiled(x):
        _cpu_kernels.library().to_f32(x.ctypes.data, out.ctypes.data, x.size, _threads(None))
    else:
        out[...] = x
    return out


def dots(x, others):
    """[x . y for y in others] + [x . x] for flat float64 arrays, on all threads
    (the same results for any thread count)."""
    kernels = _cpu_kernels.library()
    pointers = (ctypes.c_void_p * max(1, len(others)))(*[_pointer(y, np.float64) for y in others])
    out = np.zeros(len(others) + 1)
    kernels.dots(x.size, len(others), _pointer(x, np.float64), ctypes.cast(pointers, ctypes.c_void_p),
                 out.ctypes.data, _threads(None))
    return out


def combine(weights, vectors, out):
    """out = sum_j weights[j] vectors[j] for flat float64 arrays, on all threads."""
    kernels = _cpu_kernels.library()
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    pointers = (ctypes.c_void_p * len(vectors))(*[_pointer(v, np.float64) for v in vectors])
    kernels.combine(out.size, len(vectors), weights.ctypes.data, ctypes.cast(pointers, ctypes.c_void_p),
                    _pointer(out, np.float64), _threads(None))
    return out


def scale_(x, factor):
    """x *= factor."""
    if _compiled(x) and x.flags.c_contiguous:
        _cpu_kernels.library().scale(x.ctypes.data, x.size, float(factor), _threads(None))
    else:
        x *= factor
    return x


def add_(y, x):
    """y += x."""
    if _compiled(y, x) and y.flags.c_contiguous and x.flags.c_contiguous and x.shape == y.shape:
        _cpu_kernels.library().axpy(y.ctypes.data, x.ctypes.data, 1.0, y.size, _threads(None))
    else:
        y += x
    return y


def max_abs(x):
    """max |x| (NaN when x holds a NaN)."""
    if not x.size:
        return 0.0
    if _compiled(x) and x.flags.c_contiguous:
        return float(_cpu_kernels.library().max_abs(x.ctypes.data, x.size, _threads(None)))
    return max(float(x.max()), -float(x.min()))


def divide_triples(r3, eia):
    """r3 /= d3 for dense r3 (no, no, no, nv, nv, nv): (eia[i,a] + eia[j,b]) + eia[k,c]."""
    no, nv = eia.shape
    eia = np.ascontiguousarray(eia, dtype=np.float64)
    if _compiled(r3) and r3.flags.c_contiguous and r3.shape == (no,) * 3 + (nv,) * 3:
        _cpu_kernels.library().t3_divide(r3.ctypes.data, no, nv, eia.ctypes.data, _threads(None))
    else:
        d2 = eia[:, None, :, None] + eia[None, :, None, :]
        r3 /= d2[:, :, None, :, :, None] + eia[None, None, :, None, None, :]
    return r3


def project_triples(r3):
    """`_quadruples_cpu.project_triples` on all threads: symmetrize the occupied
    triples, spin-project and zero the triply repeated indices."""
    if not (_compiled(r3) and r3.flags.c_contiguous):
        return _numpy.project_triples(r3)
    source = permuted_add(np.empty(r3.shape), None, r3, 0.0, 1.0)
    for perm in _PERMS3[1:]:
        permuted_add(r3, r3, source.transpose(perm + tuple(i + 3 for i in perm)))
    del source
    nocc, nvir = r3.shape[0], r3.shape[3]
    for first, stop in prange(0, nocc, max(1, _slot_batch(nocc**2 * nvir**3))):
        part = r3[first:stop]
        spin_summation(part, part, "P3_full", -1.0 / 6.0, 1.0)
    occupied, virtual = np.arange(nocc), np.arange(nvir)
    r3[occupied, occupied, occupied, ...] = 0.0
    r3[..., virtual, virtual, virtual] = 0.0


def t3_spin_summation(a, b, slots, nvir, pattern, alpha=1.0, beta=0.0):
    spin_summation(a, b, pattern, alpha, beta)


def t4_spin_summation(source, target, nvir, pattern, alpha=1.0, beta=0.0):
    spin_summation(source, target, pattern, alpha, beta)


class CompiledQuadruplesLayout(NumpyQuadruplesLayout):
    """`NumpyQuadruplesLayout` with the compiled CPU kernels (see the module docstring)."""

    def __init__(self, *args, threads=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.threads = threads
        self._kernels = _cpu_kernels.library()
        if self._kernels is None:
            raise RuntimeError("CompiledQuadruplesLayout needs the compiled CPU core")
        virr, vloc, vsize, boff, soff, sirr, nirrep = self._tables
        arrays = dict(virr=np.ascontiguousarray(_host(virr), dtype=np.int32),
                      vloc=np.ascontiguousarray(_host(vloc), dtype=np.int32),
                      vsize=np.ascontiguousarray(_host(vsize), dtype=np.int32),
                      boff=np.ascontiguousarray(_host(boff), dtype=np.int64),
                      soff=np.ascontiguousarray(_host(soff), dtype=np.int64),
                      sirr=np.ascontiguousarray(_host(sirr), dtype=np.int32))
        self._struct = _core.structure(_core.QuadLayout, dict(nv=self.nvir, nirrep=int(nirrep)), arrays)
        self._layout = ctypes.byref(self._struct)
        self._invalid_flags = np.ascontiguousarray(_host(self._invalid), dtype=np.uint8)
        self._host_slots = np.ascontiguousarray(_host(self._slots), dtype=np.int32)

    def _n(self):
        return _threads(self.threads)

    # Gathers and scatters of occupied tiles.

    def _host_boxes(self, bounds, tile_shape, kind):
        """(rows, templates, boxes, per row, weights) of `itercc_quad_gather_boxes`
        (the host form of `QuadruplesLayout._boxes`)."""
        key = ("host boxes", bounds, tile_shape, kind)
        if key not in self._plans:
            templates, per_row, _ = self._templates(kind)
            plan = _host(self._plan(bounds, tile_shape, False)).astype(np.int64)
            slot = plan[:, 0]
            valid = slot >= 0
            index = {perm: i for i, perm in enumerate(_PERMS4)}
            perm_index = np.array([index[tuple(p)] if ok else 0
                                   for p, ok in zip(plan[:, 1:].tolist(), valid)], dtype=np.int64)
            safe = np.maximum(slot, 0)
            rows = np.empty((len(plan), 3), dtype=np.int64)
            rows[:, 0] = np.arange(len(plan)) * self.nv4
            rows[:, 1] = np.where(valid, self.offsets[safe], 0)
            rows[:, 2] = np.where(valid, perm_index * self.nirrep + self.slot_irreps[safe],
                                  len(_PERMS4) * self.nirrep)
            weights = np.ascontiguousarray([w for _, w in _gather_terms(kind)], dtype=np.float64)
            self._plans[key] = (rows, np.ascontiguousarray(_host(templates), dtype=np.int64),
                                len(plan) * per_row, per_row, weights)
        return self._plans[key]

    def gather(self, packed, tile, bounds, *, triples=False, spin=None, allowed_only=False):
        """Unpack the occupied tile `bounds` into dense `tile` (see QuadruplesLayout.gather).
        With `allowed_only`, the tile's symmetry zeros are not written (they keep
        whatever the tile held): for consumers that read allowed entries only."""
        if not tile.flags.c_contiguous:
            raise ValueError("gather needs a C-contiguous tile")
        kind = spin or ("triples" if triples else "plain")
        rows, templates, nbox, per_row, weights = self._host_boxes(bounds, tile.shape[:4], kind)
        self._kernels.quad_gather_boxes(
            _pointer(packed, np.float64), _pointer(tile, np.float64), rows.ctypes.data,
            templates.ctypes.data, nbox, per_row, len(weights), self.nvir,
            int(kind == "triples") | (2 if allowed_only else 0), weights.ctypes.data, self._n())

    def scatter(self, packed, tile, bounds, *, alpha, beta):
        plan = np.ascontiguousarray(self._plan(bounds, tile.shape[:4], True), dtype=np.int32)
        self._kernels.quad_scatter(self._layout, _pointer(tile, np.float64), plan.ctypes.data,
                                   len(plan), float(alpha), float(beta), _pointer(packed, np.float64),
                                   self._n())

    def add_rows(self, packed, rows, lists):
        """R4(o; v) += rows[row, v[map0], v[map1], v[map2], v[map3]] for every entry."""
        rows = np.ascontiguousarray(rows, dtype=np.float64)
        for entries in lists:
            entries = np.ascontiguousarray(entries, dtype=np.int32)
            self._kernels.quad_row_add(self._layout, rows.ctypes.data, entries.ctypes.data,
                                       len(entries), _pointer(packed, np.float64), self._n())

    # Dense slot batches.

    def expand(self, segment, first, stop):
        if not self.symmetric:
            return super().expand(segment, first, stop)
        dense = np.empty((stop - first,) + (self.nvir,) * 4)
        segment = np.ascontiguousarray(segment, dtype=np.float64)
        self._kernels.quad_expand(self._layout, segment.ctypes.data, first, stop - first,
                                  dense.ctypes.data, self._n())
        return dense

    def compress(self, dense, segment, first):
        if not self.symmetric:
            return super().compress(dense, segment, first)
        self._kernels.quad_compress(self._layout, _pointer(dense, np.float64), first, len(dense),
                                    _pointer(segment, np.float64), self._n())

    def project(self, residual):
        """Symmetrize occupied ties, spin-adapt and purify every slot, in place on
        the packed slots (`itercc_quad_project`: no dense rows, a quarter of the
        dense arithmetic, the same values entry by entry)."""
        if not (self.symmetric and isinstance(residual, np.ndarray) and residual.dtype == np.float64
                and residual.flags.c_contiguous and residual.shape == self.shape):
            return super().project(residual)
        if not hasattr(self, "_tie_masks"):
            masks = np.zeros(len(self.indices), dtype=np.uint8)
            for pair, tied in enumerate(self._host_ties):
                masks[tied] |= np.uint8(1 << pair)
            self._tie_masks = masks
        perms, weights, divisor = _spin_terms("project")
        self._kernels.quad_project(self._layout, residual.ctypes.data, 0, len(self.indices),
                                   self._tie_masks.ctypes.data, self._invalid_flags.ctypes.data,
                                   len(weights), perms.ctypes.data, weights.ctypes.data, 1.0, divisor,
                                   self._n())

    def _project_device(self, residual, ties, invalid):
        if not residual.flags.c_contiguous:
            return super()._project_device(residual, ties, invalid)
        for pair, tied in enumerate(ties):
            if tied.size:
                tied = np.ascontiguousarray(tied, dtype=np.int64)
                self._kernels.quad_tie(residual.ctypes.data, tied.ctypes.data, tied.size, self.nvir,
                                       pair, self._n())
        spin_summation(residual, residual, "project", threads=self.threads)
        flags = np.ascontiguousarray(_host(invalid), dtype=np.uint8)
        self._kernels.quad_purify(flags.ctypes.data, len(residual), self.nvir,
                                  _pointer(residual, np.float64), self._n())

    def divide(self, residual, eia):
        """residual /= D4 (sums of four `eia` entries), in one pass over the packed slots."""
        eia = np.ascontiguousarray(eia, dtype=np.float64)
        self._kernels.quad_divide(self._layout, _pointer(residual, np.float64), eia.ctypes.data,
                                  self._host_slots.ctypes.data, 0, len(self.indices), self._n())

    # Packed T4 blocks (no symmetry zeros) for the T4 passes.

    def _slot_sizes(self):
        """Packed length of a slot (virtual quadruples) for each slot irrep."""
        if not hasattr(self, "_packed_sizes"):
            sizes, _, _ = self._box_tables
            n = self.nirrep
            self._packed_sizes = np.array([
                sum(int(sizes[ga] * sizes[gb] * sizes[gc] * sizes[h ^ ga ^ gb ^ gc])
                    for ga in range(n) for gb in range(n) for gc in range(n)) for h in range(n)],
                dtype=np.int64)
        return self._packed_sizes

    def _pair_rows(self, p, q):
        """Packed rows (m, n) of the block T4[m, n, p, q]: grouped in classes of one
        row irrep, each a contiguous (rows x packed length) matrix."""
        key = ("pair rows", p, q)
        if key not in self._plans:
            occ = np.asarray(self.occupied_irreps, dtype=np.int64)
            nocc, sizes = self.nocc, self._slot_sizes()
            pair_irrep = occ[:, None] ^ occ[None, :]
            row_irrep = pair_irrep ^ occ[p] ^ occ[q]
            slots, orders, starts, classes = [], [], [], []
            offsets = np.zeros((nocc, nocc), dtype=np.int64)
            base = 0
            for h in range(self.nirrep):
                m, n = np.nonzero(row_irrep == h)
                if not len(m):
                    continue
                length = int(sizes[h])
                classes.append((int(pair_irrep[m[0], n[0]]), base, len(m), length))
                for j, (a, b) in enumerate(zip(m.tolist(), n.tolist())):
                    actual = (a, b, p, q)
                    slots.append(self._slot[tuple(sorted(actual))])
                    orders.append(sorted(range(4), key=actual.__getitem__))
                    offsets[a, b] = base + j * length
                    starts.append(offsets[a, b])
                base += len(m) * length
            rows = (np.ascontiguousarray(slots, dtype=np.int64),
                    np.ascontiguousarray(orders, dtype=np.int32),
                    np.ascontiguousarray(starts, dtype=np.int64))
            self._plans[key] = (rows, offsets, classes, base)
        return self._plans[key]

    def _pair_split_entries(self, p, q):
        """Entry lists (slot, row, map[4]) for the rows Y[(r, s)] (row r * nocc + s)
        of the spectator pair (p, q): one list per split (P, Q), over every
        r <= s with o[P] = (r, s), o[Q] = (p, q). A list writes each slot once
        (o determines (r, s)), and a slot takes one r per pair, so its adds
        come in split order as with one call per (r, split)."""
        key = ("pair splits", p, q)
        if key not in self._plans:
            nocc, lists = self.nocc, []
            for split in _SPLITS:
                entries = []
                for r in range(nocc):
                    for s in range(r, nocc):
                        o = [0] * 4
                        o[split[0]], o[split[1]], o[split[2]], o[split[3]] = r, s, p, q
                        if o == sorted(o):
                            entries.append((self._slot[tuple(o)], r * nocc + s, *split))
                if entries:
                    lists.append(np.ascontiguousarray(entries, dtype=np.int32))
            self._plans[key] = lists
        return self._plans[key]

    def add_oooo_pairs(self, r4, t4, W_oooo):
        """The six W_oooo terms of R4 (see `_quadruples_equations.r4_add_oooo_pairs_`)
        on packed blocks: for each spectator pair (p, q), the rows T4[m, n, p, q]
        are gathered without symmetry zeros, every class of one row irrep is a
        dense GEMM Y[(r, x)] = sum_mn W[m, n, r, x] T4[m, n, p, q] (W is totally
        symmetric, so a class maps onto itself), and the packed rows of Y are
        added to their slots."""
        from ._cpu_triples import cpu_threads

        threads = cpu_threads(None)
        nocc = self.nocc
        occ = np.asarray(self.occupied_irreps, dtype=np.int64)
        pair_irrep = occ[:, None] ^ occ[None, :]
        # W[(r, x), (m, n)] for the pairs of one pair irrep.
        W = np.asarray(W_oooo)
        weights = {}
        for g in range(self.nirrep):
            m, n = np.nonzero(pair_irrep == g)
            if len(m):
                weights[g] = np.ascontiguousarray(W[m[:, None], n[:, None], m[None, :], n[None, :]].T)
        t4 = np.ascontiguousarray(t4, dtype=np.float64)
        rows_buffer = products = None
        for p in range(nocc):
            for q in range(p, nocc):
                (slots, orders, offsets), table, classes, size = self._pair_rows(p, q)
                if rows_buffer is None or rows_buffer.size < size:
                    rows_buffer, products = np.empty(size), np.empty(size)
                self._kernels.quad_gather_rows(self._layout, t4.ctypes.data, len(slots), slots.ctypes.data,
                                               orders.ctypes.data, offsets.ctypes.data,
                                               rows_buffer.ctypes.data, self._n())
                for g, start, count, length in classes:
                    block = rows_buffer[start:start + count * length].reshape(count, length)
                    out = products[start:start + count * length].reshape(count, length)
                    einsum("xm,ms->xs", weights[g], block, out=out)
                # One call per split for all rows (r, s): enough boxes for
                # every thread also when few irreps (Cs) give few per row.
                at = np.ascontiguousarray(table.reshape(-1), dtype=np.int64)
                for entries in self._pair_split_entries(p, q):
                    self._kernels.quad_row_add_packed(self._layout, products.ctypes.data, at.ctypes.data,
                                                      entries.ctypes.data, len(entries),
                                                      _pointer(r4, np.float64), threads)

    def max_abs(self, packed):
        if not packed.size:
            return 0.0
        return float(self._kernels.max_abs(_pointer(np.ascontiguousarray(packed), np.float64),
                                           packed.size, self._n()))
