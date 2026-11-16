"""NumPy backend of the restricted CCSDTQ solver (`RestrictedCCSDTQ(backend="cpu")`).

`NumpyQuadruplesLayout` stores T4 and R4 exactly as `QuadruplesLayout` does
(same slots, offsets and in-slot order, so saved states and DIIS histories
carry over between backends) and replaces its CUDA kernels by vectorized
NumPy index maps: a slot segment expands to its dense (nv, nv, nv, nv) block
by one fancy-index copy per slot irrep, and a tile row whose occupied tuple
is unsorted is a transposed copy of its slot's block. The functions below
mirror `_quadruples_cuda.einsum`, the spin summations and `project_triples`
for NumPy arrays.

This is the correctness reference and the fallback without the compiled
core: products go through `np.einsum` and every gather, scatter and
projection makes dense copies. `_cpu_quadruples` replaces it with the
compiled kernels.
"""
from functools import lru_cache

import numpy as np

from ._quadruples_cuda import _PERMS3, _SPIN, QuadruplesLayout, _gather_terms, _slot_batch, prange


def einsum(spec, *operands, out=None, alpha=1.0, beta=0.0):
    """out = beta * out + alpha * einsum(spec, *operands) for NumPy arrays."""
    value = np.einsum(spec, *operands, optimize=True)
    if alpha != 1.0:
        # Some contractions return views of their inputs.
        value = value * alpha
    if out is None:
        return value
    if beta == 0.0:
        out[...] = value
    else:
        if beta != 1.0:
            out *= beta
        out += value
    return out


@lru_cache(maxsize=None)
def _axes(perm, lead):
    """Transpose axes with view[..., x] = source[..., x[perm[0]], x[perm[1]], ...]
    over the trailing len(perm) axes (the convention of `_SPIN` and `_ROW_ADD`)."""
    return tuple(range(lead)) + tuple(lead + int(axis) for axis in np.argsort(perm))


def _permuted_sum(source, target, terms, alpha=1.0, beta=0.0, divisor=1):
    """target = alpha * sum_t w_t source[..., x[perm_t]] / divisor + beta * target.

    `terms` holds (perm, w_t) pairs over the trailing axes; `source` may be `target`.
    """
    lead = source.ndim - len(terms[0][0])
    total = scratch = None
    for perm, weight in terms:
        view = source.transpose(_axes(tuple(perm), lead))
        if total is None:
            total = view * weight
        else:
            if scratch is None:
                scratch = np.empty_like(total)
            np.multiply(view, weight, out=scratch)
            total += scratch
    if alpha != 1.0:
        total *= alpha
    if divisor != 1:
        total /= divisor
    if beta == 0.0:
        target[...] = total
    else:
        if beta != 1.0:
            target *= beta
        target += total


def _spin_sum(source, target, pattern, alpha=1.0, beta=0.0):
    perms, weights, divisor = _SPIN[pattern]
    terms = [(perm, weight) for perm, weight in zip(perms, weights) if weight]
    _permuted_sum(source, target, terms, alpha, beta, divisor)


def t3_spin_summation(a, b, slots, nvir, pattern, alpha=1.0, beta=0.0):
    _spin_sum(a, b, pattern, alpha, beta)


def t4_spin_summation(source, target, nvir, pattern, alpha=1.0, beta=0.0):
    """target = alpha * P(source) + beta * target over the last four (virtual) axes."""
    if source.shape != target.shape:
        raise ValueError("t4_spin_summation needs arrays of one shape")
    _spin_sum(source, target, pattern, alpha, beta)


def add_transpose_(x, perm):
    """x += x.transpose(perm) in place."""
    x += x.transpose(perm)
    return x


def add_transposed(a, perm):
    """a + a.transpose(perm) as a new array."""
    return a + a.transpose(perm)


def relayout(array, order):
    """The same array stored with its axes `order` outermost first in memory."""
    return np.ascontiguousarray(array.transpose(order)).transpose([int(i) for i in np.argsort(order)])


def project_triples(r3):
    source = r3.copy()
    for perm in _PERMS3[1:]:
        r3 += source.transpose(perm + tuple(i + 3 for i in perm))
    del source
    nocc, nvir = r3.shape[0], r3.shape[3]
    for first, stop in prange(0, nocc, max(1, _slot_batch(nocc**2 * nvir**3))):
        part = r3[first:stop]
        _spin_sum(part, part, "P3_full", -1.0 / 6.0, 1.0)
    occupied, virtual = np.arange(nocc), np.arange(nvir)
    r3[occupied, occupied, occupied, ...] = 0.0
    r3[..., virtual, virtual, virtual] = 0.0


def _segment(source, window):
    """Host copy or view of the flattened elements `window` of a host or on-disk array."""
    if hasattr(source, "read"):
        return source.read(window.start, window.stop)
    return np.asarray(source).reshape(-1)[window]


def _update(array, key, values, beta):
    """array[key] = beta * array[key] + values for distinct keys (as the CUDA scatter)."""
    if beta == 0.0:
        array[key] = values
    elif beta == 1.0:
        array[key] += values
    else:
        array[key] = beta * array[key] + values


def _codes(perms):
    """One integer per row of (rows, 4) axis permutations."""
    return ((perms[:, 0] * 4 + perms[:, 1]) * 4 + perms[:, 2]) * 4 + perms[:, 3]


class NumpyQuadruplesLayout(QuadruplesLayout):
    """`QuadruplesLayout` whose packed arrays, tiles and index tables are NumPy arrays."""

    _array = staticmethod(np.asarray)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dense_indices = {}
        a, b, c, d = np.ix_(*(np.arange(self.nvir),) * 4)
        # Virtual quadruples that are zero for every closed-shell T4 (see _PURIFY).
        self._purified = (a == b) & ((b == c) | (b == d)) | (c == d) & ((a == c) | (b == c))

    def zeros(self, *, host=True):
        return np.zeros(self.size)

    # Index maps between packed slot segments and dense (nv, nv, nv, nv) rows.

    def _dense_index(self, irrep):
        """Row-major position (a, b, c, d) of every entry of a slot segment of `irrep`."""
        if irrep not in self._dense_indices:
            sizes, first, blocks = self._box_tables
            nirrep, labels = self.nirrep, self.virtual_irreps
            local = np.arange(self.nvir) - first[labels]
            a, b, c, d = np.ix_(*(np.arange(self.nvir),) * 4)
            ga, gb, gc, gd = labels[a], labels[b], labels[c], labels[d]
            offset = (blocks[((irrep * nirrep + ga) * nirrep + gb) * nirrep + gc]
                      + ((local[a] * sizes[gb] + local[b]) * sizes[gc] + local[c]) * sizes[gd] + local[d])
            allowed = (ga ^ gb ^ gc ^ gd) == irrep
            index = np.empty(int(allowed.sum()), dtype=np.intp)
            index[offset[allowed]] = np.flatnonzero(allowed)
            self._dense_indices[irrep] = index
        return self._dense_indices[irrep]

    def _groups(self, slots, first):
        """Per slot irrep: the rows of `slots` with that irrep, their packed positions
        (relative to the start of slot `first`) and the dense index of each entry."""
        irreps = self.slot_irreps[slots]
        base = self.offsets[slots] - self.offsets[first]
        for irrep in np.unique(irreps):
            chosen = np.flatnonzero(irreps == irrep)
            index = self._dense_index(int(irrep))
            yield chosen, base[chosen, None] + np.arange(index.size), index

    def _rows(self, packed, slots, first=0):
        """Dense (len(slots), nv**4) copies of `slots`; packed[0] starts slot `first`."""
        if not self.symmetric:
            return packed.reshape(-1, self.nv4)[slots - first]
        rows = np.zeros((len(slots), self.nv4))
        for chosen, positions, index in self._groups(slots, first):
            rows[chosen[:, None], index] = packed[positions]
        return rows

    def _store(self, packed, slots, rows, *, alpha=1.0, beta=0.0, first=0):
        """packed = beta * packed + alpha * rows on the allowed entries of distinct `slots`."""
        if not packed.flags.c_contiguous:
            raise ValueError("Packed quadruples must be C-contiguous")
        if not self.symmetric:
            _update(packed.reshape(-1, self.nv4), slots - first,
                    rows if alpha == 1.0 else alpha * rows, beta)
            return
        for chosen, positions, index in self._groups(slots, first):
            values = rows[chosen[:, None], index]
            if alpha != 1.0:
                values *= alpha
            _update(packed, positions, values, beta)

    # The QuadruplesLayout primitives.

    def gather(self, packed, tile, bounds, *, triples=False, spin=None, allowed_only=False):
        """Unpack the occupied tile `bounds` into dense `tile` (see QuadruplesLayout.gather)."""
        if not tile.flags.c_contiguous:
            raise ValueError("gather needs a C-contiguous tile")
        kind = spin or ("triples" if triples else "plain")
        plain = tile if kind == "plain" else np.empty_like(tile)
        rows = plain.reshape((-1,) + (self.nvir,) * 4)
        plan = self._plan(bounds, tile.shape[:4], False)
        slots = plan[:, 0].astype(np.intp)
        valid = np.flatnonzero(slots >= 0)
        if valid.size < len(rows):
            rows[slots < 0] = 0.0
        if valid.size:
            dense = self._rows(packed, slots[valid]).reshape((-1,) + (self.nvir,) * 4)
            orders = plan[valid, 1:]
            codes = _codes(orders)
            # Packed position k holds tile axis order[k].
            for code in np.unique(codes):
                chosen = np.flatnonzero(codes == code)
                order = tuple(int(axis) for axis in orders[chosen[0]])
                rows[valid[chosen]] = dense[chosen].transpose(_axes(order, 1))
            del dense
        if kind != "plain":
            _permuted_sum(plain, tile, _gather_terms(kind))

    def scatter(self, packed, tile, bounds, *, alpha, beta):
        plan = self._plan(bounds, tile.shape[:4], True)
        valid = np.flatnonzero(plan[:, 0] >= 0)
        if valid.size:
            rows = tile.reshape(-1, self.nv4)[valid]
            self._store(packed, plan[valid, 0].astype(np.intp), rows, alpha=alpha, beta=beta)

    def add_rows(self, packed, rows, lists):
        """R4(o; v) += rows[row, v[map0], v[map1], v[map2], v[map3]] for every entry."""
        for entries in lists:
            maps = entries[:, 2:]
            codes = _codes(maps)
            for code in np.unique(codes):
                chosen = np.flatnonzero(codes == code)
                perm = tuple(int(axis) for axis in maps[chosen[0]])
                values = rows[entries[chosen, 1]].transpose(_axes(perm, 1)).reshape(len(chosen), -1)
                self._store(packed, entries[chosen, 0].astype(np.intp), values, beta=1.0)

    def expand(self, segment, first, stop):
        """Dense (stop-first, nv, nv, nv, nv) copy (a view without symmetry) of a slot range."""
        shape = (stop - first,) + (self.nvir,) * 4
        if not self.symmetric:
            return segment.reshape(shape)
        return self._rows(segment, np.arange(first, stop), first).reshape(shape)

    def compress(self, dense, segment, first):
        """Write the allowed entries of a dense slot range into its segment."""
        if not self.symmetric:
            if not (dense.ctypes.data == segment.ctypes.data and dense.size == segment.size):
                segment[...] = dense.reshape(-1)
            return
        rows = dense.reshape(len(dense), -1)
        self._store(segment, np.arange(first, first + len(rows)), rows, first=first)

    def project(self, residual):
        """Symmetrize occupied ties, spin-adapt, and purify every slot."""
        for first, stop in self.batches():
            segment = residual[self.segment(first, stop)]
            dense = self.expand(segment, first, stop)
            ties = tuple(tied[(tied >= first) & (tied < stop)] - first for tied in self._host_ties)
            self._project_device(dense, ties, self._invalid[first:stop])
            self.compress(dense, segment, first)

    def _project_device(self, residual, ties, invalid):
        for axis, tied in enumerate(ties, start=1):
            if tied.size:
                original = residual[tied]
                residual[tied] = 0.5 * (original + original.swapaxes(axis, axis + 1))
        for start, stop in prange(0, len(residual), _slot_batch(self.nv4)):
            part = residual[start:stop]
            _spin_sum(part, part, "project")
        residual[invalid] = 0.0
        residual[:, self._purified] = 0.0

    def divide(self, residual, eia):
        """residual /= D4 (sums of four `eia` entries) slot by slot."""
        eia = np.asarray(eia)
        for first, stop in self.batches():
            segment = residual[self.segment(first, stop)]
            dense = self.expand(segment, first, stop)
            o = self._slots[first:stop].T
            dense /= (eia[o[0]][:, :, None, None, None] + eia[o[1]][:, None, :, None, None]
                      + eia[o[2]][:, None, None, :, None] + eia[o[3]][:, None, None, None, :])
            self.compress(dense, segment, first)

    def update_from_host(self, *args, **kwargs):
        raise ValueError("Host-staged R4 belongs to the CUDA backend")

    def from_dense(self, source, *, order=None, out=None):
        """Compress dense packed slots, optionally reordering virtual axes.

        `order[k]` is the source virtual index stored at internal position k.
        Returns the packed array and the largest discarded magnitude.
        """
        if source.shape != self.dense_shape:
            raise ValueError(f"Expected dense packed quadruples with shape {self.dense_shape}")
        out = np.empty(self.size) if out is None else out
        leakage = 0.0
        for first, stop in self.batches():
            dense = np.array(source[first:stop], dtype=np.float64)
            if order is not None:
                for axis in range(1, 5):
                    dense = np.take(dense, order, axis=axis)
            segment = out[self.segment(first, stop)]
            self.compress(dense, segment, first)
            if self.symmetric:
                kept = self.expand(segment, first, stop)
                leakage = max(leakage, float(np.abs(dense - kept).max()))
        return out, leakage

    def to_dense(self, packed, *, order=None):
        """Dense packed copy; `order` as in from_dense restores source order."""
        result = np.empty(self.dense_shape)
        inverse = None if order is None else np.argsort(order)
        for first, stop in self.batches():
            dense = self.expand(np.asarray(packed[self.segment(first, stop)]), first, stop)
            if inverse is not None:
                for axis in range(1, 5):
                    dense = np.take(dense, inverse, axis=axis)
            result[first:stop] = dense
        return result

    def accumulate_from(self, out, weight, source, source_layout, perm=None):
        """out += weight * `source` re-laid into this layout (see QuadruplesLayout)."""
        if source_layout.dense_shape != self.dense_shape:
            raise ValueError("The source quadruples use another occupied or virtual space")
        if (perm is None
                and np.array_equal(self.occupied_irreps, source_layout.occupied_irreps)
                and np.array_equal(self.virtual_irreps, source_layout.virtual_irreps)):
            for first, stop in self.batches():
                window = self.segment(first, stop)
                out[window] += weight * _segment(source, window)
            return
        for first, stop in self.batches():
            dense = source_layout.expand(_segment(source, source_layout.segment(first, stop)),
                                         first, stop)
            if perm is not None:
                for axis in range(1, 5):
                    dense = np.take(dense, perm, axis=axis)
            target = out[self.segment(first, stop)]
            part = np.empty_like(target)
            self.compress(dense, part, first)
            target += weight * part

    def max_abs(self, packed):
        return max(float(packed.max()), -float(packed.min())) if packed.size else 0.0
