"""Symmetry-packed CUDA storage for EOM/response triples directions and images.

A projected Jacobian action only reads a direction R3 and only returns the
image components inside one target irrep. Storing both in the packed
coordinates of `SingletSpace` (instead of dense occupied-triangular T3) removes
two full T3-sized device arrays per action. The packed order is exactly
`SingletSpace.indices[2]`: slots in layout order, allowed (a, b, c) ascending.

Every permutation of (a, b, c) keeps the virtual label, so symmetrization and
purification stay inside the packed set. Ground-state amplitudes are never
packed: they keep their dense support, including small off-sector leakage.
"""
from functools import cache

import numpy as np


_LOOKUP = r"""
template<class S, class T, class O>
__device__ double packed_value(const S& source, const T& table, const O& offsets,
                               long long row, long long base, int x) {
    const int rank = table[row + x];
    return rank < 0 ? 0.0 : source[base + rank];
}
"""


@cache
def _kernels():
    from cupy import ElementwiseKernel

    plane = ElementwiseKernel(
        "raw float64 source, raw int32 plan, raw int32 table, raw int32 slot_table, "
        "raw int64 offsets, int32 nv, int32 nv3, bool spin", "float64 result",
        r"""
        const int row = (i / nv3) * 4;
        const int slot = plan[row];
        if (slot < 0) {
            result = 0;
        } else {
            const int rem = i % nv3;
            const int a = rem / (nv * nv), b = (rem / nv) % nv, c = rem % nv;
            const int sa = plan[row + 1], sb = plan[row + 2], sc = plan[row + 3];
            const long long trow = static_cast<long long>(slot_table[slot]) * nv3;
            const long long base = offsets[slot];
            const double abc = packed_value(source, table, offsets, trow, base, a * sa + b * sb + c * sc);
            if (spin) {
                const double bac = packed_value(source, table, offsets, trow, base, b * sa + a * sb + c * sc);
                const double cba = packed_value(source, table, offsets, trow, base, c * sa + b * sb + a * sc);
                result = (2.0 * abc - bac) - cba;
            } else {
                result = abc;
            }
        }
        """, "itercc_packed_triples_plane", preamble=_LOOKUP)
    symmetrize = ElementwiseKernel(
        "raw float64 source, raw P positions, raw int32 table, raw int32 slot_table, "
        "raw int64 offsets, raw int8 repeats, int64 first, int32 nv, int32 nv3, int32 mode",
        "float64 result",
        r"""
        const long long p = first + i;
        const long long position = positions[p];
        const int slot = position / nv3;
        const int rem = position % nv3;
        const int a = rem / (nv * nv), b = (rem / nv) % nv, c = rem % nv;
        const long long trow = static_cast<long long>(slot_table[slot]) * nv3;
        const long long base = offsets[slot];
        const int nv2 = nv * nv;
        const double abc = source[p];
        const double bac = packed_value(source, table, offsets, trow, base, b * nv2 + a * nv + c);
        const double cba = packed_value(source, table, offsets, trow, base, c * nv2 + b * nv + a);
        const double acb = packed_value(source, table, offsets, trow, base, a * nv2 + c * nv + b);
        const double cab = packed_value(source, table, offsets, trow, base, c * nv2 + a * nv + b);
        const double bca = packed_value(source, table, offsets, trow, base, b * nv2 + c * nv + a);
        if (mode == 0) {
            // EOM direction: repeated occupied pairing with the null removal,
            // identical arithmetic to _cuda_layout.project_eom_triples.
            const int repeat = repeats[slot];
            double paired = abc;
            if (repeat == 1) paired = (abc + bac) * 0.5;
            if (repeat == 2) paired = (abc + acb) * 0.5;
            // Same summation order as the dense kernel's transposed operands.
            const double symmetric = ((abc + bac) + cba) + ((acb + bca) + cab);
            result = a == b && b == c ? 0.0 : paired - symmetric / 6.0;
        } else {
            // Residual purification, identical arithmetic to CudaTriplesLayout.purify.
            const double left = (abc + bac) + cba;
            const double right = (acb + cab) + bca;
            result = a == b && b == c ? 0.0 : abc - (left + right) / 6.0;
        }
        """, "itercc_packed_triples_symmetrize", options=("--fmad=false",), preamble=_LOOKUP)
    return plane, symmetrize


@cache
def _ladder_kernels():
    from cupy import ElementwiseKernel

    pack = ElementwiseKernel(
        "raw float64 source, raw int32 pairs, raw int32 columns, int32 nv, "
        "int32 width, int32 first, int32 sa, int32 sb, int32 sc, int32 parity, "
        "raw int32 table, raw int32 slot_table, raw int64 offsets, int32 slot0, int32 nv3",
        "float64 value",
        r"""
        const int row = first + i / width;
        const int col = columns[i % width];
        const int a = pairs[2 * row];
        const int b = pairs[2 * row + 1];
        const int slot = slot0 + col / nv;
        const int spectator = (col % nv) * sc;
        const long long trow = static_cast<long long>(slot_table[slot]) * nv3;
        const long long base = offsets[slot];
        const double ab = packed_value(source, table, offsets, trow, base, a * sa + b * sb + spectator);
        const double ba = packed_value(source, table, offsets, trow, base, b * sa + a * sb + spectator);
        value = 0.5 * (parity < 0 ? ab - ba : ab + ba);
        """, "itercc_pack_packed_ladder_columns", preamble=_LOOKUP)
    accumulate = ElementwiseKernel(
        "raw float64 even, raw float64 odd, raw int32 even_rows, raw int32 odd_rows, "
        "raw int64 even_base, raw int64 odd_base, raw int32 ranks, raw P positions, "
        "int64 first, int32 slot0, int32 nv, int32 nv2, int32 nv3, int32 orientation",
        "raw float64 target",
        r"""
        const long long p = first + i;
        const long long position = positions[p];
        const int rem = position % nv3;
        const int v0 = rem / nv2, v1 = (rem / nv) % nv, v2 = rem % nv;
        const int a = orientation == 2 ? v1 : v0;
        const int b = orientation == 0 ? v1 : v2;
        const int c = orientation == 0 ? v2 : (orientation == 1 ? v1 : v0);
        const int pair = a * nv + b;
        const int col = (static_cast<int>(position / nv3) - slot0) * nv + c;
        // Every packed entry is in the target irrep, so its pair and column
        // labels always match the dense kernel's selection test.
        const int rank = ranks[col];
        double value = even[even_base[even_rows[pair]] + rank];
        if (a != b) {
            const double exchange = odd[odd_base[odd_rows[pair]] + rank];
            value = a < b ? value + exchange : value - exchange;
        }
        target[p] = target[p] + value;
        """, "itercc_add_packed_ladder_columns")
    return pack, accumulate


@cache
def _fold_kernel():
    from cupy import ElementwiseKernel

    return ElementwiseKernel(
        "raw float64 seed, raw int32 fold, raw float64 values, raw int64 bases, "
        "raw uint8 row_irreps, raw uint8 column_irreps, raw int32 ranks, "
        "int32 no, int32 nv, int32 width, int32 j, uint8 shift, "
        "raw int32 table, raw int32 slot_table, raw int64 offsets", "raw float64 target",
        r"""
        const int iocc = i / width;
        const int entry = iocc * 5;
        const int slot = fold[entry];
        if (slot >= 0) {
            const int rem = i % width;
            const int packed = table[static_cast<long long>(slot_table[slot]) * width + rem];
            if (packed >= 0) {
                const int a = rem / (nv * nv);
                const int b = (rem / nv) % nv;
                const int c = rem % nv;
                const int sa = fold[entry + 1], sb = fold[entry + 2], sc = fold[entry + 3];
                const int row = iocc * no + j;
                const int label = row_irreps[row] ^ shift;
                const long long seed_base = static_cast<long long>(iocc) * width;
                const long long index = offsets[slot] + packed;
                int col = a * sa + b * sb + c * sc;
                double term = seed[seed_base + col];
                if (label == column_irreps[col]) term += values[bases[row] + ranks[col]];
                double updated = target[index] + term;
                const int swap = fold[entry + 4];
                if (swap != 0) {
                    col = swap == 1 ? b * sa + a * sb + c * sc : a * sa + c * sb + b * sc;
                    term = seed[seed_base + col];
                    if (label == column_irreps[col]) term += values[bases[row] + ranks[col]];
                    updated += term;
                }
                target[index] = updated;
            }
        }
        """, "itercc_fold_occupied_seed_packed")


class PackedTriplesIndex:
    """Packed-coordinate tables for one SingletSpace (one target irrep)."""

    def __init__(self, layout, labels, irrep, positions, xp):
        self.xp, self.layout, self.irrep = xp, layout, int(irrep)
        no, nv = layout.no, layout.nv
        self.no, self.nv, self.nv3 = no, nv, nv**3
        occ, vir = np.asarray(labels[:no]), np.asarray(labels[no:])
        virtual = (vir[:, None, None] ^ vir[None, :, None] ^ vir).ravel()
        slots = np.asarray(layout.indices, dtype=np.int64).reshape(-1, 3)
        slot_labels = (np.bitwise_xor.reduce(occ[slots], axis=1) if len(slots)
                       else np.zeros(0, dtype=np.int64))
        needed = sorted({int(label) ^ self.irrep for label in slot_labels})
        table = np.full((max(len(needed), 1), nv**3), -1, dtype=np.int32)
        counts = {}
        for row, label in enumerate(needed):
            selected = np.flatnonzero(virtual == label)
            table[row, selected] = np.arange(len(selected), dtype=np.int32)
            counts[label] = len(selected)
        slot_table = np.array([needed.index(int(label) ^ self.irrep) for label in slot_labels],
                              dtype=np.int32)
        sizes = np.array([counts[int(label) ^ self.irrep] for label in slot_labels], dtype=np.int64)
        offsets = np.zeros(len(slots) + 1, dtype=np.int64)
        np.cumsum(sizes, out=offsets[1:])
        self.offsets_host = offsets
        self.size = int(offsets[-1])
        if positions.size != self.size:
            raise ValueError("Packed triples tables disagree with the singlet space")
        self.table = xp.asarray(table)
        self.slot_table = xp.asarray(slot_table)
        self.offsets = xp.asarray(offsets)
        self.positions = positions
        self.repeats = xp.asarray([1 if i == j else 2 if j == k else 0
                                   for i, j, k in layout.indices], dtype=xp.int8)

    def symmetrize(self, source, mode, *, chunk=1 << 24):
        """mode 0: EOM direction projection; mode 1: residual purification.

        The output is written chunkwise; every read stays in the input, so
        the caller may pass the same array to overwrite in place only when
        `source` is not reused, which this method does by writing back in
        slot-aligned chunks after their reads complete.
        """
        _, symmetrize = _kernels()
        output = self.xp.empty_like(source) if mode == 0 else source
        offsets = self.offsets_host
        start = 0
        while start < len(offsets) - 1:
            stop = int(np.searchsorted(offsets, offsets[start] + chunk, side="right")) - 1
            stop = min(max(stop, start + 1), len(offsets) - 1)
            first, last = int(offsets[start]), int(offsets[stop])
            if last > first:
                values = symmetrize(source, self.positions, self.table, self.slot_table,
                                    self.offsets, self.repeats, first, self.nv, self.nv3, mode,
                                    size=last - first)
                output[first:last] = values
                del values
            start = stop
        return output


class PackedTriples:
    """Packed triples values with dense-like slot length for batching."""

    def __init__(self, index, data):
        if data.shape != (index.size,):
            raise ValueError("Packed triples data has the wrong size")
        self.index, self.data = index, data

    def __len__(self):
        return len(self.index.offsets_host) - 1

    @property
    def irrep(self):
        return self.index.irrep

    def zeros_like(self):
        return PackedTriples(self.index, self.index.xp.zeros(self.index.size))

    def slots(self, start, stop):
        return _PackedSlots(self, start, min(stop, len(self)))

    def plane(self, plan, spin):
        plane, _ = _kernels()
        index = self.index
        no, nv = index.no, index.nv
        value = plane(self.data, plan, index.table, index.slot_table, index.offsets,
                      nv, index.nv3, spin, size=no * no * index.nv3)
        return value.reshape(no, no, nv, nv, nv)

    def purify(self):
        self.index.symmetrize(self.data, 1)

    def fold_occupied(self, seed, plan, operands, no, nv, width, j, shift):
        index = self.index
        _fold_kernel()(seed, plan, *operands, no, nv, width, j, shift,
                       index.table, index.slot_table, index.offsets, self.data, size=seed.size)


class _PackedSlots:
    """A contiguous slot range of packed triples (ladder batch)."""

    def __init__(self, owner, start, stop):
        self.owner, self.start, self.stop = owner, start, stop

    def __len__(self):
        return self.stop - self.start
