"""Exact column compression for CUDA particle-hole triples products."""

from functools import cache

import numpy as np

from ._pair_symmetry import PairSectors


@cache
def _pack_kernel():
    from cupy import ElementwiseKernel

    return ElementwiseKernel(
        "raw float64 plane, raw int64 rows, raw int64 columns, int32 no, int32 nv, "
        "int32 j, int32 width, bool spin", "float64 result",
        r"""
        const int row = rows[i / width];
        const int col = columns[i % width];
        const int occ = row / nv;
        const int a = row % nv;
        const int b = col / nv;
        const int c = col % nv;
        const int plane_row = spin ? occ * no + j : j * no + occ;
        const long long base = static_cast<long long>(plane_row) * nv * nv * nv;
        const double abc = plane[base + (a * nv + b) * nv + c];
        if (spin) {
            const double bac = plane[base + (b * nv + a) * nv + c];
            const double cba = plane[base + (c * nv + b) * nv + a];
            result = (2.0 * abc - bac) - cba;
        } else {
            result = abc;
        }
        """, "itercc_pack_ph_line")


class CudaParticleHole:
    """Map row sectors to virtual-pair columns for a fixed occupied line."""

    def __init__(self, occupied, virtual, xp):
        self.xp = xp
        rows = (occupied[:, None] ^ virtual).ravel()
        pairs = virtual[:, None] ^ virtual
        self.sectors = PairSectors(rows)
        self.order = xp.asarray(self.sectors.order, dtype=xp.int64)
        self.columns = tuple(xp.asarray(np.flatnonzero(pairs.ravel() == label), dtype=xp.int64)
                             for label in range(8))
        # For a fixed (j, k), a row (i, a) can contribute to a column
        # (b, c) only if their XOR labels differ by the occupied (j, k) label.
        self.shifts = occupied[:, None] ^ occupied

    def product(self, matrix, *, layout=None, spin=False):
        args = matrix, self.columns, self.order
        if layout is None:
            return self.sectors.column_product(*args, xp=self.xp)
        no, nv = layout.no, layout.nv
        shape = no * nv, nv**2

        def pack(plane, rows, columns, j):
            return _pack_kernel()(plane, rows, columns, no, nv, j, len(columns), spin,
                                  size=len(rows) * len(columns)).reshape(len(rows), len(columns))

        def dense_rhs(plane, j):
            values = layout.spin_sum(plane[:, j]) if spin else plane[j]
            return values.reshape(shape)

        return self.sectors.column_product(*args, xp=self.xp, pack=pack,
                                           dense_rhs=dense_rhs, output_shape=shape)