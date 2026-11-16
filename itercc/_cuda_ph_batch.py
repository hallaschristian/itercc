"""Batch occupied lines in compact particle-hole matrix products."""

from functools import cache

import numpy as np

from ._cuda_ph_fused import CudaParticleHoleFusion


@cache
def _kernels():
    from cupy import ElementwiseKernel

    pack = ElementwiseKernel(
        "raw float64 plane, raw int64 rows, raw int64 columns, raw int64 lines, "
        "int32 no, int32 nv, int32 width, int32 count, bool spin", "float64 out",
        """
        const int row = rows[i / (width * count)];
        const int col = columns[i % width];
        const int j = lines[(i / width) % count];
        const int occ = row / nv, a = row % nv, b = col / nv, c = col % nv;
        const int plane_row = spin ? occ * no + j : j * no + occ;
        const long long base = static_cast<long long>(plane_row) * nv * nv * nv;
        const double abc = plane[base + (a * nv + b) * nv + c];
        out = spin ? (2.0 * abc - plane[base + (b * nv + a) * nv + c])
                     - plane[base + (c * nv + b) * nv + a] : abc;
        """, "itercc_pack_ph_batch")
    add = ElementwiseKernel(
        "float64 seed, raw float64 direct, raw float64 crossed, raw int64 bases, "
        "raw int32 widths, raw uint8 rows, raw uint8 columns, raw int32 ranks, "
        "int32 nv, uint8 shift, int32 line", "float64 out",
        """
        const int width = nv * nv, row = i / width, col = i % width;
        double d = 0.0, x = 0.0, swapped = 0.0;
        if ((rows[row] ^ shift) == columns[col]) {
            const long long entry = bases[row] + line * widths[row] + ranks[col];
            d = direct[entry]; x = crossed[entry];
        }
        const int a = row % nv, b = col / nv, c = col % nv;
        const int other_row = (row / nv) * nv + b, other_col = a * nv + c;
        if ((rows[other_row] ^ shift) == columns[other_col])
            swapped = crossed[bases[other_row] + line * widths[other_row] + ranks[other_col]];
        out = ((seed + d) - swapped) - 0.5 * x;
        """, "itercc_add_ph_batch", options=("--fmad=false",))
    return pack, add


class CudaParticleHoleBatch(CudaParticleHoleFusion):
    def __init__(self, occupied, virtual, ph, xp):
        super().__init__(occupied, virtual, ph, xp)
        self.groups = tuple(tuple(np.flatnonzero(ph.shifts[:, k] == shift)
                                  for shift in np.unique(ph.shifts[:, k]))
                            for k in range(len(occupied)))
        self.widths = {}
        host_rows = (occupied[:, None] ^ virtual).ravel()
        for shift in self.plans:
            widths = [len(ph.columns[int(label) ^ shift]) for label in host_rows]
            self.widths[shift] = xp.asarray(widths, dtype=xp.int32)

    def prepare(self, direct, crossed):
        ph, xp = self.ph, self.xp
        direct_blocks = ph.sectors._blocks(direct, xp, False, ph.order)
        crossed_blocks = ph.sectors._blocks(crossed, xp, False, ph.order)
        if direct_blocks is None or crossed_blocks is None:
            return None
        return _Actions(self, direct_blocks, crossed_blocks)


class _Actions:
    def __init__(self, owner, direct, crossed):
        self.owner, self.direct, self.crossed = owner, direct, crossed
        self.lines = {}

    def prepare_plane(self, plane, k):
        owner, xp = self.owner, self.owner.xp
        self.lines.clear()
        pack, _ = _kernels()
        for indices in owner.groups[k]:
            count = len(indices)
            shift = int(owner.ph.shifts[indices[0], k])
            plan = owner.plans[shift]
            direct, crossed = (xp.empty(plan.size * count) for _ in range(2))
            scratch = xp.empty(plan.scratch * count)
            dev_indices = xp.asarray(indices, dtype=xp.int64)
            for spin, blocks, target in ((True, self.direct, direct), (False, self.crossed, crossed)):
                for matrix, band in zip(blocks, plan.bands):
                    width = len(band.columns)
                    if width:
                        shape = len(band.rows), count * width
                        source = scratch[:shape[0] * shape[1]].reshape(shape)
                        pack(plane, band.rows, band.columns, dev_indices, owner.no, owner.nv,
                             width, count, spin, source)
                        start, stop = band.output.start * count, band.output.stop * count
                        xp.matmul(matrix, source, out=target[start:stop].reshape(shape))
            bases = plan.bases * count
            for rank, j in enumerate(indices):
                self.lines[int(j)] = direct, crossed, bases, owner.widths[shift], shift, rank

    def __call__(self, seed, plane, j, k):
        owner = self.owner
        direct, crossed, bases, widths, shift, rank = self.lines[j]
        _, add = _kernels()
        add(seed, direct, crossed, bases, widths, owner.rows, owner.columns, owner.ranks,
            owner.nv, shift, rank, seed)

    def release_plane(self):
        self.lines.clear()
