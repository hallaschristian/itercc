"""Particle-hole tangent products restricted to the requested output irrep."""

from functools import cache

import numpy as np

from ._cuda_ph import CudaParticleHole
from ._cuda_ph_batch import CudaParticleHoleBatch, _kernels


@cache
def _fold_kernel():
    from cupy import ElementwiseKernel

    return ElementwiseKernel(
        "raw float64 seed, raw int32 plan, raw float64 direct, raw float64 crossed, "
        "raw int64 bases, raw int32 widths, raw uint8 rows, raw uint8 columns, "
        "raw int32 ranks, int32 nv, int32 nv3, uint8 shift, int32 rank, "
        "int32 width, int32 line", "raw float64 target",
        """
        const int occ = i / nv3;
        const int fold = 5 * occ;
        const int slot = plan[fold];
        if (slot >= 0) {
            const int nv2 = nv * nv;
            const int rem = i % nv3;
            const int a = rem / nv2, b = (rem / nv) % nv, c = rem % nv;
            const int sa = plan[fold + 1], sb = plan[fold + 2], sc = plan[fold + 3];
            const int local = a * sa + b * sb + c * sc;
            const int row = occ * nv + local / nv2, col = local % nv2;
            if ((rows[row] ^ shift) == columns[col]) {
                const long long seed_base = (static_cast<long long>(occ) * width + line) * nv3;
                const long long index = static_cast<long long>(slot) * nv3 + rem;
                double value = target[index] + projected_seed(
                    seed, direct, crossed, bases, widths, ranks, nv, nv2, rank, seed_base, occ, local);
                const int swap = plan[fold + 4];
                if (swap) {
                    const int other = swap == 1 ? b * sa + a * sb + c * sc
                                                : a * sa + c * sb + b * sc;
                    value += projected_seed(seed, direct, crossed, bases, widths, ranks,
                                            nv, nv2, rank, seed_base, occ, other);
                }
                target[index] = value;
            }
        }
        """, "itercc_fold_projected_seed", options=("--fmad=false",), preamble=r"""
        template<class S, class D, class X, class B, class W, class R>
        __device__ double projected_seed(
            const S& seed, const D& direct, const X& crossed,
            const B& bases, const W& widths, const R& ranks,
            int nv, int nv2, int rank, long long seed_base, int occ, int local) {
            const int a = local / nv2, b = (local / nv) % nv, c = local % nv;
            const int row = occ * nv + a, col = b * nv + c;
            const int other_row = occ * nv + b, other_col = a * nv + c;
            const long long entry = bases[row] + rank * widths[row] + ranks[col];
            const long long swapped = bases[other_row] + rank * widths[other_row] + ranks[other_col];
            return ((seed[seed_base + local] + direct[entry]) - crossed[swapped]) - 0.5 * crossed[entry];
        }
        """)


class CudaProjectedParticleHole(CudaParticleHoleBatch):
    def __init__(self, occupied, virtual, xp):
        ph = CudaParticleHole(occupied, virtual, xp)
        super().__init__(occupied, virtual, ph, xp)
        self.all_rows = xp.arange(len(self.rows), dtype=xp.int64)
        self.host_rows = (occupied[:, None] ^ virtual).ravel()

    def prepare_tangent(self, direct, crossed, ddirect, dcrossed, irrep):
        xp = self.xp
        for shift in np.unique(self.ph.shifts) ^ irrep:
            if shift not in self.plans:
                self.plans[shift] = self._plan(int(shift))
                widths = [len(self.ph.columns[int(label) ^ shift]) for label in self.host_rows]
                self.widths[shift] = xp.asarray(widths, dtype=xp.int32)
        bands = tuple(self.ph.order[band] for band in self.ph.sectors.bands)

        def blocks(primal, tangent):
            return tuple((primal[rows[:, None], rows], tangent[rows]) for rows in bands)

        return _TangentActions(self, blocks(direct, ddirect), blocks(crossed, dcrossed), irrep)


class _TangentActions:
    def __init__(self, owner, direct, crossed, irrep):
        self.owner, self.direct, self.crossed = owner, direct, crossed
        self.irrep = irrep
        self.lines = {}

    def prepare_plane(self, plane, tangent, k, *, group=None):
        owner, xp = self.owner, self.owner.xp
        self.lines.clear()
        pack, _ = _kernels()
        groups = owner.groups[k] if group is None else (group,)
        for indices in groups:
            count = len(indices)
            shift = int(owner.ph.shifts[indices[0], k]) ^ self.irrep
            plan = owner.plans[shift]
            direct, crossed = (xp.empty(plan.size * count) for _ in range(2))
            dev_indices = xp.asarray(indices, dtype=xp.int64)
            for spin, blocks, output in ((True, self.direct, direct), (False, self.crossed, crossed)):
                for (primal, derivative), band in zip(blocks, plan.bands):
                    width = len(band.columns)
                    if not width:
                        continue
                    columns = count * width
                    start, stop = band.output.start * count, band.output.stop * count
                    destination = output[start:stop].reshape(len(band.rows), columns)
                    # The direction is exactly in the target irrep. The ground
                    # plane can contain symmetry leakage, so use all its rows.
                    packed = pack(tangent, band.rows, band.columns, dev_indices, owner.no,
                                  owner.nv, width, count, spin,
                                  size=len(band.rows) * columns).reshape(len(band.rows), columns)
                    xp.matmul(primal, packed, out=destination)
                    packed = pack(plane, owner.all_rows, band.columns, dev_indices, owner.no,
                                  owner.nv, width, count, spin,
                                  size=len(owner.all_rows) * columns).reshape(len(owner.all_rows), columns)
                    destination += derivative @ packed
            bases = plan.bases * count
            for rank, j in enumerate(indices):
                self.lines[int(j)] = direct, crossed, bases, owner.widths[shift], shift, rank

    def add(self, seed, j):
        owner = self.owner
        direct, crossed, bases, widths, shift, rank = self.lines[j]
        _, add = _kernels()
        add(seed, direct, crossed, bases, widths, owner.rows, owner.columns, owner.ranks,
            owner.nv, shift, rank, seed)

    def fold(self, target, seed, line, j, k, layout):
        owner = self.owner
        direct, crossed, bases, widths, shift, rank = self.lines[j]
        _fold_kernel()(seed, layout._fold_plan[j, k], direct, crossed, bases, widths,
                       owner.rows, owner.columns, owner.ranks, owner.nv, owner.nv**3,
                       shift, rank, seed.shape[1], line, target, size=owner.no * owner.nv**3)

    def release_plane(self):
        self.lines.clear()
