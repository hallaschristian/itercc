"""Keep particle-hole results compact through their combined seed update."""

from functools import cache
from typing import NamedTuple

import numpy as np

from ._cuda_ph import _pack_kernel


@cache
def _accumulate():
    from cupy import ElementwiseKernel

    return ElementwiseKernel(
        "float64 seed, raw float64 direct, raw float64 crossed, raw int64 bases, "
        "raw uint8 rows, raw uint8 columns, raw int32 ranks, int32 nv, uint8 shift",
        "float64 result",
        """
        const int width = nv * nv;
        const int row = i / width, col = i % width;
        double d = 0.0, x = 0.0, swapped = 0.0;
        if ((rows[row] ^ shift) == columns[col]) {
            const long long entry = bases[row] + ranks[col];
            d = direct[entry];
            x = crossed[entry];
        }
        const int a = row % nv, b = col / nv, c = col % nv;
        const int other_row = (row / nv) * nv + b;
        const int other_col = a * nv + c;
        if ((rows[other_row] ^ shift) == columns[other_col]) {
            swapped = crossed[bases[other_row] + ranks[other_col]];
        }
        result = ((seed + d) - swapped) - 0.5 * x;
        """, "itercc_compact_ph_accumulate", options=("--fmad=false",))


class _Band(NamedTuple):
    rows: object
    columns: object
    output: slice


class _Plan(NamedTuple):
    bases: object
    size: int
    scratch: int
    bands: tuple


class CudaParticleHoleFusion:
    def __init__(self, occupied, virtual, ph, xp):
        self.xp, self.ph = xp, ph
        self.no, self.nv = len(occupied), len(virtual)
        rows = (occupied[:, None] ^ virtual).ravel()
        columns = (virtual[:, None] ^ virtual).ravel()
        self.rows, self.columns = xp.asarray(rows, dtype=xp.uint8), xp.asarray(columns, dtype=xp.uint8)
        ranks = np.empty(len(columns), dtype=np.int32)
        for label in range(8):
            selected = np.flatnonzero(columns == label)
            ranks[selected] = np.arange(len(selected))
        self.ranks = xp.asarray(ranks)
        self.plans = {int(shift): self._plan(int(shift)) for shift in np.unique(ph.shifts)}

    def _plan(self, shift):
        xp, ph = self.xp, self.ph
        bases = np.empty(self.no * self.nv, dtype=np.int64)
        size, scratch, bands = 0, 0, []
        for label, band in zip(ph.sectors.labels, ph.sectors.bands):
            host_rows = ph.sectors.order[band]
            rows, columns = ph.order[band], ph.columns[int(label) ^ shift]
            count = len(rows) * len(columns)
            bases[host_rows] = size + np.arange(len(rows)) * len(columns)
            bands.append(_Band(rows, columns, slice(size, size + count)))
            size += count
            scratch = max(scratch, count)
        return _Plan(xp.asarray(bases), size, scratch, tuple(bands))

    def prepare(self, direct, crossed):
        ph, xp = self.ph, self.xp
        direct_blocks = ph.sectors._blocks(direct, xp, False, ph.order)
        crossed_blocks = ph.sectors._blocks(crossed, xp, False, ph.order)
        if direct_blocks is None or crossed_blocks is None:
            return None

        def add(seed, plane, j, k):
            shift = int(ph.shifts[j, k])
            plan = self.plans[shift]
            direct_values, crossed_values = (xp.empty(plan.size) for _ in range(2))
            scratch = xp.empty(plan.scratch)
            for spin, blocks, target in ((True, direct_blocks, direct_values),
                                         (False, crossed_blocks, crossed_values)):
                for matrix, band in zip(blocks, plan.bands):
                    width = len(band.columns)
                    if width:
                        shape = len(band.rows), width
                        packed = scratch[:shape[0] * width].reshape(shape)
                        _pack_kernel()(plane, band.rows, band.columns, self.no, self.nv, j, width, spin, packed)
                        xp.matmul(matrix, packed, out=target[band.output].reshape(shape))
            _accumulate()(seed, direct_values, crossed_values, plan.bases, self.rows,
                          self.columns, self.ranks, self.nv, shift, seed)

        return add
