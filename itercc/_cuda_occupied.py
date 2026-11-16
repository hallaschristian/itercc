"""Compact the occupied-pair ladder on exactly symmetry-supported CUDA planes."""

from functools import cache
from typing import NamedTuple

import numpy as np

from ._pair_symmetry import PairSectors


@cache
def _add_kernel():
    from cupy import ElementwiseKernel

    return ElementwiseKernel(
        "float64 current, raw float64 values, raw int64 bases, raw uint8 row_irreps, "
        "raw uint8 column_irreps, raw int32 ranks, int32 no, int32 width, int32 j, uint8 shift",
        "float64 result",
        r"""
        const int row = (i / width) * no + j;
        const int col = i % width;
        result = current;
        if ((row_irreps[row] ^ shift) == column_irreps[col]) {
            result += values[bases[row] + ranks[col]];
        }
        """, "itercc_add_compact_occupied")


@cache
def _gather_kernel():
    from cupy import ElementwiseKernel

    return ElementwiseKernel(
        "raw float64 plane, raw int64 rows, raw int64 columns, int32 plane_width, int32 width",
        "float64 result",
        "result = plane[rows[i / width] * plane_width + columns[i % width]]",
        "itercc_gather_compact_occupied")


@cache
def _fold_kernel():
    from cupy import ElementwiseKernel

    return ElementwiseKernel(
        "raw float64 seed, raw int32 fold, raw float64 values, raw int64 bases, "
        "raw uint8 row_irreps, raw uint8 column_irreps, raw int32 ranks, "
        "int32 no, int32 nv, int32 width, int32 j, uint8 shift", "raw float64 target",
        r"""
        const int iocc = i / width;
        const int entry = iocc * 5;
        const int slot = fold[entry];
        if (slot >= 0) {
            const int rem = i % width;
            const int a = rem / (nv * nv);
            const int b = (rem / nv) % nv;
            const int c = rem % nv;
            const int sa = fold[entry + 1], sb = fold[entry + 2], sc = fold[entry + 3];
            const int row = iocc * no + j;
            const int label = row_irreps[row] ^ shift;
            const long long seed_base = static_cast<long long>(iocc) * width;
            const long long index = static_cast<long long>(slot) * width + rem;
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
        """, "itercc_fold_occupied_seed")


class _Band(NamedTuple):
    rows: object
    columns: object
    destination: slice


class _PlanePlan(NamedTuple):
    bases: object
    size: int
    scratch: int
    bands: tuple


class _OccupiedPlane:
    def __init__(self, owner, values, plan, shift):
        self.owner = owner
        self.shift = shift
        self.operands = values, plan.bases, owner.row_labels, owner.column_labels, owner.ranks

    def __call__(self, seed, j):
        owner = self.owner
        _add_kernel()(seed, *self.operands, owner.no, owner.width, j, self.shift, seed)

    def fold(self, target, seed, fold_plan, j, nv):
        owner = self.owner
        if hasattr(target, "fold_occupied"):
            target.fold_occupied(seed, fold_plan, self.operands, owner.no, nv, owner.width,
                                 j, self.shift)
            return
        _fold_kernel()(seed, fold_plan, *self.operands, owner.no, nv, owner.width, j,
                       self.shift, target, size=seed.size)


class CudaOccupiedLadder:
    """Group occupied rows and select the virtual columns allowed for fixed k."""

    def __init__(self, occupied, virtual_triples, xp, *, output_irrep=0):
        self.xp = xp
        self.no, self.width = len(occupied), len(virtual_triples)
        row_labels = (occupied[:, None] ^ occupied).ravel()
        self.row_labels = xp.asarray(row_labels, dtype=xp.uint8)
        self.column_labels = xp.asarray(virtual_triples, dtype=xp.uint8)
        self.sectors = PairSectors(row_labels)
        self.order = xp.asarray(self.sectors.order, dtype=xp.int64)
        columns = [np.flatnonzero(virtual_triples == irrep) for irrep in range(8)]
        ranks = np.empty(self.width, dtype=np.int32)
        for selected in columns:
            ranks[selected] = np.arange(len(selected), dtype=np.int32)
        self.ranks = xp.asarray(ranks)
        columns = tuple(xp.asarray(selected, dtype=xp.int64) for selected in columns)
        self.shifts = occupied ^ output_irrep
        self.plans = {int(shift): self._plan(int(shift), columns) for shift in np.unique(self.shifts)}

    def _plan(self, shift, columns):
        bases = np.empty(self.no**2, dtype=np.int64)
        start, scratch, bands = 0, 0, []
        for sector, band in zip(self.sectors.labels, self.sectors.bands):
            rows = self.sectors.order[band]
            selected = columns[int(sector) ^ shift]
            width = len(selected)
            bases[rows] = start + np.arange(len(rows), dtype=np.int64) * width
            stop = start + len(rows) * width
            scratch = max(scratch, stop - start)
            bands.append(_Band(self.order[band], selected, slice(start, stop)))
            start = stop
        return _PlanePlan(self.xp.asarray(bases), start, scratch, tuple(bands))

    def product(self, matrix):
        """Return None if exact matrix support does not permit compression."""
        xp = self.xp
        blocks = self.sectors._blocks(matrix, xp, False, self.order)
        if blocks is None:
            return None

        def multiply(plane, k):
            rhs = xp.ascontiguousarray(plane)
            shift = int(self.shifts[k])
            plan = self.plans[shift]
            values = xp.empty(plan.size, dtype=matrix.dtype)
            scratch = xp.empty(plan.scratch, dtype=matrix.dtype)
            gather = _gather_kernel()
            for block, band in zip(blocks, plan.bands):
                width = len(band.columns)
                if width:
                    shape = len(block), width
                    source = scratch[:shape[0] * width].reshape(shape)
                    gather(rhs, band.rows, band.columns, self.width, width, source)
                    out = values[band.destination].reshape(shape)
                    xp.matmul(block, source, out=out)
            return _OccupiedPlane(self, values, plan, shift)

        return multiply