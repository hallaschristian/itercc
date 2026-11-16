"""Exact symmetry projection of the occupied-pair tangent contractions."""

from ._cuda_occupied import CudaOccupiedLadder, _OccupiedPlane, _gather_kernel


class CudaProjectedOccupied:
    def __init__(self, occupied, virtual, irrep, xp):
        labels = (virtual[:, None, None] ^ virtual[None, :, None] ^ virtual).ravel()
        self.owner = CudaOccupiedLadder(occupied, labels, xp, output_irrep=irrep)
        self.all_rows = xp.arange(len(occupied)**2, dtype=xp.int64)

    def prepare(self, matrix, derivative):
        owner = self.owner
        rows = tuple(owner.order[band] for band in owner.sectors.bands)
        blocks = tuple((matrix[row[:, None], row], derivative[row]) for row in rows)
        return _ProjectedOccupiedAction(owner, self.all_rows, blocks)


class _ProjectedOccupiedAction:
    def __init__(self, owner, all_rows, blocks):
        self.owner, self.all_rows, self.blocks = owner, all_rows, blocks
        xp = owner.xp
        # One 8 MiB gather buffer handles even the unrestricted ground plane.
        self.chunk = max(1, min(owner.width, (8 << 20) // (len(all_rows) * 8)))
        self.scratch = xp.empty(len(all_rows) * self.chunk)
        self.values = xp.empty(max(plan.size for plan in owner.plans.values()))

    def plane(self, ground, tangent, k):
        owner = self.owner
        shift = int(owner.shifts[k])
        plan = owner.plans[shift]
        values = self.values[:plan.size]
        gather = _gather_kernel()
        for (matrix, derivative), band in zip(self.blocks, plan.bands):
            columns = band.columns
            rows, width = len(band.rows), len(columns)
            output = values[band.destination].reshape(rows, width)
            for start in range(0, width, self.chunk):
                stop = min(start + self.chunk, width)
                selection = columns[start:stop]
                count = stop - start
                restricted = self.scratch[:rows*count].reshape(rows, count)
                gather(tangent, band.rows, selection, owner.width, count, restricted)
                value = matrix @ restricted
                # Ground amplitudes may contain real off-sector entries. Keep
                # every input row of the derivative-matrix product.
                full = self.scratch[:len(self.all_rows)*count].reshape(len(self.all_rows), count)
                gather(ground, self.all_rows, selection, owner.width, count, full)
                value += derivative @ full
                output[:, start:stop] = value
        return _OccupiedPlane(owner, values, plan, shift)
