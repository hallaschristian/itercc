"""Evaluate the virtual ladder only on a requested triples output irrep."""

import numpy as np

from ._cuda_ladder_columns import CudaLadderColumns
from ._cuda_packed import PackedTriples
from ._cuda_pairs import CudaPairPacking
from ._pair_symmetry import PairSectors


class CudaProjectedLadder:
    """Keep all input bands that can contribute to the projected output."""

    def __init__(self, layout, labels, xp):
        no, nv = layout.no, layout.nv
        vir = np.asarray(labels[no:])
        self.xp = xp
        self.virtual_labels = vir
        self.occupied_labels = np.bitwise_xor.reduce(labels[np.asarray(layout.indices)], axis=1)
        pairs = np.triu_indices(nv), np.triu_indices(nv, 1)
        self.sectors = tuple(PairSectors(vir[a] ^ vir[b]) for a, b in pairs)
        grouped = [(a[sectors.order], b[sectors.order])
                   for (a, b), sectors in zip(pairs, self.sectors)]
        self.packing = CudaPairPacking(nv, *grouped, xp)

    def add(self, target, source, w, target_irrep, batch, *, source_irrep=None):
        # For each output band, its spectator columns are fixed by the
        # target irrep. A pure source selects one input band per GEMM.
        # When the source is unrestricted, use every input row.
        matrices = self.packing.matrices(w)
        products = []
        for matrix, sectors in zip(matrices, self.sectors):
            bands = dict(zip(sectors.labels, sectors.bands))
            shift = None if source_irrep is None else source_irrep ^ target_irrep
            inputs = (slice(None),) * len(sectors.bands) if shift is None else tuple(
                bands.get(label ^ shift, slice(0, 0)) for label in sectors.labels)
            products.append(tuple((inp.start or 0, self.xp.ascontiguousarray(matrix[out, inp]))
                                  for out, inp in zip(sectors.bands, inputs)))
        # Share packing and scatter logic with the exact ground-state path.
        starts = tuple(tuple(first for first, _ in sector) for sector in products)
        blocks = tuple(tuple(block for _, block in sector) for sector in products)
        kernel = CudaLadderColumns(self.packing, self.virtual_labels, *self.sectors, *blocks,
                                   input_starts=starts)
        def rows(array, band):
            return array.slots(band.start, band.stop) if isinstance(array, PackedTriples) else array[band]

        for start in range(0, len(source), batch):
            band = slice(start, start + batch)
            kernel.add(rows(source, band), rows(target, band),
                       self.occupied_labels[band] ^ target_irrep)
