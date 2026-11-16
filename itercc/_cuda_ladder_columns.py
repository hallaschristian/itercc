"""Compact FP64 virtual-ladder products for exactly symmetric triples."""

from functools import cache

import numpy as np

from ._cuda_packed import _PackedSlots, _ladder_kernels


@cache
def _kernels():
    from cupy import ElementwiseKernel

    pack = ElementwiseKernel(
        "raw float64 source, raw int32 pairs, raw int32 columns, int32 nv, "
        "int32 width, int32 first, int32 sa, int32 sb, int32 sc, int32 parity",
        "float64 value",
        r"""
        const int row = first + i / width;
        const int col = columns[i % width];
        const int a = pairs[2 * row];
        const int b = pairs[2 * row + 1];
        const long long base = static_cast<long long>(col / nv) * nv * nv * nv
                             + (col % nv) * sc;
        const double ab = source[base + a * sa + b * sb];
        const double ba = source[base + b * sa + a * sb];
        value = 0.5 * (parity < 0 ? ab - ba : ab + ba);
        """, "itercc_pack_ladder_columns")
    accumulate = ElementwiseKernel(
        "float64 current, raw float64 even, raw float64 odd, raw int32 even_rows, "
        "raw int32 odd_rows, raw int64 even_base, raw int64 odd_base, raw uint8 pair_labels, "
        "raw uint8 column_labels, raw int32 ranks, int32 nv, int32 nv2, int32 nv3, int32 orientation",
        "float64 result",
        r"""
        const int rem = i % nv3;
        const int v0 = rem / nv2;
        const int v1 = (rem / nv) % nv;
        const int v2 = rem % nv;
        const int a = orientation == 2 ? v1 : v0;
        const int b = orientation == 0 ? v1 : v2;
        const int c = orientation == 0 ? v2 : (orientation == 1 ? v1 : v0);
        const int pair = a * nv + b;
        const int col = (i / nv3) * nv + c;
        result = current;
        if (pair_labels[pair] == column_labels[col]) {
            const int rank = ranks[col];
            double value = even[even_base[even_rows[pair]] + rank];
            if (a != b) {
                const double exchange = odd[odd_base[odd_rows[pair]] + rank];
                value = a < b ? value + exchange : value - exchange;
            }
            result += value;
        }
        """, "itercc_add_ladder_columns")
    return pack, accumulate


class CudaLadderColumns:
    def __init__(self, packing, virtual_labels, even_sectors, odd_sectors, even_blocks, odd_blocks,
                 *, input_starts=None):
        self.packing = packing
        self.xp = packing.xp
        self.virtual_labels = virtual_labels
        self.pair_labels = self.xp.asarray(virtual_labels[:, None] ^ virtual_labels,
                                           dtype=self.xp.uint8)
        self.sectors = even_sectors, odd_sectors
        if input_starts is None:
            input_starts = (tuple(band.start for band in sectors.bands) for sectors in self.sectors)
        self.blocks = tuple(tuple(zip(starts, blocks))
                            for starts, blocks in zip(input_starts, (even_blocks, odd_blocks)))

    def _columns(self, occupied_labels):
        labels = (occupied_labels[:, None] ^ self.virtual_labels).ravel()
        columns = [np.flatnonzero(labels == irrep).astype(np.int32) for irrep in range(8)]
        ranks = np.empty(labels.size, dtype=np.int32)
        for indices in columns:
            ranks[indices] = np.arange(indices.size, dtype=np.int32)
        return (self.xp.asarray(labels, dtype=self.xp.uint8), self.xp.asarray(ranks),
                tuple(self.xp.asarray(indices) for indices in columns))

    def _products(self, sectors, blocks, columns):
        bases = np.empty(len(sectors.order), dtype=np.int64)
        size = 0
        products = []
        for irrep, band, (first, block) in zip(sectors.labels, sectors.bands, blocks):
            selected = columns[irrep]
            width = len(selected)
            rows = band.stop - band.start
            bases[band] = size + width * np.arange(rows, dtype=np.int64)
            if width:
                stop = size + rows * width
                products.append((first, selected, block, slice(size, stop)))
                size = stop
        return self.xp.asarray(bases), size, products

    def _multiply(self, source, parity, strides, plan):
        pack, _ = _kernels()
        pairs = self.packing.indices[parity]
        _, size, products = plan
        packed = isinstance(source, _PackedSlots)
        if packed:
            pack_packed, _ = _ladder_kernels()
            index = source.owner.index
            extra = (index.table, index.slot_table, index.offsets, source.start, index.nv3)
        result = self.xp.empty(size, dtype=self.xp.float64)
        for first, selected, block, output in products:
            width = len(selected)
            rows, inputs = block.shape
            destination = result[output].reshape(rows, width)
            if not inputs:
                destination.fill(0.)
                continue
            if packed:
                rhs = pack_packed(source.owner.data, pairs, selected, self.packing.nv, width, first,
                                  *strides, 1 - 2 * parity, *extra, size=inputs * width)
            else:
                rhs = pack(source, pairs, selected, self.packing.nv, width, first, *strides,
                           1 - 2 * parity, size=inputs * width)
            self.xp.matmul(block, rhs.reshape(inputs, width), out=destination)
        return result

    def add(self, source, target, occupied_labels):
        _, accumulate = _kernels()
        if not isinstance(source, _PackedSlots):
            source = self.xp.ascontiguousarray(source)
        nv = self.packing.nv
        labels, ranks, columns = self._columns(occupied_labels)
        plans = [self._products(sectors, blocks, columns)
                 for sectors, blocks in zip(self.sectors, self.blocks)]
        even_base, odd_base = (plan[0] for plan in plans)
        for orientation, perm in enumerate(((0, 1, 2), (0, 2, 1), (1, 2, 0))):
            strides = tuple(nv ** (2 - axis) for axis in perm)
            even, odd = (self._multiply(source, parity, strides, plan)
                         for parity, plan in enumerate(plans))
            if isinstance(target, _PackedSlots):
                index = target.owner.index
                first, last = (int(index.offsets_host[s]) for s in (target.start, target.stop))
                if last > first:
                    _, add_packed = _ladder_kernels()
                    add_packed(even, odd, *self.packing.rows, even_base, odd_base, ranks,
                               index.positions, first, target.start, nv, nv**2, nv**3,
                               orientation, target.owner.data, size=last - first)
            else:
                accumulate(target, even, odd, *self.packing.rows, even_base, odd_base,
                           self.pair_labels, labels, ranks, nv, nv**2, nv**3, orientation, target)