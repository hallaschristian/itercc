"""Block GEMMs with an exact dense fallback for spatial pair interactions."""

import numpy as np


class PairSectors:
    """Group pair labels once and reuse contiguous row bands in each GEMM."""

    def __init__(self, labels):
        self.order = np.argsort(labels, kind="stable")
        self.inverse = np.argsort(self.order)
        self.labels, counts = np.unique(labels, return_counts=True)
        ends = np.r_[0, np.cumsum(counts)]
        self.bands = tuple(slice(int(a), int(b)) for a, b in zip(ends[:-1], ends[1:]))

    def _blocks(self, matrix, xp, grouped, order):
        if len(self.bands) <= 1:
            return None
        if grouped:
            blocks = [xp.ascontiguousarray(matrix[band, band]) for band in self.bands]
        else:
            blocks = []
            for band in self.bands:
                indices = order[band]
                blocks.append(matrix[indices[:, None], indices])
        # Equality of exact counts proves all cross-sector entries are zero.
        counts = xp.stack([xp.count_nonzero(block) for block in blocks])
        if int(xp.count_nonzero(matrix) - counts.sum()):
            return None
        return blocks

    def product(self, matrix, *, xp=np, grouped=False):
        order = None if grouped else xp.asarray(self.order)
        blocks = self._blocks(matrix, xp, grouped, order)
        if blocks is None:
            return matrix.__matmul__, False
        matrix_dtype = matrix.dtype
        inverse = None if grouped else xp.asarray(self.inverse)

        def multiply(rhs):
            if not grouped:
                rhs = xp.take(rhs, order, axis=0)
            result = xp.empty(rhs.shape, dtype=xp.result_type(matrix_dtype, rhs.dtype))
            for band, block in zip(self.bands, blocks):
                xp.matmul(block, rhs[band], out=result[band])
            if not grouped:
                return xp.take(result, inverse, axis=0)
            return result

        return multiply, True


    def column_product(self, matrix, columns, order, *, xp=np, pack=None,
                       dense_rhs=None, output_shape=None):
        """Use column support proved by the caller for each row sector."""
        blocks = self._blocks(matrix, xp, False, order)
        if blocks is None:
            def dense(rhs, shift, *args):
                return matrix @ (rhs if dense_rhs is None else dense_rhs(rhs, *args))

            return dense, False
        matrix_dtype = matrix.dtype
        sectors = tuple((int(label), order[band], block)
                        for label, band, block in zip(self.labels, self.bands, blocks))

        def multiply(rhs, shift, *args):
            shape = rhs.shape if output_shape is None else output_shape
            result = xp.zeros(shape, dtype=xp.result_type(matrix_dtype, rhs.dtype))
            for label, rows, block in sectors:
                selected = columns[label ^ shift]
                if selected.size:
                    indices = rows[:, None], selected
                    packed = rhs[indices] if pack is None else pack(rhs, rows, selected, *args)
                    result[indices] = block @ packed
            return result

        return multiply, True


def pair_product(matrix, pair_irreps, *, xp=np):
    """Build an exact blocked product for arbitrary input row ordering."""
    return PairSectors(pair_irreps).product(matrix, xp=xp)