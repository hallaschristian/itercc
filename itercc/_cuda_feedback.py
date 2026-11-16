"""Exact symmetry blocks for the two largest triples feedback contractions."""

from functools import cache

import numpy as np


@cache
def _gather():
    from cupy import ElementwiseKernel

    return ElementwiseKernel(
        "raw float64 q, raw int64 rows, raw int64 cols, int64 width, int64 columns",
        "float64 out",
        """
        const long long row = rows[i / width], col = cols[i % width];
        out = q[row * columns + col];
        """, "itercc_feedback_gather")


class _Product:
    def __init__(self, output_labels, contracted_labels, column_labels, xp):
        self.xp = xp
        self.shape = len(output_labels), len(column_labels)
        self.rows = tuple(xp.asarray(np.flatnonzero(output_labels == label)) for label in range(8))
        self.inner = tuple(xp.asarray(np.flatnonzero(contracted_labels == label)) for label in range(8))
        self.columns = tuple(xp.asarray(np.flatnonzero(column_labels == label)) for label in range(8))

    def apply(self, left, right, shift):
        prepared = self.prepare(left, shift)
        return None if prepared is None else self.multiply(prepared, right)

    def prepare(self, left, shift):
        xp = self.xp
        left = xp.ascontiguousarray(left).reshape(self.shape[0], -1)
        blocks = [(rows, self.inner[label ^ shift], label)
                  for label, rows in enumerate(self.rows) if len(rows)]
        matrices = [left[rows[:, None], inner] for rows, inner, _ in blocks]
        # Prove support of the interaction. The caller proves T3 support once.
        count = xp.stack([xp.count_nonzero(matrix) for matrix in matrices]).sum()
        if int(xp.count_nonzero(left) - count):
            return None
        return tuple((matrix, rows, inner, label)
                     for matrix, (rows, inner, label) in zip(matrices, blocks))

    def multiply(self, prepared, right, column_shift=0):
        xp = self.xp
        result = xp.zeros(self.shape, dtype=right.dtype)
        for matrix, rows, inner, label in prepared:
            cols = self.columns[label ^ column_shift]
            if len(inner) and len(cols):
                packed = _gather()(right, inner, cols, len(cols), self.shape[1],
                                   size=len(inner) * len(cols)).reshape(len(inner), len(cols))
                result[rows[:, None], cols] = matrix @ packed
        return result


class CudaFeedback:
    def __init__(self, occupied, virtual, xp):
        self.no, self.nv = len(occupied), len(virtual)
        self.occupied = occupied
        pairs = (virtual[:, None] ^ virtual).ravel()
        ov = (occupied[:, None] ^ virtual).ravel()
        oov = (occupied[:, None, None] ^ occupied[None, :, None] ^ virtual).ravel()
        vvo = (virtual[:, None, None] ^ virtual[None, :, None] ^ occupied).ravel()
        self.doubles_product = _Product(virtual, pairs, oov, xp)
        self.source_product = _Product(virtual, ov, vvo, xp)

    def doubles(self, integral, plane, m):
        value = self.doubles_product.apply(integral, plane.transpose(4, 2, 0, 1, 3),
                                           int(self.occupied[m]))
        if value is not None:
            return value.reshape(self.nv, self.no, self.no, self.nv).transpose(1, 2, 3, 0)
        return None

    def source(self, integral, plane, m):
        value = self.source_product.apply(integral.transpose(1, 0, 2),
                                          plane.transpose(1, 2, 4, 3, 0),
                                          int(self.occupied[m]))
        if value is not None:
            return value.reshape(self.nv, self.nv, self.nv, self.no).transpose(1, 2, 0, 3)
        return None
