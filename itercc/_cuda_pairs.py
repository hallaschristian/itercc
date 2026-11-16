"""Fused FP64 packing and accumulation for the CUDA virtual triples ladder."""

from functools import cache

import numpy as np


@cache
def _kernels():
    from cupy import ElementwiseKernel

    matrix = ElementwiseKernel(
        "raw float64 w, raw int32 pairs, int32 nv, int32 count, int32 parity",
        "float64 result",
        """
        const int row = i / count;
        const int col = i % count;
        const int a = pairs[2 * row];
        const int b = pairs[2 * row + 1];
        const int d = pairs[2 * col];
        const int e = pairs[2 * col + 1];
        const long long base = (static_cast<long long>(a) * nv + b) * nv * nv;
        const double direct = w[base + d * nv + e];
        const double exchange = w[base + e * nv + d];
        result = parity < 0 ? direct - exchange : (d == e ? direct : direct + exchange);
        """, "itercc_pair_matrix_fp64")
    pack = ElementwiseKernel(
        "raw float64 source, raw int32 pairs, int32 nv, int32 columns, "
        "int32 sa, int32 sb, int32 sc, int32 parity", "float64 result",
        """
        const int row = i / columns;
        const int col = i % columns;
        const int a = pairs[2 * row];
        const int b = pairs[2 * row + 1];
        const long long base = static_cast<long long>(col / nv) * nv * nv * nv
                             + (col % nv) * sc;
        const double ab = source[base + a * sa + b * sb];
        const double ba = source[base + b * sa + a * sb];
        result = 0.5 * (parity < 0 ? ab - ba : ab + ba);
        """, "itercc_pack_pairs_fp64")
    accumulate = ElementwiseKernel(
        "float64 current, raw float64 even, raw float64 odd, raw int32 even_rows, "
        "raw int32 odd_rows, int32 nv, int32 nv2, int32 nv3, int32 columns, int32 orientation",
        "float64 result",
        """
        const int rem = i % nv3;
        const int v0 = rem / nv2;
        const int v1 = (rem / nv) % nv;
        const int v2 = rem % nv;
        const int a = orientation == 2 ? v1 : v0;
        const int b = orientation == 0 ? v1 : v2;
        const int c = orientation == 0 ? v2 : (orientation == 1 ? v1 : v0);
        const int pair = a * nv + b;
        const int col = (i / nv3) * nv + c;
        double value = even[static_cast<long long>(even_rows[pair]) * columns + col];
        if (a != b) {
            const double exchange = odd[static_cast<long long>(odd_rows[pair]) * columns + col];
            value = a < b ? value + exchange : value - exchange;
        }
        result = current + value;
        """, "itercc_add_pairs_fp64")
    return matrix, pack, accumulate


class CudaPairPacking:
    def __init__(self, nv, even_indices, odd_indices, xp):
        self.xp, self.nv = xp, nv
        self.indices, self.rows = [], []
        for p, q in (even_indices, odd_indices):
            rows = np.zeros((nv, nv), dtype=np.int32)
            rows[p, q] = rows[q, p] = np.arange(len(p), dtype=np.int32)
            self.indices.append(xp.asarray(np.column_stack((p, q)), dtype=xp.int32))
            self.rows.append(xp.asarray(rows))

    def matrices(self, w):
        kernel, _, _ = _kernels()
        w = self.xp.ascontiguousarray(w)
        matrices = []
        for pairs, parity in zip(self.indices, (1, -1)):
            count = len(pairs)
            if count:
                matrix = kernel(w, pairs, self.nv, count, parity, size=count**2)
                matrices.append(matrix.reshape(count, count))
            else:
                matrices.append(self.xp.empty((0, 0), dtype=w.dtype))
        return matrices

    def add(self, source, target, even_product, odd_product):
        _, pack, accumulate = _kernels()
        source = self.xp.ascontiguousarray(source)
        nv = self.nv
        columns = len(source) * nv
        for orientation, perm in enumerate(((0, 1, 2), (0, 2, 1), (1, 2, 0))):
            strides = tuple(nv ** (2 - axis) for axis in perm)
            even_indices, odd_indices = self.indices
            rhs = pack(source, even_indices, nv, columns, *strides, 1,
                       size=len(even_indices) * columns).reshape(len(even_indices), columns)
            even = self.xp.ascontiguousarray(even_product(rhs))
            del rhs
            if len(odd_indices):
                rhs = pack(source, odd_indices, nv, columns, *strides, -1,
                           size=len(odd_indices) * columns).reshape(len(odd_indices), columns)
                odd = self.xp.ascontiguousarray(odd_product(rhs))
                del rhs
            else:
                odd = even
            accumulate(target, even, odd, *self.rows, nv, nv**2, nv**3, columns,
                       orientation, target)