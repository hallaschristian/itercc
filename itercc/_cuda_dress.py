"""Symmetry-aware, in-place T1 dressing of the CUDA interaction tensor."""

from functools import cache

import numpy as np


@cache
def _dress_kernel():
    from cupy import RawKernel

    return RawKernel(r"""
    extern "C" __global__
    void itercc_dress_axis(double *tensor, const double *t1,
                          const int *edges, const int *inputs, long long stride,
                          long long size, int n, int no, int nv, int forward)
    {
        const int targets = forward ? nv : no;
        const int to_offset = forward ? no : 0;
        const int from_offset = forward ? 0 : no;
        const long long delta = static_cast<long long>(gridDim.x) * blockDim.x;
        for (long long index = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
             index < size; index += delta) {
            const long long inner = index % stride;
            const long long group = index / stride;
            const int target = group % targets;
            const long long base = (group / targets) * n * stride + inner;
            double sum = 0.0;
            for (int p = edges[target]; p < edges[target + 1]; ++p) {
                const int source = inputs[p];
                const double amplitude = forward ? -t1[source * nv + target]
                                                 : t1[target * nv + source];
                sum += amplitude * tensor[base + (from_offset + source) * stride];
            }
            tensor[base + (to_offset + target) * stride] += sum;
        }
    }
    """, "itercc_dress_axis")


class CudaT1Dressing:
    """Apply sparse T1 rows after proving exact orbital symmetry support."""

    def __init__(self, occupied, virtual, xp):
        self.xp = xp
        self.no, self.nv = len(occupied), len(virtual)
        self.n = self.no + self.nv
        self.forbidden = xp.asarray(occupied[:, None] != virtual)
        self.forward = self._plan(virtual, occupied)
        self.backward = self._plan(occupied, virtual)

    def _plan(self, targets, sources):
        rows = [np.flatnonzero(sources == label) for label in targets]
        offsets = np.r_[0, np.cumsum([len(row) for row in rows])]
        return (self.xp.asarray(offsets, dtype=self.xp.int32),
                self.xp.asarray(np.concatenate(rows), dtype=self.xp.int32))

    def apply(self, tensor, t1):
        """Return False before modifying the tensor when T1 has off-sector entries."""
        if bool(self.xp.any(t1[self.forbidden])):
            return False
        threads = 256
        kernel = _dress_kernel()
        for axis in range(4):
            forward = axis < 2
            targets = self.nv if forward else self.no
            offsets, sources = self.forward if forward else self.backward
            elements = targets * self.n**3
            blocks = min(4096, (elements + threads - 1) // threads)
            kernel((blocks,), (threads,),
                   (tensor, t1, offsets, sources, np.int64(self.n**(3 - axis)),
                    np.int64(elements), np.int32(self.n), np.int32(self.no),
                    np.int32(self.nv), np.int32(forward)))
        return True