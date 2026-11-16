"""Verify exact point-group support of packed CUDA triples."""

from functools import cache

import numpy as np


@cache
def _support_kernel():
    from cupy import RawKernel

    return RawKernel(r"""
    extern "C" __global__
    void itercc_check_triples_support(
        const double *t3, const unsigned char *occupied,
        const unsigned char *virtuals, const long long cube,
        const long long count, int *leak)
    {
        const long long first = blockIdx.x * blockDim.x + threadIdx.x;
        const long long stride = gridDim.x * blockDim.x;
        bool found = false;
        for (long long index = first; index < count; index += stride) {
            if (t3[index] != 0.0 && occupied[index / cube] != virtuals[index % cube]) {
                found = true;
                break;
            }
        }
        if (__syncthreads_or(found) && threadIdx.x == 0) atomicExch(leak, 1);
    }
    """, "itercc_check_triples_support")


class CudaTriplesSupport:
    def __init__(self, occupied, virtual, layout, xp):
        self.xp = xp
        self.occupied_labels = np.array(
            [occupied[i] ^ occupied[j] ^ occupied[k] for i, j, k in layout.indices],
            dtype=np.uint8)
        self.occupied = xp.asarray(self.occupied_labels)
        self.cube = layout.nv**3
        pairs = virtual[:, None] ^ virtual
        self.virtual = xp.asarray(pairs[:, :, None] ^ virtual, dtype=xp.uint8)

    def supports(self, t3):
        """Prove all excluded entries are exactly zero before dropping columns."""
        if not t3.size:
            return True
        t3 = self.xp.ascontiguousarray(t3)
        leak = self.xp.zeros(1, dtype=self.xp.int32)
        threads = 256
        blocks = min(2048, (t3.size + threads - 1) // threads)
        _support_kernel()((blocks,), (threads,),
                          (t3, self.occupied, self.virtual,
                           np.int64(self.cube), np.int64(t3.size), leak))
        return not leak.item()
