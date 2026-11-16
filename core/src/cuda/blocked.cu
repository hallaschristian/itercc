// Symmetry-blocked restricted T3 on the GPU: pack and unpack, the same
// layout as core/src/cpu/blocked.cpp. Every pointer is device memory; the
// work is queued on `stream` (a cudaStream_t, e.g. CuPy's current stream).
#include <cuda_runtime.h>

#include "itercc/cuda.h"

namespace {

// One CUDA block per (slot, Ga, Gb) block of T3; threads stride over it.
template <bool Pack>
__global__ void move_blocks(const double *__restrict__ source, double *__restrict__ target,
                            int64_t nv, int64_t groups, const int64_t *__restrict__ slot_irrep,
                            const int64_t *__restrict__ block_offset,
                            const int64_t *__restrict__ count, const int64_t *__restrict__ start,
                            const int64_t *__restrict__ order)
{
    const int64_t id = blockIdx.x;
    const int64_t s = id / (groups * groups), ga = (id / groups) % groups, gb = id % groups;
    const int64_t gc = slot_irrep[s] ^ ga ^ gb;
    const int64_t na = count[ga], nb = count[gb], nc = count[gc], size = na * nb * nc;
    const int64_t *va = order + start[ga], *vb = order + start[gb], *vc = order + start[gc];
    const int64_t base = block_offset[id], nv2 = nv * nv, slot = s * nv2 * nv;
    for (int64_t e = threadIdx.x; e < size; e += blockDim.x) {
        const int64_t rc = e % nc, rb = (e / nc) % nb, ra = e / (nc * nb);
        const int64_t dense = slot + va[ra] * nv2 + vb[rb] * nv + vc[rc];
        if (Pack)
            target[base + e] = source[dense];
        else
            target[dense] = source[base + e];
    }
}

}  // namespace

extern "C" int itercc_cuda_t3_pack(const double *dense, int64_t nslots, int64_t nv,
                                   int64_t groups, const int64_t *slot_irrep,
                                   const int64_t *block_offset, const int64_t *count,
                                   const int64_t *start, const int64_t *order, double *blocked,
                                   void *stream)
{
    const int64_t blocks = nslots * groups * groups;
    if (blocks == 0) return 0;
    move_blocks<true><<<unsigned(blocks), 256, 0, cudaStream_t(stream)>>>(
        dense, blocked, nv, groups, slot_irrep, block_offset, count, start, order);
    return int(cudaGetLastError());
}

extern "C" int itercc_cuda_t3_unpack(const double *blocked, int64_t nslots, int64_t nv,
                                     int64_t groups, const int64_t *slot_irrep,
                                     const int64_t *block_offset, const int64_t *count,
                                     const int64_t *start, const int64_t *order, double *dense,
                                     void *stream)
{
    const int64_t blocks = nslots * groups * groups;
    cudaError_t status = cudaMemsetAsync(dense, 0, size_t(nslots) * nv * nv * nv * sizeof(double),
                                         cudaStream_t(stream));
    if (status != cudaSuccess || blocks == 0) return int(status);
    move_blocks<false><<<unsigned(blocks), 256, 0, cudaStream_t(stream)>>>(
        blocked, dense, nv, groups, slot_irrep, block_offset, count, start, order);
    return int(cudaGetLastError());
}
