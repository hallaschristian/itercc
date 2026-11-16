/* IterCC compiled CUDA backend (libitercc_cuda): stable C interface.
 * Kept in a separate library so CPU-only nodes never need CUDA libraries.
 * Functions return 0 on success or a CUDA/cuBLAS status code. */
#ifndef ITERCC_CUDA_H
#define ITERCC_CUDA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ITERCC_CUDA_ABI 2

int itercc_cuda_abi(void);
const char *itercc_cuda_build(void);
int itercc_cuda_device_count(int *count);

/* Toolchain check: C = A B for n x n matrices with cuBLAS and with a plain
 * kernel on the current device; max_error receives the largest difference. */
int itercc_cuda_gemm_selftest(int64_t n, double *max_error);

/* Symmetry-blocked restricted T3, the layout of itercc/_blocked.py. Tables
 * and arrays are device memory; work is queued on `stream` (cudaStream_t). */
int itercc_cuda_t3_pack(const double *dense, int64_t nslots, int64_t nv, int64_t groups,
                        const int64_t *slot_irrep, const int64_t *block_offset,
                        const int64_t *count, const int64_t *start, const int64_t *order,
                        double *blocked, void *stream);
int itercc_cuda_t3_unpack(const double *blocked, int64_t nslots, int64_t nv, int64_t groups,
                          const int64_t *slot_irrep, const int64_t *block_offset,
                          const int64_t *count, const int64_t *start, const int64_t *order,
                          double *dense, void *stream);

/* Row-major cuBLAS GEMMs on `stream` (device a, b, c). The grouped form takes
 * host descriptors as itercc_grouped_gemm in core.h. Return 0, a CUDA status,
 * 1000 + a cuBLAS status, or 2 if a size exceeds int32. */
int itercc_cuda_gemm(int transa, int transb, int64_t m, int64_t n, int64_t k, double alpha,
                     const double *a, int64_t lda, const double *b, int64_t ldb, double beta,
                     double *c, int64_t ldc, void *stream);
int itercc_cuda_grouped_gemm(int64_t count, const int64_t *desc, const double *coef,
                             const double *a, const double *b, double *c, void *stream);

#ifdef __cplusplus
}
#endif

#endif
