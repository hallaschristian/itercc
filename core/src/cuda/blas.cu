// cuBLAS dispatch for the GPU core: row-major GEMMs on a caller's stream.
// One cuBLAS handle per device is created on first use and kept.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <climits>
#include <mutex>

#include "itercc/cuda.h"

namespace {

constexpr int kMaxDevices = 64;
struct DeviceHandle {
    cublasHandle_t handle = nullptr;
    std::mutex mutex;
};
DeviceHandle handles[kMaxDevices];

bool fits(int64_t value) { return value >= 0 && value <= INT_MAX; }

int handle_for(cudaStream_t stream, cublasHandle_t *handle, std::unique_lock<std::mutex> &lock)
{
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return int(status);
    if (device < 0 || device >= kMaxDevices) return 3;
    auto &entry = handles[device];
    // Keep setStream and all subsequent enqueues together on a shared handle.
    lock = std::unique_lock<std::mutex>(entry.mutex);
    if (entry.handle == nullptr) {
        cublasStatus_t created = cublasCreate(&entry.handle);
        if (created != CUBLAS_STATUS_SUCCESS) return 1000 + int(created);
    }
    *handle = entry.handle;
    cublasStatus_t set = cublasSetStream(*handle, stream);
    return set == CUBLAS_STATUS_SUCCESS ? 0 : 1000 + int(set);
}

int row_major_gemm(cublasHandle_t handle, int transa, int transb, int64_t m, int64_t n,
                   int64_t k, double alpha, const double *a, int64_t lda, const double *b,
                   int64_t ldb, double beta, double *c, int64_t ldc)
{
    if (m == 0 || n == 0) return 0;
    if (!(fits(m) && fits(n) && fits(k) && fits(lda) && fits(ldb) && fits(ldc))) return 2;
    // Column-major C^T (n x m) = op(B)^T op(A)^T.
    const cublasOperation_t ta = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t tb = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasStatus_t status = cublasDgemm(
        handle, tb, ta, int(n), int(m), int(k), &alpha, b, int(ldb > 0 ? ldb : 1), a,
        int(lda > 0 ? lda : 1), &beta, c, int(ldc));
    return status == CUBLAS_STATUS_SUCCESS ? 0 : 1000 + int(status);
}

}  // namespace

extern "C" int itercc_cuda_gemm(int transa, int transb, int64_t m, int64_t n, int64_t k,
                                double alpha, const double *a, int64_t lda, const double *b,
                                int64_t ldb, double beta, double *c, int64_t ldc, void *stream)
{
    cublasHandle_t handle;
    std::unique_lock<std::mutex> lock;
    const int status = handle_for(cudaStream_t(stream), &handle, lock);
    if (status) return status;
    return row_major_gemm(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

// desc[11 g ..] = (transa, transb, m, n, k, a_offset, lda, b_offset, ldb, c_offset, ldc) on the
// host, coef[2 g ..] = (alpha, beta) on the host; a, b, c are device base pointers.
extern "C" int itercc_cuda_grouped_gemm(int64_t count, const int64_t *desc, const double *coef,
                                        const double *a, const double *b, double *c,
                                        void *stream)
{
    cublasHandle_t handle;
    std::unique_lock<std::mutex> lock;
    int status = handle_for(cudaStream_t(stream), &handle, lock);
    if (status) return status;
    for (int64_t g = 0; g < count; ++g) {
        const int64_t *d = desc + 11 * g;
        status = row_major_gemm(handle, int(d[0]), int(d[1]), d[2], d[3], d[4], coef[2 * g],
                                a + d[5], d[6], b + d[7], d[8], coef[2 * g + 1], c + d[9], d[10]);
        if (status) return status;
    }
    return 0;
}
