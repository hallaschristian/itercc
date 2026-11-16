// CUDA backend skeleton: library identity, device query and a cuBLAS self-test.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "itercc/cuda.h"

#define ITERCC_STRING2(x) #x
#define ITERCC_STRING(x) ITERCC_STRING2(x)

namespace {

__global__ void naive_gemm(int64_t n, const double *a, const double *b, double *c)
{
    // Column-major, as cuBLAS: C(i,j) = sum_k A(i,k) B(k,j).
    const int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
    const int64_t j = blockIdx.y;
    if (i >= n) return;
    double sum = 0.0;
    for (int64_t k = 0; k < n; ++k)
        sum += a[i + k * n] * b[k + j * n];
    c[i + j * n] = sum;
}

struct DeviceBuffer {
    double *data = nullptr;
    ~DeviceBuffer() { if (data) cudaFree(data); }
};

}  // namespace

extern "C" int itercc_cuda_abi(void) { return ITERCC_CUDA_ABI; }

extern "C" const char *itercc_cuda_build(void)
{
    return "nvcc " ITERCC_STRING(__CUDACC_VER_MAJOR__) "." ITERCC_STRING(__CUDACC_VER_MINOR__);
}

extern "C" int itercc_cuda_device_count(int *count)
{
    return int(cudaGetDeviceCount(count));
}

extern "C" int itercc_cuda_gemm_selftest(int64_t n, double *max_error)
{
    const size_t size = size_t(n) * size_t(n);
    std::vector<double> a(size), b(size), c1(size), c2(size);
    for (size_t i = 0; i < size; ++i) {
        a[i] = std::sin(0.37 * double(i));
        b[i] = std::cos(0.11 * double(i));
    }
    DeviceBuffer da, db, dc1, dc2;
    cudaError_t status = cudaSuccess;
    for (DeviceBuffer *buffer : {&da, &db, &dc1, &dc2}) {
        status = cudaMalloc(&buffer->data, size * sizeof(double));
        if (status != cudaSuccess) return int(status);
    }
    cudaMemcpy(da.data, a.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(db.data, b.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    cublasHandle_t handle;
    cublasStatus_t blas = cublasCreate(&handle);
    if (blas != CUBLAS_STATUS_SUCCESS) return 1000 + int(blas);
    const double one = 1.0, zero = 0.0;
    blas = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, int(n), int(n), int(n), &one, da.data,
                       int(n), db.data, int(n), &zero, dc1.data, int(n));
    cublasDestroy(handle);
    if (blas != CUBLAS_STATUS_SUCCESS) return 1000 + int(blas);
    naive_gemm<<<dim3(unsigned((n + 127) / 128), unsigned(n)), 128>>>(n, da.data, db.data, dc2.data);
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) return int(status);
    cudaMemcpy(c1.data(), dc1.data, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(c2.data(), dc2.data, size * sizeof(double), cudaMemcpyDeviceToHost);
    double worst = 0.0;
    for (size_t i = 0; i < size; ++i)
        worst = std::fmax(worst, std::fabs(c1[i] - c2[i]));
    *max_error = worst;
    return int(cudaGetLastError());
}
