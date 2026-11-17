// Shared deterministic tensor expansion; each destination has one owner.
#include "itercc/core.h"

extern "C" void itercc_sector_unpack(int64_t size, int64_t count,
    const int64_t *indices, const double *vector, double *out, int threads)
{
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < size; ++i) out[i] = 0.;
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < count; ++i) out[indices[i]] = vector[i];
    }
}

extern "C" void itercc_spin_unpack(int64_t count, int rank, int64_t nv, int64_t no,
    const int32_t *coordinates, const double *vector, int npermutations,
    const int32_t *permutations, const int32_t *signs, double *out, int threads)
{
    int64_t size = 1;
    for (int a = 0; a < rank; ++a) size *= nv * no;
    // Canonical coordinates have distinct particle and hole indices. Every
    // permuted destination has exactly one owner, so no atomics are needed.
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < size; ++i) out[i] = 0.;
        for (int p = 0; p < npermutations; ++p) {
            const int32_t *permutation = permutations + p * 2 * rank;
            #pragma omp for schedule(static)
            for (int64_t row = 0; row < count; ++row) {
                const int32_t *coordinate = coordinates + row * 2 * rank;
                int64_t offset = 0;
                for (int a = 0; a < 2 * rank; ++a)
                    offset = offset * (a < rank ? nv : no) + coordinate[permutation[a]];
                out[offset] = signs[p] * vector[row];
            }
        }
    }
}

