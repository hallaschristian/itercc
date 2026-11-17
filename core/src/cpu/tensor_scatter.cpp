// Shared deterministic tensor expansion; each destination has one owner.
#include "itercc/core.h"

extern "C" void itercc_signed_gather_rows(int64_t rows, int64_t width,
    const int64_t *bases, const int64_t *maps, const double *source,
    double *target, int threads)
{
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t i = 0; i < rows * width; ++i) {
        const int64_t row = i / width, base = bases[row];
        double value = 0.;
        if (base) {
            const int32_t *map = reinterpret_cast<const int32_t *>(static_cast<intptr_t>(maps[row]));
            const int32_t index = map[i % width];
            if (index) {
                const int64_t offset = (base < 0 ? -base : base) - 1
                                     + (index < 0 ? -int64_t(index) : index) - 1;
                value = source[offset];
                if ((base < 0) != (index < 0)) value = -value;
            }
        }
        target[i] = value;
    }
}

extern "C" void itercc_permuted_gather_add(int64_t count, int rank, int terms,
    const int32_t *coordinates, const int64_t *rows, const int64_t *strides,
    const double *weights, const double *source, double *target, int threads)
{
    // One thread owns each distinct output row; preserve term order exactly.
    auto gather = [&](int64_t row) {
        double sum = 0.;
        for (int term = 0; term < terms; ++term) {
            int64_t offset = 0;
            for (int axis = 0; axis < rank; ++axis)
                offset += coordinates[row * rank + axis] * strides[term * rank + axis];
            sum += weights[term] * source[offset];
        }
        target[rows[row]] += sum;
    };
    // Do not resize the OpenMP team between small gathers and the surrounding
    // GEMMs: that destroys worker reuse and can dominate the contraction time.
    if (count * rank * terms < 262144) {
        for (int64_t row = 0; row < count; ++row) gather(row);
    } else {
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int64_t row = 0; row < count; ++row) gather(row);
    }
}

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

