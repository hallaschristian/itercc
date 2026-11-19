// Shared deterministic tensor expansion; each destination has one owner.
#include "itercc/core.h"

#include <algorithm>

extern "C" void itercc_signed_gather_rows(int64_t rows, int64_t width,
    const int64_t *bases, const int64_t *maps, const double *source,
    double *target, int threads)
{
    // Tasks are fixed column chunks of one row: no division per entry, and a
    // row without a base (a repeated hole) is a plain zero fill. The chunking
    // does not depend on the thread count, and each entry has one writer.
    constexpr int64_t chunk = 8192;
    const int64_t per_row = (width + chunk - 1) / chunk, tasks = rows * per_row;
    auto run = [&](int64_t task) {
        const int64_t row = task / per_row, c0 = task % per_row * chunk;
        const int64_t c1 = std::min(width, c0 + chunk);
        double *out = target + row * width;
        const int64_t base = bases[row];
        if (!base) {
            std::fill(out + c0, out + c1, 0.);
            return;
        }
        const int32_t *map = reinterpret_cast<const int32_t *>(static_cast<intptr_t>(maps[row]));
        // source[(|base| - 1) + (|index| - 1)], negated when exactly one is negative
        const double *from = source + ((base < 0 ? -base : base) - 1);
        const double sign = base < 0 ? -1. : 1.;
        for (int64_t c = c0; c < c1; ++c) {
            const int32_t index = map[c];
            out[c] = index > 0 ? sign * from[index - 1] : index < 0 ? -sign * from[-int64_t(index) - 1] : 0.;
        }
    };
    // Small gathers stay on the caller (see itercc_permuted_gather_add on team sizes).
    if (rows * width < 32768) {
        for (int64_t task = 0; task < tasks; ++task) run(task);
    } else {
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int64_t task = 0; task < tasks; ++task) run(task);
    }
}

namespace {

// Rows [r0, r1) of a permuted gather: term by term over the rows, with the
// term's strides hoisted (Rank > 0: a fixed rank), into sums (one per row,
// from +0.0); then each row's sum is added to its target. Every row gets the
// same additions in the same order as a row-by-row loop over the terms.
template <int Rank>
void gather_rows(int64_t r0, int64_t r1, int rank, int terms, const int32_t *coordinates, const int64_t *rows,
                 const int64_t *strides, const double *weights, const double *source, double *target,
                 double *sums)
{
    const int n = Rank > 0 ? Rank : rank;
    std::fill(sums, sums + (r1 - r0), 0.);
    for (int term = 0; term < terms; ++term) {
        const int64_t *step = strides + int64_t(term) * n;
        const double w = weights[term];
        for (int64_t row = r0; row < r1; ++row) {
            const int32_t *c = coordinates + row * n;
            int64_t offset = 0;
            for (int axis = 0; axis < n; ++axis) offset += c[axis] * step[axis];
            sums[row - r0] += w * source[offset];
        }
    }
    for (int64_t row = r0; row < r1; ++row) target[rows[row]] += sums[row - r0];
}

void gather_rows_any(int64_t r0, int64_t r1, int rank, int terms, const int32_t *coordinates, const int64_t *rows,
                     const int64_t *strides, const double *weights, const double *source, double *target,
                     double *sums)
{
    switch (rank) {
    case 2: return gather_rows<2>(r0, r1, rank, terms, coordinates, rows, strides, weights, source, target, sums);
    case 4: return gather_rows<4>(r0, r1, rank, terms, coordinates, rows, strides, weights, source, target, sums);
    case 6: return gather_rows<6>(r0, r1, rank, terms, coordinates, rows, strides, weights, source, target, sums);
    case 8: return gather_rows<8>(r0, r1, rank, terms, coordinates, rows, strides, weights, source, target, sums);
    default: return gather_rows<0>(r0, r1, rank, terms, coordinates, rows, strides, weights, source, target, sums);
    }
}

}  // namespace

extern "C" void itercc_permuted_gather_add(int64_t count, int rank, int terms,
    const int32_t *coordinates, const int64_t *rows, const int64_t *strides,
    const double *weights, const double *source, double *target, int threads)
{
    // One thread owns each distinct output row; every row's terms are added
    // in order. Rows go in fixed blocks (term by term within a block: the
    // term's strides are set up once per block, not once per row and term).
    constexpr int64_t block = 2048;
    const int64_t blocks = (count + block - 1) / block;
    auto run = [&](int64_t b, double *sums) {
        const int64_t r0 = b * block, r1 = std::min(count, r0 + block);
        gather_rows_any(r0, r1, rank, terms, coordinates, rows, strides, weights, source, target, sums);
    };
    // Do not resize the OpenMP team between small gathers and the surrounding
    // GEMMs: that destroys worker reuse and can dominate the contraction time.
    if (count * rank * terms < 262144) {
        double sums[block];
        for (int64_t b = 0; b < blocks; ++b) run(b, sums);
    } else {
        #pragma omp parallel num_threads(threads)
        {
            double sums[block];
            #pragma omp for schedule(static)
            for (int64_t b = 0; b < blocks; ++b) run(b, sums);
        }
    }
}

extern "C" void itercc_spin_unpack_blocks(int64_t count, int rank, const int32_t *coordinates,
    const double *vector, int npermutations, const int32_t *permutations, const int32_t *signs,
    const int64_t *blocks, const int64_t *strides, int64_t block_size, int threads)
{
    const int axes = 2 * rank, nblocks = 1 << axes;
    #pragma omp parallel num_threads(threads)
    {
        for (int b = 0; b < nblocks; ++b) {
            double *block = reinterpret_cast<double *>(static_cast<intptr_t>(blocks[b]));
            if (block == nullptr) continue;
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < block_size; ++i) block[i] = 0.;
        }
        for (int p = 0; p < npermutations; ++p) {
            const int32_t *permutation = permutations + p * axes;
            #pragma omp for schedule(static)
            for (int64_t row = 0; row < count; ++row) {
                const int32_t *coordinate = coordinates + row * axes;
                int64_t signature = 0, offset = 0;
                for (int a = 0; a < axes; ++a) {
                    const int32_t index = coordinate[permutation[a]];
                    signature = 2 * signature + (index & 1);
                    offset += int64_t(index >> 1) * strides[a];
                }
                double *block = reinterpret_cast<double *>(static_cast<intptr_t>(blocks[signature]));
                block[offset] = signs[p] * vector[row];
            }
        }
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

