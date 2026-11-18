// Direct contractions of canonical, antisymmetric spin-orbital T4/R4.
// Sparse small-operand rows and signed lookup tables replace dense T4 tiles.
#include <cstdlib>
#include "itercc/core.h"

extern "C" void itercc_spin_quad_contract(
    int64_t count, int rank, const int32_t *coordinates,
    int64_t no, int64_t nv, const int32_t *virtual_map, const uint8_t *virtual_key,
    const int64_t *hole_map, const uint8_t *hole_key, const double *packed,
    int nexternal, const int32_t *external_axes, const int32_t *external_extents,
    const int32_t *amplitude_axes, const int64_t *starts, const int32_t *internal,
    const double *values, double alpha, double *out, int threads)
{
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t row = 0; row < count; ++row) {
        const int32_t *coordinate = coordinates + row * (2 * rank);
        int64_t group = 0;
        for (int e = 0; e < nexternal; ++e)
            group = group * external_extents[e] + coordinate[external_axes[e]];
        // Each address is affine in the sparse coefficient's internal indices.
        // Factor its output-coordinate contribution outside the summation.
        int64_t vb = 0, hb = 0, vs[4] = {}, hs[4] = {};
        for (int a = 0; a < 4; ++a) {
            vb *= nv; hb *= no;
            for (int b = 0; b < 4; ++b) { vs[b] *= nv; hs[b] *= no; }
            int p = amplitude_axes[a], h = amplitude_axes[a + 4];
            if (p >= 0) vb += coordinate[p]; else vs[-p - 1] += 1;
            if (h >= 0) hb += coordinate[h]; else hs[-h - 1] += 1;
        }
        double total = 0;
        for (int64_t t = starts[group]; t < starts[group + 1]; ++t) {
            int64_t v = vb, h = hb;
            for (int b = 0; b < 4; ++b) {
                v += vs[b] * internal[4 * t + b];
                h += hs[b] * internal[4 * t + b];
            }
            const int64_t base = hole_map[h];
            const int32_t index = virtual_map[v];
            if (base && index && hole_key[h] == virtual_key[v]) {
                const double value = packed[std::abs(base) - 1 + std::abs(index) - 1];
                total += values[t] * ((base < 0) != (index < 0) ? -value : value);
            }
        }
        out[row] += alpha * total;
    }
}

// Many terms of itercc_spin_quad_contract in one pass over the rows: each row
// takes the terms in order, adding alpha * total after each as a separate
// call would, so the results are those of one call per term (bit for bit),
// without a pass over the coordinates and outputs and a parallel region per
// term.
extern "C" void itercc_spin_quad_contract_terms(int64_t count, int rank, const int32_t *coordinates,
    int64_t no, int64_t nv, int64_t nterms, const itercc_spin_quad_term *terms, double *out, int threads)
{
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t row = 0; row < count; ++row) {
        const int32_t *coordinate = coordinates + row * (2 * rank);
        double value = out[row];
        for (int64_t k = 0; k < nterms; ++k) {
            const itercc_spin_quad_term &term = terms[k];
            int64_t group = 0;
            for (int64_t e = 0; e < term.nexternal; ++e)
                group = group * term.external_extents[e] + coordinate[term.external_axes[e]];
            int64_t vb = 0, hb = 0, vs[4] = {}, hs[4] = {};
            for (int a = 0; a < 4; ++a) {
                vb *= nv; hb *= no;
                for (int b = 0; b < 4; ++b) { vs[b] *= nv; hs[b] *= no; }
                const int64_t p = term.amplitude_axes[a], h = term.amplitude_axes[a + 4];
                if (p >= 0) vb += coordinate[p]; else vs[-p - 1] += 1;
                if (h >= 0) hb += coordinate[h]; else hs[-h - 1] += 1;
            }
            double total = 0;
            for (int64_t t = term.starts[group]; t < term.starts[group + 1]; ++t) {
                int64_t v = vb, h = hb;
                for (int b = 0; b < 4; ++b) {
                    v += vs[b] * term.internal[4 * t + b];
                    h += hs[b] * term.internal[4 * t + b];
                }
                const int64_t base = term.hole_map[h];
                const int32_t index = term.virtual_map[v];
                if (base && index && term.hole_key[h] == term.virtual_key[v]) {
                    const double x = term.packed[std::abs(base) - 1 + std::abs(index) - 1];
                    total += term.values[t] * ((base < 0) != (index < 0) ? -x : x);
                }
            }
            value += term.alpha * total;
        }
        out[row] = value;
    }
}
