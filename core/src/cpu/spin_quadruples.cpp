// Direct contractions of canonical, antisymmetric spin-orbital T4/R4.
// Sparse small-operand rows and signed lookup tables replace dense T4 tiles.
#include <algorithm>
#include <cstdlib>
#include <vector>
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

// The direct terms in dense form. For one output row and permutation, a term
// reads T4 at its amplitude with the summed letters running over the
// coefficient: its other amplitude indices (the context) are the row's. Rows
// and permutations with the same context read the same T4 values, so each
// context gathers them once, over the summed tuples of the one symmetry
// class (alpha count and irrep difference) that can complete it to a stored
// amplitude, and every row and permutation using it takes a contiguous dot
// product with its block of the coefficient. Each row then adds its terms in
// order, as itercc_spin_quad_contract_terms does.
namespace {

struct DenseItem {
    uint64_t key;    // the context: amplitude values, 0xFF for summed letters
    int64_t slot;    // row in the slot * nterms + term
    int64_t ext;     // flat index of the external values in the coefficient
};

double quad_lookup(const itercc_spin_quad_family &f, const int64_t *p, const int64_t *h, int64_t no, int64_t nv)
{
    const int64_t v = ((p[0] * nv + p[1]) * nv + p[2]) * nv + p[3];
    const int64_t w = ((h[0] * no + h[1]) * no + h[2]) * no + h[3];
    const int64_t base = f.hole_map[w];
    const int32_t index = f.virtual_map[v];
    if (!base || !index || f.hole_key[w] != f.virtual_key[v]) return 0.0;
    const double x = f.packed[std::abs(base) - 1 + std::abs(index) - 1];
    return (base < 0) != (index < 0) ? -x : x;
}

}  // namespace

extern "C" void itercc_spin_quad_dense(int64_t count, const int32_t *coordinates, int64_t no, int64_t nv,
    const int64_t *occupied_irrep, const int64_t *virtual_irrep, int64_t nfamilies,
    const itercc_spin_quad_family *families, int64_t nterms, const itercc_spin_quad_dense_term *terms,
    double *out, int threads)
{
    // Output slots: runs of rows with the same holes.
    std::vector<int64_t> bounds{0};
    for (int64_t r = 1; r < count; ++r) {
        const int32_t *x = coordinates + 8 * r, *y = x - 8;
        if (x[4] != y[4] || x[5] != y[5] || x[6] != y[6] || x[7] != y[7]) bounds.push_back(r);
    }
    bounds.push_back(count);
    const int64_t nslots = int64_t(bounds.size()) - 1;
    #pragma omp parallel num_threads(threads)
    {
        std::vector<DenseItem> items;
        std::vector<double> z, tvec;
        #pragma omp for schedule(dynamic, 1)
        for (int64_t s = 0; s < nslots; ++s) {
            const int64_t r0 = bounds[size_t(s)], r1 = bounds[size_t(s + 1)], n = r1 - r0;
            z.assign(size_t(n * nterms), 0.0);
            for (int64_t fam = 0; fam < nfamilies; ++fam) {
                const itercc_spin_quad_family &f = families[fam];
                items.clear();
                for (int64_t r = r0; r < r1; ++r) {
                    const int32_t *c = coordinates + 8 * r;
                    for (int64_t t = 0; t < nterms; ++t) {
                        const itercc_spin_quad_dense_term &term = terms[t];
                        if (term.family != fam) continue;
                        uint64_t key = 0;
                        for (int a = 0; a < 8; ++a) {
                            const int64_t axis = term.amplitude_axes[a];
                            key = (key << 8) | (axis >= 0 ? uint64_t(c[axis]) : uint64_t(0xFF));
                        }
                        int64_t ext = 0;
                        for (int64_t e = 0; e < term.nexternal; ++e)
                            ext += c[term.external_axes[e]] * term.external_strides[e];
                        items.push_back(DenseItem{key, (r - r0) * nterms + t, ext});
                    }
                }
                std::sort(items.begin(), items.end(),
                          [](const DenseItem &x, const DenseItem &y) { return x.key < y.key; });
                for (size_t i = 0; i < items.size();) {
                    size_t j = i;
                    while (j < items.size() && items[j].key == items[i].key) ++j;
                    // The context's amplitude values and symmetry class.
                    int64_t value[8], delta = f.spin_change, gamma = f.irrep;
                    for (int a = 0; a < 8; ++a) {
                        value[a] = int64_t((items[i].key >> (8 * (7 - a))) & 0xFF);
                        if (value[a] == 0xFF) continue;
                        if (a < 4) {
                            delta -= value[a] % 2 == 0;
                            gamma ^= virtual_irrep[value[a] / 2];
                        } else {
                            delta += value[a] % 2 == 0;
                            gamma ^= occupied_irrep[value[a] / 2];
                        }
                    }
                    const int64_t cls = (delta - f.class_min) * 8 + gamma;
                    if (cls < 0 || cls >= f.nclasses || f.class_start[cls + 1] == f.class_start[cls]) {
                        i = j;
                        continue;
                    }
                    const int64_t k0 = f.class_start[cls], size = f.class_start[cls + 1] - k0;
                    tvec.resize(size_t(size));
                    for (int64_t k = 0; k < size; ++k) {
                        const int32_t *in = f.internal + 4 * (k0 + k);
                        int64_t p[4], h[4];
                        for (int a = 0; a < 8; ++a) {
                            int64_t x = value[a];
                            if (x == 0xFF) {
                                // summed letter: its index among the internals
                                const itercc_spin_quad_dense_term &term = terms[items[i].slot % nterms];
                                x = in[-1 - term.amplitude_axes[a]];
                            }
                            (a < 4 ? p[a] : h[a - 4]) = x;
                        }
                        tvec[size_t(k)] = quad_lookup(f, p, h, no, nv);
                    }
                    const double *wblock = f.weights + f.weight_offset[cls];
                    for (size_t m = i; m < j; ++m) {
                        const double *w = wblock + items[m].ext * size;
                        double sum = 0.0;
                        for (int64_t k = 0; k < size; ++k) sum += w[k] * tvec[size_t(k)];
                        z[size_t(items[m].slot)] = sum;
                    }
                    i = j;
                }
            }
            for (int64_t r = 0; r < n; ++r) {
                double value = out[r0 + r];
                for (int64_t t = 0; t < nterms; ++t) value += terms[t].alpha * z[size_t(r * nterms + t)];
                out[r0 + r] = value;
            }
        }
    }
}
