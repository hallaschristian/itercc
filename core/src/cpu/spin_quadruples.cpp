// Direct contractions of canonical, antisymmetric spin-orbital T4/R4.
// Sparse small-operand rows and signed lookup tables replace dense T4 tiles.
#include <algorithm>
#include <cstdlib>
#include <vector>
#include "itercc/core.h"
#include "gemm_kernel.h"

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
// product with its block of the coefficient. Contexts of one symmetry class
// share that block, so they go together into one GEMM (the coefficient rows
// of their external values times their gathered T4 columns) when most of
// its entries are wanted, instead of a coefficient row streamed per item.
// Each row then adds its terms in order, as itercc_spin_quad_contract_terms
// does.
namespace {

struct DenseItem {
    uint64_t key;    // the context: amplitude values, 0xFF for summed letters
    int64_t slot;    // row in the slot * nterms + term
    int64_t ext;     // flat index of the external values in the coefficient
};

struct Context {
    int64_t cls;         // symmetry class of the summed tuples
    int64_t begin, end;  // its items
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
        std::vector<Context> contexts;
        std::vector<double> z, tmat, product;
        std::vector<int64_t> rows, ar, ak, bc, cc, cr;
        int64_t most_ext = 1;
        for (int64_t fam = 0; fam < nfamilies; ++fam) most_ext = std::max(most_ext, families[fam].n_ext);
        std::vector<int64_t> row_of(size_t(most_ext), -1);
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
                // Contexts (runs of equal keys) with their symmetry classes.
                contexts.clear();
                for (size_t i = 0; i < items.size();) {
                    size_t j = i;
                    while (j < items.size() && items[j].key == items[i].key) ++j;
                    int64_t delta = f.spin_change, gamma = f.irrep;
                    for (int a = 0; a < 8; ++a) {
                        const int64_t v = int64_t((items[i].key >> (8 * (7 - a))) & 0xFF);
                        if (v == 0xFF) continue;
                        if (a < 4) {
                            delta -= v % 2 == 0;
                            gamma ^= virtual_irrep[v / 2];
                        } else {
                            delta += v % 2 == 0;
                            gamma ^= occupied_irrep[v / 2];
                        }
                    }
                    const int64_t cls = (delta - f.class_min) * 8 + gamma;
                    if (cls >= 0 && cls < f.nclasses && f.class_start[cls + 1] > f.class_start[cls])
                        contexts.push_back(Context{cls, int64_t(i), int64_t(j)});
                    i = j;
                }
                std::stable_sort(contexts.begin(), contexts.end(),
                                 [](const Context &x, const Context &y) { return x.cls < y.cls; });
                for (size_t g = 0; g < contexts.size();) {
                    size_t h = g;
                    while (h < contexts.size() && contexts[h].cls == contexts[g].cls) ++h;
                    const int64_t cls = contexts[g].cls, k0 = f.class_start[cls];
                    const int64_t size = f.class_start[cls + 1] - k0, ncontexts = int64_t(h - g);
                    const double *wblock = f.weights + f.weight_offset[cls];
                    // Each context's T4 values over the class's summed tuples: column c.
                    tmat.resize(size_t(size * ncontexts));
                    int64_t wanted = 0;
                    for (int64_t c = 0; c < ncontexts; ++c) {
                        const Context &x = contexts[g + size_t(c)];
                        wanted += x.end - x.begin;
                        const DenseItem &first = items[size_t(x.begin)];
                        const itercc_spin_quad_dense_term &term = terms[first.slot % nterms];
                        double *column = tmat.data() + c * size;
                        for (int64_t k = 0; k < size; ++k) {
                            const int32_t *in = f.internal + 4 * (k0 + k);
                            int64_t p[4], q[4];
                            for (int a = 0; a < 8; ++a) {
                                int64_t v = int64_t((first.key >> (8 * (7 - a))) & 0xFF);
                                if (v == 0xFF) v = in[-1 - term.amplitude_axes[a]];   // a summed letter
                                (a < 4 ? p[a] : q[a - 4]) = v;
                            }
                            column[k] = quad_lookup(f, p, q, no, nv);
                        }
                    }
                    // The distinct external values of the class's items: the GEMM rows.
                    rows.clear();
                    for (int64_t c = 0; c < ncontexts; ++c) {
                        const Context &x = contexts[g + size_t(c)];
                        for (int64_t m = x.begin; m < x.end; ++m) {
                            const int64_t e = items[size_t(m)].ext;
                            if (row_of[size_t(e)] < 0) {
                                row_of[size_t(e)] = int64_t(rows.size());
                                rows.push_back(e);
                            }
                        }
                    }
                    const int64_t nrows = int64_t(rows.size());
                    if (size >= 8 && nrows * ncontexts <= 2 * wanted) {
                        ar.resize(size_t(nrows));
                        cr.resize(size_t(nrows));
                        for (int64_t r = 0; r < nrows; ++r) {
                            ar[size_t(r)] = rows[size_t(r)] * size;
                            cr[size_t(r)] = r * ncontexts;
                        }
                        ak.resize(size_t(size));
                        for (int64_t k = 0; k < size; ++k) ak[size_t(k)] = k;
                        bc.resize(size_t(ncontexts));
                        cc.resize(size_t(ncontexts));
                        for (int64_t c = 0; c < ncontexts; ++c) {
                            bc[size_t(c)] = c * size;
                            cc[size_t(c)] = c;
                        }
                        product.assign(size_t(nrows * ncontexts), 0.0);
                        itercc::IndirectGemm gemm{nrows, ncontexts, size, wblock, ar.data(), ak.data(), false,
                                                  tmat.data(), ak.data(), bc.data(), true,
                                                  product.data(), cr.data(), cc.data(), 1.0};
                        itercc::indirect_gemm(gemm);
                        for (int64_t c = 0; c < ncontexts; ++c) {
                            const Context &x = contexts[g + size_t(c)];
                            for (int64_t m = x.begin; m < x.end; ++m)
                                z[size_t(items[size_t(m)].slot)] =
                                    product[size_t(row_of[size_t(items[size_t(m)].ext)] * ncontexts + c)];
                        }
                    } else {                     // few wanted entries: a dot product per item
                        for (int64_t c = 0; c < ncontexts; ++c) {
                            const Context &x = contexts[g + size_t(c)];
                            const double *column = tmat.data() + c * size;
                            for (int64_t m = x.begin; m < x.end; ++m) {
                                const double *w = wblock + items[size_t(m)].ext * size;
                                double sum = 0.0;
                                for (int64_t k = 0; k < size; ++k) sum += w[k] * column[k];
                                z[size_t(items[size_t(m)].slot)] = sum;
                            }
                        }
                    }
                    for (int64_t e : rows) row_of[size_t(e)] = -1;
                    g = h;
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
