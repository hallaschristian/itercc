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

// The direct terms by opened letters (see core.h). Units are hole tuples J;
// each unit gathers its amplitude block Y once per class of the row's other
// particles B (a slot's T4 values for every summed letter), multiplies it with
// the coefficient block in one GEMM, and adds the products into every target
// slot J + x_h of this process.
//
// Diagnostic thread time of both kernels by phase (build with
// -DITERCC_PROFILE_OPENED; itercc_opened_profile prints and resets it).
#ifdef ITERCC_PROFILE_OPENED
#include <atomic>
#include <cstdio>
#include <omp.h>
namespace {
std::atomic<double> opened_time[8];
inline void opened_add(int i, double &mark)
{
    const double now = omp_get_wtime();
    double old = opened_time[i].load();
    while (!opened_time[i].compare_exchange_weak(old, old + now - mark)) {
    }
    mark = now;
}
}  // namespace
extern "C" void itercc_opened_profile()
{
    const char *names[8] = {"targets", "numbering", "Y gather", "GEMM", "scatter", "kp0 gemm", "kp0 add", "-"};
    for (int i = 0; i < 8; ++i)
        std::fprintf(stderr, "  opened %-10s %8.2f thread-s\n", names[i], opened_time[i].load());
    for (int i = 0; i < 8; ++i) opened_time[i] = 0.0;
}
#define OPENED_MARK double opened_mark = omp_get_wtime()
#define OPENED_ADD(i) opened_add((i), opened_mark)
#else
#define OPENED_MARK (void)0
#define OPENED_ADD(i) (void)0
#endif
namespace {

inline int64_t binomial4(int64_t k) { return k == 0 ? 1 : k == 1 ? 4 : 6; }

// Index of a set of hole positions (kh of 0..3) among the C(4, kh) sets.
inline int64_t pattern_of(int64_t kh, const int64_t *beta)
{
    if (kh == 0) return 0;
    if (kh == 1) return beta[0];
    static const int64_t pairs[4][4] = {{-1, 0, 1, 2}, {0, -1, 3, 4}, {1, 3, -1, 5}, {2, 4, 5, -1}};
    return pairs[beta[0]][beta[1]];
}

struct OpenedTarget {
    int64_t flat;          // x_h flat index
    int64_t key;           // particle group
    double *out;           // the target's rows in the output or in its buffer
    int64_t beta[2];       // positions of x_h in the slot (labels order)
};

// Y[i][k] (i < nb, k < K): the amplitude of slot hk[k] (signed offset + 1, 0
// for none) at the position of row i of `rows` (npf signed positions + 1 per
// B), column pk[k]. The tuples' slots and particle columns are looked up once
// per class, not once per B, and each row's reads (scattered over up to
// hundreds of slots) are prefetched kAhead tuples ahead.
inline void gather_y(int64_t nb, int64_t K, int64_t npf, const int32_t *rows, const int64_t *hk, const int64_t *pk,
                     const double *vector, double *y)
{
    constexpr int64_t kAhead = 16;
    for (int64_t i = 0; i < nb; ++i) {
        const int32_t *vrow = rows + i * npf;
        double *yi = y + i * K;
        for (int64_t k = 0; k < K; ++k) {
            if (k + kAhead < K) {
                const int64_t hn = hk[k + kAhead];
                const int32_t vn = vrow[pk[k + kAhead]];
                if (hn && vn) __builtin_prefetch(vector + (std::abs(hn) - 1 + std::abs(vn) - 1));
            }
            const int64_t h = hk[k];
            const int32_t v = vrow[pk[k]];
            if (!h || !v) {
                yi[k] = 0.0;
                continue;
            }
            const double value = vector[std::abs(h) - 1 + std::abs(v) - 1];
            yi[k] = (h < 0) != (v < 0) ? -value : value;
        }
    }
}

inline int64_t opened_alpha(const int32_t *v, int64_t n)
{
    int64_t a = 0;
    for (int64_t i = 0; i < n; ++i) a += v[i] % 2 == 0;
    return a;
}

inline int64_t opened_irrep(const int64_t *table, const int32_t *v, int64_t n)
{
    int64_t g = 0;
    for (int64_t i = 0; i < n; ++i) g ^= table[v[i] / 2];
    return g;
}

}  // namespace

extern "C" void itercc_spin_quad_opened(const itercc_spin_quad_opened_layout *layout,
    const itercc_spin_quad_opened_family *family, int64_t nunits, const int32_t *units,
    double *out, double *buffers, int threads)
{
    const itercc_spin_quad_opened_layout &L = *layout;
    const itercc_spin_quad_opened_family &F = *family;
    const int64_t kp = F.kp, kh = F.kh, no = L.no, nv = L.nv, nj = 4 - kh;
    const int64_t nbuffers = binomial4(kh) - 1;
    const int64_t nh = kh == 0 ? 1 : kh == 1 ? no : no * no;    // flat hole tuples (s_h or x_h)
    const int64_t npf = kp == 0 ? 1 : kp == 1 ? nv : nv * nv;   // flat particle tuples (s_p or x_p)
    const int64_t nxflat = npf * nh;
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t i = 0; i < nbuffers * L.buffer_rows; ++i) buffers[i] = 0.0;
    // Terms by the set of their x_h positions; B classes by B index.
    std::vector<std::vector<int64_t>> by_pattern(size_t(nbuffers + 1));
    for (int64_t t = 0; t < F.nterms; ++t) by_pattern[size_t(pattern_of(kh, F.term_beta + 2 * t))].push_back(t);
    std::vector<int8_t> class_of_b;
    if (kp > 0) {
        class_of_b.resize(size_t(F.b_start[40]));
        for (int64_t cb = 0; cb < 40; ++cb)
            for (int64_t g = F.b_start[cb]; g < F.b_start[cb + 1]; ++g) class_of_b[size_t(g)] = int8_t(cb);
    }
    #pragma omp parallel num_threads(threads)
    {
        std::vector<int64_t> hvec(static_cast<size_t>(nh)), zoff(40), znb(40), zc(40), rows_m, cols, ar, ak, bk, bc, cr, cc,
            hk, pk;
        std::vector<int32_t> m_of_x(size_t(nxflat), -1);
        std::vector<int64_t> touched, target_of(size_t(nh), -1);
        std::vector<double> y, z, a, coefs;
        std::vector<double *> dests;
        std::vector<OpenedTarget> targets;
        #pragma omp for schedule(dynamic, 1)
        for (int64_t u = 0; u < nunits; ++u) {
            OPENED_MARK;
            const int32_t *J = units + 4 * u;
            const int64_t alphaJ = opened_alpha(J, nj), irrepJ = opened_irrep(L.occupied_irrep, J, nj);
            // Amplitude slots J + s_h (signed offset + 1) and this process's target slots J + x_h.
            targets.clear();
            for (int64_t f = 0; f < nh; ++f) {
                int32_t holes[4];
                for (int64_t i = 0; i < nj; ++i) holes[i] = J[i];
                if (kh == 1) holes[3] = int32_t(f);
                if (kh == 2) {
                    holes[2] = int32_t(f / no);
                    holes[3] = int32_t(f % no);
                }
                const int64_t w = ((int64_t(holes[0]) * no + holes[1]) * no + holes[2]) * no + holes[3];
                hvec[size_t(f)] = F.hole_map[w];
                target_of[size_t(f)] = -1;
                const int64_t own = L.own[w];
                if (!own) continue;
                OpenedTarget target;
                target.flat = f;
                target.beta[0] = target.beta[1] = 0;
                for (int64_t i = 0; i < kh; ++i) {
                    int64_t pos = 0;
                    for (int64_t j = 0; j < 4; ++j) pos += j != nj + i && holes[j] < holes[nj + i];
                    target.beta[i] = pos;
                }
                const int64_t b = pattern_of(kh, target.beta);
                target.out = b == 0 ? out + (std::abs(L.hole_map[w]) - 1)
                                    : buffers + (b - 1) * L.buffer_rows + (own - 1);
                const int64_t ka = opened_alpha(holes, 4) + L.spin_change;
                const int64_t kg = opened_irrep(L.occupied_irrep, holes, 4) ^ L.irrep;
                target.key = ka >= 0 && ka <= 4 ? L.key_of[ka * 8 + kg] : -1;
                if (target.key < 0) continue;
                target_of[size_t(f)] = int64_t(targets.size());
                targets.push_back(target);
            }
            OPENED_ADD(0);
            if (targets.empty()) continue;
            if (kp == 0) {
                // B is the row: Z[x, rows] = sum_s W'[x, s] T(slot J + s_h)[rows], in column chunks.
                for (int64_t c = 0; c < F.nclasses; ++c) {
                    const int64_t s0 = F.sigma_start[c], K0 = F.sigma_start[c + 1] - s0;
                    const int64_t x0 = F.x_start[c], M0 = F.x_start[c + 1] - x0;
                    if (!K0 || !M0) continue;
                    const int64_t delta = c / 8 + F.class_min, gamma = c % 8;
                    const int64_t ka = alphaJ - delta + F.spin_change, kg = irrepJ ^ gamma ^ F.irrep;
                    const int64_t key = ka >= 0 && ka <= 4 ? L.key_of[ka * 8 + kg] : -1;
                    if (key < 0) continue;
                    const int64_t n = L.group_start[key + 1] - L.group_start[key];
                    cols.clear();
                    for (int64_t k = 0; k < K0; ++k) {
                        const int32_t *s = F.sigma + 4 * (s0 + k);
                        if (hvec[size_t(kh == 1 ? s[0] : s[0] * no + s[1])]) cols.push_back(k);
                    }
                    rows_m.clear();
                    coefs.clear();
                    dests.clear();
                    for (int64_t m = 0; m < M0; ++m) {
                        const int32_t *x = F.x + 4 * (x0 + m);
                        const int64_t t = target_of[size_t(kh == 1 ? x[0] : x[0] * no + x[1])];
                        if (t < 0 || targets[size_t(t)].key != key) continue;
                        const OpenedTarget &target = targets[size_t(t)];
                        double coef = 0.0;
                        for (int64_t term : by_pattern[size_t(pattern_of(kh, target.beta))]) {
                            const int64_t *tb = F.term_beta + 2 * term;
                            if (tb[0] == target.beta[0] && (kh < 2 || tb[1] == target.beta[1]))
                                coef += F.term_coef[term];
                        }
                        if (coef == 0.0) continue;
                        rows_m.push_back(m);
                        coefs.push_back(coef);
                        dests.push_back(target.out);
                    }
                    const int64_t M = int64_t(rows_m.size()), K = int64_t(cols.size());
                    if (!M || !K || !n) continue;
                    a.resize(size_t(M * K));
                    const double *w = F.weights + F.weight_offset[c];
                    bk.resize(size_t(K));
                    for (int64_t k = 0; k < K; ++k) {
                        const int32_t *s = F.sigma + 4 * (s0 + cols[size_t(k)]);
                        const int64_t h = hvec[size_t(kh == 1 ? s[0] : s[0] * no + s[1])];
                        bk[size_t(k)] = std::abs(h) - 1;
                        const double sign = h < 0 ? -1.0 : 1.0;
                        for (int64_t m = 0; m < M; ++m)
                            a[size_t(m * K + k)] = coefs[size_t(m)] * sign * w[rows_m[size_t(m)] * K0 + cols[size_t(k)]];
                    }
                    ar.resize(size_t(M));
                    for (int64_t m = 0; m < M; ++m) ar[size_t(m)] = m * K;
                    ak.resize(size_t(K));
                    for (int64_t k = 0; k < K; ++k) ak[size_t(k)] = k;
                    const int64_t chunk = 1024;
                    bc.resize(size_t(chunk));
                    cc.resize(size_t(chunk));
                    cr.resize(size_t(M));
                    for (int64_t m = 0; m < M; ++m) cr[size_t(m)] = m * chunk;
                    z.resize(size_t(M * chunk));
                    for (int64_t j0 = 0; j0 < n; j0 += chunk) {
                        const int64_t nc = std::min(chunk, n - j0);
                        for (int64_t j = 0; j < nc; ++j) {
                            bc[size_t(j)] = j0 + j;
                            cc[size_t(j)] = j;
                        }
                        std::fill(z.begin(), z.end(), 0.0);
                        OPENED_ADD(0);
                        itercc::IndirectGemm gemm{M, nc, K, a.data(), ar.data(), ak.data(), false,
                                                  F.vector, bk.data(), bc.data(), false,
                                                  z.data(), cr.data(), cc.data(), 1.0};
                        itercc::indirect_gemm(gemm);
                        OPENED_ADD(5);
                        for (int64_t m = 0; m < M; ++m) {
                            double *d = dests[size_t(m)] + j0;
                            const double *zm = z.data() + m * chunk;
                            for (int64_t j = 0; j < nc; ++j) d[j] += zm[j];
                        }
                        OPENED_ADD(6);
                    }
                }
                continue;
            }
            // kp >= 1. This unit's rows of each block: x whose x_h has a target here.
            touched.clear();
            auto x_hole = [&](const int32_t *x) -> int64_t {
                return kh == 0 ? 0 : kh == 1 ? x[kp] : int64_t(x[kp]) * no + x[kp + 1];
            };
            auto x_particle = [&](const int32_t *x) -> int64_t {
                return kp == 1 ? x[0] : int64_t(x[0]) * nv + x[1];
            };
            // Z blocks by class of B: offsets, B count and summed class.
            int64_t ztotal = 0;
            for (int64_t cb = 0; cb < 40; ++cb) {
                zoff[size_t(cb)] = -1;
                const int64_t nb = F.b_start[cb + 1] - F.b_start[cb];
                if (!nb) continue;
                const int64_t delta = alphaJ + F.spin_change - cb / 8, gamma = irrepJ ^ F.irrep ^ (cb % 8);
                const int64_t c = (delta - F.class_min) * 8 + gamma;
                if (delta < F.class_min || c >= F.nclasses) continue;
                const int64_t K = F.sigma_start[c + 1] - F.sigma_start[c];
                int64_t M = 0;
                for (int64_t m = F.x_start[c]; m < F.x_start[c + 1]; ++m) M += target_of[size_t(x_hole(F.x + 4 * m))] >= 0;
                if (!K || !M) continue;
                zoff[size_t(cb)] = ztotal;
                znb[size_t(cb)] = nb;
                zc[size_t(cb)] = c;
                ztotal += M * nb;
            }
            // Rows of the blocks used (a block can serve several classes of B).
            for (int64_t cb = 0; cb < 40; ++cb) {
                if (zoff[size_t(cb)] < 0) continue;
                const int64_t c = zc[size_t(cb)];
                const int32_t *first = F.x + 4 * F.x_start[c];
                if (F.x_start[c + 1] == F.x_start[c]) continue;
                if (m_of_x[size_t(x_particle(first) * nh + x_hole(first))] >= 0) continue;   // numbered already
                int32_t next = 0;
                for (int64_t m = F.x_start[c]; m < F.x_start[c + 1]; ++m) {
                    const int32_t *x = F.x + 4 * m;
                    const int64_t fh = x_hole(x);
                    if (target_of[size_t(fh)] < 0) continue;
                    const int64_t flat = x_particle(x) * nh + fh;
                    m_of_x[size_t(flat)] = next++;
                    touched.push_back(flat);
                }
            }
            z.resize(size_t(ztotal));
            OPENED_ADD(1);
            for (int64_t cb = 0; cb < 40; ++cb) {
                if (zoff[size_t(cb)] < 0) continue;
                const int64_t c = zc[size_t(cb)], nb = znb[size_t(cb)], b0 = F.b_start[cb];
                const int64_t s0 = F.sigma_start[c], K = F.sigma_start[c + 1] - s0;
                // Y[B, s] over this class of B and the class's summed tuples.
                y.resize(size_t(nb * K));
                hk.resize(size_t(K));
                pk.resize(size_t(K));
                for (int64_t k = 0; k < K; ++k) {
                    const int32_t *s = F.sigma + 4 * (s0 + k);
                    hk[size_t(k)] = hvec[size_t(x_hole(s))];
                    pk[size_t(k)] = x_particle(s);
                }
                gather_y(nb, K, npf, F.v_table + b0 * npf, hk.data(), pk.data(), F.vector, y.data());
                OPENED_ADD(2);
                ar.clear();
                for (int64_t m = F.x_start[c]; m < F.x_start[c + 1]; ++m)
                    if (target_of[size_t(x_hole(F.x + 4 * m))] >= 0)
                        ar.push_back(F.weight_offset[c] + (m - F.x_start[c]) * K);
                const int64_t M = int64_t(ar.size());
                ak.resize(size_t(K));
                bk.resize(size_t(K));
                for (int64_t k = 0; k < K; ++k) ak[size_t(k)] = bk[size_t(k)] = k;
                bc.resize(size_t(nb));
                cc.resize(size_t(nb));
                for (int64_t i = 0; i < nb; ++i) {
                    bc[size_t(i)] = i * K;
                    cc[size_t(i)] = i;
                }
                cr.resize(size_t(M));
                for (int64_t m = 0; m < M; ++m) cr[size_t(m)] = zoff[size_t(cb)] + m * nb;
                std::fill(z.begin() + zoff[size_t(cb)], z.begin() + zoff[size_t(cb)] + M * nb, 0.0);
                itercc::IndirectGemm gemm{M, nb, K, F.weights, ar.data(), ak.data(), false,
                                          y.data(), bk.data(), bc.data(), true,
                                          z.data(), cr.data(), cc.data(), 1.0};
                itercc::indirect_gemm(gemm);
                OPENED_ADD(3);
            }
            // Every row of every target: its terms' products, in term order.
            for (const OpenedTarget &target : targets) {
                const std::vector<int64_t> &terms = by_pattern[size_t(pattern_of(kh, target.beta))];
                const int32_t *rows = L.particles + 4 * L.group_start[target.key];
                const int64_t n = L.group_start[target.key + 1] - L.group_start[target.key];
                for (int64_t r = 0; r < n; ++r) {
                    const int32_t *p = rows + 4 * r;
                    double total = 0.0;
                    for (int64_t term : terms) {
                        const int64_t *tb = F.term_beta + 2 * term, *ta = F.term_alpha + 2 * term;
                        if (kh == 2 && (tb[0] != target.beta[0] || tb[1] != target.beta[1])) continue;
                        if (kh == 1 && tb[0] != target.beta[0]) continue;
                        int64_t fp = p[ta[0]], fb = 0;
                        if (kp == 2) fp = fp * nv + p[ta[1]];
                        for (int64_t i = 0; i < 4; ++i)
                            if (i != ta[0] && (kp < 2 || i != ta[1])) fb = fb * nv + p[i];
                        const int32_t g = F.b_index[fb];
                        if (g < 0) continue;
                        const int64_t cb = class_of_b[size_t(g)];
                        if (zoff[size_t(cb)] < 0) continue;
                        const int32_t m = m_of_x[size_t(fp * nh + target.flat)];
                        if (m < 0) continue;
                        total += F.term_coef[term] *
                                 z[size_t(zoff[size_t(cb)] + int64_t(m) * znb[size_t(cb)] + (g - F.b_start[cb]))];
                    }
                    target.out[r] += total;
                }
            }
            for (int64_t flat : touched) m_of_x[size_t(flat)] = -1;
            OPENED_ADD(4);
        }
    }
    // The buffers' contributions, in order of their position sets.
    if (nbuffers) {
        #pragma omp parallel for schedule(dynamic, 16) num_threads(threads)
        for (int64_t s = 0; s < L.nown; ++s) {
            double *o = out + L.own_full[s];
            const int64_t n = L.own_rows[s];
            for (int64_t b = 0; b < nbuffers; ++b) {
                const double *src = buffers + b * L.buffer_rows + L.own_compact[s];
                for (int64_t r = 0; r < n; ++r) o[r] += src[r];
            }
        }
    }
}

// Ranks two and three from the rank-four amplitude by opened letters: the
// same Z blocks as itercc_spin_quad_opened (here the summed letters, s_p and
// s_h, and the external ones, x_p and x_h, can differ in number), written
// into particle-major rows (part_offset of the particle tuple + hole_pos of
// the hole tuple).
extern "C" void itercc_spin_quad_opened_lower(const itercc_spin_quad_opened_rows *layout,
    const itercc_spin_quad_opened_family *family, int64_t nunits, const int32_t *units,
    double *out, double *buffers, int threads)
{
    const itercc_spin_quad_opened_rows &L = *layout;
    const itercc_spin_quad_opened_family &F = *family;
    const int64_t r = L.rank, no = L.no, nv = L.nv;
    const int64_t kp = F.kp, kh = F.kh, xp = F.xp, xh = F.xh;
    const int64_t nj = 4 - kh;                 // context holes (= r - xh)
    const int64_t nbuffers = xh == 0 ? 0 : xh == 1 ? r - 1 : r * (r - 1) / 2 - 1;
    const int64_t nsh = kh == 0 ? 1 : kh == 1 ? no : no * no;    // flat s_h
    const int64_t nxh = xh == 0 ? 1 : xh == 1 ? no : no * no;    // flat x_h
    const int64_t nsp = kp == 0 ? 1 : kp == 1 ? nv : nv * nv;    // flat s_p (v_table row length)
    const int64_t nxp = xp == 0 ? 1 : xp == 1 ? nv : nv * nv;    // flat x_p
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t i = 0; i < nbuffers * L.rows; ++i) buffers[i] = 0.0;
    auto pattern = [&](const int64_t *beta) -> int64_t {
        if (xh == 0) return 0;
        if (xh == 1) return beta[0];
        const int64_t p = std::min(beta[0], beta[1]), q = std::max(beta[0], beta[1]);
        return p == 0 ? q - 1 : p == 1 ? r - 2 + q - 1 : 2 * r - 4;   // pairs of r <= 3 positions
    };
    std::vector<int8_t> class_of_b(size_t(F.b_start[40]));
    for (int64_t cb = 0; cb < 40; ++cb)
        for (int64_t g = F.b_start[cb]; g < F.b_start[cb + 1]; ++g) class_of_b[size_t(g)] = int8_t(cb);
    auto s_hole = [&](const int32_t *s) -> int64_t {
        return kh == 0 ? 0 : kh == 1 ? s[kp] : int64_t(s[kp]) * no + s[kp + 1];
    };
    auto s_particle = [&](const int32_t *s) -> int64_t {
        return kp == 0 ? 0 : kp == 1 ? s[0] : int64_t(s[0]) * nv + s[1];
    };
    auto x_hole = [&](const int32_t *x) -> int64_t {
        return xh == 0 ? 0 : xh == 1 ? x[xp] : int64_t(x[xp]) * no + x[xp + 1];
    };
    auto x_particle = [&](const int32_t *x) -> int64_t {
        return xp == 0 ? 0 : xp == 1 ? x[0] : int64_t(x[0]) * nv + x[1];
    };
    #pragma omp parallel num_threads(threads)
    {
        std::vector<int64_t> hvec(static_cast<size_t>(nsh)), zoff(40), znb(40), zc(40), ar, ak, bk, bc, cr, cc, hk, pk;
        std::vector<int64_t> hole_of(static_cast<size_t>(nxh)), touched;
        std::vector<int32_t> m_of_x(size_t(nxp * nxh), -1);
        std::vector<double> y, z;
        #pragma omp for schedule(dynamic, 1)
        for (int64_t u = 0; u < nunits; ++u) {
            OPENED_MARK;
            const int32_t *J = units + 4 * u;
            int64_t alphaJ = 0, irrepJ = 0;
            for (int64_t i = 0; i < nj; ++i) {
                alphaJ += J[i] % 2 == 0;
                irrepJ ^= L.occupied_irrep[J[i] / 2];
            }
            // Amplitude slots J + s_h (signed offset + 1).
            for (int64_t f = 0; f < nsh; ++f) {
                int32_t holes[4];
                for (int64_t i = 0; i < nj; ++i) holes[i] = J[i];
                if (kh == 1) holes[3] = int32_t(f);
                if (kh == 2) {
                    holes[2] = int32_t(f / no);
                    holes[3] = int32_t(f % no);
                }
                const int64_t w = ((int64_t(holes[0]) * no + holes[1]) * no + holes[2]) * no + holes[3];
                hvec[size_t(f)] = F.hole_map[w];
            }
            // Output hole tuples J + x_h: flat sorted index (or -1).
            bool any = false;
            for (int64_t f = 0; f < nxh; ++f) {
                int32_t holes[3];
                int64_t n = 0;
                for (int64_t i = 0; i < nj; ++i) holes[n++] = J[i];
                if (xh == 1) holes[n++] = int32_t(f);
                if (xh == 2) {
                    holes[n++] = int32_t(f / no);
                    holes[n++] = int32_t(f % no);
                }
                std::sort(holes, holes + n);
                bool distinct = true;
                for (int64_t i = 1; i < n; ++i) distinct &= holes[i] != holes[i - 1];
                int64_t flat = 0;
                for (int64_t i = 0; i < n; ++i) flat = flat * no + holes[i];
                hole_of[size_t(f)] = distinct && L.hole_pos[flat] >= 0 ? flat : -1;
                any |= hole_of[size_t(f)] >= 0;
            }
            OPENED_ADD(0);
            if (!any) continue;
            // Z blocks by class of B (B: the output's particles but x_p).
            int64_t ztotal = 0;
            for (int64_t cb = 0; cb < 40; ++cb) {
                zoff[size_t(cb)] = -1;
                const int64_t nb = F.b_start[cb + 1] - F.b_start[cb];
                if (!nb) continue;
                const int64_t delta = alphaJ + F.spin_change - cb / 8, gamma = irrepJ ^ F.irrep ^ (cb % 8);
                const int64_t c = (delta - F.class_min) * 8 + gamma;
                if (delta < F.class_min || c >= F.nclasses) continue;
                const int64_t K = F.sigma_start[c + 1] - F.sigma_start[c];
                int64_t M = 0;
                for (int64_t m = F.x_start[c]; m < F.x_start[c + 1]; ++m) M += hole_of[size_t(x_hole(F.x + 4 * m))] >= 0;
                if (!K || !M) continue;
                zoff[size_t(cb)] = ztotal;
                znb[size_t(cb)] = nb;
                zc[size_t(cb)] = c;
                ztotal += M * nb;
            }
            touched.clear();
            for (int64_t cb = 0; cb < 40; ++cb) {
                if (zoff[size_t(cb)] < 0) continue;
                const int64_t c = zc[size_t(cb)];
                int32_t next = 0;
                for (int64_t m = F.x_start[c]; m < F.x_start[c + 1]; ++m) {
                    const int32_t *x = F.x + 4 * m;
                    const int64_t fh = x_hole(x);
                    if (hole_of[size_t(fh)] < 0) continue;
                    const int64_t flat = x_particle(x) * nxh + fh;
                    m_of_x[size_t(flat)] = next++;
                    touched.push_back(flat);
                }
            }
            z.resize(size_t(ztotal));
            OPENED_ADD(1);
            for (int64_t cb = 0; cb < 40; ++cb) {
                if (zoff[size_t(cb)] < 0) continue;
                const int64_t c = zc[size_t(cb)], nb = znb[size_t(cb)], b0 = F.b_start[cb];
                const int64_t s0 = F.sigma_start[c], K = F.sigma_start[c + 1] - s0;
                y.resize(size_t(nb * K));
                hk.resize(size_t(K));
                pk.resize(size_t(K));
                for (int64_t k = 0; k < K; ++k) {
                    const int32_t *s = F.sigma + 4 * (s0 + k);
                    hk[size_t(k)] = hvec[size_t(s_hole(s))];
                    pk[size_t(k)] = s_particle(s);
                }
                gather_y(nb, K, nsp, F.v_table + b0 * nsp, hk.data(), pk.data(), F.vector, y.data());
                OPENED_ADD(2);
                ar.clear();
                for (int64_t m = F.x_start[c]; m < F.x_start[c + 1]; ++m)
                    if (hole_of[size_t(x_hole(F.x + 4 * m))] >= 0)
                        ar.push_back(F.weight_offset[c] + (m - F.x_start[c]) * K);
                const int64_t M = int64_t(ar.size());
                ak.resize(size_t(K));
                bk.resize(size_t(K));
                for (int64_t k = 0; k < K; ++k) ak[size_t(k)] = bk[size_t(k)] = k;
                bc.resize(size_t(nb));
                cc.resize(size_t(nb));
                for (int64_t i = 0; i < nb; ++i) {
                    bc[size_t(i)] = i * K;
                    cc[size_t(i)] = i;
                }
                cr.resize(size_t(M));
                for (int64_t m = 0; m < M; ++m) cr[size_t(m)] = zoff[size_t(cb)] + m * nb;
                std::fill(z.begin() + zoff[size_t(cb)], z.begin() + zoff[size_t(cb)] + M * nb, 0.0);
                itercc::IndirectGemm gemm{M, nb, K, F.weights, ar.data(), ak.data(), false,
                                          y.data(), bk.data(), bc.data(), true,
                                          z.data(), cr.data(), cc.data(), 1.0};
                itercc::indirect_gemm(gemm);
                OPENED_ADD(3);
            }
            // Every output row with holes J + x_h: its terms' products, in term order.
            for (int64_t f = 0; f < nxh; ++f) {
                const int64_t hflat = hole_of[size_t(f)];
                if (hflat < 0) continue;
                int32_t tail[2] = {0, 0};
                if (xh == 1) tail[0] = int32_t(f);
                if (xh == 2) {
                    tail[0] = int32_t(f / no);
                    tail[1] = int32_t(f % no);
                }
                int64_t beta[2] = {0, 0}, alphaH = alphaJ, irrepH = irrepJ;
                for (int64_t i = 0; i < xh; ++i) {
                    int64_t pos = 0;
                    for (int64_t j = 0; j < nj; ++j) pos += J[j] < tail[i];
                    for (int64_t j = 0; j < xh; ++j) pos += j != i && tail[j] < tail[i];
                    beta[i] = pos;
                    alphaH += tail[i] % 2 == 0;
                    irrepH ^= L.occupied_irrep[tail[i] / 2];
                }
                const int64_t b = pattern(beta);
                const int64_t hole_offset = L.hole_pos[hflat];
                double *target = b == 0 ? out : buffers + (b - 1) * L.rows;
                // Particle tuples of the output rows with these holes.
                const int64_t pa = alphaH + L.spin_change, pg = irrepH ^ L.irrep;
                if (pa < 0 || pa > r) continue;
                const int64_t pclass = pa * 8 + pg;
                for (int64_t g = L.part_start[pclass]; g < L.part_start[pclass + 1]; ++g) {
                    const int32_t *p = L.part_values + 4 * g;
                    double total = 0.0;
                    for (int64_t term = 0; term < F.nterms; ++term) {
                        const int64_t *tb = F.term_beta + 2 * term, *ta = F.term_alpha + 2 * term;
                        if ((xh >= 1 && tb[0] != beta[0]) || (xh == 2 && tb[1] != beta[1])) continue;
                        int64_t fp = 0, fb = 0;
                        if (xp >= 1) fp = p[ta[0]];
                        if (xp == 2) fp = fp * nv + p[ta[1]];
                        for (int64_t i = 0; i < r; ++i)
                            if ((xp < 1 || i != ta[0]) && (xp < 2 || i != ta[1])) fb = fb * nv + p[i];
                        const int32_t gb = F.b_index[fb];
                        if (gb < 0) continue;
                        const int64_t cb = class_of_b[size_t(gb)];
                        if (zoff[size_t(cb)] < 0) continue;
                        const int32_t m = m_of_x[size_t(fp * nxh + f)];
                        if (m < 0) continue;
                        total += F.term_coef[term] *
                                 z[size_t(zoff[size_t(cb)] + int64_t(m) * znb[size_t(cb)] + (gb - F.b_start[cb]))];
                    }
                    target[L.part_offset[g] + hole_offset] += total;
                }
            }
            for (int64_t flat : touched) m_of_x[size_t(flat)] = -1;
            OPENED_ADD(4);
        }
    }
    if (nbuffers) {
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int64_t i = 0; i < L.rows; ++i) {
            double value = out[i];
            for (int64_t b = 0; b < nbuffers; ++b) value += buffers[b * L.rows + i];
            out[i] = value;
        }
    }
}
