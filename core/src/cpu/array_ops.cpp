// Whole-array operations of the CPU solvers, on all threads.
//
// The solvers' bookkeeping (symmetrizing intermediates, re-laying arrays out,
// damping and adding updates, denominators) passes over arrays of gigabytes at
// production sizes. As single-threaded NumPy expressions those passes would
// dominate a many-core node. Each kernel here computes what the NumPy
// expression computes, entry by entry with the same arithmetic, and every
// entry is written by one thread.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "itercc/core.h"

using i64 = int64_t;

namespace {

constexpr int kMaxDims = 8;

i64 leading(i64 ndim, const i64 *shape)
{
    i64 n = 1;
    for (i64 d = 0; d + 1 < ndim; ++d) n *= shape[d];
    return n;
}

// Offset of row `o` (all axes but the last, C order) through `strides`.
i64 row_offset(i64 o, i64 ndim, const i64 *shape, const i64 *strides)
{
    i64 offset = 0;
    for (i64 d = ndim - 2; d >= 0; --d) {
        offset += (o % shape[d]) * strides[d];
        o /= shape[d];
    }
    return offset;
}

}  // namespace

extern "C" int itercc_permuted_add(double *out, const double *a, const double *b, int64_t ndim,
                                   const int64_t *shape, const int64_t *b_strides, double wa,
                                   double wb, int threads)
{
    if (ndim < 1 || ndim > kMaxDims) return 1;
    const i64 inner = shape[ndim - 1], rows = leading(ndim, shape), step = b_strides[ndim - 1];
    // Long rows are cut into pieces as well: a flat array is a single row, and
    // its copy or first touch must still run on every thread.
    constexpr i64 kPiece = i64(1) << 15;
    const i64 pieces = (inner + kPiece - 1) / kPiece;
#pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
    for (i64 o = 0; o < rows; ++o)
        for (i64 p = 0; p < pieces; ++p) {
            const i64 lo = p * kPiece, hi = std::min(inner, lo + kPiece);
            double *dst = out + o * inner;
            const double *src = b + row_offset(o, ndim, shape, b_strides);
            if (a != nullptr) {
                const double *first = a + o * inner;
                for (i64 i = lo; i < hi; ++i) dst[i] = wa * first[i] + wb * src[i * step];
            } else {
                for (i64 i = lo; i < hi; ++i) dst[i] = wb * src[i * step];
            }
        }
    return 0;
}

extern "C" int itercc_symmetrize_pairs(double *x, int64_t ndim, const int64_t *shape,
                                       const int64_t *perm_strides, int threads)
{
    // x += x.transpose(perm) in place for a permutation that is its own
    // inverse: entries f and g = perm(f) form a pair (or f == g), and both get
    // x[f] + x[g]. The pair belongs to the row of its smaller entry.
    if (ndim < 1 || ndim > kMaxDims) return 1;
    const i64 inner = shape[ndim - 1], rows = leading(ndim, shape), step = perm_strides[ndim - 1];
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 o = 0; o < rows; ++o) {
        const i64 partner = row_offset(o, ndim, shape, perm_strides);
        for (i64 i = 0; i < inner; ++i) {
            const i64 f = o * inner + i, g = partner + i * step;
            if (f < g) {
                const double sum = x[f] + x[g];
                x[f] = sum;
                x[g] = sum;
            } else if (f == g) {
                x[f] = x[f] + x[f];
            }
        }
    }
    return 0;
}

extern "C" void itercc_axpy(double *y, const double *x, double alpha, int64_t n, int threads)
{
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 i = 0; i < n; ++i) y[i] = y[i] + alpha * x[i];
}

extern "C" void itercc_t3_divide(double *r3, int64_t no, int64_t nv, const double *eia, int threads)
{
    // r3[i,j,k,a,b,c] /= (eia[i,a] + eia[j,b]) + eia[k,c], as NumPy's d3.
    const i64 nv2 = nv * nv, nv3 = nv2 * nv;
#pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
    for (i64 ijk = 0; ijk < no * no * no; ++ijk)
        for (i64 a = 0; a < nv; ++a) {
            const i64 i = ijk / (no * no), j = (ijk / no) % no, k = ijk % no;
            const double *ei = eia + i * nv, *ej = eia + j * nv, *ek = eia + k * nv;
            double *row = r3 + ijk * nv3 + a * nv2;
            for (i64 b = 0; b < nv; ++b) {
                const double ab = ei[a] + ej[b];
                double *line = row + b * nv;
                for (i64 c = 0; c < nv; ++c) line[c] /= ab + ek[c];
            }
        }
}

// DIIS over vectors of gigabytes (the packed T4 history): inner products and
// linear combinations on all threads, with results independent of the thread
// count (fixed blocks, each summed in order; block sums added in block order).
namespace {
constexpr i64 kDotBlock = i64(1) << 14;
}

extern "C" void itercc_dots(int64_t n, int64_t k, const double *x, const double *const *ys, double *out,
                            int threads)
{
    // out[j] = x . ys[j] for j < k and out[k] = x . x; x is read once per block.
    const i64 blocks = (n + kDotBlock - 1) / kDotBlock;
    std::vector<double> partial(size_t(std::max<i64>(1, blocks * (k + 1))), 0.0);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 b = 0; b < blocks; ++b) {
        const i64 lo = b * kDotBlock, hi = std::min(n, lo + kDotBlock);
        double *p = partial.data() + b * (k + 1);
        for (i64 j = 0; j <= k; ++j) {
            const double *y = j < k ? ys[j] : x;
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            i64 i = lo;
            for (; i + 4 <= hi; i += 4) {
                s0 += x[i] * y[i];
                s1 += x[i + 1] * y[i + 1];
                s2 += x[i + 2] * y[i + 2];
                s3 += x[i + 3] * y[i + 3];
            }
            for (; i < hi; ++i) s0 += x[i] * y[i];
            p[j] = (s0 + s1) + (s2 + s3);
        }
    }
    for (i64 j = 0; j <= k; ++j) {
        double total = 0.0;
        for (i64 b = 0; b < blocks; ++b) total += partial[size_t(b * (k + 1) + j)];
        out[j] = total;
    }
}

// Largest absolute value; NaN if any entry is NaN (infinities give inf).
extern "C" double itercc_max_abs(const double *x, int64_t n, int threads)
{
    double best = 0.0;
    int nan = 0;
#pragma omp parallel for reduction(max : best) reduction(| : nan) schedule(static) num_threads(threads)
    for (i64 i = 0; i < n; ++i) {
        const double v = std::fabs(x[i]);
        if (v != v)
            nan = 1;
        else if (v > best)
            best = v;
    }
    return nan ? NAN : best;
}

extern "C" void itercc_combine(int64_t n, int64_t k, const double *weights, const double *const *vs,
                               double *out, int threads)
{
    // out = sum_j weights[j] vs[j], entry by entry in order of j (out may not
    // overlap any vs[j]).
    if (k <= 0) {
        std::fill(out, out + n, 0.0);
        return;
    }
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 lo = 0; lo < n; lo += kDotBlock) {
        const i64 hi = std::min(n, lo + kDotBlock);
        const double w0 = weights[0];
        const double *v0 = vs[0];
        for (i64 i = lo; i < hi; ++i) out[i] = w0 * v0[i];
        for (i64 j = 1; j < k; ++j) {
            const double w = weights[j];
            const double *v = vs[j];
            for (i64 i = lo; i < hi; ++i) out[i] = out[i] + w * v[i];
        }
    }
}

// Compact DIIS history (itercc/quadruples.py _CompactHostDIIS): errors and
// trial differences in float32, against a float64 newest error x.
extern "C" void itercc_dots_f32(int64_t n, int64_t k, const double *x, const float *const *ys,
                                double *out, int threads)
{
    // out[j] = x . ys[j] for j < k and out[k] = x . x, summed as itercc_dots.
    const i64 blocks = (n + kDotBlock - 1) / kDotBlock;
    std::vector<double> partial(size_t(std::max<i64>(1, blocks * (k + 1))), 0.0);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 b = 0; b < blocks; ++b) {
        const i64 lo = b * kDotBlock, hi = std::min(n, lo + kDotBlock);
        double *p = partial.data() + b * (k + 1);
        for (i64 j = 0; j <= k; ++j) {
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            i64 i = lo;
            if (j < k) {
                const float *y = ys[j];
                for (; i + 4 <= hi; i += 4) {
                    s0 += x[i] * double(y[i]);
                    s1 += x[i + 1] * double(y[i + 1]);
                    s2 += x[i + 2] * double(y[i + 2]);
                    s3 += x[i + 3] * double(y[i + 3]);
                }
                for (; i < hi; ++i) s0 += x[i] * double(y[i]);
            } else {
                for (; i + 4 <= hi; i += 4) {
                    s0 += x[i] * x[i];
                    s1 += x[i + 1] * x[i + 1];
                    s2 += x[i + 2] * x[i + 2];
                    s3 += x[i + 3] * x[i + 3];
                }
                for (; i < hi; ++i) s0 += x[i] * x[i];
            }
            p[j] = (s0 + s1) + (s2 + s3);
        }
    }
    for (i64 j = 0; j <= k; ++j) {
        double total = 0.0;
        for (i64 b = 0; b < blocks; ++b) total += partial[size_t(b * (k + 1) + j)];
        out[j] = total;
    }
}

extern "C" void itercc_diis_rebase(int64_t n, int64_t k, const double *weights, float *const *deltas,
                                   float *delta_new, float *error_new, double *x, const double *step,
                                   int threads)
{
    // One DIIS step in difference form. On entry x holds the newest trial
    // (input + step) and deltas[j] = trial_j - input for the k older entries.
    // With weights w (k + 1 of them, the newest last, summing to one):
    //   D = sum_j w_j deltas[j] + w_k step   (the extrapolated change of the input)
    //   x = x + (D - step)                   (the extrapolated amplitudes)
    //   deltas[j] = deltas[j] - D            (re-based on the new x)
    //   delta_new = step - D, error_new = step.
    // Without weights (null) D = step: x stays the trial (a plain Jacobi step).
    // Every entry is formed alone in a fixed order: thread-count independent.
    constexpr i64 kPiece = 512;
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 lo = 0; lo < n; lo += kPiece) {
        const i64 hi = std::min(n, lo + kPiece), m = hi - lo;
        double d[kPiece];
        const double *s = step + lo;
        if (weights == nullptr) {
            for (i64 i = 0; i < m; ++i) d[i] = s[i];
        } else if (k == 0) {
            for (i64 i = 0; i < m; ++i) d[i] = weights[0] * s[i];
        } else {
            const float *v0 = deltas[0] + lo;
            const double w0 = weights[0];
            for (i64 i = 0; i < m; ++i) d[i] = w0 * double(v0[i]);
            for (i64 j = 1; j < k; ++j) {
                const float *v = deltas[j] + lo;
                const double w = weights[j];
                for (i64 i = 0; i < m; ++i) d[i] = d[i] + w * double(v[i]);
            }
            const double wk = weights[k];
            for (i64 i = 0; i < m; ++i) d[i] = d[i] + wk * s[i];
        }
        double *xo = x + lo;
        for (i64 i = 0; i < m; ++i) xo[i] = xo[i] + (d[i] - s[i]);
        for (i64 j = 0; j < k; ++j) {
            float *v = deltas[j] + lo;
            for (i64 i = 0; i < m; ++i) v[i] = float(double(v[i]) - d[i]);
        }
        float *dn = delta_new + lo;
        for (i64 i = 0; i < m; ++i) dn[i] = float(s[i] - d[i]);
        if (error_new != nullptr) {
            float *en = error_new + lo;
            for (i64 i = 0; i < m; ++i) en[i] = float(s[i]);
        }
    }
}

namespace {

// dst[r, j] = src[r, index[j]] for rows of ncols entries: row by row on all
// threads (rows of a T3-like array: virtual pairs or cubes, each in cache).
template <typename T>
void permute_columns(const double *src, int64_t rows, int64_t ncols, const int64_t *index, T *dst,
                     int threads)
{
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 r = 0; r < rows; ++r) {
        const double *in = src + r * ncols;
        T *out = dst + r * ncols;
        for (i64 j = 0; j < ncols; ++j) out[j] = T(in[index[j]]);
    }
}

}  // namespace

extern "C" void itercc_permute_columns(const double *src, int64_t rows, int64_t ncols, const int64_t *index,
                                       double *dst, int threads)
{
    permute_columns(src, rows, ncols, index, dst, threads);
}

extern "C" void itercc_permute_columns_f32(const double *src, int64_t rows, int64_t ncols,
                                           const int64_t *index, float *dst, int threads)
{
    permute_columns(src, rows, ncols, index, dst, threads);
}

extern "C" void itercc_to_f32(const double *x, float *out, int64_t n, int threads)
{
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 lo = 0; lo < n; lo += kDotBlock) {
        const i64 hi = std::min(n, lo + kDotBlock);
        for (i64 i = lo; i < hi; ++i) out[i] = float(x[i]);
    }
}

namespace {

// x[.., t0 + t, ..] += sum_s rotation[t, s] src[.., s0 + s, ..] along `axis`
// (src may be x itself: the target and source ranges must not overlap).
void dress_axis(double *x, const double *source, int64_t n, int axis, int64_t t0, int64_t nt, int64_t s0,
                int64_t ns, const double *rotation, int threads)
{
    // x as (outer, n, inner) with the transformed axis in the middle. Each
    // task takes one outer index and a piece of the inner range: its source
    // rows (ns pieces) stay in cache while every target row is formed, and
    // each entry's sum is fixed (in order of s, then added), whatever the
    // thread count. The last axis (inner = 1) runs whole rows instead.
    if (nt <= 0 || ns <= 0 || n <= 0) return;
    i64 outer = 1, inner = 1;
    for (int d = 0; d < axis; ++d) outer *= n;
    for (int d = axis + 1; d < 4; ++d) inner *= n;
    if (inner == 1) {
#pragma omp parallel for num_threads(threads) schedule(static)
        for (i64 o = 0; o < outer; ++o) {
            double *row = x + o * n;
            const double *from = source + o * n;
            for (i64 t = 0; t < nt; ++t) {
                const double *r = rotation + t * ns;
                double sum = 0.0;
                for (i64 s = 0; s < ns; ++s) sum += r[s] * from[s0 + s];
                row[t0 + t] += sum;
            }
        }
        return;
    }
    constexpr i64 kPiece = 256;
    const i64 pieces = (inner + kPiece - 1) / kPiece;
#pragma omp parallel num_threads(threads)
    {
        alignas(64) double sum[kPiece];
#pragma omp for collapse(2) schedule(static)
        for (i64 o = 0; o < outer; ++o)
            for (i64 p = 0; p < pieces; ++p) {
                const i64 lo = p * kPiece, len = std::min(kPiece, inner - lo);
                double *base = x + o * n * inner + lo;
                const double *from = source + o * n * inner + lo;
                for (i64 t = 0; t < nt; ++t) {
                    const double *r = rotation + t * ns;
                    for (i64 i = 0; i < len; ++i) sum[i] = 0.0;
                    for (i64 s = 0; s < ns; ++s) {
                        const double w = r[s];
                        const double *src = from + (s0 + s) * inner;
                        for (i64 i = 0; i < len; ++i) sum[i] += w * src[i];
                    }
                    double *dst = base + (t0 + t) * inner;
                    for (i64 i = 0; i < len; ++i) dst[i] += sum[i];
                }
            }
    }
}

}  // namespace

extern "C" void itercc_dress_axis(double *x, int64_t n, int axis, int64_t t0, int64_t nt, int64_t s0,
                                  int64_t ns, const double *rotation, int threads)
{
    dress_axis(x, x, n, axis, t0, nt, s0, ns, rotation, threads);
}

extern "C" void itercc_dress_axis_from(double *x, const double *source, int64_t n, int axis, int64_t t0,
                                       int64_t nt, int64_t s0, int64_t ns, const double *rotation,
                                       int threads)
{
    dress_axis(x, source, n, axis, t0, nt, s0, ns, rotation, threads);
}

namespace {

// dst = src dressed with T1 along all four axes (`itercc_dress_integrals`), or
// with `ccsd` only what the CCSD residual reads (`itercc_dress_integrals_ccsd`).
// Axes 2 and 3 of one (p, q) plane: copied from src, then its occupied rows
// and columns gain t1 times the virtual ones (in place).
inline void dress_plane_23(double *out, const double *in, const int64_t *ss, i64 n, i64 no, const double *t1,
                           double *sum)
{
    const i64 nv = n - no;
    for (i64 r = 0; r < n; ++r) {
        const double *from = in + r * ss[2];
        double *to = out + r * n;
        if (ss[3] == 1)
            std::copy(from, from + n, to);
        else
            for (i64 s = 0; s < n; ++s) to[s] = from[s * ss[3]];
    }
    for (i64 i = 0; i < no; ++i) {             // axis 2: rows
        const double *t = t1 + i * nv;
        for (i64 s = 0; s < n; ++s) sum[s] = 0.0;
        for (i64 c = 0; c < nv; ++c) {
            const double w = t[c];
            const double *row = out + (no + c) * n;
            for (i64 s = 0; s < n; ++s) sum[s] += w * row[s];
        }
        double *row = out + i * n;
        for (i64 s = 0; s < n; ++s) row[s] += sum[s];
    }
    for (i64 r = 0; r < n; ++r) {              // axis 3: columns
        double *row = out + r * n;
        for (i64 i = 0; i < no; ++i) {
            const double *t = t1 + i * nv;
            double total = 0.0;
            for (i64 c = 0; c < nv; ++c) total += t[c] * row[no + c];
            sum[i] = total;
        }
        for (i64 i = 0; i < no; ++i) row[i] += sum[i];
    }
}

// The occupied-occupied block of an all-virtual (p, q) plane with its leg 2
// and leg 3 terms each from src (no virtual-virtual entry read or written).
inline void dress_block_23(double *out, const double *in, const int64_t *ss, i64 n, i64 no, const double *t1,
                           double *sum)
{
    const i64 nv = n - no;
    for (i64 r = 0; r < no; ++r) {
        const double *t = t1 + r * nv;
        for (i64 s = 0; s < no; ++s) sum[s] = 0.0;
        for (i64 c = 0; c < nv; ++c) {     // t(r, c) g(no + c, s)
            const double w = t[c];
            const double *row = in + (no + c) * ss[2];
            for (i64 s = 0; s < no; ++s) sum[s] += w * row[s * ss[3]];
        }
        const double *from = in + r * ss[2];
        double *to = out + r * n;
        for (i64 s = 0; s < no; ++s) {     // t(s, c) g(r, no + c)
            const double *u = t1 + s * nv;
            double total = 0.0;
            for (i64 c = 0; c < nv; ++c) total += u[c] * from[(no + c) * ss[3]];
            to[s] = from[s * ss[3]] + sum[s] + total;
        }
    }
}

// The entries of an all-virtual (p, q) plane with an occupied index (the
// occupied rows whole, the first no columns of the other rows) dressed along
// axes 2 and 3 from src, with the sums of dress_plane_23 in the same order
// (the same values), without writing the virtual-virtual block. Each virtual
// row is read once. `sum` holds (no + 1) n values.
inline void dress_open_23(double *out, const double *in, const int64_t *ss, i64 n, i64 no, const double *t1,
                          double *sum)
{
    const i64 nv = n - no;
    double *row = sum + no * n;
    for (i64 m = 0; m < no * n; ++m) sum[m] = 0.0;
    for (i64 c = 0; c < nv; ++c) {
        const double *from = in + (no + c) * ss[2];
        if (ss[3] == 1)
            std::copy(from, from + n, row);
        else
            for (i64 s = 0; s < n; ++s) row[s] = from[s * ss[3]];
        for (i64 i = 0; i < no; ++i) {             // axis 2: the occupied rows gain t(i, c) times this row
            const double w = t1[i * nv + c];
            double *acc = sum + i * n;
            for (i64 s = 0; s < n; ++s) acc[s] += w * row[s];
        }
        double *to = out + (no + c) * n;           // axis 3: its occupied columns
        for (i64 j = 0; j < no; ++j) {
            const double *t = t1 + j * nv;
            double total = 0.0;
            for (i64 d = 0; d < nv; ++d) total += t[d] * row[no + d];
            to[j] = row[j] + total;
        }
    }
    for (i64 i = 0; i < no; ++i) {
        const double *from = in + i * ss[2];
        const double *acc = sum + i * n;
        double *to = out + i * n;
        for (i64 s = 0; s < n; ++s) to[s] = from[s * ss[3]] + acc[s];
        for (i64 j = 0; j < no; ++j) {             // axis 3 of the dressed row
            const double *t = t1 + j * nv;
            double total = 0.0;
            for (i64 d = 0; d < nv; ++d) total += t[d] * to[no + d];
            row[j] = total;
        }
        for (i64 j = 0; j < no; ++j) to[j] += row[j];
    }
}

enum DressMode { kFull = 0, kCcsd = 1, kOpen = 2 };

// target += sum_k w[k] source_k (source_k = first + k * step) over the entries
// a partial dressing writes in an all-virtual plane: the occupied-occupied
// block (kCcsd) or every entry with an occupied index (kOpen); each sum in
// order of k, then added, as the full dressing does.
inline void add_written(double *target, const double *first, i64 step, const double *w, i64 n, i64 no,
                        DressMode mode, double *sum, i64 piece)
{
    auto add = [&](i64 lo, i64 len) {
        for (i64 c0 = 0; c0 < len; c0 += piece) {
            const i64 m1 = std::min(piece, len - c0);
            for (i64 m = 0; m < m1; ++m) sum[m] = 0.0;
            for (i64 k = 0; k < no; ++k) {
                const double wk = w[k];
                const double *from = first + k * step + lo + c0;
                for (i64 m = 0; m < m1; ++m) sum[m] += wk * from[m];
            }
            double *to = target + lo + c0;
            for (i64 m = 0; m < m1; ++m) to[m] += sum[m];
        }
    };
    for (i64 r = 0; r < no; ++r) add(r * n, mode == kOpen ? n : no);
    if (mode == kOpen)
        for (i64 r = no; r < n; ++r) add(r * n, no);
}

void dress_integrals_impl(double *dst, const double *src, const int64_t *ss, i64 n, i64 no, const double *t1,
                          int threads, DressMode mode)
{
    const i64 nv = n - no;
    if (n <= 0) return;
    const i64 n2 = n * n, n3 = n2 * n;
    constexpr i64 kPiece = 256;
    std::vector<double> down(size_t(std::max<i64>(nv * no, 1)));    // (nv, no) = -t1^T
    for (i64 a = 0; a < nv; ++a)
        for (i64 k = 0; k < no; ++k) down[size_t(a * no + k)] = -t1[k * nv + a];
    const size_t scratch = size_t(std::max<i64>((no + 1) * n, kPiece));
    // Partial dressings go plane by plane on more than 3 no threads (one
    // Xeon 8480CL socket, water aug-cc-pVQZ CCSD: 19 instead of 35 ms on 56
    // threads, 28 instead of 36 on 28; on 12 desktop threads slab by slab is
    // faster). ITERCC_DRESS_SLABS=1 or 0 forces slabs or planes.
    const char *slabs = std::getenv("ITERCC_DRESS_SLABS");
    const bool planes = slabs != nullptr && slabs[0] != '\0' ? slabs[0] == '0' : threads > 3 * no;
    if (mode != kFull && planes) {
        // Most of the work is in the planes with an occupied index, and the
        // slabs of an occupied p hold half of it: slab tasks gave at most about
        // 2 no threads work. So the occupied slabs are done one after another
        // on all threads (planes, then the axis 1 update while the slab is in
        // the last-level cache), and the planes of the other slabs are tasks of
        // their own. The same operations as slab by slab, in the same order:
        // the same results.
#pragma omp parallel num_threads(threads)
        {
            std::vector<double> sum(scratch);
            auto axis1 = [&](i64 p, i64 a) {
                const double *slab = dst + p * n3;
                double *target = dst + p * n3 + (no + a) * n2;
                if (p >= no) {
                    add_written(target, slab, n2, down.data() + a * no, n, no, mode, sum.data(), kPiece);
                    return;
                }
                for (i64 lo = 0; lo < n2; lo += kPiece) {
                    const i64 len = std::min(kPiece, n2 - lo);
                    for (i64 m = 0; m < len; ++m) sum[m] = 0.0;
                    for (i64 k = 0; k < no; ++k) {
                        const double w = -t1[k * nv + a];
                        const double *from = slab + k * n2 + lo;
                        for (i64 m = 0; m < len; ++m) sum[m] += w * from[m];
                    }
                    double *to = target + lo;
                    for (i64 m = 0; m < len; ++m) to[m] += sum[m];
                }
            };
            for (i64 p = 0; p < std::min(no, n); ++p) {
#pragma omp for schedule(dynamic, 2)
                for (i64 q = 0; q < n; ++q)
                    dress_plane_23(dst + p * n3 + q * n2, src + p * ss[0] + q * ss[1], ss, n, no, t1, sum.data());
#pragma omp for schedule(dynamic, 2)
                for (i64 a = 0; a < nv; ++a) axis1(p, a);
            }
#pragma omp for collapse(2) schedule(dynamic, 8)
            for (i64 p = no; p < n; ++p)
                for (i64 q = 0; q < n; ++q) {
                    const double *in = src + p * ss[0] + q * ss[1];
                    double *out = dst + p * n3 + q * n2;
                    if (q < no)
                        dress_plane_23(out, in, ss, n, no, t1, sum.data());
                    else if (mode == kCcsd)
                        dress_block_23(out, in, ss, n, no, t1, sum.data());
                    else
                        dress_open_23(out, in, ss, n, no, t1, sum.data());
                }
#pragma omp for collapse(2) schedule(dynamic, 16)
            for (i64 p = no; p < n; ++p)
                for (i64 a = 0; a < nv; ++a) axis1(p, a);
        }
    } else {
#pragma omp parallel num_threads(threads)
    {
        std::vector<double> sum(scratch);
#pragma omp for schedule(dynamic, 1)
        for (i64 p = 0; p < n; ++p) {
            double *slab = dst + p * n3;
            for (i64 q = 0; q < n; ++q) {
                double *out = slab + q * n2;
                const double *in = src + p * ss[0] + q * ss[1];
                if (mode == kCcsd && p >= no && q >= no)
                    dress_block_23(out, in, ss, n, no, t1, sum.data());
                else if (mode == kOpen && p >= no && q >= no)
                    dress_open_23(out, in, ss, n, no, t1, sum.data());
                else
                    dress_plane_23(out, in, ss, n, no, t1, sum.data());
            }
            if (mode != kFull && p >= no) {                // axis 1 on the entries written
                for (i64 a = 0; a < nv; ++a)
                    add_written(slab + (no + a) * n2, slab, n2, down.data() + a * no, n, no, mode, sum.data(),
                                kPiece);
                continue;
            }
            for (i64 lo = 0; lo < n2; lo += kPiece) {     // axis 1 within the slab
                const i64 len = std::min(kPiece, n2 - lo);
                for (i64 a = 0; a < nv; ++a) {
                    for (i64 m = 0; m < len; ++m) sum[m] = 0.0;
                    for (i64 k = 0; k < no; ++k) {
                        const double w = -t1[k * nv + a];
                        const double *from = slab + k * n2 + lo;
                        for (i64 m = 0; m < len; ++m) sum[m] += w * from[m];
                    }
                    double *to = slab + (no + a) * n2 + lo;
                    for (i64 m = 0; m < len; ++m) to[m] += sum[m];
                }
            }
        }
    }
    }
    if (no <= 0 || nv <= 0) return;
    if (mode == kFull) {
        dress_axis(dst, dst, n, 0, no, nv, 0, no, down.data(), threads);
        return;
    }
    // Axis 0 on the entries written: the virtual slabs gain -t1^T times the
    // occupied ones (complete by now), whole planes for an occupied q, the
    // entries written otherwise; sums in order of k, then added.
#pragma omp parallel num_threads(threads)
    {
        alignas(64) double sum[kPiece];
#pragma omp for collapse(2) schedule(static)
        for (i64 a = 0; a < nv; ++a)
            for (i64 q = 0; q < n; ++q) {
                const double *w = down.data() + a * no;
                double *target = dst + (no + a) * n3 + q * n2;
                const double *first = dst + q * n2;
                if (q >= no) {
                    add_written(target, first, n3, w, n, no, mode, sum, kPiece);
                    continue;
                }
                for (i64 lo = 0; lo < n2; lo += kPiece) {
                    const i64 len = std::min(kPiece, n2 - lo);
                    for (i64 m = 0; m < len; ++m) sum[m] = 0.0;
                    for (i64 k = 0; k < no; ++k) {
                        const double wk = w[k];
                        const double *from = first + k * n3 + lo;
                        for (i64 m = 0; m < len; ++m) sum[m] += wk * from[m];
                    }
                    double *to = target + lo;
                    for (i64 m = 0; m < len; ++m) to[m] += sum[m];
                }
            }
    }
}

}  // namespace

extern "C" void itercc_dress_integrals(double *dst, const double *src, const int64_t *ss, int64_t n, int64_t no,
                                       const double *t1, int threads)
{
    // dst = src dressed with T1 (no x nv) along all four axes: the occupied
    // rows of axes 2 and 3 gain t1 times the virtual rows, the virtual rows of
    // axes 0 and 1 gain -t1^T times the occupied rows (`dressed_integrals`).
    // src is read through its element strides ss[0..3] (a transposed view of
    // the ERIs needs no copy); dst is C-ordered. The four maps act on
    // different axes and commute, so two passes do it: slab by slab of axis 0,
    // each (p, q) plane is read once and dressed along axes 2 and 3 in cache,
    // then the slab (still cached) along axis 1; a final streaming pass dresses
    // axis 0. Every sum runs in order of its occupied or virtual index and is
    // then added, so an entry's value does not depend on the thread count.
    dress_integrals_impl(dst, src, ss, n, no, t1, threads, kFull);
}

extern "C" void itercc_dress_integrals_ccsd(double *dst, const double *src, const int64_t *ss, int64_t n,
                                            int64_t no, const double *t1, int threads)
{
    // What the T1-dressed CCSD residual reads, without reading g(vvvv): the
    // (p, q) planes with an occupied index as itercc_dress_integrals (the
    // same entries exactly); of a plane with p and q virtual only its
    // occupied-occupied block, and that without sum_ef t1(i,e) t1(j,f)
    // g(p,q,e,f), which the caller's ladder adds (`BareLadder`). The other
    // entries of those planes are left as they were.
    dress_integrals_impl(dst, src, ss, n, no, t1, threads, kCcsd);
}

extern "C" void itercc_dress_integrals_open(double *dst, const double *src, const int64_t *ss, int64_t n,
                                            int64_t no, const double *t1, int threads)
{
    // Every entry of itercc_dress_integrals with an occupied index, the same
    // values exactly; the virtual-virtual block of the planes with p and q
    // virtual, g~(vvvv), is neither written nor needed when the ladder takes
    // its sectors from itercc_ladder_sectors_t1. The bare g(vvvv) is read once,
    // for the occupied rows and columns of those planes.
    dress_integrals_impl(dst, src, ss, n, no, t1, threads, kOpen);
}

namespace {

void dress_derivative_impl(double *out, const double *g, i64 n, i64 no, const double *d, int threads, bool ccsd)
{
    const i64 nv = n - no;
    if (n <= 0) return;
    const i64 n2 = n * n;
    std::vector<double> dt(static_cast<size_t>(nv * no));     // (nv, no) = d^T
    for (i64 c = 0; c < nv; ++c)
        for (i64 i = 0; i < no; ++i) dt[static_cast<size_t>(c * no + i)] = d[i * nv + c];
#pragma omp parallel num_threads(threads)
    {
        std::vector<double> sum(static_cast<size_t>(std::max<i64>(n, 1)));
#pragma omp for collapse(2) schedule(static)
        for (i64 p = 0; p < n; ++p)
            for (i64 q = 0; q < n; ++q) {
                double *plane = out + (p * n + q) * n2;
                const double *in = g + (p * n + q) * n2;
                // CCSD: of an all-virtual (p, q) plane only the occupied block.
                const bool block = ccsd && p >= no && q >= no;
                const i64 rows = block ? no : n, cols = block ? no : n;
                for (i64 r = 0; r < rows; ++r) {           // leg 2 (rows); zero elsewhere
                    double *row = plane + r * n;
                    if (r >= no || nv <= 0) {
                        std::fill(row, row + n, 0.0);
                        continue;
                    }
                    for (i64 s = 0; s < cols; ++s) sum[s] = 0.0;
                    for (i64 c = 0; c < nv; ++c) {
                        const double w = d[r * nv + c];
                        const double *from = in + (no + c) * n;
                        for (i64 s = 0; s < cols; ++s) sum[s] += w * from[s];
                    }
                    std::copy(sum.data(), sum.data() + cols, row);
                }
                for (i64 r = 0; r < rows; ++r) {           // leg 3 (columns), vector over i
                    const double *from = in + r * n + no;
                    double *row = plane + r * n;
                    for (i64 i = 0; i < no; ++i) sum[i] = 0.0;
                    for (i64 c = 0; c < nv; ++c) {
                        const double x = from[c];
                        const double *t = dt.data() + c * no;
                        for (i64 i = 0; i < no; ++i) sum[i] += x * t[i];
                    }
                    for (i64 i = 0; i < no; ++i) row[i] += sum[i];
                }
                if (p >= no)                               // leg 0: g[k, q] planes
                    for (i64 k = 0; k < no; ++k) {
                        const double w = -d[k * nv + (p - no)];
                        const double *from = g + (k * n + q) * n2;
                        for (i64 r = 0; r < rows; ++r)
                            for (i64 s = 0; s < cols; ++s) plane[r * n + s] += w * from[r * n + s];
                    }
                if (q >= no)                               // leg 1: g[p, k] planes
                    for (i64 k = 0; k < no; ++k) {
                        const double w = -d[k * nv + (q - no)];
                        const double *from = g + (p * n + k) * n2;
                        for (i64 r = 0; r < rows; ++r)
                            for (i64 s = 0; s < cols; ++s) plane[r * n + s] += w * from[r * n + s];
                    }
            }
    }
}

}  // namespace

extern "C" void itercc_dress_derivative(double *out, const double *g, int64_t n, int64_t no, const double *d,
                                        int threads)
{
    // out = the first-order change of the T1-dressed g (C order) along d (no x
    // nv): each leg transformed once by d, the others left as they are
    // (`EOMSinglet._cache_dressing`): the virtual rows of legs 0 and 1 gain
    // -d^T times the occupied rows, the occupied rows of legs 2 and 3 gain d
    // times the virtual rows. One pass over g, no zero fill: each (p, q) plane
    // is written with its leg 2 terms, then gains its leg 3, leg 0 (p virtual;
    // the occupied slabs of g stay in cache) and leg 1 (q virtual) terms. Each
    // entry's sum runs in a fixed order, whatever the thread count.
    dress_derivative_impl(out, g, n, no, d, threads, false);
}

extern "C" void itercc_dress_derivative_ccsd(double *out, const double *g, int64_t n, int64_t no,
                                             const double *d, int threads)
{
    // `itercc_dress_derivative` where the CCSD residual reads it (the
    // derivative of `itercc_dress_integrals_ccsd`'s entries): the all-virtual
    // (p, q) planes get their occupied-occupied block only, the rest of those
    // planes is not written. The particle-particle ladder's derivative then
    // comes from the undressed g(vvvv) (`BareLadder.tangent`).
    dress_derivative_impl(out, g, n, no, d, threads, true);
}

extern "C" void itercc_gemm_narrow(const double *a, int64_t lda, const double *b, int64_t ldb, double *out,
                                   int64_t ldo, int64_t m, int64_t k, int64_t n, double alpha, double beta,
                                   int threads)
{
    // out = beta * out + alpha * a @ b for a right-hand side of few columns
    // (a: m x k, b: k x n, row strides lda, ldb, ldo; beta = 0 does not read
    // out). The pair-sector ladders multiply MB-sized blocks by one column per
    // occupied pair: memory bound, so each block row is streamed once, its dot
    // products with b's columns (transposed into one cached copy) formed four
    // rows at a time. Each entry sums in a fixed order, whatever the thread count.
    if (m <= 0 || n <= 0) return;
    std::vector<double> bt(static_cast<size_t>(n * std::max<i64>(k, 1)));
    for (i64 i = 0; i < k; ++i)
        for (i64 j = 0; j < n; ++j) bt[static_cast<size_t>(j * k + i)] = b[i * ldb + j];
    const i64 blocks = (m + 3) / 4;
#pragma omp parallel for schedule(static) num_threads(threads)
    for (i64 block = 0; block < blocks; ++block) {
        const i64 r0 = block * 4, r1 = std::min<i64>(m, r0 + 4);
        for (i64 j = 0; j < n; ++j) {
            const double *column = bt.data() + j * k;
            for (i64 r = r0; r < r1; ++r) {
                const double *row = a + r * lda;
                double sum = 0.0;
#pragma omp simd reduction(+ : sum)
                for (i64 i = 0; i < k; ++i) sum += row[i] * column[i];
                double *target = out + r * ldo + j;
                *target = beta == 0.0 ? alpha * sum : beta * *target + alpha * sum;
            }
        }
    }
}

extern "C" int itercc_project_symmetric(double *dst, const double *src, int64_t n, const int64_t *labels,
                                        double tolerance, double *stats, int threads)
{
    // dst = src without its symmetry-forbidden entries (labels XOR to nonzero),
    // in one pass on all threads, auditing the removed entries: stats = (count
    // of nonzero ones, largest magnitude, norm). The norm sums per-slab partial
    // sums in slab order, so it does not depend on the thread count. Returns 1
    // when an entry exceeds `tolerance` or is not finite (dst is then complete
    // but must not be used), else 0.
    if (n <= 0) return 0;
    const i64 n2 = n * n, n3 = n2 * n;
    std::vector<double> squares(size_t(n), 0.0), largest(size_t(n), 0.0);
    std::vector<i64> counts(size_t(n), 0);
    std::vector<int> bad(size_t(n), 0);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 p = 0; p < n; ++p) {
        double square = 0.0, peak = 0.0;
        i64 count = 0;
        int failed = 0;
        for (i64 q = 0; q < n; ++q)
            for (i64 r = 0; r < n; ++r) {
                const i64 pqr = labels[p] ^ labels[q] ^ labels[r];
                const double *in = src + p * n3 + q * n2 + r * n;
                double *out = dst + p * n3 + q * n2 + r * n;
                for (i64 s = 0; s < n; ++s) {
                    const double value = in[s];
                    if ((pqr ^ labels[s]) == 0) {
                        out[s] = value;
                        continue;
                    }
                    out[s] = 0.0;
                    const double magnitude = std::fabs(value);
                    if (!(magnitude <= tolerance)) failed = 1;        // also NaN
                    if (value != 0.0) ++count;
                    if (magnitude > peak) peak = magnitude;
                    square += value * value;
                }
            }
        squares[size_t(p)] = square;
        largest[size_t(p)] = peak;
        counts[size_t(p)] = count;
        bad[size_t(p)] = failed;
    }
    double square = 0.0, peak = 0.0;
    i64 count = 0;
    int failed = 0;
    for (i64 p = 0; p < n; ++p) {
        square += squares[size_t(p)];
        peak = std::max(peak, largest[size_t(p)]);
        count += counts[size_t(p)];
        failed |= bad[size_t(p)];
    }
    stats[0] = double(count);
    stats[1] = peak;
    stats[2] = std::sqrt(square);
    return failed;
}

extern "C" void itercc_unpack_s4(double *dst, const double *src, int64_t n, int threads)
{
    // dst[i, j, k, l] = src[ij, kl] for PySCF's 4-fold packed (pq|rs): pairs in
    // lower-triangular order, pq = p (p + 1) / 2 + q for p >= q. One row of src
    // (a few tens of KB) serves the n^2 entries of one (i, j) plane.
    if (n <= 0) return;
    const i64 n2 = n * n, npair = n * (n + 1) / 2;
    std::vector<i64> pair(static_cast<size_t>(n2));
    for (i64 k = 0; k < n; ++k)
        for (i64 l = 0; l < n; ++l) pair[size_t(k * n + l)] = k >= l ? k * (k + 1) / 2 + l : l * (l + 1) / 2 + k;
#pragma omp parallel for num_threads(threads) schedule(static)
    for (i64 ij = 0; ij < n2; ++ij) {
        const double *row = src + pair[size_t(ij)] * npair;
        double *out = dst + ij * n2;
        for (i64 kl = 0; kl < n2; ++kl) out[kl] = row[pair[size_t(kl)]];
    }
}

extern "C" void itercc_stacked_vecmat(int64_t m, int64_t kn, int64_t n, const double *w, const double *q,
                                      double alpha, double *out, int threads)
{
    // A vector times each matrix of a stack (NumPy runs it on one thread; at
    // 88 virtuals the stack is a 0.9 GB plane). Tasks are (matrix, block of
    // kBlock rows); each block's partial row is summed in order of k, and the
    // blocks are added in order.
    if (m <= 0 || n <= 0) return;
    constexpr i64 kBlock = 2048;
    const i64 blocks = std::max<i64>(1, (kn + kBlock - 1) / kBlock);
    std::vector<double> partial(size_t(m * blocks * n), 0.0);
#pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
    for (i64 i = 0; i < m; ++i)
        for (i64 b = 0; b < blocks; ++b) {
            double *p = partial.data() + (i * blocks + b) * n;
            const i64 lo = b * kBlock, hi = std::min(kn, lo + kBlock);
            for (i64 k = lo; k < hi; ++k) {
                const double wk = w[k];
                const double *row = q + (i * kn + k) * n;
                for (i64 z = 0; z < n; ++z) p[z] += wk * row[z];
            }
        }
    for (i64 i = 0; i < m; ++i)
        for (i64 z = 0; z < n; ++z) {
            double total = 0.0;
            for (i64 b = 0; b < blocks; ++b) total += partial[size_t((i * blocks + b) * n + z)];
            out[i * n + z] += alpha * total;
        }
}

extern "C" void itercc_ladder_sectors(const double *w, const int64_t *ws, const double *vv, int64_t nv,
                                      int64_t groups, const int64_t *p, const int64_t *q,
                                      const int64_t *even_start, const int64_t *even_off, const int64_t *r,
                                      const int64_t *s, const int64_t *odd_start, const int64_t *odd_off,
                                      double *even, double *odd, int accumulate, int threads)
{
    // One row of one sector block per task: a row (x_i, y_i) reads the plane
    // w[x_i, y_i, :, :] through the strides (a view of the dressed integrals
    // needs no copy) and writes only its sector's columns, so a block-diagonal
    // operator costs its blocks, not the full pair matrix. Entry arithmetic as
    // itercc_pair_matrices, then the one-body term 0.5 (vv x 1 + 1 x vv) of
    // the virtual ladder (`_cpu_triples.add_one_body`) for each ordering.
    const i64 ne = even_start[groups], nodd = odd_start[groups];
#pragma omp parallel for num_threads(threads) schedule(dynamic, 8)
    for (i64 i = 0; i < ne + nodd; ++i) {
        const bool is_even = i < ne;
        const i64 row = is_even ? i : i - ne;
        const int64_t *start = is_even ? even_start : odd_start;
        const int64_t *off = is_even ? even_off : odd_off;
        const int64_t *x = is_even ? p : r, *y = is_even ? q : s;
        i64 g = 0;
        while (start[g + 1] <= row) ++g;
        const i64 first = start[g], n = start[g + 1] - first;
        double *out = (is_even ? even : odd) + off[g] + (row - first) * n;
        const i64 xi = x[row], yi = y[row];
        const double *plane = w ? w + xi * ws[0] + yi * ws[1] : nullptr;
        for (i64 c = 0; c < n; ++c) {
            const i64 xj = x[first + c], yj = y[first + c];
            double value = 0.0;
            if (plane) {
                const double direct = plane[xj * ws[2] + yj * ws[3]];
                const double swapped = plane[yj * ws[2] + xj * ws[3]];
                value = is_even ? direct + swapped * double(xj != yj) : direct - swapped;
            }
            if (vv) {
                const double direct = 0.5 * ((yi == yj ? vv[xi * nv + xj] : 0.0)
                                             + (xi == xj ? vv[yi * nv + yj] : 0.0));
                const double swapped = 0.5 * ((yi == xj ? vv[xi * nv + yj] : 0.0)
                                              + (xi == yj ? vv[yi * nv + xj] : 0.0));
                value += is_even ? direct + swapped * double(xj != yj) : direct - swapped;
            }
            out[c] = accumulate ? out[c] + value : value;
        }
    }
}

extern "C" void itercc_ladder_sectors_t1(const double *g, const int64_t *gs, int64_t n, int64_t no,
                                         const double *t1, int64_t groups, const int64_t *p, const int64_t *q,
                                         const int64_t *even_start, const int64_t *even_off, const int64_t *r,
                                         const int64_t *s, const int64_t *odd_start, const int64_t *odd_off,
                                         const double *base_even, const double *base_odd, double *even,
                                         double *odd, int threads)
{
    // The sector blocks of the T1-dressed g~(vvvv) (as itercc_ladder_sectors
    // builds them from a dressed tensor) from those of the bare g(vvvv),
    // `base_*`, built once: g~(abcd) = g(abcd) - sum_k t(k,a) Q(k,b,c,d)
    // - sum_k t(k,b) Q(k,a,d,c) with Q(k,b,c,d) = g(k,b,c,d) - 0.5 sum_l t(l,b)
    // g(k,l,c,d), from the bare g read through its element strides gs. A row
    // (x, y) of a block then gains -sum_k [t(k,x) S(k,y,.) +- t(k,y) S(k,x,.)],
    // S the even (+) or odd (-) column combination of Q over the block's pairs:
    // o v^3 values formed first, then rank-o updates of the rows, so a pass
    // over the sectors replaces dressing g(vvvv) and reading it again.
    // Each entry sums in a fixed order, whatever the thread count.
    const i64 nv = n - no, ne = even_start[groups], nodd = odd_start[groups];
    if (nv <= 0) return;
    const i64 width = ne + nodd;
    std::unique_ptr<double[]> combos(new double[size_t(std::max<i64>(no * nv * width, 1))]);
    i64 widest = 1;
    for (i64 b = 0; b < groups; ++b) {
        widest = std::max<i64>(widest, even_start[b + 1] - even_start[b]);
        widest = std::max<i64>(widest, odd_start[b + 1] - odd_start[b]);
    }
#pragma omp parallel num_threads(threads)
    {
        std::vector<double> plane(static_cast<size_t>(nv * nv));
        std::vector<double> sum(static_cast<size_t>(widest));
#pragma omp for schedule(dynamic, 1)
        for (i64 kb = 0; kb < no * nv; ++kb) {
            const i64 k = kb / nv, b = kb % nv;
            const double *from = g + k * gs[0] + (no + b) * gs[1] + no * gs[2] + no * gs[3];
            for (i64 c = 0; c < nv; ++c)
                for (i64 d = 0; d < nv; ++d) plane[size_t(c * nv + d)] = from[c * gs[2] + d * gs[3]];
            for (i64 l = 0; l < no; ++l) {
                const double w = -0.5 * t1[l * nv + b];
                const double *gl = g + k * gs[0] + l * gs[1] + no * gs[2] + no * gs[3];
                for (i64 c = 0; c < nv; ++c)
                    for (i64 d = 0; d < nv; ++d) plane[size_t(c * nv + d)] += w * gl[c * gs[2] + d * gs[3]];
            }
            double *out = combos.get() + kb * width;
            for (i64 j = 0; j < ne; ++j) {
                const i64 x = p[j], y = q[j];
                out[j] = plane[size_t(x * nv + y)] + (x != y ? plane[size_t(y * nv + x)] : 0.0);
            }
            for (i64 j = 0; j < nodd; ++j) {
                const i64 x = r[j], y = s[j];
                out[ne + j] = plane[size_t(x * nv + y)] - plane[size_t(y * nv + x)];
            }
        }
#pragma omp for schedule(dynamic, 8)
        for (i64 i = 0; i < width; ++i) {
            const bool is_even = i < ne;
            const i64 row = is_even ? i : i - ne;
            const int64_t *start = is_even ? even_start : odd_start;
            const int64_t *off = is_even ? even_off : odd_off;
            const int64_t *x = is_even ? p : r, *y = is_even ? q : s;
            i64 blk = 0;
            while (start[blk + 1] <= row) ++blk;
            const i64 first = start[blk], m = start[blk + 1] - first;
            const i64 column = (is_even ? 0 : ne) + first;     // this block's columns in the combinations
            const size_t at = size_t(off[blk] + (row - first) * m);
            const double *base = (is_even ? base_even : base_odd) + at;
            double *target = (is_even ? even : odd) + at;
            const double sign = is_even ? 1.0 : -1.0;
            const i64 xi = x[row], yi = y[row];
            for (i64 c = 0; c < m; ++c) sum[size_t(c)] = 0.0;
            for (i64 k = 0; k < no; ++k) {
                const double wx = t1[k * nv + xi], wy = sign * t1[k * nv + yi];
                const double *sy = combos.get() + (k * nv + yi) * width + column;
                const double *sx = combos.get() + (k * nv + xi) * width + column;
                for (i64 c = 0; c < m; ++c) sum[size_t(c)] += wx * sy[c] + wy * sx[c];
            }
            for (i64 c = 0; c < m; ++c) target[c] = base[c] - sum[size_t(c)];
        }
    }
}

extern "C" void itercc_pair_matrices(const double *w, int64_t nv, int64_t ne, const int64_t *p,
                                     const int64_t *q, int64_t no, const int64_t *r, const int64_t *s,
                                     double *even, double *odd, int threads)
{
    // Row by row: a row reads one nv x nv block of w (it stays in cache), with
    // NumPy's arithmetic for the even (+ w * flag) and odd (-) matrices.
    const i64 nv2 = nv * nv;
#pragma omp parallel for num_threads(threads) schedule(dynamic, 16)
    for (i64 i = 0; i < ne + no; ++i) {
        if (i < ne) {
            const double *block = w + (p[i] * nv + q[i]) * nv2;
            double *row = even + i * ne;
            for (i64 j = 0; j < ne; ++j)
                row[j] = block[p[j] * nv + q[j]] + block[q[j] * nv + p[j]] * double(p[j] != q[j]);
        } else {
            const i64 k = i - ne;
            const double *block = w + (r[k] * nv + s[k]) * nv2;
            double *row = odd + k * no;
            for (i64 j = 0; j < no; ++j) row[j] = block[r[j] * nv + s[j]] - block[s[j] * nv + r[j]];
        }
    }
}
