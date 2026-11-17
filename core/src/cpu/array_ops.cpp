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
    const i64 nv = n - no;
    if (n <= 0) return;
    const i64 n2 = n * n, n3 = n2 * n;
    constexpr i64 kPiece = 256;
#pragma omp parallel num_threads(threads)
    {
        std::vector<double> sum(size_t(std::max<i64>(n, kPiece)));
#pragma omp for schedule(dynamic, 1)
        for (i64 p = 0; p < n; ++p) {
            double *slab = dst + p * n3;
            for (i64 q = 0; q < n; ++q) {
                const double *in = src + p * ss[0] + q * ss[1];
                double *out = slab + q * n2;
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
    if (no <= 0 || nv <= 0) return;
    std::vector<double> down(size_t(nv * no));             // (nv, no) = -t1^T
    for (i64 a = 0; a < nv; ++a)
        for (i64 k = 0; k < no; ++k) down[size_t(a * no + k)] = -t1[k * nv + a];
    dress_axis(dst, dst, n, 0, no, nv, 0, no, down.data(), threads);
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
