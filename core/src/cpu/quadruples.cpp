// Symmetry-packed restricted T4 on the CPU: the kernels of the QuadruplesLayout
// in itercc/_quadruples_cuda.py (same names without the device), for host arrays.
//
// A packed slot (i <= j <= k <= l) of irrep g holds the virtual quadruples
// (a, b, c, d) with irrep(a)^irrep(b)^irrep(c)^irrep(d) == g as consecutive
// dense blocks over (ga, gb, gc) (gd follows), each [a][b][c][d] over the
// local indices within the irreps. Virtuals are sorted by irrep, so in a dense
// row (nv^4 entries [a][b][c][d]) the allowed d of fixed (a, b, c) form one
// contiguous run, and so do their packed entries: most kernels copy those runs.
// Every output entry is written by one thread.
#include <algorithm>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include "itercc/core.h"
#include "scratch.h"

using i64 = int64_t;

namespace {

// First virtual of each irrep (virtuals are sorted by irrep).
struct Irreps {
    i64 first[8];
    explicit Irreps(const itercc_quad_layout *l)
    {
        i64 start = 0;
        for (i64 g = 0; g < 8; ++g) {
            first[g] = start;
            if (g < l->nirrep) start += l->vsize[g];
        }
    }
};

// Packed offset of the run (a, b, c, d0..) inside a slot of irrep g, d0 the
// first virtual of irrep gd = g ^ ga ^ gb ^ gc; the run has vsize[gd] entries.
inline i64 run_offset(const itercc_quad_layout *l, i64 g, i64 a, i64 b, i64 c, i64 *gd)
{
    const i64 ga = l->virr[a], gb = l->virr[b], gc = l->virr[c], n = l->nirrep;
    *gd = g ^ ga ^ gb ^ gc;
    return l->boff[((g * n + ga) * n + gb) * n + gc] +
           ((i64(l->vloc[a]) * l->vsize[gb] + l->vloc[b]) * l->vsize[gc] + l->vloc[c]) * l->vsize[*gd];
}

// Offset of (a, b, c, d) in a slot of irrep g, or -1 when forbidden.
inline i64 quad_offset(const itercc_quad_layout *l, i64 g, i64 a, i64 b, i64 c, i64 d)
{
    i64 gd;
    const i64 run = run_offset(l, g, a, b, c, &gd);
    if (l->virr[d] != gd) return -1;
    return run + l->vloc[d];
}

// Rank-four spin summation over output tiles of kEdge^4 entries. A term whose
// source has another output axis than d innermost would read with a stride
// along d; within a tile each term is read in its source's memory order (whole
// cache lines) and added into the tile's accumulator, which stays in L1. Each
// entry still adds the terms in order, as the line kernel and the CUDA kernel.
constexpr i64 kEdge = 8;
constexpr i64 kAcc1 = kEdge + 1, kAcc2 = kAcc1 * kEdge + 1;
constexpr i64 kAccSize = (kAcc2 * kEdge + 3) * kEdge;

struct Spin4 {
    i64 nv, nv2, nv3, nv4, tiles, per_slot, nterms;
    const i64 *perms;
    const double *weights;
    double alpha, divisor, beta;

    Spin4(i64 n, i64 terms, const i64 *p, const double *w, double a, double d, double b)
        : nv(n), nv2(n * n), nv3(n * n * n), nv4(n * n * n * n), tiles((n + kEdge - 1) / kEdge),
          per_slot(tiles * tiles * tiles * tiles), nterms(terms), perms(p), weights(w), alpha(a),
          divisor(d), beta(b)
    {
    }

    // One output tile of one row: out = alpha P(row) / divisor + beta out.
    void tile(const double *row, double *out, i64 index, double *acc) const
    {
        const i64 source_stride[4] = {nv3, nv2, nv, 1};
        // Padded accumulator strides: power-of-two strides would put a strided
        // term's writes into one L1 set.
        const i64 acc_stride[4] = {kAcc2 * kEdge + 3, kAcc1 * kEdge + 1, kAcc1, 1};
        i64 origin[4], extent[4], rest = index;
        for (i64 axis = 3; axis >= 0; --axis) {
            origin[axis] = (rest % tiles) * kEdge;
            extent[axis] = std::min(kEdge, nv - origin[axis]);
            rest /= tiles;
        }
        std::fill(acc, acc + kAccSize, 0.0);
        for (i64 t = 0; t < nterms; ++t) {
            // Source position k holds output axis p[k].
            const i64 *p = perms + 4 * t;
            const double w = weights[t];
            const double *base = row;
            for (i64 k = 0; k < 4; ++k) base += origin[p[k]] * source_stride[k];
            const i64 e0 = extent[p[0]], e1 = extent[p[1]], e2 = extent[p[2]], e3 = extent[p[3]];
            const i64 s0 = acc_stride[p[0]], s1 = acc_stride[p[1]], s2 = acc_stride[p[2]];
            const i64 s3 = acc_stride[p[3]];
            for (i64 i0 = 0; i0 < e0; ++i0)
                for (i64 i1 = 0; i1 < e1; ++i1)
                    for (i64 i2 = 0; i2 < e2; ++i2) {
                        const double *src = base + i0 * nv3 + i1 * nv2 + i2 * nv;
                        double *a = acc + i0 * s0 + i1 * s1 + i2 * s2;
                        if (s3 == 1)
                            for (i64 i3 = 0; i3 < e3; ++i3) a[i3] += w * src[i3];
                        else
                            for (i64 i3 = 0; i3 < e3; ++i3) a[i3 * s3] += w * src[i3];
                    }
        }
        for (i64 i0 = 0; i0 < extent[0]; ++i0)
            for (i64 i1 = 0; i1 < extent[1]; ++i1)
                for (i64 i2 = 0; i2 < extent[2]; ++i2) {
                    double *o = out + (origin[0] + i0) * nv3 + (origin[1] + i1) * nv2 +
                                (origin[2] + i2) * nv + origin[3];
                    const double *a = acc + i0 * acc_stride[0] + i1 * acc_stride[1] + i2 * acc_stride[2];
                    if (beta == 0.0)
                        for (i64 i3 = 0; i3 < extent[3]; ++i3) o[i3] = alpha * a[i3] / divisor + 0.0;
                    else
                        for (i64 i3 = 0; i3 < extent[3]; ++i3) o[i3] = alpha * a[i3] / divisor + beta * o[i3];
                }
    }
};

// Rank-four spin summation, tile by tile. In place with at least as many rows
// as threads, each thread copies a whole row into its own buffer first;
// otherwise an in-place call reads a full copy.
void spin4_tiled(const double *source, double *target, i64 nslots, i64 nv, i64 nterms,
                 const i64 *perms, const double *weights, double alpha, double divisor, double beta,
                 int threads)
{
    const Spin4 spin(nv, nterms, perms, weights, alpha, divisor, beta);
    const i64 nv4 = spin.nv4;
    if (source == target && nslots >= threads) {
#pragma omp parallel num_threads(threads)
        {
            alignas(64) double acc[kAccSize];
            double *row = itercc::scratch(0).get(size_t(nv4));
#pragma omp for schedule(dynamic, 1)
            for (i64 s = 0; s < nslots; ++s) {
                std::memcpy(row, source + s * nv4, sizeof(double) * size_t(nv4));
                for (i64 tile = 0; tile < spin.per_slot; ++tile)
                    spin.tile(row, target + s * nv4, tile, acc);
            }
        }
        return;
    }
    if (source == target) {   // in place on few rows: read a copy written on all threads
        const i64 n = nslots * nv4;
        double *dst = itercc::scratch(1).get(size_t(n));
#pragma omp parallel for num_threads(threads) schedule(static)
        for (i64 i = 0; i < n; ++i) dst[i] = source[i];
        source = dst;
    }
#pragma omp parallel num_threads(threads)
    {
        alignas(64) double acc[kAccSize];
#pragma omp for collapse(2) schedule(static)
        for (i64 s = 0; s < nslots; ++s)
            for (i64 tile = 0; tile < spin.per_slot; ++tile)
                spin.tile(source + s * nv4, target + s * nv4, tile, acc);
    }
}

}  // namespace

extern "C" void itercc_quad_gather_boxes(const double *packed, double *tile, const int64_t *rows,
                                         const int64_t *templates, int64_t nbox, int64_t per_row,
                                         int64_t terms, int64_t nv, int flags, const double *weights,
                                         int threads)
{
    // flags: 1 ring terms (their association), 2 allowed entries only (the
    // symmetry zeros of the tile are left as they are).
    const bool ring = (flags & 1) != 0, zeros = (flags & 2) == 0;
    const i64 width = 6 + 5 * terms;
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
    for (i64 x = 0; x < nbox; ++x) {
        const i64 *row = rows + 3 * (x / per_row);
        const i64 *box = templates + (row[2] * per_row + x % per_row) * width;
        const i64 na = box[1], nb = box[2], nc = box[3], d0 = box[4], nd = box[5];
        const i64 toff = row[0] + box[0];
        const double *src = packed + row[1];
        const i64 *q = box + 6;
        for (i64 a = 0; a < na; ++a)
            for (i64 b = 0; b < nb; ++b) {
                double *line = tile + toff + (a * nv + b) * nv * nv;
                for (i64 c = 0; c < nc; ++c) {
                    double *out = line + c * nv;
                    if (zeros) {
                        std::fill(out, out + d0, 0.0);
                        std::fill(out + d0 + nd, out + nv, 0.0);
                    }
                    double *run = out + d0;
                    if (ring) {   // the association of the CUDA kernels
                        const double *s0 = src + q[0] + a * q[1] + b * q[2] + c * q[3];
                        const double *s1 = src + q[5] + a * q[6] + b * q[7] + c * q[8];
                        const double *s2 = src + q[10] + a * q[11] + b * q[12] + c * q[13];
                        const i64 t0 = q[4], t1 = q[9], t2 = q[14];
                        for (i64 d = 0; d < nd; ++d) run[d] = s0[d * t0] + (s1[d * t1] + s2[d * t2]);
                    } else {
                        std::fill(run, run + nd, 0.0);
                        for (i64 t = 0; t < terms; ++t) {
                            const i64 *r = q + 5 * t;
                            const double *s = src + r[0] + a * r[1] + b * r[2] + c * r[3];
                            const i64 step = r[4];
                            const double w = weights[t];
                            if (step == 1)
                                for (i64 d = 0; d < nd; ++d) run[d] += w * s[d];
                            else
                                for (i64 d = 0; d < nd; ++d) run[d] += w * s[d * step];
                        }
                    }
                }
            }
    }
}

extern "C" void itercc_quad_scatter(const itercc_quad_layout *l, const double *tile,
                                    const int32_t *plan, int64_t nrows, double alpha, double beta,
                                    double *packed, int threads)
{
    const i64 nv = l->nv, nv4 = nv * nv * nv * nv;
    const Irreps irreps(l);
#pragma omp parallel for collapse(2) num_threads(threads) schedule(dynamic, 1)
    for (i64 r = 0; r < nrows; ++r)
        for (i64 a = 0; a < nv; ++a) {
            const i64 slot = plan[5 * r];
            if (slot < 0) continue;
            const i64 g = l->sirr[slot];
            double *base = packed + l->soff[slot];
            for (i64 b = 0; b < nv; ++b)
                for (i64 c = 0; c < nv; ++c) {
                    i64 gd;
                    double *out = base + run_offset(l, g, a, b, c, &gd);
                    const i64 n = gd < l->nirrep ? l->vsize[gd] : 0;
                    const double *in = tile + r * nv4 + ((a * nv + b) * nv + c) * nv + irreps.first[gd & 7];
                    if (beta == 0.0)
                        for (i64 d = 0; d < n; ++d) out[d] = 0.0 + alpha * in[d];
                    else
                        for (i64 d = 0; d < n; ++d) out[d] = beta * out[d] + alpha * in[d];
                }
        }
}

extern "C" void itercc_quad_expand(const itercc_quad_layout *l, const double *segment, int64_t first,
                                   int64_t nslots, double *dense, int threads)
{
    const i64 nv = l->nv, nv4 = nv * nv * nv * nv, shift = l->soff[first];
    const Irreps irreps(l);
#pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
    for (i64 s = 0; s < nslots; ++s)
        for (i64 a = 0; a < nv; ++a) {
            const i64 slot = first + s, g = l->sirr[slot];
            const double *base = segment + l->soff[slot] - shift;
            for (i64 b = 0; b < nv; ++b)
                for (i64 c = 0; c < nv; ++c) {
                    i64 gd;
                    const double *in = base + run_offset(l, g, a, b, c, &gd);
                    double *out = dense + s * nv4 + ((a * nv + b) * nv + c) * nv;
                    std::fill(out, out + nv, 0.0);
                    if (gd < l->nirrep && l->vsize[gd] > 0)
                        std::memcpy(out + irreps.first[gd], in, sizeof(double) * size_t(l->vsize[gd]));
                }
        }
}

extern "C" void itercc_quad_place(const itercc_quad_layout *l, const double *segment, int64_t first,
                                  int64_t nslots, double *dense, int threads)
{
    // itercc_quad_expand without the symmetry zeros: only the allowed entries of
    // the dense rows are written.
    const i64 nv = l->nv, nv4 = nv * nv * nv * nv, shift = l->soff[first];
    const Irreps irreps(l);
#pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
    for (i64 s = 0; s < nslots; ++s)
        for (i64 a = 0; a < nv; ++a) {
            const i64 slot = first + s, g = l->sirr[slot];
            const double *base = segment + l->soff[slot] - shift;
            for (i64 b = 0; b < nv; ++b)
                for (i64 c = 0; c < nv; ++c) {
                    i64 gd;
                    const double *in = base + run_offset(l, g, a, b, c, &gd);
                    if (gd < l->nirrep && l->vsize[gd] > 0)
                        std::memcpy(dense + s * nv4 + ((a * nv + b) * nv + c) * nv + irreps.first[gd], in,
                                    sizeof(double) * size_t(l->vsize[gd]));
                }
        }
}

extern "C" void itercc_quad_zero_allowed(const itercc_quad_layout *l, int64_t slot, double *dense,
                                         int threads)
{
    // Zero only the allowed entries of slot `slot`'s dense tile (the entries
    // its blocks write and its scatter reads), a quarter of the tile in C2v.
    const i64 nv = l->nv, g = l->sirr[slot];
    const Irreps irreps(l);
#pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
    for (i64 a = 0; a < nv; ++a)
        for (i64 b = 0; b < nv; ++b)
            for (i64 c = 0; c < nv; ++c) {
                i64 gd;
                (void)run_offset(l, g, a, b, c, &gd);
                if (gd < l->nirrep && l->vsize[gd] > 0) {
                    double *run = dense + ((a * nv + b) * nv + c) * nv + irreps.first[gd];
                    std::fill(run, run + l->vsize[gd], 0.0);
                }
            }
}

extern "C" void itercc_quad_compress(const itercc_quad_layout *l, const double *dense, int64_t first,
                                     int64_t nslots, double *segment, int threads)
{
    const i64 nv = l->nv, nv4 = nv * nv * nv * nv, shift = l->soff[first];
    const Irreps irreps(l);
#pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
    for (i64 s = 0; s < nslots; ++s)
        for (i64 a = 0; a < nv; ++a) {
            const i64 slot = first + s, g = l->sirr[slot];
            double *base = segment + l->soff[slot] - shift;
            for (i64 b = 0; b < nv; ++b)
                for (i64 c = 0; c < nv; ++c) {
                    i64 gd;
                    double *out = base + run_offset(l, g, a, b, c, &gd);
                    const double *in = dense + s * nv4 + ((a * nv + b) * nv + c) * nv;
                    if (gd < l->nirrep && l->vsize[gd] > 0)
                        std::memcpy(out, in + irreps.first[gd], sizeof(double) * size_t(l->vsize[gd]));
                }
        }
}

extern "C" void itercc_quad_purify(const uint8_t *invalid, int64_t nslots, int64_t nv, double *dense,
                                   int threads)
{
    const i64 nv4 = nv * nv * nv * nv;
#pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
    for (i64 s = 0; s < nslots; ++s)
        for (i64 a = 0; a < nv; ++a) {
            double *row = dense + s * nv4 + a * nv * nv * nv;
            if (invalid[s]) {
                std::fill(row, row + nv * nv * nv, 0.0);
                continue;
            }
            for (i64 b = 0; b < nv; ++b)
                for (i64 c = 0; c < nv; ++c) {
                    double *line = row + (b * nv + c) * nv;
                    if (a == b && b == c) {
                        std::fill(line, line + nv, 0.0);
                        continue;
                    }
                    for (i64 d = 0; d < nv; ++d)
                        if ((a == b && b == d) || (c == d && (a == c || b == c))) line[d] = 0.0;
                }
        }
}

extern "C" void itercc_quad_divide(const itercc_quad_layout *l, double *packed, const double *eia,
                                   const int32_t *slots, int64_t first, int64_t nslots, int threads)
{
    // packed /= eia[o0, a] + eia[o1, b] + eia[o2, c] + eia[o3, d] on slots first..,
    // slots[4 s ..] = o: the Jacobi step straight on the packed entries (as
    // expand, divide and compress, in one pass).
    const i64 nv = l->nv;
    const Irreps irreps(l);
#pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
    for (i64 s = 0; s < nslots; ++s)
        for (i64 a = 0; a < nv; ++a) {
            const i64 slot = first + s, g = l->sirr[slot];
            const int32_t *o = slots + 4 * s;
            const double *e0 = eia + i64(o[0]) * nv, *e1 = eia + i64(o[1]) * nv;
            const double *e2 = eia + i64(o[2]) * nv, *e3 = eia + i64(o[3]) * nv;
            double *base = packed + l->soff[slot];
            for (i64 b = 0; b < nv; ++b)
                for (i64 c = 0; c < nv; ++c) {
                    i64 gd;
                    double *run = base + run_offset(l, g, a, b, c, &gd);
                    const i64 d0 = irreps.first[gd], n = l->vsize[gd];
                    const double abc = e0[a] + e1[b] + e2[c];
                    for (i64 d = 0; d < n; ++d) run[d] /= abc + e3[d0 + d];
                }
        }
}

namespace {

// The source of one output box of a packed row: source position k holds output
// axis sel[k], so within the box the source address is affine in the output's
// local indices (base + sum_j local_j * stride[j]).
struct Affine {
    i64 base, stride[4];
};

inline Affine source_box(const itercc_quad_layout *l, i64 h, const i64 *g, const i64 *sel)
{
    const i64 N = l->nirrep;
    i64 gs[4], n[4];
    for (int k = 0; k < 4; ++k) {
        gs[k] = g[sel[k]];
        n[k] = l->vsize[gs[k]];
    }
    const i64 position[4] = {n[1] * n[2] * n[3], n[2] * n[3], n[3], 1};
    Affine a;
    a.base = l->boff[((h * N + gs[0]) * N + gs[1]) * N + gs[2]];
    for (int k = 0; k < 4; ++k) a.stride[sel[k]] = position[k];
    return a;
}

// Boxes (ga, gb, gc) of a packed row of irrep h, flattened: box b has ga = b / N^2, ...
inline void box_irreps(const itercc_quad_layout *l, i64 h, i64 box, i64 *g)
{
    const i64 N = l->nirrep;
    g[0] = box / (N * N);
    g[1] = (box / N) % N;
    g[2] = box % N;
    g[3] = h ^ g[0] ^ g[1] ^ g[2];
}

enum class BoxOp { assign, add, add_weighted };

// One output box (extents ext, C order) from its source box `a` under the
// permutation sel (source position k holds output axis sel[k]): dst(x) = /
// += / += w * src(x[sel]). The source box is read in its own memory order
// (its innermost position is contiguous) and the output box, which stays in
// L1/L2, is written with strides: reading the source along a strided output
// line would take one cache line per element. Each entry gets the same
// arithmetic as the line-by-line loops, so results are bitwise unchanged.
inline void box_permute(double *dst, const i64 *ext, const double *src, const Affine &a, const i64 *sel,
                        double w, BoxOp op)
{
    const i64 ostride[4] = {ext[1] * ext[2] * ext[3], ext[2] * ext[3], ext[3], 1};
    i64 n[4], p[4], o[4];
    for (int k = 0; k < 4; ++k) {
        n[k] = ext[sel[k]];
        p[k] = a.stride[sel[k]];        // p[3] == 1: the source's innermost position
        o[k] = ostride[sel[k]];
    }
    const double *base = src + a.base;
    const i64 step = o[3];
    for (i64 s0 = 0; s0 < n[0]; ++s0)
        for (i64 s1 = 0; s1 < n[1]; ++s1)
            for (i64 s2 = 0; s2 < n[2]; ++s2) {
                const double *x = base + s0 * p[0] + s1 * p[1] + s2 * p[2];
                double *d = dst + s0 * o[0] + s1 * o[1] + s2 * o[2];
                if (step == 1) {
                    if (op == BoxOp::assign)
                        for (i64 m = 0; m < n[3]; ++m) d[m] = x[m];
                    else if (op == BoxOp::add)
                        for (i64 m = 0; m < n[3]; ++m) d[m] += x[m];
                    else
                        for (i64 m = 0; m < n[3]; ++m) d[m] += w * x[m];
                } else {
                    if (op == BoxOp::assign)
                        for (i64 m = 0; m < n[3]; ++m) d[m * step] = x[m];
                    else if (op == BoxOp::add)
                        for (i64 m = 0; m < n[3]; ++m) d[m * step] += x[m];
                    else
                        for (i64 m = 0; m < n[3]; ++m) d[m * step] += w * x[m];
                }
            }
}

}  // namespace

extern "C" void itercc_quad_gather_rows(const itercc_quad_layout *l, const double *packed, int64_t nrows,
                                        const int64_t *row_slot, const int32_t *row_order,
                                        const int64_t *row_offset, double *rows, int threads)
{
    // Packed rows of a T4 block: row r holds T4 of slot row_slot[r] with its
    // virtual axes in the row's own occupied order (packed position k holds row
    // axis row_order[4 r + k]), stored at rows + row_offset[r] in the packed box
    // layout of the slot's irrep: only allowed entries, no symmetry zeros. Each
    // box is a strided copy of one box of the slot.
    const i64 N = l->nirrep, boxes = N * N * N;
#pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
    for (i64 r = 0; r < nrows; ++r)
        for (i64 box = 0; box < boxes; ++box) {
            const i64 slot = row_slot[r], h = l->sirr[slot];
            i64 g[4], sel[4];
            box_irreps(l, h, box, g);
            const i64 na = l->vsize[g[0]], nb = l->vsize[g[1]], nc = l->vsize[g[2]], nd = l->vsize[g[3]];
            if (!(na && nb && nc && nd)) continue;
            for (int k = 0; k < 4; ++k) sel[k] = row_order[4 * r + k];
            const Affine a = source_box(l, h, g, sel);
            const i64 ext[4] = {na, nb, nc, nd};
            box_permute(rows + row_offset[r] + l->boff[((h * N + g[0]) * N + g[1]) * N + g[2]], ext,
                        packed + l->soff[slot], a, sel, 1.0, BoxOp::assign);
        }
}

extern "C" void itercc_quad_rows_permute(const itercc_quad_layout *l, const double *in, double *out,
                                         int64_t nrows, const int64_t *row_offset,
                                         const int32_t *row_irrep, int64_t nterms, const int64_t *perms,
                                         const double *weights, int ring, int threads)
{
    // Packed rows (row r at row_offset[r], box layout of irrep row_irrep[r]):
    // out(x) = sum_t w_t in(x[p_t0], x[p_t1], x[p_t2], x[p_t3]), terms added in
    // order from zero as itercc_quad_gather_boxes does; with `ring`, the three
    // ring terms as t0 + (t1 + t2). in and out must not overlap. Each term reads
    // one box of the row per output box, affinely.
    const i64 N = l->nirrep, boxes = N * N * N;
    // Box by box in turn over all threads (not a row per thread): the terms
    // read other boxes of the same row, which then stay in the shared cache
    // (with a row per thread, 56 rows of 10 MB at 47 virtuals, each term's
    // read went to memory).
#pragma omp parallel for collapse(2) num_threads(threads) schedule(dynamic, 1)
    for (i64 r = 0; r < nrows; ++r)
        for (i64 box = 0; box < boxes; ++box) {
            const i64 h = row_irrep[r];
            i64 g[4];
            box_irreps(l, h, box, g);
            const i64 na = l->vsize[g[0]], nb = l->vsize[g[1]], nc = l->vsize[g[2]], nd = l->vsize[g[3]];
            if (!(na && nb && nc && nd)) continue;
            Affine a[24];
            const i64 count = nterms < 24 ? nterms : 24;
            for (i64 t = 0; t < count; ++t) a[t] = source_box(l, h, g, perms + 4 * t);
            const double *src = in + row_offset[r];
            double *dst = out + row_offset[r] + l->boff[((h * N + g[0]) * N + g[1]) * N + g[2]];
            if (!ring) {
                const i64 ext[4] = {na, nb, nc, nd};
                std::fill(dst, dst + na * nb * nc * nd, 0.0);
                for (i64 t = 0; t < count; ++t)
                    box_permute(dst, ext, src, a[t], perms + 4 * t, weights[t], BoxOp::add_weighted);
                continue;
            }
            for (i64 i = 0; i < na; ++i)
                for (i64 j = 0; j < nb; ++j)
                    for (i64 k = 0; k < nc; ++k) {
                        double *d = dst + ((i * nb + j) * nc + k) * nd;
                        const double *s0 = src + a[0].base + i * a[0].stride[0] + j * a[0].stride[1] + k * a[0].stride[2];
                        const double *s1 = src + a[1].base + i * a[1].stride[0] + j * a[1].stride[1] + k * a[1].stride[2];
                        const double *s2 = src + a[2].base + i * a[2].stride[0] + j * a[2].stride[1] + k * a[2].stride[2];
                        const i64 t0 = a[0].stride[3], t1 = a[1].stride[3], t2 = a[2].stride[3];
                        for (i64 m = 0; m < nd; ++m) d[m] = s0[m * t0] + (s1[m * t1] + s2[m * t2]);
                    }
        }
}

namespace {

// itercc_quad_gather_permute into rows of T (double, or float: each box is
// formed in double in a per-thread buffer and then rounded).
template <typename T>
void gather_permute(const itercc_quad_layout *l, const double *packed, int64_t nrows,
                    const int64_t *row_slot, const int32_t *row_order, const int64_t *row_offset,
                    int64_t nterms, const int64_t *perms, const double *weights, int ring, T *rows,
                    int threads)
{
    // itercc_quad_gather_rows followed by itercc_quad_rows_permute, without the
    // gathered rows: term t of row r reads the row's slot through the composed
    // permutation (slot position k holds row axis perms[t][row_order[r][k]]).
    // The gathered values are exact copies, so the results are those of the
    // two kernels.
    const i64 N = l->nirrep, boxes = N * N * N;
    // Box by box in turn over all threads, so the terms' reads of the row's
    // slot hit the shared cache (see itercc_quad_rows_permute).
#pragma omp parallel for collapse(2) num_threads(threads) schedule(dynamic, 1)
    for (i64 r = 0; r < nrows; ++r)
        for (i64 box = 0; box < boxes; ++box) {
            const i64 slot = row_slot[r], h = l->sirr[slot];
            i64 g[4];
            box_irreps(l, h, box, g);
            const i64 na = l->vsize[g[0]], nb = l->vsize[g[1]], nc = l->vsize[g[2]], nd = l->vsize[g[3]];
            if (!(na && nb && nc && nd)) continue;
            Affine a[24];
            const i64 count = nterms < 24 ? nterms : 24;
            for (i64 t = 0; t < count; ++t) {
                i64 sel[4];
                for (int k = 0; k < 4; ++k) sel[k] = perms[4 * t + row_order[4 * r + k]];
                a[t] = source_box(l, h, g, sel);
            }
            const double *src = packed + l->soff[slot];
            T *out = rows + row_offset[r] + l->boff[((h * N + g[0]) * N + g[1]) * N + g[2]];
            const i64 volume = na * nb * nc * nd;
            double *dst;
            if constexpr (std::is_same_v<T, double>) {
                dst = out;
            } else {
                dst = itercc::scratch(0).get(size_t(volume));   // released between phases
            }
            if (!ring) {
                const i64 ext[4] = {na, nb, nc, nd};
                std::fill(dst, dst + volume, 0.0);
                for (i64 t = 0; t < count; ++t) {
                    i64 sel[4];
                    for (int k = 0; k < 4; ++k) sel[k] = perms[4 * t + row_order[4 * r + k]];
                    box_permute(dst, ext, src, a[t], sel, weights[t], BoxOp::add_weighted);
                }
                if constexpr (!std::is_same_v<T, double>)
                    for (i64 x = 0; x < volume; ++x) out[x] = T(dst[x]);
                continue;
            }
            for (i64 i = 0; i < na; ++i)
                for (i64 j = 0; j < nb; ++j)
                    for (i64 k = 0; k < nc; ++k) {
                        double *d = dst + ((i * nb + j) * nc + k) * nd;
                        const double *s0 = src + a[0].base + i * a[0].stride[0] + j * a[0].stride[1] + k * a[0].stride[2];
                        const double *s1 = src + a[1].base + i * a[1].stride[0] + j * a[1].stride[1] + k * a[1].stride[2];
                        const double *s2 = src + a[2].base + i * a[2].stride[0] + j * a[2].stride[1] + k * a[2].stride[2];
                        const i64 t0 = a[0].stride[3], t1 = a[1].stride[3], t2 = a[2].stride[3];
                        for (i64 m = 0; m < nd; ++m) d[m] = s0[m * t0] + (s1[m * t1] + s2[m * t2]);
                    }
            if constexpr (!std::is_same_v<T, double>)
                for (i64 x = 0; x < volume; ++x) out[x] = T(dst[x]);
        }
}

}  // namespace

extern "C" void itercc_quad_gather_permute(const itercc_quad_layout *l, const double *packed, int64_t nrows,
                                           const int64_t *row_slot, const int32_t *row_order,
                                           const int64_t *row_offset, int64_t nterms, const int64_t *perms,
                                           const double *weights, int ring, double *rows, int threads)
{
    gather_permute(l, packed, nrows, row_slot, row_order, row_offset, nterms, perms, weights, ring, rows,
                   threads);
}

extern "C" void itercc_quad_gather_permute_f32(const itercc_quad_layout *l, const double *packed,
                                               int64_t nrows, const int64_t *row_slot,
                                               const int32_t *row_order, const int64_t *row_offset,
                                               int64_t nterms, const int64_t *perms, const double *weights,
                                               int ring, float *rows, int threads)
{
    gather_permute(l, packed, nrows, row_slot, row_order, row_offset, nterms, perms, weights, ring, rows,
                   threads);
}

extern "C" void itercc_quad_row_add_packed(const itercc_quad_layout *l, const double *rows,
                                           const int64_t *row_offset, const int32_t *entries,
                                           int64_t nentries, double *packed, int threads)
{
    // itercc_quad_row_add for packed rows: packed[slot](x) += row[x[m0], .., x[m3]]
    // with row `row` at rows + row_offset[row] in the packed box layout of the
    // slot's irrep (the same occupied indices, so the same irrep). Each target
    // box reads one box of the row, affinely.
    const i64 N = l->nirrep, boxes = N * N * N;
#pragma omp parallel for collapse(2) num_threads(threads) schedule(dynamic, 4)
    for (i64 e = 0; e < nentries; ++e)
        for (i64 box = 0; box < boxes; ++box) {
            const int32_t *entry = entries + 6 * e;
            const i64 slot = entry[0], h = l->sirr[slot];
            i64 g[4];
            box_irreps(l, h, box, g);
            const i64 na = l->vsize[g[0]], nb = l->vsize[g[1]], nc = l->vsize[g[2]], nd = l->vsize[g[3]];
            if (!(na && nb && nc && nd)) continue;
            const i64 sel[4] = {entry[2], entry[3], entry[4], entry[5]};
            const Affine a = source_box(l, h, g, sel);
            const i64 ext[4] = {na, nb, nc, nd};
            box_permute(packed + l->soff[slot] + l->boff[((h * N + g[0]) * N + g[1]) * N + g[2]], ext,
                        rows + row_offset[entry[1]], a, sel, 1.0, BoxOp::add);
        }
}

extern "C" void itercc_quad_row_add(const itercc_quad_layout *l, const double *rows,
                                    const int32_t *entries, int64_t nentries, double *packed,
                                    int threads)
{
    const i64 nv = l->nv, nv4 = nv * nv * nv * nv;
    const Irreps irreps(l);
    // Entries of one list write distinct slots.
#pragma omp parallel for collapse(2) num_threads(threads) schedule(dynamic, 1)
    for (i64 e = 0; e < nentries; ++e)
        for (i64 a = 0; a < nv; ++a) {
            const int32_t *entry = entries + 6 * e;
            const i64 slot = entry[0], g = l->sirr[slot];
            const i64 m0 = entry[2], m1 = entry[3], m2 = entry[4], m3 = entry[5];
            const double *source = rows + i64(entry[1]) * nv4;
            double *base = packed + l->soff[slot];
            i64 x[4];
            x[0] = a;
            for (i64 b = 0; b < nv; ++b)
                for (i64 c = 0; c < nv; ++c) {
                    i64 gd;
                    double *out = base + run_offset(l, g, a, b, c, &gd);
                    if (gd >= l->nirrep) continue;
                    x[1] = b;
                    x[2] = c;
                    const i64 d0 = irreps.first[gd], n = l->vsize[gd];
                    for (i64 k = 0; k < n; ++k) {
                        x[3] = d0 + k;
                        out[k] += source[((x[m0] * nv + x[m1]) * nv + x[m2]) * nv + x[m3]];
                    }
                }
        }
}

extern "C" void itercc_quad_spin(const double *source, double *target, int64_t nslots, int64_t nv,
                                 int64_t rank, int64_t nterms, const int64_t *perms,
                                 const double *weights, double alpha, double divisor, double beta,
                                 int threads)
{
    // target = alpha * (sum_t w_t source[perm_t(x)]) / divisor + beta * target over
    // rows of nv^rank entries; the terms are added in order, as in the CUDA kernel.
    i64 power[5] = {1, nv, nv * nv, nv * nv * nv, nv * nv * nv * nv};
    if (rank == 4) {
        spin4_tiled(source, target, nslots, nv, nterms, perms, weights, alpha, divisor, beta, threads);
        return;
    }
    if (source == target) {   // in place: read a copy, written (and first touched) on all threads
        const i64 n = nslots * power[rank];
        double *dst = itercc::scratch(1).get(size_t(n));
#pragma omp parallel for num_threads(threads) schedule(static)
        for (i64 i = 0; i < n; ++i) dst[i] = source[i];
        source = dst;
    }
    const i64 stride = power[rank], outer = stride / nv;   // lines of the last axis
    // Source stride of each output axis per term: output axis perm[k] reads
    // position k, whose stride is nv^(rank-1-k).
    std::vector<i64> strides(size_t(nterms * rank));
    for (i64 t = 0; t < nterms; ++t)
        for (i64 k = 0; k < rank; ++k) strides[size_t(t * rank + perms[t * rank + k])] = power[rank - 1 - k];
#pragma omp parallel num_threads(threads)
    {
        std::vector<double> acc(static_cast<size_t>(nv));
#pragma omp for collapse(2) schedule(static)
        for (i64 s = 0; s < nslots; ++s)
            for (i64 line = 0; line < outer; ++line) {
                const double *row = source + s * stride;
                // Digits of the line (all axes but the last).
                i64 digit[4] = {0, 0, 0, 0}, rest = line;
                for (i64 axis = rank - 2; axis >= 0; --axis) {
                    digit[axis] = rest % nv;
                    rest /= nv;
                }
                std::fill(acc.begin(), acc.end(), 0.0);
                for (i64 t = 0; t < nterms; ++t) {
                    const i64 *st = strides.data() + t * rank;
                    i64 base = 0;
                    for (i64 axis = 0; axis + 1 < rank; ++axis) base += digit[axis] * st[axis];
                    const double *src = row + base;
                    const i64 step = st[rank - 1];
                    const double w = weights[t];
                    if (step == 1)
                        for (i64 d = 0; d < nv; ++d) acc[size_t(d)] += w * src[d];
                    else
                        for (i64 d = 0; d < nv; ++d) acc[size_t(d)] += w * src[d * step];
                }
                double *out = target + s * stride + line * nv;
                if (beta == 0.0)
                    for (i64 d = 0; d < nv; ++d) out[d] = alpha * acc[size_t(d)] / divisor + 0.0;
                else
                    for (i64 d = 0; d < nv; ++d) out[d] = alpha * acc[size_t(d)] / divisor + beta * out[d];
            }
    }
}

extern "C" void itercc_quad_project(const itercc_quad_layout *l, double *packed, int64_t first,
                                    int64_t count, const uint8_t *ties, const uint8_t *invalid,
                                    int64_t nterms, const int64_t *perms, const double *weights,
                                    double alpha, double divisor, int threads)
{
    // The projection of packed residual slots first .. first + count - 1, in
    // place and without dense rows: occupied ties (bit p of ties[s]: x = (x + x
    // with virtual axes p, p + 1 swapped) / 2, pairs in order), the spin
    // summation x = alpha * (sum_t w_t x[perm_t]) / divisor, terms in order
    // from zero, and purification (invalid slots and entries with coinciding
    // virtuals become zero). The arithmetic of each entry is that of
    // itercc_quad_tie, itercc_quad_spin and itercc_quad_purify on dense rows.
    // Each term reads one box of the slot per output box, affinely.
    const i64 N = l->nirrep, boxes = N * N * N;
    const Irreps irreps(l);
    i64 length[8] = {0};
    for (i64 h = 0; h < N; ++h)
        for (i64 box = 0; box < boxes; ++box) {
            i64 g[4];
            box_irreps(l, h, box, g);
            length[h] += i64(l->vsize[g[0]]) * l->vsize[g[1]] * l->vsize[g[2]] * l->vsize[g[3]];
        }
    const i64 longest = *std::max_element(length, length + 8);
    const i64 count_terms = nterms < 24 ? nterms : 24;
#pragma omp parallel num_threads(threads)
    {
        double *in = itercc::scratch(0).get(size_t(2 * longest));
        double *spare = in + longest;
#pragma omp for schedule(dynamic, 1)
        for (i64 n = 0; n < count; ++n) {
            const i64 slot = first + n, h = l->sirr[slot];
            double *row = packed + l->soff[slot];
            if (invalid[n]) {
                std::fill(row, row + length[h], 0.0);
                continue;
            }
            double *src = in, *dst = spare;
            std::memcpy(src, row, sizeof(double) * size_t(length[h]));
            for (int pair = 0; pair < 3; ++pair) {
                if (!(ties[n] & (1 << pair))) continue;
                i64 swap[4] = {0, 1, 2, 3};
                std::swap(swap[pair], swap[pair + 1]);
                for (i64 box = 0; box < boxes; ++box) {
                    i64 g[4];
                    box_irreps(l, h, box, g);
                    const i64 na = l->vsize[g[0]], nb = l->vsize[g[1]], nc = l->vsize[g[2]], nd = l->vsize[g[3]];
                    if (!(na && nb && nc && nd)) continue;
                    const Affine a = source_box(l, h, g, swap);
                    const i64 own = l->boff[((h * N + g[0]) * N + g[1]) * N + g[2]];
                    for (i64 i = 0; i < na; ++i)
                        for (i64 j = 0; j < nb; ++j)
                            for (i64 k = 0; k < nc; ++k) {
                                const i64 at = own + ((i * nb + j) * nc + k) * nd;
                                const double *x = src + at;
                                const double *y = src + a.base + i * a.stride[0] + j * a.stride[1] + k * a.stride[2];
                                double *d = dst + at;
                                const i64 step = a.stride[3];
                                for (i64 m = 0; m < nd; ++m) d[m] = 0.5 * (x[m] + y[m * step]);
                            }
                }
                std::swap(src, dst);
            }
            for (i64 box = 0; box < boxes; ++box) {
                i64 g[4];
                box_irreps(l, h, box, g);
                const i64 na = l->vsize[g[0]], nb = l->vsize[g[1]], nc = l->vsize[g[2]], nd = l->vsize[g[3]];
                if (!(na && nb && nc && nd)) continue;
                Affine a[24];
                for (i64 t = 0; t < count_terms; ++t) a[t] = source_box(l, h, g, perms + 4 * t);
                double *out = row + l->boff[((h * N + g[0]) * N + g[1]) * N + g[2]];
                const i64 f0 = irreps.first[g[0]], f1 = irreps.first[g[1]], f2 = irreps.first[g[2]];
                const i64 f3 = irreps.first[g[3]];
                const i64 ext[4] = {na, nb, nc, nd};
                std::fill(out, out + na * nb * nc * nd, 0.0);
                for (i64 t = 0; t < count_terms; ++t)
                    box_permute(out, ext, src, a[t], perms + 4 * t, weights[t], BoxOp::add_weighted);
                for (i64 i = 0; i < na; ++i)
                    for (i64 j = 0; j < nb; ++j)
                        for (i64 k = 0; k < nc; ++k) {
                            double *o = out + ((i * nb + j) * nc + k) * nd;
                            const i64 va = f0 + i, vb = f1 + j, vc = f2 + k;
                            for (i64 m = 0; m < nd; ++m) {
                                const i64 vd = f3 + m;
                                const bool purified = (va == vb && (vb == vc || vb == vd)) ||
                                                      (vc == vd && (va == vc || vb == vc));
                                o[m] = purified ? 0.0 : alpha * o[m] / divisor + 0.0;
                            }
                        }
            }
        }
    }
}

extern "C" void itercc_quad_tie(double *dense, const int64_t *slots, int64_t count, int64_t nv,
                                int64_t pair, int threads)
{
    // x = (x + x with virtual axes pair, pair + 1 swapped) / 2 on the listed rows,
    // entry by entry as NumPy's 0.5 * (x + x.swapaxes(...)); the innermost loop
    // runs along d unless d is swapped, then within one (c, d) plane.
    const i64 nv2 = nv * nv, nv3 = nv2 * nv, nv4 = nv3 * nv;
#pragma omp parallel for collapse(2) num_threads(threads) schedule(dynamic, 1)
    for (i64 n = 0; n < count; ++n)
        for (i64 outer = 0; outer < nv; ++outer) {
            double *row = dense + slots[n] * nv4;
            if (pair == 0) {          // (a, b) with a = outer; (c, d) contiguous
                for (i64 q = outer + 1; q < nv; ++q) {
                    double *x = row + outer * nv3 + q * nv2, *y = row + q * nv3 + outer * nv2;
                    for (i64 o = 0; o < nv2; ++o) x[o] = y[o] = 0.5 * (x[o] + y[o]);
                }
            } else if (pair == 1) {   // (b, c) for a = outer; d contiguous
                double *plane = row + outer * nv3;
                for (i64 b = 0; b < nv; ++b)
                    for (i64 c = b + 1; c < nv; ++c) {
                        double *x = plane + b * nv2 + c * nv, *y = plane + c * nv2 + b * nv;
                        for (i64 d = 0; d < nv; ++d) x[d] = y[d] = 0.5 * (x[d] + y[d]);
                    }
            } else {                  // (c, d) for a = outer, every b
                for (i64 b = 0; b < nv; ++b) {
                    double *plane = row + outer * nv3 + b * nv2;
                    for (i64 c = 0; c < nv; ++c)
                        for (i64 d = c + 1; d < nv; ++d) {
                            const double value = 0.5 * (plane[c * nv + d] + plane[d * nv + c]);
                            plane[c * nv + d] = value;
                            plane[d * nv + c] = value;
                        }
                }
            }
        }
}
