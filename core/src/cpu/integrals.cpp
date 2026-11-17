// Integral transformations (`itercc.integrals`): the AO-to-MO transformation
// of two-electron integrals as two half transformations of GEMMs (run by the
// caller on the core's executor) around the packing, unpacking and scatter
// passes here. Every output entry has one writer; nothing depends on the
// thread count.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "itercc/core.h"

namespace {

using i64 = int64_t;

constexpr i64 kTile = 64;

// The pair (r, s), r >= s, of the lower-triangular pair index P = r(r+1)/2 + s.
inline void pair_of(i64 index, i64 &r, i64 &s)
{
    r = i64((std::sqrt(8.0 * double(index) + 1.0) - 1.0) / 2.0);
    while (r * (r + 1) / 2 > index) --r;
    while ((r + 1) * (r + 2) / 2 <= index) ++r;
    s = index - r * (r + 1) / 2;
}

}  // namespace

extern "C" void itercc_unpack_s8_s4(const double *s8, int64_t npair, double *s4, int threads)
{
    // s4 (npair x npair) = the symmetric pair matrix whose lower triangle is
    // packed row by row in s8 (PySCF's 8-fold packed integrals, aosym="s8").
    // Tiles of the lower triangle write their transposes from cache.
    const i64 tiles = (npair + kTile - 1) / kTile;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (i64 t = 0; t < tiles * (tiles + 1) / 2; ++t) {
        i64 bi = 0, bj = 0;
        pair_of(t, bi, bj);
        const i64 i0 = bi * kTile, i1 = std::min(npair, i0 + kTile);
        const i64 j0 = bj * kTile, j1 = std::min(npair, j0 + kTile);
        for (i64 i = i0; i < i1; ++i) {
            const double *row = s8 + i * (i + 1) / 2;
            const i64 jend = std::min(j1, i + 1);
            for (i64 j = j0; j < jend; ++j) s4[i * npair + j] = row[j];
        }
        for (i64 j = j0; j < j1; ++j)
            for (i64 i = std::max(i0, j + 1); i < i1; ++i) s4[j * npair + i] = s4[i * npair + j];
    }
}

extern "C" void itercc_unpack_pairs(const double *rows, int64_t items, int64_t ld, int64_t n, double *out,
                                    int threads)
{
    // out[b] (n x n, symmetric) from the packed lower triangle rows[b * ld + P],
    // P = mu(mu+1)/2 + nu.
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for (i64 b = 0; b < items; ++b)
        for (i64 mu = 0; mu < n; ++mu) {
            const double *row = rows + b * ld + mu * (mu + 1) / 2;
            double *m = out + b * n * n;
            for (i64 nu = 0; nu <= mu; ++nu) {
                m[mu * n + nu] = row[nu];
                m[nu * n + mu] = row[nu];
            }
        }
}

extern "C" void itercc_pack_pairs(const double *blocks, int64_t items, int64_t n, double *out, int64_t ld,
                                  int threads)
{
    // out[b * ld + P] = blocks[b][r][s] for P = r(r+1)/2 + s, r >= s (the
    // lower triangle of each n x n block).
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for (i64 b = 0; b < items; ++b)
        for (i64 r = 0; r < n; ++r) {
            const double *m = blocks + (b * n + r) * n;
            double *to = out + b * ld + r * (r + 1) / 2;
            for (i64 s = 0; s <= r; ++s) to[s] = m[s];
        }
}

extern "C" void itercc_scatter_pairs(const double *blocks, int64_t items, int64_t first, int64_t n,
                                     const int64_t *labels, double *out, int threads)
{
    // The blocks (rs|pq) (n x n each, of the pairs first .. first + items - 1,
    // r >= s) into out (n^4, chemists' (pq|rs) in C order) at [r][s] and
    // [s][r], from their lower triangles (p >= q, mirrored: (rs|pq) = (rs|qp)),
    // row by row with contiguous writes. With labels (may be NULL) the entries
    // whose Abelian irreps do not multiply to the totally symmetric one are
    // exact zeros.
    const i64 n2 = n * n;
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for (i64 b = 0; b < items; ++b)
        for (i64 p = 0; p < n; ++p) {
            i64 r = 0, s = 0;
            pair_of(first + b, r, s);
            const i64 g = labels ? (labels[r] ^ labels[s]) : 0;
            const double *m = blocks + b * n2;
            double *row = out + (r * n + s) * n2 + p * n;
            for (i64 q = 0; q < n; ++q) row[q] = q <= p ? m[p * n + q] : m[q * n + p];
            if (labels)
                for (i64 q = 0; q < n; ++q)
                    if ((labels[p] ^ labels[q]) != g) row[q] = 0.0;
            if (r != s) std::copy(row, row + n, out + (s * n + r) * n2 + p * n);
        }
}

extern "C" void itercc_jk_s8(const double *s8, int64_t nao, const double *dm, double *vj, double *vk,
                             int threads)
{
    // vj(mu,nu) = sum (mu nu|la si) dm(la,si), vk(mu,nu) = sum (mu la|nu si)
    // dm(la,si) for a symmetric dm (nao x nao) from the 8-fold packed (row by
    // row lower triangle of the pair matrix) real AO integrals: one pass over
    // the unique quartets, each scaled by its degeneracy and added in its
    // eight placements, then symmetrized. The quartets fall into a fixed set
    // of chunks with their own accumulators, summed in chunk order: the
    // result does not depend on the thread count. vj or vk may be NULL.
    const i64 n = nao, npair = n * (n + 1) / 2, n2 = n * n;
    const i64 total = npair * (npair + 1) / 2;
    const i64 chunks = std::max<i64>(1, std::min<i64>(64, npair));
    std::vector<double> acc(static_cast<size_t>(chunks * 2 * n2), 0.0);
    std::vector<i64> mu_of(static_cast<size_t>(npair));
    std::vector<i64> nu_of(static_cast<size_t>(npair));
    for (i64 p = 0; p < n; ++p)
        for (i64 q = 0; q <= p; ++q) {
            mu_of[size_t(p * (p + 1) / 2 + q)] = p;
            nu_of[size_t(p * (p + 1) / 2 + q)] = q;
        }
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (i64 c = 0; c < chunks; ++c) {
        double *j = acc.data() + c * 2 * n2, *k = j + n2;
        // Rows of the pair matrix in chunk c: about equal numbers of quartets.
        const i64 first = i64(std::sqrt(double(2 * total) * double(c) / double(chunks)));
        const i64 last = c + 1 == chunks ? npair : i64(std::sqrt(double(2 * total) * double(c + 1) / double(chunks)));
        for (i64 ij = std::min(first, npair); ij < std::min(last, npair); ++ij) {
            const i64 mu = mu_of[size_t(ij)], nu = nu_of[size_t(ij)];
            const double *row = s8 + ij * (ij + 1) / 2;
            const double dmn = dm[mu * n + nu] + dm[nu * n + mu];
            double jmn = 0.0;
            for (i64 kl = 0; kl <= ij; ++kl) {
                const i64 la = mu_of[size_t(kl)], si = nu_of[size_t(kl)];
                double v = row[kl];
                if (mu == nu) v *= 0.5;
                if (la == si) v *= 0.5;
                if (kl == ij) v *= 0.5;
                const double dls = dm[la * n + si] + dm[si * n + la];
                jmn += v * dls;
                j[la * n + si] += v * dmn;
                k[mu * n + la] += v * dm[nu * n + si];
                k[mu * n + si] += v * dm[nu * n + la];
                k[nu * n + la] += v * dm[mu * n + si];
                k[nu * n + si] += v * dm[mu * n + la];
            }
            j[mu * n + nu] += jmn;
        }
    }
#pragma omp parallel for schedule(static) num_threads(threads)
    for (i64 e = 0; e < n2; ++e) {
        const i64 p = e / n, q = e % n;
        double sj = 0.0, sk = 0.0;
        for (i64 c = 0; c < chunks; ++c) {
            const double *j = acc.data() + c * 2 * n2, *k = j + n2;
            sj += j[p * n + q] + j[q * n + p];
            sk += k[p * n + q] + k[q * n + p];
        }
        if (vj) vj[e] = sj;
        if (vk) vk[e] = sk;
    }
}
