// Indirect GEMM (see gemm_kernel.h).
#include "gemm_kernel.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "scratch.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#define ITERCC_GEMM_AVX512 1
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define ITERCC_GEMM_AVX2 1
#endif

#ifdef ITERCC_PROFILE_SYM
// Diagnostic knockouts of the fused driver (profile builds only; results are
// wrong while any is set): 1 skips the C updates, 2 the B packing, 4 the A
// packing, 8 the micro kernel, 16 the C prefetches. The initial mask comes
// from ITERCC_GEMM_KNOCKOUT.
namespace itercc {
std::atomic<int> gemm_knockout{[] {
    const char *text = std::getenv("ITERCC_GEMM_KNOCKOUT");
    return text != nullptr ? std::atoi(text) : 0;
}()};
}
extern "C" void itercc_gemm_knockout(int mask)
{
    itercc::gemm_knockout.store(mask);
}
#define ITERCC_KO(bit) (itercc::gemm_knockout.load(std::memory_order_relaxed) & (bit))
#else
#define ITERCC_KO(bit) 0
#endif

namespace itercc {
namespace {

using i64 = int64_t;

// Register tile MR x NR; MC rows of A and NC columns of B per packed block
// (multiples of MR and NR). Only kGemmKc affects the arithmetic.
#if defined(ITERCC_GEMM_AVX512)
constexpr i64 MR = 8, NR = 16;
#elif defined(ITERCC_GEMM_AVX2)
constexpr i64 MR = 6, NR = 8;
#else
constexpr i64 MR = 4, NR = 4;
#endif
constexpr i64 MC = 72, NC = 1024, KC = kGemmKc;
static_assert(MC % MR == 0 && NC % NR == 0, "blocking must hold whole register tiles");

// acc[i][j] = sum_k a[k][i] b[k][j] over packed micro-panels, by fused
// multiply-adds in order of k from zero.
void micro(i64 kc, const double *a, const double *b, double *acc)
{
#if defined(ITERCC_GEMM_AVX512)
    __m512d c[MR][2];
#pragma GCC unroll 8
    for (int i = 0; i < MR; ++i) c[i][0] = c[i][1] = _mm512_setzero_pd();
    for (i64 k = 0; k < kc; ++k) {
        const __m512d b0 = _mm512_loadu_pd(b), b1 = _mm512_loadu_pd(b + 8);
#pragma GCC unroll 8
        for (int i = 0; i < MR; ++i) {
            const __m512d ai = _mm512_set1_pd(a[i]);
            c[i][0] = _mm512_fmadd_pd(ai, b0, c[i][0]);
            c[i][1] = _mm512_fmadd_pd(ai, b1, c[i][1]);
        }
        a += MR;
        b += NR;
    }
#pragma GCC unroll 8
    for (int i = 0; i < MR; ++i) {
        _mm512_storeu_pd(acc + i * NR, c[i][0]);
        _mm512_storeu_pd(acc + i * NR + 8, c[i][1]);
    }
#elif defined(ITERCC_GEMM_AVX2)
    __m256d c[MR][2];
#pragma GCC unroll 6
    for (int i = 0; i < MR; ++i) c[i][0] = c[i][1] = _mm256_setzero_pd();
    for (i64 k = 0; k < kc; ++k) {
        const __m256d b0 = _mm256_loadu_pd(b), b1 = _mm256_loadu_pd(b + 4);
#pragma GCC unroll 6
        for (int i = 0; i < MR; ++i) {
            const __m256d ai = _mm256_broadcast_sd(a + i);
            c[i][0] = _mm256_fmadd_pd(ai, b0, c[i][0]);
            c[i][1] = _mm256_fmadd_pd(ai, b1, c[i][1]);
        }
        a += MR;
        b += NR;
    }
#pragma GCC unroll 6
    for (int i = 0; i < MR; ++i) {
        _mm256_storeu_pd(acc + i * NR, c[i][0]);
        _mm256_storeu_pd(acc + i * NR + 4, c[i][1]);
    }
#else
    double c[MR][NR] = {};
    for (i64 k = 0; k < kc; ++k) {
        for (int i = 0; i < MR; ++i)
            for (int j = 0; j < NR; ++j) c[i][j] = std::fma(a[i], b[j], c[i][j]);
        a += MR;
        b += NR;
    }
    for (int i = 0; i < MR; ++i)
        for (int j = 0; j < NR; ++j) acc[i * NR + j] = c[i][j];
#endif
}

// Whether the n offsets at o are consecutive (a run of memory).
inline bool consecutive(const i64 *o, i64 n)
{
    for (i64 i = 1; i < n; ++i)
        if (o[i] != o[0] + i) return false;
    return true;
}

// Groups of kRun offsets that are consecutive are copied as runs (vector
// loads) rather than gathered element by element; whether a group is a run
// does not depend on the sum, so it is found once per packing call.
constexpr i64 kRun = 8;

// mc rows (offsets ar) and kn sums (offsets ak) of an A operand as micro-panels
// [panel][k][MR] with kc sums per panel (rows past mc are zero), read in A's
// memory order; with Scale, each value times alpha. The pieces of a fused
// GEMM's chunk fill the panels' sums kn at a time (ap then points at the
// piece's first sum).
template <bool Scale, typename S = double>
void pack_a_piece(const S *a, const i64 *ar, const i64 *ak, bool rows_fast, double alpha, i64 mc,
                  i64 kn, i64 kc, double *ap)
{
    if (rows_fast) {
        // Sum by sum, each across all the panels: one pass along A's memory
        // (panel by panel, the kn sums would be kn interleaved streams, more
        // than the hardware prefetchers follow).
        thread_local std::vector<unsigned char> run;
        run.assign(size_t((mc + MR - 1) / MR), 0);
        for (i64 ip = 0, p = 0; ip < mc; ip += MR, ++p) run[size_t(p)] = mc - ip >= MR && consecutive(ar + ip, MR);
        for (i64 k = 0; k < kn; ++k) {
            const S *src = a + ak[k];
            for (i64 ip = 0, p = 0; ip < mc; ip += MR, ++p) {
                const i64 mr = std::min(MR, mc - ip);
                double *d = ap + ip * kc + k * MR;
                if (run[size_t(p)]) {
                    const S *x = src + ar[ip];
                    for (i64 i = 0; i < MR; ++i) d[i] = Scale ? alpha * double(x[i]) : double(x[i]);
                    continue;
                }
                for (i64 i = 0; i < mr; ++i)
                    d[i] = Scale ? alpha * double(src[ar[ip + i]]) : double(src[ar[ip + i]]);
                for (i64 i = mr; i < MR; ++i) d[i] = 0.0;
            }
        }
        return;
    }
    for (i64 ip = 0; ip < mc; ip += MR) {
        const i64 mr = std::min(MR, mc - ip);
        double *dst = ap + ip * kc;
        for (i64 i = 0; i < mr; ++i) {
            const S *src = a + ar[ip + i];
            for (i64 k = 0; k < kn; ++k)
                dst[k * MR + i] = Scale ? alpha * double(src[ak[k]]) : double(src[ak[k]]);
        }
        for (i64 i = mr; i < MR; ++i)
            for (i64 k = 0; k < kn; ++k) dst[k * MR + i] = 0.0;
    }
}

// kn sums (offsets bk) and nc columns (offsets bc) of a B operand as
// micro-panels [panel][k][NR] with kc sums per panel (columns past nc are
// zero), read in B's memory order.
template <bool Scale, typename S = double>
void pack_b_piece(const S *b, const i64 *bk, const i64 *bc, bool k_fast, double alpha, i64 kn, i64 nc,
                  i64 kc, double *bp)
{
    if (!k_fast) {
        // Sum by sum, each across all the panels: one pass along B's memory.
        constexpr i64 kGroup = std::min(kRun, NR);
        thread_local std::vector<unsigned char> run;
        run.assign(size_t((nc + kGroup - 1) / kGroup), 0);
        for (i64 j = 0; j < nc; j += kGroup)
            run[size_t(j / kGroup)] = nc - j >= kGroup && consecutive(bc + j, kGroup);
        for (i64 k = 0; k < kn; ++k) {
            const S *src = b + bk[k];
            for (i64 jp = 0; jp < nc; jp += NR) {
                const i64 nr = std::min(NR, nc - jp);
                double *d = bp + jp * kc + k * NR;
                for (i64 j0 = 0; j0 < nr; j0 += kGroup) {
                    double *e = d + j0;
                    if (run[size_t((jp + j0) / kGroup)]) {
                        const S *x = src + bc[jp + j0];
                        for (i64 j = 0; j < kGroup; ++j) e[j] = Scale ? alpha * double(x[j]) : double(x[j]);
                    } else {
                        const i64 n = std::min(kGroup, nr - j0);
                        for (i64 j = 0; j < n; ++j)
                            e[j] = Scale ? alpha * double(src[bc[jp + j0 + j]]) : double(src[bc[jp + j0 + j]]);
                    }
                }
                for (i64 j = nr; j < NR; ++j) d[j] = 0.0;
            }
        }
        return;
    }
    for (i64 jp = 0; jp < nc; jp += NR) {
        const i64 nr = std::min(NR, nc - jp);
        double *dst = bp + jp * kc;
        for (i64 j = 0; j < nr; ++j) {
            const S *src = b + bc[jp + j];
            for (i64 k = 0; k < kn; ++k)
                dst[k * NR + j] = Scale ? alpha * double(src[bk[k]]) : double(src[bk[k]]);
        }
        for (i64 j = nr; j < NR; ++j)
            for (i64 k = 0; k < kn; ++k) dst[k * NR + j] = 0.0;
    }
}

// Rows i0 .. i0+mc of A and sums p0 .. p0+kc as micro-panels.
void pack_a(const IndirectGemm &g, i64 i0, i64 mc, i64 p0, i64 kc, double *ap)
{
    pack_a_piece<false>(g.a, g.ar + i0, g.ak + p0, g.a_rows_fast, 1.0, mc, kc, kc, ap);
}

// Sums p0 .. p0+kc and columns j0 .. j0+nc of B as micro-panels.
void pack_b(const IndirectGemm &g, i64 p0, i64 kc, i64 j0, i64 nc, double *bp)
{
    pack_b_piece<false>(g.b, g.bk + p0, g.bc + j0, g.b_k_fast, 1.0, kc, nc, kc, bp);
}

// C += alpha * tile for mr rows (offsets cr) and nr columns (offsets cc); a
// full row of columns that are consecutive in C is updated as one run.
void update_tile(double *c, const i64 *cr, const i64 *cc, double alpha, i64 mr, i64 nr, const double *acc)
{
    bool run = nr == NR;
    for (i64 j = 1; run && j < NR; ++j) run = cc[j] == cc[0] + j;
    for (i64 i = 0; i < mr; ++i) {
        double *row = c + cr[i];
        const double *x = acc + i * NR;
        if (run) {
            double *o = row + cc[0];
            for (i64 j = 0; j < NR; ++j) o[j] = o[j] + alpha * x[j];
        } else {
            for (i64 j = 0; j < nr; ++j) row[cc[j]] = row[cc[j]] + alpha * x[j];
        }
    }
}

// C += alpha * tile for rows r0 .. r0+mr and columns c0 .. c0+nr.
void update(const IndirectGemm &g, i64 r0, i64 mr, i64 c0, i64 nr, const double *acc)
{
    update_tile(g.c, g.cr + r0, g.cc + c0, g.alpha, mr, nr, acc);
}

// Requests the C lines of a register tile (mr rows at offsets cr, nr columns
// at offsets cc) into L2 ahead of its update. C is a dense tile or a row
// buffer far larger than the caches, its rows are far apart, and an update
// that waits on them stalls the core for memory latency (the micro kernel of
// a shallow product is shorter than one miss).
void prefetch_tile(const double *c, const i64 *cr, const i64 *cc, i64 mr, i64 nr)
{
    const bool run = cc[nr - 1] == cc[0] + nr - 1;
    for (i64 i = 0; i < mr; ++i) {
        const double *row = c + cr[i];
        if (run) {
            for (i64 j = 0; j < nr; j += 8) __builtin_prefetch(row + cc[0] + j, 1, 2);
            __builtin_prefetch(row + cc[nr - 1], 1, 2);
        } else {
            for (i64 j = 0; j < nr; ++j) __builtin_prefetch(row + cc[j], 1, 2);
        }
    }
}

// The register tiles of an mc x nc block, column panel by column panel, and
// how far ahead their C lines are requested: about one micro kernel of 256
// sums ahead.
struct TileWalk {
    i64 mc, nc, rows, count, ahead;
    bool on;
    TileWalk(i64 mc_, i64 nc_, i64 kc, bool on_ = true)
        : mc(mc_), nc(nc_), rows((mc_ + MR - 1) / MR), count(rows * ((nc_ + NR - 1) / NR)),
          ahead(std::min<i64>(4, std::max<i64>(1, 256 / std::max<i64>(1, kc)))), on(on_)
    {
    }
    void prefetch(i64 t, const double *c, const i64 *cm, const i64 *cn) const
    {
        if (!on || t >= count || ITERCC_KO(16)) return;
        const i64 ir = (t % rows) * MR, jr = (t / rows) * NR;
        prefetch_tile(c, cm + ir, cn + jr, std::min(MR, mc - ir), std::min(NR, nc - jr));
    }
};

// Whether C needs prefetching in sum chunk pc of an m x nc column panel:
// always in the first chunk; in later ones only when the panel (all its rows)
// is too large to have stayed in L2 since the previous chunk. The prefetch
// instructions cost about 5% of the CPU time of a 47-virtual evaluation.
constexpr i64 kResidentC = i64(1) << 20;   // bytes
inline bool prefetch_c(i64 pc, i64 m, i64 nc) { return pc == 0 || m * nc * i64(sizeof(double)) > kResidentC; }

// Columns per packed block of B for a product of m rows and `depth` sums:
// the caller's hint (itercc_fused_contract: 256 for row-buffer outputs), else
// NC. With several chunks of KC sums, C is read and written once per chunk;
// the m x nc block of C stays in L2 between chunks only if it is small.
// ITERCC_GEMM_NC=auto (an experiment) narrows the blocks of deep products with
// many rows to hold about 1 MB of C, at the cost of packing A once more per
// block; a number fixes the width for every product. Only kGemmKc affects the
// arithmetic, so results do not depend on this.
i64 column_block(i64 m, i64 depth, i64 hint = 0)
{
    static const i64 forced = [] {
        const char *text = std::getenv("ITERCC_GEMM_NC");
        if (text == nullptr) return i64(0);
        return std::string(text) == "auto" ? i64(-1) : i64(std::atoll(text));
    }();
    if (forced > 0) return std::max(NR, forced / NR * NR);
    if (hint > 0) return std::max(NR, hint / NR * NR);
    if (forced == 0 || depth <= KC) return NC;
    constexpr i64 kBudget = i64(1) << 17;   // doubles of C held across chunks (1 MB)
    const i64 nc = std::min(NC, std::max<i64>(128, kBudget / std::max<i64>(1, m)));
    return std::max(NR, nc / NR * NR);
}

}  // namespace

void indirect_gemm(const IndirectGemm &g)
{
    if (g.m <= 0 || g.n <= 0 || g.k <= 0 || g.alpha == 0.0) return;
    const i64 kc_max = std::min(KC, g.k);
    const i64 ncb = column_block(g.m, g.k);
    const i64 nc_max = std::min(ncb, (g.n + NR - 1) / NR * NR);
    const i64 mc_max = std::min(MC, (g.m + MR - 1) / MR * MR);
    double *bp = scratch(2).get(size_t(nc_max * kc_max));
    double *ap = scratch(3).get(size_t(mc_max * kc_max));
    alignas(64) double acc[MR * NR] = {};
    for (i64 jc = 0; jc < g.n; jc += ncb) {
        const i64 nc = std::min(ncb, g.n - jc);
        for (i64 pc = 0; pc < g.k; pc += KC) {
            const i64 kc = std::min(KC, g.k - pc);
            pack_b(g, pc, kc, jc, nc, bp);
            for (i64 ic = 0; ic < g.m; ic += MC) {
                const i64 mc = std::min(MC, g.m - ic);
                const TileWalk walk(mc, nc, kc, prefetch_c(pc, g.m, nc));
                for (i64 t = 0; t < walk.ahead; ++t) walk.prefetch(t, g.c, g.cr + ic, g.cc + jc);
                pack_a(g, ic, mc, pc, kc, ap);
                i64 t = walk.ahead;
                for (i64 jr = 0; jr < nc; jr += NR) {
                    const i64 nr = std::min(NR, nc - jr);
                    for (i64 ir = 0; ir < mc; ir += MR) {
                        walk.prefetch(t++, g.c, g.cr + ic, g.cc + jc);
                        micro(kc, ap + ir * kc, bp + jr * kc, acc);
                        update(g, ic + ir, std::min(MR, mc - ir), jc + jr, nr, acc);
                    }
                }
            }
        }
    }
}

namespace {

// One fused GEMM operand side as the driver packs it: rows (offsets r) and
// sums (offsets k) of segment operand `v`, rows closer in memory when
// rows_fast; scaled by alpha when Scale.
struct FusedSide {
    const double *v;
    const i64 *r, *k;
    bool rows_fast;
    double alpha;
    bool f32;           // v holds float values
};

// pack_a_piece for a side, by its scaling and element type.
void pack_a_side(const FusedSide &x, const i64 *rows, const i64 *ak, i64 mc, i64 kn, i64 kc, double *dst)
{
    const auto *f = reinterpret_cast<const float *>(x.v);
    if (x.f32) {
        if (x.alpha == 1.0) pack_a_piece<false>(f, rows, ak, x.rows_fast, 1.0, mc, kn, kc, dst);
        else pack_a_piece<true>(f, rows, ak, x.rows_fast, x.alpha, mc, kn, kc, dst);
    } else if (x.alpha == 1.0) {
        pack_a_piece<false>(x.v, rows, ak, x.rows_fast, 1.0, mc, kn, kc, dst);
    } else {
        pack_a_piece<true>(x.v, rows, ak, x.rows_fast, x.alpha, mc, kn, kc, dst);
    }
}

// pack_b_piece for a side ([k][column]: sums faster is the opposite of rows_fast).
void pack_b_side(const FusedSide &x, const i64 *bk, const i64 *cols, i64 kn, i64 nc, i64 kc, double *dst)
{
    const auto *f = reinterpret_cast<const float *>(x.v);
    if (x.f32) {
        if (x.alpha == 1.0) pack_b_piece<false>(f, bk, cols, !x.rows_fast, 1.0, kn, nc, kc, dst);
        else pack_b_piece<true>(f, bk, cols, !x.rows_fast, x.alpha, kn, nc, kc, dst);
    } else if (x.alpha == 1.0) {
        pack_b_piece<false>(x.v, bk, cols, !x.rows_fast, 1.0, kn, nc, kc, dst);
    } else {
        pack_b_piece<true>(x.v, bk, cols, !x.rows_fast, x.alpha, kn, nc, kc, dst);
    }
}

// Rows of a fused GEMM that skinny_driver takes: two vectors of eight.
constexpr i64 kSkinnyRows = 16;

// ITERCC_GEMM_SKINNY: "0" neither skinny path, "fused" or "packed" only
// that one, else both (read once).
int skinny_mode()
{
    static const int mode = [] {
        const char *text = std::getenv("ITERCC_GEMM_SKINNY");
        if (text == nullptr || text[0] == ' ') return 3;
        const std::string value(text);
        return value == "0" ? 0 : value == "fused" ? 1 : value == "packed" ? 2 : 3;
    }();
    return mode;
}

bool skinny_enabled()
{
    return (skinny_mode() & 1) != 0;
}

#if defined(ITERCC_GEMM_AVX512) || defined(ITERCC_GEMM_AVX2)
#if defined(ITERCC_GEMM_AVX512)
using SkinnyVec = __m512d;
constexpr int kLanes = 8;
inline SkinnyVec vzero() { return _mm512_setzero_pd(); }
inline SkinnyVec vload(const double *p) { return _mm512_loadu_pd(p); }
inline SkinnyVec vset1(double x) { return _mm512_set1_pd(x); }
inline SkinnyVec vfma(SkinnyVec a, SkinnyVec b, SkinnyVec c) { return _mm512_fmadd_pd(a, b, c); }
inline void vstore(double *p, SkinnyVec x) { _mm512_store_pd(p, x); }
#else
using SkinnyVec = __m256d;
constexpr int kLanes = 4;
inline SkinnyVec vzero() { return _mm256_setzero_pd(); }
inline SkinnyVec vload(const double *p) { return _mm256_loadu_pd(p); }
inline SkinnyVec vset1(double x) { return _mm256_set1_pd(x); }
inline SkinnyVec vfma(SkinnyVec a, SkinnyVec b, SkinnyVec c) { return _mm256_fmadd_pd(a, b, c); }
inline void vstore(double *p, SkinnyVec x) { _mm256_store_pd(p, x); }
#endif

// One chunk of sums [pc, pc + kc) of skinny_driver with V vectors of rows.
template <int V>
void skinny_chunk(i64 m, i64 n, const i64 *cm, const i64 *cn, double *c, i64 nseg, const FusedSide *bs,
                  const i64 *start, i64 pc, i64 kc, const double *ap, bool rows_run)
{
    constexpr i64 kAhead = 8;   // columns between a column's prefetch and its use
    for (i64 j = 0; j < n; ++j) {
        if (j + kAhead < n) {
            for (i64 s = 0; s < nseg; ++s) {
                const i64 lo = std::max(pc, start[s]);
                if (lo < std::min(pc + kc, start[s + 1]))
                    __builtin_prefetch(bs[s].v + bs[s].r[j + kAhead] + bs[s].k[lo - start[s]], 0, 3);
            }
            __builtin_prefetch(c + cm[0] + cn[j + kAhead], 1, 3);
        }
        SkinnyVec acc[V];
        for (int v = 0; v < V; ++v) acc[v] = vzero();
        for (i64 s = 0; s < nseg; ++s) {
            const i64 lo = std::max(pc, start[s]), hi = std::min(pc + kc, start[s + 1]);
            const FusedSide &y = bs[s];
            const double *col = y.v + y.r[j];
            const i64 *bk = y.k - start[s];
            const double *a = ap + (lo - pc) * kSkinnyRows;
            if (y.alpha != 1.0) {
                for (i64 k = lo; k < hi; ++k, a += kSkinnyRows) {
                    const SkinnyVec b = vset1(y.alpha * col[bk[k]]);
                    for (int v = 0; v < V; ++v) acc[v] = vfma(vload(a + v * kLanes), b, acc[v]);
                }
            } else {
                for (i64 k = lo; k < hi; ++k, a += kSkinnyRows) {
                    const SkinnyVec b = vset1(col[bk[k]]);
                    for (int v = 0; v < V; ++v) acc[v] = vfma(vload(a + v * kLanes), b, acc[v]);
                }
            }
        }
        alignas(64) double out[V * kLanes];
        for (int v = 0; v < V; ++v) vstore(out + v * kLanes, acc[v]);
        if (rows_run) {
            double *o = c + cm[0] + cn[j];
            for (i64 i = 0; i < m; ++i) o[i] = o[i] + out[i];
        } else {
            for (i64 i = 0; i < m; ++i) c[cm[i] + cn[j]] = c[cm[i] + cn[j]] + out[i];
        }
    }
}

// fused_driver for m <= kSkinnyRows rows of doubles. With so few rows each B
// value is used once (one row block), so packing B would only copy it: B is
// read in place, column by column (a column's sums run along memory in the
// tile products, a few dozen values), against A packed [k][kSkinnyRows]. The
// arithmetic is the micro kernel's: per chunk of KC sums, fused multiply-adds
// in order of k from zero over the values scaled as the packers scale them,
// then C = C + acc. The skinny tile products of the triplet slot program
// (2998 x 7 x 7 blocks, 7 x 965 x 30, ...) spent most of their time packing
// B.
void skinny_driver(i64 m, i64 n, const i64 *cm, const i64 *cn, double *c, i64 nseg, const FusedSide *as,
                   const FusedSide *bs, const i64 *start)
{
    const i64 depth = start[nseg];
    const i64 kc_max = std::min(KC, depth);
    double *ap = scratch(3).get(size_t(kc_max * kSkinnyRows));
    const bool rows_run = consecutive(cm, m);
    const i64 vectors = (m + kLanes - 1) / kLanes;
    for (i64 pc = 0; pc < depth; pc += KC) {
        const i64 kc = std::min(KC, depth - pc);
        for (i64 s = 0; s < nseg; ++s) {
            const i64 lo = std::max(pc, start[s]), hi = std::min(pc + kc, start[s + 1]);
            const FusedSide &x = as[s];
            for (i64 k = lo; k < hi; ++k) {
                const double *src = x.v + x.k[k - start[s]];
                double *d = ap + (k - pc) * kSkinnyRows;
                for (i64 i = 0; i < m; ++i) d[i] = x.alpha != 1.0 ? x.alpha * src[x.r[i]] : src[x.r[i]];
                for (i64 i = m; i < kSkinnyRows; ++i) d[i] = 0.0;
            }
        }
        switch (vectors) {
        case 1: skinny_chunk<1>(m, n, cm, cn, c, nseg, bs, start, pc, kc, ap, rows_run); break;
        case 2: skinny_chunk<2>(m, n, cm, cn, c, nseg, bs, start, pc, kc, ap, rows_run); break;
#if !defined(ITERCC_GEMM_AVX512)
        case 3: skinny_chunk<3>(m, n, cm, cn, c, nseg, bs, start, pc, kc, ap, rows_run); break;
        case 4: skinny_chunk<4>(m, n, cm, cn, c, nseg, bs, start, pc, kc, ap, rows_run); break;
#endif
        default: break;
        }
    }
}
#define ITERCC_GEMM_SKINNY 1
#endif

// C(r, c) += sum over the segments of A_s(r, k) B_s(k, c) for r < m, c < n,
// C(r, c) = c[cm[r] + cn[c]], A_s and B_s given as sides (B's "rows" are its
// columns), sums laid end to end in chunks of KC.
void fused_driver(i64 m, i64 n, const i64 *cm, const i64 *cn, double *c, i64 nseg, const FusedSide *as,
                  const FusedSide *bs, const i64 *start, i64 nc_hint)
{
#if defined(ITERCC_GEMM_SKINNY)
    if (m <= kSkinnyRows && skinny_enabled() && !ITERCC_KO(32)) {
        bool doubles = true;
        for (i64 s = 0; s < nseg; ++s) doubles = doubles && !as[s].f32 && !bs[s].f32;
        if (doubles) return skinny_driver(m, n, cm, cn, c, nseg, as, bs, start);
    }
#endif
    const i64 depth = start[nseg];
    const i64 kc_max = std::min(KC, depth);
    const i64 ncb = column_block(m, depth, nc_hint);
    const i64 nc_max = std::min(ncb, (n + NR - 1) / NR * NR);
    const i64 mc_max = std::min(MC, (m + MR - 1) / MR * MR);
    double *bp = scratch(2).get(size_t(nc_max * kc_max));
    double *ap = scratch(3).get(size_t(mc_max * kc_max));
    alignas(64) double acc[MR * NR] = {};
    for (i64 jc = 0; jc < n; jc += ncb) {
        const i64 nc = std::min(ncb, n - jc);
        for (i64 pc = 0; pc < depth; pc += KC) {
            const i64 kc = std::min(KC, depth - pc);
            // The chunk's sums, segment piece by segment piece.
            for (i64 s = 0; s < nseg && !ITERCC_KO(2); ++s) {
                const i64 lo = std::max(pc, start[s]), hi = std::min(pc + kc, start[s + 1]);
                if (lo >= hi) continue;
                const FusedSide &x = bs[s];
                const i64 *bk = x.k + (lo - start[s]);
                pack_b_side(x, bk, x.r + jc, hi - lo, nc, kc, bp + (lo - pc) * NR);
            }
            for (i64 ic = 0; ic < m; ic += MC) {
                const i64 mc = std::min(MC, m - ic);
                const TileWalk walk(mc, nc, kc, prefetch_c(pc, m, nc));
                for (i64 t = 0; t < walk.ahead; ++t) walk.prefetch(t, c, cm + ic, cn + jc);
                for (i64 s = 0; s < nseg && !ITERCC_KO(4); ++s) {
                    const i64 lo = std::max(pc, start[s]), hi = std::min(pc + kc, start[s + 1]);
                    if (lo >= hi) continue;
                    const FusedSide &x = as[s];
                    const i64 *ak = x.k + (lo - start[s]);
                    pack_a_side(x, x.r + ic, ak, mc, hi - lo, kc, ap + (lo - pc) * MR);
                }
                i64 t = walk.ahead;
                for (i64 jr = 0; jr < nc; jr += NR) {
                    const i64 nr = std::min(NR, nc - jr);
                    for (i64 ir = 0; ir < mc; ir += MR) {
                        walk.prefetch(t++, c, cm + ic, cn + jc);
                        if (!ITERCC_KO(8)) micro(kc, ap + ir * kc, bp + jr * kc, acc);
                        if (!ITERCC_KO(1))
                            update_tile(c, cm + ic + ir, cn + jc + jr, 1.0, std::min(MR, mc - ir), nr, acc);
                    }
                }
            }
        }
    }
}

std::atomic<int> orientation{0};

// Fraction of a register-tiled m x n product that is not padding.
double tiled(i64 m, i64 n)
{
    return double(m) / double((m + MR - 1) / MR * MR) * double(n) / double((n + NR - 1) / NR * NR);
}

}  // namespace

void fused_gemm(const FusedGemm &g)
{
    const i64 n = g.j1 - g.j0;
    if (g.m <= 0 || n <= 0 || g.nseg <= 0) return;
    thread_local std::vector<i64> start;
    thread_local std::vector<FusedSide> xs, ys;
    start.resize(size_t(g.nseg + 1));
    xs.resize(size_t(g.nseg));
    ys.resize(size_t(g.nseg));
    start[0] = 0;
    for (i64 s = 0; s < g.nseg; ++s) {
        const GemmSegment &x = g.seg[s];
        start[size_t(s + 1)] = start[size_t(s)] + std::max<i64>(0, x.k);
        // A's rows from row 0; B's columns ("rows" of its side) from column j0.
        xs[size_t(s)] = FusedSide{x.a, x.ar, x.ak, x.a_rows_fast, x.alpha, x.a_f32};
        ys[size_t(s)] = FusedSide{x.b, x.bc + g.j0, x.bk, !x.b_k_fast, 1.0, x.b_f32};
    }
    if (start[size_t(g.nseg)] == 0) return;
    // C or its transpose, whichever fills the register tiles better: the
    // products and their order are the same either way (alpha stays on A's
    // values and fused multiply-adds commute), so the results are identical.
    const int mode = orientation.load(std::memory_order_relaxed);
    // A side of at most kSkinnyRows goes to the rows (skinny_driver reads the
    // other in place); otherwise whichever fills the register tiles better.
#if defined(ITERCC_GEMM_SKINNY)
    const bool skinny = std::min(g.m, n) <= kSkinnyRows && skinny_enabled();
#else
    const bool skinny = false;
#endif
    if (mode == 2 || (mode == 0 && (skinny ? n < g.m : tiled(n, g.m) > tiled(g.m, n))))
        fused_driver(n, g.m, g.cc + g.j0, g.cr, g.c, g.nseg, ys.data(), xs.data(), start.data(), g.nc);
    else
        fused_driver(g.m, n, g.cr, g.cc + g.j0, g.c, g.nseg, xs.data(), ys.data(), start.data(), g.nc);
}

bool skinny_gemm()
{
    return (skinny_mode() & 2) != 0;
}

void set_fused_orientation(int mode)
{
    orientation.store(mode, std::memory_order_relaxed);
}

const char *indirect_gemm_kernel()
{
#if defined(ITERCC_GEMM_AVX512)
    return "avx512";
#elif defined(ITERCC_GEMM_AVX2)
    return "avx2";
#else
    return "generic";
#endif
}

}  // namespace itercc
