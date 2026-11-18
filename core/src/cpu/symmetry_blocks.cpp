// Symmetry-blocked contractions on the CPU: the executor behind
// itercc/_cpu_symmetry_blocks.py (the plans come from itercc/symmetry_blocks.py).
//
// Block g of a plan (row g of `table`: rows, cols, depth, r0, c0, k0, a_base,
// b_base, o_base) computes
//
//   out[o_base + rpos[r0+r] + cpos[c0+c]] += alpha * sum_k a[a_base + a_rows[r0+r] + a_sum[k0+k]]
//                                                      * b[b_base + b_sum[k0+k] + b_cols[c0+c]]
//
// for r < rows, c < cols, k < depth (offsets in elements). Blocks write
// disjoint entries of `out`: they differ in the occupied combination or in the
// row irrep. Each block takes one route, chosen from its sizes:
//
// * small output, deep sum (T4 into R2): the sum is split into a fixed number
//   of slices whose products run in parallel and are added in order;
// * skinny (at most kSkinny rows or columns against a long other side): the
//   long operand is read in place, one L1 line of each sum index at a time,
//   so it is read once; the long side is cut into chunks;
// * shallow sums (at most kShallow): each output line is summed over k in L1
//   and added straight into `out`;
// * tiny blocks: gathered, multiplied with plain loops and added back;
// * the rest: the indirect GEMM of gemm_kernel.h, which packs the operands
//   straight from their offsets, multiplies register tiles with fused
//   multiply-adds and adds each tile straight into `out`.
//
// The arithmetic of every output entry is fixed by its route (each entry is
// formed by one thread, its sum over k in a fixed order), so blocks can be cut
// into tasks freely: tasks follow the thread count, results do not. No BLAS is
// called, so there is no second thread pool to contend with.
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "contract.h"
#include "gemm_kernel.h"
#include "itercc/core.h"
#include "scratch.h"

using i64 = int64_t;
using itercc::scratch;

#ifdef ITERCC_PROFILE_SYM
// Diagnostic timers (build with -DITERCC_PROFILE_SYM; read with itercc_sym_profile).
#include <atomic>
#include <cstdio>
#include <omp.h>
namespace {
const char *const kProfileNames[] = {"loops (thread s)",   "gemm (thread s)", "shallow (thread s)",
                                     "skinny wall",        "split wall",      "scale wall",
                                     "parallel wall"};
constexpr int kProfileCount = 7;
std::atomic<double> profile_time[kProfileCount];
void profile_add(int i, double v)
{
    double old = profile_time[i].load();
    while (!profile_time[i].compare_exchange_weak(old, old + v)) {
    }
}
}  // namespace
#define PROFILE_MARK double profile_mark = omp_get_wtime()
#define PROFILE_ADD(i) (profile_add((i), omp_get_wtime() - profile_mark), profile_mark = omp_get_wtime())
extern "C" void itercc_sym_profile(int reset)
{
    std::fprintf(stderr, "  indirect gemm kernel: %s\n", itercc::indirect_gemm_kernel());
    for (int i = 0; i < kProfileCount; ++i) {
        std::fprintf(stderr, "  %-28s %9.3f s\n", kProfileNames[i], profile_time[i].load());
        if (reset) profile_time[i] = 0.0;
    }
}
#else
#define PROFILE_MARK (void)0
#define PROFILE_ADD(i) (void)0
#endif

namespace itercc {

Scratch &scratch(int which)
{
    thread_local Scratch slots[kScratchSlots];
    return slots[which];
}

}  // namespace itercc

namespace {

constexpr i64 kNaive = i64(1) << 13;       // multiply-adds below which plain loops beat packing
constexpr i64 kSplitOutput = i64(1) << 12; // split the sum when rows * cols is at most this ...
constexpr i64 kSplitDepth = i64(1) << 11;  // ... and the depth at least this
constexpr i64 kSliceDepth = i64(1) << 9;   // depth per slice
constexpr i64 kMaxSlices = 64;
constexpr i64 kSkinny = 16;                // short side of a skinny block ...
constexpr i64 kSkinnyLong = 1024;          // ... against at least this long a side
constexpr i64 kChunk = 512;                // long-side entries per skinny task
constexpr i64 kShallow = 16;               // sums at most this deep (see shallow_block)
constexpr i64 kTasksPerThread = 4;         // tasks per thread for the dynamic schedule
constexpr i64 kMinTask = 16;               // rows or columns per task, at least
// A contraction below kParallelWork multiply-adds runs on one thread, and no
// task is cut below kMinShare: EOM and response actions make thousands of
// small contractions, which a 56-thread team made several times slower (each
// task repacks its operands). Sweep on 56 threads (water EOM-CCSDT aug-cc-pVDZ,
// CCSD polarizability cc-pVTZ): no limits 5.8-7.0 and 4.3-4.8 s; 64k and 8k
// 4.7-5.0 and 2.6-2.7 s; 128k and 32k 6.0-6.6 and 4.6-4.7 s. One thread or
// the whole team, never a smaller team: libgomp would end the surplus threads
// and start them again.
constexpr i64 kParallelWork = i64(1) << 16;
constexpr i64 kMinShare = i64(1) << 13;

// ITERCC_SYM_PARALLEL_WORK and ITERCC_SYM_MIN_SHARE override the two limits
// (multiply-adds; read once), for tuning on other machines.
i64 tuned(const char *name, i64 fallback)
{
    const char *value = std::getenv(name);
    return value != nullptr && value[0] != '\0' ? std::max<i64>(1, std::atoll(value)) : fallback;
}

struct Block {
    i64 rows, cols, depth, r0, c0, k0, a_base, b_base, o_base;
    i64 work() const { return rows * cols * depth; }
};

using Plan = itercc::ContractPlan;

Block block_at(const i64 *table, i64 g)
{
    const i64 *t = table + 9 * g;
    return Block{t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]};
}

i64 slices(const Block &g)
{
    if (g.rows * g.cols > kSplitOutput || g.depth < kSplitDepth) return 1;
    return std::min(kMaxSlices, g.depth / kSliceDepth);
}

bool skinny(const Block &g)
{
    const i64 short_side = std::min(g.rows, g.cols), long_side = std::max(g.rows, g.cols);
    return short_side <= kSkinny && long_side >= kSkinnyLong;
}

enum class Route { loops, shallow, gemm };

// B entries base + ks[k] + cs[c] (k < n, c < m) of packed rows into
// dst[k * ldk + c * ldc]. Addresses split into (row, entry) once per sum and
// once per column, never per entry: each offset is a sum of letter strides,
// those of occupied letters multiples of the row width, so the two parts add
// without carry. The values equal those of itercc_signed_gather_rows.
void packed_gather(const itercc::PackedRows &v, i64 base, const i64 *ks, i64 n, const i64 *cs, i64 m,
                   i64 ldk, i64 ldc, double *dst)
{
    thread_local std::vector<i64> col_row, col_entry;
    col_row.resize(size_t(m));
    col_entry.resize(size_t(m));
    bool one_row = true;
    for (i64 c = 0; c < m; ++c) {
        col_row[size_t(c)] = cs[c] / v.width;
        col_entry[size_t(c)] = cs[c] - col_row[size_t(c)] * v.width;
        one_row = one_row && col_row[size_t(c)] == 0;
    }
    const i64 *entries = col_entry.data();
    auto value = [&](i64 row, i64 entry) {
        const i64 b = v.bases[row];
        if (!b) return 0.;
        const int32_t index = reinterpret_cast<const int32_t *>(static_cast<intptr_t>(v.maps[row]))[entry];
        const double *from = v.source + ((b < 0 ? -b : b) - 1);
        const double sign = b < 0 ? -1. : 1.;
        return index > 0 ? sign * from[index - 1] : index < 0 ? -sign * from[-int64_t(index) - 1] : 0.;
    };
    for (i64 k = 0; k < n; ++k) {
        const i64 address = base + ks[k], row = address / v.width, entry = address - row * v.width;
        double *d = dst + k * ldk;
        if (!one_row) {
            for (i64 c = 0; c < m; ++c) d[c * ldc] = value(row + col_row[size_t(c)], entry + entries[c]);
            continue;
        }
        const i64 b = v.bases[row];
        if (!b) {
            for (i64 c = 0; c < m; ++c) d[c * ldc] = 0.;
            continue;
        }
        const int32_t *map = reinterpret_cast<const int32_t *>(static_cast<intptr_t>(v.maps[row])) + entry;
        const double *from = v.source + ((b < 0 ? -b : b) - 1);
        const double sign = b < 0 ? -1. : 1.;
        for (i64 c = 0; c < m; ++c) {
            const int32_t index = map[entries[c]];
            d[c * ldc] = index > 0 ? sign * from[index - 1] : index < 0 ? -sign * from[-int64_t(index) - 1] : 0.;
        }
    }
}

Route route(const Block &g)
{
    if (g.work() < kNaive) return Route::loops;
    return g.depth <= kShallow ? Route::shallow : Route::gemm;
}

// The block as an indirect GEMM. When out runs along rows, out^T = B^T A^T, so
// the register tiles' vector direction (columns) follows out's memory.
itercc::IndirectGemm gemm_view(const Plan &p, const Block &g)
{
    if (!p.r_fast)
        return {g.rows, g.cols, g.depth,
                p.a + g.a_base, p.a_rows + g.r0, p.a_sum + g.k0, p.a_k_major,
                p.b + g.b_base, p.b_sum + g.k0, p.b_cols + g.c0, p.b_c_major,
                p.out + g.o_base, p.rpos + g.r0, p.cpos + g.c0, p.alpha};
    return {g.cols, g.rows, g.depth,
            p.b + g.b_base, p.b_cols + g.c0, p.b_sum + g.k0, !p.b_c_major,
            p.a + g.a_base, p.a_sum + g.k0, p.a_rows + g.r0, !p.a_k_major,
            p.out + g.o_base, p.cpos + g.c0, p.rpos + g.r0, p.alpha};
}

// A restricted to sums [k1, k2): [r][k] (rows x n) or, with `k_major`, [k][r].
void gather_a(const Plan &p, const Block &g, i64 k1, i64 k2, bool k_major, double *dst)
{
    const i64 n = k2 - k1, rows = g.rows;
    const double *base = p.a + g.a_base;
    const i64 *rs = p.a_rows + g.r0, *ks = p.a_sum + g.k0 + k1;
    if (k_major) {
        for (i64 k = 0; k < n; ++k) {
            const double *src = base + ks[k];
            double *d = dst + k * rows;
            for (i64 r = 0; r < rows; ++r) d[r] = src[rs[r]];
        }
    } else {
        for (i64 r = 0; r < rows; ++r) {
            const double *src = base + rs[r];
            double *d = dst + r * n;
            for (i64 k = 0; k < n; ++k) d[k] = src[ks[k]];
        }
    }
}

// B restricted to sums [k1, k2): [k][c] (n x cols) or, with `c_major`, [c][k].
void gather_b(const Plan &p, const Block &g, i64 k1, i64 k2, bool c_major, double *dst)
{
    const i64 n = k2 - k1, cols = g.cols;
    if (p.packed_b != nullptr) {
        packed_gather(*p.packed_b, g.b_base, p.b_sum + g.k0 + k1, n, p.b_cols + g.c0, cols,
                      c_major ? 1 : cols, c_major ? n : 1, dst);
        return;
    }
    const double *base = p.b + g.b_base;
    const i64 *cs = p.b_cols + g.c0, *ks = p.b_sum + g.k0 + k1;
    if (c_major) {
        for (i64 c = 0; c < cols; ++c) {
            const double *src = base + cs[c];
            double *d = dst + c * n;
            for (i64 k = 0; k < n; ++k) d[k] = src[ks[k]];
        }
    } else {
        for (i64 k = 0; k < n; ++k) {
            const double *src = base + ks[k];
            double *d = dst + k * cols;
            for (i64 c = 0; c < cols; ++c) d[c] = src[cs[c]];
        }
    }
}

// A tiny block: gathered, x[r][c] = sum_k a[r][k] b[k][c] with plain loops in
// order of k, and added back.
void loops_block(const Plan &p, const Block &g)
{
    const i64 rows = g.rows, cols = g.cols, depth = g.depth;
    double *a = scratch(0).get(size_t(rows * depth + depth * cols + cols));
    double *b = a + rows * depth, *x = b + depth * cols;
    gather_a(p, g, 0, depth, false, a);
    gather_b(p, g, 0, depth, false, b);
    double *base = p.out + g.o_base;
    const i64 *cp = p.cpos + g.c0;
    for (i64 r = 0; r < rows; ++r) {
        std::fill(x, x + cols, 0.0);
        const double *ar = a + r * depth;
        for (i64 k = 0; k < depth; ++k) {
            const double value = ar[k];
            const double *bk = b + k * cols;
            for (i64 c = 0; c < cols; ++c) x[c] += value * bk[c];
        }
        double *o = base + p.rpos[g.r0 + r];
        for (i64 c = 0; c < cols; ++c) o[cp[c]] += p.alpha * x[c];
    }
}

// A block with a shallow sum (at most kShallow): packing would cost as much as
// the product, so each output line is summed over k in L1 and added straight
// into `out` (lines along the faster direction of `out`).
void shallow_block(const Plan &p, const Block &g)
{
    const i64 rows = g.rows, cols = g.cols, depth = g.depth;
    const bool rf = p.r_fast;
    const i64 nline = rf ? rows : cols, nother = rf ? cols : rows;
    double *a = scratch(0).get(size_t(rows * depth + depth * cols + nline));
    double *b = a + rows * depth, *line = b + depth * cols;
    // Lines along columns: a as [r][k], b as [k][c]; along rows: a as [k][r], b as [c][k].
    gather_a(p, g, 0, depth, rf, a);
    gather_b(p, g, 0, depth, rf, b);
    double *base = p.out + g.o_base;
    const i64 *line_pos = rf ? p.rpos + g.r0 : p.cpos + g.c0;
    const i64 *other_pos = rf ? p.cpos + g.c0 : p.rpos + g.r0;
    // Line i: sum_k coef[i][k] * vec[k][:], with coef = a (rows) or b^T (columns).
    const double *coef = rf ? b : a, *vec = rf ? a : b;
    for (i64 i = 0; i < nother; ++i) {
        std::fill(line, line + nline, 0.0);
        const double *ci = coef + i * depth;
        for (i64 k = 0; k < depth; ++k) {
            const double value = ci[k];
            const double *vk = vec + k * nline;
            for (i64 j = 0; j < nline; ++j) line[j] += value * vk[j];
        }
        double *o = base + other_pos[i];
        for (i64 j = 0; j < nline; ++j) o[line_pos[j]] += p.alpha * line[j];
    }
}

// The indirect GEMM of block g over sums [k1, k2) with B from packed rows: B
// is gathered in panels of columns (about 512 KB, in cache) and each panel is
// one indirect GEMM over all the sums, so every output entry gets the same sum
// in the same order as from a dense B. `swap` runs out^T = B^T A^T (as
// gemm_view does when out's rows are faster); c, cr, cc place the output.
constexpr i64 kPanel = i64(1) << 16;

void packed_indirect(const Plan &p, const Block &g, i64 k1, i64 k2, bool swap, double *c,
                     const i64 *cr, const i64 *cc, double alpha)
{
    const i64 depth = k2 - k1;
    if (depth <= 0 || g.rows <= 0 || g.cols <= 0) return;
    const i64 width = std::min(g.cols, std::max<i64>(64, kPanel / depth));
    double *panel = scratch(6).get(size_t(depth * width));
    thread_local std::vector<i64> ks, cs;
    ks.resize(size_t(depth));
    cs.resize(size_t(width));
    for (i64 j = 0; j < width; ++j) cs[size_t(j)] = j;
    for (i64 c1 = 0; c1 < g.cols; c1 += width) {
        const i64 m = std::min(width, g.cols - c1);
        packed_gather(*p.packed_b, g.b_base, p.b_sum + g.k0 + k1, depth, p.b_cols + g.c0 + c1, m, m, 1, panel);
        for (i64 k = 0; k < depth; ++k) ks[size_t(k)] = k * m;
        if (!swap)
            itercc::indirect_gemm({g.rows, m, depth, p.a + g.a_base, p.a_rows + g.r0, p.a_sum + g.k0 + k1,
                                   p.a_k_major, panel, ks.data(), cs.data(), false, c, cr, cc + c1, alpha});
        else
            itercc::indirect_gemm({m, g.rows, depth, panel, cs.data(), ks.data(), true, p.a + g.a_base,
                                   p.a_sum + g.k0 + k1, p.a_rows + g.r0, !p.a_k_major, c, cc + c1, cr, alpha});
    }
}

// The GEMM route of one block or task.
void gemm_block(const Plan &p, const Block &g)
{
    if (p.packed_b != nullptr)
        packed_indirect(p, g, 0, g.depth, p.r_fast, p.out + g.o_base, p.rpos + g.r0, p.cpos + g.c0, p.alpha);
    else
        itercc::indirect_gemm(gemm_view(p, g));
}

// A block with a small output and a deep sum: slices of the sum in parallel
// (indirect GEMMs into dense partial products), added in order into `out`.
void split_block(const Plan &p, const Block &g, i64 parts, int threads)
{
    const i64 size = g.rows * g.cols;
    double *partial = scratch(1).get(size_t(parts * size));
    std::fill(partial, partial + parts * size, 0.0);
    std::vector<i64> row_at(size_t(g.rows)), col_at(size_t(g.cols));
    for (i64 r = 0; r < g.rows; ++r) row_at[size_t(r)] = r * g.cols;
    for (i64 c = 0; c < g.cols; ++c) col_at[size_t(c)] = c;
#pragma omp parallel num_threads(threads)
    {
#pragma omp for schedule(dynamic, 1)
        for (i64 s = 0; s < parts; ++s) {
            const i64 k1 = g.depth * s / parts, k2 = g.depth * (s + 1) / parts;
            if (p.packed_b != nullptr) {
                packed_indirect(p, g, k1, k2, false, partial + s * size, row_at.data(), col_at.data(), 1.0);
                continue;
            }
            itercc::indirect_gemm({g.rows, g.cols, k2 - k1,
                                   p.a + g.a_base, p.a_rows + g.r0, p.a_sum + g.k0 + k1, p.a_k_major,
                                   p.b + g.b_base, p.b_sum + g.k0 + k1, p.b_cols + g.c0, p.b_c_major,
                                   partial + s * size, row_at.data(), col_at.data(), 1.0});
        }
        // Sum the slices in order and add the result to out.
#pragma omp for schedule(static)
        for (i64 e = 0; e < size; ++e) {
            double sum = partial[size_t(e)];
            for (i64 s = 1; s < parts; ++s) sum += partial[size_t(s * size + e)];
            const i64 r = e / g.cols, c = e % g.cols;
            p.out[g.o_base + p.rpos[g.r0 + r] + p.cpos[g.c0 + c]] += p.alpha * sum;
        }
    }
}

// A skinny block seen from its short side: `small` (short x depth, offsets
// small_short + small_sum) and `big` (depth x long, offsets big_sum + big_long);
// pos_short and pos_long place the output. Both orientations add
// small[s][k] * big[k][l] over k in order.
struct Skinny {
    const double *small, *big;
    const i64 *small_short, *small_sum, *big_sum, *big_long, *pos_short, *pos_long;
    i64 nshort, nlong, depth, o_base;
    const itercc::PackedRows *small_packed, *big_packed;   // B from packed rows (with b_base)
    i64 b_base;
};

Skinny skinny_view(const Plan &p, const Block &g)
{
    const double *b = p.packed_b != nullptr ? nullptr : p.b + g.b_base;
    if (g.rows <= g.cols)
        return Skinny{p.a + g.a_base, b, p.a_rows + g.r0, p.a_sum + g.k0, p.b_sum + g.k0,
                      p.b_cols + g.c0, p.rpos + g.r0, p.cpos + g.c0, g.rows, g.cols, g.depth, g.o_base,
                      nullptr, p.packed_b, g.b_base};
    return Skinny{b, p.a + g.a_base, p.b_cols + g.c0, p.b_sum + g.k0, p.a_sum + g.k0,
                  p.a_rows + g.r0, p.cpos + g.c0, p.rpos + g.r0, g.cols, g.rows, g.depth, g.o_base,
                  p.packed_b, nullptr, g.b_base};
}

// Entries of the `line` buffer a skinny chunk needs: one line, or with a
// packed long operand all its lines, gathered at once.
i64 skinny_line(const Skinny &s) { return s.big_packed != nullptr ? s.depth * kChunk : kChunk; }

// Long-side entries [l1, l2) of a skinny block, one sum index at a time over a
// line of long entries (the lines of later sums are then in cache, whatever
// the long operand's layout). Buffers: small (short x depth), line (kChunk),
// acc (short x kChunk).
void skinny_chunk(const Plan &p, const Skinny &s, i64 l1, i64 l2, double *small, double *line, double *acc)
{
    const i64 n = l2 - l1, ns = s.nshort, depth = s.depth;
    if (s.small_packed != nullptr)
        packed_gather(*s.small_packed, s.b_base, s.small_sum, depth, s.small_short, ns, 1, depth, small);
    else
        for (i64 i = 0; i < ns; ++i)
            for (i64 k = 0; k < depth; ++k) small[i * depth + k] = s.small[s.small_short[i] + s.small_sum[k]];
    std::fill(acc, acc + ns * n, 0.0);
    const i64 *longs = s.big_long + l1;
    if (s.big_packed != nullptr)
        packed_gather(*s.big_packed, s.b_base, s.big_sum, depth, longs, n, n, 1, line);
    for (i64 k = 0; k < depth; ++k) {
        const double *row = line + k * n;     // packed: every line gathered above
        if (s.big_packed == nullptr) {
            const double *src = s.big + s.big_sum[k];
            for (i64 l = 0; l < n; ++l) line[l] = src[longs[l]];
            row = line;
        }
        for (i64 i = 0; i < ns; ++i) {
            const double value = small[i * depth + k];
            double *ai = acc + i * n;
            for (i64 l = 0; l < n; ++l) ai[l] += value * row[l];
        }
    }
    double *base = p.out + s.o_base;
    const i64 *pl = s.pos_long + l1;
    for (i64 i = 0; i < ns; ++i) {
        double *o = base + s.pos_short[i];
        const double *ai = acc + i * n;
        for (i64 l = 0; l < n; ++l) o[pl[l]] += p.alpha * ai[l];
    }
}

}  // namespace

namespace itercc {

ContractPlan contract_plan(const double *a, const double *b, double *out, const int64_t *a_rows,
                           const int64_t *a_sum, const int64_t *b_sum, const int64_t *b_cols,
                           const int64_t *rpos, const int64_t *cpos, double alpha, int64_t flags)
{
    return ContractPlan{a, b, out, a_rows, a_sum, b_sum, b_cols, rpos, cpos, alpha,
                        (flags & 1) != 0, (flags & 2) != 0, (flags & 4) != 0};
}

void contract_serial(const ContractPlan &p, const int64_t *table, int64_t nblocks)
{
    if (p.alpha == 0.0) return;
    for (i64 g = 0; g < nblocks; ++g) {
        const Block block = block_at(table, g);
        const i64 parts = slices(block);
        if (parts > 1) {   // the same slices as in parallel, so the same sums
            split_block(p, block, parts, 1);
            continue;
        }
        if (skinny(block)) {
            const Skinny s = skinny_view(p, block);
            double *small = scratch(0).get(size_t(s.nshort * s.depth));
            double *line = scratch(2).get(size_t(skinny_line(s)));
            double *acc = scratch(1).get(size_t(s.nshort * kChunk));
            for (i64 l = 0; l < s.nlong; l += kChunk)
                skinny_chunk(p, s, l, std::min(s.nlong, l + kChunk), small, line, acc);
            continue;
        }
        switch (route(block)) {
        case Route::loops:
            loops_block(p, block);
            break;
        case Route::shallow:
            shallow_block(p, block);
            break;
        case Route::gemm:
            gemm_block(p, block);
            break;
        }
    }
}

void contract_tasks(const int64_t *table, int64_t nblocks, int64_t workers, std::vector<ContractTask> &tasks)
{
    static const i64 min_share = tuned("ITERCC_SYM_MIN_SHARE", kMinShare);
    auto task = [](i64 kind, i64 route_of, i64 first, const Block &g) {
        return ContractTask{kind, route_of, first, {g.rows, g.cols, g.depth, g.r0, g.c0, g.k0, g.a_base,
                                                    g.b_base, g.o_base}};
    };
    i64 total = 0;
    for (i64 g = 0; g < nblocks; ++g) {
        const Block block = block_at(table, g);
        const i64 parts = slices(block);
        if (parts > 1) {
            tasks.push_back(task(0, 0, parts, block));
        } else if (skinny(block)) {
            const i64 nlong = std::max(block.rows, block.cols);
            for (i64 l = 0; l < nlong; l += kChunk) tasks.push_back(task(1, 0, l, block));
        } else {
            total += block.work();
        }
    }
    // The grid pieces of itercc_sym_contract (run_contract below).
    const i64 share = std::max<i64>(min_share, total / (kTasksPerThread * std::max<i64>(1, workers)));
    for (i64 g = 0; g < nblocks; ++g) {
        const Block block = block_at(table, g);
        if (slices(block) > 1 || skinny(block)) continue;
        const Route kind = route(block);
        const i64 wanted = std::max<i64>(1, block.work() / share);
        const i64 max_r = std::max<i64>(1, block.rows / kMinTask), max_c = std::max<i64>(1, block.cols / kMinTask);
        i64 pr = 1, pc = 1;
        while (pr * pc < wanted && (pr < max_r || pc < max_c)) {
            const bool more_rows = pc >= max_c || (pr < max_r && block.rows * pc >= block.cols * pr);
            if (more_rows)
                ++pr;
            else
                ++pc;
        }
        for (i64 i = 0; i < pr; ++i)
            for (i64 j = 0; j < pc; ++j) {
                Block piece = block;
                piece.r0 += block.rows * i / pr;
                piece.rows = block.rows * (i + 1) / pr - block.rows * i / pr;
                piece.c0 += block.cols * j / pc;
                piece.cols = block.cols * (j + 1) / pc - block.cols * j / pc;
                tasks.push_back(task(2, i64(kind), 0, piece));
            }
    }
}

void contract_task(const ContractPlan &p, const ContractTask &t)
{
    if (p.alpha == 0.0) return;
    const Block g = block_at(t.block, 0);
    if (t.kind == 0) {
        split_block(p, g, t.first, 1);
    } else if (t.kind == 1) {
        const Skinny s = skinny_view(p, g);
        skinny_chunk(p, s, t.first, std::min(s.nlong, t.first + kChunk), scratch(0).get(size_t(s.nshort * s.depth)),
                     scratch(2).get(size_t(skinny_line(s))), scratch(1).get(size_t(s.nshort * kChunk)));
    } else {
        switch (Route(t.route)) {
        case Route::loops:
            loops_block(p, g);
            break;
        case Route::shallow:
            shallow_block(p, g);
            break;
        case Route::gemm:
            gemm_block(p, g);
            break;
        }
    }
}

}  // namespace itercc

extern "C" void itercc_scale(double *x, int64_t n, double beta, int threads)
{
    if (beta == 1.0) return;
    PROFILE_MARK;
    // Small arrays on the calling thread: a parallel region costs more than
    // clearing 256 KB (the executor clears every fresh output this way).
    constexpr i64 kSerial = i64(1) << 15;
    if (n <= kSerial) {
        if (beta == 0.0)
            std::fill(x, x + n, 0.0);
        else
            for (i64 i = 0; i < n; ++i) x[i] *= beta;
        PROFILE_ADD(5);
        return;
    }
    if (beta == 0.0) {
#pragma omp parallel for num_threads(threads) schedule(static)
        for (i64 i = 0; i < n; ++i) x[i] = 0.0;
    } else {
#pragma omp parallel for num_threads(threads) schedule(static)
        for (i64 i = 0; i < n; ++i) x[i] *= beta;
    }
    PROFILE_ADD(5);
}

namespace {
int run_contract(const Plan &p, int64_t nblocks, const int64_t *table, int threads);
}

extern "C" int itercc_sym_contract(int64_t nblocks, const int64_t *table, const double *a,
                                   const double *b, double *out, const int64_t *a_rows,
                                   const int64_t *a_sum, const int64_t *b_sum,
                                   const int64_t *b_cols, const int64_t *rpos,
                                   const int64_t *cpos, double alpha, int flags, int threads)
{
    if (nblocks <= 0 || alpha == 0.0) return 0;
    const Plan p{a, b, out, a_rows, a_sum, b_sum, b_cols, rpos, cpos, alpha,
                 (flags & 1) != 0, (flags & 2) != 0, (flags & 4) != 0};
    return run_contract(p, nblocks, table, threads);
}

extern "C" int itercc_sym_contract_packed(int64_t nblocks, const int64_t *table, const double *a,
                                          const double *source, const int64_t *bases,
                                          const int64_t *maps, int64_t width, double *out,
                                          const int64_t *a_rows, const int64_t *a_sum,
                                          const int64_t *b_sum, const int64_t *b_cols,
                                          const int64_t *rpos, const int64_t *cpos, double alpha,
                                          int flags, int threads)
{
    if (nblocks <= 0 || alpha == 0.0) return 0;
    const itercc::PackedRows rows{source, bases, maps, width};
    Plan p{a, nullptr, out, a_rows, a_sum, b_sum, b_cols, rpos, cpos, alpha,
           (flags & 1) != 0, (flags & 2) != 0, (flags & 4) != 0};
    p.packed_b = &rows;
    return run_contract(p, nblocks, table, threads);
}

namespace {
int run_contract(const Plan &p, int64_t nblocks, const int64_t *table, int threads)
{
    i64 all = 0;
    for (i64 g = 0; g < nblocks; ++g) all += block_at(table, g).work();
    static const i64 parallel_work = tuned("ITERCC_SYM_PARALLEL_WORK", kParallelWork);
    static const i64 min_share = tuned("ITERCC_SYM_MIN_SHARE", kMinShare);
    if (all < parallel_work) threads = 1;
    std::vector<i64> blocks;
    std::vector<std::pair<i64, i64>> chunks;   // skinny (block, first long entry)
    for (i64 g = 0; g < nblocks; ++g) {
        const Block block = block_at(table, g);
        const i64 parts = slices(block);
        if (parts > 1) {
            PROFILE_MARK;
            split_block(p, block, parts, threads);
            PROFILE_ADD(4);
        } else if (skinny(block)) {
            const i64 nlong = std::max(block.rows, block.cols);
            for (i64 l = 0; l < nlong; l += kChunk) chunks.emplace_back(g, l);
        } else {
            blocks.push_back(g);
        }
    }
    if (!chunks.empty()) {
        PROFILE_MARK;
        const i64 count = i64(chunks.size());
#pragma omp parallel num_threads(threads)
        {
#pragma omp for schedule(dynamic, 1)
            for (i64 t = 0; t < count; ++t) {
                const Block g = block_at(table, chunks[size_t(t)].first);
                const Skinny s = skinny_view(p, g);
                const i64 l1 = chunks[size_t(t)].second, l2 = std::min(s.nlong, l1 + kChunk);
                skinny_chunk(p, s, l1, l2, scratch(0).get(size_t(s.nshort * s.depth)),
                             scratch(2).get(size_t(skinny_line(s))),
                             scratch(1).get(size_t(s.nshort * kChunk)));
            }
        }
        PROFILE_ADD(3);
    }
    if (blocks.empty()) return 0;
    // Tasks: each block's route is fixed by its sizes; large blocks are cut into
    // a grid of pieces so that there are about kTasksPerThread tasks per thread.
    struct Task {
        Block block;
        Route route;
    };
    i64 total = 0;
    for (const i64 g : blocks) total += block_at(table, g).work();
    const i64 share = std::max<i64>(min_share, total / (kTasksPerThread * std::max(1, threads)));
    std::vector<Task> tasks;
    for (const i64 g : blocks) {
        const Block block = block_at(table, g);
        const Route kind = route(block);
        // pr x pc pieces, near-square in rows and columns: each piece packs its
        // rows of A and its columns of B, so a grid repacks far less than strips.
        const i64 wanted = std::max<i64>(1, block.work() / share);
        const i64 max_r = std::max<i64>(1, block.rows / kMinTask), max_c = std::max<i64>(1, block.cols / kMinTask);
        i64 pr = 1, pc = 1;
        while (pr * pc < wanted && (pr < max_r || pc < max_c)) {
            const bool more_rows = pc >= max_c || (pr < max_r && block.rows * pc >= block.cols * pr);
            if (more_rows)
                ++pr;
            else
                ++pc;
        }
        for (i64 i = 0; i < pr; ++i)
            for (i64 j = 0; j < pc; ++j) {
                Block piece = block;
                piece.r0 += block.rows * i / pr;
                piece.rows = block.rows * (i + 1) / pr - block.rows * i / pr;
                piece.c0 += block.cols * j / pc;
                piece.cols = block.cols * (j + 1) / pc - block.cols * j / pc;
                tasks.push_back(Task{piece, kind});
            }
    }
    // Largest first, so the dynamic schedule ends on small tasks.
    std::stable_sort(tasks.begin(), tasks.end(),
                     [](const Task &x, const Task &y) { return x.block.work() > y.block.work(); });
    const i64 count = i64(tasks.size());
    PROFILE_MARK;
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
    for (i64 i = 0; i < count; ++i) {
        const Task &task = tasks[size_t(i)];
        PROFILE_MARK;
        switch (task.route) {
        case Route::loops:
            loops_block(p, task.block);
            PROFILE_ADD(0);
            break;
        case Route::shallow:
            shallow_block(p, task.block);
            PROFILE_ADD(2);
            break;
        case Route::gemm:
            gemm_block(p, task.block);
            PROFILE_ADD(1);
            break;
        }
    }
    PROFILE_ADD(6);
    return 0;
}
}  // namespace

extern "C" void itercc_scratch_release(int threads)
{
    // Every thread of a team of this size frees its own buffers.
#pragma omp parallel num_threads(threads)
    for (int which = 0; which < itercc::kScratchSlots; ++which) scratch(which).release();
}
