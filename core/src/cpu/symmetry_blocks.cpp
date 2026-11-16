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
};

Skinny skinny_view(const Plan &p, const Block &g)
{
    if (g.rows <= g.cols)
        return Skinny{p.a + g.a_base, p.b + g.b_base, p.a_rows + g.r0, p.a_sum + g.k0, p.b_sum + g.k0,
                      p.b_cols + g.c0, p.rpos + g.r0, p.cpos + g.c0, g.rows, g.cols, g.depth, g.o_base};
    return Skinny{p.b + g.b_base, p.a + g.a_base, p.b_cols + g.c0, p.b_sum + g.k0, p.a_sum + g.k0,
                  p.a_rows + g.r0, p.cpos + g.c0, p.rpos + g.r0, g.cols, g.rows, g.depth, g.o_base};
}

// Long-side entries [l1, l2) of a skinny block, one sum index at a time over a
// line of long entries (the lines of later sums are then in cache, whatever
// the long operand's layout). Buffers: small (short x depth), line (kChunk),
// acc (short x kChunk).
void skinny_chunk(const Plan &p, const Skinny &s, i64 l1, i64 l2, double *small, double *line, double *acc)
{
    const i64 n = l2 - l1, ns = s.nshort, depth = s.depth;
    for (i64 i = 0; i < ns; ++i)
        for (i64 k = 0; k < depth; ++k) small[i * depth + k] = s.small[s.small_short[i] + s.small_sum[k]];
    std::fill(acc, acc + ns * n, 0.0);
    const i64 *longs = s.big_long + l1;
    for (i64 k = 0; k < depth; ++k) {
        const double *src = s.big + s.big_sum[k];
        for (i64 l = 0; l < n; ++l) line[l] = src[longs[l]];
        for (i64 i = 0; i < ns; ++i) {
            const double value = small[i * depth + k];
            double *ai = acc + i * n;
            for (i64 l = 0; l < n; ++l) ai[l] += value * line[l];
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
            double *line = scratch(2).get(size_t(kChunk));
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
            indirect_gemm(gemm_view(p, block));
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

extern "C" int itercc_sym_contract(int64_t nblocks, const int64_t *table, const double *a,
                                   const double *b, double *out, const int64_t *a_rows,
                                   const int64_t *a_sum, const int64_t *b_sum,
                                   const int64_t *b_cols, const int64_t *rpos,
                                   const int64_t *cpos, double alpha, int flags, int threads)
{
    if (nblocks <= 0 || alpha == 0.0) return 0;
    const Plan p{a, b, out, a_rows, a_sum, b_sum, b_cols, rpos, cpos, alpha,
                 (flags & 1) != 0, (flags & 2) != 0, (flags & 4) != 0};
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
                             scratch(2).get(size_t(kChunk)),
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
    const i64 share = std::max<i64>(1, total / (kTasksPerThread * std::max(1, threads)));
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
            itercc::indirect_gemm(gemm_view(p, task.block));
            PROFILE_ADD(1);
            break;
        }
    }
    PROFILE_ADD(6);
    return 0;
}

extern "C" void itercc_scratch_release(int threads)
{
    // Every thread of a team of this size frees its own buffers.
#pragma omp parallel num_threads(threads)
    for (int which = 0; which < itercc::kScratchSlots; ++which) scratch(which).release();
}
