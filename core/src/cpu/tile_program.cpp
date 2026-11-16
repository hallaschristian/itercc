// R4 tile programs: the tile loops of the CCSDTQ residual, slot by slot.
//
// The CCSDTQ equations add about a hundred contractions into each R4 tile. Run
// one call at a time, every small contraction pays a parallel region and its
// load imbalance. A tile program instead gives each thread whole slots: the
// thread's dense R4 tile stays in its caches while it adds every term with the
// single-threaded routes of contract_serial, then the tile goes to the packed
// residual. Each packed slot is written by one thread and every term's
// arithmetic is fixed, so results do not depend on the thread count.
//
// A term's plan depends only on the slot's irrep pattern, and its operands sit
// at offsets affine in the slot's occupied indices, so the program holds one
// block table per pattern and five offset coefficients per operand.
#include <omp.h>

#include <algorithm>
#include <vector>

#include "contract.h"
#include "gemm_kernel.h"
#include "itercc/core.h"
#include "scratch.h"

using i64 = int64_t;

#ifdef ITERCC_PROFILE_SYM
// Diagnostic per-term thread time (build with -DITERCC_PROFILE_SYM).
#include <atomic>
#include <cstdio>
#include <omp.h>
namespace {
constexpr int kTermSlots = 512;
std::atomic<double> term_time[kTermSlots];
}
extern "C" void itercc_tile_profile(int nterms, double *seconds, int reset)
{
    for (int t = 0; t < nterms && t < kTermSlots; ++t) {
        seconds[t] = term_time[t].load();
        if (reset) term_time[t] = 0.0;
    }
}
#endif

namespace {

i64 affine(const int64_t *coef, const int32_t *o)
{
    return coef[0] + coef[1] * o[0] + coef[2] * o[1] + coef[3] * o[2] + coef[4] * o[3];
}

// One nested level for the team mode, restored on exit. max-active-levels
// alone is not enough on GNU libgomp up to version 10 (the GCC 8.5 runtime of
// the cluster build): without nest-var every nested region gets one thread,
// so only threads / team cores would work. Restoring nest-var before the
// levels keeps both settings on runtimes that tie them together (LLVM, newer
// libgomp).
class NestedTeams {
  public:
    NestedTeams() : levels_(omp_get_max_active_levels())
    {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        nested_ = omp_get_nested();
        omp_set_nested(1);
#pragma GCC diagnostic pop
        omp_set_max_active_levels(std::max(levels_, 2));
    }
    ~NestedTeams()
    {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        omp_set_nested(nested_);
#pragma GCC diagnostic pop
        omp_set_max_active_levels(levels_);
    }
    NestedTeams(const NestedTeams &) = delete;
    NestedTeams &operator=(const NestedTeams &) = delete;

  private:
    int levels_, nested_ = 0;
};

}  // namespace

extern "C" void itercc_tile_program(const itercc_quad_layout *l, int64_t nslots, const int64_t *slots,
                                    const int32_t *occupied, const int64_t *pattern, int64_t nterms,
                                    const itercc_tile_term *terms, const double *t4, double *r4,
                                    double beta, int threads, int team)
{
    const i64 nv = l->nv, nv4 = nv * nv * nv * nv;
    team = std::max(1, std::min(team, threads));
    if (team > 1) {
        // Teams: threads / team slots at a time, each worked by a team of
        // `team` threads (a nested parallel contraction per term), so that the
        // live tiles fit the cache. The arithmetic is that of the serial path.
        const NestedTeams nesting;
#pragma omp parallel num_threads(std::max(1, threads / team))
        {
            double *tile = itercc::scratch(4).get(size_t(nv4));
            double *t4_tile = t4 != nullptr ? itercc::scratch(5).get(size_t(nv4)) : nullptr;
        bool zeroed = false;   // the terms read only allowed T4 entries: zero the tile once
#pragma omp for schedule(dynamic, 1)
            for (i64 s = 0; s < nslots; ++s) {
                const i64 slot = slots[s], kind = pattern[s];
                const int32_t *o = occupied + 4 * s;
                itercc_scale(tile, nv4, 0.0, team);
                if (t4 != nullptr) {
                    if (zeroed)
                        itercc_quad_place(l, t4 + l->soff[slot], slot, 1, t4_tile, team);
                    else
                        itercc_quad_expand(l, t4 + l->soff[slot], slot, 1, t4_tile, team);
                    zeroed = true;
                }
                for (i64 t = 0; t < nterms; ++t) {
                    const itercc_tile_term &term = terms[t];
                    const i64 count = term.plan_count[kind];
                    if (count == 0) continue;
                    const double *a = ((term.flags & 8) ? t4_tile : term.a) + affine(term.a_coef, o);
                    const double *b = ((term.flags & 16) ? t4_tile : term.b) + affine(term.b_coef, o);
                    itercc_sym_contract(count, term.blocks + 9 * term.plan_start[kind], a, b, tile,
                                        term.a_rows, term.a_sum, term.b_sum, term.b_cols, term.rpos,
                                        term.cpos, term.alpha, int(term.flags & 7), team);
                }
                const int32_t row[5] = {int32_t(slot), 0, 1, 2, 3};
                itercc_quad_scatter(l, tile, row, 1, 1.0, beta, r4, team);
            }
        }
        return;
    }
#pragma omp parallel num_threads(threads)
    {
        double *tile = itercc::scratch(4).get(size_t(nv4));
        double *t4_tile = t4 != nullptr ? itercc::scratch(5).get(size_t(nv4)) : nullptr;
        bool zeroed = false;   // the terms read only allowed T4 entries: zero the tile once
#pragma omp for schedule(dynamic, 1)
        for (i64 s = 0; s < nslots; ++s) {
            const i64 slot = slots[s], kind = pattern[s];
            const int32_t *o = occupied + 4 * s;
            std::fill(tile, tile + nv4, 0.0);
            if (t4 != nullptr) {
                if (zeroed)
                    itercc_quad_place(l, t4 + l->soff[slot], slot, 1, t4_tile, 1);
                else
                    itercc_quad_expand(l, t4 + l->soff[slot], slot, 1, t4_tile, 1);
                zeroed = true;
            }
            for (i64 t = 0; t < nterms; ++t) {
                const itercc_tile_term &term = terms[t];
                const i64 count = term.plan_count[kind];
                if (count == 0) continue;
                const double *a = ((term.flags & 8) ? t4_tile : term.a) + affine(term.a_coef, o);
                const double *b = ((term.flags & 16) ? t4_tile : term.b) + affine(term.b_coef, o);
                const itercc::ContractPlan plan =
                    itercc::contract_plan(a, b, tile, term.a_rows, term.a_sum, term.b_sum, term.b_cols,
                                          term.rpos, term.cpos, term.alpha, term.flags & 7);
#ifdef ITERCC_PROFILE_SYM
                const double start = omp_get_wtime();
#endif
                itercc::contract_serial(plan, term.blocks + 9 * term.plan_start[kind], count);
#ifdef ITERCC_PROFILE_SYM
                if (t < kTermSlots) {
                    double old = term_time[t].load(), add = omp_get_wtime() - start;
                    while (!term_time[t].compare_exchange_weak(old, old + add)) {
                    }
                }
#endif
            }
            const int32_t row[5] = {int32_t(slot), 0, 1, 2, 3};
            itercc_quad_scatter(l, tile, row, 1, 1.0, beta, r4, 1);
        }
    }
}

// Fused R4 tile programs. The terms of a tile program fall into a few output
// partitions (which tile indices the rows carry: ab|cd, ac|bd, ad|bc, a|bcd,
// ...). Per partition and row irrep, every term's product is one segment of a
// fused GEMM over the same canonical rows and columns, so the tile is read and
// written once per kGemmKc sums of all the terms together instead of once per
// term (each term's own pass over the tile cost about as much as its
// arithmetic for the shallow sums). Each entry's arithmetic is fixed by its
// block's segments, so results do not depend on the thread count or teams.

#ifdef ITERCC_PROFILE_SYM
namespace {
constexpr int kTagSlots = 64;
std::atomic<double> tag_time[kTagSlots];
}
extern "C" void itercc_fused_tile_profile(int ntags, double *seconds, int reset)
{
    for (int t = 0; t < ntags && t < kTagSlots; ++t) {
        seconds[t] = tag_time[t].load();
        if (reset) tag_time[t] = 0.0;
    }
}
#endif

namespace {

constexpr i64 kFusedPiece = 256;   // columns per team task

// The segments of a fused block for the slot with occupied indices o.
// An operand's start: base `index` (-1: the slot's T4 tile) at element offset
// `at`, in floats when base_f32 flags the base.
const double *operand(i64 index, i64 at, const double *const *bases, const uint8_t *base_f32,
                      const double *t4_tile, bool &f32)
{
    f32 = index >= 0 && base_f32 != nullptr && base_f32[index] != 0;
    if (index < 0) return t4_tile + at;
    if (f32) return reinterpret_cast<const double *>(reinterpret_cast<const float *>(bases[index]) + at);
    return bases[index] + at;
}

void resolve(const itercc_fused_block &b, const itercc_fused_segment *segments, const double *const *bases,
             const uint8_t *base_f32, const int32_t *o, const double *t4_tile,
             std::vector<itercc::GemmSegment> &out, double alpha = 1.0)
{
    static const int32_t origin[4] = {0, 0, 0, 0};
    if (o == nullptr) o = origin;
    out.resize(size_t(b.seg_count));
    for (i64 s = 0; s < b.seg_count; ++s) {
        const itercc_fused_segment &x = segments[b.seg_start + s];
        bool xf = false, yf = false;
        const double *xp = operand(x.x_base, affine(x.x_coef, o), bases, base_f32, t4_tile, xf);
        const double *yp = operand(x.y_base, affine(x.y_coef, o), bases, base_f32, t4_tile, yf);
        itercc::GemmSegment g{xp, x.x_rows, x.x_sum, (x.flags & 1) != 0,
                              yp, x.y_sum, x.y_cols, (x.flags & 2) != 0, x.depth,
                              x.alpha * alpha};
        g.a_f32 = xf;
        g.b_f32 = yf;
        out[size_t(s)] = g;
    }
}

void fused_block(const itercc_fused_block &b, const itercc_fused_segment *segments,
                 const double *const *bases, const uint8_t *base_f32, const int32_t *o,
                 const double *t4_tile, double *tile, i64 j0, i64 j1)
{
    thread_local std::vector<itercc::GemmSegment> segs;
    resolve(b, segments, bases, base_f32, o, t4_tile, segs);
#ifdef ITERCC_PROFILE_SYM
    const double start = omp_get_wtime();
#endif
    itercc::fused_gemm({b.rows, j0, j1, b.seg_count, segs.data(), tile + b.o_base, b.rpos, b.cpos});
#ifdef ITERCC_PROFILE_SYM
    if (b.tag >= 0 && b.tag < kTagSlots) {
        double old = tag_time[b.tag].load(), add = omp_get_wtime() - start;
        while (!tag_time[b.tag].compare_exchange_weak(old, old + add)) {
        }
    }
#endif
}

}  // namespace

extern "C" void itercc_fused_contract(int64_t nblocks, const itercc_fused_block *blocks,
                                      const itercc_fused_segment *segments, const double *const *bases,
                                      const uint8_t *base_f32, double *out, double alpha, int threads)
{
    // Tasks: column ranges of the blocks, about four per thread by work (a
    // range repacks its block's rows, so ranges stay at least kMinCols wide).
    // The outputs are row buffers far larger than the caches: B is packed 256
    // columns at a time (512 KB per chunk of sums, which stays in L2 while the
    // rows pass; 1024 columns fill L2), worth about 9% on the ring passes at
    // 47 virtuals. The tile programs keep the default: their A operands would
    // be packed more often.
    constexpr i64 kMinCols = 64, kRowBufferNc = 256;
    double work = 0.0;
    for (i64 b = 0; b < nblocks; ++b) {
        i64 depth = 0;
        for (i64 s = 0; s < blocks[b].seg_count; ++s) depth += segments[blocks[b].seg_start + s].depth;
        work += double(blocks[b].rows) * double(blocks[b].cols) * double(depth);
    }
    const double target = work / double(4 * std::max(1, threads)) + 1.0;
    std::vector<i64> tasks;
    for (i64 b = 0; b < nblocks; ++b) {
        const itercc_fused_block &x = blocks[b];
        i64 depth = 0;
        for (i64 s = 0; s < x.seg_count; ++s) depth += segments[x.seg_start + s].depth;
        const double own = double(x.rows) * double(x.cols) * double(depth);
        i64 pieces = std::max<i64>(1, i64(own / target));
        pieces = std::min(pieces, std::max<i64>(1, x.cols / kMinCols));
        const i64 width = (x.cols + pieces - 1) / pieces;
        for (i64 j = 0; j < x.cols; j += width) {
            tasks.push_back(b);
            tasks.push_back(j);
            tasks.push_back(std::min(x.cols, j + width));
        }
    }
    const i64 ntasks = i64(tasks.size()) / 3;
    // Always the full team: libgomp ends the surplus threads of its pool when a
    // region asks for fewer and creates new ones (with fresh per-thread GEMM
    // scratch) when a later region asks for more; thousands of calls with
    // varying task counts per evaluation made it churn through threads.
#pragma omp parallel num_threads(std::max(1, threads))
    {
        std::vector<itercc::GemmSegment> segs;
#pragma omp for schedule(dynamic, 1)
        for (i64 t = 0; t < ntasks; ++t) {
            const itercc_fused_block &x = blocks[tasks[size_t(3 * t)]];
            resolve(x, segments, bases, base_f32, nullptr, nullptr, segs, alpha);
            itercc::fused_gemm({x.rows, tasks[size_t(3 * t + 1)], tasks[size_t(3 * t + 2)], x.seg_count,
                                segs.data(), out + x.o_base, x.rpos, x.cpos, kRowBufferNc});
        }
    }
}

extern "C" void itercc_fused_orientation(int mode)
{
    itercc::set_fused_orientation(mode);
}

extern "C" void itercc_fused_tile_program(const itercc_quad_layout *l, int64_t nslots, const int64_t *slots,
                                          const int32_t *occupied, const int64_t *pattern,
                                          const int64_t *block_start, const int64_t *block_count,
                                          const itercc_fused_block *blocks,
                                          const itercc_fused_segment *segments,
                                          const double *const *bases, const uint8_t *base_f32,
                                          const double *t4, double *r4, double beta, int threads, int team)
{
    const i64 nv = l->nv, nv4 = nv * nv * nv * nv;
    team = std::max(1, std::min(team, threads));
    if (team > 1) {
        // Teams of `team` threads per slot share its tile; a team splits the
        // slot's blocks into column ranges.
        const NestedTeams nesting;
#pragma omp parallel num_threads(std::max(1, threads / team))
        {
            double *tile = itercc::scratch(4).get(size_t(nv4));
            double *t4_tile = t4 != nullptr ? itercc::scratch(5).get(size_t(nv4)) : nullptr;
            bool zeroed = false;   // the blocks read only allowed T4 entries: zero the tile once
            bool tile_zeroed = false;
            std::vector<i64> tasks, waves;
#pragma omp for schedule(dynamic, 1)
            for (i64 s = 0; s < nslots; ++s) {
                const i64 slot = slots[s], kind = pattern[s];
                const int32_t *o = occupied + 4 * s;
                // Blocks write and the scatter reads only the slot's allowed
                // entries: after one full clear, clear only those.
                if (tile_zeroed)
                    itercc_quad_zero_allowed(l, slot, tile, team);
                else
                    itercc_scale(tile, nv4, 0.0, team);
                tile_zeroed = true;
                if (t4 != nullptr) {
                    if (zeroed)
                        itercc_quad_place(l, t4 + l->soff[slot], slot, 1, t4_tile, team);
                    else
                        itercc_quad_expand(l, t4 + l->soff[slot], slot, 1, t4_tile, team);
                    zeroed = true;
                }
                // Column ranges of the blocks as tasks, in waves of one tag:
                // blocks of one tag write disjoint entries, blocks of
                // different tags the same ones (in tag order, as serially).
                const itercc_fused_block *own = blocks + block_start[kind];
                tasks.clear();
                waves.assign(1, 0);
                for (i64 b = 0; b < block_count[kind]; ++b) {
                    if (b > 0 && own[b].tag != own[b - 1].tag) waves.push_back(i64(tasks.size()) / 2);
                    for (i64 j = 0; j < own[b].cols; j += kFusedPiece) {
                        tasks.push_back(b);
                        tasks.push_back(j);
                    }
                }
                waves.push_back(i64(tasks.size()) / 2);
#pragma omp parallel num_threads(team)
                for (size_t w = 0; w + 1 < waves.size(); ++w) {
#pragma omp for schedule(dynamic, 1)
                    for (i64 t = waves[w]; t < waves[w + 1]; ++t) {
                        const itercc_fused_block &b = own[tasks[size_t(2 * t)]];
                        const i64 j0 = tasks[size_t(2 * t + 1)];
                        fused_block(b, segments, bases, base_f32, o, t4_tile, tile, j0,
                                    std::min(b.cols, j0 + kFusedPiece));
                    }
                }
                const int32_t row[5] = {int32_t(slot), 0, 1, 2, 3};
                itercc_quad_scatter(l, tile, row, 1, 1.0, beta, r4, team);
            }
        }
        return;
    }
#pragma omp parallel num_threads(threads)
    {
        double *tile = itercc::scratch(4).get(size_t(nv4));
        double *t4_tile = t4 != nullptr ? itercc::scratch(5).get(size_t(nv4)) : nullptr;
        bool zeroed = false, tile_zeroed = false;
#pragma omp for schedule(dynamic, 1)
        for (i64 s = 0; s < nslots; ++s) {
            const i64 slot = slots[s], kind = pattern[s];
            const int32_t *o = occupied + 4 * s;
            if (tile_zeroed)
                itercc_quad_zero_allowed(l, slot, tile, 1);
            else
                std::fill(tile, tile + nv4, 0.0);
            tile_zeroed = true;
            if (t4 != nullptr) {
                if (zeroed)
                    itercc_quad_place(l, t4 + l->soff[slot], slot, 1, t4_tile, 1);
                else
                    itercc_quad_expand(l, t4 + l->soff[slot], slot, 1, t4_tile, 1);
                zeroed = true;
            }
            const itercc_fused_block *own = blocks + block_start[kind];
            for (i64 b = 0; b < block_count[kind]; ++b)
                fused_block(own[b], segments, bases, base_f32, o, t4_tile, tile, 0, own[b].cols);
            const int32_t row[5] = {int32_t(slot), 0, 1, 2, 3};
            itercc_quad_scatter(l, tile, row, 1, 1.0, beta, r4, 1);
        }
    }
}
