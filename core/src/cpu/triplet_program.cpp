// Rank-four slot programs of the triplet EOM-CCSDTQ action (recorded by
// itercc/_spin_quadruples_program.py).
//
// One action contracts, for every occupied slot and projection group, a few
// dozen raw nv^4 tiles from small products and gathers each tile into the
// slot's packed rows. Here every tile is a task: one thread zeroes its tile,
// runs the products of the tile's kind with the single-threaded contraction
// routes (contract_serial: the same sums as itercc_sym_contract on any number
// of threads), and writes the tile's gathers to partial sums. The slots then
// add their tiles' partial sums in the recorded order and divide by (4!)^2,
// as the Python loop does, so the result does not depend on the thread count.
// A slot's tiles come from its class (recorded once): kinds, occupied
// permutations and gathers; slots run in waves that bound the partial sums.
//
// Products of three or four operands go through intermediates. Each one is
// computed by one tile and shared through a cache keyed by its expression and
// the occupied indices it depends on: other tiles that need it wait for its
// producer, and finished intermediates stay while they fit the byte budget.
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <cstdlib>
#include <list>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "contract.h"
#include "gemm_kernel.h"
#include "itercc/core.h"
#include "scratch.h"

namespace {

using i64 = int64_t;
using Buffer = std::shared_ptr<std::vector<double>>;

// Diagnostic thread time by phase (ITERCC_SLOT_PROFILE=1, read by
// itercc_triplet_profile): 0 zeroing, 1 products into the tile, 2 products
// into intermediates, 3 packed T4 row setup, 4 gathers, 5 cache waits and
// lookups, 6 slot sums, 7 intermediate allocation, 8 the barrier at the end
// of a wave's tiles.
constexpr int kPhases = 9;
std::atomic<double> phase_time[kPhases];
constexpr i64 kTimed = 1 << 16;        // contractions timed one by one (instruction `inter`)
std::atomic<double> contraction_time[kTimed];
std::atomic<i64> contraction_calls[kTimed];
const bool profiling = [] {
    const char *value = std::getenv("ITERCC_SLOT_PROFILE");
    return value != nullptr && value[0] == '1';
}();

void add_time(std::atomic<double> &total, double add)
{
    double old = total.load();
    while (!total.compare_exchange_weak(old, old + add)) {
    }
}

class Phase {
  public:
    explicit Phase(int which) : which_(which), start_(profiling ? omp_get_wtime() : 0.0) {}
    ~Phase()
    {
        if (!profiling) return;
        double old = phase_time[which_].load(), add = omp_get_wtime() - start_;
        while (!phase_time[which_].compare_exchange_weak(old, old + add)) {
        }
    }

  private:
    int which_;
    double start_;
};

constexpr i64 kDense = 0, kRows = 1, kInter = 2;
constexpr i64 kLookup = 1, kPublish = 2;
constexpr double kDivisor = 576.0;   // (4!)^2 antisymmetrizer terms

i64 affine(const int64_t *coef, const int64_t *fixed)
{
    return coef[0] + coef[1] * fixed[0] + coef[2] * fixed[1] + coef[3] * fixed[2] + coef[4] * fixed[3];
}

struct Key {
    i64 id, v[4];
    bool operator==(const Key &o) const
    {
        return id == o.id && v[0] == o.v[0] && v[1] == o.v[1] && v[2] == o.v[2] && v[3] == o.v[3];
    }
};

struct KeyHash {
    size_t operator()(const Key &k) const
    {
        uint64_t h = uint64_t(k.id) * 0x9E3779B97F4A7C15ull;
        for (const i64 x : k.v) h = (h ^ uint64_t(x)) * 0x100000001B3ull + (h >> 29);
        return size_t(h);
    }
};

Key key_of(const itercc_slot_intermediate &m, const int64_t *fixed)
{
    Key key{m.key, {-1, -1, -1, -1}};
    for (i64 i = 0; i < m.nvars; ++i) key.v[i] = fixed[m.positions[i]];
    return key;
}

// A large intermediate in production (the g T4 products at CuF size: about
// 2e8 multiply-adds each): the pieces of its contraction, which the tiles
// waiting for it run alongside its producer instead of sleeping. The pieces
// write disjoint entries with the arithmetic of contract_serial, so the
// intermediate is the same whoever runs them.
constexpr i64 kShareWork = i64(1) << 22;   // multiply-adds from which an intermediate is shared

struct Job {
    itercc::ContractPlan plan;
    itercc::PackedRows rows{};
    std::vector<itercc::ContractTask> tasks;
    std::atomic<i64> next{0}, done{0};

    bool open() const { return next.load(std::memory_order_relaxed) < i64(tasks.size()); }
    bool complete() const { return done.load(std::memory_order_acquire) >= i64(tasks.size()); }
    bool step()   // runs one piece if one is left
    {
        const i64 t = next.fetch_add(1, std::memory_order_relaxed);
        if (t >= i64(tasks.size())) return false;
        itercc::contract_task(plan, tasks[size_t(t)]);
        done.fetch_add(1, std::memory_order_release);
        return true;
    }
    i64 help()   // runs pieces until none is left; returns how many it ran
    {
        i64 ran = 0;
        while (step()) ++ran;
        return ran;
    }
};

// Intermediates shared between tiles: one producer per key while the others
// wait (or help with its pieces); finished buffers stay while they fit the
// byte budget (least recently used out first). A buffer in use outlives its
// eviction.
class Cache {
  public:
    explicit Cache(i64 budget) : budget_(budget) {}

    // The finished buffer of `key` (waiting while another tile computes it),
    // or null with the caller registered as its producer.
    Buffer acquire(const Key &key)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            auto found = map_.find(key);
            if (found == map_.end()) {
                map_.emplace(key, Entry{});
                ++misses_;
                return nullptr;
            }
            if (found->second.ready) {
                ++hits_;
                lru_.splice(lru_.begin(), lru_, found->second.place);
                return found->second.buffer;
            }
            const std::shared_ptr<Job> job = found->second.job;
            if (job && job->open()) {
                lock.unlock();
                {
                    const Phase timing(2);
                    helped_ += job->help();
                }
                lock.lock();
                continue;
            }
            // Its pieces are all taken (or not cut yet): help another one.
            std::shared_ptr<Job> other = open_job();
            if (other) {
                lock.unlock();
                {
                    const Phase timing(2);
                    if (other->step()) ++helped_;
                }
                lock.lock();
                continue;
            }
            const Phase timing(5);
            ready_.wait(lock);
        }
    }

    // A registered key whose producer failed (out of memory): the entry goes,
    // and a waiting tile becomes its producer instead.
    void abandon(const Key &key)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto found = map_.find(key);
            if (found != map_.end() && !found->second.ready) map_.erase(found);
        }
        ready_.notify_all();
    }

    // One piece of any intermediate being produced; false when none is left.
    // A producer runs these while helpers finish its own last pieces.
    bool help_other()
    {
        std::shared_ptr<Job> job;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job = open_job();
        }
        if (!job || !job->step()) return false;
        ++helped_;
        return true;
    }

    // The pieces of an intermediate being produced, for the tiles that wait.
    void attach(const Key &key, const std::shared_ptr<Job> &job)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++shared_;
            auto found = map_.find(key);
            if (found != map_.end()) found->second.job = job;
            open_.push_back(job);
        }
        ready_.notify_all();
    }

    // The producer's buffer: kept when it fits the budget, else dropped (a
    // waiting tile then computes it itself).
    void publish(const Key &key, const Buffer &buffer)
    {
        const i64 bytes = i64(buffer->size() * sizeof(double));
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto found = map_.find(key);
            if (bytes > budget_) {
                map_.erase(found);
            } else {
                while (bytes_ + bytes > budget_ && !lru_.empty()) {
                    auto oldest = map_.find(lru_.back());
                    bytes_ -= i64(oldest->second.buffer->size() * sizeof(double));
                    map_.erase(oldest);
                    lru_.pop_back();
                }
                lru_.push_front(key);
                found->second.buffer = buffer;
                found->second.ready = true;
                found->second.place = lru_.begin();
                found->second.job.reset();
                bytes_ += bytes;
                peak_ = std::max(peak_, bytes_);
            }
        }
        ready_.notify_all();
    }

    i64 hits() const { return hits_; }
    i64 misses() const { return misses_; }
    i64 peak() const { return peak_; }
    i64 shared() const { return shared_; }
    i64 helped() const { return helped_.load(); }

  private:
    // A job with pieces left (the lock held); finished ones leave the list.
    std::shared_ptr<Job> open_job()
    {
        while (!open_.empty() && !open_.front()->open()) open_.pop_front();
        for (const auto &job : open_)
            if (job->open()) return job;
        return nullptr;
    }

    struct Entry {
        Buffer buffer;
        bool ready = false;
        std::list<Key>::iterator place;
        std::shared_ptr<Job> job;   // its pieces while it is produced
    };
    std::mutex mutex_;
    std::condition_variable ready_;
    std::unordered_map<Key, Entry, KeyHash> map_;
    std::list<Key> lru_;   // finished entries, most recent first
    i64 budget_, bytes_ = 0, peak_ = 0, hits_ = 0, misses_ = 0, shared_ = 0;
    std::atomic<i64> helped_{0};   // pieces run by tiles that waited
    std::deque<std::shared_ptr<Job>> open_;   // jobs being produced, oldest first
};

// A plan block of a fused product: its output block and where it came from.
struct Piece {
    i64 r0, c0, rows, cols, o_base, member;
    const int64_t *block;
    const itercc_slot_instruction *x;
    const double *a, *b;
};

struct Worker {
    std::vector<Buffer> bound;   // the tile's intermediates
    std::vector<char> owner;     // ... that it produces for the cache
    std::vector<int64_t> bases, maps;
    std::vector<Piece> pieces;
    std::vector<itercc::GemmSegment> segments;
};

// The symmetry-allowed entries of a compact tile (irreps of its stored axes
// whose product is `target`), each a run of its last axis: its products
// write and its gathers read no others. A chain keeps every irrep, so
// clearing the whole compact tile cleared four times too much (C2v) and
// pulled all of it through the caches.
// A short run of zeros as a loop (a memset call per run of a few doubles
// cost more than the stores).
__attribute__((optimize("no-tree-loop-distribute-patterns")))
inline void zero_run(double *x, i64 n)
{
    for (i64 i = 0; i < n; ++i) x[i] = 0.0;
}

void clear_allowed(double *tile, const itercc_slot_tile_axes &ax, i64 nirrep, i64 target)
{
    const i64 lead = ax.naxes - 1;
    const int64_t (*st)[9] = ax.starts;
    const int64_t *step = ax.strides;
    i64 g[4] = {0, 0, 0, 0};
    for (;;) {
        i64 last = target;
        for (i64 a = 0; a < lead; ++a) last ^= g[a];
        if (last < nirrep && st[lead][last + 1] > st[lead][last]) {
            const i64 lo = st[lead][last], n = st[lead][last + 1] - lo;
            if (lead == 0) {
                zero_run(tile + lo, n);
            } else {
                for (i64 i = st[0][g[0]]; i < st[0][g[0] + 1]; ++i) {
                    double *x = tile + i * step[0] + lo;
                    if (lead == 1) {
                        zero_run(x, n);
                        continue;
                    }
                    for (i64 j = st[1][g[1]]; j < st[1][g[1] + 1]; ++j) {
                        double *y = x + j * step[1];
                        if (lead == 2) {
                            zero_run(y, n);
                            continue;
                        }
                        for (i64 k = st[2][g[2]]; k < st[2][g[2] + 1]; ++k) zero_run(y + k * step[2], n);
                    }
                }
            }
        }
        i64 a = lead - 1;
        while (a >= 0 && ++g[a] == nirrep) g[a--] = 0;
        if (a < 0) break;
    }
}

// The packed T4 rows of descriptor `index` for the tile's occupied tuple:
// rows over the hole letters' ranges in C order, bases and map addresses from
// the layout's hole tables (as _spin_quadruples_storage.signed_rows).
itercc::PackedRows packed_rows(const itercc_slot_program &p, i64 index, const int64_t *fixed,
                               const double *const *vectors, Worker &w)
{
    const itercc_slot_rows &d = p.rows[index];
    const int64_t *hole_bases = d.which ? p.direction_bases : p.ground_bases;
    const int32_t *hole_keys = d.which ? p.direction_keys : p.ground_keys;
    const i64 n = p.holes, nv = p.nv;
    i64 lo[4];
    for (int k = 0; k < 4; ++k) lo[k] = d.fixed[k] >= 0 ? fixed[d.fixed[k]] : d.start[k];
    const i64 total = d.count[0] * d.count[1] * d.count[2] * d.count[3];
    w.bases.resize(size_t(total));
    w.maps.resize(size_t(total));
    i64 r = 0;
    for (i64 a = 0; a < d.count[0]; ++a)
        for (i64 b = 0; b < d.count[1]; ++b)
            for (i64 c = 0; c < d.count[2]; ++c)
                for (i64 e = 0; e < d.count[3]; ++e, ++r) {
                    const i64 at = (((2 * (lo[0] + a) + d.spins[0]) * n + 2 * (lo[1] + b) + d.spins[1]) * n
                                    + 2 * (lo[2] + c) + d.spins[2]) * n + 2 * (lo[3] + e) + d.spins[3];
                    const i64 base = hole_bases[at];
                    w.bases[size_t(r)] = base;
                    w.maps[size_t(r)] = base != 0 ? d.pointers[hole_keys[at]] : 0;
                }
    return itercc::PackedRows{vectors[d.which], w.bases.data(), w.maps.data(), nv * nv * nv * nv};
}

const double *operand(const itercc_slot_operand &o, const int64_t *fixed, const double *const *bases,
                      const Worker &w, const double *tile)
{
    if (o.kind == kDense) return bases[o.index] + affine(o.coef, fixed);
    if (o.kind == kInter) return w.bound[size_t(o.index)]->data();
    if (o.kind == kRows) return nullptr;
    return tile + o.coef[0];
}

// A tile of a wave: its slot, template (class tile), partial sums (in the
// wave's buffer), plan set and occupied tuple.
struct Tile {
    i64 slot, tmpl, partial, plans;
    int64_t fixed[4];
};

// The products of fusion group `group` of the tile's kind as fused GEMMs:
// their plan blocks with the same output block (equal offset tables, so the
// same entries) become the segments of one GEMM, in instruction order.
void run_fused(const itercc_slot_program &p, const itercc_slot_template &m, const Tile &job, i64 group,
               const itercc_slot_instruction *code, const int64_t *tables, const int64_t *counts,
               const double *const *bases, double *tile, Worker &w)
{
    const int64_t *g = p.fuse_groups + 2 * (p.fuse_start[m.kind] + group);
    w.pieces.clear();
    for (i64 i = 0; i < g[1]; ++i) {
        const itercc_slot_instruction &x = code[p.fuse_members[g[0] + i]];
        const int64_t *table = reinterpret_cast<const int64_t *>(tables[x.plan]);
        const double *a = operand(x.a, job.fixed, bases, w, tile), *b = operand(x.b, job.fixed, bases, w, tile);
        for (i64 k = 0; k < counts[x.plan]; ++k) {
            const int64_t *t = table + 9 * k;
            w.pieces.push_back(Piece{t[3], t[4], t[0], t[1], t[8], i, t, &x, a, b});
        }
    }
    std::stable_sort(w.pieces.begin(), w.pieces.end(), [](const Piece &u, const Piece &v) {
        if (u.r0 != v.r0) return u.r0 < v.r0;
        if (u.c0 != v.c0) return u.c0 < v.c0;
        if (u.rows != v.rows) return u.rows < v.rows;
        if (u.cols != v.cols) return u.cols < v.cols;
        return u.o_base < v.o_base;
    });
    for (size_t i = 0; i < w.pieces.size();) {
        size_t j = i;
        w.segments.clear();
        while (j < w.pieces.size() && w.pieces[j].r0 == w.pieces[i].r0 && w.pieces[j].c0 == w.pieces[i].c0
               && w.pieces[j].rows == w.pieces[i].rows && w.pieces[j].cols == w.pieces[i].cols
               && w.pieces[j].o_base == w.pieces[i].o_base) {
            const Piece &u = w.pieces[j];
            const int64_t *t = u.block;   // rows, cols, depth, r0, c0, k0, a_base, b_base, o_base
            itercc::GemmSegment segment{u.a + t[6], u.x->a_rows + t[3], u.x->a_sum + t[5], (u.x->flags & 1) != 0,
                                        u.b + t[7], u.x->b_sum + t[5], u.x->b_cols + t[4], (u.x->flags & 2) != 0,
                                        t[2], u.x->alpha};
            w.segments.push_back(segment);
            ++j;
        }
        const Piece &u = w.pieces[i];
        const double start = profiling ? omp_get_wtime() : 0.0;
        itercc::fused_gemm({u.rows, 0, u.cols, i64(w.segments.size()), w.segments.data(),
                            tile + u.x->out.coef[0] + u.o_base, u.x->rpos + u.r0, u.x->cpos + u.c0});
        if (profiling) {
            // The fused GEMM's time shared among its products by their sums.
            const double spent = omp_get_wtime() - start;
            double depth = 0.0;
            for (size_t s = i; s < j; ++s) depth += double(w.pieces[s].block[2]);
            for (size_t s = i; s < j; ++s) {
                const i64 id = w.pieces[s].x->inter;
                if (id < 0 || id >= kTimed) continue;
                add_time(contraction_time[id], spent * double(w.pieces[s].block[2]) / std::max(depth, 1.0));
                ++contraction_calls[id];
            }
        }
        i = j;
    }
}

// The gathers of one part: rows[r] = sum over the terms k, in order, of
// weights[k] * tile[offset of row r's coordinates under term k] (a compact
// tile's chains through their rank tables). Terms run outermost, each over
// every row with its steps, members and tables hoisted: row r's sum gets the
// same additions in the same order as a row-by-row loop (from +0.0), so the
// partial sums are those of itercc_permuted_gather_add bit for bit.
template <int Chains, int First, int Second>
void gather_terms(const itercc_slot_terms &t, const int32_t *coordinates, i64 count, const double *tile,
                  double *rows)
{
    const i64 nv = t.nv;
    for (i64 k = 0; k < t.count; ++k) {
        const int64_t *step = t.offsets + 4 * k;
        const i64 s0 = step[0], s1 = step[1], s2 = step[2], s3 = step[3];
        const int64_t *m0 = t.members + 8 * k, *m1 = m0 + 4;
        const int64_t *table0 = Chains > 0 ? t.tables[0] : nullptr, *table1 = Chains > 1 ? t.tables[1] : nullptr;
        const i64 stride0 = Chains > 0 ? t.strides[0] : 0, stride1 = Chains > 1 ? t.strides[1] : 0;
        const double w = t.weights[k];
        for (i64 r = 0; r < count; ++r) {
            const int32_t *c = coordinates + 4 * r;
            i64 at = c[0] * s0 + c[1] * s1 + c[2] * s2 + c[3] * s3;
            if constexpr (Chains > 0) {
                i64 flat = c[m0[0]] * nv + c[m0[1]];
                if constexpr (First == 3) flat = flat * nv + c[m0[2]];
                at += table0[flat] * stride0;
            }
            if constexpr (Chains > 1) {
                i64 flat = c[m1[0]] * nv + c[m1[1]];
                if constexpr (Second == 3) flat = flat * nv + c[m1[2]];
                at += table1[flat] * stride1;
            }
            rows[r] += w * tile[at];
        }
    }
}

void gather_part(const itercc_slot_terms &t, const int32_t *coordinates, i64 count, const double *tile,
                 double *rows)
{
    std::fill(rows, rows + count, 0.0);
    const i64 first = t.nchains > 0 ? t.lengths[0] : 0, second = t.nchains > 1 ? t.lengths[1] : 0;
    if (t.nchains == 0) return gather_terms<0, 0, 0>(t, coordinates, count, tile, rows);
    if (t.nchains == 1 && first == 2) return gather_terms<1, 2, 0>(t, coordinates, count, tile, rows);
    if (t.nchains == 1 && first == 3) return gather_terms<1, 3, 0>(t, coordinates, count, tile, rows);
    if (t.nchains == 2 && first == 2 && second == 2) return gather_terms<2, 2, 2>(t, coordinates, count, tile, rows);
    if (t.nchains == 2 && first == 2 && second == 3) return gather_terms<2, 2, 3>(t, coordinates, count, tile, rows);
    if (t.nchains == 2 && first == 3 && second == 2) return gather_terms<2, 3, 2>(t, coordinates, count, tile, rows);
    // Any other chain shape (a chain of four, say): the general loop.
    for (i64 r = 0; r < count; ++r) {
        const int32_t *c = coordinates + 4 * r;
        double sum = 0.;
        for (i64 k = 0; k < t.count; ++k) {
            const int64_t *step = t.offsets + 4 * k;
            i64 at = c[0] * step[0] + c[1] * step[1] + c[2] * step[2] + c[3] * step[3];
            for (i64 g = 0; g < t.nchains; ++g) {       // a compact tile's chains
                const int64_t *member = t.members + 8 * k + 4 * g;
                i64 flat = 0;
                for (i64 j = 0; j < t.lengths[g]; ++j) flat = flat * t.nv + c[member[j]];
                at += t.tables[g][flat] * t.strides[g];
            }
            sum += t.weights[k] * tile[at];
        }
        rows[r] = sum;
    }
}

// One tile: its products, then its gathers into `partial`. With `fused`, the
// products of fusion groups wait until the other instructions are done (the
// intermediates they read stay bound) and run as fused GEMMs.
void run_tile(const itercc_slot_program &p, const Tile &job, const double *const *bases,
              const double *const *vectors, Cache &cache, i64 cacheable, double *tile, Worker &w,
              double *partial, bool fused, int workers, i64 share_work)
{
    const i64 nv = p.nv, nv4 = nv * nv * nv * nv;
    const itercc_slot_template &m = p.templates[job.tmpl];
    {
        const Phase timing(0);
        static const bool runs = [] {
            const char *value = std::getenv("ITERCC_TILE_RUNS");
            return value == nullptr || value[0] != '0';
        }();
        i64 target = p.tile_irrep;
        for (int k = 0; k < 4; ++k) target ^= p.occupied_irreps[job.fixed[k]];
        // Only the symmetry-allowed entries, dense tile or compact, where the
        // kind's stored axes keep each irrep as one run.
        const itercc_slot_tile_axes *axes = p.tile_axes != nullptr ? p.tile_axes + m.kind : nullptr;
        if (axes == nullptr || axes->naxes == 0 || !runs)
            std::fill(tile, tile + p.tile_sizes[m.kind], 0.0);
        else
            clear_allowed(tile, *axes, p.nirrep, target);
    }
    const itercc_slot_instruction *code = p.instructions + p.kind_start[m.kind];
    const i64 ncode = p.kind_count[m.kind];
    const itercc_slot_intermediate *inters = p.intermediates + p.inter_start[m.kind];
    w.bound.assign(size_t(p.inter_count[m.kind]), nullptr);
    w.owner.assign(size_t(p.inter_count[m.kind]), 0);
    const int64_t *tables = p.plan_table + p.plan_start[job.plans];
    const int64_t *counts = p.plan_count + p.plan_start[job.plans];
    for (i64 pc = 0; pc < ncode;) {
        const itercc_slot_instruction &x = code[pc];
        if (x.op == kLookup) {
            const size_t k = size_t(x.inter);
            if (w.bound[k]) {
                pc = x.next;
                continue;
            }
            if (inters[k].size * i64(sizeof(double)) <= cacheable) {
                Buffer hit = cache.acquire(key_of(inters[k], job.fixed));
                if (hit) {
                    w.bound[k] = std::move(hit);
                    pc = x.next;
                    continue;
                }
                w.owner[k] = 1;
            }
            ++pc;
            continue;
        }
        if (x.op == kPublish) {
            const size_t k = size_t(x.inter);
            if (w.owner[k]) {
                cache.publish(key_of(inters[k], job.fixed), w.bound[k]);
                w.owner[k] = 0;
            }
            ++pc;
            continue;
        }
        if (fused && x.next >= 0) {   // a product of a fusion group: below
            ++pc;
            continue;
        }
        double *out;
        if (x.out.kind == kInter) {
            const Phase timing(7);
            Buffer fresh = std::make_shared<std::vector<double>>(size_t(inters[x.out.index].size), 0.0);
            out = fresh->data();
            w.bound[size_t(x.out.index)] = std::move(fresh);
        } else {
            out = tile + x.out.coef[0];
        }
        itercc::ContractPlan plan = itercc::contract_plan(
            operand(x.a, job.fixed, bases, w, tile), operand(x.b, job.fixed, bases, w, tile), out, x.a_rows,
            x.a_sum, x.b_sum, x.b_cols, x.rpos, x.cpos, x.alpha, x.flags);
        itercc::PackedRows rows{};
        if (x.b.kind == kRows) {
            const Phase timing(3);
            rows = packed_rows(p, x.b.index, job.fixed, vectors, w);
            plan.packed_b = &rows;
        }
        const i64 count = counts[x.plan];
        if (count > 0) {
            const Phase timing(x.out.kind == kInter ? 2 : 1);
            const double start = profiling ? omp_get_wtime() : 0.0;
            const int64_t *table = reinterpret_cast<const int64_t *>(tables[x.plan]);
            i64 work = 0;
            if (x.out.kind == kInter && w.owner[size_t(x.out.index)] && workers > 1)
                for (i64 g = 0; g < count; ++g) work += table[9 * g] * table[9 * g + 1] * table[9 * g + 2];
            if (work >= share_work) {
                // Shared with the tiles that wait for this intermediate.
                auto shared = std::make_shared<Job>();
                shared->plan = plan;
                if (plan.packed_b != nullptr) {
                    shared->rows = rows;
                    shared->plan.packed_b = &shared->rows;
                }
                itercc::contract_tasks(table, count, workers, shared->tasks);
                cache.attach(key_of(inters[size_t(x.out.index)], job.fixed), shared);
                shared->help();
                // Helpers finish the last pieces: meanwhile, run other pieces.
                while (!shared->complete())
                    if (!cache.help_other()) std::this_thread::yield();
            } else {
                itercc::contract_serial(plan, table, count);
            }
            if (profiling && x.inter >= 0 && x.inter < kTimed) {
                double old = contraction_time[x.inter].load(), add = omp_get_wtime() - start;
                while (!contraction_time[x.inter].compare_exchange_weak(old, old + add)) {
                }
                ++contraction_calls[x.inter];
            }
        }
        ++pc;
    }
    if (fused) {
        const Phase timing(1);
        for (i64 group = 0; group < p.fuse_count[m.kind]; ++group)
            run_fused(p, m, job, group, code, tables, counts, bases, tile, w);
    }
    // Gathers in the order and arithmetic of itercc_permuted_gather_add.
    const Phase timing(4);
    const itercc_slot_group *groups = p.groups + p.key_group_start[p.slots[job.slot].key];
    double *sums = partial;
    for (i64 e = 0; e < m.part_count; ++e) {
        const itercc_slot_part &part = p.parts[m.part_start + e];
        const itercc_slot_group &g = groups[part.group];
        gather_part(p.terms[part.terms], g.coordinates, g.count, tile, sums);
        sums += g.count;
    }
}

// The partial sums of one tile (its gathers' rows).
i64 tile_rows(const itercc_slot_program &p, const itercc_slot_template &m, const itercc_slot_group *groups)
{
    i64 rows = 0;
    for (i64 e = 0; e < m.part_count; ++e) rows += groups[p.parts[m.part_start + e].group].count;
    return rows;
}

// A slot's rows: its tiles' partial sums in order, then the divisor.
void finish_slot(const itercc_slot_program &p, const itercc_slot_info &x, const Tile *tiles, i64 count,
                 const double *partial, double *result)
{
    const itercc_slot_group *groups = p.groups + p.key_group_start[x.key];
    double *out = result + x.start;
    for (i64 j = 0; j < count; ++j) {
        const itercc_slot_template &m = p.templates[tiles[j].tmpl];
        const double *sums = partial + tiles[j].partial;
        for (i64 e = 0; e < m.part_count; ++e) {
            const itercc_slot_group &g = groups[p.parts[m.part_start + e].group];
            for (i64 r = 0; r < g.count; ++r) out[g.rows[r]] += sums[r];
            sums += g.count;
        }
    }
    for (i64 r = 0; r < x.length; ++r) out[r] /= kDivisor;
}

// The tiles of slot s: template, occupied tuple, plan set and partial sums
// from `at` on. False when a tile has no plan.
bool slot_tiles(const itercc_slot_program &p, i64 s, i64 &at, std::vector<Tile> &tiles)
{
    const itercc_slot_info &x = p.slots[s];
    const itercc_slot_group *groups = p.groups + p.key_group_start[x.key];
    const i64 n = p.nirrep, codes = n * n * n * n;
    for (i64 t = p.class_start[x.cls]; t < p.class_start[x.cls] + p.class_count[x.cls]; ++t) {
        const itercc_slot_template &m = p.templates[t];
        Tile tile{s, t, at, 0, {0, 0, 0, 0}};
        i64 code = 0;
        for (int k = 0; k < 4; ++k) {
            tile.fixed[k] = x.occupied[m.positions[k]];
            code = code * n + p.occupied_irreps[tile.fixed[k]];
        }
        tile.plans = p.plan_id[m.kind * codes + code];
        if (tile.plans < 0) return false;
        at += tile_rows(p, m, groups);
        tiles.push_back(tile);
    }
    return true;
}

}  // namespace

extern "C" int itercc_triplet_slots(const itercc_slot_program *p, int64_t nslots, int64_t partial_limit,
                                    const double *const *bases, const double *ground,
                                    const double *direction, double *result, int64_t cache_bytes,
                                    int64_t *stats, int threads, int fused, int64_t share_work,
                                    void (*progress)(int64_t done, int64_t total))
{
    if (share_work <= 0) share_work = kShareWork;
    const i64 nv = p->nv, nv4 = nv * nv * nv * nv;
    threads = std::max(1, threads);
    std::vector<Tile> tiles;
    for (i64 s = 0; s < nslots; ++s) {   // every tile has its plans before any slot runs
        i64 at = 0;
        tiles.clear();
        if (!slot_tiles(*p, s, at, tiles)) return 1;
    }
    Cache cache(cache_bytes);
    const double *const vectors[2] = {ground, direction};
    std::vector<i64> first;   // each slot's first tile in the wave, then the end
    std::vector<i64> order;   // the wave's tiles in the order they run
    // Every partial sum is written by its tile before its slot reads it, so
    // the buffer is not cleared (a zero fill of a 1.5 GB wave on one thread
    // cost water cc-pVDZ about 0.8 s); its pages fault in on the threads.
    std::unique_ptr<double[]> partial;
    i64 capacity = 0;
    for (i64 s0 = 0; s0 < nslots;) {
        tiles.clear();
        first.clear();
        i64 used = 0, s1 = s0;
        while (s1 < nslots) {
            const size_t before = tiles.size();
            i64 at = used;
            slot_tiles(*p, s1, at, tiles);
            if (s1 > s0 && at > partial_limit) {
                tiles.resize(before);   // the next wave starts with this slot
                break;
            }
            first.push_back(i64(before));
            used = at;
            ++s1;
        }
        first.push_back(i64(tiles.size()));
        if (capacity < used) {
            partial.reset();
            partial.reset(new (std::nothrow) double[size_t(used)]);
            if (!partial) return 2;
            capacity = used;
        }
        // Tiles run by occupied tuple, T4 hole pair (positions 2 and 3) first:
        // the intermediates of g with T4 (one per hole pair, about 5 MB at CuF
        // size) and of the other operands are then used by neighbouring
        // tiles, while the cache still holds them. Only the order changes:
        // every slot adds its tiles' partial sums in their own order.
        order.resize(tiles.size());
        std::iota(order.begin(), order.end(), i64(0));
        static const bool sorted = [] {
            const char *value = std::getenv("ITERCC_SLOT_SORT");
            return value == nullptr || value[0] != '0';
        }();
        if (sorted)
        std::stable_sort(order.begin(), order.end(), [&tiles](i64 x, i64 y) {
            const int64_t *u = tiles[size_t(x)].fixed, *v = tiles[size_t(y)].fixed;
            return std::tie(u[2], u[3], u[0], u[1]) < std::tie(v[2], v[3], v[0], v[1]);
        });
        const i64 ntiles = i64(tiles.size()), nwave = s1 - s0;
        // Out of memory in a tile (an intermediate, a thread's scratch): the
        // keys it was producing are abandoned (their waiters produce them),
        // the remaining tiles are skipped, and the call returns 2.
        std::atomic<bool> failed{false};
#pragma omp parallel num_threads(threads)
        {
            Worker w;
            double *tile = nullptr;
            try {
                tile = itercc::scratch(4).get(size_t(nv4));
            } catch (const std::bad_alloc &) {
                failed = true;
            }
#pragma omp for schedule(dynamic, 1) nowait
            for (i64 j = 0; j < ntiles; ++j) {
                if (failed.load(std::memory_order_relaxed)) continue;
                const Tile &job = tiles[size_t(order[size_t(j)])];
                try {
                    run_tile(*p, job, bases, vectors, cache, cache_bytes, tile, w, partial.get() + job.partial,
                             fused != 0, threads, share_work);
                } catch (const std::bad_alloc &) {
                    const itercc_slot_template &m = p->templates[job.tmpl];
                    const itercc_slot_intermediate *inters = p->intermediates + p->inter_start[m.kind];
                    for (size_t k = 0; k < w.owner.size(); ++k)
                        if (w.owner[k]) cache.abandon(key_of(inters[k], job.fixed));
                    w.bound.clear();
                    w.owner.clear();
                    failed = true;
                }
            }
            {
                const Phase timing(8);
#pragma omp barrier
            }
            if (!failed.load()) {
#pragma omp for schedule(dynamic, 1)
                for (i64 s = 0; s < nwave; ++s) {
                    const Phase timing(6);
                    finish_slot(*p, p->slots[s0 + s], tiles.data() + first[size_t(s)],
                                first[size_t(s + 1)] - first[size_t(s)], partial.get(), result);
                }
            }
        }
        if (failed.load()) return 2;
        s0 = s1;
        if (progress != nullptr) progress(s0, nslots);   // between waves, on the calling thread
    }
    if (stats != nullptr) {
        stats[0] = cache.hits();
        stats[1] = cache.misses();
        stats[2] = cache.peak();
        stats[3] = cache.shared();
        stats[4] = cache.helped();
    }
    return 0;
}

extern "C" void itercc_triplet_profile(double *seconds, int reset)
{
    for (int i = 0; i < kPhases; ++i) {
        seconds[i] = phase_time[i].load();
        if (reset) phase_time[i] = 0.0;
    }
}

extern "C" void itercc_triplet_contraction_profile(int64_t count, double *seconds, int64_t *calls, int reset)
{
    for (i64 i = 0; i < std::min<i64>(count, kTimed); ++i) {
        seconds[i] = contraction_time[i].load();
        calls[i] = contraction_calls[i].load();
        if (reset) {
            contraction_time[i] = 0.0;
            contraction_calls[i] = 0;
        }
    }
}
