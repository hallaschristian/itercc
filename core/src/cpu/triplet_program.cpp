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
#include <cstdlib>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
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
// lookups, 6 slot sums, 7 intermediate allocation.
constexpr int kPhases = 8;
std::atomic<double> phase_time[kPhases];
constexpr i64 kTimed = 1 << 16;        // contractions timed one by one (instruction `inter`)
std::atomic<double> contraction_time[kTimed];
std::atomic<i64> contraction_calls[kTimed];
const bool profiling = [] {
    const char *value = std::getenv("ITERCC_SLOT_PROFILE");
    return value != nullptr && value[0] == '1';
}();

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

// Intermediates shared between tiles: one producer per key while the others
// wait; finished buffers stay while they fit the byte budget (least recently
// used out first). A buffer in use outlives its eviction.
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
            ready_.wait(lock);
        }
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
                bytes_ += bytes;
                peak_ = std::max(peak_, bytes_);
            }
        }
        ready_.notify_all();
    }

    i64 hits() const { return hits_; }
    i64 misses() const { return misses_; }
    i64 peak() const { return peak_; }

  private:
    struct Entry {
        Buffer buffer;
        bool ready = false;
        std::list<Key>::iterator place;
    };
    std::mutex mutex_;
    std::condition_variable ready_;
    std::unordered_map<Key, Entry, KeyHash> map_;
    std::list<Key> lru_;   // finished entries, most recent first
    i64 budget_, bytes_ = 0, peak_ = 0, hits_ = 0, misses_ = 0;
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
        itercc::fused_gemm({u.rows, 0, u.cols, i64(w.segments.size()), w.segments.data(),
                            tile + u.x->out.coef[0] + u.o_base, u.x->rpos + u.r0, u.x->cpos + u.c0});
        i = j;
    }
}

// One tile: its products, then its gathers into `partial`. With `fused`, the
// products of fusion groups wait until the other instructions are done (the
// intermediates they read stay bound) and run as fused GEMMs.
void run_tile(const itercc_slot_program &p, const Tile &job, const double *const *bases,
              const double *const *vectors, Cache &cache, i64 cacheable, double *tile, Worker &w,
              double *partial, bool fused)
{
    const i64 nv = p.nv, nv4 = nv * nv * nv * nv;
    const itercc_slot_template &m = p.templates[job.tmpl];
    {
        const Phase timing(0);
        std::fill(tile, tile + nv4, 0.0);
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
                Buffer hit;
                {
                    const Phase timing(5);
                    hit = cache.acquire(key_of(inters[k], job.fixed));
                }
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
            itercc::contract_serial(plan, reinterpret_cast<const int64_t *>(tables[x.plan]), count);
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
        const itercc_slot_terms &t = p.terms[part.terms];
        for (i64 r = 0; r < g.count; ++r) {
            const int32_t *c = g.coordinates + 4 * r;
            double sum = 0.;
            for (i64 k = 0; k < t.count; ++k) {
                const int64_t *step = t.offsets + 4 * k;
                const i64 at = c[0] * step[0] + c[1] * step[1] + c[2] * step[2] + c[3] * step[3];
                sum += t.weights[k] * tile[at];
            }
            sums[r] = sum;
        }
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
                                    int64_t *stats, int threads, int fused)
{
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
            partial.reset(new double[size_t(used)]);
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
#pragma omp parallel num_threads(threads)
        {
            Worker w;
            double *tile = itercc::scratch(4).get(size_t(nv4));
#pragma omp for schedule(dynamic, 1)
            for (i64 j = 0; j < ntiles; ++j) {
                const Tile &job = tiles[size_t(order[size_t(j)])];
                run_tile(*p, job, bases, vectors, cache, cache_bytes, tile, w, partial.get() + job.partial,
                         fused != 0);
            }
#pragma omp for schedule(dynamic, 1)
            for (i64 s = 0; s < nwave; ++s) {
                const Phase timing(6);
                finish_slot(*p, p->slots[s0 + s], tiles.data() + first[size_t(s)],
                            first[size_t(s + 1)] - first[size_t(s)], partial.get(), result);
            }
        }
        s0 = s1;
    }
    if (stats != nullptr) {
        stats[0] = cache.hits();
        stats[1] = cache.misses();
        stats[2] = cache.peak();
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
