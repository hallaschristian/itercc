// Heat-bath selected configuration interaction: determinant selection,
// the sparse variational Hamiltonian, Epstein-Nesbet second-order energies,
// the one-particle density matrix and <S^2>.
//
// A determinant is an alpha and a beta occupation string of W 64-bit words
// each (orbital p is bit p % 64 of word p / 64), with the alpha creators
// ordered before the beta creators and each string in increasing orbital
// order. Double excitations come from lists of integrals sorted by
// magnitude (Holmes, Tubman, Umrigar, JCTC 12, 3674 (2016)): for each pair
// of occupied orbitals, a scan stops at the first integral below the
// threshold. Sums whose terms arrive in a thread-dependent order (PT2
// numerators, the density matrix, <S^2>) accumulate in 128-bit fixed point,
// so every result is the same for any thread count.
#include "itercc/core.h"

#include <omp.h>
#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <numeric>
#include <type_traits>
#include <vector>

namespace {

using u64 = uint64_t;
using i128 = __int128;

constexpr double FIXED_SCALE = 1208925819614629174706176.0;   // 2^80
constexpr double NUMERATOR_SCALE = 4503599627370496.0;          // 2^52 for int64 PT2 numerators
inline int64_t to_numerator(double x) { return static_cast<int64_t>(x * NUMERATOR_SCALE); }
inline double from_numerator(int64_t v) { return static_cast<double>(v) / NUMERATOR_SCALE; }
inline i128 to_fixed(double x) { return static_cast<i128>(x * FIXED_SCALE); }
inline double from_fixed(i128 v) { return static_cast<double>(v) / FIXED_SCALE; }

inline int popc(u64 x) { return __builtin_popcountll(x); }
inline int lowest(u64 x) { return __builtin_ctzll(x); }

inline u64 mix(u64 x)
{
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

template <int W> struct Det {
    u64 w[2 * W];            // alpha words, then beta words
    const u64 *a() const { return w; }
    const u64 *b() const { return w + W; }
    u64 *a() { return w; }
    u64 *b() { return w + W; }
};

template <int W> inline bool same(const Det<W> &x, const Det<W> &y)
{
    for (int k = 0; k < 2 * W; ++k)
        if (x.w[k] != y.w[k]) return false;
    return true;
}

template <int W> inline bool before(const Det<W> &x, const Det<W> &y)
{
    for (int k = 0; k < 2 * W; ++k)
        if (x.w[k] != y.w[k]) return x.w[k] < y.w[k];
    return false;
}

// Zobrist keys: a determinant's hash is the XOR of one random key per
// occupied spin orbital, so an excitation updates it with two or four XORs
// (excitations() filters PT2 batches by hash before forming a determinant).
struct Zobrist {
    u64 key[2][256];
    Zobrist()
    {
        u64 state = 0x2545f4914f6cdd1dULL;
        for (int s = 0; s < 2; ++s)
            for (int p = 0; p < 256; ++p) {
                state += 0x9e3779b97f4a7c15ULL;
                key[s][p] = mix(state);
            }
    }
};
inline const Zobrist &zobrist()
{
    static const Zobrist table;
    return table;
}

template <int W> inline u64 hash_det(const Det<W> &x)
{
    const Zobrist &z = zobrist();
    u64 h = 0;
    for (int s = 0; s < 2; ++s)
        for (int k = 0; k < W; ++k)
            for (u64 bits = x.w[s * W + k]; bits; bits &= bits - 1) h ^= z.key[s][(k << 6) + lowest(bits)];
    return h;
}

template <int W> inline u64 hash_string(const u64 *s)
{
    u64 h = 0x51afd7ed558ccd25ULL;
    for (int k = 0; k < W; ++k) h = mix(h ^ (s[k] + 0x632be59bd9b4e019ULL * (k + 1)));
    return h;
}

inline bool test(const u64 *s, int p) { return (s[p >> 6] >> (p & 63)) & 1ULL; }
inline void flip(u64 *s, int p) { s[p >> 6] ^= 1ULL << (p & 63); }

// Number of set bits of s at positions [lo, hi).
template <int W> inline int count_range(const u64 *s, int lo, int hi)
{
    if (hi <= lo) return 0;
    int wl = lo >> 6, wh = hi >> 6;
    u64 low_mask = ~((1ULL << (lo & 63)) - 1);
    if (wl == wh) return popc(s[wl] & low_mask & ((1ULL << (hi & 63)) - 1));
    int n = popc(s[wl] & low_mask);
    for (int k = wl + 1; k < wh; ++k) n += popc(s[k]);
    if ((hi & 63) && wh < W) n += popc(s[wh] & ((1ULL << (hi & 63)) - 1));
    return n;
}

// Sign of a+_r a_p on string s (p occupied, r empty, p != r).
template <int W> inline double phase(const u64 *s, int p, int r)
{
    int lo = p < r ? p : r, hi = p < r ? r : p;
    return (count_range<W>(s, lo + 1, hi) & 1) ? -1.0 : 1.0;
}

template <int W> inline int occupied(const u64 *s, int *out)
{
    int n = 0;
    for (int k = 0; k < W; ++k)
        for (u64 x = s[k]; x; x &= x - 1) out[n++] = (k << 6) + lowest(x);
    return n;
}

constexpr int MAX_ORB = 256;

struct Entry {               // one integral of a heat-bath list
    double v;
    uint16_t r, s;
};

struct SinglesEntry {
    double bound;
    int32_t r;
};

}  // namespace

struct itercc_sci_context {
    int64_t n, na, nb;
    int words;
    std::vector<double> h, eri;      // eri[((p n + q) n + r) n + s] = (pq|rs)
    std::vector<int64_t> irrep;
    std::vector<double> J, K;        // J[p n + q] = (pp|qq), K[p n + q] = (pq|qp)
    std::vector<double> Dk, Xk;      // Dk[(p n + r) n + k] = (pr|kk), Xk = (pk|kr)
    std::vector<int64_t> os_start;   // opposite spin: pair p n + q -> entries over (r, s)
    std::vector<Entry> os;
    std::vector<int64_t> ss_start;   // same spin: pair p n + q (p < q) -> entries r < s
    std::vector<Entry> ss;
    std::vector<int64_t> single_start;
    std::vector<SinglesEntry> singles;

    double E(int64_t p, int64_t q, int64_t r, int64_t s) const
    {
        return eri[((p * n + q) * n + r) * n + s];
    }
};

namespace {

using Ctx = itercc_sci_context;

// <x| H |x> without the constant.
inline double diagonal_energy(const Ctx &c, const int *oa, int na, const int *ob, int nb)
{
    const int64_t n = c.n;
    double e = 0.0;
    for (int i = 0; i < na; ++i) {
        int p = oa[i];
        e += c.h[p * n + p];
        for (int j = 0; j < i; ++j) e += c.J[p * n + oa[j]] - c.K[p * n + oa[j]];
        for (int j = 0; j < nb; ++j) e += c.J[p * n + ob[j]];
    }
    for (int i = 0; i < nb; ++i) {
        int p = ob[i];
        e += c.h[p * n + p];
        for (int j = 0; j < i; ++j) e += c.J[p * n + ob[j]] - c.K[p * n + ob[j]];
    }
    return e;
}

// <x'| H |x> for the excitation p -> r in the string whose occupied orbitals
// are `same` (the other spin's are `other`), without the phase.
inline double single_value(const Ctx &c, int p, int r, const int *same, int ns, const int *other, int no)
{
    const int64_t n = c.n;
    const double *D = &c.Dk[(p * n + r) * n];
    const double *X = &c.Xk[(p * n + r) * n];
    double v = c.h[p * n + r];
    for (int i = 0; i < ns; ++i) v += D[same[i]] - X[same[i]];
    for (int i = 0; i < no; ++i) v += D[other[i]];
    return v;
}

template <int W> inline int set_bits(const u64 *m, int *out, int limit)
{
    int n = 0;
    for (int k = 0; k < W && n < limit; ++k)
        for (u64 x = m[k]; x && n < limit; x &= x - 1) out[n++] = (k << 6) + lowest(x);
    return n;
}

// <y| H |x> for distinct determinants, zero beyond double excitations.
// occ (may be null): x's occupied alpha orbitals (na of them), then beta.
template <int W> double element(const Ctx &c, const Det<W> &x, const Det<W> &y, const int *occ = nullptr,
                                int occ_na = 0, int occ_nb = 0)
{
    u64 da[W], db[W];
    int ca = 0, cb = 0;
    for (int k = 0; k < W; ++k) {
        da[k] = x.a()[k] ^ y.a()[k];
        db[k] = x.b()[k] ^ y.b()[k];
        ca += popc(da[k]);
        cb += popc(db[k]);
    }
    if (ca + cb > 4 || ca + cb == 0) return 0.0;
    u64 hole[W], part[W];
    int holes[2], parts[2];
    if (cb == 0 || ca == 0) {
        const u64 *xs = ca ? x.a() : x.b(), *ys = ca ? y.a() : y.b(), *d = ca ? da : db;
        for (int k = 0; k < W; ++k) {
            hole[k] = xs[k] & d[k];
            part[k] = ys[k] & d[k];
        }
        set_bits<W>(hole, holes, 2);
        set_bits<W>(part, parts, 2);
        if (ca + cb == 2) {
            const double sign = phase<W>(xs, holes[0], parts[0]);
            if (occ) {
                const int *same = ca ? occ : occ + occ_na, *other = ca ? occ + occ_na : occ;
                const int ns = ca ? occ_na : occ_nb, no = ca ? occ_nb : occ_na;
                return sign * single_value(c, holes[0], parts[0], same, ns, other, no);
            }
            int oa[MAX_ORB], ob[MAX_ORB];
            int na = occupied<W>(xs, oa), nb = occupied<W>(ca ? x.b() : x.a(), ob);
            return sign * single_value(c, holes[0], parts[0], oa, na, ob, nb);
        }
        int p = holes[0], q = holes[1], r = parts[0], s = parts[1];
        u64 t[W];
        std::memcpy(t, xs, sizeof(t));
        double sign = phase<W>(t, p, r);
        flip(t, p);
        flip(t, r);
        sign *= phase<W>(t, q, s);
        return sign * (c.E(p, r, q, s) - c.E(p, s, q, r));
    }
    if (ca != 2 || cb != 2) return 0.0;
    int p = 0, r = 0, q = 0, s = 0;
    for (int k = 0; k < W; ++k) {
        hole[k] = x.a()[k] & da[k];
        part[k] = y.a()[k] & da[k];
    }
    set_bits<W>(hole, &p, 1);
    set_bits<W>(part, &r, 1);
    for (int k = 0; k < W; ++k) {
        hole[k] = x.b()[k] & db[k];
        part[k] = y.b()[k] & db[k];
    }
    set_bits<W>(hole, &q, 1);
    set_bits<W>(part, &s, 1);
    return phase<W>(x.a(), p, r) * phase<W>(x.b(), q, s) * c.E(p, r, q, s);
}

// Every determinant y connected to x with |<y|H|x>| >= threshold:
// emit(y, value, hash of y), for the y whose hash passes accept(hash) (tested
// before y and its matrix element are formed). hx is x's hash.
template <int W, class Accept, class Emit>
void excitations(const Ctx &c, const Det<W> &x, u64 hx, double threshold, Accept &&accept, Emit &&emit)
{
    const int64_t n = c.n;
    const u64(&z)[2][256] = zobrist().key;
    int oa[MAX_ORB], ob[MAX_ORB];
    const int na = occupied<W>(x.a(), oa), nb = occupied<W>(x.b(), ob);
    Det<W> y;
    // Opposite-spin doubles: alpha p -> r, beta q -> s, value (pr|qs).
    for (int i = 0; i < na; ++i)
        for (int j = 0; j < nb; ++j) {
            const int p = oa[i], q = ob[j];
            const u64 hpq = hx ^ z[0][p] ^ z[1][q];
            const int64_t pair = p * n + q;
            for (int64_t e = c.os_start[pair]; e < c.os_start[pair + 1]; ++e) {
                const Entry &en = c.os[e];
                if (std::fabs(en.v) < threshold) break;
                if (test(x.a(), en.r) || test(x.b(), en.s)) continue;
                const u64 hy = hpq ^ z[0][en.r] ^ z[1][en.s];
                if (!accept(hy)) continue;
                y = x;
                flip(y.a(), p);
                flip(y.a(), en.r);
                flip(y.b(), q);
                flip(y.b(), en.s);
                emit(y, phase<W>(x.a(), p, en.r) * phase<W>(x.b(), q, en.s) * en.v, hy);
            }
        }
    // Same-spin doubles p < q -> r < s, value (pr|qs) - (ps|qr).
    for (int spin = 0; spin < 2; ++spin) {
        const int *occ = spin ? ob : oa;
        const int no = spin ? nb : na;
        const u64 *xs = spin ? x.b() : x.a();
        const u64 *zs = z[spin];
        for (int i = 0; i < no; ++i)
            for (int j = i + 1; j < no; ++j) {
                const int p = occ[i], q = occ[j];
                const u64 hpq = hx ^ zs[p] ^ zs[q];
                const int64_t pair = p * n + q;
                for (int64_t e = c.ss_start[pair]; e < c.ss_start[pair + 1]; ++e) {
                    const Entry &en = c.ss[e];
                    if (std::fabs(en.v) < threshold) break;
                    if (test(xs, en.r) || test(xs, en.s)) continue;
                    const u64 hy = hpq ^ zs[en.r] ^ zs[en.s];
                    if (!accept(hy)) continue;
                    y = x;
                    u64 *ys = spin ? y.b() : y.a();
                    double sign = phase<W>(ys, p, en.r);
                    flip(ys, p);
                    flip(ys, en.r);
                    sign *= phase<W>(ys, q, en.s);
                    flip(ys, q);
                    flip(ys, en.s);
                    emit(y, sign * en.v, hy);
                }
            }
    }
    // Singles, screened by an upper bound of |<y|H|x>| sorted per hole.
    for (int spin = 0; spin < 2; ++spin) {
        const int *occ = spin ? ob : oa, *other = spin ? oa : ob;
        const int no = spin ? nb : na, nother = spin ? na : nb;
        const u64 *xs = spin ? x.b() : x.a();
        const u64 *zs = z[spin];
        for (int i = 0; i < no; ++i) {
            const int p = occ[i];
            for (int64_t e = c.single_start[p]; e < c.single_start[p + 1]; ++e) {
                const SinglesEntry &en = c.singles[e];
                if (en.bound < threshold) break;
                if (test(xs, en.r)) continue;
                const u64 hy = hx ^ zs[p] ^ zs[en.r];
                if (!accept(hy)) continue;
                double v = phase<W>(xs, p, en.r) * single_value(c, p, en.r, occ, no, other, nother);
                if (std::fabs(v) < threshold) continue;
                y = x;
                u64 *ys = spin ? y.b() : y.a();
                flip(ys, p);
                flip(ys, en.r);
                emit(y, v, hy);
            }
        }
    }
}

// Open-addressing index of a fixed determinant array (indices, -1 empty).
template <int W> struct DetIndex {
    const Det<W> *dets = nullptr;
    std::vector<int64_t> slot;
    u64 mask = 0;

    void build(const Det<W> *d, int64_t count, int threads)
    {
        dets = d;
        u64 cap = 16;
        while (cap < static_cast<u64>(2 * count + 16)) cap <<= 1;
        mask = cap - 1;
        slot.assign(cap, -1);
#pragma omp parallel for schedule(static) num_threads(threads)
        for (int64_t i = 0; i < count; ++i) {
            u64 k = hash_det<W>(d[i]) & mask;
            for (;;) {
                int64_t expected = -1;
                if (__atomic_compare_exchange_n(&slot[k], &expected, i, false, __ATOMIC_RELAXED,
                                                __ATOMIC_RELAXED))
                    break;
                k = (k + 1) & mask;
            }
        }
    }

    int64_t find(const Det<W> &x, u64 h) const
    {
        for (u64 k = h & mask;; k = (k + 1) & mask) {
            const int64_t i = slot[k];
            if (i < 0) return -1;
            if (same(dets[i], x)) return i;
        }
    }
};

struct Spinlock {
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
    void lock()
    {
        while (flag.test_and_set(std::memory_order_acquire)) __builtin_ia32_pause();
    }
    void unlock() { flag.clear(std::memory_order_release); }
};

// A determinant hash map split into independently locked shards, with
// `nvalues` fixed-point accumulators of type V per key (zero: a set). Each
// shard is one array of entries (the key's 2W words, then the accumulators),
// so an insert touches one cache line or two; an empty entry has its first
// word all ones (no determinant has its first 64 alpha orbitals occupied:
// itercc_sci_create needs fewer than 64 alpha electrons).
template <int W, class V = i128> struct ShardedMap {
    static constexpr u64 EMPTY = ~0ULL;
    struct Shard {
        Spinlock lock;
        u64 *table = nullptr;
        size_t cap = 0, count = 0;
    };
    int nvalues;
    size_t stride;                       // words per entry, even (16-byte aligned accumulators)
    std::vector<Shard> shards;
    static constexpr int SHARD_BITS = 9;

    explicit ShardedMap(int nv)
        : nvalues(nv), stride((2 * W + nv * sizeof(V) / sizeof(u64) + 1) & ~size_t(1)),
          shards(size_t(1) << SHARD_BITS)
    {
    }
    ~ShardedMap() { clear(); }
    ShardedMap(const ShardedMap &) = delete;
    ShardedMap &operator=(const ShardedMap &) = delete;

    static size_t shard_of(u64 h) { return h >> (64 - SHARD_BITS); }
    V *values_of(u64 *row) const { return reinterpret_cast<V *>(row + 2 * W); }
    static const Det<W> &key_of(const u64 *row) { return *reinterpret_cast<const Det<W> *>(row); }

    // Tables of 2 MB or more are 2 MB aligned and ask for transparent huge
    // pages: random inserts over gigabytes of 4 KB pages miss the TLB.
    u64 *allocate(size_t cap) const
    {
        constexpr size_t HUGE_PAGE = size_t(1) << 21;
        size_t bytes = cap * stride * sizeof(u64);
        const size_t align = bytes >= HUGE_PAGE ? HUGE_PAGE : 64;
        bytes = (bytes + align - 1) & ~(align - 1);
        u64 *t = static_cast<u64 *>(std::aligned_alloc(align, bytes));
        if (!t) throw std::bad_alloc();
        if (align == HUGE_PAGE) madvise(t, bytes, MADV_HUGEPAGE);
        for (size_t k = 0; k < cap; ++k) t[k * stride] = EMPTY;
        return t;
    }

    void resize(Shard &s, size_t cap)
    {
        u64 *table = allocate(cap);
        const u64 mask = cap - 1;
        for (size_t i = 0; i < s.cap; ++i) {
            const u64 *row = s.table + i * stride;
            if (row[0] == EMPTY) continue;
            u64 k = hash_det<W>(key_of(row)) & mask;
            while (table[k * stride] != EMPTY) k = (k + 1) & mask;
            std::memcpy(table + k * stride, row, stride * sizeof(u64));
        }
        std::free(s.table);
        s.table = table;
        s.cap = cap;
    }

    // Adds `add` (nvalues entries, may be null for a set) to key x's
    // accumulators; the caller holds the shard's lock.
    void insert_locked(Shard &s, const Det<W> &x, u64 h, const V *add)
    {
        if (2 * (s.count + 1) > s.cap) resize(s, s.cap ? 2 * s.cap : 64);
        const u64 mask = s.cap - 1;
        u64 k = h & mask;
        u64 *row = s.table + k * stride;
        while (row[0] != EMPTY && !same(key_of(row), x)) {
            k = (k + 1) & mask;
            row = s.table + k * stride;
        }
        V *vals = values_of(row);
        if (row[0] == EMPTY) {
            std::memcpy(row, x.w, sizeof(x.w));
            for (int v = 0; v < nvalues; ++v) vals[v] = 0;
            ++s.count;
        }
        for (int v = 0; v < nvalues; ++v) vals[v] += add[v];
    }

    void add(const Det<W> &x, u64 h, const V *add)
    {
        Shard &s = shards[shard_of(h)];
        s.lock.lock();
        insert_locked(s, x, h, add);
        s.lock.unlock();
    }

    // f(key, accumulators) for every entry of shard s.
    template <class F> void for_each(size_t s, F &&f) const
    {
        const Shard &sh = shards[s];
        for (size_t k = 0; k < sh.cap; ++k) {
            u64 *row = sh.table + k * stride;
            if (row[0] != EMPTY) f(key_of(row), values_of(row));
        }
    }

    // Per-thread staging: records grouped by shard and inserted DEPTH at a
    // time under one lock acquisition (the lock cache line moves between
    // cores once per DEPTH inserts), each flush prefetching its records'
    // entries before inserting them. flush_all() before the map is read.
    struct Stage {
        static constexpr int DEPTH = 64;
        ShardedMap &map;
        std::vector<Det<W>> keys;
        std::vector<u64> hashes;
        std::vector<V> values;
        std::vector<uint8_t> fill;

        explicit Stage(ShardedMap &m)
            : map(m), keys(m.shards.size() * DEPTH), hashes(m.shards.size() * DEPTH),
              values(m.shards.size() * DEPTH * m.nvalues), fill(m.shards.size(), 0)
        {
        }

        void add(const Det<W> &x, u64 h, const V *add)
        {
            const size_t s = shard_of(h), slot = s * DEPTH + fill[s];
            keys[slot] = x;
            hashes[slot] = h;
            for (int v = 0; v < map.nvalues; ++v) values[slot * map.nvalues + v] = add[v];
            if (++fill[s] == DEPTH) flush(s);
        }

        void flush(size_t s)
        {
            const int f = fill[s];
            if (!f) return;
            Shard &sh = map.shards[s];
            sh.lock.lock();
            if (sh.cap) {
                const u64 mask = sh.cap - 1;
                for (int t = 0; t < f; ++t)
                    __builtin_prefetch(sh.table + (hashes[s * DEPTH + t] & mask) * map.stride, 1);
            }
            for (int t = 0; t < f; ++t) {
                const size_t slot = s * DEPTH + t;
                map.insert_locked(sh, keys[slot], hashes[slot], &values[slot * map.nvalues]);
            }
            sh.lock.unlock();
            fill[s] = 0;
        }

        void flush_all()
        {
            for (size_t s = 0; s < fill.size(); ++s) flush(s);
        }
    };

    size_t size() const
    {
        size_t total = 0;
        for (const Shard &s : shards) total += s.count;
        return total;
    }

    void clear()
    {
        for (Shard &s : shards) {
            std::free(s.table);
            s.table = nullptr;
            s.cap = s.count = 0;
        }
    }
};

struct SpaceBase {
    virtual ~SpaceBase() = default;
};

template <int W> struct Space : SpaceBase {
    const Ctx *ctx;
    std::vector<Det<W>> dets;
    DetIndex<W> index;
    // Unique alpha and beta strings (sorted), each determinant's string ids,
    // the determinants of each string (increasing index) and the strings one
    // single excitation away from each string.
    bool strings = false;
    std::vector<u64> astr, bstr;
    std::vector<int32_t> aid, bid;
    std::vector<int64_t> amajor_start, bmajor_start, asingle_start, bsingle_start;
    std::vector<int32_t> amajor, bmajor, asingle, bsingle;

    Space(const Ctx *c, const u64 *data, int64_t count, int threads) : ctx(c), dets(count)
    {
        std::memcpy(dets.data(), data, sizeof(Det<W>) * count);
        index.build(dets.data(), count, threads);
    }

    void unique_strings(int spin, std::vector<u64> &strs, std::vector<int32_t> &ids, int threads)
    {
        const int64_t count = dets.size();
        std::vector<int64_t> order(count);
        std::iota(order.begin(), order.end(), 0);
        auto key = [&](int64_t i) { return dets[i].w + spin * W; };
        std::sort(order.begin(), order.end(), [&](int64_t i, int64_t j) {
            const u64 *x = key(i), *y = key(j);
            for (int k = 0; k < W; ++k)
                if (x[k] != y[k]) return x[k] < y[k];
            return i < j;
        });
        ids.assign(count, 0);
        strs.clear();
        for (int64_t t = 0; t < count; ++t) {
            const u64 *x = key(order[t]);
            if (t == 0 || std::memcmp(x, key(order[t - 1]), sizeof(u64) * W) != 0)
                strs.insert(strs.end(), x, x + W);
            ids[order[t]] = static_cast<int32_t>(strs.size() / W - 1);
        }
        (void)threads;
    }

    static void group(const std::vector<int32_t> &ids, int64_t nstr, std::vector<int64_t> &start,
                      std::vector<int32_t> &members)
    {
        start.assign(nstr + 1, 0);
        for (int32_t id : ids) ++start[id + 1];
        for (int64_t s = 0; s < nstr; ++s) start[s + 1] += start[s];
        members.resize(ids.size());
        std::vector<int64_t> fill(start.begin(), start.end() - 1);
        for (int64_t i = 0; i < static_cast<int64_t>(ids.size()); ++i) members[fill[ids[i]]++] = static_cast<int32_t>(i);
    }

    // Strings one single excitation from each string, among the unique strings.
    void singles_graph(const std::vector<u64> &strs, std::vector<int64_t> &start, std::vector<int32_t> &out,
                       int threads)
    {
        const int64_t nstr = strs.size() / W;
        u64 cap = 16;
        while (cap < static_cast<u64>(2 * nstr + 16)) cap <<= 1;
        const u64 mask = cap - 1;
        std::vector<int64_t> table(cap, -1);
        for (int64_t s = 0; s < nstr; ++s) {
            u64 k = hash_string<W>(&strs[s * W]) & mask;
            while (table[k] >= 0) k = (k + 1) & mask;
            table[k] = s;
        }
        const int n = static_cast<int>(ctx->n);
        std::vector<std::vector<int32_t>> lists(nstr);
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads)
        for (int64_t s = 0; s < nstr; ++s) {
            const u64 *x = &strs[s * W];
            u64 t[W];
            int occ[MAX_ORB];
            const int no = occupied<W>(x, occ);
            for (int i = 0; i < no; ++i)
                for (int r = 0; r < n; ++r) {
                    // Any irrep: the alpha half of an opposite-spin double may
                    // change it when the beta half compensates.
                    if (test(x, r)) continue;
                    std::memcpy(t, x, sizeof(t));
                    flip(t, occ[i]);
                    flip(t, r);
                    for (u64 k = hash_string<W>(t) & mask;; k = (k + 1) & mask) {
                        const int64_t m = table[k];
                        if (m < 0) break;
                        if (std::memcmp(&strs[m * W], t, sizeof(t)) == 0) {
                            lists[s].push_back(static_cast<int32_t>(m));
                            break;
                        }
                    }
                }
            std::sort(lists[s].begin(), lists[s].end());
        }
        start.assign(nstr + 1, 0);
        for (int64_t s = 0; s < nstr; ++s) start[s + 1] = start[s] + lists[s].size();
        out.resize(start[nstr]);
        for (int64_t s = 0; s < nstr; ++s) std::copy(lists[s].begin(), lists[s].end(), out.begin() + start[s]);
    }

    void build_strings(int threads)
    {
        if (strings) return;
        unique_strings(0, astr, aid, threads);
        unique_strings(1, bstr, bid, threads);
        group(aid, astr.size() / W, amajor_start, amajor);
        group(bid, bstr.size() / W, bmajor_start, bmajor);
        singles_graph(astr, asingle_start, asingle, threads);
        singles_graph(bstr, bsingle_start, bsingle, threads);
        strings = true;
    }

    template <class F> void for_each_alpha_excited(int64_t i, F &&f) const
    {
        // Same alpha string: beta differs (single or double).
        const int32_t a = aid[i];
        for (int64_t t = amajor_start[a]; t < amajor_start[a + 1]; ++t) {
            const int32_t j = amajor[t];
            if (j == i) continue;
            int d = 0;
            for (int k = 0; k < W; ++k) d += popc(dets[i].b()[k] ^ dets[j].b()[k]);
            if (d <= 4) f(j, 0, d);
        }
    }

    // Calls f(j, alpha difference bits, beta difference bits) for every
    // determinant j != i connected to i by at most a double excitation.
    template <class F> void for_each_connected(int64_t i, F &&f) const
    {
        const Det<W> &x = dets[i];
        for_each_alpha_excited(i, f);
        const int32_t b = bid[i];
        for (int64_t t = bmajor_start[b]; t < bmajor_start[b + 1]; ++t) {
            const int32_t j = bmajor[t];
            if (j == i) continue;
            int d = 0;
            for (int k = 0; k < W; ++k) d += popc(x.a()[k] ^ dets[j].a()[k]);
            if (d <= 4) f(j, d, 0);
        }
        const int32_t a = aid[i];
        for (int64_t u = asingle_start[a]; u < asingle_start[a + 1]; ++u) {
            const int32_t a2 = asingle[u];
            for (int64_t t = amajor_start[a2]; t < amajor_start[a2 + 1]; ++t) {
                const int32_t j = amajor[t];
                int d = 0;
                for (int k = 0; k < W; ++k) d += popc(x.b()[k] ^ dets[j].b()[k]);
                if (d == 2) f(j, 2, 2);
            }
        }
    }
};

// The upper triangle (columns j > i, increasing) and the diagonal. The
// product runs over STRIPES fixed row ranges of about equal nonzeros: each
// stripe writes its rows' U x part directly and its U^T x part into its own
// buffer (entries from the stripe's first row on), and the buffers are added
// in stripe order, so the result does not depend on the thread count.
template <int W, class F> void for_each_upper(const Space<W> &sp, int64_t i, int64_t from, F &&f)
{
    // Every determinant j >= from (> i) connected to i: the per-string lists
    // hold increasing determinant indices, so each scan starts at `from`.
    const Det<W> &x = sp.dets[i];
    auto after = [&](const std::vector<int32_t> &list, int64_t lo, int64_t hi) {
        return std::lower_bound(list.begin() + lo, list.begin() + hi, static_cast<int32_t>(from)) - list.begin();
    };
    const int32_t a = sp.aid[i], b = sp.bid[i];
    for (int64_t t = after(sp.amajor, sp.amajor_start[a], sp.amajor_start[a + 1]); t < sp.amajor_start[a + 1]; ++t) {
        const int32_t j = sp.amajor[t];
        int d = 0;
        for (int k = 0; k < W; ++k) d += popc(x.b()[k] ^ sp.dets[j].b()[k]);
        if (d <= 4) f(j, 0, d);
    }
    for (int64_t t = after(sp.bmajor, sp.bmajor_start[b], sp.bmajor_start[b + 1]); t < sp.bmajor_start[b + 1]; ++t) {
        const int32_t j = sp.bmajor[t];
        int d = 0;
        for (int k = 0; k < W; ++k) d += popc(x.a()[k] ^ sp.dets[j].a()[k]);
        if (d <= 4) f(j, d, 0);
    }
    for (int64_t u = sp.asingle_start[a]; u < sp.asingle_start[a + 1]; ++u) {
        const int32_t a2 = sp.asingle[u];
        for (int64_t t = after(sp.amajor, sp.amajor_start[a2], sp.amajor_start[a2 + 1]); t < sp.amajor_start[a2 + 1];
             ++t) {
            const int32_t j = sp.amajor[t];
            int d = 0;
            for (int k = 0; k < W; ++k) d += popc(x.b()[k] ^ sp.dets[j].b()[k]);
            if (d == 2) f(j, 2, 2);
        }
    }
}

// Rows are kept in chunks of CHUNK rows, each with its own arrays (no copy
// into one array after the build, so a build peaks at the matrix's size), and
// an extension releases each old chunk once its rows are copied.
struct Hamiltonian {
    static constexpr int64_t STRIPES = 64;
    static constexpr int64_t CHUNK = 512;
    int64_t n = 0, nnz = 0;
    std::vector<double> diag;
    std::vector<std::vector<int64_t>> start;   // per chunk: CHUNK + 1 (or fewer) local row offsets
    std::vector<std::vector<int32_t>> col;
    std::vector<std::vector<double>> val;
    std::vector<int64_t> stripe;               // STRIPES + 1 row boundaries
    std::vector<int64_t> offset;               // each stripe's buffer offset
    std::vector<double> buffer;

    int64_t row_length(int64_t i) const
    {
        const auto &st = start[i / CHUNK];
        return st[i % CHUNK + 1] - st[i % CHUNK];
    }
};

struct Buffer {
    int64_t rows = 0, width = 0;
    std::vector<u64> data;
};

template <int W> Space<W> *as_space(void *p) { return static_cast<Space<W> *>(static_cast<SpaceBase *>(p)); }

template <int W> void diagonal_impl(Space<W> &sp, double *out, int threads)
{
    const Ctx &c = *sp.ctx;
    const int64_t count = sp.dets.size();
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t i = 0; i < count; ++i) {
        int oa[MAX_ORB], ob[MAX_ORB];
        int na = occupied<W>(sp.dets[i].a(), oa), nb = occupied<W>(sp.dets[i].b(), ob);
        out[i] = diagonal_energy(c, oa, na, ob, nb);
    }
}

// The Hamiltonian of the space; with `old` (the Hamiltonian of its first
// n_old determinants, unchanged), old rows keep their elements and gain
// only the columns of the new determinants, so only elements that involve a
// new determinant are evaluated.
template <int W>
Hamiltonian *hamiltonian_impl(Space<W> &sp, double drop, int threads, const Hamiltonian *old = nullptr,
                              int64_t n_old = 0)
{
    sp.build_strings(threads);
    const Ctx &c = *sp.ctx;
    const int64_t count = sp.dets.size();
    if (!old) n_old = 0;
    auto *H = new Hamiltonian;
    H->n = count;
    H->diag.resize(count);
    if (old) {
        std::copy(old->diag.begin(), old->diag.begin() + n_old, H->diag.begin());
#pragma omp parallel for schedule(static) num_threads(threads)
        for (int64_t i = n_old; i < count; ++i) {
            int oa[MAX_ORB], ob[MAX_ORB];
            int na = occupied<W>(sp.dets[i].a(), oa), nb = occupied<W>(sp.dets[i].b(), ob);
            H->diag[i] = diagonal_energy(c, oa, na, ob, nb);
        }
    } else {
        diagonal_impl(sp, H->diag.data(), threads);
    }
    const int64_t chunk = Hamiltonian::CHUNK;
    const int64_t nchunks = (count + chunk - 1) / chunk;
    H->start.resize(nchunks);
    H->col.resize(nchunks);
    H->val.resize(nchunks);
    // The extension reads old chunks and releases each one after its rows are
    // copied (old rows are a prefix, chunked the same way).
    auto *old_mut = const_cast<Hamiltonian *>(old);
#pragma omp parallel num_threads(threads)
    {
        std::vector<std::pair<int32_t, double>> row;
#pragma omp for schedule(dynamic, 1)
        for (int64_t ch = 0; ch < nchunks; ++ch) {
            const int64_t lo = ch * chunk, hi = std::min(count, lo + chunk);
            auto &st = H->start[ch];
            auto &cc = H->col[ch];
            auto &cv = H->val[ch];
            st.assign(hi - lo + 1, 0);
            for (int64_t i = lo; i < hi; ++i) {
                row.clear();
                int occ[2 * MAX_ORB];
                const int na = occupied<W>(sp.dets[i].a(), occ);
                const int nb = occupied<W>(sp.dets[i].b(), occ + na);
                const bool kept = i < n_old;
                for_each_upper(sp, i, kept ? n_old : i + 1, [&](int32_t j, int, int) {
                    const double v = element<W>(c, sp.dets[i], sp.dets[j], occ, na, nb);
                    if (std::fabs(v) > drop) row.emplace_back(j, v);
                });
                std::sort(row.begin(), row.end(),
                          [](const std::pair<int32_t, double> &x, const std::pair<int32_t, double> &y) {
                              return x.first < y.first;
                          });
                if (kept) {         // the old row's columns all precede the new ones
                    const auto &os = old->start[ch];
                    const int64_t a = os[i - lo], b = os[i - lo + 1];
                    cc.insert(cc.end(), old->col[ch].begin() + a, old->col[ch].begin() + b);
                    cv.insert(cv.end(), old->val[ch].begin() + a, old->val[ch].begin() + b);
                }
                for (auto &e : row) {
                    cc.push_back(e.first);
                    cv.push_back(e.second);
                }
                st[i - lo + 1] = static_cast<int64_t>(cc.size());
            }
            cc.shrink_to_fit();
            cv.shrink_to_fit();
            if (old_mut && ch < static_cast<int64_t>(old_mut->col.size())) {
                std::vector<int64_t>().swap(old_mut->start[ch]);
                std::vector<int32_t>().swap(old_mut->col[ch]);
                std::vector<double>().swap(old_mut->val[ch]);
            }
        }
    }
    int64_t nnz = 0;
    for (int64_t ch = 0; ch < nchunks; ++ch) nnz += H->col[ch].size();
    H->nnz = nnz;
    H->stripe.assign(Hamiltonian::STRIPES + 1, count);
    H->stripe[0] = 0;
    {
        int64_t i = 0, cumulative = 0;          // nonzeros of rows before i
        for (int64_t s = 1; s < Hamiltonian::STRIPES; ++s) {
            const int64_t goal = (nnz + count) * s / Hamiltonian::STRIPES;
            while (i < count && cumulative + i < goal) cumulative += H->row_length(i++);
            H->stripe[s] = i;
        }
    }
    H->offset.assign(Hamiltonian::STRIPES + 1, 0);
    for (int64_t s = 0; s < Hamiltonian::STRIPES; ++s)
        H->offset[s + 1] = H->offset[s] + (count - H->stripe[s]);
    H->buffer.assign(H->offset[Hamiltonian::STRIPES], 0.0);
    return H;
}

template <int W> Buffer *select_impl(Space<W> &sp, const double *cmax, double eps, int partners, int threads)
{
    const Ctx &c = *sp.ctx;
    const int64_t count = sp.dets.size();
    ShardedMap<W, int64_t> found(0);
#pragma omp parallel num_threads(threads)
    {
        typename ShardedMap<W, int64_t>::Stage stage(found);
#pragma omp for schedule(dynamic, 16)
        for (int64_t i = 0; i < count; ++i) {
            if (cmax[i] <= 0.0) continue;
            auto all = [](u64) { return true; };
            excitations<W>(c, sp.dets[i], hash_det<W>(sp.dets[i]), eps / cmax[i], all,
                           [&](const Det<W> &y, double, u64 h) {
                if (sp.index.find(y, h) >= 0) return;
                stage.add(y, h, nullptr);
                if (partners) {
                    Det<W> z;
                    std::memcpy(z.a(), y.b(), sizeof(u64) * W);
                    std::memcpy(z.b(), y.a(), sizeof(u64) * W);
                    const u64 hz = hash_det<W>(z);
                    if (sp.index.find(z, hz) < 0) stage.add(z, hz, nullptr);
                }
            });
        }
        stage.flush_all();
    }
    auto *out = new Buffer;
    out->width = 2 * W;
    std::vector<Det<W>> all;
    all.reserve(found.size());
    for (size_t s = 0; s < found.shards.size(); ++s)
        found.for_each(s, [&](const Det<W> &key, const int64_t *) { all.push_back(key); });
    found.clear();
    std::sort(all.begin(), all.end(), [](const Det<W> &x, const Det<W> &y) { return before(x, y); });
    out->rows = all.size();
    out->data.resize(all.size() * 2 * W);
    if (!all.empty()) std::memcpy(out->data.data(), all.data(), sizeof(Det<W>) * all.size());
    return out;
}

template <int W>
void pt2_impl(Space<W> &sp, const double *coeffs, int64_t nroots, const double *energies, double eps,
              int64_t batches, double *out, double *norm, int64_t *stats, int threads)
{
    const Ctx &c = *sp.ctx;
    const int64_t count = sp.dets.size();
    std::vector<double> cmax(count, 0.0);
    for (int64_t k = 0; k < nroots; ++k)
        for (int64_t i = 0; i < count; ++i) cmax[i] = std::max(cmax[i], std::fabs(coeffs[k * count + i]));
    std::vector<i128> total(nroots, 0), total_norm(nroots, 0);
    int64_t externals = 0, largest = 0;
    const char *profile_mode = std::getenv("ITERCC_SCI_PROFILE");
    const bool profile = profile_mode != nullptr;
    const bool no_map = profile && profile_mode[0] == '2';     // diagnostic: generation alone
    double generate_s = 0.0, pass_s = 0.0;
    int64_t contributions = 0;
    for (int64_t batch = 0; batch < batches; ++batch) {
        ShardedMap<W, int64_t> map(static_cast<int>(nroots));
        double t0 = omp_get_wtime();
#pragma omp parallel num_threads(threads) reduction(+ : contributions)
        {
            typename ShardedMap<W, int64_t>::Stage stage(map);
#pragma omp for schedule(dynamic, 16)
            for (int64_t i = 0; i < count; ++i) {
                if (cmax[i] <= 0.0) continue;
                int64_t add[16];
                auto in_batch = [&](u64 h) {
                    return batches == 1 || static_cast<int64_t>((h >> 20) % static_cast<u64>(batches)) == batch;
                };
                excitations<W>(c, sp.dets[i], hash_det<W>(sp.dets[i]), eps / cmax[i], in_batch,
                               [&](const Det<W> &y, double v, u64 h) {
                    for (int64_t k = 0; k < nroots; ++k) add[k] = to_numerator(v * coeffs[k * count + i]);
                    if (!no_map) stage.add(y, h, add);
                    ++contributions;
                });
            }
            stage.flush_all();
        }
        double t1 = omp_get_wtime();
        generate_s += t1 - t0;
        const int64_t nshards = map.shards.size();
        std::vector<i128> part(nshards * nroots, 0), part_norm(nshards * nroots, 0);
        std::vector<int64_t> found(nshards, 0);
#pragma omp parallel for schedule(dynamic, 4) num_threads(threads)
        for (int64_t s = 0; s < nshards; ++s) {
            map.for_each(s, [&](const Det<W> &y, const int64_t *vals) {
                if (sp.index.find(y, hash_det<W>(y)) >= 0) return;   // in the space: not external
                ++found[s];
                int oa[MAX_ORB], ob[MAX_ORB];
                int na = occupied<W>(y.a(), oa), nb = occupied<W>(y.b(), ob);
                const double hyy = diagonal_energy(c, oa, na, ob, nb);
                for (int64_t r = 0; r < nroots; ++r) {
                    const double num = from_numerator(vals[r]);
                    const double amplitude = num / (energies[r] - hyy);
                    part[s * nroots + r] += to_fixed(num * amplitude);
                    part_norm[s * nroots + r] += to_fixed(amplitude * amplitude);
                }
            });
        }
        int64_t size = 0;
        for (int64_t s = 0; s < nshards; ++s) {
            size += found[s];
            for (int64_t r = 0; r < nroots; ++r) {
                total[r] += part[s * nroots + r];
                total_norm[r] += part_norm[s * nroots + r];
            }
        }
        externals += size;
        largest = std::max(largest, size);
        pass_s += omp_get_wtime() - t1;
    }
    if (profile)
        std::fprintf(stderr, "itercc_sci_pt2: generate %.2f s, final pass %.2f s, %lld contributions, %lld externals\n",
                     generate_s, pass_s, static_cast<long long>(contributions), static_cast<long long>(externals));
    for (int64_t r = 0; r < nroots; ++r) {
        out[r] = from_fixed(total[r]);
        if (norm) norm[r] = from_fixed(total_norm[r]);
    }
    if (stats) {
        stats[0] = externals;
        stats[1] = largest;
    }
}

// Stochastic estimate of E2[eps] - E2[eps_det] (Sharma, Holmes, Jeanmairet,
// Alavi, Umrigar, JCTC 13, 1595 (2017), eq. 7): each batch draws `samples`
// determinants with replacement, p_i proportional to max_k |c_ik|, and
// evaluates the unbiased estimator of the PT2 energy over the terms above
// eps and, from the same samples, over those above eps_det.
template <int W>
void pt2_stochastic_impl(Space<W> &sp, const double *coeffs, int64_t nroots, const double *energies, double eps,
                         double eps_det, int64_t samples, int64_t nbatches, u64 seed, double *mean, double *error,
                         double *norm, int64_t *stats, int threads)
{
    const Ctx &c = *sp.ctx;
    const int64_t count = sp.dets.size();
    std::vector<double> cmax(count, 0.0), cdf(count + 1, 0.0);
    for (int64_t k = 0; k < nroots; ++k)
        for (int64_t i = 0; i < count; ++i) cmax[i] = std::max(cmax[i], std::fabs(coeffs[k * count + i]));
    for (int64_t i = 0; i < count; ++i) cdf[i + 1] = cdf[i] + cmax[i];
    const double total_weight = cdf[count];
    const double nd = static_cast<double>(samples);
    std::vector<double> values(nbatches * nroots, 0.0), norms(nbatches * nroots, 0.0);
    int64_t externals = 0;
    for (int64_t batch = 0; batch < nbatches; ++batch) {
        u64 state = mix(seed ^ (0x9e3779b97f4a7c15ULL * static_cast<u64>(batch + 1)));
        std::vector<int64_t> drawn(samples);
        for (int64_t s = 0; s < samples; ++s) {
            state += 0x9e3779b97f4a7c15ULL;
            const double u = static_cast<double>(mix(state) >> 11) * (1.0 / 9007199254740992.0);
            int64_t i = std::upper_bound(cdf.begin() + 1, cdf.end(), u * total_weight) - (cdf.begin() + 1);
            drawn[s] = std::min<int64_t>(i, count - 1);
        }
        std::sort(drawn.begin(), drawn.end());
        std::vector<std::pair<int64_t, double>> distinct;
        for (int64_t s = 0; s < samples;) {
            int64_t t = s;
            while (t < samples && drawn[t] == drawn[s]) ++t;
            distinct.emplace_back(drawn[s], static_cast<double>(t - s));
            s = t;
        }
        const int nv = static_cast<int>(4 * nroots);
        ShardedMap<W> map(nv);
#pragma omp parallel num_threads(threads)
        {
        typename ShardedMap<W>::Stage stage(map);
#pragma omp for schedule(dynamic, 1)
        for (int64_t t = 0; t < static_cast<int64_t>(distinct.size()); ++t) {
            const int64_t i = distinct[t].first;
            const double w = distinct[t].second, p = cmax[i] / total_weight;
            if (cmax[i] <= 0.0) continue;
            i128 add[64];
            auto all = [](u64) { return true; };
            excitations<W>(c, sp.dets[i], hash_det<W>(sp.dets[i]), eps / cmax[i], all,
                           [&](const Det<W> &y, double v, u64 h) {
                const bool kept = std::fabs(v) * cmax[i] >= eps_det;
                for (int64_t k = 0; k < nroots; ++k) {
                    const double term = v * coeffs[k * count + i];
                    const i128 a = to_fixed(w * term / p);
                    const i128 b = to_fixed((w * (nd - 1.0) / p - w * w / (p * p)) * term * term);
                    add[4 * k] = a;
                    add[4 * k + 1] = b;
                    add[4 * k + 2] = kept ? a : 0;
                    add[4 * k + 3] = kept ? b : 0;
                }
                stage.add(y, h, add);
            });
        }
        stage.flush_all();
        }
        externals += map.size();
        const int64_t nshards = map.shards.size();
        std::vector<i128> part(nshards * nroots, 0), part_norm(nshards * nroots, 0);
#pragma omp parallel for schedule(dynamic, 4) num_threads(threads)
        for (int64_t s = 0; s < nshards; ++s) {
            map.for_each(s, [&](const Det<W> &y, const i128 *vals) {
                if (sp.index.find(y, hash_det<W>(y)) >= 0) return;   // in the space: not external
                int oa[MAX_ORB], ob[MAX_ORB];
                int na = occupied<W>(y.a(), oa), nb = occupied<W>(y.b(), ob);
                const double hyy = diagonal_energy(c, oa, na, ob, nb);
                for (int64_t r = 0; r < nroots; ++r) {
                    const i128 *q = vals + 4 * r;
                    const double a = from_fixed(q[0]), b = from_fixed(q[1]);
                    const double ad = from_fixed(q[2]), bd = from_fixed(q[3]);
                    const double inverse = 1.0 / (energies[r] - hyy);
                    const double square = ((a * a + b) - (ad * ad + bd)) / (nd * (nd - 1.0));
                    part[s * nroots + r] += to_fixed(square * inverse);
                    part_norm[s * nroots + r] += to_fixed(square * inverse * inverse);
                }
            });
        }
        for (int64_t r = 0; r < nroots; ++r) {
            i128 sum = 0, sum_norm = 0;
            for (int64_t s = 0; s < nshards; ++s) {
                sum += part[s * nroots + r];
                sum_norm += part_norm[s * nroots + r];
            }
            values[batch * nroots + r] = from_fixed(sum);
            norms[batch * nroots + r] = from_fixed(sum_norm);
        }
    }
    for (int64_t r = 0; r < nroots; ++r) {
        double m = 0.0, v = 0.0;
        for (int64_t b = 0; b < nbatches; ++b) m += values[b * nroots + r];
        m /= nbatches;
        for (int64_t b = 0; b < nbatches; ++b) v += (values[b * nroots + r] - m) * (values[b * nroots + r] - m);
        mean[r] = m;
        if (norm) {
            double nm = 0.0;
            for (int64_t b = 0; b < nbatches; ++b) nm += norms[b * nroots + r];
            norm[r] = nm / nbatches;
        }
        error[r] = nbatches > 1 ? std::sqrt(v / (nbatches - 1) / nbatches) : 0.0;
    }
    if (stats) stats[0] = externals;
}

template <int W>
void rdm1_impl(Space<W> &sp, const double *coeffs, int64_t nroots, const double *weights, double *out, int threads)
{
    sp.build_strings(threads);
    const int64_t n = sp.ctx->n, count = sp.dets.size();
    const int nt = threads > 0 ? threads : omp_get_max_threads();
    std::vector<i128> acc(static_cast<size_t>(nt) * n * n, 0);
#pragma omp parallel num_threads(nt)
    {
        i128 *mine = &acc[static_cast<size_t>(omp_get_thread_num()) * n * n];
#pragma omp for schedule(dynamic, 64)
        for (int64_t i = 0; i < count; ++i) {
            const Det<W> &x = sp.dets[i];
            double cc = 0.0;
            for (int64_t k = 0; k < nroots; ++k) cc += weights[k] * coeffs[k * count + i] * coeffs[k * count + i];
            const i128 diag = to_fixed(cc);
            int occ[MAX_ORB];
            for (int spin = 0; spin < 2; ++spin) {
                int no = occupied<W>(x.w + spin * W, occ);
                for (int t = 0; t < no; ++t) mine[occ[t] * n + occ[t]] += diag;
            }
            // Singles to later determinants: alpha (same beta) and beta (same alpha).
            auto visit = [&](int32_t j, int ca, int cb) {
                if (j < i || ca + cb != 2) return;
                const Det<W> &y = sp.dets[j];
                const int spin = ca ? 0 : 1;
                const u64 *xs = x.w + spin * W, *ys = y.w + spin * W;
                u64 hole[W], part[W];
                for (int k = 0; k < W; ++k) {
                    hole[k] = xs[k] & ~ys[k];
                    part[k] = ys[k] & ~xs[k];
                }
                int p = 0, r = 0;
                set_bits<W>(hole, &p, 1);
                set_bits<W>(part, &r, 1);
                double cij = 0.0;
                for (int64_t k = 0; k < nroots; ++k) cij += weights[k] * coeffs[k * count + i] * coeffs[k * count + j];
                const i128 v = to_fixed(phase<W>(xs, p, r) * cij);
                mine[r * n + p] += v;
                mine[p * n + r] += v;
            };
            sp.for_each_alpha_excited(i, visit);
            const int32_t b = sp.bid[i];
            for (int64_t t = sp.bmajor_start[b]; t < sp.bmajor_start[b + 1]; ++t) {
                const int32_t j = sp.bmajor[t];
                if (j == i) continue;
                int d = 0;
                for (int k = 0; k < W; ++k) d += popc(x.a()[k] ^ sp.dets[j].a()[k]);
                if (d == 2) visit(j, 2, 0);
            }
        }
    }
    for (int64_t e = 0; e < n * n; ++e) {
        i128 sum = 0;
        for (int t = 0; t < nt; ++t) sum += acc[static_cast<size_t>(t) * n * n + e];
        out[e] = from_fixed(sum);
    }
}

// <S^2> of M_S = 0 states (equal alpha and beta counts).
template <int W> void s2_impl(Space<W> &sp, const double *coeffs, int64_t nroots, double *out, int threads)
{
    sp.build_strings(threads);
    const int64_t count = sp.dets.size();
    const int nt = threads > 0 ? threads : omp_get_max_threads();
    std::vector<i128> acc(static_cast<size_t>(nt) * nroots, 0);
#pragma omp parallel num_threads(nt)
    {
        i128 *mine = &acc[static_cast<size_t>(omp_get_thread_num()) * nroots];
#pragma omp for schedule(dynamic, 64)
        for (int64_t i = 0; i < count; ++i) {
            const Det<W> &x = sp.dets[i];
            int open = 0;
            for (int k = 0; k < W; ++k) open += popc(x.b()[k] & ~x.a()[k]);
            for (int64_t r = 0; r < nroots; ++r)
                mine[r] += to_fixed(open * coeffs[r * count + i] * coeffs[r * count + i]);
            // Spin exchange: alpha q -> p, beta p -> q.
            const int32_t a = sp.aid[i];
            for (int64_t u = sp.asingle_start[a]; u < sp.asingle_start[a + 1]; ++u) {
                const int32_t a2 = sp.asingle[u];
                for (int64_t t = sp.amajor_start[a2]; t < sp.amajor_start[a2 + 1]; ++t) {
                    const int32_t j = sp.amajor[t];
                    const Det<W> &y = sp.dets[j];
                    u64 ah[W], ap[W], bh[W], bp[W];
                    int d = 0;
                    for (int k = 0; k < W; ++k) {
                        ah[k] = x.a()[k] & ~y.a()[k];
                        ap[k] = y.a()[k] & ~x.a()[k];
                        bh[k] = x.b()[k] & ~y.b()[k];
                        bp[k] = y.b()[k] & ~x.b()[k];
                        d += popc(x.b()[k] ^ y.b()[k]);
                    }
                    if (d != 2) continue;
                    int q = 0, p = 0, bq = -1, bpp = -1;
                    set_bits<W>(ah, &q, 1);
                    set_bits<W>(ap, &p, 1);
                    set_bits<W>(bh, &bq, 1);
                    set_bits<W>(bp, &bpp, 1);
                    if (bq != p || bpp != q) continue;
                    const double e = -phase<W>(x.a(), q, p) * phase<W>(x.b(), p, q);
                    for (int64_t r = 0; r < nroots; ++r)
                        mine[r] += to_fixed(e * coeffs[r * count + i] * coeffs[r * count + j]);
                }
            }
        }
    }
    for (int64_t r = 0; r < nroots; ++r) {
        i128 sum = 0;
        for (int t = 0; t < nt; ++t) sum += acc[static_cast<size_t>(t) * nroots + r];
        out[r] = from_fixed(sum);
    }
}

template <int W> void partners_impl(Space<W> &sp, int64_t *out, int threads)
{
    const int64_t count = sp.dets.size();
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t i = 0; i < count; ++i) {
        Det<W> z;
        std::memcpy(z.a(), sp.dets[i].b(), sizeof(u64) * W);
        std::memcpy(z.b(), sp.dets[i].a(), sizeof(u64) * W);
        out[i] = sp.index.find(z, hash_det<W>(z));
    }
}

template <class F> auto dispatch(int words, F &&f)
{
    switch (words) {
    case 1: return f(std::integral_constant<int, 1>{});
    case 2: return f(std::integral_constant<int, 2>{});
    case 3: return f(std::integral_constant<int, 3>{});
    default: return f(std::integral_constant<int, 4>{});
    }
}

struct SpaceHandle {
    const Ctx *ctx;
    SpaceBase *space;
};

void sort_list(std::vector<Entry> &v, int64_t lo, int64_t hi)
{
    std::sort(v.begin() + lo, v.begin() + hi, [](const Entry &x, const Entry &y) {
        const double ax = std::fabs(x.v), ay = std::fabs(y.v);
        if (ax != ay) return ax > ay;
        if (x.r != y.r) return x.r < y.r;
        return x.s < y.s;
    });
}

}  // namespace

extern "C" {

void *itercc_sci_create(int64_t norb, int64_t nalpha, int64_t nbeta, const double *h, const double *eri,
                        const int64_t *irrep, double cutoff, int threads)
{
    // Fewer than 64 alpha electrons: the hash maps mark empty entries with an
    // all-ones first word, which no determinant then has.
    if (norb < 1 || norb > MAX_ORB || nalpha < 0 || nbeta < 0 || nalpha > norb || nbeta > norb || nalpha >= 64)
        return nullptr;
    auto *c = new (std::nothrow) Ctx;
    if (!c) return nullptr;
    const int64_t n = norb;
    c->n = n;
    c->na = nalpha;
    c->nb = nbeta;
    c->words = static_cast<int>((n + 63) / 64);
    c->h.assign(h, h + n * n);
    c->eri.assign(eri, eri + n * n * n * n);
    c->irrep.assign(n, 0);
    if (irrep) c->irrep.assign(irrep, irrep + n);
    c->J.resize(n * n);
    c->K.resize(n * n);
    for (int64_t p = 0; p < n; ++p)
        for (int64_t q = 0; q < n; ++q) {
            c->J[p * n + q] = c->E(p, p, q, q);
            c->K[p * n + q] = c->E(p, q, q, p);
        }
    c->Dk.resize(n * n * n);
    c->Xk.resize(n * n * n);
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for (int64_t p = 0; p < n; ++p)
        for (int64_t r = 0; r < n; ++r)
            for (int64_t k = 0; k < n; ++k) {
                c->Dk[(p * n + r) * n + k] = c->E(p, r, k, k);
                c->Xk[(p * n + r) * n + k] = c->E(p, k, k, r);
            }
    // Heat-bath lists. Opposite spin: every ordered pair (p, q), entries (r, s).
    std::vector<int64_t> os_count(n * n, 0), ss_count(n * n, 0);
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (int64_t pq = 0; pq < n * n; ++pq) {
        const int64_t p = pq / n, q = pq % n;
        int64_t cnt = 0, sc = 0;
        for (int64_t r = 0; r < n; ++r)
            for (int64_t s = 0; s < n; ++s) {
                if (r != p && s != q && std::fabs(c->E(p, r, q, s)) >= cutoff) ++cnt;
                if (p < q && r < s && r != p && r != q && s != p && s != q
                    && std::fabs(c->E(p, r, q, s) - c->E(p, s, q, r)) >= cutoff)
                    ++sc;
            }
        os_count[pq] = cnt;
        ss_count[pq] = sc;
    }
    c->os_start.assign(n * n + 1, 0);
    c->ss_start.assign(n * n + 1, 0);
    for (int64_t pq = 0; pq < n * n; ++pq) {
        c->os_start[pq + 1] = c->os_start[pq] + os_count[pq];
        c->ss_start[pq + 1] = c->ss_start[pq] + ss_count[pq];
    }
    c->os.resize(c->os_start[n * n]);
    c->ss.resize(c->ss_start[n * n]);
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (int64_t pq = 0; pq < n * n; ++pq) {
        const int64_t p = pq / n, q = pq % n;
        int64_t at = c->os_start[pq], st = c->ss_start[pq];
        for (int64_t r = 0; r < n; ++r)
            for (int64_t s = 0; s < n; ++s) {
                const double v = c->E(p, r, q, s);
                if (r != p && s != q && std::fabs(v) >= cutoff)
                    c->os[at++] = Entry{v, static_cast<uint16_t>(r), static_cast<uint16_t>(s)};
                if (p < q && r < s && r != p && r != q && s != p && s != q) {
                    const double w = v - c->E(p, s, q, r);
                    if (std::fabs(w) >= cutoff) c->ss[st++] = Entry{w, static_cast<uint16_t>(r), static_cast<uint16_t>(s)};
                }
            }
        sort_list(c->os, c->os_start[pq], c->os_start[pq + 1]);
        sort_list(c->ss, c->ss_start[pq], c->ss_start[pq + 1]);
    }
    // Singles p -> r of equal irrep: bound |h_pr| + sum_k (2 |(pr|kk)| + |(pk|kr)|).
    c->single_start.assign(n + 1, 0);
    std::vector<std::vector<SinglesEntry>> lists(n);
    for (int64_t p = 0; p < n; ++p) {
        for (int64_t r = 0; r < n; ++r) {
            if (r == p || c->irrep[r] != c->irrep[p]) continue;
            double b = std::fabs(c->h[p * n + r]);
            for (int64_t k = 0; k < n; ++k)
                b += 2.0 * std::fabs(c->Dk[(p * n + r) * n + k]) + std::fabs(c->Xk[(p * n + r) * n + k]);
            if (b >= cutoff) lists[p].push_back(SinglesEntry{b, static_cast<int32_t>(r)});
        }
        std::sort(lists[p].begin(), lists[p].end(), [](const SinglesEntry &x, const SinglesEntry &y) {
            return x.bound != y.bound ? x.bound > y.bound : x.r < y.r;
        });
        c->single_start[p + 1] = c->single_start[p] + lists[p].size();
    }
    for (int64_t p = 0; p < n; ++p) c->singles.insert(c->singles.end(), lists[p].begin(), lists[p].end());
    return c;
}

void itercc_sci_free(void *ctx) { delete static_cast<Ctx *>(ctx); }

int64_t itercc_sci_words(const void *ctx) { return static_cast<const Ctx *>(ctx)->words; }

int64_t itercc_sci_list_sizes(const void *ctx, int64_t *out)
{
    const Ctx *c = static_cast<const Ctx *>(ctx);
    out[0] = c->os.size();
    out[1] = c->ss.size();
    out[2] = c->singles.size();
    return static_cast<int64_t>((c->os.size() + c->ss.size()) * sizeof(Entry) + c->singles.size() * sizeof(SinglesEntry)
                                + (c->eri.size() + 2 * c->Dk.size()) * sizeof(double));
}

void *itercc_sci_space(const void *ctx, const uint64_t *dets, int64_t count, int threads)
{
    const Ctx *c = static_cast<const Ctx *>(ctx);
    if (count >= (int64_t(1) << 31)) return nullptr;
    SpaceBase *s = dispatch(c->words, [&](auto w) -> SpaceBase * {
        return new Space<decltype(w)::value>(c, dets, count, threads);
    });
    return new SpaceHandle{c, s};
}

void itercc_sci_space_free(void *space)
{
    auto *s = static_cast<SpaceHandle *>(space);
    delete s->space;
    delete s;
}

void itercc_sci_diagonal(void *space, double *out, int threads)
{
    auto *s = static_cast<SpaceHandle *>(space);
    dispatch(s->ctx->words, [&](auto w) {
        diagonal_impl(*as_space<decltype(w)::value>(s->space), out, threads);
        return 0;
    });
}

void *itercc_sci_hamiltonian(void *space, double drop, int threads)
{
    auto *s = static_cast<SpaceHandle *>(space);
    return dispatch(s->ctx->words, [&](auto w) -> void * {
        return hamiltonian_impl(*as_space<decltype(w)::value>(s->space), drop, threads);
    });
}

void *itercc_sci_hamiltonian_extend(void *space, const void *old, int64_t n_old, double drop, int threads)
{
    auto *s = static_cast<SpaceHandle *>(space);
    const auto *H = static_cast<const Hamiltonian *>(old);
    if (!H || H->n != n_old) return nullptr;
    return dispatch(s->ctx->words, [&](auto w) -> void * {
        auto &sp = *as_space<decltype(w)::value>(s->space);
        if (n_old > static_cast<int64_t>(sp.dets.size())) return nullptr;
        return hamiltonian_impl(sp, drop, threads, H, n_old);
    });
}

int64_t itercc_sci_ham_nnz(const void *ham) { return static_cast<const Hamiltonian *>(ham)->nnz; }

void itercc_sci_ham_free(void *ham) { delete static_cast<Hamiltonian *>(ham); }

void itercc_sci_ham_diagonal(const void *ham, double *out)
{
    const auto *H = static_cast<const Hamiltonian *>(ham);
    std::copy(H->diag.begin(), H->diag.end(), out);
}

void itercc_sci_matvec(void *ham, const double *x, double *y, int threads)
{
    auto *H = static_cast<Hamiltonian *>(ham);
    const int64_t count = H->n;
#pragma omp parallel num_threads(threads)
    {
#pragma omp for schedule(dynamic, 1)
        for (int64_t s = 0; s < Hamiltonian::STRIPES; ++s) {
            const int64_t lo = H->stripe[s], hi = H->stripe[s + 1];
            double *buf = H->buffer.data() + H->offset[s] - lo;      // buf[j] for j >= lo
            std::fill(buf + lo, buf + count, 0.0);
            for (int64_t i = lo; i < hi; ++i) {
                const int64_t ch = i / Hamiltonian::CHUNK, local = i % Hamiltonian::CHUNK;
                const int64_t *st = H->start[ch].data();
                const int32_t *col = H->col[ch].data();
                const double *val = H->val[ch].data();
                const double xi = x[i];
                double v = H->diag[i] * xi;
                for (int64_t t = st[local]; t < st[local + 1]; ++t) {
                    v += val[t] * x[col[t]];
                    buf[col[t]] += val[t] * xi;
                }
                y[i] = v;
            }
        }
#pragma omp for schedule(static)
        for (int64_t j = 0; j < count; ++j) {
            double v = y[j];
            for (int64_t s = 0; s < Hamiltonian::STRIPES && H->stripe[s] <= j; ++s)
                v += H->buffer[H->offset[s] + (j - H->stripe[s])];
            y[j] = v;
        }
    }
}

void *itercc_sci_select(void *space, const double *cmax, double eps, int partners, int threads)
{
    auto *s = static_cast<SpaceHandle *>(space);
    return dispatch(s->ctx->words, [&](auto w) -> void * {
        return select_impl(*as_space<decltype(w)::value>(s->space), cmax, eps, partners, threads);
    });
}

int64_t itercc_sci_buffer_rows(const void *buffer) { return static_cast<const Buffer *>(buffer)->rows; }

void itercc_sci_buffer_copy(const void *buffer, uint64_t *out)
{
    const auto *b = static_cast<const Buffer *>(buffer);
    std::copy(b->data.begin(), b->data.end(), out);
}

void itercc_sci_buffer_free(void *buffer) { delete static_cast<Buffer *>(buffer); }

void itercc_sci_pt2(void *space, const double *coeffs, int64_t nroots, const double *energies, double eps,
                    int64_t batches, double *out, double *norm, int64_t *stats, int threads)
{
    auto *s = static_cast<SpaceHandle *>(space);
    if (nroots < 1 || nroots > 16 || batches < 1) return;
    dispatch(s->ctx->words, [&](auto w) {
        pt2_impl(*as_space<decltype(w)::value>(s->space), coeffs, nroots, energies, eps, batches, out, norm, stats,
                 threads);
        return 0;
    });
}

void itercc_sci_pt2_stochastic(void *space, const double *coeffs, int64_t nroots, const double *energies,
                               double eps, double eps_det, int64_t samples, int64_t nbatches, int64_t seed,
                               double *mean, double *error, double *norm, int64_t *stats, int threads)
{
    auto *s = static_cast<SpaceHandle *>(space);
    if (nroots < 1 || nroots > 16 || samples < 2 || nbatches < 1) return;
    dispatch(s->ctx->words, [&](auto w) {
        pt2_stochastic_impl(*as_space<decltype(w)::value>(s->space), coeffs, nroots, energies, eps, eps_det,
                            samples, nbatches, static_cast<u64>(seed), mean, error, norm, stats, threads);
        return 0;
    });
}

void itercc_sci_rdm1(void *space, const double *coeffs, int64_t nroots, const double *weights, double *out,
                     int threads)
{
    auto *s = static_cast<SpaceHandle *>(space);
    dispatch(s->ctx->words, [&](auto w) {
        rdm1_impl(*as_space<decltype(w)::value>(s->space), coeffs, nroots, weights, out, threads);
        return 0;
    });
}

void itercc_sci_s2(void *space, const double *coeffs, int64_t nroots, double *out, int threads)
{
    auto *s = static_cast<SpaceHandle *>(space);
    dispatch(s->ctx->words, [&](auto w) {
        s2_impl(*as_space<decltype(w)::value>(s->space), coeffs, nroots, out, threads);
        return 0;
    });
}

void itercc_sci_partners(void *space, int64_t *out, int threads)
{
    auto *s = static_cast<SpaceHandle *>(space);
    dispatch(s->ctx->words, [&](auto w) {
        partners_impl(*as_space<decltype(w)::value>(s->space), out, threads);
        return 0;
    });
}

}  // extern "C"
