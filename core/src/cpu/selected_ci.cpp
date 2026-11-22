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
#if defined(__SSSE3__)
#include <immintrin.h>
#endif

namespace {

using u64 = uint64_t;
using i128 = __int128;

constexpr double FIXED_SCALE = 1208925819614629174706176.0;   // 2^80
constexpr double NUMERATOR_SCALE = 4503599627370496.0;          // 2^52 for int64 PT2 numerators
inline int64_t to_numerator(double x) { return static_cast<int64_t>(x * NUMERATOR_SCALE); }
inline double from_numerator(int64_t v) { return static_cast<double>(v) / NUMERATOR_SCALE; }
// trunc(x * 2^80) as static_cast<i128> gives it, from the bits of x: the cast
// calls libgcc's soft-float __fixdfti, and the stochastic PT2 converts eight
// values per term. |x| < 2^47 (the fixed-point range); NaN and infinities
// are outside it as for the cast.
inline i128 to_fixed(double x)
{
    u64 bits;
    std::memcpy(&bits, &x, sizeof bits);
    const int exponent = static_cast<int>((bits >> 52) & 0x7FF);
    if (exponent == 0) return 0;                       // zero and subnormals (|x| 2^80 < 1)
    const u64 mantissa = (bits & ((u64(1) << 52) - 1)) | (u64(1) << 52);
    // x 2^80 = mantissa 2^(exponent - 1075 + 80)
    const int shift = exponent - 995;
    i128 v;
    if (shift >= 0)
        v = static_cast<i128>(mantissa) << shift;
    else if (shift > -53)
        v = static_cast<i128>(mantissa >> -shift);
    else
        return 0;
    return (bits >> 63) ? -v : v;
}
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

// The PT2 batch of an external with hash h, from bits 32-54: hash maps take
// their slots from the low bits and their shards from the top 9, so a batch
// spreads over whole tables. (Bits 20-23 were slot bits too once a shard held
// 2^21 slots: each batch then crowded one sixteenth of every table and inserts
// went quadratic above ~1 million entries per shard, 8.6e9 externals at 16
// batches.)
inline int64_t batch_of(u64 h, int64_t batches)
{
    return static_cast<int64_t>((((h >> 32) & 0x7FFFFFULL) * static_cast<u64>(batches)) >> 23);
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
    // eri: (pq|rs) once per 8-fold permutation class (real orbitals), at
    // pair(pair(p, q), pair(r, s)): an eighth of the n^4 array (126 orbitals:
    // 0.26 instead of 2 GB per context), at the same build speed. Axial
    // (complex, Lz-adapted) orbitals keep real integrals with only the
    // symmetry (pq|rs) = (rs|pq) used here, at pair(p n + q, r n + s): half
    // the array. The formulas below take the bra orbital of each electron
    // first or rely on (pq|rs) = (qp|sr), which complex orbitals with real
    // integrals keep.
    std::vector<double> h, eri;
    bool axial = false;
    // Reflection y -> -y of axial orbitals: orbital p -> mirror[p] (an
    // involution, with phase +1); empty when not given.
    std::vector<int32_t> mirror;
    std::vector<int64_t> irrep;
    std::vector<double> J, K;        // J[p n + q] = (pp|qq), K[p n + q] = (pq|qp)
    std::vector<double> JK;          // J - K: same-spin pairs in diagonal energies (one load)
    std::vector<double> Dk, Xk;      // Dk[(p n + r) n + k] = (pr|kk), Xk = (pk|kr)
    std::vector<int64_t> os_start;   // opposite spin: pair p n + q -> entries over (r, s)
    std::vector<Entry> os;
    std::vector<int64_t> ss_start;   // same spin: pair p n + q (p < q) -> entries r < s
    std::vector<Entry> ss;
    std::vector<int64_t> single_start;
    std::vector<SinglesEntry> singles;

    static int64_t pair(int64_t p, int64_t q)
    {
        const int64_t a = p > q ? p : q, b = p > q ? q : p;
        return a * (a + 1) / 2 + b;
    }
    double E(int64_t p, int64_t q, int64_t r, int64_t s) const
    {
        return axial ? eri[pair(p * n + q, r * n + s)] : eri[pair(pair(p, q), pair(r, s))];
    }
};

namespace {

using Ctx = itercc_sci_context;

// <x| H |x> without the constant.
// sum_k row[idx[k]] in four partial sums: a single accumulator makes every
// add wait on the previous one (about 150 per diagonal energy).
inline double gathered_sum(const double *row, const int *idx, int m)
{
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    int k = 0;
    for (; k + 4 <= m; k += 4) {
        s0 += row[idx[k]];
        s1 += row[idx[k + 1]];
        s2 += row[idx[k + 2]];
        s3 += row[idx[k + 3]];
    }
    for (; k < m; ++k) s0 += row[idx[k]];
    return (s0 + s1) + (s2 + s3);
}

inline double diagonal_energy(const Ctx &c, const int *oa, int na, const int *ob, int nb)
{
    const int64_t n = c.n;
    double e = 0.0;
    for (int i = 0; i < na; ++i) {
        const int p = oa[i];
        e += c.h[p * n + p] + gathered_sum(&c.JK[p * n], oa, i) + gathered_sum(&c.J[p * n], ob, nb);
    }
    for (int i = 0; i < nb; ++i) {
        const int p = ob[i];
        e += c.h[p * n + p] + gathered_sum(&c.JK[p * n], ob, i);
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
template <int W>
double element_strings(const Ctx &c, const u64 *xa, const u64 *xb, const u64 *ya, const u64 *yb,
                       const int *occ = nullptr, int occ_na = 0, int occ_nb = 0)
{
    struct View {
        const u64 *pa, *pb;
        const u64 *a() const { return pa; }
        const u64 *b() const { return pb; }
    } x{xa, xb}, y{ya, yb};
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

template <int W> double element(const Ctx &c, const Det<W> &x, const Det<W> &y, const int *occ = nullptr,
                                int occ_na = 0, int occ_nb = 0)
{
    return element_strings<W>(c, x.a(), x.b(), y.a(), y.b(), occ, occ_na, occ_nb);
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

// excitations() with the hash of each y's alpha-beta swap as well (hxs: x's
// swap's): accept(hy, hys) and emit(y, value, hy, hys). An alpha change of y
// is a beta change of its swap.
template <int W, class Accept, class Emit>
void excitations_paired(const Ctx &c, const Det<W> &x, u64 hx, u64 hxs, double threshold, Accept &&accept,
                        Emit &&emit)
{
    const int64_t n = c.n;
    const u64(&z)[2][256] = zobrist().key;
    int oa[MAX_ORB], ob[MAX_ORB];
    const int na = occupied<W>(x.a(), oa), nb = occupied<W>(x.b(), ob);
    Det<W> y;
    for (int i = 0; i < na; ++i)
        for (int j = 0; j < nb; ++j) {
            const int p = oa[i], q = ob[j];
            const u64 hpq = hx ^ z[0][p] ^ z[1][q], spq = hxs ^ z[1][p] ^ z[0][q];
            const int64_t pair = p * n + q;
            for (int64_t e = c.os_start[pair]; e < c.os_start[pair + 1]; ++e) {
                const Entry &en = c.os[e];
                if (std::fabs(en.v) < threshold) break;
                if (test(x.a(), en.r) || test(x.b(), en.s)) continue;
                const u64 hy = hpq ^ z[0][en.r] ^ z[1][en.s], hys = spq ^ z[1][en.r] ^ z[0][en.s];
                if (!accept(hy, hys)) continue;
                y = x;
                flip(y.a(), p);
                flip(y.a(), en.r);
                flip(y.b(), q);
                flip(y.b(), en.s);
                emit(y, phase<W>(x.a(), p, en.r) * phase<W>(x.b(), q, en.s) * en.v, hy, hys);
            }
        }
    for (int spin = 0; spin < 2; ++spin) {
        const int *occ = spin ? ob : oa;
        const int no = spin ? nb : na;
        const u64 *xs = spin ? x.b() : x.a();
        const u64 *zs = z[spin], *zo = z[1 - spin];
        for (int i = 0; i < no; ++i)
            for (int j = i + 1; j < no; ++j) {
                const int p = occ[i], q = occ[j];
                const u64 hpq = hx ^ zs[p] ^ zs[q], spq = hxs ^ zo[p] ^ zo[q];
                const int64_t pair = p * n + q;
                for (int64_t e = c.ss_start[pair]; e < c.ss_start[pair + 1]; ++e) {
                    const Entry &en = c.ss[e];
                    if (std::fabs(en.v) < threshold) break;
                    if (test(xs, en.r) || test(xs, en.s)) continue;
                    const u64 hy = hpq ^ zs[en.r] ^ zs[en.s], hys = spq ^ zo[en.r] ^ zo[en.s];
                    if (!accept(hy, hys)) continue;
                    y = x;
                    u64 *ys = spin ? y.b() : y.a();
                    double sign = phase<W>(ys, p, en.r);
                    flip(ys, p);
                    flip(ys, en.r);
                    sign *= phase<W>(ys, q, en.s);
                    flip(ys, q);
                    flip(ys, en.s);
                    emit(y, sign * en.v, hy, hys);
                }
            }
    }
    for (int spin = 0; spin < 2; ++spin) {
        const int *occ = spin ? ob : oa, *other = spin ? oa : ob;
        const int no = spin ? nb : na, nother = spin ? na : nb;
        const u64 *xs = spin ? x.b() : x.a();
        const u64 *zs = z[spin], *zo = z[1 - spin];
        for (int i = 0; i < no; ++i) {
            const int p = occ[i];
            for (int64_t e = c.single_start[p]; e < c.single_start[p + 1]; ++e) {
                const SinglesEntry &en = c.singles[e];
                if (en.bound < threshold) break;
                if (test(xs, en.r)) continue;
                const u64 hy = hx ^ zs[p] ^ zs[en.r], hys = hxs ^ zo[p] ^ zo[en.r];
                if (!accept(hy, hys)) continue;
                double v = phase<W>(xs, p, en.r) * single_value(c, p, en.r, occ, no, other, nother);
                if (std::fabs(v) < threshold) continue;
                y = x;
                u64 *ys = spin ? y.b() : y.a();
                flip(ys, p);
                flip(ys, en.r);
                emit(y, v, hy, hys);
            }
        }
    }
}

// -1, 0 or +1 as y's alpha string orders before, equal to or after its beta
// string: a determinant with alpha <= beta represents its swap pair in PT2.
template <int W> inline int string_order(const Det<W> &y)
{
    for (int k = W - 1; k >= 0; --k)
        if (y.a()[k] != y.b()[k]) return y.a()[k] < y.b()[k] ? -1 : 1;
    return 0;
}

template <int W> inline Det<W> swapped(const Det<W> &y)
{
    Det<W> z;
    std::memcpy(z.a(), y.b(), sizeof(u64) * W);
    std::memcpy(z.b(), y.a(), sizeof(u64) * W);
    return z;
}

// PT2 over swap pairs (states of spin parity sigma in a space closed under
// swaps): the numerator of an external a and of its swap a' differ by the
// factor sigma, as do their contributions from an internal pair, so only
// representatives a (alpha <= beta) are accumulated, from representative
// internals i (partner >= i): each excitation y of i adds s v c_i to the
// representative of y, s = 1 for y itself and sigma for a swap (which stands
// for the swap of i's excitation, an excitation of i's partner); y = y'
// gains from i and its partner (1 + sigma) v c_i; a closed internal (i = i')
// generates both members of each pair and adds only to representatives.
// Open representatives count twice in E2. Half the generation and half the
// hash-map entries of the determinant sums, the same E2.
struct PairTerm {
    bool keep = false;
    double factor = 1.0;
    bool swap = false;
};

template <int W> inline PairTerm pair_term(const Det<W> &y, bool open_i, double sigma)
{
    PairTerm t;
    const int order = string_order<W>(y);
    if (order < 0) {
        t.keep = true;
    } else if (order > 0) {
        t.keep = open_i;
        t.swap = true;
        t.factor = sigma;
    } else {
        t.factor = open_i ? 1.0 + sigma : 1.0;
        t.keep = t.factor != 0.0;
    }
    return t;
}

// Open addressing over the space's determinants: each slot holds a 32-bit
// hash tag and the determinant's index (-1 empty), so a lookup of a
// determinant outside the space (most PT2 externals and selection
// candidates) compares tags and rarely reads a determinant.
template <int W> struct DetIndex {
    const Det<W> *dets = nullptr;
    std::vector<int64_t> slot;
    u64 mask = 0;

    static int64_t entry(u64 h, int64_t i) { return static_cast<int64_t>(((h >> 32) << 32) | static_cast<u64>(i)); }

    void build(const Det<W> *d, int64_t count, int threads)
    {
        dets = d;
        u64 cap = 16;
        while (cap < static_cast<u64>(2 * count + 16)) cap <<= 1;
        mask = cap - 1;
        slot.assign(cap, -1);
#pragma omp parallel for schedule(static) num_threads(threads)
        for (int64_t i = 0; i < count; ++i) {
            const u64 h = hash_det<W>(d[i]);
            u64 k = h & mask;
            for (;;) {
                int64_t expected = -1;
                if (__atomic_compare_exchange_n(&slot[k], &expected, entry(h, i), false, __ATOMIC_RELAXED,
                                                __ATOMIC_RELAXED))
                    break;
                k = (k + 1) & mask;
            }
        }
    }

    int64_t find(const Det<W> &x, u64 h) const
    {
        const u64 tag = h >> 32;
        for (u64 k = h & mask;; k = (k + 1) & mask) {
            const int64_t s = slot[k];
            if (s == -1) return -1;
            if ((static_cast<u64>(s) >> 32) == tag) {
                const int64_t i = static_cast<int64_t>(static_cast<u64>(s) & 0xFFFFFFFFULL);
                if (same(dets[i], x)) return i;
            }
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
// shard keeps its entries (the key's 2W words, its hash, the accumulators)
// in one dense array in insertion order, and an open-addressing table of
// 8-byte slots (a 32-bit hash tag, the entry's index + 1; 0 empty) at most
// half full. New entries are appended (sequential writes), lookups compare
// tags before keys, scans read the filled entries only, and a reset (the next
// PT2 batch) clears the slots and keeps both arrays. Entry order depends on
// the thread schedule: readers sort the keys or sum the fixed-point values,
// which is exact in any order.
template <int W, class V = i128> struct ShardedMap {
    struct Shard {
        Spinlock lock;
        u64 *slots = nullptr;
        size_t cap = 0;                  // slots (a power of two)
        u64 *entries = nullptr;
        size_t count = 0, room = 0;      // entries used and allocated
    };
    int nvalues;
    size_t stride;                       // words per entry: key, hash, pad, accumulators (16-byte aligned)
    std::vector<Shard> shards;
    static constexpr int SHARD_BITS = 9;

    explicit ShardedMap(int nv)
        : nvalues(nv), stride(2 * W + 2 + nv * sizeof(V) / sizeof(u64)), shards(size_t(1) << SHARD_BITS)
    {
    }
    ~ShardedMap() { clear(); }
    ShardedMap(const ShardedMap &) = delete;
    ShardedMap &operator=(const ShardedMap &) = delete;

    static size_t shard_of(u64 h) { return h >> (64 - SHARD_BITS); }
    static u64 tag_of(u64 h) { return (h >> 20) & 0xFFFFFFFFULL; }
    u64 *entry(const Shard &s, size_t i) const { return s.entries + i * stride; }
    V *values_of(u64 *row) const { return reinterpret_cast<V *>(row + 2 * W + 2); }
    static const Det<W> &key_of(const u64 *row) { return *reinterpret_cast<const Det<W> *>(row); }
    // The hash stored after a key passed to a for_each callback.
    static u64 hash_of(const Det<W> &key) { return reinterpret_cast<const u64 *>(&key)[2 * W]; }

    // Arrays of 2 MB or more are 2 MB aligned and ask for transparent huge
    // pages: random accesses over gigabytes of 4 KB pages miss the TLB.
    static u64 *allocate(size_t words)
    {
        constexpr size_t HUGE_PAGE = size_t(1) << 21;
        size_t bytes = std::max<size_t>(words, 1) * sizeof(u64);
        const size_t align = bytes >= HUGE_PAGE ? HUGE_PAGE : 64;
        bytes = (bytes + align - 1) & ~(align - 1);
        u64 *t = static_cast<u64 *>(std::aligned_alloc(align, bytes));
        if (!t) throw std::bad_alloc();
        if (align == HUGE_PAGE) madvise(t, bytes, MADV_HUGEPAGE);
        return t;
    }

    void rehash(Shard &s, size_t cap)
    {
        u64 *slots = allocate(cap);
        std::memset(slots, 0, cap * sizeof(u64));
        const u64 mask = cap - 1;
        for (size_t i = 0; i < s.count; ++i) {
            const u64 h = entry(s, i)[2 * W];
            u64 k = h & mask;
            while (slots[k]) k = (k + 1) & mask;
            slots[k] = (tag_of(h) << 32) | (i + 1);
        }
        std::free(s.slots);
        s.slots = slots;
        s.cap = cap;
    }

    void reserve_entries(Shard &s, size_t room)
    {
        u64 *entries = allocate(room * stride);
        if (s.count) std::memcpy(entries, s.entries, s.count * stride * sizeof(u64));
        std::free(s.entries);
        s.entries = entries;
        s.room = room;
    }

    // Adds `add` (nvalues entries, may be null for a set) to key x's
    // accumulators; the caller holds the shard's lock.
    void insert_locked(Shard &s, const Det<W> &x, u64 h, const V *add)
    {
        if (2 * (s.count + 1) > s.cap) rehash(s, s.cap ? 2 * s.cap : 128);
        const u64 mask = s.cap - 1, tag = tag_of(h);
        u64 k = h & mask;
        u64 *row = nullptr;
        for (;; k = (k + 1) & mask) {
            const u64 slot = s.slots[k];
            if (!slot) break;
            if ((slot >> 32) == tag) {
                u64 *candidate = entry(s, (slot & 0xFFFFFFFFULL) - 1);
                if (same(key_of(candidate), x)) {
                    row = candidate;
                    break;
                }
            }
        }
        if (!row) {
            if (s.count == s.room) reserve_entries(s, s.room ? 2 * s.room : 64);
            row = entry(s, s.count);
            std::memcpy(row, x.w, sizeof(x.w));
            row[2 * W] = h;
            V *vals = values_of(row);
            for (int v = 0; v < nvalues; ++v) vals[v] = 0;
            s.slots[k] = (tag << 32) | (s.count + 1);
            ++s.count;
        }
        V *vals = values_of(row);
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
        for (size_t i = 0; i < sh.count; ++i) {
            u64 *row = entry(sh, i);
            f(key_of(row), values_of(row));
        }
    }

    // Per-thread staging: records grouped by shard and inserted DEPTH at a
    // time under one lock acquisition (the lock cache line moves between
    // cores once per DEPTH inserts), each flush prefetching its records'
    // slots before inserting them. flush_all() before the map is read. 16
    // keeps a thread's staging (512 shards) near its L2 cache with the PT2's
    // eight 16-byte accumulators per record (64 made the stochastic PT2 1.6x
    // slower; 8 slowed the deterministic one).
    struct Stage {
        static constexpr int DEPTH = 16;
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
                // Slots first, then the entries their home slots point to
                // (an insert into an existing entry waits on both).
                const u64 mask = sh.cap - 1;
                for (int t = 0; t < f; ++t) __builtin_prefetch(sh.slots + (hashes[s * DEPTH + t] & mask), 1);
                for (int t = 0; t < f; ++t) {
                    const u64 h = hashes[s * DEPTH + t], slot = sh.slots[h & mask];
                    if (slot && (slot >> 32) == tag_of(h)) __builtin_prefetch(map.entry(sh, (slot & 0xFFFFFFFFULL) - 1), 1);
                }
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
            std::free(s.slots);
            std::free(s.entries);
            s.slots = s.entries = nullptr;
            s.cap = s.count = s.room = 0;
        }
    }

    // Empties every shard but keeps its arrays, for the next batch of a PT2:
    // a fresh map grows each shard by doubling (rehashing and faulting in new
    // pages, several GB per batch at CuF size), while batches are alike in
    // size. Only the slots are cleared.
    void reset(int threads)
    {
#pragma omp parallel for schedule(dynamic, 4) num_threads(threads)
        for (size_t i = 0; i < shards.size(); ++i) {
            Shard &sh = shards[i];
            if (sh.slots) std::memset(sh.slots, 0, sh.cap * sizeof(u64));
            sh.count = 0;
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
    // The other string's id of each member, in the alpha-major (amajor_b)
    // and beta-major (bmajor_a) orders: the searches read these contiguous
    // ids and the small string tables instead of the determinants.
    std::vector<int32_t> amajor_b, bmajor_a;
    // Summed group sizes of each string's single-excitation neighbours: the
    // cost of finding a determinant's opposite-spin doubles from that side.
    std::vector<int64_t> asingle_cost, bsingle_cost;

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
        const int64_t nastr = astr.size() / W;
        amajor_b.resize(amajor.size());
        bmajor_a.resize(bmajor.size());
#pragma omp parallel for schedule(static) num_threads(threads)
        for (int64_t t = 0; t < static_cast<int64_t>(amajor.size()); ++t) {
            amajor_b[t] = bid[amajor[t]];
            bmajor_a[t] = aid[bmajor[t]];
        }
        auto costs = [&](const std::vector<int64_t> &single_start, const std::vector<int32_t> &single,
                         const std::vector<int64_t> &major_start, std::vector<int64_t> &out) {
            const int64_t nstr = static_cast<int64_t>(single_start.size()) - 1;
            out.assign(nstr, 0);
#pragma omp parallel for schedule(static) num_threads(threads)
            for (int64_t s = 0; s < nstr; ++s) {
                int64_t c = 0;
                for (int64_t u = single_start[s]; u < single_start[s + 1]; ++u)
                    c += major_start[single[u] + 1] - major_start[single[u]];
                out[s] = c;
            }
        };
        costs(asingle_start, asingle, amajor_start, asingle_cost);
        costs(bsingle_start, bsingle, bmajor_start, bsingle_cost);
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
template <int W, class F> void for_each_upper(const Space<W> &sp, int64_t i, int64_t from, F &&f,
                                               int64_t *stats = nullptr)
{
    // Every determinant j >= from (> i) connected to i: the per-string lists
    // hold increasing determinant indices, so each scan starts at `from`.
    const Det<W> &x = sp.dets[i];
    auto after = [&](const std::vector<int32_t> &list, int64_t lo, int64_t hi) {
        return std::lower_bound(list.begin() + lo, list.begin() + hi, static_cast<int32_t>(from)) - list.begin();
    };
    const int32_t a = sp.aid[i], b = sp.bid[i];
    {
        const int64_t t0 = after(sp.amajor, sp.amajor_start[a], sp.amajor_start[a + 1]);
        if (stats) stats[0] += sp.amajor_start[a + 1] - t0;
        const u64 *xb = x.b();
        const u64 *xa_self = &sp.astr[static_cast<int64_t>(a) * W];
        for (int64_t t = t0; t < sp.amajor_start[a + 1]; ++t) {
            const u64 *yb = &sp.bstr[static_cast<int64_t>(sp.amajor_b[t]) * W];
            int d = 0;
            for (int k = 0; k < W; ++k) d += popc(xb[k] ^ yb[k]);
            if (d <= 4) {
                f(sp.amajor[t], xa_self, yb);
                if (stats) ++stats[1];
            }
        }
    }
    {
        const int64_t t0 = after(sp.bmajor, sp.bmajor_start[b], sp.bmajor_start[b + 1]);
        if (stats) stats[2] += sp.bmajor_start[b + 1] - t0;
        const u64 *xa = x.a();
        const u64 *xb_self = &sp.bstr[static_cast<int64_t>(b) * W];
        for (int64_t t = t0; t < sp.bmajor_start[b + 1]; ++t) {
            const u64 *ya = &sp.astr[static_cast<int64_t>(sp.bmajor_a[t]) * W];
            int d = 0;
            for (int k = 0; k < W; ++k) d += popc(xa[k] ^ ya[k]);
            if (d <= 4) {
                f(sp.bmajor[t], ya, xb_self);
                if (stats) ++stats[3];
            }
        }
    }
    // Opposite-spin doubles j = (a2, b2), a2 one excitation from a and b2 one
    // from b: from the side whose neighbours' groups are shorter in total,
    // walk each neighbour's group (from `from` on, contiguous other-string
    // ids) and test the other string against the singles of this
    // determinant's other string, marked in a per-thread bitmap.
    static thread_local std::vector<u64> amark, bmark;
    const bool alpha_side = sp.asingle_cost[a] <= sp.bsingle_cost[b];
    const auto &near_start = alpha_side ? sp.asingle_start : sp.bsingle_start;
    const auto &near = alpha_side ? sp.asingle : sp.bsingle;
    const auto &far_start = alpha_side ? sp.bsingle_start : sp.asingle_start;
    const auto &far = alpha_side ? sp.bsingle : sp.asingle;
    const auto &major_start = alpha_side ? sp.amajor_start : sp.bmajor_start;
    const auto &major = alpha_side ? sp.amajor : sp.bmajor;
    const auto &major_other = alpha_side ? sp.amajor_b : sp.bmajor_a;
    auto &mark = alpha_side ? bmark : amark;
    const int32_t own = alpha_side ? a : b, other = alpha_side ? b : a;
    const size_t mark_words = (alpha_side ? sp.bstr.size() : sp.astr.size()) / W / 64 + 1;
    if (mark.size() < mark_words) mark.assign(mark_words, 0);
    const int64_t f_lo = far_start[other], f_hi = far_start[other + 1];
    for (int64_t v = f_lo; v < f_hi; ++v) mark[far[v] >> 6] |= u64(1) << (far[v] & 63);
    for (int64_t u = near_start[own]; u < near_start[own + 1]; ++u) {
        const int32_t s2 = near[u];
        const int64_t g_lo = major_start[s2], g_hi = major_start[s2 + 1];
        const int64_t first = after(major, g_lo, g_hi);
        if (stats) {
            ++stats[4];
            stats[5] += g_hi - first;
        }
        for (int64_t t = first; t < g_hi; ++t) {
            const int32_t o = major_other[t];
            if ((mark[o >> 6] >> (o & 63)) & 1) {
                const u64 *s2_str = &(alpha_side ? sp.astr : sp.bstr)[static_cast<int64_t>(s2) * W];
                const u64 *o_str = &(alpha_side ? sp.bstr : sp.astr)[static_cast<int64_t>(o) * W];
                f(major[t], alpha_side ? s2_str : o_str, alpha_side ? o_str : s2_str);
                if (stats) ++stats[7];
            }
        }
    }
    for (int64_t v = f_lo; v < f_hi; ++v) mark[far[v] >> 6] = 0;
}

// Column indices are stored as the gaps between a row's consecutive columns
// (the first from the row index), minus one, in a grouped variable-byte code:
// a control byte holds the byte lengths (1-4) of the next four gaps, followed
// by their little-endian bytes (a row's last group is padded with zero gaps).
// Sorted spaces give mostly one- and two-byte gaps (CuF, 1.8 million
// determinants: about 1.5 bytes per column instead of 4), and the gaps' byte
// positions follow from the control byte alone, so a row decodes without a
// chain through every entry (with SSSE3, one byte shuffle per group). Readers
// load 16 bytes per group: each chunk's code ends with 16 padding bytes.
namespace columns {
struct Tables {
    uint8_t length[256];            // bytes of a group's four gaps
    uint8_t offset[256][4];         // each gap's first byte within the group
    alignas(16) uint8_t shuffle[256][16];   // the group's bytes -> four little-endian uint32 gaps
    Tables()
    {
        for (int c = 0; c < 256; ++c) {
            int at = 0;
            for (int k = 0; k < 4; ++k) {
                const int bytes = ((c >> (2 * k)) & 3) + 1;
                offset[c][k] = static_cast<uint8_t>(at);
                for (int b = 0; b < 4; ++b)
                    shuffle[c][4 * k + b] = b < bytes ? static_cast<uint8_t>(at + b) : 0x80;
                at += bytes;
            }
            length[c] = static_cast<uint8_t>(at);
        }
    }
};
inline const Tables &tables()
{
    static const Tables t;
    return t;
}
constexpr uint32_t MASK[4] = {0xFFu, 0xFFFFu, 0xFFFFFFu, 0xFFFFFFFFu};
constexpr int PADDING = 16;

inline uint32_t load32(const uint8_t *p)
{
    uint32_t x;
    std::memcpy(&x, p, 4);
    return x;
}

// Appends row i's code for its m increasing columns (all > i).
inline void encode(std::vector<uint8_t> &code, int64_t i, const int32_t *col, int64_t m)
{
    const int64_t groups = (m + 3) >> 2;
    const size_t base = code.size();
    code.resize(base + groups, 0);
    uint32_t previous = static_cast<uint32_t>(i);
    for (int64_t t = 0; t < 4 * groups; ++t) {
        uint32_t gap = 0;
        if (t < m) {
            gap = static_cast<uint32_t>(col[t]) - previous - 1;
            previous = static_cast<uint32_t>(col[t]);
        }
        const int bytes = gap < (1u << 8) ? 1 : gap < (1u << 16) ? 2 : gap < (1u << 24) ? 3 : 4;
        code[base + (t >> 2)] |= static_cast<uint8_t>((bytes - 1) << (2 * (t & 3)));
        for (int b = 0; b < bytes; ++b) code.push_back(static_cast<uint8_t>(gap >> (8 * b)));
    }
}

// Row i's m columns into out (room for m rounded up to a multiple of 4).
inline void decode(const uint8_t *p, int64_t i, int64_t m, int32_t *out)
{
    const Tables &T = tables();
    const int64_t groups = (m + 3) >> 2;
    const uint8_t *data = p + groups;
#if defined(__SSSE3__)
    const __m128i one = _mm_set1_epi32(1);
    __m128i base = _mm_set1_epi32(static_cast<int32_t>(i));
    for (int64_t g = 0; g < groups; ++g) {
        const unsigned c = p[g];
        __m128i d = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(data)),
                                     _mm_load_si128(reinterpret_cast<const __m128i *>(T.shuffle[c])));
        d = _mm_add_epi32(d, one);
        d = _mm_add_epi32(d, _mm_slli_si128(d, 4));
        d = _mm_add_epi32(d, _mm_slli_si128(d, 8));
        d = _mm_add_epi32(d, base);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(out + 4 * g), d);
        base = _mm_shuffle_epi32(d, 0xFF);
        data += T.length[c];
    }
    return;
#endif
    uint32_t j = static_cast<uint32_t>(i);
    for (int64_t g = 0; g < groups; ++g) {
        const unsigned c = p[g];
        const uint8_t *o = T.offset[c];
        const uint32_t d0 = load32(data + o[0]) & MASK[c & 3];
        const uint32_t d1 = load32(data + o[1]) & MASK[(c >> 2) & 3];
        const uint32_t d2 = load32(data + o[2]) & MASK[(c >> 4) & 3];
        const uint32_t d3 = load32(data + o[3]) & MASK[c >> 6];
        data += T.length[c];
        j += d0 + 1;
        out[4 * g] = static_cast<int32_t>(j);
        j += d1 + 1;
        out[4 * g + 1] = static_cast<int32_t>(j);
        j += d2 + 1;
        out[4 * g + 2] = static_cast<int32_t>(j);
        j += d3 + 1;
        out[4 * g + 3] = static_cast<int32_t>(j);
    }
}
}  // namespace columns

// Rows are kept in chunks of CHUNK rows, each with its own arrays (no copy
// into one array after the build, so a build peaks at the matrix's size), and
// an extension releases each old chunk once its rows are copied.
//
// With `single` the off-diagonal values are stored as float (products still
// accumulate in double): a third less memory and memory traffic per product,
// with relative errors of 6e-8 in each element (about 1e-8 Eh in energies).
struct Hamiltonian {
    static constexpr int64_t STRIPES = 64;
    static constexpr int64_t CHUNK = 512;
    int64_t n = 0, nnz = 0;
    bool single = false;
    std::vector<double> diag;
    std::vector<std::vector<int64_t>> start;   // per chunk: CHUNK + 1 (or fewer) local row offsets
    std::vector<std::vector<uint32_t>> code_start;   // per chunk: each row's first byte in code
    std::vector<std::vector<uint8_t>> code;          // per chunk: the rows' column code (see columns)
    std::vector<std::vector<double>> val;      // values (single == false)
    std::vector<std::vector<float>> val32;     // values (single == true)
    std::vector<int64_t> stripe;               // STRIPES + 1 row boundaries
    std::vector<int64_t> offset;               // each stripe's buffer offset (rows)
    std::vector<double> buffer;                // allocated by the first product, width vectors per row
    int64_t buffer_width = 0;

    int64_t bytes() const
    {
        int64_t b = static_cast<int64_t>(diag.size() + buffer.size()) * 8;
        for (const auto &v : start) b += static_cast<int64_t>(v.capacity()) * 8;
        for (const auto &v : code_start) b += static_cast<int64_t>(v.capacity()) * 4;
        for (const auto &v : code) b += static_cast<int64_t>(v.capacity());
        for (const auto &v : val) b += static_cast<int64_t>(v.capacity()) * 8;
        for (const auto &v : val32) b += static_cast<int64_t>(v.capacity()) * 4;
        return b;
    }

    int64_t row_length(int64_t i) const
    {
        const auto &st = start[i / CHUNK];
        return st[i % CHUNK + 1] - st[i % CHUNK];
    }

    // Row i's columns into out (room for row_length(i) rounded up to 4).
    void columns_of(int64_t i, int32_t *out) const
    {
        const int64_t ch = i / CHUNK, local = i % CHUNK;
        columns::decode(code[ch].data() + code_start[ch][local], i, start[ch][local + 1] - start[ch][local], out);
    }

    int64_t longest = 0;                       // longest row (set by finish_stripes)
    int64_t longest_row() const { return longest; }
};

// Appends one row (its columns, increasing) to a chunk's column code.
inline void append_row_code(std::vector<uint8_t> &code, std::vector<uint32_t> &code_start, int64_t local, int64_t i,
                            const int32_t *col, int64_t m)
{
    code_start[local] = static_cast<uint32_t>(code.size());
    columns::encode(code, i, col, m);
    code_start[local + 1] = static_cast<uint32_t>(code.size());
}

inline void finish_chunk_code(std::vector<uint8_t> &code)
{
    code.insert(code.end(), columns::PADDING, 0);
    code.shrink_to_fit();
}

struct Buffer {
    int64_t rows = 0, width = 0;
    std::vector<u64> data;
};

template <int W> Space<W> *as_space(void *p) { return static_cast<Space<W> *>(static_cast<SpaceBase *>(p)); }

// Stripes of about equal rows plus nonzeros for the product, and their buffers.
void finish_stripes(Hamiltonian &H)
{
    const int64_t count = H.n;
    H.stripe.assign(Hamiltonian::STRIPES + 1, count);
    H.stripe[0] = 0;
    int64_t i = 0, cumulative = 0;          // nonzeros of rows before i
    for (int64_t s = 1; s < Hamiltonian::STRIPES; ++s) {
        const int64_t goal = (H.nnz + count) * s / Hamiltonian::STRIPES;
        while (i < count && cumulative + i < goal) cumulative += H.row_length(i++);
        H.stripe[s] = i;
    }
    H.offset.assign(Hamiltonian::STRIPES + 1, 0);
    for (int64_t s = 0; s < Hamiltonian::STRIPES; ++s) H.offset[s + 1] = H.offset[s] + (count - H.stripe[s]);
    H.longest = 0;
    for (int64_t r = 0; r < count; ++r) H.longest = std::max(H.longest, H.row_length(r));
    std::vector<double>().swap(H.buffer);
    H.buffer_width = 0;
}

// y = H x for K vectors at once (x, y: count x K, row-major), reading H once.
// Each of the fixed STRIPES row ranges is one task: its rows' own entries go
// to y, the transposed (lower-triangle) ones to the stripe's buffer, and the
// buffers are added in stripe order (results independent of the thread count).
// Entries ahead whose vector and buffer lines are requested early (16-64 measured
// alike on a 12-core desktop; 32 is the middle).
static constexpr int64_t PREFETCH = 32;

template <class V, int K>
void spmm(Hamiltonian &H, const std::vector<std::vector<V>> &vals, const double *x, double *y, int threads)
{
    const int64_t count = H.n;
    if (H.buffer_width != K) {
        H.buffer.assign(H.offset[Hamiltonian::STRIPES] * K, 0.0);
        H.buffer_width = K;
    }
    const int64_t longest = H.longest_row();
#pragma omp parallel num_threads(threads)
    {
        std::vector<int32_t> columns(longest + 4);      // one row's decoded columns
#pragma omp for schedule(dynamic, 1)
        for (int64_t s = 0; s < Hamiltonian::STRIPES; ++s) {
            const int64_t lo = H.stripe[s], hi = H.stripe[s + 1];
            double *buf = H.buffer.data() + (H.offset[s] - lo) * K;      // buf[j * K + r] for j >= lo
            std::fill(buf + lo * K, buf + count * K, 0.0);
            for (int64_t i = lo; i < hi; ++i) {
                const int64_t ch = i / Hamiltonian::CHUNK, local = i % Hamiltonian::CHUNK;
                const int64_t *st = H.start[ch].data();
                const int64_t m = st[local + 1] - st[local];
                const V *val = vals[ch].data() + st[local];
                const int32_t *col = columns.data();
                H.columns_of(i, columns.data());
                double xi[K], v[K];
                for (int r = 0; r < K; ++r) {
                    xi[r] = x[i * K + r];
                    v[r] = H.diag[i] * xi[r];
                }
                for (int64_t t = 0; t < m; ++t) {
                    // The columns are known ahead: request the vector and
                    // buffer lines early (the product waits on these reads).
                    if (t + PREFETCH < m) {
                        const int64_t jp = col[t + PREFETCH];
                        __builtin_prefetch(x + jp * K, 0, 1);
                        __builtin_prefetch(buf + jp * K, 1, 1);
                    }
                    const int64_t j = col[t];
                    const double h = val[t];
                    for (int r = 0; r < K; ++r) {
                        v[r] += h * x[j * K + r];
                        buf[j * K + r] += h * xi[r];
                    }
                }
                for (int r = 0; r < K; ++r) y[i * K + r] = v[r];
            }
        }
#pragma omp for schedule(static)
        for (int64_t j = 0; j < count; ++j) {
            double v[K];
            for (int r = 0; r < K; ++r) v[r] = y[j * K + r];
            for (int64_t s = 0; s < Hamiltonian::STRIPES && H.stripe[s] <= j; ++s) {
                const double *b = H.buffer.data() + (H.offset[s] + (j - H.stripe[s])) * K;
                for (int r = 0; r < K; ++r) v[r] += b[r];
            }
            for (int r = 0; r < K; ++r) y[j * K + r] = v[r];
        }
    }
}

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
                              int64_t n_old = 0, bool single = false)
{
    sp.build_strings(threads);
    const Ctx &c = *sp.ctx;
    const int64_t count = sp.dets.size();
    if (!old) n_old = 0;
    auto *H = new Hamiltonian;
    H->n = count;
    H->single = single;
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
    H->code_start.resize(nchunks);
    H->code.resize(nchunks);
    H->val.resize(nchunks);
    H->val32.resize(nchunks);
    // The extension reads old chunks and releases each one after its rows are
    // copied (old rows are a prefix, chunked the same way; their values are
    // converted when the precisions differ).
    auto *old_mut = const_cast<Hamiltonian *>(old);
    // ITERCC_SCI_PROFILE=3: connection-search counters on stderr; =4 also
    // skips the matrix elements (search cost alone; the matrix is not usable).
    const char *profile_mode = std::getenv("ITERCC_SCI_PROFILE");
    const bool profile = profile_mode && (profile_mode[0] == '3' || profile_mode[0] == '4');
    const bool search_only = profile_mode && profile_mode[0] == '4';
    std::vector<int64_t> totals(8, 0);
    const double t_start = omp_get_wtime();
#pragma omp parallel num_threads(threads)
    {
        std::vector<std::pair<int32_t, double>> row;
        std::vector<int32_t> cols;
        int64_t local_stats[8] = {0};
        int64_t *stats = profile ? local_stats : nullptr;
#pragma omp for schedule(dynamic, 1)
        for (int64_t ch = 0; ch < nchunks; ++ch) {
            const int64_t lo = ch * chunk, hi = std::min(count, lo + chunk);
            auto &st = H->start[ch];
            auto &cs = H->code_start[ch];
            auto &code = H->code[ch];
            auto &cv = H->val[ch];
            auto &cf = H->val32[ch];
            st.assign(hi - lo + 1, 0);
            cs.assign(hi - lo + 1, 0);
            int64_t entries = 0;
            for (int64_t i = lo; i < hi; ++i) {
                row.clear();
                int occ[2 * MAX_ORB];
                const int na = occupied<W>(sp.dets[i].a(), occ);
                const int nb = occupied<W>(sp.dets[i].b(), occ + na);
                const bool kept = i < n_old;
                for_each_upper(sp, i, kept ? n_old : i + 1, [&](int32_t j, const u64 *ya, const u64 *yb) {
                    if (search_only) return;
                    const double v = element_strings<W>(c, sp.dets[i].a(), sp.dets[i].b(), ya, yb, occ, na, nb);
                    if (std::fabs(v) > drop) row.emplace_back(j, v);
                }, stats);
                std::sort(row.begin(), row.end(),
                          [](const std::pair<int32_t, double> &x, const std::pair<int32_t, double> &y) {
                              return x.first < y.first;
                          });
                cols.clear();
                if (kept) {         // the old row's columns all precede the new ones
                    const auto &os = old->start[ch];
                    const int64_t a = os[i - lo], b = os[i - lo + 1];
                    cols.resize(b - a + 4);
                    old->columns_of(i, cols.data());
                    cols.resize(b - a);
                    if (old->single) {
                        const float *ov = old->val32[ch].data();
                        for (int64_t t = a; t < b; ++t) {
                            if (single) cf.push_back(ov[t]);
                            else cv.push_back(ov[t]);
                        }
                    } else {
                        const double *ov = old->val[ch].data();
                        for (int64_t t = a; t < b; ++t) {
                            if (single) cf.push_back(static_cast<float>(ov[t]));
                            else cv.push_back(ov[t]);
                        }
                    }
                }
                for (auto &e : row) {
                    cols.push_back(e.first);
                    if (single) cf.push_back(static_cast<float>(e.second));
                    else cv.push_back(e.second);
                }
                append_row_code(code, cs, i - lo, i, cols.data(), static_cast<int64_t>(cols.size()));
                entries += static_cast<int64_t>(cols.size());
                st[i - lo + 1] = entries;
            }
            finish_chunk_code(code);
            cv.shrink_to_fit();
            cf.shrink_to_fit();
            if (old_mut && ch < static_cast<int64_t>(old_mut->code.size())) {
                std::vector<int64_t>().swap(old_mut->start[ch]);
                std::vector<uint32_t>().swap(old_mut->code_start[ch]);
                std::vector<uint8_t>().swap(old_mut->code[ch]);
                std::vector<double>().swap(old_mut->val[ch]);
                if (ch < static_cast<int64_t>(old_mut->val32.size())) std::vector<float>().swap(old_mut->val32[ch]);
            }
        }
        if (profile) {
#pragma omp critical
            for (int k = 0; k < 8; ++k) totals[k] += local_stats[k];
        }
    }
    if (profile)
        std::fprintf(stderr,
                     "itercc_sci_hamiltonian: %lld rows (%lld new) in %.2f s%s: same-alpha scanned %lld hits %lld; "
                     "same-beta scanned %lld hits %lld; opposite-spin groups %lld scanned %lld merged %lld hits %lld\n",
                     static_cast<long long>(count), static_cast<long long>(count - n_old), omp_get_wtime() - t_start,
                     search_only ? " (search only)" : "", static_cast<long long>(totals[0]),
                     static_cast<long long>(totals[1]), static_cast<long long>(totals[2]),
                     static_cast<long long>(totals[3]), static_cast<long long>(totals[4]),
                     static_cast<long long>(totals[5]), static_cast<long long>(totals[6]),
                     static_cast<long long>(totals[7]));
    int64_t nnz = 0;
    for (int64_t ch = 0; ch < nchunks; ++ch) nnz += H->start[ch].back();
    H->nnz = nnz;
    finish_stripes(*H);
    return H;
}

template <int W> void partners_impl(Space<W> &sp, int64_t *out, int threads);

// The image of a determinant under the orbital reflection (mirror[p] for
// each occupied p in both strings) and its phase: the sign of the
// permutation that orders each string's reflected creators again.
template <int W> inline double mirror_det(const Ctx &c, const Det<W> &x, Det<W> &out)
{
    int inversions = 0;
    for (int spin = 0; spin < 2; ++spin) {
        const u64 *xs = x.w + spin * W;
        u64 *ys = out.w + spin * W;
        for (int k = 0; k < W; ++k) ys[k] = 0;
        for (int k = 0; k < W; ++k)
            for (u64 bits = xs[k]; bits; bits &= bits - 1) {
                const int q = c.mirror[(k << 6) + lowest(bits)];
                // Earlier (lower) orbitals whose images lie above q.
                const int wq = q >> 6, bq = q & 63;
                int above = popc(bq == 63 ? 0 : ys[wq] >> (bq + 1));
                for (int m = wq + 1; m < W; ++m) above += popc(ys[m]);
                inversions += above;
                ys[wq] |= u64(1) << bq;
            }
    }
    return inversions & 1 ? -1.0 : 1.0;
}

// Symmetry-adapted basis over the orbits of the space's determinants under
// the alpha-beta swap T (spin parity sigma_t; none for 0) and the orbital
// reflection M (parity sigma_m; none for 0; needs the context's mirror). A
// state of these parities has amplitudes c(gD) phase_g(D) = chi(g) c(D), so
// it lives on one function per orbit, the normalized projection
//   P' = sum_g chi(g) g D = sum_K coef_K K   (D the orbit's lowest index),
// absent when it vanishes (a closed D for sigma_t = -1, ...). With H invariant
// under the group, <P|H|Q> = sum_{K in Q} coef_K <D|H|K> / sqrt(coef_D coef_E)
// (E Q's representative, coef_D = <D|P'>, |P'|^2 = |G| coef_D). Row P needs
// only the K > D (each K of an orbit Q >= P has K >= Q's representative >= D):
// the upper-triangle search of D's row, on one row per orbit; members of P
// itself go to the diagonal. Orbits are numbered by their representatives, so
// appending complete orbits keeps the old numbers. Spin pairs halve the rows
// and about halve the nonzeros; the reflection halves Sigma states again.
struct OrbitMap {
    std::vector<int64_t> reps;      // orbit -> representative determinant
    std::vector<int32_t> orbit;     // determinant -> its orbit (-1: none)
    std::vector<double> coef;       // coefficient of the determinant in its orbit's P'
};

template <int W>
bool orbit_map(Space<W> &sp, double sigma_t, double sigma_m, OrbitMap &om, int threads)
{
    const Ctx &c = *sp.ctx;
    const int64_t count = sp.dets.size();
    const bool use_t = sigma_t != 0.0, use_m = sigma_m != 0.0;
    if (use_m && c.mirror.empty()) return false;
    std::vector<int64_t> partner(count), image(count), rep(count);
    std::vector<double> phase(count, 1.0), self(count, 0.0);
    if (use_t) partners_impl(sp, partner.data(), threads);
    bool closed = true;
#pragma omp parallel for schedule(static) num_threads(threads) reduction(&& : closed)
    for (int64_t i = 0; i < count; ++i) {
        if (!use_t) partner[i] = i;
        image[i] = i;
        if (use_m) {
            Det<W> y;
            phase[i] = mirror_det<W>(c, sp.dets[i], y);
            image[i] = sp.index.find(y, hash_det<W>(y));
        }
        if (partner[i] < 0 || image[i] < 0) closed = false;
    }
    if (!closed) return false;
    // Images of D: e, T, M, TM (TM D = T applied to M D, phase that of M).
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t i = 0; i < count; ++i) {
        const int64_t t = partner[i], m = image[i], tm = partner[image[i]];
        rep[i] = std::min(std::min(i, t), std::min(m, tm));
        double cs = 1.0;
        if (use_t && t == i) cs += sigma_t;
        if (use_m && m == i) cs += sigma_m * phase[i];
        if (use_t && use_m && tm == i) cs += sigma_t * sigma_m * phase[i];
        self[i] = cs;
    }
    om.orbit.assign(count, -1);
    om.coef.assign(count, 0.0);
    om.reps.clear();
    for (int64_t i = 0; i < count; ++i)
        if (rep[i] == i && std::fabs(self[i]) > 0.5) {
            om.orbit[i] = static_cast<int32_t>(om.reps.size());
            om.reps.push_back(i);
        }
    const int64_t norbits = om.reps.size();
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t P = 0; P < norbits; ++P) {
        const int64_t i = om.reps[P];
        const int32_t o = static_cast<int32_t>(P);
        auto add = [&](int64_t k, double v) {
            om.orbit[k] = o;
            om.coef[k] += v;
        };
        // coef[i] is written by the e term (and by any g that fixes D).
        om.coef[i] = 0.0;
        add(i, 1.0);
        if (use_t) add(partner[i], sigma_t);
        if (use_m) add(image[i], sigma_m * phase[i]);
        if (use_t && use_m) add(partner[image[i]], sigma_t * sigma_m * phase[i]);
    }
    return true;
}

// The Hamiltonian over the orbits of the space (spin parity sigma_t,
// reflection parity sigma_m; see OrbitMap); with `old` (that of the first
// n_old determinants, complete orbits) old rows keep their elements and gain
// the new orbits' columns.
template <int W>
Hamiltonian *pair_hamiltonian_impl(Space<W> &sp, double sigma_t, double sigma_m, double drop, int threads,
                                   const Hamiltonian *old = nullptr, int64_t n_old = 0, bool single = false)
{
    sp.build_strings(threads);
    const Ctx &c = *sp.ctx;
    OrbitMap pm;
    if (!orbit_map(sp, sigma_t, sigma_m, pm, threads)) return nullptr;
    const int64_t count = pm.reps.size();
    if (!old) n_old = 0;
    const int64_t old_pairs = old ? std::lower_bound(pm.reps.begin(), pm.reps.end(), n_old) - pm.reps.begin() : 0;
    if (old && old->n != old_pairs) return nullptr;
    auto *H = new Hamiltonian;
    H->n = count;
    H->single = single;
    H->diag.assign(count, 0.0);
    if (old) std::copy(old->diag.begin(), old->diag.begin() + old_pairs, H->diag.begin());
    const int64_t chunk = Hamiltonian::CHUNK;
    const int64_t nchunks = (count + chunk - 1) / chunk;
    H->start.resize(nchunks);
    H->code_start.resize(nchunks);
    H->code.resize(nchunks);
    H->val.resize(nchunks);
    H->val32.resize(nchunks);
    auto *old_mut = const_cast<Hamiltonian *>(old);
#pragma omp parallel num_threads(threads)
    {
        std::vector<std::pair<int32_t, double>> row;
        std::vector<int32_t> cols;
#pragma omp for schedule(dynamic, 1)
        for (int64_t ch = 0; ch < nchunks; ++ch) {
            const int64_t lo = ch * chunk, hi = std::min(count, lo + chunk);
            auto &st = H->start[ch];
            auto &cs = H->code_start[ch];
            auto &code = H->code[ch];
            auto &cv = H->val[ch];
            auto &cf = H->val32[ch];
            st.assign(hi - lo + 1, 0);
            cs.assign(hi - lo + 1, 0);
            int64_t entries = 0;
            for (int64_t P = lo; P < hi; ++P) {
                const int64_t i = pm.reps[P];
                row.clear();
                int occ[2 * MAX_ORB];
                const int na = occupied<W>(sp.dets[i].a(), occ);
                const int nb = occupied<W>(sp.dets[i].b(), occ + na);
                const bool kept = P < old_pairs;
                double partner_term = 0.0;
                for_each_upper(sp, i, kept ? n_old : i + 1, [&](int32_t j, const u64 *ya, const u64 *yb) {
                    const int32_t Q = pm.orbit[j];
                    if (Q < P || pm.coef[j] == 0.0) return;  // lower triangle (or no amplitude)
                    const double v = element_strings<W>(c, sp.dets[i].a(), sp.dets[i].b(), ya, yb, occ, na, nb);
                    if (v == 0.0) return;
                    if (Q == P) {
                        partner_term += pm.coef[j] * v;
                        return;
                    }
                    row.emplace_back(Q, pm.coef[j] * v);
                });
                if (!kept) {
                    int oa[MAX_ORB], ob[MAX_ORB];
                    const int nao = occupied<W>(sp.dets[i].a(), oa), nbo = occupied<W>(sp.dets[i].b(), ob);
                    H->diag[P] = diagonal_energy(c, oa, nao, ob, nbo) + partner_term / pm.coef[i];
                }
                std::sort(row.begin(), row.end(),
                          [](const std::pair<int32_t, double> &x, const std::pair<int32_t, double> &y) {
                              return x.first < y.first;
                          });
                cols.clear();
                if (kept) {         // the old row's columns all precede the new ones
                    const auto &os = old->start[ch];
                    const int64_t a = os[P - lo], b = os[P - lo + 1];
                    cols.resize(b - a + 4);
                    old->columns_of(P, cols.data());
                    cols.resize(b - a);
                    if (old->single) {
                        const float *ov = old->val32[ch].data();
                        for (int64_t t = a; t < b; ++t) {
                            if (single) cf.push_back(ov[t]);
                            else cv.push_back(ov[t]);
                        }
                    } else {
                        const double *ov = old->val[ch].data();
                        for (int64_t t = a; t < b; ++t) {
                            if (single) cf.push_back(static_cast<float>(ov[t]));
                            else cv.push_back(ov[t]);
                        }
                    }
                }
                const double cd = pm.coef[i];
                for (size_t t = 0; t < row.size();) {
                    const int32_t Q = row[t].first;
                    double v = 0.0;
                    for (; t < row.size() && row[t].first == Q; ++t) v += row[t].second;
                    v /= std::sqrt(cd * pm.coef[pm.reps[Q]]);
                    if (std::fabs(v) <= drop) continue;
                    cols.push_back(Q);
                    if (single) cf.push_back(static_cast<float>(v));
                    else cv.push_back(v);
                }
                append_row_code(code, cs, P - lo, P, cols.data(), static_cast<int64_t>(cols.size()));
                entries += static_cast<int64_t>(cols.size());
                st[P - lo + 1] = entries;
            }
            finish_chunk_code(code);
            cv.shrink_to_fit();
            cf.shrink_to_fit();
            if (old_mut && ch < static_cast<int64_t>(old_mut->code.size())) {
                std::vector<int64_t>().swap(old_mut->start[ch]);
                std::vector<uint32_t>().swap(old_mut->code_start[ch]);
                std::vector<uint8_t>().swap(old_mut->code[ch]);
                std::vector<double>().swap(old_mut->val[ch]);
                if (ch < static_cast<int64_t>(old_mut->val32.size())) std::vector<float>().swap(old_mut->val32[ch]);
            }
        }
    }
    int64_t nnz = 0;
    for (int64_t ch = 0; ch < nchunks; ++ch) nnz += H->start[ch].back();
    H->nnz = nnz;
    finish_stripes(*H);
    return H;
}

// The matrix of the one-electron operator sum_pq d_pq E_pq (d symmetric, n x n)
// in the space: diagonal sum_p d_pp n_p, and the single excitations, found
// among the determinants sharing the other spin's string.
template <int W>
Hamiltonian *onebody_impl(Space<W> &sp, const double *d, double drop, int threads, Hamiltonian *old = nullptr,
                          int64_t n_old = 0)
{
    sp.build_strings(threads);
    const int64_t n = sp.ctx->n, count = sp.dets.size();
    if (!old) n_old = 0;
    auto *H = new Hamiltonian;
    H->n = count;
    H->diag.resize(count);
    if (old) std::copy(old->diag.begin(), old->diag.begin() + n_old, H->diag.begin());
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t i = n_old; i < count; ++i) {
        double v = 0.0;
        for (int spin = 0; spin < 2; ++spin)
            for (int k = 0; k < W; ++k)
                for (u64 bits = sp.dets[i].w[spin * W + k]; bits; bits &= bits - 1) {
                    const int p = (k << 6) + lowest(bits);
                    v += d[p * n + p];
                }
        H->diag[i] = v;
    }
    const int64_t chunk = Hamiltonian::CHUNK, nchunks = (count + chunk - 1) / chunk;
    H->start.resize(nchunks);
    H->code_start.resize(nchunks);
    H->code.resize(nchunks);
    H->val.resize(nchunks);
#pragma omp parallel num_threads(threads)
    {
        std::vector<std::pair<int32_t, double>> row;
        std::vector<int32_t> cols;
#pragma omp for schedule(dynamic, 1)
        for (int64_t ch = 0; ch < nchunks; ++ch) {
            const int64_t lo = ch * chunk, hi = std::min(count, lo + chunk);
            auto &st = H->start[ch];
            auto &cs = H->code_start[ch];
            auto &code = H->code[ch];
            auto &cv = H->val[ch];
            st.assign(hi - lo + 1, 0);
            cs.assign(hi - lo + 1, 0);
            int64_t entries = 0;
            for (int64_t i = lo; i < hi; ++i) {
                row.clear();
                const Det<W> &x = sp.dets[i];
                const bool kept = i < n_old;
                const int32_t from = static_cast<int32_t>(kept ? n_old : i + 1);
                auto visit = [&](int32_t j, int spin) {
                    const u64 *xs = x.w + spin * W, *ys = sp.dets[j].w + spin * W;
                    u64 hole[W], part[W];
                    for (int k = 0; k < W; ++k) {
                        hole[k] = xs[k] & ~ys[k];
                        part[k] = ys[k] & ~xs[k];
                    }
                    int pp = 0, rr = 0;
                    set_bits<W>(hole, &pp, 1);
                    set_bits<W>(part, &rr, 1);
                    const double v = phase<W>(xs, pp, rr) * d[pp * n + rr];
                    if (std::fabs(v) > drop) row.emplace_back(j, v);
                };
                auto after = [&](const std::vector<int32_t> &list, int64_t a, int64_t b) {
                    return std::lower_bound(list.begin() + a, list.begin() + b, from) - list.begin();
                };
                const int32_t a = sp.aid[i], b = sp.bid[i];
                for (int64_t t = after(sp.amajor, sp.amajor_start[a], sp.amajor_start[a + 1]);
                     t < sp.amajor_start[a + 1]; ++t) {
                    const int32_t j = sp.amajor[t];
                    int dd = 0;
                    for (int k = 0; k < W; ++k) dd += popc(x.b()[k] ^ sp.dets[j].b()[k]);
                    if (dd == 2) visit(j, 1);
                }
                for (int64_t t = after(sp.bmajor, sp.bmajor_start[b], sp.bmajor_start[b + 1]);
                     t < sp.bmajor_start[b + 1]; ++t) {
                    const int32_t j = sp.bmajor[t];
                    int dd = 0;
                    for (int k = 0; k < W; ++k) dd += popc(x.a()[k] ^ sp.dets[j].a()[k]);
                    if (dd == 2) visit(j, 0);
                }
                std::sort(row.begin(), row.end(),
                          [](const std::pair<int32_t, double> &u, const std::pair<int32_t, double> &v) {
                              return u.first < v.first;
                          });
                cols.clear();
                if (kept) {         // the old row's columns all precede the new ones
                    const auto &os = old->start[ch];
                    const int64_t a = os[i - lo], b = os[i - lo + 1];
                    cols.resize(b - a + 4);
                    old->columns_of(i, cols.data());
                    cols.resize(b - a);
                    cv.insert(cv.end(), old->val[ch].begin() + a, old->val[ch].begin() + b);
                }
                for (auto &e : row) {
                    cols.push_back(e.first);
                    cv.push_back(e.second);
                }
                append_row_code(code, cs, i - lo, i, cols.data(), static_cast<int64_t>(cols.size()));
                entries += static_cast<int64_t>(cols.size());
                st[i - lo + 1] = entries;
            }
            finish_chunk_code(code);
            cv.shrink_to_fit();
            if (old && ch < static_cast<int64_t>(old->code.size())) {
                std::vector<int64_t>().swap(old->start[ch]);
                std::vector<uint32_t>().swap(old->code_start[ch]);
                std::vector<uint8_t>().swap(old->code[ch]);
                std::vector<double>().swap(old->val[ch]);
            }
        }
    }
    int64_t nnz = 0;
    for (int64_t ch = 0; ch < nchunks; ++ch) nnz += H->start[ch].back();
    H->nnz = nnz;
    finish_stripes(*H);
    return H;
}

// Determinants outside the space a = E_rp x (single excitations) with
// |d_pr| cmax_i >= eps (and their alpha-beta swaps with `partners`).
template <int W>
Buffer *select_onebody_impl(Space<W> &sp, const double *d, const double *cmax, double eps, int partners, int threads)
{
    const int64_t n = sp.ctx->n, count = sp.dets.size();
    std::vector<std::vector<std::pair<double, int>>> lists(n);
    for (int64_t q = 0; q < n; ++q) {
        for (int64_t r = 0; r < n; ++r)
            if (r != q && d[q * n + r] != 0.0) lists[q].emplace_back(std::fabs(d[q * n + r]), static_cast<int>(r));
        std::sort(lists[q].begin(), lists[q].end(),
                  [](const std::pair<double, int> &u, const std::pair<double, int> &v) {
                      return u.first != v.first ? u.first > v.first : u.second < v.second;
                  });
    }
    const Zobrist &z = zobrist();
    ShardedMap<W, int64_t> found(0);
#pragma omp parallel num_threads(threads)
    {
        typename ShardedMap<W, int64_t>::Stage stage(found);
#pragma omp for schedule(dynamic, 16)
        for (int64_t i = 0; i < count; ++i) {
            if (cmax[i] <= 0.0) continue;
            const double t = eps / cmax[i];
            const Det<W> &x = sp.dets[i];
            const u64 hx = hash_det<W>(x);
            for (int spin = 0; spin < 2; ++spin) {
                const u64 *xs = x.w + spin * W;
                for (int k = 0; k < W; ++k)
                    for (u64 bits = xs[k]; bits; bits &= bits - 1) {
                        const int q = (k << 6) + lowest(bits);
                        for (const auto &e : lists[q]) {
                            if (e.first < t) break;
                            if (test(xs, e.second)) continue;
                            Det<W> y = x;
                            flip(y.w + spin * W, q);
                            flip(y.w + spin * W, e.second);
                            const u64 hy = hx ^ z.key[spin][q] ^ z.key[spin][e.second];
                            if (sp.index.find(y, hy) >= 0) continue;
                            stage.add(y, hy, nullptr);
                            if (partners) {
                                Det<W> w;
                                std::memcpy(w.a(), y.b(), sizeof(u64) * W);
                                std::memcpy(w.b(), y.a(), sizeof(u64) * W);
                                const u64 hw = hash_det<W>(w);
                                if (sp.index.find(w, hw) < 0) stage.add(w, hw, nullptr);
                            }
                        }
                    }
            }
        }
        stage.flush_all();
    }
    auto *out = new Buffer;
    out->width = 2 * W;
    std::vector<Det<W>> all;
    all.reserve(found.size());
    for (size_t sh = 0; sh < found.shards.size(); ++sh)
        found.for_each(sh, [&](const Det<W> &key, const int64_t *) { all.push_back(key); });
    std::sort(all.begin(), all.end(), [](const Det<W> &u, const Det<W> &v) { return before(u, v); });
    out->rows = all.size();
    out->data.resize(all.size() * 2 * W);
    if (!all.empty()) std::memcpy(out->data.data(), all.data(), sizeof(Det<W>) * all.size());
    return out;
}

// The distinct rows of `candidates` that are not in the space (sorted).
template <int W> Buffer *difference_impl(Space<W> &sp, const u64 *candidates, int64_t count, int threads)
{
    const Det<W> *rows = reinterpret_cast<const Det<W> *>(candidates);
    ShardedMap<W, int64_t> found(0);
#pragma omp parallel num_threads(threads)
    {
        typename ShardedMap<W, int64_t>::Stage stage(found);
#pragma omp for schedule(static)
        for (int64_t i = 0; i < count; ++i) {
            const u64 h = hash_det<W>(rows[i]);
            if (sp.index.find(rows[i], h) < 0) stage.add(rows[i], h, nullptr);
        }
        stage.flush_all();
    }
    auto *out = new Buffer;
    out->width = 2 * W;
    std::vector<Det<W>> all;
    all.reserve(found.size());
    for (size_t sh = 0; sh < found.shards.size(); ++sh)
        found.for_each(sh, [&](const Det<W> &key, const int64_t *) { all.push_back(key); });
    std::sort(all.begin(), all.end(), [](const Det<W> &u, const Det<W> &v) { return before(u, v); });
    out->rows = all.size();
    out->data.resize(all.size() * 2 * W);
    if (!all.empty()) std::memcpy(out->data.data(), all.data(), sizeof(Det<W>) * all.size());
    return out;
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

template <int W> void partners_impl(Space<W> &sp, int64_t *out, int threads);

// Representatives of the swap pairs (sigma != 0): partner[i] >= i (and, for
// sigma < 0, partner[i] != i); open[i]: a distinct partner. Empty for sigma 0.
template <int W>
void pair_representatives(Space<W> &sp, double sigma, std::vector<uint8_t> &rep, std::vector<uint8_t> &open,
                          int threads)
{
    rep.clear();
    open.clear();
    if (sigma == 0.0) return;
    const int64_t count = sp.dets.size();
    std::vector<int64_t> partner(count);
    partners_impl(sp, partner.data(), threads);
    rep.assign(count, 0);
    open.assign(count, 0);
    for (int64_t i = 0; i < count; ++i) {
        const int64_t j = partner[i];
        open[i] = j >= 0 && j != i;
        rep[i] = j > i || (j == i && sigma > 0) || j < 0;
    }
}

template <int W>
void pt2_impl(Space<W> &sp, const double *coeffs, int64_t nroots, const double *energies, double eps,
              int64_t batches, double *out, double *norm, int64_t *stats, int threads, const double *screen = nullptr,
              double sigma = 0.0)
{
    const Ctx &c = *sp.ctx;
    const int64_t count = sp.dets.size();
    std::vector<uint8_t> rep, open;
    pair_representatives(sp, sigma, rep, open, threads);
    const bool paired = sigma != 0.0;
    // Each term H_ai c_i is kept when |H_ai| screen_i >= eps, screen_i = max_k
    // |c_ik| unless the caller fixes it (then the doubles kept do not depend
    // on the coefficients, and E2 is smooth in them).
    std::vector<double> cmax(count, 0.0);
    if (screen)
        for (int64_t i = 0; i < count; ++i) cmax[i] = std::fabs(screen[i]);
    else
        for (int64_t k = 0; k < nroots; ++k)
            for (int64_t i = 0; i < count; ++i) cmax[i] = std::max(cmax[i], std::fabs(coeffs[k * count + i]));
    std::vector<i128> total(nroots, 0), total_norm(nroots, 0);
    int64_t externals = 0, largest = 0;
    const char *profile_mode = std::getenv("ITERCC_SCI_PROFILE");
    const bool profile = profile_mode != nullptr;
    const bool no_map = profile && profile_mode[0] == '2';     // diagnostic: generation alone
    double generate_s = 0.0, pass_s = 0.0;
    int64_t contributions = 0;
    ShardedMap<W, int64_t> map(static_cast<int>(nroots));       // kept across batches (reset)
    for (int64_t batch = 0; batch < batches; ++batch) {
        if (batch) map.reset(threads);
        double t0 = omp_get_wtime();
#pragma omp parallel num_threads(threads) reduction(+ : contributions)
        {
            typename ShardedMap<W, int64_t>::Stage stage(map);
#pragma omp for schedule(dynamic, 16)
            for (int64_t i = 0; i < count; ++i) {
                if (cmax[i] <= 0.0) continue;
                int64_t add[16];
                if (paired) {
                    if (!rep[i]) continue;
                    const bool open_i = open[i];
                    auto in_batch2 = [&](u64 h, u64 hs) {
                        return batches == 1 || batch_of(h, batches) == batch || batch_of(hs, batches) == batch;
                    };
                    const Det<W> xs = swapped<W>(sp.dets[i]);
                    excitations_paired<W>(c, sp.dets[i], hash_det<W>(sp.dets[i]), hash_det<W>(xs), eps / cmax[i],
                                          in_batch2, [&](const Det<W> &y, double v, u64 h, u64 hs) {
                        const PairTerm t = pair_term<W>(y, open_i, sigma);
                        if (!t.keep) return;
                        const u64 key_hash = t.swap ? hs : h;
                        if (batches > 1 && batch_of(key_hash, batches) != batch) return;
                        for (int64_t k = 0; k < nroots; ++k)
                            add[k] = to_numerator(t.factor * v * coeffs[k * count + i]);
                        if (!no_map) {
                            if (t.swap) stage.add(swapped<W>(y), key_hash, add);
                            else stage.add(y, key_hash, add);
                        }
                        ++contributions;
                    });
                    continue;
                }
                auto in_batch = [&](u64 h) {
                    return batches == 1 || batch_of(h, batches) == batch;
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
                if (sp.index.find(y, map.hash_of(y)) >= 0) return;   // in the space: not external
                const bool both = paired && string_order<W>(y) != 0;       // stands for its swap too
                found[s] += both ? 2 : 1;
                int oa[MAX_ORB], ob[MAX_ORB];
                int na = occupied<W>(y.a(), oa), nb = occupied<W>(y.b(), ob);
                const double hyy = diagonal_energy(c, oa, na, ob, nb);
                for (int64_t r = 0; r < nroots; ++r) {
                    const double num = from_numerator(vals[r]);
                    const double amplitude = num / (energies[r] - hyy);
                    const i128 e2 = to_fixed(num * amplitude), n2 = to_fixed(amplitude * amplitude);
                    part[s * nroots + r] += both ? 2 * e2 : e2;
                    part_norm[s * nroots + r] += both ? 2 * n2 : n2;
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
                         double *norm, int64_t *stats, int threads, double sigma = 0.0)
{
    const Ctx &c = *sp.ctx;
    const int64_t count = sp.dets.size();
    std::vector<uint8_t> rep, open;
    pair_representatives(sp, sigma, rep, open, threads);
    const bool paired = sigma != 0.0;
    std::vector<double> cmax(count, 0.0), cdf(count + 1, 0.0);
    for (int64_t k = 0; k < nroots; ++k)
        for (int64_t i = 0; i < count; ++i) cmax[i] = std::max(cmax[i], std::fabs(coeffs[k * count + i]));
    if (paired)             // sample the representatives (each stands for its pair)
        for (int64_t i = 0; i < count; ++i)
            if (!rep[i]) cmax[i] = 0.0;
    for (int64_t i = 0; i < count; ++i) cdf[i + 1] = cdf[i] + cmax[i];
    const double total_weight = cdf[count];
    const double nd = static_cast<double>(samples);
    std::vector<double> values(nbatches * nroots, 0.0), norms(nbatches * nroots, 0.0);
    int64_t externals = 0;
    const bool profile = std::getenv("ITERCC_SCI_PROFILE") != nullptr;
    double generate_s = 0.0, pass_s = 0.0, largest_task = 0.0;
    int64_t terms = 0, distinct_total = 0;
    ShardedMap<W> map(static_cast<int>(4 * nroots));             // kept across batches (reset)
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
        if (batch) map.reset(threads);
        const double t0 = omp_get_wtime();
        distinct_total += distinct.size();
#pragma omp parallel num_threads(threads) reduction(+ : terms) reduction(max : largest_task)
        {
        typename ShardedMap<W>::Stage stage(map);
#pragma omp for schedule(dynamic, 1)
        for (int64_t t = 0; t < static_cast<int64_t>(distinct.size()); ++t) {
            const double task_start = profile ? omp_get_wtime() : 0.0;
            const int64_t i = distinct[t].first;
            const double w = distinct[t].second, p = cmax[i] / total_weight;
            if (cmax[i] <= 0.0) continue;
            i128 add[64];
            auto record = [&](const Det<W> &y, double v, u64 h) {
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
                ++terms;
            };
            if (paired) {
                // An open internal i may reach both members of an external
                // pair, a and its swap a'; the estimator squares i's whole
                // contribution to a, so both terms are added together, when a
                // (the representative) is generated, and a' alone only when a
                // is not among i's terms. A closed internal adds a's term only.
                const bool open_i = open[i];
                const double threshold = eps / cmax[i];
                const Det<W> &x = sp.dets[i];
                int occ[2 * MAX_ORB];
                const int na = occupied<W>(x.a(), occ), nb = occupied<W>(x.b(), occ + na);
                auto partner_value = [&](const Det<W> &z) {    // <z|H|x> if a term of i (above threshold), else 0
                    int d = 0;
                    for (int k = 0; k < W; ++k) d += popc(x.a()[k] ^ z.a()[k]) + popc(x.b()[k] ^ z.b()[k]);
                    if (d == 0 || d > 4) return 0.0;
                    const double u = element_strings<W>(c, x.a(), x.b(), z.a(), z.b(), occ, na, nb);
                    return std::fabs(u) >= threshold ? u : 0.0;
                };
                auto all2 = [](u64, u64) { return true; };
                const Det<W> xs = swapped<W>(x);
                excitations_paired<W>(c, x, hash_det<W>(x), hash_det<W>(xs), threshold, all2,
                                      [&](const Det<W> &y, double v, u64 h, u64 hs) {
                    const PairTerm t = pair_term<W>(y, open_i, sigma);
                    if (!t.keep) return;
                    // Each term's screening tests the term itself (kept: above eps_det).
                    double v1 = t.factor * v, v2 = 0.0;
                    bool kept1 = std::fabs(v) * cmax[i] >= eps_det, kept2 = false;
                    if (open_i && string_order<W>(y) != 0) {
                        const Det<W> other = swapped<W>(y);
                        const double u = partner_value(other);
                        if (t.swap && u != 0.0) return;          // added with the representative
                        if (!t.swap && u != 0.0) {
                            v2 = sigma * u;
                            kept2 = std::fabs(u) * cmax[i] >= eps_det;
                        }
                    }
                    for (int64_t k = 0; k < nroots; ++k) {
                        const double ck = coeffs[k * count + i];
                        const double term = (v1 + v2) * ck;
                        const double kept_term = ((kept1 ? v1 : 0.0) + (kept2 ? v2 : 0.0)) * ck;
                        const double scale = w * (nd - 1.0) / p - w * w / (p * p);
                        add[4 * k] = to_fixed(w * term / p);
                        add[4 * k + 1] = to_fixed(scale * term * term);
                        add[4 * k + 2] = to_fixed(w * kept_term / p);
                        add[4 * k + 3] = to_fixed(scale * kept_term * kept_term);
                    }
                    if (t.swap) stage.add(swapped<W>(y), hs, add);
                    else stage.add(y, h, add);
                    ++terms;
                });
                if (profile) largest_task = std::max(largest_task, omp_get_wtime() - task_start);
                continue;
            }
            excitations<W>(c, sp.dets[i], hash_det<W>(sp.dets[i]), eps / cmax[i], [](u64) { return true; }, record);
            if (profile) largest_task = std::max(largest_task, omp_get_wtime() - task_start);
        }
        stage.flush_all();
        }
        const double t1 = omp_get_wtime();
        generate_s += t1 - t0;
        externals += map.size();
        const int64_t nshards = map.shards.size();
        std::vector<i128> part(nshards * nroots, 0), part_norm(nshards * nroots, 0);
#pragma omp parallel for schedule(dynamic, 4) num_threads(threads)
        for (int64_t s = 0; s < nshards; ++s) {
            map.for_each(s, [&](const Det<W> &y, const i128 *vals) {
                if (sp.index.find(y, map.hash_of(y)) >= 0) return;   // in the space: not external
                const double weight = paired && string_order<W>(y) != 0 ? 2.0 : 1.0;
                int oa[MAX_ORB], ob[MAX_ORB];
                int na = occupied<W>(y.a(), oa), nb = occupied<W>(y.b(), ob);
                const double hyy = diagonal_energy(c, oa, na, ob, nb);
                for (int64_t r = 0; r < nroots; ++r) {
                    const i128 *q = vals + 4 * r;
                    const double a = from_fixed(q[0]), b = from_fixed(q[1]);
                    const double ad = from_fixed(q[2]), bd = from_fixed(q[3]);
                    const double inverse = 1.0 / (energies[r] - hyy);
                    const double square = weight * ((a * a + b) - (ad * ad + bd)) / (nd * (nd - 1.0));
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
        pass_s += omp_get_wtime() - t1;
    }
    if (profile)
        std::fprintf(stderr, "itercc_sci_pt2_stochastic: %lld batches, %lld distinct samples, %lld terms, %lld externals; "
                     "generate %.2f s (largest task %.3f s), final pass %.2f s\n", static_cast<long long>(nbatches),
                     static_cast<long long>(distinct_total), static_cast<long long>(terms),
                     static_cast<long long>(externals), generate_s, largest_task, pass_s);
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
            // Spin exchange: alpha q -> p, beta p -> q, for q occupied in alpha
            // only and p in beta only; the partner is found by direct lookup.
            int aonly[MAX_ORB], bonly[MAX_ORB];
            int na1 = 0, nb1 = 0;
            for (int k = 0; k < W; ++k) {
                for (u64 bits = x.a()[k] & ~x.b()[k]; bits; bits &= bits - 1) aonly[na1++] = (k << 6) + lowest(bits);
                for (u64 bits = x.b()[k] & ~x.a()[k]; bits; bits &= bits - 1) bonly[nb1++] = (k << 6) + lowest(bits);
            }
            for (int u = 0; u < na1; ++u)
                for (int v = 0; v < nb1; ++v) {
                    const int q = aonly[u], p = bonly[v];
                    Det<W> y = x;
                    flip(y.a(), q);
                    flip(y.a(), p);
                    flip(y.b(), p);
                    flip(y.b(), q);
                    const int64_t j = sp.index.find(y, hash_det<W>(y));
                    if (j < 0) continue;
                    const double e = -phase<W>(x.a(), q, p) * phase<W>(x.b(), p, q);
                    for (int64_t r = 0; r < nroots; ++r)
                        mine[r] += to_fixed(e * coeffs[r * count + i] * coeffs[r * count + j]);
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
                        const int64_t *irrep, int64_t axial, double cutoff, int threads)
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
    c->axial = axial != 0;
    if (c->axial) {
        // One representative per pair of ordered pairs: (p n + q) >= (r n + s).
        const int64_t nn = n * n;
        c->eri.assign(nn * (nn + 1) / 2, 0.0);
#pragma omp parallel for schedule(static) num_threads(threads)
        for (int64_t a = 0; a < nn; ++a)
            std::memcpy(&c->eri[a * (a + 1) / 2], &eri[a * nn], (a + 1) * sizeof(double));
    } else {
        // One representative per class: p >= q, r >= s, pair(p, q) >= pair(r, s).
        const int64_t pairs = n * (n + 1) / 2;
        c->eri.assign(pairs * (pairs + 1) / 2, 0.0);
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
        for (int64_t p = 0; p < n; ++p)
            for (int64_t q = 0; q <= p; ++q) {
                const int64_t pq = Ctx::pair(p, q);
                for (int64_t r = 0; r <= p; ++r)
                    for (int64_t t = 0; t <= r; ++t) {
                        const int64_t rs = Ctx::pair(r, t);
                        if (rs > pq) break;
                        c->eri[Ctx::pair(pq, rs)] = eri[((p * n + q) * n + r) * n + t];
                    }
            }
    }
    c->irrep.assign(n, 0);
    if (irrep) c->irrep.assign(irrep, irrep + n);
    c->J.resize(n * n);
    c->K.resize(n * n);
    for (int64_t p = 0; p < n; ++p)
        for (int64_t q = 0; q < n; ++q) {
            c->J[p * n + q] = c->E(p, p, q, q);
            c->K[p * n + q] = c->E(p, q, q, p);
        }
    c->JK.resize(n * n);
    for (int64_t t = 0; t < n * n; ++t) c->JK[t] = c->J[t] - c->K[t];
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

void *itercc_sci_hamiltonian(void *space, double drop, int single, int threads)
{
    auto *s = static_cast<SpaceHandle *>(space);
    return dispatch(s->ctx->words, [&](auto w) -> void * {
        return hamiltonian_impl(*as_space<decltype(w)::value>(s->space), drop, threads, nullptr, 0, single != 0);
    });
}

void *itercc_sci_hamiltonian_extend(void *space, const void *old, int64_t n_old, double drop, int single, int threads)
{
    auto *s = static_cast<SpaceHandle *>(space);
    const auto *H = static_cast<const Hamiltonian *>(old);
    if (!H || H->n != n_old) return nullptr;
    return dispatch(s->ctx->words, [&](auto w) -> void * {
        auto &sp = *as_space<decltype(w)::value>(s->space);
        if (n_old > static_cast<int64_t>(sp.dets.size())) return nullptr;
        return hamiltonian_impl(sp, drop, threads, H, n_old, single != 0);
    });
}

void *itercc_sci_hamiltonian_pairs(void *space, double sigma_t, double sigma_m, double drop, int single, int threads)
{
    auto *s = static_cast<SpaceHandle *>(space);
    return dispatch(s->ctx->words, [&](auto w) -> void * {
        return pair_hamiltonian_impl(*as_space<decltype(w)::value>(s->space), sigma_t, sigma_m, drop, threads,
                                     nullptr, 0, single != 0);
    });
}

void *itercc_sci_hamiltonian_pairs_extend(void *space, const void *old, int64_t n_old, double sigma_t,
                                          double sigma_m, double drop, int single, int threads)
{
    auto *s = static_cast<SpaceHandle *>(space);
    const auto *H = static_cast<const Hamiltonian *>(old);
    if (!H) return nullptr;
    return dispatch(s->ctx->words, [&](auto w) -> void * {
        auto &sp = *as_space<decltype(w)::value>(s->space);
        if (n_old > static_cast<int64_t>(sp.dets.size())) return nullptr;
        return pair_hamiltonian_impl(sp, sigma_t, sigma_m, drop, threads, H, n_old, single != 0);
    });
}

int64_t itercc_sci_orbits(void *space, double sigma_t, double sigma_m, int64_t *orbit, double *coef, int threads)
{
    auto *s = static_cast<SpaceHandle *>(space);
    return dispatch(s->ctx->words, [&](auto w) -> int64_t {
        auto &sp = *as_space<decltype(w)::value>(s->space);
        OrbitMap om;
        if (!orbit_map(sp, sigma_t, sigma_m, om, threads)) return -1;
        const int64_t count = sp.dets.size();
        for (int64_t i = 0; i < count; ++i) {
            orbit[i] = om.orbit[i];
            coef[i] = om.coef[i];
        }
        return static_cast<int64_t>(om.reps.size());
    });
}

int itercc_sci_set_mirror(void *ctx, const int64_t *mirror)
{
    auto *c = static_cast<Ctx *>(ctx);
    const int64_t n = c->n;
    if (!mirror) {
        c->mirror.clear();
        return 0;
    }
    for (int64_t p = 0; p < n; ++p)
        if (mirror[p] < 0 || mirror[p] >= n || mirror[mirror[p]] != p) return -1;
    c->mirror.assign(mirror, mirror + n);
    return 0;
}

int64_t itercc_sci_ham_nnz(const void *ham) { return static_cast<const Hamiltonian *>(ham)->nnz; }

void itercc_sci_ham_free(void *ham) { delete static_cast<Hamiltonian *>(ham); }

void itercc_sci_ham_diagonal(const void *ham, double *out)
{
    const auto *H = static_cast<const Hamiltonian *>(ham);
    std::copy(H->diag.begin(), H->diag.end(), out);
}

void itercc_sci_matmat(void *ham, int64_t k, const double *x, double *y, int threads)
{
    auto *H = static_cast<Hamiltonian *>(ham);
    switch (k) {
    case 1: return H->single ? spmm<float, 1>(*H, H->val32, x, y, threads) : spmm<double, 1>(*H, H->val, x, y, threads);
    case 2: return H->single ? spmm<float, 2>(*H, H->val32, x, y, threads) : spmm<double, 2>(*H, H->val, x, y, threads);
    case 3: return H->single ? spmm<float, 3>(*H, H->val32, x, y, threads) : spmm<double, 3>(*H, H->val, x, y, threads);
    case 4: return H->single ? spmm<float, 4>(*H, H->val32, x, y, threads) : spmm<double, 4>(*H, H->val, x, y, threads);
    default: {
        // Larger blocks in groups of up to four columns.
        const int64_t count = H->n;
        std::vector<double> xs(count * 4), ys(count * 4);
        for (int64_t c0 = 0; c0 < k; c0 += 4) {
            const int64_t m = std::min<int64_t>(4, k - c0);
            for (int64_t i = 0; i < count; ++i)
                for (int64_t r = 0; r < m; ++r) xs[i * m + r] = x[i * k + c0 + r];
            itercc_sci_matmat(ham, m, xs.data(), ys.data(), threads);
            for (int64_t i = 0; i < count; ++i)
                for (int64_t r = 0; r < m; ++r) y[i * k + c0 + r] = ys[i * m + r];
        }
    }
    }
}

void itercc_sci_matvec(void *ham, const double *x, double *y, int threads) { itercc_sci_matmat(ham, 1, x, y, threads); }

int64_t itercc_sci_ham_bytes(const void *ham) { return static_cast<const Hamiltonian *>(ham)->bytes(); }

void itercc_sci_ham_dense(const void *ham, double *out)
{
    const auto *H = static_cast<const Hamiltonian *>(ham);
    const int64_t n = H->n;
    std::fill(out, out + n * n, 0.0);
    std::vector<int32_t> col(H->longest_row() + 4);
    for (int64_t i = 0; i < n; ++i) {
        out[i * n + i] = H->diag[i];
        const int64_t ch = i / Hamiltonian::CHUNK, local = i % Hamiltonian::CHUNK;
        const int64_t *st = H->start[ch].data();
        H->columns_of(i, col.data());
        for (int64_t t = st[local]; t < st[local + 1]; ++t) {
            const double v = H->single ? static_cast<double>(H->val32[ch][t]) : H->val[ch][t];
            const int64_t j = col[t - st[local]];
            out[i * n + j] = v;
            out[j * n + i] = v;
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

void itercc_sci_pt2_pairs(void *space, const double *coeffs, int64_t nroots, const double *energies, double eps,
                          int64_t batches, double sigma, double *out, double *norm, int64_t *stats, int threads)
{
    auto *s = static_cast<SpaceHandle *>(space);
    if (nroots < 1 || nroots > 16 || batches < 1) return;
    dispatch(s->ctx->words, [&](auto w) {
        pt2_impl(*as_space<decltype(w)::value>(s->space), coeffs, nroots, energies, eps, batches, out, norm, stats,
                 threads, nullptr, sigma);
        return 0;
    });
}

void itercc_sci_pt2_stochastic_pairs(void *space, const double *coeffs, int64_t nroots, const double *energies,
                                     double eps, double eps_det, int64_t samples, int64_t nbatches, int64_t seed,
                                     double sigma, double *mean, double *error, double *norm, int64_t *stats,
                                     int threads)
{
    auto *s = static_cast<SpaceHandle *>(space);
    if (nroots < 1 || nroots > 16 || samples < 2 || nbatches < 1) return;
    dispatch(s->ctx->words, [&](auto w) {
        pt2_stochastic_impl(*as_space<decltype(w)::value>(s->space), coeffs, nroots, energies, eps, eps_det,
                            samples, nbatches, static_cast<u64>(seed), mean, error, norm, stats, threads, sigma);
        return 0;
    });
}

void itercc_sci_pt2_screened(void *space, const double *coeffs, int64_t nroots, const double *energies,
                             const double *screen, double eps, int64_t batches, double *out, double *norm,
                             int64_t *stats, int threads)
{
    auto *s = static_cast<SpaceHandle *>(space);
    if (nroots < 1 || nroots > 16 || batches < 1 || !screen) return;
    dispatch(s->ctx->words, [&](auto w) {
        pt2_impl(*as_space<decltype(w)::value>(s->space), coeffs, nroots, energies, eps, batches, out, norm, stats,
                 threads, screen);
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

void *itercc_sci_onebody(void *space, const double *d, double drop, int threads)
{
    auto *s = static_cast<SpaceHandle *>(space);
    return dispatch(s->ctx->words, [&](auto w) -> void * {
        return onebody_impl(*as_space<decltype(w)::value>(s->space), d, drop, threads);
    });
}

void *itercc_sci_onebody_extend(void *space, const double *d, void *old, int64_t n_old, double drop, int threads)
{
    auto *s = static_cast<SpaceHandle *>(space);
    auto *H = static_cast<Hamiltonian *>(old);
    if (!H || H->n != n_old || H->single) return nullptr;
    return dispatch(s->ctx->words, [&](auto w) -> void * {
        auto &sp = *as_space<decltype(w)::value>(s->space);
        if (n_old > static_cast<int64_t>(sp.dets.size())) return nullptr;
        return onebody_impl(sp, d, drop, threads, H, n_old);
    });
}

void *itercc_sci_select_onebody(void *space, const double *d, const double *cmax, double eps, int partners,
                                int threads)
{
    auto *s = static_cast<SpaceHandle *>(space);
    return dispatch(s->ctx->words, [&](auto w) -> void * {
        return select_onebody_impl(*as_space<decltype(w)::value>(s->space), d, cmax, eps, partners, threads);
    });
}

void *itercc_sci_difference(void *space, const uint64_t *candidates, int64_t count, int threads)
{
    auto *s = static_cast<SpaceHandle *>(space);
    return dispatch(s->ctx->words, [&](auto w) -> void * {
        return difference_impl(*as_space<decltype(w)::value>(s->space), candidates, count, threads);
    });
}

void itercc_sci_parity(int64_t count, const int64_t *partner, double sign, const double *x, double *out,
                       int threads)
{
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t i = 0; i < count; ++i) out[i] = 0.5 * (x[i] + sign * x[partner[i]]);
}

// Block vectors (n x k, row-major: column r of row i at i * k + r) for the
// response solver. Column sums use fixed chunks added in order, so they do not
// depend on the thread count.
static constexpr int64_t BLOCK_CHUNK = 1 << 14;

void itercc_sci_block_dot(int64_t n, int64_t k, const double *a, const double *b, double *out, int threads)
{
    const int64_t chunks = (n + BLOCK_CHUNK - 1) / BLOCK_CHUNK;
    std::vector<double> part(chunks * k, 0.0);
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t c = 0; c < chunks; ++c) {
        const int64_t lo = c * BLOCK_CHUNK, hi = std::min(n, lo + BLOCK_CHUNK);
        double acc[16] = {0.0};
        for (int64_t i = lo; i < hi; ++i)
            for (int64_t r = 0; r < k; ++r) acc[r] += a[i * k + r] * b[i * k + r];
        for (int64_t r = 0; r < k; ++r) part[c * k + r] = acc[r];
    }
    for (int64_t r = 0; r < k; ++r) out[r] = 0.0;
    for (int64_t c = 0; c < chunks; ++c)
        for (int64_t r = 0; r < k; ++r) out[r] += part[c * k + r];
}

void itercc_sci_block_axpby(int64_t n, int64_t k, const double *alpha, const double *x, const double *beta, double *y,
                            int threads)
{
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t i = 0; i < n; ++i)
        for (int64_t r = 0; r < k; ++r) y[i * k + r] = alpha[r] * x[i * k + r] + beta[r] * y[i * k + r];
}

void itercc_sci_block_divide(int64_t n, int64_t k, const double *d, double shift, double floor, const double *x,
                             double *z, int threads)
{
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t i = 0; i < n; ++i) {
        double q = d[i] - shift;
        if (std::fabs(q) < floor) q = q < 0 ? -floor : floor;
        const double inv = 1.0 / q;
        for (int64_t r = 0; r < k; ++r) z[i * k + r] = x[i * k + r] * inv;
    }
}

void itercc_sci_block_deflate(int64_t n, int64_t k, const double *c, double *v, int threads)
{
    // v[:, r] -= c (c . v[:, r])
    const int64_t chunks = (n + BLOCK_CHUNK - 1) / BLOCK_CHUNK;
    std::vector<double> part(chunks * k, 0.0);
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t ch = 0; ch < chunks; ++ch) {
        const int64_t lo = ch * BLOCK_CHUNK, hi = std::min(n, lo + BLOCK_CHUNK);
        double acc[16] = {0.0};
        for (int64_t i = lo; i < hi; ++i)
            for (int64_t r = 0; r < k; ++r) acc[r] += c[i] * v[i * k + r];
        for (int64_t r = 0; r < k; ++r) part[ch * k + r] = acc[r];
    }
    double dots[16] = {0.0};
    for (int64_t ch = 0; ch < chunks; ++ch)
        for (int64_t r = 0; r < k; ++r) dots[r] += part[ch * k + r];
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int64_t i = 0; i < n; ++i)
        for (int64_t r = 0; r < k; ++r) v[i * k + r] -= c[i] * dots[r];
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
