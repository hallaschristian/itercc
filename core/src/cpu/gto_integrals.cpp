// Gaussian AO integrals (`itercc.basis`, `itercc.integrals`): overlap,
// kinetic energy, nuclear attraction, dipole and two-electron repulsion
// integrals over contracted spherical Gaussians, by McMurchie-Davidson
// (Hermite expansion coefficients E, Hermite Coulomb integrals R from the
// Boys function), in libcint's conventions: Cartesian components xx, xy, xz,
// yy, yz, zz, the coefficients carry the radial normalization, and c2s
// (per angular momentum, ncart x nsph) turns Cartesian blocks into spherical
// ones. Every output entry has one writer.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "itercc/core.h"

namespace {

using i64 = int64_t;

constexpr double kPi = 3.14159265358979323846;

// F_m(t), m = 0 .. mmax, into f.
void boys(int mmax, double t, double *f)
{
    if (t > 45.0) {
        // F_0 = sqrt(pi / t) erf(sqrt(t)) / 2 with erf = 1 to 1e-20, then up
        // with the exp(-t) term (stable while 2m + 1 < 2t; the asymptotic form
        // alone loses digits at high m: F_16(45) to 4e-7 relative).
        const double et = std::exp(-t);
        f[0] = 0.5 * std::sqrt(kPi / t);
        for (int m = 1; m <= mmax; ++m) f[m] = ((2 * m - 1) * f[m - 1] - et) / (2.0 * t);
        return;
    }
    const double et = std::exp(-t);
    double term = 1.0 / (2 * mmax + 1), sum = term;
    for (int k = 1; k < 400; ++k) {
        term *= 2.0 * t / (2 * mmax + 2 * k + 1);
        sum += term;
        if (term < 1e-17 * sum) break;
    }
    f[mmax] = et * sum;
    for (int m = mmax - 1; m >= 0; --m) f[m] = (2.0 * t * f[m + 1] + et) / (2 * m + 1);
}

struct Comp {
    int x, y, z;
};

std::vector<Comp> components(int l)
{
    std::vector<Comp> out;
    for (int lx = l; lx >= 0; --lx)
        for (int ly = l - lx; ly >= 0; --ly) out.push_back({lx, ly, l - lx - ly});
    return out;
}

// 1D Hermite expansion coefficients E(i, j, t) of a primitive pair,
// i <= la, j <= lb, t <= i + j; stride (lb + 1) (la + lb + 1).
struct Hermite1 {
    int la, lb, nt;
    std::vector<double> e;
    double &at(int i, int j, int t) { return e[size_t((i * (lb + 1) + j) * nt + t)]; }
    double get(int i, int j, int t) const
    {
        return t < 0 || t > i + j ? 0.0 : e[size_t((i * (lb + 1) + j) * nt + t)];
    }
    void build(int la_, int lb_, double p, double xpa, double xpb, double eab)
    {
        la = la_;
        lb = lb_;
        nt = la + lb + 1;
        e.assign(size_t((la + 1) * (lb + 1) * nt), 0.0);
        const double h = 0.5 / p;
        at(0, 0, 0) = eab;
        for (int i = 0; i < la; ++i)
            for (int t = 0; t <= i + 1; ++t)
                at(i + 1, 0, t) = h * get(i, 0, t - 1) + xpa * get(i, 0, t) + (t + 1) * get(i, 0, t + 1);
        for (int i = 0; i <= la; ++i)
            for (int j = 0; j < lb; ++j)
                for (int t = 0; t <= i + j + 1; ++t)
                    at(i, j + 1, t) = h * get(i, j, t - 1) + xpb * get(i, j, t) + (t + 1) * get(i, j, t + 1);
    }
};

// Hermite Coulomb integrals R(t, u, v) for t + u + v <= L at exponent alpha
// and displacement (x, y, z): cube of side L + 1.
void hermite_r(int L, double alpha, double x, double y, double z, std::vector<double> &r,
               std::vector<double> &prev, double *f)
{
    const int s = L + 1;
    const size_t cube = size_t(s) * s * s;
    r.assign(cube, 0.0);
    prev.assign(cube, 0.0);
    boys(L, alpha * (x * x + y * y + z * z), f);
    auto at = [s](int t, int u, int v) { return size_t((t * s + u) * s + v); };
    double scale = 1.0;
    for (int n = 0; n < L; ++n) scale *= -2.0 * alpha;
    for (int n = L; n >= 0; --n) {
        std::swap(r, prev);                          // prev: order n + 1
        r[0] = scale * f[n];
        for (int total = 1; total <= L - n; ++total)
            for (int t = total; t >= 0; --t)
                for (int u = total - t; u >= 0; --u) {
                    const int v = total - t - u;
                    double value;
                    if (t > 0)
                        value = (t > 1 ? (t - 1) * prev[at(t - 2, u, v)] : 0.0) + x * prev[at(t - 1, u, v)];
                    else if (u > 0)
                        value = (u > 1 ? (u - 1) * prev[at(t, u - 2, v)] : 0.0) + y * prev[at(t, u - 1, v)];
                    else
                        value = (v > 1 ? (v - 1) * prev[at(t, u, v - 2)] : 0.0) + z * prev[at(t, u, v - 1)];
                    r[at(t, u, v)] = value;
                }
        if (n > 0) scale /= -2.0 * alpha;
    }
}

struct Shells {
    i64 n;
    const int32_t *l, *atom, *nprim;
    const i64 *prim0, *ao_off, *c2s_off;
    const double *exps, *coefs, *xyz, *c2s;
};

// Spherical block (na x nb) = c2s_a^T cart (ca x cb) c2s_b.
void to_spherical(const Shells &sh, int la, int lb, const double *cart, double *sph)
{
    const int ca = (la + 1) * (la + 2) / 2, cb = (lb + 1) * (lb + 2) / 2;
    const int na = 2 * la + 1, nb = 2 * lb + 1;
    const double *ta = sh.c2s + sh.c2s_off[la], *tb = sh.c2s + sh.c2s_off[lb];
    std::vector<double> half(size_t(ca * nb), 0.0);
    for (int i = 0; i < ca; ++i)
        for (int k = 0; k < cb; ++k) {
            const double v = cart[i * cb + k];
            if (v == 0.0) continue;
            for (int j = 0; j < nb; ++j) half[size_t(i * nb + j)] += v * tb[k * nb + j];
        }
    for (int m = 0; m < na; ++m)
        for (int j = 0; j < nb; ++j) {
            double s = 0.0;
            for (int i = 0; i < ca; ++i) s += ta[i * na + m] * half[size_t(i * nb + j)];
            sph[m * nb + j] = s;
        }
}

}  // namespace

extern "C" void itercc_int1e(int kind, int64_t nshell, const int32_t *shell_l, const int32_t *shell_atom,
                             const int32_t *shell_nprim, const int64_t *shell_prim0, const double *exps,
                             const double *coefs, const double *atom_xyz, const double *charges, int64_t natm,
                             const double *origin, const double *c2s, const int64_t *c2s_off,
                             const int64_t *ao_off, int64_t nao, double *out, int threads)
{
    // kind 0 overlap, 1 kinetic energy, 2 nuclear attraction (-sum_C Z_C /
    // |r - C|), 3 dipole (x, y, z of r - origin: out holds three nao^2
    // matrices). Shell pairs a >= b in parallel, both triangles written.
    const Shells sh{nshell, shell_l, shell_atom, shell_nprim, shell_prim0, ao_off, c2s_off,
                    exps, coefs, atom_xyz, c2s};
    const int nmat = kind == 3 ? 3 : 1;
    const i64 npairs = nshell * (nshell + 1) / 2;
#pragma omp parallel num_threads(threads)
    {
        Hermite1 ex, ey, ez;
        std::vector<double> r, prev, cart, sph;
        double f[64];
#pragma omp for schedule(dynamic, 1)
        for (i64 pair = 0; pair < npairs; ++pair) {
            i64 a = i64((std::sqrt(8.0 * double(pair) + 1.0) - 1.0) / 2.0);
            while (a * (a + 1) / 2 > pair) --a;
            while ((a + 1) * (a + 2) / 2 <= pair) ++a;
            const i64 b = pair - a * (a + 1) / 2;
            const int la = sh.l[a], lb = sh.l[b];
            const auto ca = components(la), cb = components(lb);
            const int nca = int(ca.size()), ncb = int(cb.size());
            const double *A = sh.xyz + 3 * sh.atom[a], *B = sh.xyz + 3 * sh.atom[b];
            const double ab2 = (A[0] - B[0]) * (A[0] - B[0]) + (A[1] - B[1]) * (A[1] - B[1]) +
                               (A[2] - B[2]) * (A[2] - B[2]);
            cart.assign(size_t(nmat * nca * ncb), 0.0);
            const int extra = kind == 1 ? 2 : 0;
            for (int ip = 0; ip < sh.nprim[a]; ++ip)
                for (int jp = 0; jp < sh.nprim[b]; ++jp) {
                    const double ea = sh.exps[sh.prim0[a] + ip], eb = sh.exps[sh.prim0[b] + jp];
                    const double p = ea + eb, mu = ea * eb / p;
                    const double coef = sh.coefs[sh.prim0[a] + ip] * sh.coefs[sh.prim0[b] + jp];
                    double P[3];
                    for (int k = 0; k < 3; ++k) P[k] = (ea * A[k] + eb * B[k]) / p;
                    const double kab = std::exp(-mu * ab2);
                    ex.build(la, lb + extra, p, P[0] - A[0], P[0] - B[0], 1.0);
                    ey.build(la, lb + extra, p, P[1] - A[1], P[1] - B[1], 1.0);
                    ez.build(la, lb + extra, p, P[2] - A[2], P[2] - B[2], 1.0);
                    const double s1 = std::sqrt(kPi / p);
                    if (kind == 2) {
                        const int L = la + lb;
                        for (i64 c = 0; c < natm; ++c) {
                            const double *C = sh.xyz + 3 * c;
                            hermite_r(L, p, P[0] - C[0], P[1] - C[1], P[2] - C[2], r, prev, f);
                            const int s = L + 1;
                            const double pref = -charges[c] * 2.0 * kPi / p * kab * coef;
                            for (int i = 0; i < nca; ++i)
                                for (int j = 0; j < ncb; ++j) {
                                    const Comp &u = ca[size_t(i)], &w = cb[size_t(j)];
                                    double sum = 0.0;
                                    for (int t = 0; t <= u.x + w.x; ++t) {
                                        const double et = ex.get(u.x, w.x, t);
                                        if (et == 0.0) continue;
                                        for (int q = 0; q <= u.y + w.y; ++q) {
                                            const double eu = et * ey.get(u.y, w.y, q);
                                            if (eu == 0.0) continue;
                                            for (int v = 0; v <= u.z + w.z; ++v)
                                                sum += eu * ez.get(u.z, w.z, v) * r[size_t((t * s + q) * s + v)];
                                        }
                                    }
                                    cart[size_t(i * ncb + j)] += pref * sum;
                                }
                        }
                        continue;
                    }
                    for (int i = 0; i < nca; ++i)
                        for (int j = 0; j < ncb; ++j) {
                            const Comp &u = ca[size_t(i)], &w = cb[size_t(j)];
                            const double sx = ex.get(u.x, w.x, 0) * s1, sy = ey.get(u.y, w.y, 0) * s1,
                                         sz = ez.get(u.z, w.z, 0) * s1;
                            const double pref = kab * coef;
                            if (kind == 0) {
                                cart[size_t(i * ncb + j)] += pref * sx * sy * sz;
                            } else if (kind == 1) {
                                auto kin = [&](const Hermite1 &e, int i1, int j1) {
                                    const double s0 = e.get(i1, j1, 0) * s1, s2 = e.get(i1, j1 + 2, 0) * s1;
                                    const double sm2 = j1 >= 2 ? e.get(i1, j1 - 2, 0) * s1 : 0.0;
                                    return -2.0 * eb * eb * s2 + eb * (2 * j1 + 1) * s0 - 0.5 * j1 * (j1 - 1) * sm2;
                                };
                                cart[size_t(i * ncb + j)] += pref * (kin(ex, u.x, w.x) * sy * sz +
                                                                     sx * kin(ey, u.y, w.y) * sz +
                                                                     sx * sy * kin(ez, u.z, w.z));
                            } else {
                                const double d[3] = {(ex.get(u.x, w.x, 1) + (P[0] - origin[0]) * ex.get(u.x, w.x, 0)) * s1,
                                                     (ey.get(u.y, w.y, 1) + (P[1] - origin[1]) * ey.get(u.y, w.y, 0)) * s1,
                                                     (ez.get(u.z, w.z, 1) + (P[2] - origin[2]) * ez.get(u.z, w.z, 0)) * s1};
                                cart[size_t((0 * nca + i) * ncb + j)] += pref * d[0] * sy * sz;
                                cart[size_t((1 * nca + i) * ncb + j)] += pref * sx * d[1] * sz;
                                cart[size_t((2 * nca + i) * ncb + j)] += pref * sx * sy * d[2];
                            }
                        }
                }
            const int na = 2 * la + 1, nb = 2 * lb + 1;
            sph.assign(size_t(na * nb), 0.0);
            for (int mat = 0; mat < nmat; ++mat) {
                to_spherical(sh, la, lb, cart.data() + size_t(mat * nca * ncb), sph.data());
                double *o = out + mat * nao * nao;
                for (int i = 0; i < na; ++i)
                    for (int j = 0; j < nb; ++j) {
                        const double v = sph[size_t(i * nb + j)];
                        o[(sh.ao_off[a] + i) * nao + sh.ao_off[b] + j] = v;
                        o[(sh.ao_off[b] + j) * nao + sh.ao_off[a] + i] = v;
                    }
            }
        }
    }
}

namespace {

// F_m(T) on a grid, for the two-electron integrals: Taylor expansion about
// the nearest grid point (dF_m/dT = -F_{m+1}), then down with exp(-T).
constexpr int kBoysOrders = 48;               // m + 8 Taylor terms <= this
constexpr double kBoysStep = 0.05, kBoysMax = 45.0;
constexpr int kBoysPoints = int(kBoysMax / kBoysStep) + 2;
constexpr int kTaylor = 8;

const std::vector<double> &boys_table()
{
    static const std::vector<double> table = [] {
        std::vector<double> t(size_t(kBoysPoints) * kBoysOrders);
        double f[kBoysOrders];
        for (int k = 0; k < kBoysPoints; ++k) {
            boys(kBoysOrders - 1, k * kBoysStep, f);
            for (int m = 0; m < kBoysOrders; ++m) t[size_t(k) * kBoysOrders + m] = f[m];
        }
        return t;
    }();
    return table;
}

void boys_fast(int mmax, double t, double *f)
{
    if (t >= kBoysMax || mmax + kTaylor >= kBoysOrders) {
        boys(mmax, t, f);
        return;
    }
    const std::vector<double> &table = boys_table();
    const int k = int(t / kBoysStep + 0.5);
    const double dt = k * kBoysStep - t;
    const double *row = table.data() + size_t(k) * kBoysOrders + mmax;
    double sum = 0.0, term = 1.0;
    for (int j = 0; j < kTaylor; ++j) {
        sum += row[j] * term;
        term *= dt / (j + 1);
    }
    f[mmax] = sum;
    const double et = std::exp(-t);
    for (int m = mmax - 1; m >= 0; --m) f[m] = (2.0 * t * f[m + 1] + et) / (2 * m + 1);
}

const std::vector<Comp> &comps_of(int l)
{
    static const std::vector<std::vector<Comp>> table = [] {
        std::vector<std::vector<Comp>> t;
        for (int k = 0; k <= 12; ++k) t.push_back(components(k));
        return t;
    }();
    return table[size_t(l)];
}

// A shell pair's primitive pairs: exponent sum, centre, prefactor (the two
// contraction coefficients times exp(-mu |AB|^2)) and Hermite coefficients
// E(i, j, t) for x, y and z at offset e.
struct PrimPair {
    double p, px, py, pz, k;
    size_t e;
};

struct ShellPair {
    i64 a, b;
    int la, lb, nt;
    std::vector<PrimPair> prims;
    std::vector<double> e;
    double ex(size_t base, int axis, int i, int j, int t) const
    {
        return e[base + size_t(axis) * size_t((la + 1) * (lb + 1) * nt) + size_t((i * (lb + 1) + j) * nt + t)];
    }
};

ShellPair make_pair_data(const Shells &sh, i64 a, i64 b)
{
    ShellPair sp;
    sp.a = a;
    sp.b = b;
    sp.la = sh.l[a];
    sp.lb = sh.l[b];
    sp.nt = sp.la + sp.lb + 1;
    const double *A = sh.xyz + 3 * sh.atom[a], *B = sh.xyz + 3 * sh.atom[b];
    double ab2 = 0.0;
    for (int k = 0; k < 3; ++k) ab2 += (A[k] - B[k]) * (A[k] - B[k]);
    const size_t block = size_t((sp.la + 1) * (sp.lb + 1) * sp.nt);
    Hermite1 h;
    for (int ip = 0; ip < sh.nprim[a]; ++ip)
        for (int jp = 0; jp < sh.nprim[b]; ++jp) {
            const double ea = sh.exps[sh.prim0[a] + ip], eb = sh.exps[sh.prim0[b] + jp];
            const double p = ea + eb;
            const double k = std::exp(-ea * eb / p * ab2) * sh.coefs[sh.prim0[a] + ip] * sh.coefs[sh.prim0[b] + jp];
            if (std::fabs(k) < 1e-20) continue;               // negligible primitive pair
            PrimPair pp{p, (ea * A[0] + eb * B[0]) / p, (ea * A[1] + eb * B[1]) / p, (ea * A[2] + eb * B[2]) / p, k,
                        sp.e.size()};
            const double P[3] = {pp.px, pp.py, pp.pz};
            for (int axis = 0; axis < 3; ++axis) {
                h.build(sp.la, sp.lb, p, P[axis] - A[axis], P[axis] - B[axis], 1.0);
                sp.e.insert(sp.e.end(), h.e.begin(), h.e.begin() + long(block));
            }
            sp.prims.push_back(pp);
        }
    return sp;
}

struct EriWork {
    std::vector<double> r, prev, w, cart, tmp, sph;
    double f[kBoysOrders + 16];
};

// Hermite Coulomb integrals R(t, u, v), t + u + v <= L, in the cube of side
// L + 1 (only the entries of each order that the next one reads are written).
const double *hermite_r_fast(int L, double alpha, double x, double y, double z, EriWork &wk)
{
    const int s = L + 1;
    const size_t cube = size_t(s) * s * s;
    if (wk.r.size() < cube) {
        wk.r.resize(cube);
        wk.prev.resize(cube);
    }
    boys_fast(L, alpha * (x * x + y * y + z * z), wk.f);
    double *cur = wk.r.data(), *old = wk.prev.data();
    auto at = [s](int t, int u, int v) { return size_t((t * s + u) * s + v); };
    double scale = 1.0;
    for (int n = 0; n < L; ++n) scale *= -2.0 * alpha;
    for (int n = L; n >= 0; --n) {
        std::swap(cur, old);                           // old: order n + 1
        cur[0] = scale * wk.f[n];
        for (int total = 1; total <= L - n; ++total)
            for (int t = total; t >= 0; --t)
                for (int u = total - t; u >= 0; --u) {
                    const int v = total - t - u;
                    double value;
                    if (t > 0)
                        value = (t > 1 ? (t - 1) * old[at(t - 2, u, v)] : 0.0) + x * old[at(t - 1, u, v)];
                    else if (u > 0)
                        value = (u > 1 ? (u - 1) * old[at(t, u - 2, v)] : 0.0) + y * old[at(t, u - 1, v)];
                    else
                        value = (v > 1 ? (v - 1) * old[at(t, u, v - 2)] : 0.0) + z * old[at(t, u, v - 1)];
                    cur[at(t, u, v)] = value;
                }
        if (n > 0) scale /= -2.0 * alpha;
    }
    return cur;
}

// Cartesian (ab|cd) of the shell pairs bra = (a, b), ket = (c, d).
void eri_cart_fast(const ShellPair &bra, const ShellPair &ket, EriWork &wk)
{
    const auto &ca = comps_of(bra.la), &cb = comps_of(bra.lb), &cc = comps_of(ket.la), &cd = comps_of(ket.lb);
    const int nca = int(ca.size()), ncb = int(cb.size()), ncc = int(cc.size()), ncd = int(cd.size());
    const int lab = bra.la + bra.lb, lcd = ket.la + ket.lb, L = lab + lcd, s = L + 1, sb = lab + 1;
    const int ncdp = ncc * ncd;
    wk.cart.assign(size_t(nca * ncb * ncdp), 0.0);
    wk.w.resize(size_t(sb * sb * sb * ncdp));
    const double twopi25 = 2.0 * std::pow(kPi, 2.5);
    for (const PrimPair &pp : bra.prims)
        for (const PrimPair &qq : ket.prims) {
            const double p = pp.p, q = qq.p, pq = p + q;
            const double pref = twopi25 / (p * q * std::sqrt(pq)) * pp.k * qq.k;
            if (std::fabs(pref) < 1e-22) continue;
            const double alpha = p * q / pq;
            const double *r = hermite_r_fast(L, alpha, pp.px - qq.px, pp.py - qq.py, pp.pz - qq.pz, wk);
            for (int t = 0; t <= lab; ++t)
                for (int u = 0; u <= lab - t; ++u)
                    for (int v = 0; v <= lab - t - u; ++v) {
                        double *wrow = wk.w.data() + size_t(((t * sb + u) * sb + v) * ncdp);
                        for (int k = 0; k < ncc; ++k)
                            for (int l = 0; l < ncd; ++l) {
                                const Comp &g = cc[size_t(k)], &h = cd[size_t(l)];
                                double sum = 0.0;
                                for (int tau = 0; tau <= g.x + h.x; ++tau) {
                                    const double e1 = ket.ex(qq.e, 0, g.x, h.x, tau);
                                    if (e1 == 0.0) continue;
                                    for (int nu = 0; nu <= g.y + h.y; ++nu) {
                                        const double e2 = e1 * ket.ex(qq.e, 1, g.y, h.y, nu);
                                        if (e2 == 0.0) continue;
                                        const double sgn = ((tau + nu) & 1) ? -e2 : e2;
                                        const double *rr = r + size_t(((t + tau) * s + u + nu) * s + v);
                                        for (int phi = 0; phi <= g.z + h.z; ++phi) {
                                            const double e3 = ket.ex(qq.e, 2, g.z, h.z, phi);
                                            sum += ((phi & 1) ? -sgn : sgn) * e3 * rr[phi];
                                        }
                                    }
                                }
                                wrow[k * ncd + l] = sum;
                            }
                    }
            for (int i = 0; i < nca; ++i)
                for (int j = 0; j < ncb; ++j) {
                    const Comp &u1 = ca[size_t(i)], &w1 = cb[size_t(j)];
                    double *o = wk.cart.data() + size_t((i * ncb + j) * ncdp);
                    for (int t = 0; t <= u1.x + w1.x; ++t) {
                        const double e1 = bra.ex(pp.e, 0, u1.x, w1.x, t);
                        if (e1 == 0.0) continue;
                        for (int u = 0; u <= u1.y + w1.y; ++u) {
                            const double e2 = e1 * bra.ex(pp.e, 1, u1.y, w1.y, u);
                            if (e2 == 0.0) continue;
                            for (int v = 0; v <= u1.z + w1.z; ++v) {
                                const double e3 = pref * e2 * bra.ex(pp.e, 2, u1.z, w1.z, v);
                                if (e3 == 0.0) continue;
                                const double *wr = wk.w.data() + size_t(((t * sb + u) * sb + v) * ncdp);
                                for (int m = 0; m < ncdp; ++m) o[m] += e3 * wr[m];
                            }
                        }
                    }
                }
        }
}

// Spherical (ab|cd) from wk.cart into wk.sph: four one-index transformations.
void eri_spherical_fast(const Shells &sh, int la, int lb, int lc, int ld, EriWork &wk)
{
    const int ls[4] = {la, lb, lc, ld};
    int dims[4] = {(la + 1) * (la + 2) / 2, (lb + 1) * (lb + 2) / 2, (lc + 1) * (lc + 2) / 2, (ld + 1) * (ld + 2) / 2};
    std::vector<double> *src = &wk.cart, *dst = &wk.tmp;
    for (int axis = 0; axis < 4; ++axis) {
        const int nc = dims[axis], ns = 2 * ls[axis] + 1;
        const double *t = sh.c2s + sh.c2s_off[ls[axis]];
        int before = 1, after = 1;
        for (int k = 0; k < axis; ++k) before *= dims[k];
        for (int k = axis + 1; k < 4; ++k) after *= dims[k];
        dst->assign(size_t(before * ns * after), 0.0);
        for (int x = 0; x < before; ++x)
            for (int i = 0; i < nc; ++i) {
                const double *in = src->data() + size_t((x * nc + i) * after);
                for (int m = 0; m < ns; ++m) {
                    const double coef = t[i * ns + m];
                    if (coef == 0.0) continue;
                    double *out = dst->data() + size_t((x * ns + m) * after);
                    for (int y = 0; y < after; ++y) out[y] += coef * in[y];
                }
            }
        dims[axis] = ns;
        std::swap(src, dst);
    }
    if (src != &wk.sph) wk.sph.swap(*src);
}

}  // namespace

extern "C" void itercc_int2e_s8(int64_t nshell, const int32_t *shell_l, const int32_t *shell_atom,
                                const int32_t *shell_nprim, const int64_t *shell_prim0, const double *exps,
                                const double *coefs, const double *atom_xyz, const double *c2s,
                                const int64_t *c2s_off, const int64_t *ao_off, int64_t nao, double screen,
                                double *out, int threads)
{
    // (ij|kl) of the spherical AOs, 8-fold packed (out[I(I+1)/2 + J], I =
    // i(i+1)/2 + j >= J = k(k+1)/2 + l, i >= j, k >= l), shell quartets with
    // a Schwarz bound below `screen` set to zero. Shell-pair data (primitive
    // pairs and their Hermite coefficients) are built once; each packed entry
    // belongs to one shell quartet (its two shell pairs), computed by one task.
    const Shells sh{nshell, shell_l, shell_atom, shell_nprim, shell_prim0, ao_off, c2s_off,
                    exps, coefs, atom_xyz, c2s};
    (void)nao;
    const i64 npairs = nshell * (nshell + 1) / 2;
    std::vector<ShellPair> pairs(static_cast<size_t>(npairs));
    std::vector<double> bound(static_cast<size_t>(npairs), 0.0);
    (void)boys_table();
#pragma omp parallel num_threads(threads)
    {
        EriWork wk;
#pragma omp for schedule(dynamic, 1)
        for (i64 k = 0; k < npairs; ++k) {
            i64 a = i64((std::sqrt(8.0 * double(k) + 1.0) - 1.0) / 2.0);
            while (a * (a + 1) / 2 > k) --a;
            while ((a + 1) * (a + 2) / 2 <= k) ++a;
            pairs[size_t(k)] = make_pair_data(sh, a, k - a * (a + 1) / 2);
        }
#pragma omp for schedule(dynamic, 1)
        for (i64 k = 0; k < npairs; ++k) {           // Schwarz: max |(ab|ab)|^(1/2)
            const ShellPair &sp = pairs[size_t(k)];
            eri_cart_fast(sp, sp, wk);
            eri_spherical_fast(sh, sp.la, sp.lb, sp.la, sp.lb, wk);
            const int nab = (2 * sp.la + 1) * (2 * sp.lb + 1);
            double peak = 0.0;
            for (int i = 0; i < nab; ++i) peak = std::max(peak, std::fabs(wk.sph[size_t(i * nab + i)]));
            bound[size_t(k)] = std::sqrt(peak);
        }
#pragma omp for schedule(dynamic, 1)
        for (i64 bra = 0; bra < npairs; ++bra)
            for (i64 ket = 0; ket <= bra; ++ket) {
                const ShellPair &bp = pairs[size_t(bra)], &kp = pairs[size_t(ket)];
                const i64 a = bp.a, b = bp.b, c = kp.a, d = kp.b;
                const int na = 2 * bp.la + 1, nb = 2 * bp.lb + 1, nc = 2 * kp.la + 1, nd = 2 * kp.lb + 1;
                const bool skip = bound[size_t(bra)] * bound[size_t(ket)] < screen;
                if (!skip) {
                    eri_cart_fast(bp, kp, wk);
                    eri_spherical_fast(sh, bp.la, bp.lb, kp.la, kp.lb, wk);
                }
                for (int i = 0; i < na; ++i)
                    for (int j = 0; j < nb; ++j) {
                        const i64 ii = sh.ao_off[a] + i, jj = sh.ao_off[b] + j;
                        if (a == b && jj > ii) continue;
                        const i64 I = ii * (ii + 1) / 2 + jj;
                        for (int k2 = 0; k2 < nc; ++k2)
                            for (int l = 0; l < nd; ++l) {
                                const i64 kk = sh.ao_off[c] + k2, ll = sh.ao_off[d] + l;
                                if (c == d && ll > kk) continue;
                                const i64 J = kk * (kk + 1) / 2 + ll;
                                if (bra == ket && J > I) continue;
                                const i64 hi = std::max(I, J), lo = std::min(I, J);
                                out[hi * (hi + 1) / 2 + lo] =
                                    skip ? 0.0 : wk.sph[size_t(((i * nb + j) * nc + k2) * nd + l)];
                            }
                    }
            }
    }
}
