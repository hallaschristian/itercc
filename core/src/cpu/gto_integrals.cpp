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
    if (t > 45.0) {                                  // exp(-t) below 3e-20: the asymptotic form
        f[0] = 0.5 * std::sqrt(kPi / t);
        for (int m = 1; m <= mmax; ++m) f[m] = f[m - 1] * (2 * m - 1) / (2.0 * t);
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

// Cartesian (ab|cd) of contracted shells a, b, c, d into cart (nca, ncb, ncc, ncd).
void eri_cart(const Shells &sh, i64 a, i64 b, i64 c, i64 d, std::vector<double> &cart, Hermite1 *eab,
              Hermite1 *ecd, std::vector<double> &r, std::vector<double> &prev, std::vector<double> &w, double *f)
{
    const int la = sh.l[a], lb = sh.l[b], lc = sh.l[c], ld = sh.l[d];
    const auto ca = components(la), cb = components(lb), cc = components(lc), cd = components(ld);
    const int nca = int(ca.size()), ncb = int(cb.size()), ncc = int(cc.size()), ncd = int(cd.size());
    const int lab = la + lb, lcd = lc + ld, L = lab + lcd, s = L + 1;
    const int sb = lab + 1;                                  // bra Hermite cube side
    cart.assign(size_t(nca * ncb * ncc * ncd), 0.0);
    const double *A = sh.xyz + 3 * sh.atom[a], *B = sh.xyz + 3 * sh.atom[b];
    const double *C = sh.xyz + 3 * sh.atom[c], *D = sh.xyz + 3 * sh.atom[d];
    double ab2 = 0.0, cd2 = 0.0;
    for (int k = 0; k < 3; ++k) {
        ab2 += (A[k] - B[k]) * (A[k] - B[k]);
        cd2 += (C[k] - D[k]) * (C[k] - D[k]);
    }
    const int ncdp = ncc * ncd;
    w.resize(size_t(sb * sb * sb * ncdp));
    for (int ip = 0; ip < sh.nprim[a]; ++ip)
        for (int jp = 0; jp < sh.nprim[b]; ++jp) {
            const double ea = sh.exps[sh.prim0[a] + ip], eb = sh.exps[sh.prim0[b] + jp];
            const double p = ea + eb;
            const double kab = std::exp(-ea * eb / p * ab2) * sh.coefs[sh.prim0[a] + ip] * sh.coefs[sh.prim0[b] + jp];
            double P[3];
            for (int k = 0; k < 3; ++k) P[k] = (ea * A[k] + eb * B[k]) / p;
            for (int k = 0; k < 3; ++k) eab[k].build(la, lb, p, P[k] - A[k], P[k] - B[k], 1.0);
            for (int kp = 0; kp < sh.nprim[c]; ++kp)
                for (int lp = 0; lp < sh.nprim[d]; ++lp) {
                    const double ec = sh.exps[sh.prim0[c] + kp], ed = sh.exps[sh.prim0[d] + lp];
                    const double q = ec + ed;
                    const double kcd = std::exp(-ec * ed / q * cd2) * sh.coefs[sh.prim0[c] + kp] *
                                       sh.coefs[sh.prim0[d] + lp];
                    double Q[3];
                    for (int k = 0; k < 3; ++k) Q[k] = (ec * C[k] + ed * D[k]) / q;
                    for (int k = 0; k < 3; ++k) ecd[k].build(lc, ld, q, Q[k] - C[k], Q[k] - D[k], 1.0);
                    const double alpha = p * q / (p + q);
                    hermite_r(L, alpha, P[0] - Q[0], P[1] - Q[1], P[2] - Q[2], r, prev, f);
                    const double pref = 2.0 * std::pow(kPi, 2.5) / (p * q * std::sqrt(p + q)) * kab * kcd;
                    // w(t, u, v; cd) = sum (-1)^(tau+nu+phi) E^cd R(t + tau, u + nu, v + phi)
                    for (int t = 0; t <= lab; ++t)
                        for (int u = 0; u <= lab - t; ++u)
                            for (int v = 0; v <= lab - t - u; ++v)
                                for (int k = 0; k < ncc; ++k)
                                    for (int l = 0; l < ncd; ++l) {
                                        const Comp &g = cc[size_t(k)], &h = cd[size_t(l)];
                                        double sum = 0.0;
                                        for (int tau = 0; tau <= g.x + h.x; ++tau) {
                                            const double e1 = ecd[0].get(g.x, h.x, tau);
                                            if (e1 == 0.0) continue;
                                            for (int nu = 0; nu <= g.y + h.y; ++nu) {
                                                const double e2 = e1 * ecd[1].get(g.y, h.y, nu);
                                                if (e2 == 0.0) continue;
                                                for (int phi = 0; phi <= g.z + h.z; ++phi) {
                                                    const double e3 = e2 * ecd[2].get(g.z, h.z, phi);
                                                    const double sign = ((tau + nu + phi) & 1) ? -1.0 : 1.0;
                                                    sum += sign * e3 * r[size_t(((t + tau) * s + u + nu) * s + v + phi)];
                                                }
                                            }
                                        }
                                        w[size_t(((t * sb + u) * sb + v) * ncdp + k * ncd + l)] = sum;
                                    }
                    for (int i = 0; i < nca; ++i)
                        for (int j = 0; j < ncb; ++j) {
                            const Comp &u1 = ca[size_t(i)], &w1 = cb[size_t(j)];
                            double *o = cart.data() + size_t((i * ncb + j) * ncdp);
                            for (int t = 0; t <= u1.x + w1.x; ++t) {
                                const double e1 = eab[0].get(u1.x, w1.x, t);
                                if (e1 == 0.0) continue;
                                for (int u = 0; u <= u1.y + w1.y; ++u) {
                                    const double e2 = e1 * eab[1].get(u1.y, w1.y, u);
                                    if (e2 == 0.0) continue;
                                    for (int v = 0; v <= u1.z + w1.z; ++v) {
                                        const double e3 = pref * e2 * eab[2].get(u1.z, w1.z, v);
                                        if (e3 == 0.0) continue;
                                        const double *wr = w.data() + size_t(((t * sb + u) * sb + v) * ncdp);
                                        for (int m = 0; m < ncdp; ++m) o[m] += e3 * wr[m];
                                    }
                                }
                            }
                        }
                }
        }
}

// Spherical (ab|cd) from the Cartesian block: four one-index transformations.
void eri_spherical(const Shells &sh, int la, int lb, int lc, int ld, const std::vector<double> &cart,
                   std::vector<double> &tmp, std::vector<double> &sph)
{
    const int c[4] = {(la + 1) * (la + 2) / 2, (lb + 1) * (lb + 2) / 2, (lc + 1) * (lc + 2) / 2,
                      (ld + 1) * (ld + 2) / 2};
    const int n[4] = {2 * la + 1, 2 * lb + 1, 2 * lc + 1, 2 * ld + 1};
    const int ls[4] = {la, lb, lc, ld};
    std::vector<double> cur(cart);
    int dims[4] = {c[0], c[1], c[2], c[3]};
    for (int axis = 0; axis < 4; ++axis) {
        const double *t = sh.c2s + sh.c2s_off[ls[axis]];
        int before = 1, after = 1;
        for (int k = 0; k < axis; ++k) before *= dims[k];
        for (int k = axis + 1; k < 4; ++k) after *= dims[k];
        tmp.assign(size_t(before * n[axis] * after), 0.0);
        for (int x = 0; x < before; ++x)
            for (int i = 0; i < c[axis]; ++i) {
                const double *src = cur.data() + size_t((x * c[axis] + i) * after);
                for (int m = 0; m < n[axis]; ++m) {
                    const double coef = t[i * n[axis] + m];
                    if (coef == 0.0) continue;
                    double *dst = tmp.data() + size_t((x * n[axis] + m) * after);
                    for (int y = 0; y < after; ++y) dst[y] += coef * src[y];
                }
            }
        dims[axis] = n[axis];
        cur.swap(tmp);
    }
    sph.swap(cur);
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
    // a Schwarz bound below `screen` set to zero. Each packed entry belongs to
    // one shell quartet (its two shell pairs), computed by one task.
    const Shells sh{nshell, shell_l, shell_atom, shell_nprim, shell_prim0, ao_off, c2s_off,
                    exps, coefs, atom_xyz, c2s};
    const i64 npairs = nshell * (nshell + 1) / 2;
    std::vector<i64> pa(static_cast<size_t>(npairs));
    std::vector<i64> pb(static_cast<size_t>(npairs));
    for (i64 a = 0, k = 0; a < nshell; ++a)
        for (i64 b = 0; b <= a; ++b, ++k) {
            pa[size_t(k)] = a;
            pb[size_t(k)] = b;
        }
    const i64 npair_ao = nao * (nao + 1) / 2;
    std::vector<double> bound(static_cast<size_t>(npairs), 0.0);
#pragma omp parallel num_threads(threads)
    {
        Hermite1 eab[3], ecd[3];
        std::vector<double> r, prev, w, cart, tmp, sph;
        double f[80];
#pragma omp for schedule(dynamic, 1)
        for (i64 k = 0; k < npairs; ++k) {           // Schwarz: max |(ab|ab)|^(1/2)
            const i64 a = pa[size_t(k)], b = pb[size_t(k)];
            eri_cart(sh, a, b, a, b, cart, eab, ecd, r, prev, w, f);
            eri_spherical(sh, sh.l[a], sh.l[b], sh.l[a], sh.l[b], cart, tmp, sph);
            const int na = 2 * sh.l[a] + 1, nb = 2 * sh.l[b] + 1;
            double peak = 0.0;
            for (int i = 0; i < na * nb; ++i) peak = std::max(peak, std::fabs(sph[size_t(i * na * nb + i)]));
            bound[size_t(k)] = std::sqrt(peak);
        }
#pragma omp for schedule(dynamic, 1)
        for (i64 bra = 0; bra < npairs; ++bra)
            for (i64 ket = 0; ket <= bra; ++ket) {
                const i64 a = pa[size_t(bra)], b = pb[size_t(bra)], c = pa[size_t(ket)], d = pb[size_t(ket)];
                const int na = 2 * sh.l[a] + 1, nb = 2 * sh.l[b] + 1, nc = 2 * sh.l[c] + 1, nd = 2 * sh.l[d] + 1;
                const bool skip = bound[size_t(bra)] * bound[size_t(ket)] < screen;
                if (!skip) {
                    eri_cart(sh, a, b, c, d, cart, eab, ecd, r, prev, w, f);
                    eri_spherical(sh, sh.l[a], sh.l[b], sh.l[c], sh.l[d], cart, tmp, sph);
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
                                    skip ? 0.0 : sph[size_t(((i * nb + j) * nc + k2) * nd + l)];
                            }
                    }
            }
    }
    (void)npair_ao;
}
