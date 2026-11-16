// Element-wise work of the restricted CCSDT triples residual on the CPU.
// The GEMMs between these passes run on the GEMM executor (itercc_sym_contract);
// see itercc/_cpu_triples.py.
#include <cstdlib>
#include <cstring>
#include <vector>

#include "itercc/core.h"

using i64 = int64_t;

// Plane P[x,y] = T(x,y,k) from packed T3, with the operands of the crossed
// and direct particle-hole GEMMs. slot[xy] is the packed slot (or -1) and
// stride[3 xy..] the source strides of the tile's (d,b,c) axes; pos maps the
// row (l,d) to its particle-hole sector position.
//   crossed[pos(y,d), x, b, c] = P[x,y,d,b,c]
//   direct[pos(x,d), y, b, c] = 2 P[x,y,d,b,c] - P[x,y,b,d,c] - P[x,y,c,b,d]
extern "C" void itercc_triples_plane(const double *t3, i64 no, i64 nv, const i64 *slot,
                                     const i64 *stride, const i64 *pos, double *plane,
                                     double *crossed, double *direct, int threads)
{
    const i64 nv2 = nv * nv, nv3 = nv2 * nv;
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for (i64 xy = 0; xy < no * no; ++xy) {
        for (i64 d = 0; d < nv; ++d) {
            const i64 x = xy / no, y = xy % no;
            double *p = plane + (xy * nv + d) * nv2;
            double *cr = crossed + (pos[y * nv + d] * no + x) * nv2;
            double *dr = direct + (pos[x * nv + d] * no + y) * nv2;
            const i64 s = slot[xy];
            if (s < 0) {
                std::memset(p, 0, nv2 * sizeof(double));
                std::memset(cr, 0, nv2 * sizeof(double));
                std::memset(dr, 0, nv2 * sizeof(double));
                continue;
            }
            const double *src = t3 + s * nv3;
            const i64 t0 = stride[3 * xy], t1 = stride[3 * xy + 1], t2 = stride[3 * xy + 2];
            for (i64 b = 0; b < nv; ++b) {
                for (i64 c = 0; c < nv; ++c) {
                    const double v = src[d * t0 + b * t1 + c * t2];
                    const double v_bdc = src[b * t0 + d * t1 + c * t2];
                    const double v_cbd = src[c * t0 + b * t1 + d * t2];
                    p[b * nv + c] = v;
                    cr[b * nv + c] = v;
                    dr[b * nv + c] = 2.0 * v - v_bdc - v_cbd;
                }
            }
        }
    }
}

// Seeds S[i,j,a,b,c] of one k plane, in place over the occupied-ladder
// product when have_t3 (seeds holds it on entry):
//   S = O + A[j,a,b,i,c] + D[pos(i,a),j,b,c] - C[pos(i,b),j,a,c]
//       - 0.5 C[pos(i,a),j,b,c] - B[i,j,a,b,c],
// and S = A - B without T3.
extern "C" void itercc_triples_seeds(i64 no, i64 nv, const i64 *pos, int have_t3,
                                     const double *a_seed, const double *b_seed,
                                     const double *direct, const double *crossed, double *seeds,
                                     int threads)
{
    const i64 nv2 = nv * nv;
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for (i64 ij = 0; ij < no * no; ++ij) {
        for (i64 a = 0; a < nv; ++a) {
            const i64 i = ij / no, j = ij % no;
            const i64 base = (ij * nv + a) * nv2;
            for (i64 b = 0; b < nv; ++b) {
                const double *arow = a_seed + (((j * nv + a) * nv + b) * no + i) * nv;
                for (i64 c = 0; c < nv; ++c) {
                    const i64 at = base + b * nv + c;
                    double value = have_t3 ? seeds[at] : 0.0;
                    value += arow[c];
                    if (have_t3) {
                        value += direct[((pos[i * nv + a] * no + j) * nv + b) * nv + c];
                        value -= crossed[((pos[i * nv + b] * no + j) * nv + a) * nv + c];
                        value -= 0.5 * crossed[((pos[i * nv + a] * no + j) * nv + b) * nv + c];
                    }
                    value -= b_seed[at];
                    seeds[at] = value;
                }
            }
        }
    }
}

// Add the seeds of one k plane to packed R3. Target t (slot target[t])
// collects contributions start[t] .. start[t+1]-1, each (source tile, q0, q1,
// q2): R[u0,u1,u2] += seeds[src][u0 q0 + u1 q1 + u2 q2]. Each thread owns
// whole rows R[slot][u0].
extern "C" void itercc_triples_scatter(i64 nv, i64 ntargets, const i64 *target,
                                       const i64 *start, const i64 *contrib,
                                       const double *seeds, double *residual, int threads)
{
    const i64 nv2 = nv * nv, nv3 = nv2 * nv;
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for (i64 t = 0; t < ntargets; ++t) {
        for (i64 u0 = 0; u0 < nv; ++u0) {
            double *row = residual + target[t] * nv3 + u0 * nv2;
            for (i64 n = start[t]; n < start[t + 1]; ++n) {
                const i64 *e = contrib + 4 * n;
                const double *src = seeds + e[0] * nv3 + u0 * e[1];
                const i64 q1 = e[2], q2 = e[3];
                for (i64 u1 = 0; u1 < nv; ++u1)
                    for (i64 u2 = 0; u2 < nv; ++u2)
                        row[u1 * nv + u2] += src[u1 * q1 + u2 * q2];
            }
        }
    }
}

// Feedback plane for first occupied index m: q[i,j,x,y,z] is the spin sum
// 2F(x,y,z) - F(y,x,z) - F(z,y,x) of F = T(m,i,j), and X[i,j,a,x,y] = q[i,j,x,a,y].
extern "C" void itercc_feedback_plane(const double *t3, i64 no, i64 nv, const i64 *slot,
                                      const i64 *stride, double *q, double *x_plane, int threads)
{
    const i64 nv2 = nv * nv, nv3 = nv2 * nv;
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for (i64 ij = 0; ij < no * no; ++ij) {
        for (i64 xx = 0; xx < nv; ++xx) {
            double *qrow = q + (ij * nv + xx) * nv2;
            double *xp = x_plane + ij * nv3 + xx * nv;
            const i64 s = slot[ij];
            if (s < 0) {
                std::memset(qrow, 0, nv2 * sizeof(double));
                for (i64 y = 0; y < nv; ++y)
                    std::memset(xp + y * nv2, 0, nv * sizeof(double));
                continue;
            }
            const double *src = t3 + s * nv3;
            const i64 t0 = stride[3 * ij], t1 = stride[3 * ij + 1], t2 = stride[3 * ij + 2];
            for (i64 y = 0; y < nv; ++y) {
                for (i64 z = 0; z < nv; ++z) {
                    const double value = 2.0 * src[xx * t0 + y * t1 + z * t2]
                                         - src[y * t0 + xx * t1 + z * t2]
                                         - src[z * t0 + y * t1 + xx * t2];
                    qrow[y * nv + z] = value;
                    xp[y * nv2 + z] = value;
                }
            }
        }
    }
}

// Ladder operands of n slots for the three pair orientations o:
//   sym[e,o,s,c] = (Q[p,q,c] + Q[q,p,c]) / 2 over pairs (pe, qe),
//   anti[e,o,s,c] = (Q[r,s,c] - Q[s,r,c]) / 2 over pairs (ro, so),
// where Q is the slot tile read with strides orient[3 o .. 3 o + 2].
extern "C" void itercc_ladder_gather(const double *t3, i64 n, i64 nv, i64 ne, const i64 *pe,
                                     const i64 *qe, i64 nodd, const i64 *ro, const i64 *so,
                                     const i64 *orient, double *sym, double *anti, int threads)
{
    const i64 nv3 = nv * nv * nv;
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for (i64 s = 0; s < n; ++s) {
        for (i64 o = 0; o < 3; ++o) {
            const double *tile = t3 + s * nv3;
            const i64 t0 = orient[3 * o], t1 = orient[3 * o + 1], t2 = orient[3 * o + 2];
            for (i64 e = 0; e < ne; ++e) {
                const i64 p = pe[e], q = qe[e];
                double *out = sym + ((e * 3 + o) * n + s) * nv;
                for (i64 c = 0; c < nv; ++c)
                    out[c] = 0.5 * (tile[p * t0 + q * t1 + c * t2] + tile[q * t0 + p * t1 + c * t2]);
            }
            for (i64 e = 0; e < nodd; ++e) {
                const i64 p = ro[e], q = so[e];
                double *out = anti + ((e * 3 + o) * n + s) * nv;
                for (i64 c = 0; c < nv; ++c)
                    out[c] = 0.5 * (tile[p * t0 + q * t1 + c * t2] - tile[q * t0 + p * t1 + c * t2]);
            }
        }
    }
}

// Add the ladder products back: R[s][w] += sum_o V_o[w[perm_o0], w[perm_o1],
// w[perm_o2]], where V_o[x,y,c] is the symmetric product of pair {x,y} plus
// the signed antisymmetric product (zero on x = y).
extern "C" void itercc_ladder_scatter(i64 n, i64 nv, const i64 *even_row, const i64 *odd_row,
                                      const double *odd_sign, const i64 *perm, const double *sym,
                                      const double *anti, double *residual, int threads)
{
    const i64 nv2 = nv * nv, nv3 = nv2 * nv;
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for (i64 s = 0; s < n; ++s) {
        for (i64 w0 = 0; w0 < nv; ++w0) {
            double *row = residual + s * nv3 + w0 * nv2;
            for (i64 w1 = 0; w1 < nv; ++w1) {
                for (i64 w2 = 0; w2 < nv; ++w2) {
                    const i64 w[3] = {w0, w1, w2};
                    double total = 0.0;
                    for (i64 o = 0; o < 3; ++o) {
                        const i64 x = w[perm[3 * o]], y = w[perm[3 * o + 1]], c = w[perm[3 * o + 2]];
                        const i64 pair = x * nv + y;
                        total += sym[((even_row[pair] * 3 + o) * n + s) * nv + c];
                        const double sign = odd_sign[pair];
                        if (sign != 0.0)
                            total += sign * anti[((odd_row[pair] * 3 + o) * n + s) * nv + c];
                    }
                    row[w1 * nv + w2] += total;
                }
            }
        }
    }
}

// Remove the fully symmetric virtual component of every slot and zero the
// all-equal virtual entries.
extern "C" void itercc_purify(double *r3, i64 nslots, i64 nv, int threads)
{
    const i64 nv2 = nv * nv, nv3 = nv2 * nv;
#pragma omp parallel num_threads(threads)
    {
        std::vector<double> buffer(nv3);
#pragma omp for schedule(static)
        for (i64 s = 0; s < nslots; ++s) {
            double *t = r3 + s * nv3;
            for (i64 a = 0; a < nv; ++a)
                for (i64 b = 0; b < nv; ++b)
                    for (i64 c = 0; c < nv; ++c) {
                        const double sum = t[a * nv2 + b * nv + c] + t[b * nv2 + a * nv + c]
                                           + t[c * nv2 + b * nv + a] + t[a * nv2 + c * nv + b]
                                           + t[c * nv2 + a * nv + b] + t[b * nv2 + c * nv + a];
                        buffer[a * nv2 + b * nv + c] = t[a * nv2 + b * nv + c] - sum / 6.0;
                    }
            std::memcpy(t, buffer.data(), nv3 * sizeof(double));
            for (i64 d = 0; d < nv; ++d)
                t[d * nv2 + d * nv + d] = 0.0;
        }
    }
}

// T3 Jacobi step, in place: r3[s] *= scale, then r3[s] /= (occupied[s] - virtual).
extern "C" void itercc_triples_step(double *r3, i64 nslots, i64 nv3, const double *occupied,
                                    const double *virtual_sums, double scale, int threads)
{
#pragma omp parallel for schedule(static) num_threads(threads)
    for (i64 s = 0; s < nslots; ++s) {
        double *t = r3 + s * nv3;
        const double e = occupied[s];
        for (i64 w = 0; w < nv3; ++w) {
            t[w] *= scale;
            t[w] /= e - virtual_sums[w];
        }
    }
}
