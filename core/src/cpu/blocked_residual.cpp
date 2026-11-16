// Restricted CCSDT residual pieces on symmetry-blocked T3 (CPU).
// The algebra is that of itercc/_cpu_triples.py; only allowed entries are
// stored, so the GEMM column ranges shrink with the symmetry.
#include <algorithm>
#include <cstring>
#include <vector>

#include "itercc/core.h"

using i64 = int64_t;

namespace {

const i64 kPerm[3][3] = {{0, 1, 2}, {0, 2, 1}, {1, 2, 0}};

// Position of allowed entry (a,b,c) inside a slot of irrep gamma.
inline i64 tpos(const itercc_blocked_t3 *t, i64 gamma, i64 a, i64 b, i64 c)
{
    const i64 g = t->groups, gb = t->irrep[b], gc = t->irrep[c];
    return t->toff[(gamma * g + t->irrep[a]) * g + gb]
           + (t->rank[a] * t->count[gb] + t->rank[b]) * t->count[gc] + t->rank[c];
}

// Visit every allowed (a, b, c) of a slot of irrep gamma in storage order.
template <class Visit>
inline void for_each_entry(const itercc_blocked_t3 *t, i64 gamma, Visit visit)
{
    i64 position = 0;
    for (i64 ga = 0; ga < t->groups; ++ga)
        for (i64 gb = 0; gb < t->groups; ++gb) {
            const i64 gc = gamma ^ ga ^ gb;
            const i64 *va = t->order + t->start[ga], *vb = t->order + t->start[gb];
            const i64 *vc = t->order + t->start[gc];
            for (i64 ra = 0; ra < t->count[ga]; ++ra)
                for (i64 rb = 0; rb < t->count[gb]; ++rb)
                    for (i64 rc = 0; rc < t->count[gc]; ++rc)
                        visit(position++, va[ra], vb[rb], vc[rc]);
        }
}

// Value of the tile T(x,y,k) at output coordinates (d,b,c): the packed slot
// read through the from_sorted permutation f (w[f[m]] = output m).
inline double tile_value(const itercc_blocked_t3 *t, const double *tile, i64 gamma,
                         const i64 *f, i64 d, i64 b, i64 c)
{
    i64 w[3];
    w[f[0]] = d, w[f[1]] = b, w[f[2]] = c;
    return tile[tpos(t, gamma, w[0], w[1], w[2])];
}

// Index of the virtual pair (b,c) in the pair list of its irrep.
inline i64 pair_index(const itercc_blocked_t3 *t, const itercc_blocked_plane *p, i64 b, i64 c)
{
    const i64 gb = t->irrep[b], gc = t->irrep[c], tau = gb ^ gc;
    return p->poff[tau * t->groups + gb] + t->rank[b] * t->count[gc] + t->rank[c];
}

// Start of the particle-hole operand row (l,d) at the column block of j.
inline i64 ph_at(const itercc_blocked_t3 *t, const itercc_blocked_plane *p, i64 l, i64 d, i64 j)
{
    const i64 nv = t->nv, sector = p->occ_irrep[l] ^ t->irrep[d];
    const i64 kappa = sector ^ p->occ_irrep[p->k];
    return p->ph_base[sector] + p->ph_row[l * nv + d] * p->ph_ncols[sector]
           + p->jcol[kappa * p->no + j];
}

}  // namespace

extern "C" void itercc_bt3_plane(const itercc_blocked_t3 *t, const itercc_blocked_plane *p,
                                 const double *t3, double *plane, double *crossed, double *direct,
                                 int threads)
{
    const i64 g = t->groups, no = p->no, gk = p->occ_irrep[p->k];
#pragma omp parallel for collapse(2) schedule(dynamic) num_threads(threads)
    for (i64 xy = 0; xy < no * no; ++xy) {
        for (i64 gd = 0; gd < g; ++gd) {
            const i64 x = xy / no, y = xy % no;
            const i64 gxy = p->occ_irrep[x] ^ p->occ_irrep[y], gamma = gxy ^ gk, tau = gamma ^ gd;
            const i64 slot = p->tile_slot[xy];
            const i64 *f = p->tile_perm + 3 * xy;
            const double *tile = slot >= 0 ? t3 + t->slot_offset[slot] : nullptr;
            double *prow = plane + p->pair_base[gxy] + p->pair_row[xy] * p->ntriple[gamma];
            const i64 *vd = t->order + t->start[gd];
            for (i64 rd = 0; rd < t->count[gd]; ++rd) {
                const i64 d = vd[rd];
                double *crow = crossed + ph_at(t, p, y, d, x);
                double *drow = direct + ph_at(t, p, x, d, y);
                for (i64 gb = 0; gb < g; ++gb) {
                    const i64 gc = tau ^ gb, nb = t->count[gb], nc = t->count[gc];
                    const i64 *vb = t->order + t->start[gb], *vc = t->order + t->start[gc];
                    const i64 pbase = p->poff[tau * g + gb];
                    const i64 tbase = t->toff[(gamma * g + gd) * g + gb] + rd * nb * nc;
                    for (i64 rb = 0; rb < nb; ++rb)
                        for (i64 rc = 0; rc < nc; ++rc) {
                            const i64 b = vb[rb], c = vc[rc];
                            double v = 0.0, v_bdc = 0.0, v_cbd = 0.0;
                            if (tile) {
                                v = tile_value(t, tile, gamma, f, d, b, c);
                                v_bdc = tile_value(t, tile, gamma, f, b, d, c);
                                v_cbd = tile_value(t, tile, gamma, f, c, b, d);
                            }
                            prow[tbase + rb * nc + rc] = v;
                            crow[pbase + rb * nc + rc] = v;
                            drow[pbase + rb * nc + rc] = 2.0 * v - v_bdc - v_cbd;
                        }
                }
            }
        }
    }
}

extern "C" void itercc_bt3_seeds(const itercc_blocked_t3 *t, const itercc_blocked_plane *p,
                                 int have_t3, const double *a_seed, const double *b_seed,
                                 const double *direct, const double *crossed, double *seeds,
                                 int threads)
{
    const i64 g = t->groups, no = p->no, nv = t->nv, gk = p->occ_irrep[p->k];
#pragma omp parallel for collapse(2) schedule(dynamic) num_threads(threads)
    for (i64 n = 0; n < no * no; ++n) {
        for (i64 ga = 0; ga < g; ++ga) {
            const i64 xy = p->pair_order[n], i = xy / no, j = xy % no;
            const i64 gij = p->occ_irrep[i] ^ p->occ_irrep[j], gamma = gij ^ gk;
            double *srow = seeds + p->pair_base[gij] + p->pair_row[xy] * p->ntriple[gamma];
            const i64 *va = t->order + t->start[ga];
            for (i64 gb = 0; gb < g; ++gb) {
                const i64 gc = gamma ^ ga ^ gb, nb = t->count[gb], nc = t->count[gc];
                const i64 *vb = t->order + t->start[gb], *vc = t->order + t->start[gc];
                const i64 base = t->toff[(gamma * g + ga) * g + gb];
                for (i64 ra = 0; ra < t->count[ga]; ++ra) {
                    const i64 a = va[ra];
                    // B row (i,j,a): irrep gij ^ irrep(a); its columns are the pairs (b,c).
                    const i64 gb_row = gij ^ ga;
                    const double *brow = b_seed + p->b_base[gb_row]
                                         + p->b_row[(i * no + j) * nv + a] * p->b_ncols[gb_row];
                    for (i64 rb = 0; rb < nb; ++rb) {
                        const i64 b = vb[rb];
                        // A row (j,a,b): irrep(j) ^ irrep(a) ^ irrep(b); columns (i,c).
                        const i64 ga_row = p->occ_irrep[j] ^ ga ^ gb;
                        const double *arow = a_seed + p->a_base[ga_row]
                                             + p->a_row[(j * nv + a) * nv + b] * p->a_ncols[ga_row];
                        for (i64 rc = 0; rc < nc; ++rc) {
                            const i64 c = vc[rc], at = base + (ra * nb + rb) * nc + rc;
                            double value = have_t3 ? srow[at] : 0.0;
                            value += arow[p->a_col[i * nv + c]];
                            if (have_t3) {
                                const i64 dpos = ph_at(t, p, i, a, j) + pair_index(t, p, b, c);
                                value += direct[dpos];
                                value -= crossed[ph_at(t, p, i, b, j) + pair_index(t, p, a, c)];
                                value -= 0.5 * crossed[dpos];
                            }
                            value -= brow[pair_index(t, p, b, c)];
                            srow[at] = value;
                        }
                    }
                }
            }
        }
    }
}

extern "C" void itercc_bt3_scatter(const itercc_blocked_t3 *t, const itercc_blocked_plane *p,
                                   i64 ntargets, const i64 *target, const i64 *start,
                                   const i64 *contrib, const double *seeds, double *r3, int threads)
{
    const i64 g = t->groups, no = p->no;
#pragma omp parallel for collapse(2) schedule(dynamic) num_threads(threads)
    for (i64 n = 0; n < ntargets; ++n) {
        for (i64 ga = 0; ga < g; ++ga) {
            const i64 slot = target[n], gt = t->slot_irrep[slot];
            double *tile = r3 + t->slot_offset[slot];
            const i64 *va = t->order + t->start[ga];
            for (i64 gb = 0; gb < g; ++gb) {
                const i64 gc = gt ^ ga ^ gb, nb = t->count[gb], nc = t->count[gc];
                const i64 *vb = t->order + t->start[gb], *vc = t->order + t->start[gc];
                const i64 base = t->toff[(gt * g + ga) * g + gb];
                for (i64 ra = 0; ra < t->count[ga]; ++ra)
                    for (i64 rb = 0; rb < nb; ++rb)
                        for (i64 rc = 0; rc < nc; ++rc) {
                            const i64 u[3] = {va[ra], vb[rb], vc[rc]};
                            const i64 at = base + (ra * nb + rb) * nc + rc;
                            double value = tile[at];
                            for (i64 e = start[n]; e < start[n + 1]; ++e) {
                                const i64 *entry = contrib + 4 * e, xy = entry[0];
                                i64 w[3];
                                w[entry[1]] = u[0], w[entry[2]] = u[1], w[entry[3]] = u[2];
                                const i64 gxy = p->occ_irrep[xy / no] ^ p->occ_irrep[xy % no];
                                value += seeds[p->pair_base[gxy] + p->pair_row[xy] * p->ntriple[gt]
                                               + tpos(t, gt, w[0], w[1], w[2])];
                            }
                            tile[at] = value;
                        }
            }
        }
    }
}

extern "C" void itercc_bt3_feedback_plane(const itercc_blocked_t3 *t, i64 no, const i64 *tile_slot,
                                          const i64 *tile_perm, const double *t3, double *q,
                                          double *x_plane, int threads)
{
    const i64 nv = t->nv, nv2 = nv * nv, nv3 = nv2 * nv;
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for (i64 ij = 0; ij < no * no; ++ij) {
        for (i64 xx = 0; xx < nv; ++xx) {
            double *qrow = q + (ij * nv + xx) * nv2;
            double *xp = x_plane ? x_plane + ij * nv3 + xx * nv : nullptr;
            const i64 slot = tile_slot[ij];
            const i64 *f = tile_perm + 3 * ij;
            const i64 gamma = slot >= 0 ? t->slot_irrep[slot] : -1;
            const double *tile = slot >= 0 ? t3 + t->slot_offset[slot] : nullptr;
            for (i64 y = 0; y < nv; ++y)
                for (i64 z = 0; z < nv; ++z) {
                    double value = 0.0;
                    if (tile && (t->irrep[xx] ^ t->irrep[y] ^ t->irrep[z]) == gamma)
                        value = 2.0 * tile_value(t, tile, gamma, f, xx, y, z)
                                - tile_value(t, tile, gamma, f, y, xx, z)
                                - tile_value(t, tile, gamma, f, z, y, xx);
                    qrow[y * nv + z] = value;
                    if (xp) xp[y * nv2 + z] = value;
                }
        }
    }
}

// Operands of the three pair orientations: row = pair (p,q), column = (slot,
// o, z): sym = (Q[p,q,z] + Q[q,p,z]) / 2 and anti = (Q[r,s,z] - Q[s,r,z]) / 2
// with Q the slot tile in orientation o (Q[i0,i1,i2] = T[w], w[perm_o[m]] = i_m).
extern "C" void itercc_bt3_ladder_gather(const itercc_blocked_t3 *t, const itercc_blocked_ladder *l,
                                         const double *t3, double *sym, double *anti, int threads)
{
    const i64 g = t->groups, neven = l->even_start[g], nodd = l->odd_start[g];
#pragma omp parallel for schedule(dynamic, 4) num_threads(threads)
    for (i64 row = 0; row < neven + nodd; ++row) {
        const bool even = row < neven;
        const i64 e = even ? row : row - neven;
        const i64 p = even ? l->even_p[e] : l->odd_r[e], q = even ? l->even_q[e] : l->odd_s[e];
        const i64 pi = t->irrep[p] ^ t->irrep[q];
        const i64 local = e - (even ? l->even_start[pi] : l->odd_start[pi]), ncols = l->ncols[pi];
        double *out = (even ? sym + l->sym_base[pi] : anti + l->anti_base[pi]) + local * ncols;
        const double sign = even ? 1.0 : -1.0;
        for (i64 s = 0; s < l->nslots; ++s) {
            const i64 slot = l->first_slot + s, gs = t->slot_irrep[slot], gz = pi ^ gs;
            const double *tile = t3 + t->slot_offset[slot];
            const i64 nz = t->count[gz];
            const i64 *vz = t->order + t->start[gz];
            for (i64 o = 0; o < 3; ++o) {
                double *column = out + l->colstart[pi * l->nslots + s] + o * nz;
                const i64 *perm = kPerm[o];
                for (i64 rz = 0; rz < nz; ++rz) {
                    i64 w[3], v[3];
                    w[perm[0]] = p, w[perm[1]] = q, w[perm[2]] = vz[rz];
                    v[perm[0]] = q, v[perm[1]] = p, v[perm[2]] = vz[rz];
                    column[rz] = 0.5 * (tile[tpos(t, gs, w[0], w[1], w[2])]
                                        + sign * tile[tpos(t, gs, v[0], v[1], v[2])]);
                }
            }
        }
    }
}

// R[slot][w] += sum_o V_o[w[perm_o0], w[perm_o1], w[perm_o2]], V_o the
// symmetric product of the pair plus the signed antisymmetric product.
extern "C" void itercc_bt3_ladder_scatter(const itercc_blocked_t3 *t, const itercc_blocked_ladder *l,
                                          const double *sym, const double *anti, double *r3,
                                          int threads)
{
    const i64 nv = t->nv;
#pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (i64 s = 0; s < l->nslots; ++s) {
        const i64 slot = l->first_slot + s, gs = t->slot_irrep[slot];
        double *tile = r3 + t->slot_offset[slot];
        for_each_entry(t, gs, [&](i64 position, i64 a, i64 b, i64 c) {
            const i64 w[3] = {a, b, c};
            double total = 0.0;
            for (i64 o = 0; o < 3; ++o) {
                const i64 *perm = kPerm[o];
                const i64 x = w[perm[0]], y = w[perm[1]], z = w[perm[2]];
                const i64 pi = t->irrep[x] ^ t->irrep[y], pair = x * nv + y, ncols = l->ncols[pi];
                const i64 column = l->colstart[pi * l->nslots + s] + o * t->count[t->irrep[z]]
                                   + t->rank[z];
                total += sym[l->sym_base[pi] + l->even_row[pair] * ncols + column];
                const double sign = l->odd_sign[pair];
                if (sign != 0.0)
                    total += sign * anti[l->anti_base[pi] + l->odd_row[pair] * ncols + column];
            }
            tile[position] += total;
        });
    }
}

// Remove the fully symmetric virtual component of every slot (sum of the six
// permutations, in the order of the dense kernel) and zero T[d,d,d].
extern "C" void itercc_bt3_purify(const itercc_blocked_t3 *t, double *r3, int threads)
{
    i64 largest = 0;
    for (i64 s = 0; s < t->nslots; ++s) {
        const i64 end = s + 1 < t->nslots ? t->slot_offset[s + 1] : t->size;
        largest = std::max(largest, end - t->slot_offset[s]);
    }
#pragma omp parallel num_threads(threads)
    {
        std::vector<double> buffer(largest);
#pragma omp for schedule(dynamic)
        for (i64 s = 0; s < t->nslots; ++s) {
            const i64 gs = t->slot_irrep[s];
            double *tile = r3 + t->slot_offset[s];
            i64 used = 0;
            for_each_entry(t, gs, [&](i64 position, i64 a, i64 b, i64 c) {
                const double sum = tile[tpos(t, gs, a, b, c)] + tile[tpos(t, gs, b, a, c)]
                                   + tile[tpos(t, gs, c, b, a)] + tile[tpos(t, gs, a, c, b)]
                                   + tile[tpos(t, gs, c, a, b)] + tile[tpos(t, gs, b, c, a)];
                buffer[position] = tile[position] - sum / 6.0;
                used = position + 1;
            });
            std::memcpy(tile, buffer.data(), used * sizeof(double));
            for (i64 d = 0; d < t->nv; ++d)
                if (t->irrep[d] == gs)
                    tile[tpos(t, gs, d, d, d)] = 0.0;
        }
    }
}
