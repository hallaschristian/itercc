// Symmetry-blocked restricted T3: pack, unpack and leakage audit.
//
// Dense-slot layout: T[s, a, b, c] for packed occupied slots s and virtual
// orbitals in their original order. Blocked layout: for slot s with occupied
// irrep G_s and every (Ga, Gb), the block of virtuals with irreps
// (Ga, Gb, Gc = G_s ^ Ga ^ Gb), row-major over their ranks within each irrep,
// at block_offset[(s * groups + Ga) * groups + Gb]. Virtuals of irrep G are
// order[start[G] .. start[G] + count[G] - 1] (original indices).
#include <cmath>
#include <cstring>

#include "itercc/core.h"

using i64 = int64_t;

extern "C" void itercc_t3_pack(const double *dense, i64 nslots, i64 nv, i64 groups,
                               const i64 *slot_irrep, const i64 *block_offset, const i64 *count,
                               const i64 *start, const i64 *order, double *blocked, int threads)
{
    const i64 nv2 = nv * nv, nv3 = nv2 * nv;
#pragma omp parallel for collapse(2) schedule(dynamic) num_threads(threads)
    for (i64 s = 0; s < nslots; ++s) {
        for (i64 ga = 0; ga < groups; ++ga) {
            const i64 gs = slot_irrep[s];
            const double *slot = dense + s * nv3;
            for (i64 gb = 0; gb < groups; ++gb) {
                const i64 gc = gs ^ ga ^ gb;
                const i64 na = count[ga], nb = count[gb], nc = count[gc];
                const i64 *va = order + start[ga], *vb = order + start[gb], *vc = order + start[gc];
                double *out = blocked + block_offset[(s * groups + ga) * groups + gb];
                for (i64 ra = 0; ra < na; ++ra)
                    for (i64 rb = 0; rb < nb; ++rb) {
                        const double *row = slot + va[ra] * nv2 + vb[rb] * nv;
                        double *dst = out + (ra * nb + rb) * nc;
                        for (i64 rc = 0; rc < nc; ++rc)
                            dst[rc] = row[vc[rc]];
                    }
            }
        }
    }
}

// Inverse of itercc_t3_pack: every dense entry is written, forbidden ones as zero.
extern "C" void itercc_t3_unpack(const double *blocked, i64 nslots, i64 nv, i64 groups,
                                 const i64 *slot_irrep, const i64 *block_offset,
                                 const i64 *count, const i64 *start, const i64 *order,
                                 double *dense, int threads)
{
    const i64 nv2 = nv * nv, nv3 = nv2 * nv;
#pragma omp parallel for schedule(dynamic) num_threads(threads)
    for (i64 s = 0; s < nslots; ++s) {
        double *slot = dense + s * nv3;
        std::memset(slot, 0, nv3 * sizeof(double));
        const i64 gs = slot_irrep[s];
        for (i64 ga = 0; ga < groups; ++ga)
            for (i64 gb = 0; gb < groups; ++gb) {
                const i64 gc = gs ^ ga ^ gb;
                const i64 na = count[ga], nb = count[gb], nc = count[gc];
                const i64 *va = order + start[ga], *vb = order + start[gb], *vc = order + start[gc];
                const double *in = blocked + block_offset[(s * groups + ga) * groups + gb];
                for (i64 ra = 0; ra < na; ++ra)
                    for (i64 rb = 0; rb < nb; ++rb) {
                        double *row = slot + va[ra] * nv2 + vb[rb] * nv;
                        const double *src = in + (ra * nb + rb) * nc;
                        for (i64 rc = 0; rc < nc; ++rc)
                            row[vc[rc]] = src[rc];
                    }
            }
    }
}

// Largest |T[s,a,b,c]| over symmetry-forbidden entries (irreps of a, b, c not
// multiplying to the slot irrep); NaN if any forbidden entry is NaN.
extern "C" double itercc_t3_leakage(const double *dense, i64 nslots, i64 nv,
                                    const i64 *slot_irrep, const i64 *virtual_irrep, int threads)
{
    const i64 nv2 = nv * nv, nv3 = nv2 * nv;
    double worst = 0.0;
    int nan = 0;
    // Branch-free over c: forbidden entries contribute |T|, allowed ones 0.
#pragma omp parallel for collapse(2) reduction(max : worst) reduction(| : nan) \
    schedule(static) num_threads(threads)
    for (i64 s = 0; s < nslots; ++s) {
        for (i64 a = 0; a < nv; ++a) {
            const double *slot = dense + s * nv3 + a * nv2;
            const i64 ga = virtual_irrep[a] ^ slot_irrep[s];
            for (i64 b = 0; b < nv; ++b) {
                const i64 gab = ga ^ virtual_irrep[b];
                const double *row = slot + b * nv;
                double row_worst = 0.0;
                int row_nan = 0;
                for (i64 c = 0; c < nv; ++c) {
                    const double v = (virtual_irrep[c] != gab) ? std::fabs(row[c]) : 0.0;
                    row_nan |= (v != v);
                    row_worst = v > row_worst ? v : row_worst;
                }
                nan |= row_nan;
                worst = row_worst > worst ? row_worst : worst;
            }
        }
    }
    return nan ? NAN : worst;
}
