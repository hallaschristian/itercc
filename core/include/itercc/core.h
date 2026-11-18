/* IterCC compiled core: the stable C interface used by the Python package.
 *
 * All arrays are C-contiguous: float64 data unless a function or flag says
 * float32, int64 indices except the int32 layout tables. Every output
 * entry is written by one thread in a fixed order, so results do not depend
 * on the thread count, and the library is built with -ffp-contract=off so
 * they do not depend on the target instruction set either.
 *
 * Bump ITERCC_CORE_ABI whenever a signature or a function's meaning changes;
 * the Python loader refuses a library with another ABI number.
 */
#ifndef ITERCC_CORE_H
#define ITERCC_CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ITERCC_CORE_ABI 31

/* Each row gathers through a signed one-based int32 map. bases encode a
 * signed one-based source base, zero means an entirely zero row. maps are
 * intptr_t addresses carried in int64 slots (64-bit host). */
void itercc_signed_gather_rows(int64_t rows, int64_t width, const int64_t *bases,
    const int64_t *maps, const double *source, double *target, int threads);

/* Ordered weighted gathers into distinct packed rows. coordinates: count x
 * rank int32, strides: terms x rank int64 in elements; negative strides OK.
 * Caller guarantees valid coordinates/rows and nonoverlapping source/target. */
void itercc_permuted_gather_add(int64_t count, int rank, int terms,
    const int32_t *coordinates, const int64_t *rows, const int64_t *strides,
    const double *weights, const double *source, double *target, int threads);

/* Expand a sector vector into a zeroed flat tensor. Indices must be distinct
 * and in [0,size), owned and validated by the caller's immutable layout. */
void itercc_sector_unpack(int64_t size, int64_t count, const int64_t *indices,
    const double *vector, double *out, int threads);

/* Expand canonical spin-orbital amplitudes, first touching the dense output
 * in parallel. Coordinates: count x (2*rank); permutations: npermutations x
 * (2*rank). Distinct canonical orbits are a precondition (no overlapping writes). */
void itercc_spin_unpack(int64_t count, int rank, int64_t nv, int64_t no,
    const int32_t *coordinates, const double *vector, int npermutations,
    const int32_t *permutations, const int32_t *signs, double *out, int threads);

/* Binary small-operand / canonical spin-orbital T4 contraction. Coordinates
 * are count x (2*rank) int32. Signed maps encode address+1 (zero is forbidden).
 * Sparse coefficient rows are starts[group]..starts[group+1], with four int32
 * internal indices per entry. amplitude_axes[8] selects output coordinates
 * (>=0) or internal indices (-1-index). Every output is owned by one thread. */
void itercc_spin_quad_contract(int64_t count, int rank, const int32_t *coordinates,
    int64_t no, int64_t nv, const int32_t *virtual_map, const uint8_t *virtual_key,
    const int64_t *hole_map, const uint8_t *hole_key, const double *packed,
    int nexternal, const int32_t *external_axes, const int32_t *external_extents,
    const int32_t *amplitude_axes, const int64_t *starts, const int32_t *internal,
    const double *values, double alpha, double *out, int threads);

/* One term of itercc_spin_quad_contract (its arguments, eight bytes each). */
typedef struct {
    const int32_t *virtual_map;
    const uint8_t *virtual_key;
    const int64_t *hole_map;
    const uint8_t *hole_key;
    const double *packed;
    int64_t nexternal;
    int64_t external_axes[8], external_extents[8], amplitude_axes[8];
    const int64_t *starts;
    const int32_t *internal;
    const double *values;
    double alpha;
} itercc_spin_quad_term;

/* The terms in order in one pass over the rows: the results of one
 * itercc_spin_quad_contract call per term, bit for bit. */
void itercc_spin_quad_contract_terms(int64_t count, int rank, const int32_t *coordinates, int64_t no,
    int64_t nv, int64_t nterms, const itercc_spin_quad_term *terms, double *out, int threads);

/* The direct terms in dense form (spin_quadruples.cpp). A family is one
 * coefficient on one packed T4 (the maps, `packed`, the layout's spin change
 * and irrep): its summed tuples (four spin-orbital indices each, `internal`)
 * sorted by symmetry class, class (delta - class_min) * 8 + gamma with delta
 * the alpha count of the summed particles minus that of the summed holes and
 * gamma the irrep product; class_start (nclasses + 1 entries) cuts them, and
 * class c's coefficient block is weights[weight_offset[c] ..], n_ext rows of
 * external values by its tuples. */
typedef struct {
    const int32_t *virtual_map;
    const uint8_t *virtual_key;
    const int64_t *hole_map;
    const uint8_t *hole_key;
    const double *packed;
    int64_t spin_change, irrep, class_min, nclasses;
    const int64_t *class_start;
    const int32_t *internal;
    const double *weights;
    const int64_t *weight_offset;
    int64_t n_ext;
} itercc_spin_quad_family;

/* One term (permutation) of a family: the external values at row positions
 * external_axes with flat index sum value * stride, the amplitude as in
 * itercc_spin_quad_contract (amplitude_axes: row position, or -1 - the index
 * of a summed letter), and its factor. */
typedef struct {
    int64_t family, nexternal;
    int64_t external_axes[4], external_strides[4];
    int64_t amplitude_axes[8];
    double alpha;
} itercc_spin_quad_dense_term;

/* out[row] += sum_t alpha_t sum_k W_t[ext][k] T4(context(row, t) + tuple k) for
 * rank-four rows (eight int32 coordinates, slots in runs of equal holes), each
 * context gathering its T4 values once; each row adds its terms in order.
 * occupied_irrep and virtual_irrep give each spatial orbital's irrep. */
void itercc_spin_quad_dense(int64_t count, const int32_t *coordinates, int64_t no, int64_t nv,
    const int64_t *occupied_irrep, const int64_t *virtual_irrep, int64_t nfamilies,
    const itercc_spin_quad_family *families, int64_t nterms, const itercc_spin_quad_dense_term *terms,
    double *out, int threads);

/* Integral transformations (integrals.cpp, itercc.integrals.mo_eri). s4 (npair
 * x npair) from the row-packed lower triangle s8; out[b] (n x n, symmetric)
 * from packed lower triangles rows[b * ld + P]; out[b * ld + P] = the lower
 * triangle of blocks[b]; blocks (rs|pq) of the pairs first.. into the
 * chemists' n^4 out at [r][s] and [s][r], symmetry-forbidden entries zero when
 * labels is not NULL. P = r(r+1)/2 + s, r >= s. */
void itercc_unpack_s8_s4(const double *s8, int64_t npair, double *s4, int threads);
void itercc_unpack_pairs(const double *rows, int64_t items, int64_t ld, int64_t n, double *out, int threads);
void itercc_pack_pairs(const double *blocks, int64_t items, int64_t n, double *out, int64_t ld, int threads);
void itercc_scatter_pairs(const double *blocks, int64_t items, int64_t first, int64_t n,
                          const int64_t *labels, double *out, int threads);
/* vj = J[dm], vk = K[dm] (nao x nao, either may be NULL) of a symmetric
 * density from 8-fold packed real AO integrals; independent of threads. */
void itercc_jk_s8(const double *s8, int64_t nao, const double *dm, double *vj, double *vk, int threads);
/* Gaussian AO integrals (gto_integrals.cpp, itercc.integrals): kind 0
 * overlap, 1 kinetic, 2 nuclear attraction, 3 dipole of r - origin (three
 * matrices) over spherical contracted shells (libcint conventions), and the
 * 8-fold packed two-electron integrals with Schwarz screening. */
void itercc_int1e(int kind, int64_t nshell, const int32_t *shell_l, const int32_t *shell_atom,
                  const int32_t *shell_nprim, const int64_t *shell_prim0, const double *exps, const double *coefs,
                  const double *atom_xyz, const double *charges, int64_t natm, const double *origin,
                  const double *c2s, const int64_t *c2s_off, const int64_t *ao_off, int64_t nao, double *out,
                  int threads);
void itercc_int2e_s8(int64_t nshell, const int32_t *shell_l, const int32_t *shell_atom, const int32_t *shell_nprim,
                     const int64_t *shell_prim0, const double *exps, const double *coefs, const double *atom_xyz,
                     const double *c2s, const int64_t *c2s_off, const int64_t *ao_off, int64_t nao, double screen,
                     double *out, int threads);

/* Library identity. */
int itercc_core_abi(void);
const char *itercc_core_build(void);      /* compiler, target and non-default options */

/* Restricted CCSDT triples, one occupied plane at a time (see
 * itercc/_cpu_triples.py for the algebra). */
void itercc_triples_plane(const double *t3, int64_t no, int64_t nv, const int64_t *slot,
                          const int64_t *stride, const int64_t *pos, double *plane,
                          double *crossed, double *direct, int threads);
void itercc_triples_seeds(int64_t no, int64_t nv, const int64_t *pos, int have_t3,
                          const double *a_seed, const double *b_seed, const double *direct,
                          const double *crossed, double *seeds, int threads);
void itercc_triples_scatter(int64_t nv, int64_t ntargets, const int64_t *target,
                            const int64_t *start, const int64_t *contrib, const double *seeds,
                            double *residual, int threads);
void itercc_feedback_plane(const double *t3, int64_t no, int64_t nv, const int64_t *slot,
                           const int64_t *stride, double *q, double *x_plane, int threads);
void itercc_ladder_gather(const double *t3, int64_t n, int64_t nv, int64_t ne,
                          const int64_t *pe, const int64_t *qe, int64_t nodd, const int64_t *ro,
                          const int64_t *so, const int64_t *orient, double *sym, double *anti,
                          int threads);
void itercc_ladder_scatter(int64_t n, int64_t nv, const int64_t *even_row,
                           const int64_t *odd_row, const double *odd_sign, const int64_t *perm,
                           const double *sym, const double *anti, double *residual, int threads);
void itercc_purify(double *r3, int64_t nslots, int64_t nv, int threads);
void itercc_triples_step(double *r3, int64_t nslots, int64_t nv3, const double *occupied,
                         const double *virtual_sums, double scale, int threads);

/* Vector operations for the solver loop. */
double itercc_max_abs(const double *x, int64_t n, int threads);

/* Symmetry-blocked restricted T3 (see core/src/cpu/blocked.cpp and
 * itercc/_blocked.py for the layout). */
void itercc_t3_pack(const double *dense, int64_t nslots, int64_t nv, int64_t groups,
                    const int64_t *slot_irrep, const int64_t *block_offset, const int64_t *count,
                    const int64_t *start, const int64_t *order, double *blocked, int threads);
void itercc_t3_unpack(const double *blocked, int64_t nslots, int64_t nv, int64_t groups,
                      const int64_t *slot_irrep, const int64_t *block_offset,
                      const int64_t *count, const int64_t *start, const int64_t *order,
                      double *dense, int threads);
double itercc_t3_leakage(const double *dense, int64_t nslots, int64_t nv,
                         const int64_t *slot_irrep, const int64_t *virtual_irrep, int threads);

/* Blocked T3 for the residual. The entry (a,b,c) of a slot of irrep G sits at
 *   toff[(G * groups + Ga) * groups + Gb]
 *     + (rank[a] * count[Gb] + rank[b]) * count[Gc] + rank[c]
 * from slot_offset[slot]; virtuals of irrep G are order[start[G] + rank]. */
typedef struct {
    int64_t nv, groups, nslots, size;
    const int64_t *slot_irrep, *slot_offset;   /* nslots */
    const int64_t *irrep, *rank;               /* nv */
    const int64_t *count, *start;              /* groups */
    const int64_t *order;                      /* nv */
    const int64_t *toff;                       /* groups^3 */
} itercc_blocked_t3;

/* Virtual ladder on a chunk of slots. Pairs p <= q (even) and r < s (odd)
 * are listed by pair irrep pi; sector pi holds rows even_start[pi] ..
 * even_start[pi+1]-1. Its operand matrix (rows x ncols[pi], at sym_base[pi]
 * or anti_base[pi]) has columns (slot, orientation o, third index z of irrep
 * pi ^ G_slot), from colstart[pi * nslots + slot - first_slot]. even_row and
 * odd_row give the sector-local row of pair {x,y}; odd_sign is +1 for x < y,
 * -1 for x > y and 0 on the diagonal. */
typedef struct {
    int64_t first_slot, nslots;
    const int64_t *even_p, *even_q, *even_start;
    const int64_t *odd_r, *odd_s, *odd_start;
    const int64_t *even_row, *odd_row;
    const double *odd_sign;
    const int64_t *colstart, *ncols, *sym_base, *anti_base;
} itercc_blocked_ladder;

/* One occupied plane k of the blocked triples residual.
 * Occupied pairs (x,y) are grouped by pair irrep G: pair_order lists them by
 * sector (pair_start, groups + 1), pair_row[x*no+y] is the sector-local row.
 * The plane and seed matrices of sector G (rows: pairs, columns: triples of
 * irrep G ^ G_k in slot order, ntriple[] wide) start at pair_base[G].
 * Particle-hole rows (l,d) are grouped by irrep s = irrep(l) ^ irrep(d);
 * ph_row[l*nv+d] is the sector-local row and the operand matrix of sector s
 * (ph_ncols[s] wide) starts at ph_base[s]. Its columns are (j, b, c) with
 * irrep s ^ G_k: column jcol[kappa*no+j] + pair index of (b,c), kappa the
 * column irrep, where the pairs (b,c) of irrep tau start block Gb at
 * poff[tau*groups+Gb] and run over (rank b, rank c). tile_slot and
 * tile_perm give the packed slot (or -1) and from_sorted permutation of each
 * tile (x,y,k). */
typedef struct {
    int64_t no, k;
    const int64_t *occ_irrep;                          /* no */
    const int64_t *tile_slot, *tile_perm;              /* no*no, no*no*3 */
    const int64_t *pair_order, *pair_start, *pair_row, *pair_base;
    const int64_t *ph_row, *ph_base, *ph_ncols;
    const int64_t *jcol, *poff, *ntriple;
    /* T2-driven seeds as block matrices. A: rows (j,a,b) of irrep G
     * (a_row[(j*nv+a)*nv+b] sector-local), columns (i,c) of irrep G ^ G_k
     * (a_col[i*nv+c] local to the irrep of (i,c)), matrix G at a_base[G],
     * a_ncols[G] wide. B: rows (i,j,a) of irrep G (b_row[(i*no+j)*nv+a]),
     * columns the pairs (b,c) of irrep G ^ G_k, at b_base[G], b_ncols[G] wide. */
    const int64_t *a_row, *a_col, *a_base, *a_ncols;
    const int64_t *b_row, *b_base, *b_ncols;
} itercc_blocked_plane;

/* plane: sector matrices of T(x,y,k); crossed[s][(l,d), (j,b,c)] = T(j,l,k;d,b,c);
 * direct[s][(l,d), (j,b,c)] = 2 T(l,j,k;d,b,c) - T(l,j,k;b,d,c) - T(l,j,k;c,b,d). */
void itercc_bt3_plane(const itercc_blocked_t3 *t, const itercc_blocked_plane *p,
                      const double *t3, double *plane, double *crossed, double *direct,
                      int threads);
/* Seeds of plane k in the plane layout (in place over the occupied-ladder
 * product when have_t3): S = O + A[(j,a,b),(i,c)] + D[(i,a),(j,b,c)]
 * - C[(i,b),(j,a,c)] - 0.5 C[(i,a),(j,b,c)] - B[(i,j,a),(b,c)], with the
 * T2-driven A and B in the block layout of itercc_blocked_plane. */
void itercc_bt3_seeds(const itercc_blocked_t3 *t, const itercc_blocked_plane *p, int have_t3,
                      const double *a_seed, const double *b_seed, const double *direct,
                      const double *crossed, double *seeds, int threads);
/* Add the seeds of plane k to blocked R3: target slot target[n] collects
 * contributions start[n] .. start[n+1]-1, each (pair x*no+y, t0, t1, t2):
 * R[u] += S(x,y)[w] with w[t_m] = u_m. */
void itercc_bt3_scatter(const itercc_blocked_t3 *t, const itercc_blocked_plane *p,
                        int64_t ntargets, const int64_t *target, const int64_t *start,
                        const int64_t *contrib, const double *seeds, double *r3, int threads);
/* Feedback plane m from blocked T3: dense q[i,j,x,y,z] (spin sum of T(m,i,j))
 * and, unless x_plane is null, X[i,j,a,x,y] = q[i,j,x,a,y], as
 * itercc_feedback_plane. */
void itercc_bt3_feedback_plane(const itercc_blocked_t3 *t, int64_t no, const int64_t *tile_slot,
                               const int64_t *tile_perm, const double *t3, double *q,
                               double *x_plane, int threads);

void itercc_bt3_ladder_gather(const itercc_blocked_t3 *t, const itercc_blocked_ladder *l,
                              const double *t3, double *sym, double *anti, int threads);
void itercc_bt3_ladder_scatter(const itercc_blocked_t3 *t, const itercc_blocked_ladder *l,
                               const double *sym, const double *anti, double *r3, int threads);
void itercc_bt3_purify(const itercc_blocked_t3 *t, double *r3, int threads);

/* Free the per-thread scratch buffers of the kernels (kept between calls) on a
 * team of `threads` threads and the calling thread. */
void itercc_scratch_release(int threads);
/* x = beta x (beta 0 writes zeros). */
void itercc_scale(double *x, int64_t n, double beta, int threads);

/* Symmetry-blocked contraction (itercc/symmetry_blocks.py plans). Row g of
 * table (nblocks x 9: rows, cols, depth, r0, c0, k0, a_base, b_base, o_base):
 *   out[o_base + rpos[r0+r] + cpos[c0+c]] +=
 *       alpha * sum_k a[a_base + a_rows[r0+r] + a_sum[k0+k]] * b[b_base + b_sum[k0+k] + b_cols[c0+c]]
 * Blocks must write disjoint entries of out. flags: 1 when a's rows vary
 * faster in memory than its sums, 2 when b's sums vary faster than its
 * columns, 4 when out's rows vary faster than its columns (gather and
 * scatter orders only). Returns 0. */
int itercc_sym_contract(int64_t nblocks, const int64_t *table, const double *a, const double *b,
                        double *out, const int64_t *a_rows, const int64_t *a_sum,
                        const int64_t *b_sum, const int64_t *b_cols, const int64_t *rpos,
                        const int64_t *cpos, double alpha, int flags, int threads);

/* itercc_sym_contract with b given as the signed rows of itercc_signed_gather_rows
 * (rows of `width` entries; b's offsets address that dense row-major layout),
 * read in place: the same results as gathering b first and contracting. */
int itercc_sym_contract_packed(int64_t nblocks, const int64_t *table, const double *a,
                               const double *source, const int64_t *bases, const int64_t *maps,
                               int64_t width, double *out, const int64_t *a_rows,
                               const int64_t *a_sum, const int64_t *b_sum, const int64_t *b_cols,
                               const int64_t *rpos, const int64_t *cpos, double alpha, int flags,
                               int threads);

/* Symmetry-packed restricted T4 (QuadruplesLayout of itercc/_quadruples_cuda.py):
 * virtual irreps virr, local index vloc and irrep sizes vsize (int32), block
 * offsets boff[((g * nirrep + ga) * nirrep + gb) * nirrep + gc] within a slot
 * of irrep g, slot offsets soff and slot irreps sirr. Virtuals are sorted by
 * irrep. Dense rows are nv^4 entries [a][b][c][d]. */
typedef struct {
    int64_t nv, nirrep;
    const int32_t *virr, *vloc, *vsize;
    const int64_t *boff, *soff;
    const int32_t *sirr;
} itercc_quad_layout;

/* Tile rows from packed slots, box by box (rows: tile offset, packed slot
 * offset, template; templates of 6 + 5 terms fields per box). flags: 1 ring
 * terms, 2 allowed entries only (symmetry zeros left untouched). */
void itercc_quad_gather_boxes(const double *packed, double *tile, const int64_t *rows,
                              const int64_t *templates, int64_t nbox, int64_t per_row,
                              int64_t terms, int64_t nv, int flags, const double *weights,
                              int threads);
/* packed = beta packed + alpha tile on the allowed entries of rows with a
 * slot (plan: nrows x 5 int32, slot first; one row per slot). */
void itercc_quad_scatter(const itercc_quad_layout *l, const double *tile, const int32_t *plan,
                         int64_t nrows, double alpha, double beta, double *packed, int threads);
/* Dense rows of slots first.. from their packed segment (starting at slot
 * first), forbidden entries zero; and back (allowed entries only). */
void itercc_quad_expand(const itercc_quad_layout *l, const double *segment, int64_t first,
                        int64_t nslots, double *dense, int threads);
void itercc_quad_compress(const itercc_quad_layout *l, const double *dense, int64_t first,
                          int64_t nslots, double *segment, int threads);
/* itercc_quad_expand writing only the allowed entries (forbidden ones untouched). */
void itercc_quad_place(const itercc_quad_layout *l, const double *segment, int64_t first,
                       int64_t nslots, double *dense, int threads);
/* Zero rows of invalid slots and entries with three equal virtuals (as the
 * CUDA purify). */
void itercc_quad_purify(const uint8_t *invalid, int64_t nslots, int64_t nv, double *dense,
                        int threads);
/* packed[slot](x) += rows[row, x[m0], x[m1], x[m2], x[m3]] for entries
 * (slot, row, m0..m3) (int32, 6 per entry), distinct slots. */
void itercc_quad_row_add(const itercc_quad_layout *l, const double *rows, const int32_t *entries,
                         int64_t nentries, double *packed, int threads);
/* Packed rows of a T4 block: row r is slot row_slot[r] with its virtual axes in
 * the row's occupied order (packed position k holds row axis row_order[4r+k]),
 * stored at rows + row_offset[r] in the packed box layout of the slot's irrep. */
void itercc_quad_gather_rows(const itercc_quad_layout *l, const double *packed, int64_t nrows,
                             const int64_t *row_slot, const int32_t *row_order,
                             const int64_t *row_offset, double *rows, int threads);
/* Packed rows (row r at row_offset[r], box layout of irrep row_irrep[r]):
 * out(x) = sum_t w_t in(x[perm_t]) in order from zero (perms: nterms x 4), or
 * with ring = 1 the three ring terms as t0 + (t1 + t2). No overlap. */
void itercc_quad_rows_permute(const itercc_quad_layout *l, const double *in, double *out,
                              int64_t nrows, const int64_t *row_offset, const int32_t *row_irrep,
                              int64_t nterms, const int64_t *perms, const double *weights, int ring,
                              int threads);
/* itercc_quad_gather_rows then itercc_quad_rows_permute (row irreps those of
 * the slots), in one pass from the packed slots: the same values. */
void itercc_quad_gather_permute(const itercc_quad_layout *l, const double *packed, int64_t nrows,
                                const int64_t *row_slot, const int32_t *row_order, const int64_t *row_offset,
                                int64_t nterms, const int64_t *perms, const double *weights, int ring,
                                double *rows, int threads);
/* Zero the allowed entries of slot `slot`'s dense nv^4 tile. */
void itercc_quad_zero_allowed(const itercc_quad_layout *l, int64_t slot, double *dense, int threads);
/* itercc_quad_gather_permute with the rows written as floats (each value is
 * formed in double, then rounded). */
void itercc_quad_gather_permute_f32(const itercc_quad_layout *l, const double *packed, int64_t nrows,
                                    const int64_t *row_slot, const int32_t *row_order,
                                    const int64_t *row_offset, int64_t nterms, const int64_t *perms,
                                    const double *weights, int ring, float *rows, int threads);
/* itercc_quad_row_add for packed rows (row r at rows + row_offset[r]). */
void itercc_quad_row_add_packed(const itercc_quad_layout *l, const double *rows,
                                const int64_t *row_offset, const int32_t *entries,
                                int64_t nentries, double *packed, int threads);

/* packed /= eia[o0,a] + eia[o1,b] + eia[o2,c] + eia[o3,d] for slots first ..
 * first+nslots (slots: 4 int32 occupied indices per slot), in one pass. */
void itercc_quad_divide(const itercc_quad_layout *l, double *packed, const double *eia,
                        const int32_t *slots, int64_t first, int64_t nslots, int threads);

/* Whole-array operations (array_ops.cpp). out (C order over shape, ndim <= 8)
 * = wa a + wb b, b read through b_strides (elements; a transposed view, say);
 * a may be out, or NULL for out = wb b. b must not overlap out. */
int itercc_permuted_add(double *out, const double *a, const double *b, int64_t ndim,
                        const int64_t *shape, const int64_t *b_strides, double wa, double wb,
                        int threads);
/* x += x.transpose(perm) in place for a permutation that is its own inverse;
 * perm_strides[d] is the C-order stride of axis perm[d]. */
int itercc_symmetrize_pairs(double *x, int64_t ndim, const int64_t *shape,
                            const int64_t *perm_strides, int threads);
/* out[j] = x . ys[j] (j < k) and out[k] = x . x, in fixed blocks summed in
 * order: the same results on any number of threads. */
void itercc_dots(int64_t n, int64_t k, const double *x, const double *const *ys, double *out,
                 int threads);
/* out = sum_j weights[j] vs[j] (k vectors, in order of j; out overlaps none). */
void itercc_combine(int64_t n, int64_t k, const double *weights, const double *const *vs, double *out,
                    int threads);
/* y += alpha x. */
void itercc_axpy(double *y, const double *x, double alpha, int64_t n, int threads);
/* Compact DIIS history (float32 errors and trial differences):
 * itercc_dots with float32 ys; out = (float) x; and one DIIS step in
 * difference form: with x the newest trial, deltas[j] (k older entries) trial
 * minus the input and weights (k + 1, newest last; null for a plain step),
 * D = sum_j w_j deltas[j] + w_k step, x += D - step, deltas[j] -= D,
 * delta_new = step - D and error_new = step (error_new may be null). */
void itercc_dots_f32(int64_t n, int64_t k, const double *x, const float *const *ys, double *out,
                     int threads);
void itercc_to_f32(const double *x, float *out, int64_t n, int threads);
/* dst[r, j] = src[r, index[j]] for `rows` rows of `ncols` entries (C order);
 * the _f32 variant rounds to float. */
void itercc_permute_columns(const double *src, int64_t rows, int64_t ncols, const int64_t *index,
                            double *dst, int threads);
void itercc_permute_columns_f32(const double *src, int64_t rows, int64_t ncols, const int64_t *index,
                                float *dst, int threads);
void itercc_diis_rebase(int64_t n, int64_t k, const double *weights, float *const *deltas,
                        float *delta_new, float *error_new, double *x, const double *step,
                        int threads);
/* One-index transform of a dense n^4 tensor (C order) in place along `axis`:
 * x[.., t0 + t, ..] += sum_s rotation[t, s] x[.., s0 + s, ..] for t < nt and
 * s < ns (rotation nt x ns, row-major), each sum formed in order of s and then
 * added. The ranges [t0, t0 + nt) and [s0, s0 + ns) must not overlap. */
void itercc_dress_axis(double *x, int64_t n, int axis, int64_t t0, int64_t nt, int64_t s0, int64_t ns,
                       const double *rotation, int threads);
/* itercc_dress_axis reading the source rows from `source` (same shape as x):
 * x[.., t0 + t, ..] += sum_s rotation[t, s] source[.., s0 + s, ..] (the
 * derivative legs of the T1 dressing in EOM and response actions). */
void itercc_dress_axis_from(double *x, const double *source, int64_t n, int axis, int64_t t0, int64_t nt,
                            int64_t s0, int64_t ns, const double *rotation, int threads);
/* out[i, z] += alpha sum_k w[k] q[i, k, z] (q: m x kn x n, out: m x n, C order).
 * The sum runs in fixed blocks of k, added in order: the same result on any
 * number of threads. */
void itercc_stacked_vecmat(int64_t m, int64_t kn, int64_t n, const double *w, const double *q, double alpha,
                           double *out, int threads);
/* Pair matrices of w (nv^4, C order): even[i, j] = w[p_i, q_i, p_j, q_j]
 * + w[p_i, q_i, q_j, p_j] * (p_j != q_j) for ne pairs (p, q), and
 * odd[i, j] = w[r_i, s_i, r_j, s_j] - w[r_i, s_i, s_j, r_j] for no pairs (r, s). */
void itercc_pair_matrices(const double *w, int64_t nv, int64_t ne, const int64_t *p, const int64_t *q,
                          int64_t no, const int64_t *r, const int64_t *s, double *even, double *odd,
                          int threads);
/* dst (n^4, C order) = src (read through element strides ss[0..3]) dressed
 * with t1 (no x nv) along all four axes: the occupied rows of axes 2 and 3
 * gain t1 times the virtual rows, the virtual rows of axes 0 and 1 gain
 * -t1^T times the occupied rows. Two passes over the tensor. */
void itercc_dress_integrals(double *dst, const double *src, const int64_t *ss, int64_t n, int64_t no,
                            const double *t1, int threads);
/* What the T1-dressed CCSD residual reads, never reading g(vvvv): the (p, q)
 * planes with an occupied index as itercc_dress_integrals; of a plane with p
 * and q virtual only the occupied-occupied block, without
 * sum_ef t1(i,e) t1(j,f) g(p,q,e,f). Other entries are left as they were. */
void itercc_dress_integrals_ccsd(double *dst, const double *src, const int64_t *ss, int64_t n, int64_t no,
                                 const double *t1, int threads);
/* Every entry of itercc_dress_integrals with an occupied index, the same
 * values; the virtual-virtual block of the planes with p and q virtual is not
 * written (the ladder sectors come from itercc_ladder_sectors_t1). */
void itercc_dress_integrals_open(double *dst, const double *src, const int64_t *ss, int64_t n, int64_t no,
                                 const double *t1, const double *contracted, int threads);
/* contracted (nv, nv, nv, no) = sum_d g(a,b,c,d) t1(j,d) from the sector
 * blocks of g(vvvv) (pair irreps exact); odd_row (nv x nv) is the odd row of
 * a pair x < y. For itercc_dress_integrals_open. */
void itercc_sectors_contract_t1(int64_t nv, int64_t no, const double *t1, int64_t groups, const int64_t *p,
                                const int64_t *q, const int64_t *even_start, const int64_t *even_off,
                                const int64_t *r, const int64_t *s, const int64_t *odd_start,
                                const int64_t *odd_off, const int64_t *odd_row, const double *even,
                                const double *odd, double *out, int threads);
/* out (n^4) = the first-order change of the T1-dressed g (C order) along d
 * (no x nv): -d^T times the occupied rows added to the virtual rows of legs 0
 * and 1, d times the virtual rows added to the occupied rows of legs 2 and 3,
 * each leg transformed once from g, in one pass. out need not be initialized. */
void itercc_dress_derivative(double *out, const double *g, int64_t n, int64_t no, const double *d, int threads);
void itercc_gemm_narrow(const double *a, int64_t lda, const double *b, int64_t ldb, double *out, int64_t ldo,
                        int64_t m, int64_t k, int64_t n, double alpha, double beta, int threads);
void itercc_dress_derivative_ccsd(double *out, const double *g, int64_t n, int64_t no, const double *d,
                                  int threads);
/* dst (n^4, C order) = src without the entries whose Abelian labels XOR to
 * nonzero; stats = (count of nonzero removed entries, largest magnitude, norm).
 * Returns 1 when a removed entry exceeds tolerance or is not finite, else 0. */
int itercc_project_symmetric(double *dst, const double *src, int64_t n, const int64_t *labels,
                             double tolerance, double *stats, int threads);
/* dst[i, j, k, l] = src[ij, kl] for PySCF's 4-fold packed two-electron
 * integrals (n(n+1)/2 x n(n+1)/2, lower-triangular pair order). */
void itercc_unpack_s4(double *dst, const double *src, int64_t n, int threads);
/* The symmetry-sector blocks of the even and odd pair matrices above, packed
 * one sector after another: for each sector g (rows and columns
 * start[g]..start[g+1] of the sector-ordered pairs) a square block at off[g].
 * w is read through element strides ws[0..3] (may be NULL: no integral
 * part); vv (nv x nv, may be NULL) adds the one-body ladder term
 * 0.5 (vv x 1 + 1 x vv) to each pair ordering. accumulate != 0 adds into
 * the blocks. Only the diagonal sector blocks are formed: exact for an
 * operator that is block-diagonal in pair irreps. */
void itercc_ladder_sectors(const double *w, const int64_t *ws, const double *vv, int64_t nv, int64_t groups,
                           const int64_t *p, const int64_t *q, const int64_t *even_start,
                           const int64_t *even_off, const int64_t *r, const int64_t *s,
                           const int64_t *odd_start, const int64_t *odd_off, double *even, double *odd,
                           int accumulate, int threads);
/* The sector blocks (as itercc_ladder_sectors) of the T1-dressed g(vvvv)
 * from those of the bare g(vvvv) (base_*) and the bare g (n^4, element
 * strides gs): g~(abcd) = g(abcd) - sum_k t(k,a) Q(k,b,c,d) - sum_k t(k,b)
 * Q(k,a,d,c), Q(k,b,c,d) = g(k,b,c,d) - 0.5 sum_l t(l,b) g(k,l,c,d). */
void itercc_ladder_sectors_t1(const double *g, const int64_t *gs, int64_t n, int64_t no, const double *t1,
                              int64_t groups, const int64_t *p, const int64_t *q, const int64_t *even_start,
                              const int64_t *even_off, const int64_t *r, const int64_t *s,
                              const int64_t *odd_start, const int64_t *odd_off, const double *base_even,
                              const double *base_odd, double *even, double *odd, int threads);
/* r3[i,j,k,a,b,c] /= (eia[i,a] + eia[j,b]) + eia[k,c] (dense no^3 nv^3). */
void itercc_t3_divide(double *r3, int64_t no, int64_t nv, const double *eia, int threads);

/* One term of an R4 tile program. Operands: base arrays a and b (with flag 8
 * or 16, the slot's dense T4 tile instead) at element offsets
 * c[0] + c[1] i + c[2] j + c[3] k + c[4] l for the slot's occupied indices.
 * The slot's irrep pattern p selects rows plan_start[p] .. + plan_count[p]
 * of `blocks` (9 columns as itercc_sym_contract, o_base relative to the R4
 * tile). Flags 1, 2 and 4 are the memory orders of itercc_sym_contract. */
typedef struct {
    const double *a, *b;
    int64_t a_coef[5], b_coef[5];
    const int64_t *a_rows, *a_sum, *b_sum, *b_cols, *rpos, *cpos;
    const int64_t *plan_start, *plan_count, *blocks;
    double alpha;
    int64_t flags;
} itercc_tile_term;

/* The R4 tile terms slot by slot, slots in parallel: for each slot (a sorted
 * occupied tuple: occupied[4 s ..], irrep pattern pattern[s]) a dense nv^4 R4
 * tile starts at zero, every term adds its blocks (T4 operands read the
 * slot's dense tile, expanded from packed t4, which may be NULL when no term
 * reads it), and the tile's allowed entries go to the packed slot as
 * r4 = beta r4 + tile. With team > 1, threads / team slots run at a time,
 * each on a team of `team` threads (fewer live tiles; the same results). */
void itercc_tile_program(const itercc_quad_layout *l, int64_t nslots, const int64_t *slots,
                         const int32_t *occupied, const int64_t *pattern, int64_t nterms,
                         const itercc_tile_term *terms, const double *t4, double *r4, double beta,
                         int threads, int team);

/* One product of a fused R4 tile block: alpha X(r, k) Y(k, c) over k < depth,
 * X(r, k) = x[x_rows[r] + x_sum[k]], Y(k, c) = y[y_sum[k] + y_cols[c]], where
 * x is bases[x_base] (-1: the slot's dense T4 tile) at the element offset
 * x_coef[0] + x_coef[1] i + .. + x_coef[4] l for the slot's occupied indices,
 * and y likewise. Flag 1: x's rows are closer in memory than its sums; flag
 * 2: y's sums are closer than its columns. */
typedef struct {
    int64_t x_base, y_base;
    int64_t x_coef[5], y_coef[5];
    const int64_t *x_rows, *x_sum, *y_sum, *y_cols;
    int64_t depth;
    double alpha;
    int64_t flags;
} itercc_fused_segment;

/* One fused block: out[o_base + rpos[r] + cpos[c]] += the sum of segments
 * seg_start .. seg_start + seg_count - 1, for r < rows and c < cols, as one
 * GEMM over the segments' sums laid end to end. In a tile program, a
 * pattern's blocks are sorted by `tag` and blocks of one tag must write
 * disjoint entries (teams run them at once, tag after tag). */
typedef struct {
    int64_t rows, cols, o_base;
    const int64_t *rpos, *cpos;
    int64_t seg_start, seg_count, tag;
} itercc_fused_block;

/* Fused blocks that write disjoint entries of `out`, in parallel (each split
 * into column ranges by the thread count; the results do not depend on it).
 * Segment operands are bases[x_base] + x_coef[0] (the other coefficients are
 * unused) and every segment's alpha is multiplied by `alpha`. A base with
 * base_f32[base] != 0 (base_f32 may be NULL) holds float values, at offsets
 * in floats; they are widened to double as the GEMMs pack them. */
void itercc_fused_contract(int64_t nblocks, const itercc_fused_block *blocks,
                           const itercc_fused_segment *segments, const double *const *bases,
                           const uint8_t *base_f32, double *out, double alpha, int threads);

/* Orientation of the fused GEMMs: 0 picks the better register tiling per
 * product (default), 1 keeps C, 2 transposes it (diagnostic: the results are
 * the same). */
void itercc_fused_orientation(int mode);
/* itercc_tile_program with the terms fused: slot s runs blocks
 * block_start[p] .. + block_count[p] of its irrep pattern p. */
void itercc_fused_tile_program(const itercc_quad_layout *l, int64_t nslots, const int64_t *slots,
                               const int32_t *occupied, const int64_t *pattern,
                               const int64_t *block_start, const int64_t *block_count,
                               const itercc_fused_block *blocks, const itercc_fused_segment *segments,
                               const double *const *bases, const uint8_t *base_f32, const double *t4,
                               double *r4, double beta, int threads, int team);

/* Projection of packed residual slots first .. first + count - 1 in place:
 * occupied ties (bit p of ties[s - first] swaps virtual axes p, p + 1 as
 * itercc_quad_tie), the spin summation alpha (sum_t w_t x[perm_t]) / divisor
 * (as itercc_quad_spin) and purification (invalid[s - first] zeroes the slot;
 * entries with coinciding virtuals as itercc_quad_purify). */
void itercc_quad_project(const itercc_quad_layout *l, double *packed, int64_t first, int64_t count,
                         const uint8_t *ties, const uint8_t *invalid, int64_t nterms,
                         const int64_t *perms, const double *weights, double alpha, double divisor,
                         int threads);
/* dense[s] = (dense[s] + dense[s] with virtual axes pair, pair + 1 swapped) / 2
 * for the rows s in slots (pair 0, 1 or 2). */
void itercc_quad_tie(double *dense, const int64_t *slots, int64_t count, int64_t nv, int64_t pair,
                     int threads);
/* target = alpha (sum_t w_t source[perm_t]) / divisor + beta target over rows of
 * nv^rank entries (perms: nterms x rank, terms added in order). Source and
 * target must be the same array or not overlap. */
void itercc_quad_spin(const double *source, double *target, int64_t nslots, int64_t nv,
                      int64_t rank, int64_t nterms, const int64_t *perms, const double *weights,
                      double alpha, double divisor, double beta, int threads);

/* Rank-four slot programs of the triplet EOM-CCSDTQ action, recorded by
 * itercc/_spin_quadruples_program.py. Every field is eight bytes wide.
 *
 * An operand is a dense base (bases[index] at the element offset coef[0] +
 * coef[1] f0 + .. + coef[4] f3 for the tile's occupied tuple f), packed T4
 * rows (descriptor `index`, only as the second operand), an intermediate of
 * the tile (number `index`) or the tile (offset coef[0]). */
typedef struct {
    int64_t kind;        /* 0 dense, 1 packed T4 rows, 2 intermediate, 3 tile */
    int64_t index;
    int64_t coef[5];
} itercc_slot_operand;

/* op 0 contracts out += alpha a b by the kind's contraction `plan` (an
 * intermediate output is a fresh zeroed buffer); op 1 looks intermediate
 * `inter` up (known: continue at instruction `next`); op 2 publishes it. */
typedef struct {
    int64_t op, inter, next, plan;
    itercc_slot_operand a, b, out;
    const int64_t *a_rows, *a_sum, *b_sum, *b_cols, *rpos, *cpos;
    int64_t flags;
    double alpha;
} itercc_slot_instruction;

/* An intermediate: expression `key`, `size` elements, and the positions in the
 * tile's occupied tuple of the nvars indices its value depends on. */
typedef struct {
    int64_t key, size, nvars, positions[4];
} itercc_slot_intermediate;

/* Packed T4 rows (ground T4 for which = 0, the direction for 1): rows over the
 * four hole letters in C order, letter k at occupied index fixed[k] of the
 * tile (fixed[k] >= 0) or over start[k] .. + count[k], with hole spin
 * spins[k]; `pointers` holds the signed map address of every sector. */
typedef struct {
    int64_t which, spins[4], fixed[4], start[4], count[4];
    const int64_t *pointers;
} itercc_slot_rows;

/* A slot class's tile: its kind, the positions in the slot's occupied tuple
 * of the tile's occupied indices, and its gathers parts[part_start ..
 * + part_count]. A gather adds, for each row r of the slot's spin group
 * `group` (rows and four virtual coordinates c), sum_t weights[t]
 * tile[c . offsets[t]] over the terms of `terms` to the slot's row rows[r]. */
typedef struct {
    int64_t kind, positions[4], part_start, part_count;
} itercc_slot_template;

typedef struct {
    int64_t group, terms;
} itercc_slot_part;

typedef struct {
    const int64_t *offsets;   /* four tile strides per term */
    const double *weights;
    int64_t count;
} itercc_slot_terms;

typedef struct {
    const int64_t *rows;          /* rows within the slot */
    const int32_t *coordinates;   /* four virtual indices per row */
    int64_t count;
} itercc_slot_group;

/* A slot: class, particle key (its spin groups start at key_group_start[key]),
 * rows start .. + length of the result and occupied tuple. */
typedef struct {
    int64_t cls, key, start, length, occupied[4];
} itercc_slot_info;

typedef struct {
    int64_t nv, holes, nirrep;   /* virtual orbitals; spin-orbital occupied orbitals; irreps */
    const int64_t *kind_start, *kind_count, *inter_start, *inter_count;
    const itercc_slot_instruction *instructions;
    const itercc_slot_intermediate *intermediates;
    const itercc_slot_rows *rows;
    /* plan set p: block tables plan_table[plan_start[p] + c] (addresses) with
     * plan_count[..] blocks for contraction c of its kind; plan_id[kind *
     * nirrep^4 + code] is the plan set of a kind for the irrep pattern code
     * ((g0 nirrep + g1) nirrep + g2) nirrep + g3 of a tile's occupied tuple. */
    const int64_t *plan_start, *plan_table, *plan_count, *plan_id;
    const int64_t *ground_bases;
    const int32_t *ground_keys;
    const int64_t *direction_bases;
    const int32_t *direction_keys;
    const int64_t *occupied_irreps;
    const int64_t *class_start, *class_count;   /* tiles of a class: templates[class_start ..] */
    const itercc_slot_template *templates;
    const itercc_slot_part *parts;
    const itercc_slot_terms *terms;
    const int64_t *key_group_start;
    const itercc_slot_group *groups;
    const itercc_slot_info *slots;
    /* Fusion groups of kind k: fuse_groups[fuse_start[k] .. + fuse_count[k]],
     * each (first, count) of fuse_members (instruction numbers within the
     * kind): products into the tile with equal output offset tables. */
    const int64_t *fuse_start, *fuse_count, *fuse_groups, *fuse_members;
} itercc_slot_program;

/* Slots 0 .. nslots - 1 of `result` (zero on entry), in waves whose partial
 * sums stay within partial_limit doubles (or one slot): every tile on one
 * thread (a wave's tiles in order of their occupied tuples, positions 2, 3,
 * 0, 1), then each slot's rows get its tiles' partial sums in order, divided
 * by (4!)^2. Intermediates are shared through a cache of cache_bytes, the
 * pieces of those of at least share_work multiply-adds (0: 2^22) also with
 * the tiles waiting for them (the same values whoever runs them); stats (five
 * entries) gets the cache's hits, misses and peak bytes, the intermediates
 * shared in pieces and the pieces run by waiting tiles. With `fused`, the products of a
 * fusion group run after the tile's other instructions as fused GEMMs, one
 * per block of output rows and columns (other sums, so other rounding);
 * without, every product runs in place, as the Python loop does. The results
 * do not depend on the thread count. Returns 0, or 1 (nothing done) for a
 * tile without a plan. */
int itercc_triplet_slots(const itercc_slot_program *p, int64_t nslots, int64_t partial_limit,
                         const double *const *bases, const double *ground, const double *direction,
                         double *result, int64_t cache_bytes, int64_t *stats, int threads, int fused,
                         int64_t share_work);
/* Diagnostic thread seconds of itercc_triplet_slots by phase (with
 * ITERCC_SLOT_PROFILE=1): zeroing, products into tiles, into intermediates,
 * packed row setup, gathers, cache waits, slot sums, intermediate allocation,
 * the barrier after a wave's tiles (nine entries). */
void itercc_triplet_profile(double *seconds, int reset);
/* ... and by contraction (the contraction number of each instruction). */
void itercc_triplet_contraction_profile(int64_t count, double *seconds, int64_t *calls, int reset);

#ifdef __cplusplus
}
#endif

#endif
