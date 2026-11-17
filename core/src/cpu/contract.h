// Symmetry-block contractions inside the CPU core (see symmetry_blocks.cpp).
#ifndef ITERCC_CONTRACT_H
#define ITERCC_CONTRACT_H

#include <cstdint>

namespace itercc {

// One contraction of a plan: operands, offset tables and memory orders.
// Block g (row of a 9-column table: rows, cols, depth, r0, c0, k0, a_base,
// b_base, o_base) adds
//   out[o_base + rpos[r0+r] + cpos[c0+c]] += alpha * sum_k a[a_base + a_rows[r0+r] + a_sum[k0+k]]
//                                                       * b[b_base + b_sum[k0+k] + b_cols[c0+c]].
// B given as signed rows of a packed vector instead of a dense array (see
// itercc_signed_gather_rows): the dense element at address row * width + e is
// sign * source[|bases[row]| - 1 + |map[e]| - 1], map = maps[row] (int32,
// one-based, signed), negated when exactly one of bases[row] and map[e] is
// negative, and zero when either is zero. The dense array never exists: the
// executor gathers each piece of B it needs into its own scratch.
struct PackedRows {
    const double *source;
    const int64_t *bases;   // per dense row
    const int64_t *maps;    // per dense row: address of `width` int32 entries
    int64_t width;          // dense entries per row
};

struct ContractPlan {
    const double *a, *b;
    double *out;
    const int64_t *a_rows, *a_sum, *b_sum, *b_cols, *rpos, *cpos;
    double alpha;
    bool a_k_major, b_c_major, r_fast;   // a's rows faster than its sums; b's sums faster
                                          // than its columns; out's rows faster than columns
    const PackedRows *packed_b = nullptr; // B from packed rows (b unused); offsets as for dense b
};

ContractPlan contract_plan(const double *a, const double *b, double *out, const int64_t *a_rows,
                           const int64_t *a_sum, const int64_t *b_sum, const int64_t *b_cols,
                           const int64_t *rpos, const int64_t *cpos, double alpha, int64_t flags);

// Every block on the calling thread, each by its size route: the same
// arithmetic as itercc_sym_contract on any number of threads.
void contract_serial(const ContractPlan &p, const int64_t *table, int64_t nblocks);

}  // namespace itercc

#endif
