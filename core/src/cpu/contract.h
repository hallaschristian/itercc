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
struct ContractPlan {
    const double *a, *b;
    double *out;
    const int64_t *a_rows, *a_sum, *b_sum, *b_cols, *rpos, *cpos;
    double alpha;
    bool a_k_major, b_c_major, r_fast;   // a's rows faster than its sums; b's sums faster
                                          // than its columns; out's rows faster than columns
};

ContractPlan contract_plan(const double *a, const double *b, double *out, const int64_t *a_rows,
                           const int64_t *a_sum, const int64_t *b_sum, const int64_t *b_cols,
                           const int64_t *rpos, const int64_t *cpos, double alpha, int64_t flags);

// Every block on the calling thread, each by its size route: the same
// arithmetic as itercc_sym_contract on any number of threads.
void contract_serial(const ContractPlan &p, const int64_t *table, int64_t nblocks);

}  // namespace itercc

#endif
