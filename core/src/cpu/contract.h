// Symmetry-block contractions inside the CPU core (see symmetry_blocks.cpp).
#ifndef ITERCC_CONTRACT_H
#define ITERCC_CONTRACT_H

#include <cstdint>
#include <vector>

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

// One piece of a contraction: a block with a deep sum (split into slices,
// kind 0), a chunk of a skinny block's long side (kind 1), or a piece of a
// block's rows and columns by its route (kind 2). The pieces of a plan write
// disjoint entries, and each entry gets the arithmetic of contract_serial.
struct ContractTask {
    int64_t kind, route, first;   // first: slices (kind 0) or first long entry (kind 1)
    int64_t block[9];             // rows, cols, depth, r0, c0, k0, a_base, b_base, o_base
};

// The pieces of a plan as itercc_sym_contract cuts it for `workers` threads
// (pieces of about work / (4 workers) multiply-adds), appended to `tasks`.
void contract_tasks(const int64_t *table, int64_t nblocks, int64_t workers, std::vector<ContractTask> &tasks);

// One piece on the calling thread (split slices also on the calling thread).
void contract_task(const ContractPlan &p, const ContractTask &t);

}  // namespace itercc

#endif
