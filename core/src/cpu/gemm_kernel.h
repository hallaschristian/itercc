// Indirect GEMM: the contraction kernel of the CPU core.
//
// C(r, c) += alpha * sum_k A(r, k) B(k, c) for operands addressed through
// offset tables, as the symmetry-block plans produce them:
//   A(r, k) = a[ar[r] + ak[k]],  B(k, c) = b[bk[k] + bc[c]],  C(r, c) = c[cr[r] + cc[c]].
// It follows the BLIS structure, with the gathers folded into the packing:
// operands are packed straight from their scattered addresses into
// cache-resident micro-panels, a register-blocked fused multiply-add
// microkernel forms MR x NR tiles, and each tile is added straight into C
// through cr/cc (no gathered copies and no product buffer).
//
// Each C entry receives alpha * (its sum over k) once per chunk of kGemmKc
// sums, each sum formed by fused multiply-adds in order of k from zero. So the
// result does not depend on the instruction set (AVX-512, AVX2 or the
// portable std::fma kernel), on how rows and columns are split between calls,
// or on the thread count.
#ifndef ITERCC_GEMM_KERNEL_H
#define ITERCC_GEMM_KERNEL_H

#include <cstdint>

namespace itercc {

struct IndirectGemm {
    int64_t m, n, k;
    const double *a;
    const int64_t *ar, *ak;
    bool a_rows_fast;    // A's rows are closer in memory than its sums (pack along rows)
    const double *b;
    const int64_t *bk, *bc;
    bool b_k_fast;       // B's sums are closer in memory than its columns
    double *c;
    const int64_t *cr, *cc;
    double alpha;
};

// Sums per chunk (fixes the rounding). -DITERCC_GEMM_KC=N builds a variant
// for experiments (results then differ from the default build in rounding).
#ifndef ITERCC_GEMM_KC
#define ITERCC_GEMM_KC 256
#endif
constexpr int64_t kGemmKc = ITERCC_GEMM_KC;

// Runs on the calling thread; uses scratch slots 2 and 3.
void indirect_gemm(const IndirectGemm &g);

// One product of a fused GEMM: alpha * sum_k A(r, k) B(k, c), with
// A(r, k) = a[ar[r] + ak[k]] and B(k, c) = b[bk[k] + bc[c]].
// An operand flagged f32 holds float values (a or b then points at float data,
// with offsets in floats); they are widened to double as they are packed, so
// the products and sums are those of the double values.
struct GemmSegment {
    const double *a;
    const int64_t *ar, *ak;
    bool a_rows_fast;
    const double *b;
    const int64_t *bk, *bc;
    bool b_k_fast;
    int64_t k;
    double alpha;
    bool a_f32 = false, b_f32 = false;
};

// Fused GEMM: C(r, c) += sum_s sum_k alpha_s A_s(r, k) B_s(k, c) for rows
// r < m and columns j0 <= c < j1 (C(r, c) = c[cr[r] + cc[c]]; column c reads
// bc[c] of every segment). The segments' sums are laid end to end and cut into
// chunks of kGemmKc, so C is read and written once per chunk of all the
// segments together rather than once per segment. alpha_s scales A_s as it is
// packed. Each entry's arithmetic depends only on the segments (and their
// order), not on j0, j1 or the thread: column ranges can go to different
// threads. Runs on the calling thread; uses scratch slots 2 and 3.
struct FusedGemm {
    int64_t m, j0, j1;
    int64_t nseg;
    const GemmSegment *seg;
    double *c;
    const int64_t *cr, *cc;
    int64_t nc = 0;     // columns per packed block of B (0: the default)
};

void fused_gemm(const FusedGemm &g);

// Orientation of fused GEMMs: 0 picks per product (the default), 1 keeps C,
// 2 transposes it. A diagnostic: every choice gives the same results.
void set_fused_orientation(int mode);

// "avx512", "avx2" or "generic".
const char *indirect_gemm_kernel();

}  // namespace itercc

#endif
