# Architecture

Design notes, correctness contracts and measurements. Timings refer to the
development system, CuF with scalar X2C (13 correlated occupied spatial
orbitals; 29, 47, 88 or 131 virtuals depending on the basis), on a 12-core
workstation, two-socket Sapphire Rapids nodes (2 x 56 cores) and NVIDIA
A100/H100 GPUs. "13o/47v" means 13 occupied and 47 virtual orbitals.

| Area | Modules | Responsibility |
| --- | --- | --- |
| Hamiltonian boundary | `hamiltonian.py`, `pyscf_interface.py` | Integral conventions and optional exact-integral RHF adapter |
| Ground-state tensor equations | `restricted.py`, `_restricted_equations.py`, `_triples_layout.py` | T1 dressing, spin-adapted residuals, occupied-triangular T3, virtual pair ladder, iteration and DIIS |
| Independent implementations | `solver.py`, `_equations.py`, `tests/oracle` | Spin-orbital tensor path and determinant-space prototype |
| Symmetry and transforms | `spatial_symmetry.py`, `orbital_rotation.py`, `_pair_symmetry.py` | Orbital labels, leakage audits, amplitude rotations and symmetry-aware packing |
| EOM and response | `eom.py`, `_directional.py`, `_eom_triples.py`, `response.py` | Exact directional Jacobian, singlet projection, Davidson and shifted GMRES, CC Lagrangian response |
| Compiled core | `core/`, `_core.py`, `_cpu_kernels.py` | C++/OpenMP CPU kernels and the CUDA backend behind a stable C interface, built with CMake per instruction-set level |
| CPU threading | `_cpu_triples.py` | Ground-state triples residual and T3 feedback on the CPU: plane-wide GEMMs with the compiled kernels or a NumPy thread pool |
| CCSDTQ | `quadruples.py`, `_quadruples_equations.py`, `_quadruples_cuda.py`, `_quadruples_cpu.py`, `_cpu_quadruples.py`, `symmetry_blocks.py`, `_cpu_symmetry_blocks.py` | Restricted CCSDTQ on CUDA (`backend="cuda"`) or the CPU (`backend="cpu"`): packed T4 layout, tile equations, symmetry-blocked products |
| GPU | `_cuda_*.py` | Ground and tangent contractions, tiling, index maps, assembly and cached source intermediates |
| Checkpoints | `checkpoint_io.py`, `RestrictedCCSDTQ.save_state`/`load_state` | Atomic array snapshots and resumable solver states |

## Correctness contracts

- Ground-state CCSDT updates T1, T2 and T3 at each iteration. Energy change and raw residual tolerances govern convergence. The default DIIS history holds T1 and T2. `diis_triples=True` also extrapolates T3 and adds full triples history.
- The restricted solver's public residuals use the same layout as its amplitudes. Spatial T2 has paired occupied/virtual permutation symmetry. Repeated occupied T3 slots require the matching virtual-axis permutations. Conversion to spin orbitals or checkpoint reordering must preserve these conventions.
- Singlet EOM linearizes the residual at a validated ground-state snapshot. The projected space uses Abelian XOR labels when supplied. With labels, the integrals and ground amplitudes are symmetry-projected for the action when their leakage stays within `symmetry_tolerance` (default 2e-9; 0 keeps the original operators, whose leakage then enters every action). Root acceptance uses fresh full-space residuals of the acting operators. The cached primal dressing and feedback are tied to the frozen reference.
- The GPU has separate general and projected paths. A projected contraction may skip output rows that cannot contribute. Ground amplitudes and integrals with small off-sector components can still contribute to allowed output and must retain their input support. Keep the general path for raw checks and off-ground mixed derivatives.
- Checkpoint payloads are NumPy arrays written atomically (`checkpoint_io`). Readers must check orbital frame, dimensions, method, and input/control fingerprints before resuming; `RestrictedCCSDTQ.load_state` checks the layout, the precision and a hash of the Hamiltonian, and a CUDA state carries its autotuned GEMM routes so that a resumed solve rounds as the saved one would have.

## CPU threading

CPU ground-state residuals (`residual_backend="cpu"` with the default pair
ladder) use `_cpu_triples.py`. For each occupied index k, the T2-driven
seeds, the particle-hole products and the occupied ladder of the whole plane
T(x,y,k) are five GEMMs, and the one-body virtual term is part of the virtual
ladder matrix. The T3 feedback uses two GEMMs per plane T(m,i,j).

CCSD, CCSDT and CCSDTQ share one contraction engine on the CPU: the GEMM
executor of the compiled core (`itercc_sym_contract`, the indirect GEMM of
`gemm_kernel.h`; see "CCSDTQ on the CPU"). `RestrictedCC` passes
`_cpu_triples.CompiledContract` as the `contract` of
`_restricted_equations`, whose `add_product`/`plus_product` add each product
of the lower ranks and triples sources straight into its target (no einsum
temporaries, no transposed copies); `matmul`/`product_into` run the plane
GEMMs, the pair-sector ladders and the T3 feedback; the symmetry-blocked path
runs its sector GEMMs and T2 seeds as executor block tables that read the
operands in place. No OpenBLAS call remains on these paths, so every OpenMP
thread works on the products (NumPy's OpenBLAS stops at 64 threads and keeps
a second thread pool spinning between calls), and the results do not depend
on the thread count. NumPy, CuPy and the dual numbers of EOM and response
keep their arithmetic: the helpers evaluate exactly the expressions they
replace when the `contract` has no `into`. On 12 local cores this took CCSD
from 0.132 to 0.066 s per evaluation at 47 virtuals and 0.904 to 0.542 s at
88, and CCSDT from 0.78 to 0.48 s at 29 virtuals; before, 12 threads were no
faster than 4. On a Sapphire Rapids node (13 occupied, seconds per
evaluation, old -> new; 14, 28 and 56 threads on one socket with its memory,
112 spread over both with interleaved memory):

| | 14 | 28 | 56 | 112 |
|---|---|---|---|---|
| CCSD 88v | 0.66 -> 0.29 | 0.60 -> 0.22 | 0.66 -> 0.19 | 0.94 -> 0.19 |
| CCSD 131v | 2.65 -> 1.14 | 2.36 -> 0.83 | 2.27 -> 0.68 | 2.64 -> 0.66 |
| CCSDT 47v | 1.14 -> 1.11 | 0.92 -> 0.79 | 4.44 -> 0.67 | 2.63 -> 0.76 |
| CCSDT 88v | 8.65 -> 8.43 | 6.45 -> 5.67 | 11.06 -> 4.29 | 10.46 -> 4.81 |
| CCSDT 131v | | 25.0 -> 21.0 | 30.8 -> 16.4 | 26.4 -> 17.3 |

The OpenBLAS path got slower past 28 threads; the executor keeps scaling
within a socket. A second socket adds nothing for CCSD and CCSDT (CCSDTQ at
13o/29v: 18.7, 13.9 and 11.3 s at 28, 56 and 112 threads), so independent
solves run one per socket: one socket now beats the whole node before this
change and an A100 (7.9 s at 88v, 26.6 s at 131v).
`_cpu_quadruples.copy` also copied strided views (such as
`RestrictedCC.g`, a transposed view of the ERIs) through
`np.ascontiguousarray` on one thread before its parallel pass, a full n^4
copy per T1 dressing (0.8 GB at 88 virtuals); it now copies in one parallel
pass.

The element-wise work (copies out of packed T3, spin sums, seed assembly,
the scatter into packed R3, the ladder operands, purification, and the T3
DIIS, update step and residual maximum in `solve`) runs in the OpenMP
kernels of the compiled core (see below). `cpu_threads` (the solver's
`cpu_threads=`, which reaches every kernel of an evaluation through
`_cpu_triples.thread_scope`, else `ITERCC_CPU_THREADS`) defaults to one
thread per physical core of the process's affinity: no BLAS runs on these
paths, so NumPy's OpenBLAS count (capped at 64) no longer bounds them.
Hyperthreads do not help, and a team on every logical CPU of a shared
machine stalls at each barrier: in a 24-CPU WSL VM on 12 cores, 24 threads
made a CCSDT Jacobian action 4.6x slower than 12. The
NumPy fallback (no compiler, or `ITERCC_CPU_KERNELS=0`) overlaps its
element-wise work with the GEMMs of the neighbouring plane (default 2 workers).

Every output entry is written by one thread in a fixed order, so results do
not depend on the thread count. Custom triples assemblers and
`pair_ladder=False` keep the generic per-(j,k) loop, which is also the
reference in `tests/test_cpu_triples.py`.

With orbital labels and exactly symmetric amplitudes, ground-state residuals
run on symmetry-blocked T3 (`_blocked_residual.py`); see the compiled core
section. The doubles particle-particle ladder uses the same pair-sector
GEMMs (`doubles_ladder`) for CCSD (`RestrictedCC(..., rank=2)`), CCSDT and
EOM ground checks. On the CPU, EOM and response Jacobian actions
(`_cpu_tangent.py`) use the product rule instead of dual-number loops: the
triples residual is bilinear in (t2, triples sources) plus linear in T3 with
operators from the intermediates, and the T3 feedback is bilinear in T3 and
the dressed (f, g), so each tangent is two calls of the compiled functions.
`ITERCC_CPU_TANGENT=0` restores the dual-number loop (the test reference).

The symmetry-blocked paths need exact zeros: a single rounding-level
off-symmetry entry makes a block check fail and the product run dense. The
prepared CuF integrals carry 5e-11 of one-electron leakage (the ground
amplitudes 5e-11 in T1), which kept the CCSDT ground state on dense T3 and
every EOM/response product dense (the particle-hole, occupied- and
pair-ladder matrices about nirrep times larger than their blocks). So
`RestrictedCC(..., orbital_irreps=..., symmetry_tolerance=2e-9)` projects
integrals whose leakage stays within the tolerance (as `RestrictedCCSDTQ`
does) and the forbidden entries of starting amplitudes (`_symmetric`); the
default 0 keeps the original integrals, which raw-Hamiltonian audits and
refinements need (a driver can project explicitly for its main solve). And
`EOMSinglet`/`DipoleResponse(..., symmetry_tolerance=2e-9)` act with the
projected integrals and ground amplitudes when both stay within it (EOM
already refused integral leakage above 2e-9). The actions change by the
order of the dropped leakage, far below solve tolerances; 0 keeps the
original operators, and `integral_symmetry_leakage`,
`amplitude_symmetry_leakage` and `symmetry_projected` record what happened.
Totally symmetric directions (A1 EOM roots, the totally symmetric dipole
and every property solve of response) then run the triples part of the
action on blocked T3 (`_blocked_residual.triples` with the product-rule
ladder matrices; ground T3 packed once per engine), and the ground-state
parts of the action (feedback increments, ladder base) are cached per
engine. On CuF 13o/29v (12 threads) an A1 action went from 2.24 to 0.88 s and
a CCSDT ground-state residual from 1.0 to 0.5 s.

A direction of irrep G != 0 uses that R3 = S(t2; vooo, vvvo) + L(operators;
T3) is linear in each pairing, so dR3 = R3(t2, dT3; im, dvooo, dvvvo, W) +
R3(dt2, T3; dim, vooo, vvvo, dW). XOR-ing every occupied label with G moves
each T3 slot's irrep by G (three occupied indices) and leaves pair irreps
unchanged (two): in those labels dT3, the derivative sources (three or one
occupied index), t2 and every ground operator are totally symmetric, so the
first pairing runs on the blocked kernels unchanged
(`_cpu_tangent._blocked_layout(..., irrep)`). The second (dT2's seeds, and
the derivative operators on the ground T3) stays on dense-slot T3; its
operators couple sector s to s ^ G only, and `_sector_blocks` turns such a
matrix into rectangular blocks (G = 0 is the old block-diagonal case), so its
particle-hole and pair-ladder GEMMs are blocked too. A non-A1 action went from
1.95 to 1.39 s. The rest of the gap to A1 is that second pairing's dense-slot
plane, seed and scatter passes and its occupied ladder: blocking them would
need kernels that read one blocked layout (the ground T3) and write another
(the shifted residual).

The spectrum code shares the same CPU machinery. `DirectionalBackend.einsum`
(the dual numbers of the EOM and response lower ranks) runs two-operand
primal and tangent products on the GEMM executor (`CompiledContract`). EOM
and response now cache the ground-state T1 dressing on the CPU as on CUDA
(`cache_dressing` defaults on with the compiled core) and form each action's
derivative legs with `itercc_dress_axis_from`: before, every CPU action
re-dressed the n^4 integrals as dual numbers in NumPy. The Davidson and GMRES
subspace products (`krylov.Columns`: overlaps, combinations, basis changes and
the projected-matrix updates) run on the executor for real float64 NumPy
blocks. EOM-CCSDT actions at CuF VDZP (13o/29v, 12 local cores): 2.65 -> 2.1 s
(A1) and 2.9 -> 2.3 s (B1), images equal to 1e-17. The finite-difference
EOM-CCSDTQ (`eom_ccsdtq.FiniteDifferenceEOM`, formerly CUDA only) takes the
CPU `RestrictedCCSDTQ` too (dense T4 layout), so its actions use the compiled
CPU CCSDTQ residual; the four-electron FCI check runs on both backends.

`lower_residuals` started R2 as `0.5 * g[v, v, o, o].transpose(2, 3, 0, 1)`,
which keeps the view's transposed memory order, so every R2 update wrote
through strides; on NumPy R2 now starts C-ordered. The T3 feedback of the
dense-slot path stays on the dense GEMMs (`_feedback_terms`, two executor GEMMs
per plane, about 0.6 TF per evaluation at 88 virtuals). The symmetry-blocked
path (`_blocked_residual.feedback`) writes each spin-summed plane
q = P T3(m, i, j) once and runs the six feedback products as executor block
tables with per-operand irreps, which read q and the integral slices in place:
no relayout copy of q and none of the GEMM work on symmetry zeros (10o/60v C2v:
0.68 -> 0.43 s per evaluation, results equal to rounding). An earlier blocked
version gathered its blocks with NumPy fancy indexing on one thread and was
slower than the dense GEMMs. EOM and response actions of projected operators
use it for both tangent terms, dT3 in the occupied labels shifted by the
direction's irrep and the derivative integrals on the ground T3
(`_cpu_tangent.feedback`, `EOMSinglet._exact_labels`). The
reversed-axis update `vvvo -= y.transpose(3, 2, 1, 0)` runs on all threads
(`subtract_reversed`); a single thread took about 70 ms per plane on it at 88
virtuals. So does the R1 term, a vector times each matrix of the plane
(`subtract_vecmat`, `itercc_stacked_vecmat`, fixed blocks summed in order):
NumPy's matmul of a vector with a stack runs on one thread whatever the BLAS
threads, about 1 s per evaluation at 88 virtuals. The pair matrices of the
ladders (`_ladder_sectors`, `itercc_pair_matrices`) are formed row by row from
one cached nv x nv block of W instead of four-index fancy indexing (0.47 ->
0.15 s per call at 88 virtuals). Restricted CCSDT on a Sapphire Rapids node
(cpu-sym, seconds per evaluation, 13 occupied) before and after these fixes:
47 virtuals 1.81 -> 1.3-1.4 (56 threads), 88 virtuals 11.3 -> 7.4, ~106
virtuals 20.8 -> 12.7 (112 pinned threads); an A100 takes 7.9 and 13 s. At
~131 virtuals the node took 31.0 s before the pair-matrix kernel (43.9 s
before all of them; A100 26.6 s). The T1 dressing of the integrals (`dressed_integrals`,
`itercc_dress_axis`) copies g on all threads and transforms each axis in
place, a task per outer index and piece of the inner range so the source rows
stay in cache; NumPy's copy, `moveaxis` and `tensordot` moved the n^4 tensor
several times on one thread for 9 GF of arithmetic (1.4 s per evaluation at 88
virtuals; 0.77 -> 0.13 s at 13o/75v on 12 cores).

## Compiled core

`core/` is a CMake project (C++17). `libitercc_core.<target>.so` holds the
OpenMP CPU kernels and `libitercc_cuda.so` the CUDA backend (sm_80, sm_90
and PTX), kept separate so CPU-only nodes never load CUDA libraries. Python
reaches both through the plain C interface in `core/include/itercc/*.h` with
ctypes (`_core.py`); each library reports an ABI number that the loader
checks, so a stale build is never used. Bump the ABI with any signature
change.

Targets are named by x86-64 level (`x86-64-v3` for AVX2, `x86-64-v4` for
AVX-512), and the loader picks the best one the running CPU supports. A
`native` build is trusted only from `ITERCC_CORE_DIR` or the checkout's
`core/build`. The libraries are built with `-ffp-contract=off` (and
`--fmad=false` for CUDA), so results do not depend on the instruction set.
`core/build_levels.sh` builds the x86-64-v3 and v4 levels (and the CUDA
library); use the system GCC there so that its libstdc++ and
libgomp match every node at run time. Without a prebuilt library,
`_cpu_kernels.py` compiles the same sources on first use into
`ITERCC_KERNEL_CACHE` (default `~/.cache/itercc`), keyed by sources,
compiler and CPU model.

`_blocked.py` defines symmetry-blocked restricted T3: per packed occupied
slot, only the virtual triples whose irreps multiply to the slot's occupied
irrep, stored as contiguous (Ga, Gb, Gc) blocks over irrep-sorted virtuals
(3.97x smaller than dense slots for CuF in C2v). Pack, unpack and a leakage
audit run in the core on the CPU (`itercc_t3_*`) and the GPU
(`itercc_cuda_t3_*`, on the caller's stream).

The CPU core neither links nor calls a BLAS: every product runs on its own
GEMM executor (see "CPU threading"). The BLAS registry of earlier versions
(NumPy's CBLAS registered at load time for `itercc_gemm`) was removed in
ABI 14 with the other exports nothing called; ABI 15 put the CCSD/CCSDT DIIS
on the CCSDTQ host-DIIS kernels (`dots`, `combine`, `permuted_add`). The
build string (`itercc_core_build`; record it with benchmark results)
names non-default builds (`kc=N`, `profile`).
The CUDA library keeps one cuBLAS handle per device and runs its GEMMs on
the stream it is given (CuPy's current stream).

## CCSDTQ on the CPU

`RestrictedCCSDTQ(..., backend="cpu")` runs the CUDA backend's equations on
NumPy arrays (`mycc.xp`), with the same packed T4 layout, so saved states
and DIIS histories move between backends. `_quadruples_equations._kernels`
picks the kernel module by array type. With the compiled core,
`_cpu_quadruples` provides:

* `CompiledQuadruplesLayout`: the `QuadruplesLayout` primitives (box
  gathers of plain, ring and spin-summed tiles; scatters; row additions;
  expand/compress; tie symmetrization, spin projection and purification;
  the Jacobi step) as the OpenMP kernels of `core/src/cpu/quadruples.cpp`,
  which copy the contiguous d-runs of each (a, b, c);
* `spin_summation`: P(T) over 8^4 output tiles, each term read in its
  source's memory order (an in-place call copies one row per thread);
* `einsum`: two-operand products into `out` on the symmetry-block executor
  with one irrep (gather, GEMM and an alpha-scaled add into `out`);
* `blocked_einsum`: `CpuSymmetryBlocks` for the tile and block engines (one
  irrep without orbital labels). It keeps the planning of `SymmetryBlocks`,
  and `core/src/cpu/symmetry_blocks.cpp` runs each plan's block table:
  skinny blocks (at most 16 rows or columns against a long side) read the
  long operand in place once; shallow sums (at most 16) add each output line
  straight from L1; the rest go to the indirect GEMM of
  `core/src/cpu/gemm_kernel.cpp`, a BLIS-style kernel that packs operands
  straight from their offset tables into cache-resident micro-panels,
  multiplies register tiles with fused multiply-adds (AVX-512 8x16, AVX2
  6x8, portable std::fma) and adds each tile straight into the output (no
  gathered copies, no product buffer). It is as fast as OpenBLAS on large
  dense GEMMs and faster on the contractions' shallow and indirect shapes.
  No BLAS is called, so there is no second thread pool. Packing reads the
  operands sum by sum across all panels when their rows (or columns) are the
  closer ones in memory (panel by panel, a 256-sum chunk would be 256
  interleaved streams), and copies groups of 8 consecutive offsets as runs.
  The C lines of each register tile are prefetched into L2 about one micro
  kernel ahead: C is a tile or a row buffer far larger than the caches, and
  a shallow product's micro kernel is shorter than a memory miss (worth about
  10% on the 2|2 tile partitions at 13o/47v). `ITERCC_GEMM_NC=auto` (an
  experiment, off by default) narrows the column blocks of deep products so
  that their C block stays in L2 across the 256-sum chunks. Each output entry is
  formed by one thread with a fixed order of summation (sums in chunks of
  256, fused multiply-adds in order of k), so blocks are cut into tasks by
  thread count while results stay identical for any thread count and on
  AVX2 and AVX-512 builds alike.
* `_cpu_tile_program.py` and `core/src/cpu/tile_program.cpp`: with one-slot
  tiles, the two R4 tile loops of `compute_r4_tri` run in one compiled call
  each. The loop body (the same Python function the tile loop calls) is
  played once per slot with a recorder, which captures each term's plan and
  the offsets of its operands in named base arrays; later evaluations only
  pass fresh base addresses. A term's plan depends only on the slot's irrep
  pattern and its operand offsets are affine in the slot's occupied indices,
  so recording replays the body once per pattern (at most nirrep^4) plus
  five times for the offset coefficients, not once per slot (7315 slots at
  19 occupied orbitals). Slots run in parallel; each keeps a dense R4 tile
  (and the slot's T4 tile) live. Slots run on teams of threads only when the
  tiles would exceed a memory budget (`team_size`: an eighth of the host
  memory by default). Teams were first sized to keep the tiles in cache.
  Their GEMMs did run faster (40 against 30 GF/s per thread), but at 47
  virtuals a Sapphire Rapids socket took 2.4x the wall time with teams of
  five: barriers, per-slot parallel regions and the serial tile work.
  Every term's arithmetic is the same on any team or thread count.
* Fused tile programs (`TileProgram._fuse`, `itercc_fused_tile_program`):
  run one term at a time, each of the 106 terms made its own read-modify-
  write pass over the slot's tile. That cost about 2 ns per entry per term
  whatever the depth of its sum, so the shallow terms (sums over 2-9
  indices: 11% of the flops) took half the time. At 12 threads each thread
  ran 3.6x slower than alone, because the passes were memory traffic.
  The terms fall into seven output partitions (which tile indices the rows
  carry: ab|cd, ac|bd, ad|bc and a|bcd, b|acd, c|abd, d|abc). Oriented so
  that the rows carry the first tile index, the terms of one partition and
  row irrep share their rows and columns. So each (partition, row irrep)
  is one GEMM whose sum runs through every term in turn (`fused_gemm`:
  segments with their own operands, offset tables permuted to canonical
  rows and columns, and alpha folded into the packing). The tile is then
  updated once per 256 sums of all the terms together. Each product takes
  whichever orientation fills the register tiles better; that choice, the
  column ranges teams split blocks into, and the thread count leave the
  results unchanged. Teams run the blocks of one partition at once and the
  partitions in turn. On the 6o/29v model the tile programs went from 2.7
  to 1.5 s per evaluation at 12 threads (4.7 s per evaluation). The 2|2
  partitions (the vvvv-like depth of 460-510) run at about 32 GF/s per
  core. The 1|3 partitions are bound by packing: each element of their large
  operand serves only the few rows of one irrep. `ITERCC_TILE_FUSED=0`
  runs one pass per term.
* Whole-array bookkeeping (`array_ops.cpp`): symmetrizations such as
  `W += W.transpose(p)` (pairwise in place for self-inverse p), `t3 +
  t3.transpose(p)`, re-layouts, the solve loop's damping and update, the
  dense-T3 and packed-T4 Jacobi divides and `project_triples` run on all
  threads with NumPy's arithmetic. As single-threaded NumPy passes over
  arrays of gigabytes they would dominate a many-core node.

The T4 passes read whole blocks of T4 (all values of one or two occupied
indices). On the CPU they work on packed blocks (`_cpu_packed.py`): each
block row (T4[m, p, q, r], or T4[m, n, p, q]) holds only its allowed virtual
quadruples, in the slot box layout of the row's irrep (`PackedRows`,
`quad_gather_rows`), and ring and spin sums are formed on the packed rows
(`quad_rows_permute`). These kernels address every box affinely (the source
of an output box under a permutation is one box with fixed strides).

* W_oooo pass: W is totally symmetric, so each class of rows of one irrep is
  one dense GEMM Y = W T on the indirect GEMM; `quad_row_add_packed` adds
  the packed rows of Y to their slots.
* Ring and T4-into-lower-rank passes: their contractions split the virtual
  quadruple between operands (one virtual of the block is summed against W),
  so `build_plan` plans at box granularity. With the irreps of every index
  fixed (each operand totally symmetric up to its irrep; the search fixes an
  operand's last free index from the others), every address in a dense
  array or a packed row is affine in the local indices, and each block gets
  explicit offset segments. Blocks that differ only in the irreps of summed
  indices write the same output entries. They are the segments of one fused
  GEMM (`Plan._fuse`, `itercc_fused_contract`, the same engine as the fused
  tile programs). All output-disjoint fused blocks of a contraction run in one
  parallel call, split into column ranges by the thread count. The box-level
  blocks were small (about 10^5 multiply-adds each), and running them as
  successive waves took 3,600 parallel calls per evaluation at 6o, where the
  fused blocks take 550 (`ITERCC_PACKED_FUSED=0` runs the waves). Plans
  depend on the spectators only through each operand's XOR of their irreps
  and are shared between solvers with the same orbital symmetry.
* T4 into lower ranks forms both spin sums of a block straight from the
  packed slots (`quad_gather_permute`: each term reads the row's slot through
  the composed permutation), without first gathering the plain rows. The
  P4_201 sum (2 - (01) - (02) - (03) on the virtual positions) commutes with
  every permutation of positions 1-3, so the P4_201 rows of the pair (q, p)
  are those of (p, q) with the last two virtual positions exchanged: one
  gather per pair p <= q serves both orders, read through plans with swapped
  letters. The Fock term (1/6) f[m, e] S[m, i, j, k, e, a, b, c] is the n = j
  row of the t1 term t1[m, j, e, n] S[...], so one contraction adds both.
  At 13o/47v on one socket the pass went from 27.4 to 23.8 s (gathers 1352
  -> 1040, contractions 4056 -> 3380).
* Kernels whose terms read several boxes of the same row (`quad_gather_permute`,
  `quad_rows_permute`) hand out boxes to all threads in turn, so the threads
  work through one or two rows at a time and the terms' reads hit the shared
  cache; with a row per thread, 56 rows of 10 MB were live at 47 virtuals and
  every term read went to memory.
* `itercc_fused_contract` (the ring and lower-rank contractions, whose outputs
  are row buffers) packs B 256 columns at a time: 1024 columns of 256 sums
  fill L2. Ring contractions 12.1 -> 11.0 s at 13o/47v; the tile programs keep
  1024 (their A operands would be packed more often: 42.1 -> 44.2 s at 256).
* The projection of the packed residual (occupied ties, the 24-term spin
  sum, purification) runs on the packed slots (`quad_project`): no dense
  rows, and a quarter of the dense arithmetic with the same values. It went
  from 0.54 to 0.06 s per evaluation at 6o.

On the 6o/29v model these passes went from 4.4 s to 2.1 s per evaluation
with packed blocks, then to about 1.7 s with the fused plans, the one-pass
spin sums and the packed projection (12 threads).
Without orbital labels or the compiled core, the dense blocked versions in
`_quadruples_equations.py` run (with symmetry zeros skipped in the gathers
when the contractions are blocked).

Large arrays are first written by parallel kernels (amplitudes passed in
or resumed are copied on all threads), so their pages spread over both
sockets' memory. On the 12-core development machine every phase speeds up
about 3x from 1 to 12 threads: the code is bound by memory bandwidth, which a
dual-channel desktop saturates with a few cores. A two-socket node has about
25 times the bandwidth, so cutting traffic is what pays (hence the packed T4
blocks above; the dense passes moved about 3.8 TB per evaluation at 19o/29v,
mostly symmetry zeros).

On a two-socket Sapphire Rapids node (2 × 56 cores) the socket, not the node,
is the unit:

* **One socket.** A socket with local memory reached 235 GB/s in a y += x
  probe; the whole node with unpinned threads reached only 190-220 GB/s. A
  13o/29v CuF evaluation took 15.4 s on one socket.
* **The whole node.** The same evaluation took 16.8 s with interleaved pages,
  12.4 s with 112 pinned threads (OMP_PROC_BIND=spread) and 45 s with 112
  unpinned threads.
* **Unpinned 112 threads.** Even plain streaming kernels (zeroing a buffer,
  adding packed rows) ran 10-40x slower, a pattern consistent with pages moving
  between the sockets; interleaved or bound memory avoids it.
* **Production setup.** Independent solves (finite-field points) therefore
  run one per socket, with `numactl --cpunodebind=N --membind=N`.

Two thread-count traps on such nodes:

* NumPy's OpenBLAS stops at 64 threads, and `cpu_threads()` follows the BLAS
  count by default, so set `ITERCC_CPU_THREADS`.
* With OMP_PROC_BIND set, libgomp binds the calling thread to one core at
  initialization. The CPU count is therefore read once at import
  (`_cpu_triples._PROCESS_CPUS`), but threads started afterwards inherit the
  binding.

The fused tile programs scale to all 112 cores; the T4 passes do so only
with local or interleaved pages.

Two more fixes from the same measurements:

* `itercc_permuted_add` (a copy when `a` is None, and the parallel first
  touch of incoming amplitudes) ran its rows in parallel, and a flat array
  is a single row, so packed T4 was copied, and its pages placed, by one
  thread. It now also cuts rows into pieces. The bandwidth probe of
  `benchmarks.ccsdtq_scaling` first touched its arrays this way, so its
  whole-node figures above are low.
* The host T4 DIIS reuses the arrays of the entry it drops: the trial-T4 copy
  for the next copy, and the error for the next R4, zeroed on all threads.
  Previously it allocated two packed T4 per evaluation (16.6 GB each at 47
  virtuals), and first touched them through page faults.
* The T4 history dominated host memory: with six entries a solve held about
  13 packed T4 (T4, R4, a spare trial and 5-6 entries of a float64 trial and
  error), 492 GB at 47 virtuals in Cs, more than one socket. The compiled CPU
  backend now keeps it compact (`_CompactHostDIIS`, `ITERCC_DIIS_COMPACT=0`
  for the float64 history): each entry is a float32 error (used only in inner
  products, against the float64 newest error; `itercc_dots_f32`) and a
  float32 trial minus the current T4. A step extrapolates in that difference
  form and re-bases the differences on the new T4 in one pass
  (`itercc_diis_rebase`), with the newest entry's exact float64 step. The
  rounding is relative to the differences between iterates, which vanish as
  the solve converges, and a plain Jacobi step stays exact, so the fixed point
  is unchanged; iterates follow the float64 history to about 1e-8 relative
  (`tests/test_quadruples.py`). The float64 step becomes the next R4. A solve
  now holds about 8 packed T4 (T4, R4, six entries, one spare entry).
* T4 into lower ranks gathered the spin-summed rows of whole chunks of m (about
  0.5 GB at 47 virtuals), and eight contractions then streamed them from
  memory. The rows are now cut into blocks of (m, n) of about
  `ITERCC_LOWER_BLOCK_MB` (`_cpu_packed.lower_blocks`; 0 restores the chunks),
  so a block is gathered and read by all its contractions while it is in the
  last-level cache. Plans depend on a block only through the irreps of its
  occupied ranges and are shared between alike blocks. On the desktop (four
  16 MB L3 slices) blocks gain nothing (13o/29v: 15.1 s with chunks, 15.7 s
  with 64 MB blocks, 23.5 s with 4 MB), and on Sapphire Rapids (one 105 MB
  L3 per socket; 13o/47v) they lose too: the pass took 26-31 s
  with chunks against 58-62 s with 64 or 96 MB blocks and about 85 s with 32
  MB, ten times as many gathers and parallel GEMMs, each too small for 56
  threads. The default is therefore 0 (chunks); block-level fusion needs a
  single compiled driver in which each thread owns whole blocks.

A VTune hotspot profile of a 13o/47v evaluation on one socket (driverless
perf sampling; the memory and microarchitecture analyses need the sampling
driver) split the CPU time into packing 28%, idle threads at OpenMP barriers
21%, the micro kernel 19%, the T4 gathers 14%, the C prefetch 5% and clearing
tiles 3%, and counted about 98,000 threads in 15 minutes. libgomp ends the
surplus threads of its pool when a parallel region asks for fewer threads and
creates new ones, with new per-thread scratch, when a later one asks for more;
`itercc_fused_contract` asked for min(threads, tasks), thousands of times per
evaluation. Every region now uses the full team. The C prefetch runs only in
a product's first sum chunk or when its C panel (all rows by the column
block) exceeds 1 MB, the size that stays in L2 between chunks, and a slot's
tile clears only its allowed entries after the first slot
(`itercc_quad_zero_allowed`). T4 into lower ranks runs its pairs p <= q in
rounds whose pairs share no index (`_pair_rounds`, round-robin): such pairs
write disjoint r2 entries, r3 slabs [:, :, j] and W[..., i, j], so a round's
pairs run at once on teams of about 8 threads (`ITERCC_LOWER_TEAMS`) with the
same results for any team count. On 12 desktop cores three teams took the pass
from 15.4 to 13.6 s. The ring pass does the same with sorted triples in rounds
whose triples share at most one index (`_triple_rounds`; a triple adds to the
slots holding it plus one index, so two triples meet only if they share two),
16 rounds of 22-30 triples at 13 occupied. W_oooo pairs cannot: every pair
reaches every slot through its free pair (r, x). On Sapphire Rapids (13o/47v,
one socket, A/B on one node) the full team per
fused contraction, the cheaper prefetch and the allowed-only clearing took an
evaluation from 106.2 to 102.2 s and 107.5 to 105.0 s, and the concurrent
passes from 130.6 to 121.6 s and 135.5 to 129.8 s (that node ran about 25%
slower than the first).

`RestrictedCCSDTQ(..., precision="single")` (CPU only) lets the memory-bound R4 passes read float32 copies of
their largest operands: T3 and its spin sum in both tile programs, and the
spin-summed T4 rows of T4 into lower ranks (`itercc_quad_gather_permute_f32`
forms each box in double and rounds it). The fused GEMMs widen float32
operands to double as they pack them (`base_f32` flags of
`itercc_fused_contract` and `itercc_fused_tile_program`), so every product and
sum is formed in double and T1-T4 stay double. Emulated float32 storage on
small models (3o/6v, 4o/10v) shows what such rounding costs: T4 alone moves
the converged energy by 1e-13 to 6e-13 Eh and puts the residual floor at
2e-11 to 5e-11; T3 and T4 by 3e-12 to 1.5e-11 Eh with floors of 3e-10 to
8e-10, both in the same number of iterations as double. Rounding T1 and T2 as
well costs 2e-10 to 5e-10 Eh, a floor near 1e-8 and more iterations, so they
stay double. Earlier GPU emulation of float32 T4 and R4 storage for the CuF
tensor gave 2.6e-12 Eh and 3e-5 au. Float64 final iterations are therefore not
needed with float32 operands; only residual tolerances below about 1e-9 are
out of reach. Measured at 13o/47v on one socket (same socket
each): 105.9 -> 102.9 s and 111.1 -> 107.6 s per evaluation (about 3%). The
lower-rank contractions, which stream the rows, gained 35% (18.2 -> 11.8 s);
the tile programs 5% (42.0 -> 39.8 s) and the gathers nothing: those are
limited by indirect addressing and short runs, not by bytes, as on the GPU.
`precision="double"` stays the default.

At 47 virtuals on one socket, 13o/47v evaluations took 176 s with teams and
113-116 s slot-parallel (tile programs 43 s, T4 into lower ranks 32 s, rings
17 s, W_oooo 7 s). The solve loop around them (host DIIS, the update) added
about 25 s before the two fixes. What remains is bound by memory traffic:

* T4 into lower ranks writes the spin-summed rows, about 640 GB per
  evaluation, and five contractions read them back.
* The 1|3 tile partitions gather T3-sized operands for every slot.

Knockouts of the fused GEMM driver (profile build: `ITERCC_PROFILE_SYM`,
`itercc_gemm_knockout`; the speed workdir's `tools/knockout_profile.py` and
`kernel_census.py --profile-build --knockout`) split a 13o/47v evaluation on
one socket as follows:

* Tile programs (43 s): the micro kernel takes 10 s and the C updates 7 s;
  the rest is packing. The 1|3 partitions of the T4-driven program (705 of its
  1350 thread-s) are almost all packing: their twelve W_ooooov x T3 terms
  gather four T3 slabs (all m, n for one occupied index) three times per
  slot, about 0.4 GB per slot, at a quarter of the socket bandwidth.
* Ring contractions (12 s): micro kernel 3 s, C updates 1.5 s.
* Lower-rank contractions (13-14 s): no measurable micro-kernel time; they
  stream the spin-summed rows.

The next levers are therefore structural: T3 slabs shared between slots (a
slab-major pass over the W_ooooov x T3 terms, or a compact irrep-blocked T3),
fewer passes over C, and lower-rank contractions run on row pieces that stay
in cache. Larger sum chunks are a build option (`-DITERCC_GEMM_KC=N`; the
speed workdir's `kernel_census.py --kc N`): KC 384 took the tile programs from
42.0 to 40.2 s (the evaluation 105.4 -> 104.0 s), KC 512 gave about the same;
the default stays 256, since the chunk fixes the rounding of every result.

Pair-blocked T3 (the default; `ITERCC_TILE_BLOCKED_T3=0` turns it off)
targets the short runs of those gathers. On a Sapphire Rapids socket at
13o/47v C2v it took the tile programs from 41.3 to 37.2 s and the evaluation
from 88.9 to 85.0 s, for two T3-sized copies of memory. In a product, the address of a T3 entry is an
occupied part plus v1 nv^2 + v2 nv + v3, and a GEMM panel of free virtuals
reads it in runs of nv/nirrep doubles (one irrep of v3). `PairBlocks` stores
the last two virtual axes irrep pair by irrep pair (C order inside a block;
`itercc_permute_columns`, or its float32 variant for `precision="single"`).
The address is then the occupied part plus v1 nv^2 + position[v2 nv + v3].
That is still a free part plus a summed part whenever v2 and v3 are both free
or both summed, so the term's offset tables map through the permutation
(`PairBlocks.remap`, `TileProgram._blocked_sides`) and the fused program
reads the blocked copy as an extra base. T3 is symmetric under permutations
of its (occupied, virtual) pairs, to rounding (`project_triples` sums six
permutations), so in this mode the T3 views of the W_vooo, W_ovovvo and
W_ooooov terms take their pairs with the summed virtual first and the free
ones in output order (`_pair_ordered`). Free blocks then run over
(nv/nirrep)^2 doubles. 66 of the 72 T3 terms read pair-blocked T3 or its
spin sum; only the six W_ovovvo x c_t3 terms of the T4 program, whose spin
sum lacks the pair symmetry, read the dense copy. The copies change only
addresses: the residuals are bit for bit those of the pair-ordered terms on
the dense arrays, and within 1e-12 of the default order.

On the CPU, W_vvoooo and W_vvvvoo are re-laid out with their occupied
indices outermost before the tile loops (same index order), since every
tile fixes those indices. The CPU default `block_size` is 1: a one-slot R4
tile (nv^4 entries) stays in cache through the terms added to it. About 90%
of an evaluation runs in compiled code. At 6o/29v and 12 threads an
evaluation takes 3.7 s (5.9 s before the fusions): 1.5 s in the tile
programs and 1.7 s in the T4 passes. Both are now bound by reading their
operands: per slot, the 1|3 partitions gather about 40 MB of T3-sized
operands for 1.4 MB of tile.

`_quadruples_cpu` (NumPy only) is the reference and the fallback without the
core (`ITERCC_CPU_KERNELS=0`); `tests/test_cpu_quadruples.py` checks the
compiled layout against it, and `tests/test_quadruples.py` runs the PySCF
residual and FCI checks on both backends.

The remaining planned work is the CCSDT residual on blocked T3 on the GPU and
blocked or off-diagonal symmetry for non-totally-symmetric tangents.

## GPU optimization boundaries

CUDA accelerates the rank-six triples work through reusable tiles and FP64 GEMMs.
Projected EOM and response actions keep the triples direction and image in
the packed target-irrep coordinates of `SingletSpace` (`_cuda_packed.py`);
plane gathers, ladder packing, the compact occupied fold, ladder accumulation
and purification read or write those coordinates directly. The compiled seed
leaves the virtual one-body term to the pair ladder, where it is added to W
and dW as 1/2(vv x 1 + 1 x vv) after the six-ordering fold. EOM uses direct tangent assembly and can reuse ground-state intermediates across actions. For large projected planes the code groups occupied rows, bounds packed feedback columns and streams particle-hole workspaces. Nonzero-frequency response may need separate real and imaginary actions. The full eigensolver or response solve also holds Krylov state, so one-action peak memory is only a lower-scope measurement.

Use the determinant and general tensor paths to check new schedules before interpreting timings. Shared components should stay usable without PySCF.
