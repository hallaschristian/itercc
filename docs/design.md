# Design

How IterCC gets its speed and how it keeps results exact. For usage see the
[guide](guide.md); for measurements, [benchmarks](benchmarks.md).

## Layers

| Layer | Modules | What it does |
| --- | --- | --- |
| Inputs | `pyscf_interface.py`, `hamiltonian.py`, `spatial_symmetry.py` | MO integrals, Abelian labels, symmetry audits and projection |
| CCSD/CCSDT | `restricted.py`, `_restricted_equations.py`, `_triples_layout.py`, `_blocked*.py`, `_cpu_triples.py` | Spin-adapted closed-shell residuals on packed, symmetry-blocked T3 |
| CCSDTQ | `quadruples.py`, `_quadruples_*.py`, `_cpu_quadruples.py`, `_cpu_tile_program.py`, `symmetry_blocks.py` | Packed T4, tile equations, compiled tile programs, host/device DIIS, checkpoints |
| EOM and response | `eom.py`, `response.py`, `_directional.py`, `_cpu_tangent.py`, `krylov.py` | Exact Jacobian actions, Davidson, shifted GMRES, CC Lagrangian response |
| Compiled core | `core/` (C++17/OpenMP, CUDA), `_core.py`, `_cpu_kernels.py` | GEMM executor, element-wise kernels, CUDA kernels, behind a versioned C ABI |
| Runtime | `_runtime.py` | Threads, host memory and the registry of ITERCC_* switches, shared by every method |
| GPU | `_cuda_*.py`, `gemm_contract.py` | CuPy/cuBLAS contractions, fused layout kernels, autotuned GEMM routes |
| References | `solver.py` (`TensorCC`), `tests/oracle` | Independent spin-orbital and determinant-space implementations for tests |

## Canonical machinery

Each concern has one implementation, and every method uses it:

| Concern | The one way | Where |
| --- | --- | --- |
| Threads | Every solver and engine takes `cpu_threads` (engines inherit their ground-state solver's) and runs its public calls in `thread_scope(cpu_threads)`; kernels below call `cpu_threads(None)` | `_runtime.threaded`, `cpu_threads`, `thread_scope` |
| Switches | Each ITERCC_* variable is registered once with its default and when it is read; the guide's table is generated from the registry | `_runtime.SWITCHES`, `switch`, `enabled`, `integer` |
| Host memory | Physical, cgroup, Slurm and NUMA limits; budgets in bytes, options in `_gib` | `_runtime.host_memory_limit`, `resident_bytes` |
| Compiled core | Loaded once and checked against the ABI; None without it, and every caller has the NumPy path | `_cpu_kernels.library()`, `_core.cuda()` |
| CPU products | The core's GEMM executor (`itercc_sym_contract`): symmetry-blocked with orbital labels (`CpuSymmetryBlocks`), else the one dense engine (`dense_products`, behind `product_into`, `CompiledContract` and `bounded_einsum`) | `_cpu_symmetry_blocks.py`, `_cpu_quadruples.py`, `_cpu_triples.py` |
| CUDA products | Symmetry blocks (fused, direct or grouped-GEMM plans) and autotuned GEMM routes | `symmetry_blocks.py`, `gemm_contract.py` |
| Long vectors | Scale, add, dots and combinations on all threads (never NumPy BLAS between kernels) | `_cpu_quadruples` array operations |
| T1 dressing | One pass over g for the dressing and for its derivative, in every method | `itercc_dress_integrals*`, `itercc_dress_derivative` |
| Amplitudes | Packed, symmetry-blocked storage per rank | `TriplesLayout`/`BlockedTriples`, `QuadruplesLayout`, `PackedSpinQuadruples` |
| Jacobians | Product-rule tangents of the residual code; the ground-state parts are formed once per engine | `Directional`, `_cpu_tangent.py` |
| Repeated products | Recorded once per symmetry pattern, replayed by C++ for every occupied slot | `TileProgram` (CCSDTQ), `SlotProgram` (triplet) |
| Iterative solvers | Davidson with thick restarts and certification; shifted GMRES with recycled solutions | `krylov.davidson` (all EOM engines, through `EOMSinglet.solve`), `krylov.gmres` |

Deliberate exceptions, measured on water cc-pVDZ (12 threads):

- **CCSDTQ's T1-T3 equations** run on full T3 with the PySCF-derived
  intermediates that its T4 equations share, not on the blocked CCSDT
  engine: 2% of an evaluation (0.045 of 2.1 s for four), while the T4
  passes take the rest on the tile programs.
- **Response right-hand sides** (`DipoleResponse._operator_residual`) loop
  over occupied pairs in Python: 0.07 of 4.2 s for a CCSDT polarizability.
- **Triplet ranks 1-3** run the Python-driven tile loop (`contract_tile` on
  the symmetry engine); only rank four, nearly all of a large action, is the
  compiled slot program: 0.63 of 2.98 s per action here, a few percent on
  CuF (13 occupied, 47 virtual).
- **Finite-difference EOM-CCSDTQ** (`eom_ccsdtq.py`) differentiates the
  CCSDTQ residual numerically: one residual per action for a nonzero target
  irrep, two for the totally symmetric one, each with the symmetry blocks of
  the subgroup that keeps the target sector (`sector_labels`), not the
  analytic tangent of every quadruples term.
- **CUDA CCSDT** keeps its dense-slot T3 layout (`CudaTriplesLayout`), and
  its fused kernels stay options because their support masks are
  ground-state specific; `restricted.cuda_options()` is the one production
  set (every fused path) for ground-state solves.

## Where the speed comes from

1. **Spin adaptation and packing.** Amplitudes are spatial-orbital, closed-shell
   quantities: T3 is stored for ordered occupied triples i <= j <= k (about 1/6
   of the dense array) and T4 for i <= j <= k <= l (about 1/24). Each restricted
   residual is a fraction of the spin-orbital work.
2. **Abelian symmetry everywhere.** With orbital labels every amplitude holds
   only its symmetry-allowed blocks (T3 about nirrep times smaller), and every
   product of the ground state, the EOM Jacobian and the response runs block by
   block. Integrals are projected onto exact symmetry so that rounding-level
   leakage never forces a dense fallback.
3. **Its own GEMM executor.** The CPU core multiplies operands in place through
   offset tables: a BLIS-style kernel packs straight from the strided or
   gathered operands into cache-resident panels (AVX-512, AVX2 or portable FMA
   micro-kernels) and adds each register tile directly into the output. No
   transposed copies, no einsum temporaries, no BLAS thread pool: it keeps
   scaling across a 56-core socket where NumPy's BLAS path stopped at 28 threads.
4. **Compiled tile programs for CCSDTQ.** The roughly hundred T4 residual terms
   are recorded once per symmetry pattern (operand offsets are affine in the
   occupied indices) and replayed by C++ for every occupied slot, fused into
   few GEMMs that read pair-blocked T3 and write each R4 tile once. Independent
   T4 passes run concurrently on thread teams.
5. **Exact Jacobians without derivatives by hand.** EOM and response actions
   are product-rule tangents of the same compiled residual code, so a Jacobian
   action costs about one or two ground-state residual evaluations, and the
   ground-state intermediates are cached per engine.
6. **Recycling in the iterative solvers.** Davidson restarts keep Ritz vectors
   with their images (no repeated actions); response keeps every converged
   solution with its image and starts each new frequency from the least-squares
   combination of earlier ones.
7. **Memory-bound passes are streamed.** Element-wise passes over T4 run in one
   parallel sweep each; the T4 DIIS history can live in host memory as compact
   float32 differences with the same fixed point.

## Exactness

- Every reduction uses a fixed blocking and every output element is written by
  one thread in a fixed order, and all builds use `-ffp-contract=off`: results
  are bitwise identical for any thread count and any instruction-set level.
- Convergence is judged on raw residuals of every rank, not only on the energy.
  EOM roots are accepted only after a fresh full-space residual evaluation.
- CCSDTQ checkpoints carry the amplitudes, the DIIS history, a hash of the
  Hamiltonian and (on CUDA) the autotuned GEMM routes, so a resumed solve
  reproduces the uninterrupted one bit for bit.
- The compiled core reports an ABI number that the loader checks, and a build
  is skipped on a CPU lacking an instruction-set extension it was compiled for.

## Tests as the specification

The pure-NumPy paths are the reference: every compiled kernel and tile program
is tested against them (bit for bit where the arithmetic is the same), the
restricted equations against an independent spin-orbital implementation and
PySCF, and EOM and response against determinant-space references built from
the full transformed Hamiltonian (`tests/oracle`). Detailed engineering notes,
including the optimizations that were tried and rejected, are in the git
history of `docs/architecture.md` (release 0.7.0).
