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
| GPU | `_cuda_*.py`, `gemm_contract.py` | CuPy/cuBLAS contractions, fused layout kernels, autotuned GEMM routes |
| References | `solver.py` (`TensorCC`), `tests/oracle` | Independent spin-orbital and determinant-space implementations for tests |

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
