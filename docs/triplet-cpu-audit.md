# Triplet CPU machinery audit — 2026-11-17

Scope: doubles performance and remaining obstacles before another production
triplet solve. CCSDT and CCSDTQ currently follow different implementations.
No production jobs were launched, interrupted, or modified for this audit.

## Implemented and measured

The CPU core now expands canonical spin-orbital amplitudes with OpenMP,
including parallel initialization of the output. A canonical excitation and
each of its signed permutations own distinct output entries; no atomic
updates or changes to the equations are needed. NumPy remains the reference
and fallback for unsupported dtypes/backends.

On the full CuF scalar-X2C ANO-RCC-VDZP checkpoint (26 correlated electrons,
29 virtual orbitals, B1 sector), 56 Sapphire Rapids cores, 8 GiB intermediate
cache budget, and the same normalized random rank-1–4 direction:

| CCSDTQ doubles output | NumPy expansion | Compiled expansion |
|---|---:|---:|
| Action wall time | 15.2791 s | 5.66574 s |
| Peak process RSS | 67.0146 GiB | 67.3482 GiB |
| Canonical doubles outputs | 33,558 | 33,558 |
| Maximum numerical difference | — | 0 (bitwise equal) |

This is a 2.70x incremental improvement over today's optimized packed
contraction path, not the much older tiled solver. Profiling reduced
expansion from about 9.4 s to 0.327 s. Remaining doubles work is mainly
symmetry-blocked contractions (3.31 s including planning), direct R4 feedback
(1.03 s including lookup setup), and T1 dressing (0.64 s). Timings include
profiling overhead and first-use plans. Complete worker times were
96.66/98.93 s because loading and construction dominate this one-action
probe; repeated EOM actions reuse that setup.

For a singles-only trial, the earlier zero-work fix already removed the zero
T1 derivative of g(oovv) and its entire ground-T4 doubles contraction.
The previous packed/tiled variants both take about 3 s on that case.
Signed T4 lookup maps are now built lazily so that this case also avoids
unused maps. No separate speedup is claimed for this small change.

The inherited Davidson memory guard counted the logical dense T4 shape
even with packed storage. It now uses the engine's conservative tiled
workspace estimate plus packed search vectors and images. A budget-limited
four-electron triplet solve is validated against FCI. The constructor's
workspace estimate remains conservative, not a peak-RSS guarantee.

Validation: eight canonical-expansion checks (ranks 1–4, one/three threads,
strided input and empty sectors), seven triplet/FCI/dense-reference checks,
and six shared EOM memory/audit/core-loader checks passed. Core ABI is 21;
stale prebuilt libraries are rejected.

## Remaining priorities

1. **Batch the CCSDTQ rank-four work across occupied slots.** The scalar
   occupied-slot loop repeatedly forms raw virtual tiles and projects them
   into canonical output. The earlier benchmark covers eight slots, not a
   full rank-four action. Fast triples and doubles do not establish a
   practical full eigensolve time. Use the shared compiled tile-program/GEMM
   machinery for batches, reuse contractions across occupied tuples, and
   profile several spin/irrep slot classes. A full action or representative
   measured throughput is a production launch gate.

2. **Compile the spin projector.** A full 360,194,362-component CuF vector
   takes 31.95 s per projection; its idempotence defect is 1.09e-16 and peak
   RSS for this standalone probe is 12.27 GiB. About 30.57 s is in the packed
   rank-four projector, mostly NumPy indexed gather/add operations. A native
   route plan grouped by destination (and by source for the transpose) can
   give each output one owner and preserve accumulation order without
   atomics. This is a candidate, not an implemented or measured speedup.

3. **Integrate the CCSDT triplet CPU path into the main package.** The running
   aug-QZ campaign imports its frozen `benchmarks.cuf.cuf_triplet_ccpy`
   adapter and CCpy dependencies. That adapter is absent from this checkout's
   `itercc` API; it is separate from the new analytic `TripletEOMCCSDTQ`.
   The actual campaign sources were inspected. Its `--cpu-slabs` route
   already uses the shared bounded helpers, including applicable compiled
   dense CPU GEMMs. However, `sigma` enables raw triples packing only for
   CUDA, and `_ladder_active` also explicitly excludes NumPy. CPU execution
   therefore misses packed triples and pair-ladder savings. Port their
   arithmetic to shared CPU kernels, propagate irrep labels through the
   adapter, and validate against its independent spin-block reference.
   CCSDTQ timing improvements must not be attributed to this CCSDT driver.

4. **Reduce certification cost without weakening it.** The CCSDT probe runs
   the right action, discards it, and separately evaluates every spin block
   for a raw audit. That audit follows the full residual path rather than
   the CUDA raw-output shortcut. Retain/stream suitable raw diagnostics
   during the action and optimize audit contractions, while preserving an
   independent final certificate and spin/symmetry leakage checks. Do not
   replace full certification with a projected residual alone.

5. **Preserve expensive work on restart.** The production pilot saves Ritz
   vectors every few iterations, not the Davidson basis/images or a partially
   completed rank-four action. A timeout during the first long action loses
   its work. Restart records must bind the Hamiltonian, source/operator
   identity, spin/irrep sector, direction, and completed output ranges before
   saved images can safely be reused.

Already shared: Davidson uses compiled CPU dots/matrix products, block
orthogonalization and thick restart with reused images. Replacing that
machinery is lower priority than action and projector costs.

## Evidence

### Shared machinery follow-up (November 17)

The native canonical expansion now lives in `core/src/cpu/tensor_scatter.cpp`
and is dispatched by `itercc._tensor_scatter`, not a quadruples-specific
implementation. `itercc._tensor_symmetry` owns signed permutation conventions.
Compatibility imports preserve earlier internal users. The canonical native
kernel's arithmetic and traversal are unchanged from the 2.70x doubles result.

| Consumer | Shared route and applicability |
| --- | --- |
| Triplet EOM-CCSDTQ, rank 1–4 expansion | `unpack_antisymmetric`: native float64 CPU; common complex/device/reference fallback |
| Singlet EOM-CCSD/CCSDT | `SingletSpace.unpack` uses `unpack_sector`; retains its spatial pair/triples projectors |
| CCSD/CCSDT static and optical response | Same `SingletSpace` and Jacobian machinery as singlet EOM, including complex fallback |
| CPU/GPU triplet tiled actions | Shared `IntermediateCache`, with action-local ownership and a byte budget |
| All EOM users of `HostDavidson` | Forward the engine's actual storage estimate, with transfer allowance; packed T4 is not charged as a logical dense tensor |
| Ground CCSD/CCSDT/CCSDTQ and singlet EOM-CCSDTQ | Already use the compiled tensor/GEMM/tile/Krylov infrastructure; their spatial layouts must not use spin-orbital antisymmetric expansion |

Sector expansion is a new rank-independent parallel zero/scatter primitive.
Local four-thread microbenchmarks were bitwise exact: 16 million output doubles,
one quarter populated, median 19.04 ms NumPy versus 17.48 ms native (1.09x).
The 262,144-entry case gave 0.1034 versus 0.0844 ms (1.23x). These are expansion
timings, not solver speedups. Single-thread and one-eighth-filled cases were
slower; automatic dispatch therefore keeps NumPy for fewer than four workers,
less than one-quarter population, fewer than 262,144 output elements, or fewer
than 65,536 output elements per worker. Complex/GPU paths also retain their
backend implementation. Dispatch thresholds are conservative heuristics, not
a guarantee of speedup on every cluster topology.

Cache replacement now subtracts the previous allocation before inserting a
new value, fixing double-counted bytes for a reused key. An oversized replacement
invalidates the old value. The current action normally inserts only cache misses;
these semantics make broader reuse safe without enlarging its memory budget.

Core ABI is now 22. CMake, source-distribution packaging and runtime compilation
include the shared translation unit. Old binaries are rejected and rebuilt or
use the existing reference fallback. Frozen jobs are not modified.

The separate frozen CCpy triplet-CCSDT adapter uses partially antisymmetric
spin blocks, not these fully antisymmetric canonical coordinates. Its missing
CPU triples-packing/pair-ladder integration is still separate work; this
refactoring does not claim that those optimizations are implemented or that
already submitted jobs inherit them. Likewise, occupied-slot batching and a
compiled spin projector remain future work.

The benchmark driver is `benchmarks/tensor_scatter_cpu.py`.
The EOM/response/quadruples/core and expansion run passed 93 tests (50 optional
or backend tests skipped); after cache extraction, all 9 selected cache/triplet
tests passed again, including FCI roots and six-electron dense Jacobian parity.
GPU execution was not retimed or certified in this follow-up. Source SHA256
manifest and raw test receipts are stored alongside the benchmark JSON.

The raw test receipts, benchmark JSON, source manifests and the cluster jobs
behind these numbers are kept outside the repository. New production solves remain
on hold; these jobs are bounded diagnostics only.
