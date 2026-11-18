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

1. **Batch the CCSDTQ rank-four work across occupied slots.** The compiled
   slot program (Evidence below) now runs every tile of an action in the core,
   about 6.5x faster on water cc-pVDZ and 3.3x on a 16-slot CuF sample (0.34-0.44
   s per slot on the first 400 slots); the g-T4 intermediates, computed one
   per thread while other tiles wait, now limit it. The scalar
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

### Compiled rank-four slot program (November 17)

`itercc/_spin_quadruples_program.py` records the rank-four slots once and
runs every raw tile of an action in the compiled core (`itercc_triplet_slots`,
core ABI 28). A tile kind (projection group and raw spin pattern) is
recorded with its occupied tuple symbolic: dense operands at offsets affine
in the tuple, packed T4 rows from the hole tables, intermediates keyed by
their expression, plans per irrep pattern. A slot class (hole spins, particle
spin groups, repeated occupied indices) fixes a slot's tiles, so the core
expands each slot from its class. Every tile runs on one thread with the
single-threaded contraction routes; intermediates are shared between tiles
through a cache with one producer per key; each tile's gathers go to partial
sums that the slot adds in the loop's order. The products into a tile with
equal output offset tables (same partition of the tile's virtual indices)
run as one fused GEMM per output block after the tile's intermediates, so
the tile is read and written once per block instead of once per product.
`ITERCC_TRIPLET_FUSED=0` runs them one by one in place, which gives the
loop's action bit for bit; fused, the action agrees with it to about 3e-16
(relative). Either way the result does not depend on the thread count.
`ITERCC_TRIPLET_PROGRAM=0` runs the loop and `ITERCC_SLOT_PROFILE=1` times
the program's phases.

Water, 12 desktop threads, one model and ground state, the same random
direction (seconds per action after the first; every column uses the tile
trees below):

| Basis | Slot loop | Compiled, unfused | Compiled, fused | First action (fused, with recording) |
| --- | ---: | ---: | ---: | ---: |
| 6-31G | 8.29 | 0.49 | 0.47 | 1.95 |
| cc-pVDZ | 26.41 | 4.91 | 4.04-4.32 | 5.80 |

Unfused output equals the loop's bit for bit; fused differs by at most
2.5e-16 relative. Before the tree change below the compiled cc-pVDZ action
took 5.2 s. In the fused cc-pVDZ action the slots take 2.2-2.5 s, the direct
packed terms 1.1 s and ranks one to three 0.6 s. On one thread the slots take
13.1 s (products into tiles 7.6 s, intermediates 1.9 s, gathers 0.6 s); on 12
threads the products cost 2.3x their single-thread time, as the two memory
channels saturate.

Tile trees: `contraction_schedule` used to pick the tree with the smallest
peak intermediate. It now counts an intermediate's multiply-adds once per
tiles that share it (no**(fixed occupied letters it does not read)), among
trees whose largest intermediate stays within a tile (nv**4). Two diagrams
change. goovv t2 t4 (`mnef,aeij,bcdfklmn`) contracts g with T4 first, an
intermediate per (k, l) shared through the cache, instead of a packed-T4
product per tile: on one thread its 7,038 products took 9.4 s, now 432
intermediates take 0.64 s. gvovv t2 t3 (`amef,beij,cdfklm`) likewise. Summed
over the diagrams, amortized multiply-adds per tile fall 2.0x at water size
and 6.6x at CuF size. The loop and the program share the trees; against the
old trees the action changes at round-off. Without `intermediate_cache_gib`
the program's cache gets half of what the memory budget leaves beside the
engine's estimate (at least 256 MiB): with 256 MiB the cc-pVDZ slots took
4.9 s instead of 2.7 s (unfused).

The CuF layout has 14,235 slots with about 445 tiles each (6.3 million tiles
per action) and up to 10 million partial sums (79 MB) per slot; slots run in
waves of at most 256 MiB of partial sums.

CuF, the same 16-slot general-input quadruples sample as below (full
checkpoint, 56 Sapphire Rapids cores, 160 GiB, 8 GiB intermediate cache,
source `100e276`, one allocation, each case in a fresh worker that runs the
sample twice; no cProfile):

| Case | First action | Second action | Peak RSS (GiB) | Output against the loop |
| --- | ---: | ---: | ---: | --- |
| Slot loop (new trees) | 47.22 | 43.26 | 80.9 | — |
| Compiled, unfused | 34.43 | 25.21 | 111.1 | bitwise equal |
| Compiled, fused | 30.58 | 20.80 | 112.2 | 2.3e-17 relative |

The first compiled action includes recording the whole 14,235-slot layout
(8.1 s: 896 kinds, 28 slot classes, 187,696 plan sets). The loop with the new
trees agrees with the `42cc1e4` loop (old trees, 35.48 s with cProfile) to
8.6e-18 relative, but is slower on this sample: its 8 GiB cache fills with the
g-T4 intermediates (5.6 MB each here) and misses rise from 19,847 to 42,832.
Thread seconds of the fused second action (56 threads x 20.8 s = 1,165):
intermediates 327, products into tiles 70, cache waits 33, gathers 3, tile
zeroing 2. So intermediates dominate (the cache is full: 12,353 hits, 40,885
misses), and about 60% of the thread time is idle, mostly at the barriers
between waves of about three slots each (10 million partial sums per slot).

An offline model explains most of it: the real `SlotProgram` built for the
CuF layout (checkpoint irreps, 13 occupied, 29 virtual, B1) on shape-only
operands, and a sequential walk over its tiles with the cache as an LRU
(model estimates, not measurements). At CuF size the tile byte budget cut the
summed hole pair (m, n) of packed T4 into 28 slabs, so `mnef,bcdfklmn->bcdekl`
became 28 full-size intermediates per hole pair and 28 products into each
tile: 43,232 of them in the 16-slot sample. Read in place, packed T4 needs no
slab, and both the loop and the program now keep its summed letters whole
(an explicit `occupied_slab` keeps slabs). Multiply-adds and packed-T4 gathers
of the model, 16-slot sample, 8 GiB cache:

| Trees and slabs | Tile products | Intermediates | Packed-T4 gathers |
| --- | ---: | ---: | ---: |
| Old trees, slabs (`42cc1e4`) | 4.18e11 | 1.5e10 | 4.8e10 |
| New trees, slabs (`100e276`) | 1.11e11 | 3.92e11 | 4.8e10 |
| New trees, whole T4 | 5.2e10 | 2.18e11 | 2.7e10 |

For a whole action (6.49 million tiles) the old trees need 5.5e14
multiply-adds and 6.1e13 gathers. With whole T4 the new trees need 7.0e13 and
4.5e11 if every intermediate is kept (262 GiB), 2.9e14 and 2.7e13 with a
16 GiB cache in slot order, and 1.6e14 and 1.1e13 with the same cache when a
wave's tiles run sorted by occupied tuple (the T4 hole pair first) and waves
hold 200 slots. The executor now runs each wave's tiles in that order (only
the order of independent tiles changes) and sizes waves by the cache budget.

Measured with `d32cc04` on the same 16-slot sample and allocation type
(56 cores, 200 GiB, 8 GiB cache; seconds of the second action, the first in
parentheses):

| Case | `d32cc04` | `100e276` | Peak RSS (GiB) | Output |
| --- | ---: | ---: | ---: | --- |
| Slot loop | 25.91 (24.73) | 43.26 (47.22) | 81.3 | — |
| Compiled, unfused | 11.76 (16.91) | 25.21 (34.43) | 83.7 | the loop's, bit for bit |
| Compiled, fused | 10.68 (15.59) | 20.80 (30.58) | 83.9 | 2.2e-17 relative to the loop |

Cache misses fell from 40,885 to 10,541 and recording from 8.1 s to 4.0 s.
Against the `42cc1e4` loop (35.48 s, old trees and slabs) the fused program
is 3.3x faster. The fused outputs of `d32cc04` and `100e276` differ by
1.5e-17 relative. On the first 400 slots (8.5 million R4 rows, one action,
recording included) the fused program took 177.6 s with an 8 GiB cache
(91 GiB peak RSS) and 134.3 s with 32 GiB (135 GiB), with the same output bit
for bit: 0.44 and 0.34 s per slot, against 0.67 s per slot in the 16-slot
sample. Thread seconds (8 GiB): intermediates 3,287, cache waits 1,608,
products into tiles 848, gathers 98, tile zeroing 77, of 9,945; with 32 GiB,
2,128 and 884 for the first two. The g-T4 intermediates, each computed on one
thread while other tiles wait for it, are now the largest cost.

### Packed T4 rows in the GEMMs (November 17, `0afcdff`)

The blocked GEMMs now read packed T4 in place: a T4 operand is `SignedRows`
and `itercc_sym_contract_packed` gathers each piece of B a route needs into
its own scratch, so no dense T4 slab is expanded. The same 16-slot
general-input quadruples sample was run on the full CuF checkpoint with
three fresh workers in one 56-core Sapphire Rapids allocation (128 GiB, 8 GiB
intermediate cache, profiling on):

| Source and occupied slab | Seconds | Speedup | Peak RSS (GiB) | Output against the first row |
| --- | ---: | ---: | ---: | --- |
| Dense expansion `889eef3`, width 2 | 63.17 | 1.00x | 75.81 | — |
| Packed rows `42cc1e4`, width 2 | 41.49 | 1.52x | 75.71 | bitwise equal |
| Packed rows `42cc1e4`, automatic width | 35.48 | 1.78x | 75.51 | max abs 8.7e-19 |

The first row repeats the 63.05 s width-2 result below, so the allocations
are comparable. Without the dense slab, automatic (byte-budget) width is the
fastest setting here, as it is on water cc-pVDZ (12 desktop threads, full
action: 25.9 s automatic, 28.4 s width 2, 36.9 s width 1). The width-2
recommendation below holds only for sources before `0afcdff`. These are
component timings of 16 slots, not the 14,235-slot action.

### CuF slab-width validation (November 17, latest CPU kernels)

The `889eef3` update is numerically valid on the full 26-electron/29-virtual
CuF checkpoint, but its byte-budget-sized occupied slabs are slower here than
the preceding native-gather version `461b1e5`. Fresh paired workers used the
same 56-core Sapphire Rapids allocation, 128 GiB requested memory and 8 GiB
intermediate cache. Each row is one paired sample, not a whole eigensolve.

| Output/configuration | Reference seconds | Candidate seconds | Peak RSS, reference → candidate (GiB) | Maximum absolute error |
| --- | ---: | ---: | ---: | ---: |
| Quadruples, general input, 16 slots, default wider slabs | 76.31 | 91.77 | 75.82 → 75.63 | 8.67e-19 |
| Quadruples, singles-only input, 128 slots, defaults | 108.11 | 102.58 | 48.82 → 48.81 | 0, bitwise equal |
| Complete triples output, general input, defaults | 136.19 | 136.24 | 72.56 → 72.56 | 0, bitwise equal |
| Quadruples, general input, 16 slots, candidate slab width 2 | 72.06 | 63.05 | 75.93 → 75.82 | 0, bitwise equal |

The wider slabs reduced contraction calls but increased both expansion and
compiled contraction time. Width 2 retains the newer kernels and metadata
caches while recovering the faster tiling for this model. The matched narrow
slab result is 1.143x faster than the previous native-gather version. Complete
triples performance is unchanged. Do not multiply speedups from different
occupied-slot samples or infer a full eigensolve speedup.

`TripletEOMCCSDTQ(..., occupied_slab=2)` now forwards the same explicit width
to every shared tiled contraction, including lower-rank feedback and both
quadruples projection paths. `None` preserves automatic byte-budget sizing;
no existing CPU/GPU default was changed. Both CuF drivers expose and record
`--occupied-slab`. The initial width-2 measurement used an external wrapper
setting that identical contraction argument; dense six-electron Jacobian
checks also pass through the public option for widths 1, 2 and None.

Benchmark jobs: 61589064, 61589068, 61589071 and 61591673. Frozen sources,
profiles, numerical comparisons and raw logs are archived under
`itercc_triplet_latest_20261117` in the cluster scratch area. The initial
latest-source suite passed 27 focused tests; the public-width dense-Jacobian
check passed all three cases. GPU performance was not measured.

The exact deployment snapshot `dcc1950` was then checked with the public
`--occupied-slab 2` flag against the live job's frozen `ef5a57b` source, again
using paired fresh workers on the same 56-core allocation. Profiling was off.

| Output | Live-source seconds | Deployment seconds | Speedup | Peak RSS, live → deployment (GiB) |
| --- | ---: | ---: | ---: | ---: |
| Quadruples, general input, 4 complete occupied slots | 161.92 | 22.10 | 7.33x | 71.89 → 71.97 |
| Complete triples output, general input | 142.28 | 134.85 | 1.055x | 72.45 → 72.55 |

Both outputs were bitwise identical. These are component timings; the
quadruples sample is not the full 14,235-slot action. Jobs 61594003/61594005
preserve the final comparisons, and `summary.json` / `benchmark-acceptance.json`
in the same archive record every paired result, including the rejected wider
slabs. Worker wall times including setup were 355.58 → 101.36 s for the
quadruples pair and 227.81 → 220.99 s for triples. Setup variability means
the entire worker-wall reduction cannot be attributed to the action kernels.

### Native packed-tile gathers (November 17)

Profiling a general rank-1–4 direction on the full 26-electron/29-virtual CuF
checkpoint found that packed T4/R4 expansion consumed 112.15 s of a 131.29 s
four-slot quadruples action. Dense virtual tiles were repeatedly zeroed and
filled with Python/NumPy indexed assignments. This is different from the
previous optimization of complete lower-rank canonical amplitude expansion.

The shared CPU core now provides two rank-independent primitives in
`tensor_scatter.cpp` / `_tensor_scatter.py`:

- `signed_gather_rows`: expand rows through signed, one-based address maps,
  including zero rows, with parallel first-touch and no temporary value gathers.
  `PackedSpinQuadruples.expand` uses it for both T4 and R4 in every applicable
  CPU storage consumer. Ground/direction layouts share immutable virtual maps;
  the constructor's memory estimate includes their worst-case signed-map size.
- `PermutedGather`: accumulate ordered, weighted permutations directly into
  distinct packed result rows. The doubles, triples and quadruples projections
  use one implementation. It supports strided sources, including reversed axes,
  and preserves the NumPy sum order. Small gathers execute serial native loops
  without resizing the OpenMP team used by surrounding contractions.

No numerical screening or equations were changed. GPU, complex and unsupported
array layouts retain their reference paths. Core ABI 23 prevents loading an old
binary. Coordinate plans are action-local; address maps contain no amplitudes.

The occupied-slot batching experiment was rejected: on the initial matched
sample it increased the singles-only time from 11.87 to 29.56 s, and general
direction peak memory from 71.86 to 79.48 GiB with little time benefit. The
batching code was removed, not left enabled by default. An intermediate gather
prototype also exposed expensive OpenMP team resizing; that was corrected
before the final comparisons.

Final comparisons use fresh reference/native workers on the same 56-core,
384-GiB allocation with an 8-GiB intermediate cache. Reference switches are
`--numpy-projection --numpy-tile-expansion`; the standard driver is
`benchmarks/spin_quadruples_cpu.py`. This isolates the new gather kernels from
the previously validated packed contractions. The current production spectrum
job retains its frozen earlier code.

| Measured output on full CuF checkpoint | Reference (s) | Native (s) | Speedup | Peak process RSS, reference → native (GiB) |
| --- | ---: | ---: | ---: | ---: |
| Quadruples, general rank-1–4 direction, 4 occupied slots | 111.11 | 23.71 | 4.69x | 71.95 → 71.94 |
| Quadruples, singles-only direction, 16 occupied slots | 12.43 | 10.91 | 1.14x | 45.09 → 45.10 |
| Complete triples output, general rank-1–4 direction | 172.65 | 168.63 | 1.02x | 72.45 → 72.56 |

All three outputs were bitwise identical (maximum absolute difference zero).
The quadruples measurements cover selected complete occupied slots, not the
entire quadruples action or eigensolve. Complete triples performance is roughly
unchanged. These single paired samples establish the measured kernel improvement;
they do not establish a whole-solver speedup. In the final general-vector profile,
packed expansion took 5.68 s and compiled contractions took 9.66 s.

Jobs 61563382, 61563383 and 61563385 produced the singles-only, general-vector
and complete-triples receipts respectively. Their fresh-worker wall times,
including setup, were 98.73 → 90.25 s, 289.84 → 100.21 s and 251.31 → 247.05 s.
Setup varied substantially in the general-vector pair, so its worker-wall gain
cannot all be attributed to the new kernels. The 26-electron/29-virtual input
SHA256 is `9d0b98a2dd4bea77f52d0cde1b3fc2f5bf3692cf7c78325d4799cd7acc839bed`;
normalized directions use seed 714. `final-summary.json`, profiles and source
manifests preserve the raw results and exact benchmark source.

Validation: 25 focused expansion/projection, FCI triplet, six-electron dense
Jacobian and loader checks passed; 13 final kernel checks cover all supported
spin signatures, empty/reversed occupied slices, thread reproducibility,
alias/dtype/stride rejection, and shared virtual-map compatibility. GPU execution
was not rebenchmarked. The full-model timing and comparison receipts are kept
outside the repository.

### Earlier shared machinery follow-up (November 17, ABI 22)

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

This earlier change introduced core ABI 22. CMake, source-distribution packaging and runtime compilation
include the shared translation unit. Old binaries are rejected and rebuilt or
use the existing reference fallback. Frozen jobs are not modified.

The separate frozen CCpy triplet-CCSDT adapter uses partially antisymmetric
spin blocks, not these fully antisymmetric canonical coordinates. Its missing
CPU triples-packing/pair-ladder integration is still separate work; this
refactoring does not claim that those optimizations are implemented or that
already submitted jobs inherit them. The later native packed-tile gather work
above adds compiled projection and records why occupied-slot batching was rejected.

The benchmark driver is `benchmarks/tensor_scatter_cpu.py`.
The EOM/response/quadruples/core and expansion run passed 93 tests (50 optional
or backend tests skipped); after cache extraction, all 9 selected cache/triplet
tests passed again, including FCI roots and six-electron dense Jacobian parity.
GPU execution was not retimed or certified in this follow-up. Source SHA256
manifest and raw test receipts are stored alongside the benchmark JSON.

The raw test receipts, benchmark JSON, source manifests and the cluster jobs
behind these numbers are kept outside the repository. New production solves remain
on hold; these jobs are bounded diagnostics only.
