# User guide

- [Inputs](#inputs)
- [CCSD and CCSDT](#ccsd-and-ccsdt)
- [CCSDTQ](#ccsdtq)
- [Excited states (EOM-CCSDT)](#excited-states-eom-ccsdt)
- [Polarizabilities (linear response)](#polarizabilities-linear-response)
- [Spin-orbit coupling](#spin-orbit-coupling)
- [GPUs](#gpus)
- [Performance and tuning](#performance-and-tuning)
- [How results are validated](#how-results-are-validated)

## Inputs

All solvers take a real, orthonormal, spin-free MO Hamiltonian:

| Argument | Meaning |
| --- | --- |
| `h` | one-electron matrix (n x n) |
| `eri` | two-electron integrals in chemists' notation, `eri[p,q,r,s] = (pq\|rs)` (n^4) |
| `nocc` | correlated occupied orbitals; occupied orbitals come first |
| `constant` | nuclear repulsion plus frozen-core energy |
| `orbital_irreps` | optional Abelian irrep (0..7, XOR products) of each orbital |

From PySCF, `itercc.pyscf_interface.from_rhf(mf, frozen=k)` returns all of them
for a converged RHF (or scalar-X2C RHF): `model.h`, `model.eri`, `model.nocc`,
`model.constant`, `model.orbital_irreps` (when the molecule was built with
`symmetry=True`; linear molecules get their D2h or C2v subgroup labels) and
`model.mo_coeff`. `dipole_integrals(model, mol)` gives the dipole matrices and
their irreps for response calculations.

Symmetry is free speed: with `orbital_irreps` and `symmetry_tolerance=1e-8`
the integrals are projected onto their exact symmetry blocks (they deviate only
by rounding) and every product skips the symmetry-forbidden blocks. Results do
not change.

## CCSD and CCSDT

```python
cc = RestrictedCC(model.h, model.eri, model.nocc, model.constant,
                  orbital_irreps=model.orbital_irreps, symmetry_tolerance=1e-8)   # rank=2: CCSD
result = cc.solve(energy_tol=1e-10, residual_tol=1e-8)
result.energy, result.correlation_energy, result.converged
t1, t2, t3 = result.amplitudes        # t1[i,a], t2[i,j,a,b], t3[slot,a,b,c] (slots: cc.triples_indices)
```

A solve is converged when the energy change is below `energy_tol` and the
largest raw residual element of every rank is below `residual_tol`.

| `solve` option | Default | Use |
| --- | --- | --- |
| `energy_tol`, `residual_tol` | 1e-11, 1e-9 | 1e-12 and 1e-10 before EOM or response (required) |
| `max_iter` | 100 | residual evaluations |
| `diis_size` | 6 | DIIS history; T1 and T2 by default |
| `diis_triples` | False | also extrapolate T3 (more memory, sometimes fewer iterations) |
| `damping`, `level_shift` | 0 | for difficult cases (stretched bonds, near-degeneracies) |
| `initial` | None (MP2 doubles, zero triples) | `(t1, t2, t3)` from an earlier solve, e.g. a nearby geometry |
| `callback(record)` | | called after every residual evaluation |

## CCSDTQ

```python
solver = RestrictedCCSDTQ(model.h, model.eri, model.nocc, model.constant,
                          orbital_irreps=model.orbital_irreps, backend="cpu")    # default: "cuda"
t1, t2, t3 = ccsdt.amplitudes
result = solver.solve(initial=(t1, t2, solver.expand_triples(t3)),            # T4 starts at zero
                      energy_tol=1e-10, residual_tol=1e-7, diis_size=8, damping=0.2)
```

T4 is stored only for ordered occupied quadruples i <= j <= k <= l and, with
labels, only for symmetry-allowed virtual quadruples. On the CPU the T4 terms run
as compiled tile programs on symmetry-blocked GEMMs.

- **Diffuse basis sets**: a stiff T4 mode can make the T4 residual grow while
  the energy settles. Put T4 into DIIS: `diis_quadruples="host"` (history in
  host memory; compact float32 entries on the CPU with the same fixed point),
  with `diis_size=8` to 10 and `damping=0.2`.
- **Checkpoints**: `solve(checkpoint=lambda s: solver.save_state("state_dir", s))`
  writes the full state (amplitudes and DIIS history) after every iteration;
  `state, _ = solver.load_state("state_dir")` and `solve(resume=state)` continue
  the iteration exactly, in another process and on either backend.
- **`precision="single"`** (CPU): the memory-bound passes read float32 copies of
  T3 and T4 rows; all arithmetic stays double. About 3% faster, moves energies by
  about 1e-11 Eh, and residuals below about 1e-9 become unreachable.
- **GPU memory**: `solver.device_bytes(diis_size=8, diis_quadruples="host")`
  estimates the peak. If the residual does not fit, `host_r4=True, block_size=2`
  keeps R4 in host memory.

## Excited states (EOM-CCSDT)

```python
eom = EOMSinglet(cc, ground.amplitudes, orbital_irreps=model.orbital_irreps, irrep=2)
states = eom.solve(nroots=3, residual_tol=1e-6)
states.excitation_energies, states.total_energies, states.converged
```

- Singlet EOM-EE-CCSDT with full singles, doubles and triples; `method="ccsd"`
  with a `rank=2` ground state gives EOM-CCSD.
- One Abelian sector per solve (`irrep`, PySCF numbering: C2v A1, A2, B1, B2 =
  0..3; D2h Ag, B1g, B2g, B3g, Au, B1u, B2u, B3u = 0..7). Without labels only
  `irrep=0` (all states) is available.
- The ground state must be converged to a residual of 1e-10.
- The Jacobian actions are exact analytic derivatives of the CCSDT residual; a
  restarted non-Hermitian Davidson solver finds the roots, and every root is
  accepted only after a fresh full-space residual check.
- `residual_tol=1e-6` put excitation energies within about 3e-8 Eh of tightly
  converged roots on water, with about 30% fewer Jacobian actions than 1e-9.
- `max_space` (default 20, at least `2 * nroots + 1`) bounds the Davidson
  subspace; `memory_budget_gib` rejects a space that would not fit.
- Not implemented: left eigenvectors, transition moments, oscillator
  strengths. Experimental: triplet (spin-flip) EOM-CCSDTQ (`TripletEOMCCSDTQ`).

For CPU triplet EOM-CCSDTQ, `occupied_tiles=True, packed_spin=True` uses the
compiled canonical spin-orbital contractions when the CPU core is available.
They avoid dense T4 expansions in lower-rank feedback and part of the
quadruples action. Five nonlinear quadruples terms reuse small effective
coefficients. The remaining products use the shared symmetry-blocked CPU
executor, including their intermediate contractions. Set
`packed_contractions=False` for the earlier tiled reference, or `True` to
require the compiled path (it raises if unavailable).

Cluster runs can set `intermediate_cache_gib=8` (default 0) to retain repeated
products within each Jacobian action. This is a byte cap on cached arrays,
not on total process memory. It is included in the existing memory estimate;
set `memory_budget_gib` for the full allocation. Cached direction-dependent
products are discarded between actions. `last_intermediate_cache` reports
hits, misses and peak cached array bytes. Ground-state effective coefficients
and canonical address maps are reused for the lifetime of the EOM engine.
On the CPU the rank-four slots then run as one compiled program: every raw
tile of an action on its own thread, intermediates shared between tiles
through the cache (large ones computed in pieces that waiting tiles help
with), and the direct packed T4 terms by opened letters (for each tuple of
the amplitude's occupied letters that come from the output row, the
amplitude block gathered once per class of the row's other particles and
one GEMM with the coefficient block; ranks two and three alike). Tile products form only the entries their projection
reads (letters of each antisymmetric subset in increasing spin-orbital
order), in ranks two and three among the occupied letters as well. Its waves
of slots hold `partial_sums_gib` of partial sums (default: as much as the
cache, at most 32 GiB), so a cache can be as large as an action's distinct
intermediates (262 GiB on CuF) without waves of that size; without a cache
setting the program takes half of what `memory_budget_gib` leaves beside the
engine's estimate (which counts the program's tiles and buffers). Progress
callbacks arrive after each wave. The action agrees with the earlier Python slot loop to round-off, and
bit for bit with `ITERCC_TRIPLET_FUSED=0`; the switches below fall back.
An action can also be shared by processes, one per socket (or node):
`itercc._slot_split.Partners(engine, builder, kwargs, parts=2)` starts
workers that build the same engine, deals the rank-four slots out in blocks
and the tiles of ranks two and three in turn, and adds the workers' rows to
each `sigma`: the one-process action bit for bit in rank four, to rounding in
ranks two and three (the CuF triplet driver's `--sockets 2`, its launcher
binding itself to NUMA node 0).
Packed T4 is read in place, so by default (`occupied_slab=None`) its summed
occupied letters stay whole. A positive `occupied_slab`, for example 2, still
cuts them into slabs of at most that width, which with packed T4 only repeats
work (an intermediate and a product per slab); keep the default unless
benchmarking. Both CuF triplet benchmark drivers expose this as
`--occupied-slab` and record the setting. It changes contraction tiling, not
the Hamiltonian or equations.
Canonical CPU input expansion also uses the compiled core. The packed
engine's Davidson memory check accounts for its actual packed/tiled storage
instead of a logical dense rank-four tensor. See the
[triplet CPU audit](triplet-cpu-audit.md) for measured doubles timings and
remaining CCSDT/CCSDTQ integration work.

`python -m benchmarks.spin_quadruples_cpu` compares this path with the tiled
reference on a saved CuF DZ checkpoint. Its triples sample retains all
virtuals and internal occupied sums while restricting external holes; its
quadruples sample retains complete canonical occupied slots. These timings
are bounded action benchmarks, not full eigensolve speedups. Both versions
must use the same input, vector, output sample and thread allocation.

## Polarizabilities (linear response)

```python
dipoles, dipole_irreps = dipole_integrals(model, mol)
response = DipoleResponse(cc, ground.amplitudes, dipoles,
                          orbital_irreps=model.orbital_irreps, dipole_irreps=dipole_irreps)
static = response.polarizability(float("inf")).tensor_au
at_1064 = response.polarizability(1064.0).tensor_au
damped = response.polarizability(532.0, damping_Eh=0.001).tensor_au      # complex
```

- Full CCSDT linear response (the CC Lagrangian, with the ground multipliers
  folded in by an extra zero-frequency solve), clamped nuclei and fixed
  orbitals, in atomic units. `method="ccsd"` gives CCSD response.
- `dipoles` are the electronic dipole matrices (-r) in the same orbital frame
  as the amplitudes; for scalar X2C, supply picture-changed dipoles.
- Keep one `DipoleResponse` for a sweep of wavelengths: every converged
  solution can seed later solves. A positive `recycle_gib` budget that is too
  small for the image store still enables projected real dipole guesses.
- For preemptible jobs, pass `checkpoint=ResponseCheckpoint(directory, identity)`
  to `DipoleResponse`. Use an identity derived from the Hamiltonian, ground
  amplitudes, dipoles and solver version, and a separate directory per writer.
  Restarted solves check each saved dipole vector against the current residual.
- `tol` is the relative residual of each linear solve (default 1e-8). On water,
  `tol=1e-6` reproduced tightly converged tensors to about 2e-8 relative.

## Spin-orbit coupling

`spin_orbit_state_interaction` diagonalizes singlet and triplet states coupled
by a one-electron SOC operator, from transition spin densities that the caller
supplies. `x2c_spin_orbit(mf, mo_coeff=model.mo_coeff)` builds the bare
one-electron X2C SOC operator (spin-traceless) in the correlated MO basis.

Conventions: densities store D_alpha[p,q] - D_beta[p,q] with
D[p,q] = <bra|a_q^+ a_p|ket>, axis order (bra, ket, p, q); singlet-triplet
densities at M = 0 in both directions and triplet-triplet densities at M = +1,
with |1,+1> = S+|1,0>/sqrt(2). The three operator matrices multiply the Pauli
matrices (no factor 1/2). The result's `basis` labels states as
(S, root, M). `spin_orbit_from_components` accepts the Cartesian matrix elements
directly.

## GPUs

Every method also runs on NVIDIA GPUs through CuPy (`pip install cupy-cuda12x`):

```python
from itercc.restricted import cuda_options
cc = RestrictedCC(..., orbital_irreps=labels, **cuda_options())   # every fused CUDA path
# without orbital labels: **cuda_options(labelled=False)
EOMSinglet(..., backend="cuda");  DipoleResponse(..., backend="cuda");  RestrictedCCSDTQ(..., backend="cuda")
```

Use data-centre GPUs with full double-precision throughput (A100, H100). A
56-core CPU socket runs CCSDT faster than an A100 (see
[benchmarks](benchmarks.md)); the GPU path pays off when CPU nodes are scarce or
for CCSDTQ problems that fit in device memory. CUDA results are deterministic,
and saved CCSDTQ states carry the autotuned GEMM choices so resumes are bit for
bit.

## Performance and tuning

Tensor expansion, signed permutations and bounded intermediate caching are
shared package machinery, rather than method-specific copies. EOM and response
use the same sector-expansion dispatcher; canonical spin-orbital amplitudes use
its antisymmetric counterpart. Spatial and spin-orbital symmetries remain
distinct. See the [reuse and validation matrix](triplet-cpu-audit.md#shared-machinery-follow-up-november-17)
for applicability, measured timings and remaining integration work.
Packed spin T4/R4 tile expansion and rank-2–4 projection also use shared native
gather kernels; see the [packed-tile measurements](triplet-cpu-audit.md#native-packed-tile-gathers-november-17).

- **Threads**: one per physical core of the process's CPU affinity. Set
  `ITERCC_CPU_THREADS` to the cores you were allocated, or pass `cpu_threads=`
  to a solver: `RestrictedCC`, `RestrictedCCSDTQ`, `EOMSinglet`,
  `DipoleResponse` and `TripletEOMCCSDTQ` all take it, and an excited-state
  engine inherits its ground-state solver's count (the kernels ignore
  `OMP_NUM_THREADS`; hyperthreads make them slower). Do not set `OMP_PROC_BIND`.
- **Sockets**: run one calculation per socket with its memory bound to it,
  `numactl --cpunodebind=0 --membind=0`. A second socket gives one calculation
  little, while two independent calculations per node nearly double throughput.
- **Reproducibility**: all builds use `-ffp-contract=off` and fixed-block
  reductions, so results are bitwise identical for any thread count and
  instruction-set level.
- **Compiled core**: found in `ITERCC_CORE_DIR`, a source checkout's
  `core/build`, per-level builds installed in the package
  (`libitercc_core.x86-64-v4.so`, `-v3`, `-v2`; the best one the CPU supports
  loads), or else compiled on first use into `ITERCC_KERNEL_CACHE` (default
  `~/.cache/itercc`, keyed by CPU model). For clusters, build the portable levels
  once: `bash core/build_levels.sh` (add `ITERCC_CUDA=ON` for the CUDA library).

Every switch is registered in `itercc/_runtime.py` (`SWITCHES`); "Read" says when it
takes effect: on each call, when a solver or engine is built, at import, when the
core loads, or inside the core (read once by C++).

| Environment variable | Read | Effect |
| --- | --- | --- |
| `ITERCC_CPU_THREADS` | call | kernel threads when no `cpu_threads` option is given (default: physical cores in the affinity mask) |
| `ITERCC_CPU_KERNELS` | call | `0`: NumPy path only (no compiled core, no warning) |
| `ITERCC_CORE_DIR` | load | directories searched first for a prebuilt core |
| `ITERCC_KERNEL_CACHE` | load | where first-use core builds go (default `~/.cache/itercc`) |
| `ITERCC_CXX` | load | the compiler of first-use core builds (default `c++`, then `g++`) |
| `ITERCC_MALLOC` | load | `0`: leave glibc's malloc thresholds alone |
| `ITERCC_BASIS_PATH` | call | extra directories of basis-set files |
| `ITERCC_NATIVE_TRANSFORM` | call | `0`: PySCF's integral transformation in `from_rhf` instead of the core's |
| `ITERCC_BLOCKED` | construction | `0`: CCSDT on dense instead of symmetry-blocked T3 |
| `ITERCC_BARE_LADDER` | construction | `0` or `1`: particle-particle ladder without or with the cached bare sectors (default: always in ground states, from 30 virtuals in EOM/response actions) |
| `ITERCC_OPEN_DRESSING` | call | `0`: CCSDT T1 dressing writes the dressed g(vvvv) instead of ladder sectors |
| `ITERCC_DRESS_SLABS` | core | `1` or `0`: CCSD T1 dressing by slabs or by plane tasks (default: plane tasks with more than 3 threads per occupied orbital) |
| `ITERCC_CPU_TANGENT` | construction | `0`: EOM/response actions on the generic path |
| `ITERCC_TILE_BLOCKED_T3` | call | `0`: CCSDTQ tile programs read dense T3 |
| `ITERCC_TILE_FUSED` | call | `0`: CCSDTQ tile programs make one pass per term |
| `ITERCC_TILE_MEMORY` | call | CCSDTQ tile memory budget in bytes (default: host memory / 8) |
| `ITERCC_TILE_TEAM` | call | CCSDTQ threads per tile-program team |
| `ITERCC_LOWER_TEAMS` | call | CCSDTQ concurrent teams of the T4-into-lower-rank passes |
| `ITERCC_TEAM_SCHEDULE` | call | `rounds`: CCSDTQ teams in fixed rounds (diagnostic) |
| `ITERCC_LOWER_CHUNK` | call | occupied rows per chunk of the CCSDTQ T4-into-lower-rank pass (default: 2 GiB) |
| `ITERCC_DIIS_COMPACT` | call | `0`: float64 instead of compact float32 host T4 DIIS history |
| `ITERCC_TRIPLET_PROGRAM` | call | `0`: triplet EOM-CCSDTQ rank-four slots in the Python loop instead of the compiled program |
| `ITERCC_TRIPLET_FUSED` | call | `0`: triplet slot program adds products into a tile one by one (the loop's action bit for bit) |
| `ITERCC_DIRECT_DENSE` | call | `0`: triplet direct packed T4 terms by per-row sparse lookups instead of contexts |
| `ITERCC_SLOT_POSITIONS` | call | `0`: triplet slot-program intermediates and tiles keep virtual axes in orbital order, not irrep order |
| `ITERCC_HOLES_FIRST` | call | `0`: triplet dense rank-three tensors with the occupied axes innermost, not outermost |
| `ITERCC_PAIR_SUMS` | call | `0`: triplet tile products sum both orders of an exchangeable pair of summed letters instead of x < y twice |
| `ITERCC_COMPACT_TILES` | call | `0`: triplet slot-program tiles stored whole (nv^4), not only their ordered entries |
| `ITERCC_TILE_CHAINS_INNER` | call | `0`: compact slot-program tiles store their axes in letter order, not chains innermost |
| `ITERCC_INTER_CHAINS_INNER` | call | `0`: compact slot-program intermediates store their axes in letter order, not chains innermost |
| `ITERCC_COMPACT_INTERMEDIATES` | call | `0`: triplet slot-program intermediates stored whole, not only their ordered entries |
| `ITERCC_ORDERED_INTERMEDIATES` | call | `0`: triplet slot-program intermediates formed at every entry, not only where their readers read |
| `ITERCC_DIRECT_OPENED` | call | `0`: triplet rank-four direct packed T4 terms by contexts per output slot (`ITERCC_DIRECT_DENSE`) instead of one GEMM per hole tuple and class |
| `ITERCC_SLOT_SHARE_WORK` | call | multiply-adds from which a slot-program intermediate is computed in shared pieces (default 2^22) |
| `ITERCC_SLOT_PROFILE` | core | `1`: slot-program phase timers (`itercc_triplet_profile`) |
| `ITERCC_SLOT_SORT` | core | `0`: slot-program tiles in slot order (diagnostic) |
| `ITERCC_TILE_RUNS` | core | `0`: slot-program tiles (dense or compact) cleared whole, not only where symmetry allows entries (diagnostic) |
| `ITERCC_SYM_MIN_SHARE` | core | GEMM executor: least work per thread of a shared block |
| `ITERCC_SYM_PARALLEL_WORK` | core | GEMM executor: work from which one block runs on all threads |
| `ITERCC_GEMM_NC` | core | `auto`: narrower GEMM column blocks for deep products (experiment) |
| `ITERCC_GEMM_KNOCKOUT` | core | GEMM micro-kernel stages to skip (timing diagnostic only) |
| `ITERCC_GEMM_AUTOTUNE` | import | `0`: CUDA GEMM routes chosen statically instead of timed |
| `ITERCC_BOUNDED_GEMM` | import | `1`: CUDA slab einsum products through the GEMM router |
| `ITERCC_SLAB_MIB` | import | `bounded_einsum`: MiB per slab intermediate |
| `ITERCC_SYMMETRY_FUSED_DEPTH` | import | CUDA symmetry blocks: deepest fused-kernel sum |
| `ITERCC_SYMMETRY_FUSED_SKINNY` | import | CUDA symmetry blocks: widest skinny fused product |
| `ITERCC_SYMMETRY_DIRECT_DEPTH` | import | CUDA symmetry blocks: deepest direct-kernel sum |

## How results are validated

The test suite (`pytest`, about 1030 tests) checks residuals and energies
against PySCF's RCCSDT and RCCSDTQ, exact diagonalization and an independent
spin-orbital implementation; EOM roots and polarizabilities against
determinant-space references and finite-field derivatives; and every compiled
kernel against the NumPy path, bit for bit where the arithmetic is the same.
EOM-CCSDT on LiH/STO-3G agrees with NWChem's TCE to 7e-12 Eh. The comparison
runs in [benchmarks](benchmarks.md) agree with PySCF, ccpy, NWChem and ORCA to the
convergence thresholds used.
