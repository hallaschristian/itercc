# Changelog

## Unreleased

- Triplet EOM-CCSDTQ tile products form only the tile entries their
  projection reads. A projection group takes only the permutations that keep
  each antisymmetric subset of external letters increasing, so a raw tile is
  read only where those letters are in increasing spin-orbital order (x < y,
  or x <= y when x's spin bit is the smaller): half of an antisymmetric
  pair, five sixths of a triple. The symmetry-block planner now takes
  ordering constraints on free letters (`_composite(..., ordered)`) and
  builds GEMMs over the kept tuples only; the slot program and the Python
  loop pass the constraints of each projection group and raw spin pattern
  (`ordered_letters`). The water cc-pVDZ action is bit for bit the one
  before; on the CuF layout the tile products of an action fall from
  6.7e13 to 1.7e13 multiply-adds (offline model), and the first 400 CuF
  slots take 62.7 s instead of 99.0 s (slot program 50.0 s instead of
  87.0 s; output within 3.4e-16 of the earlier one).
- `TripletEOMCCSDTQ(partial_sums_gib=...)`: the slot program's waves hold
  that many bytes of partial sums (default: as many as the cache, at most
  32 GiB) instead of always as many as the intermediate cache, so a cache
  can hold every distinct intermediate of an action (262 GiB on CuF, where
  each is then computed once: the modelled action falls from 1.2e14 to
  2.0e13 multiply-adds) without waves of the same size.
- Triplet EOM-CCSDTQ actions split over processes (`itercc/_slot_split.py`):
  every rank-four slot writes only its own rows with values that do not
  depend on the other slots of the process, so `Partners` deals the slots
  out in blocks of 16 to workers (one per NUMA node, bound with numactl)
  that build the same engine, and `sigma` adds their rows: the action of
  one process bit for bit. The slot program runs any table of slots
  (`SlotProgram.run(slots=...)`, no core change); the CuF triplet driver
  takes `--sockets N`.
- One runtime layer for every method (`itercc/_runtime.py`, documented in
  docs/design.md "Canonical machinery"). Threads: `RestrictedCCSDTQ`,
  `EOMSinglet`, `DipoleResponse` and `TripletEOMCCSDTQ` take a `cpu_threads`
  option like `RestrictedCC` (excited-state engines inherit the ground-state
  solver's), and every solve, action and residual call runs in one thread
  scope (`_runtime.threaded`); before, CCSDTQ, EOM and response kernels
  re-read the environment on every call and the EOM's internal solver
  dropped the option. Switches: every ITERCC_* variable is registered with
  its default and when it is read; the table in docs/guide.md is generated
  from the registry (nine switches were undocumented), and a test checks
  that the package and the core read nothing else. `ITERCC_BARE_LADDER`
  means the same in ground states and actions (`0` off, `1` on);
  `ITERCC_CPU_TANGENT` is read once per engine, so the ground-residual
  audit and the actions take one path. The `_scan_pool` and the
  ERI-leakage check follow the thread count.
- Machinery gaps closed: the triplet actions' T1-dressing derivative runs
  the singlet actions' one-pass kernel (`itercc_dress_derivative`) instead
  of four NumPy tensordots; two-operand products into new arrays in
  `bounded_einsum` (triplet coefficient tables, tiles without a symmetry
  engine) run on the core's GEMM executor; the package has one dense
  product engine (`_cpu_quadruples.dense_products`) instead of one per
  thread count plus a CCSDTQ copy; CCSDTQ forms the spin-summed T3 once for
  its T1-T3 equations and T3 intermediates instead of four times; on CUDA,
  fused and direct symmetry-block plans run without cuBLAS 12.5 (only
  grouped-GEMM plans then take the dense fallback) instead of the whole
  engine falling back.
- `restricted.cuda_options()` is the production CUDA CCSDT configuration
  (every fused path; `labelled=False` without orbital labels), used by the
  guide, llms.txt and benchmarks/ccsdt_backends.py instead of hand-written
  flag sets (the guide's set lacked the column paths).
- Measured on water (12 threads, same desktop, before -> after): CCSDTQ
  aug-cc-pVDZ 2.07 -> 1.75 s per iteration, EOM-CCSDT aug-cc-pVDZ 6.45 ->
  5.76 s, CCSD response cc-pVTZ 4.00 -> 3.42 s, CCSDT response cc-pVDZ 5.28
  -> 4.81 s, CCSD aug-cc-pVTZ and CCSDT cc-pVTZ iterations 3-6% faster; a
  triplet EOM-CCSDTQ action (cc-pVDZ) 3.10 -> 2.98 s. Energies agree to
  1e-13 with the same iteration counts.
- docs/benchmarks.md, its charts, README and llms.txt rerun for IterCC on one
  Xeon 8480CL socket (whole node held, 56 threads) after the review: most
  iterations 2-8% faster (CCSD aug-cc-pVQZ 59 -> 57 ms, CCSDT aug-cc-pVTZ
  0.139 -> 0.130 s, CCSDTQ aug-cc-pVDZ 0.496 -> 0.484 s), EOM-CCSDT
  aug-cc-pVDZ 5.2 -> 4.8 s and cc-pVTZ 13.3 -> 12.4 s; the other programs'
  records are unchanged.

- Triplet slot program, propagated: progress callbacks (`slot_finished`, as
  the CuF drivers print them) arrive after each wave of slots again instead
  of once at the end (`itercc_triplet_slots` takes a progress function, core
  ABI 32); the engine's memory estimate counts the program's per-thread
  tiles, dense direct-term buffers and partial sums; docs/guide.md and
  llms.txt describe the compiled path, its switches and why `occupied_slab`
  should stay at its default with packed T4.
- Triplet EOM-CCSDTQ rank-four slots run as one compiled program on the CPU
  (`itercc_triplet_slots`, core ABI 28): water cc-pVDZ on 12 threads takes
  4.0-4.3 s per action instead of 26.4 s (6-31G: 0.47 s instead of 8.3 s).
  Tile kinds and slot classes are recorded once (the first action takes
  5.8 s); every tile then runs on one thread with the single-threaded
  contraction routes, intermediates are shared between tiles through a cache
  with one producer per key, the products into a tile with the same output
  blocks run as fused GEMMs, and each slot adds its tiles' partial sums in the
  loop's order. The action does not depend on the thread count; it agrees
  with the Python slot loop to round-off, and bit for bit with
  `ITERCC_TRIPLET_FUSED=0`. `ITERCC_TRIPLET_PROGRAM=0` runs the loop. Tile
  contraction trees now count an intermediate's work once per tiles that
  share it: goovv t2 t4 contracts g with T4 first (one intermediate per T4
  hole pair instead of a packed-T4 product per tile), which changes the
  action at round-off. Without `intermediate_cache_gib`, the program's cache
  takes half of what the memory budget leaves free (at least 256 MiB).
  On a 16-slot CuF sample (56 cores, 8 GiB cache) the fused program takes
  20.8 s against 43.3 s for the loop; the loop itself is slower there than
  with the old trees (35.5 s), as the g-T4 intermediates crowd its cache.
  Packed T4 read in place keeps its summed occupied letters whole (at CuF
  size the tile byte budget had cut them into 28 slabs, each its own g-T4
  intermediate and product per tile); `occupied_slab` still sets slabs. The
  program runs a wave's tiles sorted by occupied tuple, T4 hole pair first,
  sizes waves by the intermediate cache budget and no longer clears the
  partial-sum buffer.
  On the 16-slot CuF sample this takes the fused program from 20.8 s to
  10.7 s and the loop from 43.3 s to 25.9 s.
- Direct packed spin-orbital terms (`PackedBinaryAction.add`) run all their
  terms in one pass over the rows (`itercc_spin_quad_contract_terms`, core
  ABI 29) instead of one parallel pass per diagram, derivative and
  permutation (144 at rank four), with the same results bit for bit: water
  cc-pVDZ triplet actions take 3.9 s instead of 4.3 s.
- Large intermediates of the triplet slot program (at least 2^22
  multiply-adds, such as the g-T4 products at CuF size) are computed in the
  pieces of the threaded executor, and the tiles waiting for one run its
  pieces with its producer instead of sleeping (core ABI 30). The pieces
  write disjoint entries with the single-threaded arithmetic, so the action
  is unchanged bit for bit; `ITERCC_SLOT_SHARE_WORK` sets the threshold.
  On the first 400 CuF slots (56 cores) this takes 145.5 s instead of
  177.6 s with an 8 GiB cache and 128.5 s instead of 134.3 s with 32 GiB;
  the 16-slot sample was slower there (12.6 s instead of 10.7 s), but so
  were the phases that sharing does not touch, by about the same 16%, so
  most of that difference was the allocation. A producer whose last pieces
  are with helpers now runs pieces of other intermediates being produced
  instead of waiting, and so does a tile whose intermediate has no pieces
  left. In one allocation, sharing is neutral on the 16-slot sample
  (10.3 s either way) and takes the first 400 slots from 151.8 s to 136.4 s.
- Direct packed T4 terms that read the same T4 entries in the same pattern
  share one coefficient table (`merged_diagrams`: at rank four 10 diagrams
  in 5 tables, such as gvovo T4 with (goovv t2) T4 and gvvvv T4 with
  (goovv t2) T4), so each entry is looked up once instead of once per
  diagram, and the ground-state tables are kept across actions. The action
  changes at round-off (1.9e-15 relative on water cc-pVDZ, which takes 3.7 s
  instead of 3.95 s). On the first 400 CuF slots the direct terms take 25.4 s
  instead of 41.6 s (121.2 s instead of 135.1 s in all).
- Rank-four direct packed T4 terms run by contexts
  (`itercc_spin_quad_dense`, core ABI 31): the T4 indices of a row and
  permutation that are not summed form a context; each context gathers its
  T4 values once, only over the summed tuples of the one symmetry class that
  can complete it to a stored amplitude, and every row and permutation using
  it takes a dot product with a dense block of the coefficient. The sparse
  per-row lookups mostly read amplitudes that cannot exist. The same nonzero
  products are added in the same order, so the action is the sparse path's
  bit for bit; `ITERCC_DIRECT_DENSE=0` keeps the sparse path. The slot
  program passes the direct terms 2^22 rows at a time (the kernel runs a
  call's slots in parallel). On the first 400 CuF slots (56 cores) they take
  5.4 s instead of 24.2 s, and the whole probe 99.0 s instead of 121.2 s.
- Benchmarks: IterCC re-timed on one 56-core cluster socket (node held
  whole, 56 threads at most). Three cc-pVTZ EOM-CCSDT roots take 13.3 s
  instead of 15.1 s, CCSD polarizabilities in cc-pVTZ 7.3 s instead of
  8.3 s, and aug-cc-pVQZ CCSD from RHF 2.6 s and 15 GiB instead of 4.3 s
  and 21 GiB (faster AO-to-MO transform); times per iteration are unchanged
  within noise. docs/benchmarks.md no longer quotes 112-thread runs.
- Faster triplet EOM-CCSDTQ actions on the CPU (packed spin, occupied
  tiles): water cc-pVDZ on 12 threads takes 26.5 s per action instead of
  47 s, with the same action bit for bit. A 16-slot CuF quadruples sample
  on 56 cores takes 35.5 s instead of 63.2 s; see docs/triplet-cpu-audit.md.
  Automatic occupied slab width is now the fastest setting on both systems.
  The blocked GEMMs now read packed T4 in place. A T4 operand is passed as
  `SignedRows` (the signed rows of `signed_gather_rows`, described by the
  shape and strides of the dense tile), and `itercc_sym_contract_packed`
  gathers each piece of B a route needs into its own scratch: GEMM panels of
  about 512 KB, skinny chunks, and the small and shallow blocks. The dense
  tile is never written; before, one action wrote and read back about
  112 GiB of it, half of it zero rows, from 15 GiB of distinct rows.
  Addresses split into row and entry once per sum and once per column, so no
  entry needs a division. Core ABI 27.
  A summed occupied axis of a T4 slab is now as wide as a quarter of the tile
  cache allows, not two slots, so one deeper product replaces several that
  each read and write the whole output tile. Dense expansion (GPU, fallbacks)
  works in fixed column chunks, and the rows of a repeated hole are a plain
  zero fill. The blocked engine keeps each plan's kernel addresses and keys
  per-tile labels by identity. An action also resolves its thread count once
  instead of on every kernel call. Packed-row descriptors come from hole
  lookup tables in a few array operations and are reused within an action
  (3.8 s of Python in a 35 s CuF sample). The direct packed terms of
  consecutive slots run in one call, and each slot reuses one raw tile
  buffer. Results are unchanged bit for bit.
  What remains is about 16 s of small GEMMs, most of them shallow ones
  (summing over one occupied index) that are limited by writing their output
  tile, plus about 10 s of Python scheduling the tiles.
- A calculation can now run without PySCF: `itercc.molecule.Molecule`
  (geometry; NWChem-format basis sets from `ITERCC_BASIS_PATH`, the bundled
  `itercc/basis_sets` (STO-3G, Pople 3-21G to 6-311G**, cc-pVXZ and
  aug-cc-pVXZ up to 5Z, def2-SVP to QZVPP; about 2 MB, from PySCF's Basis Set
  Exchange files, see THIRD_PARTY.md) or an installed PySCF's data files;
  elements that need an effective core potential are refused; Abelian point
  group and symmetry-adapted basis from `itercc.symmetry`), the package's own Gaussian integrals
  (`itercc.integrals`: overlap, kinetic, nuclear attraction, dipole and
  8-fold packed ERIs by McMurchie-Davidson in the core, libcint's
  conventions, equal to libcint to 1e-13 through g functions), native
  closed-shell RHF (`itercc.scf.rhf` and `run`: J/K from the core, DIIS, one
  eigenproblem per irrep) and `from_scf` for the solvers. Water RHF agrees
  with PySCF to 1e-12 Eh and CCSD/CCSDT to their convergence. The integrals
  are still 4-5x slower than libcint (cc-pVQZ: 0.39 instead of 0.09 s).
- IterCC transforms the integrals itself: `itercc.integrals.mo_eri` turns
  AO two-electron integrals (8-fold or 4-fold packed, or full) into the
  (pq|rs) the solvers take, as two half transformations of large GEMMs on
  the core's executor (`itercc_unpack_s8_s4`, `itercc_unpack_pairs`,
  `itercc_pack_pairs`, `itercc_scatter_pairs`), written once and with the
  symmetry-forbidden entries exact zeros. `from_rhf` uses it for in-core AO
  integrals instead of PySCF's ao2mo and the unpacking (water aug-cc-pVQZ on
  56 cores: 0.84 instead of 2.0 s; `ITERCC_NATIVE_TRANSFORM=0` restores
  ao2mo). Integrals without a nonzero forbidden entry pass the solvers'
  symmetry projection without a copy, and the solvers check the integrals
  for nonfinite entries in one compiled pass (NumPy's single-threaded scan
  was 0.6 s of the 0.83 s setup for water aug-cc-pVQZ). Core ABI 24.
- Symmetry-blocked contractions below 64k multiply-adds run on one thread,
  and none is cut into tasks of fewer than 8k: EOM and response actions make
  thousands of small ones, which a 56-thread team made slower than 14 threads
  (each task repacks its operands). Water on 56 threads: EOM-CCSDT
  aug-cc-pVDZ 4.7-5.0 instead of 5.8-7.0 s, CCSD polarizability cc-pVTZ
  2.6-2.7 instead of 4.3-4.8 s. One thread or the whole team, never a smaller
  team, which would make libgomp end and restart threads
  (`ITERCC_SYM_PARALLEL_WORK`, `ITERCC_SYM_MIN_SHARE` override the limits).
  Same results.
- The open CCSDT dressing no longer reads g(vvvv) at all: the products of its
  planes with T1 that the occupied rows and columns take come from the bare
  ladder sector blocks (`LadderSectors.contract_t1`,
  `itercc_sectors_contract_t1`). Core ABI 23.
- Cluster benchmark scripts request at most 56 cores (one socket of the
  nodes used) and no longer run on whole nodes; the 112-thread points are
  gone from the thread-scaling records.
- CCSDT no longer dresses the all-virtual block of the integrals. On the
  symmetry-blocked path g~(vvvv) enters only through the sector blocks of the
  ladders, which now come from those of the bare g(vvvv), built once, and
  the rank-o T1 terms (`LadderSectors.dressed`, `itercc_ladder_sectors_t1`);
  the dressing (`itercc_dress_integrals_open`) writes every other entry with
  the same values as before and reads g(vvvv) once. The T3 ladder matrix
  takes the sector blocks over instead of copying them. Water on one 56-core
  socket: aug-cc-pVQZ 0.99 instead of 1.25 s per iteration (dressing 89
  instead of 245 ms, sectors 23 instead of 45 ms; 24x from 56 threads instead
  of 22x), cc-pVQZ 0.24 instead of 0.29 s. `ITERCC_OPEN_DRESSING=0` restores
  the full dressing. Core ABI 19.
- Iterations no longer fault their memory in anew (Linux, glibc). glibc
  maps large blocks on their own and unmaps them when freed, and returns
  freed heap to the kernel at thresholds it moves at run time, so every
  iteration's temporaries came back zero-filled through page faults, in
  some processes and not others. IterCC now fixes the thresholds (blocks up
  to 32 MiB from the heap, up to 1 GiB of free heap kept), takes every block
  from the heap while a CCSDT solve runs and gives the free pages back when
  it ends (`malloc_trim`), and CCSDT keeps one array of dressed integrals
  across iterations (6.8 GB per iteration in water aug-cc-pVQZ). Water CCSDT
  on one 56-core socket: cc-pVQZ 0.29 instead of 0.36 s per iteration,
  aug-cc-pVTZ 0.15 instead of 0.17 s; without symmetry in aug-cc-pVDZ, 23 ms
  in every run instead of 24 or 35 ms. Same peak memory with symmetry. CCSD,
  EOM and CCSDTQ solves keep glibc's mapping of large blocks (no gain, and
  slower or larger with the heap alone). `ITERCC_MALLOC=0`, or glibc's
  `MALLOC_MMAP_THRESHOLD_`, `MALLOC_TRIM_THRESHOLD_` or `MALLOC_MMAP_MAX_`,
  leaves the allocator alone.
- A process that finds the core sources and compiler unchanged loads the
  runtime build without hashing the sources and running the compiler
  (about 80 ms off every small calculation).
- The CCSD dressing runs plane by plane on more than three threads per
  occupied orbital: slab by slab gave only about two per occupied orbital
  work (water aug-cc-pVQZ on one 56-core socket: 19 instead of 35 ms for the
  dressing, 58 instead of 78 ms per iteration). Same operations in the same
  order, so the same results; `ITERCC_DRESS_SLABS=1` or `0` forces either.
- CCSD polarizabilities about 2x faster and EOM-CCSDT and CCSDT response
  actions about 30% faster (water, 12 cores: CCSD cc-pVTZ static and 532 nm
  4.3 instead of 9.4 s; EOM-CCSDT aug-cc-pVDZ, three roots, 7.4 instead of
  10.5 s). Jacobian actions take the particle-particle ladder from the
  undressed g(vvvv) through its derivative (`BareLadder.tangent`, exact for
  directions of any irrep) instead of three dense products with the dressed
  g(vvvv) and its derivative; for CCSD the derivative of the dressed
  integrals skips the all-virtual planes (`itercc_dress_derivative_ccsd`,
  into a buffer kept between actions). From 30 virtual orbitals: below, the
  dense ladder is faster (`ITERCC_BARE_LADDER=0` or `1` forces either). The
  sector scan of the dense-slot triples no longer starts a thread pool per
  call. Core ABI 18.
- The pair-sector ladders of CCSD and of EOM and response actions multiply
  large blocks by one column per occupied pair; with up to 16 columns they
  now stream through `itercc_gemm_narrow` (fixed summation order) at the
  memory bandwidth instead of a fifth of it (water cc-pVQZ CCSD, 12 cores:
  62 instead of 94 ms per iteration).
- CCSDTQ plans its symmetry-blocked T4 products 1.8x faster (water 6-31G*:
  2.3 instead of 3.2 s for the whole calculation; the plans are unchanged).
- CCSD up to 5x faster per iteration (water aug-cc-pVTZ, 12 cores: 33 ms
  instead of 0.16 s; cc-pVTZ 9 instead of 24 ms). The T1-dressed integrals
  are no longer formed in their all-virtual block: the CCSD dressing
  (`itercc_dress_integrals_ccsd`) never
  reads g(vvvv) and writes only the entries the residual uses, into one
  array reused across iterations, and the particle-particle ladder takes the
  undressed g(vvvv) through pair sectors built once, with the T1 terms as
  o^3 v^3 corrections (`BareLadder`). Energies are unchanged to 1e-12 Eh.
- EOM and response actions form the T1 derivative of the dressed integrals
  in one pass without a zero fill (`itercc_dress_derivative`; 2x faster for
  that step; a water cc-pVTZ CCSD polarizability, static and 532 nm, 10.4
  instead of 12.6 s).
- `from_rhf` no longer imports all of PySCF: its density-fitting check read a
  missing attribute through PySCF's lazy `__getattr__` (0.25 s, most of a
  small molecule's first calculation).
- Setup 8-10x faster: the symmetry projection of the integrals runs in one
  pass on all threads (`itercc_project_symmetric`; NumPy's masked copies moved
  the n^4 tensor several times on one thread), and `from_rhf` unpacks PySCF's
  packed MO integrals on all threads (`itercc_unpack_s4`). Water aug-cc-pVTZ:
  `RestrictedCC(...)` 0.10 instead of 0.8-1.0 s. Core ABI 17.
- CCSDT a third to three quarters faster per iteration on many cores (water,
  12 cores: 0.71 instead of 1.17 s in aug-cc-pVTZ, 0.12 instead of 0.18 s in
  cc-pVTZ, 5 instead of 19 ms in 6-31G, where symmetry no longer costs). The
  doubles and triples ladders build their symmetry-sector blocks straight from
  the dressed integrals in one compiled pass (`itercc_ladder_sectors`), with
  the T2 term as small per-sector GEMMs: no copies of g(vvvv), no dense pair
  matrices. The T1 dressing takes two passes over the integrals instead of
  five (`itercc_dress_integrals`). Core ABI 16.
- CCSDTQ on the CPU puts T4 into DIIS by default (`diis_quadruples="auto"`):
  diffuse basis sets converge without settings (water aug-cc-pVDZ: 15
  iterations; before, damped Jacobi on T4 had not converged after 200).
  `diis_quadruples=False` restores the previous behaviour.
- The host T4 DIIS, the Davidson and the GMRES vector products use the core's
  fixed-order sums instead of NumPy's BLAS: OpenBLAS threads left spinning
  slowed the following kernels (small CCSDTQ iterations 2.6x on 12 threads,
  a response sweep about 15%).
- Benchmarks: a one-core comparison that adds ORCA 6.1.1 (AUTOCI CCSDT and
  CCSDTQ; `compare.py --code orca`, `benchmarks/compare_single_core.sh`).
- Benchmarks rerun on one 56-core cluster CPU socket per calculation, up to
  aug-cc-pVQZ, with CCSD (adding Psi4 1.11), CCSD and CCSDT polarizabilities,
  NWChem CCSDTQ and thread scaling to 112 threads
  (`benchmarks/cluster_campaign.sh` with `cluster_runs.txt`;
  `build_nwchem.sh` builds NWChem 7.3.1 with CCSDTQ and CCSDT response;
  `NWCHEM_MPIRUN` sets NWChem's launcher). The desktop records stay in
  `benchmarks/results/`.

## 0.8.0

- `from_rhf` returns the Abelian `orbital_irreps` of the correlated orbitals
  (linear molecules: their D2h or C2v subgroup), and
  `pyscf_interface.dipole_integrals(model, mol)` gives the dipole matrices and
  their irreps, so a PySCF calculation feeds every solver without helper code.
- Documentation rewritten: a short README, one user guide, a design page, and
  `llms.txt`, a complete usage guide for AI assistants.
- `benchmarks/compare.py`: timings against PySCF, ccpy and NWChem on the same
  machine, with the figures and tables of `docs/benchmarks.md`.

## 0.7.0

First standalone release of the IterCC package.

- Compiled CPU core (`core/`, C++17/OpenMP, own GEMM kernels) behind a
  versioned C ABI: built on first use from the sources an installation
  carries, or ahead of time per instruction-set level (`core/build_levels.sh`);
  a native build is skipped on CPUs lacking an extension it was compiled for.
- CCSD and CCSDT on symmetry-blocked T3, including the T3 feedback products
  (CCSDT with 131 virtuals: 11.8 -> 9.2 s per evaluation on one Sapphire
  Rapids socket).
- CCSDTQ on the CPU: packed T4, compiled tile programs reading pair-blocked
  T3 by default, concurrent thread teams in the T4 passes, compact float32
  host T4 DIIS history, symmetry-blocked T3-level products.
- EOM-CCSDT and response actions on compiled product-rule tangents with
  symmetry-blocked feedback; Davidson iterations form each Ritz vector and
  residual once.
- CUDA: saved CCSDTQ states carry the autotuned GEMM routes (bit-exact
  resumes in a new process); host-staged R4 through page-locked buffers;
  the T3 denominator is formed on the fly.
- DIIS falls back to its newest entries when the history is nearly
  dependent; reductions use fixed blocks, so CPU results do not depend on the
  thread count.
