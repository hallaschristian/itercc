# Changelog

## Unreleased

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
