# Changelog

## Unreleased

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
