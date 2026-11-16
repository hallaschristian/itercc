# Singlet excited states (EOM-EE-CCSDT)

The EOM-EE-CCSDT ground cluster and response manifold both contain singles,
doubles and triples. Use `method="ccsd"` with a converged CCSD ground state
(`RestrictedCC(..., rank=2)`) for comparison. The same API runs on the CPU
and on CUDA.

## Usage

```python
from itercc import EOMSinglet, RestrictedCC

# h and chemists' eri describe the orthonormal correlated MO Hamiltonian;
# constant includes nuclear repulsion and the frozen-core energy.
cc = RestrictedCC(h, eri, nocc, constant, orbital_irreps=labels, symmetry_tolerance=1e-8)
ground = cc.solve(energy_tol=1e-12, residual_tol=1e-10, diis_size=8)
assert ground.converged

eom = EOMSinglet(cc, ground.amplitudes, orbital_irreps=labels, irrep=2)
states = eom.solve(nroots=2, residual_tol=1e-8, max_space=16)
assert states.converged
print(states.excitation_energies)   # hartree, relative to the CCSDT ground state
print(states.total_energies)        # ground energy plus excitation energy
```

On a GPU, pass `backend="cuda"` to `EOMSinglet` (Jacobian vectors stay on the
device) and run the ground state with the CUDA residual:

```python
cc = RestrictedCC(h, eri, nocc, constant, residual_backend="cuda",
                  cuda_layout_fusion=True, cuda_crossed_fusion=True,
                  cuda_pair_fusion=True, cuda_pool_cache=True)
ground = cc.solve(energy_tol=1e-12, residual_tol=3e-11, diis_size=6,
                  diis_triples=True, max_iter=250)
eom = EOMSinglet(cc, ground.amplitudes, backend="cuda", orbital_irreps=labels, irrep=2)
states = eom.solve(nroots=2, residual_tol=1e-8, max_space=16, memory_budget_gib=6.5)
```

`irrep` selects one Abelian sector (XOR labels of D2h or a subgroup; in
PySCF's C2v order A1, A2, B1, B2 are 0, 1, 2, 3). Ground-state residuals are
checked against the acting Hamiltonian before EOM (`ground_residual_tol`,
default 1e-10). The final eigenpair residual is evaluated afresh, including
components outside the selected sector, and a root is accepted only after
that full residual passes the requested threshold. Iteration limits return
`converged=False`; unconverged energies are not results.

With orbital labels, the integrals and ground amplitudes are audited for
symmetry leakage. When both stay within `symmetry_tolerance` (default 2e-9),
the Jacobian acts with their symmetry projections, so every operator is
exactly block-diagonal and the actions run on symmetry-blocked kernels. 0
keeps the original operators, whose leakage then enters every action.

## Architecture and performance

`_directional.py` applies exact algebraic product rules to the restricted
residual equations: no finite differences, fitted excitation energies or
perturbative triples. On the CPU the action is assembled from compiled
product-rule tangents (`_cpu_tangent.py`): the triples tangent and the T3
feedback run on symmetry-blocked T3, a non-A1 direction in occupied labels
shifted by its irrep, and the ground-state parts of each action are cached
per engine. `_directional.DirectionalCudaLayout` applies the linear CUDA
packing, spin-sum, crossed-update and triples assembly kernels separately to
primal and response arrays; the GPU caches the ground T1-dressed Hamiltonian
and differentiates its leg transforms.

`eom.SingletSpace` stores only allowed Abelian-sector elements. Every vector
is projected onto paired T2 symmetry and the physical spin-adapted T3
subspace. On CUDA with the projected tangent path (`packed_triples=True`,
the default) the triples direction and its image stay in these packed
coordinates during the action; `packed_triples=False` restores dense
storage. Nonzero target irreps are tested on CPU and GPU.

The restarted non-Hermitian Davidson solver keeps vectors and images on the
chosen device and diagonalizes only the small projected matrix on the CPU.
Complex intermediate Ritz pairs are handled through real/imaginary
correction directions; only effectively real converged roots are accepted.
On restart the retained Ritz vectors reuse their images, formed from the
stored images with the same coefficients, so no action is repeated. Each
iteration forms every root's Ritz vector and residual once: the residual
norm and, for a pending root, its correction come from the same pass, the
Ritz vector's norm comes from its coefficients (the basis is orthonormal),
and a correction, already projected and orthonormalized against the basis,
is appended without a second Gram-Schmidt pass. `guesses` and
`checkpoint_callback(info, roots)` support restarts; a driver should bind a
restart to the integrals, ground checkpoint, method, symmetry and solver
controls. A byte-based working-set estimate (`memory_budget_gib`) rejects an
oversized Davidson space before allocating its basis.

## Validation

- Analytic actions agree with central residual differences at interacting
  trial amplitudes, are linear in their directions, and agree CPU/CUDA.
- The entire physical singlet CCSDT Jacobian on an interacting four-electron
  problem agrees with direct determinant-space evaluation of the transformed
  Hamiltonian commutator; its first two roots are checked independently.
- Two-electron singlet total energies agree with exact diagonalization.
- EOM-CCSD singlet roots agree with PySCF on Be/6-31G.
- Full EOM-CCSDT on all-electron LiH/STO-3G agreed with NWChem 7.0 TCE to
  7.4e-12 hartree in the excitation energy (0.1329098748064 versus
  0.1329098748138).

## Scope

Implemented: neutral, spin-singlet vertical excitation and total energies.
Experimental: triplet (spin-flip) EOM-CCSDTQ (`itercc.TripletEOMCCSDTQ`) and
a finite-difference singlet EOM-CCSDTQ (`itercc.eom_ccsdtq.FiniteDifferenceEOM`). Not
implemented: left eigenvectors, transition moments, oscillator strengths and
gradients.

For linear molecules the Abelian labels alone do not establish a
Sigma/Pi/Delta assignment: partner degeneracies and an axial analysis are
separate checks. Along a scan, track roots by state overlaps; the energy
order alone is not a state identifier. Large-basis EOM needs more memory
than the ground state because of its Davidson vectors (`max_space` vectors
and as many images).
