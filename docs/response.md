# Wavelength-dependent polarizability (CCSDT linear response)

`DipoleResponse` evaluates the clamped-nuclei, fixed-orbital electronic
dipole polarizability with full linear-response CCSDT, or CCSD with
`method="ccsd"` and a matching CCSD ground state. The input dipoles are three
real symmetric length-gauge MO matrices of the electronic dipole operator
(-r for electrons) in exactly the correlated orbital frame of the supplied
ground-state amplitudes. The field enters as H(F) = H(0) - sum_i F_i dipole_i.
The output tensor is in atomic units. For scalar-X2C calculations supply the
picture-changed electronic dipoles.

```python
from itercc import DipoleResponse

engine = DipoleResponse(cc, ground.amplitudes, dipoles,
                        orbital_irreps=labels, dipole_irreps=(2, 3, 0))
static = engine.polarizability(float("inf"))
infrared = engine.polarizability(1064.0)
damped = engine.polarizability(532.0, damping_Eh=0.001)
print(infrared.tensor_au)
```

`dipole_irreps` gives the Abelian irrep of each Cartesian component (needed
with orbital labels; `examples/_molecule.py` determines them from the
matrices). The frequency is hc divided by the wavelength. A positive damping
parameter sets the imaginary part of the frequency in hartree and yields a
complex tensor; an infinite wavelength and zero damping give the static
limit. A shifted solve that cannot meet its requested relative residual
raises an exception. `backend="cuda"` keeps the contractions and Krylov
vectors on the device; NumPy solves only the small projected least-squares
problems.

## Method

The two shifted right-hand systems use the same full CCSDT Jacobian as EOM.
The CC Lagrangian includes derivatives of the energy and residual. The
stationary multiplier satisfies lambda A = -E_T, so an additional zero-shift
solve evaluates each contraction lambda G as -E_T A^-1 G. This retains the
CC Lagrangian contribution without constructing a transpose or summing over
excited states. The mixed residual derivatives use centered polynomial
formulas: the one-body BCH residual is quadratic and the two-body BCH
residual quartic in the amplitudes, so the formulas are exact for those
degrees apart from floating-point error. For two directions of the same
non-totally-symmetric irrep with projected operators, the symmetry-allowed
part of the shifted tangent is odd in the step, and a two-point formula is
exact (half the shifted tangents).

The tensor convention and stationary CC Lagrangian follow
[Yuan et al., frequency-dependent relativistic CC response](https://arxiv.org/abs/2307.14296).

## Performance

With orbital labels, the Jacobian acts with the symmetry-projected integrals
and ground amplitudes when their leakage stays within `symmetry_tolerance`
(default 2e-9; see `EOMSinglet`). Rounding-level leakage otherwise makes
every symmetry block check fail and every product run dense. The totally
symmetric dipole and every property solve then run the triples part of each
action on symmetry-blocked T3 (13 occupied, 29 virtual orbitals: 0.88 s per
action instead of 2.24 s); the other dipoles run on blocked T3 in shifted
occupied labels and blocked GEMMs (1.39 s).

Every solve of one symmetry sector uses the same Jacobian A: the +omega and
-omega dipole solves of a wavelength, the zero-shift property solves, and
the solves of every other wavelength differ only in the shift and the
right-hand side. The engine therefore keeps every converged solution x of
(A - shift) x = b per sector with its image A x = b + shift x, which the
solved system supplies without an action (`krylov.RecycledImages`). Each
solve starts from the least-squares solution of its system in the span of
all earlier solutions: along a sweep this extrapolates in frequency, and the
-omega solve of a wavelength builds on its +omega solution.
`DipoleResponse(..., recycle_gib=None)` gives the stores half of the host
memory left after the engine and one solve's Krylov basis on the CPU (split
over the solved sectors, at most 64 solutions each) and none on CUDA; set it
explicitly on CUDA, where the vectors live in device memory, or to 0 to
start each system from the previous wavelength's solution only. On the
13-occupied, 29-virtual system (tolerance 1e-6) a sweep of the static point,
1064, 780 and 650 nm took 107, 148, 98 and 93 Jacobian actions; from the
third wavelength on, the -omega solves start below 1e-4 and take 9 to 11
actions.

## Validation

Tests compare the full Cartesian tensor against exact determinant
diagonalization in a two-electron problem at zero and finite frequencies,
including a damped point near a pole. They also compare a four-electron
full-CCSDT static component against a fixed-orbital finite-field derivative
and verify the one-body residual against the general CCSDT equations.

A finite-field check must use the same zero-field orbital frame and apply
the dipole directly to the one-electron integrals: reoptimizing the orbitals
in each field changes the finite-field quantity in a truncated CC model.
