# Spin-orbit state interaction

The state-interaction layer diagonalizes a scalar singlet and triplet model
space with a first-order one-electron spin-orbit coupling (SOC) operator. The
SOC term can include a separately constructed spin-orbit mean-field
contribution. Use a common orthonormal orbital frame and a common energy zero
for all states. Include both components of each scalar Pi or Delta state,
along with any nearby multiplets relevant to the desired spectrum.

```python
from itercc import spin_orbit_state_interaction
from itercc.pyscf_interface import x2c_spin_orbit

pauli = x2c_spin_orbit(scalar_x2c_rhf, mo_coeff=correlated_mo_coeff)
result = spin_orbit_state_interaction(
    singlet_energies, triplet_energies, pauli, density_st, density_ts, density_tt
)
```

The optional PySCF adapter copies the scalar X2C approximation, recontraction
and auxiliary basis settings into the spin-orbital helper. It returns the
spin-traceless part of the resulting one-electron Hamiltonian, preserving the
original scalar X2C energies. With `from_rhf`, pass the returned `mo_coeff`
to match the frozen-core and correlated orbital ordering. The adapter
supplies bare one-electron SOC; two-electron SOC needs a separate effective
operator.

## Conventions

The density tensors store D_alpha[p,q] - D_beta[p,q] with
D[p,q] = <bra|a_q^dagger a_p|ket>. Axis order is (bra, ket, p, q). Supply
both singlet-triplet directions at M=0 and triplet-triplet densities at
M=+1. Choose the ladder phase |1,+1> = S+ |1,0> / sqrt(2). The Pauli
components multiply sigma_x, sigma_y and sigma_z without an additional
factor of 1/2. The output basis labels are (S, root index within S, M), and
the eigenvectors' columns are the SOC states in that basis.

`spin_orbit_from_components` accepts the Cartesian matrix elements directly
as arrays of shape (bra, ket, 3). This also supports implementations that
contract the similarity-transformed SOC operator with the left and right
states without materializing a transition density for each root pair.

EOM-CC inputs require consistently normalized biorthogonal left and right
roots. The two directions can differ before Hermitization: the
implementation averages the projected operator with its adjoint and reports
the maximum element of the difference before averaging. See
[Epifanovsky et al.](https://doi.org/10.1063/1.4927785) and
[Pokhilko et al.](https://doi.org/10.1063/1.5108762) for the EOM-CC SOC and
spinless density conventions. IterCC's native EOM solver provides right
singlet roots; left states and triplet transition densities come from the
caller.

## Implementation

Density contractions use tiled BLAS and reconstruct the magnetic components
in the small model space. Real densities use real BLAS. `backend="cuda"`
accepts device-resident densities and stages host arrays tile by tile; the
default CPU backend suits small state spaces. Both backends diagonalize the
small state-interaction matrix with CPU LAPACK.

The two-electron test checks the actual CCSD ground amplitudes, biorthogonal
EOM transition densities and all-M exact diagonalization. Convergence with
the number of retained spin-free roots is a separate check for a spectrum.
