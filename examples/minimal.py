"""The smallest complete IterCC run: CCSDT, one singlet EOM root and the static
polarizability of a two-orbital model Hamiltonian (NumPy only)."""
import numpy as np

from itercc import DipoleResponse, EOMSinglet, RestrictedCC

h = np.diag([-1.0, 0.5])            # one occupied and one virtual spatial orbital
eri = np.zeros((2, 2, 2, 2))        # chemists' notation (pq|rs)
cc = RestrictedCC(h, eri, nocc=1)
ground = cc.solve()
assert ground.converged

eom = EOMSinglet(cc, ground.amplitudes)
roots = eom.solve(nroots=1, max_space=3)
assert roots.converged

dipoles = np.zeros((3, 2, 2))       # electronic dipole operator, MO basis
dipoles[0, 0, 1] = dipoles[0, 1, 0] = 1.0
response = DipoleResponse(cc, ground.amplitudes, dipoles)
static = response.polarizability(float("inf"))

print(f"CCSDT energy       {ground.energy:.10f} Eh")
print(f"excitation energy  {roots.excitation_energies[0]:.10f} Eh")
print(f"alpha_xx (static)  {static.tensor_au[0, 0].real:.6f} au")
