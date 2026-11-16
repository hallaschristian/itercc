"""Water: CCSDT ground state, EOM-CCSDT excitation energies in two symmetry
sectors, and the static and 532 nm CCSDT polarizability, on the CPU with C2v
symmetry. Requires PySCF.

    python examples/water_ccsdt.py [basis]
"""
import sys
import time

import numpy as np
from pyscf import gto, scf

from itercc import DipoleResponse, EOMSinglet, RestrictedCC
from itercc.pyscf_interface import dipole_integrals, from_rhf

basis = sys.argv[1] if len(sys.argv) > 1 else "cc-pvdz"
mol = gto.M(atom="O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469", basis=basis,
            symmetry=True, verbose=0)
mf = scf.RHF(mol).run(conv_tol=1e-12)
model = from_rhf(mf, frozen=1)                  # freeze O 1s; MO integrals and C2v orbital labels
labels = model.orbital_irreps
print(f"{basis}: {model.nocc} correlated occupied, {len(model.h) - model.nocc} virtual orbitals")

start = time.perf_counter()
cc = RestrictedCC(model.h, model.eri, model.nocc, model.constant,
                  orbital_irreps=labels, symmetry_tolerance=1e-8)
ground = cc.solve(energy_tol=1e-12, residual_tol=1e-10, max_iter=200)   # EOM/response need 1e-10
assert ground.converged
print(f"CCSDT energy {ground.energy:.10f} Eh ({len(ground.iterations)} iterations, "
      f"{time.perf_counter() - start:.1f} s)")

for irrep in (2, 3):                            # B1, B2 (PySCF's C2v ids: A1 0, A2 1, B1 2, B2 3)
    start = time.perf_counter()
    eom = EOMSinglet(cc, ground.amplitudes, orbital_irreps=labels, irrep=irrep)
    states = eom.solve(nroots=2, residual_tol=1e-6)
    assert states.converged
    energies = ", ".join(f"{e * 27.211386:.3f}" for e in states.excitation_energies)
    print(f"EOM-CCSDT irrep {irrep}: {energies} eV ({time.perf_counter() - start:.1f} s)")

start = time.perf_counter()
dipoles, dipole_irreps = dipole_integrals(model, mol)
response = DipoleResponse(cc, ground.amplitudes, dipoles, orbital_irreps=labels,
                          dipole_irreps=dipole_irreps)
for wavelength in (float("inf"), 532.0):        # nm; inf is the static limit
    alpha = response.polarizability(wavelength, tol=1e-6).tensor_au.real
    print(f"alpha({wavelength:g} nm): diagonal {np.round(np.diag(alpha), 4)} au, "
          f"isotropic {np.trace(alpha) / 3:.4f} au")
print(f"response {time.perf_counter() - start:.1f} s")
