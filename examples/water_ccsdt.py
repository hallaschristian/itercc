"""Water: full CCSDT ground state, singlet EOM-CCSDT roots in two C2v sectors,
and the static and 532 nm CCSDT electronic polarizability, on the CPU.

With orbital labels the integrals are projected onto their exact C2v blocks
(`symmetry_tolerance`), so every product of the solver, the Jacobian and the
response runs on symmetry-blocked kernels. Requires PySCF.

    python examples/water_ccsdt.py [basis]
"""
import sys
import time

import numpy as np

from _molecule import prepare
from itercc import DipoleResponse, EOMSinglet, RestrictedCC

basis = sys.argv[1] if len(sys.argv) > 1 else "cc-pvdz"
water = """O  0.000  0.000  0.117
           H  0.000  0.757 -0.469
           H  0.000 -0.757 -0.469"""
m = prepare(water, basis, frozen=1)
print(f"{basis}: {m['nocc']} correlated occupied, {len(m['h']) - m['nocc']} virtual orbitals; "
      f"RHF {m['scf_energy']:.10f} Eh")

start = time.perf_counter()
cc = RestrictedCC(m["h"], m["eri"], m["nocc"], m["constant"], orbital_irreps=m["labels"],
                  symmetry_tolerance=1e-8)
ground = cc.solve(energy_tol=1e-12, residual_tol=1e-10, diis_size=8, max_iter=200)  # EOM audits 1e-10
assert ground.converged
print(f"CCSDT energy {ground.energy:.10f} Eh "
      f"({len(ground.iterations)} iterations, {time.perf_counter() - start:.1f} s)")

for irrep in (0, 3):                # two Abelian sectors (C2v: A1 and B2 in PySCF's order)
    start = time.perf_counter()
    eom = EOMSinglet(cc, ground.amplitudes, orbital_irreps=m["labels"], irrep=irrep)
    states = eom.solve(nroots=2, residual_tol=1e-6, max_space=12)
    assert states.converged
    energies = ", ".join(f"{e:.6f}" for e in states.excitation_energies)
    print(f"EOM-CCSDT irrep {irrep}: {energies} Eh ({time.perf_counter() - start:.1f} s)")

start = time.perf_counter()
response = DipoleResponse(cc, ground.amplitudes, m["dipoles"], orbital_irreps=m["labels"],
                          dipole_irreps=m["dipole_irreps"])
for wavelength in (float("inf"), 532.0):
    tensor = response.polarizability(wavelength, tol=1e-6).tensor_au.real
    print(f"alpha({wavelength:g} nm) diagonal {np.round(np.diag(tensor), 4)} au, "
          f"isotropic {np.trace(tensor) / 3:.4f} au")
print(f"response {time.perf_counter() - start:.1f} s")
