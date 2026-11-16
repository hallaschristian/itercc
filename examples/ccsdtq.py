"""Water: CCSDT, then full CCSDTQ started from the CCSDT amplitudes, on the CPU
(or a GPU with --cuda). Requires PySCF.

    python examples/ccsdtq.py [basis] [--cuda]
"""
import sys
import time

from pyscf import gto, scf

from itercc import RestrictedCC, RestrictedCCSDTQ
from itercc.pyscf_interface import from_rhf

arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
basis = arguments[0] if arguments else "6-31g"
backend = "cuda" if "--cuda" in sys.argv else "cpu"
mol = gto.M(atom="O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469", basis=basis,
            symmetry=True, verbose=0)
model = from_rhf(scf.RHF(mol).run(conv_tol=1e-12), frozen=1)
print(f"{basis}: {model.nocc} correlated occupied, {len(model.h) - model.nocc} virtual orbitals")

cc = RestrictedCC(model.h, model.eri, model.nocc, model.constant,
                  orbital_irreps=model.orbital_irreps, symmetry_tolerance=1e-8)
ccsdt = cc.solve(energy_tol=1e-10, residual_tol=1e-8)
assert ccsdt.converged
print(f"CCSDT  {ccsdt.energy:.10f} Eh")

start = time.perf_counter()
solver = RestrictedCCSDTQ(model.h, model.eri, model.nocc, model.constant,
                          orbital_irreps=model.orbital_irreps, backend=backend)
t1, t2, t3 = ccsdt.amplitudes
result = solver.solve(initial=(t1, t2, solver.expand_triples(t3)),     # T4 starts at zero
                      energy_tol=1e-10, residual_tol=1e-7, diis_size=8, damping=0.2)
assert result.converged
print(f"CCSDTQ {result.energy:.10f} Eh ({len(result.iterations)} iterations, "
      f"{time.perf_counter() - start:.1f} s on {backend}); "
      f"quadruples contribution {result.energy - ccsdt.energy:+.3e} Eh")
