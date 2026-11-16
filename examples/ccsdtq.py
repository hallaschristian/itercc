"""Water in a small basis: CCSDT, then full CCSDTQ started from the CCSDT
amplitudes, on the CPU (compiled tile programs, symmetry-blocked T4).
Pass --cuda to run CCSDTQ on a GPU with CuPy instead. Requires PySCF.

    python examples/ccsdtq.py [basis] [--cuda]
"""
import sys
import time

from _molecule import prepare
from itercc import RestrictedCC, RestrictedCCSDTQ

arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
basis = arguments[0] if arguments else "6-31g"
backend = "cuda" if "--cuda" in sys.argv else "cpu"
water = """O  0.000  0.000  0.117
           H  0.000  0.757 -0.469
           H  0.000 -0.757 -0.469"""
m = prepare(water, basis, frozen=1)
print(f"{basis}: {m['nocc']} correlated occupied, {len(m['h']) - m['nocc']} virtual orbitals")

cc = RestrictedCC(m["h"], m["eri"], m["nocc"], m["constant"], orbital_irreps=m["labels"],
                  symmetry_tolerance=1e-8)
ccsdt = cc.solve(energy_tol=1e-10, residual_tol=1e-8, diis_size=8, max_iter=200)
assert ccsdt.converged
print(f"CCSDT  {ccsdt.energy:.10f} Eh")

start = time.perf_counter()
quadruples = RestrictedCCSDTQ(m["h"], m["eri"], m["nocc"], m["constant"],
                              orbital_irreps=m["labels"], backend=backend)
t1, t2, t3 = ccsdt.amplitudes
result = quadruples.solve(initial=(t1, t2, quadruples.expand_triples(t3)), energy_tol=1e-10,
                          residual_tol=1e-7, diis_size=8, damping=0.2, max_iter=200)
assert result.converged
print(f"CCSDTQ {result.energy:.10f} Eh ({len(result.iterations)} evaluations, "
      f"{time.perf_counter() - start:.1f} s on {backend}); "
      f"quadruples correction {result.energy - ccsdt.energy:+.3e} Eh")
