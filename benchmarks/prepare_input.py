"""Write a Hamiltonian archive (and optionally converged CCSDT amplitudes) for
the benchmark scripts, from a PySCF RHF calculation. Requires PySCF.

    python -m benchmarks.prepare_input "O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469" \\
        cc-pvtz water.npz --frozen 1 --ground water_ccsdt.npz

The archive holds h, eri (chemists' notation), nocc, constant and
orbital_irreps (Abelian XOR labels of D2h or a subgroup; all zero without
symmetry). `--ground` also solves CCSDT with the symmetry-projected
integrals and stores t1, t2 and t3 (IterCC's packed restricted T3).
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("atom", help="PySCF geometry string (Angstrom) or an .xyz file")
    parser.add_argument("basis")
    parser.add_argument("output", type=Path)
    parser.add_argument("--frozen", type=int, default=0, help="frozen core spatial orbitals")
    parser.add_argument("--no-symmetry", action="store_true")
    parser.add_argument("--ground", type=Path, help="also write converged CCSDT amplitudes here")
    args = parser.parse_args()

    from pyscf import gto, scf
    from pyscf.scf import hf_symm

    from itercc import RestrictedCC
    from itercc.pyscf_interface import from_rhf

    symmetry = not args.no_symmetry
    mol = gto.M(atom=args.atom, basis=args.basis, symmetry=symmetry, verbose=0)
    if symmetry and mol.groupname in ("Dooh", "Coov"):
        mol = gto.M(atom=args.atom, basis=args.basis, verbose=0,
                    symmetry="D2h" if mol.groupname == "Dooh" else "C2v")
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    model = from_rhf(mf, frozen=args.frozen)
    labels = (np.asarray(hf_symm.get_orbsym(mol, mf.mo_coeff), dtype=np.int64)[args.frozen:]
              if symmetry else np.zeros(model.mo_coeff.shape[1], dtype=np.int64))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, h=model.h, eri=model.eri, nocc=model.nocc, constant=model.constant,
             orbital_irreps=labels)
    print(f"{args.output}: {model.nocc} correlated occupied, {len(model.h) - model.nocc} virtual "
          f"orbitals, point group {mol.groupname}, RHF {mf.e_tot:.10f} Eh")
    if args.ground is not None:
        cc = RestrictedCC(model.h, model.eri, model.nocc, model.constant, orbital_irreps=labels,
                          symmetry_tolerance=1e-8)
        result = cc.solve(energy_tol=1e-12, residual_tol=1e-10, diis_size=8, max_iter=300)
        if not result.converged:
            raise SystemExit("CCSDT did not converge")
        t1, t2, t3 = result.amplitudes
        np.savez(args.ground, t1=t1, t2=t2, t3=t3)
        print(f"{args.ground}: CCSDT {result.energy:.10f} Eh, {len(result.iterations)} iterations")


if __name__ == "__main__":
    main()
