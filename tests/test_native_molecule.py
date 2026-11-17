"""Fully native molecules: basis sets, symmetry, integrals and RHF against PySCF."""
import numpy as np
import pytest

from itercc import _cpu_kernels

pytestmark = pytest.mark.skipif(_cpu_kernels.library() is None, reason="compiled core unavailable")
pyscf = pytest.importorskip("pyscf")          # basis-set data and the reference

WATER = "O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469"
ETHYLENE = "C 0 0 0.667; C 0 0 -0.667; H 0 0.923 1.237; H 0 -0.923 1.237; H 0 0.923 -1.237; H 0 -0.923 -1.237"
HOCL = "O 0 0 0; H 0.96 0 0; Cl -0.4 1.6 0"            # Cs (plane xy)
CHFCLBR = "C 0 0 0; H 0.6 0.6 0.6; F -0.7 0.7 -0.6; Cl 0.8 -0.8 -0.7; Br -0.9 -0.9 0.9"


@pytest.mark.parametrize("atoms, basis, group", [(WATER, "cc-pvdz", "C2v"), (ETHYLENE, "6-31g*", "D2h"),
                                                  (HOCL, "sto-3g", "Cs")])
def test_symmetry_and_native_rhf(atoms, basis, group):
    from pyscf import gto, scf

    from itercc.molecule import Molecule
    from itercc.scf import run

    mol = Molecule(atoms, basis)
    assert mol.symmetry.group == group
    u = np.hstack([block for _, block in mol.symmetry.so_basis])
    np.testing.assert_allclose(u.T @ u, np.eye(mol.nao), atol=1e-12)              # orthonormal, complete
    result, hcore, eri = run(mol, conv_tol=1e-11)
    reference = scf.RHF(gto.M(atom=atoms, basis=basis, verbose=0)).run(conv_tol=1e-11)
    assert result.converged and abs(result.energy - reference.e_tot) < 1e-9
    # Every orbital is pure in its irrep.
    for irrep, block in mol.symmetry.so_basis:
        if block.shape[1]:
            other = np.hstack([b for i, b in mol.symmetry.so_basis if i != irrep])
            s = __import__("itercc.integrals", fromlist=["overlap"]).overlap(mol.basis)
            orbitals = result.mo_coeff[:, result.orbital_irreps == irrep]
            assert np.abs(other.T @ s @ orbitals).max() < 1e-8


def test_native_water_ccsd_matches_the_pyscf_path():
    from pyscf import gto, scf

    from itercc import RestrictedCC
    from itercc.molecule import Molecule
    from itercc.pyscf_interface import from_rhf
    from itercc.scf import from_scf, run

    mol = Molecule(WATER, "cc-pvdz")
    result, hcore, eri = run(mol, conv_tol=1e-12)
    energies = []
    for model in (from_scf(result, hcore, eri, mol.energy_nuc(), frozen=1),
                  from_rhf(scf.RHF(gto.M(atom=WATER, basis="cc-pvdz", symmetry=True, verbose=0)).run(conv_tol=1e-12),
                           frozen=1)):
        cc = RestrictedCC(model.h, model.eri, model.nocc, model.constant, orbital_irreps=model.orbital_irreps,
                          symmetry_tolerance=1e-8, rank=3)
        energies.append(cc.solve(energy_tol=1e-11, residual_tol=1e-9).energy)
    assert abs(energies[0] - energies[1]) < 1e-9


def test_low_symmetry_and_basis_lookup():
    from itercc.molecule import Molecule, parse_nwchem

    mol = Molecule(CHFCLBR, "sto-3g")
    assert mol.symmetry.group == "C1" and mol.symmetry.nirrep == 1
    parsed = parse_nwchem("H    SP\n  1.0   0.5   0.25\n  0.2   0.5   0.75\n")
    assert [entry[0] for entry in parsed["H"]] == [0, 1]
    with pytest.raises(FileNotFoundError):
        Molecule(WATER, "no-such-basis")


@pytest.mark.parametrize("basis", ["def2-tzvp", "6-311g**"])
def test_bundled_basis_sets_give_pyscf_energies(basis, monkeypatch):
    from pathlib import Path

    from pyscf import gto, scf

    import itercc.molecule as molecule
    from itercc.scf import run

    bundled = Path(molecule.__file__).parent / "basis_sets"
    monkeypatch.setattr(molecule, "_basis_directories", lambda: [bundled])   # the package's files only
    mol = molecule.Molecule(WATER, basis)
    result, _, _ = run(mol, conv_tol=1e-11)
    reference = scf.RHF(gto.M(atom=WATER, basis=basis, verbose=0)).run(conv_tol=1e-11)
    assert mol.nao == reference.mol.nao and abs(result.energy - reference.e_tot) < 1e-9


def test_effective_core_potentials_are_refused():
    from itercc.molecule import Molecule

    with pytest.raises(NotImplementedError):
        Molecule("Rb 0 0 0; H 0 0 2.4", "def2-svp", symmetry=False)
