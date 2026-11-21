"""Native spin-free X2C (sfX2C-1e): relativistic integrals, the decontracted
basis, the one-electron Hamiltonian, picture-changed dipoles and RHF energies,
against PySCF's `scf.RHF(mol).sfx2c1e()` where PySCF is installed."""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from itercc import _cpu_kernels

pytestmark = pytest.mark.skipif(_cpu_kernels.library() is None, reason="compiled core unavailable")

CUF = "Cu 0 0 0; F 0 0 1.76"
CUF_RHF = -1752.7103132582324     # PySCF 2.14 sfX2C-1e RHF of the CuF production model (aug-cc-pVDZ-DK, C2v)
SYSTEMS = {"Cu+": ("Cu 0 0 0", 1), "CuF": (CUF, 0)}


@lru_cache(maxsize=None)
def native(name):
    """(Molecule, RHFResult) of a system, with the bundled aug-cc-pVDZ-DK only."""
    import itercc.molecule as molecule
    from itercc.scf import run

    bundled = Path(molecule.__file__).parent / "basis_sets"
    atoms, charge = SYSTEMS[name]
    with mock.patch.object(molecule, "_basis_directories", lambda: [bundled]):
        mol = molecule.Molecule(atoms, "aug-cc-pVDZ-DK", charge=charge)
    result, _, _ = run(mol, x2c=True, conv_tol=1e-12, conv_tol_grad=1e-9)     # dipoles: first order in it
    assert result.converged
    return mol, result


def test_relativistic_integrals_match_libcint():
    pytest.importorskip("pyscf")
    from pyscf import gto

    from itercc import integrals
    from itercc.basis import Basis

    mol = gto.M(atom="Cu 0 0 0; F 0 0.3 1.76", basis="aug-cc-pvdz-dk", verbose=0)   # through f, off axis
    b = Basis.from_pyscf(mol)
    pvp = mol.intor("int1e_pnucp")
    np.testing.assert_allclose(integrals.pvp(b), pvp, rtol=0, atol=1e-13 * np.abs(pvp).max())
    origin = (0.1, -0.2, 0.3)
    with mol.with_common_orig(origin):
        prp = mol.intor("int1e_sprsp").reshape(3, 4, mol.nao, mol.nao)[:, 3]      # the spin-free part
    np.testing.assert_allclose(integrals.prp(b, origin), prp, rtol=0, atol=1e-13 * np.abs(prp).max())
    assert np.array_equal(integrals.pvp(b, threads=1), integrals.pvp(b, threads=3))
    assert np.array_equal(integrals.prp(b, origin, threads=1), integrals.prp(b, origin, threads=3))


def test_decontracted_basis_spans_the_contracted_one():
    from itercc import integrals

    mol, _ = native("CuF")
    primitives, contraction = mol.basis.decontracted()
    assert all(len(shell.exponents) == 1 for shell in primitives.shells)
    np.testing.assert_allclose(contraction.T @ integrals.overlap(primitives) @ contraction,
                               integrals.overlap(mol.basis), rtol=0, atol=1e-13)
    pyscf = pytest.importorskip("pyscf")
    from itercc.basis import Basis

    reference = pyscf.gto.M(atom=CUF, basis="aug-cc-pvdz-dk", verbose=0)
    pmol, _ = reference.decontract_basis(aggregate=True)
    ours, _ = Basis.from_pyscf(reference).decontracted()
    assert ours.nao == pmol.nao
    for atom in range(pmol.natm):
        for l in range(pmol._bas[:, 1].max() + 1):
            theirs = sorted(float(pmol.bas_exp(i)[0]) for i in range(pmol.nbas)
                            if pmol.bas_atom(i) == atom and pmol.bas_angular(i) == l)
            mine = sorted(float(s.exponents[0]) for s in ours.shells if s.atom == atom and s.l == l)
            assert mine == theirs


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_hamiltonian_and_picture_changed_dipoles_match_pyscf(name):
    pytest.importorskip("pyscf")
    from pyscf import gto, lib, scf

    from itercc import integrals
    from itercc.basis import Basis

    atoms, charge = SYSTEMS[name]
    mol = gto.M(atom=atoms, basis="aug-cc-pvdz-dk", charge=charge, verbose=0)
    reference = scf.RHF(mol).sfx2c1e()
    x2c = integrals.SpinFreeX2C(Basis.from_pyscf(mol))
    np.testing.assert_allclose(x2c.hcore(), reference.get_hcore(), rtol=0, atol=1e-8)
    origin = (0.0, 0.1, -0.2)
    with mol.with_common_orig(origin):
        xmol = reference.with_x2c.get_xmol()[0]
        small = xmol.intor_symmetric("int1e_sprsp").reshape(3, 4, xmol.nao, xmol.nao)[:, 3]
        dipole = reference.with_x2c.picture_change(("int1e_r", small * (0.5 / lib.param.LIGHT_SPEED) ** 2))
    np.testing.assert_allclose(x2c.dipole(origin), dipole, rtol=0, atol=1e-11)
    assert np.abs(dipole - integrals.dipole(x2c.basis, origin)).max() > 1e-6       # the picture change matters


def test_native_cuf_reproduces_the_production_model():
    _, result = native("CuF")
    assert abs(result.energy - CUF_RHF) < 1e-8


@pytest.mark.parametrize("name", sorted(SYSTEMS))
def test_x2c_rhf_energies_match_pyscf(name):
    pytest.importorskip("pyscf")
    from pyscf import gto, scf

    atoms, charge = SYSTEMS[name]
    mol, result = native(name)
    assert mol.symmetry.group == {"Cu+": "D2h", "CuF": "C2v"}[name]
    reference = scf.RHF(gto.M(atom=atoms, basis="aug-cc-pvdz-dk", charge=charge, symmetry=True, verbose=0))
    reference = reference.sfx2c1e().run(conv_tol=1e-12)
    assert reference.converged and abs(result.energy - reference.e_tot) < 1e-8


def test_picture_changed_dipole_moment_matches_pyscf():
    from itercc.scf import dipole_integrals

    mol, result = native("CuF")
    nocc = mol.nelectron // 2
    model = SimpleNamespace(mo_coeff=result.mo_coeff, orbital_irreps=result.orbital_irreps)
    moments = {}
    for x2c in (False, True):
        dipoles, irreps = dipole_integrals(mol, model, x2c=x2c)
        origin = mol.charges @ mol.coords / mol.charges.sum()
        electronic = 2.0 * np.einsum("xii->x", dipoles[:, :nocc, :nocc])
        moments[x2c] = electronic + mol.charges @ (mol.coords - origin)
        assert irreps[2] == 0 and len({irreps[0], irreps[1], 0}) == 3                # z totally symmetric
        np.testing.assert_allclose(moments[x2c][:2], 0.0, atol=1e-10)
    assert abs(moments[True][2] - moments[False][2]) > 1e-7                  # 2.3e-7 au for CuF
    pytest.importorskip("pyscf")
    from pyscf import gto, scf

    reference = scf.RHF(gto.M(atom=CUF, basis="aug-cc-pvdz-dk", symmetry=True, verbose=0)).sfx2c1e()
    reference.run(conv_tol=1e-12, conv_tol_grad=1e-9)
    for x2c in (False, True):
        expected = reference.dip_moment(unit="AU", picture_change=x2c, verbose=0)
        assert abs(np.linalg.norm(moments[x2c]) - np.linalg.norm(expected)) < 2e-8                 # 1.3e-9 au
