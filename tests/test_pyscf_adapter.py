"""The exact-integral adapter accepts RHF but not Kohn-Sham objects (no SCF is run)."""
import numpy as np
import pytest


def test_from_rhf_rejects_kohn_sham_reference():
    pytest.importorskip("pyscf")
    from pyscf import dft, gto
    from itercc.pyscf_interface import from_rhf

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74; H 0 0 3; H 0 0 3.74", basis="sto-3g", verbose=0)
    mf = dft.RKS(mol)
    # Pretend convergence without an SCF: the type check must reject it first.
    mf.converged = True
    mf.mo_coeff = np.eye(mol.nao)
    mf.mo_occ = np.array([2., 2., 0., 0.])
    with pytest.raises(ValueError, match="RHF"):
        from_rhf(mf, frozen=1)


def test_x2c_pauli_operator_matches_spin_orbital_core():
    np = pytest.importorskip("numpy")
    pytest.importorskip("pyscf")
    from pyscf import gto, scf
    from itercc.pyscf_interface import x2c_spin_orbit

    mol = gto.M(atom="Li 0 0 0; F 0 0 1.56", basis="sto-3g", verbose=0)
    scalar = scf.RHF(mol).x2c()
    scalar.with_x2c.xuncontract = False
    spin = scf.GHF(mol).x2c()
    spin.with_x2c.xuncontract = False
    full = spin.get_hcore()
    n = mol.nao_nr()
    trace = 0.5 * (full[:n, :n] + full[n:, n:])
    x, y, z = x2c_spin_orbit(scalar)
    reconstructed = np.block([[trace + z, x - 1j*y], [x + 1j*y, trace - z]])
    np.testing.assert_allclose(reconstructed, full, rtol=0, atol=1e-12)
    assert np.linalg.norm(y) > 1e-6

    coeff = np.random.default_rng(1).normal(size=(n, 4))
    np.testing.assert_allclose(x2c_spin_orbit(scalar, mo_coeff=coeff),
                               coeff.T @ np.stack((x, y, z)) @ coeff, atol=1e-13)


WATER = "O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469"


def test_from_rhf_returns_orbital_labels_and_dipoles_follow_them():
    pytest.importorskip("pyscf")
    from pyscf import gto, scf
    from pyscf.scf import hf_symm
    from itercc.pyscf_interface import dipole_integrals, from_rhf

    mol = gto.M(atom=WATER, basis="6-31g", symmetry=True, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-10)
    model = from_rhf(mf, frozen=1)
    np.testing.assert_array_equal(model.orbital_irreps, hf_symm.get_orbsym(mol, mf.mo_coeff)[1:])

    dipoles, irreps = dipole_integrals(model, mol)
    assert dipoles.shape == (3,) + model.h.shape
    np.testing.assert_allclose(dipoles, dipoles.transpose(0, 2, 1), rtol=0, atol=1e-12)
    assert sorted(irreps) == [0, 2, 3]                    # C2v: A1, B1, B2
    pair = model.orbital_irreps[:, None] ^ model.orbital_irreps[None, :]
    for component, irrep in zip(dipoles, irreps):
        assert np.max(np.abs(component[pair != irrep])) < 1e-10

    plain_mol = gto.M(atom=WATER, basis="6-31g", verbose=0)
    plain = from_rhf(scf.RHF(plain_mol).run(conv_tol=1e-10), frozen=1)
    assert plain.orbital_irreps is None
    assert dipole_integrals(plain, plain_mol)[1] is None
    np.testing.assert_allclose(np.abs(dipole_integrals(plain, plain_mol)[0]).sum(),
                               np.abs(dipoles).sum(), rtol=1e-8)


def test_linear_molecules_get_abelian_subgroup_labels():
    pytest.importorskip("pyscf")
    from pyscf import gto, scf
    from itercc import RestrictedCC
    from itercc.pyscf_interface import from_rhf

    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="6-31g", symmetry=True, verbose=0)
    assert mol.groupname == "Dooh"
    model = from_rhf(scf.RHF(mol).run(conv_tol=1e-10), frozen=2)
    assert model.orbital_irreps.min() >= 0 and model.orbital_irreps.max() <= 7
    assert len(set(model.orbital_irreps.tolist())) > 2
    cc = RestrictedCC(model.h, model.eri, model.nocc, model.constant, rank=2,
                      orbital_irreps=model.orbital_irreps, symmetry_tolerance=1e-8)
    assert cc.symmetry_projected                          # the labels block the integrals
