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
