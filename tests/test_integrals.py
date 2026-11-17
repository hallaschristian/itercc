"""The native AO-to-MO transformation (itercc.integrals) against PySCF's ao2mo."""
import numpy as np
import pytest

from itercc import _cpu_kernels

pytestmark = pytest.mark.skipif(_cpu_kernels.library() is None, reason="compiled core unavailable")


def _water(basis="cc-pvdz"):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf
    from pyscf.scf import hf_symm

    mol = gto.M(atom="O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469", basis=basis, symmetry=True,
                verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    labels = np.asarray(hf_symm.get_orbsym(mol, mf.mo_coeff), dtype=np.int64)
    return pyscf, mol, mf, labels


@pytest.mark.parametrize("form", ["s8", "s4", "s1"])
def test_mo_eri_matches_ao2mo(form):
    pyscf, mol, mf, labels = _water()
    from itercc.integrals import mo_eri

    coeff = mf.mo_coeff[:, 1:]
    n = coeff.shape[1]
    reference = pyscf.ao2mo.restore(1, pyscf.ao2mo.kernel(mf._eri, coeff), n)
    ao = {"s8": mf._eri, "s4": pyscf.ao2mo.restore(4, mf._eri, mol.nao),
          "s1": pyscf.ao2mo.restore(1, mf._eri, mol.nao)}[form]
    for threads in (1, 3):
        plain = mo_eri(ao, coeff, threads=threads, batch_bytes=1 << 16)   # several batches
        np.testing.assert_allclose(plain, reference, rtol=0, atol=1e-12)
    symmetric = mo_eri(ao, coeff, orbital_irreps=labels[1:])
    total = labels[1:, None, None, None] ^ labels[None, 1:, None, None] ^ labels[None, None, 1:, None] \
        ^ labels[None, None, None, 1:]
    assert np.all(symmetric[total != 0] == 0.0)                          # exact zeros
    np.testing.assert_allclose(symmetric[total == 0], reference[total == 0], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(symmetric, symmetric.transpose(1, 0, 2, 3))
    np.testing.assert_array_equal(symmetric, symmetric.transpose(0, 1, 3, 2))


def test_from_rhf_with_and_without_the_native_transform(monkeypatch):
    _, _, mf, _ = _water()
    from itercc.pyscf_interface import from_rhf

    native = from_rhf(mf, frozen=1)
    monkeypatch.setenv("ITERCC_NATIVE_TRANSFORM", "0")
    pyscf_side = from_rhf(mf, frozen=1)
    np.testing.assert_allclose(native.eri, pyscf_side.eri, rtol=0, atol=1e-12)
    np.testing.assert_allclose(native.h, pyscf_side.h, rtol=0, atol=1e-12)    # BLAS rounding


def test_clean_integrals_are_kept_without_a_copy():
    from itercc.spatial_symmetry import _project_eri

    labels = np.array([0, 1, 0, 1, 2])
    total = labels[:, None, None, None] ^ labels[None, :, None, None] ^ labels[None, None, :, None] \
        ^ labels[None, None, None, :]
    eri = np.where(total == 0, np.random.default_rng(2).standard_normal((5,) * 4), 0.0)
    kept, audit = _project_eri(eri, labels, 1e-8)
    assert kept is eri and audit.count == 0
    leaky = eri + np.where(total != 0, 1e-12, 0.0)
    projected, audit = _project_eri(leaky, labels, 1e-8)
    assert projected is not leaky and audit.count == np.count_nonzero(total) and np.all(projected == eri)


def test_jk_matches_pyscf():
    _, mol, mf, _ = _water()
    from itercc.scf import jk

    rng = np.random.default_rng(5)
    dm = rng.standard_normal((mol.nao,) * 2)
    dm = dm + dm.T
    reference = mf.get_jk(mol, dm)
    for threads in (1, 4):
        vj, vk = jk(mf._eri, dm, threads)
        np.testing.assert_allclose(vj, reference[0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(vk, reference[1], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(jk(mf._eri, dm, 1)[1], jk(mf._eri, dm, 4)[1])   # fixed sums


def test_native_rhf_and_ccsd_match_pyscf():
    _, mol, mf, labels = _water()
    from itercc import RestrictedCC
    from itercc.pyscf_interface import from_rhf
    from itercc.scf import from_scf, integrals_from_pyscf, rhf

    hcore, overlap, eri, nelectron, enuc, so_basis = integrals_from_pyscf(mol)
    result = rhf(hcore, overlap, eri, nelectron, enuc, so_basis=so_basis, conv_tol=1e-12)
    assert result.converged and abs(result.energy - mf.e_tot) < 1e-10
    np.testing.assert_array_equal(result.orbital_irreps, labels)
    energies = []
    for model in (from_scf(result, hcore, eri, enuc, frozen=1), from_rhf(mf, frozen=1)):
        cc = RestrictedCC(model.h, model.eri, model.nocc, model.constant, orbital_irreps=model.orbital_irreps,
                          symmetry_tolerance=1e-8, rank=2)
        energies.append(cc.solve(energy_tol=1e-11, residual_tol=1e-9).energy)
    assert abs(energies[0] - energies[1]) < 1e-9


@pytest.mark.parametrize("basis", ["sto-3g", "cc-pvdz", "aug-cc-pvdz"])
def test_native_gaussian_integrals_match_libcint(basis):
    pytest.importorskip("pyscf")
    from pyscf import gto

    from itercc import integrals
    from itercc.basis import Basis

    mol = gto.M(atom="O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469", basis=basis, verbose=0)
    b = Basis.from_pyscf(mol)
    np.testing.assert_allclose(integrals.overlap(b), mol.intor("int1e_ovlp"), rtol=0, atol=1e-12)
    np.testing.assert_allclose(integrals.kinetic(b), mol.intor("int1e_kin"), rtol=0, atol=1e-12)
    np.testing.assert_allclose(integrals.nuclear(b), mol.intor("int1e_nuc"), rtol=0, atol=1e-11)
    np.testing.assert_allclose(integrals.dipole(b), mol.intor("int1e_r"), rtol=0, atol=1e-12)
    for threads in (1, 3):
        np.testing.assert_allclose(integrals.eri_s8(b, threads=threads), mol.intor("int2e", aosym="s8"),
                                   rtol=0, atol=1e-12)


def test_spherical_transform_matches_libcint():
    pytest.importorskip("pyscf")
    from pyscf import gto

    from itercc.basis import cart2sph

    for l in range(2, 7):
        np.testing.assert_allclose(cart2sph(l), gto.cart2sph(l, normalized="sp"), rtol=0, atol=1e-13)
