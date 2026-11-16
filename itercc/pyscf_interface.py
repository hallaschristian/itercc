"""Optional PySCF RHF/X2C integral adapter."""

from dataclasses import dataclass

import numpy as np

from .hamiltonian import Hamiltonian


@dataclass(frozen=True)
class SpatialIntegrals:
    h: np.ndarray
    eri: np.ndarray
    nocc: int
    constant: float
    mo_coeff: np.ndarray

    def spin_hamiltonian(self):
        return Hamiltonian.restricted(self.h, self.eri, self.nocc, self.constant)


def from_rhf(mf, *, frozen=0):
    """Extract real MO integrals for an RHF state with a lowest-occupied frozen core.

    Uses the same hcore (including any scalar X2C and field) and ERI source as
    the supplied mean-field object. PySCF is required only for this adapter.
    """
    from pyscf import ao2mo, scf

    from pyscf.dft.rks import KohnShamDFT

    # RKS subclasses RHF; its get_veff would put the XC potential in a frozen core.
    if not isinstance(mf, scf.hf.RHF) or isinstance(mf, KohnShamDFT) or not mf.converged:
        raise ValueError("Expected converged RHF (or scalar-X2C RHF)")
    if getattr(mf, "with_df", None) is not None:
        raise ValueError("Density fitting is outside this exact-integral adapter")
    coeff = np.asarray(mf.mo_coeff)
    occ = np.asarray(mf.mo_occ)
    nocc = int(np.count_nonzero(occ))
    if (np.iscomplexobj(coeff) or not np.all(occ[:nocc] == 2)
            or not np.all(occ[nocc:] == 0)):
        raise ValueError("Expected real, occupied-first, doubly occupied RHF orbitals")
    if not isinstance(frozen, (int, np.integer)) or not 0 <= frozen < nocc < len(occ):
        raise ValueError("Expected nonempty correlated occupied and virtual spaces")
    active = np.asarray(coeff[:, frozen:], order="F")
    core = coeff[:, :frozen]
    hcore = mf.get_hcore()
    h = hcore
    constant = mf.energy_nuc()
    if frozen:
        dm = 2 * (core @ core.T)
        potential = mf.get_veff(dm=dm)
        h = hcore + potential
        constant += np.einsum("pq,qp->", hcore + 0.5 * potential, dm)
    source = mf._eri if mf._eri is not None else mf.mol
    eri = ao2mo.restore(1, ao2mo.kernel(source, active), active.shape[1])
    return SpatialIntegrals(active.T @ h @ active, eri, nocc - frozen, float(constant), active)


def x2c_spin_orbit(mf, *, mo_coeff=None):
    """Build bare one-electron X2C Pauli SOC with the scalar helper's settings.

    Pass the correlated MO coefficients from from_rhf to obtain SOC in
    the same orbital order. Omitting them returns the matrices in the AO
    basis. The spin trace is discarded so that the original scalar
    Hamiltonian can be used for state interaction.
    """
    from pyscf.x2c.sfx2c1e import SpinFreeX2CHelper
    from pyscf.x2c.x2c import SpinOrbitalX2CHelper

    scalar = getattr(mf, "with_x2c", None)
    if type(scalar) is not SpinFreeX2CHelper or "get_xmat" in vars(scalar):
        raise ValueError("expected a standard PySCF scalar-X2C helper")
    if mf.mol.has_ecp():
        raise ValueError("bare molecular X2C requires an all-electron basis")

    helper = SpinOrbitalX2CHelper(mf.mol)
    helper.approx = scalar.approx
    helper.xuncontract = scalar.xuncontract
    helper.basis = scalar.basis
    h = helper.get_hcore()
    n = h.shape[0] // 2
    aa, ab, ba, bb = h[:n, :n], h[:n, n:], h[n:, :n], h[n:, n:]
    pauli = np.stack((0.5 * (ab + ba), (ba - ab) / (2j), 0.5 * (aa - bb)))
    if not np.isfinite(pauli).all() or np.max(abs(pauli - pauli.conj().swapaxes(1, 2))) > 1e-9:
        raise RuntimeError("X2C SOC is not finite and Hermitian")
    pauli = 0.5 * (pauli + pauli.conj().swapaxes(1, 2))
    if mo_coeff is None:
        return pauli
    coeff = np.asarray(mo_coeff)
    if coeff.ndim != 2 or coeff.shape[0] != n or not np.isfinite(coeff).all():
        raise ValueError("expected a finite AO-by-MO coefficient matrix")
    return coeff.conj().T @ pauli @ coeff

