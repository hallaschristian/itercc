"""Optional PySCF adapter: correlated MO integrals, Abelian orbital labels and
dipole matrices from an RHF (or scalar-X2C RHF) calculation, and X2C SOC."""

from dataclasses import dataclass

import numpy as np

from .hamiltonian import Hamiltonian
from . import _runtime


@dataclass(frozen=True)
class SpatialIntegrals:
    h: np.ndarray
    eri: np.ndarray
    nocc: int
    constant: float
    mo_coeff: np.ndarray
    orbital_irreps: np.ndarray | None = None

    def spin_hamiltonian(self):
        return Hamiltonian.restricted(self.h, self.eri, self.nocc, self.constant)


def from_rhf(mf, *, frozen=0):
    """Extract real MO integrals for an RHF state with a lowest-occupied frozen core.

    Returns `SpatialIntegrals` with h, eri (chemists' notation), nocc (correlated
    occupied orbitals), constant (nuclear repulsion plus frozen-core energy),
    mo_coeff (the correlated MOs) and orbital_irreps: the Abelian irrep (0..7,
    D2h or a subgroup in PySCF's numbering) of each correlated orbital when the
    molecule was built with symmetry, else None. Linear molecules get the
    labels of their D2h or C2v subgroup. Uses the same hcore (including any
    scalar X2C and field) and ERI source as the supplied mean-field object.
    PySCF is required only for this adapter.
    """
    from pyscf import ao2mo, scf

    from pyscf.dft.rks import KohnShamDFT

    # RKS subclasses RHF; its get_veff would put the XC potential in a frozen core.
    if not isinstance(mf, scf.hf.RHF) or isinstance(mf, KohnShamDFT) or not mf.converged:
        raise ValueError("Expected converged RHF (or scalar-X2C RHF)")
    if _attribute(mf, "with_df") is not None:
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
    labels = _orbital_irreps(mf, frozen)
    eri = _mo_eri(mf, active, labels)
    return SpatialIntegrals(active.T @ h @ active, eri, nocc - frozen, float(constant), active, labels)


def _mo_eri(mf, active, labels):
    """The (pq|rs) of the active orbitals: IterCC's own transformation of the
    in-core AO integrals (`itercc.integrals.mo_eri`; 2-4x faster than ao2mo and
    its unpacking on 56 cores, symmetry-forbidden entries exact zeros), else
    PySCF's ao2mo (AO integrals not held in memory, no compiled core, or
    ITERCC_NATIVE_TRANSFORM=0)."""
    from pyscf import ao2mo

    from . import _cpu_kernels

    if (mf._eri is not None and _runtime.enabled("ITERCC_NATIVE_TRANSFORM")
            and _cpu_kernels.library() is not None):
        from .integrals import mo_eri

        return mo_eri(mf._eri, active, orbital_irreps=labels)
    source = mf._eri if mf._eri is not None else mf.mol
    return _full_eri(ao2mo.kernel(source, active), active.shape[1])


def _attribute(obj, name):
    """obj.name, or None, without PySCF's lazy `__getattr__`: a missing
    attribute of a mean-field object imports every PySCF module first (about
    0.25 s, most of `from_rhf` for a small molecule)."""
    try:
        return object.__getattribute__(obj, name)
    except AttributeError:
        return None


def _full_eri(packed, n):
    """The n^4 (pq|rs) from PySCF's 4-fold packed MO integrals: unpacked on all
    threads by the compiled core (ao2mo.restore runs on one)."""
    from pyscf import ao2mo

    from . import _cpu_kernels

    library = _cpu_kernels.library()
    packed = np.asarray(packed)
    npair = n * (n + 1) // 2
    if library is None or packed.shape != (npair, npair) or packed.dtype != np.float64:
        return ao2mo.restore(1, packed, n)
    from ._runtime import cpu_threads

    packed = np.ascontiguousarray(packed)
    full = np.empty((n,) * 4)
    library.unpack_s4(full.ctypes.data, packed.ctypes.data, n, cpu_threads(None))
    return full


def _orbital_irreps(mf, frozen):
    mol = mf.mol
    if not mol.symmetry:
        return None
    from pyscf.scf import hf_symm

    labels = np.asarray(hf_symm.get_orbsym(mol, mf.mo_coeff), dtype=np.int64)
    if mol.groupname in ("Dooh", "Coov"):
        labels = labels % 10                     # PySCF's D2h / C2v ids of the linear irreps
    if labels.shape != (len(mf.mo_occ),) or labels.min() < 0 or labels.max() > 7:
        return None
    return labels[frozen:].copy()


def dipole_integrals(model, mol, *, origin=None):
    """Electronic dipole matrices for `DipoleResponse`, and their irreps.

    Returns (dipoles, dipole_irreps): the three length-gauge matrices of the
    electronic dipole operator -r (atomic units) in the correlated orbital
    frame of `model` (from `from_rhf`), with the origin at the centre of
    nuclear charge unless given (Bohr), and the Abelian irrep of each
    Cartesian component (None when `model.orbital_irreps` is None). For
    scalar-X2C references supply picture-changed dipoles instead.
    """
    charges = mol.atom_charges()
    if origin is None:
        origin = charges @ mol.atom_coords() / charges.sum()
    with mol.with_common_orig(origin):
        r = mol.intor_symmetric("int1e_r")
    coeff = np.asarray(model.mo_coeff)
    dipoles = -np.einsum("pi,xpq,qj->xij", coeff, r, coeff)
    if model.orbital_irreps is None:
        return dipoles, None
    return dipoles, tuple(_operator_irrep(d, model.orbital_irreps) for d in dipoles)


def _operator_irrep(matrix, labels):
    """The Abelian irrep of a one-body operator: its nonzero elements couple
    orbitals whose labels XOR to it."""
    pair = labels[:, None] ^ labels[None, :]
    scale = np.max(np.abs(matrix))
    found = {int(g) for g in np.unique(pair[np.abs(matrix) > 1e-8 * max(scale, 1.0)])}
    if len(found) > 1:
        raise ValueError(f"the operator spans several irreps {sorted(found)}; check the orbital labels")
    return found.pop() if found else 0


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

