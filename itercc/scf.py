"""Native restricted Hartree-Fock on the compiled core.

`rhf` solves closed-shell RHF from AO integrals (one-electron matrices and
8-fold packed two-electron integrals): Coulomb and exchange matrices from the
core (`itercc_jk_s8`), Pulay DIIS, and with a symmetry-adapted basis one
eigenproblem per irrep, so every orbital carries an exact Abelian irrep.
`from_scf` then gives the correlated-orbital integrals the coupled-cluster
solvers take, with the native transformation (`itercc.integrals.mo_eri`).
The AO integrals still come from elsewhere (`integrals_from_pyscf`) until the
package computes them itself.
"""
from dataclasses import dataclass

import numpy as np

from . import _cpu_kernels
from ._runtime import cpu_threads


def jk(eri_s8, dm, threads=None):
    """(J, K) of a symmetric density `dm` from 8-fold packed real AO integrals:
    J(mn) = sum (mn|ls) dm(ls), K(mn) = sum (ml|ns) dm(ls)."""
    library = _cpu_kernels.library()
    if library is None:
        raise RuntimeError("the native SCF needs the compiled core")
    dm = np.ascontiguousarray(dm, dtype=np.float64)
    n = dm.shape[0]
    eri_s8 = np.ascontiguousarray(eri_s8, dtype=np.float64)
    npair = n * (n + 1) // 2
    if dm.shape != (n, n) or eri_s8.shape != (npair * (npair + 1) // 2,):
        raise ValueError("dm must be square and eri_s8 the 8-fold packed integrals of its basis")
    vj, vk = np.empty((n, n)), np.empty((n, n))
    library.jk_s8(eri_s8.ctypes.data, n, dm.ctypes.data, vj.ctypes.data, vk.ctypes.data, cpu_threads(threads))
    return vj, vk


@dataclass(frozen=True)
class RHFResult:
    energy: float
    mo_coeff: np.ndarray          # (AO, orbitals), by orbital energy
    mo_energy: np.ndarray
    mo_occ: np.ndarray            # 2 or 0
    orbital_irreps: np.ndarray    # Abelian irrep id of each orbital (None without symmetry)
    converged: bool
    iterations: int


def _blocks(overlap, so_basis):
    """Per irrep: (irrep id, SO coefficients U (AO x m), orthonormalizer X (m x k))."""
    blocks = []
    for irrep, u in so_basis:
        u = np.asarray(u, dtype=np.float64)
        if u.shape[1] == 0:
            continue
        s = u.T @ overlap @ u
        w, v = np.linalg.eigh(s)
        keep = w > 1e-10 * w.max()
        blocks.append((int(irrep), u, v[:, keep] / np.sqrt(w[keep])))
    return blocks


def _diagonalize(fock, blocks):
    """Orbital energies, coefficients (AO x all) and irreps, by energy."""
    energies, coeffs, irreps = [], [], []
    for irrep, u, x in blocks:
        f = x.T @ (u.T @ fock @ u) @ x
        e, c = np.linalg.eigh(0.5 * (f + f.T))
        energies.append(e)
        coeffs.append(u @ (x @ c))
        irreps.append(np.full(len(e), irrep, dtype=np.int64))
    energies, coeffs, irreps = np.concatenate(energies), np.hstack(coeffs), np.concatenate(irreps)
    order = np.argsort(energies, kind="stable")
    return energies[order], coeffs[:, order], irreps[order]


def rhf(hcore, overlap, eri_s8, nelectron, energy_nuc=0.0, *, so_basis=None, conv_tol=1e-10,
        conv_tol_grad=None, max_iter=100, diis_space=8, dm0=None, threads=None):
    """Closed-shell RHF. `so_basis`: [(irrep id, AO x m coefficients), ...] of
    an orthonormal symmetry-adapted basis (for example PySCF's
    zip(mol.irrep_id, mol.symm_orb)); without it one block and no irreps.
    Starts from the core Hamiltonian, or `dm0`. Converged when the energy
    changes by less than `conv_tol` and the orbital gradient (largest entry of
    FDS - SDF in an orthonormal basis) is below `conv_tol_grad` (default
    sqrt(conv_tol))."""
    hcore = np.asarray(hcore, dtype=np.float64)
    overlap = np.asarray(overlap, dtype=np.float64)
    n = len(hcore)
    if nelectron % 2 or not 0 < nelectron // 2 <= n:
        raise ValueError("closed-shell RHF needs an even number of electrons that fit the basis")
    nocc = nelectron // 2
    symmetric = so_basis is not None
    blocks = _blocks(overlap, so_basis if symmetric else [(0, np.eye(n))])
    x_full = np.zeros((n, sum(x.shape[1] for _, _, x in blocks)))
    column = 0
    for _, u, x in blocks:
        x_full[:, column:column + x.shape[1]] = u @ x
        column += x.shape[1]
    conv_tol_grad = np.sqrt(conv_tol) if conv_tol_grad is None else conv_tol_grad
    if dm0 is None:
        _, c, _ = _diagonalize(hcore, blocks)
        dm = 2.0 * c[:, :nocc] @ c[:, :nocc].T
    else:
        dm = np.asarray(dm0, dtype=np.float64)
    focks, errors = [], []
    last, converged = None, False
    for iteration in range(1, max_iter + 1):
        vj, vk = jk(eri_s8, dm, threads)
        fock = hcore + vj - 0.5 * vk
        energy = 0.5 * float(np.einsum("ij,ji->", dm, hcore + fock)) + energy_nuc
        error = x_full.T @ (fock @ dm @ overlap - overlap @ dm @ fock) @ x_full
        gradient = float(np.abs(error).max())
        if last is not None and abs(energy - last) < conv_tol and gradient < conv_tol_grad:
            converged = True
            break
        last = energy
        focks.append(fock)
        errors.append(error)
        del focks[:-diis_space], errors[:-diis_space]
        if len(focks) > 1:
            m = len(focks)
            b = -np.ones((m + 1, m + 1))
            b[m, m] = 0.0
            for i in range(m):
                for j in range(i + 1):
                    b[i, j] = b[j, i] = float(np.vdot(errors[i], errors[j]))
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            weights = np.linalg.lstsq(b, rhs, rcond=None)[0][:m]
            fock = sum(w * f for w, f in zip(weights, focks))
        _, c, _ = _diagonalize(fock, blocks)
        dm = 2.0 * c[:, :nocc] @ c[:, :nocc].T
    # Canonical orbitals of the final Fock matrix.
    energies, coeffs, irreps = _diagonalize(hcore + vj - 0.5 * vk, blocks)
    occ = np.zeros(len(energies))
    occ[:nocc] = 2.0
    return RHFResult(energy, coeffs, energies, occ, irreps if symmetric else None, converged, iteration)


def from_scf(result, hcore, eri_s8, energy_nuc=0.0, *, frozen=0, threads=None):
    """Correlated-orbital integrals of a converged `rhf` result, as
    `pyscf_interface.from_rhf` returns them: h (frozen-core dressed), eri
    (chemists' notation, `integrals.mo_eri`), nocc, constant (nuclear
    repulsion plus frozen-core energy), mo_coeff and orbital_irreps."""
    from .integrals import mo_eri
    from .pyscf_interface import SpatialIntegrals

    if not result.converged:
        raise ValueError("Expected a converged RHF")
    nocc = int(np.count_nonzero(result.mo_occ))
    if not isinstance(frozen, (int, np.integer)) or not 0 <= frozen < nocc < len(result.mo_occ):
        raise ValueError("Expected nonempty correlated occupied and virtual spaces")
    hcore = np.asarray(hcore, dtype=np.float64)
    active = np.asarray(result.mo_coeff[:, frozen:], order="C")
    h, constant = hcore, float(energy_nuc)
    if frozen:
        core = result.mo_coeff[:, :frozen]
        dm = 2.0 * core @ core.T
        vj, vk = jk(eri_s8, dm, threads)
        potential = vj - 0.5 * vk
        h = hcore + potential
        constant += float(np.einsum("pq,qp->", hcore + 0.5 * potential, dm))
    labels = None if result.orbital_irreps is None else result.orbital_irreps[frozen:].copy()
    eri = mo_eri(eri_s8, active, orbital_irreps=labels, threads=threads)
    return SpatialIntegrals(active.T @ h @ active, eri, nocc - frozen, constant, active, labels)


def integrals_from_pyscf(mol):
    """(hcore, overlap, 8-fold packed ERIs, electrons, nuclear repulsion,
    symmetry-adapted basis or None) of a PySCF molecule, from libcint: the
    AO integrals `rhf` takes until the package computes its own."""
    hcore = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
    so_basis = list(zip(mol.irrep_id, mol.symm_orb)) if mol.symmetry else None
    return (hcore, mol.intor_symmetric("int1e_ovlp"), mol.intor("int2e", aosym="s8"), mol.nelectron,
            mol.energy_nuc(), so_basis)


def run(molecule, *, threads=None, screen=1e-14, **options):
    """Native RHF of a `molecule.Molecule`: the package's own AO integrals
    (`itercc.integrals`), symmetry-adapted basis and `rhf`. Returns
    (RHFResult, hcore, 8-fold packed ERIs) for `from_scf`."""
    from . import integrals

    b = molecule.basis
    hcore = integrals.kinetic(b, threads) + integrals.nuclear(b, threads)
    overlap = integrals.overlap(b, threads)
    eri = integrals.eri_s8(b, screen=screen, threads=threads)
    so = molecule.symmetry.so_basis if molecule.symmetry is not None else None
    result = rhf(hcore, overlap, eri, molecule.nelectron, molecule.energy_nuc(), so_basis=so, threads=threads,
                 **options)
    return result, hcore, eri
