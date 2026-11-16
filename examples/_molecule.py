"""Shared PySCF setup for the examples: RHF, correlated MO integrals, Abelian
orbital labels and electronic dipole matrices in the same orbital frame."""
import numpy as np
from pyscf import gto, scf
from pyscf.scf import hf_symm

from itercc.pyscf_interface import from_rhf


def irrep_of(matrix, labels, threshold=1e-8):
    """The Abelian irrep of a one-body operator: its nonzero elements couple
    orbitals whose labels XOR to it."""
    pair = labels[:, None] ^ labels[None, :]
    found = {int(g) for g in np.unique(pair[np.abs(matrix) > threshold])}
    if len(found) != 1:
        raise ValueError(f"The operator spans several irreps: {sorted(found)}")
    return found.pop()


def prepare(atom, basis, *, frozen=0, symmetry=True):
    """Model integrals for `RestrictedCC` and friends.

    Returns a dict with h, eri (chemists' notation), nocc (correlated occupied
    spatial orbitals), constant (nuclear repulsion plus frozen core), labels
    (one XOR irrep per correlated orbital, D2h or a subgroup), dipoles (the
    electronic dipole operator -r, origin at the centre of nuclear charge) and
    dipole_irreps (the irrep of each Cartesian component).
    """
    mol = gto.M(atom=atom, basis=basis, symmetry=symmetry, verbose=0)
    if symmetry and mol.groupname in ("Dooh", "Coov"):      # Abelian subgroup for linear molecules
        mol = gto.M(atom=atom, basis=basis, symmetry="D2h" if mol.groupname == "Dooh" else "C2v",
                    verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    model = from_rhf(mf, frozen=frozen)
    coeff = model.mo_coeff
    if symmetry:
        labels = np.asarray(hf_symm.get_orbsym(mol, mf.mo_coeff), dtype=np.int64)[frozen:]
    else:
        labels = np.zeros(coeff.shape[1], dtype=np.int64)
    charges = mol.atom_charges()
    origin = charges @ mol.atom_coords() / charges.sum()
    with mol.with_common_orig(origin):
        r = mol.intor_symmetric("int1e_r")
    dipoles = -np.einsum("pi,xpq,qj->xij", coeff, r, coeff)
    return dict(h=model.h, eri=model.eri, nocc=model.nocc, constant=model.constant,
                labels=labels, dipoles=dipoles,
                dipole_irreps=tuple(irrep_of(d, labels) for d in dipoles),
                scf_energy=mf.e_tot)
