"""Build reproducible closed-shell active-space Hamiltonians.

These routines serve validation calculations that compare coupled cluster with
selected CI or FCI on precisely the same spatial-orbital Hamiltonian.
"""

import numpy as np

from .spatial_symmetry import orbital_labels


def fold_closed_shell(h, eri, constant, count):
    """Integrate out `count` lowest closed-shell spatial orbitals.

    Returns the valence Hamiltonian and the additive scalar energy. The input
    ERIs use chemists' notation and all returned arrays have independent,
    contiguous storage suitable for a solver checkpoint.
    """
    h = np.asarray(h, dtype=np.float64)
    eri = np.asarray(eri, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or eri.shape != (len(h),) * 4:
        raise ValueError("Incompatible spatial h and ERI shapes")
    if not isinstance(count, (int, np.integer)) or not 0 <= count < len(h):
        raise ValueError("count must retain at least one spatial orbital")
    occupied = np.arange(count)
    coulomb = eri[:, :, occupied, occupied].sum(axis=-1)
    exchange = np.einsum("piiq->pq", eri[:, :count, :count, :], optimize=True)
    fock_shift = 2*coulomb - exchange
    updated_constant = float(constant) + float(np.trace((2*h + fock_shift)[:count, :count]))
    return (np.ascontiguousarray((h + fock_shift)[count:, count:]),
            np.ascontiguousarray(eri[count:, count:, count:, count:]), updated_constant)


def select_active_orbitals(h, eri, constant, labels, occupied_count, occupied, virtual):
    """Select occupied and virtual orbitals while keeping their solver order.

    `occupied` indexes the occupied block and `virtual` indexes the virtual
    block. The return value contains the sliced Hamiltonian, new occupied
    count, and Abelian labels. Repeated or cross-block indices are rejected so
    that a saved active model remains unambiguous.
    """
    h = np.asarray(h, dtype=np.float64)
    eri = np.asarray(eri, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or eri.shape != (len(h),) * 4:
        raise ValueError("Incompatible spatial h and ERI shapes")
    labels = orbital_labels(labels, len(h))
    if not isinstance(occupied_count, (int, np.integer)) or not 0 < occupied_count < len(h):
        raise ValueError("Invalid occupied_count")
    occupied, virtual = np.asarray(occupied, dtype=int), np.asarray(virtual, dtype=int)
    if occupied.ndim != 1 or virtual.ndim != 1 or not len(occupied) or not len(virtual):
        raise ValueError("Select at least one occupied and one virtual orbital")
    if (np.any(occupied < 0) or np.any(occupied >= occupied_count) or np.unique(occupied).size != len(occupied)
            or np.any(virtual < 0) or np.any(virtual >= len(h)-occupied_count)
            or np.unique(virtual).size != len(virtual)):
        raise ValueError("Active indices must be unique and belong to their orbital block")
    keep = np.r_[occupied, occupied_count + virtual]
    return (np.ascontiguousarray(h[np.ix_(keep, keep)]),
            np.ascontiguousarray(eri[np.ix_(keep, keep, keep, keep)]), float(constant),
            len(occupied), labels[keep])
