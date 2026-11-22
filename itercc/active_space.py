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


def natural_virtuals(density, nocc, count, *, fock=None, orbital_irreps=None, orbital_lz=None, mirror=None):
    """Truncate the virtual space to its `count` most occupied natural orbitals.

    `density` (n x n, e.g. a state-averaged selected-CI density) is
    diagonalized in its virtual block within each symmetry label (C2v
    `orbital_irreps`, or the m of axial orbitals `orbital_lz` with their
    reflection `mirror`: the block averaged with its reflection and the same
    rotation for m and -m, so pairs are kept or dropped together). The
    `count` virtuals of largest occupation are kept (frozen natural
    virtuals); with `fock` they are semicanonicalized within each label.
    The occupied orbitals are unchanged.

    Returns (coefficients, labels, mirror, discarded): the n x (nocc + count)
    columns of the new orbitals in the old ones (occupied first, then the
    virtuals by orbital energy or occupation, each +m orbital followed by its
    -m partner), their labels, their reflection map (axial; else None), and
    the discarded occupation. Apply with
    `selected_ci.rotate_integrals(h, eri, coefficients, axial=...)`.
    """
    density = np.asarray(density, dtype=np.float64)
    n = density.shape[0]
    if density.shape != (n, n) or not 0 < nocc < n:
        raise ValueError("Expected a square density and 0 < nocc < n")
    axial = orbital_lz is not None
    if axial and mirror is None:
        raise ValueError("axial orbitals need their mirror map")
    labels = (np.asarray(orbital_lz, dtype=np.int64) if axial
              else np.zeros(n, dtype=np.int64) if orbital_irreps is None
              else np.asarray(orbital_irreps, dtype=np.int64))
    virtual = np.arange(nocc, n)
    if axial:
        mirror = np.asarray(mirror, dtype=np.int64)
        if (mirror[virtual] < nocc).any():
            raise ValueError("the reflection must map virtual orbitals to virtual orbitals")
        density = 0.5 * (density + density[np.ix_(mirror, mirror)])
    # Natural virtuals per label (axial: per m >= 0, the -m block by reflection).
    groups = []          # (occupation, label, vector over all orbitals, partner vector or None)
    for value in np.unique(labels[virtual]):
        if axial and value < 0:
            continue
        idx = virtual[labels[virtual] == value]
        w, v = np.linalg.eigh(density[np.ix_(idx, idx)])
        for k in range(len(idx)):
            vec = np.zeros(n)
            vec[idx] = v[:, k]
            partner = None
            if axial and value > 0:
                partner = np.zeros(n)
                partner[mirror[idx]] = v[:, k]
            groups.append((w[k], value, vec, partner))
    groups.sort(key=lambda g: -g[0])
    kept, total = [], 0
    for g in groups:
        size = 1 if g[3] is None else 2
        if total + size > count:
            continue
        kept.append(g)
        total += size
        if total == count:
            break
    if total != count:
        raise ValueError(f"cannot keep exactly {count} virtuals with pairs kept together")
    discarded = float(sum(g[0] * (1 if g[3] is None else 2) for g in groups) - sum(
        g[0] * (1 if g[3] is None else 2) for g in kept))
    # Semicanonical virtuals within each label (the same rotation for a pair's members).
    columns = []          # (energy or -occupation, label, vector, partner)
    for value in sorted({g[1] for g in kept}):
        block = [g for g in kept if g[1] == value]
        vecs = np.array([g[2] for g in block]).T
        if fock is not None:
            e, r = np.linalg.eigh(vecs.T @ np.asarray(fock) @ vecs)
        else:
            e, r = -np.array([g[0] for g in block]), np.eye(len(block))
        new = vecs @ r
        partners = None if block[0][3] is None else np.array([g[3] for g in block]).T @ r
        for k in range(len(block)):
            columns.append((e[k], value, new[:, k], None if partners is None else partners[:, k]))
    columns.sort(key=lambda c: (c[0], c[1]))
    coeff = np.zeros((n, nocc + count))
    coeff[:nocc, :nocc] = np.eye(n)[:nocc, :nocc]
    new_labels = list(labels[:nocc])
    new_mirror = list(mirror[:nocc]) if axial else None
    col = nocc
    for e, value, vec, partner in columns:
        coeff[:, col] = vec
        new_labels.append(value)
        if partner is None:
            if axial:
                new_mirror.append(col)
            col += 1
        else:
            coeff[:, col + 1] = partner
            new_labels.append(-value)
            new_mirror.extend([col + 1, col])
            col += 2
    return (coeff, np.asarray(new_labels, dtype=np.int64),
            None if new_mirror is None else np.asarray(new_mirror, dtype=np.int64), discarded)
