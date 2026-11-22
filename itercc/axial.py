"""Axial (Lz-adapted) orbitals of linear molecules.

A molecule on the z axis keeps the angular momentum about the axis: orbitals
of m = 0 (sigma), +-1 (pi), +-2 (delta), ... Real orbitals come in pairs of
cos(m phi) and sin(m phi) type (C2v: A1 or B1, and A2 or B2); their complex
combinations (c +- i s) / sqrt(2) are Lz eigenfunctions. In those orbitals
the one- and two-electron integrals stay real, with the symmetries
(pq|rs) = (rs|pq) = (qp|sr) but no longer (pq|rs) = (qp|rs), and a
determinant's M_L, the sum of its orbitals' m, is conserved: an additive
label finer than the C2v irreps (A1 holds Sigma+ and Delta states, M_L = 0
holds Sigma states only).

`axial_orbitals` pairs the orbitals of a real calculation, `axial_integrals`
transforms h and eri into the complex orbitals, and `axial_natural_orbitals`
gives natural orbitals that keep the pairing. `SelectedCI(...,
orbital_lz=...)` takes the result.
"""
from dataclasses import dataclass

import numpy as np

from .basis import cart2sph, cart_components


def _polynomial_map(l, action):
    """The (nsph x nsph) matrix of a linear map of degree-l polynomials,
    `action(lx, ly, lz) -> {(lx', ly', lz'): coefficient}`, on the real
    spherical functions of `cart2sph`."""
    comps = cart_components(l)
    index = {c: i for i, c in enumerate(comps)}
    c2s = cart2sph(l)
    cart = np.zeros((len(comps), len(comps)))
    for column, comp in enumerate(comps):
        for target, value in action(*comp).items():
            cart[index[target], column] += value
    image = cart @ c2s
    matrix, *_ = np.linalg.lstsq(c2s, image, rcond=None)
    if np.abs(c2s @ matrix - image).max() > 1e-10:
        raise RuntimeError("the map leaves the spherical functions")
    return matrix


def _derivative(lx, ly, lz):
    """d/dphi = x d/dy - y d/dx on a monomial."""
    out = {}
    if ly:
        out[(lx + 1, ly - 1, lz)] = float(ly)
    if lx:
        out[(lx - 1, ly + 1, lz)] = out.get((lx - 1, ly + 1, lz), 0.0) - float(lx)
    return out


def _reflection(lx, ly, lz):
    """y -> -y on a monomial."""
    return {(lx, ly, lz): float((-1) ** ly)}


def _ao_operator(basis, action):
    nao = basis.nao
    out = np.zeros((nao, nao))
    offsets = basis.ao_offsets()
    cache = {}
    for k, shell in enumerate(basis.shells):
        if shell.l not in cache:
            cache[shell.l] = _polynomial_map(shell.l, action) if shell.l else np.array([[_scalar(action)]])
        o = int(offsets[k])
        n = 2 * shell.l + 1
        out[o:o + n, o:o + n] = cache[shell.l]
    return out


def _scalar(action):
    return sum(action(0, 0, 0).values())


@dataclass
class AxialOrbitals:
    """real: (n x n) orthogonal; the given orbitals times `real` are the same
    orbitals with each sin-type partner set exactly to the rotated cos-type
    one (in place, so labels and order carry over). transform: (n x n)
    complex; the axial orbitals in those aligned real ones (a cos orbital
    c at i and its partner s at j become (c + i s)/sqrt(2) at i, m > 0, and
    (c - i s)/sqrt(2) at j). lz: each axial orbital's m. mirror: the index
    of each orbital's image under the reflection y -> -y (m -> -m)."""
    real: np.ndarray
    transform: np.ndarray
    lz: np.ndarray
    mirror: np.ndarray


def _purified(a, blocks, tol):
    """(n x n) orthogonal: within each block, every orbital projected onto
    the |m| eigenspace (of the block's Lz^2 = a^T a) that holds most of it,
    then symmetrically orthonormalized: the nearest orbitals of one |m| each.
    SCF and natural orbitals mix orbitals of different |m| and equal C2v
    irrep (sigma and delta) at the level of their convergence."""
    n = a.shape[0]
    out = np.zeros((n, n))
    for block in blocks:
        block = np.asarray(block, dtype=np.int64)
        sub = a[:, block]
        lsq = sub.T @ sub                                  # Lz^2 within the block's span
        w, v = np.linalg.eigh(0.5 * (lsq + lsq.T))
        m = np.rint(np.sqrt(np.maximum(w, 0.0)))
        if np.abs(w - m * m).max() > 1e-3:
            raise ValueError("the orbitals' span is not closed under rotations about the axis")
        projected = np.zeros((len(block), len(block)))
        for value in np.unique(m):
            space = v[:, m == value]
            weight = np.einsum("ik,ik->i", space, space)
            projected[:, weight > 0.5] = (space @ space.T)[:, weight > 0.5]
        if np.abs(projected.T @ projected - np.eye(len(block))).max() > 0.5:
            raise ValueError("the orbitals are far from |m| eigenfunctions")
        u, _, vt = np.linalg.svd(projected)
        out[np.ix_(block, block)] = u @ vt
    if np.abs(out - np.eye(n)).max() > tol ** 0.5:
        raise ValueError("purifying the orbitals would change them by more than sqrt(tol)")
    return out


def axial_orbitals(basis, coefficients, *, overlap=None, blocks=None, tol=1e-6):
    """Pair the orbitals `coefficients` (AO x n, orthonormal) of a molecule
    whose atoms lie on the z axis into Lz eigenfunctions. Each orbital
    should have one |m| and a definite reflection parity (symmetry-adapted
    canonical or natural orbitals of a cylindrically symmetric state), and
    the span of each of `blocks` (index arrays, default all orbitals: pass
    the occupied and virtual ones to keep them apart) closed under rotations
    about the axis. Mixing of different |m| at the level of an SCF's
    convergence (below sqrt(tol) per coefficient) is projected out first;
    `real` includes that rotation."""
    from .integrals import overlap as ao_overlap

    coords = np.asarray(basis.coords, dtype=np.float64)
    if np.abs(coords[:, :2]).max() > 1e-8:
        raise ValueError("axial orbitals need every atom on the z axis")
    c = np.asarray(coefficients, dtype=np.float64)
    s = ao_overlap(basis) if overlap is None else np.asarray(overlap, dtype=np.float64)
    n = c.shape[1]
    if np.abs(c.T @ s @ c - np.eye(n)).max() > 1e-8:
        raise ValueError("the orbitals must be orthonormal")
    d_ao = _ao_operator(basis, _derivative)
    r_ao = _ao_operator(basis, _reflection)
    purify = _purified(c.T @ s @ (d_ao @ c), [np.arange(n)] if blocks is None else blocks, tol)
    c = c @ purify
    image = d_ao @ c                                      # d/dphi of each orbital (AO)
    a = c.T @ s @ image                                   # in the orbitals: antisymmetric
    full = np.einsum("pi,pq,qi->i", image, s, image)      # m^2 over the whole AO space
    kept = np.einsum("ji,ji->i", a, a)                    # m^2 within the orbitals
    if np.abs(full - kept).max() > tol:
        raise ValueError("the orbitals' span is not closed under rotations about the axis")
    m = np.rint(np.sqrt(np.maximum(full, 0.0))).astype(np.int64)
    if np.abs(full - m * m).max() > tol:
        raise ValueError("an orbital mixes different |m|")
    lsq = a.T @ a
    if np.abs(lsq - np.diag(np.diag(lsq))).max() > tol:
        raise ValueError("the orbitals mix different |m|")
    parity = np.einsum("pi,pq,qr,ri->i", c, s, r_ao, c)
    if np.abs(np.abs(parity) - 1.0).max() > tol:
        raise ValueError("an orbital has no definite parity under y -> -y")
    real = purify.copy()
    transform = np.zeros((n, n), dtype=np.complex128)
    lz = np.zeros(n, dtype=np.int64)
    mirror = np.arange(n)
    for k in np.flatnonzero(m == 0):
        if parity[k] < 0:
            raise ValueError("an m = 0 orbital odd under y -> -y")
        transform[k, k] = 1.0
    for value in np.unique(m[m > 0]):
        cos = np.flatnonzero((m == value) & (parity > 0))
        sin = np.flatnonzero((m == value) & (parity < 0))
        if len(cos) != len(sin):
            raise ValueError(f"|m| = {value}: {len(cos)} cos-type but {len(sin)} sin-type orbitals")
        # Partner of cos orbital c: s = -(d/dphi c) / m, within the sin-type span.
        partners = -a[np.ix_(sin, cos)] / value               # (sin x cos), orthogonal
        if np.abs(partners.T @ partners - np.eye(len(cos))).max() > tol:
            raise ValueError(f"|m| = {value}: the sin-type orbitals do not span the partners")
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-np.abs(partners))
        for t, k in zip(rows, cols):
            i, j = cos[k], sin[t]
            real[:, j] = purify[:, sin] @ partners[:, k]
            transform[i, i] = transform[i, j] = 1.0 / np.sqrt(2.0)
            transform[j, i] = 1j / np.sqrt(2.0)
            transform[j, j] = -1j / np.sqrt(2.0)
            lz[i], lz[j] = value, -value
            mirror[i], mirror[j] = j, i
    return AxialOrbitals(real, transform, lz, mirror)


def _pairs(transform):
    """(m = 0 indices, cos indices i, partner indices j) of a pair transform."""
    n = transform.shape[0]
    single, cos, sin = [], [], []
    for i in range(n):
        column = np.flatnonzero(np.abs(transform[:, i]) > 1e-12)
        if len(column) == 1:
            if column[0] != i or abs(transform[i, i] - 1.0) > 1e-12:
                raise ValueError("not an axial pair transform")
            single.append(i)
        elif abs(transform[i, i].imag) < 1e-12 and transform[i, i].real > 0 and transform[:, i].imag.max() > 0:
            j = int(column[column != i][0])
            cos.append(i)
            sin.append(j)
    return np.array(single, dtype=np.int64), np.array(cos, dtype=np.int64), np.array(sin, dtype=np.int64)


def _pair_axis(re, im, axis, cos, sin, conjugate):
    """Apply the pair transform to one index of a complex array (re, im) in place:
    new_i = (x_i + i x_j) / sqrt(2), new_j = (x_i - i x_j) / sqrt(2) for a
    ket index, the complex conjugate coefficients for a bra index."""
    sign = -1.0 if conjugate else 1.0
    r = np.moveaxis(re, axis, 0)
    m = np.moveaxis(im, axis, 0)
    ar, ai, br, bi = r[cos].copy(), m[cos].copy(), r[sin].copy(), m[sin].copy()
    f = 1.0 / np.sqrt(2.0)
    r[cos] = f * (ar - sign * bi)
    m[cos] = f * (ai + sign * br)
    r[sin] = f * (ar + sign * bi)
    m[sin] = f * (ai - sign * br)


def axial_integrals(h, eri, orbitals, *, tol=1e-6):
    """h and chemists' eri (real orbitals, as aligned by `orbitals.real`) in
    the axial orbitals of `orbitals` (`AxialOrbitals`): real arrays, with the
    M_L-forbidden entries set to exactly zero after checking they are below
    `tol` (a frozen core's potential carries its SCF's small departures from
    cylindrical symmetry into h), and h averaged with its mirror image."""
    t = orbitals.transform
    lz = np.asarray(orbitals.lz)
    _, cos, sin = _pairs(t)
    h = np.asarray(h, dtype=np.float64)
    n = h.shape[0]
    hc = t.conj().T @ h @ t
    if np.abs(hc.imag).max() > tol:
        raise ValueError("h is not real in the axial orbitals (orbitals not cylindrically symmetric?)")
    hc = hc.real.copy()
    forbidden = lz[:, None] != lz[None, :]
    if np.abs(hc[forbidden]).max(initial=0.0) > tol:
        raise ValueError("h couples orbitals of different m")
    hc[forbidden] = 0.0
    mirror = np.asarray(orbitals.mirror)
    hc = 0.5 * (hc + hc[np.ix_(mirror, mirror)])
    re = np.array(np.asarray(eri, dtype=np.float64).reshape(n, n, n, n))
    im = np.zeros_like(re)
    for axis, conjugate in ((0, True), (1, False), (2, True), (3, False)):
        _pair_axis(re, im, axis, cos, sin, conjugate)
    if np.abs(im).max() > tol:
        raise ValueError("eri is not real in the axial orbitals (orbitals not cylindrically symmetric?)")
    del im
    change = (-lz[:, None, None, None] + lz[None, :, None, None] - lz[None, None, :, None]
              + lz[None, None, None, :])
    forbidden = change != 0
    if np.abs(re[forbidden]).max(initial=0.0) > tol:
        raise ValueError("eri violates M_L conservation")
    re[forbidden] = 0.0
    return hc, re


def axial_natural_orbitals(density, lz, mirror):
    """(rotation, occupations): natural orbitals of a one-particle density in
    axial orbitals, averaged with its mirror image (the cylindrical average
    of a Pi or Delta state's single M_L component), diagonalized per m with
    the same real rotation for m and -m, so the pairing and labels carry
    over. Columns of `rotation` are the new orbitals in the old ones, by
    decreasing occupation within each m; `rotate_integrals(...,
    axial=True)` applies it. The same matrix rotates the aligned real
    orbitals (cos and sin partners alike) into their real counterparts."""
    density = np.asarray(density, dtype=np.float64)
    lz, mirror = np.asarray(lz), np.asarray(mirror)
    sym = 0.5 * (density + density[np.ix_(mirror, mirror)])
    n = len(lz)
    rotation = np.zeros((n, n))
    occupations = np.zeros(n)
    for value in np.unique(np.abs(lz)):
        plus = np.flatnonzero(lz == value)
        w, v = np.linalg.eigh(sym[np.ix_(plus, plus)])
        order = np.argsort(-w, kind="stable")
        w, v = w[order], v[:, order]
        for k in range(v.shape[1]):                     # deterministic signs
            big = np.argmax(np.abs(v[:, k]))
            if v[big, k] < 0:
                v[:, k] = -v[:, k]
        rotation[np.ix_(plus, plus)] = v
        occupations[plus] = w
        if value:
            minus = mirror[plus]
            rotation[np.ix_(minus, minus)] = v
            occupations[minus] = w
    return rotation, occupations


def axial_dipoles(dipoles, orbitals, *, tol=1e-8):
    """The x, y, z dipole matrices (real orbitals, aligned by `orbitals.real`)
    in the axial orbitals: z (Delta m = 0) and x (Delta m = +-1) are real and
    symmetric there; y is imaginary, and is returned as zeros (by the
    cylindrical symmetry alpha_yy = alpha_xx, which `SelectedCIResponse`
    fills in for axial orbitals)."""
    t = orbitals.transform
    lz = np.asarray(orbitals.lz)
    out = []
    for k, d in enumerate(np.asarray(dipoles, dtype=np.float64)):
        dc = t.conj().T @ d @ t
        if k == 1:
            if np.abs(dc.real).max() > tol:
                raise ValueError("the y dipole is not imaginary in the axial orbitals")
            out.append(np.zeros_like(d))
            continue
        if np.abs(dc.imag).max() > tol:
            raise ValueError("a dipole component is not real in the axial orbitals")
        dc = dc.real
        allowed = np.abs(lz[:, None] - lz[None, :]) == (1 if k == 0 else 0)
        if np.abs(dc[~allowed]).max(initial=0.0) > tol:
            raise ValueError("a dipole component breaks the axial selection rule")
        dc[~allowed] = 0.0
        out.append(0.5 * (dc + dc.T))
    return tuple(out)
