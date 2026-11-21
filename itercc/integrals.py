"""Native integrals on the compiled core: Gaussian AO integrals and transformations.

`mo_eri` turns atomic-orbital two-electron integrals into the molecular-orbital
integrals the solvers take, replacing PySCF's `ao2mo` (and the unpacking of its
packed output): two half transformations, each a pair of large GEMMs per batch
of index pairs on the core's executor, around compiled packing and scatter
passes. The result is written once, in the solvers' layout, and with orbital
irreps the symmetry-forbidden entries are exact zeros, so the solvers' symmetry
projection keeps it without a copy.

The Gaussian AO integrals (`overlap`, `kinetic`, `nuclear`, `dipole`, `pvp`,
`prp`, `eri_s8`) come from the core's McMurchie-Davidson kernels, and
`SpinFreeX2C` builds the scalar-relativistic (sfX2C-1e) one-electron
Hamiltonian and picture-changed dipoles from them.
"""
import numpy as np

from . import _cpu_kernels
from ._cpu_triples import matmul, product_into
from ._runtime import cpu_threads


def _pairs(n):
    return n * (n + 1) // 2


def _ao_pair_matrix(ao_eri, nao, library, threads):
    """The symmetric (pair x pair) AO integral matrix (mu nu|lambda sigma),
    mu >= nu, lambda >= sigma, from 8-fold packed, 4-fold packed or full input."""
    npair = _pairs(nao)
    ao_eri = np.asarray(ao_eri, dtype=np.float64)
    if ao_eri.ndim == 1:
        if ao_eri.size != npair * (npair + 1) // 2:
            raise ValueError("8-fold packed AO integrals of the wrong size")
        s4 = np.empty((npair, npair))
        ao_eri = np.ascontiguousarray(ao_eri)          # held while the kernel reads it
        library.unpack_s8_s4(ao_eri.ctypes.data, npair, s4.ctypes.data, threads)
        return s4
    if ao_eri.shape == (npair, npair):
        return np.ascontiguousarray(ao_eri)
    if ao_eri.shape == (nao,) * 4:
        rows, cols = np.tril_indices(nao)
        return np.ascontiguousarray(ao_eri[rows, cols][:, rows, cols])
    raise ValueError("AO integrals must be 8-fold packed, (pair x pair) or (n, n, n, n)")


def _half(rows, coeff, coeff_t, nao, nmo, batch, library, threads, sink):
    """For every packed row of `rows` (pairs of AO indices), the n x n block
    C^T M C of its symmetric AO matrix M, handed to sink(first, blocks)."""
    count = len(rows)
    square = np.empty((batch, nao, nao))
    half = np.empty((batch, nao, nmo))
    blocks = np.empty((batch, nmo, nmo))
    for first in range(0, count, batch):
        size = min(batch, count - first)
        part = rows[first:first + size]
        library.unpack_pairs(part.ctypes.data, size, rows.shape[1], nao, square.ctypes.data, threads)
        matmul(square[:size].reshape(size * nao, nao), coeff, half[:size].reshape(size * nao, nmo),
               threads=threads)
        if not product_into("mk,ikn->imn", coeff_t, half[:size], blocks[:size], 1.0, 0.0, threads):
            np.matmul(coeff_t, half[:size], out=blocks[:size])
        sink(first, blocks[:size])


def mo_eri(ao_eri, coeff, *, orbital_irreps=None, threads=None, batch_bytes=256 << 20):
    """(pq|rs) of the orbitals `coeff` (AO x n), chemists' notation, a C-ordered
    float64 n^4 array, from AO two-electron integrals: 8-fold packed (PySCF's
    `mf._eri`, `mol.intor("int2e", aosym="s8")`), 4-fold packed (pair x pair)
    or a full (N, N, N, N) array. With `orbital_irreps` (the Abelian irrep of
    each orbital) the symmetry-forbidden entries are exact zeros. About
    2 N^5 multiply-adds for N basis functions and orbitals, on all threads."""
    library = _cpu_kernels.library()
    if library is None:
        raise RuntimeError("the native integral transformation needs the compiled core")
    coeff = np.ascontiguousarray(coeff, dtype=np.float64)
    if coeff.ndim != 2:
        raise ValueError("coeff must be (AO x orbitals)")
    nao, nmo = coeff.shape
    threads = cpu_threads(threads)
    labels = None
    if orbital_irreps is not None:
        labels = np.ascontiguousarray(orbital_irreps, dtype=np.int64)
        if labels.shape != (nmo,) or labels.min() < 0 or labels.max() > 7:
            raise ValueError("orbital_irreps must hold one Abelian irrep (0..7) per orbital")
    coeff_t = np.ascontiguousarray(coeff.T)
    s4 = _ao_pair_matrix(ao_eri, nao, library, threads)
    npair, mpair = _pairs(nao), _pairs(nmo)
    batch = max(1, min(npair, batch_bytes // (8 * nao * max(nao, nmo))))
    # First half: (mu nu|lambda sigma) -> (mu nu|rs), packed over r >= s.
    half = np.empty((npair, mpair))

    def store(first, blocks):
        library.pack_pairs(blocks.ctypes.data, len(blocks), nmo, half[first:].ctypes.data, mpair, threads)

    _half(s4, coeff, coeff_t, nao, nmo, batch, library, threads, store)
    del s4
    # Second half over the rows (rs|..): (rs|pq) for all pq, written at [r, s] and [s, r].
    from ._cpu_quadruples import copy

    rows = copy(half.T)
    del half
    out = np.empty((nmo,) * 4)

    def scatter(first, blocks):
        library.scatter_pairs(blocks.ctypes.data, len(blocks), first, nmo,
                              None if labels is None else labels.ctypes.data, out.ctypes.data, threads)

    _half(rows, coeff, coeff_t, nao, nmo, batch, library, threads, scatter)
    return out


def _shell_arrays(basis):
    """The flat arrays the core's integral kernels take for a `basis.Basis`."""
    from .basis import cart2sph

    shells = basis.shells
    lmax = max(s.l for s in shells)
    l = np.array([s.l for s in shells], dtype=np.int32)
    atom = np.array([s.atom for s in shells], dtype=np.int32)
    nprim = np.array([len(s.exponents) for s in shells], dtype=np.int32)
    prim0 = np.zeros(len(shells), dtype=np.int64)
    prim0[1:] = np.cumsum(nprim[:-1])
    exps = np.concatenate([s.exponents for s in shells]).astype(np.float64)
    coefs = np.concatenate([s.coefficients for s in shells]).astype(np.float64)
    blocks = [np.ascontiguousarray(cart2sph(k)) for k in range(lmax + 1)]
    c2s_off = np.zeros(lmax + 1, dtype=np.int64)
    c2s_off[1:] = np.cumsum([b.size for b in blocks[:-1]])
    c2s = np.concatenate([b.ravel() for b in blocks])
    xyz = np.ascontiguousarray(basis.coords, dtype=np.float64)
    return dict(l=l, atom=atom, nprim=nprim, prim0=prim0, exps=exps, coefs=coefs, xyz=xyz, c2s=c2s,
                c2s_off=c2s_off, ao_off=basis.ao_offsets(), nao=basis.nao)


def _int1e(basis, kind, origin=(0.0, 0.0, 0.0), threads=None):
    library = _cpu_kernels.library()
    a = _shell_arrays(basis)
    nao = a["nao"]
    out = np.empty((3 if kind in (3, 5) else 1, nao, nao))
    charges = np.ascontiguousarray(basis.charges, dtype=np.float64)
    origin = np.ascontiguousarray(origin, dtype=np.float64)
    library.int1e(kind, len(a["l"]), a["l"].ctypes.data, a["atom"].ctypes.data, a["nprim"].ctypes.data,
                  a["prim0"].ctypes.data, a["exps"].ctypes.data, a["coefs"].ctypes.data, a["xyz"].ctypes.data,
                  charges.ctypes.data, len(charges), origin.ctypes.data, a["c2s"].ctypes.data,
                  a["c2s_off"].ctypes.data, a["ao_off"].ctypes.data, nao, out.ctypes.data, cpu_threads(threads))
    return out if kind in (3, 5) else out[0]


def overlap(basis, threads=None):
    """<mu|nu> of the spherical AOs of `basis` (a `basis.Basis`)."""
    return _int1e(basis, 0, threads=threads)


def kinetic(basis, threads=None):
    """<mu|-1/2 nabla^2|nu>."""
    return _int1e(basis, 1, threads=threads)


def nuclear(basis, threads=None):
    """<mu|-sum_C Z_C / |r - C||nu>."""
    return _int1e(basis, 2, threads=threads)


def dipole(basis, origin=(0.0, 0.0, 0.0), threads=None):
    """(3, nao, nao) <mu|r - origin|nu> (x, y, z)."""
    return _int1e(basis, 3, origin, threads=threads)


def pvp(basis, threads=None):
    """<p mu|V|p nu> = sum_k <d_k mu|-sum_C Z_C / |r - C||d_k nu>, point
    nuclei (libcint's int1e_pnucp): the small-component potential of X2C."""
    return _int1e(basis, 4, threads=threads)


def prp(basis, origin=(0.0, 0.0, 0.0), threads=None):
    """(3, nao, nao) <p mu|r - origin|p nu> = sum_k <d_k mu|(r - origin)_c|d_k nu>
    (c = x, y, z): the spin-free part of libcint's int1e_sprsp, the
    small-component block of the X2C picture-changed dipole."""
    return _int1e(basis, 5, origin, threads=threads)


def eri_s8(basis, *, screen=1e-14, threads=None):
    """(mu nu|la si), 8-fold packed (PySCF's aosym="s8" layout), shell quartets
    with a Schwarz bound below `screen` zero."""
    library = _cpu_kernels.library()
    a = _shell_arrays(basis)
    nao = a["nao"]
    npair = nao * (nao + 1) // 2
    out = np.empty(npair * (npair + 1) // 2)
    library.int2e_s8(len(a["l"]), a["l"].ctypes.data, a["atom"].ctypes.data, a["nprim"].ctypes.data,
                     a["prim0"].ctypes.data, a["exps"].ctypes.data, a["coefs"].ctypes.data, a["xyz"].ctypes.data,
                     a["c2s"].ctypes.data, a["c2s_off"].ctypes.data, a["ao_off"].ctypes.data, nao, float(screen),
                     out.ctypes.data, cpu_threads(threads))
    return out


LIGHT_SPEED = 137.03599967994
"""The speed of light in atomic units that X2C uses: PySCF's
`lib.param.LIGHT_SPEED` (CODATA 2006's 1/alpha, 137.035999679, with its
uncertainty digits appended), so X2C energies agree with PySCF's; CODATA 2018's
137.035999084 moves the RHF energy of CuF by about 5e-8 Eh."""


class SpinFreeX2C:
    """Spin-free one-electron exact two-component Hamiltonian (sfX2C-1e) of a
    `basis.Basis`: PySCF's `scf.RHF(mol).sfx2c1e()` (approx="1e",
    xuncontract=True, point nuclei).

    The modified Dirac equation is solved once in the decontracted basis
    (`Basis.decontracted`; S, T, V and W = pVp of its primitives):

        [V  T            ] [A]   [S  0          ] [A]
        [T  W / 4c^2 - T ] [B] = [0  T / 2c^2   ] [B] E

    Its electronic (upper) half gives the decoupling X = B A^-1 and the
    renormalization R = S^-1/2 (S^-1/2 S~ S^-1/2)^-1/2 S^1/2, S~ = S + X^T T X
    / 2c^2. `hcore()` is the Foldy-Wouthuysen one-electron Hamiltonian R^T (V +
    TX + X^T T - X^T T X + X^T W X / 4c^2) R, formed from the electronic
    eigenpairs as PySCF does, projected onto the contracted functions by the
    contraction coefficients. The two-electron integrals stay the
    nonrelativistic Coulomb ones, as in PySCF. `dipole()` gives picture-changed
    dipole integrals (`picture_change`); `itercc.integrals.dipole` stays the
    unrelaxed (nonrelativistic-picture) operator."""

    def __init__(self, basis, *, light_speed=LIGHT_SPEED, threads=None):
        self.basis = basis
        self.light_speed = float(light_speed)
        self.threads = threads
        self.primitives, self.contraction = basis.decontracted()
        p = self.primitives
        self.s, self.t = overlap(p, threads), kinetic(p, threads)
        self.v, self.w = nuclear(p, threads), pvp(p, threads)
        self._xr = None

    def _contract(self, matrix):
        c = self.contraction
        return c.T @ matrix @ c

    def _electronic(self):
        """Energies and large and small components of the electronic
        solutions of the modified Dirac equation (normalized in its metric)."""
        from scipy.linalg import LinAlgError, eigh

        c, n = self.light_speed, len(self.s)
        h, m = np.zeros((2 * n, 2 * n)), np.zeros((2 * n, 2 * n))
        h[:n, :n] = self.v
        h[:n, n:] = h[n:, :n] = self.t
        h[n:, n:] = self.w * (0.25 / c**2) - self.t
        m[:n, :n] = self.s
        m[n:, n:] = self.t * (0.5 / c**2)
        try:
            e, a = eigh(h, m)
            return e[n:], a[:n, n:], a[n:, n:]
        except LinAlgError:                  # a linearly dependent primitive basis: PySCF's fallback
            d, u = eigh(m)
            keep = d > 1e-9
            u = u[:, keep] / np.sqrt(d[keep])
            e, a = eigh(u.T @ h @ u)
            a = u @ a
            keep = e > -c**2
            return e[keep], a[:n, keep], a[n:, keep]

    def hcore(self):
        """The sfX2C-1e one-electron Hamiltonian over the contracted AOs."""
        from scipy.linalg import eigh

        e, large, _ = self._electronic()
        w, u = eigh(large.T @ self.s @ large)
        keep = w > 1e-14
        r = (u[:, keep] / np.sqrt(w[keep])) @ u[:, keep].T @ large.T @ self.s
        return self._contract((r.T * e) @ r)

    def decoupling(self):
        """(X, R) over the primitive AOs."""
        if self._xr is None:
            from scipy.linalg import eigh, solve

            _, large, small = self._electronic()
            if large.shape[0] == large.shape[1]:
                x = solve(large.T, small.T).T
            else:
                x = small @ solve(large.T @ large, large.T)
            c = self.light_speed
            s = self.s
            tilde = s + x.T @ self.t @ x * (0.5 / c**2)
            w, v = eigh(s)
            keep = w > 1e-14
            v, root = v[:, keep], np.sqrt(w[keep])
            middle = (v.T @ tilde @ v) / np.outer(root, root)
            w1, v1 = eigh(middle)
            keep1 = w1 > 1e-14
            middle = (v1[:, keep1] / np.sqrt(w1[keep1])) @ v1[:, keep1].T
            r = v @ (middle / root[:, None] * root[None, :]) @ v.T
            self._xr = (x, r)
        return self._xr

    def picture_change(self, large, small):
        """The X2C picture of an even one-electron operator over the contracted
        AOs: `large` its matrix (or stack of matrices) over the primitive AOs,
        `small` the same of <sigma.p mu|O|sigma.p nu> (its spin-free part,
        without the 1 / 4c^2): R^T (O_LL + X^T O_SS X / 4c^2) R, contracted."""
        x, r = self.decoupling()
        scale = 0.25 / self.light_speed**2
        large, small = np.asarray(large, dtype=np.float64), np.asarray(small, dtype=np.float64)
        out = [self._contract(r.T @ (o + scale * (x.T @ w @ x)) @ r)
               for o, w in zip(large.reshape(-1, *large.shape[-2:]), small.reshape(-1, *small.shape[-2:]))]
        return np.array(out).reshape(large.shape[:-2] + out[0].shape)

    def dipole(self, origin=(0.0, 0.0, 0.0)):
        """(3, nao, nao) picture-changed <mu|r - origin|nu> (x, y, z); the
        electronic dipole operator is its negative."""
        p = self.primitives
        return self.picture_change(dipole(p, origin, self.threads), prp(p, origin, self.threads))
