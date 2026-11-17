"""Native integral transformations on the compiled core.

`mo_eri` turns atomic-orbital two-electron integrals into the molecular-orbital
integrals the solvers take, replacing PySCF's `ao2mo` (and the unpacking of its
packed output): two half transformations, each a pair of large GEMMs per batch
of index pairs on the core's executor, around compiled packing and scatter
passes. The result is written once, in the solvers' layout, and with orbital
irreps the symmetry-forbidden entries are exact zeros, so the solvers' symmetry
projection keeps it without a copy.
"""
import numpy as np

from . import _cpu_kernels
from ._cpu_triples import cpu_threads, matmul, product_into


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
        library.unpack_s8_s4(np.ascontiguousarray(ao_eri).ctypes.data, npair, s4.ctypes.data, threads)
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
    if library is None or not hasattr(library, "scatter_pairs"):
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
