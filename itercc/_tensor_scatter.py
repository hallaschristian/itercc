"""Shared tensor expansion for EOM, response and spin-orbital methods.

Canonical antisymmetric coordinates and restricted spatial coordinates have
DIFFERENT symmetries. Use expand_antisymmetric only for the former; use
unpack_sector for distinct flat indices and apply the caller's projector.
Native helpers share the core loader, thread controls and NumPy fallback.
Index tables are internal, validated layout data: in bounds and unique
(canonical permutation orbits must also be disjoint).
"""
from functools import lru_cache
from math import prod

import numpy as np

from . import _cpu_kernels
from ._cpu_triples import cpu_threads


def unpack_sector(vector, indices, shape, *, xp=np, threads=None):
    """Expand distinct flat sector indices, retaining zeros outside the sector.

    Real float64 CPU tensors use parallel first-touch and scatter for large,
    sufficiently populated outputs with at least four workers. Small/sparse
    tensors and smaller thread pools avoid the parallel zeroing overhead;
    complex and device tensors retain the backend implementation.
    """
    if vector.ndim != 1 or indices.shape != vector.shape:
        raise ValueError("Sector vector and indices must be matching one-dimensional arrays")
    size = prod(shape)
    if (xp is np and isinstance(vector, np.ndarray) and vector.dtype == np.float64
            and size >= 2**18 and vector.size * 4 >= size
            and 4 <= cpu_threads(threads) <= size // 2**16):
        kernel = _cpu_kernels.library()
        if kernel is not None:
            values = np.ascontiguousarray(vector)
            positions = np.ascontiguousarray(indices, dtype=np.int64)
            result = np.empty(shape, dtype=np.float64)
            kernel.sector_unpack(size, values.size, positions.ctypes.data, values.ctypes.data,
                                 result.ctypes.data, cpu_threads(threads))
            return result
    result = xp.zeros(shape, dtype=xp.result_type(vector.dtype, xp.float64))
    result.reshape(-1)[indices] = vector
    return result


@lru_cache(maxsize=4)
def _antisymmetric_permutations(rank):
    from itertools import product
    from ._tensor_symmetry import signed_permutations
    rows, signs = [], []
    for (p, sp), (h, sh) in product(signed_permutations(rank), repeat=2):
        rows.append(p + tuple(rank + a for a in h))
        signs.append(sp * sh)
    return np.asarray(rows, dtype=np.int32), np.asarray(signs, dtype=np.int32)


def expand_antisymmetric(vector, coordinates, shape, *, threads=None):
    """Compiled expansion of canonical spin-orbital amplitudes, or None.

    Coordinates must enumerate distinct canonical particle/hole combinations;
    every permutation then writes a unique entry. This internal helper shares
    the CPU core and thread controls with the other tensor kernels.
    """
    rank = len(shape) // 2
    if not (isinstance(vector, np.ndarray) and vector.dtype == np.float64
            and 1 <= rank <= 4 and len(shape) == 2 * rank):
        return None
    kernel = _cpu_kernels.library()
    if kernel is None:
        return None
    nv, no = shape[0], shape[rank]
    if tuple(shape) != (nv,) * rank + (no,) * rank or coordinates.shape != (2 * rank, vector.size):
        raise ValueError("Canonical amplitudes and coordinates have incompatible shapes")
    coordinates = np.ascontiguousarray(coordinates.T, dtype=np.int32)
    vector = np.ascontiguousarray(vector)
    permutations, signs = _antisymmetric_permutations(rank)
    result = np.empty(shape, dtype=np.float64)
    kernel.spin_unpack(vector.size, rank, nv, no, coordinates.ctypes.data, vector.ctypes.data,
                       len(signs), permutations.ctypes.data, signs.ctypes.data, result.ctypes.data,
                       cpu_threads(threads))
    return result


def unpack_antisymmetric(vector, coordinates, shape, *, xp=np, threads=None):
    """Expand canonical spin-orbital coordinates on CPU or device.

    Backend selection and the reference fallback live here so callers do not
    duplicate permutation conventions or decide which native kernel to use.
    """
    if xp is np:
        compiled = expand_antisymmetric(vector, coordinates, shape, threads=threads)
        if compiled is not None:
            return compiled
    rank = len(shape) // 2
    permutations, signs = _antisymmetric_permutations(rank)
    result = xp.zeros(shape, dtype=vector.dtype)
    for axes, sign in zip(permutations, signs):
        result[tuple(coordinates[int(axis)] for axis in axes)] = int(sign) * vector
    return result


