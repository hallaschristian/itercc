"""Shared tensor scatter/gather for EOM, response and spin-orbital methods.

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
from ._runtime import cpu_threads


def signed_gather_rows(vector, bases, maps, width, *, threads=None):
    """Gather signed one-based address maps into dense rows, or return None.

    Layout-owned maps must be contiguous int32 arrays of the same width and
    addresses must be valid for vector. A zero base ignores its map entirely.
    This primitive is independent of excitation rank, spin and spatial symmetry.
    """
    if not (isinstance(vector, np.ndarray) and vector.dtype == np.float64
            and vector.ndim == 1 and vector.flags.c_contiguous and np.dtype(np.intp).itemsize == 8):
        return None
    kernel = _cpu_kernels.library()
    if kernel is None:
        return None
    bases = np.ascontiguousarray(bases, dtype=np.int64)
    if len(maps) != len(bases):
        raise ValueError("Each gathered row needs a base and address map")
    for base, mapping in zip(bases, maps):
        if base and not (isinstance(mapping, np.ndarray) and mapping.dtype == np.int32
                         and mapping.shape == (width,) and mapping.flags.c_contiguous):
            raise ValueError("Nonzero rows need contiguous int32 maps of the requested width")
    pointers = np.asarray([0 if m is None else m.ctypes.data for m in maps], dtype=np.int64)
    output = np.empty((len(bases), width), dtype=np.float64)
    kernel.signed_gather_rows(len(bases), width, bases.ctypes.data, pointers.ctypes.data,
                              vector.ctypes.data, output.ctypes.data, cpu_threads(threads))
    return output


class SignedRows:
    """A float64 tensor kept as the signed rows of `signed_gather_rows`.

    Row r of `width` entries reads `vector` through `maps[r]` from base
    `bases[r]` (zero: a zero row). The rows form a row-major array of shape
    `lengths + entry_shape`; `shape` and `strides` describe that array with its
    axes permuted by `axes`, as if it existed. The CPU symmetry-blocked engine
    reads the rows in place (`itercc_sym_contract_packed`); `dense()` gathers
    them. Maps are layout-owned int32 arrays (see `signed_gather_rows`), given
    as a list or, with `pointers`, as their addresses (int64 per row, zero for
    a zero row) with `maps` any object that keeps them alive.
    """
    dtype = np.dtype(np.float64)

    def __init__(self, vector, bases, maps, width, lengths, entry_shape, axes, *, pointers=None):
        self.vector = vector
        self.bases = np.ascontiguousarray(bases, dtype=np.int64)
        self.maps = maps
        self.width = int(width)
        if pointers is None:
            for base, mapping in zip(self.bases, maps):
                if base and not (isinstance(mapping, np.ndarray) and mapping.dtype == np.int32
                                 and mapping.shape == (self.width,) and mapping.flags.c_contiguous):
                    raise ValueError("Nonzero rows need contiguous int32 maps of the row width")
            pointers = [0 if m is None else m.ctypes.data for m in maps]
        self.pointers = np.ascontiguousarray(pointers, dtype=np.int64)
        layout = tuple(lengths) + tuple(entry_shape)
        if prod(entry_shape) != self.width or prod(lengths) != len(self.bases):
            raise ValueError("Signed rows do not match their dense shape")
        steps = [8] * len(layout)
        for i in range(len(layout) - 2, -1, -1):
            steps[i] = steps[i + 1] * layout[i + 1]
        self._layout, self._axes = layout, tuple(axes)
        self.shape = tuple(layout[a] for a in axes)
        self.strides = tuple(steps[a] for a in axes)
        self.ndim = len(self.shape)
        self.nbytes = self.bases.nbytes + self.pointers.nbytes

    def dense(self):
        """The tensor itself (gathered on the compiled core)."""
        kernel = _cpu_kernels.library()
        if kernel is None:
            raise RuntimeError("Signed rows need the compiled core")
        rows = np.empty((len(self.bases), self.width), dtype=np.float64)
        kernel.signed_gather_rows(len(self.bases), self.width, self.bases.ctypes.data, self.pointers.ctypes.data,
                                  self.vector.ctypes.data, rows.ctypes.data, cpu_threads(None))
        return rows.reshape(self._layout).transpose(self._axes)


@lru_cache(maxsize=256)
def _gather_steps(terms, strides):
    offsets = np.zeros((len(terms), len(strides)), dtype=np.int64)
    for row, (axes, _) in enumerate(terms):
        offsets[row, list(axes)] = strides
    return offsets, np.asarray([weight for _, weight in terms], dtype=np.float64)


class PermutedGather:
    """Reusable coordinates for ordered antisymmetry/sector gathers on the CPU.

    Rows are distinct and coordinates are in bounds, guaranteed by the layout.
    Negative source strides are supported. No source tensor or result is cached.
    """
    def __init__(self, rows, coordinates):
        self.rows = np.ascontiguousarray(rows, dtype=np.int64)
        self.coordinates = np.ascontiguousarray(np.asarray(coordinates).T, dtype=np.int32)
        if self.rows.ndim != 1 or self.coordinates.ndim != 2 or len(self.coordinates) != len(self.rows):
            raise ValueError("Gather coordinates and rows must have the same length")

    def add(self, source, target, terms, *, threads=None):
        if not (isinstance(source, np.ndarray) and isinstance(target, np.ndarray)
                and source.dtype == target.dtype == np.float64
                and target.ndim == 1 and target.flags.c_contiguous and target.flags.writeable):
            return False
        kernel = _cpu_kernels.library()
        if kernel is None:
            return False
        if (source.ndim != self.coordinates.shape[1] or np.may_share_memory(source, target)
                or any(stride % 8 for stride in source.strides)):
            return False
        terms = tuple((tuple(axes), float(weight)) for axes, weight in terms)
        if any(sorted(axes) != list(range(source.ndim)) for axes, _ in terms):
            raise ValueError("Gather terms must permute all source axes")
        strides, weights = _gather_steps(terms, tuple(s // 8 for s in source.strides))
        kernel.permuted_gather_add(len(self.rows), source.ndim, len(terms),
                                  self.coordinates.ctypes.data, self.rows.ctypes.data,
                                  strides.ctypes.data, weights.ctypes.data, source.ctypes.data,
                                  target.ctypes.data, cpu_threads(threads))
        return True


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


