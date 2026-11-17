"""Direct packed spin-orbital contractions using the shared compiled CPU core.

Hamiltonian/T4 terms and five factorizable nonlinear terms enter here. Other
terms retain the shared tile executor. The dense spin-orbital implementation
stays independent.
"""
from functools import lru_cache
from math import factorial

import numpy as np

from . import _cpu_kernels
from ._cpu_triples import cpu_threads
from ._tensor_symmetry import signed_permutations
from ._spin_quadruples_tiles import _permutation_pairs, projection_diagrams
from .bounded_einsum import einsum


class PackedLookup:
    """Signed canonical addresses, shared by every tensor with this layout."""

    def __init__(self, layout, virtual=None):
        no, nv = 2 * layout.no, 2 * layout.nv
        keys = {key: i for i, key in enumerate(sorted(layout.groups))}
        if len(keys) > 255:
            raise ValueError("Too many spin/spatial sectors for the packed lookup")
        if virtual is None:
            self.virtual_map = np.zeros(nv**4, dtype=np.int32)
            self.virtual_key = np.zeros(nv**4, dtype=np.uint8)
            for key, particles in layout.groups.items():
                if len(particles) >= np.iinfo(np.int32).max:
                    raise OverflowError("Packed virtual addresses exceed int32")
                for order, sign in signed_permutations(4):
                    positions = np.ravel_multi_index(particles[:, order].T, (nv,) * 4)
                    self.virtual_map[positions] = sign * np.arange(1, len(particles) + 1)
                    self.virtual_key[positions] = keys[key]
        else:
            self.virtual_map, self.virtual_key = virtual.virtual_map, virtual.virtual_key
        self.hole_map = np.zeros(no**4, dtype=np.int64)
        self.hole_key = np.zeros(no**4, dtype=np.uint8)
        # Holes are much smaller than the virtual table. Build them in batches.
        holes = np.asarray(list(layout.slots), dtype=np.int32).reshape(-1, 4)
        offsets = np.asarray([part.start + 1 for _, part in layout.slots.values()], dtype=np.int64)
        sectors = np.asarray([keys[key] for key, _ in layout.slots.values()], dtype=np.uint8)
        for order, sign in signed_permutations(4):
            positions = np.ravel_multi_index(holes[:, order].T, (no,) * 4)
            self.hole_map[positions] = sign * offsets
            self.hole_key[positions] = sectors


@lru_cache(maxsize=None)
def packed_diagrams(rank):
    """Binary T4 terms and products reducible to a small effective coefficient."""
    selected = []
    for equivalent, multiplicity, diagrams in projection_diagrams(rank):
        for diagram in diagrams:
            coefficient, spec, names = diagram
            if names[-1] != "t4":
                continue
            lhs, output = spec.split("->")
            inputs = lhs.split(",")
            small, amplitude = inputs[:-1], inputs[-1]
            retained = set(output + amplitude)
            labels = "".join(dict.fromkeys(c for term in small for c in term if c in retained))
            if len(labels) > 4:
                continue  # leave six-index coefficients on the GEMM tile path
            # A product's retained indices inherit only the antisymmetries of
            # their original operands, not arbitrary halves of the new tensor.
            halves = tuple("".join(c for c in half if c in labels)
                           for term in small for half in (term[:len(term)//2], term[len(term)//2:]))
            selected.append((equivalent, multiplicity, diagram, ",".join(small) + "->" + labels,
                             labels + "," + amplitude + "->" + output, halves))
    return tuple(selected)


@lru_cache(maxsize=None)
def supported_diagrams(rank):
    return frozenset(item[2] for item in packed_diagrams(rank))


def _coefficient_rows(tensor, subscript, amplitude, output, halves):
    """Exact sparse rows; no numerical threshold or integral screening."""
    external = tuple(c for c in subscript if c in output)
    internal = tuple(c for c in subscript if c not in output)
    coordinates = np.array(np.nonzero(tensor)).T
    keep = np.ones(len(coordinates), dtype=bool)
    factor = 1
    # Contracted pairs antisymmetric in BOTH operands are summed only once.
    for first in halves:
        for second in (amplitude[:4], amplitude[4:]):
            group = [subscript.index(c) for c in first if c in internal and c in second]
            factor *= factorial(len(group))
            for a, b in zip(group, group[1:]):
                keep &= coordinates[:, a] < coordinates[:, b]
    coordinates = coordinates[keep]
    values = np.ascontiguousarray(tensor[tuple(coordinates.T)])
    extents = np.array([tensor.shape[subscript.index(c)] for c in external], dtype=np.int32)
    if external:
        groups = np.ravel_multi_index(coordinates[:, [subscript.index(c) for c in external]].T,
                                     tuple(extents))
        count = int(np.prod(extents))
    else:
        groups, count = np.zeros(len(coordinates), dtype=np.int64), 1
    order = np.argsort(groups, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(groups, minlength=count))].astype(np.int64)
    inside = np.zeros((len(coordinates), 4), dtype=np.int32)
    inside[:, :len(internal)] = coordinates[order][:, [subscript.index(c) for c in internal]]
    return external, internal, extents, starts, inside, values[order], factor


class PackedBinaryAction:
    """Action-local coefficient tables; immutable amplitude address maps are reused."""

    def __init__(self, engine, blocks, direction):
        self.engine = engine
        self.vectors = tuple(None if v is None else np.ascontiguousarray(v, dtype=np.float64)
                             for v in (engine.amplitudes[3], direction))
        self.blocks = blocks
        self.rows = {}

    def _maps(self, derivative):
        maps = self.engine.__dict__.setdefault("_packed_cpu_maps", [None, None])
        if maps[derivative] is None:
            layout = (self.engine._ground_t4, self.engine._direction_t4)[derivative]
            maps[derivative] = PackedLookup(layout, maps[1 - derivative])
        return maps[derivative]

    def _coefficient(self, names, spec, derivative):
        if len(names) == 1:
            pair = self.blocks[names[0]]
            return pair.tangent if derivative == 0 else pair.value
        # Ground-state intermediates are immutable for this EOM engine. Their
        # direction-dependent derivatives live only for the current action.
        if not hasattr(self.engine, "_packed_coefficients"):
            self.engine._packed_coefficients = {}
        cache = self.engine._packed_coefficients
        key = names, spec
        if derivative == 1:
            if key not in cache:
                cache[key] = einsum(np, spec, *(self.blocks[name].value for name in names))
            return cache[key]
        result = None
        for active, name in enumerate(names):
            if self.blocks[name].tangent is None:
                continue
            operands = [self.blocks[name].tangent if i == active else self.blocks[name].value
                        for i, name in enumerate(names)]
            part = einsum(np, spec, *operands)
            if result is None:
                result = part
            else:
                result += part
        return result

    def add(self, rank, coordinates, result):
        engine = self.engine
        coordinates = np.ascontiguousarray(coordinates, dtype=np.int32)
        if coordinates.shape != (result.size, 2 * rank) or not result.flags.c_contiguous:
            raise ValueError("Packed contraction needs contiguous canonical coordinates and output")
        if not len(result):
            return
        kernel = _cpu_kernels.library().spin_quad_contract
        for equivalent, multiplicity, diagram, small_spec, spec, halves in packed_diagrams(rank):
            coefficient, original_spec, names = diagram
            lhs, output = spec.split("->")
            small, amplitude = lhs.split(",")
            for derivative in range(2):
                vector = self.vectors[derivative]
                if vector is None:
                    continue
                key = original_spec, derivative
                if key not in self.rows:
                    tensor = self._coefficient(names[:-1], small_spec, derivative)
                    self.rows[key] = (None if tensor is None else
                                      _coefficient_rows(tensor, small, amplitude, output, halves))
                if self.rows[key] is None:
                    continue
                external, internal, extents, starts, inside, values, factor = self.rows[key]
                if not len(values):
                    continue
                maps = self._maps(derivative)
                for (p, sp), (h, sh) in _permutation_pairs(rank, equivalent):
                    positions = p + tuple(rank + i for i in h)
                    axes = np.array([positions[output.index(c)] for c in external], dtype=np.int32)
                    amp_axes = np.array([positions[output.index(c)] if c in output
                                         else -1 - internal.index(c) for c in amplitude], dtype=np.int32)
                    alpha = coefficient * multiplicity * factor * sp * sh / factorial(rank)**2
                    kernel(len(result), rank, coordinates.ctypes.data, 2 * engine.nocc, 2 * engine.nvir,
                           maps.virtual_map.ctypes.data, maps.virtual_key.ctypes.data,
                           maps.hole_map.ctypes.data, maps.hole_key.ctypes.data, vector.ctypes.data,
                           len(external), axes.ctypes.data, extents.ctypes.data, amp_axes.ctypes.data,
                           starts.ctypes.data, inside.ctypes.data, values.ctypes.data, alpha,
                           result.ctypes.data, cpu_threads(None))
