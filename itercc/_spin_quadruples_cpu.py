"""Direct packed spin-orbital contractions using the shared compiled CPU core.

Hamiltonian/T4 terms and five factorizable nonlinear terms enter here. Other
terms retain the shared tile executor. The dense spin-orbital implementation
stays independent.
"""
from functools import lru_cache
from math import factorial

import numpy as np

from . import _cpu_kernels
from ._runtime import cpu_threads
from ._tensor_symmetry import signed_permutations
from ._spin_quadruples_tiles import _permutation_pairs, projection_diagrams
from .bounded_einsum import einsum
from . import _runtime


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


@lru_cache(maxsize=None)
def merged_diagrams(rank):
    """The packed diagrams grouped by the T4 entries they read: diagrams of one
    projection group whose coefficients, after renaming the summed letters in
    the order of the T4 subscript, have the same letters, T4 pattern and
    antisymmetric summed pairs read the same entries in the same pattern, so
    their coefficients add into one table (one lookup per entry instead of
    one per diagram; at CuF size gvovo T4 and (goovv t2) T4 each made 9,300
    lookups per output row).

    Each group is (equivalent, labels, amplitude, output, pairs, members): the
    merged coefficient has subscript `labels` (output letters, then the
    summed letters P, Q, R, S), `pairs` are its antisymmetric summed pairs as
    `_coefficient_rows` halves, and members are (scale, diagram, small_spec,
    order): the diagram's coefficient (`small_spec`) transposed by `order`,
    times `scale` (its coefficient and multiplicity)."""
    groups = {}
    for equivalent, multiplicity, diagram, small_spec, spec, halves in packed_diagrams(rank):
        coefficient, _, _ = diagram
        lhs, output = spec.split("->")
        labels, amplitude = lhs.split(",")
        internal = [c for c in amplitude if c in labels and c not in output]
        if any(c not in output and c not in internal for c in labels) or len(internal) > 4:
            internal = None
        if internal is None:
            key = ("alone", len(groups))
            groups[key] = (equivalent, labels, amplitude, output, halves, [(coefficient * multiplicity, diagram,
                                                                           small_spec, tuple(range(len(labels))))])
            continue
        rename = {c: c for c in output} | {c: "PQRS"[k] for k, c in enumerate(internal)}
        canonical = sorted(labels, key=lambda c: output.index(c) if c in output else len(output) + internal.index(c))
        merged_labels = "".join(rename[c] for c in canonical)
        order = tuple(labels.index(c) for c in canonical)
        merged_amplitude = "".join(rename.get(c, c) for c in amplitude)
        # Only summed letters of one half that share a T4 half matter (the
        # factor and ordering of _coefficient_rows).
        pairs = []
        for half in halves:
            for side in (amplitude[:4], amplitude[4:]):
                group = "".join(rename[c] for c in half if c in internal and c in side)
                if len(group) > 1:
                    pairs.append(group)
        pairs = tuple(sorted(pairs))
        key = (equivalent, merged_labels, merged_amplitude, output, pairs)
        if key not in groups:
            groups[key] = (equivalent, merged_labels, merged_amplitude, output, pairs, [])
        groups[key][5].append((coefficient * multiplicity, diagram, small_spec, order))
    return tuple((*group[:5], tuple(group[5])) for group in groups.values())


def _pair_factor(pairs):
    """The summed antisymmetric pairs counted once: their orderings."""
    factor = 1
    for pair in pairs:
        factor *= factorial(len(pair))
    return factor


def dense_enabled():
    """Rank-four direct terms by contexts (itercc_spin_quad_dense) unless
    ITERCC_DIRECT_DENSE=0 (then the sparse per-row lookups)."""
    return _runtime.enabled("ITERCC_DIRECT_DENSE")


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


_i8 = np.int64
FAMILY = np.dtype([("virtual_map", _i8), ("virtual_key", _i8), ("hole_map", _i8), ("hole_key", _i8),
                   ("packed", _i8), ("spin_change", _i8), ("irrep", _i8), ("class_min", _i8), ("nclasses", _i8),
                   ("class_start", _i8), ("internal", _i8), ("weights", _i8), ("weight_offset", _i8),
                   ("n_ext", _i8)])
DENSE_TERM = np.dtype([("family", _i8), ("nexternal", _i8), ("external_axes", _i8, 4),
                       ("external_strides", _i8, 4), ("amplitude_axes", _i8, 8), ("alpha", np.float64)])
TERM = np.dtype([("virtual_map", _i8), ("virtual_key", _i8), ("hole_map", _i8), ("hole_key", _i8),
                 ("packed", _i8), ("nexternal", _i8), ("external_axes", _i8, 8),
                 ("external_extents", _i8, 8), ("amplitude_axes", _i8, 8), ("starts", _i8),
                 ("internal", _i8), ("values", _i8), ("alpha", np.float64)])


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

    def _merged_tensor(self, derivative, members):
        """The merged coefficient of a group (see `merged_diagrams`), or None."""
        tensor = None
        for scale, diagram, small_spec, order in members:
            part = self._coefficient(diagram[2][:-1], small_spec, derivative)
            if part is None:
                continue
            part = scale * np.transpose(part, order)
            tensor = part if tensor is None else tensor + part
        return None if tensor is None else np.ascontiguousarray(tensor)

    def _merged_rows(self, rank, number, derivative, labels, amplitude, output, pairs, members):
        """Sparse rows of a merged coefficient (see `merged_diagrams`), or None.
        The ground-state ones (derivative 1) do not depend on the direction and
        are kept on the engine across actions."""
        key = rank, number, derivative
        kept = self.engine.__dict__.setdefault("_packed_merged_rows", {}) if derivative == 1 else self.rows
        if key in kept:
            return kept[key]
        tensor = self._merged_tensor(derivative, members)
        kept[key] = None if tensor is None else _coefficient_rows(tensor, labels, amplitude, output, pairs)
        return kept[key]

    def _dense_family(self, rank, number, derivative, labels, amplitude, output, pairs, members):
        """Class tables of a merged coefficient for itercc_spin_quad_dense, or
        None: its summed tuples (in the order of `labels`, ordered within each
        antisymmetric pair as `_coefficient_rows` keeps them) sorted by
        symmetry class, and its blocks of external values by tuples."""
        key = "dense", rank, number, derivative
        kept = self.engine.__dict__.setdefault("_packed_merged_rows", {}) if derivative == 1 else self.rows
        if key in kept:
            return kept[key]
        tensor = self._merged_tensor(derivative, members)
        if tensor is None:
            kept[key] = None
            return None
        engine = self.engine
        occupied = np.asarray(engine._spatial_labels["o"], dtype=np.int64)
        virtual = np.asarray(engine._spatial_labels["v"], dtype=np.int64)
        internal = [c for c in labels if c not in output]
        particle = [amplitude.index(c) < 4 for c in internal]
        n_ext = int(np.prod(tensor.shape[:len(labels) - len(internal)]))
        dims = tensor.shape[len(labels) - len(internal):]
        tuples = np.stack(np.meshgrid(*(np.arange(d) for d in dims), indexing="ij"), axis=-1).reshape(-1, len(dims))
        keep = np.ones(len(tuples), dtype=bool)
        for pair in pairs:
            columns = [internal.index(c) for c in pair]
            for a, b in zip(columns, columns[1:]):
                keep &= tuples[:, a] < tuples[:, b]
        flat = np.flatnonzero(keep)
        tuples = tuples[keep]
        delta = np.zeros(len(tuples), dtype=np.int64)
        gamma = np.zeros(len(tuples), dtype=np.int64)
        for column, is_particle in enumerate(particle):
            spins = tuples[:, column] % 2 == 0
            if is_particle:
                delta += spins
                gamma ^= virtual[tuples[:, column] // 2]
            else:
                delta -= spins
                gamma ^= occupied[tuples[:, column] // 2]
        class_min = -sum(not x for x in particle)
        nclasses = (len(internal) + 1) * 8
        classes = (delta - class_min) * 8 + gamma
        order = np.argsort(classes, kind="stable")
        counts = np.bincount(classes, minlength=nclasses)
        class_start = np.ascontiguousarray(np.r_[0, np.cumsum(counts)], dtype=np.int64)
        values = np.zeros((len(tuples), 4), dtype=np.int32)
        values[:, :len(internal)] = tuples[order]
        weights_all = tensor.reshape(n_ext, -1)[:, flat[order]]
        blocks, offsets, at = [], [], 0
        for c in range(nclasses):
            block = np.ascontiguousarray(weights_all[:, class_start[c]:class_start[c + 1]])
            offsets.append(at)
            blocks.append(block.reshape(-1))
            at += block.size
        weights = np.ascontiguousarray(np.concatenate(blocks)) if at else np.zeros(1)
        layout = (engine._ground_t4, engine._direction_t4)[derivative]
        maps = self._maps(derivative)
        family = dict(arrays=(class_start, values, weights, np.asarray(offsets, dtype=np.int64)),
                      spin_change=layout.spin_change, irrep=layout.irrep, class_min=class_min,
                      nclasses=nclasses, n_ext=n_ext, maps=maps, factor=_pair_factor(pairs),
                      internal=tuple(internal))
        kept[key] = family
        return family

    def _add_dense(self, rank, coordinates, result, library):
        """The rank-four direct terms by contexts (see itercc_spin_quad_dense)."""
        engine = self.engine
        families, terms = [], []
        for number, (equivalent, labels, amplitude, output, pairs, members) in enumerate(merged_diagrams(rank)):
            for derivative in range(2):
                vector = self.vectors[derivative]
                if vector is None:
                    continue
                family = self._dense_family(rank, number, derivative, labels, amplitude, output, pairs, members)
                if family is None:
                    continue
                maps = family["maps"]
                class_start, values, weights, offsets = family["arrays"]
                index = len(families)
                families.append((maps.virtual_map.ctypes.data, maps.virtual_key.ctypes.data,
                                 maps.hole_map.ctypes.data, maps.hole_key.ctypes.data, vector.ctypes.data,
                                 family["spin_change"], family["irrep"], family["class_min"], family["nclasses"],
                                 class_start.ctypes.data, values.ctypes.data, weights.ctypes.data,
                                 offsets.ctypes.data, family["n_ext"]))
                internal = family["internal"]
                external = [c for c in labels if c in output]
                dims = [2 * (engine.nvir if output.index(c) < rank else engine.nocc) for c in external]
                strides = [int(np.prod(dims[k + 1:])) for k in range(len(dims))]
                for (p, sp), (h, sh) in _permutation_pairs(rank, equivalent):
                    positions = p + tuple(rank + i for i in h)
                    axes = [positions[output.index(c)] for c in external]
                    amp_axes = [positions[output.index(c)] if c in output else -1 - internal.index(c)
                                for c in amplitude]
                    terms.append((index, len(external), tuple(axes) + (0,) * (4 - len(axes)),
                                  tuple(strides) + (0,) * (4 - len(strides)), tuple(amp_axes),
                                  family["factor"] * sp * sh / factorial(rank)**2))
        if not terms:
            return
        family_table = np.array(families, dtype=FAMILY)
        term_table = np.array(terms, dtype=DENSE_TERM)
        occupied = np.ascontiguousarray(engine._spatial_labels["o"], dtype=np.int64)
        virtual = np.ascontiguousarray(engine._spatial_labels["v"], dtype=np.int64)
        library.spin_quad_dense(len(result), coordinates.ctypes.data, 2 * engine.nocc, 2 * engine.nvir,
                                occupied.ctypes.data, virtual.ctypes.data, len(family_table),
                                family_table.ctypes.data, len(term_table), term_table.ctypes.data,
                                result.ctypes.data, cpu_threads(None))

    def add(self, rank, coordinates, result):
        engine = self.engine
        coordinates = np.ascontiguousarray(coordinates, dtype=np.int32)
        if coordinates.shape != (result.size, 2 * rank) or not result.flags.c_contiguous:
            raise ValueError("Packed contraction needs contiguous canonical coordinates and output")
        if not len(result):
            return
        library = _cpu_kernels.library()
        terms = []          # one itercc_spin_quad_term per kernel call, in order
        dense = (rank == 4 and dense_enabled()
                 and 2 * max(engine.nocc, engine.nvir) < 255)
        if dense:
            self._add_dense(rank, coordinates, result, library)
            return
        for number, (equivalent, labels, amplitude, output, pairs, members) in enumerate(merged_diagrams(rank)):
            for derivative in range(2):
                vector = self.vectors[derivative]
                if vector is None:
                    continue
                rows = self._merged_rows(rank, number, derivative, labels, amplitude, output, pairs, members)
                if rows is None:
                    continue
                external, internal, extents, starts, inside, values, factor = rows
                if not len(values):
                    continue
                maps = self._maps(derivative)
                for (p, sp), (h, sh) in _permutation_pairs(rank, equivalent):
                    positions = p + tuple(rank + i for i in h)
                    axes = np.array([positions[output.index(c)] for c in external], dtype=np.int32)
                    amp_axes = np.array([positions[output.index(c)] if c in output
                                         else -1 - internal.index(c) for c in amplitude], dtype=np.int32)
                    alpha = factor * sp * sh / factorial(rank)**2
                    terms.append((maps.virtual_map.ctypes.data, maps.virtual_key.ctypes.data,
                                  maps.hole_map.ctypes.data, maps.hole_key.ctypes.data, vector.ctypes.data,
                                  len(external), tuple(axes) + (0,) * (8 - len(axes)),
                                  tuple(extents) + (0,) * (8 - len(extents)), tuple(amp_axes),
                                  starts.ctypes.data, inside.ctypes.data, values.ctypes.data, alpha))
        if terms:
            # Every term in one pass over the rows (the arrays above stay alive
            # in self.rows, the maps and the vectors).
            table = np.array(terms, dtype=TERM)
            library.spin_quad_contract_terms(len(result), rank, coordinates.ctypes.data, 2 * engine.nocc,
                                             2 * engine.nvir, len(table), table.ctypes.data,
                                             result.ctypes.data, cpu_threads(None))
