"""Direct packed spin-orbital contractions using the shared compiled CPU core.

Hamiltonian/T4 terms and five factorizable nonlinear terms enter here. Other
terms retain the shared tile executor. The dense spin-orbital implementation
stays independent.
"""
from functools import lru_cache
from itertools import combinations, permutations
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


def opened_enabled():
    """Rank-four direct terms by opened letters (itercc_spin_quad_opened)
    unless ITERCC_DIRECT_OPENED=0 (then `dense_enabled` decides)."""
    return _runtime.enabled("ITERCC_DIRECT_OPENED")


def _order_sign(keys):
    """The parity of sorting distinct `keys`: +1 or -1."""
    sign = 1
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if keys[i] > keys[j]:
                sign = -sign
    return sign


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
OPENED_LAYOUT = np.dtype([(name, _i8) for name in (
    "no", "nv", "occupied_irrep", "virtual_irrep", "spin_change", "irrep", "hole_map", "own", "key_of",
    "group_start", "particles", "nown", "own_full", "own_compact", "own_rows", "buffer_rows")])
OPENED_FAMILY = np.dtype([(name, _i8) for name in (
    "kp", "kh", "xp", "xh", "hole_map", "vector", "spin_change", "irrep", "class_min", "nclasses", "sigma_start", "sigma",
    "x_start", "x", "x_row", "weights", "weight_offset", "b_start", "b_index", "v_table", "nterms",
    "term_alpha", "term_beta", "term_coef")])
OPENED_ROWS = np.dtype([(name, _i8) for name in (
    "no", "nv", "rank", "occupied_irrep", "virtual_irrep", "spin_change", "irrep", "rows", "part_start",
    "part_values", "part_offset", "hole_pos")])
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

    def add_rank_four(self, result, slots):
        """The rank-four direct terms of the output slots `slots` (indices of
        the direction layout's slots, increasing) added into `result`, the
        whole packed R4: one GEMM per hole tuple and class
        (itercc_spin_quad_opened), else by contexts per output slot or by
        per-row lookups (`add`) in runs of consecutive slots."""
        engine = self.engine
        layout = engine._direction_t4
        slots = np.asarray(slots, dtype=np.int64)
        if not len(slots):
            return
        if opened_enabled() and self._add_opened(result, slots):
            return
        # The direct packed terms of consecutive slots (windows follow each
        # other), at least 2^22 rows per call: the dense kernel runs a call's
        # slots in parallel (2^18 rows were 12 CuF slots for 56 threads), and
        # each row's sums do not depend on the chunks.
        items = list(layout.slots.items())
        chosen = [items[i] for i in slots]
        following = np.r_[True, np.diff(slots) == 1]          # rows follow the previous slot's
        done, start = len(chosen), 0
        while start < done:
            stop, count = start, 0
            while stop < done and count < 1 << 22 and (stop == start or following[stop]):
                count += len(layout.groups[chosen[stop][1][0]])
                stop += 1
            parts = []
            for holes, (key, _) in chosen[start:stop]:
                particles = layout.groups[key]
                coordinates = np.empty((len(particles), 8), dtype=np.int32)
                coordinates[:, :4], coordinates[:, 4:] = particles, holes
                parts.append(coordinates)
            first, last = chosen[start][1][1].start, chosen[stop - 1][1][1].stop
            self.add(4, parts[0] if len(parts) == 1 else np.concatenate(parts), result[first:last])
            start = stop

    def _opened_particles(self, kp):
        """B tables of `kp` opened particles (see itercc_spin_quad_opened): the
        increasing (4 - kp)-tuples of virtual spin orbitals sorted by class
        (alpha count * 8 + irrep), their starts, the index of each flat tuple,
        and the signed row + 1 of B followed by each flat s_p in its particle
        group. Kept on the engine (the virtual maps are shared)."""
        engine = self.engine
        cache = engine.__dict__.setdefault("_opened_particles", {})
        if kp in cache:
            return cache[kp]
        nv = 2 * engine.nvir
        virtual = np.asarray(engine._spatial_labels["v"], dtype=np.int64)
        tuples = np.array(list(combinations(range(nv), 4 - kp)), dtype=np.int64).reshape(-1, 4 - kp)
        classes = (tuples % 2 == 0).sum(axis=1) * 8 + np.bitwise_xor.reduce(virtual[tuples // 2], axis=1)
        order = np.argsort(classes, kind="stable")
        tuples, classes = tuples[order], classes[order]
        b_start = np.r_[0, np.cumsum(np.bincount(classes, minlength=40))].astype(np.int64)
        flat = np.ravel_multi_index(tuples.T, (nv,) * (4 - kp)).astype(np.int64)
        b_index = np.full(nv ** (4 - kp), -1, dtype=np.int32)
        b_index[flat] = np.arange(len(tuples), dtype=np.int32)
        virtual_map = self._maps(1).virtual_map
        v_table = np.ascontiguousarray(
            virtual_map[flat[:, None] * nv**kp + np.arange(nv**kp, dtype=np.int64)[None, :]], dtype=np.int32)
        values = np.zeros((len(tuples), 4), dtype=np.int32)
        values[:, :4 - kp] = tuples
        cache[kp] = b_start, b_index, v_table, values
        return cache[kp]

    def _opened_layout(self, slots):
        """The output side of itercc_spin_quad_opened for the target `slots`
        (kept on the engine for the last set of slots)."""
        engine = self.engine
        layout = engine._direction_t4
        kept = engine.__dict__.get("_opened_layout")
        if kept is not None and np.array_equal(kept[0], slots):
            return kept[1]
        no = 2 * engine.nocc
        groups = engine.__dict__.get("_opened_groups")
        if groups is None:
            keys = sorted(layout.groups)
            key_of = np.full(40, -1, dtype=np.int64)
            for i, (alpha, irrep) in enumerate(keys):
                key_of[alpha * 8 + irrep] = i
            group_start = np.r_[0, np.cumsum([len(layout.groups[k]) for k in keys])].astype(np.int64)
            particles = np.ascontiguousarray(np.concatenate([layout.groups[k] for k in keys]), dtype=np.int32)
            groups = engine._opened_groups = key_of, group_start, particles
        key_of, group_start, particles = groups
        items = list(layout.slots.items())
        holes = np.array([items[i][0] for i in slots], dtype=np.int64).reshape(-1, 4)
        full = np.array([items[i][1][1].start for i in slots], dtype=np.int64)
        rows = np.array([items[i][1][1].stop - items[i][1][1].start for i in slots], dtype=np.int64)
        compact = np.r_[0, np.cumsum(rows)[:-1]].astype(np.int64)
        own = np.zeros(no**4, dtype=np.int64)
        for order in permutations(range(4)):
            own[np.ravel_multi_index(holes[:, order].T, (no,) * 4)] = compact + 1
        units = np.ascontiguousarray(holes, dtype=np.int32)
        occupied = np.ascontiguousarray(engine._spatial_labels["o"], dtype=np.int64)
        virtual = np.ascontiguousarray(engine._spatial_labels["v"], dtype=np.int64)
        struct = np.zeros(1, dtype=OPENED_LAYOUT)
        maps = self._maps(1)
        values = dict(no=no, nv=2 * engine.nvir, occupied_irrep=occupied.ctypes.data,
                      virtual_irrep=virtual.ctypes.data, spin_change=layout.spin_change, irrep=layout.irrep,
                      hole_map=maps.hole_map.ctypes.data, own=own.ctypes.data, key_of=key_of.ctypes.data,
                      group_start=group_start.ctypes.data, particles=particles.ctypes.data, nown=len(slots),
                      own_full=full.ctypes.data, own_compact=compact.ctypes.data, own_rows=rows.ctypes.data,
                      buffer_rows=int(rows.sum()))
        for name, value in values.items():
            struct[name] = value
        # The arrays stay alive with the struct.
        built = dict(struct=struct, units=units, keep=(own, full, rows, compact, occupied, virtual, maps),
                     buffer_rows=int(rows.sum()))
        engine._opened_layout = (np.array(slots, copy=True), built)
        return built

    def _opened_tables(self, rank, number, derivative, equivalent, labels, amplitude, output, pairs, members):
        """Class tables of a merged coefficient of `rank` for
        itercc_spin_quad_opened(_lower) (None: no coefficient; False: not of
        the opened form). The ground-state ones (derivative 1) are kept on
        the engine."""
        key = "opened", rank, number, derivative
        engine = self.engine
        kept = engine.__dict__.setdefault("_packed_merged_rows", {}) if derivative == 1 else self.rows
        if key in kept:
            return kept[key]
        internal = [c for c in labels if c not in output]
        external = [c for c in labels if c in output]
        particle = [amplitude.index(c) < 4 for c in internal]
        kp, kh = sum(particle), len(internal) - sum(particle)
        outer = [output.index(c) < rank for c in external]
        xp, xh = sum(outer), len(outer) - sum(outer)
        if (max(kp, kh, xp, xh) > 2 or outer != sorted(outer, reverse=True)
                or rank - xp != 4 - kp or rank - xh != 4 - kh or (rank == 4 and (xp, xh) != (kp, kh))):
            kept[key] = False
            return False
        tensor = self._merged_tensor(derivative, members)
        if tensor is None:
            kept[key] = None
            return None
        occupied = np.asarray(engine._spatial_labels["o"], dtype=np.int64)
        virtual = np.asarray(engine._spatial_labels["v"], dtype=np.int64)
        target = engine._direction_t4
        source = (engine._ground_t4, engine._direction_t4)[derivative]
        factor = _pair_factor(pairs)
        # Terms: positions of the external letters, and coefficients with the
        # sign that brings the amplitude to (context increasing, summed letters).
        merged = {}
        for (p, sp), (h, sh) in _permutation_pairs(rank, equivalent):
            positions = p + tuple(rank + i for i in h)
            axes = [positions[output.index(c)] for c in external]
            alpha = tuple(a for a in axes if a < rank)
            beta = tuple(a - rank for a in axes if a >= rank)
            amp = [positions[output.index(c)] if c in output else -1 - internal.index(c) for c in amplitude]
            signs = [_order_sign([(0, a) if a >= 0 else (1, -1 - a) for a in half]) for half in (amp[:4], amp[4:])]
            merged[alpha, beta] = merged.get((alpha, beta), 0.0) + (
                factor * sp * sh / factorial(rank)**2 * signs[0] * signs[1])
        terms = [(alpha, beta, coef) for (alpha, beta), coef in merged.items() if coef != 0.0]
        term_alpha = np.zeros((max(len(terms), 1), 2), dtype=np.int64)
        term_beta = np.zeros((max(len(terms), 1), 2), dtype=np.int64)
        for t, (alpha, beta, _) in enumerate(terms):
            term_alpha[t, :len(alpha)], term_beta[t, :len(beta)] = alpha, beta
        term_coef = np.array([coef for _, _, coef in terms], dtype=np.float64)
        # Summed tuples by class, particles first.
        n_ext = int(np.prod(tensor.shape[:len(external)]))
        dims = tensor.shape[len(external):]
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
            alphas = (tuples[:, column] % 2 == 0).astype(np.int64)
            delta += alphas if is_particle else -alphas
            gamma ^= (virtual if is_particle else occupied)[tuples[:, column] // 2]
        class_min, nclasses = -kh, (len(internal) + 1) * 8
        classes = (delta - class_min) * 8 + gamma
        order = np.argsort(classes, kind="stable")
        sigma_start = np.r_[0, np.cumsum(np.bincount(classes, minlength=nclasses))].astype(np.int64)
        columns = ([i for i in range(len(internal)) if particle[i]]
                   + [i for i in range(len(internal)) if not particle[i]])
        sigma = np.zeros((len(tuples), 4), dtype=np.int32)
        sigma[:, :len(internal)] = tuples[order][:, columns]
        sigma_flat = flat[order]
        # External tuples (particles first): distinct, increasing where every
        # term takes them increasing, paired with the summed class they meet.
        grid = (np.stack(np.meshgrid(*(np.arange(d) for d in tensor.shape[:len(external)]), indexing="ij"),
                         axis=-1).reshape(-1, len(external)) if external else np.zeros((1, 0), dtype=np.int64))
        valid = np.ones(len(grid), dtype=bool)
        for first, size, index in ((0, xp, 0), (xp, xh, 1)):
            if size == 2:
                increasing = all(term[index][0] < term[index][1] for term in terms)
                valid &= grid[:, first] < grid[:, first + 1] if increasing else grid[:, first] != grid[:, first + 1]
        dx = np.zeros(len(grid), dtype=np.int64)
        gx = np.zeros(len(grid), dtype=np.int64)
        for column in range(len(external)):
            alphas = (grid[:, column] % 2 == 0).astype(np.int64)
            is_particle = column < xp
            dx += alphas if is_particle else -alphas
            gx ^= (virtual if is_particle else occupied)[grid[:, column] // 2]
        ds = dx - (target.spin_change - source.spin_change)
        gs = gx ^ target.irrep ^ source.irrep
        paired = (ds - class_min) * 8 + gs
        valid &= (ds >= class_min) & (paired < nclasses)
        x_flat = np.flatnonzero(valid)
        x_class = paired[valid]
        order = np.argsort(x_class, kind="stable")
        x_flat, x_class = x_flat[order], x_class[order]
        x_start = np.r_[0, np.cumsum(np.bincount(x_class, minlength=nclasses))].astype(np.int64)
        x = np.zeros((len(x_flat), 4), dtype=np.int32)
        x[:, :len(external)] = grid[x_flat]
        x_row = np.full(n_ext, -1, dtype=np.int32)
        x_row[x_flat] = np.arange(len(x_flat)) - x_start[x_class]
        matrix = tensor.reshape(n_ext, -1)
        blocks, offsets, at = [], [], 0
        for c in range(nclasses):
            block = np.ascontiguousarray(matrix[np.ix_(x_flat[x_start[c]:x_start[c + 1]],
                                                       sigma_flat[sigma_start[c]:sigma_start[c + 1]])])
            offsets.append(at)
            blocks.append(block.reshape(-1))
            at += block.size
        weights = np.ascontiguousarray(np.concatenate(blocks)) if at else np.zeros(1)
        b_start, b_index, v_table, _ = self._opened_particles(kp) if kp else (
            np.zeros(41, dtype=np.int64), np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.int32), None)
        tables = dict(kp=kp, kh=kh, xp=xp, xh=xh, source=source, class_min=class_min, nclasses=nclasses,
                      arrays=(sigma_start, sigma, x_start, x, x_row, weights,
                              np.asarray(offsets, dtype=np.int64), b_start, b_index, v_table,
                              term_alpha, term_beta, term_coef))
        kept[key] = tables
        return tables

    def _opened_units(self, kh):
        """Hole tuples J of (4 - kh) increasing occupied spin orbitals."""
        cache = self.engine.__dict__.setdefault("_opened_units", {})
        if kh not in cache:
            tuples = np.array(list(combinations(range(2 * self.engine.nocc), 4 - kh)), dtype=np.int32)
            units = np.zeros((len(tuples), 4), dtype=np.int32)
            units[:, :4 - kh] = tuples.reshape(len(tuples), 4 - kh)
            cache[kh] = units
        return cache[kh]

    def _add_opened(self, result, slots):
        """itercc_spin_quad_opened for every rank-four family (False: a family
        is not of the opened form; nothing is added then)."""
        families = []
        for number, (equivalent, labels, amplitude, output, pairs, members) in enumerate(merged_diagrams(4)):
            for derivative in range(2):
                if self.vectors[derivative] is None:
                    continue
                tables = self._opened_tables(4, number, derivative, equivalent, labels, amplitude, output,
                                             pairs, members)
                if tables is False:
                    return False
                if tables is not None:
                    families.append((derivative, tables))
        if not families:
            return True
        library = _cpu_kernels.library()
        built = self._opened_layout(slots)
        engine = self.engine
        need = max((3 if t["kh"] == 1 else 5 if t["kh"] == 2 else 0) for _, t in families) * built["buffer_rows"]
        # Up to five rows of the output per position set of the opened holes
        # (14 GB at CuF size): mapped for this call only, not kept between
        # actions (the kernel clears them on all threads anyway).
        buffers = np.empty(max(need, 1))
        threads = cpu_threads(None)
        for derivative, tables in families:
            struct = self._opened_struct(derivative, tables)
            units = built["units"] if tables["kh"] == 0 else self._opened_units(tables["kh"])
            library.spin_quad_opened(built["struct"].ctypes.data, struct.ctypes.data, len(units),
                                     units.ctypes.data, result.ctypes.data, buffers.ctypes.data, threads)
        return True

    def _opened_struct(self, derivative, tables):
        """The itercc_spin_quad_opened_family of a family's tables on this
        action's vector (the tables keep the arrays alive)."""
        (sigma_start, sigma, x_start, x, x_row, weights, offsets, b_start, b_index, v_table,
         term_alpha, term_beta, term_coef) = tables["arrays"]
        maps = self._maps(derivative)
        vector = self.vectors[derivative]
        struct = np.zeros(1, dtype=OPENED_FAMILY)
        values = dict(kp=tables["kp"], kh=tables["kh"], xp=tables["xp"], xh=tables["xh"],
                      hole_map=maps.hole_map.ctypes.data,
                      vector=vector.ctypes.data, spin_change=tables["source"].spin_change,
                      irrep=tables["source"].irrep, class_min=tables["class_min"],
                      nclasses=tables["nclasses"], sigma_start=sigma_start.ctypes.data,
                      sigma=sigma.ctypes.data, x_start=x_start.ctypes.data, x=x.ctypes.data,
                      x_row=x_row.ctypes.data, weights=weights.ctypes.data, weight_offset=offsets.ctypes.data,
                      b_start=b_start.ctypes.data, b_index=b_index.ctypes.data, v_table=v_table.ctypes.data,
                      nterms=len(term_coef), term_alpha=term_alpha.ctypes.data,
                      term_beta=term_beta.ctypes.data, term_coef=term_coef.ctypes.data)
        for name, value in values.items():
            struct[name] = value
        return struct

    def _opened_rows(self, rank):
        """The particle-major rows of `rank` (two or three) for
        itercc_spin_quad_opened_lower, kept on the engine."""
        engine = self.engine
        cache = engine.__dict__.setdefault("_opened_rows", {})
        if rank in cache:
            return cache[rank]
        no, nv = 2 * engine.nocc, 2 * engine.nvir
        indices = np.asarray(engine.space.indices[rank - 1])
        part_start, _, _, part_values = self._opened_particles(4 - rank)
        rows = indices.shape[1]
        particles = np.ravel_multi_index(indices[:rank].astype(np.int64), (nv,) * rank)
        holes = np.ravel_multi_index(indices[rank:].astype(np.int64), (no,) * rank)
        first = np.full(nv**rank, -1, dtype=np.int64)
        unique, start = np.unique(particles, return_index=True)
        first[unique] = start
        hole_pos = np.full(no**rank, -1, dtype=np.int64)
        hole_pos[holes] = np.arange(rows) - first[particles]
        part_offset = first[np.ravel_multi_index(part_values[:, :rank].T.astype(np.int64), (nv,) * rank)]
        occupied = np.ascontiguousarray(engine._spatial_labels["o"], dtype=np.int64)
        virtual = np.ascontiguousarray(engine._spatial_labels["v"], dtype=np.int64)
        struct = np.zeros(1, dtype=OPENED_ROWS)
        values = dict(no=no, nv=nv, rank=rank, occupied_irrep=occupied.ctypes.data,
                      virtual_irrep=virtual.ctypes.data, spin_change=engine._direction_t4.spin_change,
                      irrep=engine._direction_t4.irrep, rows=rows, part_start=part_start.ctypes.data,
                      part_values=part_values.ctypes.data, part_offset=part_offset.ctypes.data,
                      hole_pos=hole_pos.ctypes.data)
        for name, value in values.items():
            struct[name] = value
        cache[rank] = dict(struct=struct, rows=rows, keep=(part_offset, hole_pos, occupied, virtual, part_values))
        return cache[rank]

    def add_lower(self, rank, out):
        """The direct packed terms of rank two or three into `out` (that
        rank's canonical rows): by opened letters (itercc_spin_quad_opened_lower)
        unless ITERCC_DIRECT_OPENED=0 or a family is not of that form, else by
        per-row lookups (`add`)."""
        if opened_enabled() and self._add_opened_lower(rank, out):
            return
        self.add(rank, self.engine.space.indices[rank - 1].T, out)

    def _add_opened_lower(self, rank, out):
        families = []
        for number, (equivalent, labels, amplitude, output, pairs, members) in enumerate(merged_diagrams(rank)):
            for derivative in range(2):
                if self.vectors[derivative] is None:
                    continue
                tables = self._opened_tables(rank, number, derivative, equivalent, labels, amplitude, output,
                                             pairs, members)
                if tables is False:
                    return False
                if tables is not None:
                    families.append((derivative, tables))
        if not families:
            return True
        library = _cpu_kernels.library()
        layout = self._opened_rows(rank)
        patterns = max({0: 1, 1: rank, 2: rank * (rank - 1) // 2}[t["xh"]] for _, t in families)
        buffers = np.empty(max((patterns - 1) * layout["rows"], 1))
        threads = cpu_threads(None)
        for derivative, tables in families:
            struct = self._opened_struct(derivative, tables)
            units = self._opened_units(tables["kh"])
            library.spin_quad_opened_lower(layout["struct"].ctypes.data, struct.ctypes.data, len(units),
                                           units.ctypes.data, out.ctypes.data, buffers.ctypes.data, threads)
        return True

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
