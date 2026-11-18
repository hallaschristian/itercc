"""Compiled rank-four slot program of the triplet EOM-CCSDTQ action on the CPU.

For every occupied slot and projection group, the rank-four part of one
action contracts a few dozen raw virtual tiles (one per external spin pattern
and occupied permutation), each from up to a few hundred small products, and
gathers every tile into the slot's packed rows. Driven from Python, each
product pays tens of microseconds of interpreter work and its own parallel
region, and the GIL keeps the slots serial. `SlotProgram` records the work
once and runs every tile of an action in the compiled core
(`itercc_triplet_slots`): tiles in parallel, each on one thread with the
single-threaded contraction routes and its raw tile in that thread's caches.

* A tile kind is a projection group and a raw spin pattern. Its products
  depend on the tile's occupied tuple only through operand offsets (affine in
  it: the dense spin blocks), the packed T4 rows they select and the irrep
  pattern of their plans, so a kind is recorded once with the tuple symbolic.
  Plans are made per distinct contraction and irrep pattern of the occupied
  indices it reads, for every pattern the slots need.
* A slot class (hole spins, particle spin groups present, repeated occupied
  indices) fixes a slot's tiles: their kinds, occupied permutations and
  gathers. Classes are recorded once; the core expands each slot's tiles.
* Intermediates (the inner products of three- and four-operand diagrams) are
  shared between tiles through one cache keyed by their expression and the
  occupied indices they depend on; a tile waits for one that another thread
  is computing, and helps with it when it is large: its producer cuts the
  contraction into the pieces of the threaded executor, which the waiting
  tiles run too (the g T4 intermediates at CuF size, about 2e8 multiply-adds
  each, were computed on one thread while dozens of tiles waited).
* Tile products form only the entries their projection reads (letters of
  an antisymmetric subset in increasing spin-orbital order), so they read an
  intermediate only at such entries: an intermediate is formed only where
  every product that reads it, in any kind, reads (`_order_intermediates`;
  the g T4 products of the rings, read at b < c < d, shrink six-fold).
* Intermediates and tiles keep their virtual axes in irrep order (orbitals
  sorted by irrep, `PositionedView`): a symmetry class is a contiguous run,
  so a product reads whole cache lines of them instead of every fourth value
  (C2v orbitals in energy order). Composites and sums are those of the plain
  layout, so the action does not change.
* Each tile's gathers go to partial sums, added to the slot's rows in the
  order of the Python loop and divided by (4!)^2 as it does, so the action is
  the loop's bit for bit, on any number of threads.

Slots run in waves that bound the partial sums (as many bytes as the
intermediate cache, at least `_PARTIAL_DOUBLES` doubles); a wave's tiles run
sorted by occupied tuple, so tiles that share intermediates run together.
"""
import ctypes
from itertools import product
from math import prod

import numpy as np

from . import _cpu_kernels, _runtime
from ._spin_quadruples_tiles import (_PACKED_TILE_BYTES, _packed_cuts, ordered_letters, _tree_irrep, contraction_schedule,
                                     external_routes, routes)
from ._tensor_scatter import SignedRows, _gather_steps
from .symmetry_blocks import PositionedView

DENSE, ROWS, INTER, TILE = range(4)
_VIRTUAL = frozenset("abcdefgh")      # virtual letters of the diagrams
CONTRACT, LOOKUP, PUBLISH = range(3)
_ALL = (None, None)
_FIXED = "@"                   # (_FIXED, q): the bounds of the tile's q-th occupied index
_PARTIAL_DOUBLES = 1 << 25      # partial sums of a wave of slots (256 MiB)

_i8 = np.int64
_OPERAND = [("kind", _i8), ("index", _i8), ("coef", _i8, 5)]
INSTRUCTION = np.dtype([("op", _i8), ("inter", _i8), ("next", _i8), ("plan", _i8),
                        ("a", _OPERAND), ("b", _OPERAND), ("out", _OPERAND),
                        ("a_rows", _i8), ("a_sum", _i8), ("b_sum", _i8), ("b_cols", _i8),
                        ("rpos", _i8), ("cpos", _i8), ("flags", _i8), ("alpha", np.float64)])
INTERMEDIATE = np.dtype([("key", _i8), ("size", _i8), ("nvars", _i8), ("positions", _i8, 4)])
ROWS_DESCRIPTOR = np.dtype([("which", _i8), ("spins", _i8, 4), ("fixed", _i8, 4), ("start", _i8, 4),
                            ("count", _i8, 4), ("pointers", _i8)])
TEMPLATE = np.dtype([("kind", _i8), ("positions", _i8, 4), ("part_start", _i8), ("part_count", _i8)])
PART = np.dtype([("group", _i8), ("terms", _i8)])
TERMS = np.dtype([("offsets", _i8), ("weights", _i8), ("count", _i8)])
GROUP = np.dtype([("rows", _i8), ("coordinates", _i8), ("count", _i8)])
SLOT = np.dtype([("cls", _i8), ("key", _i8), ("start", _i8), ("length", _i8), ("occupied", _i8, 4)])

_ARRAYS = ("kind_start", "kind_count", "inter_start", "inter_count", "instructions", "intermediates", "rows",
           "plan_start", "plan_table", "plan_count", "plan_id", "ground_bases", "ground_keys",
           "direction_bases", "direction_keys", "occupied_irreps", "class_start", "class_count",
           "templates", "parts", "terms", "key_group_start", "groups", "slots", "fuse_start", "fuse_count",
           "fuse_groups", "fuse_members", "tile_runs")


class ProgramStruct(ctypes.Structure):
    """itercc_slot_program of core.h."""
    _fields_ = ([("nv", ctypes.c_int64), ("holes", ctypes.c_int64), ("nirrep", ctypes.c_int64)]
                + [(name, ctypes.c_void_p) for name in _ARRAYS] + [("tile_irrep", ctypes.c_int64)])


_PROGRESS = ctypes.CFUNCTYPE(None, ctypes.c_int64, ctypes.c_int64)


class Unsupported(Exception):
    """A product the compiled program cannot run (the Python loop runs the action)."""


_STORE = np.zeros(1)


def _like(shape, strides):
    """A float64 array of this shape and strides for planning (never read)."""
    return np.lib.stride_tricks.as_strided(_STORE, shape=tuple(shape), strides=tuple(strides), writeable=True)


def _rows_like(shape, strides):
    stub = object.__new__(SignedRows)
    stub.shape, stub.strides, stub.ndim = tuple(shape), tuple(strides), len(shape)
    return stub


def _c_strides(shape):
    steps, step = [], 8
    for n in reversed(shape):
        steps.append(step)
        step *= n
    return tuple(reversed(steps))


def _symbolic(key):
    """(canonical key, fixed positions): the tile's occupied indices in `key`
    become variables numbered by first appearance."""
    positions = []

    def walk(x):
        if isinstance(x, tuple):
            if len(x) == 2 and x[0] == _FIXED:
                if x[1] not in positions:
                    positions.append(x[1])
                return ("var", positions.index(x[1]))
            return tuple(walk(y) for y in x)
        return x
    return walk(key), tuple(positions)


class _Kind:
    """The recorded products of one tile kind (see the module docstring)."""

    def __init__(self):
        self.code = []          # instruction tuples
        self.inters = []        # (template id, positions, shape)
        self.contracts = []     # (contraction number, fixed positions of its labels)


class SlotProgram:
    """The rank-four slots of a `TripletEOMCCSDTQ` engine as one compiled
    program (see the module docstring). Built for one set of tangent operands
    present and one set of operand shapes (`matches`)."""

    def __init__(self, engine, blocks, groups, presence):
        self.library = _cpu_kernels.library()
        if self.library is None or not hasattr(self.library, "triplet_slots"):
            raise Unsupported
        if engine._symmetry_engine is None or engine.nocc < 1:
            raise Unsupported
        self.blocked = engine._symmetry_engine
        self.no, self.nv = engine.nocc, engine.nvir
        self.nirrep = int(self.blocked.nirrep)
        self.irrep = engine.space.irrep
        # Stored position of each virtual orbital in the program's own arrays
        # (intermediates, tiles): irrep order, stable (ITERCC_SLOT_POSITIONS=0: none).
        virtual = np.asarray(engine._spatial_labels["v"], dtype=np.int64)
        self._vpos = None
        if _runtime.enabled("ITERCC_SLOT_POSITIONS") and np.any(np.diff(virtual) < 0):
            self._vpos = np.empty(len(virtual), dtype=np.int64)
            self._vpos[np.argsort(virtual, kind="stable")] = np.arange(len(virtual))
        self.slab = engine.occupied_slab
        self.occupied_irreps = np.ascontiguousarray(engine._spatial_labels["o"], dtype=np.int64)
        if self.occupied_irreps.size and (self.occupied_irreps.min() < 0
                                          or self.occupied_irreps.max() >= self.nirrep):
            raise Unsupported
        self.groups = groups                   # projection groups with their tile diagrams
        self.presence = presence
        self.signature = self._signature(blocks)
        self._bases, self._base_shapes = {}, {}
        for name, pair in sorted(blocks.items()):
            for side, array in enumerate((pair.value, pair.tangent)):
                if array is not None:
                    if array.dtype != np.float64:
                        raise Unsupported
                    self._bases[name, side] = len(self._bases)
                    self._base_shapes[name, side] = (array.shape, array.strides)
        self._rows = {}                        # T4 row descriptors
        self._templates = {}                   # intermediate expressions
        self._kinds, self._kind_of = [], {}
        self._contractions = {}                # distinct contraction -> number
        self._contraction_keys = []            # number -> its identity in `_contractions`
        self._contraction_info = []            # (spec, operand arrays, labels, irreps, structure, ordered)
        self._plans_of = {}                    # (contraction, irreps of its occupied indices) -> plan
        self._plan_sets = []
        self._keep = []
        self._prepare(engine)

    @staticmethod
    def _signature(blocks):
        return tuple((name, side, array.shape, array.strides, array.dtype.char)
                     for name, pair in sorted(blocks.items())
                     for side, array in enumerate((pair.value, pair.tangent)) if array is not None)

    def matches(self, blocks, presence):
        return presence == self.presence and self._signature(blocks) == self.signature

    # ---- recording tile kinds -------------------------------------------

    def _dense(self, name, side, signature, letters, selected):
        array_shape, array_strides = self._base_shapes.get((name, side), (None, None))
        if array_shape is None or len(array_shape) != len(letters):
            raise Unsupported
        coef = [0] * 5
        shape, strides = [], []
        for c, spin, n, step in zip(letters, signature, array_shape, array_strides):
            if step % 8 or step < 0:
                raise Unsupported
            step //= 8
            coef[0] += spin * step                  # the spin block: [spin::2]
            extent, step = (n - spin + 1) // 2, 2 * step
            bound = selected.get(c)
            if bound is None:
                shape.append(extent)
            elif bound[0] == _FIXED:
                coef[1 + bound[1]] += step
                shape.append(1)
            else:
                coef[0] += step * bound[0]
                shape.append(bound[1] - bound[0])
            strides.append(8 * step)
        return (DENSE, self._bases[name, side], tuple(shape), tuple(strides), tuple(coef))

    def _t4(self, which, signature, letters, selected):
        cuts = []
        for c in letters[4:]:
            bound = selected.get(c)
            if bound is None:
                cuts.append((-1, 0, self.no))
            elif bound[0] == _FIXED:
                cuts.append((bound[1], 0, 1))
            else:
                cuts.append((-1, bound[0], bound[1] - bound[0]))
        key = (which, tuple(signature), tuple(cuts))
        index = self._rows.setdefault(key, len(self._rows))
        lengths = tuple(count for _, _, count in cuts)
        layout = lengths + (self.nv,) * 4
        steps = _c_strides(layout)
        axes = (4, 5, 6, 7, 0, 1, 2, 3)
        return (ROWS, index, tuple(layout[a] for a in axes), tuple(steps[a] for a in axes), None)

    def _contract(self, kind, spec, a, b, out, alpha, beta, selected, irreps, ordered=()):
        if a[0] == ROWS:                       # packed rows are read as the second operand
            if b[0] == ROWS:
                raise Unsupported
            inputs, output = spec.split("->")
            first, second = inputs.split(",")
            spec, a, b, irreps = f"{second},{first}->{output}", b, a, (irreps[1], irreps[0])
        letters = set(spec) - set(",->")
        labels, positions = [], []
        for c in sorted(letters):
            bound = selected.get(c)
            if bound is None:
                continue
            if bound[0] == _FIXED:
                if bound[1] not in positions:
                    positions.append(bound[1])
                bound = (_FIXED, positions.index(bound[1]))
            labels.append((c, bound))
        irreps = (int(irreps[0]), int(irreps[1]))
        # The program's own arrays (intermediates, the tile) keep virtual axes
        # in irrep order (see the module docstring).
        sides = spec.replace("->", ",").split(",")
        operands = tuple((x[0] == ROWS, x[2], x[3],
                          self._vpos is not None and x[0] in (INTER, TILE)
                          and tuple(c in _VIRTUAL and n == self.nv for c, n in zip(side, x[2])))
                         for x, side in zip((a, b, out), sides))
        identity = (spec, operands, tuple(labels), irreps, tuple(ordered))
        number = self._number(identity)
        kind.code.append((CONTRACT, -1, -1, len(kind.contracts), a, b, out, float(alpha), beta))
        kind.contracts.append((number, tuple(positions)))

    def _number(self, identity):
        """The contraction number of an identity (spec, operands, labels,
        irreps, ordered), registered on first use."""
        number = self._contractions.get(identity)
        if number is None:
            spec, operands, labels, irreps, ordered = identity
            number = self._contractions[identity] = len(self._contraction_info)
            arrays = tuple(_rows_like(shape, strides) if rows else
                           PositionedView(_like(shape, strides), [self._vpos if v else None for v in placed])
                           if placed else _like(shape, strides)
                           for rows, shape, strides, placed in operands)
            self._contraction_info.append([spec, arrays, labels, irreps, None, ordered])
            self._contraction_keys.append(identity)
        return number

    def _order_intermediates(self):
        """Form each intermediate only where every product reading it reads.

        A product writes only the entries of its output allowed by its
        ordering constraints, so it reads an operand only where the
        constraints among the operand's letters that are also output letters
        hold. An intermediate (one template, shared through the cache by tiles
        of every kind) keeps the constraints all its readers share (a strict
        one where every reader's is strict), found from the tile products down
        through nested intermediates; its product is then recorded with them.
        The entries it forms are those it formed before, with the same
        arithmetic, so the action does not change."""
        found = {}                      # template identity -> {(x, y): strict}

        def readers():
            needed = {}                 # identity -> list of constraint dicts, one per reader
            for kind in self._kinds:
                for op, _, _, plan, a, b, out, _, _ in kind.code:
                    if op != CONTRACT:
                        continue
                    info = self._contraction_info[kind.contracts[plan][0]]
                    spec = info[0]
                    letters_a, letters_b = spec.split("->")[0].split(",")
                    output = spec.split("->")[1]
                    if out[0] == TILE:
                        constraints = {(x, y): strict for x, y, strict in info[5]}
                    else:
                        constraints = found.get(kind.inters[out[1]][0], {})
                    for operand, letters in ((a, letters_a), (b, letters_b)):
                        if operand[0] != INTER:
                            continue
                        kept = {(x, y): strict for (x, y), strict in constraints.items()
                                if x in letters and y in letters and x in output and y in output}
                        needed.setdefault(kind.inters[operand[1]][0], []).append(kept)
            result = {}
            for identity, lists in needed.items():
                common = dict(lists[0])
                for other in lists[1:]:
                    common = {pair: strict and other[pair] for pair, strict in common.items() if pair in other}
                if common:
                    result[identity] = common
            return result

        for _ in range(8):              # nested intermediates: a few levels
            updated = readers()
            if updated == found:
                break
            found = updated
        else:
            return                      # no fixed point: keep every entry
        if not found:
            return
        for kind in self._kinds:
            for op, _, _, plan, a, b, out, _, _ in kind.code:
                if op != CONTRACT or out[0] != INTER:
                    continue
                constraints = found.get(kind.inters[out[1]][0])
                if not constraints:
                    continue
                number, positions = kind.contracts[plan]
                spec, operands, labels, irreps, ordered = self._contraction_keys[number]
                ordered = tuple(sorted((x, y, strict) for (x, y), strict in constraints.items()))
                kind.contracts[plan] = (self._number((spec, operands, labels, irreps, ordered)), positions)

    def _record(self, group, raw):
        """The products of contract_tile for one projection group and raw spin
        pattern, with the tile's occupied tuple symbolic."""
        no, nv = self.no, self.nv
        kind = _Kind()
        tile_shape = (nv,) * 4 + (1,) * 4
        tile_strides = _c_strides(tile_shape)
        known = {}                                  # symbolic expression -> intermediate
        for coefficient, spec, names in self.groups[group][2]:
            lhs, output = spec.split("->")
            inputs = lhs.split(",")
            cut = {c: q for q, c in enumerate(output[4:])}
            t4 = names.index("t4") if "t4" in names else None
            if t4 is None or self.slab is None:      # packed T4 is read whole (see contract_tile)
                slabs = ((),)
            else:
                slabs = _packed_cuts(inputs[t4][4:], tuple((c, 0, 1) for c in cut), no, nv, _PACKED_TILE_BYTES,
                                     self.slab)
            for derivative in range(len(names)):
                if not self.presence[names[derivative]]:
                    continue
                for signatures in routes(spec, raw, derivative):
                    for pieces in slabs:
                        selected = {c: (s.start, s.stop) for c, s in pieces}
                        selected.update({c: (_FIXED, q) for c, q in cut.items()})

                        def operand(i):
                            side = int(i == derivative)
                            if names[i] == "t4":
                                return self._t4(side, signatures[i], inputs[i], selected)
                            return self._dense(names[i], side, signatures[i], inputs[i], selected)

                        def key_of(tree):
                            if isinstance(tree, int):
                                return (names[tree], tree == derivative, signatures[tree],
                                        tuple(selected.get(c, _ALL) for c in inputs[tree]))
                            return tree[0], key_of(tree[1]), key_of(tree[2])

                        irreps = tuple(self.irrep if i == derivative else 0 for i in range(len(names)))

                        def evaluate(tree):
                            if isinstance(tree, int):
                                return operand(tree)
                            key = key_of(tree)
                            hit = known.get(key)
                            if hit is not None:      # computed earlier in the tile
                                return hit
                            template, positions = _symbolic(key)
                            identity = self._templates.setdefault(template, len(self._templates))
                            number = len(kind.inters)
                            kind.inters.append(None)        # numbered before its operands
                            look = len(kind.code)
                            kind.code.append([LOOKUP, number, -1, -1, None, None, None, 0.0, 0.0])
                            inner, left, right = tree
                            a, b = evaluate(left), evaluate(right)
                            sa, sb = inner.split("->")[0].split(",")
                            extents = dict(zip(sa, a[2])) | dict(zip(sb, b[2]))
                            shape = tuple(extents[c] for c in inner.split("->")[1])
                            kind.inters[number] = (identity, positions, shape)
                            value = (INTER, number, shape, _c_strides(shape), None)
                            self._contract(kind, inner, a, b, value, 1.0, 0.0, selected,
                                           (_tree_irrep(left, irreps), _tree_irrep(right, irreps)))
                            kind.code.append((PUBLISH, number, -1, -1, None, None, None, 0.0, 0.0))
                            kind.code[look][2] = len(kind.code)
                            known[key] = value
                            return value

                        tile = (TILE, 0, tile_shape, tile_strides, None)
                        # Only the entries the projection reads (`ordered_letters`).
                        ordered = ordered_letters(self.groups[group][0], raw, output)
                        if len(names) == 2:
                            self._contract(kind, spec, operand(0), operand(1), tile, coefficient, 1.0,
                                           selected, irreps, ordered)
                            continue
                        if len(names) < 2:
                            raise Unsupported
                        tree = contraction_schedule(spec, no, nv, tuple(cut))
                        binary, left, right = tree
                        order = tuple(output.index(c) for c in binary.split("->")[1])
                        view = (TILE, 0, tuple(tile_shape[o] for o in order),
                                tuple(tile_strides[o] for o in order), None)
                        a, b = evaluate(left), evaluate(right)
                        self._contract(kind, binary, a, b, view, coefficient, 1.0, selected,
                                       (_tree_irrep(left, irreps), _tree_irrep(right, irreps)), ordered)
        return kind

    def _kind(self, group, raw):
        number = self._kind_of.get((group, raw))
        if number is None:
            number = self._kind_of[group, raw] = len(self._kinds)
            self._kinds.append(self._record(group, raw))
        return number

    # ---- planning ----------------------------------------------------------

    def _plan(self, number, irreps):
        """Plan of a contraction for the irreps of its occupied indices."""
        found = self._plans_of.get((number, irreps))
        if found is not None:
            return found
        info = self._contraction_info[number]
        spec, arrays, labels, operand_irreps, structure, ordered = info
        occupied = self.occupied_irreps
        current = {c: (np.asarray([irreps[bound[1]]], dtype=np.int64) if bound[0] == _FIXED
                       else occupied[bound[0]:bound[1]]) for c, bound in labels}
        found_structure, plan, _ = self.blocked._entry(spec, *arrays, current, operand_irreps, ordered)
        if not found_structure:
            raise Unsupported
        if structure is None:
            info[4] = found_structure
        elif found_structure is not structure:
            raise Unsupported                  # its offset tables differ between irrep patterns
        plan = np.ascontiguousarray(plan, dtype=np.int64)
        self._keep.append(plan)
        found = self._plans_of[number, irreps] = (plan.ctypes.data if len(plan) else 0, len(plan))
        return found

    def _plan_set(self, kind_number, code):
        """Plans of a kind's contractions for one irrep pattern (code) of the
        tile's occupied tuple."""
        n = self.nirrep
        pattern = (code // n**3, code // n**2 % n, code // n % n, code % n)
        found = [self._plan(number, tuple(pattern[q] for q in positions))
                 for number, positions in self._kinds[kind_number].contracts]
        self._plan_sets.append(found)
        return len(self._plan_sets) - 1

    # ---- slot classes ------------------------------------------------------

    def _spin_groups(self, layout, key):
        """(particle spins, rows in the slot, virtual coordinates) per spin
        pattern present, in the order of the Python loop."""
        particles = layout.groups[key]
        signatures = particles % 2
        found = []
        for pspin in product((0, 1), repeat=4):
            mask = np.flatnonzero(np.all(signatures == pspin, axis=1))
            if mask.size:
                rows = np.ascontiguousarray(mask, dtype=np.int64)
                coordinates = particles[mask] // 2
                if self._vpos is not None:           # the tile's stored positions
                    coordinates = self._vpos[coordinates]
                coordinates = np.ascontiguousarray(coordinates, dtype=np.int32)
                self._keep.extend((rows, coordinates))
                found.append((pspin, rows, coordinates))
        return found

    def _class(self, spins, pspins, equal, terms_of):
        """Tiles of a slot class in the loop's order: (kind, occupied positions,
        gathers as (spin group number, terms number)). The representative
        occupied tuple `equal` (each index's first position) has the class's
        repeated indices, so its tiles' occupied tuples are positions."""
        tiles = []
        for group, (equivalent, multiplicity, diagrams) in enumerate(self.groups):
            if not diagrams:
                continue
            scatter = {}
            for number, pspin in enumerate(pspins):
                for raw, placements in external_routes(4, pspin + spins, equivalent):
                    for p, h, sign in placements:
                        fixed = tuple(equal[i] for i in h)
                        by_group = scatter.setdefault((raw, fixed), {})
                        by_group.setdefault(pspin, (number, []))[1].append((p, sign))
            for (raw, fixed), by_group in scatter.items():
                parts = []
                for number, permutations in by_group.values():
                    terms = tuple((p, multiplicity * sign) for p, sign in permutations)
                    parts.append((number, terms_of.setdefault(terms, len(terms_of))))
                tiles.append((self._kind(group, raw), fixed, parts))
        return tiles

    def _prepare(self, engine):
        layout = engine._direction_t4
        nv, n = self.nv, self.nirrep
        keys, key_groups, classes, class_tiles, terms_of = {}, [], {}, [], {}
        slots = np.zeros(max(1, len(layout.slots)), dtype=SLOT)
        for s, (holes, (key, window)) in enumerate(layout.slots.items()):
            k = keys.get(key)
            if k is None:
                k = keys[key] = len(key_groups)
                key_groups.append(self._spin_groups(layout, key))
            occupied = tuple(h // 2 for h in holes)
            spins = tuple(h % 2 for h in holes)
            equal = tuple(occupied.index(o) for o in occupied)
            pspins = tuple(pspin for pspin, _, _ in key_groups[k])
            c = classes.get((spins, pspins, equal))
            if c is None:
                c = classes[spins, pspins, equal] = len(class_tiles)
                class_tiles.append(self._class(spins, pspins, equal, terms_of))
            slots[s] = (c, k, window.start, window.stop - window.start, occupied)
        self.nslots = len(layout.slots)
        if _runtime.enabled("ITERCC_ORDERED_INTERMEDIATES"):
            self._order_intermediates()
        # Plans for every (kind, irrep pattern) the slots need.
        plan_id = np.full((max(1, len(self._kinds)), n**4), -1, dtype=np.int64)
        templates, parts, class_start, class_count = [], [], [], []
        weights = np.array([n**3, n**2, n, 1], dtype=np.int64)
        for c, tiles in enumerate(class_tiles):
            class_start.append(len(templates))
            class_count.append(len(tiles))
            for kind, fixed, own in tiles:
                templates.append((kind, fixed, len(parts), len(own)))
                parts.extend(own)
            if not tiles or not self.nslots:
                continue
            members = slots[:self.nslots][slots["cls"][:self.nslots] == c]
            kinds = np.array([kind for kind, _, _ in tiles], dtype=np.int64)
            positions = np.array([fixed for _, fixed, _ in tiles], dtype=np.int64)
            irreps = self.occupied_irreps[members["occupied"][:, positions]]       # slots x tiles x 4
            codes = irreps @ weights
            needed = np.unique(kinds[None, :] * n**4 + codes)
            for pair in needed[plan_id.reshape(-1)[needed] < 0]:
                kind, code = divmod(int(pair), n**4)
                plan_id[kind, code] = self._plan_set(kind, code)
        self.plan_id = plan_id
        self.slots = slots
        self.templates = np.zeros(max(1, len(templates)), dtype=TEMPLATE)
        if templates:
            self.templates["kind"] = [t[0] for t in templates]
            self.templates["positions"] = [t[1] for t in templates]
            self.templates["part_start"] = [t[2] for t in templates]
            self.templates["part_count"] = [t[3] for t in templates]
        self.parts = np.array(parts or [(0, 0)], dtype=PART)
        self.class_start = np.asarray(class_start or [0], dtype=np.int64)
        self.class_count = np.asarray(class_count or [0], dtype=np.int64)
        terms = np.zeros(max(1, len(terms_of)), dtype=TERMS)
        tile_steps = (nv**3, nv**2, nv, 1)
        for permutations, number in terms_of.items():
            offsets, factors = _gather_steps(permutations, tile_steps)
            offsets = np.ascontiguousarray(offsets, dtype=np.int64)
            factors = np.ascontiguousarray(factors, dtype=np.float64)
            self._keep.extend((offsets, factors))
            terms[number] = (offsets.ctypes.data, factors.ctypes.data, len(permutations))
        self.terms = terms
        groups, key_group_start = [], []
        for found in key_groups:
            key_group_start.append(len(groups))
            groups.extend((rows.ctypes.data, coordinates.ctypes.data, len(rows)) for _, rows, coordinates in found)
        self.groups_table = np.array(groups or [(0, 0, 0)], dtype=GROUP)
        self.key_group_start = np.asarray(key_group_start or [0], dtype=np.int64)
        self._finish(engine)

    # ---- the core program --------------------------------------------------

    @staticmethod
    def _operand(x):
        value = np.zeros(1, dtype=_OPERAND)[0]
        if x is not None:
            value["kind"], value["index"] = x[0], x[1]
            if x[4] is not None:
                value["coef"] = x[4]
        return value

    def _fusion(self, kind):
        """Fusion group (within the kind) of each instruction, -1 for none: the
        products into the tile with dense operands, grouped by their output
        offset tables (equal tables: a block of each addresses the same tile
        entries in the same order, so one fused GEMM sums them)."""
        groups, found = {}, []
        for op, _, _, plan, a, b, out, _, _ in kind.code:
            group = -1
            if op == CONTRACT and out[0] == TILE and a[0] != ROWS and b[0] != ROWS:
                structure = self._contraction_info[kind.contracts[plan][0]][4]
                tables = self._tables.get(id(structure))
                if tables is None:
                    tables = self._tables[id(structure)] = (
                        structure["rpos"].tobytes() + b"|" + structure["cpos"].tobytes(), structure)
                group = groups.setdefault(tables[0], len(groups))
            found.append(group)
        return found, len(groups)

    def _finish(self, engine):
        instructions, intermediates = [], []
        kind_start, kind_count, inter_start, inter_count = [], [], [], []
        fuse_start, fuse_count, fuse_groups, members = [], [], [], []
        self._tables = {}
        for kind in self._kinds:
            kind_start.append(len(instructions))
            kind_count.append(len(kind.code))
            inter_start.append(len(intermediates))
            inter_count.append(len(kind.inters))
            for identity, positions, shape in kind.inters:
                intermediates.append((identity, prod(shape), len(positions),
                                      tuple(positions) + (0,) * (4 - len(positions))))
            fusion, ngroups = self._fusion(kind)
            fuse_start.append(len(fuse_groups))
            fuse_count.append(ngroups)
            for group in range(ngroups):
                own = [i for i, g in enumerate(fusion) if g == group]
                fuse_groups.append((len(members), len(own)))
                members.extend(own)
            for (op, inter, after, plan, a, b, out, alpha, beta), group in zip(kind.code, fusion):
                row = np.zeros(1, dtype=INSTRUCTION)[0]
                row["op"], row["inter"], row["next"], row["plan"] = op, inter, after, plan
                if op == CONTRACT:
                    row["next"] = group                          # its fusion group, -1 for none
                    row["inter"] = kind.contracts[plan][0]       # its contraction (for profiles)
                    structure = self._contraction_info[kind.contracts[plan][0]][4]
                    if structure is None:
                        raise Unsupported          # never planned: no slot runs this kind
                    row["a"], row["b"], row["out"] = self._operand(a), self._operand(b), self._operand(out)
                    for name in ("a_rows", "a_sum", "b_sum", "b_cols", "rpos", "cpos"):
                        row[name] = structure[name].ctypes.data
                    row["flags"], row["alpha"] = self.blocked.flags(structure), alpha
                    if beta not in (0.0, 1.0) or (beta == 0.0) != (out[0] == INTER):
                        raise Unsupported
                instructions.append(row)
        self.instructions = np.array(instructions or [np.zeros(1, INSTRUCTION)[0]], dtype=INSTRUCTION)
        self.intermediates = np.array(intermediates or [(0, 0, 0, (0, 0, 0, 0))], dtype=INTERMEDIATE)
        self.fuse_start = np.asarray(fuse_start or [0], dtype=np.int64)
        self.fuse_count = np.asarray(fuse_count or [0], dtype=np.int64)
        self.fuse_groups = np.ascontiguousarray(np.asarray(fuse_groups or [(0, 0)], dtype=np.int64).reshape(-1, 2))
        self.fuse_members = np.asarray(members or [0], dtype=np.int64)
        self._tables = None
        self.kind_start = np.asarray(kind_start or [0], dtype=np.int64)
        self.kind_count = np.asarray(kind_count or [0], dtype=np.int64)
        self.inter_start = np.asarray(inter_start or [0], dtype=np.int64)
        self.inter_count = np.asarray(inter_count or [0], dtype=np.int64)
        sizes = [len(found) for found in self._plan_sets]
        self.plan_start = np.asarray(np.r_[0, np.cumsum(sizes)][:-1] if sizes else [0], dtype=np.int64)
        self.plan_table = np.asarray([t for found in self._plan_sets for t, _ in found] or [0], dtype=np.int64)
        self.plan_count = np.asarray([c for found in self._plan_sets for _, c in found] or [0], dtype=np.int64)
        # T4 rows: both layouts' hole tables and complete map tables.
        rows = np.zeros(max(1, len(self._rows)), dtype=ROWS_DESCRIPTOR)
        layouts = (engine._ground_t4, engine._direction_t4)
        tables = [layout._hole_tables() for layout in layouts]
        n = 2 * self.no
        for (which, signature, cuts), index in self._rows.items():
            layout = layouts[which]
            bases, keys, sectors = tables[which]
            pspin, hspin = tuple(signature[:4]), tuple(signature[4:])
            table = layout._map_pointers.get(pspin)
            if table is None:
                table = layout._map_pointers[pspin] = np.zeros(len(sectors), dtype=np.int64)
            axes = [2 * np.arange(self.no, dtype=np.int64) + s for s in hspin]
            every = (((axes[0][:, None, None, None] * n + axes[1][None, :, None, None]) * n
                      + axes[2][None, None, :, None]) * n + axes[3][None, None, None, :]).reshape(-1)
            live = bases[every] != 0
            for owner in np.unique(keys[every][live]):
                if table[owner] == 0:
                    table[owner] = layout._signed_map(pspin, sectors[owner]).ctypes.data
            row = rows[index]
            row["which"], row["spins"] = which, hspin
            row["fixed"] = [q for q, _, _ in cuts]
            row["start"] = [s for _, s, _ in cuts]
            row["count"] = [c for _, _, c in cuts]
            row["pointers"] = table.ctypes.data
            self._keep.append(table)
        self.rows = rows
        self.reads_ground = any(which == 0 for which, _, _ in self._rows)
        holes = [(np.ascontiguousarray(b, dtype=np.int64), np.ascontiguousarray(k, dtype=np.int32))
                 for b, k, _ in tables]
        arrays = dict(kind_start=self.kind_start, kind_count=self.kind_count, inter_start=self.inter_start,
                      inter_count=self.inter_count, instructions=self.instructions,
                      intermediates=self.intermediates, rows=self.rows, plan_start=self.plan_start,
                      plan_table=self.plan_table, plan_count=self.plan_count, plan_id=self.plan_id,
                      ground_bases=holes[0][0], ground_keys=holes[0][1], direction_bases=holes[1][0],
                      direction_keys=holes[1][1], occupied_irreps=self.occupied_irreps,
                      class_start=self.class_start, class_count=self.class_count, templates=self.templates,
                      parts=self.parts, terms=self.terms, key_group_start=self.key_group_start,
                      groups=self.groups_table, slots=self.slots, fuse_start=self.fuse_start,
                      fuse_count=self.fuse_count, fuse_groups=self.fuse_groups, fuse_members=self.fuse_members)
        # The tile's virtual positions of each irrep as one run (irrep order:
        # its products and gathers touch only the symmetry-allowed entries);
        # none when an irrep's positions are not contiguous.
        virtual = np.asarray(engine._spatial_labels["v"], dtype=np.int64)
        stored = virtual if self._vpos is None else virtual[np.argsort(virtual, kind="stable")]
        runs = np.zeros((self.nirrep, 2), dtype=np.int64)
        contiguous = self.nirrep > 1
        for g in range(self.nirrep):
            where = np.flatnonzero(stored == g)
            if len(where):
                runs[g] = where[0], where[-1] + 1
                contiguous &= len(where) == where[-1] + 1 - where[0]
        arrays["tile_runs"] = runs
        for name, array in arrays.items():
            if not array.flags.c_contiguous:
                raise ValueError(f"{name} is not contiguous")
        self.struct = ProgramStruct(nv=self.nv, holes=n, nirrep=self.nirrep,
                                    **{name: arrays[name].ctypes.data for name in _ARRAYS},
                                    tile_irrep=int(self.irrep))
        if not contiguous:
            self.struct.tile_runs = None
        self._arrays = arrays

    def run(self, blocks, ground, direction, result, *, cache_bytes, threads, slot_limit=None, fused=True,
            share_work=0, progress=None, slots=None, partial_bytes=None):
        """Add the projected rank-four slots into `result` (zero on entry: the
        slots' rows get the Python loop's values). `slots` (increasing slot
        indices) runs only those slots, `slot_limit` the first ones; each slot
        writes only its own rows, with the same values whichever slots run with
        it. `progress(done)` is called after each wave of slots. Returns the
        intermediate cache statistics."""
        pointers = np.zeros(max(1, len(self._bases)), dtype=np.int64)
        for name, pair in blocks.items():
            for side, array in enumerate((pair.value, pair.tangent)):
                if (name, side) in self._bases:
                    pointers[self._bases[name, side]] = array.ctypes.data
        for vector in (ground, direction):
            if vector is not None and not (vector.dtype == np.float64 and vector.flags.c_contiguous):
                raise ValueError("Packed T4 vectors must be contiguous float64 arrays")
        if (ground is None and self.reads_ground) or (direction is None and self.presence["t4"]):
            raise ValueError("The program reads a missing T4 vector")
        if not (result.dtype == np.float64 and result.flags.c_contiguous):
            raise ValueError("The rank-four result must be a contiguous float64 array")
        struct = self.struct
        if slots is not None:
            # The program walks the slot records it is given: a table of the chosen ones.
            chosen = np.ascontiguousarray(self.slots[np.asarray(slots, dtype=np.int64)])
            struct = ProgramStruct.from_buffer_copy(self.struct)
            struct.slots = chosen.ctypes.data
            nslots = len(chosen)
        else:
            nslots = self.nslots if slot_limit is None else min(int(slot_limit), self.nslots)
        stats = np.zeros(8, dtype=np.int64)
        callback = None if progress is None else _PROGRESS(lambda done, total: progress(int(done)))
        # Waves hold as many partial sums as the intermediate cache holds
        # bytes (at least 256 MiB): larger waves let more tiles that share
        # intermediates run together.
        partial = max(_PARTIAL_DOUBLES, int(cache_bytes if partial_bytes is None else partial_bytes) // 8)
        status = self.library.triplet_slots(
            ctypes.byref(struct), nslots, partial, pointers.ctypes.data,
            None if ground is None else ground.ctypes.data,
            None if direction is None else direction.ctypes.data,
            result.ctypes.data, int(cache_bytes), stats.ctypes.data, int(threads), int(bool(fused)),
            int(share_work), ctypes.cast(callback, ctypes.c_void_p) if callback is not None else None)
        if status != 0:
            raise RuntimeError(f"The compiled slot program failed (status {status})")
        return dict(hits=int(stats[0]), misses=int(stats[1]), peak_bytes=int(stats[2]), shared=int(stats[3]),
                    helped=int(stats[4]))
