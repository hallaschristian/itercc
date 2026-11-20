"""Compiled R4 tile loops of the CPU CCSDTQ residual (`itercc_tile_program`).

`compute_r4_tri` adds about a hundred symmetry-blocked contractions into each
R4 tile. Called from Python one at a time, every small contraction pays the
interpreter and an OpenMP parallel region. `TileProgram` records the loop body
once per solver and runs it in the core: each thread takes whole slots, keeps
its R4 tile in cache while it adds every term with the single-threaded
contraction routes, and writes the tile to the packed residual.

The body is the same Python function the tile loop calls (a sequence of
`tile(spec, a, b, out=..., alpha=..., beta=...)` calls for one tile). A term's
plan depends only on the slot's irrep pattern (at most nirrep**4 of them), and
its operands are views at offsets affine in the slot's occupied indices. So
the body is played with a recorder once per pattern (plans) and at five index
tuples (the offset coefficients, checked at every pattern's slot), not once
per slot: recording stays cheap and small for thousands of slots. Later runs
only pass fresh base addresses. One-slot tiles (block_size 1) only.

A T3-like base can also come as a pair-blocked copy (`_cpu_packed.PairBlocks`:
the last two virtual axes stored irrep pair by irrep pair). The fused program
reads it in every term whose last two letters of that base are both free or
both summed: the term's offset tables map through the permutation, and the
GEMM packs runs of a whole irrep pair block (about (nv/nirrep)^2 doubles)
instead of single virtual rows (nv/nirrep). Other terms read the dense copy.
"""
import ctypes
import functools

import numpy as np

from . import _core, _cpu_kernels
from . import _runtime
from ._cpu_packed import output_sector


@functools.lru_cache(maxsize=None)
def memory_bytes():
    """Host memory this process may use (`_cpu_triples.host_memory_limit`)."""
    from ._cpu_triples import host_memory_limit

    return host_memory_limit()


def team_size(threads, live_bytes, budget_bytes=None):
    """Threads per slot. Each running slot keeps `live_bytes` of tiles. One
    thread per slot is fastest whenever the tiles fit: teams share a slot's
    tiles in cache and their GEMMs run faster, but each slot then pays its
    parallel regions, barriers between partitions and the serial parts of its
    tile work. Measured at 47 virtuals, teams of five on a Sapphire Rapids
    socket took 2.4x the wall time of slot-parallel tiles (with 78 MB of tiles
    per thread), and teams of twelve on a 12-core desktop took 1.3x. So
    teams only bound the tiles' memory: as many slots run at once as fit
    `budget_bytes` (default: an eighth of the host memory; ITERCC_TILE_MEMORY
    in bytes). ITERCC_TILE_TEAM fixes the team size."""
    override = _runtime.integer("ITERCC_TILE_TEAM")
    if override:
        return max(1, min(threads, override))
    if budget_bytes is None:
        budget_bytes = _runtime.integer("ITERCC_TILE_MEMORY") or 0
        if not budget_bytes:
            available = memory_bytes()
            # When unknown, run only one slot at a time to bound memory.
            budget_bytes = live_bytes if available is None else available // 8
    concurrent = max(1, min(threads, budget_bytes // max(1, live_bytes)))
    return max(1, -(-threads // concurrent))


class Unsupported(Exception):
    """The body has a product without a blocked plan or an operand outside the bases."""


class _Term:
    __slots__ = ("spec", "alpha", "structure", "flags", "a_base", "b_base", "a_coef", "b_coef",
                 "starts", "counts", "blocks")


class _Recorded:
    def __init__(self, signature, terms):
        self.signature = signature
        self.terms = terms
        self.fused = {}                   # blocked base names -> _Fused (False: not fusable)


class _Fused:
    """The terms of a recorded program as fused blocks (itercc_fused_tile_program)."""

    def __init__(self, block_start, block_count, blocks, segments, keep, partitions, tables=None):
        self.block_start, self.block_count = block_start, block_count
        self.blocks, self.segments = blocks, segments
        self.keep = keep                  # offset tables the blocks and segments point into
        self.partitions = partitions      # row letters of each tag
        # Per pattern, the blocks as (tag, rpos, cpos, segments), each segment
        # with its row and column offset tables: for `TileProgram._sector`.
        self.tables = tables
        self.sectors = {}                 # (labels, targets) -> (pattern, _Fused)


def _pack_fused(block_start, block_count, blocks, segments, keep, partitions, tables=None):
    """A _Fused from Python block and segment tuples."""
    fused_blocks = (_core.FusedBlock * max(1, len(blocks)))()
    for target, values in zip(fused_blocks, blocks):
        (target.rows, target.cols, target.rpos, target.cpos, target.seg_start, target.seg_count,
         target.tag) = values
        target.o_base = 0
    fused_segments = (_core.FusedSegment * max(1, len(segments)))()
    for target, values in zip(fused_segments, segments):
        target.x_base, target.y_base = values[0], values[1]
        target.x_coef[:] = values[2]
        target.y_coef[:] = values[3]
        target.x_rows, target.x_sum, target.y_sum, target.y_cols = values[4:8]
        target.depth, target.alpha, target.flags = values[8:11]
    return _Fused(np.ascontiguousarray(block_start, dtype=np.int64),
                  np.ascontiguousarray(block_count, dtype=np.int64), fused_blocks, fused_segments,
                  keep, partitions, tables)


def blocked_t3_enabled():
    """Pair-blocked T3 for the tile programs, unless ITERCC_TILE_BLOCKED_T3=0
    (Sapphire Rapids, 13o/47v C2v: tile programs 41.3 -> 37.2 s, the
    evaluation -4.4%; residuals within 1e-12 of the dense pair order)."""
    return _runtime.enabled("ITERCC_TILE_BLOCKED_T3")


def fusion_enabled():
    """Fused tile programs unless ITERCC_TILE_FUSED=0 (then one pass per term)."""
    return _runtime.enabled("ITERCC_TILE_FUSED")


class TileProgram:
    """Runs R4 tile-loop bodies slot-parallel in the compiled core (see the module docstring)."""

    def __init__(self, solver):
        self.layout = solver.layout
        self.blocks = solver._tile_blocks
        self.library = _cpu_kernels.library()
        nocc, nvir = solver.nocc, solver.nvir
        self.nocc = nocc
        indices = np.asarray(self.layout.indices, dtype=np.int64).reshape(-1, 4)
        self.slots = np.ascontiguousarray(np.arange(len(indices), dtype=np.int64))
        self.occupied = np.ascontiguousarray(indices, dtype=np.int32)
        irreps = np.asarray(self.layout.occupied_irreps, dtype=np.int64)
        codes = (((irreps[indices[:, 0]] * 8 + irreps[indices[:, 1]]) * 8
                  + irreps[indices[:, 2]]) * 8 + irreps[indices[:, 3]])
        unique, first, self.pattern = np.unique(codes, return_index=True, return_inverse=True)
        self.pattern = np.ascontiguousarray(self.pattern, dtype=np.int64)
        self.representatives = [tuple(int(x) for x in indices[f]) for f in first]
        self._out = np.zeros((1, 1, 1, 1) + (nvir,) * 4)     # the tiles' shape and strides
        self._t4 = np.zeros((1, 1, 1, 1) + (nvir,) * 4)
        self._recorded = {}
        self._failed = set()
        self._pairs = None
        self._sector_classes = {}         # OutputSector key -> (virtual labels, class per slot)

    def pairs(self):
        """The pair blocking of T3-like bases (virtual irreps of the layout)."""
        if self._pairs is None:
            from ._cpu_packed import PairBlocks

            self._pairs = PairBlocks(self.layout.virtual_irreps, self.layout.nirrep)
        return self._pairs

    def blocked_enabled(self):
        """Whether the tile programs read pair-blocked T3 copies (`blocked`)."""
        return blocked_t3_enabled()

    def blocked(self, arrays, *, single=False):
        """Pair-blocked copies of `arrays` (name -> T3-shaped array) for `run`
        (float32 with `single`), or None with ITERCC_TILE_BLOCKED_T3=0."""
        if not blocked_t3_enabled():
            return None
        return {name: self.pairs().permute(array, single=single, threads=self.blocks.threads)
                for name, array in arrays.items()}

    def run(self, key, body, bases, r4, beta, *, t4=None, narrow=None, blocked=None):
        """Run `body(tile, i0, i1, j0, j1, k0, k1, l0, l1, out[, t4_tile])` for every
        slot in the core and add the tiles to packed `r4` (r4 = beta r4 + tile).
        `bases` names the arrays the body's operands are views of; with `t4`
        the body reads the slot's dense T4 tile. Returns False (nothing done)
        when the body cannot be compiled, so the caller runs its Python loop.
        `narrow` maps base names to float32 arrays of the same shape that the
        fused program reads instead (widened to double as its GEMMs pack them).
        `blocked` maps base names to their `pairs().permute` copies (float64 or
        float32), read by the terms that can (`_blocked_sides`)."""
        if key in self._failed:
            return False
        bases = {name: array for name, array in bases.items() if array is not None}
        signature = tuple((name, array.shape, array.strides, array.dtype.char)
                          for name, array in sorted(bases.items()))
        recorded = self._recorded.get(key)
        if recorded is None or recorded.signature != signature:
            try:
                recorded = self._record(body, bases, signature, t4 is not None)
            except Unsupported:
                self._failed.add(key)
                return False
            self._recorded[key] = recorded
        names = [name for name, *_ in signature]
        nv = self._out.shape[-1]
        blocked = {name: array for name, array in (blocked or {}).items()
                   if name in bases and array.shape == bases[name].shape and bases[name].shape[-2:] == (nv,) * 2
                   and array.dtype in (np.float64, np.float32) and array.flags.c_contiguous
                   and bases[name].flags.c_contiguous}
        swapped = tuple(sorted(blocked))
        fused = recorded.fused.get(swapped)
        if fused is None:
            try:
                fused = self._fuse(recorded, names, swapped)
            except Unsupported:
                fused = False
            recorded.fused[swapped] = fused
        from ._runtime import cpu_threads

        if not (r4.flags.c_contiguous and r4.dtype == np.float64):
            raise ValueError("The packed residual must be a C-contiguous float64 array")
        packed = None if t4 is None else np.ascontiguousarray(t4, dtype=np.float64)
        threads = cpu_threads(self.blocks.threads)
        # Each running slot keeps a dense R4 tile (and a T4 tile) live.
        live = self._out.nbytes * (2 if t4 is not None else 1)
        self.team = team_size(threads, live)
        if fused and fusion_enabled():
            pattern = self.pattern
            sector = output_sector(self.layout)
            if sector is not None and fused.tables is not None:
                pattern, fused = self._sector(fused, self.sector_classes(sector))
            narrow = {name: array for name, array in (narrow or {}).items()
                      if name in bases and array.shape == bases[name].shape and array.dtype == np.float32
                      and array.flags.c_contiguous and bases[name].flags.c_contiguous}
            arrays = [narrow.get(n, bases[n]) for n in names] + [blocked[n] for n in swapped]
            pointers = (ctypes.c_void_p * max(1, len(arrays)))(*[a.ctypes.data for a in arrays])
            flags = (ctypes.c_uint8 * max(1, len(arrays)))(*[int(a.dtype == np.float32) for a in arrays])
            self.library.fused_tile_program(
                self.layout._layout, len(self.slots), self.slots.ctypes.data, self.occupied.ctypes.data,
                pattern.ctypes.data, fused.block_start.ctypes.data, fused.block_count.ctypes.data,
                ctypes.cast(fused.blocks, ctypes.c_void_p), ctypes.cast(fused.segments, ctypes.c_void_p),
                ctypes.cast(pointers, ctypes.c_void_p), ctypes.cast(flags, ctypes.c_void_p),
                None if packed is None else packed.ctypes.data, r4.ctypes.data, float(beta), threads,
                self.team)
            return True
        terms = (_core.TileTerm * len(recorded.terms))()
        for slot, term in zip(terms, recorded.terms):
            slot.a = None if term.a_base < 0 else bases[names[term.a_base]].ctypes.data
            slot.b = None if term.b_base < 0 else bases[names[term.b_base]].ctypes.data
            slot.a_coef[:] = term.a_coef
            slot.b_coef[:] = term.b_coef
            structure = term.structure
            slot.a_rows, slot.a_sum, slot.b_sum, slot.b_cols, slot.rpos, slot.cpos = (
                structure[name].ctypes.data for name in ("a_rows", "a_sum", "b_sum", "b_cols", "rpos", "cpos"))
            slot.plan_start, slot.plan_count, slot.blocks = (
                term.starts.ctypes.data, term.counts.ctypes.data, term.blocks.ctypes.data)
            slot.alpha = term.alpha
            slot.flags = term.flags
        self.library.tile_program(self.layout._layout, len(self.slots), self.slots.ctypes.data,
                                  self.occupied.ctypes.data, self.pattern.ctypes.data, len(terms),
                                  ctypes.cast(terms, ctypes.c_void_p),
                                  None if packed is None else packed.ctypes.data, r4.ctypes.data,
                                  float(beta), threads, self.team)
        return True

    def _blocked_sides(self, recorded, names, swapped):
        """{(term number, side): (base index, free table, sum table)} for the
        term operands that read a pair-blocked base. The pair part of an
        address (its offset mod nv^2: the last two virtual axes of the base)
        must come from one table: the offset coefficients, the block outer
        offsets and one of the two tables are multiples of nv^2. Then the
        address mod nv^2 is the other table's entry mod nv^2, and
        `PairBlocks.remap` of both tables gives the blocked address."""
        if not swapped:
            return {}
        pairs = self.pairs()
        nv2 = self._out.shape[-1] ** 2
        tables, sides = {}, {}

        def remapped(table):
            hit = tables.get(id(table))
            if hit is None:
                hit = tables[id(table)] = (pairs.remap(table), table)   # keeps the key alive
            return hit[0]

        for number, term in enumerate(recorded.terms):
            st = term.structure
            for side, (base, coef, free, total, column) in enumerate((
                    (term.a_base, term.a_coef, st["a_rows"], st["a_sum"], 6),
                    (term.b_base, term.b_coef, st["b_cols"], st["b_sum"], 7))):
                if base < 0 or names[base] not in swapped:
                    continue
                if np.any(free % nv2) and np.any(total % nv2):
                    continue                     # the pair part split between the tables
                if any(int(c) % nv2 for c in coef):
                    continue
                if term.blocks.size and np.any(term.blocks[:, column] % nv2):
                    continue
                sides[number, side] = (len(names) + swapped.index(names[base]), remapped(free),
                                       remapped(total))
        return sides

    def _fuse(self, recorded, names=None, swapped=()):
        """The recorded terms as fused blocks. A term's rows carry some of the
        tile's virtual indices (its output partition: ab|cd, ac|bd, a|bcd, ...);
        oriented so that the rows carry the tile's first virtual index (operands
        swapped otherwise), the terms of one partition and row irrep share
        their rows and columns. Each such block becomes one fused GEMM over
        canonical rows and columns (sorted by tile offset), every term one
        segment of it with its offset tables permuted to the canonical order.
        Operands of the bases named in `swapped` that can read their blocked
        copies (`_blocked_sides`) do, as extra bases after `names`."""
        from .symmetry_blocks import _parse

        shape = self._out.shape
        keep, orders, permuted, canonical = [], {}, {}, {}
        partitions, info = {}, []
        for term in recorded.terms:
            sa, sb, so = _parse(term.spec)
            moving = [c for c, n in zip(so, shape) if n > 1]
            rows = "".join(c for c in moving if c in sa and c not in sb)
            cols = "".join(c for c in moving if c in sb and c not in sa)
            normal = not moving or moving[0] in rows    # one virtual: every extent is 1
            info.append((normal, partitions.setdefault(rows if normal else cols, len(partitions))))

        def order_of(table, start, count):
            key = (id(table), start, count)
            hit = orders.get(key)
            if hit is None:
                keys = table[start:start + count]
                order = np.argsort(keys, kind="stable")
                if (order == np.arange(count)).all():
                    hit = (None, keys, key)
                else:
                    hit = (order, keys[order], key)
                orders[key] = hit
            return hit

        def address(table, start, order):
            """(address, entries) of a block's offset table in canonical order."""
            if order[0] is None:
                return table.ctypes.data + 8 * start, table[start:start + len(order[1])]
            key = (id(table), start, order[2])
            hit = permuted.get(key)
            if hit is None:
                hit = permuted[key] = np.ascontiguousarray(
                    table[start:start + len(order[0])][order[0]], dtype=np.int64)
                keep.append(hit)
            return hit.ctypes.data, hit

        def canon(keys):
            code = keys.tobytes()
            hit = canonical.get(code)
            if hit is None:
                hit = canonical[code] = np.ascontiguousarray(keys, dtype=np.int64)
                keep.append(hit)
            return hit, code

        sides = self._blocked_sides(recorded, names, swapped)
        keep.extend(table for side in sides.values() for table in side[1:])
        block_start, block_count, blocks, segments, tables = [], [], [], [], []
        for kind in range(len(self.representatives)):
            local, where = [], {}
            for number, (term, (normal, tag)) in enumerate(zip(recorded.terms, info)):
                st = term.structure
                a_index, a_free, a_sum = sides.get((number, 0), (term.a_base, st["a_rows"], st["a_sum"]))
                b_index, b_free, b_sum = sides.get((number, 1), (term.b_base, st["b_cols"], st["b_sum"]))
                table = term.blocks[term.starts[kind]:term.starts[kind] + term.counts[kind]]
                for rows, cols, depth, r0, c0, k0, a_base, b_base, o_base in table.tolist():
                    if o_base != 0:
                        raise Unsupported
                    if normal:
                        r_order, c_order = order_of(st["rpos"], r0, rows), order_of(st["cpos"], c0, cols)
                        x = (a_index, term.a_coef, a_base, a_free, r0, a_sum)
                        y = (b_index, term.b_coef, b_base, b_free, c0, b_sum)
                        flags = int(term.flags & 1) | int(term.flags & 2)
                    else:
                        r_order, c_order = order_of(st["cpos"], c0, cols), order_of(st["rpos"], r0, rows)
                        x = (b_index, term.b_coef, b_base, b_free, c0, b_sum)
                        y = (a_index, term.a_coef, a_base, a_free, r0, a_sum)
                        flags = (0 if term.flags & 2 else 1) | (0 if term.flags & 1 else 2)
                    (rpos, r_code), (cpos, c_code) = canon(r_order[1]), canon(c_order[1])
                    (x_rows, x_table), (y_cols, y_table) = address(x[3], x[4], r_order), address(y[3], y[4], c_order)
                    segment = (x[0], y[0], [x[1][0] + x[2], *x[1][1:]], [y[1][0] + y[2], *y[1][1:]],
                               x_rows, x[5].ctypes.data + 8 * k0, y[5].ctypes.data + 8 * k0, y_cols,
                               depth, term.alpha, flags, x_table, y_table)
                    block = where.get((tag, r_code, c_code))
                    if block is None:
                        block = where[(tag, r_code, c_code)] = len(local)
                        local.append((tag, rpos, cpos, []))
                    local[block][3].append(segment)
            local.sort(key=lambda block: block[0])         # stable: tag, then first use
            # Blocks of one tag run at once in teams: their rows must not overlap.
            for tag in {block[0] for block in local}:
                rows = [block[1] for block in local if block[0] == tag]
                if sum(map(len, rows)) != len(np.unique(np.concatenate(rows))):
                    raise Unsupported
            block_start.append(len(blocks))
            block_count.append(len(local))
            tables.append(local)
            for tag, rpos, cpos, own in local:
                blocks.append((len(rpos), len(cpos), rpos.ctypes.data, cpos.ctypes.data,
                               len(segments), len(own), tag))
                segments.extend(own)
        return _pack_fused(block_start, block_count, blocks, segments, keep,
                           {tag: rows for rows, tag in partitions.items()}, tables)

    def sector_classes(self, sector):
        """(virtual labels, class per slot) of an `OutputSector` for `_sector`:
        per slot the XOR its virtual labels need, irrep ^ (its occupied labels)."""
        hit = self._sector_classes.get(sector.key)
        if hit is None:
            classes = sector.irrep ^ np.bitwise_xor.reduce(sector.occupied[self.occupied], axis=1)
            hit = self._sector_classes[sector.key] = (sector.virtual, np.ascontiguousarray(classes,
                                                                                         dtype=np.int64))
        return hit

    def _sector(self, fused, sector):
        """`fused` restricted to the tile entries of one symmetry sector: for
        `sector` = (virtual labels, slot classes) the entries (a, b, c, d) of
        slot s whose labels XOR to classes[s] (the EOM sector of a finer group
        than the layout's). An entry's offset is its rows' plus its columns'
        (disjoint base-nv digits of the dense tile), so a block's rows and
        columns take the classes of their digits and each block splits into
        the row and column class pairs that reach the sector: about half the
        rows times half the columns of the block, the summed dimension whole.
        Returns (pattern per slot, _Fused); the other entries are not formed."""
        virtual, classes = sector
        key = (virtual.tobytes(), classes.tobytes())
        hit = fused.sectors.get(key)
        if hit is not None:
            return hit
        nv = len(virtual)

        def digits(pos):
            label = np.zeros(len(pos), dtype=np.int64)
            for _ in range(4):
                label ^= virtual[pos % nv]
                pos = pos // nv
            return label

        refined, pattern = {}, np.empty(len(self.pattern), dtype=np.int64)
        for s, (kind, target) in enumerate(zip(self.pattern.tolist(), classes.tolist())):
            pattern[s] = refined.setdefault((kind, target), len(refined))
        keep = [fused.keep]
        block_start, block_count, blocks, segments = [], [], [], []
        for kind, target in refined:
            count = 0
            block_start.append(len(blocks))
            for tag, rpos, cpos, own in fused.tables[kind]:
                r_class, c_class = digits(rpos), digits(cpos)
                for alpha in np.unique(r_class).tolist():
                    rows = np.flatnonzero(r_class == alpha)
                    cols = np.flatnonzero(c_class == (alpha ^ target))
                    if not (rows.size and cols.size):
                        continue
                    sub_r = np.ascontiguousarray(rpos[rows])
                    sub_c = np.ascontiguousarray(cpos[cols])
                    keep.extend((sub_r, sub_c))
                    first = len(segments)
                    for x in own:
                        x_rows = np.ascontiguousarray(x[11][rows], dtype=np.int64)
                        y_cols = np.ascontiguousarray(x[12][cols], dtype=np.int64)
                        keep.extend((x_rows, y_cols))
                        segments.append((*x[:4], x_rows.ctypes.data, x[5], x[6], y_cols.ctypes.data, *x[8:11]))
                    blocks.append((rows.size, cols.size, sub_r.ctypes.data, sub_c.ctypes.data, first,
                                   len(own), tag))
                    count += 1
            block_count.append(count)
        hit = fused.sectors[key] = (np.ascontiguousarray(pattern),
                                    _pack_fused(block_start, block_count, blocks, segments, keep,
                                                fused.partitions))
        return hit

    def _locate(self, array, names, bases, tile):
        """(base index, element offset) of an operand: into the T4 tile (base -1)
        or into one of the bases."""
        if not (isinstance(array, np.ndarray) and array.dtype == np.float64):
            raise Unsupported
        address = array.ctypes.data
        candidates = ([(-1, tile)] if tile is not None else []) + list(enumerate(bases[n] for n in names))
        for index, base in candidates:
            if base.ctypes.data <= address < base.ctypes.data + base.nbytes:
                offset = address - base.ctypes.data
                if offset % 8:
                    raise Unsupported
                return index, offset // 8
        raise Unsupported

    def _calls(self, body, key, with_t4):
        """The tile calls of the body for occupied indices `key` (i, j, k, l)."""
        calls = []

        def record(spec, a, b, out=None, alpha=1.0, beta=0.0):
            calls.append((spec, a, b, float(alpha), float(beta)))
        i, j, k, l = key
        body(record, i, i + 1, j, j + 1, k, k + 1, l, l + 1, self._out,
             *((self._t4,) if with_t4 else ()))
        return calls

    def _record(self, body, bases, signature, with_t4):
        names = [name for name, *_ in signature]
        tile = self._t4 if with_t4 else None
        # Offsets at the origin and one step along each occupied index: the
        # affine coefficients of every operand.
        unit = [(0, 0, 0, 0)] + [tuple(int(d == axis) for d in range(4)) for axis in range(4)]
        if self.nocc < 2:
            unit = unit[:1]
        probes = [self._calls(body, key, with_t4) for key in unit]
        first = probes[0]
        terms = []
        for number, (spec, a, b, alpha, beta) in enumerate(first):
            if beta != (0.0 if number == 0 else 1.0):
                raise Unsupported              # the tile must start at zero, then accumulate
            term = _Term()
            term.spec, term.alpha = spec, alpha
            located = [[self._locate(x, names, bases, tile) for x in call[1:3]]
                       for call in (probe[number] for probe in probes)]
            for side, (base_name, coef_name) in enumerate((("a_base", "a_coef"), ("b_base", "b_coef"))):
                bases_seen = {entry[side][0] for entry in located}
                if len(bases_seen) != 1:
                    raise Unsupported
                origin = located[0][side][1]
                coef = [origin] + [(located[1 + axis][side][1] - origin) if len(located) > 1 else 0
                                   for axis in range(4)]
                setattr(term, base_name, located[0][side][0])
                setattr(term, coef_name, coef)
            terms.append(term)
        for probe in probes:
            if [call[0] for call in probe] != [term.spec for term in terms]:
                raise Unsupported
        # One plan per irrep pattern, from that pattern's representative slot.
        occupied = self.layout.occupied_irreps
        plans = [[] for _ in terms]
        for key in self.representatives:
            calls = self._calls(body, key, with_t4)
            if len(calls) != len(terms):
                raise Unsupported
            labels = {c: occupied[x:x + 1] for c, x in zip("ijkl", key)}
            for term, plan_list, (spec, a, b, alpha, beta) in zip(terms, plans, calls):
                if spec != term.spec or alpha != term.alpha:
                    raise Unsupported
                for x, base, coef in ((a, term.a_base, term.a_coef), (b, term.b_base, term.b_coef)):
                    index, offset = self._locate(x, names, bases, tile)
                    if index != base or offset != coef[0] + sum(c * v for c, v in zip(coef[1:], key)):
                        raise Unsupported      # the operand offsets are not affine in the slot
                structure, plan = self.blocks.plan_for(spec, a, b, self._out, labels=labels)
                if not structure:
                    raise Unsupported
                if not plan_list:
                    term.structure, term.flags = structure, self.blocks.flags(structure)
                    term.flags |= (8 if term.a_base < 0 else 0) | (16 if term.b_base < 0 else 0)
                elif structure is not term.structure:
                    raise Unsupported
                plan_list.append(np.asarray(plan, dtype=np.int64).reshape(-1, 9))
        for term, plan_list in zip(terms, plans):
            counts = np.array([len(plan) for plan in plan_list], dtype=np.int64)
            term.counts = counts
            term.starts = np.ascontiguousarray(np.r_[0, np.cumsum(counts)[:-1]], dtype=np.int64)
            term.blocks = np.ascontiguousarray(np.concatenate(plan_list).reshape(-1, 9), dtype=np.int64)
        return _Recorded(signature, terms)
