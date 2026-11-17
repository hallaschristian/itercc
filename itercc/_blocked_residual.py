"""Restricted CCSDT residual terms on symmetry-blocked T3 (compiled core).

Same algebra as `_cpu_triples.py`, with T3 and R3 in the `BlockedTriples`
layout. Because only allowed entries exist, every GEMM is restricted in its
columns as well as its rows: for C2v the products shrink about fourfold
relative to the dense-slot path, as do the bytes the kernels move.
"""

import ctypes
import os

import numpy as np

from . import _core, _cpu_kernels
from ._cpu_triples import LADDER_BUDGET, _feedback_terms, _ladder_chunk, _ladder_matrix, _ladder_sectors, _seed_sources


class NotBlockable(ValueError):
    """An operator is not block-diagonal in the irreps (use the dense-slot path)."""


def _sectors(labels, groups):
    """Stable order by label, sector starts (groups + 1) and sector-local rows."""
    order = np.argsort(labels, kind="stable").astype(np.int64)
    start = np.searchsorted(labels[order], np.arange(groups + 1)).astype(np.int64)
    row = np.empty(len(labels), dtype=np.int64)
    row[order] = np.arange(len(labels)) - np.repeat(start[:-1], np.diff(start))
    return order, start, row


def _diagonal_blocks(matrix, order, start, groups):
    """Packed diagonal sector blocks of `matrix` (rows and columns by `order`)."""
    grouped = matrix[np.ix_(order, order)]
    blocks, offsets, total, kept = [], [], 0, 0
    for sector in range(groups):
        a, b = int(start[sector]), int(start[sector + 1])
        block = np.ascontiguousarray(grouped[a:b, a:b])
        kept += np.count_nonzero(block)
        blocks.append(block.reshape(-1))
        offsets.append(total)
        total += (b - a) ** 2
    if kept != np.count_nonzero(grouped):
        raise NotBlockable("A residual operator couples different irreps")
    return (np.concatenate(blocks) if blocks else np.empty(0)), offsets


class PlaneTables:
    """Tables of the per-plane kernels for one blocked layout and occupied irreps."""

    def __init__(self, blocked, occupied_irreps):
        layout, groups, nv = blocked.layout, blocked.groups, blocked.nv
        no = layout.no
        self.blocked, self.no, self.groups = blocked, no, groups
        self.occ = np.ascontiguousarray(occupied_irreps, dtype=np.int64)
        count = blocked.count
        pair_labels = (self.occ[:, None] ^ self.occ[None, :]).reshape(-1)
        self.pair_order, self.pair_start, self.pair_row = _sectors(pair_labels, groups)
        ph_labels = (self.occ[:, None] ^ blocked.virtual_irreps[None, :]).reshape(-1)
        self.ph_order, self.ph_start, self.ph_row = _sectors(ph_labels, groups)
        gs = np.arange(groups)
        self.npair = np.array([sum(int(count[gb] * count[gb ^ tau]) for gb in gs)
                               for tau in gs], dtype=np.int64)
        self.poff = np.zeros((groups, groups), dtype=np.int64)
        for tau in gs:
            sizes = count * count[gs ^ tau]
            self.poff[tau] = np.cumsum(sizes) - sizes
        self.ntriple = np.array([sum(int(count[ga] * count[gb] * count[gamma ^ ga ^ gb])
                                     for ga in gs for gb in gs) for gamma in gs], dtype=np.int64)
        widths = self.npair[gs[:, None] ^ self.occ[None, :]]          # (kappa, j)
        self.jcol = np.ascontiguousarray(np.cumsum(widths, axis=1) - widths, dtype=np.int64)
        self.ncol = widths.sum(axis=1).astype(np.int64)
        unit = layout.slots.astype(np.int64)
        self.third = [(np.ascontiguousarray(unit[:, :, k]).reshape(-1),
                       np.ascontiguousarray(layout.from_sorted[:, :, k], dtype=np.int64).reshape(-1))
                      for k in range(no)]
        self.first = [(np.ascontiguousarray(unit[m]).reshape(-1),
                       np.ascontiguousarray(layout.from_sorted[m], dtype=np.int64).reshape(-1))
                      for m in range(no)]
        self.scatter = [self._scatter(layout, k) for k in range(no)]
        # T2-driven seeds: A rows (j,a,b), columns (i,c); B rows (i,j,a).
        vir = blocked.virtual_irreps
        self.a_order, self.a_start, self.a_row = _sectors(
            (self.occ[:, None, None] ^ vir[None, :, None] ^ vir[None, None, :]).reshape(-1), groups)
        self.ic_order, self.ic_start, self.a_col = _sectors(
            (self.occ[:, None] ^ vir[None, :]).reshape(-1), groups)
        self.b_order, self.b_start, self.b_row = _sectors(
            (pair_labels[:, None] ^ vir[None, :]).reshape(-1), groups)
        self.d_of = [blocked.order[blocked.start[x]:blocked.start[x] + count[x]] for x in gs]
        self.l_of = [np.flatnonzero(self.occ == x) for x in gs]
        pairs = []                                   # (b, c) of each pair irrep, list order
        for tau in gs:
            b_list, c_list = [], []
            for gb in gs:
                vb, vc = self.d_of[gb], self.d_of[gb ^ tau]
                b_list.append(np.repeat(vb, len(vc)))
                c_list.append(np.tile(vc, len(vb)))
            pairs.append((np.concatenate(b_list), np.concatenate(c_list)))
        self.pair_lists = pairs
        n_pairs, n_ph = np.diff(self.pair_start), np.diff(self.ph_start)
        n_arow, n_ic, n_brow = np.diff(self.a_start), np.diff(self.ic_start), np.diff(self.b_start)
        self.planes = []
        for k in range(no):
            gk = int(self.occ[k])
            widths = self.ntriple[gs ^ gk]
            pair_base = (np.cumsum(n_pairs * widths) - n_pairs * widths).astype(np.int64)
            ph_ncols = np.ascontiguousarray(self.ncol[gs ^ gk])
            ph_base = (np.cumsum(n_ph * ph_ncols) - n_ph * ph_ncols).astype(np.int64)
            a_ncols = np.ascontiguousarray(n_ic[gs ^ gk], dtype=np.int64)
            a_base = (np.cumsum(n_arow * a_ncols) - n_arow * a_ncols).astype(np.int64)
            b_ncols = np.ascontiguousarray(self.npair[gs ^ gk])
            b_base = (np.cumsum(n_brow * b_ncols) - n_brow * b_ncols).astype(np.int64)
            slot, perm = self.third[k]
            tables = _core.structure(_core.BlockedPlane, dict(no=no, k=k), dict(
                occ_irrep=self.occ, tile_slot=slot, tile_perm=perm, pair_order=self.pair_order,
                pair_start=self.pair_start, pair_row=self.pair_row, pair_base=pair_base,
                ph_row=self.ph_row, ph_base=ph_base, ph_ncols=ph_ncols, jcol=self.jcol,
                poff=self.poff, ntriple=self.ntriple, a_row=self.a_row, a_col=self.a_col,
                a_base=a_base, a_ncols=a_ncols, b_row=self.b_row, b_base=b_base, b_ncols=b_ncols))
            sizes = dict(plane=int((n_pairs * widths).sum()), ph=int((n_ph * ph_ncols).sum()),
                         pair=(n_pairs, widths, pair_base), rhs=(n_ph, ph_ncols, ph_base),
                         a=(int((n_arow * a_ncols).sum()), a_base, a_ncols),
                         b=(int((n_brow * b_ncols).sum()), b_base, b_ncols))
            self.planes.append((tables, sizes))

    def _seed_plans(self, k, nv):
        """Executor tables of plane k's T2-seed products (they depend on k only
        through its irrep): for each, (block table, offset tables)."""
        gk = int(self.occ[k])
        cache = self.__dict__.setdefault("_seed_cache", {})
        if gk in cache:
            return cache[gk]
        no = self.no
        _, sizes = self.planes[k]
        _, a_base, _ = sizes["a"]
        _, b_base, _ = sizes["b"]
        plans = []
        for kind in "ab":
            blocks, columns = [], [[] for _ in range(6)]
            r0 = c0 = k0 = 0
            for gamma in range(self.groups):
                if kind == "a":
                    # A[r, (i,c)] = sum_d vvvo_rows[rows[r], d] t2(i, k, d, c)
                    rows = self.a_order[self.a_start[gamma]:self.a_start[gamma + 1]]
                    summed = self.d_of[gamma]
                    cols = self.ic_order[self.ic_start[gamma ^ gk]:self.ic_start[(gamma ^ gk) + 1]]
                    i, c = np.divmod(cols, nv)
                    parts = (rows * nv, summed, summed * nv, i * no * nv * nv + c)
                    base = a_base[gamma]
                else:
                    # B[r, (b,c)] = sum_l vooo_rows[rows[r], l] t2(l, k, b, c)
                    rows = self.b_order[self.b_start[gamma]:self.b_start[gamma + 1]]
                    summed = self.l_of[gamma]
                    pb, pc = self.pair_lists[gamma ^ gk]
                    cols = pb * nv + pc
                    parts = (rows * no, summed, summed * no * nv * nv, cols)
                    base = b_base[gamma]
                if not (len(rows) and len(cols) and len(summed)):
                    continue        # an empty sum leaves the (zeroed) block at zero
                blocks.append((len(rows), len(cols), len(summed), r0, c0, k0, 0, 0, int(base)))
                for column, part in zip(columns, (*parts, np.arange(len(rows)) * len(cols),
                                                  np.arange(len(cols)))):
                    column.append(np.asarray(part, dtype=np.int64))
                r0, c0, k0 = r0 + len(rows), c0 + len(cols), k0 + len(summed)
            table = np.array(blocks, dtype=np.int64).reshape(-1, 9)
            # Offset tables: a_rows, a_sum, b_sum, b_cols, rpos, cpos.
            offsets = tuple(np.ascontiguousarray(np.concatenate(c) if c else np.zeros(1, np.int64))
                            for c in columns)
            plans.append((table, offsets))
        cache[gk] = plans
        return plans

    def t2_seeds(self, vvvo_rows, vooo_rows, t2, k, a_out, b_out, threads=None):
        """Block products A = vvvo t2 and B = vooo t2 of plane k (see core.h).

        With the compiled core, each is one call of the GEMM executor that
        reads the operands' blocks in place (no gathered copies).
        """
        library = _cpu_kernels.library()
        nv = t2.shape[-1]
        if (library is not None and t2.flags.c_contiguous and t2.dtype == np.float64
                and vvvo_rows.flags.c_contiguous and vooo_rows.flags.c_contiguous):
            from ._cpu_triples import cpu_threads

            threads = cpu_threads(threads)
            _, sizes = self.planes[k]
            plane = t2.ctypes.data + 8 * k * nv * nv          # t2(., k, ., .)
            for (table, offsets), rows, out, total in zip(
                    self._seed_plans(k, nv), (vvvo_rows, vooo_rows), (a_out, b_out),
                    (sizes["a"][0], sizes["b"][0])):
                if total:
                    library.scale(out.ctypes.data, int(total), 0.0, threads)
                if len(table):
                    library.sym_contract(len(table), table.ctypes.data, rows.ctypes.data, plane,
                                         out.ctypes.data, *(x.ctypes.data for x in offsets),
                                         1.0, 0, threads)
            return
        gk = int(self.occ[k])
        _, sizes = self.planes[k]
        _, a_base, a_ncols = sizes["a"]
        _, b_base, b_ncols = sizes["b"]
        t2k = t2[:, k]                                    # (i or l, d or b, c)
        for gamma in range(self.groups):
            rows = self.a_order[self.a_start[gamma]:self.a_start[gamma + 1]]
            d = self.d_of[gamma]
            columns = self.ic_order[self.ic_start[gamma ^ gk]:self.ic_start[(gamma ^ gk) + 1]]
            if len(rows) and len(columns):
                left = vvvo_rows[np.ix_(rows, d)] if len(d) else np.zeros((len(rows), 0))
                right = t2k.transpose(1, 0, 2).reshape(len(t2k[0]), -1)[d][:, columns]
                np.matmul(left, right, out=a_out[a_base[gamma]:a_base[gamma] + len(rows) * len(columns)]
                          .reshape(len(rows), len(columns)))
            rows = self.b_order[self.b_start[gamma]:self.b_start[gamma + 1]]
            l = self.l_of[gamma]
            pb, pc = self.pair_lists[gamma ^ gk]
            if len(rows) and len(pb):
                left = vooo_rows[np.ix_(rows, l)]
                right = t2k[l][:, pb, pc]
                np.matmul(left, right, out=b_out[b_base[gamma]:b_base[gamma] + len(rows) * len(pb)]
                          .reshape(len(rows), len(pb)))

    @staticmethod
    def _scatter(layout, k):
        """Targets of plane k and their (pair, t0, t1, t2) contributions."""
        no = layout.no
        contributions = {}
        for i in range(no):
            for j in range(no):
                slot = int(layout.slots[i, j, k])
                if slot < 0:
                    continue
                ts = [int(x) for x in layout.to_sorted[i, j, k]]
                found = contributions.setdefault(slot, [])
                found.append((i * no + j, *ts))
                a, b, c = layout.indices[slot]
                if a == b:                      # second virtual permutation
                    found.append((i * no + j, ts[1], ts[0], ts[2]))
                elif b == c:
                    found.append((i * no + j, ts[0], ts[2], ts[1]))
        targets = np.array(sorted(contributions), dtype=np.int64)
        lists = [contributions[t] for t in targets]
        start = np.cumsum([0] + [len(x) for x in lists]).astype(np.int64)
        flat = np.array([e for x in lists for e in x], dtype=np.int64).reshape(-1)
        return targets, start, flat


def triples(blocked, tables, t2, t3, have_t3, im, vooo, vvvo, g, o, v, threads, *,
            ladder_budget=LADDER_BUDGET, wvvvv=None, ladder_g=None, ladder_owned=False):
    """Blocked R3 from T2 and blocked T3, as `_cpu_triples.triples` (`wvvvv`
    replaces the ladder matrix built from g, t2 and im, as there; `ladder_g` =
    (LadderSectors, even, odd), the sector blocks of g(vvvv), builds the ladder
    matrix in sector form instead, in those blocks when `ladder_owned`).

    Raises NotBlockable when an intermediate is not symmetry-blocked; the
    caller then uses the dense-slot path.
    """
    from ._cpu_triples import _plane_matrices, _Rows

    library = _cpu_kernels.library()
    no, nv, groups = tables.no, blocked.nv, blocked.groups
    layout = blocked.c_layout()
    residual = np.zeros(blocked.size)
    vvvo_rows, vooo_rows = _seed_sources(vooo, vvvo, no, nv)
    a_seed = np.zeros(max(sizes["a"][0] for _, sizes in tables.planes))
    b_seed = np.zeros(max(sizes["b"][0] for _, sizes in tables.planes))
    largest_plane = max(sizes["plane"] for _, sizes in tables.planes)
    largest_rhs = max(sizes["ph"] for _, sizes in tables.planes)
    seeds = np.empty(largest_plane)
    if have_t3:
        occupied, _, _ = _plane_matrices(im, _Rows(None, no, nv), no, nv)
        direct = np.ascontiguousarray(im.ovvo.transpose(3, 1, 0, 2)).reshape(no * nv, no * nv)
        direct *= 0.25
        crossed = np.ascontiguousarray(im.ovov.transpose(2, 1, 0, 3)).reshape(no * nv, no * nv)
        occ_mats, occ_off = _diagonal_blocks(occupied, tables.pair_order, tables.pair_start, groups)
        dir_mats, ph_off = _diagonal_blocks(direct, tables.ph_order, tables.ph_start, groups)
        cro_mats, _ = _diagonal_blocks(crossed, tables.ph_order, tables.ph_start, groups)
        plane = np.empty(largest_plane)
        crossed_rhs, direct_rhs = np.empty(largest_rhs), np.empty(largest_rhs)
        direct_out, crossed_out = np.empty(largest_rhs), np.empty(largest_rhs)
    else:
        direct_out = crossed_out = seeds
    for k in range(no):
        tables.t2_seeds(vvvo_rows, vooo_rows, t2, k, a_seed, b_seed, threads)
        plane_tables, sizes = tables.planes[k]
        if have_t3:
            library.bt3_plane(ctypes.addressof(layout), ctypes.addressof(plane_tables),
                              t3.ctypes.data, plane.ctypes.data, crossed_rhs.ctypes.data,
                              direct_rhs.ctypes.data, threads)
            n_pairs, widths, pair_base = sizes["pair"]
            _grouped(library, zip(n_pairs, widths), occ_off, occ_mats, plane, pair_base, seeds,
                     threads)
            n_ph, ph_ncols, ph_base = sizes["rhs"]
            _grouped(library, zip(n_ph, ph_ncols), ph_off, dir_mats, direct_rhs, ph_base,
                     direct_out, threads)
            _grouped(library, zip(n_ph, ph_ncols), ph_off, cro_mats, crossed_rhs, ph_base,
                     crossed_out, threads)
        library.bt3_seeds(ctypes.addressof(layout), ctypes.addressof(plane_tables), int(have_t3),
                          a_seed.ctypes.data, b_seed.ctypes.data, direct_out.ctypes.data,
                          crossed_out.ctypes.data, seeds.ctypes.data, threads)
        targets, start, contributions = tables.scatter[k]
        library.bt3_scatter(ctypes.addressof(layout), ctypes.addressof(plane_tables), len(targets),
                            targets.ctypes.data, start.ctypes.data, contributions.ctypes.data,
                            seeds.ctypes.data, residual.ctypes.data, threads)
    if have_t3:
        if wvvvv is None and ladder_g is not None:
            structure, g_even, g_odd = ladder_g
            packed = (structure, *structure.ladder_matrix((g_even, g_odd), im.vv, t2, g[o, o, v, v],
                                                          threads, inplace=ladder_owned))
            ladder(blocked, residual, t3, None, threads, budget=ladder_budget, packed=packed)
        else:
            if wvvvv is None:
                wvvvv = _ladder_matrix(g, o, v, t2, im, nv, threads)
            ladder(blocked, residual, t3, wvvvv, threads, budget=ladder_budget)
    purify(blocked, residual, threads)
    return residual


# The feedback products of one plane m (`_cpu_triples._feedback_terms`), with
# q[i,j,x,y,z] the spin-summed T3(m,i,j,x,y,z) and the integral slices
# g3 = g[v,m,v,v], oovv = g[o,m,v,v] and gf = -g[o,m,o,v] + 0.5 delta f[m,v]:
# (spec, left operand, right operand, output, alpha).
_FEEDBACK = (("ijxay,byx->ijab", "q", "g3", "r2", 1.0),
             ("ijxay,lyx->alij", "q", "oovv", "vooo", 1.0),
             ("ijxay,jyx->ia", "q", "oovv", "r1", 1.0),
             ("ijxya,jyx->ia", "q", "oovv", "r1", -0.5),
             ("jnx,ijxyz->inyz", "gf", "q", "r2", 1.0),
             ("jdx,ijxyz->zydi", "oovv", "q", "vvvo", -1.0))


def _feedback_engine(tables, threads, shift=0):
    """The symmetry-blocked executor for the feedback products in the orbital
    labels (the tables' occupied labels XOR `shift`), cached on the tables."""
    engines = tables.__dict__.setdefault("_feedback_engines", {})
    key = (int(shift), threads)
    engine = engines.get(key)
    if engine is None:
        from ._cpu_symmetry_blocks import CpuSymmetryBlocks

        labels = dict({c: tables.occ ^ int(shift) for c in "ijln"},
                      **{c: tables.blocked.virtual_irreps for c in "abdxyz"})
        engine = engines[key] = CpuSymmetryBlocks(labels, "", tables.groups, fallback=None,
                                                  threads=threads)
    return engine


def feedback(blocked, tables, r1, r2, vooo, vvvo, f, g, t3, o, v, threads, *, shift=0,
             operator_irrep=0):
    """T3 terms of R1/R2 and the triples sources from blocked T3, as `_cpu_triples.feedback`.

    Per plane m the spin-summed q = P T3(m, i, j) is written dense (zero
    outside its symmetry blocks), and the products run on the symmetry-blocked
    executor, which reads q and the integral slices in place through offset
    tables: no relayout copy of q, and none of the symmetry-zero GEMM work
    (about 3/4 of it in C2v). The dense GEMMs of `_cpu_triples._feedback_terms`
    remain for outputs the executor cannot take.

    Tangents (`_cpu_tangent.feedback`): T3 of irrep `shift` in the tables of
    the occupied labels shifted by it (`_cpu_tangent._blocked_layout`), and f
    and g of irrep `operator_irrep`, exactly; the outputs then have the irrep
    of both together. In the orbital labels q has the irrep of the tables'
    occupied label of m, the integral slices that XOR shift XOR operator_irrep.
    """
    library = _cpu_kernels.library()
    no, nv = tables.no, blocked.nv
    layout = blocked.c_layout()
    outputs = dict(r1=r1, r2=r2, vooo=vooo, vvvo=vvvo)
    engine = _feedback_engine(tables, threads, shift)
    blocked_products = all(isinstance(x, np.ndarray) and x.dtype == np.float64 and x.flags.writeable
                           for x in (*outputs.values(), g, f))
    q = np.empty((no, no, nv, nv, nv))
    x_plane = None if blocked_products else np.empty((no, no, nv, nv, nv))
    for m in range(no):
        slot, perm = tables.first[m]
        library.bt3_feedback_plane(ctypes.addressof(layout), no, slot.ctypes.data, perm.ctypes.data,
                                   t3.ctypes.data, q.ctypes.data,
                                   None if x_plane is None else x_plane.ctypes.data, threads)
        if x_plane is not None:
            _feedback_terms(m, q, x_plane, r1, r2, vooo, vvvo, f, g, o, v)
            continue
        gf = -g[o, m, o, v]                              # (j, n, x), a new array
        for j in range(no):
            gf[j, j] += 0.5 * f[m, v]
        operands = dict(q=q, g3=g[v, m, v, v], oovv=g[o, m, v, v], gf=gf)
        plane = int(tables.occ[m])
        operator = plane ^ int(shift) ^ int(operator_irrep)
        irreps = dict(q=plane, g3=operator, oovv=operator, gf=operator)
        for spec, left, right, out, alpha in _FEEDBACK:
            engine.einsum(spec, operands[left], operands[right], outputs[out], alpha, 1.0,
                          irreps=(irreps[left], irreps[right]))


_SECTOR_PLANS = {}


def _sector_plan(sizes, a_offsets, bases):
    """(block table, offset tables, zeroed regions) of `_grouped`, cached by shape."""
    key = (sizes, a_offsets, bases)
    plan = _SECTOR_PLANS.get(key)
    if plan is not None:
        return plan
    blocks, columns, regions = [], [[] for _ in range(6)], []
    r0 = c0 = 0
    for (m, n), a_offset, base in zip(sizes, a_offsets, bases):
        if not (m and n):
            continue
        # Row-major A (m x m) at a_offset; X and out (m x n) at base.
        blocks.append((m, n, m, r0, c0, r0, a_offset, base, base))
        rows, cols = np.arange(m, dtype=np.int64), np.arange(n, dtype=np.int64)
        for column, part in zip(columns, (rows * m, rows, rows * n, cols, rows * n, cols)):
            column.append(part)
        regions.append((base, m * n))
        r0, c0 = r0 + m, c0 + n
    table = np.array(blocks, dtype=np.int64).reshape(-1, 9)
    offsets = tuple(np.concatenate(c) if c else np.zeros(1, np.int64) for c in columns)
    if len(_SECTOR_PLANS) > 4096:
        _SECTOR_PLANS.clear()
    plan = _SECTOR_PLANS[key] = (table, offsets, tuple(regions))
    return plan


def _grouped(library, sizes, a_offsets, matrices, operands, bases, out, threads=None):
    """out[base] = block @ operand[base] for each sector, (m x m) @ (m x n), in one
    call of the core's GEMM executor (the contraction engine of the CPU CCSDTQ
    solver: no BLAS, all OpenMP threads, sectors and tiles in parallel)."""
    from ._cpu_triples import cpu_threads

    threads = cpu_threads(threads)
    sizes = tuple((int(m), int(n)) for m, n in sizes)
    table, offsets, regions = _sector_plan(sizes, tuple(int(x) for x in a_offsets),
                                           tuple(int(x) for x in bases))
    if not len(table):
        return
    for start, length in regions:           # the executor adds into out
        library.scale(out.ctypes.data + 8 * start, length, 0.0, threads)
    library.sym_contract(len(table), table.ctypes.data, matrices.ctypes.data, operands.ctypes.data,
                         out.ctypes.data, *(x.ctypes.data for x in offsets), 1.0, 0, threads)


def _sector_matrices(matrix, start, groups):
    """Diagonal blocks of a sector-ordered matrix, packed one after another."""
    blocks, offsets, total = [], [], 0
    for pi in range(groups):
        a, b = int(start[pi]), int(start[pi + 1])
        blocks.append(np.ascontiguousarray(matrix[a:b, a:b]).reshape(-1))
        offsets.append(total)
        total += (b - a) ** 2
    return (np.concatenate(blocks) if blocks else np.empty(0)), offsets


def ladder(blocked, residual, t3, wvvvv, threads, *, budget=LADDER_BUDGET, packed=None):
    """R3 += sum over the three virtual pairs of W(ab,de) T(..de..), blocked.

    `wvvvv` must be block-diagonal in pair irreps (the symmetric Hamiltonian);
    `residual` and `t3` are blocked arrays of `blocked`. `packed` =
    (LadderSectors, even, odd) gives W's sector blocks instead (`wvvvv` unused).
    """
    library = _cpu_kernels.library()
    groups, nv = blocked.groups, blocked.nv
    irrep = blocked.virtual_irreps
    if packed is not None:
        structure, even_mats, odd_mats = packed
        if structure.groups != groups or structure.nv != nv:
            raise ValueError("The ladder sectors belong to another layout")
        p, q, r, s = structure.pairs
        (_, _, even_start, even_off, _), (_, _, odd_start, odd_off, _) = structure.even, structure.odd
        even_label, odd_label = irrep[p] ^ irrep[q], irrep[r] ^ irrep[s]
    else:
        (p, q, r, s), even_blocks, odd_blocks = _ladder_sectors(wvvvv, irrep)
        # One block for pairs of several sectors (the matrix couples sectors), or
        # blocks between different sectors: not block-diagonal.
        for (x, y), found in (((p, q), even_blocks), ((r, s), odd_blocks)):
            if (any(rows != cols for rows, cols, _ in found)
                    or (len(found) == 1 and len(np.unique(irrep[x] ^ irrep[y])) > 1)):
                raise NotBlockable("The ladder matrix is not block-diagonal in pair irreps")
        p, q, r, s = (np.ascontiguousarray(x, dtype=np.int64) for x in (p, q, r, s))
        even_label, odd_label = irrep[p] ^ irrep[q], irrep[r] ^ irrep[s]
        even_start = np.searchsorted(even_label, np.arange(groups + 1)).astype(np.int64)
        odd_start = np.searchsorted(odd_label, np.arange(groups + 1)).astype(np.int64)
        even = _dense_sector_matrix(even_blocks, len(p))
        odd = _dense_sector_matrix(odd_blocks, len(r))
        even_mats, even_off = _sector_matrices(even, even_start, groups)
        odd_mats, odd_off = _sector_matrices(odd, odd_start, groups)
    even_row = np.zeros((nv, nv), dtype=np.int64)
    even_row[p, q] = even_row[q, p] = np.arange(len(p)) - even_start[even_label]
    odd_row = np.zeros((nv, nv), dtype=np.int64)
    odd_row[r, s] = odd_row[s, r] = np.arange(len(r)) - odd_start[odd_label]
    odd_sign = np.zeros((nv, nv))
    odd_sign[r, s], odd_sign[s, r] = 1.0, -1.0
    n_even, n_odd = np.diff(even_start), np.diff(odd_start)
    layout = blocked.c_layout()
    nslots = len(blocked.slot_irrep)
    chunk = _ladder_chunk(max(1, nv // groups), nslots, len(p) + len(r), budget)
    for first in range(0, nslots, chunk):
        count = min(chunk, nslots - first)
        third = blocked.count[np.arange(groups)[:, None] ^ blocked.slot_irrep[first:first + count]]
        widths = 3 * third                                   # (pi, slot) columns
        colstart = np.ascontiguousarray(np.cumsum(widths, axis=1) - widths, dtype=np.int64)
        ncols = widths.sum(axis=1).astype(np.int64)
        sym_size, anti_size = n_even * ncols, n_odd * ncols
        sym_base = (np.cumsum(sym_size) - sym_size).astype(np.int64)
        anti_base = (np.cumsum(anti_size) - anti_size).astype(np.int64)
        tables = _core.structure(_core.BlockedLadder, dict(first_slot=first, nslots=count), dict(
            even_p=p, even_q=q, even_start=even_start, odd_r=r, odd_s=s, odd_start=odd_start,
            even_row=even_row, odd_row=odd_row, odd_sign=odd_sign, colstart=colstart,
            ncols=ncols, sym_base=sym_base, anti_base=anti_base))
        sym_in, anti_in = np.empty(int(sym_size.sum())), np.empty(int(anti_size.sum()))
        library.bt3_ladder_gather(ctypes.addressof(layout), ctypes.addressof(tables),
                                  t3.ctypes.data, sym_in.ctypes.data, anti_in.ctypes.data, threads)
        sym_out, anti_out = np.empty_like(sym_in), np.empty_like(anti_in)
        _grouped(library, zip(n_even, ncols), even_off, even_mats, sym_in, sym_base, sym_out, threads)
        _grouped(library, zip(n_odd, ncols), odd_off, odd_mats, anti_in, anti_base, anti_out,
                 threads)
        del sym_in, anti_in
        library.bt3_ladder_scatter(ctypes.addressof(layout), ctypes.addressof(tables),
                                   sym_out.ctypes.data, anti_out.ctypes.data,
                                   residual.ctypes.data, threads)


def _dense_sector_matrix(blocks, size):
    """The sector-ordered matrix from `_ladder_sectors` blocks (diagonal or whole)."""
    if len(blocks) == 1 and blocks[0][0] == slice(None):
        return blocks[0][2]
    matrix = np.zeros((size, size))
    for rows, cols, block in blocks:
        matrix[rows, cols] = block
    return matrix


def purify(blocked, residual, threads):
    """Remove the fully symmetric virtual component of each slot, in place."""
    _cpu_kernels.library().bt3_purify(ctypes.addressof(blocked.c_layout()),
                                      residual.ctypes.data, threads)
