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

    def _seed_plans(self, k, nv, source=0):
        """Executor tables of plane k's T2-seed products (they depend on k only
        through its irrep): for each, (block table, offset tables). `source`:
        the irrep of vooo and vvvo in these tables' labels (T2 then has the
        irrep of the seeds XOR source); it selects the summed indices."""
        gk, source = int(self.occ[k]), int(source)
        cache = self.__dict__.setdefault("_seed_cache", {})
        if (gk, source) in cache:
            return cache[gk, source]
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
                    summed = self.d_of[gamma ^ source]
                    cols = self.ic_order[self.ic_start[gamma ^ gk]:self.ic_start[(gamma ^ gk) + 1]]
                    i, c = np.divmod(cols, nv)
                    parts = (rows * nv, summed, summed * nv, i * no * nv * nv + c)
                    base = a_base[gamma]
                else:
                    # B[r, (b,c)] = sum_l vooo_rows[rows[r], l] t2(l, k, b, c)
                    rows = self.b_order[self.b_start[gamma]:self.b_start[gamma + 1]]
                    summed = self.l_of[gamma ^ source]
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
        cache[gk, source] = plans
        return plans

    def t2_seeds(self, vvvo_rows, vooo_rows, t2, k, a_out, b_out, threads=None, source=0):
        """Block products A = vvvo t2 and B = vooo t2 of plane k (see core.h).

        With the compiled core, each is one call of the GEMM executor that
        reads the operands' blocks in place (no gathered copies). `source`: the
        irrep of vooo and vvvo in these tables' labels (`_seed_plans`).
        """
        library = _cpu_kernels.library()
        nv = t2.shape[-1]
        if (library is not None and t2.flags.c_contiguous and t2.dtype == np.float64
                and vvvo_rows.flags.c_contiguous and vooo_rows.flags.c_contiguous):
            from ._runtime import cpu_threads

            threads = cpu_threads(threads)
            _, sizes = self.planes[k]
            plane = t2.ctypes.data + 8 * k * nv * nv          # t2(., k, ., .)
            for (table, offsets), rows, out, total in zip(
                    self._seed_plans(k, nv, source), (vvvo_rows, vooo_rows), (a_out, b_out),
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
            d = self.d_of[gamma ^ int(source)]
            columns = self.ic_order[self.ic_start[gamma ^ gk]:self.ic_start[(gamma ^ gk) + 1]]
            if len(rows) and len(columns):
                left = vvvo_rows[np.ix_(rows, d)] if len(d) else np.zeros((len(rows), 0))
                right = t2k.transpose(1, 0, 2).reshape(len(t2k[0]), -1)[d][:, columns]
                np.matmul(left, right, out=a_out[a_base[gamma]:a_base[gamma] + len(rows) * len(columns)]
                          .reshape(len(rows), len(columns)))
            rows = self.b_order[self.b_start[gamma]:self.b_start[gamma + 1]]
            l = self.l_of[gamma ^ int(source)]
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
            ladder_budget=LADDER_BUDGET, wvvvv=None, ladder_g=None, ladder_owned=False,
            purified=True):
    """Blocked R3 from T2 and blocked T3, as `_cpu_triples.triples` (`wvvvv`
    replaces the ladder matrix built from g, t2 and im, as there; `ladder_g` =
    (LadderSectors, even, odd), the sector blocks of g(vvvv), builds the ladder
    matrix in sector form instead, in those blocks when `ladder_owned`;
    `purified=False` leaves the projection `purify` to the caller).

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
    if purified:
        purify(blocked, residual, threads)
    return residual


def _shifted_blocks(matrix, out_order, out_start, in_order, in_start, groups, shift):
    """The blocks of `matrix` (rows: output indices, columns: input indices)
    from input sector x ^ shift to output sector x, packed row-major one after
    another, and their offsets by output sector. Raises NotBlockable when any
    entry outside them is nonzero (the check is exact)."""
    blocks, offsets, total, kept = [], [], 0, 0
    for x in range(groups):
        rows = out_order[out_start[x]:out_start[x + 1]]
        y = x ^ int(shift)
        block = np.ascontiguousarray(matrix[np.ix_(rows, in_order[in_start[y]:in_start[y + 1]])])
        kept += np.count_nonzero(block)
        blocks.append(block.reshape(-1))
        offsets.append(total)
        total += block.size
    if kept != np.count_nonzero(matrix):
        raise NotBlockable("An operator is not of the direction's irrep")
    return (np.concatenate(blocks) if blocks else np.empty(0)), offsets


def triples_shifted(source, target, t2, t3, have_t3, im, vooo, vvvo, o, v, threads, *, shift,
                    wvvvv, residual, ladder_budget=LADDER_BUDGET):
    """Adds to `residual` (blocked in `target`) the R3 of a T2 and operators
    (im, W) of irrep `shift` with totally symmetric vooo, vvvo and T3: the
    derivative-operator half R3(dt2, T3; dim, vooo, vvvo, dW) of a Jacobian
    action of irrep `shift` (`_cpu_tangent.triples`). Not purified.

    `source` = (BlockedTriples, PlaneTables) of the orbital labels, in which T3
    is blocked; `target` = those of the occupied labels XOR shift, where the
    result is totally symmetric. Every T3 plane is gathered with the source
    tables and assembled and scattered with the target ones: their virtual
    triple and pair orders agree, and the products map input sectors to output
    sectors as the operators do: pair sector x ^ shift -> x (pair labels do not
    see the occupied shift) and particle-hole sector s -> s (the shift moves
    the output rows by the operator's irrep). In the target labels vooo and
    vvvo have irrep `shift`, which selects the seeds' summed indices. Raises
    NotBlockable, before writing anything, when an operator is not exactly of
    irrep `shift`."""
    from ._cpu_triples import _plane_matrices, _Rows

    (blocked_in, tables_in), (blocked_out, tables_out) = source, target
    library = _cpu_kernels.library()
    no, nv, groups, shift = tables_out.no, blocked_out.nv, blocked_out.groups, int(shift)
    if blocked_in.groups != groups or tables_in.no != no:
        raise ValueError("The source and target layouts do not match")
    layout_in, layout_out = blocked_in.c_layout(), blocked_out.c_layout()
    if have_t3:
        occupied, _, _ = _plane_matrices(im, _Rows(None, no, nv), no, nv)
        direct = np.ascontiguousarray(im.ovvo.transpose(3, 1, 0, 2)).reshape(no * nv, no * nv)
        direct *= 0.25
        crossed = np.ascontiguousarray(im.ovov.transpose(2, 1, 0, 3)).reshape(no * nv, no * nv)
        occ_mats, occ_off = _shifted_blocks(occupied, tables_out.pair_order, tables_out.pair_start,
                                            tables_in.pair_order, tables_in.pair_start, groups, shift)
        ph = (tables_out.ph_order, tables_out.ph_start, tables_in.ph_order, tables_in.ph_start, groups, 0)
        dir_mats, ph_off = _shifted_blocks(direct, *ph)
        cro_mats, _ = _shifted_blocks(crossed, *ph)
        del occupied, direct, crossed
        prepared = _shifted_ladder(blocked_out, wvvvv, shift)
        plane = np.empty(max(sizes["plane"] for _, sizes in tables_in.planes))
        crossed_rhs, direct_rhs = (np.empty(max(sizes["ph"] for _, sizes in tables_in.planes))
                                   for _ in range(2))
        largest_rhs = max(sizes["ph"] for _, sizes in tables_out.planes)
        direct_out, crossed_out = np.empty(largest_rhs), np.empty(largest_rhs)
    vvvo_rows, vooo_rows = _seed_sources(vooo, vvvo, no, nv)
    a_seed = np.zeros(max(sizes["a"][0] for _, sizes in tables_out.planes))
    b_seed = np.zeros(max(sizes["b"][0] for _, sizes in tables_out.planes))
    seeds = np.empty(max(sizes["plane"] for _, sizes in tables_out.planes))
    if not have_t3:
        direct_out = crossed_out = seeds
    for k in range(no):
        tables_out.t2_seeds(vvvo_rows, vooo_rows, t2, k, a_seed, b_seed, threads, source=shift)
        planes_in, sizes_in = tables_in.planes[k]
        planes_out, sizes_out = tables_out.planes[k]
        if have_t3:
            library.bt3_plane(ctypes.addressof(layout_in), ctypes.addressof(planes_in), t3.ctypes.data,
                              plane.ctypes.data, crossed_rhs.ctypes.data, direct_rhs.ctypes.data,
                              threads)
            n_pairs, widths_out, base_out = sizes_out["pair"]
            _, widths_in, base_in = sizes_in["pair"]
            n_out, ncols_out, ph_out = sizes_out["rhs"]
            n_in, ncols_in, ph_in = sizes_in["rhs"]
            xs = np.arange(groups)
            if not (np.array_equal(widths_out, widths_in[xs ^ shift]) and np.array_equal(ncols_out, ncols_in)):
                raise ValueError("The source and target planes do not match")
            _products(library, [(n_pairs[x], widths_out[x], n_pairs[x ^ shift], occ_off[x],
                                 base_in[x ^ shift], base_out[x]) for x in xs],
                      occ_mats, plane, seeds, threads)
            ph_blocks = [(n_out[s], ncols_out[s], n_in[s], ph_off[s], ph_in[s], ph_out[s]) for s in xs]
            _products(library, ph_blocks, dir_mats, direct_rhs, direct_out, threads)
            _products(library, ph_blocks, cro_mats, crossed_rhs, crossed_out, threads)
        library.bt3_seeds(ctypes.addressof(layout_out), ctypes.addressof(planes_out), int(have_t3),
                          a_seed.ctypes.data, b_seed.ctypes.data, direct_out.ctypes.data,
                          crossed_out.ctypes.data, seeds.ctypes.data, threads)
        targets, start, contributions = tables_out.scatter[k]
        library.bt3_scatter(ctypes.addressof(layout_out), ctypes.addressof(planes_out), len(targets),
                            targets.ctypes.data, start.ctypes.data, contributions.ctypes.data,
                            seeds.ctypes.data, residual.ctypes.data, threads)
    if have_t3:
        _ladder_shifted(blocked_in, blocked_out, residual, t3, prepared, threads, budget=ladder_budget)
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


def _block_plan(blocks):
    """(block table, offset tables, zeroed regions) of `_products`, cached by shape.

    Each block (m, n, k, a_offset, b_base, o_base) is out (m x n, row-major at
    o_base) = A (m x k, row-major at a_offset) @ B (k x n, row-major at b_base);
    k = 0 leaves its output zero."""
    plan = _SECTOR_PLANS.get(blocks)
    if plan is not None:
        return plan
    table, columns, regions = [], [[] for _ in range(6)], []
    r0 = c0 = k0 = 0
    for m, n, k, a_offset, b_base, o_base in blocks:
        if not (m and n):
            continue
        regions.append((o_base, m * n))
        if not k:
            continue
        table.append((m, n, k, r0, c0, k0, a_offset, b_base, o_base))
        rows, cols = np.arange(m, dtype=np.int64), np.arange(n, dtype=np.int64)
        depth = np.arange(k, dtype=np.int64)
        for column, part in zip(columns, (rows * k, depth, depth * n, cols, rows * n, cols)):
            column.append(part)
        r0, c0, k0 = r0 + m, c0 + n, k0 + k
    table = np.array(table, dtype=np.int64).reshape(-1, 9)
    offsets = tuple(np.concatenate(c) if c else np.zeros(1, np.int64) for c in columns)
    if len(_SECTOR_PLANS) > 4096:
        _SECTOR_PLANS.clear()
    plan = _SECTOR_PLANS[blocks] = (table, offsets, tuple(regions))
    return plan


def _products(library, blocks, matrices, operands, out, threads=None):
    """The `_block_plan` products of `blocks` in one call of the core's GEMM
    executor (the contraction engine of the CPU CCSDTQ solver: no BLAS, all
    OpenMP threads, sectors and tiles in parallel); every output block is
    written (zero where its sum is empty)."""
    from ._runtime import cpu_threads

    threads = cpu_threads(threads)
    table, offsets, regions = _block_plan(tuple(tuple(int(x) for x in block) for block in blocks))
    for start, length in regions:           # the executor adds into out
        library.scale(out.ctypes.data + 8 * start, length, 0.0, threads)
    if len(table):
        library.sym_contract(len(table), table.ctypes.data, matrices.ctypes.data,
                             operands.ctypes.data, out.ctypes.data,
                             *(x.ctypes.data for x in offsets), 1.0, 0, threads)


def _grouped(library, sizes, a_offsets, matrices, operands, bases, out, threads=None):
    """out[base] = block @ operand[base] for each sector, (m x m) @ (m x n)."""
    _products(library, [(m, n, m, a_offset, base, base)
                        for (m, n), a_offset, base in zip(sizes, a_offsets, bases)],
              matrices, operands, out, threads)


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


def _shifted_ladder(blocked, wvvvv, shift):
    """(shift, pair lists, even and odd sector blocks) of a ladder matrix W of
    irrep `shift` for `_ladder_shifted`: each kind's blocks from pair sector
    x ^ shift to x packed one after another, with offsets and a presence
    flag by output sector. Raises NotBlockable when W couples other sectors."""
    groups, irrep, shift = blocked.groups, blocked.virtual_irreps, int(shift)
    (p, q, r, s), even_blocks, odd_blocks = _ladder_sectors(wvvvv, irrep)
    p, q, r, s = (np.ascontiguousarray(x, dtype=np.int64) for x in (p, q, r, s))
    kinds = []
    for (x, y), found in (((p, q), even_blocks), ((r, s), odd_blocks)):
        label = irrep[x] ^ irrep[y]
        mats, offsets, present, total = [], np.zeros(groups, dtype=np.int64), np.zeros(groups, bool), 0
        for rows, cols, block in found:
            whole = rows.start is None
            if whole or int(label[rows.start] ^ label[cols.start]) != shift:
                if np.any(block):
                    raise NotBlockable("The ladder matrix is not of the direction's irrep")
                continue
            sector = int(label[rows.start])
            mats.append(np.ascontiguousarray(block).reshape(-1))
            offsets[sector], present[sector] = total, True
            total += block.size
        kinds.append((np.concatenate(mats) if mats else np.empty(0), offsets, present))
    return shift, (p, q, r, s), kinds[0], kinds[1]


def _ladder_shifted(source, blocked, residual, t3, prepared, threads, *, budget=LADDER_BUDGET):
    """`ladder` for a W of irrep `shift` (`_shifted_ladder`) on a T3 blocked
    in `source` (orbital labels), into `residual` blocked in `blocked` (the
    occupied labels XOR shift): pair rows are gathered with the source's slot
    irreps and scattered with the target's; the columns of a slot agree."""
    library = _cpu_kernels.library()
    groups, nv = blocked.groups, blocked.nv
    irrep = blocked.virtual_irreps
    shift, (p, q, r, s), (even_mats, even_off, even_present), (odd_mats, odd_off, odd_present) = prepared
    even_label, odd_label = irrep[p] ^ irrep[q], irrep[r] ^ irrep[s]
    even_start = np.searchsorted(even_label, np.arange(groups + 1)).astype(np.int64)
    odd_start = np.searchsorted(odd_label, np.arange(groups + 1)).astype(np.int64)
    even_row = np.zeros((nv, nv), dtype=np.int64)
    even_row[p, q] = even_row[q, p] = np.arange(len(p)) - even_start[even_label]
    odd_row = np.zeros((nv, nv), dtype=np.int64)
    odd_row[r, s] = odd_row[s, r] = np.arange(len(r)) - odd_start[odd_label]
    odd_sign = np.zeros((nv, nv))
    odd_sign[r, s], odd_sign[s, r] = 1.0, -1.0
    n_even, n_odd = np.diff(even_start), np.diff(odd_start)
    xs = np.arange(groups)
    nslots = len(blocked.slot_irrep)
    chunk = _ladder_chunk(max(1, nv // groups), nslots, len(p) + len(r), budget)

    def tables(layout, first, count):
        third = layout.count[xs[:, None] ^ layout.slot_irrep[first:first + count]]
        widths = 3 * third                                   # (pi, slot) columns
        colstart = np.ascontiguousarray(np.cumsum(widths, axis=1) - widths, dtype=np.int64)
        ncols = widths.sum(axis=1).astype(np.int64)
        sym_size, anti_size = n_even * ncols, n_odd * ncols
        sym_base = (np.cumsum(sym_size) - sym_size).astype(np.int64)
        anti_base = (np.cumsum(anti_size) - anti_size).astype(np.int64)
        structure = _core.structure(_core.BlockedLadder, dict(first_slot=first, nslots=count), dict(
            even_p=p, even_q=q, even_start=even_start, odd_r=r, odd_s=s, odd_start=odd_start,
            even_row=even_row, odd_row=odd_row, odd_sign=odd_sign, colstart=colstart,
            ncols=ncols, sym_base=sym_base, anti_base=anti_base))
        return structure, ncols, sym_base, anti_base, int(sym_size.sum()), int(anti_size.sum())

    for first in range(0, nslots, chunk):
        count = min(chunk, nslots - first)
        gather, ncols_in, sym_in_base, anti_in_base, sym_in_size, anti_in_size = tables(source, first, count)
        scatter, ncols_out, sym_out_base, anti_out_base, sym_out_size, anti_out_size = tables(
            blocked, first, count)
        if not np.array_equal(ncols_out, ncols_in[xs ^ shift]):
            raise ValueError("The source and target ladder columns do not match")
        sym_in, anti_in = np.empty(sym_in_size), np.empty(anti_in_size)
        library.bt3_ladder_gather(ctypes.addressof(source.c_layout()), ctypes.addressof(gather),
                                  t3.ctypes.data, sym_in.ctypes.data, anti_in.ctypes.data, threads)
        sym_out, anti_out = np.empty(sym_out_size), np.empty(anti_out_size)
        _products(library, [(n_even[x], ncols_out[x], n_even[x ^ shift] if even_present[x] else 0,
                             even_off[x], sym_in_base[x ^ shift], sym_out_base[x]) for x in xs],
                  even_mats, sym_in, sym_out, threads)
        _products(library, [(n_odd[x], ncols_out[x], n_odd[x ^ shift] if odd_present[x] else 0,
                             odd_off[x], anti_in_base[x ^ shift], anti_out_base[x]) for x in xs],
                  odd_mats, anti_in, anti_out, threads)
        del sym_in, anti_in
        library.bt3_ladder_scatter(ctypes.addressof(blocked.c_layout()), ctypes.addressof(scatter),
                                   sym_out.ctypes.data, anti_out.ctypes.data, residual.ctypes.data,
                                   threads)


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
