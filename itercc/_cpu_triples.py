"""CPU assembly of the restricted CCSD and CCSDT residual terms.

The generic loop in `RestrictedCC._triples` builds one (j, k) seed at a time
with small GEMMs and tile-sized NumPy operations; on the CPU that is
memory-bound and uses a few cores. Here:

* the seeds of a whole k plane come from five large GEMMs, and the one-body
  virtual term is folded into the ladder matrix;
* the T3 feedback uses two GEMMs per occupied plane;
* with the compiled core (`_cpu_kernels`) every GEMM runs on its GEMM
  executor (`product_into`, `matmul`: all threads, no BLAS), one fused pass
  per plane builds the GEMM operands and one assembles the seeds, and each
  thread owns whole rows of R3 in the scatter. `CompiledContract` puts the
  CCSD and lower-rank products on the same executor;
* without it, the GEMMs run in NumPy and copies, spin sums and transposed
  sums on a Python thread pool, overlapping the GEMMs of the neighbouring
  planes.

Thread counts come from `cpu_threads` (a solver's `thread_scope`,
ITERCC_CPU_THREADS, or one per core of the affinity mask). Each task writes its own part of the
output in a fixed order, so the result does not depend on the thread count.
"""

from concurrent.futures import ThreadPoolExecutor, wait
from functools import partial
import threading

import numpy as np

from . import _cpu_kernels as _kernels
from ._pair_symmetry import PairSectors

_ORIENTATIONS = ((0, 1, 2), (0, 2, 1), (1, 2, 0))
_SLOT_BATCH = 8         # slots per ladder or purification operation


# Threads and host memory live in `_runtime` (re-exported for older imports).
from ._runtime import (_PROCESS_CORES, _PROCESS_CPU_SET, _PROCESS_CPUS, _available_cores,  # noqa: F401,E402
                       _available_cpus, _cores, _id_list, cpu_threads, host_memory_limit,
                       resident_bytes, thread_scope)


# Products on the core's GEMM executor -------------------------------------
#
# The restricted CCSD and CCSDT products run on the contraction engine of the
# CPU CCSDTQ solver (`itercc_sym_contract`, core/src/cpu/symmetry_blocks.cpp):
# it reads operands in place through offset tables, adds register tiles
# straight into the output, uses every OpenMP thread (NumPy's OpenBLAS stops
# at 64 threads and keeps a second thread pool spinning between calls) and
# gives the same results for any thread count. Without the compiled core the
# NumPy expressions run instead.

def _engine():
    """The package's dense product engine (`_cpu_quadruples.dense_products`),
    or None without the core."""
    from ._cpu_quadruples import dense_products

    return dense_products()


def _float64(*arrays):
    return all(isinstance(x, np.ndarray) and x.dtype == np.float64 for x in arrays)


def product_into(spec, a, b, out, alpha=1.0, beta=0.0, threads=None):
    """out = beta * out + alpha * einsum(spec, a, b) on the core executor.

    Returns False, with `out` untouched, without the compiled core, for other
    dtypes, or when `out` overlaps an operand.
    """
    engine = _engine()
    if (engine is None or not _float64(out, a, b) or not out.flags.writeable
            or np.may_share_memory(out, a) or np.may_share_memory(out, b)):
        return False
    engine(spec, a, b, out, float(alpha), float(beta), threads=threads)
    return True


_NARROW = 16          # right-hand sides up to this many columns: `itercc_gemm_narrow`


def _narrow(a, b, out):
    """Whether a @ b streams a large `a` (rows enough to share among threads)
    against a few columns of `b`."""
    if not (b.ndim == 2 and a.ndim == 2 and 0 < b.shape[1] <= _NARROW and a.shape[0] >= 256
            and a.shape[0] * a.shape[1] >= 1 << 16
            and all(isinstance(x, np.ndarray) and x.dtype == np.float64 for x in (a, b, out))
            and out.shape == (a.shape[0], b.shape[1]) and a.shape[1] == b.shape[0]):
        return False
    return all(x.strides[1] == 8 and x.strides[0] > 0 and x.strides[0] % 8 == 0 for x in (a, b, out))


def matmul(a, b, out, *, alpha=1.0, beta=0.0, threads=None):
    """out = beta * out + alpha * (a @ b) for a matrix `a` and a matrix or a stack of
    matrices `b`, on the core executor when compiled; returns out. A large `a`
    against a few columns (the pair-sector ladders: one column per occupied
    pair) streams through `itercc_gemm_narrow` (the executor's tiles reached a
    fifth of the memory bandwidth there)."""
    if _narrow(a, b, out):
        library = _kernels.library()
        if library is not None and out.flags.writeable:
            library.gemm_narrow(a.ctypes.data, a.strides[0] // 8, b.ctypes.data, b.strides[0] // 8,
                                out.ctypes.data, out.strides[0] // 8, a.shape[0], a.shape[1], b.shape[1],
                                float(alpha), float(beta), cpu_threads(threads))
            return out
    if a.ndim == 2 and b.ndim in (2, 3):
        spec = "ik,kj->ij" if b.ndim == 2 else "mk,ikn->imn"
        if product_into(spec, a, b, out, alpha, beta, threads):
            return out
    if alpha == 1.0 and beta == 0.0:
        return np.matmul(a, b, out=out)
    value = np.matmul(a, b)
    if beta == 0.0:
        np.multiply(value, alpha, out=out)
    else:
        out *= beta
        out += alpha * value
    return out


def _output_shape(spec, operands):
    inputs, output = spec.replace(" ", "").split("->")
    extent = {}
    for letters, x in zip(inputs.split(","), operands):
        extent.update(zip(letters, x.shape))
    return tuple(extent[c] for c in output)


class CompiledContract:
    """The `contract` of `_restricted_equations` for NumPy arrays on the CPU, with
    the products on the core executor (see `_restricted_equations.add_product`):
    `into` adds a product to an existing array without a temporary, `copy` and
    `combine` are whole-array operations on all threads."""

    LARGE = 1 << 16             # elements below which NumPy's copy is as fast

    def __init__(self, threads=None):
        self.threads = threads

    def __call__(self, spec, *operands):
        if len(operands) == 2 and _float64(*operands) and "->" in spec:
            out = np.empty(_output_shape(spec, operands))
            if product_into(spec, *operands, out, 1.0, 0.0, self.threads):
                return out
        return np.einsum(spec, *operands, optimize=True)

    def into(self, spec, a, b, out, alpha=1.0):
        return product_into(spec, a, b, out, alpha, 1.0, self.threads)

    def copy(self, x):
        if _float64(x) and x.size >= self.LARGE and _kernels.library() is not None:
            from . import _cpu_quadruples

            return _cpu_quadruples.copy(x)
        return np.array(x, order="C", copy=True)

    def combine(self, a, b, wa, wb):
        """wa * a + wb * b as a new C-ordered array (same arithmetic as NumPy), or None."""
        if not (_float64(a, b) and a.size >= self.LARGE and _kernels.library() is not None):
            return None
        from ._cpu_quadruples import permuted_add

        out = permuted_add(np.empty(a.shape), None, a, 0.0, wa)
        return permuted_add(out, out, b, 1.0, wb)


class ThreadPool:
    """Worker threads for index ranges, and two coordinators for overlapped stages.

    The executors start on first use, so a pool that only carries the thread
    count for the compiled kernels costs nothing.
    """

    def __init__(self, workers):
        self.workers = max(1, int(workers))
        self._executor = self._stages = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for executor in (self._stages, self._executor):
            if executor is not None:
                executor.shutdown()
        self._executor = self._stages = None

    def _start(self):
        if self.workers > 1 and self._executor is None:
            self._executor = ThreadPoolExecutor(self.workers)
            self._stages = ThreadPoolExecutor(2)

    def run(self, task, count):
        """Call task(start, stop) on contiguous chunks that cover range(count)."""
        self._start()
        chunks = 1 if self._executor is None else min(count, 2 * self.workers)
        if chunks <= 1:
            if count > 0:
                task(0, count)
            return
        bounds = [count * c // chunks for c in range(chunks + 1)]
        futures = [self._executor.submit(task, a, b) for a, b in zip(bounds, bounds[1:]) if b > a]
        wait(futures)
        for future in futures:
            future.result()

    def stage(self, *steps):
        """Run `steps` in order, in the background when there are workers."""
        self._start()
        if self._stages is None:
            for step in steps:
                step()
            return None
        return self._stages.submit(lambda: [step() for step in steps])

    @staticmethod
    def join(*handles):
        for handle in handles:
            if handle is not None:
                handle.result()


class _Rows:
    """Order (occupied, virtual) rows by particle-hole irrep for block GEMMs.

    `rows[i]` selects the grouped rows of occupied index i: a slice when no
    labels are given (natural order), else an index array.
    """

    def __init__(self, labels, no, nv):
        sectors = None if labels is None else PairSectors(labels)
        if sectors is None or len(sectors.bands) <= 1:
            self.sectors = None
            self.rows = [slice(i * nv, (i + 1) * nv) for i in range(no)]
        else:
            self.sectors = sectors
            self.rows = [sectors.inverse[i * nv:(i + 1) * nv] for i in range(no)]

    def blocks(self, matrix):
        """`_sector_blocks` of `matrix` in the grouped order, or the whole matrix."""
        if self.sectors is None:
            return [(slice(None), slice(None), matrix)]
        order = self.sectors.order
        grouped = np.ascontiguousarray(matrix[np.ix_(order, order)])
        blocks = _sector_blocks(self.sectors, grouped)
        return [(slice(None), slice(None), grouped)] if blocks is None else blocks


_SCAN_THREADED_BYTES = 32 << 20
_SCAN_POOL, _SCAN_LOCK = [], threading.Lock()


def _scan_pool():
    with _SCAN_LOCK:
        if not _SCAN_POOL:
            _SCAN_POOL.append(ThreadPoolExecutor(max(1, min(16, cpu_threads())),
                                                 thread_name_prefix="itercc-scan"))
    return _SCAN_POOL[0]


def _sector_blocks(sectors, grouped):
    """[(row band, column band, block)] of a matrix in sector order whose every
    nonzero entry couples a row of sector s to a column of sector s ^ G, for
    one G (G = 0: block-diagonal, a ground-state operator; G != 0: the
    derivative operator of a direction of irrep G), or None. The check is
    exact; rows whose partner sector is absent have no block (zero)."""
    labels, bands = [int(x) for x in sectors.labels], sectors.bands
    if len(bands) <= 1:
        return None
    pairs = [(i, j) for i in range(len(bands)) for j in range(len(bands))]

    def found(pair):
        return bool(np.any(grouped[bands[pair[0]], bands[pair[1]]]))
    if grouped.nbytes < _SCAN_THREADED_BYTES:
        nonzero = [found(pair) for pair in pairs]
    else:
        # Large pair matrices (about 0.6 GB at 131 virtuals) are scanned on a
        # persistent pool (NumPy releases the GIL in the reduction). A pool per
        # call started up to 16 threads several times per Jacobian action for
        # matrices of a few kB: 18% of an EOM-CCSDT action in aug-cc-pVDZ.
        nonzero = list(_scan_pool().map(found, pairs))
    shift = None
    for (i, j), found in zip(pairs, nonzero):
        if found:
            if shift is None:
                shift = labels[i] ^ labels[j]
            elif labels[i] ^ labels[j] != shift:
                return None
    shift = shift or 0
    where = {label: band for label, band in zip(labels, bands)}
    library = _kernels.library()

    def block(rows, cols):
        view = grouped[rows, cols]
        if library is not None and view.dtype == np.float64 and view.size:
            from ._cpu_quadruples import copy

            return copy(view)                   # all threads
        return np.ascontiguousarray(view)
    return [(rows, where[label ^ shift], block(rows, where[label ^ shift]))
            for label, rows in zip(labels, bands) if label ^ shift in where]


def _product(blocks, rhs, out, threads=None):
    """out[rows] = block @ rhs[columns] for each (row band, column band, block)
    of `_sector_blocks`; rows that no block reaches are zero."""
    covered = 0
    for rows, cols, block in blocks:
        matmul(block, rhs[cols], out[rows], threads=threads)
        covered += len(range(*rows.indices(len(out))))
    if covered < len(out):
        mask = np.ones(len(out), dtype=bool)
        for rows, _, _ in blocks:
            mask[rows] = False
        out[mask] = 0.0
    return out


def _gather_plane(pool, layout, t3, out, fixed, axis):
    """out[x, y] = T(fixed, x, y) (axis 0) or T(x, y, fixed) (axis 2), tile by tile."""
    no = layout.no

    def plane_tiles(start, stop):
        for index in range(start, stop):
            x, y = divmod(index, no)
            indices = (fixed, x, y) if axis == 0 else (x, y, fixed)
            slot = layout.slots[indices]
            if slot < 0:
                out[x, y] = 0.0
            else:
                out[x, y] = t3[slot].transpose(layout.from_sorted[indices])

    pool.run(plane_tiles, no * no)


# Doubles of gathered ladder buffers per chunk of the compiled paths (2 GiB).
# The gather and scatter give each thread whole slots, so chunks must hold a
# socket's worth of slots (the former 2**25 gave 10 slots at 131 virtuals).
# The chunk shapes the GEMMs, so it depends on the sizes only, never on the
# thread count or memory (the same results for any thread count).
LADDER_BUDGET = 2**28


def triples(layout, t2, t3, have_t3, im, vooo, vvvo, g, o, v, pool, *,
            ph_irreps=None, virtual_irreps=None, ladder_budget=LADDER_BUDGET, wvvvv=None):
    """Packed R3 from T2 and T3, as `RestrictedCC._triples` on the CPU.

    For each k, the seed S(i,j,k)[a,b,c] collects the T2-driven terms and the
    particle-hole and occupied-ladder products of the plane T(x,y,k), and is
    added to the canonical slot of (i,j,k) with its paired permutations.
    `wvvvv` replaces the ladder matrix built from (g, t2, im.vv); Jacobian
    actions pass the ground or tangent matrix, and then g is not used.
    """
    library = _kernels.library()
    if library is not None:
        return _triples_compiled(library, layout, t2, t3, have_t3, im, vooo, vvvo, g, o, v,
                                 pool.workers, ph_irreps, virtual_irreps, ladder_budget, wvvvv)
    no, nv = layout.no, layout.nv
    plane_shape = (no, no, nv, nv, nv)
    residual = np.zeros(layout.shape)
    rows = _Rows(ph_irreps, no, nv)
    grouped = rows.sectors is not None
    vvvo_rows, vooo_rows = _seed_sources(vooo, vvvo, no, nv)
    # Two buffer sets: GEMMs of plane k overlap the operands of k + 1 and
    # the seed assembly and scatter of k - 1.
    rhs_shape = (no * nv, no, nv, nv)
    outputs = [dict(a=np.empty((no, nv, nv, no, nv)), b=np.empty(plane_shape),
                    seeds=np.empty(plane_shape)) for _ in range(2)]
    if have_t3:
        occupied, direct_blocks, crossed_blocks = _plane_matrices(im, rows, no, nv)
        # Operands: the plane, and rows (l, d) in particle-hole order with
        # columns (j, b, c) for the crossed and direct products.
        inputs = [dict(plane=np.empty(plane_shape), crossed=np.empty(rhs_shape),
                       direct=np.empty(rhs_shape)) for _ in range(2)]
        for out in outputs:
            out.update(direct=np.empty(rhs_shape), crossed=np.empty(rhs_shape))

    def operands(buffers):
        plane, crossed_rhs, direct_rhs = buffers["plane"], buffers["crossed"], buffers["direct"]

        def plane_rows(start, stop):
            scratch = np.empty((no, nv, nv, nv)) if grouped else None
            for l in range(start, stop):
                # crossed: row (l, d) holds P[j, l, d, b, c]
                crossed_rhs[rows.rows[l]] = plane[:, l].transpose(1, 0, 2, 3)
                # direct: row (l, d) holds spin_sum(P[l, j])[d, b, c]
                source = plane[l]
                target = scratch if grouped else direct_rhs[rows.rows[l]].transpose(1, 0, 2, 3)
                np.multiply(source, 2, out=target)
                target -= source.swapaxes(1, 2)
                target -= source.swapaxes(1, 3)
                if grouped:
                    direct_rhs[rows.rows[l]] = scratch.transpose(1, 0, 2, 3)

        return plane_rows

    def gemms(k, buffers, out):
        _t2_seeds(vvvo_rows, vooo_rows, t2, k, out["a"], out["b"])
        if have_t3:
            # The occupied ladder is written straight into the seeds.
            np.matmul(occupied, buffers["plane"].reshape(no * no, nv**3),
                      out=out["seeds"].reshape(no * no, nv**3))
            _product(direct_blocks, buffers["direct"].reshape(no * nv, -1),
                     out["direct"].reshape(no * nv, -1))
            _product(crossed_blocks, buffers["crossed"].reshape(no * nv, -1),
                     out["crossed"].reshape(no * nv, -1))

    def combine(out):
        seeds, a_seed, b_seed = out["seeds"], out["a"], out["b"]

        def seed_rows(start, stop):
            tile = np.empty((nv, no, nv, nv)) if grouped and have_t3 else None
            for i in range(start, stop):
                seed = seeds[i]                               # (j, a, b, c)
                if have_t3:
                    seed += a_seed[:, :, :, i]
                    direct = (np.take(out["direct"], rows.rows[i], axis=0, out=tile) if grouped
                              else out["direct"][rows.rows[i]])          # (a, j, b, c)
                    seed += direct.transpose(1, 0, 2, 3)
                    # crossed (x, j, y, c): -T(i,b,j,a,c) - 0.5 T(i,a,j,b,c);
                    # each block is read once, so it may be scaled in place.
                    cross = (np.take(out["crossed"], rows.rows[i], axis=0, out=tile) if grouped
                             else out["crossed"][rows.rows[i]])
                    seed -= cross.transpose(1, 2, 0, 3)
                    cross *= 0.5
                    seed -= cross.transpose(1, 0, 2, 3)
                else:
                    np.copyto(seed, a_seed[:, :, :, i])
                seed -= b_seed[i]

        return seed_rows

    pairs = [(x, y) for x in range(no) for y in range(x, no)]

    def scatter(k, out):
        seeds = out["seeds"]

        def seed_tiles(start, stop):
            for x, y in pairs[start:stop]:
                for i, j in ((x, y),) if x == y else ((x, y), (y, x)):
                    slot = layout.slots[i, j, k]
                    if slot < 0:
                        continue
                    seed = seeds[i, j].transpose(layout.to_sorted[i, j, k])
                    target = residual[slot]
                    target += seed
                    # Equal occupied labels: the second virtual permutation.
                    first, second, third = layout.indices[slot]
                    if first == second:
                        target += seed.swapaxes(0, 1)
                    elif second == third:
                        target += seed.swapaxes(1, 2)

        return seed_tiles

    def stage_in(k):
        buffers = inputs[k % 2]
        return (partial(_gather_plane, pool, layout, t3, buffers["plane"], k, 2),
                partial(pool.run, operands(buffers), no))

    def stage_out(k):
        # A slot receives (i, j) and (j, i) of one unordered pair, so pairs
        # partition the writes of one plane; planes are scattered in order.
        out = outputs[k % 2]
        return (partial(pool.run, combine(out), no), partial(pool.run, scatter(k, out), len(pairs)))

    if have_t3:
        pool.join(pool.stage(*stage_in(0)))
    for k in range(no):
        handles = []
        if have_t3 and k + 1 < no:
            handles.append(pool.stage(*stage_in(k + 1)))
        if k:
            handles.append(pool.stage(*stage_out(k - 1)))
        try:
            gemms(k, inputs[k % 2] if have_t3 else None, outputs[k % 2])
        finally:
            pool.join(*handles)
    pool.join(pool.stage(*stage_out(no - 1)))
    del outputs
    if have_t3:
        del inputs
        if wvvvv is None:
            wvvvv = _ladder_matrix(g, o, v, t2, im, nv)
        pair_ladder(residual, t3, wvvvv, pool, virtual_irreps=virtual_irreps, budget=ladder_budget)
    purify(residual, pool)
    return residual


def _seed_sources(vooo, vvvo, no, nv):
    """GEMM operands of the T2-driven seeds: rows (j,a,b) by d, and (i,j,a) by l."""
    vvvo_rows = _transposed_copy(vvvo, (3, 0, 1, 2)).reshape(no * nv * nv, nv)
    vooo_rows = _transposed_copy(vooo, (2, 3, 0, 1)).reshape(no * no * nv, no)
    return vvvo_rows, vooo_rows


def _transposed_copy(x, axes):
    """np.ascontiguousarray(x.transpose(axes)), on all threads when compiled."""
    from . import _cpu_quadruples

    view = x.transpose(axes)
    if x.dtype == np.float64 and _cpu_quadruples.available():
        return _cpu_quadruples.permuted_add(np.empty(view.shape), None, view, 0.0, 1.0)
    return np.ascontiguousarray(view)


def _t2_seeds(vvvo_rows, vooo_rows, t2, k, a_seed, b_seed, threads=None):
    """A[j,a,b,i,c] = sum_d W(abdj) t2(ikdc) and B[i,j,a,b,c] = sum_l W(alij) t2(lkbc)."""
    no, nv = t2.shape[0], t2.shape[2]
    t2k = t2[:, k]
    # The executor reads t2(i, k, d, c) in place (no transposed copy).
    if not product_into("rd,idc->ric", vvvo_rows, t2k, a_seed.reshape(no * nv * nv, no, nv),
                        threads=threads):
        np.matmul(vvvo_rows, t2k.transpose(1, 0, 2).reshape(nv, no * nv),
                  out=a_seed.reshape(no * nv * nv, no * nv))
    if not product_into("rl,lbc->rbc", vooo_rows, t2k, b_seed.reshape(no * no * nv, nv, nv),
                        threads=threads):
        np.matmul(vooo_rows, t2k.reshape(no, nv * nv), out=b_seed.reshape(no * no * nv, nv * nv))


def _plane_matrices(im, rows, no, nv):
    """Occupied-ladder matrix and the blocked direct and crossed particle-hole matrices."""
    # The one-body occupied term is folded into the occupied ladder.
    woooo = im.oooo * 0.5
    for j in range(no):
        woooo[:, j, :, j] -= 0.5 * im.oo
    occupied = np.ascontiguousarray(woooo.transpose(2, 3, 0, 1)).reshape(no**2, no**2)
    direct = np.ascontiguousarray(im.ovvo.transpose(3, 1, 0, 2)).reshape(no * nv, no * nv)
    direct *= 0.25
    crossed = np.ascontiguousarray(im.ovov.transpose(2, 1, 0, 3)).reshape(no * nv, no * nv)
    return occupied, rows.blocks(direct), rows.blocks(crossed)


def _ladder_matrix(g, o, v, t2, im, nv, threads=None):
    """W(ab,de) of the virtual ladder, with the one-body virtual term folded in.

    Summed over the six occupied permutations, the seed term
    0.5 vv(ad) T(i,j,k; d,b,c) acts once on each virtual index. Over the
    three ladder pairs that is 0.5 (vv x 1 + 1 x vv) on every pair.
    """
    wvvvv = _ladder_base(g, o, v, t2, nv, threads)
    add_one_body(wvvvv, im.vv)
    return wvvvv


def _ladder_base(g, o, v, t2, nv, threads=None):
    """g(vvvv) + sum_lm g(lm,de) t2(lm,ab): the ladder matrix without its one-body term."""
    wvvvv = None
    if _engine() is not None and _float64(g, t2):
        wvvvv = CompiledContract(threads).copy(g[v, v, v, v])
        if not product_into("lmde,lmab->abde", g[o, o, v, v], t2, wvvvv, 1.0, 1.0, threads):
            wvvvv = None
    if wvvvv is None:
        wvvvv = np.einsum("lmde,lmab->abde", g[o, o, v, v], t2, optimize=True)
        wvvvv += g[v, v, v, v]
    return wvvvv


def add_one_body(w, vv):
    """w(ab,de) += 0.5 (vv(a,d) delta(b,e) + delta(a,d) vv(b,e)), along the diagonals
    (no nv^4 temporaries)."""
    half = 0.5 * vv
    for b in range(w.shape[1]):
        w[:, b, :, b] += half
    for a in range(w.shape[0]):
        w[a, :, a, :] += half


def _ladder_sectors(wvvvv, virtual_irreps):
    """Pair lists in sector order and the blocked symmetric/antisymmetric matrices."""
    nv = wvvvv.shape[0]
    p, q = np.triu_indices(nv)
    r, s = np.triu_indices(nv, 1)
    even_sectors = odd_sectors = None
    if virtual_irreps is not None:
        even_sectors = PairSectors(virtual_irreps[p] ^ virtual_irreps[q])
        odd_sectors = PairSectors(virtual_irreps[r] ^ virtual_irreps[s])
        p, q = p[even_sectors.order], q[even_sectors.order]
        r, s = r[odd_sectors.order], s[odd_sectors.order]
    library = _kernels.library()
    if library is not None and wvvvv.dtype == np.float64 and wvvvv.shape == (nv,) * 4:
        # The same entries and arithmetic on all threads (four-index fancy
        # indexing reads W at random on one thread: about 0.5 s per call at
        # 88 virtuals).
        if wvvvv.flags.c_contiguous:
            w = wvvvv
        else:                   # g(vvvv) of the dressed integrals: copied on all threads
            from ._cpu_quadruples import copy

            w = copy(wvvvv)
        p, q, r, s = (np.ascontiguousarray(x, dtype=np.int64) for x in (p, q, r, s))
        even, odd = np.empty((len(p), len(p))), np.empty((len(r), len(r)))
        library.pair_matrices(w.ctypes.data, nv, len(p), p.ctypes.data, q.ctypes.data, len(r),
                              r.ctypes.data, s.ctypes.data, even.ctypes.data, odd.ctypes.data,
                              cpu_threads(None))
    else:
        even = wvvvv[p[:, None], q[:, None], p, q]
        even += wvvvv[p[:, None], q[:, None], q, p] * (p != q)
        odd = wvvvv[r[:, None], s[:, None], r, s]
        odd -= wvvvv[r[:, None], s[:, None], s, r]

    def blocks(sectors, matrix):
        found = None if sectors is None else _sector_blocks(sectors, matrix)
        return [(slice(None), slice(None), matrix)] if found is None else found

    return (p, q, r, s), blocks(even_sectors, even), blocks(odd_sectors, odd)


class LadderSectors:
    """The pair sectors of the virtual ladders, for operators that are
    block-diagonal in pair irreps (exactly symmetric integrals and T1).

    Pairs are ordered by sector as in `_ladder_sectors`; `build` forms only the
    diagonal sector blocks of the even and odd pair matrices, packed one sector
    after another in the layout the GEMM executor reads, straight from a
    strided view of g(vvvv): no copy of the integrals, no dense pair matrix.
    """

    def __init__(self, virtual_irreps, groups):
        virtual_irreps = np.asarray(virtual_irreps, dtype=np.int64)
        nv = len(virtual_irreps)
        self.nv, self.groups = nv, int(groups)
        pairs = []
        for x, y in (np.triu_indices(nv), np.triu_indices(nv, 1)):
            label = virtual_irreps[x] ^ virtual_irreps[y]
            order = np.argsort(label, kind="stable")
            x, y, label = (np.ascontiguousarray(a[order], dtype=np.int64) for a in (x, y, label))
            start = np.searchsorted(label, np.arange(self.groups + 1)).astype(np.int64)
            size = np.diff(start) ** 2
            offset = (np.cumsum(size) - size).astype(np.int64)
            pairs.append((x, y, start, offset, int(size.sum())))
        self.even, self.odd = pairs

    @property
    def pairs(self):
        return self.even[0], self.even[1], self.odd[0], self.odd[1]

    def empty(self):
        return np.empty(self.even[4]), np.empty(self.odd[4])

    def build(self, w=None, *, vv=None, out=None, accumulate=False, threads=None):
        """(even, odd) sector blocks of w (a 4-index float64 view, or None) plus
        the one-body term of `vv`, into `out` (added when `accumulate`)."""
        even, odd = self.empty() if out is None else out
        strides = None
        if w is not None:
            if w.dtype != np.float64 or w.shape != (self.nv,) * 4 or any(x % 8 for x in w.strides):
                raise ValueError("expected a float64 nv^4 view of the integrals")
            strides = np.asarray([x // 8 for x in w.strides], dtype=np.int64)
        vv = None if vv is None else np.ascontiguousarray(vv, dtype=np.float64)
        (p, q, es, eo, _), (r, s, os_, oo, _) = self.even, self.odd
        _kernels.library().ladder_sectors(
            None if w is None else w.ctypes.data, None if strides is None else strides.ctypes.data,
            None if vv is None else vv.ctypes.data, self.nv, self.groups, p.ctypes.data,
            q.ctypes.data, es.ctypes.data, eo.ctypes.data, r.ctypes.data, s.ctypes.data,
            os_.ctypes.data, oo.ctypes.data, even.ctypes.data, odd.ctypes.data, int(accumulate),
            cpu_threads(threads))
        return even, odd

    def dressed(self, bare, g, no, t1, *, out=None, threads=None):
        """(even, odd) sector blocks of the T1-dressed g(vvvv) from `bare`, those
        of the bare g(vvvv) (`build`), and the bare integrals `g` (a float64
        n^4 view): g~(abcd) = g(abcd) - sum_k t(k,a) Q(k,b,c,d)
        - sum_k t(k,b) Q(k,a,d,c), Q(k,b,c,d) = g(k,b,c,d) - 0.5 sum_l t(l,b)
        g(k,l,c,d) (`itercc_ladder_sectors_t1`): one pass over the sectors and
        o v^3 terms instead of dressing g(vvvv) and reading it again."""
        if g.dtype != np.float64 or g.ndim != 4 or any(x % 8 for x in g.strides):
            raise ValueError("expected a float64 n^4 view of the integrals")
        even, odd = self.empty() if out is None else out
        strides = np.asarray([x // 8 for x in g.strides], dtype=np.int64)
        t1 = np.ascontiguousarray(t1, dtype=np.float64)
        (p, q, es, eo, _), (r, s, os_, oo, _) = self.even, self.odd
        _kernels.library().ladder_sectors_t1(
            g.ctypes.data, strides.ctypes.data, g.shape[0], int(no), t1.ctypes.data, self.groups,
            p.ctypes.data, q.ctypes.data, es.ctypes.data, eo.ctypes.data, r.ctypes.data, s.ctypes.data,
            os_.ctypes.data, oo.ctypes.data, bare[0].ctypes.data, bare[1].ctypes.data, even.ctypes.data,
            odd.ctypes.data, cpu_threads(threads))
        return even, odd

    def contract_t1(self, sectors, t1, *, threads=None):
        """(nv, nv, nv, no) products sum_d g(a,b,c,d) t1(j,d) of the g(vvvv)
        whose sector blocks are `sectors` (`itercc_sectors_contract_t1`): what
        the occupied rows and columns of its dressed planes take from it."""
        t1 = np.ascontiguousarray(t1, dtype=np.float64)
        no = t1.shape[0]
        if t1.shape != (no, self.nv):
            raise ValueError("t1 must be (occupied, virtual)")
        if getattr(self, "_odd_row", None) is None:
            (_, _, _, _, _), (r, s, _, _, _) = self.even, self.odd
            table = np.full((self.nv, self.nv), -1, dtype=np.int64)
            table[r, s] = np.arange(len(r))
            self._odd_row = table
        out = np.empty((self.nv, self.nv, self.nv, no))
        (p, q, es, eo, _), (r, s, os_, oo, _) = self.even, self.odd
        _kernels.library().sectors_contract_t1(
            self.nv, no, t1.ctypes.data, self.groups, p.ctypes.data, q.ctypes.data, es.ctypes.data,
            eo.ctypes.data, r.ctypes.data, s.ctypes.data, os_.ctypes.data, oo.ctypes.data,
            self._odd_row.ctypes.data, sectors[0].ctypes.data, sectors[1].ctypes.data, out.ctypes.data,
            cpu_threads(threads))
        return out

    def blocks(self, matrices, which):
        """[(rows, rows, block)] views of packed blocks, for `_product`."""
        _, _, start, offset, _ = self.even if which == "even" else self.odd
        found = []
        for g in range(self.groups):
            a, b = int(start[g]), int(start[g + 1])
            if b > a:
                n = b - a
                found.append((slice(a, b), slice(a, b),
                              matrices[offset[g]:offset[g] + n * n].reshape(n, n)))
        return found

    def ladder_matrix(self, g_sectors, vv, t2, g_oovv, threads=None, *, inplace=False):
        """Sector blocks of W(ab,de) = g(ab,de) + sum_lm g(lm,de) t2(lm,ab)
        + 0.5 (vv x 1 + 1 x vv) (`_ladder_matrix`) from the blocks of g: the
        t2 term as one small GEMM per sector, ranked by the occupied pairs.
        `inplace` adds the terms to `g_sectors` instead of a copy."""
        from ._cpu_quadruples import copy

        even, odd = g_sectors if inplace else (copy(x) for x in g_sectors)
        self.build(None, vv=vv, out=(even, odd), accumulate=True, threads=threads)
        no = t2.shape[0]
        for (x, y, start, offset, _), matrices, sign in ((self.even, even, 1.0), (self.odd, odd, -1.0)):
            for g in range(self.groups):
                a, b = int(start[g]), int(start[g + 1])
                if b == a:
                    continue
                xs, ys, n = x[a:b], y[a:b], b - a
                left = np.ascontiguousarray(t2[:, :, xs, ys].reshape(no * no, n).T)
                swapped = g_oovv[:, :, ys, xs]
                if sign > 0:
                    swapped = swapped * (xs != ys)
                right = np.ascontiguousarray((g_oovv[:, :, xs, ys] + sign * swapped).reshape(no * no, n))
                matmul(left, right, matrices[offset[g]:offset[g] + n * n].reshape(n, n), beta=1.0,
                       threads=threads)
        return even, odd


def doubles_ladder(gvvvv, t2, *, virtual_irreps=None, sectors=None, blocks=None):
    """0.5 sum_cd g(ab,cd) t2(ij,cd) through the pair sectors of the T3 ladder.

    g commutes with pair interchange, so t2(ij) splits into its symmetric and
    antisymmetric (c,d) parts, each multiplied by a half-size matrix that is
    block-diagonal in pair irreps when `virtual_irreps` are given. Only i <= j
    is computed: the product L(ij,ab) satisfies L(ji,ba) = L(ij,ab). Exact for
    any t2; about 16x fewer operations than the dense contraction with C2v.
    `sectors` = (LadderSectors, even, odd) supplies the sector blocks of g
    already built, or `blocks` what `_ladder_sectors` returns (`gvvvv` is then
    not read).
    """
    no, nv = t2.shape[0], t2.shape[2]
    if sectors is not None:
        structure, even, odd = sectors
        p, q, r, s = structure.pairs
        even_blocks, odd_blocks = structure.blocks(even, "even"), structure.blocks(odd, "odd")
    elif blocks is not None:
        (p, q, r, s), even_blocks, odd_blocks = blocks
    else:
        (p, q, r, s), even_blocks, odd_blocks = _ladder_sectors(gvvvv, virtual_irreps)
    i, j = np.triu_indices(no)
    pairs = t2[i, j]                                          # (pairs, c, d)
    sym_in = np.ascontiguousarray((pairs[:, p, q] + pairs[:, q, p]).T) * 0.5
    anti_in = np.ascontiguousarray((pairs[:, r, s] - pairs[:, s, r]).T) * 0.5
    sym_out = _product(even_blocks, sym_in, np.empty_like(sym_in))
    anti_out = _product(odd_blocks, anti_in, np.empty_like(anti_in))
    upper = np.empty((len(i), nv, nv))
    upper[:, p, q] = sym_out.T
    upper[:, q, p] = sym_out.T
    upper[:, r, s] += anti_out.T
    upper[:, s, r] -= anti_out.T
    ladder = np.empty_like(t2)
    ladder[i, j] = upper
    ladder[j, i] = upper.transpose(0, 2, 1)
    ladder *= 0.5
    return ladder


class BareLadder:
    """The CCSD ladder 0.5 sum_cd g~(ab,cd) t2(ij,cd) of the T1-dressed integrals
    g~ without dressing g(vvvv), which is most of a pass over n^4 per iteration.

    Axes 2 and 3 keep their virtual rows, so g~(abcd) = g(abcd)
    - sum_k t(ka) g(kbcd) - sum_k t(kb) g(akcd) + sum_kl t(ka) t(lb) g(klcd), and
    with the pair symmetry of g and t2 the ladder is L(g) - B - B(ji,ba) with
        B(ij,ab) = sum_k t(ka) Q(ij,k,b),  Q(ij,k,b) = Y(ij,k,b) - 0.5 sum_l W(ij,k,l) t(lb),
        Y(ij,k,b) = 0.5 sum_cd g(kbcd) t2(ijcd),  W(ij,kl) = 0.5 sum_cd g(klcd) t2(ijcd):
    the undressed ladder from pair sectors built once (`structure`, a
    LadderSectors, else `_ladder_sectors` with `virtual_irreps`) and o^3 v^3
    corrections. Exact for any T1. The undressed ladder also takes
    t1(ic) t1(jd), adding 0.5 sum_ef g(abef) t1(ie) t1(jf): the part of the
    dressed g(abij) that `dressed_integrals(..., ccsd=True)` leaves out.
    """

    def __init__(self, g, no, *, structure=None, virtual_irreps=None, threads=None):
        n = g.shape[0]
        o, v = slice(0, no), slice(no, n)
        nv = n - no
        self.no, self.nv, self.threads, self.virtual_irreps = no, nv, threads, virtual_irreps
        # (cd, kb) and (cd, kl): the right-hand sides of Y and W.
        self.ovvv = np.ascontiguousarray(g[o, v, v, v].transpose(2, 3, 0, 1)).reshape(nv * nv, no * nv)
        self.oovv = np.ascontiguousarray(g[o, o, v, v].transpose(2, 3, 0, 1)).reshape(nv * nv, no * no)
        if structure is not None:
            self.sectors, self.blocks = (structure, *structure.build(g[v, v, v, v], threads=threads)), None
        else:
            self.sectors, self.blocks = None, _ladder_sectors(g[v, v, v, v], virtual_irreps)

    def __call__(self, t1, t2, *, tau=True):
        """The ladder; `tau=False` leaves out the t1 t1 term (for dressed
        integrals that keep it in g(abij): exactly 0.5 sum_cd g~(abcd) t2(ijcd))."""
        t1 = np.ascontiguousarray(t1, dtype=np.float64)
        ladder = self._ladder(t2 + t1[:, None, :, None] * t1[None, :, None, :] if tau else t2)
        q, _ = self._corrections(t1, t2)
        self._subtract(ladder, t1, q)
        return ladder

    def tangent(self, t1, t2, dt1, dt2, *, tau=True):
        """The derivative of `__call__(t1, t2, tau=tau)` along (dt1, dt2), exact:
        L(g, dtau) - dB - dB(ji,ba), dB = dt1^T Q + t1^T dQ,
        dQ = dY - 0.5 (dW t1 + W dt1). The ladder takes the undressed g(vvvv)
        in the pair sectors of the ground state for a direction of any irrep
        (`doubles_ladder` is exact for any t2); Q and W at (t1, t2) are kept
        for the next direction at the same amplitudes."""
        no, nv, threads = self.no, self.nv, self.threads
        t1 = np.ascontiguousarray(t1, dtype=np.float64)
        dt1 = np.zeros_like(t1) if dt1 is None else np.ascontiguousarray(dt1, dtype=np.float64)
        dt2 = np.zeros((no, no, nv, nv)) if dt2 is None else np.ascontiguousarray(dt2, dtype=np.float64)
        if tau:
            dtau = dt2 + dt1[:, None, :, None] * t1[None, :, None, :]
            dtau += t1[:, None, :, None] * dt1[None, :, None, :]
            ladder = self._ladder(dtau)
            del dtau
        else:
            ladder = self._ladder(dt2)
        point = getattr(self, "_point", None)
        if point is None or not (np.array_equal(point[0], t1) and np.array_equal(point[1], t2)):
            self._point = point = (t1.copy(), np.array(t2, dtype=np.float64), *self._corrections(t1, t2))
        q, w = point[2], point[3]
        dq, dw = self._corrections(t1, dt2)              # dY - 0.5 dW t1, and dW
        matmul(w.reshape(no**3, no), dt1, dq.reshape(no**3, nv), alpha=-0.5, beta=1.0, threads=threads)
        self._subtract(ladder, dt1, q)
        self._subtract(ladder, t1, dq)
        return ladder

    def _ladder(self, t2):
        return doubles_ladder(None, t2, virtual_irreps=self.virtual_irreps, sectors=self.sectors,
                              blocks=self.blocks)

    def _corrections(self, t1, t2):
        """(Q, W) of `t2`: Q(ij, kb) = Y - 0.5 W t1, W(ij, kl), linear in t2."""
        no, nv, threads = self.no, self.nv, self.threads
        rows = np.ascontiguousarray(t2, dtype=np.float64).reshape(no * no, nv * nv)
        q = matmul(rows, self.ovvv, np.empty((no * no, no * nv)), alpha=0.5, threads=threads)
        w = matmul(rows, self.oovv, np.empty((no * no, no * no)), alpha=0.5, threads=threads)
        matmul(w.reshape(no**3, no), t1, q.reshape(no**3, nv), alpha=-0.5, beta=1.0, threads=threads)
        return q, w

    def _subtract(self, ladder, t1, q):
        """ladder -= B + B(ji,ba), B(ij,ab) = sum_k t1(k,a) Q(ij,k,b)."""
        no, nv, threads = self.no, self.nv, self.threads
        # B as (a, ij, b) = t1^T @ Q with Q as (k, ij b).
        columns = np.ascontiguousarray(q.reshape(no * no, no, nv).transpose(1, 0, 2)).reshape(no, no * no * nv)
        b = matmul(np.ascontiguousarray(t1.T), columns, np.empty((nv, no * no * nv)), threads=threads)
        b = b.reshape(nv, no, no, nv).transpose(1, 2, 0, 3)
        ladder -= b
        ladder -= b.transpose(1, 0, 3, 2)


def _ladder_chunk(nv, count, pairs, budget):
    per_slot = 2 * 3 * nv * pairs
    return max(1, min(count, budget // max(1, per_slot)))


def purify(residual, pool):
    """Remove the null fully symmetric virtual component, as TriplesLayout.purify."""
    if not len(residual):
        return
    diag = np.arange(residual.shape[-1])

    def purify_slots(start, stop):
        for s0 in range(start, stop, _SLOT_BATCH):
            tiles = residual[s0:min(stop, s0 + _SLOT_BATCH)]
            symmetric = tiles + tiles.swapaxes(-3, -2)
            symmetric += tiles.swapaxes(-3, -1)
            symmetric += symmetric.swapaxes(-2, -1)
            tiles -= symmetric / 6
            tiles[:, diag, diag, diag] = 0.0

    pool.run(purify_slots, len(residual))


def pair_ladder(residual, t3, wvvvv, pool, *, virtual_irreps=None, budget=2**25):
    """Add sum_de W(ab,de) T(..de..) for the three virtual pairs of every slot.

    W commutes with pair interchange, so each orientation splits into the
    symmetric and antisymmetric pair sectors. Each slot chunk is one GEMM per
    sector over all three orientations; `budget` bounds its buffers (doubles).
    """
    nv = t3.shape[-1]
    (p, q, r, s), even_blocks, odd_blocks = _ladder_sectors(wvvvv, virtual_irreps)
    ne, nodd = len(p), len(r)
    orient = [(0, *(a + 1 for a in perm)) for perm in _ORIENTATIONS]
    back = [(0, *(perm.index(a) + 1 for a in range(3))) for perm in _ORIENTATIONS]
    chunk = _ladder_chunk(nv, len(t3), ne + nodd, budget)
    for c0 in range(0, len(t3), chunk):
        n = min(chunk, len(t3) - c0)
        sym_in, anti_in = np.empty((ne, 3, n, nv)), np.empty((nodd, 3, n, nv))

        def gather(start, stop):
            for s0 in range(start, stop, _SLOT_BATCH):
                s1 = min(stop, s0 + _SLOT_BATCH)
                tiles = t3[c0 + s0:c0 + s1]
                for o in range(3):
                    oriented = tiles.transpose(orient[o])
                    column = sym_in[:, o, s0:s1].transpose(1, 0, 2)
                    np.add(oriented[:, p, q], oriented[:, q, p], out=column)
                    column *= 0.5
                    column = anti_in[:, o, s0:s1].transpose(1, 0, 2)
                    np.subtract(oriented[:, r, s], oriented[:, s, r], out=column)
                    column *= 0.5

        pool.run(gather, n)
        sym_out = _product(even_blocks, sym_in.reshape(ne, -1), np.empty((ne, 3 * n * nv)))
        anti_out = _product(odd_blocks, anti_in.reshape(nodd, -1), np.empty((nodd, 3 * n * nv)))
        del sym_in, anti_in
        sym_out = sym_out.reshape(ne, 3, n, nv)
        anti_out = anti_out.reshape(nodd, 3, n, nv)

        def scatter(start, stop):
            tiles = np.empty((_SLOT_BATCH, nv, nv, nv))
            for s0 in range(start, stop, _SLOT_BATCH):
                s1 = min(stop, s0 + _SLOT_BATCH)
                block = tiles[:s1 - s0]
                target = residual[c0 + s0:c0 + s1]
                for o in range(3):
                    symmetric = sym_out[:, o, s0:s1].transpose(1, 0, 2)
                    antisymmetric = anti_out[:, o, s0:s1].transpose(1, 0, 2)
                    block[:, p, q] = symmetric
                    block[:, q, p] = symmetric
                    block[:, r, s] += antisymmetric
                    block[:, s, r] -= antisymmetric
                    target += block.transpose(back[o])

        pool.run(scatter, n)


def feedback(r1, r2, vooo, vvvo, f, g, layout, t3, o, v, pool):
    """Add all T3 terms to R1/R2 and the triples sources, as `triples_feedback`.

    For each first occupied index m, q is the spin-summed plane T(m,i,j)
    and X[i,j,a,x,y] = q[i,j,x,a,y]. Two GEMMs per plane give every term;
    the next plane is gathered while they run.
    """
    library = _kernels.library()
    if library is not None:
        _feedback_compiled(library, r1, r2, vooo, vvvo, f, g, layout, t3, o, v, pool.workers)
        return
    no, nv = layout.no, layout.nv
    shape = (no, no, nv, nv, nv)
    buffers = [dict(plane=np.empty(shape), q=np.empty(shape), x=np.empty(shape)) for _ in range(2)]

    def spin(buffer):
        plane, q, x_plane = buffer["plane"], buffer["q"], buffer["x"]

        def spin_rows(start, stop):
            for i in range(start, stop):
                source, target = plane[i], q[i]                   # (j, x, y, z)
                np.multiply(source, 2, out=target)
                target -= source.swapaxes(1, 2)
                target -= source.swapaxes(1, 3)
                x_plane[i] = target.transpose(0, 2, 1, 3)

        return spin_rows

    def stage(m):
        buffer = buffers[m % 2]
        return (partial(_gather_plane, pool, layout, t3, buffer["plane"], m, 0),
                partial(pool.run, spin(buffer), no))

    pool.join(pool.stage(*stage(0)))
    for m in range(no):
        handle = pool.stage(*stage(m + 1)) if m + 1 < no else None
        try:
            _feedback_terms(m, buffers[m % 2]["q"], buffers[m % 2]["x"], r1, r2, vooo, vvvo,
                            f, g, o, v)
        finally:
            pool.join(handle)


def _feedback_terms(m, q, x_plane, r1, r2, vooo, vvvo, f, g, o, v):
    """The two GEMMs of plane m: q[i,j,x,y,z] spin-summed T(m,i,j), X[i,j,a,x,y] = q[i,j,x,a,y]."""
    no, nv = q.shape[0], q.shape[-1]
    oovv = g[o, m, v, v]            # (l, d, e); also g(j, b, c) of the R1 term
    g3 = g[v, m, v, v]              # (b, c, d)
    g4 = g[o, m, o, v]              # (k, j, c)
    # X @ [G3 | G5]: R2 sum_cd g(bcd) q(ijdac) and vooo sum_de g(lde) q(ijead).
    # The G5 columns also give R1's 2 q(ijcab) term on their (j, j) diagonal.
    right = np.concatenate([g3.transpose(2, 1, 0).reshape(nv * nv, nv),
                            oovv.transpose(2, 1, 0).reshape(nv * nv, no)], axis=1)
    x_rows = x_plane.reshape(no * no * nv, nv * nv)
    direct = r2.flags.c_contiguous and _engine() is not None and _float64(r2, x_rows)
    if direct:
        # The executor adds the G3 columns straight into R2 (C-ordered rows (i,j,a)).
        matmul(x_rows, right[:, :nv], r2.reshape(no * no * nv, nv), beta=1.0)
        z5 = matmul(x_rows, right[:, nv:], np.empty((no * no * nv, no))).reshape(no, no, nv, no)
    else:
        z = matmul(x_rows, right, np.empty((no * no * nv, nv + no))).reshape(no, no, nv, nv + no)
        r2 += z[..., :nv]
        z5 = z[..., nv:]
    vooo += z5.transpose(2, 3, 0, 1)
    r1 += np.einsum("ijaj->ia", z5)
    subtract_vecmat(r1, oovv.transpose(0, 2, 1).reshape(-1), q.reshape(no, no * nv * nv, nv), 0.5)
    # [G4'; G6] @ q[i]: R2 -sum_kc g(kjc) q(ikcab) + 0.5 sum_c f(mc) q(ijcab),
    # and vvvo -sum_le g(lde) q(jleba).
    left = np.empty((no + nv, no * nv))
    left[:no] = -g4.transpose(1, 0, 2).reshape(no, no * nv)
    for j in range(no):
        left[j, j * nv:(j + 1) * nv] += 0.5 * f[m, v]
    left[no:] = oovv.transpose(1, 0, 2).reshape(nv, no * nv)
    q_rows = q.reshape(no, no * nv, nv * nv)
    if direct:
        matmul(left[:no], q_rows, r2.reshape(no, no, nv * nv), beta=1.0)
        y6 = matmul(left[no:], q_rows, np.empty((no, nv, nv * nv)))
    else:
        y = matmul(left, q_rows, np.empty((no, no + nv, nv * nv)))
        r2 += y[:, :no].reshape(no, no, nv, nv)
        y6 = y[:, no:]
    subtract_reversed(vvvo, y6.reshape(no, nv, nv, nv))


def dressed_integrals(g0, t1, threads=None, *, ccsd=False, open=False, contracted=None, out=None):
    """A C-ordered copy of the integrals g0 dressed with T1 along all four axes
    (`solver._dress_axis` in turn: axes 0 and 1 add -t1.T times the occupied
    rows to the virtual ones, axes 2 and 3 t1 times the virtual rows to the
    occupied ones), copied and dressed on all threads; None without the
    compiled kernels. NumPy's copy and tensordot moved the whole tensor several
    times on one thread (about 1.4 s per evaluation at 88 virtuals) for 9 GF of
    arithmetic. With `ccsd` only what the CCSD residual reads, never reading
    g(vvvv) (`itercc_dress_integrals_ccsd`): in the planes g(a, b, ., .) of two
    virtual indices only the occupied-occupied block, and that without
    sum_ef t1(i,e) t1(j,f) g(abef), which `BareLadder` adds; the rest of those
    planes is left unset. With `open` every entry with an occupied index,
    exactly as the full dressing (`itercc_dress_integrals_open`); g~(vvvv) is
    left unset, for a caller that takes the ladder sectors from
    `LadderSectors.dressed`; with `contracted` (`LadderSectors.contract_t1`)
    g(vvvv) is not read either. `out` (a C-ordered float64 n^4 array) receives
    the fused dressing when given."""
    library = _kernels.library()
    if (library is None or not all(isinstance(x, np.ndarray) and x.dtype == np.float64 for x in (g0, t1))
            or g0.ndim != 4 or t1.ndim != 2):
        return None
    from . import _cpu_quadruples

    no, nv = t1.shape
    n = g0.shape[0]
    if g0.shape != (n,) * 4 or n != no + nv:
        return None
    if ccsd and any(s % 8 for s in g0.strides):
        raise ValueError("the CCSD dressing needs the compiled kernel and a float64 g0")
    if open and (ccsd or any(s % 8 for s in g0.strides)):
        raise ValueError("the open dressing needs the compiled kernel and a float64 g0")
    threads = cpu_threads(threads)
    up = np.ascontiguousarray(t1, dtype=np.float64)            # (no, nv)
    if all(stride % 8 == 0 for stride in g0.strides):
        # Fused: two passes over the tensor instead of a copy and four (the
        # dressing is memory bound; the four axis maps commute).
        g = (out if out is not None and out.shape == (n,) * 4 and out.dtype == np.float64
             and out.flags.c_contiguous and out is not g0 else np.empty((n,) * 4))
        strides = np.asarray([stride // 8 for stride in g0.strides], dtype=np.int64)
        if open:
            if contracted is not None and (contracted.shape != (nv, nv, nv, no) or contracted.dtype != np.float64
                                           or not contracted.flags.c_contiguous):
                raise ValueError("contracted must be the C-ordered (nv, nv, nv, no) products")
            library.dress_integrals_open(g.ctypes.data, g0.ctypes.data, strides.ctypes.data, n, no, up.ctypes.data,
                                         None if contracted is None else contracted.ctypes.data, threads)
            return g
        kernel = library.dress_integrals_ccsd if ccsd else library.dress_integrals
        kernel(g.ctypes.data, g0.ctypes.data, strides.ctypes.data, n, no, up.ctypes.data, threads)
        return g
    g = _cpu_quadruples.copy(g0)
    down = np.ascontiguousarray(-t1.T, dtype=np.float64)       # (nv, no)
    for axis in (0, 1):
        library.dress_axis(g.ctypes.data, n, axis, no, nv, 0, no, down.ctypes.data, threads)
    for axis in (2, 3):
        library.dress_axis(g.ctypes.data, n, axis, 0, no, no, nv, up.ctypes.data, threads)
    return g


def subtract_vecmat(out, w, q, alpha):
    """out -= alpha * (w @ q[i] for each matrix q[i]) on all threads when
    compiled (NumPy's matmul of a vector with a stack runs on one thread: about
    1 s per evaluation at 88 virtuals for the feedback's R1 term)."""
    library = _kernels.library()
    if (library is not None and out.flags.c_contiguous and q.flags.c_contiguous
            and all(x.dtype == np.float64 for x in (out, w, q)) and q.ndim == 3
            and out.shape == (q.shape[0], q.shape[2]) and w.shape == (q.shape[1],)):
        w = np.ascontiguousarray(w)
        library.stacked_vecmat(q.shape[0], q.shape[1], q.shape[2], w.ctypes.data, q.ctypes.data,
                               -float(alpha), out.ctypes.data, cpu_threads(None))
    else:
        out -= alpha * np.matmul(w, q)


def subtract_reversed(vvvo, y):
    """vvvo -= y.transpose(3, 2, 1, 0): on all threads when compiled (a
    plane's y is tens of MB, and a single thread misses the cache on nearly
    every element of the reversed order)."""
    from . import _cpu_quadruples

    view = y.transpose(3, 2, 1, 0)
    if vvvo.flags.c_contiguous and _cpu_quadruples.available():
        _cpu_quadruples.permuted_add(vvvo, vvvo, view, 1.0, -1.0)
    else:
        vvvo -= view


# Compiled path ------------------------------------------------------------


def _tables(layout):
    """Per-plane kernel tables, cached on the layout.

    For plane k (third index) and plane m (first index): the packed slot of
    each (x, y) tile and its source strides. For the scatter of plane k: the
    R3 slots it reaches, each with its (seed tile, q0, q1, q2) contributions.
    """
    cached = getattr(layout, "_cpu_kernel_tables", None)
    if cached is not None:
        return cached
    no, nv = layout.no, layout.nv
    unit = np.array([nv * nv, nv, 1], dtype=np.int64)
    slots = layout.slots.astype(np.int64)
    strides = unit[layout.from_sorted]                     # (no, no, no, 3)
    third = [(np.ascontiguousarray(slots[:, :, k]).reshape(-1),
              np.ascontiguousarray(strides[:, :, k]).reshape(-1)) for k in range(no)]
    first = [(np.ascontiguousarray(slots[m]).reshape(-1),
              np.ascontiguousarray(strides[m]).reshape(-1)) for m in range(no)]
    scatter = []
    for k in range(no):
        contributions = {}
        for i in range(no):
            for j in range(no):
                slot = int(layout.slots[i, j, k])
                if slot < 0:
                    continue
                # R[slot] += S[i,j].transpose(to_sorted): R[u] += S[sum_m u_m q_m]
                q = unit[layout.to_sorted[i, j, k]]
                found = contributions.setdefault(slot, [])
                found.append((i * no + j, *q))
                a, b, c = layout.indices[slot]
                if a == b:                  # second virtual permutation
                    found.append((i * no + j, q[1], q[0], q[2]))
                elif b == c:
                    found.append((i * no + j, q[0], q[2], q[1]))
        targets = np.array(sorted(contributions), dtype=np.int64)
        lists = [contributions[t] for t in targets]
        start = np.cumsum([0] + [len(x) for x in lists]).astype(np.int64)
        flat = np.array([e for x in lists for e in x], dtype=np.int64).reshape(-1)
        scatter.append((targets, start, flat))
    layout._cpu_kernel_tables = (third, first, scatter)
    return layout._cpu_kernel_tables


def _triples_compiled(lib, layout, t2, t3, have_t3, im, vooo, vvvo, g, o, v, threads,
                      ph_irreps, virtual_irreps, ladder_budget, wvvvv=None):
    """`triples` with the element-wise passes in the OpenMP kernels."""
    ptr = _kernels.pointer
    no, nv = layout.no, layout.nv
    plane_shape = (no, no, nv, nv, nv)
    residual = np.zeros(layout.shape)
    rows = _Rows(ph_irreps, no, nv)
    position = (np.arange(no * nv, dtype=np.int64) if rows.sectors is None
                else rows.sectors.inverse.astype(np.int64))
    third, _, scatter = _tables(layout)
    vvvo_rows, vooo_rows = _seed_sources(vooo, vvvo, no, nv)
    a_seed, b_seed = np.empty((no, nv, nv, no, nv)), np.empty(plane_shape)
    seeds = np.empty(plane_shape)       # holds the occupied ladder before the seed pass
    direct_out = crossed_out = seeds    # placeholders when there is no T3
    if have_t3:
        t3 = np.ascontiguousarray(t3, dtype=np.float64)
        occupied, direct_blocks, crossed_blocks = _plane_matrices(im, rows, no, nv)
        rhs_shape = (no * nv, no, nv, nv)
        plane = np.empty(plane_shape)
        crossed_rhs, direct_rhs = np.empty(rhs_shape), np.empty(rhs_shape)
        direct_out, crossed_out = np.empty(rhs_shape), np.empty(rhs_shape)
    for k in range(no):
        _t2_seeds(vvvo_rows, vooo_rows, t2, k, a_seed, b_seed, threads)
        if have_t3:
            slot, stride = third[k]
            lib.triples_plane(ptr(t3), no, nv, ptr(slot), ptr(stride), ptr(position),
                              ptr(plane), ptr(crossed_rhs), ptr(direct_rhs), threads)
            matmul(occupied, plane.reshape(no * no, nv**3), seeds.reshape(no * no, nv**3),
                   threads=threads)
            _product(direct_blocks, direct_rhs.reshape(no * nv, -1), direct_out.reshape(no * nv, -1),
                     threads)
            _product(crossed_blocks, crossed_rhs.reshape(no * nv, -1),
                     crossed_out.reshape(no * nv, -1), threads)
        lib.triples_seeds(no, nv, ptr(position), int(have_t3), ptr(a_seed), ptr(b_seed),
                          ptr(direct_out), ptr(crossed_out), ptr(seeds), threads)
        targets, start, contributions = scatter[k]
        lib.triples_scatter(nv, len(targets), ptr(targets), ptr(start), ptr(contributions),
                            ptr(seeds), ptr(residual), threads)
    del a_seed, b_seed, seeds, direct_out, crossed_out
    if have_t3:
        del plane, crossed_rhs, direct_rhs
        if wvvvv is None:
            wvvvv = _ladder_matrix(g, o, v, t2, im, nv, threads)
        _ladder_compiled(lib, residual, t3, wvvvv, threads, virtual_irreps, ladder_budget)
    if len(residual):
        lib.purify(ptr(residual), len(residual), nv, threads)
    return residual


def _ladder_compiled(lib, residual, t3, wvvvv, threads, virtual_irreps, budget):
    """`pair_ladder` with the gather and scatter in the OpenMP kernels."""
    ptr = _kernels.pointer
    nv = t3.shape[-1]
    (p, q, r, s), even_blocks, odd_blocks = _ladder_sectors(wvvvv, virtual_irreps)
    p, q, r, s = (np.ascontiguousarray(x, dtype=np.int64) for x in (p, q, r, s))
    ne, nodd = len(p), len(r)
    even_row = np.zeros((nv, nv), dtype=np.int64)
    even_row[p, q] = even_row[q, p] = np.arange(ne)
    odd_row = np.zeros((nv, nv), dtype=np.int64)
    odd_row[r, s] = odd_row[s, r] = np.arange(nodd)
    odd_sign = np.zeros((nv, nv))
    odd_sign[r, s], odd_sign[s, r] = 1.0, -1.0
    unit = (nv * nv, nv, 1)
    orient = np.array([[unit[a] for a in perm] for perm in _ORIENTATIONS], dtype=np.int64)
    perms = np.array(_ORIENTATIONS, dtype=np.int64)
    chunk = _ladder_chunk(nv, len(t3), ne + nodd, budget)
    for c0 in range(0, len(t3), chunk):
        n = min(chunk, len(t3) - c0)
        sym_in, anti_in = np.empty((ne, 3, n, nv)), np.empty((nodd, 3, n, nv))
        lib.ladder_gather(ptr(t3[c0:c0 + n]), n, nv, ne, ptr(p), ptr(q), nodd, ptr(r), ptr(s),
                          ptr(orient), ptr(sym_in), ptr(anti_in), threads)
        sym_out = _product(even_blocks, sym_in.reshape(ne, -1), np.empty((ne, 3 * n * nv)), threads)
        anti_out = _product(odd_blocks, anti_in.reshape(nodd, -1), np.empty((nodd, 3 * n * nv)),
                            threads)
        del sym_in, anti_in
        lib.ladder_scatter(n, nv, ptr(even_row), ptr(odd_row), ptr(odd_sign), ptr(perms),
                           ptr(sym_out), ptr(anti_out), ptr(residual[c0:c0 + n]), threads)


def _feedback_compiled(lib, r1, r2, vooo, vvvo, f, g, layout, t3, o, v, threads):
    """`feedback` with the spin-summed planes built by an OpenMP kernel."""
    ptr = _kernels.pointer
    no, nv = layout.no, layout.nv
    _, first, _ = _tables(layout)
    t3 = np.ascontiguousarray(t3, dtype=np.float64)
    q, x_plane = np.empty((no, no, nv, nv, nv)), np.empty((no, no, nv, nv, nv))
    for m in range(no):
        slot, stride = first[m]
        lib.feedback_plane(ptr(t3), no, nv, ptr(slot), ptr(stride), ptr(q), ptr(x_plane), threads)
        _feedback_terms(m, q, x_plane, r1, r2, vooo, vvvo, f, g, o, v)
