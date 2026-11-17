"""Full iterative closed-shell CCSDT with occupied-triangular spatial T3."""

from contextlib import nullcontext
import ctypes
from functools import partial
from numbers import Integral
import os
import warnings

import numpy as np

from . import _blocked_residual as _blocked
from . import _cpu_triples as _cpu
from . import _restricted_equations as eq
from ._pair_symmetry import PairSectors, pair_product
from ._triples_layout import NoTriples, TriplesLayout
from .hamiltonian import _real64
from .solver import Iteration, Result, _DIIS, _dress_axis
from .spatial_symmetry import orbital_labels


def _fock(h, g, o, xp=np):
    return h + 2 * xp.einsum("piqi->pq", g[:, o, :, o]) - xp.einsum("piiq->pq", g[:, o, o, :])


def _near_zero_triples_denominator(occupied_sums, virtual_sums):
    if not occupied_sums:
        return False
    # Canonical orbitals usually give disjoint ranges. Overlapping spectra
    # still need the exact per-denominator check.
    lower, upper = virtual_sums.min(), virtual_sums.max()
    if lower - max(occupied_sums) >= 1e-12 or min(occupied_sums) - upper >= 1e-12:
        return False
    return any(np.min(np.abs(energy - virtual_sums)) < 1e-12 for energy in occupied_sums)


def _direct_ladder(residual, t3, wvvvv, batch, *, xp=np):
    """Three pair orientations per canonical occupied triple, in BLAS batches."""
    nv = t3.shape[-1]
    matrix = xp.ascontiguousarray(wvvvv).reshape(nv * nv, nv * nv)
    for start in range(0, len(t3), batch):
        source = t3[start : start + batch]
        for a, b, c in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
            rhs = source.transpose(a + 1, b + 1, 0, c + 1).reshape(nv * nv, -1)
            tile = (matrix @ rhs).reshape(nv, nv, len(source), nv).transpose(2, 0, 1, 3)
            axes = tuple((a, b, c).index(axis) + 1 for axis in range(3))
            residual[start : start + batch] += tile.transpose(0, *axes)


def _pair_ladder(residual, t3, wvvvv, batch, *, xp=np, virtual_irreps=None, fuse_cuda=False,
                 supported_occupied_irreps=None, axial=None, axial_tolerance=0.0):
    """Apply the virtual ladder in the symmetric and antisymmetric pair sectors.

    W(ab,de) commutes with pair interchange (ab -> ba, de -> ed).
    The two half-size GEMMs replace one full pair-space GEMM per orientation.
    Supplied occupied labels require a prior exact support check of T3.
    """
    nv = t3.shape[-1]
    wvvvv = xp.asarray(wvvvv)
    if axial is not None and supported_occupied_irreps is not None:
        if axial.apply(residual, t3, wvvvv, supported_occupied_irreps, batch, tolerance=axial_tolerance):
            return
    even_sectors = odd_sectors = packing = None
    if virtual_irreps is None and not fuse_cuda:
        p, q = xp.triu_indices(nv)
        r, s = xp.triu_indices(nv, 1)
    else:
        p, q = np.triu_indices(nv)
        r, s = np.triu_indices(nv, 1)
        if virtual_irreps is not None:
            even_sectors = PairSectors(virtual_irreps[p] ^ virtual_irreps[q])
            odd_sectors = PairSectors(virtual_irreps[r] ^ virtual_irreps[s])
            p, q = p[even_sectors.order], q[even_sectors.order]
            r, s = r[odd_sectors.order], s[odd_sectors.order]
        if fuse_cuda:
            from ._cuda_pairs import CudaPairPacking

            packing = CudaPairPacking(nv, (p, q), (r, s), xp)
        else:
            p, q, r, s = map(xp.asarray, (p, q, r, s))
    if packing is not None:
        even, odd = packing.matrices(wvvvv)
    else:
        even = wvvvv[p[:, None], q[:, None], p, q]
        even += wvvvv[p[:, None], q[:, None], q, p] * (p != q)
        odd = wvvvv[r[:, None], s[:, None], r, s]
        odd -= wvvvv[r[:, None], s[:, None], s, r]
    del wvvvv
    even_product, odd_product = even.__matmul__, odd.__matmul__
    compact = None
    if supported_occupied_irreps is not None:
        if even_sectors is None or packing is None:
            raise ValueError("Compact ladder products require fused symmetry blocks")
        even_blocks = even_sectors._blocks(even, xp, True, None)
        odd_blocks = odd_sectors._blocks(odd, xp, True, None)
        if even_blocks is not None and odd_blocks is not None:
            from ._cuda_ladder_columns import CudaLadderColumns

            compact = CudaLadderColumns(packing, virtual_irreps, even_sectors, odd_sectors,
                                        even_blocks, odd_blocks)
            del even_product, odd_product
    if even_sectors is not None and compact is None:
        # The pair packing and scattering already use grouped indices.
        even_product, _ = even_sectors.product(even, xp=xp, grouped=True)
        odd_product, _ = odd_sectors.product(odd, xp=xp, grouped=True)
    del even, odd
    resident = isinstance(residual, xp.ndarray)
    for start in range(0, len(t3), batch):
        source = xp.asarray(t3[start : start + batch])
        target = residual[start : start + batch] if resident else xp.zeros_like(source)
        if compact is not None:
            compact.add(source, target, supported_occupied_irreps[start:start + batch])
        elif packing is not None:
            packing.add(source, target, even_product, odd_product)
        else:
            columns = len(source) * nv
            tile = xp.empty_like(source)
            for perm in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
                oriented = source.transpose(0, *(a + 1 for a in perm))
                rhs = 0.5 * (oriented[:, p, q] + oriented[:, q, p])
                rhs = rhs.transpose(1, 0, 2).reshape(len(p), columns)
                symmetric = even_product(rhs).reshape(len(p), len(source), nv).transpose(1, 0, 2)
                tile[:, p, q] = tile[:, q, p] = symmetric
                rhs = 0.5 * (oriented[:, r, s] - oriented[:, s, r])
                rhs = rhs.transpose(1, 0, 2).reshape(len(r), columns)
                antisymmetric = odd_product(rhs).reshape(len(r), len(source), nv).transpose(1, 0, 2)
                tile[:, r, s] += antisymmetric
                tile[:, s, r] -= antisymmetric
                axes = tuple(perm.index(axis) + 1 for axis in range(3))
                target += tile.transpose(0, *axes)
        if not resident:
            residual[start : start + batch] += target.get()


class _BlockDIIS(_DIIS):
    """Store rank arrays directly. With `compiled` (the CPU core is present),
    large arrays are added and mixed, and arrays of every size dotted, on all
    threads with the kernels of the CCSDTQ host DIIS
    (`_cpu_quadruples.permuted_add`, `dots`, `combine`; fixed blocks, so the
    results do not depend on the thread count), else in bounded NumPy chunks.
    The new error is read once per update for all its dots.
    """

    LARGE = 1 << 16

    def __init__(self, size, compiled=False, threads=None):
        super().__init__(size, np)
        self.compiled, self.threads = bool(compiled), threads

    def _threaded(self, *arrays):
        return (self.compiled and arrays[0].size >= self.LARGE
                and all(a.flags.c_contiguous and a.dtype == np.float64 for a in arrays))

    def _compiled_dots(self, *arrays):
        # Dots of every size: OpenBLAS ddot rounds by its thread count.
        return self.compiled and all(a.flags.c_contiguous and a.dtype == np.float64 for a in arrays)

    def _dot(self, left, right):
        return self._dots(right, [left])[0]

    def _dots(self, error, errors):
        """[e . error for e in errors] + [error . error], each rank of `error`
        read once (the compiled `dots` kernel: fixed blocks, so the same bits
        for any thread count)."""
        from . import _cpu_quadruples

        total = np.zeros(len(errors) + 1)
        for rank, a in enumerate(error):
            others = [e[rank] for e in errors]
            if self._compiled_dots(a, *others):
                with _cpu.thread_scope(self.threads):
                    total += _cpu_quadruples.dots(a.reshape(-1), [b.reshape(-1) for b in others])
            else:
                flat = a.ravel()
                total += [np.dot(b.ravel(), flat) for b in others] + [np.dot(flat, flat)]
        return total

    def _add(self, a, b):
        if not self._threaded(a, b):
            return a + b
        from . import _cpu_quadruples

        with _cpu.thread_scope(self.threads):
            return _cpu_quadruples.permuted_add(np.empty_like(a), a, b, 1.0, 1.0)

    def take_step(self, amplitudes, steps):
        return self.update(tuple(self._add(a, step) for a, step in zip(amplitudes, steps)), steps)

    def update(self, amplitudes, steps):
        if self.size < 2:
            return amplitudes
        weights = self._append(amplitudes, steps, self._dot, self._dots)
        if weights is None:
            return amplitudes
        result = []
        for rank, amplitude in enumerate(amplitudes):
            stored = [entry[0][rank] for entry in self.history]
            if self._threaded(*stored):
                from . import _cpu_quadruples

                with _cpu.thread_scope(self.threads):
                    output = _cpu_quadruples.combine(weights, [a.reshape(-1) for a in stored],
                                                     np.empty(amplitude.size))
                result.append(output.reshape(amplitude.shape))
                continue
            vectors = [entry[0][rank].ravel() for entry in self.history]
            output = np.empty(amplitude.size)
            chunk = 1 << 20
            buffer = np.empty(min(chunk, output.size))
            for start in range(0, output.size, chunk):
                end = min(start + chunk, output.size)
                part = output[start:end]
                np.multiply(vectors[0][start:end], weights[0], out=part)
                scratch = buffer[:end - start]
                for weight, vector in zip(weights[1:], vectors[1:]):
                    np.multiply(vector[start:end], weight, out=scratch)
                    part += scratch
            result.append(output.reshape(amplitude.shape))
        return tuple(result)


class _SymmetricBlockDIIS(_BlockDIIS):
    """Store compact triples history with a dense fallback on exact leakage."""

    def __init__(self, size, columns, compiled=False, threads=None):
        super().__init__(size, compiled, threads)
        self.columns = columns

    def take_step(self, amplitudes, steps):
        if self.size < 2 or self.columns is None:
            return super().take_step(amplitudes, steps)
        packed = self.columns.pack(amplitudes[-1])
        packed_steps = self.columns.pack(steps[-1]) if packed is not None else None
        if packed_steps is None:
            for i, (vector, error) in enumerate(self.history):
                self.history[i] = ((*vector[:2], self.columns.unpack(vector[-1])),
                                   (*error[:2], self.columns.unpack(error[-1])))
                del vector, error
            self.columns = None
            return super().take_step(amplitudes, steps)
        # Form the proposed triples in the compact representation. The only
        # dense output needed is the result passed to the next residual.
        packed += packed_steps
        compact = (amplitudes[0] + steps[0], amplitudes[1] + steps[1], packed)
        mixed = self.update(compact, (*steps[:2], packed_steps))
        return (*mixed[:2], self.columns.unpack(mixed[-1]))


class RestrictedCC:
    """Spin-restricted full CCSDT for real orthonormal spatial MO integrals.

    h is one-electron, eri is in chemists' notation (pq|rs), and the constant
    includes nuclear repulsion and any frozen-core contribution. Orbitals are
    occupied first. The caller keeps the input integrals fixed during use.

    Amplitudes and residuals are spatial T1[i,a], T2[i,j,a,b] and packed
    T3[slot,a,b,c]. Slots are indexed by triples_indices in lexicographic
    i <= j <= k order. The identically zero i == j == k slots are omitted.
    `rank=2` gives restricted CCSD: T3 has no slots (shape (0, nv, nv, nv)).
    """

    def __init__(self, h, eri, nocc, constant=0.0, *, ladder_batch=4, pair_ladder=True,
                 ladder_backend="cpu", residual_backend="cpu", orbital_irreps=None,
                 cuda_pair_fusion=False, cuda_layout_fusion=False,
                 cuda_ph_columns=False, cuda_ladder_columns=False,
                 cuda_occupied_columns=False, cuda_axial=None, axial_projection_tolerance=0.0,
                 cuda_jacobi=False, cuda_t1_dressing=False, host_diis_columns=False,
                 cuda_crossed_fusion=False, cuda_feedback_columns=False, cuda_assembly_columns=False,
                 cuda_ph_fusion=False, cuda_pool_cache=False, cuda_ph_batch=False,
                 cpu_threads=None, rank=3, symmetry_tolerance=0.0):
        if rank not in (2, 3):
            raise ValueError("rank must be 2 (CCSD) or 3 (CCSDT)")
        h, eri = _real64(h), _real64(eri)
        if h.ndim != 2 or h.shape[0] != h.shape[1] or eri.shape != (len(h),) * 4:
            raise ValueError("Incompatible spatial h and ERI shapes")
        if not isinstance(nocc, Integral) or not 0 < nocc < len(h):
            raise ValueError("Expected nonempty occupied and virtual spatial spaces")
        if not isinstance(ladder_batch, Integral) or ladder_batch < 1:
            raise ValueError("ladder_batch must be a positive integer")
        if not (np.isfinite(h).all() and np.isfinite(eri).all() and np.isfinite(constant)):
            raise ValueError("Hamiltonian contains nonfinite data")
        if not np.isfinite(symmetry_tolerance) or symmetry_tolerance < 0:
            raise ValueError("Expected a finite nonnegative symmetry tolerance")
        # With orbital labels and `symmetry_tolerance` > 0 (for example 2e-9),
        # integrals whose symmetry-forbidden entries stay within it (integral
        # rounding) are projected: the symmetry-blocked CPU path needs exact
        # zeros, and rounding-level leakage keeps every solve on the dense
        # path. The default 0 keeps the original integrals, as the raw-
        # Hamiltonian audits and refinements of the field drivers require
        # (they project explicitly for their main solve).
        self.symmetry_tolerance = float(symmetry_tolerance)
        self.integral_symmetry_leakage, self.symmetry_projected = None, False
        if orbital_irreps is not None and symmetry_tolerance > 0:
            from .spatial_symmetry import project_integrals

            try:
                h, eri, audited = project_integrals(h, eri, orbital_irreps, tolerance=symmetry_tolerance)
            except ValueError:
                pass
            else:
                self.integral_symmetry_leakage = tuple(item.maximum for item in audited)
                self.symmetry_projected = True
        self.h, self.g = h, eri.transpose(0, 2, 1, 3)
        self.o, self.v = slice(nocc), slice(nocc, None)
        self.ladder_batch = ladder_batch
        if ladder_backend not in ("cpu", "cuda") or residual_backend not in ("cpu", "cuda"):
            raise ValueError("Choose 'cpu' or 'cuda' for each backend")
        self._xp = np
        if cuda_pool_cache and residual_backend != "cuda":
            raise ValueError("Device allocation caching requires CUDA residuals")
        self._cuda_pool_cache = cuda_pool_cache
        if residual_backend == "cuda" or ladder_backend == "cuda":
            import cupy

        if residual_backend == "cuda":
            self._xp = cupy
            self._pool = cupy.cuda.MemoryPool()
        if cuda_jacobi and residual_backend != "cuda":
            raise ValueError("Device Jacobi requires CUDA residuals")
        self._cuda_jacobi = cuda_jacobi
        self.rank = rank
        if cuda_layout_fusion:
            if residual_backend != "cuda" or rank == 2:
                raise ValueError("Fused triples layout requires CUDA residuals and triples")
            from ._cuda_layout import CudaTriplesLayout

            self.layout = CudaTriplesLayout(nocc, len(h) - nocc, cupy)
        elif rank == 2:
            self.layout = NoTriples(nocc, len(h) - nocc)
        else:
            self.layout = TriplesLayout(nocc, len(h) - nocc)
        if cuda_crossed_fusion and not cuda_layout_fusion:
            raise ValueError("The fused crossed update requires the CUDA triples layout")
        self._crossed_update = (self.layout.subtract_crossed_fused if cuda_crossed_fusion
                                else self.layout.subtract_crossed)
        self.triples_indices = self.layout.indices
        if residual_backend == "cpu" and ladder_backend == "cuda" and not pair_ladder:
            raise ValueError("A CPU residual requires the packed pair ladder for CUDA offload")
        ladder = _pair_ladder if pair_ladder else _direct_ladder
        ladder_xp = cupy if ladder_backend == "cuda" or residual_backend == "cuda" else np
        options = dict(xp=ladder_xp)
        if cuda_pair_fusion:
            if not pair_ladder or ladder_xp is np:
                raise ValueError("Fused pair packing requires the CUDA pair ladder")
            options["fuse_cuda"] = True
        self._ph_irreps = self._ph_columns = self._column_support = None
        self._cuda_ladder_columns = cuda_ladder_columns
        self._occupied_columns = None
        self._fold_occupied = cuda_occupied_columns and cuda_layout_fusion
        if orbital_irreps is not None:
            if not pair_ladder:
                raise ValueError("Symmetry blocks require the packed pair ladder")
            labels = orbital_labels(orbital_irreps, len(h)).copy()
            options["virtual_irreps"] = labels[nocc:].copy()
            self._ph_irreps = (labels[:nocc, None] ^ labels[nocc:]).ravel()
        if (cuda_ph_columns or cuda_ladder_columns or cuda_occupied_columns
                or cuda_feedback_columns or cuda_assembly_columns):
            if self._xp is np or orbital_irreps is None:
                raise ValueError("Compressed columns require CUDA residuals and orbital labels")
            if cuda_ladder_columns and not cuda_pair_fusion:
                raise ValueError("Compressed ladder columns require fused CUDA pairs")
            from ._cuda_support import CudaTriplesSupport

            self._column_support = CudaTriplesSupport(labels[:nocc], labels[nocc:], self.layout, cupy)
        self._feedback = None
        if cuda_feedback_columns:
            from ._cuda_feedback import CudaFeedback

            self._feedback = CudaFeedback(labels[:nocc], labels[nocc:], cupy)
        self._assembly = None
        if cuda_assembly_columns:
            from ._cuda_assembly import CudaAssembly

            self._assembly = CudaAssembly(labels[:nocc], labels[nocc:], cupy)
        if cuda_ph_columns:
            from ._cuda_ph import CudaParticleHole

            self._ph_columns = CudaParticleHole(labels[:nocc], labels[nocc:], cupy)
        self._ph_fusion = None
        if cuda_ph_batch and not cuda_ph_fusion:
            raise ValueError("Batched particle-hole products require particle-hole fusion")
        self._ph_batch = cuda_ph_batch
        if cuda_ph_fusion:
            if not cuda_ph_columns:
                raise ValueError("Particle-hole fusion requires CUDA column products")
            from ._cuda_ph_fused import CudaParticleHoleFusion

            fusion = CudaParticleHoleFusion
            if cuda_ph_batch:
                from ._cuda_ph_batch import CudaParticleHoleBatch

                fusion = CudaParticleHoleBatch
            self._ph_fusion = fusion(labels[:nocc], labels[nocc:], self._ph_columns, cupy)
        if cuda_occupied_columns:
            from ._cuda_occupied import CudaOccupiedLadder

            virtual = labels[nocc:]
            triples = (virtual[:, None, None] ^ virtual[None, :, None] ^ virtual).ravel()
            self._occupied_columns = CudaOccupiedLadder(labels[:nocc], triples, cupy)
        if cuda_axial is not None:
            if not cuda_ladder_columns:
                raise ValueError("Axial blocking requires the exact CUDA ladder support check")
            if np.any(labels > 3):
                raise ValueError("Axial blocking requires C2v orbital labels 0..3")
            if not np.isfinite(axial_projection_tolerance) or axial_projection_tolerance < 0:
                raise ValueError("Choose a finite nonnegative axial projection tolerance")
            from ._cuda_axial import CudaAxialLadder

            momentum, partners = map(np.asarray, cuda_axial)
            if momentum.shape != (self.layout.nv,) or partners.shape != momentum.shape:
                raise ValueError("Expected an angular label and partner for each virtual orbital")
            options.update(axial=CudaAxialLadder(momentum, labels[nocc:], partners, cupy),
                           axial_tolerance=axial_projection_tolerance)
        self._t1_dressing = None
        if cuda_t1_dressing:
            if self._xp is np or orbital_irreps is None:
                raise ValueError("Sparse T1 dressing requires CUDA residuals and orbital labels")
            from ._cuda_dress import CudaT1Dressing

            self._t1_dressing = CudaT1Dressing(labels[:nocc], labels[nocc:], cupy)
        self._diis_columns = None
        if host_diis_columns:
            if orbital_irreps is None:
                raise ValueError("Compact host DIIS requires orbital labels")
            from ._diis_columns import host_columns

            self._diis_columns = host_columns(labels[:nocc], labels[nocc:], self.layout, cpu_threads)
        self._ladder = partial(ladder, **options)
        # CPU residuals run on the compiled kernels (_cpu_triples):
        # `cpu_threads` (else ITERCC_CPU_THREADS, else one per core) is the thread
        # count of every kernel of an evaluation (`_cpu.thread_scope`).
        if cpu_threads is not None and (not isinstance(cpu_threads, Integral) or cpu_threads < 1):
            raise ValueError("cpu_threads must be a positive integer")
        self._cpu_threads = cpu_threads
        # A CUDA ladder (ladder_backend="cuda") needs the generic assembly.
        self._cpu_fast = pair_ladder and ladder_xp is np
        self._virtual_irreps = options.get("virtual_irreps")
        # With orbital labels, CPU residuals run on symmetry-blocked T3
        # (_blocked_residual) whenever the amplitudes are exactly symmetric.
        self._labels = labels if orbital_irreps is not None else None
        self._blocked_plan = None
        self._integrals_symmetric = self._ladder_structure = self._bare_ladder = None
        self._dressing_buffer, self._hold_dressing = None, False
        self._cpu_blocked = os.environ.get("ITERCC_BLOCKED", "1") != "0"
        self._contract = partial(eq.contract, xp=self._xp)
        if self._xp is np and _cpu._kernels.library() is not None:
            # CCSD and CCSDT products on the GEMM executor of the CPU CCSDTQ solver.
            self._contract = _cpu.CompiledContract(cpu_threads)
        self.device_pool_reserved_bytes = 0
        self.fock = _fock(h, self.g, self.o)
        self.reference_energy = float(constant + np.trace((h + self.fock)[self.o, self.o]))
        self.shapes = ((nocc, self.layout.nv), (nocc, nocc, self.layout.nv, self.layout.nv),
                       self.layout.shape)

    def _validated(self, amplitudes, *, triples=True):
        arrays = tuple(map(_real64, amplitudes))
        shapes = self.shapes if triples else self.shapes[:2]
        if tuple(a.shape for a in arrays) != shapes:
            raise ValueError(f"Expected amplitude shapes {shapes}")
        t1, t2 = arrays[:2]
        if not (np.isfinite(t1).all() and np.isfinite(t2).all()):
            raise ValueError("Expected finite amplitudes")
        if not np.allclose(t2, t2.transpose(1, 0, 3, 2), atol=1e-12, rtol=1e-9):
            raise ValueError("T2 must be symmetric under paired occupied/virtual permutation")
        if not triples:
            return arrays
        t3 = arrays[2]
        if not all(np.isfinite(tile).all() for tile in t3):
            raise ValueError("Expected finite amplitudes")
        for (i, j, k), tile in zip(self.triples_indices, t3):
            if i == j and not np.allclose(tile, tile.swapaxes(0, 1), atol=1e-12, rtol=1e-9):
                raise ValueError("T3 repeated occupied slots must have matching virtual symmetry")
            if j == k and not np.allclose(tile, tile.swapaxes(1, 2), atol=1e-12, rtol=1e-9):
                raise ValueError("T3 repeated occupied slots must have matching virtual symmetry")
        return arrays

    def _dressed(self, t1, ccsd=False):
        """(f, g) dressed with T1; `ccsd`: only what the CCSD residual reads
        (`_cpu.dressed_integrals`), for `_bare_ladder_for`."""
        o, v = self.o, self.v
        xp = self._xp
        h = xp.array(self.h, copy=True)
        _dress_axis(h, 0, v, o, -t1.T, xp)
        _dress_axis(h, 1, o, v, t1, xp)
        if xp is np and self._t1_dressing is None:
            g = _cpu.dressed_integrals(self.g, t1, self._cpu_threads, ccsd=ccsd,
                                       out=self._dressing_output() if ccsd else None)
            if g is not None:
                return _fock(h, g, o, xp=xp), g
        g = xp.array(self.g, order="C", copy=True)
        if self._t1_dressing is None or not self._t1_dressing.apply(g, t1):
            for axis in (0, 1):
                _dress_axis(g, axis, v, o, -t1.T, xp)
            for axis in (2, 3):
                _dress_axis(g, axis, o, v, t1, xp)
        return _fock(h, g, o, xp=xp), g

    def _triples(self, t2, t3, have_t3, im, vooo, vvvo, g):
        layout, o, v = self.layout, self.o, self.v
        xp, contract = self._xp, self._contract
        residual = xp.zeros(layout.shape)
        no, nv = layout.no, layout.nv
        actions = None if self._assembly is None else self._assembly.prepare(t2, vvvo, im.vv)
        if have_t3:
            support = self._column_support
            supported = support is not None and support.supports(t3)
            # Fold the one-body occupied contraction into the occupied
            # ladder and prepare the two particle-hole matrices once.
            woooo = im.oooo * 0.5
            for j in range(no):
                woooo[:, j, :, j] -= 0.5 * im.oo
            occupied = xp.ascontiguousarray(woooo.transpose(2, 3, 0, 1)).reshape(no**2, no**2)
            compact_occupied = (self._occupied_columns.product(occupied)
                                if supported and self._occupied_columns is not None else None)
            direct = xp.ascontiguousarray(im.ovvo.transpose(3, 1, 0, 2)).reshape(no * nv, no * nv)
            direct *= 0.25
            crossed = xp.ascontiguousarray(im.ovov.transpose(2, 1, 0, 3)).reshape(no * nv, no * nv)
            direct_product, crossed_product = direct.__matmul__, crossed.__matmul__
            ph_shifts = None
            ph = self._ph_columns
            compact_ph = (self._ph_fusion.prepare(direct, crossed)
                          if supported and self._ph_fusion is not None else None)
            if compact_ph is None:
                if ph is not None and supported:
                    direct_product, _ = ph.product(direct, layout=layout, spin=True)
                    crossed_product, _ = ph.product(crossed, layout=layout)
                    ph_shifts = ph.shifts
                elif self._ph_irreps is not None:
                    direct_product, _ = pair_product(direct, self._ph_irreps, xp=xp)
                    crossed_product, _ = pair_product(crossed, self._ph_irreps, xp=xp)
            del direct, crossed
        # The seed need only be materialized for one ordered (j,k) line.
        # Assemble the six paired permutations directly into packed output.
        for k in range(layout.no):
            if have_t3:
                plane = layout.plane(t3, k, axis=2, xp=xp)
                if compact_ph is not None and self._ph_batch:
                    compact_ph.prepare_plane(plane, k)
                if compact_occupied is None:
                    occ_ladder = (occupied @ plane.reshape(no**2, nv**3)).reshape(plane.shape)
                else:
                    occ_ladder = compact_occupied(plane, k)
            for j in range(layout.no):
                seed = None if actions is None else actions.seed(t2, j, k)
                if seed is None:
                    seed = contract("abd,idc->iabc", vvvo[:, :, :, j], t2[:, k])
                seed -= contract("ali,lbc->iabc", vooo[:, :, :, j], t2[:, k])
                if have_t3:
                    left = plane[:, j]
                    virtual = None if actions is None or not supported else actions.one_body(plane, j, k)
                    seed += 0.5 * (contract("ad,idbc->iabc", im.vv, left) if virtual is None else virtual)
                    del virtual
                    if compact_ph is not None:
                        compact_ph(seed, plane, j, k)
                    elif ph_shifts is None:
                        rhs = layout.spin_sum(left).reshape(no * nv, nv**2)
                        seed += direct_product(rhs).reshape(seed.shape)
                        del rhs
                        tile = crossed_product(plane[j].reshape(no * nv, nv**2))
                    else:
                        shift = ph_shifts[j, k]
                        seed += direct_product(plane, shift, j).reshape(seed.shape)
                        tile = crossed_product(plane, shift, j)
                    # The two crossed diagrams differ only by the output
                    # a/b permutation. Apply the matrix once.
                    if compact_ph is None:
                        self._crossed_update(seed, tile.reshape(seed.shape))
                        del tile
                    if compact_occupied is None:
                        seed += occ_ladder[:, j]
                    elif self._fold_occupied:
                        layout.add_seed(residual, seed, j, k, occupied=occ_ladder)
                        continue
                    else:
                        occ_ladder(seed, j)
                layout.add_seed(residual, seed, j, k)
            if have_t3:
                del plane, occ_ladder, left
                if compact_ph is not None and self._ph_batch:
                    compact_ph.release_plane()
        if have_t3:
            wvvvv = contract("lmde,lmab->abde", g[o, o, v, v], t2)
            wvvvv += g[v, v, v, v]
            if supported and self._cuda_ladder_columns:
                self._ladder(residual, t3, wvvvv, self.ladder_batch,
                             supported_occupied_irreps=support.occupied_labels)
            else:
                self._ladder(residual, t3, wvvvv, self.ladder_batch)
        layout.purify(residual, xp=xp)
        return residual

    def _evaluate(self, amplitudes, have_t3, *, triples=None, tangent_feedback=None):
        with _cpu.thread_scope(self._cpu_threads):
            return self._evaluate_scoped(amplitudes, have_t3, triples=triples,
                                         tangent_feedback=tangent_feedback)

    def _evaluate_scoped(self, amplitudes, have_t3, *, triples=None, tangent_feedback=None):
        xp, contract = self._xp, self._contract
        t1, t2, t3 = amplitudes
        o, v = self.o, self.v
        bare = self._bare_ladder_for(t1, t2, t3)
        if bare is not None:
            # CCSD: g(vvvv) stays undressed and unread (`_cpu.BareLadder`).
            f, g = self._dressed(t1, ccsd=True)
            ladder_g = None
            ladder = lambda _, t2: bare(t1, t2)                  # noqa: E731
        else:
            f, g = self._dressed(t1)
            ladder_g = self._ladder_g(g, t1) if xp is np and self._cpu_fast else None
            ladder = (partial(_cpu.doubles_ladder, virtual_irreps=self._virtual_irreps, sectors=ladder_g)
                      if xp is np and self._cpu_fast else None)
        r1, r2, intermediates, spin_g, c2 = eq.lower_residuals(f, g, t2, o, v, contract,
                                                               vvvv_ladder=ladder)
        if not len(t3):
            return r1, r2 + r2.transpose(1, 0, 3, 2), xp.zeros(self.layout.shape)
        vooo, vvvo = eq.triples_sources(f, g, t2, c2, o, v, contract)
        threaded = (xp is np and self._cpu_fast and triples is None
                    and tangent_feedback is None)
        with (_cpu.ThreadPool(_cpu.cpu_threads(self._cpu_threads)) if threaded
              else nullcontext()) as pool:
            blocked = self._blocked_t3(t1, t2, t3, pool.workers) if threaded else None
            if have_t3:
                if blocked is not None:
                    _blocked.feedback(*blocked[:2], r1, r2, vooo, vvvo, f, g, blocked[2], o, v,
                                      pool.workers)
                elif threaded:
                    _cpu.feedback(r1, r2, vooo, vvvo, f, g, self.layout, t3, o, v, pool)
                elif tangent_feedback is None:
                    eq.triples_feedback(r1, r2, vooo, vvvo, f, g, self.layout, t3, o, v,
                                        contract=contract, xp=xp,
                                        feedback=(self._feedback if self._feedback is not None
                                                  and self._column_support.supports(t3) else None))
                else:
                    tangent_feedback(r1, r2, vooo, vvvo, f, g, self.layout, t3, o, v)
                intermediates.ovvo = eq.add_product(contract, intermediates.ovvo,
                                                    "mled,miea->ladi", spin_g, c2, 0.5)
                intermediates.ovov = eq.add_product(contract, intermediates.ovov,
                                                    "mlde,imea->laid", g[o, o, v, v], t2, -0.5)
            r2 = r2 + r2.transpose(1, 0, 3, 2)
            if blocked is not None:
                layout, tables, t3b = blocked
                try:
                    r3 = _blocked.triples(layout, tables, t2, t3b, have_t3, intermediates, vooo,
                                          vvvo, g, o, v, pool.workers, ladder_g=ladder_g)
                    return r1, r2, layout.unpack(r3, threads=pool.workers)
                except _blocked.NotBlockable:
                    self._cpu_blocked = False       # an operator couples irreps: stay dense
            if threaded:
                return r1, r2, _cpu.triples(self.layout, t2, t3, have_t3, intermediates, vooo,
                                            vvvo, g, o, v, pool, ph_irreps=self._ph_irreps,
                                            virtual_irreps=self._virtual_irreps)
        assemble = self._triples if triples is None else triples
        return r1, r2, assemble(t2, t3, have_t3, intermediates, vooo, vvvo, g)

    def _nonzero(self, t3):
        """Whether T3 has a nonzero entry (a NaN counts), on all threads when it can."""
        compiled = self._compiled_kernels()
        if (compiled is not None and t3.size and isinstance(t3, np.ndarray)
                and t3.flags.c_contiguous and t3.dtype == np.float64):
            return compiled[0].max_abs(t3.ctypes.data, t3.size, compiled[1]) != 0.0
        return bool(np.any(t3))

    def _residuals(self, amplitudes, *, device_step=None):
        if self._xp is np:
            return self._evaluate(amplitudes, self._nonzero(amplitudes[2]))
        xp, pool = self._xp, self._pool
        success = False
        device_amplitudes = residuals = None
        try:
            with xp.cuda.using_allocator(pool.malloc):
                device_amplitudes = tuple(xp.asarray(a) for a in amplitudes)
                have_t3 = self._nonzero(amplitudes[2])
                residuals = self._evaluate(device_amplitudes, have_t3)
                del device_amplitudes
                if device_step is None:
                    result = tuple(r.get() for r in residuals)
                else:
                    jacobi, *controls = device_step
                    result = jacobi.finish(residuals, *controls)
                self.device_pool_reserved_bytes = pool.total_bytes()
                del residuals
                success = True
                return result
        finally:
            device_amplitudes = residuals = None
            if not success or not self._cuda_pool_cache:
                pool.free_all_blocks()

    def _symmetric_integrals(self, threads=None):
        """Whether the integrals have no symmetry-forbidden entry (checked once)."""
        if self._integrals_symmetric is None:
            from .spatial_symmetry import has_integral_leakage

            self._integrals_symmetric = bool(self.symmetry_projected or not has_integral_leakage(
                self.h, self.g.transpose(0, 2, 1, 3), self._labels,
                threads=_cpu.cpu_threads(threads if threads is not None else self._cpu_threads)))
        return self._integrals_symmetric

    def _dressing_output(self):
        """During `solve`, the n^4 array the CCSD dressing of each
        evaluation writes into (nothing reads it after the evaluation, so one
        array serves every iteration and its pages are faulted in once: about
        a quarter of a water aug-cc-pVTZ iteration), released when `solve`
        returns; None otherwise."""
        if not self._hold_dressing:
            return None
        if self._dressing_buffer is None:
            self._dressing_buffer = np.empty(self.g.shape)
        return self._dressing_buffer

    def _bare_ladder_for(self, t1, t2, t3):
        """The CCSD particle-particle ladder without dressing g(vvvv)
        (`_cpu.BareLadder`, built once) when the compiled CPU path evaluates
        plain CCSD amplitudes with this class's dressing; else None (CCSDT
        reads the dressed g(vvvv) in its T3 ladder; EOM and response dress
        dual numbers). ITERCC_BARE_LADDER=0 turns it off."""
        if (type(t1) is not np.ndarray or type(t2) is not np.ndarray or len(t3) or self._xp is not np
                or not self._cpu_fast or self._t1_dressing is not None or "_dressed" in vars(self)
                or type(self)._dressed is not RestrictedCC._dressed
                or t1.dtype != np.float64 or t2.dtype != np.float64
                or not hasattr(_cpu._kernels.library(), "dress_integrals_ccsd")
                or self.g.dtype != np.float64 or os.environ.get("ITERCC_BARE_LADDER", "1") == "0"):
            return None
        if self._bare_ladder is None:
            structure = None
            if (self._labels is not None and self._cpu_blocked and self._virtual_irreps is not None
                    and self._symmetric_integrals()):
                if self._ladder_structure is None:
                    from ._blocked import irrep_groups

                    no = self.layout.no
                    self._ladder_structure = _cpu.LadderSectors(self._labels[no:], irrep_groups(self._labels))
                structure = self._ladder_structure
            self._bare_ladder = _cpu.BareLadder(self.g, self.layout.no, structure=structure,
                                                virtual_irreps=self._virtual_irreps, threads=self._cpu_threads)
        return self._bare_ladder

    def _ladder_g(self, g, t1):
        """(LadderSectors, even, odd): the sector blocks of the dressed g(vvvv), for
        the doubles and triples ladders, or None. With symmetric integrals and a
        T1 that is zero in every forbidden entry the dressed Hamiltonian is
        exactly block-diagonal in pair irreps, so its blocks are all of it."""
        if (self._labels is None or not self._cpu_blocked or self._virtual_irreps is None
                or _cpu._kernels.library() is None or not isinstance(g, np.ndarray)
                or g.dtype != np.float64):
            return None
        no = self.layout.no
        occupied, virtual = self._labels[:no], self._labels[no:]
        if np.any(t1[occupied[:, None] != virtual]) or not self._symmetric_integrals():
            return None
        if self._ladder_structure is None:
            from ._blocked import irrep_groups

            self._ladder_structure = _cpu.LadderSectors(virtual, irrep_groups(self._labels))
        v = self.v
        return (self._ladder_structure, *self._ladder_structure.build(g[v, v, v, v]))

    def _blocked_t3(self, t1, t2, t3, threads):
        """(layout, plane tables, blocked T3) when the blocked CPU path applies, else None.

        It applies with orbital labels, the compiled core (its GEMM executor
        runs the products) and amplitudes that are exactly zero in every
        symmetry-forbidden entry, so the blocked residual equals the dense one.
        Exactly symmetric inputs give exactly symmetric residuals, so solves
        stay on this path.
        """
        if self._labels is None or not self._cpu_blocked or not len(t3):
            return None
        if _cpu._kernels.library() is None:
            return None
        if self._blocked_plan is None:
            from ._blocked import BlockedTriples
            from ._blocked_residual import PlaneTables

            if not self._symmetric_integrals(threads):
                self._cpu_blocked = False
                return None

            no = self.layout.no
            occupied, virtual = self._labels[:no], self._labels[no:]
            layout = BlockedTriples(self.layout, occupied, virtual)
            occ2, vir2 = occupied[:, None] ^ occupied, virtual[:, None] ^ virtual
            self._blocked_plan = (layout, PlaneTables(layout, occupied),
                                  occupied[:, None] != virtual,
                                  occ2[:, :, None, None] != vir2[None, None])
        layout, tables, t1_forbidden, t2_forbidden = self._blocked_plan
        if np.any(t1[t1_forbidden]) or np.any(t2[t2_forbidden]):
            return None
        t3 = np.ascontiguousarray(t3, dtype=np.float64)
        if layout.leakage(t3, threads=threads) != 0.0:
            return None
        return layout, tables, layout.pack(t3, threads=threads)

    def _symmetric(self, amplitudes):
        """NumPy amplitudes with their symmetry-forbidden entries removed when
        the integrals were projected and the removed entries stay within
        `symmetry_tolerance` (a start from an unprojected solve, say), so the
        solve runs on the blocked path; otherwise unchanged."""
        if not self.symmetry_projected or self._xp is not np or len(amplitudes) != 3:
            return amplitudes
        from .spatial_symmetry import project_amplitudes

        try:
            return project_amplitudes(amplitudes if len(amplitudes[2]) else amplitudes[:2],
                                      self._labels, tolerance=self.symmetry_tolerance)[0] + (
                () if len(amplitudes[2]) else (amplitudes[2],))
        except ValueError:
            return amplitudes

    def _warn_dense_start(self, amplitudes):
        """Warn when a labeled CPU solve starts off the blocked path because the
        initial amplitudes (a saved raw-Hamiltonian state, say) have nonzero
        symmetry-forbidden entries: every evaluation would then run dense."""
        if (self._labels is None or self._xp is not np or not self._cpu_fast
                or not len(amplitudes[2]) or _cpu._kernels.library() is None):
            return
        threads = _cpu.cpu_threads(self._cpu_threads)
        if self._blocked_t3(*amplitudes, threads) is None and self._cpu_blocked:
            warnings.warn("The initial amplitudes have nonzero symmetry-forbidden entries, so "
                          "the solve runs on the dense CPU path (symmetry_tolerance > 0 "
                          "projects them).", RuntimeWarning, stacklevel=3)

    def _compiled_kernels(self):
        """(library, threads) of the CPU kernels for the host passes of a solve
        (residual maxima, Jacobi step, DIIS), or None without the core. GPU
        residuals arrive as host arrays, so their solves use them too."""
        library = _cpu._kernels.library()
        return None if library is None else (library, _cpu.cpu_threads(self._cpu_threads))

    def release_device_memory(self):
        """Release cached free device allocations after a batch of evaluations."""
        if self._xp is not np:
            self._pool.free_all_blocks()

    def residuals(self, amplitudes):
        """Unpreconditioned spatial residuals in the amplitude layout."""
        return self._residuals(self._validated(amplitudes))

    def _energy(self, t1, t2):
        o, v = self.o, self.v
        tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
        eri = self.g[o, o, v, v]
        return float(2 * np.einsum("ia,ia->", self.fock[o, v], t1)
                     + 2 * np.einsum("ijab,ijab->", eri, tau)
                     - np.einsum("ijba,ijab->", eri, tau))

    def correlation_energy(self, amplitudes):
        t1, t2, _ = self._validated(amplitudes)
        return self._energy(t1, t2)

    def solve(self, *, initial=None, energy_tol=1e-11, residual_tol=1e-9,
              max_iter=100, diis_size=6, diis_triples=False, level_shift=0.0,
              damping=0.0, callback=None, checkpoint_callback=None):
        """Iterate all amplitudes, with a raw spatial residual test at every rank.

        DIIS extrapolates T1/T2 by default to bound the T3 history. T3 is
        updated on every step. Use diis_triples=True to extrapolate all three
        ranks. This keeps one full amplitude and step per DIIS history slot.
        callback receives each Iteration record after evaluating its residual.
        checkpoint_callback receives the one-based evaluation index, record,
        and current amplitudes before the update. Treat the arrays as read-only
        and copy them to retain an in-memory snapshot after the callback.
        """
        if not (np.isfinite(energy_tol) and energy_tol > 0
                and np.isfinite(residual_tol) and residual_tol > 0):
            raise ValueError("Expected positive finite convergence thresholds")
        if (not isinstance(max_iter, Integral) or max_iter < 1
                or not isinstance(diis_size, Integral) or diis_size < 0
                or not 0 <= damping < 1 or not np.isfinite(level_shift) or level_shift < 0):
            raise ValueError("Invalid iteration controls")
        o, v = self.o, self.v
        occ, vir = self.fock.diagonal()[o], self.fock.diagonal()[v] + level_shift
        d1 = occ[:, None] - vir
        d2 = d1[:, None, :, None] + d1[None, :, None, :]
        d3v = vir[:, None, None] + vir[None, :, None] + vir[None, None, :]
        d3o = [sum(occ[p] for p in indices) for indices in self.triples_indices]
        if (np.min(np.abs(d1)) < 1e-12 or np.min(np.abs(d2)) < 1e-12
                or _near_zero_triples_denominator(d3o, d3v)):
            raise ValueError("Near-zero orbital denominator. Use a positive level shift.")
        if initial is None:
            amplitudes = (self.fock[o, v] / d1, self.g[o, o, v, v] / d2, np.zeros(self.layout.shape))
        else:
            amplitudes = tuple(a.copy() for a in self._validated(initial))
            amplitudes = self._symmetric(amplitudes)
            self._warn_dense_start(amplitudes)
        del initial
        jacobi = None
        if self._cuda_jacobi:
            from ._cuda_jacobi import CudaJacobi

            jacobi = CudaJacobi(self._xp, d1, d2, d3o, d3v, damping)
        compiled = self._compiled_kernels()
        if diis_triples and self._diis_columns is not None:
            diis = _SymmetricBlockDIIS(diis_size, self._diis_columns, compiled is not None,
                                       self._cpu_threads)
        else:
            # Without diis_triples it extrapolates (T1, T2) only.
            diis = _BlockDIIS(diis_size, compiled is not None, self._cpu_threads)
        d3o_array = np.array(d3o, dtype=np.float64)
        d3v = np.ascontiguousarray(d3v)
        history = []
        last, converged = np.inf, False
        self._hold_dressing = True             # `_dressing_output`
        for iteration in range(max_iter):
            t1, t2, t3 = amplitudes
            correlation = self._energy(t1, t2)
            can_update = iteration + 1 < max_iter
            if diis_triples and can_update:
                diis.reserve_slot()
            change = correlation - last
            energy_ready = abs(change) <= energy_tol
            device_gate = jacobi is not None and (energy_ready or not can_update)
            residuals = steps = None
            if not device_gate:
                residuals = self._residuals(amplitudes)
                r3 = residuals[2]
                if compiled is not None and r3.flags.c_contiguous and r3.dtype == np.float64:
                    triples_max = compiled[0].max_abs(r3.ctypes.data, r3.size, compiled[1])
                else:
                    triples_max = np.max([np.max(np.abs(tile)) for tile in r3], initial=0.0)
                maxima = (float(np.max(np.abs(residuals[0]))), float(np.max(np.abs(residuals[1]))),
                          float(triples_max))
                del r3
            else:
                maxima, steps = self._residuals(amplitudes, device_step=(
                    jacobi, energy_ready, residual_tol, can_update))
            if not np.isfinite((correlation, *maxima)).all():
                raise FloatingPointError("Nonfinite coupled-cluster iterate")
            info = Iteration(self.reference_energy + correlation, change, maxima)
            history.append(info)
            if callback is not None:
                callback(info)
            if checkpoint_callback is not None:
                checkpoint_callback(iteration + 1, info, amplitudes)
            if energy_ready and max(maxima) <= residual_tol:
                converged = True
                break
            if can_update:
                if not device_gate:
                    r1, r2, r3 = residuals
                    step1, step2 = (1 - damping) * r1 / d1, (1 - damping) * r2 / d2
                    if (compiled is not None and r3.size and r3.flags.c_contiguous
                            and r3.dtype == np.float64):
                        compiled[0].triples_step(r3.ctypes.data, len(r3), r3[0].size,
                                                 d3o_array.ctypes.data, d3v.ctypes.data,
                                                 1 - damping, compiled[1])
                    else:
                        for tile, e in zip(r3, d3o):
                            tile *= 1 - damping
                            tile /= e - d3v
                else:
                    step1, step2, r3 = steps
                if diis_triples:
                    amplitudes = diis.take_step(amplitudes, (step1, step2, r3))
                else:
                    t1, t2 = diis.update((t1 + step1, t2 + step2), (step1, step2))
                    if (compiled is not None and t3.size and t3.flags.c_contiguous
                            and r3.flags.c_contiguous and t3.dtype == r3.dtype == np.float64):
                        compiled[0].axpy(t3.ctypes.data, r3.ctypes.data, 1.0, t3.size, compiled[1])
                    else:
                        t3 += r3
                    amplitudes = t1, t2, t3
                del r3
            del residuals, steps
            last = correlation
        self._hold_dressing, self._dressing_buffer = False, None
        return Result(self.reference_energy + correlation, correlation, amplitudes, converged, tuple(history))
