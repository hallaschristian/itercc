"""Matrix-free spin-singlet EOM-CCSDT with analytic CPU/CUDA Jacobian actions.

Nonzero Abelian target irreps are supported. Ground-state support masks must
never be used on excitation vectors. Triplet/SF sectors are explicitly absent.
"""
from copy import copy
from dataclasses import dataclass
from functools import cache, partial
import os
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from . import _restricted_equations as eq
from ._cuda_packed import PackedTriples
from .krylov import davidson
from ._directional import Directional, DirectionalBackend, DirectionalCudaLayout
from ._triples_layout import TriplesLayout
from .restricted import RestrictedCC, _pair_ladder, _fock

_BARE_MIN_VIRTUALS = 30     # `EOMSinglet._bare_ladder`: the dense ladder is faster below


@cache
def _cuda_triples_diagonal():
    from cupy import ElementwiseKernel

    return ElementwiseKernel(
        "I position, raw float64 energies, raw int32 slots, int32 no, int32 nv",
        "float64 gap",
        """
        const int nv2 = nv * nv, nv3 = nv2 * nv;
        const long long slot = 3LL * (position / nv3);
        const int rem = position % nv3;
        const int a = rem / nv2, b = (rem / nv) % nv, c = rem % nv;
        gap = energies[no + a] + energies[no + b] + energies[no + c]
              - energies[slots[slot]] - energies[slots[slot + 1]] - energies[slots[slot + 2]];
        """, "itercc_triples_orbital_diagonal")


class SingletSpace:
    """Sector-compressed coordinates with projection onto the physical subspace.

    Paired T2 symmetry and the spin-adapted T3 null relation are enforced at
    every unpack. The redundant packed metric is positive on this range.
    """
    def __init__(self, solver, orbital_irreps=None, irrep=0, *, xp=np, triples=True):
        self.xp, self.layout = xp, TriplesLayout(solver.layout.no, solver.layout.nv)
        self.shapes = solver.shapes if triples else solver.shapes[:2]
        n = solver.layout.no + solver.layout.nv
        labels = np.zeros(n, dtype=int) if orbital_irreps is None else np.asarray(orbital_irreps)
        if labels.shape != (n,) or labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels > 7)):
            raise ValueError("Expected Abelian XOR orbital irreps 0..7")
        if not isinstance(irrep, (int, np.integer)) or not 0 <= irrep <= 7:
            raise ValueError("Expected target Abelian irrep 0..7")
        if orbital_irreps is None and irrep != 0:
            raise ValueError("Nonzero target irrep requires orbital labels")
        self.irrep, self.labels = int(irrep), labels.copy()
        no = self.layout.no
        occ, vir = labels[:no], labels[no:]
        masks = [(occ[:, None] ^ vir) == irrep,
                 (occ[:, None, None, None] ^ occ[None, :, None, None]
                  ^ vir[None, None, :, None] ^ vir[None, None, None, :]) == irrep]
        if triples:
            virtual = vir[:, None, None] ^ vir[None, :, None] ^ vir[None, None, :]
            # One occupied slot at a time avoids an additional full T3 mask.
            indices = []
            stride = self.layout.nv**3
            dtype = (np.int32 if xp is not np and len(self.layout.indices) * stride
                     <= np.iinfo(np.int32).max else np.int64)
            for slot, (i, j, k) in enumerate(self.layout.indices):
                allowed = np.flatnonzero((virtual ^ occ[i] ^ occ[j] ^ occ[k]).ravel() == irrep)
                indices.append(allowed.astype(dtype, copy=False) + slot * stride)
            masks.append(None)
        self.indices = [xp.asarray(np.flatnonzero(mask.ravel())) for mask in masks[:2]]
        if triples:
            self.indices.append(xp.asarray(np.concatenate(indices) if indices else np.array([], dtype=int)))
        self.sizes = tuple(len(index) for index in self.indices)
        self.size = sum(self.sizes)
        self._repeats = None
        self._packed = None
        if triples and xp is not np:
            self._repeats = xp.asarray(
                [1 if i == j else 2 if j == k else 0 for i, j, k in self.layout.indices],
                dtype=xp.int8)
        if not self.size:
            raise ValueError("Requested excitation sector is empty")

    def packed_index(self):
        """Device tables mapping packed triples to (slot, abc); built once."""
        if self._packed is None:
            from ._cuda_packed import PackedTriplesIndex
            self._packed = PackedTriplesIndex(self.layout, self.labels, self.irrep,
                                              self.indices[2], self.xp)
        return self._packed

    def unpack(self, vector, *, triples=True, packed=False):
        """Project a packed vector onto the physical subspace.

        With packed=True on CUDA, the triples component stays in packed
        coordinates (a PackedTriples) instead of a dense T3-sized array.
        """
        xp = self.xp
        if vector.shape != (self.size,):
            raise ValueError("Excitation vector has the wrong dimension")
        arrays, start = [], 0
        ranks = len(self.shapes) if triples else 2
        packed = (packed and ranks == 3 and xp is not np and vector.dtype == xp.float64)
        for shape, index, size in zip(self.shapes[:ranks], self.indices[:ranks], self.sizes[:ranks]):
            if packed and len(arrays) == 2:
                owner = self.packed_index()
                source = xp.ascontiguousarray(vector[start:start + size])
                arrays.append(PackedTriples(owner, owner.symmetrize(source, 0)))
                start += size
                continue
            array = xp.zeros(shape, dtype=xp.result_type(vector.dtype, xp.float64))
            array.reshape(-1)[index] = vector[start:start + size]
            arrays.append(array)
            start += size
        arrays[1] = 0.5 * (arrays[1] + arrays[1].transpose(1, 0, 3, 2))
        if len(arrays) == 3 and not packed:
            if self._repeats is None:
                for tile, (i, j, k) in zip(arrays[2], self.layout.indices):
                    if i == j:
                        tile[...] = 0.5 * (tile + tile.swapaxes(0, 1))
                    elif j == k:
                        tile[...] = 0.5 * (tile + tile.swapaxes(1, 2))
                self.layout.purify(arrays[2], xp=xp)
            else:
                from ._cuda_layout import project_eom_triples
                project_eom_triples(arrays[2], self._repeats, self.layout.nv)
        return tuple(arrays)

    def pack(self, arrays):
        return self.xp.concatenate([array.data if isinstance(array, PackedTriples)
                                    else array.reshape(-1)[index]
                                    for array, index in zip(arrays, self.indices)])

    def project(self, vector):
        # Packed triples avoid a dense T3-sized temporary on every projection.
        return self.pack(self.unpack(vector, packed=True))

    def orbital_diagonal(self, fock):
        xp = self.xp
        energies = xp.diag(xp.asarray(fock))
        no = self.layout.no
        gap = energies[no:][None, :] - energies[:no, None]
        doubles = gap[:, None, :, None] + gap[None, :, None, :]
        parts = [gap.reshape(-1)[self.indices[0]], doubles.reshape(-1)[self.indices[1]]]
        if len(self.shapes) == 3:
            nv = self.layout.nv
            positions = self.indices[2]
            if xp is not np:
                triples = xp.asarray(np.asarray(self.layout.indices, dtype=np.int32).reshape(-1, 3))
                diagonal = xp.empty(self.size, dtype=energies.dtype)
                n1, n2 = self.sizes[:2]
                diagonal[:n1], diagonal[n1:n1+n2] = parts
                _cuda_triples_diagonal()(positions, energies, triples, no, nv, diagonal[n1+n2:])
                return diagonal
            stride = nv**3
            slot = positions // stride
            a = positions % stride // self.layout.nv**2
            b = positions % self.layout.nv**2 // self.layout.nv
            c = positions % self.layout.nv
            triples = xp.asarray(np.array(self.layout.indices, dtype=int).reshape(-1, 3))
            if len(positions):
                i, j, k = triples[slot].T
                parts.append(energies[no + a] + energies[no + b] + energies[no + c]
                             - energies[i] - energies[j] - energies[k])
            else:
                parts.append(xp.zeros(0))
        return xp.concatenate(parts)


def _cpu_dressing_kernel():
    """The compiled CPU kernels when they offer `dress_axis_from`, else None."""
    from . import _cpu_kernels

    library = _cpu_kernels.library()
    return library if library is not None and hasattr(library, "dress_axis_from") else None


class EOMSinglet:
    """Full singlet EOM-CCSDT (or explicitly selected CCSD development mode).

    With orbital labels, the integrals and the ground amplitudes are audited
    for symmetry leakage (the integrals must stay within 2e-9). When both stay
    within `symmetry_tolerance`, the Jacobian acts with their symmetry
    projections: every operator is then exactly block-diagonal, so the
    symmetry-blocked GEMMs apply (about nirrep times fewer flops than the
    dense ones that rounding-level leakage forces), and the actions change by
    the order of the dropped leakage, far below any solve tolerance.
    `symmetry_tolerance=0` keeps the original operators.
    `integral_symmetry_leakage`, `amplitude_symmetry_leakage` and
    `symmetry_projected` record what happened.
    """
    def __init__(self, solver, amplitudes, *, orbital_irreps=None, irrep=0,
                 backend="cpu", method="ccsdt", ground_residual_tol=1e-10, symmetry_tolerance=2e-9,
                 cache_dressing=None, fused_cuda_layout=True, tangent_triples=True,
                 triples_batch=None, projected_ladder=True, projected_particle_hole=True,
                 compiled_seed=True, fused_seed_fold=None, projected_occupied=None, stream_particle_hole=True, tangent_feedback=None,
                 packed_triples=True):
        if method not in ("ccsd", "ccsdt") or backend not in ("cpu", "cuda"):
            raise ValueError("Choose CCSD/CCSDT and CPU/CUDA")
        if not np.isfinite(ground_residual_tol) or ground_residual_tol <= 0:
            raise ValueError("Expected a positive ground-state residual tolerance")
        triples = method == "ccsdt"
        host = solver._validated(amplitudes if triples else amplitudes[:2], triples=triples)
        xp = np
        if backend == "cuda":
            import cupy as xp
        self.xp, self.method = xp, method
        self.tangent_triples = tangent_triples
        self.compiled_seed = compiled_seed
        self.fused_seed_fold = fused_seed_fold
        # Whole-plane row grouping helps once the dense plane is large.
        if projected_occupied is None:
            projected_occupied = solver.layout.no**2 * solver.layout.nv**3 * 8 >= 256 << 20
        self.projected_occupied = projected_occupied
        self.stream_particle_hole = stream_particle_hole
        if tangent_feedback is None:
            tangent_feedback = solver.layout.no**2 * solver.layout.nv**3 * 8 >= (256 << 20)
        self.tangent_feedback = tangent_feedback
        # Keep projected CUDA directions and images in packed triples
        # coordinates when every tangent stage has a packed implementation.
        self.packed_triples = packed_triples
        self._tangent_feedback = None
        self._occupied_plans = {}
        if triples_batch is not None and (not isinstance(triples_batch, int) or triples_batch < 1):
            raise ValueError("Triples batch must be a positive integer")
        self.triples_batch = triples_batch
        if not np.isfinite(symmetry_tolerance) or symmetry_tolerance < 0:
            raise ValueError("Expected a finite nonnegative symmetry tolerance")
        h, eri = solver.h, solver.g.transpose(0, 2, 1, 3)
        self.integral_symmetry_leakage = self.amplitude_symmetry_leakage = None
        self.symmetry_projected = False
        self.ground_energy = solver.reference_energy + solver._energy(*host[:2])
        if orbital_irreps is not None:
            from .spatial_symmetry import project_amplitudes, project_integrals
            projected_h, projected_eri, audited = project_integrals(h, eri, orbital_irreps, tolerance=2e-9)
            self.integral_symmetry_leakage = tuple(item.maximum for item in audited)
            if symmetry_tolerance and max(self.integral_symmetry_leakage) <= symmetry_tolerance:
                try:
                    projected, audited = project_amplitudes(host, orbital_irreps,
                                                            tolerance=symmetry_tolerance)
                except ValueError:      # leaky amplitudes: keep the original operators
                    projected = None
                if projected is not None:
                    self.amplitude_symmetry_leakage = tuple(item.maximum for item in audited)
                    h, eri, host = projected_h, projected_eri, projected
                    self.symmetry_projected = True
            del projected_h, projected_eri
        # A fresh proxy guarantees that ground-state-specific fused support
        # masks cannot silently discard Pi/Delta or other response sectors.
        primal = RestrictedCC(h, eri, solver.layout.no,
                             solver.reference_energy - float(np.trace(
                                 (solver.h + solver.fock)[solver.o, solver.o])))
        self.space = SingletSpace(primal, orbital_irreps, irrep, xp=xp, triples=method == "ccsdt")
        self.diagonal = self.space.orbital_diagonal(primal.fock)
        self._primal = primal
        self.proxy = copy(primal)
        self.ad = DirectionalBackend(xp)
        self.proxy._xp = self.ad
        if cache_dressing is None:
            # Cache the ground-state dressing and differentiate only its legs
            # (on the CPU with the compiled one-index transform); without it,
            # every action re-dresses the n^4 integrals as dual numbers.
            cache_dressing = backend == "cuda" or _cpu_dressing_kernel() is not None
        if not cache_dressing:
            self.proxy.h, self.proxy.g = xp.asarray(primal.h), xp.asarray(primal.g)
        self.proxy._contract = partial(eq.contract, xp=self.ad)
        self.proxy._ladder = partial(_pair_ladder, xp=self.ad)
        # Freeze the linearization point without a second device allocation.
        self.amplitudes = tuple(a.copy() if xp is np else xp.asarray(a) for a in host)
        if xp is not np and fused_cuda_layout and triples:
            from ._cuda_layout import CudaTriplesLayout
            from ._directional import DirectionalCudaLayout
            self.proxy.layout = DirectionalCudaLayout(CudaTriplesLayout(
                solver.layout.no, solver.layout.nv, xp))
            self.proxy._crossed_update = self.proxy.layout.subtract_crossed
        self._projected_ladder = None
        self._projected_particle_hole = None
        self.projected_particle_hole = projected_particle_hole
        self.projected_ladder = projected_ladder
        if (xp is not np and fused_cuda_layout and triples and orbital_irreps is not None
                and len(self.amplitudes[2]) and len(np.unique(orbital_irreps)) > 1):
            from ._cuda_projected_ladder import CudaProjectedLadder
            self._projected_ladder = CudaProjectedLadder(self.space.layout, self.space.labels, xp)
            from ._cuda_projected_ph import CudaProjectedParticleHole
            labels = self.space.labels
            no = solver.layout.no
            self._projected_particle_hole = CudaProjectedParticleHole(labels[:no], labels[no:], xp)
        if cache_dressing:
            self._cache_dressing()
            if self._projected_ladder is not None:
                from ._cuda_eom_feedback import CudaTangentFeedback
                no = self.space.layout.no
                labels = self.space.labels
                self._tangent_feedback = CudaTangentFeedback(labels[:no], labels[no:], xp)
        self.proxy._tangent_ladder = self._bare_ladder(getattr(self.space, "labels", None))
        residual = self._ground_residual(cache_dressing)
        maxima = [max(float(xp.max(r)), -float(xp.min(r))) if r.size else 0. for r in residual]
        maximum = max(maxima)
        if not np.isfinite(maxima).all() or maximum > ground_residual_tol:
            raise ValueError(f"Ground-state residual {maximum:.3e} exceeds {ground_residual_tol:.3e}")
        self.ground_residual_max = maximum
        self.matvec_count, self.matvec_seconds = 0, 0.0

    def _ground_residual(self, cache_dressing):
        """Audit the original ground residual without allocating zero tangents."""
        from ._directional import DirectionalCudaLayout

        if (self.xp is np and self.method == "ccsdt" and type(self.proxy.layout) is TriplesLayout
                and os.environ.get("ITERCC_CPU_TANGENT", "1") != "0"):
            # CPU: the plain (compiled) residual of the original Hamiltonian.
            return self._primal.residuals(self.amplitudes)
        if not isinstance(self.proxy.layout, DirectionalCudaLayout):
            return tuple(r.value for r in self._evaluate(tuple(Directional(a) for a in self.amplitudes)))
        plain = copy(self.proxy)
        xp = self.xp
        plain._xp = xp
        plain.layout = self.proxy.layout.base
        plain._contract = partial(eq.contract, xp=xp)
        plain._ladder = partial(_pair_ladder, xp=xp, fuse_cuda=True)
        plain._crossed_update = plain.layout.subtract_crossed_fused
        if cache_dressing:
            def dressed(t1):
                return tuple(operator.value for operator in self.proxy._dressed(Directional(t1)))
            plain._dressed = dressed
        return plain._evaluate(self.amplitudes, True)

    def _cache_dressing(self):
        """T1 excitation generators commute, so differentiate the dressed
        Hamiltonian by applying each infinitesimal leg transform once.
        Cache primal operators; never recompute their four-leg transform.
        """
        xp, o, v = self.xp, self.proxy.o, self.proxy.v
        plain = copy(self.proxy)
        plain._xp = xp
        f0, g0 = RestrictedCC._dressed(plain, self.amplitudes[0])
        h0 = xp.array(plain.h, copy=True)
        from .solver import _dress_axis
        _dress_axis(h0, 0, v, o, -self.amplitudes[0].T, xp)
        _dress_axis(h0, 1, o, v, self.amplitudes[0], xp)

        spare = {}

        def derivative_legs(tensor, response, creation, annihilation, ccsd=False):
            library = _cpu_dressing_kernel() if xp is np else None
            if (library is not None and tensor.ndim == 4 and tensor.dtype == np.float64
                    and tensor.flags.c_contiguous and response.dtype == np.float64):
                # On all threads, each output row formed once (no tensordot copies).
                from ._cpu_triples import cpu_threads

                n, no = tensor.shape[0], response.shape[0]
                nv = n - no
                threads = cpu_threads(None)
                kernel = getattr(library, "dress_derivative_ccsd" if ccsd else "dress_derivative", None)
                if (tuple(creation), tuple(annihilation)) == ((0, 1), (2, 3)) and kernel is not None:
                    # All four legs in one pass over g, no zero fill; `ccsd`: not
                    # the all-virtual planes beyond their occupied block, which
                    # only the particle-particle ladder reads (`_bare_ladder`).
                    output = spare.get(ccsd) if ccsd else None
                    # The CCSD output leaves most of the all-virtual planes
                    # unwritten: kept between actions while nothing else holds
                    # the last one (a fresh n^4 array faulted in about a quarter
                    # of its pages per action).
                    if output is None or output.shape != tensor.shape or sys.getrefcount(output) > 3:
                        output = np.empty_like(tensor)
                        if ccsd:
                            spare[ccsd] = output
                    d = np.ascontiguousarray(response)
                    kernel(output.ctypes.data, tensor.ctypes.data, n, no, d.ctypes.data, threads)
                    return output
                output = np.zeros_like(tensor)
                down = np.ascontiguousarray(-response.T)      # (nv, no)
                up = np.ascontiguousarray(response)           # (no, nv)
                for axis in creation:
                    library.dress_axis_from(output.ctypes.data, tensor.ctypes.data, n, axis, no, nv, 0, no,
                                            down.ctypes.data, threads)
                for axis in annihilation:
                    library.dress_axis_from(output.ctypes.data, tensor.ctypes.data, n, axis, 0, no, no, nv,
                                            up.ctypes.data, threads)
                return output
            output = xp.zeros_like(tensor)
            for axis in creation:
                source = xp.moveaxis(tensor, axis, 0)
                target = xp.moveaxis(output, axis, 0)
                target[v] -= xp.tensordot(response.T, source[o], axes=1)
            for axis in annihilation:
                source = xp.moveaxis(tensor, axis, 0)
                target = xp.moveaxis(output, axis, 0)
                target[o] += xp.tensordot(response, source[v], axes=1)
            return output

        shifted = self._shifted_dressing = {}

        def dressed_at(t1):
            """(h, f, g) dressed with another plain T1 (response's mixed
            derivatives evaluate at shifted amplitudes): the plain dressing
            (compiled on the CPU), the last one kept."""
            if shifted.get("t1") is not t1:
                f1, g1 = RestrictedCC._dressed(plain, t1)
                h1 = xp.array(plain.h, copy=True)
                _dress_axis(h1, 0, v, o, -t1.T, xp)
                _dress_axis(h1, 1, o, v, t1, xp)
                shifted.update(t1=t1, operators=(h1, f1, g1))
            return shifted["operators"]

        def dressed(t1, ccsd=False):
            if t1.value is self.amplitudes[0]:
                h, f, g = h0, f0, g0
            elif (isinstance(t1.value, xp.ndarray) and t1.value.dtype == np.float64
                  and (t1.tangent is None or t1.tangent.dtype == np.float64)):
                # The same leg transforms at any T1 (the generators commute).
                h, f, g = dressed_at(t1.value)
            else:
                return RestrictedCC._dressed(self.proxy, t1)
            if t1.tangent is None:
                return Directional(f), Directional(g)
            dh = derivative_legs(h, t1.tangent, (0,), (1,))
            dg = derivative_legs(g, t1.tangent, (0, 1), (2, 3), ccsd=ccsd)
            return Directional(f, _fock(dh, dg, o, xp=xp)), Directional(g, dg)

        self.proxy._dressed = dressed

    def release_shifted(self):
        """Drop what actions at shifted amplitudes left cached (their dressed
        integrals, ladder matrix, feedback increments and amplitude references:
        about a T3, an n^4 and an nv^4 array); the ground-state entries stay."""
        getattr(self, "_shifted_dressing", {}).clear()
        ground = getattr(self, "_cpu_ground", {})
        for name, entries in ground.items():
            if isinstance(entries, list):
                del entries[1:]
        ground.pop("blocked direction", None)

    def _evaluate(self, amplitudes):
        if self.method == "ccsdt":
            # Always include T3 diagrams, even for a zero ground-state T3:
            # their derivatives with respect to R3 can be nonzero.
            return self.proxy._evaluate(amplitudes, True)
        ladder = self.proxy._tangent_ladder
        # With the ladder from the undressed g(vvvv), the dressing's derivative
        # skips what only the dressed ladder would read.
        f, g = self.proxy._dressed(amplitudes[0], ccsd=ladder is not None)
        r1, r2, *_ = eq.lower_residuals(f, g, amplitudes[1], self.proxy.o, self.proxy.v, self.proxy._contract,
                                        vvvv_ladder=None if ladder is None else partial(ladder, amplitudes[0]))
        return r1, r2 + r2.transpose(1, 0, 3, 2)

    def _bare_ladder(self, labels):
        """`proxy._tangent_ladder`: the particle-particle ladder of dual-number
        amplitudes from the undressed g(vvvv) (`_cpu_triples.BareLadder`, built
        once, exactly 0.5 sum_cd g~(abcd) t2(ijcd) and its derivative) instead
        of three dense products with the dressed g(vvvv) and its derivative
        per action; None off the compiled CPU path, with fewer than
        `_BARE_MIN_VIRTUALS` virtual orbitals (water CCSD polarizabilities, 12
        cores: 25 virtuals 1.1 instead of 1.0 s, 36 virtuals 1.9 instead of
        3.7 s) or with ITERCC_BARE_LADDER=0 (=1 forces it)."""
        from . import _cpu_triples as _cpu

        g = self.proxy.g
        choice = os.environ.get("ITERCC_BARE_LADDER", "")
        if (self.xp is not np or _cpu._kernels.library() is None or not isinstance(g, np.ndarray)
                or g.dtype != np.float64 or choice == "0"):
            return None
        no = self.proxy.layout.no
        if choice != "1" and len(g) - no < _BARE_MIN_VIRTUALS:
            return None
        structure = virtual = None
        if labels is not None:
            labels = np.asarray(labels, dtype=np.int64)
            virtual = labels[no:]
            if self.symmetry_projected and len(np.unique(labels)) > 1:
                # Exactly symmetric integrals: only the diagonal sector blocks.
                from ._blocked import irrep_groups

                structure = _cpu.LadderSectors(virtual, irrep_groups(labels))
        bare = _cpu.BareLadder(g, no, structure=structure, virtual_irreps=virtual)
        contract = self.proxy._contract

        def ladder(t1, g_vvvv, t2):
            value = (t1.value, t2.value)
            tangent = (t1.tangent, t2.tangent)
            if not all(isinstance(x, np.ndarray) and x.dtype == np.float64 for x in value + tangent
                       if x is not None):
                return 0.5 * contract("abcd,ijcd->ijab", g_vvvv, t2)     # complex: dense
            # The dressed g keeps the t1 t1 term in g(abij): no tau.
            point = getattr(ladder, "point", None)
            if point is None or not (np.array_equal(point[0], value[0]) and np.array_equal(point[1], value[1])):
                ladder.point = point = (value[0].copy(), value[1].copy(), bare(*value, tau=False))
            if tangent[0] is None and tangent[1] is None:
                return Directional(point[2].copy())
            return Directional(point[2].copy(), bare.tangent(*value, *tangent, tau=False))

        return ladder

    def sigma(self, vector):
        if self.xp.iscomplexobj(vector):
            value = self.sigma(vector.real)
            if bool(self.xp.any(vector.imag)):
                value = value + 1j * self.sigma(vector.imag)
            return value
        start = perf_counter()
        direction = self.space.unpack(vector, packed=self._packed_path(self.space.irrep))
        result = self.space.pack(self._tangent(self.amplitudes, direction, irrep=self.space.irrep))
        del direction
        if not bool(self.xp.isfinite(result).all()):
            raise FloatingPointError("Nonfinite EOM Jacobian action")
        if self.xp is not np:
            self.xp.cuda.get_current_stream().synchronize()
        self.matvec_count += 1
        self.matvec_seconds += perf_counter() - start
        return result

    def _packed_path(self, irrep):
        """True when the projected CUDA action reads and writes packed triples.

        That requires the tangent triples assembly with the compiled seed, the
        compact occupied fold and the projected virtual ladder for `irrep`.
        """
        return bool(self.packed_triples and self.xp is not np and irrep is not None
                    and self.method == "ccsdt" and self.tangent_triples
                    and len(self.amplitudes[2]) and self.compiled_seed
                    and self.projected_occupied and self.projected_ladder
                    and self._projected_ladder is not None
                    and isinstance(self.proxy.layout, DirectionalCudaLayout))

    def _exact_labels(self, t1, dt1, irrep):
        """The orbital labels when the feedback operators are exactly symmetric
        (projected integrals and ground T1, so the dressed f and g are totally
        symmetric) and their derivatives exactly of the direction's irrep (dT1
        of that irrep), else None (`_cpu_tangent.feedback`). A shifted point
        T + h L of a non-A1 L has a T1 of mixed irreps and gets None."""
        labels = getattr(self.space, "labels", None)
        if labels is None or not self.symmetry_projected or not isinstance(t1, np.ndarray):
            return None
        no = t1.shape[0]
        pair = np.asarray(labels[:no])[:, None] ^ np.asarray(labels[no:])[None, :]
        if np.any(t1[pair != 0]) or (dt1 is not None and np.any(dt1[pair != int(irrep or 0)])):
            return None
        return labels

    def _tangent(self, amplitudes, direction, *, irrep=None):
        dual = tuple(Directional(t, r) for t, r in zip(amplitudes, direction))
        if (self.tangent_triples and self.method == "ccsdt" and len(dual[2]) and self.xp is np
                and type(self.proxy.layout) is TriplesLayout
                and os.environ.get("ITERCC_CPU_TANGENT", "1") != "0"):
            # CPU: product-rule actions with the compiled triples and feedback.
            from . import _cpu_tangent

            # The ground parts of each action are computed once (`ground`).
            ground = self.__dict__.setdefault("_cpu_ground", {})
            labels = getattr(self.space, "labels", None)
            r1, r2, r3 = self.proxy._evaluate(
                dual, True,
                tangent_feedback=partial(_cpu_tangent.feedback, ground=ground, proxy=self.proxy,
                                         labels=self._exact_labels(amplitudes[0], direction[0], irrep),
                                         irrep=irrep),
                triples=partial(_cpu_tangent.triples, self.proxy, labels=labels, ground=ground,
                                irrep=irrep))
            return r1.tangent, r2.tangent, r3
        if self.tangent_triples and self.method == "ccsdt" and len(dual[2]):
            from ._eom_triples import triples_tangent
            projected = self._projected_ladder if self.projected_ladder and irrep is not None else None
            particle_hole = self._projected_particle_hole if self.projected_particle_hole and irrep is not None else None
            occupied = None
            if self.projected_occupied and self.compiled_seed and irrep is not None and self._projected_ladder is not None:
                if irrep not in self._occupied_plans:
                    from ._cuda_eom_occupied import CudaProjectedOccupied
                    no = self.space.layout.no
                    labels = self.space.labels
                    self._occupied_plans[irrep] = CudaProjectedOccupied(labels[:no], labels[no:], irrep, self.xp)
                occupied = self._occupied_plans[irrep]
            feedback = None
            if (self.tangent_feedback and self._tangent_feedback is not None
                    and irrep is not None and amplitudes is self.amplitudes):
                feedback = partial(self._tangent_feedback.apply, irrep=irrep)
            r1, r2, r3 = self.proxy._evaluate(dual, True, tangent_feedback=feedback,
                triples=partial(triples_tangent, self.proxy, self.xp, batch=self.triples_batch,
                                projected=projected, irrep=irrep, particle_hole=particle_hole,
                                compiled_seed=self.compiled_seed, fused_seed_fold=self.fused_seed_fold,
                                occupied_plan=occupied, stream_particle_hole=self.stream_particle_hole))
            return r1.tangent, r2.tangent, r3
        values = self._evaluate(dual)
        return tuple(self.xp.zeros_like(value.value) if value.tangent is None
                     else value.tangent for value in values)

    def full_residual_norm(self, vector, energy):
        """Residual norm of the engine's full operator, including components
        outside the target irrep, so leakage is not hidden by packing. With
        `symmetry_projected` the engine acts with projected integrals and
        amplitudes, so integral leakage itself is not audited here.
        """
        start = perf_counter()
        directions = self.space.unpack(vector.real)
        sigma = self._tangent(self.amplitudes, directions)
        self.matvec_count += 1
        if self.xp.iscomplexobj(vector) and bool(self.xp.any(vector.imag)):
            imaginary = self.space.unpack(vector.imag)
            extra = self._tangent(self.amplitudes, imaginary)
            sigma = [s + 1j * v for s, v in zip(sigma, extra)]
            directions = [r + 1j * i for r, i in zip(directions, imaginary)]
            self.matvec_count += 1
        squared = 0.
        for image, direction in zip(sigma, directions):
            residual = image - energy * direction
            squared += float(self.xp.vdot(residual, residual).real)
        self.matvec_seconds += perf_counter() - start
        if not np.isfinite(squared):
            raise FloatingPointError("Nonfinite full EOM residual audit")
        return float(np.sqrt(squared))

    def solve(self, nroots=3, *, residual_tol=1e-8, max_iter=100,
              max_space=20, guesses=None, callback=None, memory_budget_gib=None,
              checkpoint_callback=None, thick_restart=True, olsen=False):
        """Restarted real non-Hermitian block Davidson with audited Ritz roots.

        Uses the shared `itercc.krylov.davidson`: the basis and images stay in
        one preallocated block on the selected device, restarts keep the
        current and previous Ritz vectors (thick_restart) without new actions,
        and only the small projected matrix crosses to CPU. Converged roots
        are certified by a fresh full-space action.
        """
        xp, space = self.xp, self.space
        if (not isinstance(nroots, int) or nroots < 1 or max_space < 2 * nroots + 1
                or max_iter < 1 or not np.isfinite(residual_tol) or residual_tol <= 0):
            raise ValueError("Invalid root count, subspace, iteration or residual controls")
        estimated = (2 * max_space + 6) * space.size * 8
        estimated += 12 * sum(np.prod(shape, dtype=np.int64) for shape in space.shapes) * 8
        # The proxy is a RestrictedCC (h) or, for spin-flip engines, a TensorCC.
        orbitals = len(self.proxy.h if hasattr(self.proxy, "h") else self.proxy.hamiltonian.h)
        estimated += 4 * orbitals**4 * 8
        if memory_budget_gib is not None and estimated > memory_budget_gib * 1024**3:
            raise MemoryError(f"Estimated EOM working set {estimated / 1024**3:.2f} GiB exceeds budget")
        outcome = davidson(lambda vector: self.sigma(vector), self.diagonal, space.project,
                           size=space.size, nroots=nroots, xp=xp, residual_tol=residual_tol,
                           max_iter=max_iter, max_space=max_space, guesses=guesses,
                           callback=callback, checkpoint_callback=checkpoint_callback,
                           certify=self.full_residual_norm, matvecs=lambda: self.matvec_count,
                           retain_previous=thick_restart, olsen=olsen)
        values = outcome.values
        return EOMResult(outcome.converged, values, self.ground_energy + values, outcome.vectors,
                         outcome.residual_norms, outcome.history, self.method, space.irrep,
                         self.matvec_count, self.matvec_seconds)


class HostDavidson:
    """Keep the Davidson basis on the host while a CUDA engine applies the Jacobian."""

    xp = np
    solve = EOMSinglet.solve

    def __init__(self, engine, *, project=None, audit=None):
        self.engine = engine
        self.proxy, self.method = engine.proxy, engine.method
        self.ground_energy = engine.ground_energy
        self.diagonal = engine.diagonal.get()
        self.transfer_seconds = 0.0
        self.space = SimpleNamespace(size=engine.space.size, shapes=engine.space.shapes,
                                     irrep=engine.space.irrep,
                                     project=project if project is not None else self._project)
        if audit is not None:
            self.full_residual_norm = audit

    def _project(self, vector):
        return self.engine.space.project(self.engine.xp.asarray(vector)).get()

    def full_residual_norm(self, vector, energy):
        return self.engine.full_residual_norm(self.engine.xp.asarray(vector), energy)

    @property
    def matvec_count(self):
        return self.engine.matvec_count

    @property
    def matvec_seconds(self):
        return self.engine.matvec_seconds

    def sigma(self, vector):
        start = perf_counter()
        device = self.engine.xp.asarray(vector, blocking=True)
        uploaded = perf_counter()
        image = self.engine.sigma(device)
        download_start = perf_counter()
        result = image.get()
        self.transfer_seconds += uploaded - start + perf_counter() - download_start
        return result


@dataclass(frozen=True)
class EOMResult:
    converged: bool
    excitation_energies: np.ndarray
    total_energies: np.ndarray
    vectors: tuple
    residual_norms: tuple
    iterations: tuple
    method: str
    irrep: int
    matvecs: int
    matvec_seconds: float
