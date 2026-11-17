"""Full iterative restricted CCSDTQ ground states on a single CUDA device or in NumPy."""
from copy import copy
import ctypes
from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import combinations_with_replacement, permutations
from math import isfinite
from numbers import Integral
import os
from types import SimpleNamespace

import numpy as np

from .checkpoint_io import read_array_directory, to_device, write_array_directory
from .hamiltonian import _real64
from .restricted import RestrictedCC
from . import _cpu_kernels, _cpu_quadruples
from .solver import Iteration, Result, _DIIS
from .spatial_symmetry import orbital_labels, project_integrals

STATE_FORMAT = "itercc-rccsdtq-state-1"


def _host_axpy(value, weight, total):
    total += weight * value


def _host_copy(source):
    """A new float64 NumPy array from a host, memory-mapped or saved (lazy) array."""
    from .checkpoint_io import SavedArray

    return np.asarray(source).astype(np.float64, copy=not isinstance(source, SavedArray))


class _FullRankDIIS(_DIIS):
    """Keep trial vectors and errors as device arrays without flattening copies."""

    def __init__(self, size, xp):
        super().__init__(size, xp)
        self._axpy = _host_axpy if xp is np else xp.ElementwiseKernel(
            "float64 value, float64 weight", "float64 total",
            "total += weight * value", "itercc_ccsdtq_diis_axpy",
        )

    def _dot(self, left, right):
        cp = self.xp
        return sum(cp.dot(a.reshape(-1), b.reshape(-1)) for a, b in zip(left, right))

    def take_step(self, amplitudes, updates):
        self.reserve_slot()
        trial = tuple(a + r for a, r in zip(amplitudes, updates))
        weights = self._append(trial, updates, self._dot)
        if weights is None:
            return trial
        mixed = tuple(self.xp.zeros_like(a) for a in trial)
        for weight, (vector, _) in zip(weights, self.history):
            for target, value in zip(mixed, vector):
                self._axpy(value, weight, target)
        return mixed


class _PackedTriplesDIIS(_DIIS):
    """T1–T3 DIIS that stores each occupied triple i <= j <= k of T3 once.

    Entries are weighted by the square root of the number of distinct
    occupied permutations, so every inner product equals the one of the full
    arrays and the extrapolation matches full-array DIIS to rounding.
    """

    def __init__(self, size, cp, nocc):
        super().__init__(size, cp)
        keys = np.asarray(list(combinations_with_replacement(range(nocc), 3)),
                          dtype=np.int64).reshape(-1, 3)
        self._keys = tuple(cp.asarray(column) for column in keys.T)
        weight = np.sqrt([len(set(permutations(key))) for key in map(tuple, keys)])
        self._weight = cp.asarray(weight)[:, None, None, None]
        if cp is np:
            # Compiled passes (all threads): packing copies runs of slabs
            # (i, j, k0..k1) of one weight; unpacking writes every slab of the
            # full T3 from its sorted slab, transposed, as the permutation loop
            # below leaves it (the last permutation reaching a slab wins).
            self._runs = []           # [start, stop, i, j, k0, weight]
            for s, (key, w) in enumerate(zip(keys.tolist(), weight.tolist())):
                last = self._runs[-1] if self._runs else None
                if last is not None and last[1] == s and last[2:4] == key[:2] and last[5] == w:
                    last[1] = s + 1
                else:
                    self._runs.append([s, s + 1, *key, w])
            source = {}
            for perm in permutations(range(3)):
                for s, key in enumerate(keys.tolist()):
                    source[tuple(key[p] for p in perm)] = (s, perm)
            self._sources = sorted(source.items())

    def _compiled(self, *arrays):
        return (self.xp is np and _cpu_quadruples.available()
                and all(a.dtype == np.float64 and a.flags.c_contiguous for a in arrays))

    def pack(self, t1, t2, t3):
        if self._compiled(t3):
            n1, n2 = t1.size, t2.size
            out = np.empty(n1 + n2 + len(self._weight) * t3[0, 0, 0].size)
            out[:n1], out[n1:n1 + n2] = t1.ravel(), t2.ravel()
            packed = out[n1 + n2:].reshape((-1,) + t3.shape[3:])
            for start, stop, i, j, k, w in self._runs:
                _cpu_quadruples.permuted_add(packed[start:stop].reshape(-1), None,
                                             t3[i, j, k:k + stop - start].reshape(-1), 0.0, w)
            return out
        i, j, k = self._keys
        return self.xp.concatenate((t1.ravel(), t2.ravel(), (t3[i, j, k] * self._weight).ravel()))

    def unpack(self, vector, like):
        t1, t2, t3 = like
        n1, n2 = t1.size, t2.size
        if self._compiled(vector, t3):
            packed = vector[n1 + n2:].reshape((-1,) + t3.shape[3:])
            full = np.empty_like(t3)
            weight = self._weight.reshape(-1)
            for target, (s, perm) in self._sources:
                # Times 1 / w: within an ulp of the NumPy path's x / w.
                _cpu_quadruples.permuted_add(full[target], None, packed[s].transpose(perm), 0.0,
                                             1.0 / float(weight[s]))
            return (vector[:n1].reshape(t1.shape).copy(), vector[n1:n1 + n2].reshape(t2.shape).copy(),
                    full)
        packed = vector[n1 + n2:].reshape((-1,) + t3.shape[3:]) / self._weight
        full = self.xp.empty_like(t3)
        for perm in permutations(range(3)):
            # t3[P(ijk), P(abc)] = t3[ijk, abc] for every permutation P.
            full[tuple(self._keys[p] for p in perm)] = packed.transpose(0, *(1 + p for p in perm))
        return (vector[:n1].reshape(t1.shape).copy(), vector[n1:n1 + n2].reshape(t2.shape).copy(),
                full)

    def _dots(self, error, errors):
        """[e . error for e in errors] + [error . error]: fixed-block compiled
        dots on the CPU (OpenBLAS ddot rounds by its thread count)."""
        if self._compiled(error, *errors):
            return _cpu_quadruples.dots(error, errors)
        values = [self.xp.dot(e, error) for e in errors] + [self.xp.dot(error, error)]
        return [float(v) for v in values]

    # T4 is iterated outside this history, so the T1–T3 errors of earlier
    # evaluations belong to other T4 and can go stale. An error this many times
    # larger than the smallest stored one (in norm) means the problem moved
    # under the history, for example converged CCSDT amplitudes with T4 = 0
    # followed by the first T4: their near-zero error would pin the
    # extrapolation to the stale amplitudes until it leaves the history.
    STALE_RATIO = 100.0

    def update(self, amplitudes, steps):
        if self.size < 2:
            return amplitudes
        vector, error = self.pack(*amplitudes), self.pack(*steps)
        if self.history:
            norm = float(self._dots(error, [])[-1])
            if norm > self.STALE_RATIO**2 * float(np.min(np.diag(self.gram))):
                self.history, self.gram = [], np.empty((0, 0))
        weights = self._append(vector, error, self.xp.dot, self._dots)
        if weights is None:
            return amplitudes
        return self.unpack(self.mix(weights, [previous for previous, _ in self.history]), amplitudes)

    def mix(self, weights, vectors):
        """sum_j weights[j] vectors[j] (on all threads with the compiled core)."""
        if self.xp is np and _cpu_quadruples.available():
            return _cpu_quadruples.combine(weights, vectors, np.empty_like(vectors[0]))
        mixed = self.xp.zeros_like(vectors[0])
        for weight, vector in zip(weights, vectors):
            mixed += weight * vector
        return mixed


class _HostQuadruplesDIIS(_DIIS):
    """DIIS over T1–T4 whose T4 history lives in host memory.

    Damped Jacobi on T4 alone diverges when the T4 equations have a mode that
    the orbital-energy denominators misjudge (for example the collective
    excitations into a manifold of diffuse virtuals); DIIS over all four ranks
    extrapolates such modes away, as restarted GMRES would on a linear system,
    and keeps every error vector consistent with the T4 it belongs to.

    T1–T3 are packed as in `_PackedTriplesDIIS` and stay on the device. The T4
    trial vectors and errors (packed layout vectors) are host arrays: inner
    products and the extrapolation run on the host chunk by chunk, and only
    the extrapolated T4 goes back to the device. The device therefore holds
    no more than with T1–T3 DIIS; the history costs 2 * size packed T4 of
    host memory. History entries are ((T1–T3 vector, T4 vector), (T1–T3
    error, T4 error)). With NumPy (backend="cpu") the "device" is host memory
    as well and the extrapolation is written straight into T4.
    """
    # Host passes without the compiled core go in cache-sized chunks: the new
    # error is read once against every stored error, and the extrapolation
    # accumulates in place (BLAS axpy).
    CHUNK = 2**21           # doubles (16 MiB)

    def __init__(self, size, cp, nocc):
        super().__init__(size, cp)
        self._lower = _PackedTriplesDIIS(size, cp, nocc)
        self._mixed = None      # reused host buffer for the extrapolated T4
        # NumPy backend: the T4 arrays of the entry dropped for a new one are
        # reused (its trial T4 for the next copy, its error through
        # `recycle_error` for the next R4) instead of allocating, and first
        # touching, two packed T4 every evaluation.
        self._spare = []
        self.recycle_error = None

    def reserve_slot(self):
        if self.size >= 2 and len(self.history) == self.size:
            vector, error = self.history[0]
            super().reserve_slot()
            # Only arrays this solve owns (not read-only, e.g. mapped from a saved state).
            reusable = [a for a in (vector[1], error[1]) if isinstance(a, np.ndarray)
                        and a.flags.writeable and a.flags.c_contiguous and a.flags.owndata]
            if any(a is vector[1] for a in reusable) and (self.xp is not np
                                                          or _cpu_quadruples.available()):
                self._spare[:] = [vector[1]]       # take_step's copy of the trial T4 reuses it
            if (self.xp is np and self.recycle_error is not None
                    and any(a is error[1] for a in reusable)):
                self.recycle_error(error[1])

    def _t4_dots(self, error):
        """Inner products of a host T4 error with every stored T4 error and itself."""
        others = [previous[1] for _, previous in self.history]
        if _cpu_quadruples.available() and error.flags.c_contiguous and all(
                o.flags.c_contiguous for o in others):
            return _cpu_quadruples.dots(error, others)       # one pass on all threads
        dots = np.zeros(len(others) + 1)
        for s in range(0, error.size, self.CHUNK):
            part = error[s:s + self.CHUNK]
            for index, other in enumerate(others):
                dots[index] += np.dot(part, other[s:s + self.CHUNK])
            dots[-1] += np.dot(part, part)
        return dots

    def take_step(self, lower, lower_steps, t4, t4_step):
        """Record one step and return the next amplitudes (T1, T2, T3, T4).

        `lower`/`lower_steps` are T1–T3 and their (damped, preconditioned)
        steps on the device; `t4` is the device T4 with its step already added
        (the trial T4) and `t4_step` that step as a host array, which the
        history keeps.
        """
        from scipy.linalg.blas import daxpy

        cp = self.xp
        compiled = cp is np and _cpu_quadruples.available() and t4.flags.c_contiguous
        spare = self._spare.pop() if self._spare and self._spare[-1].shape == t4.shape else None
        if compiled and spare is not None:
            trial = _cpu_quadruples.permuted_add(spare, None, t4, 0.0, 1.0)
        elif cp is not np and spare is not None and t4.flags.c_contiguous:
            trial = t4.get(out=spare)       # no fresh (first-touched) host T4 per step
        else:
            trial = _cpu_quadruples.copy(t4) if compiled else t4.copy() if cp is np else cp.asnumpy(t4)
        vector = (self._lower.pack(*(a + s for a, s in zip(lower, lower_steps))), trial)
        error = (self._lower.pack(*lower_steps), t4_step)
        # The solve reserves a slot before allocating the new R4 and trial T4.
        t4_dots = iter(self._t4_dots(t4_step))
        lower_dots = {id(previous): float(cp.dot(previous[0], error[0])) for _, previous in self.history}
        lower_dots[id(error)] = float(cp.dot(error[0], error[0]))

        def dot(previous, current):
            return cp.asarray(lower_dots[id(previous)] + next(t4_dots))
        weights = self._append(vector, error, dot)
        if weights is None:
            return (*self._lower.unpack(vector[0], lower), t4)
        mixed = self._lower.mix(weights, [previous[0] for previous, _ in self.history])
        if cp is np:
            out = t4            # the history holds a copy of the trial T4
        else:
            if self._mixed is None or self._mixed.size != t4.size:
                self._mixed = np.empty(t4.size)
            out = self._mixed
        stored = [previous[1] for previous, _ in self.history]
        if (_cpu_quadruples.available() and (compiled or cp is not np)
                and all(isinstance(a, np.ndarray) and a.dtype == np.float64 and a.flags.c_contiguous
                        for a in stored)):
            # The host history on all threads (also for a device T4).
            _cpu_quadruples.combine(weights, stored, out)
            if out is not t4:
                t4.set(out)
            return (*self._lower.unpack(mixed, lower), t4)
        for s in range(0, t4.size, self.CHUNK):
            part = out[s:s + self.CHUNK]
            np.multiply(self.history[0][0][1][s:s + self.CHUNK], weights[0], out=part)
            for weight, (previous, _) in zip(weights[1:], self.history[1:]):
                daxpy(previous[1][s:s + self.CHUNK], part, a=weight)
        if out is not t4:
            t4.set(out)
        return (*self._lower.unpack(mixed, lower), t4)

    def restore(self, history, gram, base=None):
        """A saved history: float64 entries as they are; entries of the compact
        history (float32, trial T4 minus the T4 it was saved with) as trial T4
        again, from `base`, the resumed T4."""
        cp = self.xp
        base_host = None
        restored = []
        for v, e in history:
            trial = v[1]
            if np.dtype(trial.dtype) == np.float32:
                if base is None:
                    raise ValueError("A compact T4 history needs the resumed T4 to convert")
                if base_host is None:
                    base_host = base if isinstance(base, np.ndarray) else cp.asnumpy(base)
                    base_host = base_host.reshape(-1)
                trial = np.empty(base_host.size)
                for s in range(0, base_host.size, self.CHUNK):
                    np.add(base_host[s:s + self.CHUNK], v[1][s:s + self.CHUNK], out=trial[s:s + self.CHUNK])
            restored.append(((cp.asarray(v[0]), np.asarray(trial, dtype=np.float64)),
                             (cp.asarray(e[0]), np.asarray(e[1], dtype=np.float64))))
        self.history = restored
        self.gram = np.array(gram, dtype=np.float64).reshape(len(self.history), len(self.history))


class _CompactHostDIIS(_HostQuadruplesDIIS):
    """`_HostQuadruplesDIIS` of the compiled CPU backend with a float32 T4 history.

    Each entry keeps two float32 packed T4: its error (used only in inner
    products, against the float64 newest error) and its trial minus the
    current amplitudes. A step extrapolates in that difference form and
    re-bases the differences on the new amplitudes in one pass over the
    history (`itercc_diis_rebase`); the newest entry enters with its exact
    float64 step. The rounding is therefore relative to the differences
    between iterates, which vanish as the iteration converges, and a plain
    Jacobi step is exact, so the fixed point and the convergence are those of
    the float64 history. An entry costs one packed T4 instead of two, and the
    float64 step goes back to the solver as the next R4: about 2 + size packed
    T4 in all (T4, R4, the history and one spare entry), against 3 + 2 * size.
    `ITERCC_DIIS_COMPACT=0` selects the float64 history.
    """

    def __init__(self, size, cp, nocc):
        super().__init__(size, cp, nocc)
        self._spare32 = []

    def reserve_slot(self):
        if self.size >= 2 and len(self.history) == self.size:
            vector, error = self.history[0]
            _DIIS.reserve_slot(self)
            for part in (vector[1], error[1]):
                if part.dtype == np.float32 and part.flags.writeable and part.flags.owndata:
                    self._spare32.append(part)

    def _float32(self, n):
        while self._spare32:
            spare = self._spare32.pop()
            if spare.size == n:
                return spare
        return np.empty(n, dtype=np.float32)

    def take_step(self, lower, lower_steps, t4, t4_step):
        library = _cpu_kernels.library()
        threads = _cpu_quadruples._threads(None)
        self.reserve_slot()
        n = t4.size
        deltas = [vector[1] for vector, _ in self.history]
        errors = [error[1] for _, error in self.history]
        delta_new, error_new = self._float32(n), self._float32(n)
        t4_dots = np.zeros(len(errors) + 1)
        library.dots_f32(n, len(errors), t4_step.ctypes.data, _pointers(errors), t4_dots.ctypes.data,
                         threads)
        cp = self.xp
        vector = (self._lower.pack(*(a + s for a, s in zip(lower, lower_steps))), delta_new)
        error = (self._lower.pack(*lower_steps), error_new)
        if cp is np and _cpu_quadruples.available():
            # The core's dots, not NumPy's BLAS: OpenBLAS threads left spinning
            # after a call stole the cores of every following kernel (about
            # 50 ms per iteration on 12 threads), and these sums do not depend
            # on the thread count.
            lower_dots = list(_cpu_quadruples.dots(error[0], [previous[0] for _, previous in self.history]))
        else:
            lower_dots = [float(cp.dot(previous[0], error[0])) for _, previous in self.history]
            lower_dots.append(float(cp.dot(error[0], error[0])))
        dots = iter(lower_dots[i] + t4_dots[i] for i in range(len(lower_dots)))
        weights = self._append(vector, error, lambda previous, current: next(dots))
        w = None if weights is None else np.ascontiguousarray(weights, dtype=np.float64)
        library.diis_rebase(n, len(deltas), None if w is None else w.ctypes.data, _pointers(deltas),
                            delta_new.ctypes.data, error_new.ctypes.data, t4.ctypes.data,
                            t4_step.ctypes.data, threads)
        if (self.recycle_error is not None and t4_step.flags.writeable and t4_step.flags.owndata
                and t4_step.flags.c_contiguous):
            self.recycle_error(t4_step)
        if weights is None:
            return (*self._lower.unpack(vector[0], lower), t4)
        mixed = self._lower.mix(weights, [previous[0] for previous, _ in self.history])
        return (*self._lower.unpack(mixed, lower), t4)

    def restore(self, history, gram, base=None):
        """A saved history: compact entries as they are, float64 ones (trial T4
        and error) converted to differences from `base`, the resumed T4."""
        restored = []
        for v, e in history:
            if np.dtype(v[1].dtype) == np.float32:
                delta = np.array(v[1], dtype=np.float32)
            else:
                if base is None:
                    raise ValueError("A float64 T4 history needs the resumed T4 to convert")
                trial, delta = np.asarray(v[1]), np.empty(base.size, dtype=np.float32)
                for s in range(0, base.size, self.CHUNK):
                    delta[s:s + self.CHUNK] = trial[s:s + self.CHUNK] - base[s:s + self.CHUNK]
            restored.append(((np.asarray(v[0], dtype=np.float64), delta),
                             (np.asarray(e[0], dtype=np.float64), np.array(e[1], dtype=np.float32))))
        self.history = restored
        self.gram = np.array(gram, dtype=np.float64).reshape(len(restored), len(restored))


def _pointers(arrays):
    """A ctypes array of the data pointers of `arrays` (at least one slot)."""
    pointers = (ctypes.c_void_p * max(1, len(arrays)))(*[a.ctypes.data for a in arrays])
    return ctypes.cast(pointers, ctypes.c_void_p)


@dataclass
class CCSDTQState:
    """Everything needed to continue `RestrictedCCSDTQ.solve` exactly.

    `amplitudes` are the next iterate in the solver's internal format: T1–T3
    in the internal orbital order and T4 in `layout` form. `evaluations`
    counts the residual evaluations already done; `previous_correlation` is
    the correlation energy of the last one.
    """
    evaluations: int
    amplitudes: tuple
    previous_correlation: float
    iterations: tuple
    diis_history: list
    diis_gram: np.ndarray
    settings: dict

    def history_bytes(self):
        return sum(part.nbytes for pair in self.diis_history for value in pair
                   for part in (value if isinstance(value, tuple) else (value,)))

    def without_history(self):
        """The same state with an empty DIIS history: resuming it continues the
        amplitudes and receipts and rebuilds the DIIS subspace (for a T4
        history too large to write with every checkpoint)."""
        return replace(self, diis_history=[], diis_gram=np.empty((0, 0)))


class RestrictedCCSDTQ:
    """Real closed-shell CCSDTQ for chemist-order integrals in the MO basis.

    T1 and T2 use IterCC's restricted ordering and T3 is fully expanded. T4 is
    a flat array in the form of `self.layout`: occupied tuples i <= j <= k <= l
    in lexicographic order, each followed by its virtual quadruples. With
    `orbital_irreps` (one Abelian label 0..7 per orbital, products by XOR)
    only symmetry-allowed T4 and R4 entries are stored; the integrals are
    audited with `symmetry_tolerance`, virtual orbitals are ordered by irrep
    internally, and T1–T3 are returned in the caller's orbital order. Without
    labels T4 is the dense packed array (slots, nv, nv, nv, nv) flattened.
    Returned amplitudes are float64 arrays of the backend (CuPy or NumPy). With `host_r4=True`, the
    rank-four residual is a NumPy array.

    `backend="cpu"` runs the same equations and layout with NumPy arrays
    (amplitudes and residuals are NumPy arrays; CuPy is not imported);
    `host_r4` is a CUDA option and is rejected there.

    `block_size` is the occupied edge of the R4 tiles the equations work on.
    It defaults to 4 on CUDA and 1 on the CPU, where a one-slot tile (nv^4
    entries) stays in the last-level cache through the many terms added to it.

    `precision="single"` (CPU backend) lets the memory-bound R4 passes read
    float32 copies of their largest operands: T3 and its spin sum in the tile
    programs, and the spin-summed T4 rows of the T4-into-lower-ranks pass. The
    GEMMs widen the values to double as they pack them, so every product and
    sum is formed in double and T1-T4 stay double; the residual moves by the
    float32 rounding of those operands (about 1e-7 relative to their terms).
    """

    def __init__(self, h, eri, nocc, constant=0.0, *, block_size=None, host_r4=False,
                 orbital_irreps=None, symmetry_tolerance=1e-9, backend="cuda", precision="double"):
        from . import _quadruples_equations

        if precision not in ("double", "single"):
            raise ValueError("precision must be 'double' or 'single'")
        if precision == "single" and backend != "cpu":
            raise ValueError("precision='single' is a mode of the CPU backend")
        self.precision = precision

        if backend == "cuda":
            import cupy as xp

            from ._quadruples_cuda import QuadruplesLayout, project_triples
        elif backend == "cpu":
            xp = np
            from ._quadruples_cpu import NumpyQuadruplesLayout as QuadruplesLayout, project_triples

            if _cpu_quadruples.available():     # the compiled CPU kernels
                QuadruplesLayout = _cpu_quadruples.CompiledQuadruplesLayout
                project_triples = _cpu_quadruples.project_triples

            if host_r4:
                raise ValueError("host_r4 stages R4 of the CUDA backend in host memory; "
                                 "backend='cpu' keeps every array there")
        else:
            raise ValueError("backend must be 'cuda' or 'cpu'")
        if block_size is None:
            block_size = 4 if backend == "cuda" else 1
        if not isinstance(block_size, Integral) or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        h, eri = _real64(h), _real64(eri)
        n = len(h)
        if not isinstance(nocc, Integral) or not 0 < nocc < n:
            raise ValueError("Expected nonempty occupied and virtual spatial spaces")
        labels = np.zeros(n, dtype=np.int64)
        self.integral_symmetry_leakage = None
        self._tolerance = float(symmetry_tolerance)
        if orbital_irreps is not None:
            labels = orbital_labels(orbital_irreps, n).astype(np.int64)
            h, eri, self.integral_symmetry_leakage = project_integrals(
                h, eri, labels, tolerance=symmetry_tolerance)
        self.orbital_irreps = None if orbital_irreps is None else labels
        order = np.r_[np.arange(nocc), nocc + np.argsort(labels[nocc:], kind="stable")]
        self._virtual_order = self._virtual_inverse = None
        if np.any(order != np.arange(n)):
            self._virtual_order = order[nocc:] - nocc
            self._virtual_inverse = np.argsort(self._virtual_order)
            h = np.ascontiguousarray(h[np.ix_(order, order)])
            eri = np.ascontiguousarray(eri[np.ix_(order, order, order, order)])
        self._reference = RestrictedCC(h, eri, nocc, constant)
        self._constant = float(constant)
        self.backend = backend
        self.xp = xp
        self._host_r4 = host_r4
        self._equations = _quadruples_equations
        self._project_triples = project_triples
        self.nocc = self._reference.o.stop
        self.nmo = self._reference.h.shape[0]
        self.nvir = self.nmo - self.nocc
        self.blksize = min(int(block_size), self.nocc)
        internal = labels[order]
        self.layout = QuadruplesLayout(self.nocc, self.nvir, internal[:nocc], internal[nocc:])
        self.shapes = self._reference.shapes[:2] + (
            (self.nocc,) * 3 + (self.nvir,) * 3, self.layout.shape)
        self.reference_energy = self._reference.reference_energy

        # Symmetry-blocked GEMMs for the R4 tile terms: cuBLAS 12.5 grouped GEMM
        # on the GPU (with orbital labels), the compiled executor on the CPU
        # (one irrep without labels). Plain NumPy contracts the dense blocks.
        self._tile_blocks = self._block_blocks = self._tile_program = None
        engine = None
        if backend == "cuda" and self.layout.symmetric:
            from . import symmetry_blocks
            from ._quadruples_cuda import einsum as dense_einsum

            if symmetry_blocks.available():
                def engine(labels, combinations, nirrep):
                    return symmetry_blocks.SymmetryBlocks(labels, combinations, nirrep,
                                                          fallback=dense_einsum)
        elif backend == "cpu" and _cpu_quadruples.available():
            engine = _cpu_quadruples.blocked_einsum
        if engine is not None:
            labels = dict(m=internal[:nocc], n=internal[:nocc],
                          **{c: internal[nocc:] for c in "abcdef"})
            self._tile_blocks = engine(labels, "ijkl", self.layout.nirrep)
            occupied = {c: internal[:nocc] for c in "ijklmnx"}
            self._block_blocks = engine(dict(labels, **occupied), "", self.layout.nirrep)
        if backend == "cpu" and self._tile_blocks is not None and self.blksize == 1:
            # One-slot tiles: the R4 tile loops run slot-parallel in the core.
            from ._cpu_tile_program import TileProgram

            self._tile_program = TileProgram(self)

        self._dressing = copy(self._reference)
        self._dressing.h = xp.asarray(self._reference.h)
        self._dressing.g = xp.asarray(self._reference.g, order="C")
        self._dressing._xp = xp

    # Conversion between the caller's arrays and the internal form.

    def _to_backend(self, source):
        """A new float64 array of the backend from a host, device, mapped or saved array."""
        if self.xp is np:
            copy = _host_copy(source)
            if self._compiled_cpu() and copy.size >= 2**20 and copy.flags.c_contiguous:
                # First touch on all threads, so the pages of large amplitudes are
                # spread over the sockets' memory (a NumPy copy puts them all on
                # the calling thread's socket for the whole solve).
                return _cpu_quadruples.permuted_add(np.empty(copy.shape), None, copy, 0.0, 1.0)
            return copy
        return to_device(source if source.dtype == np.float64 else np.asarray(source, dtype=np.float64))

    def _host(self, array):
        """A NumPy view or copy of a backend array."""
        return array if self.xp is np else self.xp.asnumpy(array)

    def _reorder(self, arrays, *, inverse=False):
        """Move T1–T3 (or R1–R3) between caller and internal virtual order."""
        if self._virtual_order is None:
            return tuple(arrays)
        xp = self.xp
        index = xp.asarray(self._virtual_inverse if inverse else self._virtual_order)
        result = []
        for rank, array in enumerate(arrays, 1):
            for axis in range(rank, 2 * rank):
                array = xp.take(array, index, axis=axis)
            result.append(array)
        return tuple(result)

    def _quadruples(self, t4, copy_arrays):
        xp = self.xp
        layout = self.layout
        if t4.dtype.kind not in "iuf":
            raise ValueError("Expected a real T4 array")
        if t4.shape == layout.shape:
            if isinstance(t4, xp.ndarray):
                return (xp.array(t4, dtype=xp.float64, copy=True) if copy_arrays
                        else xp.ascontiguousarray(t4, dtype=xp.float64))
            return self._to_backend(t4)
        if t4.shape == layout.dense_shape:
            if isinstance(t4, xp.ndarray) and not layout.symmetric and self._virtual_order is None:
                flat = t4.reshape(-1)
                return (xp.array(flat, dtype=xp.float64, copy=True) if copy_arrays
                        else xp.ascontiguousarray(flat, dtype=xp.float64))
            out, leakage = layout.from_dense(t4, order=self._virtual_order)
            if not isfinite(leakage) or leakage > self._tolerance:
                raise ValueError(f"T4 symmetry leakage {leakage:.6g} exceeds {self._tolerance:.6g}")
            return out
        raise ValueError(f"Expected T4 with shape {layout.shape} or {layout.dense_shape}")

    def _as_amplitudes(self, arrays, *, copy_arrays=False):
        xp = self.xp
        lower = []
        for a, shape in zip(arrays[:3], self.shapes[:3]):
            a = xp.asarray(a)
            if a.shape != shape or a.dtype.kind not in "iuf":
                raise ValueError(f"Expected a real amplitude array with shape {shape}")
            lower.append(xp.array(a, dtype=xp.float64, order="C", copy=True) if copy_arrays
                         else xp.ascontiguousarray(a, dtype=xp.float64))
        lower = self._reorder(lower)
        if len(arrays) == 3:
            return lower
        return (*lower, self._quadruples(arrays[3], copy_arrays))

    def caller_amplitudes(self, amplitudes):
        """T1–T3 in the caller's orbital order; T4 stays in `layout` form."""
        return (*self._reorder(amplitudes[:3], inverse=True), *amplitudes[3:])

    def expand_triples(self, packed):
        """Expand a T3 array from an IterCC restricted CCSDT checkpoint."""
        layout = self._reference.layout
        packed = np.asarray(packed, dtype=np.float64)
        if packed.shape != layout.shape:
            raise ValueError(f"Expected packed triples with shape {layout.shape}")
        expanded = np.empty(self.shapes[2], dtype=np.float64)
        for k in range(self.nocc):
            expanded[:, :, k] = layout.plane(packed, k, axis=2)
        return expanded

    def quadruples_to_dense(self, t4):
        """Dense packed (slots, nv, nv, nv, nv) T4 in the caller's virtual order."""
        return self.layout.to_dense(t4, order=self._virtual_order)

    def source_layout(self, signature):
        """Layout and internal virtual order of quadruples saved with `layout_signature`.

        `order[k]` is the caller's virtual index at internal position k. A
        signature of None (or without labels) means the dense packed layout.
        """
        QuadruplesLayout = type(self.layout)     # the layout class of this backend

        labels = None if signature is None else signature.get("orbital_irreps")
        if signature is not None and (signature["nocc"], signature["nvir"]) != (self.nocc, self.nvir):
            raise ValueError("The saved state belongs to another orbital space")
        if labels is None:
            return QuadruplesLayout(self.nocc, self.nvir), np.arange(self.nvir)
        labels = orbital_labels(labels, self.nmo).astype(np.int64)
        order = np.argsort(labels[self.nocc:], kind="stable")
        return QuadruplesLayout(self.nocc, self.nvir, labels[:self.nocc],
                                labels[self.nocc:][order]), order

    def combine_states(self, states):
        """Initial amplitudes sum_k w_k T_k from states solved for related Hamiltonians.

        `states` holds (weight, amplitudes, signature) with T1–T3 in the
        caller's orbital order and T4 packed as described by `signature`
        (from `layout_signature`; None or a dense packed T4 means the dense
        layout). Sources may use other symmetry labels, for example zero-field
        C2v states for a transverse-field Cs solve; entries this solver's
        labels forbid are dropped. Arrays may live on the device, in host
        memory or on disk. Returns caller-order T1–T3 and T4 in this solver's
        layout as device arrays, ready for `solve(initial=...)`.
        """
        xp = self.xp
        convert = np.asarray if xp is np else to_device
        lower = [xp.zeros(shape) for shape in self.shapes[:3]]
        t4 = self.layout.zeros()
        mine = np.arange(self.nvir) if self._virtual_order is None else self._virtual_order
        for weight, amplitudes, signature in states:
            for total, part in zip(lower, amplitudes[:3]):
                total += weight * convert(part)
            source = amplitudes[3]
            if signature is None or tuple(source.shape) == self.layout.dense_shape:
                layout, order = self.source_layout(None)
            else:
                layout, order = self.source_layout(signature)
            perm = np.argsort(order)[mine]
            self.layout.accumulate_from(t4, float(weight), source, layout,
                                        None if np.array_equal(perm, np.arange(self.nvir)) else perm)
        return (*lower, t4)

    def residuals(self, amplitudes):
        """Projected raw residuals: R1–R3 in caller order, R4 in `layout` form."""
        if len(amplitudes) != 4:
            raise ValueError("Expected T1, T2, T3, and T4")
        return self.caller_amplitudes(self._residuals(self._as_amplitudes(amplitudes)))

    @property
    def host_r4(self):
        return self._host_r4

    @host_r4.setter
    def host_r4(self, value):
        if value and self.backend != "cuda":
            raise ValueError("host_r4 is an option of the CUDA backend")
        self._host_r4 = bool(value)

    def device_bytes(self, *, host_r4=None, diis_size=6, diis_quadruples=False):
        """Estimate the peak device bytes of `solve` (amplitudes, residuals,
        intermediates, tile buffers and DIIS history); about 10% is added for
        allocator fragmentation."""
        from ._quadruples_cuda import _slot_batch

        no, nv, b = self.nocc, self.nvir, self.blksize
        host_r4 = self._host_r4 if host_r4 is None else host_r4
        t4 = self.layout.size
        t3 = no**3 * nv**3
        nv4 = nv**4
        # Always resident: reference and T1-dressed integrals, the small
        # intermediates (W_vvvv, W_vvvo, ...) and the symmetry engine's tables.
        fixed = 2 * (no + nv)**4 + 5 * nv4 + 2 * no * nv**3 + 8 * no**2 * nv**2
        # From the T4 pass through the R4 terms: T3, R3 and two more T3-sized
        # arrays, W_oovvvo, W_ovovvo, W_vvvvoo and W_vvoooo.
        lower = 6 * t3 + nv4 * no**2 + nv**2 * no**4
        if self._tile_program is not None and self._tile_program.blocked_enabled():
            # Pair-blocked copies of T3 and its spin sum (float32 when single).
            lower += t3 if self.precision == "single" else 2 * t3
        # R4 tile loops: the output tile, two operand tiles and one sorted or
        # reshaped operand copy.
        ladder = 4 * b**4 * nv4
        # T4 pass (t4_into_lower_ranks_): the spin-summed block and the
        # symmetry-allowed part of it (a fraction `allowed`) gathered for a GEMM.
        chunk = max(1, min(no, (2 * 2**30) // (8 * no * nv4)))
        allowed = t4 / max(1, self.layout.dense_shape[0] * nv4)
        fused = (1 + allowed) * chunk * no * nv4
        if host_r4:     # tile loops: (nocc, b, b, b) blocks with einsum copies
            ring = pairs = 2 * no * b**3 * nv4 + b**4 * nv4
        else:           # spectator triples and pairs, see _quadruples_equations
            ring = 5 * no * nv4
            pairs = (no**2 + no) * nv4
        # layout.project of R4: a dense batch and three copies of a slot batch.
        slots = len(self.layout.indices)
        project = (min(slots, self.layout.transfer_slots) + 3 * min(slots, _slot_batch(nv4))) * nv4
        if diis_quadruples is True and diis_size >= 2:
            history = 2 * diis_size * (t4 + t3 + no**2 * nv**2 + no * nv)
        elif diis_size >= 2:     # T1–T3 on the device (also with a host T4 history)
            packed = (no + 2) * (no + 1) * no // 6 * nv**3
            history = 2 * diis_size * (packed + no**2 * nv**2 + no * nv)
        else:
            history = 0
        words = (t4 * (1 if host_r4 else 2) + fixed + history
                 + max(lower + max(ladder, fused, ring, pairs), 4 * t3 + project))
        return int(1.1 * 8 * words)

    def new_quadruples_residual(self, t4, *, zeroed=True):
        """A packed R4: a recycled one when the host DIIS has one to spare,
        zeroed on all threads unless `zeroed=False` (the caller overwrites
        every entry first)."""
        spares = getattr(self, "_r4_spares", None)
        if spares:
            r4 = spares.pop()
            if zeroed:
                self._scale_(r4, 0.0)
            return r4
        return self.layout.zeros(host=self._host_r4)

    def _residuals(self, amplitudes):
        t1, t2, t3, t4 = amplitudes
        eq = self._equations
        fock, eri = self._dressing._dressed(t1)
        imds = SimpleNamespace(t1_fock=fock, t1_eris=eri)
        eq.intermediates_t1t2(self, imds, t2)
        r1, r2 = eq.compute_r1r2(self, imds, t2)
        eq.r1r2_add_t3_(self, imds, r1, r2, t3)

        eq.intermediates_t3(self, imds, t2)
        eq.intermediates_t3_add_t3(self, imds, t3)
        r3 = eq.compute_r3(self, imds, t2, t3)
        eq.intermediates_t4_tri(self, imds, t2, t3)
        # One pass over T4 feeds r2, r3 and the T4-equation intermediates.
        eq.t4_into_lower_ranks_(self, imds, r2, r3, t4)
        self._release_workspaces()
        r2 += r2.transpose(1, 0, 3, 2)
        self._project_triples(r3)
        eq.finish_intermediates_t4(imds)
        imds.t1_fock = imds.t1_eris = None
        r4 = eq.compute_r4_tri(self, imds, t2, t3, t4)
        self._release_workspaces()
        self.layout.project(r4)
        return r1, r2, r3, r4

    def _release_workspaces(self):
        # The symmetry engines keep their largest operand workspace between
        # calls; hand it back between phases with different big temporaries.
        for engine in (self._tile_blocks, self._block_blocks):
            if engine is not None:
                engine.release_workspace()

    def _eia(self, level_shift):
        diag = self._reference.fock.diagonal()
        eia = diag[:self.nocc, None] - diag[None, self.nocc:] - level_shift
        if np.any(np.abs(eia) < 1e-14):
            raise ValueError("Near-zero singles denominator")
        return self.xp.asarray(eia)

    def _compiled_cpu(self):
        return self.backend == "cpu" and _cpu_quadruples.available()

    def _scale_(self, x, factor):
        if self._compiled_cpu() and isinstance(x, np.ndarray):
            _cpu_quadruples.scale_(x, factor)
        else:
            x *= factor

    def _add_(self, y, x):
        if self._compiled_cpu() and isinstance(y, np.ndarray):
            _cpu_quadruples.add_(y, x)
        else:
            y += x

    def _precondition(self, residuals, eia, d2, d3=None):
        r1, r2, r3, r4 = residuals
        r1 /= eia
        r2 /= d2
        # r3 /= d3 with the T3 denominators formed on the fly (no T3-sized
        # array kept for the whole solve), with the same rounding.
        if d3 is not None:
            r3 /= d3
        elif self.xp is np:
            _cpu_quadruples.divide_triples(r3, eia)
        else:
            from ._quadruples_cuda import divide_triples

            divide_triples(r3, eia)
        if not self._host_r4:
            self.layout.divide(r4, eia)

    def _initial(self, initial, eia, d2, copy=True):
        xp = self.xp
        if initial is None:
            ref = self._reference
            return (xp.asarray(ref.fock[ref.o, ref.v]) / eia,
                    xp.asarray(ref.g[ref.o, ref.o, ref.v, ref.v]) / d2,
                    xp.zeros(self.shapes[2]), self.layout.zeros())
        if len(initial) not in (3, 4):
            raise ValueError("Expected T1, T2, T3, and optionally T4")
        amps = self._as_amplitudes(initial, copy_arrays=copy)
        if len(amps) == 3:
            amps += (self.layout.zeros(),)
        return amps

    @staticmethod
    def _denominators(eia):
        """(d2, None): the T3 denominators are formed as `_precondition` divides."""
        return eia[:, None, :, None] + eia[None, :, None, :], None

    def _maxima(self, residuals):
        if self._compiled_cpu():
            return (*(_cpu_quadruples.max_abs(r) for r in residuals[:3]),
                    self.layout.max_abs(residuals[3]))
        return (*(max(float(r.max()), -float(r.min())) for r in residuals[:3]),
                self.layout.max_abs(residuals[3]))

    def solve(self, *, initial=None, max_iter=100, energy_tol=1e-11,
              residual_tol=1e-9, level_shift=0.0, damping=0.0, diis_size=6,
              diis_quadruples="auto", callback=None, checkpoint=None, resume=None,
              copy_initial=True):
        """Iterate T1–T4 with damped Jacobi updates and DIIS.

        Pass three starting arrays to initialize T4 to zero, or four to start
        from given amplitudes (T4 in `layout` form or dense packed form).
        `max_iter` bounds the residual evaluations of this call. The callback
        takes (evaluation_number, Iteration) after each residual evaluation;
        returning True stops the iteration with the evaluated amplitudes.
        `checkpoint(state)` receives a CCSDTQState after every update, before
        the next evaluation; `save_state` writes it, and passing the state
        read by `load_state` as `resume` continues the iteration exactly,
        including the DIIS history (on CUDA, in a new process too: the state
        carries the autotuned GEMM routes, reused on the same GPU model). The last residual and energy both refer
        to the returned amplitudes. DIIS always includes T1–T3; whether it
        includes T4 is `diis_quadruples`. The default "auto" is "host" on the
        CPU backend (diffuse basis sets have a stiff T4 mode that damped Jacobi
        steps on T4 do not converge; the compact host history costs about
        (diis_size + 1) / 2 T4 of memory) and False on CUDA, where T4 then
        takes damped Jacobi steps. `diis_quadruples=True` includes T4 with the history on
        the device (a substantially larger pool); `diis_quadruples="host"`
        includes T4 with its history in host memory (2 * diis_size packed T4; the
        compact float32 history of the compiled CPU backend: diis_size + 1),
        which also works with host-staged R4. Include T4 whenever the T4
        Jacobi iteration has an unstable mode (a quadruples residual that
        grows while the energy settles). A resumed state without a DIIS
        history (see `CCSDTQState.without_history`) restarts the subspace.
        With `copy_initial=False`, device
        arrays in `initial` (for example from `combine_states`) are iterated in
        place instead of copied, which saves one T4 of device memory; the
        caller must not use them afterwards.
        """
        xp = self.xp
        if isinstance(diis_quadruples, str) and diis_quadruples == "auto":
            diis_quadruples = "host" if self.backend == "cpu" else False
        if (not isinstance(max_iter, Integral) or max_iter < 1
                or not isinstance(diis_size, Integral) or diis_size < 0
                or not np.isfinite((energy_tol, residual_tol, level_shift, damping)).all()
                or energy_tol <= 0 or residual_tol <= 0 or not 0 <= damping < 1
                or diis_quadruples not in (False, True, "host")):
            raise ValueError("Invalid CCSDTQ iteration settings")
        host_diis = diis_quadruples == "host" and diis_size >= 2
        if self._host_r4 and diis_quadruples is True and diis_size >= 2:
            raise ValueError('Host-staged R4 needs diis_quadruples="host" to include T4 in DIIS')
        settings = dict(level_shift=float(level_shift), damping=float(damping),
                        diis_size=int(diis_size),
                        diis_quadruples="host" if diis_quadruples == "host" else bool(diis_quadruples))
        eia = self._eia(level_shift)
        d2, d3 = self._denominators(eia)
        # An ordered occupied/virtual spectrum guarantees nonzero higher-rank
        # denominators once singles denominators have been checked.
        if float(eia.max()) > 0.0 and float(eia.min()) < 0.0:
            raise ValueError("Occupied and virtual orbital denominators must have one sign")
        full_diis = diis_quadruples is True and diis_size >= 2
        compact = (host_diis and self._compiled_cpu()
                   and os.environ.get("ITERCC_DIIS_COMPACT", "1") != "0")
        diis = (_CompactHostDIIS(diis_size, xp, self.nocc) if compact
                else _HostQuadruplesDIIS(diis_size, xp, self.nocc) if host_diis
                else _FullRankDIIS(diis_size, xp) if full_diis
                else _PackedTriplesDIIS(diis_size, xp, self.nocc))
        self._r4_spares = []
        if host_diis and self._compiled_cpu():
            diis.recycle_error = self._r4_spares.append
        if resume is None:
            amplitudes = self._initial(initial, eia, d2, copy_initial)
            previous, history, done = None, [], 0
        else:
            if initial is not None:
                raise ValueError("Pass either initial amplitudes or a state to resume")
            if resume.settings != settings:
                raise ValueError(f"The state used {resume.settings}, not {settings}")
            amplitudes = tuple(self._to_backend(a) for a in resume.amplitudes[:3])
            amplitudes += (self._quadruples(resume.amplitudes[3], True),)
            previous, history, done = resume.previous_correlation, list(resume.iterations), resume.evaluations
            if host_diis:
                diis.restore(resume.diis_history, resume.diis_gram, base=amplitudes[3])
            else:
                diis.history = [(_to_device(self._to_backend, vector), _to_device(self._to_backend, error))
                                for vector, error in resume.diis_history]
                diis.gram = np.array(resume.diis_gram, dtype=np.float64).reshape(
                    len(diis.history), len(diis.history))
        converged = False
        for count in range(1, max_iter + 1):
            evaluation = done + count
            if host_diis:
                diis.reserve_slot()
            corr = self._reference._energy(self._host(amplitudes[0]), self._host(amplitudes[1]))
            residuals = self._residuals(amplitudes)
            maxima = self._maxima(residuals)
            if not np.isfinite((corr, *maxima)).all():
                raise FloatingPointError("Nonfinite CCSDTQ energy or residual")
            delta = float("inf") if previous is None else abs(corr - previous)
            info = Iteration(self.reference_energy + corr, delta, maxima)
            history.append(info)
            stop = callback is not None and callback(evaluation, info) is True
            converged = delta < energy_tol and max(maxima) < residual_tol
            if converged or stop or count == max_iter:
                break
            self._precondition(residuals, eia, d2, d3)
            updates = residuals
            if self._host_r4:
                # With host DIIS the host residual is overwritten by the T4 step it records.
                self.layout.update_from_host(amplitudes[3], residuals[3], eia, 1.0 - damping,
                                             keep_step=host_diis)
                updates = residuals[:3]
            if damping:
                for update in updates:
                    self._scale_(update, 1.0 - damping)
                del update
            if host_diis:
                t4, t4_step = amplitudes[3], residuals[3]
                if not self._host_r4:
                    self._add_(t4, t4_step)         # the trial T4
                    t4_step = self._host(t4_step)   # a new R4 every evaluation
                    residuals, updates = residuals[:3], None
                amplitudes = diis.take_step(amplitudes[:3], residuals[:3], t4, t4_step)
                del t4, t4_step
            elif full_diis:
                amplitudes = diis.take_step(amplitudes, residuals)
            else:
                for amplitude, update in zip(amplitudes, updates):
                    self._add_(amplitude, update)
                amplitudes = (*diis.update(amplitudes[:3], residuals[:3]), amplitudes[3])
                del update
            previous = corr
            del residuals, updates
            if checkpoint is not None:
                checkpoint(CCSDTQState(evaluation, amplitudes, previous, tuple(history),
                                       list(diis.history), diis.gram.copy(), settings))
        self._r4_spares = []
        return Result(self.reference_energy + corr, corr, self.caller_amplitudes(amplitudes),
                      converged, tuple(history))

    # Restartable solver state.

    def layout_signature(self):
        """Identify the T4 form and orbital order a saved state depends on."""
        return dict(nocc=self.nocc, nvir=self.nvir, size=self.layout.size,
                    orbital_irreps=None if self.orbital_irreps is None else self.orbital_irreps.tolist())

    def _checkpoint_hamiltonian(self):
        """Fingerprint the exact internal Hamiltonian, once and outside the residual loop."""
        if not hasattr(self, "_checkpoint_hamiltonian_sha256"):
            reference = self._reference
            digest = sha256(np.asarray(self._constant, dtype="<f8").tobytes())
            digest.update(np.ascontiguousarray(reference.h, dtype="<f8"))
            eri = reference.g.transpose(0, 2, 1, 3)
            for slab in eri:
                digest.update(np.ascontiguousarray(slab, dtype="<f8"))
            self._checkpoint_hamiltonian_sha256 = digest.hexdigest()
        return self._checkpoint_hamiltonian_sha256

    def save_state(self, path, state, metadata=None, *, writer=None):
        """Atomically write a CCSDTQState (device arrays are streamed).

        With a `checkpoint_io.BackgroundArrayWriter`, the state is copied to
        host memory and written by its thread while the solve continues.
        """
        arrays = {f"t{rank}": array for rank, array in enumerate(state.amplitudes, 1)}
        shapes = []
        for index, (vector, error) in enumerate(state.diis_history):
            for kind, value in (("vector", vector), ("error", error)):
                parts = value if isinstance(value, tuple) else (value,)
                shapes.append(len(parts))
                for part_index, part in enumerate(parts):
                    arrays[f"diis_{index}_{kind}_{part_index}"] = part
        arrays["diis_gram"] = np.asarray(state.diis_gram, dtype=np.float64)
        iterations = [dict(energy=i.energy, delta_energy=i.delta_energy if isfinite(i.delta_energy) else None,
                           residual_max=list(i.residual_max)) for i in state.iterations]
        meta = dict(format=STATE_FORMAT, evaluations=state.evaluations,
                    previous_correlation=state.previous_correlation, settings=state.settings,
                    layout=self.layout_signature(), iterations=iterations,
                    diis_parts=shapes, metadata=metadata or {}, precision=self.precision,
                    hamiltonian_sha256=self._checkpoint_hamiltonian())
        if self.backend == "cuda":
            from . import gemm_contract

            meta["gemm_routes"] = gemm_contract.routes()
        if writer is None:
            write_array_directory(path, arrays, meta)
        else:
            writer.submit(path, arrays, meta)

    def load_state(self, path):
        """Read a state written by save_state for this Hamiltonian and layout."""
        arrays, meta = read_array_directory(path, lazy=True)
        if meta.get("format") != STATE_FORMAT or meta["layout"] != self.layout_signature():
            raise ValueError("The saved CCSDTQ state belongs to another layout or format")
        if ("hamiltonian_sha256" in meta
                and meta["hamiltonian_sha256"] != self._checkpoint_hamiltonian()):
            raise ValueError("The saved CCSDTQ state belongs to another Hamiltonian")
        if meta.get("precision", self.precision) != self.precision:
            raise ValueError("The saved CCSDTQ state uses a different precision")
        history, parts = [], iter(meta["diis_parts"])
        for index in range(len(meta["diis_parts"]) // 2):
            pair = []
            for kind in ("vector", "error"):
                values = tuple(arrays[f"diis_{index}_{kind}_{p}"] for p in range(next(parts)))
                pair.append(values if len(values) > 1 else values[0])
            history.append(tuple(pair))
        iterations = tuple(Iteration(row["energy"], float("inf") if row["delta_energy"] is None
                                     else row["delta_energy"], tuple(row["residual_max"]))
                           for row in meta["iterations"])
        amplitudes = tuple(arrays[f"t{rank}"] for rank in range(1, 5))
        if tuple(a.shape for a in amplitudes) != self.shapes:
            raise ValueError("The saved CCSDTQ amplitudes do not match the layout")
        state = CCSDTQState(meta["evaluations"], amplitudes, meta["previous_correlation"],
                            iterations, history, arrays["diis_gram"], meta["settings"])
        if self.backend == "cuda" and meta.get("gemm_routes"):
            from . import gemm_contract

            # The saved run's autotuned GEMM routes: the resumed iteration
            # rounds as the saved one would have (on the same GPU model).
            gemm_contract.restore_routes(meta["gemm_routes"])
        return state, meta["metadata"]


def _to_device(convert, value):
    if isinstance(value, tuple):
        return tuple(convert(part) for part in value)
    return convert(value)
