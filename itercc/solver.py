"""T1-dressed spin-orbital tensor CCSD and full iterative CCSDT."""

from dataclasses import dataclass
from itertools import product
from numbers import Integral

import numpy as np

from . import _equations
from .hamiltonian import Hamiltonian, _real64


@dataclass(frozen=True)
class Iteration:
    energy: float
    delta_energy: float
    residual_max: tuple[float, ...]


@dataclass(frozen=True)
class Result:
    energy: float
    correlation_energy: float
    amplitudes: tuple[np.ndarray, ...]
    converged: bool
    iterations: tuple[Iteration, ...]

    @property
    def residual_max(self):
        return self.iterations[-1].residual_max


def _dress_axis(tensor, axis, target, source, rotation, xp, *, operand=None):
    legs = xp.moveaxis(tensor, axis, 0)
    source_legs = legs if operand is None else xp.moveaxis(operand, axis, 0)
    legs[target] += xp.tensordot(rotation, source_legs[source], axes=1)


class _PairLadder:
    """Evaluate 1/2 <ab||de> t[de,...] using antisymmetric pairs and GEMM."""

    def __init__(self, nv, xp):
        self.xp = xp
        self.p, self.q = xp.triu_indices(nv, 1)
        self._eri = None
        self._packed = None

    def __call__(self, eri, amplitude):
        if self._eri is not eri:
            p, q = self.p, self.q
            self._packed = eri[p[:, None], q[:, None], p, q]
            self._eri = eri
        p, q = self.p, self.q
        tail = amplitude.shape[2:]
        result = self.xp.zeros_like(amplitude)
        if p.size:
            amplitudes = amplitude[p, q].reshape(len(p), -1)
            unique = (self._packed @ amplitudes).reshape((len(p), *tail))
            result[p, q] = unique
            result[q, p] = -unique
        return result


def _dense_ladder(eri, amplitude, xp):
    return 0.5 * xp.tensordot(eri, amplitude, axes=((2, 3), (0, 1)))


class _DIIS:
    def __init__(self, size, xp):
        self.size = size
        self.xp = xp
        self.history = []
        self.gram = np.empty((0, 0))

    def reserve_slot(self):
        if self.size >= 2 and len(self.history) == self.size:
            self.history.pop(0)
            self.gram = self.gram[1:, 1:]

    def _append(self, vector, error, dot, dots=None):
        """Store (vector, error) and return the extrapolation weights (None:
        take the plain step). `dots(error, errors)`, when given, returns
        [e . error for e in errors] + [error . error] in one pass over `error`."""
        self.reserve_slot()
        n = len(self.history) + 1
        gram = np.empty((n, n))
        gram[:-1, :-1] = self.gram
        if dots is not None:
            dots = np.asarray(dots(error, [previous for _, previous in self.history]), dtype=float)
        else:
            dots = [dot(previous, error) for _, previous in self.history]
            dots.append(dot(error, error))
            dots = self.xp.stack(dots)
            if self.xp is not np:
                dots = dots.get()
        gram[-1, :-1] = gram[:-1, -1] = dots[:-1]
        gram[-1, -1] = dots[-1]
        self.gram = gram
        self.history.append((vector, error))
        if n == 1 or not gram[-1, -1]:
            return None
        # A nearly dependent history (singular system, or weights with an L1
        # norm above 100) extrapolates from its newest entries instead, the
        # older ones weighted zero, rather than switching DIIS off until the
        # dependent entries age out.
        for first in range(n - 1):
            k = n - first
            block = gram[first:, first:]
            system = np.ones((k + 1, k + 1))
            system[:k, :k] = block / np.max(np.diag(block))
            system[-1, -1] = 0
            rhs = np.zeros(k + 1)
            rhs[-1] = 1
            try:
                weights = np.linalg.solve(system, rhs)[:-1]
            except np.linalg.LinAlgError:
                continue
            if np.isfinite(weights).all() and np.linalg.norm(weights, ord=1) <= 100:
                if first:
                    weights = np.concatenate([np.zeros(first), weights])
                return weights
        return None

    def update(self, amplitudes, steps):
        if self.size < 2:
            return amplitudes
        vector = self.xp.concatenate([t.ravel() for t in amplitudes])
        error = self.xp.concatenate([t.ravel() for t in steps])
        weights = self._append(vector, error, self.xp.dot)
        if weights is None:
            return amplitudes
        mixed = self.xp.zeros_like(vector)
        for weight, (previous, _) in zip(weights, self.history):
            mixed += weight * previous
        cuts = np.cumsum([t.size for t in amplitudes[:-1]]).tolist()
        return tuple(x.reshape(t.shape) for x, t in zip(self.xp.split(mixed, cuts), amplitudes))


class TensorCC:
    """Dense reference-quality tensor implementation in real spin orbitals.

    CCSDT updates all three ranks together. The Hamiltonian arrays must remain
    fixed for the life of this solver. Use restricted() to set up a closed shell.
    """

    def __init__(self, hamiltonian: Hamiltonian, *, rank=3, pair_ladder=True, factored_triples=True, backend="cpu"):
        if rank not in (2, 3):
            raise ValueError("Supported ranks are 2 (CCSD) and 3 (CCSDT)")
        if backend == "cpu":
            self.xp = np
        elif backend == "cuda":
            import cupy

            self.xp = cupy
        else:
            raise ValueError("backend must be 'cpu' or 'cuda'")
        self.hamiltonian = hamiltonian
        self._h, self._g = map(self.xp.asarray, (hamiltonian.h, hamiltonian.g))
        self.rank = rank
        self.pair_ladder = pair_ladder
        self._triples = _equations.triples_factored if factored_triples else _equations.triples
        no = hamiltonian.nocc
        nv = len(hamiltonian.h) - no
        self.slices = {"o": slice(no), "v": slice(no, None)}
        self.shapes = tuple((nv,) * r + (no,) * r for r in range(1, rank + 1))
        self.fock = hamiltonian.fock()
        self._fock = self.xp.asarray(self.fock)
        self.reference_energy = hamiltonian.reference_energy()
        self._blocks = {
            "".join(key): tuple(self.slices[axis] for axis in key)
            for key in product("ov", repeat=4)
        }
        self._fblocks = {
            a + b: (self.slices[a], self.slices[b]) for a, b in product("ov", repeat=2)
        }

    def _validated(self, amplitudes):
        arrays = tuple(_real64(t) for t in amplitudes)
        if tuple(t.shape for t in arrays) != self.shapes or not all(
            np.isfinite(t).all() for t in arrays
        ):
            raise ValueError(f"Expected finite amplitudes with shapes {self.shapes}")
        for rank, tensor in enumerate(arrays, start=1):
            for start in (0, rank):
                for axis in range(start, start + rank - 1):
                    if not np.allclose(tensor, -tensor.swapaxes(axis, axis + 1),
                                       atol=1e-12, rtol=1e-9):
                        raise ValueError(f"T{rank} must be separately antisymmetric "
                                         "in the virtual and occupied indices")
        return arrays

    def _dressed(self, t1, *, one_body=False):
        xp = self.xp
        o, v = self.slices["o"], self.slices["v"]
        # e^-T1 H e^T1 rotates creator legs by I-T1 and annihilator
        # legs by I+T1. Since the orbital T1 matrix squares to zero the
        # orbital exponentials are exact at first order.
        h, g = self._h.copy(), self._g.copy()
        _dress_axis(h, 0, v, o, -t1, xp)
        _dress_axis(h, 1, o, v, t1.T, xp)
        for axis in (0, 1):
            _dress_axis(g, axis, v, o, -t1, xp)
        for axis in (2, 3):
            _dress_axis(g, axis, o, v, t1.T, xp)
        f = h + xp.einsum("piqi->pq", g[:, o, :, o])
        return (f, g, h) if one_body else (f, g)

    def _residuals(self, amplitudes):
        t1, t2 = amplitudes[:2]
        t3 = amplitudes[2] if self.rank == 3 else None
        f, g = self._dressed(t1)
        f = {key: f[index] for key, index in self._fblocks.items()}
        g = {key: g[index] for key, index in self._blocks.items()}
        xp = self.xp
        if self.pair_ladder:
            ladder = _PairLadder(t1.shape[0], xp)
        else:
            def ladder(eri, amplitude):
                return _dense_ladder(eri, amplitude, xp)

        r1 = _equations.singles(f, g, t2, t3, ladder, xp.einsum)
        r2 = _equations.doubles(f, g, t2, t3, ladder, xp.einsum)
        if t3 is None:
            return r1, r2
        return r1, r2, self._triples(f, g, t2, t3, ladder, xp.einsum)

    def residuals(self, amplitudes):
        """Return projected, unpreconditioned residuals in virtual-first order.

        T2 and T3 must be antisymmetric separately in virtual and occupied
        indices. The corresponding residuals use the same convention.
        """
        arrays = tuple(map(self.xp.asarray, self._validated(amplitudes)))
        return self._host_arrays(self._residuals(arrays))

    def _host_arrays(self, arrays):
        return arrays if self.xp is np else tuple(t.get() for t in arrays)

    def _correlation_energy(self, amplitudes):
        t1, t2 = amplitudes[:2]
        o, v = self.slices["o"], self.slices["v"]
        integrals = self._g[o, o, v, v]
        return float(self.xp.einsum("ia,ai->", self._fock[o, v], t1)
                     + 0.25 * self.xp.einsum("ijab,abij->", integrals, t2)
                     + 0.5 * self.xp.einsum("ijab,ai,bj->", integrals, t1, t1, optimize=True))

    def correlation_energy(self, amplitudes):
        return self._correlation_energy(tuple(map(self.xp.asarray, self._validated(amplitudes))))

    def _denominators(self, level_shift):
        o, v = self.slices["o"], self.slices["v"]
        diag = self.fock.diagonal()
        occupied, virtual = diag[o], diag[v] + level_shift
        denominators = []
        for rank in range(1, self.rank + 1):
            ev, eo = virtual, occupied
            for _ in range(1, rank):
                ev = np.add.outer(ev, virtual)
                eo = np.add.outer(eo, occupied)
            denominator = np.subtract.outer(eo, ev).transpose(
                tuple(range(rank, rank * 2)) + tuple(range(rank))
            )
            if np.min(np.abs(denominator)) < 1e-12:
                raise ValueError("Near-zero orbital denominator. Use a positive level shift.")
            denominators.append(self.xp.asarray(denominator))
        return denominators

    def solve(self, *, initial=None, energy_tol=1e-11, residual_tol=1e-9,
              max_iter=80, diis_size=6, level_shift=0.0, damping=0.0):
        """Solve with Jacobi preconditioning and Pulay DIIS.

        Convergence requires both a small energy change and the requested
        maximum raw residual at every rank. Always inspect result.converged.
        """
        if not (np.isfinite(energy_tol) and np.isfinite(residual_tol)
                and energy_tol > 0 and residual_tol > 0):
            raise ValueError("Energy and residual tolerances must be positive and finite")
        if (not isinstance(max_iter, Integral) or max_iter < 1
                or not isinstance(diis_size, Integral) or diis_size < 0
                or not 0 <= damping < 1 or not np.isfinite(level_shift) or level_shift < 0):
            raise ValueError("Invalid iteration controls")
        denominators = self._denominators(level_shift)
        if initial is None:
            o, v = self.slices["o"], self.slices["v"]
            guess = [self._fock[v, o] / denominators[0],
                     self._g[v, v, o, o] / denominators[1]]
            if self.rank == 3:
                guess.append(self.xp.zeros(self.shapes[2]))
            amplitudes = tuple(guess)
        else:
            amplitudes = tuple(map(self.xp.asarray, self._validated(initial)))
        diis = _DIIS(diis_size, self.xp)
        history = []
        last = np.inf
        converged = False
        for iteration in range(max_iter):
            correlation = self._correlation_energy(amplitudes)
            residuals = self._residuals(amplitudes)
            maxima = self.xp.stack([self.xp.max(self.xp.abs(r)) for r in residuals])
            if self.xp is not np:
                maxima = maxima.get()
            maxima = tuple(map(float, maxima))
            if not np.isfinite((correlation, *maxima)).all():
                raise FloatingPointError("Nonfinite coupled-cluster iterate")
            change = correlation - last
            history.append(Iteration(self.reference_energy + correlation, change, maxima))
            if abs(change) <= energy_tol and max(maxima) <= residual_tol:
                converged = True
                break
            if iteration + 1 < max_iter:
                steps = tuple((1 - damping) * r / d for r, d in zip(residuals, denominators))
                candidate = tuple(t + step for t, step in zip(amplitudes, steps))
                amplitudes = diis.update(candidate, steps)
            last = correlation
        return Result(self.reference_energy + correlation, correlation,
                      self._host_arrays(amplitudes), converged, tuple(history))

