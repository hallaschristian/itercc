"""Restarted Krylov solvers shared by every EOM and response method.

Both solvers keep their search space in one preallocated block of rows on
the device of the supplied vectors (or in host memory for NumPy vectors).
Orthogonalization, projected-matrix updates and Ritz/solution assembly are
then single matrix-vector or matrix-matrix products, and every restart is a
change of basis in coefficient space that needs no extra operator action.

* `davidson`: real non-Hermitian block Davidson with thick (GD+k) restarts.
  A restart keeps the current Ritz vectors and the previous iteration's Ritz
  vectors, which retains the convergence history a plain Ritz restart drops.
* `gmres`: right-preconditioned GMRES with deflated restarting (GMRES-DR).
  A restart keeps the harmonic Ritz vectors nearest the origin together with
  the residual, so slow modes found in one cycle stay deflated in the next.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np


def _host(xp):
    return np.asarray if xp is np else xp.asnumpy


def _product(xp, a, b):
    """a @ b; real float64 NumPy blocks run on the CPU core's GEMM executor
    (all threads, no OpenBLAS pool; the engine of the CC residuals)."""
    if (xp is np and a.dtype == np.float64 and b.dtype == np.float64 and a.ndim in (1, 2)
            and b.ndim in (1, 2) and a.size and b.size):
        from ._cpu_triples import matmul

        left = a.reshape(1, -1) if a.ndim == 1 else a
        right = b.reshape(-1, 1) if b.ndim == 1 else b
        out = np.empty((left.shape[0], right.shape[1]))
        matmul(left, right, out)
        if a.ndim == 1 and b.ndim == 1:
            return out[0, 0]
        if a.ndim == 1:
            return out[0]
        return out[:, 0] if b.ndim == 1 else out
    return a @ b


class Columns:
    """A preallocated block of equal-length vectors stored as contiguous rows."""

    def __init__(self, xp, size, capacity, dtype, *, chunk_bytes=64 << 20):
        self.xp = xp
        self.data = xp.empty((capacity, size), dtype=dtype)
        self.count = 0
        self.chunk_bytes = chunk_bytes

    @property
    def capacity(self):
        return self.data.shape[0]

    def block(self, stop=None):
        return self.data[:self.count if stop is None else stop]

    def overlaps(self, vector, stop=None):
        """Return conj(Q) @ vector for the first `stop` rows as a host array."""
        block = self.block(stop)
        if not len(block):
            return np.zeros(0, dtype=np.result_type(block.dtype, vector.dtype))
        xp = self.xp
        if xp.iscomplexobj(block):
            return _host(xp)((block @ vector.conj()).conj())
        if xp.iscomplexobj(vector):
            return (_host(xp)(_product(xp, block, xp.ascontiguousarray(vector.real)))
                    + 1j * _host(xp)(_product(xp, block, xp.ascontiguousarray(vector.imag))))
        return _host(xp)(_product(xp, block, vector))

    def combine(self, coefficients, stop=None):
        """Return sum_i c_i q_i without forming complex copies of a real block."""
        xp = self.xp
        coefficients = np.asarray(coefficients)
        block = self.block(len(coefficients) if stop is None else stop)
        if np.iscomplexobj(coefficients) and not xp.iscomplexobj(block):
            real = _product(xp, xp.asarray(np.ascontiguousarray(coefficients.real)), block)
            if not np.any(coefficients.imag):
                return real
            return real + 1j * _product(xp, xp.asarray(np.ascontiguousarray(coefficients.imag)), block)
        return _product(xp, xp.asarray(coefficients.astype(block.dtype, copy=False)), block)

    def orthogonalize(self, vector, stop=None):
        """Two-pass classical Gram-Schmidt in place; returns the coefficients."""
        stop = self.count if stop is None else stop
        total = np.zeros(stop, dtype=np.result_type(self.data.dtype, vector.dtype))
        if not stop:
            return total
        for _ in range(2):
            overlap = self.overlaps(vector, stop)
            vector -= self.combine(overlap, stop)
            total += overlap
        return total

    def append(self, vector):
        if self.count >= self.capacity:
            raise IndexError("Krylov block is full")
        self.data[self.count] = vector
        self.count += 1

    def transform(self, coefficients):
        """In place: row i <- sum_l Y[l, i] row l, for an m x r matrix Y with r <= m.

        Column chunks bound the workspace; each chunk is read before it is
        overwritten, so no second full block is allocated.
        """
        xp = self.xp
        coefficients = np.asarray(coefficients)
        m, r = coefficients.shape
        if r > m or m > self.count:
            raise ValueError("Basis change must not enlarge the block")
        mixed = xp.asarray(coefficients.T.astype(self.data.dtype, copy=False))
        size = self.data.shape[1]
        step = max(1, self.chunk_bytes // max(1, m * self.data.itemsize))
        for start in range(0, size, step):
            stop = min(size, start + step)
            self.data[:r, start:stop] = _product(xp, mixed, self.data[:m, start:stop])
        self.count = r


class RecycledImages:
    """Directions z with their images A z under one real linear operator,
    collected over many solves of shifted systems (A - s) x = b, to start
    each new solve from its best approximation in their span. A converged
    solution enters at no action: A x = b + s x up to its residual.

    The directions are kept orthonormal (Z^T Z = 1; each image W follows the
    same combinations, W = A Z), with the Gram matrices W^T W and Z^T W. For a
    shift s and right-hand side b, `guess` minimizes ||b - (W - s Z) y|| over
    y through the normal equations, at no operator action: every shift reuses
    every image (A is shift-free), so a sweep of frequencies, the +-omega
    pair of one frequency and the zero-shift solves of many right-hand sides
    all build on the directions found before. Real directions only: a complex
    direction enters as its real and imaginary parts (A is real).
    """

    def __init__(self, xp, size, capacity):
        self.xp = xp
        self.z = Columns(xp, size, capacity, np.float64)
        self.w = Columns(xp, size, capacity, np.float64)
        self.ww = np.zeros((capacity, capacity))
        self.zw = np.zeros((capacity, capacity))     # zw[i, j] = z_i . w_j
        self.skipped = 0                             # directions refused when full

    @property
    def count(self):
        return self.z.count

    def add(self, z, w, *, drop=1e-8):
        """Record a real direction z with w = A z (neither is modified); a
        direction (nearly) in the span already, or a full store, is skipped."""
        xp = self.xp
        if self.z.count >= self.z.capacity:
            self.skipped += 1
            return False
        scale = float(xp.linalg.norm(z))
        if not scale > 0 or not np.isfinite(scale):
            return False
        z, w = z.astype(np.float64, copy=True), w.astype(np.float64, copy=True)
        coefficients = self.z.orthogonalize(z)
        if len(coefficients):
            w -= self.w.combine(coefficients)
        length = float(xp.linalg.norm(z))
        if length <= drop * scale:
            return False
        z /= length
        w /= length
        k = self.z.count
        self.ww[:k, k] = self.ww[k, :k] = self.w.overlaps(w)
        self.zw[:k, k] = self.z.overlaps(w)
        self.zw[k, :k] = self.w.overlaps(z)
        self.ww[k, k] = float(xp.dot(w, w))
        self.zw[k, k] = float(xp.dot(z, w))
        self.z.append(z)
        self.w.append(w)
        return True

    def guess(self, rhs, shift=0.0, *, rcond=1e-12):
        """(x, estimated residual norm) of min ||rhs - (A - shift) x|| over
        x in span(Z), or (None, ||rhs||) for an empty store."""
        xp, k = self.xp, self.z.count
        norm = float(xp.linalg.norm(rhs))
        if not k or norm == 0:
            return None, norm
        s = complex(shift)
        ww, zw = self.ww[:k, :k], self.zw[:k, :k]
        # (W - sZ)^H (W - sZ) with Z^T Z = 1, and (W - sZ)^H b.
        normal = ww - s * zw.T - np.conj(s) * zw + abs(s) ** 2 * np.eye(k)
        right = self.w.overlaps(rhs) - np.conj(s) * self.z.overlaps(rhs)
        if not (np.iscomplexobj(rhs) or s.imag):
            normal, right = normal.real, np.real(right)
        y = np.linalg.lstsq(normal, right, rcond=rcond)[0]
        residual = max(norm**2 - float(np.real(np.vdot(right, y))), 0.0) ** 0.5
        return self.z.combine(y), residual


def orthonormal_columns(matrix, *, drop=1e-10):
    """Orthonormal basis of the columns in order (CGS2), dropping dependent ones.

    Returns (basis, kept) where kept[i] tells whether input column i added a
    direction, so leading spans are preserved.
    """
    matrix = np.asarray(matrix)
    basis, kept = [], []
    for column in matrix.T:
        scale = np.linalg.norm(column)
        if scale == 0 or not np.isfinite(scale):
            kept.append(False)
            continue
        vector = column / scale
        for _ in range(2):
            for q in basis:
                vector = vector - np.vdot(q, vector) * q
        length = np.linalg.norm(vector)
        if length <= drop:
            kept.append(False)
            continue
        basis.append(vector / length)
        kept.append(True)
    dtype = matrix.dtype if np.issubdtype(matrix.dtype, np.inexact) else np.float64
    if not basis:
        return np.zeros((matrix.shape[0], 0), dtype=dtype), kept
    return np.stack(basis, axis=1), kept


def _real_parts(column):
    column = np.asarray(column)
    if np.iscomplexobj(column):
        if not np.any(column.imag):
            return [column.real]
        return [column.real, column.imag]
    return [column]


@dataclass(frozen=True)
class DavidsonOutcome:
    converged: bool
    values: np.ndarray
    vectors: tuple
    residual_norms: tuple
    history: tuple


def davidson(sigma, diagonal, project, *, size, nroots, xp, residual_tol=1e-8, max_iter=100,
             max_space=20, guesses=None, callback=None, checkpoint_callback=None, certify=None,
             matvecs=None, retain_previous=True, olsen=False, precondition=None, minimum=1e-7):
    """Lowest `nroots` roots with real part above `minimum` of a real non-Hermitian operator.

    `sigma(v)` applies the operator to a physical vector, `project(v)` maps
    any vector onto the physical subspace and `diagonal` preconditions the
    correction, unless `precondition(residual, energy, vector)` returns it.
    Converged roots are certified with `certify(root, energy)` (a fresh,
    independent residual norm) before acceptance. With `olsen`, the diagonal
    correction is made orthogonal to the Ritz vector (Olsen's correction).
    """
    if (not isinstance(nroots, (int, np.integer)) or nroots < 1 or max_space < 2 * nroots + 1
            or max_iter < 1 or not np.isfinite(residual_tol) or residual_tol <= 0):
        raise ValueError("Invalid root count, subspace, iteration or residual controls")
    tohost = _host(xp)
    count = matvecs if matvecs is not None else (lambda: None)
    basis = Columns(xp, size, max_space, xp.float64)
    images = Columns(xp, size, max_space, xp.float64)
    projected = np.zeros((max_space, max_space))

    def orthonormal(vector, others=()):
        """Project onto the physical space, then CGS2 against the basis and `others`."""
        candidate = project(xp.asarray(vector, dtype=xp.float64))
        if candidate is vector or getattr(candidate, "base", None) is vector:
            candidate = candidate.copy()
        for _ in range(2):
            candidate -= basis.combine(basis.overlaps(candidate))
            for other in others:
                candidate -= float(xp.dot(other, candidate)) * other
        length = float(xp.linalg.norm(candidate))
        if length < 1e-13:
            return None
        candidate /= length
        return candidate

    def append(vector):
        candidate = orthonormal(vector)
        if candidate is not None:
            basis.append(candidate)

    if guesses is not None:
        parts = [part for guess in guesses
                 for part in ((guess.real, guess.imag) if xp.iscomplexobj(guess) else (guess,))]
        if len(parts) > max_space:
            raise ValueError("Too many initial guesses")
        for part in parts:
            append(part)
    for index in np.argsort(tohost(diagonal)):
        if basis.count >= min(max_space, 2 * nroots + 2):
            break
        unit = xp.zeros(size)
        unit[int(index)] = 1
        append(unit)
    if basis.count < nroots:
        raise ValueError("Requested more roots than the physical sector supports")

    def ritz(column):
        return basis.combine(column)

    def ritz_pair(column, energy):
        if not np.iscomplexobj(column):
            energy = float(np.real(energy))
        vector = ritz(column)
        image = images.combine(column)
        image -= energy * vector
        return vector, image

    history, previous = [], None
    start = perf_counter()
    for iteration in range(1, max_iter + 1):
        m, known = basis.count, images.count
        while images.count < m:
            images.append(sigma(basis.data[images.count]))
        if known < m:
            projected[:m, known:m] = tohost(_product(xp, basis.block(), images.data[known:m].T))
            if known:
                projected[known:m, :known] = tohost(_product(xp, basis.data[known:m], images.data[:known].T))
        energies, vectors = np.linalg.eig(projected[:m, :m])
        valid = [i for i in np.argsort(energies.real) if energies[i].real > minimum]
        if len(valid) < nroots:
            raise RuntimeError(f"Insufficient positive Ritz roots at iteration {iteration}: {energies}")
        chosen = valid[:nroots]
        omega = energies[chosen]
        # One Ritz vector and residual per root and iteration: the residual
        # norm and, for a pending root, its correction (orthogonal to the
        # whole current basis, including directions a restart is about to
        # drop) come from the same pass. The basis is orthonormal, so the
        # coefficients carry the norm of the Ritz vector.
        keep = checkpoint_callback is not None
        columns, lengths, corrections, ritz_vectors = [], [], [], []
        for energy, column in zip(omega, vectors[:, chosen].T):
            if np.iscomplexobj(column) and not np.any(column.imag):
                column = column.real
            column = column / float(np.linalg.norm(column))
            columns.append(column)
            vector, residual = ritz_pair(column, energy)
            lengths.append(float(xp.linalg.norm(residual)))
            if keep:
                ritz_vectors.append(vector)
            if lengths[-1] > residual_tol:
                if precondition is not None:
                    residual = precondition(residual, float(np.real(energy)), vector)
                else:
                    denominator = float(np.real(energy)) - diagonal
                    denominator = xp.where(xp.abs(denominator) < 1e-3,
                                           xp.where(denominator < 0, -1e-3, 1e-3), denominator)
                    residual /= denominator
                    if olsen and not xp.iscomplexobj(residual):
                        # Olsen: t = (theta - D)^-1 (r - eps x) with x . t = 0.
                        scaled = vector / denominator
                        ratio = float(xp.dot(vector, residual)) / float(xp.dot(vector, scaled))
                        residual -= ratio * scaled
                        del scaled
                    del denominator
                parts = (residual.real, residual.imag) if xp.iscomplexobj(residual) else (residual,)
                for part in parts:
                    candidate = orthonormal(part, corrections)
                    if candidate is not None:
                        corrections.append(candidate)
                del parts
            del vector, residual
        info = dict(iteration=iteration, excitation_Eh=np.real(omega).tolist(),
                    excitation_imaginary_Eh=np.imag(omega).tolist(), residual_norm=lengths,
                    subspace=m, matvecs=count(), elapsed_seconds=perf_counter() - start)
        history.append(info)
        if callback is not None:
            callback(info)
        if keep:
            checkpoint_callback(info, tuple(ritz_vectors))
        if max(lengths) <= residual_tol and max(abs(np.imag(omega))) <= residual_tol:
            roots = tuple(ritz_vectors) if keep else tuple(ritz(column) for column in columns)
            final = (tuple(certify(root, float(np.real(energy))) for root, energy in zip(roots, omega))
                     if certify is not None else tuple(lengths))
            if max(final) <= residual_tol:
                return DavidsonOutcome(True, np.real(omega), roots, final, tuple(history))
            raise RuntimeError(f"Full-space EOM residual {max(final):.3e} "
                               f"exceeds tolerance {residual_tol:.3e}")
        del ritz_vectors
        if not corrections:
            raise RuntimeError("Davidson subspace stagnated before convergence")
        needed = min(len(corrections), max_space - nroots)
        if m + needed > max_space:
            # Thick restart: current Ritz vectors, then (GD+k) the previous
            # iteration's Ritz vectors, all formed in coefficient space.
            kept = [part for column in columns for part in _real_parts(column)]
            room = max_space - needed - len(kept)
            if retain_previous and previous is not None and room > 0:
                extra = []
                for column in previous:
                    padded = np.zeros(m, dtype=column.dtype)
                    padded[:len(column)] = column
                    extra.extend(_real_parts(padded))
                kept.extend(extra[:room])
            change, _ = orthonormal_columns(np.stack(kept, axis=1))
            change = change[:, :max(nroots, max_space - needed)]
            r = change.shape[1]
            basis.transform(change)
            images.transform(change)
            projected[:r, :r] = change.T @ projected[:m, :m] @ change
            columns = [change.T @ column for column in columns]
            m = r
        for candidate in corrections:
            # Already projected and orthonormal to the basis: a thick restart
            # keeps a subspace of the span the corrections were made against.
            if basis.count < max_space:
                basis.append(candidate)
        del corrections
        if basis.count == m:
            raise RuntimeError("Davidson subspace stagnated before convergence")
        previous = columns
    roots = tuple(ritz(column) for column in columns)
    return DavidsonOutcome(False, omega, roots, tuple(lengths), tuple(history))


def default_deflation(width):
    return min(10, width // 3)


def _harmonic(hessenberg, nq, j, k, residual=None):
    """Orthonormal coefficient basis keeping k harmonic Ritz vectors nearest 0.

    With S = Q[:, :j] and B S = Q H, harmonic Ritz pairs satisfy
    H^H H g = theta H[:j]^H g. The returned Y spans [g_1..g_k], H g and the
    residual coefficients, so B (Q Y[:j, :k']) stays inside span(Q Y).
    """
    from scipy.linalg import eig

    full = hessenberg[:nq, :j]
    square = hessenberg[:j, :j]
    values, vectors = eig(full.conj().T @ full, square.conj().T)
    finite = np.flatnonzero(np.isfinite(values))
    order = finite[np.argsort(np.abs(values[finite]), kind="stable")]
    real = not np.iscomplexobj(hessenberg)
    picked = []
    for index in order:
        if len(picked) >= k:
            break
        column = vectors[:, index]
        picked.extend(_real_parts(column) if real else [column])
    if not picked:
        return None, 0
    directions = np.stack(picked, axis=1)
    padded = np.zeros((nq, directions.shape[1]), dtype=directions.dtype)
    padded[:j] = directions
    leading_basis, _ = orthonormal_columns(padded)
    # In exact arithmetic span(H G, residual) adds one direction to span(G).
    # Rank-reveal the complement so rounding noise never claims a column.
    rest = [full @ directions]
    if residual is not None:
        rest.append(np.asarray(residual)[:nq, None])
    rest = np.concatenate(rest, axis=1)
    scale = np.linalg.norm(rest, axis=0)
    rest = rest[:, scale > 0] / scale[scale > 0]
    for _ in range(2):
        rest = rest - leading_basis @ (leading_basis.conj().T @ rest)
    left, singular, _ = np.linalg.svd(rest, full_matrices=False)
    change = np.concatenate([leading_basis, left[:, singular > 1e-9]], axis=1)
    if real:
        change = change.real
    return change, leading_basis.shape[1]


def gmres(action, rhs, diagonal, xp, *, tol, restart, maxiter, initial=None, deflation=None):
    """Right-preconditioned GMRES with deflated restarting.

    Solves action(x) = rhs with right preconditioning by `diagonal` (an array
    dividing each vector, or any callable applying M^-1). `restart` bounds
    the basis to restart + 1 vectors. `deflation` harmonic Ritz vectors are
    kept across restarts (0 gives classical restarted GMRES). Restarts use
    the implicit residual, so the only extra action is the final true
    residual check (a drifted implicit residual continues the iteration).
    Returns (x, relative true residual, Arnoldi steps).
    """
    norm_rhs = float(xp.linalg.norm(rhs))
    if norm_rhs == 0:
        return xp.zeros_like(rhs), 0., 0
    inverse = diagonal if callable(diagonal) else (lambda vector: vector / diagonal)
    dtype = rhs.dtype
    x = xp.zeros_like(rhs)
    residual = rhs.copy()
    if initial is not None:
        trial = rhs - action(initial)
        if float(xp.linalg.norm(trial)) < norm_rhs:
            x, residual = initial.astype(dtype, copy=True), trial
        del trial
    if float(xp.linalg.norm(residual)) <= tol * norm_rhs:
        return x, float(xp.linalg.norm(residual)) / norm_rhs, 0
    width = int(min(restart, rhs.size))
    k = default_deflation(width) if deflation is None else max(0, min(int(deflation), (width - 2) // 2))
    capacity = width + 1
    space = Columns(xp, rhs.size, capacity, dtype)
    hessenberg = np.zeros((capacity, capacity), dtype=np.result_type(dtype, np.float64))
    coefficients = np.zeros(capacity, dtype=hessenberg.dtype)
    state = dict(nq=0, j=0)

    def begin(vector, keep=False):
        """Start a cycle from `vector` (consumed); `keep` retains the current
        deflation rows and their Hessenberg block and appends the new direction."""
        coefficients[:] = 0
        if keep:
            coefficients[:state["nq"]] = space.orthogonalize(vector)
        else:
            space.count, state["nq"], state["j"] = 0, 0, 0
            hessenberg[:] = 0
        beta = float(xp.linalg.norm(vector))
        if beta > 1e-300:
            vector /= beta
            space.append(vector)
            coefficients[state["nq"]] = beta
            state["nq"] += 1

    def least_squares():
        nq, j = state["nq"], state["j"]
        target = coefficients[:nq]
        if not j:
            return np.zeros(0, dtype=hessenberg.dtype), float(np.linalg.norm(target))
        solution = np.linalg.lstsq(hessenberg[:nq, :j], target, rcond=None)[0]
        return solution, float(np.linalg.norm(target - hessenberg[:nq, :j] @ solution))

    def deflated_space(residual_coefficients):
        nq, j = state["nq"], state["j"]
        if not k or not j:
            return None, 0
        change, leading = _harmonic(hessenberg, nq, j, k, residual_coefficients)
        if change is None or not leading or change.shape[1] > capacity - 2:
            return None, 0
        return change, leading

    def apply_change(change, leading):
        nq, j = state["nq"], state["j"]
        new = change.conj().T @ hessenberg[:nq, :j] @ change[:j, :leading]
        space.transform(change)
        r = change.shape[1]
        hessenberg[:] = 0
        hessenberg[:r, :leading] = new
        state["nq"], state["j"] = r, leading

    begin(residual)
    del residual
    steps, checked = 0, -1      # Arnoldi steps; steps at the last true-residual check
    while True:
        solution, estimate = least_squares()
        breakdown = False
        while estimate > .8 * tol * norm_rhs and state["nq"] < capacity and steps < maxiter:
            nq, j = state["nq"], state["j"]
            w = action(inverse(space.data[j]))
            steps += 1
            hessenberg[:nq, j] = space.orthogonalize(w, nq)
            length = float(xp.linalg.norm(w))
            hessenberg[nq, j] = length
            state["j"] = j + 1
            if length > 1e-14:
                w /= length
                space.append(w)
                state["nq"] = nq + 1
            del w
            solution, estimate = least_squares()
            if length <= 1e-14:
                breakdown = True
                break
        nq, j = state["nq"], state["j"]
        if j:
            x += inverse(space.combine(solution, j))
        remaining = coefficients[:nq] - hessenberg[:nq, :j] @ solution
        finished = estimate <= .8 * tol * norm_rhs or breakdown or steps >= maxiter or not j
        if finished:
            true = rhs - action(x)
            relative = float(xp.linalg.norm(true)) / norm_rhs
            # No step since the last check: the deflation space already
            # holds the true residual and another check cannot improve x.
            if relative <= tol or steps >= maxiter or not j or steps == checked:
                del space
                return x, relative, steps
            checked = steps
            # The implicit residual drifted: keep the deflation space and
            # continue from the true residual.
            change, leading = deflated_space(None)
            if change is not None:
                apply_change(change, leading)
            begin(true, keep=change is not None)
            del true
            continue
        change, leading = deflated_space(remaining)
        if change is None:
            # Classical restart on the implicit residual Q @ remaining.
            vector = space.combine(remaining, nq)
            begin(vector)
            del vector
            continue
        apply_change(change, leading)
        coefficients[:] = 0
        coefficients[:change.shape[1]] = change.conj().T @ remaining
