"""Heat-bath selected configuration interaction (near-full CI) on the compiled core.

`SelectedCI` grows a determinant space by heat-bath selection (Holmes,
Tubman, Umrigar, JCTC 12, 3674 (2016)): every determinant `a` with
|H_ai c_i| >= eps for some determinant `i` of the space and root `c` is
added, the space's Hamiltonian is diagonalized, and the cycle repeats until
few determinants are new. Epstein-Nesbet second-order energies over the
determinants outside the space (screened at `pt2_epsilon`) estimate the
remaining correlation, and a sequence of decreasing `epsilons` extrapolates
the variational energies linearly in the PT2 correction to full CI
(`extrapolate`, `extrapolate_gap`).

Spaces of M_S = 0 contain each determinant's alpha-beta swap, and
`spin_parity` keeps states of even (+1: singlets, quintets) or odd (-1:
triplets) total spin: with alpha creators before beta creators, the swap
maps a spin-S, M_S = 0 state to (-1)^S times itself. Orbital irreps (Abelian
labels, XOR products) keep every determinant in the references' irrep.

The C kernels (core/src/cpu/selected_ci.cpp) hold the integrals, the sorted
excitation lists, the sparse Hamiltonian (upper triangle, 12 bytes per
nonzero) and the PT2 hash maps; results do not depend on the thread count.
"""

from dataclasses import dataclass, field
import time

import numpy as np

from . import _cpu_kernels, _runtime, krylov


def _address(array):
    return array.ctypes.data if array is not None else None


@dataclass
class Stage:
    """One converged variational space."""
    epsilon: float
    determinants: int
    nonzeros: int                   # upper-triangle off-diagonal elements
    energies: np.ndarray            # variational total energies of the roots
    pt2: np.ndarray                 # Epstein-Nesbet corrections (zeros without PT2)
    s2: np.ndarray                  # <S^2> of the roots
    cycles: int
    seconds: dict = field(default_factory=dict)
    pt2_externals: int = 0
    pt2_error: np.ndarray = None        # standard error of a stochastic PT2 part
    pt2_norm: np.ndarray = None         # first-order norm sum_a (num_a / (E - H_aa))^2

    @property
    def total(self):
        return self.energies + self.pt2

    @property
    def renormalized_pt2(self):
        """E2 / (1 + norm), the renormalized correction: closer to linear in
        the variational energy, so extrapolations bend less."""
        norm = np.zeros_like(self.pt2) if self.pt2_norm is None else self.pt2_norm
        return self.pt2 / (1.0 + norm)


@dataclass
class SCIResult:
    stages: list
    determinants: np.ndarray        # final space (rows of 2 * words uint64)
    vectors: np.ndarray             # final roots, nroots x determinants
    constant: float

    @property
    def energies(self):
        return self.stages[-1].energies

    def extrapolate(self, root=0, points=None, renormalized=False):
        """Full-CI estimate of `root`: the variational energies of the last
        `points` stages (default all with PT2) fitted linearly against their
        PT2 corrections (renormalized ones with `renormalized`), evaluated at
        zero correction. Returns (energy, spread), the spread being the
        change from dropping the stage with the largest correction (zero
        with two points)."""
        corrections = [(s.renormalized_pt2 if renormalized else s.pt2)[root] for s in self.stages]
        return _extrapolate([s.energies[root] for s in self.stages], corrections, points)


def _extrapolate(energies, corrections, points=None):
    e, p = np.asarray(energies, float), np.asarray(corrections, float)
    use = np.flatnonzero(p != 0.0)
    if points is not None:
        use = use[-points:]
    if len(use) < 2:
        raise ValueError("Extrapolation needs at least two stages with PT2 corrections")
    fit = np.polyfit(p[use], e[use], 1)[1]
    spread = 0.0
    if len(use) > 2:
        spread = abs(np.polyfit(p[use[1:]], e[use[1:]], 1)[1] - fit)
    return float(fit), float(spread)


def extrapolate_gap(lower, upper, *, root_lower=0, root_upper=0, points=None, renormalized=False):
    """Full-CI estimate of E(upper root) - E(lower root) from two results
    whose stages share epsilons: the variational gaps fitted against the PT2
    gaps (the stages' corrections largely cancel in the gap). Returns
    (gap, spread)."""
    if len(lower.stages) != len(upper.stages):
        raise ValueError("The results need the same number of stages")
    gaps = [b.energies[root_upper] - a.energies[root_lower] for a, b in zip(lower.stages, upper.stages)]
    pick = (lambda s: s.renormalized_pt2) if renormalized else (lambda s: s.pt2)
    corrections = [pick(b)[root_upper] - pick(a)[root_lower] for a, b in zip(lower.stages, upper.stages)]
    return _extrapolate(gaps, corrections, points)


@dataclass
class SCIPolarizability:
    """Dipole polarizabilities of the selected-CI ground state in one space.

    `tensors_au[k]` is alpha_ij at `frequencies_au[k]` (au), by linear
    response in the fixed space: sum over its states n of
    <0|mu_i|n><n|mu_j|0> (1/(E_n - E_0 - w) + 1/(E_n - E_0 + w)), the exact
    response of the variational state in that space. `dipole_au` is
    <0|mu|0> of the dipole matrices as given (electronic -r)."""
    determinants: int
    energy: float
    dipole_au: np.ndarray
    frequencies_au: tuple
    tensors_au: np.ndarray
    residuals: np.ndarray            # relative residuals of the linear solves
    seconds: float = 0.0
    pt2: float = 0.0                 # zero-field PT2 of the ground state at the stage (0 without PT2)
    space: np.ndarray = None         # the determinants of the space


class SelectedCI:
    """Selected CI of a real orbital Hamiltonian (h, chemists' eri (pq|rs))
    with `nelec` electrons (an even count with M_S = 0, or (nalpha, nbeta))."""

    def __init__(self, h, eri, nelec, constant=0.0, *, orbital_irreps=None, cutoff=1e-12, cpu_threads=None):
        h = np.ascontiguousarray(h, dtype=np.float64)
        n = h.shape[0]
        if h.shape != (n, n) or not np.allclose(h, h.T, atol=1e-10):
            raise ValueError("h must be a real symmetric matrix")
        eri = np.ascontiguousarray(np.asarray(eri, dtype=np.float64).reshape(n, n, n, n))
        if n > 256:
            raise ValueError("At most 256 orbitals")
        if isinstance(nelec, (tuple, list)):
            nalpha, nbeta = (int(x) for x in nelec)
        else:
            if int(nelec) % 2:
                raise ValueError("An odd electron count needs (nalpha, nbeta)")
            nalpha = nbeta = int(nelec) // 2
        if not (0 <= nalpha <= n and 0 <= nbeta <= n):
            raise ValueError("Electron counts must fit the orbitals")
        if nalpha >= 64:
            raise ValueError("At most 63 alpha electrons")
        self._h, self._eri, self._cutoff = h, eri, float(cutoff)
        self._labels_given = orbital_irreps is not None
        self.kernels = _cpu_kernels.library()
        if self.kernels is None:
            raise RuntimeError("SelectedCI needs the compiled core (see itercc._cpu_kernels)")
        self.norb, self.nalpha, self.nbeta = n, nalpha, nbeta
        self.constant = float(constant)
        self.cpu_threads = _runtime.thread_option(cpu_threads)
        self.irreps = (np.zeros(n, dtype=np.int64) if orbital_irreps is None
                       else np.ascontiguousarray(orbital_irreps, dtype=np.int64))
        if self.irreps.shape != (n,) or self.irreps.min() < 0 or self.irreps.max() > 7:
            raise ValueError("orbital_irreps must hold one Abelian irrep 0..7 per orbital")
        self._ctx = self.kernels.sci_create(n, nalpha, nbeta, h.ctypes.data, eri.ctypes.data,
                                            self.irreps.ctypes.data, float(cutoff), self._threads())
        if not self._ctx:
            raise MemoryError("Could not create the selected-CI context")
        self.words = int(self.kernels.sci_words(self._ctx))
        sizes = np.zeros(3, dtype=np.int64)
        self.context_bytes = int(self.kernels.sci_list_sizes(self._ctx, sizes.ctypes.data))
        self.list_sizes = tuple(int(x) for x in sizes)

    def __del__(self):
        ctx = getattr(self, "_ctx", None)
        if ctx:
            self.kernels.sci_free(ctx)
            self._ctx = None

    def _threads(self):
        return self.cpu_threads if self.cpu_threads is not None else _runtime.cpu_threads()

    # ----- determinants -------------------------------------------------------------------
    def determinant(self, alpha, beta):
        """The row of a determinant with occupied alpha and beta orbitals."""
        row = np.zeros(2 * self.words, dtype=np.uint64)
        for offset, orbitals in ((0, alpha), (self.words, beta)):
            for p in orbitals:
                if not 0 <= p < self.norb:
                    raise ValueError("Orbital index out of range")
                row[offset + p // 64] |= np.uint64(1) << np.uint64(p % 64)
        if [self.occupations(row)[k].size for k in (0, 1)] != [self.nalpha, self.nbeta]:
            raise ValueError("A determinant needs nalpha and nbeta distinct orbitals")
        return row

    def occupations(self, row):
        """(alpha, beta) occupied orbitals of a determinant row."""
        out = []
        for offset in (0, self.words):
            bits = [p for p in range(self.norb) if (int(row[offset + p // 64]) >> (p % 64)) & 1]
            out.append(np.array(bits, dtype=np.int64))
        return tuple(out)

    def irrep(self, row):
        alpha, beta = self.occupations(row)
        label = 0
        for p in (*alpha, *beta):
            label ^= int(self.irreps[p])
        return label

    def aufbau(self):
        """The determinant with the lowest orbitals occupied."""
        return self.determinant(range(self.nalpha), range(self.nbeta))

    def single_excitations(self, row, irrep=None):
        """All alpha and beta single excitations of `row` (optionally of one
        irrep): rows, for building excited-state references."""
        alpha, beta = (list(x) for x in self.occupations(row))
        out = []
        for spin, occ in ((0, alpha), (1, beta)):
            for p in occ:
                for r in range(self.norb):
                    if r in occ:
                        continue
                    new = [q for q in occ if q != p] + [r]
                    det = self.determinant(new, beta) if spin == 0 else self.determinant(alpha, new)
                    if irrep is None or self.irrep(det) == irrep:
                        out.append(det)
        return np.array(out, dtype=np.uint64).reshape(-1, 2 * self.words)

    def swapped(self, rows):
        rows = np.asarray(rows, dtype=np.uint64).reshape(-1, 2 * self.words)
        return np.ascontiguousarray(np.concatenate([rows[:, self.words:], rows[:, :self.words]], axis=1))

    # ----- one space ----------------------------------------------------------------------
    def _space(self, dets):
        dets = np.ascontiguousarray(dets, dtype=np.uint64)
        handle = self.kernels.sci_space(self._ctx, dets.ctypes.data, len(dets), self._threads())
        if not handle:
            raise MemoryError("Could not index the determinant space")
        return handle

    def diagonal(self, dets):
        space = self._space(dets)
        try:
            out = np.empty(len(dets))
            self.kernels.sci_diagonal(space, out.ctypes.data, self._threads())
            return out + self.constant
        finally:
            self.kernels.sci_space_free(space)

    def _unique(self, rows):
        rows = np.ascontiguousarray(rows, dtype=np.uint64).reshape(-1, 2 * self.words)
        view = rows.view([("w", np.uint64, (2 * self.words,))]).ravel()
        _, first = np.unique(view, return_index=True)
        return rows[np.sort(first)]

    @_runtime.threaded
    def solve(self, nroots=1, *, references=None, spin_parity=None, epsilons=(1e-3, 3e-4, 1e-4, 5e-5),
              pt2_epsilon=None, pt2_batches=1, pt2_deterministic=None, pt2_samples=200, pt2_sample_batches=20,
              seed=0, residual_tol=1e-5, max_space=None, max_cycles=12, new_fraction=0.01,
              max_determinants=None, callback=None, checkpoint=None, initial=None):
        """Roots `0..nroots-1` of the selected space for each epsilon in
        turn (each space grows from the previous one). `references` (rows;
        default the aufbau determinant) fix the irrep; `spin_parity` +1/-1
        selects even/odd total spin (needs nalpha == nbeta). With
        `pt2_epsilon`, each stage gets Epstein-Nesbet corrections in
        `pt2_batches` passes (more passes, less memory). With
        `pt2_deterministic` (> pt2_epsilon) the correction is semistochastic:
        exact above `pt2_deterministic`, plus E2[pt2_epsilon] -
        E2[pt2_deterministic] estimated from `pt2_sample_batches` batches of
        `pt2_samples` determinants (seeded by `seed`; `stage.pt2_error`).
        `max_determinants` ends the series at the first stage that would
        outgrow it (that stage keeps the space reached so far). Davidson
        stops at `residual_tol` (the energy error is about its square over
        the gap to the next root: 1e-5 gives about 1e-9 Eh).
        `checkpoint(stage, determinants, vectors)` runs after each stage;
        `initial=(determinants, vectors)` (from a checkpoint or a result;
        vectors may be None) starts the series from that space instead of
        the references, e.g. to resume with the remaining epsilons."""
        if spin_parity not in (None, 1, -1):
            raise ValueError("spin_parity must be None, +1 or -1")
        if spin_parity is not None and self.nalpha != self.nbeta:
            raise ValueError("Spin parity needs equal alpha and beta counts")
        threads = self._threads()
        refs = self.aufbau()[None] if references is None else np.asarray(references, dtype=np.uint64)
        refs = refs.reshape(-1, 2 * self.words)
        target = {self.irrep(r) for r in refs}
        if len(target) != 1:
            raise ValueError("The references must share one irrep")
        partners = spin_parity is not None
        if spin_parity == -1 and all((r[:self.words] == r[self.words:]).all() for r in refs):
            # Closed-shell determinants have no odd-spin component: start from
            # their single excitations of the same irrep instead.
            refs = np.concatenate([self.single_excitations(r, irrep=target.copy().pop()) for r in refs])
            if len(refs) == 0:
                raise ValueError("No open-shell determinant of the references' irrep")
        dets = self._unique(np.concatenate([refs, self.swapped(refs)]) if partners else refs)
        vectors = None
        if initial is not None:
            dets = np.ascontiguousarray(initial[0], dtype=np.uint64).reshape(-1, 2 * self.words)
            if initial[1] is not None:
                vectors = np.ascontiguousarray(initial[1], dtype=np.float64).reshape(nroots, len(dets))
        # Grow the reference space until it holds the roots comfortably.
        while initial is None and len(dets) < 4 * nroots + 4:
            space = self._space(dets)
            try:
                buffer = self.kernels.sci_select(space, np.ones(len(dets)).ctypes.data, 1e-2, int(partners), threads)
            finally:
                self.kernels.sci_space_free(space)
            new = self._take(buffer)
            if len(new) == 0:
                break
            dets = np.concatenate([dets, new])
        # The Hamiltonian of the previous cycle's space (a prefix of the
        # current one) is extended rather than rebuilt; it is released before
        # PT2, whose hash maps need the memory.
        try:
            return self._series(dets, vectors, nroots, epsilons, spin_parity, partners, pt2_epsilon, pt2_batches,
                                pt2_deterministic, pt2_samples, pt2_sample_batches, seed, residual_tol, max_space,
                                max_cycles, new_fraction, max_determinants, callback, checkpoint)
        finally:
            if self._kept is not None:
                self.kernels.sci_ham_free(self._kept)
                self._kept = None

    _kept = None
    _kept_n = 0

    def _hamiltonian(self, space, size):
        threads = self._threads()
        if self._kept is not None and self._kept_n <= size:
            ham = self.kernels.sci_hamiltonian_extend(space, self._kept, self._kept_n, 1e-15, threads)
        else:
            ham = self.kernels.sci_hamiltonian(space, 1e-15, threads)
        if self._kept is not None:
            self.kernels.sci_ham_free(self._kept)
            self._kept = None
        if not ham:
            raise MemoryError("Could not build the Hamiltonian")
        return ham

    def _series(self, dets, vectors, nroots, epsilons, spin_parity, partners, pt2_epsilon, pt2_batches,
                pt2_deterministic, pt2_samples, pt2_sample_batches, seed, residual_tol, max_space, max_cycles,
                new_fraction, max_determinants, callback, checkpoint):
        threads = self._threads()
        stages = []
        capped = False
        for eps in epsilons:
            cycles = 0
            seconds = dict(hamiltonian=0.0, davidson=0.0, select=0.0, pt2=0.0)
            while True:
                cycles += 1
                space = self._space(dets)
                try:
                    start = time.perf_counter()
                    ham = self._hamiltonian(space, len(dets))
                    nnz = int(self.kernels.sci_ham_nnz(ham))
                    seconds["hamiltonian"] += time.perf_counter() - start
                    self._kept, self._kept_n = ham, len(dets)
                    start = time.perf_counter()
                    energies, vectors = self._diagonalize(ham, space, len(dets), nroots, vectors, spin_parity,
                                                          residual_tol, max_space)
                    seconds["davidson"] += time.perf_counter() - start
                    start = time.perf_counter()
                    cmax = np.ascontiguousarray(np.max(np.abs(vectors), axis=0))
                    buffer = self.kernels.sci_select(space, cmax.ctypes.data, float(eps), int(partners), threads)
                    new = self._take(buffer)
                    seconds["select"] += time.perf_counter() - start
                    capped = max_determinants is not None and len(dets) + len(new) > max_determinants
                    done = len(new) <= new_fraction * len(dets) or cycles >= max_cycles or capped
                    if done and capped and cycles == 1 and stages and stages[-1].determinants == len(dets):
                        pass        # the cap stopped growth: the previous stage already holds this space
                    elif done:
                        s2 = np.zeros(nroots)
                        if self.nalpha == self.nbeta:
                            self.kernels.sci_s2(space, vectors.ctypes.data, nroots, s2.ctypes.data, threads)
                        pt2 = np.zeros(nroots)
                        pt2_error = np.zeros(nroots)
                        pt2_norm = np.zeros(nroots)
                        externals = 0
                        if pt2_epsilon is not None:
                            self.kernels.sci_ham_free(self._kept)
                            self._kept = None
                            start = time.perf_counter()
                            stats = np.zeros(2, dtype=np.int64)
                            electronic = np.ascontiguousarray(energies - self.constant)
                            exact_eps = pt2_epsilon if pt2_deterministic is None else pt2_deterministic
                            self.kernels.sci_pt2(space, vectors.ctypes.data, nroots, electronic.ctypes.data,
                                                 float(exact_eps), int(pt2_batches), pt2.ctypes.data,
                                                 pt2_norm.ctypes.data, stats.ctypes.data, threads)
                            externals = int(stats[0])
                            if pt2_deterministic is not None and pt2_deterministic > pt2_epsilon:
                                mean = np.zeros(nroots)
                                norm = np.zeros(nroots)
                                self.kernels.sci_pt2_stochastic(
                                    space, vectors.ctypes.data, nroots, electronic.ctypes.data, float(pt2_epsilon),
                                    float(pt2_deterministic), int(pt2_samples), int(pt2_sample_batches), int(seed),
                                    mean.ctypes.data, pt2_error.ctypes.data, norm.ctypes.data, stats.ctypes.data,
                                    threads)
                                pt2 = pt2 + mean
                                pt2_norm = pt2_norm + norm
                            seconds["pt2"] += time.perf_counter() - start
                        stage = Stage(float(eps), len(dets), nnz, energies.copy(), pt2, s2, cycles, seconds,
                                      externals, pt2_error, pt2_norm)
                        stages.append(stage)
                        if callback is not None:
                            callback(stage)
                        if checkpoint is not None:
                            checkpoint(stage, dets, vectors)
                finally:
                    self.kernels.sci_space_free(space)
                if done:
                    break
                dets = np.concatenate([dets, new])
                vectors = np.concatenate([vectors, np.zeros((nroots, len(new)))], axis=1)
            if capped:
                break
        return SCIResult(stages, dets, vectors, self.constant)

    def _take(self, buffer):
        try:
            rows = int(self.kernels.sci_buffer_rows(buffer))
            out = np.empty((rows, 2 * self.words), dtype=np.uint64)
            if rows:
                self.kernels.sci_buffer_copy(buffer, out.ctypes.data)
            return out
        finally:
            self.kernels.sci_buffer_free(buffer)

    def _diagonalize(self, ham, space, size, nroots, previous, spin_parity, residual_tol, max_space):
        threads = self._threads()
        diagonal = np.empty(size)
        self.kernels.sci_ham_diagonal(ham, diagonal.ctypes.data)
        if spin_parity is not None:
            partner = np.empty(size, dtype=np.int64)
            self.kernels.sci_partners(space, partner.ctypes.data, threads)
            if (partner < 0).any():
                raise RuntimeError("The space is not closed under alpha-beta swaps")

            def project(v):
                return 0.5 * (v + spin_parity * v[partner])
        else:
            def project(v):
                return v
        if size <= 2 * nroots + 1 or size <= 400:
            # Small spaces: a dense eigensolver on the projected Hamiltonian.
            basis = np.eye(size)
            dense = np.empty((size, size))
            for j in range(size):
                y = np.empty(size)
                self.kernels.sci_matvec(ham, np.ascontiguousarray(basis[j]).ctypes.data, y.ctypes.data, threads)
                dense[:, j] = y
            if spin_parity is not None:
                P = np.array([project(basis[j]) for j in range(size)]).T
                values, vecs = np.linalg.eigh(P @ dense @ P)
                keep = [k for k in range(size) if np.linalg.norm(P @ vecs[:, k]) > 0.5]
                values, vecs = values[keep], vecs[:, keep]
            else:
                values, vecs = np.linalg.eigh(dense)
            if len(values) < nroots:
                raise ValueError("The space holds fewer states than roots")
            return values[:nroots] + self.constant, np.ascontiguousarray(vecs[:, :nroots].T)

        def sigma(v):
            v = np.ascontiguousarray(v, dtype=np.float64)
            y = np.empty_like(v)
            self.kernels.sci_matvec(ham, v.ctypes.data, y.ctypes.data, threads)
            return y

        guesses = []
        if previous is not None:
            guesses = [project(np.ascontiguousarray(v)) for v in previous]
        else:
            for index in np.argsort(diagonal):
                g = np.zeros(size)
                g[index] = 1.0
                g = project(g)
                if np.linalg.norm(g) > 0.1 and all(abs(np.dot(g, o)) < 0.5 for o in guesses):
                    guesses.append(g / np.linalg.norm(g))
                if len(guesses) >= 2 * nroots:
                    break
        space_size = max_space or max(4 * nroots + 4, 24)
        outcome = krylov.davidson(sigma, diagonal, project, size=size, nroots=nroots, xp=np,
                                  residual_tol=residual_tol, max_iter=200, max_space=space_size,
                                  guesses=guesses, minimum=-np.inf)
        if not outcome.converged:
            raise RuntimeError("Davidson did not converge")
        roots = np.array([np.asarray(v).real for v in outcome.vectors])
        roots /= np.linalg.norm(roots, axis=1)[:, None]
        return np.asarray(outcome.values).real + self.constant, np.ascontiguousarray(roots)

    # ----- properties ---------------------------------------------------------------------
    @_runtime.threaded
    def density_matrix(self, result, weights=None):
        """Spin-summed one-particle density matrix of the final roots,
        averaged with `weights` (default equal)."""
        nroots = len(result.vectors)
        w = np.full(nroots, 1.0 / nroots) if weights is None else np.asarray(weights, dtype=np.float64)
        space = self._space(result.determinants)
        try:
            out = np.empty((self.norb, self.norb))
            vectors = np.ascontiguousarray(result.vectors)
            self.kernels.sci_rdm1(space, vectors.ctypes.data, nroots, np.ascontiguousarray(w).ctypes.data,
                                  out.ctypes.data, self._threads())
            return out
        finally:
            self.kernels.sci_space_free(space)

    def natural_orbitals(self, result, weights=None):
        """(occupations, coefficients): eigenvectors of the averaged density
        matrix within each irrep, by decreasing occupation (columns in the
        current orbital basis), and the irreps of the new orbitals."""
        rdm = self.density_matrix(result, weights)
        occupations = np.zeros(self.norb)
        coefficients = np.zeros((self.norb, self.norb))
        labels = np.zeros(self.norb, dtype=np.int64)
        columns = []
        for label in np.unique(self.irreps):
            idx = np.flatnonzero(self.irreps == label)
            values, vecs = np.linalg.eigh(rdm[np.ix_(idx, idx)])
            for k in range(len(idx)):
                columns.append((values[k], label, idx, vecs[:, k]))
        columns.sort(key=lambda item: -item[0])
        for col, (value, label, idx, vec) in enumerate(columns):
            occupations[col] = value
            coefficients[idx, col] = vec
            labels[col] = label
        return occupations, coefficients, labels

    # ----- polarizabilities -------------------------------------------------------------
    def _dipole_irreps(self, dipoles, dipole_irreps):
        from .finite_field import _irreps
        return _irreps(dipoles, self.irreps if self._labels_given else None, dipole_irreps)

    def _perturbed(self, h):
        """A context of the same two-electron Hamiltonian with another h (no labels)."""
        return SelectedCI(h, self._eri, (self.nalpha, self.nbeta), self.constant, cutoff=self._cutoff,
                          cpu_threads=self.cpu_threads)

    def _one_body(self, d):
        """A context of the one-electron operator sum_pq d_pq E_pq alone (its matrix in a space)."""
        return SelectedCI(d, np.zeros_like(self._eri), (self.nalpha, self.nbeta), 0.0, cutoff=self._cutoff,
                          cpu_threads=self.cpu_threads)

    def _select_from(self, context, dets, cmax, eps, partners):
        """Determinants outside `dets` with |X_ai| cmax_i >= eps for the
        operator X of `context` (its Hamiltonian: H, or a one-body operator)."""
        space = context._space(dets)
        try:
            buffer = context.kernels.sci_select(space, np.ascontiguousarray(cmax).ctypes.data, float(eps),
                                                int(partners), self._threads())
        finally:
            context.kernels.sci_space_free(space)
        return context._take(buffer)

    def _response(self, dets, operators, frequencies, spin_parity, tol, guess, irreps):
        """Ground state and linear response in the fixed space `dets`:
        (SCIPolarizability, ground vector, solutions of the first frequency
        per component (None for a zero component))."""
        start = time.perf_counter()
        threads = self._threads()
        dets = np.ascontiguousarray(dets, dtype=np.uint64)
        n = len(dets)
        space = self._space(dets)
        handles = []
        try:
            ham = self.kernels.sci_hamiltonian(space, 1e-15, threads)
            handles.append((self.kernels, ham))
            if guess is not None and len(guess) <= n:
                guess = np.concatenate([guess, np.zeros(n - len(guess))])[None]
            else:
                guess = None
            energies, vectors = self._diagonalize(ham, space, n, 1, guess, spin_parity, min(tol, 1e-9), None)
            e0 = energies[0] - self.constant
            c0 = vectors[0]
            diagonal = np.empty(n)
            self.kernels.sci_ham_diagonal(ham, diagonal.ctypes.data)
            partner = None
            if spin_parity is not None:
                partner = np.empty(n, dtype=np.int64)
                self.kernels.sci_partners(space, partner.ctypes.data, threads)
            sources = []
            for op in operators:
                op_space = op._space(dets)
                try:
                    op_ham = op.kernels.sci_hamiltonian(op_space, 0.0, threads)
                finally:
                    op.kernels.sci_space_free(op_space)
                handles.append((op.kernels, op_ham))
                y = np.empty(n)
                op.kernels.sci_matvec(op_ham, np.ascontiguousarray(c0).ctypes.data, y.ctypes.data, threads)
                sources.append(y)
            moments = np.array([c0 @ y for y in sources])
            sources = [y - m * c0 for y, m in zip(sources, moments)]

            def project(v):
                v = v - (c0 @ v) * c0
                if partner is not None:
                    v = 0.5 * (v + spin_parity * v[partner])
                return v

            tensors = np.zeros((len(frequencies), 3, 3))
            residuals = np.zeros((len(frequencies), 3, 2))
            first = [None, None, None]
            for f, w in enumerate(frequencies):
                for j in range(3):
                    if np.linalg.norm(sources[j]) == 0:
                        continue
                    solutions = []
                    for side, shift in enumerate((e0 + w, e0 - w) if w else (e0,)):
                        def action(v, shift=shift):
                            v = project(np.ascontiguousarray(v))
                            out = np.empty(n)
                            self.kernels.sci_matvec(ham, v.ctypes.data, out.ctypes.data, threads)
                            return project(out - shift * v)
                        precondition = diagonal - shift
                        precondition = np.where(np.abs(precondition) < 1e-3, np.copysign(1e-3, precondition),
                                                precondition)
                        x, residual, _ = krylov.gmres(action, sources[j], precondition, np, tol=tol, restart=40,
                                                      maxiter=2000)
                        residuals[f, j, side] = residual
                        solutions.append(project(x))
                    if f == 0:
                        first[j] = solutions[0]
                    for i in range(3):
                        if irreps is not None and irreps[i] != irreps[j]:
                            continue
                        value = sum(sources[i] @ x for x in solutions)
                        tensors[f, i, j] = 2 * value if not w else value
                tensors[f] = 0.5 * (tensors[f] + tensors[f].T)
            pol = SCIPolarizability(n, energies[0], moments, tuple(frequencies), tensors, residuals,
                                    time.perf_counter() - start)
            return pol, c0, first
        finally:
            for kernels, handle in handles:
                kernels.sci_ham_free(handle)
            self.kernels.sci_space_free(space)

    @_runtime.threaded
    def polarizability(self, dipoles, *, frequencies=(0.0,), references=None, spin_parity=1,
                       epsilons=(1e-3, 3e-4, 1e-4), dipole_irreps=None, pt2_epsilon=None, max_determinants=None,
                       max_cycles=8, new_fraction=0.01, tol=1e-9, spaces=None, callback=None):
        """Dipole polarizabilities alpha_ij(w) of the selected-CI ground state.

        Each stage's space holds the ground state's selected determinants at
        that epsilon and the response determinants: the ground state and the
        static response vectors x_j (normalized) are solved in the space, and
        every determinant a with |<a|mu_j|i> c_i| >= eps or
        |<a|H|i> x_ji| >= eps for some i of the space is added, until fewer
        than `new_fraction` are new (at most `max_cycles` rounds). In the
        final space alpha comes from linear response: alpha_ij(w) =
        <0|mu_i (H - E_0 - w)^-1 Q mu_j|0> + (w -> -w), Q projecting out
        |0> (GMRES to relative residual `tol`): the exact polarizability of
        the space's variational state (the zero-field limit of its
        finite-field energies), equal to full CI in the full space.
        `frequencies` (au) below the first excitation give dynamic
        polarizabilities. `dipoles` are three symmetric one-electron
        matrices in the orbitals of h (for example the electronic -r of
        `pyscf_interface.dipole_integrals`). With `spaces` (a list of
        determinant arrays, for example `[r.determinants for r in ...]`),
        only the linear response runs, in each given space. Returns a list
        of `SCIPolarizability`, one per stage or space (`.space` holds its
        determinants)."""
        dipoles = _symmetric(dipoles)
        frequencies = tuple(float(w) for w in frequencies)
        if not frequencies or any(not np.isfinite(w) or w < 0 for w in frequencies):
            raise ValueError("Frequencies must be finite and nonnegative")
        irreps = self._dipole_irreps(dipoles, dipole_irreps)
        operators = [self._one_body(d) for d in dipoles]
        partners = spin_parity is not None
        results = []
        guess = None
        if spaces is not None:
            for dets in spaces:
                pol, guess, _ = self._response(dets, operators, frequencies, spin_parity, tol, guess, irreps)
                pol.space = np.asarray(dets)
                results.append(pol)
                if callback is not None:
                    callback(pol)
            return results
        ground = self.solve(1, references=references, spin_parity=spin_parity, epsilons=epsilons,
                            pt2_epsilon=pt2_epsilon, max_determinants=max_determinants)
        union = None
        static = (0.0,) + tuple(w for w in frequencies if w)
        for stage in ground.stages:
            base = ground.determinants[:stage.determinants]
            union = base if union is None else self._unique(np.concatenate([union, base]))
            for cycle in range(max_cycles):
                pol, guess, first = self._response(union, operators, static, spin_parity, tol, guess, irreps)
                # Every component: the dipole couplings bring in its irrep's determinants.
                fresh = [self._select_from(op, union, np.abs(guess), stage.epsilon, partners) for op in operators]
                xs = [np.abs(x) / np.linalg.norm(x) for x in first if x is not None and np.linalg.norm(x) > 0]
                if xs:
                    fresh.append(self._select_from(self, union, np.max(xs, axis=0), stage.epsilon, partners))
                fresh = [f for f in fresh if len(f)]
                new = self._unique(np.concatenate(fresh)) if fresh else np.empty((0, 2 * self.words), np.uint64)
                done = len(new) <= new_fraction * len(union) or cycle == max_cycles - 1 or (
                    max_determinants is not None and len(union) + len(new) > max_determinants)
                if done:
                    break
                union = np.concatenate([union, new])
            keep = [0] + [k for k, w in enumerate(static) if w in frequencies and w]
            if 0.0 not in frequencies:
                keep = keep[1:]
            pol.frequencies_au = tuple(static[k] for k in keep)
            pol.tensors_au = pol.tensors_au[keep]
            pol.residuals = pol.residuals[keep]
            pol.pt2 = float(stage.pt2[0])
            pol.space = union.copy()
            results.append(pol)
            if callback is not None:
                callback(pol)
        return results

    def field_solver_factory(self, determinants, *, spin_parity=1, residual_tol=1e-9):
        """`solver_factory(h, orbital_irreps)` for `finite_field_polarizability`:
        each field point diagonalizes its Hamiltonian in the fixed space
        `determinants` (for example a `polarizability` result's `.space`), so the field
        energies vary smoothly and their second differences give that
        space's polarizability (equal to `polarizability`'s static tensor in
        the same space, up to the field extrapolation)."""
        dets = np.ascontiguousarray(determinants, dtype=np.uint64)
        parent = self

        class _Iteration:
            def __init__(self, delta):
                self.delta_energy = delta

        class _Result:
            def __init__(self, energy, vector):
                self.energy, self.converged = energy, True
                self.iterations = [_Iteration(0.0)]
                self.amplitudes = (vector,)

        class _FixedSpace:
            def __init__(self, h):
                self.sci = parent._perturbed(h)

            def solve(self, initial=None, **options):
                guess = None if initial is None else np.asarray(initial[0], dtype=np.float64)[None]
                res = self.sci.solve(1, references=dets[:1], spin_parity=spin_parity, epsilons=(1e3,),
                                     new_fraction=1.0, initial=(dets, guess), residual_tol=residual_tol)
                if len(res.determinants) != len(dets):
                    raise RuntimeError("The fixed space grew")
                return _Result(float(res.energies[0]), res.vectors[0].copy())

        return lambda h, orbital_irreps=None: _FixedSpace(h)


def _symmetric(matrices):
    m = np.asarray(matrices, dtype=np.float64)
    if m.ndim != 3 or m.shape[0] != 3 or not np.allclose(m, m.transpose(0, 2, 1), atol=1e-10):
        raise ValueError("Expected three symmetric dipole matrices")
    return np.ascontiguousarray(m)


def rotate_integrals(h, eri, coefficients):
    """h and chemists' eri in the orbitals `coefficients` (columns)."""
    c = np.asarray(coefficients, dtype=np.float64)
    n = c.shape[0]
    h2 = c.T @ h @ c
    g = np.asarray(eri, dtype=np.float64).reshape(n, n, n, n)
    g = np.tensordot(g, c, axes=([3], [0]))
    g = np.tensordot(g, c, axes=([2], [0])).transpose(0, 1, 3, 2)
    g = np.tensordot(g, c, axes=([1], [0])).transpose(0, 3, 1, 2)
    g = np.tensordot(c, g, axes=([0], [0]))
    return h2, np.ascontiguousarray(g)
