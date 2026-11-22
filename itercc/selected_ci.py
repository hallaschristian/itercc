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
maps a spin-S, M_S = 0 state to (-1)^S times itself. The variational
problem then lives on one function per swap pair, (D + sign D') / sqrt(2)
(`spin_pairs`, the default): half the rows and about half the nonzeros of
the determinant Hamiltonian, and Davidson vectors of half the length; the
roots come back as determinant vectors. Orbital irreps (Abelian
labels, XOR products) keep every determinant in the references' irrep. For
linear molecules, `orbital_lz` (the m of complex, Lz-adapted orbitals from
`itercc.axial`) keeps every determinant's M_L instead (sums of m), a finer
label than C2v's: M_L = 0 holds Sigma states, where A1 also holds Delta.

Dipole polarizabilities come from linear response in a selected space
(`SelectedCIResponse`, with the `DipoleResponse` interface and results;
`SelectedCI.polarizability` grows the space with dipole- and
response-coupled determinants).

The C kernels (core/src/cpu/selected_ci.cpp) hold the integrals, the sorted
excitation lists, the sparse Hamiltonian (upper triangle; values in single
precision from `SINGLE_FROM` determinants, column gaps in about 1.5 bytes),
the one-body operator matrices, the selections and the PT2 hash maps;
results do not depend on the thread count.
"""

from dataclasses import dataclass, field
import time

import numpy as np

from . import _core, _cpu_kernels, _cpu_quadruples, _runtime, krylov
from .response import HC_EH_NM, Polarizability, _shifted_guesses


# Spaces from this size store the Hamiltonian's values in single precision
# (hamiltonian_precision="auto"): a third less memory and memory traffic.
SINGLE_FROM = 1_000_000
# Intermediate rounds of a selection only steer it: their roots (and the
# polarizability's static responses) are solved loosely, and each stage's
# final space is solved again to the requested tolerance. (A two-root CuF
# series at 1e-4 throughout: Davidson 25% faster, energies within 3e-8 Eh.)
# (Looser steering selects worse response spaces: at 5e-2 the polarizability
# of a CuF space of 0.9 million determinants fell by 0.05 au.)
_SELECT_GROUND_TOL = 1e-4
_SELECT_RESPONSE_TOL = 1e-2


# From this size a stage's first growth sorts the grown space by alpha string,
# then beta string: connected determinants then lie close together, and
# products with the Hamiltonian, limited by their scattered reads, run 1.5x
# faster (CuF, 1.8 and 3.5 million determinants). Later rounds append.
SORT_FROM = 200_000


def _locality_order(dets, words):
    """The order of `dets` by alpha string, then beta string."""
    keys = [dets[:, words + k] for k in range(words)] + [dets[:, k] for k in range(words)]
    return np.lexsort(keys)


# Determinants in the exactly inverted block of the preconditioner.
BLOCK_PRECONDITIONER = 1500


# With memory_budget_gib="auto" the series may use this fraction of the
# host's memory limit (`_runtime.host_memory_limit`: cgroup, Slurm
# allocation, NUMA nodes) for the next space's matrices and vectors.
MEMORY_FRACTION = 0.85


class _MemoryPlan:
    """Projected memory of a grown space against a budget (bytes, or None).

    A space's matrices grow faster than the space (each determinant finds
    more connected ones): bytes ~ n^p, p fitted to the last two builds of
    the series that differ by 10% or more (CuF, 27 to 43 million
    determinants: 1.26), else `EXPONENT`. Besides its matrices a space
    holds `per_determinant` bytes per determinant (solver vectors, product
    buffers, indices). glibc keeps blocks below 32 MiB that are freed during
    an extension in per-thread heaps (`_core.keep_freed_memory`), so an
    extension can hold the old and the new matrices at once: when both would
    not fit, the old ones are freed (and the free heap returned to the
    kernel) and the new ones built from scratch."""

    EXPONENT = 1.35

    def __init__(self, budget):
        self.budget = budget
        self.history = []               # (determinants, matrix bytes)

    def record(self, size, nbytes):
        self.history.append((int(size), int(nbytes)))

    def projected(self, size):
        """Matrix bytes of a space of `size` determinants (0 before any build)."""
        if not self.history:
            return 0
        n1, b1 = self.history[-1]
        exponent = self.EXPONENT
        for n0, b0 in reversed(self.history[:-1]):
            if n1 >= 1.1 * n0 and b0 > 0:
                exponent = min(max(np.log(b1 / b0) / np.log(n1 / n0), 1.0), 1.6)
                break
        return b1 * (size / n1) ** exponent

    def check(self, size, per_determinant, held=0):
        """(fits, extends): whether a space of `size` determinants fits the
        budget, and whether it also fits next to `held` bytes of the current
        matrices (so they can be extended)."""
        if self.budget is None or not self.history:
            return True, True
        need = self.projected(size) + per_determinant * size
        return need <= self.budget, need + held <= self.budget


class _Pairs:
    """The symmetry-adapted basis of a space closed under alpha-beta swaps
    (spin parity `sign`) and, with `mirror_sign`, the reflection of axial
    orbitals: one function per orbit of determinants (the normalized
    projection of its lowest-index member; orbits whose projection vanishes,
    such as closed determinants for sign -1, are left out), in the order of
    those members (the core's `sci_hamiltonian_pairs`)."""

    def __init__(self, sci, space, size, sign, mirror_sign=None):
        orbit = np.empty(size, dtype=np.int64)
        coef = np.empty(size)
        self.sign = float(sign or 0.0)
        self.mirror_sign = float(mirror_sign or 0.0)
        count = int(sci.kernels.sci_orbits(space, self.sign, self.mirror_sign, orbit.ctypes.data, coef.ctypes.data,
                                           sci._threads()))
        if count < 0:
            raise RuntimeError("The space is not closed under the alpha-beta swap and reflection")
        self.size, self.count, self.orbit = size, count, orbit
        self.index = np.flatnonzero(orbit >= 0)
        self.member_orbit = orbit[self.index]
        self.member_coef = coef[self.index]
        self.norm = np.sqrt(np.bincount(self.member_orbit, self.member_coef ** 2, minlength=count))
        _, first = np.unique(self.member_orbit, return_index=True)
        self.reps = self.index[first]

    def to_pairs(self, vectors):
        """(k, size) determinant vectors -> (k, orbits): their projections."""
        v = np.asarray(vectors, dtype=np.float64)
        out = np.empty((len(v), self.count))
        for k in range(len(v)):
            out[k] = np.bincount(self.member_orbit, self.member_coef * v[k, self.index],
                                 minlength=self.count) / self.norm
        return out

    def to_full(self, vectors):
        """(k, orbits) -> (k, size) determinant vectors (zero outside orbits)."""
        v = np.asarray(vectors, dtype=np.float64)
        scale = self.member_coef / self.norm[self.member_orbit]
        out = np.zeros((len(v), self.size))
        for k in range(len(v)):
            out[k, self.index] = scale * v[k, self.member_orbit]
        return out

    def rows(self, full_rows):
        """The orbits holding any of the determinant indices `full_rows`."""
        found = self.orbit[np.asarray(full_rows, dtype=np.int64)]
        return np.unique(found[found >= 0])

    def determinants(self, dets, rows):
        """The determinants of the orbits `rows`: their representatives in
        order, then the other members (so an orbit space built from them
        numbers the orbits as `rows`)."""
        rows = np.asarray(rows, dtype=np.int64)
        chosen = np.zeros(self.count, dtype=bool)
        chosen[rows] = True
        reps = self.reps[rows]
        members = self.index[chosen[self.member_orbit]]
        others = np.setdiff1d(members, reps)
        return np.ascontiguousarray(np.concatenate([dets[reps], dets[others]]))


class _BlockPreconditioner:
    """(H - e)^-1 approximated by the exact inverse on a block of leading
    determinants (`rows`, by weight in the vectors being solved for) and by
    the diagonal elsewhere: the leading determinants carry the strong
    couplings that slow diagonal-preconditioned solvers. With `pairs` the
    rows are spin pairs (of determinants `determinants`)."""

    def __init__(self, sci, determinants, diagonal, rows, pairs=None):
        self.diagonal, self.rows = np.ascontiguousarray(diagonal, dtype=np.float64), np.ascontiguousarray(rows, dtype=np.int64)
        self.kernels, self.threads = sci.kernels, sci._threads()
        m = len(self.rows)
        block_dets = (np.ascontiguousarray(determinants[self.rows]) if pairs is None
                      else pairs.determinants(determinants, self.rows))
        space = sci._space(block_dets)
        try:
            ham = (sci.kernels.sci_hamiltonian(space, 0.0, 0, sci._threads()) if pairs is None
                   else sci.kernels.sci_hamiltonian_pairs(space, pairs.sign, pairs.mirror_sign, 0.0, 0,
                                                          sci._threads()))
            try:
                dense = np.empty((m, m))
                sci.kernels.sci_ham_dense(ham, dense.ctypes.data)
            finally:
                sci.kernels.sci_ham_free(ham)
        finally:
            sci.kernels.sci_space_free(space)
        self.values, self.vectors = np.linalg.eigh(dense)

    @classmethod
    def leading(cls, sci, determinants, diagonal, weights, size=None, pairs=None):
        """The block of the `size` determinants (pairs) with the largest weight (max |v| over the given vectors)."""
        size = BLOCK_PRECONDITIONER if size is None else size
        n = len(diagonal)
        if size <= 0 or n == 0:
            return None
        if n <= size:
            rows = np.arange(n)
        else:
            rows = np.sort(np.argpartition(-np.asarray(weights), size - 1)[:size])
        return cls(sci, determinants, diagonal, rows, pairs)

    def denominators(self, values, e, positive):
        d = values - e
        if positive:
            return np.maximum(np.abs(d), 1e-3)
        return np.where(np.abs(d) < 1e-3, np.copysign(1e-3, d), d)

    def apply(self, r, e, positive):
        """M^-1 r with M = H - e (positive: |H - e|, a positive definite
        preconditioner) for r of shape (n,) or (n, k)."""
        if not positive and r.dtype == np.float64 and r.flags.c_contiguous and self.kernels is not None:
            z = np.empty_like(r)        # the diagonal part on all threads
            k = 1 if r.ndim == 1 else r.shape[1]
            self.kernels.sci_block_divide(len(self.diagonal), k, self.diagonal.ctypes.data, float(e), 1e-3,
                                          r.ctypes.data, z.ctypes.data, self.threads)
        else:
            d = self.denominators(self.diagonal, e, positive)
            z = r / (d if r.ndim == 1 else d[:, None])
        block = self.vectors.T @ r[self.rows]
        dv = self.denominators(self.values, e, positive)
        z[self.rows] = self.vectors @ (block / (dv if r.ndim == 1 else dv[:, None]))
        return z


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

    `polarizabilities[k]` is the `response.Polarizability` (wavelength_nm,
    frequency_Eh, damping_Eh, tensor_au, solves) at the k-th wavelength, from
    `SelectedCIResponse` in that space. `dipole_au` is <0|mu|0> of the dipole
    matrices as given (electronic -r)."""
    determinants: int
    energy: float
    dipole_au: np.ndarray
    polarizabilities: tuple
    seconds: float = 0.0
    pt2: float = 0.0                 # zero-field PT2 of the ground state at the stage (0 without PT2)
    space: np.ndarray = None         # the determinants of the space
    hamiltonian_bytes: int = 0       # compiled Hamiltonian and dipole matrices of the space

    @property
    def tensors_au(self):
        """The tensors stacked (real without damping)."""
        tensors = np.array([pol.tensor_au for pol in self.polarizabilities])
        return tensors.real if all(pol.damping_Eh == 0 for pol in self.polarizabilities) else tensors


class SelectedCI:
    """Selected CI of a real orbital Hamiltonian (h, chemists' eri (pq|rs))
    with `nelec` electrons (an even count with M_S = 0, or (nalpha, nbeta))."""

    def __init__(self, h, eri, nelec, constant=0.0, *, orbital_irreps=None, orbital_lz=None, orbital_mirror=None,
                 cutoff=1e-12, cpu_threads=None, hamiltonian_precision="auto", memory_budget_gib="auto",
                 spin_pairs=True):
        h = np.ascontiguousarray(h, dtype=np.float64)
        n = h.shape[0]
        if h.shape != (n, n) or not np.allclose(h, h.T, atol=1e-10):
            raise ValueError("h must be a real symmetric matrix")
        if orbital_irreps is not None and orbital_lz is not None:
            raise ValueError("Give orbital_irreps (real orbitals) or orbital_lz (axial orbitals), not both")
        self.axial = orbital_lz is not None
        eri = np.ascontiguousarray(np.asarray(eri, dtype=np.float64).reshape(n, n, n, n))
        # The core keeps one integral per permutation class (8-fold for real
        # orbitals, 4-fold for axial ones): check a sample.
        p, q, r, s = np.random.default_rng(0).integers(0, n, size=(4, min(4096, n ** 4)))
        sample = eri[p, q, r, s]
        tolerance = 1e-10 * max(1.0, float(np.abs(sample).max()))
        images = ((r, s, p, q), (q, p, s, r)) if self.axial else ((q, p, r, s), (p, q, s, r), (r, s, p, q))
        if any(np.abs(eri[a, b, c, d] - sample).max() > tolerance for a, b, c, d in images):
            raise ValueError("eri must have the 4-fold permutational symmetry of axial orbitals" if self.axial
                             else "eri must have the 8-fold permutational symmetry of real orbitals")
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
        if hamiltonian_precision not in ("auto", "single", "double"):
            raise ValueError('hamiltonian_precision must be "auto", "single" or "double"')
        self.hamiltonian_precision = hamiltonian_precision
        if memory_budget_gib != "auto" and memory_budget_gib is not None and not float(memory_budget_gib) > 0:
            raise ValueError('memory_budget_gib must be "auto", None or positive')
        self.memory_budget_gib = memory_budget_gib
        self.spin_pairs = bool(spin_pairs)
        self._h, self._eri, self._cutoff = h, eri, float(cutoff)
        self._labels_given = orbital_irreps is not None or orbital_lz is not None
        self.kernels = _cpu_kernels.library()
        if self.kernels is None:
            raise RuntimeError("SelectedCI needs the compiled core (see itercc._cpu_kernels)")
        self.norb, self.nalpha, self.nbeta = n, nalpha, nbeta
        self.constant = float(constant)
        self.cpu_threads = _runtime.thread_option(cpu_threads)
        if self.axial:
            self.irreps = np.ascontiguousarray(orbital_lz, dtype=np.int64)
            if self.irreps.shape != (n,):
                raise ValueError("orbital_lz must hold one m per orbital")
        else:
            self.irreps = (np.zeros(n, dtype=np.int64) if orbital_irreps is None
                           else np.ascontiguousarray(orbital_irreps, dtype=np.int64))
            if self.irreps.shape != (n,) or self.irreps.min() < 0 or self.irreps.max() > 7:
                raise ValueError("orbital_irreps must hold one Abelian irrep 0..7 per orbital")
        self._ctx = self.kernels.sci_create(n, nalpha, nbeta, h.ctypes.data, eri.ctypes.data,
                                            self.irreps.ctypes.data, int(self.axial), float(cutoff),
                                            self._threads())
        if not self._ctx:
            raise MemoryError("Could not create the selected-CI context")
        self.words = int(self.kernels.sci_words(self._ctx))
        self.mirror = None
        if orbital_mirror is not None:
            if not self.axial:
                raise ValueError("orbital_mirror needs axial orbitals (orbital_lz)")
            mirror = np.ascontiguousarray(orbital_mirror, dtype=np.int64)
            if mirror.shape != (n,) or (self.irreps[mirror] != -self.irreps).any():
                raise ValueError("orbital_mirror must map each orbital to one of opposite m")
            if self.kernels.sci_set_mirror(self._ctx, mirror.ctypes.data) != 0:
                raise ValueError("orbital_mirror must be an involution")
            self.mirror = mirror
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

    def _memory_plan(self):
        budget = self.memory_budget_gib
        if budget == "auto":
            limit = _runtime.host_memory_limit()
            budget = None if limit is None else MEMORY_FRACTION * limit / 2**30
        return _MemoryPlan(None if budget is None or not np.isfinite(budget) else float(budget) * 2**30)

    def _release(self, handle):
        """Free a compiled matrix and return the free heap to the kernel."""
        self.kernels.sci_ham_free(handle)
        _core.release_free_memory()

    def _single(self, size):
        """1 when a Hamiltonian of `size` determinants stores single-precision values."""
        return int(self.hamiltonian_precision == "single"
                   or (self.hamiltonian_precision == "auto" and size >= SINGLE_FROM))

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
        """The determinant's irrep (XOR of its orbitals' labels), or its M_L
        (their sum) in axial orbitals."""
        alpha, beta = self.occupations(row)
        if self.axial:
            return int(sum(int(self.irreps[p]) for p in (*alpha, *beta)))
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

    def mirrored(self, rows, chunk=1 << 20):
        """The reflections of determinant rows (orbital p -> orbital_mirror[p]):
        each row's bits unpacked, permuted and packed again (in chunks)."""
        rows = np.ascontiguousarray(rows, dtype=np.uint64).reshape(-1, 2 * self.words)
        bits = 64 * self.words
        perm = np.arange(2 * bits)
        for offset in (0, bits):
            perm[offset + self.mirror] = offset + np.arange(self.norb)      # new bit mirror[p] <- old bit p
        out = np.empty_like(rows)
        for start in range(0, len(rows), chunk):
            block = rows[start:start + chunk].view(np.uint8)
            unpacked = np.unpackbits(block, axis=1, bitorder="little")
            packed = np.packbits(unpacked[:, perm], axis=1, bitorder="little")
            out[start:start + chunk] = np.ascontiguousarray(packed).view(np.uint64)
        return out

    def _close(self, rows, mirror):
        """rows with their reflections (distinct) when `mirror`."""
        if not mirror or len(rows) == 0:
            return rows
        return self._unique(np.concatenate([rows, self.mirrored(rows)]))

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
              max_determinants=None, callback=None, checkpoint=None, initial=None, mirror_parity=None):
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
        Each stage starts from the previous stage's converged roots: it selects
        first and builds the Hamiltonian of the grown space only.
        `max_determinants` ends the series at the first stage that would
        outgrow it (that stage keeps the space reached so far), and so does
        the memory budget (`memory_budget_gib`): the series stops before a
        space whose projected Hamiltonian and vectors would not fit. From
        `SORT_FROM` determinants a stage's first growth sorts the grown space
        by alpha and beta string (faster products), so a stage's space is not a
        prefix of the next one's; `checkpoint` receives each stage's space.
        Each stage's roots are solved to Davidson residual `residual_tol`
        (the energy error is about its square over the gap to the next root:
        1e-5 gives about 1e-9 Eh); the rounds before, which only steer the
        selection, stop at 1e-4.
        `checkpoint(stage, determinants, vectors)` runs after each stage;
        `initial=(determinants, vectors)` (from a checkpoint or a result;
        vectors may be None) starts the series from that space instead of
        the references, e.g. to resume with the remaining epsilons: the roots
        are solved there first, unless `initial=(determinants, vectors,
        stage)` passes the checkpoint's stage too (its converged roots then
        select directly, exactly as the uninterrupted series).
        `mirror_parity` +1/-1 (axial orbitals with `orbital_mirror`, M_L = 0
        references) keeps Sigma+ or Sigma- states: spaces then contain each
        determinant's reflection, and the problem lives on one function per
        orbit under the swap and the reflection (half the rows again)."""
        if spin_parity not in (None, 1, -1):
            raise ValueError("spin_parity must be None, +1 or -1")
        if mirror_parity not in (None, 1, -1):
            raise ValueError("mirror_parity must be None, +1 or -1")
        if mirror_parity is not None and self.mirror is None:
            raise ValueError("mirror_parity needs orbital_mirror")
        if spin_parity is not None and self.nalpha != self.nbeta:
            raise ValueError("Spin parity needs equal alpha and beta counts")
        threads = self._threads()
        refs = self.aufbau()[None] if references is None else np.asarray(references, dtype=np.uint64)
        refs = refs.reshape(-1, 2 * self.words)
        target = {self.irrep(r) for r in refs}
        if len(target) != 1:
            raise ValueError("The references must share one irrep")
        if mirror_parity is not None and target != {0}:
            raise ValueError("mirror_parity needs M_L = 0 references")
        mirror = mirror_parity is not None
        partners = spin_parity is not None
        if spin_parity == -1 and all((r[:self.words] == r[self.words:]).all() for r in refs):
            # Closed-shell determinants have no odd-spin component: start from
            # their single excitations of the same irrep instead.
            refs = np.concatenate([self.single_excitations(r, irrep=target.copy().pop()) for r in refs])
            if len(refs) == 0:
                raise ValueError("No open-shell determinant of the references' irrep")
        dets = self._unique(np.concatenate([refs, self.swapped(refs)]) if partners else refs)
        dets = self._close(dets, mirror)
        vectors = given = None
        if initial is not None:
            dets = np.ascontiguousarray(initial[0], dtype=np.uint64).reshape(-1, 2 * self.words)
            if initial[1] is not None:
                vectors = np.ascontiguousarray(initial[1], dtype=np.float64).reshape(nroots, len(dets))
                if len(initial) > 2 and initial[2] is not None:
                    given = initial[2]
                    if given.determinants != len(dets) or len(given.energies) != nroots:
                        raise ValueError("The initial stage does not match the initial space and roots")
        # Grow the reference space until it holds the roots comfortably: count
        # the symmetry-adapted functions (closed determinants carry no odd
        # spin, reflection-invariant ones no Sigma- part).
        while initial is None:
            space = self._space(dets)
            try:
                if self._functions(space, len(dets), spin_parity, mirror_parity) >= 4 * nroots + 4:
                    break
                ones = np.ones(len(dets))          # held while the kernel reads it
                buffer = self.kernels.sci_select(space, ones.ctypes.data, 1e-2, int(partners), threads)
            finally:
                self.kernels.sci_space_free(space)
            new = self._close(self._take(buffer), mirror)
            if len(new) == 0:
                break
            dets = np.concatenate([dets, new])
        # The Hamiltonian of the previous cycle's space (a prefix of the
        # current one) is extended rather than rebuilt; it is released before
        # PT2, whose hash maps need the memory.
        try:
            return self._series(dets, vectors, given, nroots, epsilons, spin_parity, partners, pt2_epsilon, pt2_batches,
                                pt2_deterministic, pt2_samples, pt2_sample_batches, seed, residual_tol, max_space,
                                max_cycles, new_fraction, max_determinants, callback, checkpoint, mirror_parity)
        finally:
            if self._kept is not None:
                self.kernels.sci_ham_free(self._kept)
                self._kept = None

    _kept = None
    _kept_n = 0

    def _functions(self, space, size, spin_parity, mirror_parity):
        """The number of basis functions of a space for the given parities
        (orbits with a nonzero projection; determinants without them)."""
        if spin_parity is None and mirror_parity is None:
            return size
        orbit, coef = np.empty(size, dtype=np.int64), np.empty(size)
        count = int(self.kernels.sci_orbits(space, float(spin_parity or 0), float(mirror_parity or 0),
                                            orbit.ctypes.data, coef.ctypes.data, self._threads()))
        return size if count < 0 else count

    def _hamiltonian(self, space, size, group=None):
        """The space's Hamiltonian (over the orbits of `group` = (spin parity,
        reflection parity), 0 for an unused operation, if given), extending
        the kept one of the previous cycle."""
        threads = self._threads()
        single = self._single(size)
        extend = self._kept is not None and self._kept_n <= size
        if group is None:
            ham = (self.kernels.sci_hamiltonian_extend(space, self._kept, self._kept_n, 1e-15, single, threads)
                   if extend else self.kernels.sci_hamiltonian(space, 1e-15, single, threads))
        else:
            t, m = (float(x) for x in group)
            ham = (self.kernels.sci_hamiltonian_pairs_extend(space, self._kept, self._kept_n, t, m, 1e-15, single,
                                                             threads)
                   if extend else self.kernels.sci_hamiltonian_pairs(space, t, m, 1e-15, single, threads))
        if self._kept is not None:
            self._release(self._kept)
            self._kept = None
        if not ham:
            raise MemoryError("Could not build the Hamiltonian")
        return ham

    def _series(self, dets, vectors, given, nroots, epsilons, spin_parity, partners, pt2_epsilon, pt2_batches,
                pt2_deterministic, pt2_samples, pt2_sample_batches, seed, residual_tol, max_space, max_cycles,
                new_fraction, max_determinants, callback, checkpoint, mirror_parity=None):
        threads = self._threads()
        stages = []
        capped = False
        fresh = None            # rows without amplitudes yet (first-order fill)
        mirror = mirror_parity is not None
        # The reflection's orbits include the swap whenever the spin parity is kept.
        sign = spin_parity if (spin_parity is not None and (self.spin_pairs or mirror)) else None
        group = None
        if sign is not None or mirror:
            group = (0 if sign is None else sign, 0 if mirror_parity is None else mirror_parity)
        plan = self._memory_plan()
        # Davidson's subspace and images, the roots' product buffers (about
        # STRIPES/2 + 1 rows each), residuals, indices and determinant copies.
        subspace = max_space or max(4 * nroots + 4, 24)
        per_determinant = 8 * (2 * subspace + 37 * nroots + 6) + 48 * self.words
        # Roots converged in the current space select without a Hamiltonian:
        # those of the previous stage, or of a given checkpoint stage.
        solved = refined = given is not None
        steer = max(residual_tol, _SELECT_GROUND_TOL)
        energies = None if given is None else np.asarray(given.energies, dtype=np.float64)
        nnz = 0 if given is None else int(given.nonzeros)
        held = 0
        if nnz:
            plan.record(len(dets), nnz * (8 if self._single(len(dets)) else 12) + 16 * len(dets))
        for eps in epsilons:
            cycles = 0
            seconds = dict(hamiltonian=0.0, davidson=0.0, select=0.0, pt2=0.0)
            while True:
                cycles += 1
                space = self._space(dets)
                try:
                    if not solved:
                        start = time.perf_counter()
                        ham = self._hamiltonian(space, len(dets), group)
                        nnz = int(self.kernels.sci_ham_nnz(ham))
                        held = int(self.kernels.sci_ham_bytes(ham))
                        plan.record(len(dets), held)
                        seconds["hamiltonian"] += time.perf_counter() - start
                        self._kept, self._kept_n = ham, len(dets)
                        start = time.perf_counter()
                        energies, vectors = self._diagonalize(ham, space, len(dets), nroots, vectors, spin_parity,
                                                              steer, max_space, fresh, dets, pairs=group)
                        fresh = None
                        solved, refined = True, steer <= residual_tol
                        seconds["davidson"] += time.perf_counter() - start
                    start = time.perf_counter()
                    cmax = np.ascontiguousarray(np.max(np.abs(vectors), axis=0))
                    buffer = self.kernels.sci_select(space, cmax.ctypes.data, float(eps), int(partners), threads)
                    new = self._close(self._take(buffer), mirror)
                    seconds["select"] += time.perf_counter() - start
                    kept = held + 8 * 34 * nroots * len(dets) if self._kept is not None else 0
                    fits, extends = plan.check(len(dets) + len(new), per_determinant, kept)
                    capped = (max_determinants is not None and len(dets) + len(new) > max_determinants) or not fits
                    done = len(new) <= new_fraction * len(dets) or cycles >= max_cycles or capped
                    if done and capped and cycles == 1 and stages and stages[-1].determinants == len(dets):
                        pass        # the cap stopped growth: the previous stage already holds this space
                    elif done:
                        if not refined:         # the stage's space: its roots to residual_tol
                            start = time.perf_counter()
                            energies, vectors = self._diagonalize(ham, space, len(dets), nroots, vectors,
                                                                  spin_parity, residual_tol, max_space, None, dets,
                                                                  pairs=group)
                            refined = True
                            seconds["davidson"] += time.perf_counter() - start
                        s2 = np.zeros(nroots)
                        if self.nalpha == self.nbeta:
                            self.kernels.sci_s2(space, vectors.ctypes.data, nroots, s2.ctypes.data, threads)
                        pt2 = np.zeros(nroots)
                        pt2_error = np.zeros(nroots)
                        pt2_norm = np.zeros(nroots)
                        externals = 0
                        if pt2_epsilon is not None:
                            if self._kept is not None:
                                self._release(self._kept)
                                self._kept = None
                            start = time.perf_counter()
                            stats = np.zeros(2, dtype=np.int64)
                            electronic = np.ascontiguousarray(energies - self.constant)
                            exact_eps = pt2_epsilon if pt2_deterministic is None else pt2_deterministic
                            if group is None:
                                self.kernels.sci_pt2(space, vectors.ctypes.data, nroots, electronic.ctypes.data,
                                                     float(exact_eps), int(pt2_batches), pt2.ctypes.data,
                                                     pt2_norm.ctypes.data, stats.ctypes.data, threads)
                            else:       # each orbit of externals once
                                self.kernels.sci_pt2_orbits(space, vectors.ctypes.data, nroots, electronic.ctypes.data,
                                                            float(exact_eps), int(pt2_batches), float(group[0]),
                                                            float(group[1]), pt2.ctypes.data, pt2_norm.ctypes.data,
                                                            stats.ctypes.data, threads)
                            externals = int(stats[0])
                            if pt2_deterministic is not None and pt2_deterministic > pt2_epsilon:
                                mean = np.zeros(nroots)
                                norm = np.zeros(nroots)
                                common = (space, vectors.ctypes.data, nroots, electronic.ctypes.data, float(pt2_epsilon),
                                          float(pt2_deterministic), int(pt2_samples), int(pt2_sample_batches),
                                          int(seed))
                                outputs = (mean.ctypes.data, pt2_error.ctypes.data, norm.ctypes.data,
                                           stats.ctypes.data, threads)
                                if group is None:
                                    self.kernels.sci_pt2_stochastic(*common, *outputs)
                                else:
                                    self.kernels.sci_pt2_stochastic_orbits(*common, float(group[0]), float(group[1]),
                                                                           *outputs)
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
                old = len(dets)
                dets = np.concatenate([dets, new])
                vectors = np.concatenate([vectors, np.zeros((nroots, len(new)))], axis=1)
                if cycles == 1 and len(dets) >= SORT_FROM:
                    # A stage's first growth sorts the grown space (the new
                    # determinants among the old) and builds it afresh.
                    order = _locality_order(dets, self.words)
                    dets = np.ascontiguousarray(dets[order])
                    vectors = np.ascontiguousarray(vectors[:, order])
                    fresh = np.flatnonzero(order >= old)
                    if self._kept is not None:
                        self._release(self._kept)
                        self._kept = None
                else:
                    fresh = np.arange(old, len(dets))
                    if not extends and self._kept is not None:      # rebuilt: old and new would not fit
                        self._release(self._kept)
                        self._kept = None
                solved = False
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

    def _first_order(self, ham, vectors, rows, diagonal):
        """`vectors` (roots x size) whose entries at `rows` are new (zero): those
        get first-order amplitudes (H c)_a / (E - H_aa), E the Rayleigh
        quotient of the old part, from one pass over H for all roots."""
        k = len(vectors)
        threads = self._threads()
        block = _cpu_quadruples.copy(np.asarray(vectors, dtype=np.float64).T)
        image = np.empty_like(block)
        self.kernels.sci_matmat(ham, k, block.ctypes.data, image.ctypes.data, threads)
        out = np.array(vectors, dtype=np.float64)
        numerators, norms = np.zeros(k), np.zeros(k)
        if k <= 16:                 # the block-dot kernel's width
            self.kernels.sci_block_dot(len(block), k, block.ctypes.data, image.ctypes.data,
                                       numerators.ctypes.data, threads)
            self.kernels.sci_block_dot(len(block), k, block.ctypes.data, block.ctypes.data, norms.ctypes.data,
                                       threads)
        else:
            numerators, norms = np.einsum("ir,ir->r", block, image), np.einsum("ir,ir->r", block, block)
        for r in range(k):
            energy = numerators[r] / max(norms[r], 1e-300)
            denominator = energy - diagonal[rows]
            denominator = np.where(np.abs(denominator) < 1e-3, np.copysign(1e-3, denominator), denominator)
            out[r, rows] = image[rows, r] / denominator
        return out

    def _diagonalize(self, ham, space, size, nroots, previous, spin_parity, residual_tol, max_space, fill_from=None,
                     dets=None, pairs=None):
        """The lowest `nroots` roots of `ham` (determinant vectors back).
        With `pairs` = (spin parity, reflection parity), `ham` is the
        Hamiltonian over that group's orbits: the roots are solved there and
        expanded."""
        if pairs is not None and pairs is not False:
            basis = _Pairs(self, space, size, *pairs)
            if previous is not None:
                previous = basis.to_pairs(previous)
            if fill_from is not None:
                rows = np.arange(fill_from, size) if np.ndim(fill_from) == 0 else fill_from
                fill_from = basis.rows(rows)
            energies, vectors = self._solve_roots(ham, space, basis.count, nroots, previous, None, residual_tol,
                                                  max_space, fill_from, dets, basis)
            return energies, np.ascontiguousarray(basis.to_full(vectors))
        return self._solve_roots(ham, space, size, nroots, previous, spin_parity, residual_tol, max_space, fill_from,
                                 dets, None)

    def _solve_roots(self, ham, space, size, nroots, previous, spin_parity, residual_tol, max_space, fill_from, dets,
                     block_pairs):
        threads = self._threads()
        diagonal = np.empty(size)
        self.kernels.sci_ham_diagonal(ham, diagonal.ctypes.data)
        if previous is not None and fill_from is not None:
            # `fill_from`: the first new row (the rest are new), or the new rows' indices.
            rows = np.arange(fill_from, size) if np.ndim(fill_from) == 0 else np.asarray(fill_from, dtype=np.int64)
            if 0 < len(rows) < size:
                previous = self._first_order(ham, previous, rows, diagonal)
        if spin_parity is not None:
            partner = np.empty(size, dtype=np.int64)
            self.kernels.sci_partners(space, partner.ctypes.data, threads)
            if (partner < 0).any():
                raise RuntimeError("The space is not closed under alpha-beta swaps")

            def project(v):
                v = np.ascontiguousarray(v, dtype=np.float64)
                out = np.empty_like(v)
                self.kernels.sci_parity(size, partner.ctypes.data, float(spin_parity), v.ctypes.data,
                                        out.ctypes.data, threads)
                return out
        else:
            def project(v):
                return v
        if size <= 2 * nroots + 1 or size <= 400:
            # Small spaces: a dense eigensolver on the projected Hamiltonian.
            basis = np.eye(size)
            dense = np.empty((size, size))
            for j in range(size):
                y = np.empty(size)
                unit = np.ascontiguousarray(basis[j])
                self.kernels.sci_matvec(ham, unit.ctypes.data, y.ctypes.data, threads)
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
        weights = np.max(np.abs(np.asarray(guesses)), axis=0) if guesses else -diagonal
        block = (_BlockPreconditioner.leading(self, dets, diagonal, weights, pairs=block_pairs)
                 if dets is not None else None)

        def precondition(residual, energy, vector):
            # Olsen's correction t = M^-1 r - e M^-1 x with x . t = 0, M the block
            # approximation of H - energy (an exact block alone would return x).
            u = block.apply(residual, energy, positive=False)
            w = block.apply(vector, energy, positive=False)
            ratio = _dot(vector, u) / _dot(vector, w)
            return _cpu_quadruples.permuted_add(u, u, w, -1.0, ratio)        # -(u - ratio w)

        def sigma_block(rows):
            # The new directions of several roots in one pass over H (transposes on all threads).
            block_in = _cpu_quadruples.copy(np.asarray(rows, dtype=np.float64).T)
            block_out = np.empty_like(block_in)
            self.kernels.sci_matmat(ham, block_in.shape[1], block_in.ctypes.data, block_out.ctypes.data, threads)
            return _cpu_quadruples.copy(block_out.T)

        outcome = krylov.davidson(sigma, diagonal, project, size=size, nroots=nroots, xp=np,
                                  residual_tol=residual_tol, max_iter=200, max_space=space_size,
                                  guesses=guesses, minimum=-np.inf,
                                  precondition=precondition if block is not None else None,
                                  sigma_block=sigma_block if nroots > 1 else None)
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
            w = np.ascontiguousarray(w)
            self.kernels.sci_rdm1(space, vectors.ctypes.data, nroots, w.ctypes.data,
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
        if self.axial:
            # |Delta M_L| of each component (`axial.axial_dipoles`: x and z, y zero).
            return (1, 1, 0)
        from .finite_field import _irreps
        return _irreps(dipoles, self.irreps if self._labels_given else None, dipole_irreps)

    def _perturbed(self, h):
        """A context of the same two-electron Hamiltonian with another h (no labels)."""
        return SelectedCI(h, self._eri, (self.nalpha, self.nbeta), self.constant, cutoff=self._cutoff,
                          cpu_threads=self.cpu_threads, hamiltonian_precision=self.hamiltonian_precision,
                          memory_budget_gib=self.memory_budget_gib)

    def _difference(self, space, candidates):
        """The distinct rows of `candidates` outside the indexed space (compiled)."""
        candidates = np.ascontiguousarray(candidates, dtype=np.uint64).reshape(-1, 2 * self.words)
        if not len(candidates):
            return candidates
        return self._take(self.kernels.sci_difference(space, candidates.ctypes.data, len(candidates),
                                                      self._threads()))

    @_runtime.threaded
    def polarizability(self, dipoles, wavelengths_nm=(np.inf,), *, damping_Eh=0.0, references=None, spin_parity=1,
                       epsilons=(1e-3, 3e-4, 1e-4), dipole_irreps=None, pt2_epsilon=None, max_determinants=None,
                       max_cycles=8, new_fraction=0.01, tol=1e-5, ground_tol=1e-6, spaces=None, callback=None,
                       solve_options=None, initial=None, mirror_parity=None):
        """Dipole polarizabilities of the selected-CI ground state, one
        `SCIPolarizability` per stage (or per given space).

        Each stage's space holds the ground state's selected determinants at
        that epsilon and the response determinants: the ground state and the
        static response vectors x_j (normalized) are solved loosely in the
        space (they only steer the selection), and
        every determinant a with |<a|mu_j|i> c_i| >= eps or
        |<a|H|i> x_ji| >= eps for some i of the space is added (compiled
        selections), until fewer than `new_fraction` are new (at most
        `max_cycles` rounds); each round extends the previous round's
        Hamiltonian and dipole matrices. In the final space the ground state
        is solved to Davidson residual `ground_tol` and `SelectedCIResponse` gives the
        tensors at `wavelengths_nm` (inf: static) with `damping_Eh`, as
        `DipoleResponse` does: the exact response of the space's variational
        state, equal to full CI in the full space. `dipoles` are three
        symmetric one-electron matrices in the orbitals of h (for example
        the electronic -r of `pyscf_interface.dipole_integrals`). With
        `spaces` (determinant arrays) only the response runs, in each.
        `solve_options` go to the ground state's `solve` (for example
        `pt2_batches` or `pt2_deterministic` with `pt2_epsilon`: each
        record's `pt2` is then its stage's correction, for extrapolating the
        tensors against it). `initial` (determinants, for example a
        record's `.space`, or (determinants, ground vector)) continues a
        series: the ground state's selection starts from that space, with
        the given `epsilons`. Like `max_determinants`, the memory budget
        (`memory_budget_gib`) ends the series before a space that would not
        fit. In axial orbitals pass `axial.axial_dipoles` (x and z; the
        response spaces then hold M_L = 0 and +-1, and alpha_yy = alpha_xx);
        `mirror_parity=1` adds the reflection (a Sigma+ ground state: its x
        and z responses are even too)."""
        dipoles = _symmetric(dipoles)
        wavelengths = tuple(float(w) for w in np.atleast_1d(wavelengths_nm))
        irreps = self._dipole_irreps(dipoles, dipole_irreps)
        partners = spin_parity is not None
        mirror = mirror_parity is not None
        threads = self._threads()
        results = []

        def finish(response, start, pt2=0.0):
            response.refine(ground_tol)
            pols = tuple(response.polarizability(w, damping_Eh=damping_Eh, tol=tol) for w in wavelengths)
            record = SCIPolarizability(response.size, response.energy, response.dipole_au, pols,
                                       time.perf_counter() - start, float(pt2), response.determinants,
                                       response.hamiltonian_bytes)
            results.append(record)
            if callback is not None:
                callback(record)

        if spaces is not None:
            response = None
            for dets in spaces:
                start = time.perf_counter()
                response = SelectedCIResponse(self, dipoles, dets, spin_parity=spin_parity, dipole_irreps=irreps,
                                              previous=response, residual_tol=ground_tol,
                                              mirror_parity=mirror_parity)
                finish(response, start)
            if response is not None:
                response.close()
            return results
        if initial is not None and not isinstance(initial, tuple):
            initial = (initial, None)
        stage_spaces = []

        def keep(stage, dets, vectors):
            stage_spaces.append((dets.copy(), vectors[0].copy()))

        ground = self.solve(1, references=references, spin_parity=spin_parity, epsilons=epsilons,
                            pt2_epsilon=pt2_epsilon, max_determinants=max_determinants, initial=initial,
                            checkpoint=keep, mirror_parity=mirror_parity, **(solve_options or {}))
        union, response = None, None
        plan = self._memory_plan()
        # Block PCG on the dipole components (about 8 vectors each) or the
        # ground state's Davidson subspace, plus indices and determinant copies.
        per_determinant = 8 * 52 + 48 * self.words
        try:
            for stage, (base, base_vector) in zip(ground.stages, stage_spaces):
                start = time.perf_counter()
                order = None
                if union is None:
                    union = base
                else:
                    grown = np.concatenate([union, self._difference(response._space, base)])
                    fits, _ = plan.check(len(grown), per_determinant)
                    if (max_determinants is not None and len(grown) > max_determinants) or not fits:
                        break       # the cap ends the series; earlier stages stand
                    union = grown
                    if len(union) >= SORT_FROM:      # each stage starts from a sorted space
                        order = _locality_order(union, self.words)
                        union = np.ascontiguousarray(union[order])
                for cycle in range(max_cycles):
                    guess = None
                    if response is None:        # this stage's ground vector
                        guess = base_vector.copy()
                    rebuild = cycle > 0 and not extends
                    response = SelectedCIResponse(self, dipoles, union, spin_parity=spin_parity,
                                                  dipole_irreps=irreps, previous=response, guess=guess,
                                                  residual_tol=_SELECT_GROUND_TOL,
                                                  order=order if cycle == 0 else None, rebuild=rebuild,
                                                  mirror_parity=mirror_parity)
                    plan.record(len(union), response.hamiltonian_bytes)
                    c0 = np.ascontiguousarray(np.abs(response.ground))
                    fresh = []
                    for d in dipoles:
                        if not np.any(d):
                            continue
                        d = np.ascontiguousarray(d)
                        buffer = self.kernels.sci_select_onebody(response._space, d.ctypes.data, c0.ctypes.data,
                                                                 float(stage.epsilon), int(partners), threads)
                        fresh.append(self._close(self._take(buffer), mirror))
                    statics = response.statics(_SELECT_RESPONSE_TOL)
                    xs = [np.abs(x) / np.linalg.norm(x) for x in statics if x is not None and np.linalg.norm(x) > 0]
                    if xs:
                        cmax = np.ascontiguousarray(np.max(xs, axis=0))
                        buffer = self.kernels.sci_select(response._space, cmax.ctypes.data, float(stage.epsilon),
                                                         int(partners), threads)
                        fresh.append(self._close(self._take(buffer), mirror))
                    fresh = [f for f in fresh if len(f)]
                    new = self._difference(response._space, np.concatenate(fresh)) if fresh else union[:0]
                    fits, extends = plan.check(len(union) + len(new), per_determinant, response.hamiltonian_bytes)
                    capped = not fits or (max_determinants is not None and len(union) + len(new) > max_determinants)
                    if len(new) <= new_fraction * len(union) or cycle == max_cycles - 1 or capped:
                        break
                    union = np.concatenate([union, new])
                if capped and cycle == 0:
                    # The cap stopped the stage before any response determinants
                    # joined its ground space: no polarizability of this stage.
                    break
                finish(response, start, stage.pt2[0])
        finally:
            if response is not None:
                response.close()
        return results

    def field_solver_factory(self, determinants, *, spin_parity=1, residual_tol=1e-9, pt2_epsilon=None,
                             pt2_batches=1, renormalized=False, pt2_screen=None):
        """`solver_factory(h, orbital_irreps)` for `finite_field_polarizability`:
        each field point diagonalizes its Hamiltonian in the fixed space
        `determinants` (for example a `polarizability` result's `.space`), so the field
        energies vary smoothly and their second differences give that
        space's polarizability (equal to `polarizability`'s static tensor in
        the same space, up to the field extrapolation). With `pt2_epsilon`
        each energy adds the deterministic Epstein-Nesbet correction of its
        field's state (renormalized with `renormalized`), in `pt2_batches`
        passes: the second differences then give the PT2-corrected
        polarizability of the space, which converges with the space much
        faster than the variational one (the screening makes E2 piecewise
        smooth in the field: each dropped term is below pt2_epsilon^2 over
        an excitation energy). Terms H_ai c_i are kept by |H_ai| |c_i| >=
        pt2_epsilon, so the kept set follows the field's state and E2 jumps
        where terms cross the threshold; `pt2_screen` (one magnitude per
        determinant, for example `SelectedCIResponse.field_screen`) fixes
        the screening instead (|H_ai| screen_i >= pt2_epsilon at every
        field), so the doubles kept are the same at every field and E2 is
        smooth in it."""
        if self.axial:
            raise NotImplementedError("polarizabilities in axial orbitals are not supported; use real orbitals")
        dets = np.ascontiguousarray(determinants, dtype=np.uint64)
        parent = self
        screen = None
        if pt2_screen is not None:
            screen = np.ascontiguousarray(pt2_screen, dtype=np.float64)
            if screen.shape != (len(dets),) or pt2_epsilon is None:
                raise ValueError("pt2_screen needs pt2_epsilon and one entry per determinant")

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
                                     new_fraction=1.0, initial=(dets, guess), residual_tol=residual_tol,
                                     pt2_epsilon=pt2_epsilon if screen is None else None, pt2_batches=pt2_batches)
                if len(res.determinants) != len(dets):
                    raise RuntimeError("The fixed space grew")
                energy = float(res.energies[0])
                if pt2_epsilon is None:
                    return _Result(energy, res.vectors[0].copy())
                if screen is None:
                    pt2, norm = res.stages[-1].pt2[0], res.stages[-1].pt2_norm[0]
                else:
                    pt2, norm = self._screened(res)
                return _Result(energy + float(pt2 / (1.0 + norm) if renormalized else pt2), res.vectors[0].copy())

            def _screened(self, res):
                sci = self.sci
                vector = np.ascontiguousarray(res.vectors[0])
                electronic = np.array([res.energies[0] - sci.constant])
                out, norm, stats = np.zeros(1), np.zeros(1), np.zeros(2, dtype=np.int64)
                space = sci._space(dets)
                try:
                    sci.kernels.sci_pt2_screened(space, vector.ctypes.data, 1, electronic.ctypes.data,
                                                 screen.ctypes.data, float(pt2_epsilon), int(pt2_batches),
                                                 out.ctypes.data, norm.ctypes.data, stats.ctypes.data,
                                                 sci._threads())
                finally:
                    sci.kernels.sci_space_free(space)
                return out[0], norm[0]

        return lambda h, orbital_irreps=None: _FixedSpace(h)


def _symmetric(matrices):
    m = np.asarray(matrices, dtype=np.float64)
    if m.ndim != 3 or m.shape[0] != 3 or not np.allclose(m, m.transpose(0, 2, 1), atol=1e-10):
        raise ValueError("Expected three symmetric dipole matrices")
    return np.ascontiguousarray(m)


class SelectedCIResponse:
    """Electric dipole response of a selected-CI ground state in a fixed
    determinant space, with the interface of `DipoleResponse`.

    The ground state of `sci`'s Hamiltonian is solved in `determinants`
    (rows; closed under alpha-beta swaps with `spin_parity`), the dipole
    operators' matrices come from the compiled one-body kernel, and
    `polarizability(wavelength_nm, damping_Eh=..., tol=...)` returns a
    `response.Polarizability`: alpha_ij = sum over the space's states n of
    <0|mu_i|n><n|mu_j|0> (1/(E_n - E_0 - w - i g) + 1/(E_n - E_0 + w + i g)),
    from solves of (H - E_0 -+ (w + i g)) x = Q mu_j|0> (Q projects out |0>).
    Real shifts use block preconditioned conjugate gradients on all
    components at once (one pass over H per step for all of them; the
    operator is positive definite below the first excitation); complex
    shifts, or a real shift above an excitation, use GMRES per component.
    Solves start from the earlier solutions of the same component
    (`response._shifted_guesses`), so wavelength scans get cheap. With
    `previous` (a response whose space is a prefix of this one) its
    Hamiltonian and dipole matrices are extended instead of rebuilt (unless
    `rebuild`), its solutions start the new solves, and `previous` is
    closed; matrices that are not extended are freed before the new ones are
    built.

    With `spin_parity` (and `sci.spin_pairs`) the Hamiltonian, ground state
    and response vectors live on one function per alpha-beta swap pair (the
    dipole operators are spin-free), with `mirror_parity` per orbit under
    swap and reflection (axial orbitals; a Sigma+ ground state and the x and
    z components, which are even); `ground`, `statics` and `static` return
    determinant vectors. In axial orbitals give `axial.axial_dipoles` (y
    zero): alpha_yy is set to alpha_xx.
    """

    def __init__(self, sci, dipoles, determinants, *, spin_parity=1, dipole_irreps=None, guess=None,
                 residual_tol=1e-9, previous=None, order=None, rebuild=False, mirror_parity=None):
        self.sci, self.kernels, self.spin_parity = sci, sci.kernels, spin_parity
        self.mirror_parity = mirror_parity
        if mirror_parity is not None and sci.mirror is None:
            raise ValueError("mirror_parity needs orbital_mirror")
        self._group = None
        if mirror_parity is not None or (spin_parity is not None and sci.spin_pairs):
            self._group = (0 if spin_parity is None else spin_parity, 0 if mirror_parity is None else mirror_parity)
        self._basis = None
        threads = sci._threads()
        self.determinants = np.ascontiguousarray(determinants, dtype=np.uint64).reshape(-1, 2 * sci.words)
        self.size = n = len(self.determinants)
        self._dipoles = _symmetric(dipoles)
        self.irreps = sci._dipole_irreps(self._dipoles, dipole_irreps)
        self._ham, self._space, self._ops, self._guesses = None, None, [None, None, None], {}
        prefix = (order is None and previous is not None and previous.size <= n
                  and np.array_equal(previous.determinants, self.determinants[:previous.size]))
        extend = prefix and not rebuild and previous._ham is not None and previous._group == self._group
        carried = {}
        if order is not None and previous is not None:
            # A re-sorted space: `order` maps its rows to those of the previous
            # space padded with the new determinants; the previous ground
            # state and solutions, permuted, start this space's solves.
            order = np.asarray(order)

            def moved(v):
                return np.concatenate([v, np.zeros(n - len(v), dtype=v.dtype)])[order]

            guess = moved(previous.ground)
            carried = {key: [(moved(previous._to_full(x)), shift) for x, shift in stored]
                       for key, stored in previous._guesses.items()}
        elif previous is not None and prefix:
            # The previous space is a prefix: its solutions, as determinant
            # vectors padded with zeros, start this space's solves.
            carried = {key: [(previous._to_full(x), shift) for x, shift in stored]
                       for key, stored in previous._guesses.items()}
        try:
            if previous is not None and not extend:
                if prefix and guess is None:    # rebuilt: the previous solutions still start this space's solves
                    guess = previous.ground
                previous._release_matrices()
            self._space = sci._space(self.determinants)
            single = sci._single(n)
            if extend:
                if guess is None:
                    guess = previous.ground
                if self._group is None:
                    self._ham = self.kernels.sci_hamiltonian_extend(self._space, previous._ham, previous.size, 1e-15,
                                                                    single, threads)
                else:
                    self._ham = self.kernels.sci_hamiltonian_pairs_extend(
                        self._space, previous._ham, previous.size, float(self._group[0]), float(self._group[1]),
                        1e-15, single, threads)
            elif self._group is None:
                self._ham = self.kernels.sci_hamiltonian(self._space, 1e-15, single, threads)
            else:
                self._ham = self.kernels.sci_hamiltonian_pairs(self._space, float(self._group[0]),
                                                               float(self._group[1]), 1e-15, single, threads)
            if not self._ham:
                raise MemoryError("Could not build the Hamiltonian")
            for j, d in enumerate(self._dipoles):
                if not np.any(d):
                    continue
                d = np.ascontiguousarray(d)
                old = previous._ops[j] if extend else None
                if old is not None:
                    op = self.kernels.sci_onebody_extend(self._space, d.ctypes.data, old, previous.size, 0.0, threads)
                else:
                    op = self.kernels.sci_onebody(self._space, d.ctypes.data, 0.0, threads)
                if not op:
                    raise MemoryError("Could not build a dipole matrix")
                self._ops[j] = op
            if previous is not None:
                previous.close()
            if self._group is not None:
                self._basis = _Pairs(sci, self._space, n, *self._group)
            self._m = n if self._basis is None else self._basis.count      # length of the internal vectors
            self._diagonal = np.empty(self._m)
            self.kernels.sci_ham_diagonal(self._ham, self._diagonal.ctypes.data)
            self._partner = None
            if spin_parity is not None and self._basis is None:
                self._partner = np.empty(n, dtype=np.int64)
                self.kernels.sci_partners(self._space, self._partner.ctypes.data, threads)
            fill_from = None
            if guess is not None and len(guess) <= n:
                fill_from = len(guess)
                guess = np.concatenate([guess, np.zeros(n - len(guess))])[None]
            else:
                guess = None
            self._ground_state(guess, residual_tol, fill_from)
        except BaseException:
            self.close()
            raise
        # The previous space's solutions (determinant vectors, zero-padded or
        # permuted) start this space's solves.
        for key, stored in carried.items():
            self._guesses[key] = [(self._to_internal(x if len(x) == n else
                                                     np.concatenate([x, np.zeros(n - len(x), dtype=x.dtype)])), s)
                                  for x, s in stored]

    def _to_full(self, v):
        """An internal (orbit-basis) vector as a determinant vector."""
        if self._basis is None:
            return v
        if np.iscomplexobj(v):
            return self._to_full(v.real) + 1j * self._to_full(v.imag)
        return np.ascontiguousarray(self._basis.to_full(np.asarray(v, dtype=np.float64)[None])[0])

    def _to_internal(self, v):
        """A determinant vector projected onto the internal (orbit) basis."""
        if self._basis is None:
            return v
        if np.iscomplexobj(v):
            return self._to_internal(v.real) + 1j * self._to_internal(v.imag)
        return np.ascontiguousarray(self._basis.to_pairs(np.asarray(v, dtype=np.float64)[None])[0])

    def _ground_state(self, guess, residual_tol, fill_from=None):
        sci, threads = self.sci, self.sci._threads()
        energies, vectors = sci._diagonalize(self._ham, self._space, self.size, 1, guess, self.spin_parity,
                                             residual_tol, None, fill_from, self.determinants, pairs=self._group)
        self.residual_tol = residual_tol
        self.energy = float(energies[0])
        self._e0 = self.energy - sci.constant
        self.ground = np.ascontiguousarray(vectors[0])           # determinant vector
        self._g = self._to_internal(self.ground)
        self._g = self._g / np.linalg.norm(self._g)
        # Q mu_j |0> from the compiled one-body matrices (determinants), as internal vectors.
        self.sources, moments = [], np.zeros(3)
        for j, op in enumerate(self._ops):
            if op is None:
                self.sources.append(None)
                continue
            y = np.empty(self.size)
            self.kernels.sci_matvec(op, self.ground.ctypes.data, y.ctypes.data, threads)
            moments[j] = _dot(self.ground, y)
            self.sources.append(self._project(self._to_internal(y)))
        self.dipole_au = moments

    def refine(self, residual_tol):
        """Solve the ground state again to `residual_tol`, from the current
        one (when tighter than its current tolerance), and update the
        dipole sources; earlier response solutions stay as starting points."""
        if residual_tol < self.residual_tol:
            self._ground_state(self.ground[None], residual_tol)

    @property
    def hamiltonian_bytes(self):
        """Bytes held by the compiled Hamiltonian and dipole matrices."""
        handles = [h for h in (self._ham, *self._ops) if h is not None]
        return int(sum(self.kernels.sci_ham_bytes(h) for h in handles))

    def _release_matrices(self):
        """Free the compiled Hamiltonian and dipole matrices (the vectors stay)."""
        if self._ham is not None:
            self.kernels.sci_ham_free(self._ham)
            self._ham = None
        for j, op in enumerate(self._ops):
            if op is not None:
                self.kernels.sci_ham_free(op)
                self._ops[j] = None
        _core.release_free_memory()

    def close(self):
        """Release the compiled Hamiltonian, dipole matrices and space."""
        kernels = getattr(self, "kernels", None)
        if getattr(self, "_ham", None) is not None:
            kernels.sci_ham_free(self._ham)
            self._ham = None
        for j, op in enumerate(getattr(self, "_ops", ())):
            if op is not None:
                kernels.sci_ham_free(op)
                self._ops[j] = None
        if getattr(self, "_space", None) is not None:
            kernels.sci_space_free(self._space)
            self._space = None

    def __del__(self):
        self.close()

    def _project_real(self, v):
        threads = self.sci._threads()
        v = np.ascontiguousarray(v, dtype=np.float64)
        if self._partner is not None:
            out = np.empty_like(v)
            self.kernels.sci_parity(len(v), self._partner.ctypes.data, float(self.spin_parity), v.ctypes.data,
                                    out.ctypes.data, threads)
            v = out
        overlap = _dot(self._g, v)
        return _axpy(-overlap, self._g, v)

    def _project(self, v):
        if np.iscomplexobj(v):
            return self._project_real(v.real) + 1j * self._project_real(v.imag)
        return self._project_real(v)

    def _apply(self, v):
        v = np.ascontiguousarray(v, dtype=np.float64)      # held while the kernel reads it
        out = np.empty(self._m)
        self.kernels.sci_matvec(self._ham, v.ctypes.data, out.ctypes.data, self.sci._threads())
        return out

    def _initial(self, j, rhs, shift):
        stored = self._guesses.get((j, np.iscomplexobj(np.asarray(shift))), [])
        if not stored:
            return None
        initial, _ = _shifted_guesses(rhs, shift, stored, np)
        return initial

    def _store(self, j, x, shift):
        stored = self._guesses.setdefault((j, np.iscomplexobj(np.asarray(shift))), [])
        stored.append((x, shift))
        del stored[:-4]

    def _gmres(self, j, shift, tol, restart, maxiter):
        rhs = self.sources[j]
        if np.iscomplexobj(np.asarray(shift)):
            rhs = rhs.astype(np.complex128)
        initial = self._initial(j, rhs, shift)

        def action(v):
            v = self._project(v)
            if np.iscomplexobj(v):
                image = self._apply(v.real) + 1j * self._apply(v.imag)
            else:
                image = self._apply(v)
            return self._project(image - (self._e0 + shift) * v)

        precondition = self._diagonal - (self._e0 + shift)
        floor = 1e-3 if np.iscomplexobj(precondition) else np.copysign(1e-3, precondition)
        precondition = np.where(np.abs(precondition) < 1e-3, floor, precondition)
        x, residual, steps = krylov.gmres(action, rhs, precondition, np, tol=tol, restart=restart, maxiter=maxiter,
                                          initial=initial)
        x = self._project(x)
        self._store(j, x, shift)
        r = rhs - action(x)
        return x, dict(component=j, shift=complex(shift), relative_residual=float(residual), steps=int(steps),
                       method="gmres"), r

    def _pcg(self, js, shift, tol, maxiter):
        """Block preconditioned conjugate gradients for the components `js`
        at a real shift: {j: (x, report)}, or None when the operator is not
        positive definite there (an excitation below the shift)."""
        K, threads = self.kernels, self.sci._threads()
        n, k = self._m, len(js)
        e = self._e0 + float(shift)
        ones, minus_ones = np.ones(k), -np.ones(k)
        B = np.ascontiguousarray(np.stack([self.sources[j] for j in js], axis=1))
        X = np.zeros((n, k))
        for c, j in enumerate(js):
            guess = self._initial(j, self.sources[j], float(shift))
            if guess is not None:
                X[:, c] = guess
        # |d - e| floored: a positive diagonal preconditioner, with the exact
        # (absolute) inverse on the leading determinants of the sources and
        # of the starting vectors.
        precondition = np.maximum(np.abs(self._diagonal - e), 1e-3)
        weights = np.zeros(n)
        for c in range(k):
            for v in (B[:, c], X[:, c]):
                top = np.max(np.abs(v))
                if top > 0:
                    weights = np.maximum(weights, np.abs(v) / top)
        block = _BlockPreconditioner.leading(self.sci, self.determinants, self._diagonal, weights,
                                             pairs=self._basis)

        def dot(a, b):
            out = np.empty(k)
            K.sci_block_dot(n, k, a.ctypes.data, b.ctypes.data, out.ctypes.data, threads)
            return out

        def axpby(alpha, x, beta, y):
            K.sci_block_axpby(n, k, np.ascontiguousarray(alpha, dtype=np.float64).ctypes.data, x.ctypes.data,
                              np.ascontiguousarray(beta, dtype=np.float64).ctypes.data, y.ctypes.data, threads)

        def apply(v, out):              # out = Q (H - e) v
            K.sci_matmat(self._ham, k, v.ctypes.data, out.ctypes.data, threads)
            axpby(np.full(k, -e), v, ones, out)
            K.sci_block_deflate(n, k, self._g.ctypes.data, out.ctypes.data, threads)

        def precondition_into(r, z):    # z = Q M^-1 r
            K.sci_block_divide(n, k, precondition.ctypes.data, 0.0, 1e-3, r.ctypes.data, z.ctypes.data, threads)
            if block is not None:
                dv = block.denominators(block.values, e, True)
                z[block.rows] = block.vectors @ ((block.vectors.T @ r[block.rows]) / dv[:, None])
            K.sci_block_deflate(n, k, self._g.ctypes.data, z.ctypes.data, threads)

        K.sci_block_deflate(n, k, self._g.ctypes.data, X.ctypes.data, threads)
        b_norm = np.sqrt(dot(B, B))
        b_norm[b_norm == 0] = 1.0
        Q = np.empty_like(X)
        apply(X, Q)
        R = B.copy()
        axpby(minus_ones, Q, ones, R)
        if np.any(np.sqrt(dot(R, R)) > b_norm):     # a poor guess: start from zero
            X[:] = 0.0
            R = B.copy()
        Z = np.empty_like(X)
        precondition_into(R, Z)
        P = Z.copy()
        rz = dot(R, Z)
        steps = 0
        while True:
            relative = np.sqrt(np.maximum(dot(R, R), 0.0)) / b_norm
            active = relative > 0.8 * tol
            if not active.any() or steps >= maxiter:
                break
            apply(P, Q)
            steps += 1
            pq = dot(P, Q)
            if np.any(pq[active] <= 0):
                return None
            alpha = np.where(active, rz / np.where(pq != 0, pq, 1.0), 0.0)
            axpby(alpha, P, ones, X)
            axpby(-alpha, Q, ones, R)
            precondition_into(R, Z)
            rz_new = dot(R, Z)
            beta = np.where(active, rz_new / np.where(rz != 0, rz, 1.0), 0.0)
            rz = np.where(active, rz_new, rz)
            axpby(ones, Z, beta, P)
        # True residuals of the final block.
        apply(X, Q)
        R = B.copy()
        axpby(minus_ones, Q, ones, R)
        relative = np.sqrt(np.maximum(dot(R, R), 0.0)) / b_norm
        out = {}
        for c, j in enumerate(js):
            x = self._project(np.ascontiguousarray(X[:, c]))
            self._store(j, x, float(shift))
            out[j] = (x, dict(component=j, shift=complex(shift), relative_residual=float(relative[c]),
                              steps=int(steps), method="pcg"), np.ascontiguousarray(R[:, c]))
        return out

    def _solve_many(self, shift, tol, restart, maxiter, components=None):
        """{j: (x, report, residual)} for every nonzero component (or `components`)."""
        js = [j for j in (range(3) if components is None else components) if self.sources[j] is not None]
        if not js:
            return {}
        if not np.iscomplexobj(np.asarray(shift)):
            found = self._pcg(js, shift, tol, maxiter)
            if found is not None:
                return found
        return {j: self._gmres(j, shift, tol, restart, maxiter) for j in js}

    def statics(self, tol=1e-8, maxiter=1000):
        """The static response vectors (H - E_0)^-1 Q mu_j|0>, j = 0..2 (None for a zero
        component), as determinant vectors."""
        found = self._solve_many(0.0, tol, 40, maxiter)
        return [self._to_full(found[j][0]) if j in found else None for j in range(3)]

    def static(self, j, tol=1e-8, restart=40, maxiter=1000):
        """The static response vector of component j (None for a zero component)."""
        if self.sources[j] is None:
            return None
        return self._to_full(self._solve_many(0.0, tol, restart, maxiter, components=[j])[j][0])

    def field_screen(self, field_au, tol=1e-6):
        """|c_0| + F max_j |x_j|: a bound on the magnitudes of the states in
        uniform fields up to F (first order: c(F) = c_0 - F x_j), the fixed
        PT2 screening for `SelectedCI.field_solver_factory(pt2_screen=)`."""
        screen = np.abs(self.ground)
        for x in self.statics(tol):
            if x is not None:
                screen = np.maximum(screen, np.abs(self.ground) + float(field_au) * np.abs(x))
        return screen

    def polarizability(self, wavelength_nm, *, damping_Eh=0.0, tol=1e-5, restart=40, maxiter=1000, callback=None,
                       reuse_guesses=True):
        """The Cartesian electronic tensor in au (complex), as
        `DipoleResponse.polarizability`; use +inf for static. Solves stop at
        relative residual `tol` (their reports are the result's `solves`);
        the tensor is the stationary (Hylleraas) form b_i.x_j + x_i.r_j per
        sign of the shift, r_j the residual of x_j, so its error is second
        order in the solve error. `reuse_guesses=False` starts them cold."""
        wavelength_nm, damping_Eh = float(wavelength_nm), float(damping_Eh)
        if not wavelength_nm > 0 or not np.isfinite(damping_Eh) or damping_Eh < 0 or not 0 < tol < 1:
            raise ValueError("Invalid wavelength, damping or tolerance")
        if not reuse_guesses:
            self._guesses.clear()
        frequency = HC_EH_NM / wavelength_nm
        shift = frequency if damping_Eh == 0 else complex(frequency, damping_Eh)
        plus = self._solve_many(shift, tol, restart, maxiter)
        minus = plus if shift == 0 else self._solve_many(-shift, tol, restart, maxiter)
        reports = [r for j in sorted(plus) for r in ((plus[j][1],) if shift == 0 else (plus[j][1], minus[j][1]))]
        if callback is not None:
            for report in reports:
                callback(report)
        tensor = np.zeros((3, 3), dtype=np.complex128)
        for i in range(3):
            for j in range(3):
                if i not in plus or j not in plus:
                    continue
                if self.irreps is not None and self.irreps[i] != self.irreps[j]:
                    continue
                tensor[i, j] = sum(np.dot(self.sources[i], side[j][0]) + np.dot(side[i][0], side[j][2])
                                   for side in (plus, minus))
        tensor = 0.5 * (tensor + tensor.T)
        if self.sci.axial and self.sources[1] is None and self.sources[0] is not None:
            tensor[1, 1] = tensor[0, 0]          # cylindrical symmetry: y as x
        return Polarizability(wavelength_nm, frequency, damping_Eh, tensor, tuple(reports))


def _dot(x, y):
    """x . y of flat float64 arrays on the compiled kernels (deterministic)."""
    from ._cpu_quadruples import dots
    return float(dots(np.ascontiguousarray(x, dtype=np.float64), [np.ascontiguousarray(y, dtype=np.float64)])[0])


def _axpy(a, x, y):
    """a x + y as a new array, on the compiled kernels."""
    from ._cpu_quadruples import combine
    out = np.empty_like(y)
    return combine(np.array([a, 1.0]), [np.ascontiguousarray(x), np.ascontiguousarray(y)], out)


def rotate_integrals(h, eri, coefficients, *, orbital_irreps=None, axial=False):
    """h and chemists' eri in the orbitals `coefficients` (columns, real); the
    two-electron part by the native transformation (`integrals.mo_eri`, with
    exact zeros for the symmetry-forbidden entries given `orbital_irreps`).
    `axial=True` takes integrals of axial orbitals (4-fold symmetric only)
    and transforms them index by index."""
    from .integrals import mo_eri
    c = np.ascontiguousarray(coefficients, dtype=np.float64)
    n = c.shape[0]
    h2 = c.T @ np.asarray(h, dtype=np.float64) @ c
    if axial:
        g = np.asarray(eri, dtype=np.float64).reshape(n, n, n, n)
        for _ in range(4):           # contract the leading index, move it last
            g = np.tensordot(g, c, axes=([0], [0]))
        return h2, np.ascontiguousarray(g)
    g = mo_eri(np.ascontiguousarray(eri, dtype=np.float64).reshape(n, n, n, n), c, orbital_irreps=orbital_irreps)
    return h2, np.ascontiguousarray(g.reshape(c.shape[1], c.shape[1], c.shape[1], c.shape[1]))
