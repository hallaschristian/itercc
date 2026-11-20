"""Static dipole polarizabilities from ground-state energies in finite fields.

Analytic response (`DipoleResponse`) covers CCSD and CCSDT. Any ground-state
method, CCSDTQ included, also gives the static tensor as the second derivative
of its energy in a uniform field, with the same fixed orbitals: each point
solves the Hamiltonian with h - F (n . dipoles) for a field F along a unit
vector n, and

    alpha_nn(F) = -(E(+F) + E(-F) - 2 E(0)) / F**2 = alpha_nn + O(F**2),
    mu_n(F)     = -(E(+F) - E(-F)) / (2 F)         = mu_n + O(F**2).

Several field strengths are extrapolated to zero field as a polynomial in F**2
(Richardson), which removes the hyperpolarizability terms. Off-diagonal
elements come from the lines n = (e_i + e_j) / sqrt(2):
alpha_ij = alpha_nn - (alpha_ii + alpha_jj) / 2.

With orbital labels, each field solve runs in the labels of the subgroup the
field conserves (`spatial_symmetry.conserved_labels`); a component that is not
totally symmetric needs only +F (a symmetry operation maps the field to -F),
and elements between components of different irreps vanish without solves.
Each point starts from the amplitudes predicted by the solved points of its
line (`continuation`), which saves most of the iterations.
"""
from dataclasses import dataclass
from math import isfinite, sqrt
from time import perf_counter

import numpy as np

from .continuation import continuation_plan, lagrange_weights, weighted_sum
from .spatial_symmetry import conserved_labels, operator_irreps, orbital_labels

AXES = "xyz"
# Water CCSDT against analytic response: 1e-3/2e-3 au leaves 3e-8 au of F**4 error with
# aug-cc-pVDZ (2e-3/4e-3: 2e-7), and these tolerances 1e-7 to 1e-6 au of convergence
# error (1e-12/1e-10: up to 1e-5 au, 20% fewer iterations).
DEFAULT_FIELDS = (1e-3, 2e-3)
DEFAULT_SOLVE = dict(energy_tol=1e-13, residual_tol=1e-11)


@dataclass(frozen=True)
class FieldPoint:
    """One ground-state solve: `direction` is "zero", an axis ("x") or a pair line ("xz")."""
    direction: str
    field_au: float
    energy_Eh: float
    converged: bool
    evaluations: int
    seconds: float


@dataclass(frozen=True)
class FiniteFieldPolarizability:
    """Static polarizability (au) of the Hamiltonian's energy, clamped nuclei and fixed orbitals.

    `tensor_au` is extrapolated to zero field (with one field strength it is
    the centered value). Elements that were not solved are NaN (diagonal
    elements of an off-diagonal request are solved too); elements between
    components of different irreps are exactly zero. `dipole_au` is -dE/dF of
    the given Hamiltonian, the electronic dipole of its correlated orbitals
    (add the nuclear and frozen-core dipoles for the total); it is exactly
    zero for components that are not totally symmetric and NaN for axes that
    were not solved. `centered_au` holds one tensor per field strength before
    extrapolation.
    """
    tensor_au: np.ndarray
    dipole_au: np.ndarray
    fields_au: tuple
    centered_au: tuple
    energy_Eh: float
    points: tuple

    @property
    def converged(self):
        return all(p.converged for p in self.points)

    @property
    def extrapolation_au(self):
        """Zero-field tensor minus the centered tensor of the smallest field: the
        size of the F**2 correction, an upper estimate of the centered value's
        field error."""
        return self.tensor_au - self.centered_au[int(np.argmin(self.fields_au))]


def _components(components, irreps):
    """Requested (i, j) pairs, i <= j, and the zero pairs that symmetry fixes."""
    if components is None:
        wanted = {(i, j) for i in range(3) for j in range(i, 3)}
    else:
        wanted = set()
        for name in components:
            if len(name) != 2 or any(c not in AXES for c in name):
                raise ValueError(f"Unknown tensor component {name!r}; use for example 'xx' or 'xz'")
            i, j = sorted(AXES.index(c) for c in name)
            wanted.add((i, j))
    zero = {(i, j) for i, j in wanted if i != j and irreps is not None and irreps[i] != irreps[j]}
    return sorted(wanted - zero), zero


def _irreps(dipoles, labels, dipole_irreps):
    if labels is None:
        return None
    if dipole_irreps is not None:
        irreps = tuple(int(g) for g in dipole_irreps)
        if len(irreps) != 3 or any(not 0 <= g <= 7 for g in irreps):
            raise ValueError("Expected three dipole irreps 0..7")
        return irreps
    irreps = []
    for d in dipoles:
        found = operator_irreps(d, labels, tolerance=1e-8 * max(1.0, float(np.max(np.abs(d)))))
        if len(found) > 1:
            raise ValueError(f"A dipole component spans several irreps {found}; check the orbital labels")
        irreps.append(found[0] if found else 0)
    return tuple(irreps)


def _combine(solver, plan):
    """Initial amplitudes sum_k w_k T_k for `solver` from [(state, weight)]."""
    if hasattr(solver, "combine_states"):
        return solver.combine_states([(w, amplitudes, signature) for (amplitudes, signature), w in plan])
    xp = np
    first = plan[0][0][0][0]
    if type(first).__module__.split(".")[0] == "cupy":
        import cupy as xp
    return weighted_sum([(w, amplitudes) for (amplitudes, _), w in plan], xp=xp)


def finite_field_polarizability(solver_factory, h, dipoles, *, orbital_irreps=None, dipole_irreps=None,
                                fields=DEFAULT_FIELDS, components=None, initial=None, warm_start=True,
                                solve_options=None, callback=None):
    """Static polarizability tensor -d2E/dF_i dF_j from energies at finite fields.

    `solver_factory(h, orbital_irreps)` returns a ground-state solver of the
    perturbed one-electron matrix `h` with the given labels, for example
    `lambda h, labels: RestrictedCCSDTQ(h, eri, nocc, constant,
    orbital_irreps=labels, backend="cpu")`; its `solve(**options)` returns a
    `Result`. `h` is the zero-field one-electron matrix and `dipoles` the
    three electronic dipole matrices (-r) in the same orbitals, as for
    `DipoleResponse` (for scalar X2C, picture-changed). `orbital_irreps` and
    `dipole_irreps` (found from the dipoles when omitted) enable the symmetry
    savings. `fields` are the strengths (au); with two or more the tensor is
    extrapolated to zero field. `components` restricts the tensor to elements
    such as ("xx", "zz"). `initial` starts the zero-field solve: amplitudes,
    or a callable that receives the zero-field solver (for example
    `lambda s: (t1, t2, s.expand_triples(t3))` from CCSDT for CCSDTQ). With
    `warm_start`, the solved states of the current line are kept to predict
    its next point (up to four states besides the zero-field one; pass False
    to keep none). `solve_options` go to every `solve` (default energy_tol
    1e-13 and residual_tol 1e-11: the second differences divide energy errors
    by F**2). `callback(point)` receives each `FieldPoint` as it is solved.

    Raises RuntimeError when a solve does not converge.
    """
    dipoles = np.asarray(dipoles, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or dipoles.shape != (3, *h.shape):
        raise ValueError("Expected a square h and three dipole matrices of its shape")
    fields = tuple(float(f) for f in fields)
    if not fields or any(not isfinite(f) or f <= 0 for f in fields) or len(set(fields)) != len(fields):
        raise ValueError("Expected distinct positive finite field strengths")
    labels = None if orbital_irreps is None else orbital_labels(orbital_irreps, len(h))
    irreps = _irreps(dipoles, labels, dipole_irreps)
    wanted, zero = _components(components, irreps)
    options = dict(DEFAULT_SOLVE, **(solve_options or {}))
    if "initial" in options:
        raise ValueError("Pass the zero-field start as initial=, not in solve_options")

    points = []

    def solve(direction, field, operator, line_labels, start):
        solver = solver_factory(h - field * operator if field else h, line_labels)
        begin = perf_counter()
        result = solver.solve(initial=start(solver), **options)
        point = FieldPoint(direction, field, float(result.energy), bool(result.converged),
                           len(result.iterations), perf_counter() - begin)
        points.append(point)
        if callback is not None:
            callback(point)
        if not result.converged:
            change = result.iterations[-1].delta_energy if result.iterations else None
            detail = f" (last energy change {change:.1e} Eh)" if change is not None and isfinite(change) else ""
            raise RuntimeError(f"The {direction} solve at field {field:g} au did not converge{detail}; "
                               "raise max_iter or adjust the solve_options")
        signature = solver.layout_signature() if hasattr(solver, "combine_states") else None
        return point.energy_Eh, (result.amplitudes, signature) if warm_start else None

    e0, zero_state = solve("zero", 0.0, None, labels,
                           lambda solver: initial(solver) if callable(initial) else initial)

    def line(direction, vector):
        """Signed-field energies of one line, {field: energy}, and whether -F mirrors +F."""
        operator = np.tensordot(vector, dipoles, axes=1)
        if irreps is None:
            symmetric, line_labels = False, None
        else:
            irrep = irreps[int(np.flatnonzero(vector)[0])]
            symmetric, line_labels = irrep != 0, conserved_labels(labels, [irrep])
        signed = [f for f in fields] + ([] if symmetric else [-f for f in fields])
        available = {0.0: zero_state} if warm_start else {}
        energies = {0.0: e0}
        for field in signed:
            plan = continuation_plan(available, field) if warm_start else []
            start = (lambda solver, plan=plan: _combine(solver, plan) if plan else None)
            energies[field], state = solve(direction, field, operator, line_labels, start)
            if warm_start:
                available[field] = state
        if symmetric:
            energies.update({-f: energies[f] for f in fields})
        return energies, symmetric

    weights = lagrange_weights([f * f for f in fields], 0.0)
    centered = np.full((len(fields), 3, 3), np.nan)
    dipole_centered = np.full((len(fields), 3), np.nan)
    needed = sorted({i for pair in wanted for i in pair})
    for i in needed:
        vector = np.eye(3)[i]
        energies, symmetric = line(AXES[i], vector)
        for k, f in enumerate(fields):
            centered[k, i, i] = -(energies[f] + energies[-f] - 2 * e0) / f**2
            dipole_centered[k, i] = 0.0 if symmetric else -(energies[f] - energies[-f]) / (2 * f)
    for i, j in wanted:
        if i == j:
            continue
        vector = (np.eye(3)[i] + np.eye(3)[j]) / sqrt(2)
        energies, _ = line(AXES[i] + AXES[j], vector)
        for k, f in enumerate(fields):
            along = -(energies[f] + energies[-f] - 2 * e0) / f**2
            centered[k, i, j] = centered[k, j, i] = along - (centered[k, i, i] + centered[k, j, j]) / 2
    for i, j in zero:
        centered[:, i, j] = centered[:, j, i] = 0.0
    tensor = np.einsum("k,kij->ij", weights, centered)
    dipole = np.einsum("k,ki->i", weights, dipole_centered)
    return FiniteFieldPolarizability(tensor, dipole, fields, tuple(centered), e0, tuple(points))
