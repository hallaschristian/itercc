"""Linear-response CCSDTQ dipole polarizabilities on the finite-difference
CCSDTQ Jacobian (compiled CPU solver).

`DipoleResponse` forms the CC linear response function from right-hand
Jacobian solves: the dipole solves (A -+ omega) X(+-omega) = -xi, and for each
tensor component one zero-shift solve that contracts the ground multipliers
implicitly (lambda.G = -E_T.A^-1.G). `QuadruplesDipoleResponse` keeps that
algorithm (and its GMRES, shift recycling and warm starts) and supplies every
ingredient from evaluations of the complete T1-T4 residual of
`RestrictedCCSDTQ`, so no CCSDTQ left-hand or Lambda equations are needed:

* Jacobian actions: `eom_ccsdtq.FiniteDifferenceEOM` in each sector (one
  residual per action in a sector other than the totally symmetric one).
* xi_p = Res(T; h + p) - Res(T; h): the residual is linear in the one-electron
  Hamiltonian, which it reads only through the T1-dressed Fock matrix, so the
  perturbed Hamiltonian is the solver's own with h + p swapped in.
* The mixed second derivatives of the property right-hand side,
  d xi_p/dT [d] and d2 Res/dT2 [L, R], from residuals at T + a d with fixed
  multiples a of one step: the BCH residual is a polynomial of degree four in
  T and xi_p one of degree two, so the stencils below are exact (they only
  round). In a sector G other than the totally symmetric one the totally
  symmetric part of Res(T + a d) holds only the even powers of a, which
  halves the evaluations; there the directions are written into the ground
  amplitudes' (zero) sector entries and cleared afterwards, so no copy of T is
  made.
* The energy terms are those of the closed-shell CC energy in T1 and T2.

A sector G != 0 needs a solver whose orbital labels keep G totally symmetric
(`spatial_symmetry.sector_labels(labels, G)`), with the ground amplitudes in
its layout (`solver.combine_states`); the zero-shift property solves run on
the solver of sector 0 (the full group) when one is given.
"""
from numbers import Integral

import numpy as np

from . import _runtime
from .eom_ccsdtq import FiniteDifferenceEOM, QuadruplesSector
from .hamiltonian import _real64
from .response import DipoleResponse


class _Sector:
    """A symmetry sector: its solver, Jacobian engine and packings."""

    def __init__(self, irrep, solver, amplitudes, labels, step):
        self.irrep = int(irrep)
        self.solver = solver
        self.engine = FiniteDifferenceEOM(solver, amplitudes, labels, irrep, step=step)
        self.space = self.engine.space
        self._labels = labels
        self._total = self.space if self.irrep == 0 else None
        no = solver.nocc
        order = getattr(solver, "_virtual_order", None)
        self.order = (np.arange(solver.nmo) if order is None
                      else np.r_[np.arange(no), no + np.asarray(order)])
        reference = solver._reference
        o, v = reference.o, reference.v
        g = np.asarray(reference.g)[o, o, v, v]
        self.energy_weight = 2 * g - g.swapaxes(2, 3)
        self.fock_ov = np.array(np.asarray(reference.fock)[o, v])
        self.o, self.v = o, v

    @property
    def total(self):
        """The totally symmetric entries of this solver's arrays (packing only)."""
        if self._total is None:
            self._total = QuadruplesSector(self.solver, self._labels, 0)
            self._total.diagonal = None         # a vector we never read
        return self._total

    def internal(self, matrix):
        return np.ascontiguousarray(matrix[np.ix_(self.order, self.order)])

    def residual(self, vector=None, scale=0.0, field=None):
        """Res(T + scale vector; h + field): `vector` packed in this sector,
        `field` a one-electron matrix in the solver's internal order."""
        solver, engine = self.solver, self.engine
        dressing = solver._dressing
        bare = dressing.h
        if field is not None:
            dressing.h = bare + field
        try:
            if vector is None or scale == 0:
                return solver._residuals(engine.amplitudes)
            if engine.one_sided:
                # The ground amplitudes' sector entries are zero: write the
                # displacement there and clear it again (no copy of T).
                amplitudes = engine.amplitudes
                engine.space.shift(amplitudes, vector, scale, assign=True)
                try:
                    return solver._residuals(amplitudes)
                finally:
                    engine.space.clear(amplitudes)
            shifted = engine._shifted()
            engine.space.shift(shifted, vector, scale)
            return solver._residuals(shifted)
        finally:
            dressing.h = bare

    def lower(self, vector, ranks=2):
        """T1 (and T2) parts of a packed vector, in the solver's internal order."""
        arrays = tuple(np.zeros(shape) for shape in self.solver.shapes[:ranks])
        self.space.shift(arrays, np.ascontiguousarray(vector, dtype=np.float64), 1.0)
        return arrays


class QuadruplesDipoleResponse(DipoleResponse):
    """Fixed-orbital LR-CCSDTQ electric dipole response (real frequencies).

    `solvers` maps each needed irrep to a CPU `RestrictedCCSDTQ`
    (precision="double") whose orbital labels keep that irrep totally
    symmetric, and `amplitudes` maps the same irreps to the converged ground
    amplitudes in that solver's layout (or is one tuple for every solver, when
    they share a layout). A single solver may be passed without symmetry.
    `labels` are the caller's orbital irreps (None: no symmetry) and
    `dipoles` three symmetric caller-order MO matrices with `dipole_irreps`,
    as for `DipoleResponse`. `axes` limits the Cartesian axes solved (for
    example (0, 2) for a linear molecule, whose yy equals xx); the tensor
    entries of other axes stay zero. Sector 0 is needed for the property
    solves; without its own solver the first given one runs them with
    central differences.

    Each tensor component costs the dipole solves of `DipoleResponse` plus
    a handful of residual evaluations for its right-hand side: five (four
    static) in a sector G != 0 and ten (seven static) in sector 0.
    """

    STEP = 0.04         # stencil step for unit-max-norm directions

    def __init__(self, solvers, amplitudes, dipoles, labels=None, *, dipole_irreps=None, axes=None,
                 recycle_gib=0.0, checkpoint=None, step=5e-5):
        if not isinstance(solvers, dict):
            solvers = {0: solvers}
        if not solvers:
            raise ValueError("Expected at least one CCSDTQ solver")
        first = next(iter(solvers.values()))
        nmo = first.nmo
        for solver in solvers.values():
            if solver.xp is not np:
                raise ValueError("LR-CCSDTQ runs on the compiled CPU solver (backend='cpu')")
            if getattr(solver, "precision", "double") != "double":
                raise ValueError("LR-CCSDTQ needs precision='double' solvers")
            if solver.nmo != nmo or solver.nocc != first.nocc:
                raise ValueError("The solvers must share one orbital space")
        raw = _real64(dipoles)
        if (raw.shape != (3, nmo, nmo) or not np.isfinite(raw).all()
                or np.max(np.abs(raw - raw.transpose(0, 2, 1))) > 1e-10):
            raise ValueError("Expected three finite real symmetric MO dipoles")
        if labels is None:
            labels = np.zeros(nmo, dtype=np.int64)
            if dipole_irreps not in (None, (0, 0, 0), [0, 0, 0]):
                raise ValueError("Nonzero dipole irreps require orbital labels")
        labels = np.asarray(labels, dtype=np.int64)
        sectors = (0, 0, 0) if dipole_irreps is None else tuple(int(k) for k in dipole_irreps)
        if len(sectors) != 3 or any(not 0 <= k <= 7 for k in sectors):
            raise ValueError("Supply one Abelian irrep for each Cartesian dipole")
        for axis, sector in enumerate(sectors):
            forbidden = (labels[:, None] ^ labels) != sector
            if np.max(np.abs(raw[axis, forbidden]), initial=0.) > 2e-9:
                raise ValueError("Dipole has the wrong orbital symmetry")
        axes = tuple(range(3)) if axes is None else tuple(int(a) for a in axes)
        if any(not 0 <= a < 3 for a in axes):
            raise ValueError("Axes are 0, 1 and 2")
        self.active_axes = tuple(axis in axes and bool(np.any(raw[axis])) for axis in range(3))
        needed = {0} | {sectors[a] for a in range(3) if self.active_axes[a]}
        missing = needed - {0} - set(solvers)
        if missing:
            raise ValueError(f"No solver for the dipole sectors {sorted(missing)}")
        if not isinstance(amplitudes, dict):
            amplitudes = {k: amplitudes for k in solvers}

        self.method = "ccsdtq"
        self.xp = self.krylov_xp = np
        self.sectors = sectors
        self.labels = labels
        self._sector = {}
        for k in sorted(needed):
            if k in solvers:
                self._sector[k] = _Sector(k, solvers[k], amplitudes[k], labels, step)
            else:   # property solves on another sector's solver, central differences
                key = next(iter(solvers))
                self._sector[k] = _Sector(k, solvers[key], amplitudes[key], labels, step)
        zero = self._sector[0]
        self.cpu_threads = zero.engine.cpu_threads
        self.spaces = {k: s.space for k, s in self._sector.items()}
        self.diagonal = {k: s.engine.diagonal for k, s in self._sector.items()}
        # Perturbations -mu in each sector solver's internal order; they must
        # be totally symmetric there (that solver keeps their irrep).
        self.perturbations = -raw
        self._fields = {}
        for axis in range(3):
            if not self.active_axes[axis]:
                continue
            sector = self._sector[sectors[axis]]
            field = sector.internal(-raw[axis])
            kept = sector.solver.orbital_irreps
            if kept is not None:
                internal = np.asarray(kept)[sector.order]
                if np.max(np.abs(field[(internal[:, None] ^ internal) != 0]), initial=0.) > 2e-9:
                    raise ValueError(f"The solver of sector {sectors[axis]} does not keep dipole {axis} "
                                     "totally symmetric; use spatial_symmetry.sector_labels")
            self._fields[axis] = field
        self.rhs = tuple(self._xi(axis) if self.active_axes[axis] else None for axis in range(3))
        self.checkpoint = checkpoint
        self.deflation = None
        if not np.isfinite(recycle_gib) or recycle_gib < 0:
            raise ValueError("recycle_gib must be a nonnegative number of GiB")
        self.recycle_gib = float(recycle_gib)
        self._guesses = {}
        self._dipole_shifts = {}
        self._recycled = {}

    # Ingredients from residual evaluations.

    @_runtime.threaded
    def _xi(self, axis):
        """xi_p = Res(T; h + p) - Res(T; h) in the dipole's sector."""
        sector = self._sector[self.sectors[axis]]
        value = sector.space.pack(sector.residual(field=self._fields[axis]))
        if sector.irrep == 0:       # the converged residual (zero by symmetry otherwise)
            value -= sector.space.pack(sector.residual())
        return value

    def _accumulate(self, sector, terms):
        """sum of weight * (totally symmetric part of Res(T + scale d; h + field))."""
        result = None
        for weight, direction, scale, field in terms:
            if weight == 0:
                continue
            value = sector.total.pack(sector.residual(direction, scale, field))
            value *= weight
            if result is None:
                result = value
            else:
                result += value
            del value
        return result

    def _scale(self, *vectors):
        largest = max(float(np.max(np.abs(v))) for v in vectors if v is not None and v.size)
        return self.STEP / max(1.0, largest)

    def _property_rhs(self, i, j, left, right):
        """d xi_pj/dT [left] + d xi_pi/dT [right] + d2 Res/dT2 [left, right], totally symmetric."""
        sector = self._sector[self.sectors[i]]
        fi, fj = self._fields[i], self._fields[j]
        same = left is right
        u = left + right
        w = None if same else left - right
        s = self._scale(u, w)
        terms = []
        if sector.irrep != 0:
            # Totally symmetric part of Res(T + a d) = a^2/2 R2[d,d] + a^4/24 R4[d^4] + Res(T):
            # R2[d,d] = (16 E(s) - E(2s) - 15 Res(T)) / (6 s^2); Res(T + s d; h + p) adds s dxi_p[d].
            # R2[L,R] = (R2[u,u] - R2[w,w]) / 4, u = L + R, w = L - R (Res(T) cancels).
            c = 1.0 / (24 * s * s)
            if i == j:
                terms += [(1 / s, u, s, fi), (16 * c - 1 / s, u, s, None)]
            else:
                terms += [(1 / s, left, s, fj), (-1 / s, left, s, None),
                          (1 / s, right, s, fi), (-1 / s, right, s, None), (16 * c, u, s, None)]
            terms.append((-c, u, 2 * s, None))
            if same:
                terms.append((-15 * c, None, 0.0, None))
            else:
                terms += [(-16 * c, w, s, None), (c, w, 2 * s, None)]
        else:
            # Everything is totally symmetric: central stencils, exact for the quartic residual:
            # R2[d,d] = (-E(2s) + 16 E(s) - 30 E(0) + 16 E(-s) - E(-2s)) / (12 s^2),
            # dxi_p[d] = (F(s) - E(s) - F(-s) + E(-s)) / (2 s), F with h + p.
            c = 1.0 / (48 * s * s)
            half = 1 / (2 * s)
            if i == j:
                terms += [(half, u, s, fi), (-half, u, -s, fi),
                          (16 * c - half, u, s, None), (16 * c + half, u, -s, None)]
            else:
                terms += [(half, left, s, fj), (-half, left, -s, fj), (-half, left, s, None),
                          (half, left, -s, None), (half, right, s, fi), (-half, right, -s, fi),
                          (-half, right, s, None), (half, right, -s, None),
                          (16 * c, u, s, None), (16 * c, u, -s, None)]
            terms += [(-c, u, 2 * s, None), (-c, u, -2 * s, None)]
            if same:
                terms.append((-30 * c, None, 0.0, None))
            else:
                terms += [(-16 * c, w, s, None), (-16 * c, w, -s, None), (c, w, 2 * s, None),
                          (c, w, -2 * s, None)]
        return self._accumulate(sector, terms)

    def _energy_gradient(self, vector):
        """dE/dT [z] for a totally symmetric packed vector z."""
        sector = self._sector[0]
        z1, z2 = sector.lower(vector)
        t1 = np.asarray(sector.engine.amplitudes[0])
        w = sector.energy_weight
        return float(2 * np.einsum("ia,ia->", sector.fock_ov, z1) + np.einsum("ijab,ijab->", w, z2)
                     + np.einsum("ijab,ia,jb->", w, z1, t1) + np.einsum("ijab,ia,jb->", w, t1, z1))

    def _direct(self, i, j, left, right):
        """The energy's explicit part: 2 p_i.R1 + 2 p_j.L1 + E''[L, R]."""
        sector = self._sector[self.sectors[i]]
        o, v = sector.o, sector.v
        (l1,), (r1,) = sector.lower(left, 1), sector.lower(right, 1)
        w = sector.energy_weight
        return float(2 * np.einsum("ia,ia->", self._fields[i][o, v], r1)
                     + 2 * np.einsum("ia,ia->", self._fields[j][o, v], l1)
                     + np.einsum("ijab,ia,jb->", w, l1, r1) + np.einsum("ijab,ia,jb->", w, r1, l1))

    # DipoleResponse hooks.

    def _action(self, space, vector):
        return self._sector[space.irrep].engine.sigma(vector)

    @_runtime.threaded
    def _component(self, i, j, minus_i, plus_j, controls, reports):
        if np.iscomplexobj(minus_i) or np.iscomplexobj(plus_j):
            raise ValueError("LR-CCSDTQ takes real frequencies (no damping)")
        combined = self._property_rhs(i, j, minus_i, plus_j)
        direct = self._direct(i, j, minus_i, plus_j)
        correction = self._solve(0, 0., combined, *controls, reports, ("property", i, j))
        del combined
        return complex(self._energy_gradient(correction) - direct)

    def polarizability(self, wavelength_nm, *, damping_Eh=0., tol=1e-8, restart=16, maxiter=160,
                       callback=None, reuse_guesses=True, deflation=None):
        """As `DipoleResponse.polarizability`, for real frequencies only."""
        if damping_Eh != 0:
            raise ValueError("LR-CCSDTQ takes real frequencies (no damping)")
        if isinstance(restart, bool) or not isinstance(restart, Integral):
            raise ValueError("Invalid linear-solver controls")
        return super().polarizability(wavelength_nm, tol=tol, restart=restart, maxiter=maxiter,
                                      callback=callback, reuse_guesses=reuse_guesses, deflation=deflation)
