"""Clamped-nuclei, fixed-orbital coupled-cluster electric dipole response."""
from dataclasses import dataclass
from functools import partial
from numbers import Integral

import numpy as np

from . import _restricted_equations as eq
from .eom import EOMSinglet, SingletSpace
from .hamiltonian import _real64
from .krylov import RecycledImages, gmres
from .solver import _dress_axis


HC_EH_NM = 45.5633525291


def _gmres(action, rhs, diagonal, xp, *, tol, restart, maxiter, initial=None, deflation=None):
    """Right-preconditioned GMRES with deflated restarts (see itercc.krylov.gmres).

    `restart` bounds the device basis to restart + 1 vectors; `deflation`
    harmonic Ritz vectors survive each restart (0 restores classical GMRES(m)).
    """
    return gmres(action, rhs, diagonal, xp, tol=tol, restart=restart, maxiter=maxiter,
                 initial=initial, deflation=deflation)


@dataclass(frozen=True)
class Polarizability:
    wavelength_nm: float
    frequency_Eh: float
    damping_Eh: float
    tensor_au: np.ndarray
    solves: tuple


class DipoleResponse:
    """Full CCSDT linear response via shifted right-hand Jacobian solves.

    For each response pair, contract the ground multiplier implicitly:
    lambda.G = -E_T.A^-1.G. This requires an additional zero-shift solve
    and retains the CC Lagrangian terms without a stored left Jacobian.

    Every solve of a symmetry sector uses the same Jacobian A, whatever its
    shift (the frequency, +-) or right-hand side (dipole or property). Each
    converged solution x of (A - s) x = b is therefore kept with its image
    A x = b + s x, which costs no action (`krylov.RecycledImages`, within
    `recycle_gib` of memory), and each new solve starts from its least-squares
    solution in the span of all earlier solutions of the sector: along a
    sweep this extrapolates in frequency, and the -omega solve of a
    wavelength builds on its +omega solution. Solutions, not Krylov
    directions: a direction costs as much memory as a solution, every
    recorded direction is orthogonalized against the store, and a full
    sweep has only two or three solutions per sector and wavelength.
    `recycle_gib=None` takes half of the host memory left after the engine
    and one solve's Krylov basis on the CPU, and keeps no store on CUDA
    (device memory); 0 turns it off.
    """
    RECYCLE_MAX = 64            # solutions per sector
    RECYCLE_WORK = 40           # vectors held back for one solve's (complex) Krylov basis

    def __init__(self, solver, amplitudes, dipoles, *, orbital_irreps=None,
                 dipole_irreps=None, backend="cpu", method="ccsdt", recycle_gib=None,
                 symmetry_tolerance=2e-9, host_krylov=False):
        raw = _real64(dipoles)
        nmo = len(solver.h)
        if (raw.shape != (3, nmo, nmo)
                or not np.isfinite(raw).all()
                or np.max(np.abs(raw - raw.transpose(0, 2, 1))) > 1e-10):
            raise ValueError("Expected three finite real symmetric MO dipoles")
        sectors = (0, 0, 0) if dipole_irreps is None else tuple(dipole_irreps)
        if len(sectors) != 3 or any(not isinstance(k, (int, np.integer)) or not 0 <= k <= 7
                                    for k in sectors):
            raise ValueError("Supply one Abelian irrep for each Cartesian dipole")
        if orbital_irreps is None and sectors != (0, 0, 0):
            raise ValueError("Nonzero dipole irreps require orbital labels")
        # The Jacobian acts with the symmetry-projected Hamiltonian when the
        # leakage allows (see EOMSinglet); the energy terms use the same one.
        self.engine = EOMSinglet(solver, amplitudes, orbital_irreps=orbital_irreps,
                                 backend=backend, method=method,
                                 symmetry_tolerance=symmetry_tolerance)
        acting = self.engine._primal
        self.xp, self.sectors = self.engine.xp, sectors
        self.krylov_xp = np if host_krylov else self.xp
        self.spaces = {0: self.engine.space}
        self.solver = solver
        self.o, self.v = solver.o, solver.v
        self.layout = getattr(self.engine.proxy.layout, "base", self.engine.proxy.layout)
        if orbital_irreps is not None:
            labels = self.engine.space.labels
            for axis, sector in enumerate(sectors):
                forbidden = (labels[:, None] ^ labels) != sector
                if np.max(np.abs(raw[axis, forbidden]), initial=0.) > 2e-9:
                    raise ValueError("Dipole has the wrong orbital symmetry")
        self.active_axes = tuple(bool(np.any(matrix)) for matrix in raw)
        response_sectors = {k for k, active in zip(sectors, self.active_axes) if active} - {0}
        self.diagonal = {0: self.engine.diagonal}
        for sector in response_sectors:
            space = SingletSpace(acting, orbital_irreps, sector,
                                 xp=self.xp, triples=method == "ccsdt")
            self.spaces[sector] = space
            self.diagonal[sector] = space.orbital_diagonal(acting.fock)
        self.perturbations = self.xp.asarray(-raw, dtype=self.xp.float64)
        self.amplitudes = self.engine.amplitudes
        # Plain products: the compiled GEMM executor on the CPU (the acting
        # solver's contract), NumPy/CuPy einsum otherwise.
        self.contract = acting._contract if self.xp is np else partial(eq.contract, xp=self.xp)
        self.rhs = tuple(self.spaces[k].pack(self._operator_residual(p, self.amplitudes))
                         if active else None
                         for k, p, active in zip(sectors, self.perturbations, self.active_axes))
        if self.krylov_xp is not self.xp:
            self.diagonal = {sector: self._krylov_array(value)
                             for sector, value in self.diagonal.items()}
            self.rhs = tuple(self._krylov_array(b) if b is not None else None for b in self.rhs)
        self.energy_weight = self.xp.asarray(
            2 * acting.g[self.o, self.o, self.v, self.v]
            - acting.g[self.o, self.o, self.v, self.v].swapaxes(2, 3))
        self.fock_ov = self.xp.asarray(acting.fock[self.o, self.v])
        self._guesses = {}
        self.deflation = None
        if recycle_gib is None:
            recycle_gib = 0.0
            if self.krylov_xp is np:
                from ._cpu_triples import host_memory_limit, resident_bytes

                limit = host_memory_limit()
                if limit is not None:
                    work = self.RECYCLE_WORK * 8 * max(space.size for space in self.spaces.values())
                    recycle_gib = max(0.0, (limit - resident_bytes() - work) / 2) / 2**30
        if not np.isfinite(recycle_gib) or recycle_gib < 0:
            raise ValueError("recycle_gib must be a nonnegative number of GiB")
        self.recycle_gib = float(recycle_gib)
        self._recycled = {}

    def _krylov_array(self, value):
        if self.krylov_xp is not self.xp and isinstance(value, self.xp.ndarray):
            return self.xp.asnumpy(value)
        return self.krylov_xp.asarray(value)

    def _operator_residual(self, operator, amplitudes):
        """One-body contribution to the spin-adapted CC residual."""
        xp, contract, o, v = self.xp, self.contract, self.o, self.v
        t1, t2 = amplitudes[:2]
        f = operator.astype(xp.result_type(operator, t1), copy=True)
        _dress_axis(f, 0, v, o, -t1.T, xp)
        _dress_axis(f, 1, o, v, t1, xp)
        r1 = f[v, o].T + contract("kc,ikac->ia", f[o, v], 2*t2 - t2.swapaxes(2, 3))
        r2 = contract("bc,ijac->ijab", f[v, v], t2)
        r2 -= contract("kj,ikab->ijab", f[o, o], t2)
        if self.engine.method == "ccsd":
            return r1, r2 + r2.transpose(1, 0, 3, 2)
        # The CUDA layout kernels are real. Complex directions use real
        # applications and recombination before reaching this routine.
        layout, t3 = self.layout, amplitudes[2]
        r3 = xp.zeros(layout.shape)
        if len(t3):
            for m in range(layout.no):
                q = layout.spin_plane(t3, m, xp=xp)
                r2 += 0.5 * contract("c,ijcab->ijab", f[m, v], q)
            vooo = contract("ld,ijad->alij", f[o, v], t2)
            for k in range(layout.no):
                plane = layout.plane(t3, k, axis=2, xp=xp)
                for j in range(layout.no):
                    left = plane[:, j]
                    seed = -contract("ali,lbc->iabc", vooo[:, :, :, j], t2[:, k])
                    seed += 0.5 * contract("ad,idbc->iabc", f[v, v], left)
                    seed -= 0.5 * contract("li,labc->iabc", f[o, o], left)
                    layout.add_seed(r3, seed, j, k)
            layout.purify(r3, xp=xp)
        return r1, r2 + r2.transpose(1, 0, 3, 2), r3

    def _store(self, sector):
        """The stored solutions of a sector with their images (None when off
        or too small); the budget is split over the solved sectors."""
        if sector not in self._recycled:
            space = self.spaces[sector]
            budget = self.recycle_gib * 2**30 / max(1, len(self.spaces))
            capacity = int(min(self.RECYCLE_MAX, budget // (2 * 8 * max(1, space.size))))
            self._recycled[sector] = (RecycledImages(self.krylov_xp, space.size, capacity)
                                      if capacity >= 2 else None)
        return self._recycled[sector]

    def _action(self, space, vector):
        """A vector (complex vectors as two real actions)."""
        packed = self.engine._packed_path(space.irrep)
        parts = [vector.real]
        if self.xp.iscomplexobj(vector) and bool(self.xp.any(vector.imag)):
            parts.append(vector.imag)
        images = [space.pack(self.engine._tangent(self.amplitudes, space.unpack(part, packed=packed),
                                                  irrep=space.irrep))
                  for part in parts]
        return images[0] if len(images) == 1 else images[0] + 1j * images[1]

    def _solve(self, sector, shift, rhs, tol, restart, maxiter, reports, key):
        xp, space = self.krylov_xp, self.spaces[sector]
        rhs = self._krylov_array(rhs)
        if float(xp.linalg.norm(rhs)) == 0:
            return self.xp.zeros(rhs.shape, dtype=rhs.dtype)
        dtype = xp.complex128 if xp.iscomplexobj(rhs) or np.iscomplexobj(shift) else xp.float64
        rhs = rhs.astype(dtype, copy=False)
        diagonal = self.diagonal[sector] - shift
        diagonal = xp.where(xp.abs(diagonal) < 1e-3, 1e-3, diagonal)
        actions = 0

        def action(vector):
            nonlocal actions
            actions += 1
            image = self._action(space, self.xp.asarray(vector))
            return self._krylov_array(image) - shift*vector

        cache_key = (*key, dtype)
        initial = self._guesses.get(cache_key)
        prior = initial is not None
        # The best start in the span of every earlier solution of this sector,
        # which holds the previous wavelength's solution unless the store
        # filled up (then that solution alone may be the better start).
        # Without such a solution, a start must at least halve the residual
        # to pay for the action that checks it.
        store, guess_residual, stored = self._store(sector), None, 0
        if store is not None and store.count and not (prior and store.skipped):
            stored = store.count
            guess, estimate = store.guess(rhs, shift)
            norm = float(xp.linalg.norm(rhs))
            if guess is not None and estimate < (1.0 if prior else 0.5) * norm:
                initial, guess_residual = guess.astype(dtype, copy=False), estimate / norm
        vector, defect, steps = _gmres(action, rhs, diagonal, xp, tol=tol, restart=restart,
                                       maxiter=maxiter, initial=initial, deflation=self.deflation)
        reports(dict(irrep=sector, solve=key, shift_Eh=[float(np.real(shift)), float(np.imag(shift))],
                     relative_residual=defect, jacobian_actions=actions, arnoldi_steps=steps,
                     prior_wavelength_guess=prior, recycled_directions=stored,
                     recycled_guess_relative_residual=guess_residual))
        if not np.isfinite(defect) or defect > tol:
            raise RuntimeError(f"CC response solve failed for irrep {sector} at {shift} Eh "
                               f"(relative residual {defect:.3e} after {steps} iterations)")
        self._guesses[cache_key] = vector
        if store is not None:
            # A x = b + shift x up to the residual (A is real: real and
            # imaginary parts separately). A part almost in the span is
            # dropped, which bounds the image error relative to the part.
            image = rhs + shift * vector
            store.add(vector.real, image.real, drop=1e-3)
            if xp.iscomplexobj(vector) and bool(xp.any(vector.imag)):
                store.add(vector.imag, image.imag, drop=1e-3)
        return self.xp.asarray(vector)

    def _unpack(self, axis, vector):
        return self.spaces[self.sectors[axis]].unpack(vector)

    def _step(self, direction):
        maximum = max((float(self.xp.max(self.xp.abs(x))) for x in direction if x.size), default=0.)
        return .04 / max(1., maximum)

    def _one_body_mixed(self, operator, direction):
        # The one-body residual is at most quadratic in T, so this
        # centered polynomial stencil differentiates it exactly in arithmetic.
        step = self._step(direction)
        space = self.spaces[0]
        result = space.pack(self._operator_residual(operator,
                    tuple(t + step*r for t, r in zip(self.amplitudes, direction))))
        result -= space.pack(self._operator_residual(operator,
                    tuple(t - step*r for t, r in zip(self.amplitudes, direction))))
        return result / (2*step)

    def _mixed(self, left, right, odd=False):
        # The BCH residual is at most quartic in T. Its directional
        # derivative is cubic and the four-point stencil is exact for cubics.
        # With `odd` (left and right of one irrep G != A1, exactly symmetric
        # operators) the even powers of the step carry irrep G, so the packed
        # A1 part is odd in the step and the stencil needs its +h, +2h halves.
        step = self._step(left)
        space = self.spaces[0]
        result = self.xp.zeros(space.size)
        stencil = (((1, 16), (2, -2)) if odd else ((-2, 1), (-1, -8), (1, 8), (2, -1)))
        for scale, weight in stencil:
            shifted = tuple(t + scale*step*r for t, r in zip(self.amplitudes, left))
            result += weight * space.pack(self.engine._tangent(shifted, right))
        del shifted
        self.engine.release_shifted()
        return result/(12*step)

    def _parts(self, direction):
        yield 1., tuple(x.real for x in direction)
        if any(self.xp.iscomplexobj(x) and bool(self.xp.any(x.imag)) for x in direction):
            yield 1j, tuple(x.imag for x in direction)

    def _bilinear_mixed(self, left, right, odd=False):
        u, v = list(self._parts(left)), list(self._parts(right))
        real = self._mixed(u[0][1], v[0][1], odd)
        if len(u) == len(v) == 2:
            imaginary_product = self._mixed(u[1][1], v[1][1], odd)
            sum_product = self._mixed(tuple(a+b for a, b in zip(u[0][1], u[1][1])),
                                      tuple(a+b for a, b in zip(v[0][1], v[1][1])), odd)
            return real - imaginary_product + 1j*(sum_product-real-imaginary_product)
        if len(u) == 2:
            return real + 1j*self._mixed(u[1][1], v[0][1], odd)
        if len(v) == 2:
            return real + 1j*self._mixed(u[0][1], v[1][1], odd)
        return real

    def _energy_gradient(self, vector):
        t1, t2 = self.amplitudes[:2]
        u1, u2 = self.spaces[0].unpack(vector, triples=False)
        contract = self.contract
        return (2*contract("ia,ia->", self.fock_ov, u1)
                + contract("ijab,ijab->", self.energy_weight, u2)
                + contract("ijab,ia,jb->", self.energy_weight, t1, u1)
                + contract("ijab,ia,jb->", self.energy_weight, u1, t1))

    def _component(self, i, j, minus_i, plus_j, controls, reports):
        left = self._unpack(i, minus_i)
        right = left if i == j and minus_i is plus_j else self._unpack(j, plus_j)
        pi, pj = self.perturbations[i], self.perturbations[j]
        xp, contract = self.xp, self.contract
        dtype = xp.result_type(minus_i, plus_j)
        combined = xp.zeros(self.spaces[0].size, dtype=dtype)
        mixed_operators = ((pi, tuple(a+b for a,b in zip(left, right))),) if i == j else (
            (pj, left), (pi, right))
        for operator, response in mixed_operators:
            for coefficient, direction in self._parts(response):
                combined += coefficient*self._one_body_mixed(operator, direction)
        # Both directions lie in the sector of irrep self.sectors[i] (== j).
        odd = bool(self.engine.symmetry_projected and self.sectors[i] == self.sectors[j] != 0)
        combined += self._bilinear_mixed(left, right, odd)
        direct = (2*contract("ia,ia->", pi[self.o, self.v], right[0])
                  + 2*contract("ia,ia->", pj[self.o, self.v], left[0])
                  + contract("ijab,ia,jb->", self.energy_weight, left[0], right[0])
                  + contract("ijab,ia,jb->", self.energy_weight, right[0], left[0]))
        correction = self._solve(0, 0., combined, *controls, reports, ("property", i, j))
        return complex(self._energy_gradient(correction) - direct)

    def polarizability(self, wavelength_nm, *, damping_Eh=0., tol=1e-8, restart=16, maxiter=160,
                       callback=None, reuse_guesses=True, deflation=None):
        """Return the Cartesian electronic tensor in au. Use +inf for static.

        Each linear solve keeps `deflation` harmonic Ritz vectors across GMRES
        restarts (None picks min(10, restart // 3); 0 is classical GMRES(m)).
        The basis is restart + 1 vectors on the backend or the host. Solves start from
        the earlier solutions (see the class docstring);
        `reuse_guesses=False` drops them first.
        """
        wavelength_nm, damping_Eh = float(wavelength_nm), float(damping_Eh)
        if (not wavelength_nm > 0 or not np.isfinite(damping_Eh) or damping_Eh < 0
                or not np.isfinite(tol) or not 0 < tol < 1
                or isinstance(restart, bool) or not isinstance(restart, Integral) or restart < 2
                or isinstance(maxiter, bool) or not isinstance(maxiter, Integral) or maxiter < 1):
            raise ValueError("Invalid wavelength, damping or linear-solver controls")
        frequency = HC_EH_NM / wavelength_nm
        if not np.isfinite(frequency):
            raise ValueError("Wavelength gives a nonfinite frequency")
        shift = frequency if damping_Eh == 0 else complex(frequency, damping_Eh)
        controls, reports = (tol, restart, maxiter), []
        if not reuse_guesses:       # a cold start: no earlier solutions or images
            self._guesses.clear()
            self._recycled.clear()
        if deflation is not None and (isinstance(deflation, bool) or not isinstance(deflation, Integral)
                                      or deflation < 0):
            raise ValueError("Deflation must be a nonnegative integer")
        self.deflation = deflation

        def record(report):
            reports.append(report)
            if callback is not None:
                callback(report)

        def dipole_solutions(sign):
            return [self._solve(k, sign*shift, -b, *controls, record, ("dipole", i, sign))
                    if b is not None else None
                    for i, (k, b) in enumerate(zip(self.sectors, self.rhs))]

        plus = dipole_solutions(1)
        minus = plus if shift == 0 else dipole_solutions(-1)
        if shift == 0:
            for i, value in enumerate(plus):
                if value is not None:
                    key = ("dipole", i, 1, self.krylov_xp.float64)
                    self._guesses[("dipole", i, -1, self.krylov_xp.float64)] = (
                        self._guesses[key] if key in self._guesses else self._krylov_array(value))
        tensor = np.zeros((3, 3), dtype=np.complex128)
        for i in range(3):
            for j in range(i, 3):
                if not (self.active_axes[i] and self.active_axes[j]) or self.sectors[i] != self.sectors[j]:
                    continue
                component = self._component(i, j, minus[i], plus[j], controls, record)
                if i != j and shift != 0:
                    component = .5*(component + self._component(j, i, minus[j], plus[i], controls, record))
                tensor[i, j] = tensor[j, i] = component
        if not np.isfinite(tensor).all():
            raise FloatingPointError("Nonfinite polarizability")
        return Polarizability(wavelength_nm, frequency, damping_Eh, tensor, tuple(reports))
