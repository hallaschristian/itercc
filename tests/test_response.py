"""Validate LR-CC polarizabilities against independent electronic calculations."""
import os

import numpy as np
import pytest

from itercc import DipoleResponse, Hamiltonian, ResponseCheckpoint, RestrictedCC
from itercc.response import HC_EH_NM, _shifted_guesses
from test_restricted import make_model
from test_tensor_oracle import determinant_problem


def model(no, nv):
    h, eri, _, constant = make_model(no, nv)
    rng = np.random.default_rng(3437)
    dipoles = rng.normal(size=(3, len(h), len(h)))
    dipoles = .5 * (dipoles + dipoles.transpose(0, 2, 1))
    return h, .1 * eri, constant, dipoles


def ground(h, eri, no, constant):
    solver = RestrictedCC(h, eri, no, constant)
    result = solver.solve(energy_tol=1e-13, residual_tol=1e-12, diis_triples=True)
    assert result.converged
    return solver, result


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("method", ["ccsd", "ccsdt"])
def test_two_electron_wavelength_tensor_matches_exact_diagonalization(backend, method):
    if backend == "cuda" and os.environ.get("ITERCC_TEST_CUDA") != "1":
        pytest.skip("CUDA check")
    h, eri, constant, dipoles = model(1, 2)
    solver, state = ground(h, eri, 1, constant)
    response = DipoleResponse(solver, state.amplitudes, dipoles, backend=backend, method=method)
    matrix, _ = determinant_problem(Hamiltonian.restricted(h, eri, 1, constant), 1, 1)
    energies, states = np.linalg.eigh(matrix)
    moments = []
    for d in dipoles:
        operator, _ = determinant_problem(Hamiltonian.restricted(d, np.zeros_like(eri), 1), 1, 1)
        moments.append(states[:, 1:].T @ operator @ states[:, 0])
    moments = np.asarray(moments)
    gaps = energies[1:] - energies[0]
    optically_active = gaps[np.max(np.abs(moments), axis=0) > 1e-5]
    near_pole = HC_EH_NM / (optically_active.min() * .998)
    for wavelength, damping in ((np.inf, 0.), (300., .01), (near_pole, .01)):
        shift = HC_EH_NM/wavelength + 1j*damping
        expected = (moments * (2*gaps/(gaps*gaps - shift*shift))) @ moments.T
        value = response.polarizability(wavelength, damping_Eh=damping, tol=1e-10)
        np.testing.assert_allclose(value.tensor_au, expected, atol=3e-9, rtol=0)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_full_triples_static_response_matches_fixed_orbital_finite_fields(backend):
    if backend == "cuda" and os.environ.get("ITERCC_TEST_CUDA") != "1":
        pytest.skip("CUDA check")
    h, eri, constant, dipoles = model(2, 2)
    solver, state = ground(h, eri, 2, constant)
    dipoles[1:] = 0
    response = DipoleResponse(solver, state.amplitudes, dipoles, backend=backend)
    value = response.polarizability(np.inf, tol=1e-10)
    estimates = []
    for strength in (.001, .002):
        plus = ground(h - strength*dipoles[0], eri, 2, constant)[1].energy
        minus = ground(h + strength*dipoles[0], eri, 2, constant)[1].energy
        estimates.append(-((plus-state.energy)+(minus-state.energy))/strength**2)
    expected = (4*estimates[0]-estimates[1])/3
    np.testing.assert_allclose(value.tensor_au[0, 0], expected, atol=2e-8, rtol=0)
    # Independently separate the one-body residual from the full Hamiltonian.
    p = dipoles[0]
    raw = RestrictedCC(p, np.zeros_like(eri), 2).residuals(state.amplitudes)
    actual = response._operator_residual(response.xp.asarray(p), response.amplitudes)
    for component, reference in zip(actual, raw):
        actual = component if backend == "cpu" else component.get()
        np.testing.assert_allclose(actual, reference, atol=1e-13, rtol=0)


def test_unsigned_dipole_input_matches_float_perturbations():
    solver = RestrictedCC(np.diag([-1., .5]), np.zeros((2, 2, 2, 2)), 1)
    amplitudes = tuple(np.zeros(shape) for shape in solver.shapes[:2])
    dipoles = np.zeros((3, 2, 2), dtype=np.uint8)
    dipoles[0, 0, 1] = dipoles[0, 1, 0] = 1
    response = DipoleResponse(solver, amplitudes, dipoles, method="ccsd")
    np.testing.assert_array_equal(response.perturbations, -dipoles.astype(float))
    expected = np.diag([4 / 1.5, 0., 0.])
    np.testing.assert_allclose(response.polarizability(np.inf).tensor_au, expected,
                               atol=1e-13, rtol=0)


@pytest.mark.parametrize("wavelength, controls", [
    (5e-324, {}), (np.inf, {"restart": 3.5}), (np.inf, {"maxiter": True}),
])
def test_invalid_response_controls_fail_before_iteration(wavelength, controls):
    solver = RestrictedCC(np.diag([-1., .5]), np.zeros((2, 2, 2, 2)), 1)
    amplitudes = tuple(np.zeros(shape) for shape in solver.shapes[:2])
    response = DipoleResponse(solver, amplitudes, np.zeros((3, 2, 2)), method="ccsd")
    with pytest.raises(ValueError, match="Wavelength|linear-solver controls"):
        response.polarizability(wavelength, **controls)


def test_recycled_solutions_keep_the_sweep_and_save_actions():
    """A wavelength sweep (static, two frequencies, a damped point) that starts
    each solve from the stored earlier solutions gives the tensors of
    independent solves with fewer Jacobian actions; the stored images (taken
    from the solved systems, not from actions) are the Jacobian actions of
    the stored directions, and the store's guesses are exact least-squares fits."""
    from itercc.krylov import RecycledImages

    h, eri, constant, dipoles = model(2, 3)
    solver, state = ground(h, eri, 2, constant)
    sweep = ((np.inf, 0.), (1064., 0.), (780., 0.), (780., .005))
    results = {}
    for gib in (0.0, 1.0):
        response = DipoleResponse(solver, state.amplitudes, dipoles, recycle_gib=gib)
        values = [response.polarizability(w, damping_Eh=d, tol=1e-10) for w, d in sweep]
        actions = sum(s["jacobian_actions"] for v in values for s in v.solves)
        results[gib] = ([v.tensor_au for v in values], actions)
        assert (response._store(0) is None) == (gib == 0.0)
    for reference, recycled in zip(results[0.0][0], results[1.0][0]):
        np.testing.assert_allclose(recycled, reference, atol=1e-8, rtol=0)
    assert results[1.0][1] < 0.9 * results[0.0][1], (results[0.0][1], results[1.0][1])
    # Each image is exact up to its solve's residual, which orthogonalization
    # magnifies by at most 1/drop = 1e3 (a wrong shift sign or real/imaginary
    # split would be off by order one).
    stored = response._store(0)
    assert stored.count > 10 and not stored.skipped
    for z, w in zip(stored.z.block(), stored.w.block()):
        np.testing.assert_allclose(response._action(response.spaces[0], z), w,
                                   atol=1e-5 * np.linalg.norm(w), rtol=0)
    # The default budget leaves the solve's Krylov basis and half the rest
    # when the platform reports a memory limit.
    from itercc._cpu_triples import host_memory_limit

    default = DipoleResponse(solver, state.amplitudes, dipoles).recycle_gib
    assert (default > 0) == (host_memory_limit() is not None)

    # The store against a dense least-squares fit, for a complex shift.
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(30, 30)) + 10 * np.eye(30)
    store = RecycledImages(np, 30, 8)
    for _ in range(6):
        z = rng.normal(size=30)
        store.add(z, matrix @ z)
    assert not store.add(store.z.data[0] * 2.0, matrix @ store.z.data[0] * 2.0)   # in the span
    rhs, shift = rng.normal(size=30) + 1j * rng.normal(size=30), 0.3 + 0.05j
    guess, estimate = store.guess(rhs, shift)
    basis = store.z.data[:store.count].T
    shifted = (matrix - shift * np.eye(30)) @ basis
    y = np.linalg.lstsq(shifted, rhs, rcond=None)[0]
    np.testing.assert_allclose(guess, basis @ y, atol=1e-9)
    assert abs(estimate - np.linalg.norm(rhs - shifted @ y)) < 1e-8


def test_shift_projection_matches_dense_fit_without_stored_images():
    rng = np.random.default_rng(391)
    matrix = 10 * np.eye(15) + rng.standard_normal((15, 15))
    rhs = rng.standard_normal(15)
    shift = .25
    eye = np.eye(len(rhs))
    old = [(np.linalg.solve(matrix - s * eye, rhs), s) for s in (.13, -.2)]
    basis = np.stack([v for v, _ in old], axis=1)
    image = (matrix - shift * eye) @ basis
    expected = np.linalg.lstsq(image, rhs, rcond=None)[0]
    guess, estimate = _shifted_guesses(rhs, shift, old, np)
    np.testing.assert_allclose(guess, basis @ expected, atol=2e-9, rtol=0)
    np.testing.assert_allclose(estimate, np.linalg.norm(rhs - image @ expected) / np.linalg.norm(rhs),
                               atol=1e-10, rtol=0)


def test_dipole_checkpoint_resumes_with_a_true_residual_check(tmp_path):
    h, eri, constant, dipoles = model(2, 3)
    dipoles[1:] = 0.
    solver, state = ground(h, eri, 2, constant)
    args = solver, state.amplitudes, dipoles
    cache = ResponseCheckpoint(tmp_path, "matched-ground")
    cold = DipoleResponse(*args, checkpoint=cache, recycle_gib=0.).polarizability(1100.)
    warm = DipoleResponse(*args, checkpoint=cache, recycle_gib=0.).polarizability(1100.)
    np.testing.assert_allclose(cold.tensor_au, warm.tensor_au, atol=2e-8, rtol=0)
    assert len(cold.solves) == 3
    assert all(s["checkpoint_guess"] and s["jacobian_actions"] == 1 for s in warm.solves[:2])
    assert not warm.solves[-1]["checkpoint_guess"]
    with pytest.raises(ValueError, match="checkpoint identity"):
        DipoleResponse(*args, checkpoint=ResponseCheckpoint(tmp_path, "other-ground")).polarizability(1100.)
