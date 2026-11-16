"""Shared Krylov solvers: deflated-restart GMRES and thick-restart Davidson."""
import numpy as np
import pytest

from itercc.krylov import Columns, davidson, gmres, orthonormal_columns


def operator(n=160, seed=3, small=6, complex_shift=0.):
    """Nonsymmetric operator with a few small eigenvalues that stall GMRES(m)."""
    rng = np.random.default_rng(seed)
    values = np.linspace(1., 4., n)
    values[:small] = np.geomspace(2e-3, 5e-2, small)
    basis = np.eye(n) + 0.15 * rng.standard_normal((n, n)) / np.sqrt(n)
    matrix = basis @ np.diag(values) @ np.linalg.inv(basis)
    diagonal = np.diag(matrix).copy()
    return matrix - complex_shift * np.eye(n), diagonal - complex_shift


def counted(matrix):
    calls = []

    def action(vector):
        calls.append(1)
        return matrix @ vector
    return action, calls


@pytest.mark.parametrize("deflation", [0, 4, None])
def test_gmres_matches_dense_solution(deflation):
    matrix, diagonal = operator()
    rhs = np.random.default_rng(1).standard_normal(len(matrix))
    action, calls = counted(matrix)
    x, relative, steps = gmres(action, rhs, diagonal, np, tol=1e-10, restart=12, maxiter=400,
                               deflation=deflation)
    assert relative <= 1e-10
    np.testing.assert_allclose(x, np.linalg.solve(matrix, rhs), rtol=0, atol=1e-7)
    # Restarts reuse the implicit residual: one true residual at the end.
    assert len(calls) == steps + 1


def test_deflated_restarts_beat_classical_restarts():
    matrix, diagonal = operator()
    rhs = np.random.default_rng(2).standard_normal(len(matrix))
    steps = {}
    for deflation in (0, 4):
        steps[deflation] = gmres(lambda v: matrix @ v, rhs, diagonal, np, tol=1e-9, restart=12,
                                 maxiter=600, deflation=deflation)[2]
    assert steps[4] < 0.85 * steps[0]


def test_complex_shifted_system():
    matrix, diagonal = operator(complex_shift=0.3 - 0.02j)
    rhs = np.random.default_rng(4).standard_normal(len(matrix)) + 0j
    x, relative, _ = gmres(lambda v: matrix @ v, rhs, diagonal, np, tol=1e-10, restart=12,
                           maxiter=600)
    assert relative <= 1e-10
    np.testing.assert_allclose(x, np.linalg.solve(matrix, rhs), rtol=0, atol=1e-7)


def test_initial_guess_and_iteration_limit():
    matrix, diagonal = operator()
    rhs = np.random.default_rng(6).standard_normal(len(matrix))
    exact = np.linalg.solve(matrix, rhs)
    action, calls = counted(matrix)
    x, relative, steps = gmres(action, rhs, diagonal, np, tol=1e-6, restart=10, maxiter=5,
                               initial=exact)
    assert steps == 0 and relative < 1e-12 and len(calls) == 1
    x, relative, steps = gmres(lambda v: matrix @ v, rhs, diagonal, np, tol=1e-12, restart=10,
                               maxiter=7)
    assert steps == 7 and relative > 1e-12


def test_column_block_transform_is_in_place_basis_change():
    rng = np.random.default_rng(7)
    block = Columns(np, 37, 6, np.float64, chunk_bytes=64)
    rows = rng.standard_normal((5, 37))
    for row in rows:
        block.append(row)
    change, _ = orthonormal_columns(rng.standard_normal((5, 3)))
    block.transform(change)
    assert block.count == 3
    np.testing.assert_allclose(block.block(), change.T @ rows, rtol=0, atol=1e-13)


def nonsymmetric_spectrum(n=140, seed=8):
    rng = np.random.default_rng(seed)
    values = np.concatenate([[0.31, 0.3105, 0.47], np.linspace(0.8, 6., n - 3)])
    basis = np.eye(n) + 0.1 * rng.standard_normal((n, n)) / np.sqrt(n)
    return basis @ np.diag(values) @ np.linalg.inv(basis), values


@pytest.mark.parametrize("thick, olsen", [(False, False), (True, False), (True, True)])
def test_davidson_roots_and_thick_restart(thick, olsen):
    matrix, values = nonsymmetric_spectrum()
    calls = []

    def sigma(vector):
        calls.append(1)
        return matrix @ vector

    def certify(root, energy):
        return float(np.linalg.norm(matrix @ root - energy * root))

    outcome = davidson(sigma, np.diag(matrix).copy(), lambda v: v.copy(), size=len(matrix), nroots=2,
                       xp=np, residual_tol=1e-9, max_iter=300, max_space=8, certify=certify,
                       retain_previous=thick, olsen=olsen)
    assert outcome.converged
    np.testing.assert_allclose(outcome.values, values[:2], rtol=0, atol=1e-10)
    for root, energy in zip(outcome.vectors, outcome.values):
        assert np.linalg.norm(matrix @ root - energy * root) <= 1e-9
    assert any(b["subspace"] < a["subspace"] for a, b in zip(outcome.history, outcome.history[1:]))


def test_previous_ritz_vectors_reduce_restarted_iterations():
    matrix, _ = nonsymmetric_spectrum(seed=9)
    counts = {}
    for thick in (False, True):
        calls = []

        def sigma(vector):
            calls.append(1)
            return matrix @ vector
        outcome = davidson(sigma, np.diag(matrix).copy(), lambda v: v.copy(), size=len(matrix), nroots=1,
                           xp=np, residual_tol=1e-9, max_iter=400, max_space=5, retain_previous=thick)
        assert outcome.converged
        counts[thick] = len(calls)
    assert counts[True] < counts[False]


def test_custom_preconditioner_matches_diagonal_default():
    matrix, values = nonsymmetric_spectrum(seed=10)
    diagonal = np.diag(matrix).copy()

    def precondition(residual, energy, vector):
        denominator = energy - diagonal
        denominator = np.where(np.abs(denominator) < 1e-3, np.where(denominator < 0, -1e-3, 1e-3), denominator)
        return residual / denominator

    runs = [davidson(lambda v: matrix @ v, diagonal, lambda v: v.copy(), size=len(matrix), nroots=2, xp=np,
                     residual_tol=1e-9, max_iter=200, max_space=10, precondition=pre)
            for pre in (None, precondition)]
    assert all(run.converged for run in runs)
    assert len(runs[0].history) == len(runs[1].history)
    np.testing.assert_allclose(runs[0].values, runs[1].values, rtol=0, atol=1e-12)


def test_inexact_operator_reports_its_true_residual():
    matrix, diagonal = operator()
    rng = np.random.default_rng(12)

    def noisy(vector):
        out = matrix @ vector
        return out * (1 + 1e-7 * rng.standard_normal(out.shape))
    rhs = rng.standard_normal(len(matrix))
    x, relative, steps = gmres(noisy, rhs, diagonal, np, tol=1e-11, restart=12, maxiter=150)
    assert steps == 150
    assert abs(relative - np.linalg.norm(rhs - matrix @ x) / np.linalg.norm(rhs)) < 1e-6
