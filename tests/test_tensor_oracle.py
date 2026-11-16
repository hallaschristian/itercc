"""Use the unchanged seed's determinant convention as an independent oracle."""

from itertools import combinations, permutations
from pathlib import Path
import sys

import numpy as np
import pytest

from itercc import Hamiltonian, TensorCC

sys.path.insert(0, str(Path(__file__).with_name("oracle")))
from reference_cc import excitation, solve


def determinant_problem(model, nalpha, nbeta):
    n = len(model.h)
    even, odd = range(0, n, 2), range(1, n, 2)
    masks = sorted(sum(1 << p for p in a + b)
                   for a in combinations(even, nalpha) for b in combinations(odd, nbeta))
    index = {mask: i for i, mask in enumerate(masks)}
    h = np.eye(len(masks)) * model.constant
    pairs = tuple(combinations(range(n), 2))
    for col, mask in enumerate(masks):
        occupied = tuple(p for p in range(n) if mask & (1 << p))
        for i in occupied:
            for a in range(n):
                moved = excitation(mask, (i,), (a,))
                if moved and moved[0] in index:
                    h[index[moved[0]], col] += moved[1] * model.h[a, i]
        for i, j in combinations(occupied, 2):
            for a, b in pairs:
                moved = excitation(mask, (i, j), (a, b))
                if moved and moved[0] in index:
                    h[index[moved[0]], col] += moved[1] * model.g[a, b, i, j]
    return h, masks


def odd_sign(p):
    return (-1) ** sum(i > j for i, j in combinations(p, 2))


def amplitudes_for_sector(solver, masks, rng, scale):
    no = solver.hamiltonian.nocc
    ref = (1 << no) - 1
    amplitudes = [np.zeros(shape) for shape in solver.shapes]
    excitations = []
    for row, mask in enumerate(masks):
        holes = tuple(i for i in range(no) if not mask & (1 << i))
        particles = tuple(a for a in range(no, len(solver.fock)) if mask & (1 << a))
        rank = len(holes)
        if not 0 < rank <= solver.rank:
            continue
        canonical = tuple(a - no for a in particles) + holes
        value = rng.normal(scale=scale)
        for ap in permutations(particles):
            for ip in permutations(holes):
                amplitudes[rank - 1][tuple(a - no for a in ap) + ip] = (
                    odd_sign(ap) * odd_sign(ip) * value
                )
        excitations.append((row, holes, particles, canonical, value))
    return amplitudes, excitations, ref


def apply_cluster_exponential(t, vector, order):
    result = vector.copy()
    for power in range(1, order + 1):
        vector = (t @ vector) / power
        result += vector
    return result


def interacting_model():
    rng = np.random.default_rng(395)
    h = rng.normal(scale=0.035, size=(4, 4))
    h = h + h.T + np.diag([-2.2, -1.15, 0.55, 1.2])
    factors = rng.normal(scale=0.13, size=(6, 4, 4))
    factors += factors.transpose(0, 2, 1).copy()
    return Hamiltonian.restricted(h, np.einsum("xpq,xrs->pqrs", factors, factors), 2, 0.7)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("scale", [0.0, 0.08, 0.23])
def test_unconverged_projected_residuals(rank, scale):
    model = interacting_model()
    solver = TensorCC(model, rank=rank)
    dense = TensorCC(model, rank=rank, pair_ladder=False, factored_triples=False)
    h, masks = determinant_problem(model, 2, 2)
    index = {mask: i for i, mask in enumerate(masks)}
    amplitudes, excitations, ref = amplitudes_for_sector(
        solver, masks, np.random.default_rng(8834), scale
    )
    t = np.zeros_like(h)
    for _, holes, particles, _, value in excitations:
        for col, mask in enumerate(masks):
            moved = excitation(mask, holes, particles)
            if moved and moved[0] in index:
                t[index[moved[0]], col] += moved[1] * value
    reference_vector = np.eye(len(masks))[:, index[ref]]
    ket = apply_cluster_exponential(t, reference_vector, model.nocc)
    hbar = apply_cluster_exponential(-t, h @ ket, model.nocc)
    residuals = solver.residuals(amplitudes)
    projected = [np.zeros(shape) for shape in solver.shapes]
    for row, holes, particles, _, _ in excitations:
        phase = excitation(ref, holes, particles)[1]
        for ap in permutations(particles):
            for ip in permutations(holes):
                projected[len(holes) - 1][tuple(a - model.nocc for a in ap) + ip] = (
                    odd_sign(ap) * odd_sign(ip) * phase * hbar[row]
                )
    for actual, expected, plain in zip(residuals, projected, dense.residuals(amplitudes)):
        np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=0)
        np.testing.assert_allclose(actual, plain, atol=3e-14, rtol=0)
    assert solver.reference_energy + solver.correlation_energy(amplitudes) == pytest.approx(
        hbar[index[ref]], abs=1e-13, rel=0
    )


@pytest.mark.parametrize("rank", [2, 3])
def test_converged_closed_shell_matches_determinant_oracle(rank):
    model = interacting_model()
    h, masks = determinant_problem(model, 2, 2)
    ref = (1 << model.nocc) - 1
    oracle = solve(h, masks, ref, len(model.h), max_rank=rank)
    solver = TensorCC(model, rank=rank)
    result = solver.solve(energy_tol=1e-13, residual_tol=1e-11)
    assert result.converged
    assert result.energy == pytest.approx(oracle.energy, abs=2e-12, rel=0)
    for maxima, residual in zip(result.residual_max, solver.residuals(result.amplitudes)):
        assert maxima == np.max(np.abs(residual))
    if rank == 3:
        assert np.linalg.norm(result.amplitudes[-1]) > 1e-5
        assert abs(result.energy - np.linalg.eigvalsh(h)[0]) > 1e-8


def test_iteration_limit_retains_reported_iterate():
    solver = TensorCC(interacting_model())
    result = solver.solve(max_iter=1)
    assert not result.converged
    assert len(result.iterations) == 1
    assert result.residual_max == tuple(np.max(np.abs(r)) for r in solver.residuals(result.amplitudes))


def test_initial_amplitudes_require_separate_antisymmetry():
    solver = TensorCC(interacting_model())
    amplitudes = [np.zeros(shape) for shape in solver.shapes]
    amplitudes[1][0, 1, 0, 1] = 0.01
    with pytest.raises(ValueError, match="T2 must be separately antisymmetric"):
        solver.solve(initial=amplitudes)
