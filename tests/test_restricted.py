"""Independent spin-orbital checks of the compact closed-shell solver."""

from itertools import permutations, product

import numpy as np
import pytest

from itercc import Hamiltonian, RestrictedCC, TensorCC
from itercc.restricted import _near_zero_triples_denominator


def test_triples_denominator_range_shortcut():
    vir = np.array([[[-2.5, -1.0, 0.5, 2.0]]])
    for occupied in ([], [-5.0, -4.0], [4.0, 5.0], [-4.0, 4.0],
                     [0.5 + 0.5e-12], [-2.5 - 0.5e-12], [2.0 + 2e-12]):
        expected = any(np.min(np.abs(energy - vir)) < 1e-12 for energy in occupied)
        assert _near_zero_triples_denominator(occupied, vir) == expected


def make_model(no, nv, seed=137):
    rng = np.random.default_rng(seed)
    n = no + nv
    diagonal = np.r_[np.linspace(-1.6, -0.8, no), np.linspace(0.25, 1.2, nv)]
    h = np.diag(diagonal)
    delta = rng.normal(scale=0.03, size=(n, n))
    h += delta + delta.T
    factor = rng.normal(scale=0.09, size=(n + 3, n, n))
    factor = factor + factor.transpose(0, 2, 1)
    eri = np.einsum("xpq,xrs->pqrs", factor, factor)
    return h, eri, no, 0.4


def spatial_to_spin(solver, values):
    """Expand E(ai) products by independently antisymmetrizing spin labels."""
    t1, t2, packed = values
    no, nv = t1.shape
    t3 = np.zeros((no, no, no, nv, nv, nv))
    for labels, tile in zip(solver.triples_indices, packed):
        for p in permutations(range(3)):
            t3[tuple(labels[i] for i in p)] = tile.transpose(p)
    result = []
    spin = (slice(0, None, 2), slice(1, None, 2))
    for rank, value in enumerate((t1, t2, t3), 1):
        so = np.zeros(tuple(2 * length for length in value.shape))
        for occ in product(range(2), repeat=rank):
            for p in permutations(range(rank)):
                vir = tuple(occ[i] for i in p)
                parity = sum(x > y for i, x in enumerate(p) for y in p[i + 1:])
                axes = tuple(range(rank)) + tuple(rank + i for i in p)
                so[tuple(spin[s] for s in occ + vir)] += (-1)**parity * value.transpose(axes)
        result.append(so.transpose(tuple(range(rank, 2 * rank)) + tuple(range(rank))))
    return tuple(result)


def trial_amplitudes(solver, scale, with_t3):
    rng = np.random.default_rng(59)
    t1, t2, t3 = (rng.normal(scale=scale, size=shape) for shape in solver.shapes)
    t2 = (t2 + t2.transpose(1, 0, 3, 2)) / 2
    if with_t3:
        for slot, ((i, j, k), tile) in enumerate(zip(solver.triples_indices, t3)):
            if i == j:
                t3[slot] = (tile + tile.swapaxes(0, 1)) / 2
            elif j == k:
                t3[slot] = (tile + tile.swapaxes(1, 2)) / 2
        solver.layout.purify(t3)
    else:
        t3.fill(0.0)
    return t1, t2, t3


@pytest.mark.parametrize("no,nv", [(1, 2), (2, 3), (3, 1), (3, 2), (3, 4)])
@pytest.mark.parametrize("with_t3", [False, True])
@pytest.mark.parametrize("pair_ladder", [False, True])
def test_packed_residuals_match_spin_orbital_equations(no, nv, with_t3, pair_ladder):
    model = make_model(no, nv)
    restricted = RestrictedCC(*model, ladder_batch=3, pair_ladder=pair_ladder)
    dense = TensorCC(Hamiltonian.restricted(*model))
    amplitudes = trial_amplitudes(restricted, 0.07, with_t3)
    spatial = restricted.residuals(amplitudes)
    spin = spatial_to_spin(restricted, amplitudes)
    for actual, expected in zip(spatial_to_spin(restricted, spatial), dense.residuals(spin)):
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
    assert restricted.reference_energy == pytest.approx(dense.reference_energy, abs=1e-14, rel=0)
    assert restricted.correlation_energy(amplitudes) == pytest.approx(
        dense.correlation_energy(spin), abs=1e-14, rel=0
    )


@pytest.mark.parametrize("diis_triples", [False, True])
def test_converges_all_ranks_and_can_restart(diis_triples):
    model = make_model(3, 3)
    restricted = RestrictedCC(*model)
    dense = TensorCC(Hamiltonian.restricted(*model))
    criteria = dict(energy_tol=1e-13, residual_tol=1e-11)
    partial = restricted.solve(max_iter=4, diis_triples=diis_triples, **criteria)
    assert not partial.converged
    saved = partial.amplitudes[-1].copy()
    result = restricted.solve(initial=partial.amplitudes, diis_triples=diis_triples, **criteria)
    assert result.converged
    np.testing.assert_array_equal(partial.amplitudes[-1], saved)
    assert np.linalg.norm(result.amplitudes[-1]) > 1e-5
    expected = dense.solve(**criteria)
    assert expected.converged
    assert result.energy == pytest.approx(expected.energy, abs=1e-12, rel=0)
    spin = spatial_to_spin(restricted, result.amplitudes)
    assert max(np.max(np.abs(r)) for r in dense.residuals(spin)) <= 3e-11

@pytest.mark.parametrize("no,nv,with_t3,pair_ladder", [
    (1, 2, False, True), (3, 1, False, True), (2, 3, False, True),
    (3, 4, True, True), (3, 2, True, False),
])
def test_cuda_residual_matches_cpu(no, nv, with_t3, pair_ladder):
    import os

    if os.environ.get("ITERCC_TEST_CUDA") != "1":
        pytest.skip("set ITERCC_TEST_CUDA=1 to run CUDA checks")
    pytest.importorskip("cupy")
    model = make_model(no, nv)
    cpu = RestrictedCC(*model, ladder_batch=3, pair_ladder=pair_ladder)
    gpu = RestrictedCC(*model, ladder_batch=3, pair_ladder=pair_ladder,
                       residual_backend="cuda")
    amplitudes = trial_amplitudes(cpu, 0.07, with_t3)
    for expected, actual in zip(cpu.residuals(amplitudes), gpu.residuals(amplitudes)):
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)


def test_checkpoint_callback_reports_restartable_evaluated_state():
    solver = RestrictedCC(*make_model(3, 2))
    snapshots = {}

    def save(index, record, amplitudes):
        snapshots[index] = (record, tuple(a.copy() for a in amplitudes))

    result = solver.solve(max_iter=4, diis_size=0, checkpoint_callback=save)
    assert len(snapshots) == 4
    assert snapshots[4][0] == result.iterations[-1]
    resumed = solver.solve(initial=snapshots[3][1], max_iter=2, diis_size=0)
    for expected, actual in zip(result.amplitudes, resumed.amplitudes):
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)


def test_cuda_solve_converges():
    import os

    if os.environ.get("ITERCC_TEST_CUDA") != "1":
        pytest.skip("set ITERCC_TEST_CUDA=1 to run CUDA checks")
    pytest.importorskip("cupy")
    model = make_model(2, 3)
    criteria = dict(energy_tol=1e-12, residual_tol=1e-10)
    expected = RestrictedCC(*model).solve(**criteria)
    actual = RestrictedCC(*model, residual_backend="cuda").solve(**criteria)
    assert expected.converged and actual.converged
    assert actual.energy == pytest.approx(expected.energy, abs=1e-12, rel=0)
    assert max(actual.residual_max) < 1e-10


def test_block_diis_agrees_with_flat_diis():
    from itercc.restricted import _BlockDIIS
    from itercc.solver import _DIIS

    rng = np.random.default_rng(11)
    block, flat = _BlockDIIS(3), _DIIS(3, np)
    for _ in range(7):
        amplitudes = tuple(rng.normal(size=(4,) * rank) for rank in range(1, 4))
        steps = tuple(rng.normal(size=a.shape) for a in amplitudes)
        block.reserve_slot()
        actual = block.update(amplitudes, steps)
        expected = flat.update(amplitudes, steps)
        for got, wanted in zip(actual, expected):
            np.testing.assert_allclose(got, wanted, atol=2e-14, rtol=0)


class _OvershootingTriples(RestrictedCC):
    """Triples residual scaled by 3: the same solution, but a plain Jacobi T3 step overshoots by
    a factor -2 (a stand-in for a stiff triples mode)."""

    def _residuals(self, amplitudes, *args, **kwargs):
        r1, r2, r3 = super()._residuals(amplitudes, *args, **kwargs)
        return r1, r2, 3.0 * r3


def test_auto_triples_diis_rescues_a_diverging_triples_mode():
    model = make_model(3, 3)
    criteria = dict(energy_tol=1e-12, residual_tol=1e-10, max_iter=200)
    expected = RestrictedCC(*model).solve(**criteria)
    assert expected.converged
    with pytest.raises(FloatingPointError), np.errstate(over="ignore", invalid="ignore"):
        _OvershootingTriples(*model).solve(diis_triples=False, **criteria)
    with pytest.warns(RuntimeWarning, match="extrapolating T3 in DIIS"):
        result = _OvershootingTriples(*model).solve(**criteria)
    assert result.converged
    assert result.energy == pytest.approx(expected.energy, abs=1e-10, rel=0)
    with pytest.raises(ValueError):
        RestrictedCC(*model).solve(diis_triples="yes", **criteria)
