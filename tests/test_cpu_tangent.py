"""CPU EOM/response Jacobian actions through the compiled product-rule path."""

import numpy as np
import pytest

from itercc import EOMSinglet, RestrictedCC
from test_cpu_triples import LABELS, labelled_model
from test_restricted import make_model


def sigma_pair(monkeypatch, solver, ground, labels, irrep, seed):
    """The EOM actions of two random vectors with the CPU tangent and the
    dual-number loop (the CPU tangent's second action reuses its ground parts).
    With labels, the CPU tangent must run (half of) its triples on blocked T3
    in the occupied labels shifted by the irrep, and must not keep a finished
    action's direction alive in its cache."""
    from itercc import _cpu_tangent

    rng = np.random.default_rng(seed)
    actions, vectors, packed = [], None, []
    original = _cpu_tangent._blocked_direction

    def spy(*args):
        packed.append(original(*args))
        return packed[-1]
    monkeypatch.setattr(_cpu_tangent, "_blocked_direction", spy)
    for compiled in ("1", "0"):
        monkeypatch.setenv("ITERCC_CPU_TANGENT", compiled)
        eom = EOMSinglet(solver, ground.amplitudes, orbital_irreps=labels, irrep=irrep)
        if vectors is None:
            vectors = [eom.space.project(rng.normal(size=eom.space.size)) for _ in range(2)]
        actions.append(np.concatenate([eom.sigma(vector) for vector in vectors]))
        if compiled == "1" and labels is not None:
            ground_cache = eom._cpu_ground
            assert f"blocked layout {irrep}" in ground_cache and ("blocked off", irrep) not in ground_cache
            assert packed and all(p is not None for p in packed)
            assert "blocked direction" not in ground_cache
    return actions


@pytest.mark.parametrize("irrep", [0, 1, 3])
def test_cpu_tangent_matches_dual_numbers_by_irrep(monkeypatch, irrep):
    labels = LABELS[:7]
    solver = RestrictedCC(*labelled_model(3, 4, labels))
    ground = solver.solve(energy_tol=1e-12, residual_tol=1e-10, diis_triples=True, damping=0.1)
    assert ground.converged
    compiled, reference = sigma_pair(monkeypatch, solver, ground, labels, irrep, 31 + irrep)
    np.testing.assert_allclose(compiled, reference, rtol=0,
                               atol=1e-12 * max(1.0, float(np.max(np.abs(reference)))))


def test_cpu_tangent_matches_dual_numbers_without_labels(monkeypatch):
    solver = RestrictedCC(*make_model(2, 3))
    ground = solver.solve(residual_tol=1e-11, energy_tol=1e-12)
    assert ground.converged
    compiled, reference = sigma_pair(monkeypatch, solver, ground, None, 0, 7)
    np.testing.assert_allclose(compiled, reference, rtol=0, atol=1e-13)


def leaky_model(no, nv, labels, leak=3e-11):
    """A labelled model with rounding-level one-electron symmetry leakage."""
    h, eri, no, constant = labelled_model(no, nv, labels)
    forbidden = (labels[:, None] ^ labels) != 0
    noise = np.random.default_rng(17).normal(size=h.shape)
    h = h + leak * forbidden * (noise + noise.T)
    return h, eri, no, constant


def test_projection_within_tolerance_keeps_solves_and_actions_blocked(monkeypatch):
    """Integral leakage at rounding level: RestrictedCC projects it and stays on
    the blocked T3 path; EOM acts with the projected operators (the A1 triples
    on blocked T3) and its action matches the original operators' to the
    leakage level; symmetry_tolerance=0 keeps the original operators."""
    from itercc import _cpu_kernels

    if _cpu_kernels.library() is None:
        pytest.skip("no compiled core")
    labels = LABELS[:7]
    model = leaky_model(3, 4, labels)
    solver = RestrictedCC(*model, orbital_irreps=labels, symmetry_tolerance=2e-9)
    assert solver.symmetry_projected and max(solver.integral_symmetry_leakage) > 0
    ground = solver.solve(energy_tol=1e-12, residual_tol=1e-10, diis_triples=True, damping=0.1)
    assert ground.converged and solver._cpu_blocked
    assert solver._blocked_t3(*ground.amplitudes, 2) is not None
    exact = RestrictedCC(*labelled_model(3, 4, labels), orbital_irreps=labels).solve(
        energy_tol=1e-12, residual_tol=1e-10, diis_triples=True, damping=0.1)
    assert abs(ground.energy - exact.energy) < 1e-9
    assert not RestrictedCC(*model, orbital_irreps=labels).symmetry_projected     # the default

    unprojected = RestrictedCC(*model)                 # ground state of the leaky operators
    state = unprojected.solve(energy_tol=1e-12, residual_tol=1e-10, diis_triples=True, damping=0.1)
    actions = {}
    for tolerance in (2e-9, 0.0):
        eom = EOMSinglet(unprojected, state.amplitudes, orbital_irreps=labels, symmetry_tolerance=tolerance)
        assert eom.symmetry_projected == (tolerance > 0)
        vector = eom.space.project(np.random.default_rng(5).normal(size=eom.space.size))
        actions[tolerance] = eom.sigma(vector)
        if tolerance:
            assert eom._cpu_ground["blocked ground"][0][1] is not None     # A1: blocked T3
    np.testing.assert_allclose(actions[2e-9], actions[0.0], rtol=0,
                               atol=1e-8 * max(1.0, float(np.max(np.abs(actions[0.0])))))
