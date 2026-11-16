"""Initial guesses from solved neighbouring states (field continuation)."""
import numpy as np
import pytest

from itercc import RestrictedCCSDTQ
from itercc.continuation import continuation_plan, lagrange_weights
from itercc.spatial_symmetry import conserved_labels
from test_quadruples import LABELS, cuda, symmetric_model  # noqa: F401  (fixture)


def test_lagrange_weights_reproduce_polynomials():
    nodes = (-0.5, 0.0, 0.5)
    for target in (1.0, -1.0, 0.25):
        weights = lagrange_weights(nodes, target)
        for f in (lambda x: 3.0, lambda x: 2 * x - 1, lambda x: x * x + x):
            assert abs(sum(w * f(x) for w, x in zip(weights, nodes)) - f(target)) < 1e-12
    assert lagrange_weights((-0.5, 0.0, 0.5), 1.0) == pytest.approx([1.0, -3.0, 3.0])
    with pytest.raises(ValueError):
        lagrange_weights((0.0, 0.0), 1.0)


def test_continuation_plan_uses_the_nearest_states():
    available = {0.0: "zero", 0.5: "plus", -0.5: "minus", 1.0: "far"}
    plan = continuation_plan(available, -1.0, max_nodes=2)
    assert [key for key, _ in plan] == ["minus", "zero"]
    assert [w for _, w in plan] == pytest.approx([2.0, -1.0])
    assert continuation_plan({0.0: "zero"}, 0.5) == [("zero", 1.0)]
    assert continuation_plan({}, 0.5) == []


def operator(labels, irrep, seed=4):
    rng = np.random.default_rng(seed)
    n = len(labels)
    value = rng.normal(size=(n, n))
    value = value + value.T
    return np.where((labels[:, None] ^ labels) == irrep, value, 0.0)


def solved(model, labels, field, d, **options):
    h, eri, no, constant = model
    solver = RestrictedCCSDTQ(h + field * d, eri, no, constant, block_size=2, orbital_irreps=labels)
    settings = dict(max_iter=120, damping=0.1, energy_tol=1e-11, residual_tol=1e-9, diis_size=6)
    settings.update(options)
    return solver, solver.solve(**settings)


def test_states_move_between_symmetry_layouts(cuda):
    model = symmetric_model()
    dense, result = solved(model, None, 0.0, 0.0, max_iter=3, diis_size=0)
    reference = cuda.asnumpy(result.amplitudes[3]).reshape(dense.layout.dense_shape)
    c2v = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=LABELS)
    cs = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=conserved_labels(LABELS, [2]))
    from_dense = c2v.combine_states([(1.0, result.amplitudes, None)])
    np.testing.assert_array_equal(c2v.quadruples_to_dense(from_dense[3]), reference)
    to_cs = cs.combine_states([(1.0, from_dense, c2v.layout_signature())])
    np.testing.assert_array_equal(cs.quadruples_to_dense(to_cs[3]), reference)
    twice = cs.combine_states([(2.0, from_dense, c2v.layout_signature()),
                               (-1.0, to_cs, cs.layout_signature())])
    np.testing.assert_allclose(cs.quadruples_to_dense(twice[3]), reference, rtol=0, atol=1e-15)
    for a, b in zip(twice[:3], result.amplitudes[:3]):
        np.testing.assert_allclose(cuda.asnumpy(a), cuda.asnumpy(b), rtol=0, atol=1e-15)


@pytest.mark.parametrize("irrep", [0, 2])
def test_field_continuation_converges_faster_to_the_same_state(cuda, irrep):
    model = symmetric_model()
    d = operator(LABELS, irrep)
    labels = LABELS if irrep == 0 else conserved_labels(LABELS, [irrep])
    zero_solver, zero = solved(model, LABELS, 0.0, d)
    assert zero.converged
    cold_solver, cold = solved(model, labels, 0.02, d)
    warm_solver = RestrictedCCSDTQ(model[0] + 0.02 * d, *model[1:], block_size=2, orbital_irreps=labels)
    initial = warm_solver.combine_states([(1.0, zero.amplitudes, zero_solver.layout_signature())])
    warm = warm_solver.solve(initial=initial, max_iter=120, damping=0.1, energy_tol=1e-11,
                             residual_tol=1e-9, diis_size=6)
    assert cold.converged and warm.converged
    assert abs(warm.energy - cold.energy) < 1e-9
    assert len(warm.iterations) < len(cold.iterations)
    # A line through the zero- and first-field states predicts the second field better still.
    far_solver = RestrictedCCSDTQ(model[0] + 0.04 * d, *model[1:], block_size=2, orbital_irreps=labels)
    plan = continuation_plan({0.0: (zero, zero_solver), 0.02: (warm, warm_solver)}, 0.04)
    guess = far_solver.combine_states([(w, state.amplitudes, solver.layout_signature())
                                       for (state, solver), w in plan])
    constant = far_solver.combine_states([(1.0, warm.amplitudes, warm_solver.layout_signature())])
    linear_error = max(float(abs(r).max()) for r in far_solver.residuals(guess))
    constant_error = max(float(abs(r).max()) for r in far_solver.residuals(constant))
    assert linear_error < 0.5 * constant_error


def test_weighted_sum_combines_plain_amplitude_tuples():
    from itercc.continuation import weighted_sum

    a = (np.ones(3), np.full((2, 2), 2.0))
    b = (np.zeros(3), np.ones((2, 2)))
    total = weighted_sum([(2.0, a), (-1.0, b)])
    np.testing.assert_array_equal(total[0], 2 * np.ones(3))
    np.testing.assert_array_equal(total[1], np.full((2, 2), 3.0))
    with pytest.raises(ValueError):
        weighted_sum([(1.0, a), (1.0, (np.zeros(3),))])
