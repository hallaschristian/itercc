import numpy as np

from itercc import RestrictedCC
from itercc._diis_columns import HostTriplesColumns
from itercc._triples_layout import TriplesLayout
from itercc.restricted import _BlockDIIS, _SymmetricBlockDIIS
from itercc.spatial_symmetry import project_amplitudes, project_integrals
from test_restricted import make_model, trial_amplitudes


def test_compact_diis_preserves_history_on_exact_leakage():
    no, nv = 3, 4
    layout = TriplesLayout(no, nv)
    labels = np.array([0, 1, 2, 0, 1, 2, 3])
    columns = HostTriplesColumns(labels[:no], labels[no:], layout)
    assert columns.size < np.prod(layout.shape)
    dense, compact = _BlockDIIS(3), _SymmetricBlockDIIS(3, columns)
    rng = np.random.default_rng(293)
    for i in range(5):
        vector, steps = [tuple(rng.normal(size=shape) for shape in ((no, nv), (no, no, nv, nv), layout.shape))
                         for _ in range(2)]
        vector, _ = project_amplitudes(vector, labels, tolerance=10)
        steps, _ = project_amplitudes(steps, labels, tolerance=10)
        if i == 3:
            steps[-1][0, 0, 0, 0] = 1e-18
            assert columns.pack(steps[-1]) is None
        expected, actual = dense.take_step(vector, steps), compact.take_step(vector, steps)
        for left, right in zip(expected, actual):
            np.testing.assert_allclose(left, right, atol=2e-14, rtol=0)
        assert (compact.columns is None) == (i >= 3)
    assert compact.history[0][0][-1].shape == layout.shape


def test_compact_diis_converges_the_same_full_ccsdt_problem():
    h, eri, no, constant = make_model(3, 3)
    labels = np.array([0, 1, 1, 0, 1, 1])
    h, eri, _ = project_integrals(h, eri, labels, tolerance=1)
    kw = dict(orbital_irreps=labels)
    dense, compact = RestrictedCC(h, eri, no, constant, **kw), RestrictedCC(
        h, eri, no, constant, host_diis_columns=True, **kw)
    initial, _ = project_amplitudes(trial_amplitudes(dense, 0.02, True), labels, tolerance=1)
    opts = dict(initial=initial, energy_tol=1e-12, residual_tol=1e-10, diis_size=3, diis_triples=True)
    expected, actual = dense.solve(**opts), compact.solve(**opts)
    assert expected.converged and actual.converged
    assert abs(expected.energy - actual.energy) < 1e-12
    assert max(actual.residual_max) <= 1e-10
