import numpy as np
import pytest

from itercc import RestrictedCC
from itercc.spatial_symmetry import project_amplitudes, project_integrals
from test_restricted import make_model, trial_amplitudes


def test_projection_audits_only_forbidden_entries_and_preserves_input():
    no, nv = 3, 4
    h, eri, _, constant = make_model(no, nv)
    labels = np.array([0, 1, 2, 0, 1, 2, 3])
    pairs = labels[:, None] ^ labels
    bad_h = pairs != 0
    bad_eri = pairs[:, :, None, None] != pairs
    h[bad_h], eri[bad_eri] = 4e-12, -5e-12
    h0, eri0, audit = project_integrals(h, eri, labels, tolerance=1e-11)
    for value, clean, mask, leak in zip((h, eri), (h0, eri0), (bad_h, bad_eri), audit):
        np.testing.assert_array_equal(clean[~mask], value[~mask])
        np.testing.assert_array_equal(clean[mask], 0)
        assert leak.count == np.count_nonzero(value[mask])
        assert leak.maximum == np.max(np.abs(value[mask]))
        assert leak.norm == pytest.approx(np.linalg.norm(value[mask]))
    with pytest.raises(ValueError, match="Symmetry leakage"):
        project_integrals(h, eri, labels, tolerance=4.5e-12)
    np.testing.assert_array_equal(h[bad_h], 4e-12)
    np.testing.assert_array_equal(eri[bad_eri], -5e-12)

    solver = RestrictedCC(h0, eri0, no, constant)
    amps = trial_amplitudes(solver, 0.02, True)
    pure, _ = project_amplitudes(amps, labels, tolerance=1)
    clean, audits = project_amplitudes(pure, labels, tolerance=0)
    for expected, actual, leak in zip(pure, clean, audits):
        np.testing.assert_array_equal(expected, actual)
        assert leak.count == 0
    pure[0][0, 1] = 2e-12
    with pytest.raises(ValueError, match="Symmetry leakage"):
        project_amplitudes(pure, labels, tolerance=1e-12)
    assert pure[0][0, 1] == 2e-12