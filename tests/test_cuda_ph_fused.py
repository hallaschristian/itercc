import os

import numpy as np
import pytest

from itercc import RestrictedCC
from itercc.spatial_symmetry import project_integrals, project_amplitudes
from test_restricted import make_model, trial_amplitudes

pytestmark = pytest.mark.skipif(os.environ.get("ITERCC_TEST_CUDA") != "1",
                                reason="CUDA checks require ITERCC_TEST_CUDA=1")


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_ph_fusion_matches_full_residual_and_convergence(groups):
    pytest.importorskip("cupy")
    h, eri, no, constant = make_model(3, 5)
    eri *= 0.2
    labels = np.arange(8) % groups
    h, eri, _ = project_integrals(h, eri, labels, tolerance=1)
    kwargs = dict(residual_backend="cuda", orbital_irreps=labels, cuda_layout_fusion=True, cuda_ph_columns=True)
    baseline = RestrictedCC(h, eri, no, constant, **kwargs)
    candidate = RestrictedCC(h, eri, no, constant, cuda_feedback_columns=True, cuda_assembly_columns=True, cuda_ph_fusion=True, **kwargs)
    initial, _ = project_amplitudes(trial_amplitudes(RestrictedCC(h, eri, no, constant), 0.04, True), labels, tolerance=1)
    for actual, expected in zip(candidate.residuals(initial), baseline.residuals(initial)):
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
    zero_triples = (*initial[:2], np.zeros_like(initial[2]))
    for actual, expected in zip(candidate.residuals(zero_triples), baseline.residuals(zero_triples)):
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
    settings = dict(initial=initial, diis_size=3, diis_triples=True,
                    energy_tol=1e-12, residual_tol=1e-10)
    expected, actual = baseline.solve(**settings), candidate.solve(**settings)
    assert expected.converged and actual.converged
    assert abs(expected.energy - actual.energy) < 1e-12
    assert max(actual.residual_max) <= 1e-10
    # Off-sector interactions and amplitudes must retain all contributions.
    raw = make_model(3, 5)
    baseline = RestrictedCC(*raw, **kwargs)
    candidate = RestrictedCC(*raw, cuda_feedback_columns=True, cuda_assembly_columns=True, cuda_ph_fusion=True, **kwargs)
    for amps in (initial, trial_amplitudes(RestrictedCC(h, eri, no, constant), 0.04, True)):
        for actual, expected in zip(candidate.residuals(amps), baseline.residuals(amps)):
            np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
