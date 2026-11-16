import os

import numpy as np
import pytest

from itercc import RestrictedCC
from itercc.restricted import _pair_ladder
from itercc.spatial_symmetry import project_amplitudes, project_integrals
from test_restricted import make_model, trial_amplitudes

pytestmark = pytest.mark.skipif(os.environ.get("ITERCC_TEST_CUDA") != "1",
                                reason="set ITERCC_TEST_CUDA=1 to run CUDA checks")


@pytest.mark.parametrize("nv,grouped,leak", [(1, False, False), (6, False, False),
                                             (6, True, False), (6, True, True)])
def test_cuda_pair_fusion_agrees_with_unfused_ladder(nv, grouped, leak):
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(219)
    labels = np.arange(nv) % 3 if grouped else None
    w = rng.normal(scale=0.1, size=(nv,) * 4)
    w += w.transpose(1, 0, 3, 2).copy()
    if grouped:
        pairs = labels[:, None] ^ labels
        w[pairs[:, :, None, None] != pairs] = 0
    if leak:
        w[0, 0, 0, 1] = w[0, 0, 1, 0] = 0.01
    source = cp.asarray(rng.normal(size=(5, nv, nv, nv)))
    start = cp.asarray(rng.normal(size=source.shape))
    expected, actual = start.copy(), start.copy()
    _pair_ladder(expected, source, w, 2, xp=cp, virtual_irreps=labels)
    _pair_ladder(actual, source, w, 2, xp=cp, virtual_irreps=labels, fuse_cuda=True)
    np.testing.assert_allclose(actual.get(), expected.get(), rtol=0, atol=1e-14)


@pytest.mark.parametrize("grouped", [False, True])
def test_cuda_fused_pair_ladder_matches_full_cpu_residual(grouped):
    pytest.importorskip("cupy")
    model = make_model(3, 4)
    if grouped:
        labels = np.array([0, 1, 2, 0, 1, 2, 3])
        h, eri, _ = project_integrals(model[0], model[1], labels, tolerance=1)
        model = h, eri, *model[2:]
    else:
        labels = None
    cpu = RestrictedCC(*model, ladder_batch=3)
    gpu = RestrictedCC(*model, ladder_batch=3, residual_backend="cuda",
                       orbital_irreps=labels, cuda_pair_fusion=True)
    amplitudes = trial_amplitudes(cpu, 0.07, True)
    if grouped:
        amplitudes, _ = project_amplitudes(amplitudes, labels, tolerance=1)
    for actual, expected in zip(gpu.residuals(amplitudes), cpu.residuals(amplitudes)):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)