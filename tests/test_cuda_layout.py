import os

import numpy as np
import pytest

from itercc import RestrictedCC
from itercc._cuda_layout import CudaTriplesLayout
from itercc._triples_layout import TriplesLayout
from test_restricted import make_model, trial_amplitudes

pytestmark = pytest.mark.skipif(os.environ.get("ITERCC_TEST_CUDA") != "1",
                                reason="set ITERCC_TEST_CUDA=1 to run CUDA checks")


def test_crossed_diagrams_use_one_exact_cuda_update():
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(543)
    seed, tile = rng.normal(size=(2, 3, 4, 4, 4))
    seed = seed.swapaxes(1, 3)
    expected = seed.copy()
    TriplesLayout.subtract_crossed(expected, tile.copy())
    actual = cp.asarray(seed)
    CudaTriplesLayout.subtract_crossed_fused(actual, cp.asarray(tile))
    np.testing.assert_array_equal(actual.get(), expected)


@pytest.mark.parametrize("no,nv", [(1, 1), (2, 3), (4, 5)])
def test_cuda_layout_permutations_match_reference(no, nv):
    cp = pytest.importorskip("cupy")
    baseline, fused = TriplesLayout(no, nv), CudaTriplesLayout(no, nv, cp)
    rng = np.random.default_rng(402)
    t3 = rng.normal(size=baseline.shape)
    cuda_t3 = cp.asarray(t3)
    expected, actual = t3.copy(), cuda_t3.copy()
    for k in range(no):
        for axis in (0, 2):
            np.testing.assert_array_equal(fused.plane(cuda_t3, k, axis=axis).get(),
                                          baseline.plane(t3, k, axis=axis))
        np.testing.assert_array_equal(fused.spin_plane(cuda_t3, k).get(),
                                      baseline.spin_plane(t3, k))
        for j in range(no):
            seed = rng.normal(size=(no, nv, nv, nv))
            baseline.add_seed(expected, seed, j, k)
            fused.add_seed(actual, cp.asarray(seed), j, k)
    np.testing.assert_array_equal(actual.get(), expected)
    baseline.purify(expected)
    fused.purify(actual, xp=cp)
    np.testing.assert_allclose(actual.get(), expected, atol=8e-15, rtol=0)


def test_cuda_layout_fusion_preserves_full_residual_and_iterative_energy():
    pytest.importorskip("cupy")
    model = make_model(3, 3)
    cpu = RestrictedCC(*model, ladder_batch=3)
    cuda = RestrictedCC(*model, ladder_batch=3, residual_backend="cuda",
                        cuda_pair_fusion=True, cuda_layout_fusion=True, cuda_crossed_fusion=True)
    amplitudes = trial_amplitudes(cpu, 0.07, True)
    for actual, expected in zip(cuda.residuals(amplitudes), cpu.residuals(amplitudes)):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)
    criteria = dict(energy_tol=1e-12, residual_tol=1e-10)
    expected, actual = cpu.solve(**criteria), cuda.solve(**criteria)
    assert expected.converged and actual.converged
    assert actual.energy == pytest.approx(expected.energy, abs=1e-12, rel=0)
    assert max(actual.residual_max) <= 1e-10
