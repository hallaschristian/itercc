import os

import numpy as np
import pytest

from itercc import RestrictedCC
from itercc._cuda_dress import CudaT1Dressing
from itercc._cuda_jacobi import CudaJacobi
from itercc.solver import _dress_axis
from itercc.spatial_symmetry import project_amplitudes, project_integrals
from test_restricted import make_model, trial_amplitudes

pytestmark = pytest.mark.skipif(os.environ.get("ITERCC_TEST_CUDA") != "1",
                                reason="set ITERCC_TEST_CUDA=1 to run CUDA checks")


def test_sparse_t1_dressing_and_leakage_fallback():
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(33)
    occupied, virtual = np.array([0, 1, 1]), np.array([1, 2, 0, 2, 1])
    no = len(occupied)
    source = rng.normal(scale=0.1, size=(no + len(virtual),) * 4)
    t1 = rng.normal(scale=0.1, size=(no, len(virtual)))
    t1[occupied[:, None] != virtual] = 0
    original = cp.asarray(source)
    expected, actual = original.copy(), original.copy()
    rotation = cp.asarray(t1)
    o, v = slice(no), slice(no, None)
    for axis in (0, 1):
        _dress_axis(expected, axis, v, o, -rotation.T, cp)
    for axis in (2, 3):
        _dress_axis(expected, axis, o, v, rotation, cp)
    dressing = CudaT1Dressing(occupied, virtual, cp)
    assert dressing.apply(actual, rotation)
    np.testing.assert_allclose(actual.get(), expected.get(), atol=2e-16, rtol=0)
    rotation[0, 0] = 1e-18
    actual = original.copy()
    assert not dressing.apply(actual, rotation)
    np.testing.assert_array_equal(actual.get(), source)


def test_sparse_dressing_preserves_full_residual():
    pytest.importorskip("cupy")
    labels = np.array([0, 1, 2, 0, 1, 2, 3])
    model = make_model(3, 4)
    h, eri, _ = project_integrals(*model[:2], labels, tolerance=1)
    args = h, eri, *model[2:]
    kw = dict(orbital_irreps=labels, residual_backend="cuda", cuda_pair_fusion=True,
              cuda_layout_fusion=True, cuda_ph_columns=True, cuda_ladder_columns=True)
    baseline, candidate = RestrictedCC(*args, **kw), RestrictedCC(*args, cuda_t1_dressing=True, **kw)
    amps = trial_amplitudes(RestrictedCC(*args), 0.07, True)
    amps, _ = project_amplitudes(amps, labels, tolerance=1)
    for left, right in zip(baseline.residuals(amps), candidate.residuals(amps)):
        np.testing.assert_allclose(left, right, atol=2e-14, rtol=0)


def test_device_jacobi_reports_raw_maxima_and_skips_unneeded_transfer():
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(34)
    arrays = [rng.normal(size=shape) for shape in ((2, 3), (2, 2, 3, 3), (4, 3, 3, 3))]
    d1 = -1 - rng.random(arrays[0].shape)
    d2 = -2 - rng.random(arrays[1].shape)
    d3o = -4 - rng.random(4)
    d3v = 1 + rng.random((3, 3, 3))
    stepper = CudaJacobi(cp, d1, d2, d3o, d3v, 0.2)
    maxima, steps = stepper.finish(tuple(map(cp.asarray, arrays)), False, 1e-12, True)
    np.testing.assert_array_equal(maxima, [np.max(np.abs(a)) for a in arrays])
    expected = [0.8 * a / d for a, d in zip(arrays, (d1, d2, d3o[:, None, None, None] - d3v))]
    for actual, wanted in zip(steps, expected):
        np.testing.assert_allclose(actual, wanted, rtol=2e-16, atol=1e-16)
    for ready, allowed in ((True, True), (False, False)):
        observed, skipped = stepper.finish(tuple(map(cp.asarray, arrays)), ready, 10, allowed)
        assert skipped is None and maxima == observed
    arrays[2][0, 0, 0, 0] = np.nan
    observed, skipped = stepper.finish(tuple(map(cp.asarray, arrays)), False, 1e-12, True)
    assert np.isinf(observed[2]) and skipped is None


@pytest.mark.parametrize("no,nv", [(1, 2), (3, 3)])
def test_device_jacobi_preserves_convergence_and_checkpoint_callback(no, nv):
    pytest.importorskip("cupy")
    model = make_model(no, nv)
    baseline = RestrictedCC(*model, residual_backend="cuda")
    candidate = RestrictedCC(*model, residual_backend="cuda", cuda_jacobi=True)
    settings = dict(energy_tol=1e-12, residual_tol=1e-10, diis_size=3, diis_triples=True)
    expected = baseline.solve(**settings)
    snapshots = []
    actual = candidate.solve(**settings, checkpoint_callback=lambda i, info, _: snapshots.append((i, info)))
    assert expected.converged and actual.converged
    assert actual.energy == pytest.approx(expected.energy, abs=1e-12, rel=0)
    assert max(actual.residual_max) <= 1e-10
    assert len(snapshots) == len(actual.iterations)
    assert snapshots[-1][1] == actual.iterations[-1]