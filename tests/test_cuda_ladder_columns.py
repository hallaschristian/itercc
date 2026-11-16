import os

import numpy as np
import pytest

from itercc import RestrictedCC
from itercc._cuda_ladder_columns import CudaLadderColumns
from itercc._cuda_support import CudaTriplesSupport
from itercc.restricted import _pair_ladder
from itercc.spatial_symmetry import project_amplitudes, project_integrals
from test_restricted import make_model, trial_amplitudes

pytestmark = pytest.mark.skipif(os.environ.get("ITERCC_TEST_CUDA") != "1",
                                reason="set ITERCC_TEST_CUDA=1 to run CUDA checks")


@pytest.mark.parametrize("matrix_leak", [False, True])
def test_compact_virtual_ladder_preserves_supported_values(matrix_leak, monkeypatch):
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(81)
    labels = np.array([0, 0, 1, 1, 2, 3])
    occupied = np.array([0, 2, 1, 3, 1])
    nv = len(labels)
    pairs = labels[:, None] ^ labels
    w = rng.normal(scale=0.1, size=(nv,) * 4)
    w += w.transpose(1, 0, 3, 2).copy()
    w[pairs[:, :, None, None] != pairs] = 0
    if matrix_leak:
        w[0, 0, 0, 2] = w[0, 0, 2, 0] = 0.01
    source = rng.normal(size=(len(occupied), nv, nv, nv))
    source[occupied[:, None, None, None] != (pairs[:, :, None] ^ labels)] = 0
    source = cp.asarray(source)
    expected = cp.asarray(rng.normal(scale=0.1, size=source.shape))
    actual = expected.copy()
    options = dict(xp=cp, virtual_irreps=labels, fuse_cuda=True)
    calls = []
    original = CudaLadderColumns.add

    def count(self, source, target, labels):
        calls.append(len(source))
        return original(self, source, target, labels)

    monkeypatch.setattr(CudaLadderColumns, "add", count)
    _pair_ladder(expected, source, w, 2, **options)
    _pair_ladder(actual, source, w, 2, supported_occupied_irreps=occupied, **options)
    np.testing.assert_allclose(actual.get(), expected.get(), atol=1e-14, rtol=0)
    assert calls == ([] if matrix_leak else [2, 2, 1])


@pytest.mark.parametrize("t3_leak", [False, True])
def test_virtual_and_ph_columns_share_one_exact_support_check(t3_leak, monkeypatch):
    pytest.importorskip("cupy")
    model = make_model(3, 4)
    labels = np.array([0, 1, 2, 0, 1, 2, 3])
    h, eri, _ = project_integrals(*model[:2], labels, tolerance=1)
    args = h, eri, *model[2:]
    options = dict(orbital_irreps=labels, residual_backend="cuda", ladder_batch=3,
                   cuda_pair_fusion=True, cuda_layout_fusion=True, cuda_ph_columns=True)
    row_only = RestrictedCC(*args, **options)
    compact = RestrictedCC(*args, cuda_ladder_columns=True, **options)
    amplitudes = trial_amplitudes(RestrictedCC(*args), 0.07, True)
    amplitudes, _ = project_amplitudes(amplitudes, labels, tolerance=1)
    if t3_leak:
        amplitudes[2][0, 0, 0, 0] = 1e-18
    expected = row_only.residuals(amplitudes)

    support_calls = []
    batch_calls = []
    orig_check, orig_add = CudaTriplesSupport.supports, CudaLadderColumns.add

    def check(self, t3):
        result = orig_check(self, t3)
        support_calls.append(result)
        return result

    def add(self, source, target, labels):
        batch_calls.append(len(source))
        return orig_add(self, source, target, labels)

    monkeypatch.setattr(CudaTriplesSupport, "supports", check)
    monkeypatch.setattr(CudaLadderColumns, "add", add)
    for left, right in zip(compact.residuals(amplitudes), expected):
        np.testing.assert_allclose(left, right, atol=2e-14, rtol=0)
    assert support_calls == [not t3_leak]
    assert batch_calls == ([] if t3_leak else [3, 3, 1])