import os

import numpy as np
import pytest

from itercc import RestrictedCC
from itercc._cuda_occupied import CudaOccupiedLadder
from itercc.spatial_symmetry import project_amplitudes, project_integrals
from test_restricted import make_model, trial_amplitudes

pytestmark = pytest.mark.skipif(os.environ.get("ITERCC_TEST_CUDA") != "1",
                                reason="set ITERCC_TEST_CUDA=1 to run CUDA checks")


def test_compact_occupied_product_matches_dense_for_every_line():
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(207)
    occupied = np.array([0, 1, 2, 0, 3])
    column_labels = np.array([0, 3, 2, 1, 1, 2, 0, 0, 4])
    no, width = len(occupied), len(column_labels)
    row_labels = (occupied[:, None] ^ occupied).ravel()
    matrix = rng.normal(size=(no**2,) * 2)
    matrix[row_labels[:, None] != row_labels] = 0
    plan = CudaOccupiedLadder(occupied, column_labels, cp)
    apply = plan.product(cp.asarray(matrix))
    assert apply is not None
    for k, shift in enumerate(occupied):
        rhs = rng.normal(size=(no**2, width))
        rhs[(row_labels[:, None] ^ shift) != column_labels] = 0
        dense = (matrix @ rhs).reshape(no, no, width)
        add = apply(cp.asarray(rhs), k)
        for j in range(no):
            initial = rng.normal(size=(width, no))
            actual = cp.asarray(initial).T
            add(actual, j)
            np.testing.assert_allclose(actual.get(), initial.T + dense[:, j], atol=2e-14, rtol=0)
    matrix[0, 1] = 1e-18
    assert plan.product(cp.asarray(matrix)) is None


@pytest.mark.parametrize("leakage", [False, True])
@pytest.mark.parametrize("fold_fusion", [False, True])
def test_occupied_compression_preserves_full_cuda_residual(leakage, fold_fusion, monkeypatch):
    pytest.importorskip("cupy")
    model = make_model(3, 4)
    labels = np.array([0, 1, 2, 0, 1, 2, 3])
    h, eri, _ = project_integrals(*model[:2], labels, tolerance=1)
    args = h, eri, *model[2:]
    options = dict(orbital_irreps=labels, residual_backend="cuda", ladder_batch=3,
                   cuda_pair_fusion=True, cuda_layout_fusion=fold_fusion,
                   cuda_ph_columns=True, cuda_ladder_columns=True)
    baseline = RestrictedCC(*args, **options)
    compact = RestrictedCC(*args, cuda_occupied_columns=True, **options)
    amps = trial_amplitudes(RestrictedCC(*args), 0.07, True)
    amps, _ = project_amplitudes(amps, labels, tolerance=1)
    if leakage:
        amps[2][0, 0, 0, 0] = 1e-18
    calls = []
    original = CudaOccupiedLadder.product

    def record(self, matrix):
        apply = original(self, matrix)
        calls.append(apply is not None)
        return apply

    monkeypatch.setattr(CudaOccupiedLadder, "product", record)
    for wanted, actual in zip(baseline.residuals(amps), compact.residuals(amps)):
        np.testing.assert_allclose(actual, wanted, atol=2e-14, rtol=0)
    assert calls == ([] if leakage else [True])