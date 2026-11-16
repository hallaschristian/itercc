import os

import numpy as np
import pytest

from itercc import RestrictedCC
from itercc._cuda_ph import CudaParticleHole
from itercc._triples_layout import TriplesLayout
from itercc.spatial_symmetry import project_amplitudes, project_integrals
from test_restricted import make_model, trial_amplitudes

pytestmark = pytest.mark.skipif(os.environ.get("ITERCC_TEST_CUDA") != "1",
                                reason="set ITERCC_TEST_CUDA=1 to run CUDA checks")


def test_cuda_ph_columns_preserve_supported_products_and_fallback():
    cp = pytest.importorskip("cupy")
    occupied, virtual = np.array([0, 2, 6]), np.array([0, 1, 1, 6])
    layout = TriplesLayout(len(occupied), len(virtual))
    ph = CudaParticleHole(occupied, virtual, cp)
    rng = np.random.default_rng(193)
    rows = (occupied[:, None] ^ virtual).ravel()
    columns = (virtual[:, None] ^ virtual).ravel()
    matrix = rng.normal(size=(len(rows), len(rows)))
    matrix[rows[:, None] != rows] = 0
    product, chosen = ph.product(cp.asarray(matrix))
    assert chosen
    for shift in np.unique(ph.shifts):
        rhs = rng.normal(size=(len(rows), len(columns)))
        rhs[rows[:, None] != (columns ^ shift)] = 0
        np.testing.assert_allclose(product(cp.asarray(rhs), shift).get(), matrix @ rhs,
                                   rtol=0, atol=2e-14)
    matrix[0, 1] = 1e-18
    device = cp.asarray(matrix)
    product, chosen = ph.product(device)
    assert not chosen
    rhs = cp.asarray(rng.normal(size=(len(rows), len(columns))))
    np.testing.assert_array_equal(product(rhs, 0).get(), (device @ rhs).get())
    plane = cp.asarray(rng.normal(size=(layout.no, layout.no) + (layout.nv,) * 3))
    for spin in (False, True):
        from_plane, chosen = ph.product(device, layout=layout, spin=spin)
        assert not chosen
        expected = layout.spin_sum(plane[:, 1]) if spin else plane[1]
        np.testing.assert_array_equal(from_plane(plane, 0, 1).get(),
                                      (device @ expected.reshape(rhs.shape)).get())


@pytest.mark.parametrize("leakage", [False, True])
def test_cuda_ph_columns_match_full_residual_with_and_without_leakage(leakage, monkeypatch):
    pytest.importorskip("cupy")
    no, nv = 3, 4
    model = make_model(no, nv)
    labels = np.array([0, 1, 2, 0, 1, 2, 3])
    h, eri, _ = project_integrals(*model[:2], labels, tolerance=1)
    args = h, eri, *model[2:]
    options = dict(orbital_irreps=labels, residual_backend="cuda",
                   cuda_pair_fusion=True, cuda_layout_fusion=True)
    baseline = RestrictedCC(*args, **options)
    compressed = RestrictedCC(*args, cuda_ph_columns=True, **options)
    amplitudes = trial_amplitudes(RestrictedCC(*args), 0.07, True)
    amplitudes, _ = project_amplitudes(amplitudes, labels, tolerance=1)
    if leakage:
        # A forbidden virtual label, with the repeated occupied symmetry intact.
        amplitudes[2][0, 0, 0, 0] = 1e-18

    used = []
    original = CudaParticleHole.product

    def track(self, matrix, **options):
        product, chosen = original(self, matrix, **options)
        used.append(chosen)
        return product, chosen

    monkeypatch.setattr(CudaParticleHole, "product", track)
    for expected, actual in zip(baseline.residuals(amplitudes),
                                compressed.residuals(amplitudes)):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)
    assert used == ([] if leakage else [True, True])