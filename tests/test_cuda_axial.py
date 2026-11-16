import os

import numpy as np
import pytest

from itercc._cuda_axial import CudaAxialLadder, real_pair_transform
from itercc.restricted import _pair_ladder

M = np.array([0, 1, -1, 2, -2, 0])
IRREPS = np.array([0, 2, 3, 0, 1, 0])
PARTNERS = np.array([0, 2, 1, 4, 3, 5])


@pytest.mark.parametrize("parity", [1, -1])
def test_real_axial_pair_transform_is_orthogonal(parity):
    table = real_pair_transform(M, IRREPS, PARTNERS, parity)
    index, coefficient = table.forward
    matrix = np.zeros((len(index), len(index)))
    for column, (rows, values) in enumerate(zip(index, coefficient)):
        np.add.at(matrix[:, column], rows, values)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(len(index)), atol=2e-15, rtol=0)


def invariant_interaction(rng):
    nv = len(M)
    raw = rng.normal(scale=0.1, size=(nv,) * 4)
    raw += raw.transpose(1, 0, 3, 2).copy()
    w = np.zeros_like(raw)
    for theta in np.arange(17) * (2 * np.pi / 17):
        u = np.eye(nv)
        for i in np.flatnonzero(M > 0):
            j = PARTNERS[i]
            co, si = np.cos(M[i] * theta), np.sin(M[i] * theta)
            u[np.ix_([i, j], [i, j])] = [[co, -si], [si, co]]
        w += np.einsum("pa,qb,rd,se,abde->pqrs", u, u, u, u, raw, optimize=True) / 17
    pair_labels = IRREPS[:, None] ^ IRREPS
    w[pair_labels[:, :, None, None] != pair_labels] = 0.0
    return w


def test_cuda_axial_ladder_and_fallback():
    if os.environ.get("ITERCC_TEST_CUDA") != "1":
        pytest.skip("set ITERCC_TEST_CUDA=1 to run CUDA checks")
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(31)
    nv = len(M)
    w = invariant_interaction(rng)
    pair_labels = IRREPS[:, None] ^ IRREPS
    occupied = np.array([0, 2, 3, 1, 0])
    source = rng.normal(scale=0.03, size=(len(occupied), nv, nv, nv))
    source[occupied[:, None, None, None] != (pair_labels[:, :, None] ^ IRREPS)] = 0
    source, w = cp.asarray(source), cp.asarray(w)
    actual, expected = cp.zeros_like(source), cp.zeros_like(source)
    axial = CudaAxialLadder(M, IRREPS, PARTNERS, cp)
    _pair_ladder(expected, source, w, 2, xp=cp, virtual_irreps=IRREPS, fuse_cuda=True,
                 supported_occupied_irreps=occupied)
    assert axial.apply(actual, source, w, occupied, 2, tolerance=1e-13)
    np.testing.assert_allclose(actual.get(), expected.get(), rtol=0, atol=5e-16)
    w[0, 0, 0, 0] += 0.2
    w[0, 0, 3, 3] += 0.3
    assert not axial.apply(actual, source, w, occupied, 2, tolerance=1e-13)
    np.testing.assert_allclose(actual.get(), expected.get(), rtol=0, atol=5e-16)

def test_optional_axial_path_matches_full_residual(monkeypatch):
    if os.environ.get("ITERCC_TEST_CUDA") != "1":
        pytest.skip("set ITERCC_TEST_CUDA=1 to run CUDA checks")
    pytest.importorskip("cupy")
    from itercc import RestrictedCC
    from itercc.spatial_symmetry import project_amplitudes
    from test_restricted import trial_amplitudes

    no, nv = 3, len(M)
    h = np.diag(np.r_[[-2.0, -1.0, -1.0], [1.0, 1.2, 1.2, 1.5, 1.5, 2.0]])
    eri = np.zeros((no + nv,) * 4)
    eri[no:, no:, no:, no:] = invariant_interaction(np.random.default_rng(317)).transpose(0, 2, 1, 3)
    labels = np.r_[[0, 2, 3], IRREPS]
    kw = dict(residual_backend="cuda", ladder_batch=2, orbital_irreps=labels,
              cuda_pair_fusion=True, cuda_layout_fusion=True,
              cuda_ph_columns=True, cuda_ladder_columns=True)
    baseline = RestrictedCC(h, eri, no, **kw)
    axial = RestrictedCC(h, eri, no, cuda_axial=(M, PARTNERS), axial_projection_tolerance=1e-13, **kw)
    amps, _ = project_amplitudes(trial_amplitudes(RestrictedCC(h, eri, no), 0.07, True), labels, tolerance=1)
    amps[0].fill(0)
    amps[1].fill(0)
    used = []
    original = CudaAxialLadder.apply

    def track(self, *args, **options):
        valid = original(self, *args, **options)
        used.append(valid)
        return valid

    monkeypatch.setattr(CudaAxialLadder, "apply", track)
    for expected, actual in zip(baseline.residuals(amps), axial.residuals(amps)):
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
    assert used == [True]
