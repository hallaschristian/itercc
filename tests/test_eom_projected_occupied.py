"""Check projected occupied contractions without an iterative CC solve."""
import os

import numpy as np
import pytest


@pytest.mark.parametrize("irrep", [0, 2, 3])
def test_projected_occupied_keeps_ground_leakage_and_matches_dense(irrep):
    if os.environ.get("ITERCC_TEST_CUDA") != "1":
        pytest.skip("CUDA check")
    import cupy as cp
    from itercc._cuda_eom_occupied import CudaProjectedOccupied

    no, nv = 3, 5
    occ = np.array([0, 2, 3])
    vir = np.array([0, 0, 2, 2, 3])
    rng = np.random.default_rng(731)
    output_rows = (occ[:, None] ^ occ).ravel()
    columns = (vir[:, None, None] ^ vir[None, :, None] ^ vir).ravel()
    matrix, derivative = rng.normal(size=(2, no**2, no**2))
    ground, direction = rng.normal(size=(2, no**2, nv**3))
    owner = CudaProjectedOccupied(occ, vir, irrep, cp)
    action = owner.prepare(cp.asarray(matrix), cp.asarray(derivative))
    action.chunk = 7
    for k in range(no):
        mask = (output_rows[:, None] ^ columns ^ occ[k]) == irrep
        tangent = np.where(mask, direction, 0.)
        expected = np.where(mask, matrix @ tangent + derivative @ ground, 0.)
        occupied = action.plane(cp.asarray(ground), cp.asarray(tangent), k)
        result = cp.zeros((no, no, nv**3))
        for j in range(no):
            line = cp.zeros((no, nv, nv, nv))
            occupied(line, j)
            result[:, j] = line.reshape(no, nv**3)
        np.testing.assert_allclose(result.get().reshape(expected.shape), expected, atol=2e-14, rtol=0)
