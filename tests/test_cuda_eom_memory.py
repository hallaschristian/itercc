"""Check the fused CUDA gap construction and compact packed indices."""
import os

import numpy as np
import pytest

from itercc import RestrictedCC
from itercc.eom import SingletSpace
from test_restricted import make_model


def test_cuda_triples_diagonal_matches_cpu():
    if os.environ.get("ITERCC_TEST_CUDA") != "1":
        pytest.skip("CUDA check")
    import cupy as cp

    solver = RestrictedCC(*make_model(3, 4))
    labels = np.array([0, 2, 3, 0, 2, 2, 3])
    for irrep in (0, 2, 3):
        cpu = SingletSpace(solver, labels, irrep)
        gpu = SingletSpace(solver, labels, irrep, xp=cp)
        assert gpu.indices[2].dtype == cp.int32
        np.testing.assert_array_equal(gpu.indices[2].get(), cpu.indices[2])
        np.testing.assert_allclose(gpu.orbital_diagonal(solver.fock).get(),
                                   cpu.orbital_diagonal(solver.fock), rtol=0, atol=1e-14)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_zero_dipole_skips_unused_excitation_sectors(backend):
    if backend == "cuda" and os.environ.get("ITERCC_TEST_CUDA") != "1":
        pytest.skip("CUDA check")
    from itercc import DipoleResponse

    gap = 1.5
    solver = RestrictedCC(np.diag([-1., .5]), np.zeros((2,) * 4), 1)
    amplitudes = (np.zeros((1, 1)), np.zeros((1, 1, 1, 1)))
    dipoles = np.zeros((3, 2, 2))
    dipoles[0, 0, 1] = dipoles[0, 1, 0] = 1.
    response = DipoleResponse(solver, amplitudes, dipoles, orbital_irreps=[0, 2],
                              dipole_irreps=(2, 3, 0), backend=backend, method="ccsd")
    assert set(response.spaces) == {0, 2}
    assert response.rhs[1] is response.rhs[2] is None
    assert response.diagonal[0] is response.engine.diagonal
    actual = response.polarizability(np.inf, tol=1e-11).tensor_au
    np.testing.assert_allclose(actual, np.diag([4/gap, 0., 0.]), atol=1e-12, rtol=0)
