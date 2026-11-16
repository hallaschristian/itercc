"""Davidson restarts reuse Ritz images; roots still match the dense projected Jacobian."""
import numpy as np

from itercc import RestrictedCC
from itercc.eom import EOMSinglet
from test_restricted import make_model


def test_restarted_davidson_reuses_images_and_matches_dense_spectrum():
    solver = RestrictedCC(*make_model(2, 3))
    ground = solver.solve(residual_tol=1e-11, energy_tol=1e-12)
    assert ground.converged
    eom = EOMSinglet(solver, ground.amplitudes)
    space = eom.space
    basis = np.linalg.qr(np.stack([space.project(np.eye(space.size)[i])
                                   for i in range(space.size)], axis=1))[0]
    rank = np.linalg.matrix_rank(basis, tol=1e-10)
    basis = np.linalg.svd(basis)[0][:, :rank] if rank < basis.shape[1] else basis
    dense = basis.T @ np.stack([eom.sigma(column) for column in basis.T], axis=1)
    exact = np.sort(np.linalg.eigvals(dense).real)
    exact = exact[exact > 1e-7][:2]
    calls = []
    sigma = eom.sigma

    def counted(vector):
        calls.append(1)
        return sigma(vector)

    eom.sigma = counted
    before = eom.matvec_count
    # A tiny subspace forces several restarts. Its iteration count is chaotic:
    # ground amplitudes perturbed at 1e-15 needed 160 to 288 iterations.
    result = eom.solve(nroots=2, max_space=6, residual_tol=1e-9, max_iter=400)
    assert result.converged
    np.testing.assert_allclose(result.excitation_energies, exact, atol=1e-9, rtol=0)
    history = result.iterations
    restarts = sum(later["subspace"] < earlier["subspace"]
                   for earlier, later in zip(history, history[1:]))
    assert restarts >= 1
    # Fresh certification adds one full action per root after the last iteration.
    assert result.matvecs - before == len(calls) + 2
