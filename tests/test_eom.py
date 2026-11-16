import os
import numpy as np
import pytest

from itercc import RestrictedCC
from itercc.eom import EOMSinglet
from test_restricted import make_model, trial_amplitudes
from test_restricted import spatial_to_spin
from test_tensor_oracle import determinant_problem, excitation, apply_cluster_exponential
from itercc import Hamiltonian
from itertools import combinations


@pytest.mark.parametrize("no,nv", [(1, 2), (2, 2), (3, 2)])
def test_analytic_jacobian_action_matches_residual_derivative(no, nv):
    solver = RestrictedCC(*make_model(no, nv))
    amplitudes = trial_amplitudes(solver, .035, True)
    eom = EOMSinglet(solver, amplitudes, ground_residual_tol=1)
    rng = np.random.default_rng(713)
    vector = eom.space.project(rng.normal(size=eom.space.size))
    vector /= np.linalg.norm(vector)
    direction = eom.space.unpack(vector)
    step = 1e-5
    plus = solver.residuals(tuple(t + step * r for t, r in zip(amplitudes, direction)))
    minus = solver.residuals(tuple(t - step * r for t, r in zip(amplitudes, direction)))
    expected = eom.space.pack(tuple((p - m) / (2 * step) for p, m in zip(plus, minus)))
    np.testing.assert_allclose(eom.sigma(vector), expected, atol=2e-10, rtol=0)
    # The exact derivative is linear in the direction, including T3 feedback.
    other = eom.space.project(rng.normal(size=eom.space.size))
    np.testing.assert_allclose(eom.sigma(vector + other), eom.sigma(vector) + eom.sigma(other),
                               atol=2e-13, rtol=0)


def test_cuda_analytic_jacobian_matches_cpu():
    if os.environ.get("ITERCC_TEST_CUDA") != "1":
        pytest.skip("CUDA check")
    cp = pytest.importorskip("cupy")
    solver = RestrictedCC(*make_model(3, 2))
    amplitudes = trial_amplitudes(solver, .035, True)
    cpu = EOMSinglet(solver, amplitudes, ground_residual_tol=1)
    gpu = EOMSinglet(solver, amplitudes, ground_residual_tol=1, backend="cuda")
    vector = cpu.space.project(np.random.default_rng(32).normal(size=cpu.space.size))
    np.testing.assert_allclose(gpu.sigma(cp.asarray(vector)).get(), cpu.sigma(vector),
                               atol=3e-13, rtol=0)


def physical_basis(space):
    projector = np.column_stack([space.project(v) for v in np.eye(space.size)])
    u, s, _ = np.linalg.svd(projector)
    return u[:, s > 1e-10]


def test_full_ccsdt_excited_roots_match_independent_determinant_commutator():
    h, eri, no, constant = make_model(2, 2)
    model = h, .1 * eri, no, constant
    solver = RestrictedCC(*model)
    ground = solver.solve(energy_tol=1e-13, residual_tol=1e-12, diis_triples=True)
    assert ground.converged
    eom = EOMSinglet(solver, ground.amplitudes)
    basis = physical_basis(eom.space)
    spin_model = Hamiltonian.restricted(*model)
    h, masks = determinant_problem(spin_model, 2, 2)
    lookup = {mask: i for i, mask in enumerate(masks)}
    no, nso = spin_model.nocc, len(spin_model.h)
    reference = np.eye(len(masks))[:, lookup[(1 << no) - 1]]

    def operator(spatial):
        amplitudes = spatial_to_spin(solver, spatial)
        result = np.zeros_like(h)
        for rank in (1, 2, 3):
            for holes in combinations(range(no), rank):
                for particles in combinations(range(no, nso), rank):
                    value = amplitudes[rank - 1][tuple(p - no for p in particles) + holes]
                    if value == 0:
                        continue
                    for column, mask in enumerate(masks):
                        moved = excitation(mask, holes, particles)
                        if moved is not None and moved[0] in lookup:
                            result[lookup[moved[0]], column] += moved[1] * value
        return result

    t = operator(ground.amplitudes)
    exp_t = apply_cluster_exponential(t, np.eye(len(h)), no)
    hbar = apply_cluster_exponential(-t, h @ exp_t, no)
    operators = [operator(eom.space.unpack(v)) for v in basis.T]
    columns = np.column_stack([r @ reference for r in operators])
    commutators = np.column_stack([(hbar @ r - r @ hbar) @ reference for r in operators])
    independent = np.linalg.lstsq(columns, commutators, rcond=1e-12)[0]
    tensor = np.column_stack([basis.T @ eom.sigma(v) for v in basis.T])
    np.testing.assert_allclose(tensor, independent, atol=2e-13, rtol=0)
    expected = np.sort(np.linalg.eigvals(independent).real)[:2]
    result = eom.solve(nroots=2, max_space=14, residual_tol=1e-10)
    assert result.converged
    np.testing.assert_allclose(result.excitation_energies, expected, atol=2e-10, rtol=0)
    assert max(result.residual_norms) < 1e-10


def test_ccsd_singlet_excitation_energies_match_pyscf():
    pytest.importorskip("pyscf")
    from pyscf import gto, scf, cc, lib
    from itercc.pyscf_interface import from_rhf
    lib.num_threads(1)
    mf = scf.RHF(gto.M(atom="Be 0 0 0", basis="6-31g", verbose=0))
    mf.conv_tol = 1e-13
    mf.kernel()
    reference = cc.CCSD(mf)
    reference.conv_tol, reference.conv_tol_normt = 1e-13, 1e-11
    reference.kernel()
    assert reference.converged
    problem = from_rhf(mf)
    solver = RestrictedCC(problem.h, problem.eri, problem.nocc, problem.constant)
    amplitudes = (reference.t1, reference.t2, np.zeros(solver.shapes[2]))
    eom = EOMSinglet(solver, amplitudes, method="ccsd")
    actual = eom.solve(nroots=2, max_space=24, residual_tol=1e-9)
    comparator = reference.EOMEESinglet()
    comparator.conv_tol = 1e-10
    expected, _ = comparator.kernel(nroots=2)
    assert actual.converged and all(comparator.converged)
    np.testing.assert_allclose(actual.excitation_energies, expected, atol=2e-9, rtol=0)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_nontotally_symmetric_response_is_preserved(backend):
    if backend == "cuda" and os.environ.get("ITERCC_TEST_CUDA") != "1":
        pytest.skip("CUDA check")
    from itercc.spatial_symmetry import project_integrals
    h, eri, no, constant = make_model(2, 2)
    labels = np.array([0, 1, 0, 1])
    h, eri, _ = project_integrals(h, eri * .1, labels, tolerance=1)
    solver = RestrictedCC(h, eri, no, constant)
    ground = solver.solve(energy_tol=1e-13, residual_tol=1e-12, diis_triples=True)
    assert ground.converged
    eom = EOMSinglet(solver, ground.amplitudes, orbital_irreps=labels, irrep=1, backend=backend)
    xp = eom.xp
    vector = eom.space.project(xp.asarray(np.random.default_rng(139).normal(size=eom.space.size)))
    direction = tuple(np.asarray(v) if backend == "cpu" else v.get() for v in eom.space.unpack(vector))
    step = 1e-5
    plus = solver.residuals(tuple(t + step * r for t, r in zip(ground.amplitudes, direction)))
    minus = solver.residuals(tuple(t - step * r for t, r in zip(ground.amplitudes, direction)))
    expected = eom.space.pack(tuple(xp.asarray((p - m) / (2 * step)) for p, m in zip(plus, minus)))
    np.testing.assert_allclose(np.asarray(eom.sigma(vector)) if backend == "cpu" else eom.sigma(vector).get(),
                               np.asarray(expected) if backend == "cpu" else expected.get(),
                               atol=3e-10, rtol=0)
    assert np.linalg.norm(direction[0]) > 0


def test_memory_budget_prevents_large_subspace_allocation():
    solver = RestrictedCC(*make_model(1, 2))
    ground = solver.solve(energy_tol=1e-13, residual_tol=1e-11)
    eom = EOMSinglet(solver, ground.amplitudes)
    with pytest.raises(MemoryError):
        eom.solve(1, memory_budget_gib=1e-12)


def test_two_electron_full_ccsdt_singlets_match_exact_diagonalization():
    from pyscf import gto, scf, fci, lib
    from itercc.pyscf_interface import from_rhf
    lib.num_threads(1)
    mf = scf.RHF(gto.M(atom="He 0 0 0", basis="6-31g", verbose=0))
    mf.conv_tol = 1e-13
    mf.kernel()
    model = from_rhf(mf)
    solver = RestrictedCC(model.h, model.eri, model.nocc, model.constant)
    ground = solver.solve(energy_tol=1e-13, residual_tol=1e-12)
    eom = EOMSinglet(solver, ground.amplitudes)
    result = eom.solve(2, residual_tol=1e-11, max_space=5)
    exact, _ = fci.direct_spin0.kernel(model.h, model.eri, len(model.h), 2,
                                     ecore=model.constant, nroots=3, tol=1e-13)
    assert result.converged
    np.testing.assert_allclose(result.total_energies, exact[1:], atol=2e-11, rtol=0)


def test_cpu_jacobian_owns_its_frozen_amplitudes():
    solver = RestrictedCC(*make_model(2, 2))
    amplitudes = trial_amplitudes(solver, .035, True)
    eom = EOMSinglet(solver, amplitudes, ground_residual_tol=1, cache_dressing=True)
    vector = np.random.default_rng(632).normal(size=eom.space.size)
    expected = eom.sigma(vector)
    for amplitude in amplitudes:
        amplitude.fill(0.)
    np.testing.assert_array_equal(eom.sigma(vector), expected)


def test_eom_rejects_projected_root_that_fails_full_space_audit(monkeypatch):
    solver = RestrictedCC(np.diag([-1., .5]), np.zeros((2, 2, 2, 2)), 1)
    amplitudes = tuple(np.zeros(shape) for shape in solver.shapes[:2])
    eom = EOMSinglet(solver, amplitudes, method="ccsd")
    monkeypatch.setattr(eom, "full_residual_norm", lambda root, energy: 1e-3)
    with pytest.raises(RuntimeError, match="Full-space EOM residual"):
        eom.solve(1, max_space=3, residual_tol=1e-8)
