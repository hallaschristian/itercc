"""Validate state interaction against all-M determinant-space EOM-CCSD."""

from itertools import combinations
import os

import numpy as np
import pytest

from itercc import Hamiltonian, TensorCC, spin_orbit_from_components, spin_orbit_state_interaction
from test_tensor_oracle import apply_cluster_exponential, determinant_problem, excitation


def _one_body(operators, masks):
    """Matrix elements of a_p^dagger a_q for all spin-orbital pairs."""
    index = {mask: i for i, mask in enumerate(masks)}
    matrices = np.zeros((operators, operators, len(masks), len(masks)))
    for col, mask in enumerate(masks):
        for p in range(operators):
            for q in range(operators):
                moved = excitation(mask, (q,), (p,))
                if moved and moved[0] in index:
                    matrices[p, q, index[moved[0]], col] = moved[1]
    return matrices


@pytest.fixture(scope="module")
def interacting_problem():
    rng = np.random.default_rng(681)
    nmo = 3
    factors = rng.normal(scale=0.15, size=(6, nmo, nmo))
    factors += factors.swapaxes(1, 2).copy()
    h = np.diag([-1.8, 0.4, 0.7])
    model = Hamiltonian.restricted(h, np.einsum("xpq,xrs->pqrs", factors, factors), 1)
    ground = TensorCC(model, rank=2).solve(energy_tol=1e-13, residual_tol=1e-12)
    assert ground.converged
    t1, t2 = ground.amplitudes

    masks = [sum(1 << p for p in occ) for occ in combinations(range(2 * nmo), 2)]
    index = {mask: i for i, mask in enumerate(masks)}
    dim = len(masks)
    scalar = np.zeros((dim, dim))
    for na, nb in ((1, 1), (2, 0), (0, 2)):
        sector, sector_masks = determinant_problem(model, na, nb)
        idx = [index[mask] for mask in sector_masks]
        scalar[np.ix_(idx, idx)] = sector

    one_body = _one_body(2 * nmo, masks)
    cluster = np.einsum("ai,aimn->mn", t1, one_body[2:, :2])
    for a, b in combinations(range(2, 2 * nmo), 2):
        for col, mask in enumerate(masks):
            moved = excitation(mask, (0, 1), (a, b))
            if moved:
                cluster[index[moved[0]], col] += moved[1] * t2[a - 2, b - 2, 0, 1]
    exp_t = apply_cluster_exponential(cluster, np.eye(dim), 2)
    exp_minus_t = apply_cluster_exponential(-cluster, np.eye(dim), 2)
    np.testing.assert_allclose(exp_minus_t @ exp_t, np.eye(dim), atol=2e-15)

    plus = sum(one_body[2 * p, 2 * p + 1] for p in range(nmo))
    ms0 = [i for i, mask in enumerate(masks) if (mask & 0b010101).bit_count() == 1]
    e, vectors = np.linalg.eigh(scalar[np.ix_(ms0, ms0)])
    v = np.eye(dim)[:, ms0] @ vectors
    is_triplet = np.linalg.norm(plus @ v, axis=0) > 1
    s, t = v[:, ~is_triplet], v[:, is_triplet]
    # Exercise complex phases as well as the real scalar-orbital convention.
    s = s * np.exp(1j * rng.normal(size=s.shape[1]))
    t = t * np.exp(1j * rng.normal(size=t.shape[1]))
    tplus = plus @ t / np.sqrt(2)
    tminus = plus.T @ t / np.sqrt(2)
    basis = np.column_stack((s, np.stack((tplus, t, tminus), axis=2).reshape(dim, -1)))
    np.testing.assert_allclose(basis.conj().T @ basis, np.eye(dim), atol=2e-14)

    energies = np.concatenate((e[~is_triplet], np.repeat(e[is_triplet], 3)))
    np.testing.assert_allclose(e[0], ground.energy, atol=2e-12)
    eom_left = basis.conj().T @ exp_t
    eom_right = exp_minus_t @ basis
    np.testing.assert_allclose((exp_minus_t @ scalar @ exp_t) @ eom_right,
                               eom_right * energies, atol=2e-12)
    np.testing.assert_allclose(eom_left @ eom_right, np.eye(dim), atol=2e-14)
    assert np.max(abs(eom_left - eom_right.conj().T)) > 1e-3
    dm_operator = (one_body[::2, ::2] - one_body[1::2, 1::2]).swapaxes(0, 1)
    dm_operator = exp_minus_t @ dm_operator @ exp_t

    def transition(bra, ket):
        return (eom_left[bra] @ dm_operator @ eom_right[:, ket]).transpose(2, 3, 0, 1)

    ns, nt = s.shape[1], t.shape[1]
    i_s, i_0, i_plus = slice(ns), slice(ns + 1, None, 3), slice(ns, None, 3)
    pauli = rng.normal(scale=0.01, size=(3, nmo, nmo)) + 0.01j * rng.normal(size=(3, nmo, nmo))
    pauli = pauli + pauli.conj().swapaxes(1, 2)
    x, y, z = pauli
    so_h = np.empty((2 * nmo, 2 * nmo), complex)
    so_h[::2, ::2], so_h[1::2, 1::2] = z, -z
    so_h[::2, 1::2], so_h[1::2, ::2] = x - 1j * y, x + 1j * y
    direct = scalar + np.einsum("pq,pqij->ij", so_h, one_body)
    args = (e[~is_triplet], e[is_triplet], pauli, transition(i_s, i_0),
            transition(i_0, i_s), transition(i_plus, i_plus))
    assert ns == 6 and nt == 3
    return args, basis, direct


def test_state_interaction_matches_exact_two_electron_eom_ccsd(interacting_problem):
    args, basis, direct = interacting_problem
    result = spin_orbit_state_interaction(*args)
    np.testing.assert_allclose(result.hamiltonian, basis.conj().T @ direct @ basis, atol=2e-13)
    np.testing.assert_allclose(result.energies, np.linalg.eigvalsh(direct), atol=2e-13)
    assert result.max_antihermitian_Eh < 1e-13
    es, et, pauli, *densities = args
    components = [np.einsum("xpq,...qp->...x", pauli, block) for block in densities]
    direct_components = spin_orbit_from_components(es, et, *components)
    np.testing.assert_allclose(direct_components.hamiltonian, result.hamiltonian, atol=2e-14)


def test_directed_density_averaging_and_strided_input(interacting_problem):
    args, _, _ = interacting_problem
    es, et, pauli, st, ts, tt = args
    ts = np.zeros_like(ts)
    result = spin_orbit_state_interaction(es, et, pauli, st, ts, tt)
    hermitian = spin_orbit_state_interaction(*args)
    assert result.max_antihermitian_Eh > 1e-3
    ns = len(es)
    np.testing.assert_allclose(result.hamiltonian[:ns, ns:], hermitian.hamiltonian[:ns, ns:] / 2)
    empty = np.empty((0, 0, 3, 3))
    isolated = spin_orbit_state_interaction([2., 1.], [], pauli, np.empty((2, 0, 3, 3)),
                                          np.empty((0, 2, 3, 3)), empty)
    np.testing.assert_array_equal(isolated.energies, [1., 2.])


@pytest.mark.skipif(os.environ.get("ITERCC_TEST_CUDA") != "1", reason="CUDA check")
def test_state_interaction_cuda(interacting_problem):
    cp = pytest.importorskip("cupy")
    args, _, _ = interacting_problem
    cpu = spin_orbit_state_interaction(*args)
    gpu = spin_orbit_state_interaction(*(cp.asarray(arg) for arg in args), backend="cuda")
    np.testing.assert_allclose(cp.asnumpy(gpu.hamiltonian), cpu.hamiltonian, atol=2e-14)
    np.testing.assert_allclose(cp.asnumpy(gpu.energies), cpu.energies, atol=2e-14)
    staged = spin_orbit_state_interaction(*args, backend="cuda")
    np.testing.assert_allclose(cp.asnumpy(staged.hamiltonian), cpu.hamiltonian, atol=2e-14)
