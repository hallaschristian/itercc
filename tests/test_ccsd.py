"""Restricted CCSD (rank=2) and the pair-sector doubles ladder."""

import numpy as np
import pytest

from itercc import Hamiltonian, RestrictedCC, TensorCC
from itercc._cpu_triples import doubles_ladder
from test_restricted import make_model


@pytest.mark.parametrize("irreps", [None, np.array([1, 2, 3, 0, 3, 0, 1])])
def test_doubles_ladder_matches_einsum(irreps):
    rng = np.random.default_rng(21)
    no, nv = 3, 7
    w = rng.normal(size=(nv,) * 4)
    w += w.transpose(1, 0, 3, 2).copy()                  # pair interchange
    if irreps is not None:
        pair = irreps[:, None] ^ irreps[None, :]
        w[pair[:, :, None, None] != pair[None, None]] = 0.0
    t2 = rng.normal(size=(no, no, nv, nv))
    t2 = t2 + t2.transpose(1, 0, 3, 2)
    expected = 0.5 * np.einsum("abcd,ijcd->ijab", w, t2)
    np.testing.assert_allclose(doubles_ladder(w, t2, virtual_irreps=irreps), expected,
                               rtol=0, atol=1e-13)


def test_restricted_ccsd_matches_spin_orbital_ccsd():
    model = make_model(3, 4)
    criteria = dict(energy_tol=1e-12, residual_tol=1e-10)
    restricted = RestrictedCC(*model, rank=2)
    assert restricted.shapes[2] == (0, 4, 4, 4)
    result = restricted.solve(**criteria)
    expected = TensorCC(Hamiltonian.restricted(*model), rank=2).solve(**criteria)
    assert result.converged and expected.converged
    assert result.energy == pytest.approx(expected.energy, abs=1e-11, rel=0)


def test_ccsd_residual_is_the_ccsdt_residual_without_triples():
    model = make_model(3, 3)
    ccsd, ccsdt = RestrictedCC(*model, rank=2), RestrictedCC(*model)
    rng = np.random.default_rng(5)
    t1 = 0.05 * rng.normal(size=(3, 3))
    t2 = 0.05 * rng.normal(size=(3, 3, 3, 3))
    t2 = 0.5 * (t2 + t2.transpose(1, 0, 3, 2))
    r_ccsd = ccsd.residuals((t1, t2, np.zeros(ccsd.shapes[2])))
    r_ccsdt = ccsdt.residuals((t1, t2, np.zeros(ccsdt.shapes[2])))
    for left, right in zip(r_ccsd[:2], r_ccsdt[:2]):
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-14)
