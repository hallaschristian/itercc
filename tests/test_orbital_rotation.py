from itertools import permutations

import numpy as np
import pytest

from itercc import RestrictedCC
from itercc.orbital_rotation import rotate_amplitudes
from test_restricted import make_model, trial_amplitudes


@pytest.mark.parametrize("virtuals", [2, 4])
def test_packed_gauge_transport_matches_full_tensor_rotation(virtuals):
    no, nv = 3, 4
    solver = RestrictedCC(*make_model(no, nv))
    amps = trial_amplitudes(solver, 0.07, True)
    rng = np.random.default_rng(42)
    uo, uv = (np.linalg.qr(rng.normal(size=(n, n)))[0] for n in (no, nv))
    uv = uv[:, :virtuals]
    actual = rotate_amplitudes(amps, uo, uv, batch=3)

    full = np.zeros((no, no, no, nv, nv, nv))
    for label, tile in zip(solver.triples_indices, amps[2]):
        for p in permutations(range(3)):
            full[tuple(label[k] for k in p)] = tile.transpose(p)
    expected = np.einsum("ijkabc,iI,jJ,kK,aA,bB,cC->IJKABC", full,
                         uo, uo, uo, uv, uv, uv, optimize=True)
    for label, tile in zip(solver.triples_indices, actual[2]):
        np.testing.assert_allclose(tile, expected[label], rtol=0, atol=1e-14)
    np.testing.assert_allclose(actual[0], uo.T @ amps[0] @ uv, rtol=0, atol=1e-14)
    np.testing.assert_allclose(actual[1], np.einsum("ijab,iI,jJ,aA,bB->IJAB", amps[1],
                                                  uo, uo, uv, uv, optimize=True),
                               rtol=0, atol=1e-14)
    if virtuals == nv:
        for got, wanted in zip(rotate_amplitudes(actual, uo.T, uv.T), amps):
            np.testing.assert_allclose(got, wanted, rtol=0, atol=1e-14)
