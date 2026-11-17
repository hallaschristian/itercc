"""Compiled canonical expansion agrees with direct signed permutation scatter."""
from itertools import combinations, permutations, product

import numpy as np
import pytest

from itercc import _cpu_quadruples, _tensor_scatter


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
@pytest.mark.parametrize("threads", [1, 3])
def test_spin_unpack_matches_reference(rank, threads, monkeypatch):
    if not _cpu_quadruples.available():
        pytest.skip("Compiled CPU core unavailable")
    no, nv = 4, 5
    coordinates = np.array([v + o for v in combinations(range(nv), rank)
                            for o in combinations(range(no), rank)][::2], dtype=np.int32).T
    vector = np.random.default_rng(716).normal(size=coordinates.shape[1] * 2)[::2]
    expected = np.zeros((nv,) * rank + (no,) * rank)
    for p, h in product(permutations(range(rank)), repeat=2):
        inversions = sum(a > b for order in (p, h) for a, b in combinations(order, 2))
        axes = p + tuple(rank + i for i in h)
        expected[tuple(coordinates[i] for i in axes)] = (-1)**inversions * vector
    actual = _cpu_quadruples.expand_antisymmetric(vector, coordinates, expected.shape, threads=threads)
    np.testing.assert_array_equal(actual, expected)
    empty = _cpu_quadruples.expand_antisymmetric(vector[:0], coordinates[:, :0], expected.shape,
                                                threads=threads)
    np.testing.assert_array_equal(empty, np.zeros_like(expected))
    # The common dispatcher also owns complex and no-compiler fallbacks.
    np.testing.assert_array_equal(
        _tensor_scatter.unpack_antisymmetric(vector.astype(complex) * (1 + .4j),
                                             coordinates, expected.shape), expected * (1 + .4j))
    monkeypatch.setattr(_tensor_scatter, "expand_antisymmetric", lambda *a, **k: None)
    np.testing.assert_array_equal(
        _tensor_scatter.unpack_antisymmetric(vector, coordinates, expected.shape), expected)
