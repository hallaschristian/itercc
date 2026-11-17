from itertools import permutations

import numpy as np
import pytest

from itercc._tensor_scatter import PermutedGather


@pytest.mark.parametrize("rank", [2, 3, 4, 6])
@pytest.mark.parametrize("threads", [1, 3])
def test_ordered_gather_matches_numpy_bits(rank, threads):
    rng = np.random.default_rng(733)
    source = rng.normal(size=(9,) * rank)[::-1]  # negative source stride
    coordinates = rng.integers(0, 9, size=(rank, 5000), dtype=np.int32)
    rows = rng.choice(8192, 5000, replace=False)
    terms = tuple((p, (1., -4., 24.)[i % 3]) for i, p in enumerate(permutations(range(rank))))
    result = rng.normal(size=8192)
    expected = result.copy()
    subtotal = np.zeros(len(rows))
    for p, weight in terms:
        subtotal += weight * source[tuple(coordinates[i] for i in p)]
    expected[rows] += subtotal
    plan = PermutedGather(rows, coordinates)
    if not plan.add(source, result, terms, threads=threads):
        pytest.skip("Native CPU kernel unavailable")
    np.testing.assert_array_equal(result, expected)


def test_gather_rejects_unsupported_alias_and_dtype():
    plan = PermutedGather(np.arange(4), np.arange(4, dtype=np.int32).reshape(1, 4))
    values = np.ones(4)
    assert not plan.add(values, values, (((0,), 1.),))
    assert not plan.add(values.astype(complex), np.zeros(4), (((0,), 1.),))
    unaligned_stride = np.ndarray((4,), dtype=np.float64, buffer=bytearray(16), strides=(1,))
    assert not plan.add(unaligned_stride, np.zeros(4), (((0,), 1.),))
