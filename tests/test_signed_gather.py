from itertools import product

import numpy as np
import pytest

from itercc._spin_quadruples_storage import PackedSpinQuadruples
from itercc._tensor_scatter import signed_gather_rows


def test_shared_signed_row_gather():
    values = np.arange(10, dtype=float) - 3
    maps = [np.array([1, -2, 0, 4], dtype=np.int32), None,
            np.array([-1, 2, -3, 0], dtype=np.int32)]
    actual = signed_gather_rows(values, [1, 0, -4], maps, 4, threads=3)
    if actual is None:
        pytest.skip("Compiled CPU core unavailable")
    np.testing.assert_array_equal(actual, [[-3, 2, 0, 0], [0, 0, 0, 0], [0, -1, 2, 0]])


def test_virtual_maps_shared_only_between_compatible_layouts():
    ground = PackedSpinQuadruples([0, 1], [0, 1, 2], 0, 0)
    direction = PackedSpinQuadruples([0, 1], [0, 1, 2], 1, 1)
    direction.share_virtual_maps(ground)
    assert direction._cpu_signed_maps is ground._cpu_signed_maps
    assert direction._virtual_maps is ground._virtual_maps
    with pytest.raises(ValueError):
        PackedSpinQuadruples([0, 1], [1, 0, 2], 0, 0).share_virtual_maps(ground)


@pytest.mark.parametrize("spin_change", [0, 1])
def test_packed_expansion_matches_numpy_for_all_spin_blocks(spin_change):
    layout = PackedSpinQuadruples(np.array([0, 1, 2]), np.array([0, 1, 2, 3]), 1, spin_change)
    vector = np.random.default_rng(739).normal(size=layout.size)
    cuts = (slice(None, None, -1), slice(0, 2), slice(1, 3), slice(0, 1))
    for signature in product((0, 1), repeat=8):
        if sum(signature[4:]) - sum(signature[:4]) != spin_change:
            continue
        layout.native_expansion = False
        expected = layout.expand(vector, signature, cuts, np)
        layout.native_expansion = True
        actual = layout.expand(vector, signature, cuts, np)
        np.testing.assert_array_equal(actual, expected)
        # Empty occupied slices must retain shape and dtype.
        empty = layout.expand(vector, signature, (slice(0, 0),) + cuts[1:], np)
        assert empty.shape == (4, 4, 4, 4, 0, 2, 2, 1)
