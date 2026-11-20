import numpy as np
import pytest

from itercc.active_space import fold_closed_shell, select_active_orbitals


def test_fold_closed_shell_preserves_closed_shell_reference_energy():
    h = np.diag([-2.0, -0.7, 0.4])
    eri = np.zeros((3, 3, 3, 3))
    eri[0, 0, 0, 0] = 0.6
    folded_h, folded_eri, constant = fold_closed_shell(h, eri, 1.2, 1)
    assert folded_h.shape == (2, 2)
    assert folded_eri.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(constant, 1.2 + 2*(-2.0) + .6)


def test_select_active_orbitals_retains_block_order_and_rejects_duplicates():
    h = np.diag(np.arange(5.0))
    eri = np.zeros((5, 5, 5, 5))
    result = select_active_orbitals(h, eri, 3., [0, 1, 2, 3, 0], 2, [1], [2, 0])
    np.testing.assert_array_equal(np.diag(result[0]), [1., 4., 2.])
    assert result[3] == 1
    with pytest.raises(ValueError):
        select_active_orbitals(h, eri, 3., [0, 1, 2, 3, 0], 2, [1, 1], [0])
