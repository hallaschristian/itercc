"""Abelian labels conserved by one-electron field perturbations."""
from itertools import product

import numpy as np

from itercc.spatial_symmetry import conserved_labels, operator_irreps


def test_operator_irreps_of_c2v_dipoles():
    labels = np.array([0, 2, 1, 3, 0])          # A1, B1, A2, B2, A1
    x = np.zeros((5, 5))
    x[0, 1] = x[1, 0] = x[2, 3] = x[3, 2] = 0.3  # x is B1
    z = np.diag([0.1, 0.2, 0.3, 0.4, 0.5])
    z[0, 4] = z[4, 0] = 0.7                      # z is A1
    assert operator_irreps(x, labels, tolerance=1e-12) == [2]
    assert operator_irreps(z, labels, tolerance=1e-12) == [0]
    x[0, 3] = 1e-13
    assert operator_irreps(x, labels, tolerance=1e-12) == [2]


def test_conserved_labels_are_a_homomorphism_that_kills_the_field():
    group = np.arange(8)
    for broken in ([2], [3], [5], [1, 6], [2, 3]):
        image = conserved_labels(group, broken)
        for g, h in product(group, repeat=2):
            assert image[g ^ h] == image[g] ^ image[h]
        assert all(image[p] == 0 for p in broken)
        assert len(set(image.tolist())) == 8 >> len({int(x) for x in broken})
    # A field along x in C2v (B1) leaves the Cs labels A' = {A1, B1}, A'' = {A2, B2}.
    assert conserved_labels([0, 1, 2, 3], [2]).tolist() == [0, 1, 0, 1]
    assert conserved_labels([0, 1, 2, 3], [0]).tolist() == [0, 1, 2, 3]
