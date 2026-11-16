"""Independent energy check in a three-electron interacting model."""

from itertools import combinations

import numpy as np

from reference_cc import excitation, solve


def test_iterative_triples_recover_exact_three_electron_energy():
    # Four sites with two alpha electrons and one beta electron. Hopping
    # changes each spin independently; on-site repulsion is a two-body term.
    masks = [
        (1 << a) | (1 << b) | (1 << (c + 4))
        for a, b in combinations(range(4), 2) for c in range(4)
    ]
    index = {mask: i for i, mask in enumerate(masks)}
    h = np.zeros((len(masks), len(masks)))
    energies = [-2.0, -1.0, 0.5, 1.2] * 2
    for col, mask in enumerate(masks):
        h[col, col] = sum(e for p, e in enumerate(energies) if mask & (1 << p))
        h[col, col] += sum(u for p, u in enumerate((0.7, 1.1, 0.6, 0.9))
                           if mask & (1 << p) and mask & (1 << (p + 4)))
        for offset in (0, 4):
            for left, right, hopping in (
                (0, 1, 0.35), (1, 2, 0.28), (2, 3, 0.22),
                (0, 2, 0.41), (1, 3, 0.38), (0, 3, 0.19),
            ):
                for source, target in ((left, right), (right, left)):
                    moved = excitation(mask, (offset + source,), (offset + target,))
                    if moved is not None:
                        h[index[moved[0]], col] -= hopping * moved[1]
    exact = np.linalg.eigvalsh(h)[0]
    reference = (1 << 0) | (1 << 1) | (1 << 4)
    ccsd = solve(h, masks, reference, 8, max_rank=2)
    ccsdt = solve(h, masks, reference, 8, max_rank=3)
    assert abs(ccsdt.energy - exact) < 1e-10
    assert np.linalg.norm(ccsdt.amplitudes[ccsdt.ranks == 3]) > 1e-5
    assert abs(ccsd.energy - exact) > 1e-8
