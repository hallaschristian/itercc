import os

import numpy as np
import pytest

from itercc._pair_symmetry import PairSectors, pair_product
from itercc.restricted import _pair_ladder


@pytest.fixture(params=["numpy", "cuda"])
def xp(request):
    if request.param == "numpy":
        return np
    if os.environ.get("ITERCC_TEST_CUDA") != "1":
        pytest.skip("set ITERCC_TEST_CUDA=1 to run CUDA checks")
    return pytest.importorskip("cupy")


def as_host(array):
    return array if isinstance(array, np.ndarray) else array.get()


def test_exact_sector_product_and_cross_sector_fallback(xp):
    rng = np.random.default_rng(8)
    labels = np.array([0, 1, 1, 0, 2, 3])
    matrix = rng.normal(size=(len(labels), len(labels)))
    rhs = rng.normal(size=(len(labels), 9))
    matrix[labels[:, None] != labels] = 0.0
    multiply, blocked = pair_product(xp.asarray(matrix), labels, xp=xp)
    assert blocked
    np.testing.assert_allclose(as_host(multiply(xp.asarray(rhs))), matrix @ rhs, rtol=0, atol=1e-14)
    matrix[0, 1] = 1e-18
    multiply, blocked = pair_product(xp.asarray(matrix), labels, xp=xp)
    assert not blocked
    np.testing.assert_array_equal(as_host(multiply(xp.asarray(rhs))),
                                  as_host(xp.asarray(matrix) @ xp.asarray(rhs)))


@pytest.mark.parametrize("leakage", [False, True])
def test_sector_ladder_matches_unblocked_ladder_at_full_trial_amplitudes(xp, leakage):
    rng = np.random.default_rng(8)
    labels = np.array([0, 1, 1, 0, 2, 3])
    irreps = labels[:, None] ^ labels
    size = len(labels)
    matrix = rng.normal(scale=0.1, size=(size,) * 4)
    matrix += matrix.transpose(1, 0, 3, 2).copy()
    matrix[irreps[:, :, None, None] != irreps] = 0.0
    t3 = rng.normal(scale=0.03, size=(5, size, size, size))
    if leakage:
        matrix[0, 0, 0, 1] = matrix[0, 0, 1, 0] = 1e-8
    dense, blocked = xp.zeros_like(xp.asarray(t3)), xp.zeros_like(xp.asarray(t3))
    _pair_ladder(dense, t3, matrix, 2, xp=xp)
    _pair_ladder(blocked, t3, matrix, 2, xp=xp, virtual_irreps=labels)
    np.testing.assert_allclose(as_host(blocked), as_host(dense), rtol=0, atol=2e-15)

def test_blocked_full_residual_preserves_all_terms(monkeypatch, xp):
    from itercc import RestrictedCC
    from test_restricted import make_model, trial_amplitudes

    no, nv = 3, 4
    h, eri, _, constant = make_model(no, nv)
    labels = np.array([0, 1, 2, 0, 1, 2, 3])
    pair = labels[:, None] ^ labels
    h[pair != 0] = 0
    eri[pair[:, :, None, None] != pair] = 0
    # This checks the row-blocked dense-slot products; the symmetry-blocked
    # T3 path is covered in test_blocked_residual.py.
    monkeypatch.setenv("ITERCC_BLOCKED", "0")
    dense = RestrictedCC(h, eri, no, constant)
    backend = "cpu" if xp is np else "cuda"
    blocked = RestrictedCC(h, eri, no, constant, orbital_irreps=labels, residual_backend=backend)
    occ, vir = labels[:no], labels[no:]
    t1, t2, t3 = trial_amplitudes(dense, 0.05, True)
    t1[occ[:, None] != vir] = 0
    t2[(occ[:, None] ^ occ)[:, :, None, None] != (vir[:, None] ^ vir)] = 0
    vir3 = vir[:, None, None] ^ vir[None, :, None] ^ vir[None, None, :]
    for tile, (i, j, k) in zip(t3, dense.triples_indices):
        tile[vir3 != (occ[i] ^ occ[j] ^ occ[k])] = 0
    chosen = []

    # Both the generic products and the threaded CPU path pick exact sector
    # blocks here: direct, crossed and the two T3 ladder sectors, plus (on the
    # CPU) the two sectors of the doubles vvvv ladder. The CPU finds them with
    # `_cpu_triples._sector_blocks` (block-diagonal: no sector shift).
    if xp is np:
        from itercc import _cpu_triples

        original = _cpu_triples._sector_blocks

        def track(sectors, grouped):
            blocks = original(sectors, grouped)
            chosen.append(blocks is not None and all(rows == cols for rows, cols, _ in blocks))
            return blocks

        monkeypatch.setattr(_cpu_triples, "_sector_blocks", track)
    else:
        original = PairSectors._blocks

        def track(self, matrix, xp, grouped, order):
            blocks = original(self, matrix, xp, grouped, order)
            chosen.append(blocks is not None)
            return blocks

        monkeypatch.setattr(PairSectors, "_blocks", track)
    for expected, actual in zip(dense.residuals((t1, t2, t3)), blocked.residuals((t1, t2, t3))):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    assert chosen == [True] * (6 if xp is np else 4)

