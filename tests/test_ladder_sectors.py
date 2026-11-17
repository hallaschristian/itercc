"""The sector-form ladders and the fused T1 dressing reproduce the dense
constructions they replace."""
import numpy as np
import pytest

from itercc import _cpu_kernels, _cpu_triples as cpu
from itercc._blocked import irrep_groups

pytestmark = pytest.mark.skipif(_cpu_kernels.library() is None, reason="compiled core unavailable")

LABELS = np.array([0, 1, 2, 3, 0, 0, 1, 3, 2, 0, 1])


def symmetric_tensor(labels, rng):
    """A 4-index tensor that is zero unless the labels of its indices XOR to 0
    and that commutes with pair interchange, as the ladder matrices do."""
    n = len(labels)
    w = rng.standard_normal((n, n, n, n))
    w = w + w.transpose(1, 0, 3, 2)
    allowed = (labels[:, None, None, None] ^ labels[None, :, None, None]
               ^ labels[None, None, :, None] ^ labels[None, None, None, :]) == 0
    return np.where(allowed, w, 0.0)


def dense_blocks(w, structure, which, labels):
    (p, q, r, s), even_blocks, odd_blocks = cpu._ladder_sectors(np.ascontiguousarray(w), labels)
    blocks = even_blocks if which == "even" else odd_blocks
    matrix = np.zeros((len(p), len(p)) if which == "even" else (len(r), len(r)))
    for rows, cols, block in blocks:
        matrix[rows, cols] = block
    return matrix, ((p, q) if which == "even" else (r, s))


@pytest.mark.parametrize("threads", [1, 3])
def test_sector_blocks_equal_the_dense_pair_matrices(threads):
    rng = np.random.default_rng(3)
    nv = len(LABELS)
    big = np.zeros((nv + 2,) * 4)
    big[1:-1, 1:-1, 1:-1, 1:-1] = symmetric_tensor(LABELS, rng)
    w = big[1:-1, 1:-1, 1:-1, 1:-1]                         # a strided view, as g(vvvv)
    vv = rng.standard_normal((nv, nv))
    vv = np.where((LABELS[:, None] ^ LABELS) == 0, vv + vv.T, 0.0)
    structure = cpu.LadderSectors(LABELS, irrep_groups(LABELS))
    even, odd = structure.build(w, threads=threads)
    with_vv = structure.build(w, vv=vv, threads=threads)
    dense_w = np.ascontiguousarray(w).copy()
    cpu.add_one_body(dense_w, vv)
    for which, packed, packed_vv in (("even", even, with_vv[0]), ("odd", odd, with_vv[1])):
        reference, pairs = dense_blocks(w, structure, which, LABELS)
        reference_vv, _ = dense_blocks(dense_w, structure, which, LABELS)
        assert all(np.array_equal(a, b) for a, b in zip(pairs, structure.pairs[:2] if which == "even"
                                                       else structure.pairs[2:]))
        for rows, _, block in structure.blocks(packed, which):
            np.testing.assert_array_equal(block, reference[rows, rows])
        for rows, _, block in structure.blocks(packed_vv, which):
            np.testing.assert_allclose(block, reference_vv[rows, rows], rtol=0, atol=1e-13)
        # Everything outside the diagonal sector blocks is zero: nothing is dropped.
        covered = np.zeros_like(reference, dtype=bool)
        for rows, _, _ in structure.blocks(packed, which):
            covered[rows, rows] = True
        assert not np.any(reference[~covered])


def test_ladder_matrix_sectors_equal_the_dense_ladder_matrix():
    rng = np.random.default_rng(5)
    no, labels = 3, np.array([0, 1, 3, 0, 1, 2, 3, 0, 2, 1])
    occupied, virtual = labels[:no], labels[no:]
    nv = len(virtual)
    n = no + nv
    g = rng.standard_normal((n, n, n, n))
    g = g + g.transpose(1, 0, 3, 2)
    total = labels[:, None, None, None] ^ labels[None, :, None, None] ^ labels[None, None, :, None] \
        ^ labels[None, None, None, :]
    g = np.where(total == 0, g, 0.0)
    t2 = rng.standard_normal((no, no, nv, nv))
    t2 = t2 + t2.transpose(1, 0, 3, 2)
    t2 = np.where((occupied[:, None, None, None] ^ occupied[None, :, None, None]
                   ^ virtual[None, None, :, None] ^ virtual[None, None, None, :]) == 0, t2, 0.0)
    vv = rng.standard_normal((nv, nv))
    vv = np.where((virtual[:, None] ^ virtual) == 0, vv + vv.T, 0.0)
    o, v = slice(0, no), slice(no, n)

    class Intermediates:
        pass

    im = Intermediates()
    im.vv = vv
    dense = cpu._ladder_matrix(g, o, v, t2, im, nv)
    structure = cpu.LadderSectors(virtual, irrep_groups(labels))
    g_sectors = structure.build(g[v, v, v, v])
    even, odd = structure.ladder_matrix(g_sectors, vv, t2, g[o, o, v, v])
    for which, packed in (("even", even), ("odd", odd)):
        reference, _ = dense_blocks(dense, structure, which, virtual)
        for rows, _, block in structure.blocks(packed, which):
            np.testing.assert_allclose(block, reference[rows, rows], rtol=0, atol=1e-12)


@pytest.mark.parametrize("threads", [1, 4])
def test_fused_dressing_matches_the_axis_by_axis_dressing(threads):
    from itercc import _cpu_quadruples

    rng = np.random.default_rng(7)
    n, no = 13, 4
    eri = rng.standard_normal((n, n, n, n))
    g0 = eri.transpose(0, 2, 1, 3)                       # the solver's view of the ERIs
    t1 = 0.1 * rng.standard_normal((no, n - no))
    fused = cpu.dressed_integrals(g0, t1, threads)
    library = _cpu_kernels.library()
    reference = _cpu_quadruples.copy(g0)
    down, up = np.ascontiguousarray(-t1.T), np.ascontiguousarray(t1)
    for axis in (0, 1):
        library.dress_axis(reference.ctypes.data, n, axis, no, n - no, 0, no, down.ctypes.data, threads)
    for axis in (2, 3):
        library.dress_axis(reference.ctypes.data, n, axis, 0, no, no, n - no, up.ctypes.data, threads)
    np.testing.assert_allclose(fused, reference, rtol=0, atol=1e-13)
    np.testing.assert_array_equal(fused, cpu.dressed_integrals(g0, t1, 1))   # any thread count


def test_ccsdtq_cpu_default_puts_t4_into_diis():
    from itercc import RestrictedCCSDTQ

    rng = np.random.default_rng(2)
    n, no = 6, 2
    h = np.diag(np.arange(n, dtype=float) - 1.5)
    eri = 0.02 * rng.standard_normal((n, n, n, n))
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(2, 3, 0, 1)
    solver = RestrictedCCSDTQ(h, eri, no, backend="cpu")
    states = []
    solver.solve(max_iter=2, checkpoint=states.append)
    assert states[-1].settings["diis_quadruples"] == "host"
    states.clear()
    solver.solve(max_iter=2, diis_quadruples=False, checkpoint=states.append)
    assert states[-1].settings["diis_quadruples"] is False
