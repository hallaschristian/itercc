"""Symmetry-blocked T3 storage and the CUDA backend's grouped GEMM."""

import numpy as np
import pytest

from itercc import _core, _cpu_kernels
from itercc._blocked import BlockedTriples, irrep_groups
from itercc._triples_layout import TriplesLayout

OCCUPIED = np.array([0, 1, 2, 0, 3])
VIRTUAL = np.array([1, 2, 3, 0, 3, 0, 1])


@pytest.fixture(params=["compiled", "numpy"])
def cpu_path(request, monkeypatch):
    if request.param == "numpy":
        monkeypatch.setenv("ITERCC_CPU_KERNELS", "0")
    elif _cpu_kernels.library() is None:
        pytest.skip("no compiled core")
    return request.param


def triple_irreps(blocked):
    virtual = blocked.virtual_irreps
    return virtual[:, None, None] ^ virtual[None, :, None] ^ virtual[None, None, :]


def allowed_triples(blocked, rng):
    """A dense-slot T3 that is zero outside the allowed blocks."""
    dense = rng.normal(size=blocked.layout.shape)
    for tile, gs in zip(dense, blocked.slot_irrep):
        tile[triple_irreps(blocked) != gs] = 0.0
    return dense


def forbidden_entry(blocked):
    """Index (slot, a, b, c) of one symmetry-forbidden T3 entry."""
    for s, gs in enumerate(blocked.slot_irrep):
        found = np.argwhere(triple_irreps(blocked) != gs)
        if len(found):
            return (s, *map(int, found[0]))
    raise AssertionError("every entry is allowed")


def test_layout_keeps_only_allowed_triples():
    layout = TriplesLayout(len(OCCUPIED), len(VIRTUAL))
    blocked = BlockedTriples(layout, OCCUPIED, VIRTUAL)
    assert blocked.groups == irrep_groups(OCCUPIED, VIRTUAL) == 4
    triple = VIRTUAL[:, None, None] ^ VIRTUAL[None, :, None] ^ VIRTUAL[None, None, :]
    expected = sum(int(np.count_nonzero(triple == gs)) for gs in blocked.slot_irrep)
    assert blocked.size == expected
    assert blocked.size < 0.3 * blocked.dense_size


def test_round_trip_and_agreement(cpu_path):
    rng = np.random.default_rng(3)
    layout = TriplesLayout(len(OCCUPIED), len(VIRTUAL))
    blocked = BlockedTriples(layout, OCCUPIED, VIRTUAL)
    dense = allowed_triples(blocked, rng)
    packed = blocked.pack(dense, tolerance=0.0, threads=3)
    np.testing.assert_array_equal(blocked.unpack(packed, threads=3), dense)
    # Every block holds exactly the original entries at the sorted ranks.
    s, ga, gb = 5, 1, 3
    gc = blocked.slot_irrep[s] ^ ga ^ gb
    rows = [blocked.order[blocked.start[g]:blocked.start[g] + blocked.count[g]] for g in (ga, gb, gc)]
    np.testing.assert_array_equal(blocked.block(packed, s, ga, gb), dense[s][np.ix_(*rows)])


def test_compiled_and_numpy_packing_agree(monkeypatch):
    if _cpu_kernels.library() is None:
        pytest.skip("no compiled core")
    rng = np.random.default_rng(4)
    layout = TriplesLayout(len(OCCUPIED), len(VIRTUAL))
    blocked = BlockedTriples(layout, OCCUPIED, VIRTUAL)
    dense = rng.normal(size=layout.shape)            # forbidden entries are dropped by both
    compiled = blocked.pack(dense, threads=2), blocked.leakage(dense, threads=2)
    monkeypatch.setenv("ITERCC_CPU_KERNELS", "0")
    reference = blocked.pack(dense), blocked.leakage(dense)
    np.testing.assert_array_equal(compiled[0], reference[0])
    assert compiled[1] == reference[1]


def test_leakage_is_reported(cpu_path):
    rng = np.random.default_rng(5)
    layout = TriplesLayout(len(OCCUPIED), len(VIRTUAL))
    blocked = BlockedTriples(layout, OCCUPIED, VIRTUAL)
    dense = allowed_triples(blocked, rng)
    entry = forbidden_entry(blocked)
    dense[entry] = -3e-9
    assert blocked.leakage(dense) == 3e-9
    with pytest.raises(ValueError, match="forbidden"):
        blocked.pack(dense, tolerance=1e-9)
    dense[entry] = np.nan
    assert np.isnan(blocked.leakage(dense))
    with pytest.raises(ValueError, match="forbidden"):
        blocked.pack(dense, tolerance=1.0)


def cuda_backend():
    import os

    if os.environ.get("ITERCC_TEST_CUDA") != "1":
        pytest.skip("set ITERCC_TEST_CUDA=1 to run CUDA checks")
    cupy = pytest.importorskip("cupy")
    backend = _core.cuda()
    if backend is None:
        pytest.skip("libitercc_cuda is not built")
    return cupy, backend


def test_device_packing_matches_host():
    cupy, _ = cuda_backend()
    rng = np.random.default_rng(8)
    layout = TriplesLayout(len(OCCUPIED), len(VIRTUAL))
    blocked = BlockedTriples(layout, OCCUPIED, VIRTUAL)
    dense = rng.normal(size=layout.shape)
    packed = blocked.pack_device(cupy.asarray(dense))
    np.testing.assert_array_equal(packed.get(), blocked.pack(dense))
    np.testing.assert_array_equal(blocked.unpack_device(packed).get(),
                                  blocked.unpack(blocked.pack(dense)))


def test_device_grouped_gemm_matches_cupy():
    cupy, backend = cuda_backend()
    rng = np.random.default_rng(9)
    sizes = ((3, 4, 5), (2, 6, 3), (7, 1, 2))           # (m, n, k), transposed A in the last
    a = cupy.asarray(rng.normal(size=sum(m * k for m, _, k in sizes)))
    b = cupy.asarray(rng.normal(size=sum(k * n for _, n, k in sizes)))
    c = cupy.asarray(rng.normal(size=sum(m * n for m, n, _ in sizes)))
    original = c.copy()
    desc, coef, offsets = [], [], [0, 0, 0]
    for index, (m, n, k) in enumerate(sizes):
        transa = int(index == 2)
        desc += [transa, 0, m, n, k, offsets[0], m if transa else k, offsets[1], n, offsets[2], n]
        coef += [1.5, -1.0]
        offsets = [offsets[0] + m * k, offsets[1] + k * n, offsets[2] + m * n]
    desc, coef = np.array(desc, dtype=np.int64), np.array(coef)
    status = backend.grouped_gemm(len(sizes), desc.ctypes.data, coef.ctypes.data, a.data.ptr,
                                  b.data.ptr, c.data.ptr, cupy.cuda.get_current_stream().ptr)
    assert status == 0
    start = [0, 0, 0]
    for index, (m, n, k) in enumerate(sizes):
        left = a[start[0]:start[0] + m * k].reshape((k, m) if index == 2 else (m, k))
        left = left.T if index == 2 else left
        right = b[start[1]:start[1] + k * n].reshape(k, n)
        expected = 1.5 * left @ right - original[start[2]:start[2] + m * n].reshape(m, n)
        np.testing.assert_allclose(c[start[2]:start[2] + m * n].reshape(m, n).get(),
                                   expected.get(), rtol=0, atol=1e-13)
        start = [start[0] + m * k, start[1] + k * n, start[2] + m * n]
