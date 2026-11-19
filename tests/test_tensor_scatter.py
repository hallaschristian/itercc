"""Shared expansion preserves sector selection, dtype and physical projectors."""
from types import SimpleNamespace

import numpy as np
import pytest

from itercc import _cpu_kernels, _tensor_scatter
from itercc.eom import HostDavidson, SingletSpace, _working_bytes


@pytest.mark.parametrize("dtype", [np.float64, np.complex128, np.float32])
@pytest.mark.parametrize("compiled", [False, True])
def test_sector_expansion(dtype, compiled, monkeypatch):
    if compiled:
        if _cpu_kernels.library() is None:
            pytest.skip("Compiled CPU core unavailable")
    else:
        monkeypatch.setattr(_cpu_kernels, "library", lambda: None)
    shape = (64, 64, 64)
    # Unsorted, disjoint indices and strided values exercise layout-independent use.
    indices = np.arange(0, np.prod(shape), 3, dtype=np.int64)[::-1]
    vector = np.random.default_rng(717).normal(size=2 * len(indices))[::2].astype(dtype)
    if dtype == np.complex128:
        vector *= 1 + .3j
    expected = np.zeros(shape, dtype=np.result_type(dtype, np.float64))
    expected.ravel()[indices] = vector
    for threads in (1, 4):
        actual = _tensor_scatter.unpack_sector(vector, indices, shape, threads=threads)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
    empty = _tensor_scatter.unpack_sector(vector[:0], indices[:0], shape)
    assert not np.any(empty)


@pytest.mark.parametrize("triples", [False, True])
def test_singlet_space_uses_shared_expansion_and_retains_projection(triples, monkeypatch):
    from itercc import RestrictedCC
    from test_restricted import make_model
    from itercc import eom

    solver = RestrictedCC(*make_model(2, 3))
    space = SingletSpace(solver, [0, 1, 0, 1, 0], 1, triples=triples)
    vector = np.random.default_rng(719).normal(size=space.size)
    calls = []
    original = _tensor_scatter.unpack_sector

    def recorded(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(eom, "unpack_sector", recorded)
    actual = space.unpack(vector)
    assert len(calls) == (3 if triples else 2)

    def reference(vector, indices, shape, *, xp):
        result = xp.zeros(shape, dtype=vector.dtype)
        result.ravel()[indices] = vector
        return result

    monkeypatch.setattr(eom, "unpack_sector", reference)
    for left, right in zip(actual, space.unpack(vector)):
        np.testing.assert_array_equal(left, right)


def test_host_davidson_preserves_packed_estimate():
    # Deliberately enormous logical T4 shape: only the engine knows its storage.
    engine = SimpleNamespace(space=SimpleNamespace(size=17, shapes=((10**9,) * 8,)),
                             davidson_working_bytes=lambda n: 1234 + 16 * n * 17)
    host = object.__new__(HostDavidson)
    host.engine, host.space = engine, engine.space
    assert _working_bytes(host, 10) == _working_bytes(engine, 10) + 2 * 17 * 8


def test_legacy_spin_import_is_the_shared_implementation():
    from itercc import _cpu_quadruples, _spin_quadruples_storage, _tensor_symmetry
    assert _cpu_quadruples.expand_antisymmetric is _tensor_scatter.expand_antisymmetric
    assert _spin_quadruples_storage.signed_permutations is _tensor_symmetry.signed_permutations


@pytest.mark.parametrize("threads,stride", [(1, 4), (2, 4), (4, 8), (56, 4)])
def test_sector_dispatch_avoids_parallel_overhead(threads, stride, monkeypatch):
    def unexpected():
        raise AssertionError("Ineligible layouts must not invoke the native core")
    monkeypatch.setattr(_cpu_kernels, "library", unexpected)
    indices = np.arange(0, 2**18, stride)
    values = np.ones(len(indices))
    result = _tensor_scatter.unpack_sector(values, indices, (2**18,), threads=threads)
    assert np.count_nonzero(result) == len(indices)


@pytest.mark.parametrize("holes_first", [False, True])
def test_spin_block_unpack_matches_the_dense_unpack(holes_first):
    """unpack_spin_blocks keeps the dense unpack's nonzero spin blocks, bit for
    bit, and only those."""
    from itertools import combinations, product

    from itercc._tensor_scatter import unpack_antisymmetric, unpack_spin_blocks
    from itercc._spin_quadruples_tiles import spin_blocks

    nv, no, rank = 8, 6, 3
    rows = []
    for p in combinations(range(nv), rank):
        for h in combinations(range(no), rank):
            spins = [x % 2 for x in p + h]
            if sum(spins[rank:]) - sum(spins[:rank]) == 1:        # a spin-flip sector
                rows.append(p + h)
    coordinates = np.asarray(rows, dtype=np.int64).T
    vector = np.random.default_rng(11).normal(size=coordinates.shape[1])
    shape = (nv,) * rank + (no,) * rank
    blocked = unpack_spin_blocks(vector, coordinates, shape, holes_first=holes_first)
    if blocked is None:
        pytest.skip("compiled core unavailable")
    dense = spin_blocks(unpack_antisymmetric(vector, coordinates, shape))
    assert len(blocked.blocks) == 15
    for signature in product((0, 1), repeat=2 * rank):
        np.testing.assert_array_equal(blocked[signature], dense[signature])
