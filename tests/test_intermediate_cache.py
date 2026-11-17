import numpy as np
import pytest

from itercc._intermediate_cache import IntermediateCache


def test_replacement_eviction_and_oversize_invalidation():
    cache = IntermediateCache(64)
    small, large = np.ones(4), np.ones(8)
    cache.put("a", small)
    cache.put("a", large)
    assert cache.bytes == cache.peak_bytes == 64
    assert cache.get("a") is large
    cache.put("a", small)
    cache.put("b", small)
    assert cache.get("a") is small  # make b the oldest
    cache.put("c", small)
    assert cache.get("b") is None
    assert cache.bytes == 64
    cache.put("a", np.ones(9))  # oversized replacement must not leave stale a
    assert cache.get("a") is None
    assert cache.bytes == 32


def test_cache_budget_and_compatibility():
    from itercc._spin_quadruples_tiles import IntermediateCache as compatibility
    assert compatibility is IntermediateCache
    with pytest.raises(ValueError):
        IntermediateCache(-1)
    cache = IntermediateCache(0)
    cache.put("a", np.ones(1))
    assert not cache.entries and cache.bytes == 0
