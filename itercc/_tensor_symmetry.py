"""Permutation conventions shared by spin-orbital tensor layouts and kernels."""
from functools import lru_cache
from itertools import combinations, permutations


def parity(indices):
    return (-1) ** sum(i > j for i, j in combinations(indices, 2))


@lru_cache(maxsize=None)
def signed_permutations(rank):
    return tuple((order, parity(order)) for order in permutations(range(rank)))
