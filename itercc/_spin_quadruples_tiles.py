"""Occupied-tile execution of the spin-orbital rank-four diagrams."""
from collections import defaultdict
from functools import lru_cache
from itertools import product
from math import prod

import numpy as np

from ._spin_quadruples_diagrams import DIAGRAMS
from ._spin_quadruples_storage import signed_permutations
from .bounded_einsum import einsum, einsum_into


@lru_cache(maxsize=None)
def contraction_schedule(spec, no, nv, fixed=()):
    """Binary contraction tree, minimizing peak intermediate size first."""
    inputs, output = spec.split("->")
    inputs = tuple(inputs.split(","))
    count = len(inputs)
    lengths = {c: (1 if c in fixed else nv if c in "abcdef" else no)
               for c in set("".join(inputs))}

    @lru_cache(maxsize=None)
    def plan(mask):
        if mask.bit_count() == 1:
            pos = mask.bit_length() - 1
            return (0, 0), inputs[pos], pos
        inside = set().union(*(set(term) for i, term in enumerate(inputs) if mask >> i & 1))
        outside = set(output).union(*(set(term) for i, term in enumerate(inputs) if not mask >> i & 1))
        letters = "".join(sorted(inside & outside))
        best = None
        for left in range(1, mask):
            if left & mask != left or not left & (mask & -mask):
                continue
            right = mask ^ left
            ca, a, pa = plan(left)
            cb, b, pb = plan(right)
            cost = (max(ca[0], cb[0], prod(lengths[c] for c in letters)),
                    ca[1] + cb[1] + prod(lengths[c] for c in set(a + b)))
            if best is None or cost < best[0]:
                best = cost, letters, (a + "," + b + "->" + letters, pa, pb)
        return best

    return plan((1 << count) - 1)[2]


def _evaluate(xp, tree, operands):
    if isinstance(tree, int):
        return operands[tree]()
    spec, left, right = tree
    return einsum(xp, spec, _evaluate(xp, left, operands), _evaluate(xp, right, operands))


def tile_product(xp, spec, operands, target, coefficient, no, nv, fixed=()):
    """Evaluate only the requested occupied tile and accumulate into its output."""
    _, output = spec.split("->")
    if len(operands) == 1:
        target += coefficient * xp.einsum(spec, operands[0]())
        return
    tree = contraction_schedule(spec, no, nv, fixed)
    binary, left, right = tree
    axes = binary.split("->")[1]
    view = target.transpose(tuple(output.index(c) for c in axes))
    einsum_into(xp, view, coefficient, binary,
                _evaluate(xp, left, operands), _evaluate(xp, right, operands))


def spin_blocks(tensor):
    """Views of interleaved spin tensors indexed by their alpha/beta signatures."""
    return {signature: tensor[tuple(slice(s, None, 2) for s in signature)]
            for signature in product((0, 1), repeat=tensor.ndim)}


@lru_cache(maxsize=None)
def routes(spec, target, derivative):
    """Spin-conserving primal legs and a single alpha-particle/beta-hole tangent."""
    inputs, output = spec.split("->")
    inputs = inputs.split(",")
    internal = sorted(set("".join(inputs)) - set(output))
    fixed = dict(zip(output, target))
    selected = []
    for spins in product((0, 1), repeat=len(internal)):
        assigned = fixed | dict(zip(internal, spins))
        signatures = tuple(tuple(assigned[c] for c in term) for term in inputs)
        if all(sum(s[len(s) // 2:]) - sum(s[:len(s) // 2]) == (i == derivative)
               for i, s in enumerate(signatures)):
            selected.append(signatures)
    return tuple(selected)


@lru_cache(maxsize=None)
def external_routes(rank, signature):
    """Group antisymmetrizer terms by raw virtual and occupied spin pattern."""
    groups = defaultdict(list)
    for (p, sp), (h, sh) in product(signed_permutations(rank), repeat=2):
        axes = p + tuple(rank + i for i in h)
        raw = tuple(signature[i] for i in axes)
        groups[raw].append((p, h, sp * sh))
    return tuple((raw, tuple(rows)) for raw, rows in groups.items())


def contract_tile(xp, rank, raw, fixed, primal, tangent, no, nv, *, irrep=None,
                  labels=None, engine=None, occupied_slab=2):
    """Raw virtual tile for one external spin pattern and occupied tuple."""
    output_shape = (nv,) * rank + tuple(value.stop - value.start for value in fixed)
    result = xp.zeros(output_shape)
    # T4 enters each diagram at most once. Expanding a given spin slab
    # repeatedly across diagrams is more expensive than the contractions.
    cache = {}
    cache_bytes = 0
    max_cache_bytes = 256 * 2**20
    for coefficient, spec, names in DIAGRAMS[rank]:
        lhs, output = spec.split("->")
        inputs = lhs.split(",")
        cuts = dict(zip(output[rank:], fixed))
        for derivative in range(len(names)):
            for signatures in routes(spec, raw, derivative):
                # Split occupied sums of a packed rank-four input before
                # expansion, including feedback into the lower ranks.
                t4 = names.index("t4") if "t4" in names else None
                summed = "" if t4 is None else "".join(c for c in inputs[t4][4:] if c not in cuts)
                slabs = tuple(slice(i, min(no, i + occupied_slab))
                              for i in range(0, no, occupied_slab))
                for pieces in product(slabs, repeat=len(summed)):
                    selected = cuts | dict(zip(summed, pieces))

                    def fetch(i):
                        nonlocal cache_bytes
                        block = (tangent if i == derivative else primal)[names[i]][signatures[i]]
                        selector = tuple(selected.get(c, slice(None)) for c in inputs[i])
                        if not callable(block):
                            return block[selector]
                        key = i == derivative, signatures[i], tuple((s.start, s.stop) for s in selector)
                        if key in cache:
                            return cache[key]
                        tile = block(selector)
                        if tile.nbytes <= max_cache_bytes:
                            while cache_bytes + tile.nbytes > max_cache_bytes:
                                oldest = next(iter(cache))
                                cache_bytes -= cache.pop(oldest).nbytes
                            cache[key] = tile
                            cache_bytes += tile.nbytes
                        return tile

                    operands = tuple(lambda i=i: fetch(i) for i in range(len(names)))
                    if engine is not None and len(names) == 2 and not summed:
                        current = {c: labels["o"][cut] for c, cut in cuts.items()}
                        engine.einsum(spec, *[op() for op in operands], result, alpha=coefficient,
                                      beta=1.0, labels=current,
                                      irreps=tuple(irrep if i == derivative else 0 for i in range(2)))
                    else:
                        tile_product(xp, spec, operands, result, coefficient, no, nv, tuple(cuts))
    return result
