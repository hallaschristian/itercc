"""Initial guesses for a parameter sequence from states already solved.

Finite-field properties, potential-curve scans and similar workflows solve the
same equations at nearby values of a parameter (a field strength, a bond
length). The solution varies smoothly, so a polynomial through solved states
predicts the next one far better than a fixed reference guess: with nodes at
spacing h the starting error is O(h**n) for n nodes. These helpers only choose
nodes and weights; combining amplitudes is left to each method's own layout
(for example `RestrictedCCSDTQ.combine_states`).
"""
from math import isfinite


def lagrange_weights(nodes, target):
    """Weights w_k with sum_k w_k f(x_k) = p(target) for the interpolant p of f at `nodes`."""
    nodes = [float(x) for x in nodes]
    if len(set(nodes)) != len(nodes) or not nodes or not all(map(isfinite, nodes)):
        raise ValueError("Expected distinct finite nodes")
    weights = []
    for k, x in enumerate(nodes):
        weight = 1.0
        for j, other in enumerate(nodes):
            if j != k:
                weight *= (target - other) / (x - other)
        weights.append(weight)
    return weights


def continuation_plan(available, target, *, max_nodes=3):
    """[(key, weight)] predicting the state at `target` from solved states.

    `available` maps parameter values to anything that identifies a solved
    state. The `max_nodes` values nearest to `target` are used (fewer when
    fewer are available), giving a polynomial of degree max_nodes - 1. Returns
    an empty list when nothing is available.
    """
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    nodes = sorted(available, key=lambda x: (abs(x - target), x))[:max_nodes]
    if not nodes:
        return []
    return list(zip((available[x] for x in nodes), lagrange_weights(nodes, target)))


def weighted_sum(states, *, xp=None):
    """sum_k w_k A_k over tuples of same-shape arrays, for example CCSD or CCSDT amplitudes.

    `states` is [(weight, (A_1, A_2, ...)), ...]; arrays are converted with
    `xp.asarray` (NumPy when `xp` is None). Methods with packed or
    symmetry-compressed layouts provide their own combination, for example
    `RestrictedCCSDTQ.combine_states`.
    """
    import numpy as np

    xp = np if xp is None else xp
    total = None
    for weight, arrays in states:
        arrays = [xp.asarray(a, dtype=xp.float64) for a in arrays]
        if total is None:
            total = [weight * a for a in arrays]
        else:
            if len(arrays) != len(total) or any(a.shape != t.shape for a, t in zip(arrays, total)):
                raise ValueError("States must have matching amplitude shapes")
            for t, a in zip(total, arrays):
                t += weight * a
    if total is None:
        raise ValueError("Expected at least one state")
    return tuple(total)
