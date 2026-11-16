"""Small determinant-space reference for iterative CCSD and CCSDT.

The input is a Hamiltonian in a fixed-particle determinant basis. This oracle
evaluates the finite cluster exponentials directly so it can check a future
tensor implementation independently. It is limited to 256 determinants.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import root


@dataclass(frozen=True)
class CCResult:
    energy: float
    amplitudes: np.ndarray
    ranks: np.ndarray
    max_residual: float


def excitation(mask, holes, particles):
    """Apply a†_a a†_b ... a_j a_i with ascending hole/particle tuples."""
    phase = 1
    for orbital in holes:
        bit = 1 << orbital
        if not mask & bit:
            return None
        phase *= 1 - 2 * ((mask & (bit - 1)).bit_count() % 2)
        mask ^= bit
    for orbital in reversed(particles):
        bit = 1 << orbital
        if mask & bit:
            return None
        phase *= 1 - 2 * ((mask & (bit - 1)).bit_count() % 2)
        mask |= bit
    return mask, phase


def solve(hamiltonian, determinants, reference, norb, *, max_rank=3, tolerance=1e-10):
    """Solve the projected exp(-T) H exp(T) equations through max_rank."""
    determinants = tuple(map(int, determinants))
    if not 1 <= len(determinants) <= 256 or max_rank not in (2, 3) or norb < 1:
        raise ValueError("Expected a tiny determinant sector and rank two or three")
    if not 0 < tolerance < 1:
        raise ValueError("Expected a positive residual tolerance below one")
    index = {mask: i for i, mask in enumerate(determinants)}
    reference = int(reference)
    if reference not in index or len(index) != len(determinants) or any(
        mask < 0 or mask >= 1 << norb or mask.bit_count() != reference.bit_count()
        for mask in determinants
    ):
        raise ValueError("Invalid fixed-particle determinant basis")
    h = np.asarray(hamiltonian)
    if (
        h.shape != (len(index), len(index))
        or np.iscomplexobj(h)
        or not np.isfinite(h).all()
        or np.max(abs(h - h.T)) > 1e-12
    ):
        raise ValueError("Expected a finite real symmetric determinant Hamiltonian")

    occupied = [p for p in range(norb) if reference & (1 << p)]
    virtual = [p for p in range(norb) if not reference & (1 << p)]
    connections, projection, phases, ranks = [], [], [], []
    for target in determinants:
        removed = reference & ~target
        rank = removed.bit_count()
        if not 1 <= rank <= max_rank:
            continue
        holes = tuple(p for p in occupied if removed & (1 << p))
        particles = tuple(p for p in virtual if target & (1 << p))
        _, phase = excitation(reference, holes, particles)
        for source, mask in enumerate(determinants):
            edge = excitation(mask, holes, particles)
            if edge is not None and edge[0] in index:
                connections.append((source, index[edge[0]], len(projection), edge[1]))
        projection.append(index[target])
        phases.append(phase)
        ranks.append(rank)
    if not projection:
        raise ValueError("No reference excitations in the supplied sector")
    src, dst, amplitude, sign = np.asarray(connections).T
    reference_vector = np.zeros(len(index))
    ref_index = index[reference]
    reference_vector[ref_index] = 1.0
    limit = min(len(occupied), len(virtual))
    # Subtracting the reference diagonal reduces cancellation in the residuals.
    origin = float(h[ref_index, ref_index])
    shifted = h - origin * np.eye(len(index))

    def exponential(vector, weights):
        result = vector.copy()
        term = vector
        for k in range(1, limit + 1):
            term = np.bincount(dst, weights=weights * term[src], minlength=len(index)) / k
            result += term
        return result

    def transformed(t):
        weights = sign * t[amplitude]
        return exponential(shifted @ exponential(reference_vector, weights), -weights)

    def residual(t):
        return np.asarray(phases) * transformed(t)[projection]

    fit = root(residual, np.zeros(len(projection)), tol=min(tolerance, 1e-11))
    maximum = float(np.max(abs(residual(fit.x))))
    if not np.isfinite(maximum) or maximum > tolerance:
        raise RuntimeError(f"CC residual {maximum:.3e} exceeds tolerance: {fit.message}")
    return CCResult(float(origin + transformed(fit.x)[ref_index]), fit.x, np.asarray(ranks), maximum)
