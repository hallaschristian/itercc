"""Device-side raw residual checks and Jacobi preconditioning."""

from functools import cache

import numpy as np


@cache
def _kernels():
    from cupy import ElementwiseKernel, ReductionKernel

    maximum = ReductionKernel("float64 value", "float64 maximum",
                              "isnan(value) ? __longlong_as_double(0x7ff0000000000000LL) : fabs(value)",
                              "a > b ? a : b", "maximum = a", "0.0",
                              "itercc_max_abs_raw_residual")
    triples = ElementwiseKernel(
        "float64 raw, raw float64 occupied, raw float64 virtual_sum, int64 width, float64 weight",
        "float64 step",
        "step = (weight * raw) / (occupied[i / width] - virtual_sum[i % width])",
        "itercc_jacobi_packed_triples")
    return maximum, triples


class CudaJacobi:
    def __init__(self, xp, d1, d2, d3o, d3v, damping):
        self.xp = xp
        self.d1, self.d2, self.d3o, self.d3v = map(xp.asarray, (d1, d2, d3o, d3v))
        self.weight = 1 - damping

    def finish(self, residuals, energy_ready, residual_tol, can_update):
        """Read raw maxima first, and transfer steps only when an update is needed."""
        maximum, triples = _kernels()
        maxima = tuple(self.xp.stack([maximum(r) for r in residuals]).get().tolist())
        if (not np.isfinite(maxima).all() or not can_update
                or (energy_ready and max(maxima) <= residual_tol)):
            return maxima, None
        r1, r2, r3 = residuals
        for residual, denominator in ((r1, self.d1), (r2, self.d2)):
            residual *= self.weight
            residual /= denominator
        if r3.size:
            triples(r3, self.d3o, self.d3v, self.d3v.size, self.weight, r3)
        return maxima, tuple(r.get() for r in residuals)