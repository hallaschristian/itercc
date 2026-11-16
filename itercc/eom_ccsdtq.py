"""Experimental symmetry-sector, matrix-free EOM-CCSDTQ (CUDA or CPU).

The Jacobian action is the central finite difference of the complete T1-T4
residual of `RestrictedCCSDTQ`, so it runs on the solver's backend: CuPy
arrays with the CUDA solver, NumPy arrays with `backend="cpu"` (the compiled
CPU solver: the shared GEMM executor and the tile programs; EOM needs the
dense T4 layout, so not the packed T4 passes). The solver must use
precision="double": the difference would magnify float32 rounding.
"""
from functools import lru_cache
from itertools import product
from time import perf_counter

import numpy as np

from . import _cpu_quadruples

from itercc.eom import EOMSinglet


@lru_cache(maxsize=None)
def _cuda_kernels():
    import cupy as cp

    add = cp.ElementwiseKernel(
        "float64 delta, int32 position, float64 scale", "raw float64 target",
        "target[position] += scale * delta", "itercc_eom_quad_direction_add",
    )
    gap = cp.ElementwiseKernel(
        "int32 position, raw float64 occupied, raw float64 virtual_energy, int32 stride",
        "float64 gap", "gap = virtual_energy[position % stride] - occupied[position / stride]",
        "itercc_eom_quad_orbital_gap",
    )
    return add, gap


def _host(xp, array):
    return array if xp is np else xp.asnumpy(array)


class QuadruplesSector:
    def __init__(self, solver, labels, irrep):
        self.xp = xp = solver.xp
        self.irrep = int(irrep)
        self.shapes = solver.shapes
        self.indices = []
        self.slices = []
        self.size = 0
        no, nv = solver.nocc, solver.nvir
        labels = np.asarray(labels)
        if labels.shape != (solver.nmo,) or labels.dtype.kind not in "iu":
            raise ValueError("Expected Abelian orbital irrep labels")
        energies = solver._reference.fock.diagonal()
        diagonal = []
        # The CUDA kernels take int32 excitation indices; NumPy any size.
        index_type = np.int64 if xp is np else np.int32
        for rank, shape in enumerate(self.shapes, start=1):
            stride = nv**rank
            if xp is not np and np.prod(shape, dtype=np.int64) > np.iinfo(np.int32).max:
                raise ValueError("The CUDA EOM kernels require int32 excitation indices")
            occupied = (solver.layout.indices if rank == 4 else tuple(product(range(no), repeat=rank)))
            occupied = np.asarray(occupied, dtype=np.int32)
            sector = np.bitwise_xor.reduce(labels[occupied], axis=1) ^ irrep
            virtual_irrep = labels[no:]
            virtual_energy = energies[no:]
            for _ in range(rank-1):
                virtual_irrep = np.bitwise_xor.outer(virtual_irrep, labels[no:]).reshape(-1)
                virtual_energy = np.add.outer(virtual_energy, energies[no:]).reshape(-1)
            allowed = {s: np.flatnonzero(virtual_irrep == s).astype(index_type) for s in set(sector)}
            offsets = np.empty(len(sector) + 1, dtype=np.intp)
            offsets[0] = 0
            np.cumsum([allowed[s].size for s in sector], out=offsets[1:])
            host = np.empty(offsets[-1], dtype=index_type)
            occupied_energy = energies[occupied].sum(axis=1)
            gaps = np.empty(host.size) if xp is np else None
            for row, s in enumerate(sector):
                block = slice(offsets[row], offsets[row + 1])
                np.add(allowed[s], row * stride, out=host[block])
                if gaps is not None:
                    np.subtract(virtual_energy[allowed[s]], occupied_energy[row], out=gaps[block])
            indices = xp.asarray(host)
            self.indices.append(indices)
            self.slices.append(slice(self.size, self.size + indices.size))
            self.size += indices.size
            if xp is np:
                diagonal.append(gaps)
            else:
                diagonal.append(_cuda_kernels()[1](indices, xp.asarray(occupied_energy),
                                                   xp.asarray(virtual_energy), stride))
        self.diagonal = xp.concatenate(diagonal)

    def project(self, vector):
        # All initial guesses are singles. The residual projector, and the
        # permutation-invariant diagonal, keep every generated vector in its range.
        return vector

    def pack(self, amplitudes):
        result = self.xp.empty(self.size)
        for array, indices, block in zip(amplitudes, self.indices, self.slices):
            if self.xp is np:
                # flatiter indexes logical C order even for a strided residual.
                flat = array.reshape(-1) if array.flags.c_contiguous else array.flat
                result[block] = flat[indices]
            else:
                result[block] = array.reshape(-1)[indices]
        return result

    def shift(self, amplitudes, vector, scale):
        for amplitude, indices, block in zip(amplitudes, self.indices, self.slices):
            if self.xp is np:
                flat = amplitude.reshape(-1) if amplitude.flags.c_contiguous else amplitude.flat
                flat[indices] += scale * vector[block]     # distinct entries
            else:
                if not amplitude.flags.c_contiguous:      # reshape would copy: the shift would be lost
                    raise ValueError("Expected C-contiguous device amplitudes")
                _cuda_kernels()[0](vector[block], indices, scale, amplitude.reshape(-1))


class FiniteDifferenceEOM:
    """Numerically differentiate the complete T1–T4 residual equations of a
    CUDA or CPU `RestrictedCCSDTQ` (dense T4 layout)."""
    def __init__(self, solver, amplitudes, labels, irrep, *, step=5e-5):
        if solver._host_r4:
            raise ValueError("The EOM pilot requires device-resident R4")
        if getattr(solver, "precision", "double") != "double":
            # float32 operands round both residuals; the central difference
            # would turn that rounding into Jacobian errors of order 1e-3.
            raise ValueError("Finite-difference EOM needs a precision='double' solver")
        if solver.layout.symmetric:
            raise ValueError("EOM requires a dense T4 layout so it can perturb other symmetry sectors")
        self.xp = xp = solver.xp
        self.method = "ccsdtq"
        self.solver = solver
        self.proxy = solver._reference
        self.amplitudes = solver._as_amplitudes(amplitudes)
        self.space = QuadruplesSector(solver, labels, irrep)
        self.diagonal = self.space.diagonal
        self.ground_energy = solver.reference_energy + solver._reference._energy(
            _host(xp, self.amplitudes[0]), _host(xp, self.amplitudes[1]))
        self.step = step
        self.matvec_count = 0
        self.matvec_seconds = 0.0

    def _shifted(self):
        """Fresh copies of the ground amplitudes (on all threads on the CPU)."""
        if self.xp is np and _cpu_quadruples.available():
            return tuple(_cpu_quadruples.copy(a) for a in self.amplitudes)
        return tuple(a.copy() for a in self.amplitudes)

    def action(self, vector, step, *, omega=None):
        shifted = self._shifted()
        self.space.shift(shifted, vector, step)
        plus = self.solver._residuals(shifted)
        if omega is None:
            value = self.space.pack(plus)
            del plus
        self.space.shift(shifted, vector, -2*step)
        minus = self.solver._residuals(shifted)
        if omega is None:
            value -= self.space.pack(minus)
            value /= 2*step
            return value
        for a,b in zip(plus,minus):
            a -= b
            a /= 2*step
        self.space.shift(plus, vector, -omega)
        xp = self.xp
        return float(np.sqrt(sum(float(xp.dot(a.reshape(-1), a.reshape(-1))) for a in plus)))

    def sigma(self, vector):
        xp = self.xp
        start = perf_counter()
        if xp.iscomplexobj(vector):
            result = (self.action(xp.ascontiguousarray(vector.real), self.step)
                      + 1j*self.action(xp.ascontiguousarray(vector.imag), self.step))
        else:
            result = self.action(vector, self.step)
        if not bool(xp.isfinite(result).all()):
            raise FloatingPointError("Nonfinite EOM-CCSDTQ Jacobian image")
        if xp is not np:
            xp.cuda.get_current_stream().synchronize()
        self.matvec_count += 1
        self.matvec_seconds += perf_counter()-start
        return result

    def full_residual_norm(self, vector, omega):
        xp = self.xp
        if xp.iscomplexobj(vector) and float(xp.max(xp.abs(vector.imag))) > 1e-10:
            raise ValueError("The current CuF pilot expects real roots")
        return self.action(xp.ascontiguousarray(vector.real), self.step/2, omega=omega)

    def singles_guesses(self):
        n = self.space.indices[0].size
        if n < 4:
            raise ValueError("The one-root pilot requires at least four singles in its sector")
        order = np.argsort(_host(self.xp, self.diagonal[:n]))[:4]
        for j in order:
            guess = self.xp.zeros(self.space.size)
            guess[int(j)] = 1
            yield guess

    def solve_one(self, **options):
        guesses = self.singles_guesses()
        # Reuse the response-optimized EOM driver's restarted Davidson loop.
        return EOMSinglet.solve(self, 1, guesses=guesses, **options)


# The CUDA name of the engine (it takes CPU solvers too).
GPUFiniteDifferenceEOM = FiniteDifferenceEOM


class HostAuditEOM(FiniteDifferenceEOM):
    """Keep only one rank-four residual on device during the full audit (CUDA)."""

    def action(self, vector, step, *, omega=None):
        if omega is None or self.xp is np:
            return super().action(vector, step, omega=omega)
        cp = self.xp

        shifted = self._shifted()
        self.space.shift(shifted, vector, step)
        *positive, r4 = self.solver._residuals(shifted)
        host_r4 = cp.asnumpy(r4).reshape(-1)
        del r4

        self.space.shift(shifted, vector, -2*step)
        *negative, r4 = self.solver._residuals(shifted)
        del shifted
        for plus, minus in zip(positive, negative):
            plus -= minus
            plus /= 2*step
        del negative
        flat = r4.reshape(-1)
        count = (128 * 2**20) // flat.itemsize
        for first in range(0, flat.size, count):
            tile = flat[first:first+count]
            cp.subtract(cp.asarray(host_r4[first:first+count]), tile, out=tile)
            tile /= 2*step
        values = (*positive, r4)
        self.space.shift(values, vector, -omega)
        return float(cp.sqrt(sum(cp.dot(a.reshape(-1), a.reshape(-1)) for a in values)))
