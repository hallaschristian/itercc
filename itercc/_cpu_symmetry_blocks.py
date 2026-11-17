"""Symmetry-blocked `einsum` on the CPU: `SymmetryBlocks` plans, compiled execution.

`CpuSymmetryBlocks` takes NumPy float64 operands and keeps the planning of
`symmetry_blocks.SymmetryBlocks` (sorted composite indices, offset tables, the
GEMM list per irrep pattern of the enumerated letters). Each plan is one table
of blocks that `itercc_sym_contract` in the compiled core runs on its own
GEMM (no BLAS): blocks with a small output and a deep sum (T4 into R2) split
along the sum into a fixed number of slices, large blocks one at a time on
all threads, and the rest spread over the threads as plain loops or
single-threaded GEMMs.
The route depends only on block sizes, so results do not depend on the thread
count. Without the compiled core every call goes to the fallback.
"""
from functools import reduce

import numpy as np

from . import _cpu_kernels
from ._cpu_triples import cpu_threads
from .symmetry_blocks import SymmetryBlocks, _offsets

_COLUMNS = ("a_rows", "a_sum", "b_sum", "b_cols", "rpos", "cpos")


def numpy_einsum(spec, a, b, out=None, alpha=1.0, beta=0.0):
    """Dense fallback with the `SymmetryBlocks.einsum` calling convention."""
    value = np.einsum(spec, a, b, optimize=True)
    if out is None:
        return alpha * value if alpha != 1.0 else value
    if beta == 0.0:
        np.multiply(value, alpha, out=out)
    else:
        out *= beta
        out += alpha * value
    return out


class CpuSymmetryBlocks(SymmetryBlocks):
    """`SymmetryBlocks` for NumPy arrays (see the module docstring)."""

    def __init__(self, labels, combinations, nirrep, *, fallback=numpy_einsum, threads=None):
        super().__init__(labels, combinations, nirrep, fallback=fallback)
        self.threads = threads
        self._library = _cpu_kernels.library()
        # Labels keys of per-call labels dicts by identity; each entry keeps
        # its dict alive, so an id is not reused while it is in the map. One
        # dict lookup per call keeps this safe when threads share the engine.
        self._label_keys = {}

    def release_workspace(self):
        """Free the kernels' per-thread scratch buffers (they keep their pages
        between calls, so a phase of many products faults them in once)."""
        if self._library is not None:
            from ._cpu_triples import cpu_threads

            self._library.scratch_release(cpu_threads(self.threads))

    def _offsets(self, letters, extent, strides, composite):
        key = (composite[0], tuple(strides[c] for c in letters))
        hit = self._tables.get(key)
        if hit is None:
            hit = self._tables[key] = np.ascontiguousarray(
                _offsets(letters, extent, strides, composite[1]), dtype=np.int64)
        return hit

    def _supported(self, a, b, out):
        return (self._library is not None and out.flags.writeable
                and all(isinstance(x, np.ndarray) and x.dtype == np.float64 for x in (a, b, out)))

    def _plan(self, structure, labels, irreps):
        """Block table (rows, cols, depth, r0, c0, k0, a_base, b_base, o_base) for
        one irrep pattern of the enumerated letters and operand irreps."""
        def irrep(letters, combo):
            return reduce(np.bitwise_xor, [int(labels[c][i]) for i, c in zip(combo, letters)], 0)

        r_start, c_start, k_start = structure["r_start"], structure["c_start"], structure["k_start"]
        blocks = []
        for i_r, o_r in enumerate(structure["combos_r"]):
            g_r = irrep(structure["orow"], o_r) ^ irreps[0]
            for i_c, o_c in enumerate(structure["combos_c"]):
                g_c = irrep(structure["ocol"], o_c) ^ irreps[1]
                for kappa in range(self.nirrep):
                    rho, sigma = kappa ^ g_r, kappa ^ g_c
                    rows = int(r_start[rho + 1] - r_start[rho])
                    cols = int(c_start[sigma + 1] - c_start[sigma])
                    depth = int(k_start[kappa + 1] - k_start[kappa])
                    if rows and cols and depth:
                        blocks.append((rows, cols, depth, int(r_start[rho]), int(c_start[sigma]),
                                       int(k_start[kappa]), structure["a_outer"][i_r],
                                       structure["b_outer"][i_c], structure["obase"][i_r][i_c]))
        return np.array(blocks, dtype=np.int64).reshape(-1, 9)

    def _label_key_of(self, labels):
        if not labels:
            return ()
        found = self._label_keys.get(id(labels))
        if found is not None and found[0] is labels:
            return found[1]
        if self.nirrep == 1:                  # every label 0: the extents say it all
            key = tuple((c, len(v)) for c, v in labels.items())
        else:
            key = tuple((c, np.asarray(v, dtype=np.int64).tobytes()) for c, v in labels.items())
        if len(self._label_keys) > 4096:
            self._label_keys = {}
        self._label_keys[id(labels)] = (labels, key)
        return key

    def _entry(self, spec, a, b, out, labels, irreps):
        """(structure, block table, kernel arguments) of one call signature, cached."""
        call = (spec, a.shape, a.strides, a.dtype.char, b.shape, b.strides, b.dtype.char,
                out.shape, out.strides, out.dtype.char, int(irreps[0]), int(irreps[1]),
                self._label_key_of(labels))
        hit = self._calls.get(call)
        if hit is None:
            structure, plan = self._resolve(spec, a, b, out, labels, irreps)
            # The table and offset columns live as long as the cached plan, so
            # their addresses are resolved once instead of on every call.
            arguments = (None if not structure or not len(plan) else
                         (len(plan), plan.ctypes.data, *(structure[name].ctypes.data for name in _COLUMNS),
                          self.flags(structure)))
            hit = self._calls[call] = (structure, plan, arguments)
        return hit

    def plan_for(self, spec, a, b, out, labels=None, irreps=(0, 0)):
        """(structure, block table) of one call signature, cached, or (False, None)
        when the product has no blocked plan (`einsum` then uses the fallback)."""
        return self._entry(spec, a, b, out, labels, irreps)[:2]

    @staticmethod
    def flags(structure):
        """Memory-order flags of a structure for `itercc_sym_contract`."""
        return (int(bool(structure["ta"])) | 2 * int(bool(structure["tb"]))
                | 4 * int(bool(structure["rfast"])))

    def einsum(self, spec, a, b, out, alpha=1.0, beta=0.0, labels=None, irreps=(0, 0)):
        """out = beta * out + alpha * einsum(spec, a, b) (see `SymmetryBlocks.einsum`)."""
        structure, plan, arguments = self._entry(spec, a, b, out, labels, irreps)
        if not structure:
            if self.fallback is None:
                raise ValueError(f"No symmetry-blocked plan for {spec}")
            return self.fallback(spec, a, b, out=out, alpha=alpha, beta=beta)
        threads = cpu_threads(self.threads)
        library = self._library
        if beta != 1.0:
            if out.flags.c_contiguous:
                library.scale(out.ctypes.data, out.size, float(beta), threads)
            elif beta == 0.0:
                out[...] = 0.0
            else:
                out *= beta
        if arguments is None or alpha == 0.0:
            return out
        count, table, a_rows, a_sum, b_sum, b_cols, rpos, cpos, flags = arguments
        library.sym_contract(count, table, a.ctypes.data, b.ctypes.data, out.ctypes.data,
                             a_rows, a_sum, b_sum, b_cols, rpos, cpos, float(alpha), flags, threads)
        return out
