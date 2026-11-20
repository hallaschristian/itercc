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

from . import _cpu_kernels, _cpu_quadruples, _runtime
from ._cpu_packed import OutputSector

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
    """The EOM vector of one symmetry sector: for each rank, each occupied
    tuple's virtual tuples (increasing flat index, caller orbital order) whose
    irreps (caller `labels`) multiply with the occupied tuple's to `irrep`.

    `indices[r]` holds their flat positions in caller-order arrays (rank four:
    slot * nv^4 + virtual flat index). On the CPU `pack` and `shift` address
    the solver's own arrays, whose virtuals may be in its internal order and
    whose T4 may be symmetry-compressed by a subgroup in which `irrep` is
    totally symmetric (`spatial_symmetry.sector_labels`), through tables of
    positions within each slot for each class of slots, in parallel
    (`itercc_sector_rows`). Rank four's positions are then formed only if
    `indices` is read (7.3e9 entries, 58 GB, for CuF in aug-cc-pVDZ)."""

    def __init__(self, solver, labels, irrep):
        self.xp = xp = solver.xp
        self.irrep = int(irrep)
        self.shapes = solver.shapes
        self.slices = []
        self.size = 0
        no, nv = solver.nocc, solver.nvir
        labels = np.asarray(labels)
        if labels.shape != (solver.nmo,) or labels.dtype.kind not in "iu":
            raise ValueError("Expected Abelian orbital irrep labels")
        self.labels = labels
        layout = solver.layout
        order = getattr(solver, "_virtual_order", None)     # internal k -> caller virtual
        self._compiled = xp is np and _cpu_quadruples.available()
        if order is not None and not self._compiled:
            raise ValueError("A solver with reordered virtuals needs the compiled CPU core")
        if layout.symmetric and not self._compiled:
            raise ValueError("A symmetry-compressed T4 layout needs the compiled CPU core")
        energies = np.array(_host(np, solver._reference.fock.diagonal()))
        if order is not None:
            energies[no + order] = energies[no:].copy()      # caller order
        inverse = np.arange(nv) if order is None else np.argsort(order)   # caller -> internal
        self._rows = []       # per rank: tables of `itercc_sector_rows`
        self._caller = []     # per rank: slot sectors, allowed virtual tuples, vector starts, stride
        self._indices = None
        diagonal = []
        # The CUDA kernels take int32 excitation indices; NumPy any size.
        index_type = np.int64 if xp is np else np.int32
        for rank, shape in enumerate(self.shapes, start=1):
            stride = nv**rank
            if xp is not np and np.prod(shape, dtype=np.int64) > np.iinfo(np.int32).max:
                raise ValueError("The CUDA EOM kernels require int32 excitation indices")
            occupied = (layout.indices if rank == 4 else tuple(product(range(no), repeat=rank)))
            occupied = np.asarray(occupied, dtype=np.int32).reshape(-1, rank)
            sector = np.bitwise_xor.reduce(labels[occupied], axis=1) ^ irrep
            virtual_irrep = labels[no:]
            virtual_energy = energies[no:]
            for _ in range(rank-1):
                virtual_irrep = np.bitwise_xor.outer(virtual_irrep, labels[no:]).reshape(-1)
                virtual_energy = np.add.outer(virtual_energy, energies[no:]).reshape(-1)
            allowed = {s: np.flatnonzero(virtual_irrep == s).astype(index_type)
                       for s in set(sector.tolist())}
            del virtual_irrep
            lengths = np.array([allowed[s].size for s in sector.tolist()], dtype=np.int64)
            starts = np.r_[0, np.cumsum(lengths)].astype(np.int64)
            total = int(starts[-1])
            occupied_energy = energies[occupied].sum(axis=1)
            self._caller.append((sector, allowed, starts, stride))
            if self._compiled:
                gaps = np.empty(total)
                for row, s in enumerate(sector.tolist()):
                    np.subtract(virtual_energy[allowed[s]], occupied_energy[row],
                                out=gaps[starts[row]:starts[row + 1]])
                diagonal.append(gaps)
                self._rows.append(self._tables(solver, rank, occupied, sector, allowed, starts, inverse))
            else:
                host = np.empty(total, dtype=index_type)
                gaps = np.empty(total) if xp is np else None
                for row, s in enumerate(sector.tolist()):
                    block = slice(starts[row], starts[row + 1])
                    np.add(allowed[s], row * stride, out=host[block])
                    if gaps is not None:
                        np.subtract(virtual_energy[allowed[s]], occupied_energy[row], out=gaps[block])
                indices = xp.asarray(host)
                if self._indices is None:
                    self._indices = []
                self._indices.append(indices)
                if xp is np:
                    diagonal.append(gaps)
                else:
                    diagonal.append(_cuda_kernels()[1](indices, xp.asarray(occupied_energy),
                                                       xp.asarray(virtual_energy), stride))
            del virtual_energy
            self.slices.append(slice(self.size, self.size + total))
            self.size += total
        self.diagonal = xp.concatenate(diagonal)

    @staticmethod
    def _tables(solver, rank, occupied, sector, allowed, starts, inverse):
        """Positions of each slot's sector entries in the solver's array of
        `rank`: per class of slots one table of positions within the slot."""
        no, nv = solver.nocc, solver.nvir
        layout = solver.layout
        if rank < 4:
            offsets = np.ravel_multi_index(occupied.T.astype(np.int64), (no,) * rank) * nv**rank
            slot_class = sector.astype(np.int64)
        else:
            offsets = (np.asarray(layout.offsets[:-1], dtype=np.int64) if layout.symmetric
                       else np.arange(len(occupied), dtype=np.int64) * nv**4)
            # A compressed slot's storage also depends on its irrep in the layout's labels.
            stored = (np.asarray(layout.slot_irreps, dtype=np.int64) if layout.symmetric
                      else np.zeros(len(occupied), dtype=np.int64))
            slot_class = sector.astype(np.int64) * 8 + stored
        identity = np.array_equal(inverse, np.arange(nv))
        tables, keys = [], {}
        classes = np.empty(len(occupied), dtype=np.int64)
        for row, key in enumerate(slot_class.tolist()):
            if key not in keys:
                keys[key] = len(tables)
                internal = allowed[int(sector[row])].astype(np.int64)
                if not identity:
                    digits = np.unravel_index(internal, (nv,) * rank)
                    internal = np.ravel_multi_index(tuple(inverse[d] for d in digits), (nv,) * rank)
                if rank == 4 and layout.symmetric:
                    # Where the layout stores each entry: compress a marker slot.
                    marker = np.zeros((1,) + (nv,) * 4)
                    marker.reshape(-1)[internal] = np.arange(1, internal.size + 1, dtype=np.float64)
                    segment = np.zeros(int(layout.offsets[row + 1] - layout.offsets[row]))
                    layout.compress(marker, segment, row)
                    stored_at = np.flatnonzero(segment)
                    if stored_at.size != internal.size:
                        raise ValueError("The T4 layout drops entries of the EOM sector: its symmetry "
                                         "labels must keep the target irrep totally symmetric")
                    internal = np.empty(internal.size, dtype=np.int64)
                    internal[segment[stored_at].astype(np.int64) - 1] = stored_at
                    del marker, segment, stored_at
                tables.append(np.ascontiguousarray(internal, dtype=np.int64))
            classes[row] = keys[key]
        lengths = np.array([t.size for t in tables], dtype=np.int64)
        pointers = np.array([t.ctypes.data for t in tables], dtype=np.int64)
        return (np.ascontiguousarray(offsets, dtype=np.int64), classes, tables, lengths, pointers,
                np.ascontiguousarray(starts[:-1], dtype=np.int64))

    @property
    def indices(self):
        """Flat positions of the vector's entries in caller-order arrays."""
        if self._indices is None:
            self._indices = []
            for sector, allowed, starts, stride in self._caller:
                host = np.empty(int(starts[-1]), dtype=np.int64)
                for row, s in enumerate(sector.tolist()):
                    np.add(allowed[s], row * stride, out=host[starts[row]:starts[row + 1]])
                self._indices.append(host)
        return self._indices

    def _storage_positions(self, rank):
        """Flat positions of the vector's rank entries in the solver's array."""
        offsets, classes, tables, _, _, starts = self._rows[rank - 1]
        block = self.slices[rank - 1]
        host = np.empty(block.stop - block.start, dtype=np.int64)
        for row, (offset, c) in enumerate(zip(offsets.tolist(), classes.tolist())):
            np.add(tables[c], offset, out=host[starts[row]:starts[row] + tables[c].size])
        return host

    def _sector_rows(self, rank, array, vector, scale, mode):
        """itercc_sector_rows on a C-contiguous float64 array (False: another array)."""
        if not (isinstance(array, np.ndarray) and array.dtype == np.float64 and array.flags.c_contiguous
                and vector.flags.c_contiguous and vector.dtype == np.float64):
            return False
        offsets, classes, _, lengths, pointers, starts = self._rows[rank - 1]
        _cpu_kernels.library().sector_rows(array.ctypes.data, len(offsets), offsets.ctypes.data,
                                            classes.ctypes.data, pointers.ctypes.data, lengths.ctypes.data,
                                            starts.ctypes.data, vector.ctypes.data, float(scale), mode,
                                            _runtime.cpu_threads(None))
        return True

    def project(self, vector):
        # All initial guesses are singles. The residual projector, and the
        # permutation-invariant diagonal, keep every generated vector in its range.
        return vector

    def pack(self, amplitudes):
        result = self.xp.empty(self.size)
        for rank, (array, block) in enumerate(zip(amplitudes, self.slices), 1):
            if self.xp is np:
                if self._compiled and self._sector_rows(rank, array, result[block], 0.0, 0):
                    continue
                positions = self._storage_positions(rank) if self._compiled else self._indices[rank - 1]
                # flatiter indexes logical C order even for a strided residual.
                flat = array.reshape(-1) if array.flags.c_contiguous else array.flat
                result[block] = flat[positions]
            else:
                result[block] = array.reshape(-1)[self._indices[rank - 1]]
        return result

    def shift(self, amplitudes, vector, scale, *, assign=False):
        """amplitudes += scale * vector on the sector's entries (with `assign`:
        those entries = scale * vector)."""
        for rank, (amplitude, block) in enumerate(zip(amplitudes, self.slices), 1):
            if self.xp is np:
                part = np.ascontiguousarray(vector[block], dtype=np.float64)
                if self._compiled and self._sector_rows(rank, amplitude, part, scale, 2 if assign else 1):
                    continue
                positions = self._storage_positions(rank) if self._compiled else self._indices[rank - 1]
                flat = amplitude.reshape(-1) if amplitude.flags.c_contiguous else amplitude.flat
                if assign:
                    flat[positions] = scale * part
                else:
                    flat[positions] += scale * part     # distinct entries
            else:
                if assign:
                    raise ValueError("Assigning sector entries is a CPU operation")
                if not amplitude.flags.c_contiguous:      # reshape would copy: the shift would be lost
                    raise ValueError("Expected C-contiguous device amplitudes")
                _cuda_kernels()[0](vector[block], self._indices[rank - 1], scale, amplitude.reshape(-1))

    def clear(self, amplitudes):
        """Set the sector's entries of host amplitudes to zero."""
        empty = np.empty(0)
        for rank, amplitude in enumerate(amplitudes, 1):
            if self._compiled and self._sector_rows(rank, amplitude, empty, 0.0, 3):
                continue
            positions = self._storage_positions(rank) if self._compiled else self._indices[rank - 1]
            flat = amplitude.reshape(-1) if amplitude.flags.c_contiguous else amplitude.flat
            flat[positions] = 0.0


class FiniteDifferenceEOM:
    """Numerically differentiate the complete T1–T4 residual equations of a
    CUDA or CPU `RestrictedCCSDTQ`.

    `labels` are the caller's orbital irreps (the full Abelian group) and
    `irrep` the target sector. The solver may keep a symmetry-compressed T4
    and symmetry-blocked GEMMs for labels in which `irrep` is totally
    symmetric: `spatial_symmetry.sector_labels(labels, irrep)` (the full group
    for irrep 0, a subgroup of half its order otherwise), since T + eps R then
    keeps them (CPU only); `combine_states` brings archived ground amplitudes
    into its layout. Without labels it works on the dense layout.

    For a target irrep other than the totally symmetric one (`one_sided`, the
    CPU default) one residual per action suffices: the terms of
    Res(T + eps R) of odd order in eps lie in that sector and the even ones in
    the totally symmetric sector (R x R is totally symmetric), and Res(T) is
    zero there, so its sector part over eps is the central difference, term
    for term. The shifted amplitudes' sector entries are set to eps R, so any
    sector leakage of the stored ground amplitudes drops out; the integrals
    must be free of it (they are audited). The shift is made in the ground
    amplitudes themselves (their sector entries are eps R during an action
    and zero after it; the totally symmetric entries are never written), so
    an action holds one copy of T rather than two (107 GB of T4 for CuF in
    aug-cc-pVDZ with B1-subgroup labels); amplitudes that are not writable
    C-contiguous host arrays are copied once, here. Only the sector's
    residual entries are formed where the residual's outputs allow it
    (`_cpu_packed.OutputSector`; ITERCC_EOM_SECTOR_OUTPUTS=0 forms them all).
    """
    def __init__(self, solver, amplitudes, labels, irrep, *, step=5e-5, one_sided=None):
        if solver._host_r4:
            raise ValueError("The EOM pilot requires device-resident R4")
        self.xp = xp = solver.xp
        if one_sided is None:
            one_sided = int(irrep) != 0 and xp is np
        if one_sided and (int(irrep) == 0 or xp is not np):
            raise ValueError("One residual per action needs a nonzero target irrep on the CPU")
        if getattr(solver, "precision", "double") != "double" and not one_sided:
            # float32 operands round both residuals; for the totally symmetric
            # sector the central difference would turn that rounding of O(1)
            # entries into Jacobian errors of order 1e-3. A one-sided action's
            # sector entries are O(eps) term by term, so float32 copies move
            # it by their rounding relative to those terms.
            raise ValueError("Finite-difference EOM needs a precision='double' solver unless one-sided")
        labels = np.asarray(labels)
        if labels.shape != (solver.nmo,) or labels.dtype.kind not in "iu":
            raise ValueError("Expected Abelian orbital irrep labels")
        if solver.layout.symmetric:
            if xp is not np:
                raise ValueError("EOM requires a dense T4 layout on the GPU so it can perturb other "
                                 "symmetry sectors")
            _check_sector_labels(solver.orbital_irreps, labels, int(irrep))
        self.method = "ccsdtq"
        self.solver = solver
        self.cpu_threads = getattr(solver, "cpu_threads", None)
        self.proxy = solver._reference
        self.amplitudes = solver._as_amplitudes(amplitudes)
        self.space = QuadruplesSector(solver, labels, irrep)
        self.diagonal = self.space.diagonal
        self.ground_energy = solver.reference_energy + solver._reference._energy(
            _host(xp, self.amplitudes[0]), _host(xp, self.amplitudes[1]))
        self.step = step
        self.one_sided = bool(one_sided)
        self._sector = None
        if self.one_sided:
            _check_integral_sectors(solver, labels)
            self.amplitudes = tuple(a if _writable(a) else _cpu_quadruples.copy(np.asarray(a))
                                    for a in self.amplitudes)
            # Only the sector's residual entries are packed: during an action
            # the layout carries the sector (labels in the solver's orbital
            # order), and the R4 tile programs and ring terms form only those
            # entries.
            no = solver.nocc
            order = getattr(solver, "_virtual_order", None)
            virtual = labels[no:] if order is None else labels[no:][order]
            self._sector = OutputSector(labels[:no], virtual, irrep)
        self.matvec_count = 0
        self.matvec_seconds = 0.0

    def _shifted(self):
        """Fresh copies of the ground amplitudes (on all threads on the CPU)."""
        if self.xp is np and _cpu_quadruples.available():
            return tuple(_cpu_quadruples.copy(a) for a in self.amplitudes)
        return tuple(a.copy() for a in self.amplitudes)

    def action(self, vector, step, *, omega=None):
        if self.one_sided:
            return self._one_sided(vector, step, omega)
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

    def _one_sided(self, vector, step, omega):
        """The sector part of Res(T + step R) over step (see the class); with
        `omega`, the norm of its difference from omega R."""
        shifted = self.amplitudes
        layout = self.solver.layout
        if _runtime.enabled("ITERCC_EOM_SECTOR_OUTPUTS"):
            layout.output_sector = self._sector
        try:
            self.space.shift(shifted, vector, step, assign=True)
            residuals = self.solver._residuals(shifted)
        finally:
            self.space.clear(shifted)
            layout.output_sector = None
        value = self.space.pack(residuals)
        del residuals
        value /= step
        if omega is None:
            return value
        value -= omega * vector
        return float(np.sqrt(float(value @ value)))

    @_runtime.threaded
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

    @_runtime.threaded
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

    @_runtime.threaded
    def solve_one(self, **options):
        guesses = self.singles_guesses()
        # Reuse the response-optimized EOM driver's restarted Davidson loop.
        return EOMSinglet.solve(self, 1, guesses=guesses, **options)


def _writable(array):
    """A host array the one-sided action may shift in place (not a memory map)."""
    return (type(array) is np.ndarray and array.dtype == np.float64 and array.flags.c_contiguous
            and array.flags.writeable)


def _check_sector_labels(solver_labels, labels, irrep):
    """A symmetry-compressed solver must keep the target sector: its labels a
    function of the caller's that is XOR-linear and sends `irrep` to 0."""
    mapping = {}
    for full, kept in zip(np.asarray(labels).tolist(), np.asarray(solver_labels).tolist()):
        if mapping.setdefault(full, kept) != kept:
            raise ValueError("The solver's orbital labels are not a subgroup of the EOM labels")
    for x in mapping:
        for y in mapping:
            if x ^ y in mapping and mapping[x ^ y] != mapping[x] ^ mapping[y]:
                raise ValueError("The solver's orbital labels are not a subgroup of the EOM labels")
        if x ^ irrep in mapping and mapping[x ^ irrep] != mapping[x]:
            raise ValueError("The solver's labels separate the EOM sector from the ground state; use "
                             "spatial_symmetry.sector_labels(labels, irrep)")


def _check_integral_sectors(solver, labels):
    """One residual per action needs integrals without entries that break the
    caller's symmetry (they would put Res(T) into the target sector)."""
    from .spatial_symmetry import has_integral_leakage

    reference = solver._reference
    order = getattr(solver, "_virtual_order", None)
    internal = np.asarray(labels).copy()
    no = solver.nocc
    if order is not None:
        internal[no:] = internal[no:][order]
    if has_integral_leakage(_host(np, reference.h), _host(np, reference.g), internal):
        raise ValueError("The integrals break the orbital symmetry: project them "
                         "(spatial_symmetry.project_integrals) or pass one_sided=False")


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
