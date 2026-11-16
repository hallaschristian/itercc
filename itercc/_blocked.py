"""Symmetry-blocked storage of restricted T3.

With Abelian orbital irreps, T(i,j,k; a,b,c) of a symmetric reference is
nonzero only when irrep(a) ^ irrep(b) ^ irrep(c) equals the occupied irrep
irrep(i) ^ irrep(j) ^ irrep(k). `BlockedTriples` keeps, for each packed
occupied slot of `TriplesLayout`, only those virtual triples: for every
(Ga, Gb) the block of virtuals with irreps (Ga, Gb, Gc), row-major over their
ranks within each irrep. For C2v that is about a quarter of the dense-slot
storage, and contractions over blocks skip the forbidden three quarters.

`pack` moves a dense-slot array (slots, nv, nv, nv) into this layout and
`unpack` back; `leakage` is the largest entry that packing would drop. The
work runs in the compiled core (OpenMP) when it is available, with a NumPy
reference otherwise.
"""

import numpy as np

from . import _cpu_kernels
from ._triples_layout import TriplesLayout


def irrep_groups(*labels):
    """Smallest power of two above every label: the XOR-closed irrep count."""
    largest = max((int(np.max(x)) for x in labels if len(x)), default=0)
    groups = 1
    while groups <= largest:
        groups *= 2
    return groups


class BlockedTriples:
    """Block layout of restricted T3 by virtual irreps (see module docstring)."""

    def __init__(self, layout, occupied_irreps, virtual_irreps):
        if not isinstance(layout, TriplesLayout):
            raise TypeError("Expected a TriplesLayout")
        occupied = np.asarray(occupied_irreps, dtype=np.int64)
        virtual = np.asarray(virtual_irreps, dtype=np.int64)
        if occupied.shape != (layout.no,) or virtual.shape != (layout.nv,):
            raise ValueError("Irrep labels do not match the layout")
        if np.any(occupied < 0) or np.any(virtual < 0):
            raise ValueError("Irrep labels must be nonnegative")
        self.layout = layout
        self.nv = layout.nv
        self.groups = groups = irrep_groups(occupied, virtual)
        self.virtual_irreps = virtual
        self.order = np.argsort(virtual, kind="stable").astype(np.int64)
        self.count = np.bincount(virtual, minlength=groups).astype(np.int64)
        self.start = np.concatenate([[0], np.cumsum(self.count)[:-1]]).astype(np.int64)
        self.slot_irrep = np.array([occupied[i] ^ occupied[j] ^ occupied[k]
                                    for i, j, k in layout.indices], dtype=np.int64)
        offsets = np.empty((len(self.slot_irrep), groups, groups), dtype=np.int64)
        total = 0
        for s, gs in enumerate(self.slot_irrep):
            for ga in range(groups):
                for gb in range(groups):
                    offsets[s, ga, gb] = total
                    total += int(self.count[ga] * self.count[gb] * self.count[gs ^ ga ^ gb])
        self.block_offset = offsets
        self.size = total
        self.dense_size = len(self.slot_irrep) * self.nv**3
        self._device_tables = {}
        # Residual tables: rank of each virtual within its irrep, block
        # offsets inside a slot by slot irrep, and each slot's start.
        self.rank = np.empty(self.nv, dtype=np.int64)
        self.rank[self.order] = np.arange(self.nv) - np.repeat(self.start, self.count)
        toff = np.zeros((groups, groups, groups), dtype=np.int64)
        for gamma in range(groups):
            position = 0
            for ga in range(groups):
                for gb in range(groups):
                    toff[gamma, ga, gb] = position
                    position += int(self.count[ga] * self.count[gb] * self.count[gamma ^ ga ^ gb])
        self.toff = toff
        self.slot_offset = np.ascontiguousarray(offsets[:, 0, 0])

    def c_layout(self):
        """The itercc_blocked_t3 structure of this layout."""
        from . import _core

        cached = getattr(self, "_c_layout", None)
        if cached is None:
            cached = self._c_layout = _core.structure(
                _core.BlockedT3,
                dict(nv=self.nv, groups=self.groups, nslots=len(self.slot_irrep), size=self.size),
                dict(slot_irrep=self.slot_irrep, slot_offset=self.slot_offset,
                     irrep=self.virtual_irreps, rank=self.rank, count=self.count,
                     start=self.start, order=self.order, toff=self.toff))
        return cached

    @property
    def shape(self):
        return (self.size,)

    def block(self, blocked, slot, ga, gb):
        """View of one (Ga, Gb, Gc) block of a blocked array."""
        gc = self.slot_irrep[slot] ^ ga ^ gb
        shape = (self.count[ga], self.count[gb], self.count[gc])
        start = self.block_offset[slot, ga, gb]
        return blocked[start:start + int(np.prod(shape))].reshape(shape)

    def _arguments(self):
        ptr = _cpu_kernels.pointer
        return (len(self.slot_irrep), self.nv, self.groups, ptr(self.slot_irrep),
                ptr(self.block_offset), ptr(self.count), ptr(self.start), ptr(self.order))

    @staticmethod
    def _threads(threads):
        from ._cpu_triples import cpu_threads

        return cpu_threads(threads)

    def leakage(self, dense, *, threads=None):
        """Largest |entry| that packing would drop (NaN if one of them is NaN)."""
        dense = self._checked(dense)
        library = _cpu_kernels.library()
        if library is not None:
            ptr = _cpu_kernels.pointer
            return library.t3_leakage(ptr(dense), len(self.slot_irrep), self.nv,
                                      ptr(self.slot_irrep), ptr(self.virtual_irreps),
                                      self._threads(threads))
        virtual = self.virtual_irreps
        triple = virtual[:, None, None] ^ virtual[None, :, None] ^ virtual[None, None, :]
        worst = 0.0
        for tile, gs in zip(dense, self.slot_irrep):
            dropped = np.abs(tile[triple != gs])
            if np.isnan(dropped).any():
                return float("nan")
            worst = max(worst, float(dropped.max(initial=0.0)))
        return worst

    def pack(self, dense, *, tolerance=None, threads=None):
        """Blocked copy of a dense-slot array; with `tolerance`, check what is dropped."""
        dense = self._checked(dense)
        if tolerance is not None:
            dropped = self.leakage(dense, threads=threads)
            if not dropped <= tolerance:
                raise ValueError(f"Symmetry-forbidden T3 entries up to {dropped:.3e} "
                                 f"exceed {tolerance:.3e}")
        blocked = np.empty(self.size)
        library = _cpu_kernels.library()
        if library is not None:
            ptr = _cpu_kernels.pointer
            library.t3_pack(ptr(dense), *self._arguments(), ptr(blocked), self._threads(threads))
            return blocked
        for s in range(len(self.slot_irrep)):
            for ga in range(self.groups):
                for gb in range(self.groups):
                    gc = self.slot_irrep[s] ^ ga ^ gb
                    rows = [self.order[self.start[g]:self.start[g] + self.count[g]]
                            for g in (ga, gb, gc)]
                    self.block(blocked, s, ga, gb)[...] = dense[s][np.ix_(*rows)]
        return blocked

    def unpack(self, blocked, *, threads=None):
        """Dense-slot array of a blocked one, with every forbidden entry zero."""
        blocked = np.ascontiguousarray(blocked, dtype=np.float64)
        if blocked.shape != self.shape:
            raise ValueError(f"Expected a blocked array of shape {self.shape}")
        dense = np.empty(self.layout.shape)
        library = _cpu_kernels.library()
        if library is not None:
            ptr = _cpu_kernels.pointer
            library.t3_unpack(ptr(blocked), *self._arguments(), ptr(dense), self._threads(threads))
            return dense
        dense.fill(0.0)
        for s in range(len(self.slot_irrep)):
            for ga in range(self.groups):
                for gb in range(self.groups):
                    gc = self.slot_irrep[s] ^ ga ^ gb
                    rows = [self.order[self.start[g]:self.start[g] + self.count[g]]
                            for g in (ga, gb, gc)]
                    dense[s][np.ix_(*rows)] = self.block(blocked, s, ga, gb)
        return dense

    # GPU (CuPy arrays, compiled libitercc_cuda) ------------------------------

    def _device_call(self, name, source, target):
        import cupy

        from . import _core

        backend = _core.cuda()
        if backend is None:
            raise RuntimeError("libitercc_cuda is not built (see core/CMakeLists.txt)")
        device = cupy.cuda.Device().id
        tables = self._device_tables.get(device)
        if tables is None:
            tables = tuple(cupy.asarray(table) for table in (
                self.slot_irrep, self.block_offset, self.count, self.start, self.order))
            self._device_tables[device] = tables
        status = getattr(backend, name)(
            source.data.ptr, len(self.slot_irrep), self.nv, self.groups,
            *(table.data.ptr for table in tables), target.data.ptr,
            cupy.cuda.get_current_stream().ptr)
        if status:
            raise RuntimeError(f"{name} failed with CUDA status {status}")
        return target

    def pack_device(self, dense):
        """Blocked copy of a dense-slot CuPy array, on the current stream."""
        import cupy

        dense = cupy.ascontiguousarray(dense, dtype=cupy.float64)
        if dense.shape != self.layout.shape:
            raise ValueError(f"Expected a dense-slot array of shape {self.layout.shape}")
        return self._device_call("t3_pack", dense, cupy.empty(self.size))

    def unpack_device(self, blocked):
        """Dense-slot CuPy array of a blocked one, forbidden entries zero."""
        import cupy

        blocked = cupy.ascontiguousarray(blocked, dtype=cupy.float64)
        if blocked.shape != self.shape:
            raise ValueError(f"Expected a blocked array of shape {self.shape}")
        return self._device_call("t3_unpack", blocked, cupy.empty(self.layout.shape))

    def _checked(self, dense):
        dense = np.ascontiguousarray(dense, dtype=np.float64)
        if dense.shape != self.layout.shape:
            raise ValueError(f"Expected a dense-slot array of shape {self.layout.shape}")
        return dense
