"""Host-side exact T3 column packing for a bounded DIIS history."""

import numpy as np


class HostTriplesColumns:
    """Store only symmetry-supported triples entries, with no rounding projection."""

    def __init__(self, occupied, virtual, layout):
        self.shape = layout.shape
        cube = (virtual[:, None, None] ^ virtual[None, :, None] ^ virtual).ravel()
        groups = tuple(np.flatnonzero(cube == irrep) for irrep in range(8))
        rows, offset = [], 0
        for i, j, k in layout.indices:
            selected = groups[occupied[i] ^ occupied[j] ^ occupied[k]]
            end = offset + len(selected)
            rows.append((selected, slice(offset, end)))
            offset = end
        self.rows = tuple(rows)
        self.size = offset

    def pack(self, triples):
        result = np.empty(self.size, dtype=triples.dtype)
        for tile, (selected, target) in zip(triples, self.rows):
            flat = tile.reshape(-1)
            np.take(flat, selected, out=result[target], mode="clip")
            if np.count_nonzero(flat) != np.count_nonzero(result[target]):
                return None
        return result

    def unpack(self, packed):
        result = np.zeros(self.shape, dtype=packed.dtype)
        for tile, (selected, source) in zip(result, self.rows):
            tile.ravel()[selected] = packed[source]
        return result


class BlockedColumns:
    """The entries of HostTriplesColumns in the blocked T3 layout
    (`_blocked.BlockedTriples`), checked, packed and unpacked by the compiled
    kernels on all threads (the NumPy passes above run on one)."""

    def __init__(self, occupied, virtual, layout, threads=None):
        from ._blocked import BlockedTriples

        self.blocked = BlockedTriples(layout, occupied, virtual)
        self.shape, self.size, self.threads = layout.shape, self.blocked.size, threads

    def pack(self, triples):
        """The compact copy, or None when a forbidden entry is nonzero (or NaN)."""
        if triples.dtype != np.float64 or self.blocked.leakage(triples, threads=self.threads) != 0.0:
            return None
        return self.blocked.pack(triples, threads=self.threads)

    def unpack(self, packed):
        return self.blocked.unpack(packed, threads=self.threads)


def host_columns(occupied, virtual, layout, threads=None):
    """BlockedColumns with the compiled CPU core, else HostTriplesColumns."""
    from . import _cpu_kernels

    if _cpu_kernels.library() is not None:
        return BlockedColumns(occupied, virtual, layout, threads)
    return HostTriplesColumns(occupied, virtual, layout)