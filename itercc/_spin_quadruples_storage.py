"""Canonical spin-orbital quadruples indexed by occupied tuple and virtual irrep."""
from functools import lru_cache
from itertools import combinations, product

from ._tensor_symmetry import parity, signed_permutations

import numpy as np


def alpha_count(indices):
    return sum(i % 2 == 0 for i in indices)


def orbital_irrep(indices, labels):
    value = 0
    for index in indices:
        value ^= int(labels[index // 2])
    return value


class PackedSpinQuadruples:
    """Rank-four amplitudes with independently antisymmetric spin-orbital legs.

    Store only ordered particle and hole lists in one spin and spatial sector.
    Each hole slot owns a contiguous segment from a shared virtual table.
    """

    def __init__(self, occupied, virtual, irrep, spin_change):
        self.occupied, self.virtual = np.asarray(occupied), np.asarray(virtual)
        self.no, self.nv = len(occupied), len(virtual)
        self.irrep, self.spin_change = irrep, spin_change
        groups = {}
        for particles in combinations(range(2 * self.nv), 4):
            key = alpha_count(particles), orbital_irrep(particles, virtual)
            groups.setdefault(key, []).append(particles)
        self.groups = {key: np.array(rows, dtype=np.int32) for key, rows in groups.items()}
        self.slots = {}
        self.size = 0
        self._virtual_maps = {}
        self._cpu_signed_maps = {}
        self.native_expansion = True
        for holes in combinations(range(2 * self.no), 4):
            key = alpha_count(holes) + spin_change, orbital_irrep(holes, occupied) ^ irrep
            if key in self.groups:
                size = len(self.groups[key])
                self.slots[holes] = (key, slice(self.size, self.size + size))
                self.size += size

    def share_virtual_maps(self, other):
        """Share immutable virtual address maps across ground/direction sectors."""
        if not np.array_equal(self.virtual, other.virtual):
            raise ValueError("Shared address maps require identical virtual orbital labels")
        self._virtual_maps = other._virtual_maps
        self._cpu_signed_maps = other._cpu_signed_maps

    def _virtual_map(self, signature, key, xp):
        cache_key = signature, key, xp
        if cache_key in self._virtual_maps:
            return self._virtual_maps[cache_key]
        canonical = self.groups[key]
        # Build all placements that match this spin order. There are at most
        # 24, even though the tensor has nv**4 entries.
        expected = np.array(signature, dtype=np.int32)
        indices = []
        for order, sign in signed_permutations(4):
            permuted = canonical[:, order]
            selected = np.flatnonzero(np.all(permuted % 2 == expected, axis=1))
            if selected.size:
                position = np.ravel_multi_index((permuted[selected] // 2).T, (self.nv,) * 4)
                indices.append((xp.asarray(selected, dtype=np.int32),
                                xp.asarray(position, dtype=np.int32), sign))
        self._virtual_maps[cache_key] = tuple(indices)
        return self._virtual_maps[cache_key]

    def signed_rows(self, vector, signature, cuts, xp):
        """The occupied tile of `expand` as `SignedRows` (rows not gathered),
        or None without the compiled core or for a non-NumPy vector."""
        if not (self.native_expansion and xp is np and isinstance(vector, np.ndarray)
                and vector.dtype == np.float64 and vector.ndim == 1 and vector.flags.c_contiguous):
            return None
        from ._cpu_kernels import library
        if library() is None:
            return None
        from ._tensor_scatter import SignedRows
        lengths = tuple(len(range(*cut.indices(self.no))) for cut in cuts)
        bases, maps = [], []
        for labels in product(*(range(*cut.indices(self.no)) for cut in cuts)):
            holes = tuple(2 * i + spin for i, spin in zip(labels, signature[4:]))
            item = self.slots.get(tuple(sorted(holes))) if len(set(holes)) == 4 else None
            if item is None:
                bases.append(0)
                maps.append(None)
                continue
            key, window = item
            cache_key = signature[:4], key
            if cache_key not in self._cpu_signed_maps:
                mapping = np.zeros(self.nv**4, dtype=np.int32)
                for source, positions, sign in self._virtual_map(signature[:4], key, np):
                    mapping[positions] = sign * (source + 1)
                self._cpu_signed_maps[cache_key] = mapping
            bases.append(parity(holes) * (window.start + 1))
            maps.append(self._cpu_signed_maps[cache_key])
        return SignedRows(vector, bases, maps, self.nv**4, lengths, (self.nv,) * 4, (4, 5, 6, 7, 0, 1, 2, 3))

    def expand(self, vector, signature, cuts, xp):
        """Expand only the occupied tile requested by one contraction."""
        lengths = tuple(len(range(*cut.indices(self.no))) for cut in cuts)
        rows = self.signed_rows(vector, signature, cuts, xp)
        if rows is not None:
            return rows.dense()
        output = xp.zeros(lengths + (self.nv**4,), dtype=vector.dtype)
        occupied = (range(*cut.indices(self.no)) for cut in cuts)
        for row, labels in enumerate(product(*occupied)):
            holes = tuple(2 * i + spin for i, spin in zip(labels, signature[4:]))
            if len(set(holes)) != 4:
                continue
            sorted_holes = tuple(sorted(holes))
            item = self.slots.get(sorted_holes)
            if item is None:
                continue
            key, window = item
            values = vector[window]
            target = output.reshape(-1, self.nv**4)[row]
            phase = parity(holes)
            for source, target_positions, sign in self._virtual_map(signature[:4], key, xp):
                target[target_positions] = (sign * phase) * values[source]
        axes = tuple(range(4, 8)) + tuple(range(4))
        return output.reshape(lengths + (self.nv,) * 4).transpose(axes)

    def from_restricted(self, solver, amplitudes, xp):
        """Convert a restricted checkpoint directly into canonical spin slots."""
        if amplitudes.shape == solver.layout.dense_shape:
            packed = amplitudes
        elif not solver.layout.symmetric:
            packed = amplitudes.reshape(solver.layout.dense_shape)
        else:
            packed = solver.quadruples_to_dense(amplitudes)
        spatial_slots = {slot: row for row, slot in enumerate(solver.layout.indices)}
        result = xp.empty(self.size, dtype=np.float64)
        for holes, (key, window) in self.slots.items():
            tile = packed[spatial_slots[tuple(i // 2 for i in holes)]]
            if hasattr(tile, "get"):
                tile = tile.get()
            flat = tile.reshape(-1)
            spins = tuple(h % 2 for h in holes)
            values = np.zeros(len(self.groups[key]))
            # The expansion map is also the spin-lift gather map.
            for source, positions, sign in self._virtual_map(spins, key, np):
                values[source] += sign * flat[positions]
            result[window] = xp.asarray(values)
        return result


class PackedSpinProjector:
    """Apply the M_S=1 triplet projector without per-determinant spin maps."""

    def __init__(self, source, xp):
        if source.spin_change != 1:
            raise ValueError("The triplet projector expects M_S=1")
        self.xp = xp
        raised = PackedSpinQuadruples(source.occupied, source.virtual, source.irrep, 2)
        self.raised_size = raised.size
        self.routes = []
        virtual_routes = {}
        base = 2 * source.nv
        for key, particles in source.groups.items():
            target = raised.groups.get((key[0] + 1, key[1]))
            if target is None:
                continue
            target_ids = np.ravel_multi_index(target.T, (base,) * 4)
            maps = []
            for column in range(4):
                allowed = particles[:, column] % 2 == 1
                if column:
                    allowed &= particles[:, column - 1] != particles[:, column] - 1
                rows = np.flatnonzero(allowed)
                if rows.size:
                    flipped = particles[rows].copy()
                    flipped[:, column] -= 1
                    ids = np.ravel_multi_index(flipped.T, (base,) * 4)
                    columns = np.searchsorted(target_ids, ids)
                    maps.append((xp.asarray(rows, dtype=xp.int32), xp.asarray(columns, dtype=xp.int32)))
            virtual_routes[key] = maps

        for holes, (key, window) in source.slots.items():
            if holes in raised.slots:
                _, target = raised.slots[holes]
                self.routes.extend((window, target, rows, columns, 1)
                                   for rows, columns in virtual_routes.get(key, ()))
            for at, hole in enumerate(holes):
                if hole % 2 == 0 and hole + 1 not in holes:
                    flipped = holes[:at] + (hole + 1,) + holes[at + 1:]
                    _, target = raised.slots[flipped]
                    # Flipping an occupied spin changes the hole-order phase.
                    self.routes.append((window, target, slice(None), slice(None), -1))

    def project(self, vector):
        if not self.routes:
            return vector
        result = vector.copy()
        raised = self.xp.empty(self.raised_size, dtype=vector.dtype)
        # S- S+ has eigenvalues 0, 4, 10 and 18 in the M_S=1 rank-four space.
        for eigenvalue in (4., 10., 18.):
            raised.fill(0)
            for source, target, rows, columns, sign in self.routes:
                raised[target][columns] += sign * result[source][rows]
            for source, target, rows, columns, sign in self.routes:
                result[source][rows] -= sign / eigenvalue * raised[target][columns]
        return result
