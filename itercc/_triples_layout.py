"""Occupied-triangular storage and permutation views of restricted triples."""

from itertools import combinations_with_replacement

import numpy as np

from ._restricted_equations import spin_sum


class TriplesLayout:
    """Store T[i,j,k,a,b,c] for i <= j <= k, with i == j == k identically zero."""

    def __init__(self, no, nv):
        self.no, self.nv = no, nv
        self.indices = tuple(t for t in combinations_with_replacement(range(no), 3)
                             if t[0] != t[2])
        self.shape = (len(self.indices), nv, nv, nv)
        self.slots = np.full((no,) * 3, -1, dtype=np.intp)
        self.to_sorted = np.zeros((no,) * 3 + (3,), dtype=np.intp)
        lookup = {triple: slot for slot, triple in enumerate(self.indices)}
        for indices in np.ndindex((no,) * 3):
            perm = tuple(sorted(range(3), key=indices.__getitem__))
            self.to_sorted[indices] = perm
            self.slots[indices] = lookup.get(tuple(indices[i] for i in perm), -1)
        self.from_sorted = np.argsort(self.to_sorted, axis=-1)

    def plane(self, data, fixed, *, axis, xp=np):
        """Copy only the two-occupied-index plane requested by a contraction."""
        view = xp.empty((self.no, self.no, self.nv, self.nv, self.nv))
        for i, j in np.ndindex(self.no, self.no):
            indices = (fixed, i, j) if axis == 0 else (i, j, fixed)
            slot = self.slots[indices]
            if slot < 0:
                view[i, j].fill(0.0)
            else:
                view[i, j] = data[slot].transpose(self.from_sorted[indices])
        return view

    spin_sum = staticmethod(spin_sum)

    @staticmethod
    def feedback_sum(plane):
        return 2 * plane - plane.swapaxes(-2, -1)


    def spin_plane(self, data, fixed, *, xp=np):
        return self.spin_sum(self.plane(data, fixed, axis=0, xp=xp))

    @staticmethod
    def subtract_crossed(seed, tile):
        """Accumulate the two crossed particle-hole diagrams, consuming tile."""
        seed -= tile.swapaxes(1, 2)
        tile *= 0.5
        seed -= tile

    def add_seed(self, target, seed, j, k):
        """Fold the iabc seed into canonical R3, with equal-occupied permutations."""
        for i, tile in enumerate(seed):
            indices = (i, j, k)
            slot = self.slots[indices]
            if slot < 0:
                continue
            tile = tile.transpose(self.to_sorted[indices])
            target[slot] += tile
            # With equal occupied labels there are two virtual permutations
            # for each distinct ordered triple. The all-equal slot is omitted.
            a, b, c = self.indices[slot]
            if a == b:
                target[slot] += tile.swapaxes(0, 1)
            elif b == c:
                target[slot] += tile.swapaxes(1, 2)

    def purify(self, triples, *, xp=np):
        """Remove the null fully symmetric virtual component of each triple."""
        if not len(triples):
            return
        if (xp is np and isinstance(triples, np.ndarray) and triples.dtype == np.float64
                and triples.flags.c_contiguous):
            from . import _cpu_kernels
            from ._cpu_triples import cpu_threads

            library = _cpu_kernels.library()
            if library is not None:     # the same projection on all threads
                library.purify(triples.ctypes.data, len(triples), self.nv, cpu_threads(None))
                return
        diag = xp.arange(self.nv)
        batch = 1 if xp is np else 16
        for start in range(0, len(triples), batch):
            tiles = triples[start : start + batch]
            symmetric = tiles + tiles.swapaxes(-3, -2)
            symmetric += tiles.swapaxes(-3, -1)
            symmetric += symmetric.swapaxes(-2, -1)
            tiles -= symmetric / 6
            tiles[:, diag, diag, diag] = 0.0


class NoTriples(TriplesLayout):
    """A layout without triples slots, for restricted CCSD (T3 of shape (0, nv, nv, nv))."""

    def __init__(self, no, nv):
        super().__init__(no, nv)
        self.indices = ()
        self.shape = (0, nv, nv, nv)
        self.slots = np.full((no,) * 3, -1, dtype=np.intp)
