"""Real molecular Hamiltonians with occupied orbitals first."""

from dataclasses import dataclass
from itertools import product

import numpy as np


def _real64(value):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError("Expected real numerical data")
    return np.asarray(array, dtype=np.float64, order="C")


@dataclass(frozen=True)
class Hamiltonian:
    """One-body h and antisymmetrized <pq||rs> in physicists' spin order.

    The Hamiltonian uses g[p,q,r,s] with operators a†_p a†_q a_s a_r.
    The constant includes nuclear repulsion and any frozen-core energy.
    """

    h: np.ndarray
    g: np.ndarray
    nocc: int
    constant: float = 0.0

    def __post_init__(self):
        h, g = _real64(self.h), _real64(self.g)
        if h.ndim != 2 or h.shape[0] != h.shape[1] or g.shape != (h.shape[0],) * 4:
            raise ValueError("Incompatible h and g shapes")
        if not isinstance(self.nocc, (int, np.integer)) or not 0 < self.nocc < len(h):
            raise ValueError("Occupied and virtual spin spaces must both be nonempty")
        if not (np.isfinite(h).all() and np.isfinite(g).all() and np.isfinite(self.constant)):
            raise ValueError("Hamiltonian contains nonfinite data")
        object.__setattr__(self, "h", h)
        object.__setattr__(self, "g", g)

    @classmethod
    def restricted(cls, h, eri, nocc, constant=0.0):
        """Lift closed-shell spatial h and chemists' (pq|rs) ERIs to spin tensors.

        Within each spatial orbital the spin order is alpha then beta. Inputs
        are in an orthonormal MO basis. Fold frozen core into h and constant
        before calling. This dense backend is intended for modest MO spaces.
        """
        h, eri = _real64(h), _real64(eri)
        if h.ndim != 2 or h.shape[0] != h.shape[1] or eri.shape != (h.shape[0],) * 4:
            raise ValueError("Incompatible spatial h and ERI shapes")
        if not isinstance(nocc, (int, np.integer)) or not 0 < nocc < len(h):
            raise ValueError("Expected nonempty occupied and virtual spatial spaces")
        n = len(h)
        g = np.zeros((2 * n,) * 4)
        coulomb = eri.transpose(0, 2, 1, 3)
        exchange = coulomb.swapaxes(2, 3)
        spin = [slice(0, None, 2), slice(1, None, 2)]
        for p, q in product(spin, repeat=2):
            g[p, q, p, q] += coulomb
            g[p, q, q, p] -= exchange
        return cls(np.kron(h, np.eye(2)), g, 2 * nocc, constant)

    def fock(self):
        o = slice(self.nocc)
        return self.h + np.einsum("piqi->pq", self.g[:, o, :, o])

    def reference_energy(self):
        o = slice(self.nocc)
        return float(self.constant + np.trace(self.h[o, o])
                     + 0.5 * np.einsum("ijij->", self.g[o, o, o, o]))

