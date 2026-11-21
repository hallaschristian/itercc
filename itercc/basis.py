"""Gaussian basis sets: shells, normalization and real spherical harmonics.

The conventions are libcint's (and PySCF's), so integrals can be compared
entry by entry: a primitive of angular momentum l and exponent a carries the
radial normalization `gto_norm(l, a)`, contracted radial functions are
normalized, Cartesian components run xx, xy, xz, yy, yz, zz (x powers
descending, then y), and spherical functions are real solid harmonics in the
order m = -l .. l, except p functions, which keep x, y, z.
"""
from dataclasses import dataclass
from functools import lru_cache
from math import comb, factorial, gamma, pi, sqrt

import numpy as np


def cart_components(l):
    """(lx, ly, lz) of the Cartesian components of angular momentum l."""
    return [(lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)]


def ncart(l):
    return (l + 1) * (l + 2) // 2


def nsph(l):
    return 2 * l + 1


def gto_norm(l, exponent):
    """1 / sqrt(int r^(2l+2) exp(-2 a r^2) dr): the radial normalization."""
    return 1.0 / sqrt(gamma(l + 1.5) / (2.0 * (2.0 * exponent) ** (l + 1.5)))


def _double_factorial(n):
    return 1 if n <= 0 else n * _double_factorial(n - 2)


def _angular_overlap(a, b):
    """int (x^a / r^l)(x^b / r^l) dOmega for Cartesian exponent triples a, b."""
    e = [i + j for i, j in zip(a, b)]
    if any(x % 2 for x in e):
        return 0.0
    i, j, k = (x // 2 for x in e)
    return 4.0 * pi * _double_factorial(2 * i - 1) * _double_factorial(2 * j - 1) * _double_factorial(2 * k - 1) \
        / _double_factorial(2 * (i + j + k) + 1)


@lru_cache(maxsize=None)
def cart2sph(l):
    """(ncart x nsph) coefficients: spherical function m = column, as a
    combination of the Cartesian components, normalized over the sphere
    (libcint's c2s with its s and p factors 1/sqrt(4 pi), sqrt(3/(4 pi))
    included). Real solid harmonics after Helgaker, Joergensen and Olsen
    (6.4.47-6.4.50)."""
    comps = cart_components(l)
    index = {c: i for i, c in enumerate(comps)}
    if l == 1:                                   # x, y, z (libcint's order), normalized
        return np.eye(3) * sqrt(3.0 / (4.0 * pi))
    out = np.zeros((len(comps), 2 * l + 1))
    for column, m in enumerate(range(-l, l + 1)):
        am = abs(m)
        vm = 0.5 if m < 0 else 0.0
        coeff = {}
        for t in range((l - am) // 2 + 1):
            for u in range(t + 1):
                # v runs over vm, vm + 1, ... up to floor(am / 2 - vm) + vm
                v = vm
                while v <= (am // 2 if m >= 0 else (am - 1) // 2 + 0.5) + 1e-9:
                    two_v = int(round(2 * v))
                    c = (-1) ** (t + v - vm) * 0.25 ** t * comb(l, t) * comb(l - t, am + t) * comb(t, u) * comb(am, two_v)
                    lx = 2 * t + am - 2 * u - two_v
                    ly = 2 * u + two_v
                    lz = l - 2 * t - am
                    key = index[(lx, ly, lz)]
                    coeff[key] = coeff.get(key, 0.0) + c
                    v += 1.0
        vector = np.zeros(len(comps))
        for key, c in coeff.items():
            vector[key] = c
        norm = sqrt(sum(vector[i] * vector[j] * _angular_overlap(comps[i], comps[j])
                        for i in range(len(comps)) for j in range(len(comps))))
        out[:, column] = vector / norm
    return out


@dataclass(frozen=True)
class Shell:
    atom: int
    l: int
    exponents: np.ndarray        # (nprim,)
    coefficients: np.ndarray     # (nprim,) including gto_norm and the contraction normalization


def normalized_coefficients(l, exponents, coefficients):
    """Contraction coefficients times gto_norm, scaled so that the contracted
    radial function is normalized (PySCF's convention)."""
    exponents = np.asarray(exponents, dtype=np.float64)
    c = np.asarray(coefficients, dtype=np.float64) * np.array([gto_norm(l, a) for a in exponents])
    s = gamma(l + 1.5) / (2.0 * np.add.outer(exponents, exponents) ** (l + 1.5))
    return c / sqrt(float(c @ s @ c))


@dataclass
class Basis:
    """Shells on atoms (Bohr coordinates, nuclear charges), spherical functions."""
    shells: list
    coords: np.ndarray           # (natm, 3)
    charges: np.ndarray          # (natm,)

    @property
    def nao(self):
        return sum(nsph(s.l) for s in self.shells)

    def ao_offsets(self):
        offsets = np.zeros(len(self.shells) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([nsph(s.l) for s in self.shells])
        return offsets

    def decontracted(self):
        """(primitives, contraction): the uncontracted basis, one normalized
        primitive shell per distinct exponent of each atom and angular
        momentum (exponents equal to 9 decimals merged, steepest first, atoms
        in order), and the (primitive AO x AO) matrix whose columns are this
        basis's functions in it: AO = primitives @ contraction. PySCF's
        `mol.decontract_basis(aggregate=True)`, the basis X2C decouples in."""
        groups = {}
        for index, shell in enumerate(self.shells):
            groups.setdefault((shell.atom, shell.l), []).append(index)
        rows = {}                                  # (atom, l, rounded exponent) -> primitive shell
        primitives = []
        for (atom, l) in sorted(groups):
            exps = np.concatenate([self.shells[i].exponents for i in groups[(atom, l)]])
            _, first = np.unique(np.round(exps, 9), return_index=True)
            for exponent in sorted(exps[first], reverse=True):
                rows[(atom, l, float(np.round(exponent, 9)))] = len(primitives)
                primitives.append(Shell(atom, l, np.array([exponent]), np.array([gto_norm(l, exponent)])))
        basis = Basis(primitives, self.coords, self.charges)
        prim_offsets, offsets = basis.ao_offsets(), self.ao_offsets()
        contraction = np.zeros((basis.nao, self.nao))
        for index, shell in enumerate(self.shells):
            m = nsph(shell.l)
            for exponent, coefficient in zip(shell.exponents, shell.coefficients):
                row = rows[(shell.atom, shell.l, float(np.round(exponent, 9)))]
                value = coefficient / primitives[row].coefficients[0]
                for k in range(m):
                    contraction[prim_offsets[row] + k, offsets[index] + k] += value
        return basis, contraction

    @classmethod
    def from_pyscf(cls, mol):
        """The shells of a PySCF molecule (spherical basis), one per contraction."""
        if mol.cart:
            raise ValueError("expected a spherical PySCF basis")
        shells = []
        for b in range(mol.nbas):
            l, exps = int(mol.bas_angular(b)), np.asarray(mol.bas_exp(b))
            ctr = np.asarray(mol.bas_ctr_coeff(b))            # (nprim, nctr), normalized contractions
            for k in range(ctr.shape[1]):
                shells.append(Shell(int(mol.bas_atom(b)), l, exps.copy(), normalized_coefficients(l, exps, ctr[:, k])))
        return cls(shells, np.asarray(mol.atom_coords(), dtype=np.float64),
                   np.asarray(mol.atom_charges(), dtype=np.float64))
