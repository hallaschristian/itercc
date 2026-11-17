"""Abelian point-group symmetry of a `molecule.Molecule`: the subgroup of D2h
whose elements lie along the coordinate axes, its irreps as XOR-combinable
ids, and a symmetry-adapted (SO) basis of the spherical AOs.

An element of D2h acts on coordinates as a sign triple (sx, sy, sz): E, the
three C2 rotations, inversion and the three mirror planes. A real solid
harmonic has definite parity under each axis flip, so an AO maps onto the
same function on the image atom times a sign. The irreps of the found group
(of order 2^k) are labelled by the bit mask of the generators under which
their character is -1, so the product of two irreps is the XOR of their ids,
as the solvers' orbital labels require.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np

from .basis import cart2sph, cart_components, nsph

_OPERATIONS = [s for s in product((1, -1), repeat=3)]          # (sx, sy, sz), identity first
_NAMES = {(1, 1, 1): "E", (-1, -1, 1): "C2z", (-1, 1, -1): "C2y", (1, -1, -1): "C2x", (-1, -1, -1): "i",
          (1, 1, -1): "sxy", (1, -1, 1): "sxz", (-1, 1, 1): "syz"}


@dataclass(frozen=True)
class Symmetry:
    group: str                      # D2h subgroup name
    operations: tuple               # sign triples of the group
    generators: tuple               # sign triples; irrep id bit k: character -1 under generator k
    so_basis: list                  # [(irrep id, AO x n_irrep orthonormal coefficients)]
    nirrep: int


def _group_name(ops):
    names = {_NAMES[o] for o in ops}
    table = {frozenset({"E"}): "C1", frozenset({"E", "i"}): "Ci", frozenset({"E", "C2z"}): "C2",
             frozenset({"E", "C2y"}): "C2", frozenset({"E", "C2x"}): "C2", frozenset({"E", "sxy"}): "Cs",
             frozenset({"E", "sxz"}): "Cs", frozenset({"E", "syz"}): "Cs"}
    if frozenset(names) in table:
        return table[frozenset(names)]
    if len(names) == 8:
        return "D2h"
    if "i" in names:
        return "C2h"
    if sum(n.startswith("C2") for n in names) == 3:
        return "D2"
    return "C2v"


def _atom_map(coords, keys, op, tol):
    """image[a] = the atom at op(coords[a]) with the same key, or None."""
    image = []
    moved = coords * np.array(op, dtype=np.float64)
    for a in range(len(coords)):
        match = [b for b in range(len(coords)) if keys[b] == keys[a] and np.linalg.norm(coords[b] - moved[a]) < tol]
        if len(match) != 1:
            return None
        image.append(match[0])
    return image


def _parities(l):
    """(px, py, pz) parities of each spherical function of angular momentum l."""
    c2s = cart2sph(l)
    comps = cart_components(l)
    out = []
    for m in range(nsph(l)):
        rows = np.nonzero(np.abs(c2s[:, m]) > 1e-14)[0]
        lx, ly, lz = comps[rows[0]]
        out.append((lx % 2, ly % 2, lz % 2))
    return out


def detect(molecule, tol=1e-5):
    """The `Symmetry` of a molecule as positioned (centered on its nuclear charge)."""
    coords = molecule.coords
    basis = molecule.basis
    # Atoms are equivalent when element and shells agree.
    shells_of = [[] for _ in range(len(coords))]
    for shell in basis.shells:
        shells_of[shell.atom].append((shell.l, tuple(np.round(shell.exponents, 12)),
                                      tuple(np.round(shell.coefficients, 12))))
    keys = [(molecule.charges[a], tuple(shells_of[a])) for a in range(len(coords))]
    ops, maps = [], []
    for op in _OPERATIONS:
        image = _atom_map(coords, keys, op, tol)
        if image is not None:
            ops.append(op)
            maps.append(image)
    # Generators: greedily add elements not yet generated.
    def compose(x, y):
        return tuple(a * b for a, b in zip(x, y))

    generators, generated = [], {(1, 1, 1)}
    for op in ops:
        if op not in generated:
            generators.append(op)
            generated |= {compose(op, g) for g in generated}
    # Each group element as a bit vector over the generators.
    coordinates = {(1, 1, 1): 0}
    for k, g in enumerate(generators):
        for element, bits in list(coordinates.items()):
            coordinates[compose(element, g)] = bits | (1 << k)
    nirrep = 1 << len(generators)
    # AO bookkeeping: per atom, the AO offsets of its shells in order.
    offsets = basis.ao_offsets()
    atom_aos = [[] for _ in range(len(coords))]          # per atom: list of (start, l)
    for s, shell in enumerate(basis.shells):
        atom_aos[shell.atom].append((int(offsets[s]), shell.l))
    nao = basis.nao
    parity = {l: _parities(l) for l in {shell.l for shell in basis.shells}}
    vectors = [[] for _ in range(nirrep)]
    seen = [set() for _ in range(nirrep)]
    for atom in range(len(coords)):
        for local, (start, l) in enumerate(atom_aos[atom]):
            for m in range(nsph(l)):
                px, py, pz = parity[l][m]
                for irrep in range(nirrep):
                    v = np.zeros(nao)
                    for op, image in zip(ops, maps):
                        sign = (op[0] ** px) * (op[1] ** py) * (op[2] ** pz)
                        chi = -1 if bin(irrep & coordinates[op]).count("1") % 2 else 1
                        target_start, _ = atom_aos[image[atom]][local]
                        v[target_start + m] += chi * sign
                    norm = np.linalg.norm(v)
                    if norm < 1e-8:
                        continue
                    v /= norm
                    key = tuple(np.round(np.abs(v), 10)), tuple(np.nonzero(np.abs(v) > 1e-10)[0])
                    if key in seen[irrep]:
                        continue
                    seen[irrep].add(key)
                    vectors[irrep].append(v)
    so_basis = [(irrep, np.array(vectors[irrep]).T if vectors[irrep] else np.zeros((nao, 0)))
                for irrep in range(nirrep)]
    total = sum(u.shape[1] for _, u in so_basis)
    if total != nao:
        raise RuntimeError(f"symmetry-adapted basis has {total} functions for {nao} AOs")
    return Symmetry(_group_name(ops), tuple(ops), tuple(generators), so_basis, nirrep)
