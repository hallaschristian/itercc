"""Molecules and basis sets without PySCF: geometry, nuclear charges, Gaussian
basis sets read from NWChem-format files, and Abelian point-group symmetry.

Basis sets are read by name from `ITERCC_BASIS_PATH` (directories of
NWChem-format files, separated like PATH), then from the sets bundled in
`itercc/basis_sets` (STO-3G, Pople, cc-pVXZ, aug-cc-pVXZ, def2; see
THIRD_PARTY.md), then from the basis-set data of an installed PySCF (its files
only; no PySCF code runs). Effective core potentials are not supported: an
element whose basis comes with one is refused.
"""
import os
from pathlib import Path

import numpy as np

from .basis import Basis, Shell, normalized_coefficients
from . import _runtime

BOHR = 0.52917721092                 # Angstrom per Bohr (PySCF's value)

ELEMENTS = ["X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
            "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se",
            "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb",
            "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
            "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At",
            "Rn"]
_CHARGE = {symbol.upper(): z for z, symbol in enumerate(ELEMENTS)}
_ANGULAR = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5, "I": 6, "K": 7}


def _normalize_name(name):
    """PySCF's rule: lower case without spaces, dashes, underscores or
    parentheses, '*' as 's' (6-31G* -> 631gs)."""
    return "".join(c for c in name.lower().replace("*", "s") if c not in " -_(),")


class BasisSet(dict):
    """{element symbol: [(l, exponents, coefficient columns)]}; `ecp`: the
    elements the file gives an effective core potential."""
    ecp = frozenset()


def parse_nwchem(text):
    """`BasisSet` from NWChem basis-set text; SP blocks give an S and a P
    entry. ECP sections are skipped, their elements recorded."""
    out = {}
    current = None
    section = None
    ecp = set()
    for raw in text.splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        word = line.split()[0].upper()
        if word in ("BASIS", "ECP", "SO"):
            section, current = word, None
            continue
        if word == "END":
            section, current = None, None
            continue
        parts = line.split()
        if section in ("ECP", "SO"):
            if parts[0][0].isalpha():
                ecp.add(parts[0].capitalize())
            continue
        if parts[0][0].isalpha():
            symbol, kind = parts[0].capitalize(), parts[1].upper()
            current = (symbol, kind, [])
            out.setdefault(symbol, []).append(current)
            continue
        if current is None:
            continue
        current[2].append([float(x.replace("D", "E").replace("d", "e")) for x in parts])
    result = {}
    for symbol, blocks in out.items():
        entries = []
        for _, kind, rows in blocks:
            data = np.array(rows, dtype=np.float64)
            if kind == "SP":
                entries.append((0, data[:, 0], data[:, 1:2]))
                entries.append((1, data[:, 0], data[:, 2:3]))
            else:
                entries.append((_ANGULAR[kind], data[:, 0], data[:, 1:]))
        result[symbol] = entries
    parsed = BasisSet(result)
    parsed.ecp = frozenset(ecp)
    return parsed


def _basis_directories():
    dirs = [Path(p) for p in _runtime.switch("ITERCC_BASIS_PATH").split(os.pathsep) if p]
    dirs.append(Path(__file__).resolve().parent / "basis_sets")
    try:
        import importlib.util

        spec = importlib.util.find_spec("pyscf")
        if spec is not None and spec.submodule_search_locations:
            dirs.append(Path(list(spec.submodule_search_locations)[0]) / "gto" / "basis")
    except (ImportError, ValueError):
        pass
    return dirs


def load_basis(name):
    """The parsed basis set `name` (as `parse_nwchem`), from the first file
    whose normalized name matches in the basis directories (and one level of
    their subfolders)."""
    wanted = _normalize_name(name)
    for directory in _basis_directories():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*")) + sorted(directory.glob("*/*")):
            if path.suffix in (".dat", ".nw", ".txt", "") and _normalize_name(path.stem) == wanted and path.is_file():
                return parse_nwchem(path.read_text(encoding="utf-8", errors="replace"))
    raise FileNotFoundError(f"basis set {name!r} not found; set ITERCC_BASIS_PATH to a folder of NWChem-format files")


def parse_atoms(atom, unit="angstrom"):
    """[(symbol, (x, y, z) in Bohr)] from "O 0 0 0; H 0 1 0" or a list of
    (symbol, (x, y, z)) in `unit`."""
    scale = 1.0 / BOHR if unit.lower().startswith("a") else 1.0
    if isinstance(atom, str):
        entries = []
        for item in atom.replace("\n", ";").split(";"):
            parts = item.split()
            if parts:
                entries.append((parts[0], tuple(float(x) for x in parts[1:4])))
    else:
        entries = [(symbol, tuple(xyz)) for symbol, xyz in atom]
    out = []
    for symbol, xyz in entries:
        symbol = symbol.strip().capitalize()
        if symbol.upper() not in _CHARGE:
            raise ValueError(f"unknown element {symbol!r}")
        out.append((symbol, tuple(scale * float(x) for x in xyz)))
    return out


class Molecule:
    """A molecule with a Gaussian basis: `Molecule("O 0 0 0.117; H 0 0.757
    -0.469; H 0 -0.757 -0.469", "cc-pvdz")`. Coordinates are kept as given
    (Bohr internally); with `symmetry` the largest Abelian subgroup of D2h
    whose elements lie along the coordinate axes (after centering on the
    nuclear charge) is found, with a symmetry-adapted basis."""

    def __init__(self, atom, basis, *, unit="angstrom", charge=0, symmetry=True):
        self.atoms = parse_atoms(atom, unit)
        self.charge = int(charge)
        self.charges = np.array([_CHARGE[s.upper()] for s, _ in self.atoms], dtype=np.float64)
        coords = np.array([xyz for _, xyz in self.atoms], dtype=np.float64)
        if symmetry:
            coords = coords - self.charges @ coords / self.charges.sum()
        self.coords = coords
        self.basis_name = basis
        self.basis = self._build_basis(basis)
        self.nelectron = int(round(self.charges.sum())) - self.charge
        self.symmetry = None
        if symmetry:
            from .symmetry import detect

            self.symmetry = detect(self)

    def _build_basis(self, basis):
        tables = {}
        shells = []
        for index, (symbol, _) in enumerate(self.atoms):
            name = basis[symbol] if isinstance(basis, dict) else basis
            if name not in tables:
                tables[name] = load_basis(name)
            if symbol not in tables[name]:
                raise ValueError(f"basis {name!r} has no entry for {symbol}")
            if symbol in tables[name].ecp:
                raise NotImplementedError(f"basis {name!r} gives {symbol} an effective core potential, "
                                          "which IterCC does not support")
            for l, exps, columns in tables[name][symbol]:
                for k in range(columns.shape[1]):
                    keep = columns[:, k] != 0.0
                    shells.append(Shell(index, l, exps[keep].copy(),
                                        normalized_coefficients(l, exps[keep], columns[keep, k])))
        return Basis(shells, self.coords, self.charges)

    @property
    def nao(self):
        return self.basis.nao

    def energy_nuc(self):
        energy = 0.0
        for i in range(len(self.charges)):
            for j in range(i):
                energy += self.charges[i] * self.charges[j] / np.linalg.norm(self.coords[i] - self.coords[j])
        return float(energy)
