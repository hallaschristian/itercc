"""Audit roundoff in a real spatial orbital gauge with Abelian labels."""

from dataclasses import dataclass

import numpy as np

from ._triples_layout import TriplesLayout
from .hamiltonian import _real64


@dataclass(frozen=True)
class SymmetryLeakage:
    count: int
    maximum: float
    norm: float


def orbital_labels(labels, n):
    labels = np.asarray(labels)
    if (labels.shape != (n,) or labels.dtype.kind not in "iu"
            or np.any(labels < 0) or np.any(labels > 7)):
        raise ValueError("Provide one Abelian label 0..7 per orbital for symmetry blocks")
    return labels


def _project(array, tiles, tolerance):
    """Check all discarded entries and leave the input unchanged."""
    result = array.copy()
    count, maximum, squared = 0, 0.0, 0.0
    for index, forbidden in tiles:
        tile = result[index]
        removed = tile[forbidden]
        magnitude = float(np.max(np.abs(removed), initial=0))
        if not np.isfinite(magnitude) or magnitude > tolerance:
            raise ValueError(f"Symmetry leakage {magnitude:.6g} exceeds {tolerance:.6g}")
        count += int(np.count_nonzero(removed))
        maximum = max(maximum, magnitude)
        squared += float(removed @ removed)
        tile[forbidden] = 0
    return result, SymmetryLeakage(count, maximum, float(np.sqrt(squared)))


def _check_tolerance(tolerance):
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Expected a finite nonnegative symmetry tolerance")


def _eri_forbidden(pairs):
    return ((i, row[:, None, None] != pairs) for i, row in enumerate(pairs))


def has_integral_leakage(h, eri, labels, *, threads=1):
    """Test the exact symmetry of validated integrals without copying the ERIs."""
    if not np.any(labels):
        return False
    pairs = labels[:, None] ^ labels
    if np.any(h[pairs != 0]):
        return True
    from . import _cpu_kernels

    library = _cpu_kernels.library()
    if library is not None and eri.flags.c_contiguous:
        # An ERI plane (q,r,s) must have the same XOR as its first index.
        # This is the same layout test used for packed triples.
        labels64 = np.ascontiguousarray(labels, dtype=np.int64)
        address = labels64.ctypes.data
        return library.t3_leakage(eri.ctypes.data, len(labels), len(labels),
                                  address, address, threads) != 0.0
    return any(np.any(eri[i][mask]) for i, mask in _eri_forbidden(pairs))


def project_integrals(h, eri, labels, *, tolerance):
    """Copy and audit only symmetry-forbidden one- and two-electron entries."""
    _check_tolerance(tolerance)
    h, eri = _real64(h), _real64(eri)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or eri.shape != (len(h),) * 4:
        raise ValueError("Incompatible spatial h and ERI shapes")
    n = len(h)
    labels = orbital_labels(labels, n)
    pairs = labels[:, None] ^ labels
    h, h_audit = _project(h, [(slice(None), pairs != 0)], tolerance)
    eri, eri_audit = _project_eri(eri, labels, tolerance)
    return h, eri, (h_audit, eri_audit)


def _project_eri(eri, labels, tolerance):
    """`_project` of the ERIs: in one pass on all threads with the compiled core
    (NumPy's masked copies moved the n^4 tensor several times on one thread).
    Integrals without a nonzero forbidden entry (`itercc.integrals.mo_eri`) are
    audited and returned as they are, without a copy."""
    from . import _cpu_kernels

    library = _cpu_kernels.library()
    if library is None or eri.dtype != np.float64 or not eri.flags.c_contiguous:
        return _project(eri, _eri_forbidden(labels[:, None] ^ labels), tolerance)
    from ._cpu_triples import cpu_threads

    stats = np.zeros(3)
    labels64 = np.ascontiguousarray(labels, dtype=np.int64)
    if library.project_symmetric(None, eri.ctypes.data, len(labels), labels64.ctypes.data,
                                 float(tolerance), stats.ctypes.data, cpu_threads(None)):
        raise ValueError(f"Symmetry leakage {stats[1]:.6g} exceeds {tolerance:.6g}")
    if stats[0] == 0:
        return eri, SymmetryLeakage(0, 0.0, 0.0)
    out = np.empty_like(eri)
    library.project_symmetric(out.ctypes.data, eri.ctypes.data, len(labels), labels64.ctypes.data,
                              float(tolerance), stats.ctypes.data, cpu_threads(None))
    return out, SymmetryLeakage(int(stats[0]), float(stats[1]), float(stats[2]))


def project_amplitudes(amplitudes, labels, *, tolerance):
    """Copy and audit forbidden T1/T2(/T3) entries in the packed solver layout
    (CCSD amplitudes: T1 and T2 only)."""
    _check_tolerance(tolerance)
    t1, t2, *rest = map(_real64, amplitudes)
    no, nv = t1.shape
    layout = TriplesLayout(no, nv)
    if t2.shape != (no, no, nv, nv) or len(rest) > 1 or (rest and rest[0].shape != layout.shape):
        raise ValueError("Incompatible restricted amplitude shapes")
    labels = orbital_labels(labels, no + nv)
    occupied, virtual = labels[:no], labels[no:]
    occ2 = occupied[:, None] ^ occupied
    vir2 = virtual[:, None] ^ virtual
    vir3 = vir2[:, :, None] ^ virtual
    t1, audit1 = _project(t1, [(slice(None), occupied[:, None] != virtual)], tolerance)
    t2, audit2 = _project(t2, ((i, occ2[i, :, None, None] != vir2) for i in range(no)),
                          tolerance)
    if not rest:
        return (t1, t2), (audit1, audit2)
    t3, audit3 = _project(rest[0], ((slot, (occ2[i, j] ^ occupied[k]) != vir3)
                                    for slot, (i, j, k) in enumerate(layout.indices)), tolerance)
    return (t1, t2, t3), (audit1, audit2, audit3)

def operator_irreps(matrix, labels, *, tolerance):
    """Abelian irreps carried by the entries of a one-electron operator above tolerance."""
    matrix = np.asarray(matrix)
    labels = orbital_labels(labels, len(matrix))
    if matrix.shape != (len(labels),) * 2:
        raise ValueError("Expected a square one-electron operator")
    pairs = labels[:, None] ^ labels
    return sorted({int(irrep) for irrep in pairs[np.abs(matrix) > tolerance]})


def conserved_labels(labels, broken):
    """Labels of the subgroup under which every irrep in `broken` is totally symmetric.

    A perturbation of irrep p mixes orbitals whose labels differ by p. Each
    nonzero p is divided out with the homomorphism g -> g ^ p if g shares the
    highest bit of p, which keeps XOR products and sends p to the identity.
    For a field along x in C2v (B1 = 2) this gives the Cs labels A' = 0, A'' = 1.
    """
    labels = np.array(labels, dtype=np.int64)
    pending = [int(p) for p in broken]
    while pending:
        p = pending.pop()
        if p == 0:
            continue
        bit = 1 << (p.bit_length() - 1)
        labels = np.where(labels & bit, labels ^ p, labels)
        pending = [q ^ p if q & bit else q for q in pending]
    return labels
