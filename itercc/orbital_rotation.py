"""Transport a restricted checkpoint between occupied/virtual orbital gauges."""

import numpy as np

from ._restricted_equations import contract
from ._triples_layout import TriplesLayout
from .hamiltonian import _real64


def rotate_amplitudes(amplitudes, occupied, virtual, *, batch=8):
    """Rotate T1/T2 and packed T3, with new orbitals equal to old @ U.

    Virtual columns may select an orthonormal subspace. Occupied orbitals
    are rotated within the same space because the reference is fixed.
    """
    uo, uv = _real64(occupied), _real64(virtual)
    if (any(u.ndim != 2 or not 0 < u.shape[1] <= u.shape[0]
            or not np.allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-11, rtol=0)
            for u in (uo, uv)) or uo.shape[0] != uo.shape[1]):
        raise ValueError("Expected an occupied rotation and orthonormal virtual columns")
    no, nv = len(uo), len(uv)
    layout = TriplesLayout(no, nv)
    t1, t2, t3 = tuple(_real64(a) for a in amplitudes)
    if not all(np.isfinite(array).all() for array in (t1, t2, t3)):
        raise ValueError("Expected finite amplitudes")
    if t1.shape != (no, nv) or t2.shape != (no, no, nv, nv) or t3.shape != layout.shape:
        raise ValueError("Amplitudes and rotation sizes do not match")
    if not isinstance(batch, (int, np.integer)) or batch < 1:
        raise ValueError("Expected a positive integer batch size")
    t1_new = uo.T @ t1 @ uv
    t2_new = contract("ijab,iI,jJ,aA,bB->IJAB", t2, uo, uo, uv, uv)
    output_layout = TriplesLayout(no, uv.shape[1])
    tv = np.empty(output_layout.shape)
    for start in range(0, len(t3), batch):
        tile = slice(start, start + batch)
        tv[tile] = contract("xabc,aA,bB,cC->xABC", t3[tile], uv, uv, uv)

    t3_new = np.zeros_like(tv)
    work = np.empty(output_layout.shape[1:])
    for source_k in range(no):
        # Expand just the two occupied indices needed for this contribution.
        plane = output_layout.plane(tv, source_k, axis=2)
        rotated = contract("li,mj,lmabc->ijabc", uo, uo, plane)
        del plane
        for output, (i, j, k) in zip(t3_new, layout.indices):
            np.multiply(rotated[i, j], uo[source_k, k], out=work)
            output += work
    return t1_new, t2_new, t3_new
