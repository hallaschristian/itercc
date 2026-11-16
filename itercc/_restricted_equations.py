# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
# Adapted from the Apache-2.0 PySCF 2.14.0 spatial RCCSDT equations by
# Yu Jin and Huanchen Zhai. See LICENSE.pyscf, NOTICE.pyscf and THIRD_PARTY.md.
# The contractions use an occupied-triangular schedule; with the compiled CPU
# core the products run on its GEMM executor (`contract`, `add_product`).
"""T1-dressed restricted CCSDT contractions in occupied-first spatial order."""

from dataclasses import dataclass

import numpy as np


def contract(spec, *operands, xp=np):
    return xp.einsum(spec, *operands, optimize=True)


# A `contract` may also offer `into(spec, a, b, out, alpha)`, which adds
# alpha * einsum(spec, a, b) to `out` in place and returns False when it does
# not apply, plus `copy(x)` and `combine(a, b, wa, wb)` (wa a + wb b) on all
# threads: the CPU path passes `_cpu_triples.CompiledContract`, the GEMM
# executor of the CPU CCSDTQ solver. Without them (NumPy, CuPy, the dual
# numbers of EOM and response) each helper evaluates exactly the expression it
# stands for, so those paths keep their arithmetic.

def add_product(contract, out, spec, a, b, alpha=1.0):
    """out += alpha * einsum(spec, a, b); returns out (rebind it: a dual-number
    `out` may be replaced)."""
    into = getattr(contract, "into", None)
    if into is not None and into(spec, a, b, out, alpha):
        return out
    value = contract(spec, a, b)
    if alpha == 1.0:
        out += value
    elif alpha == -1.0:
        out -= value
    elif alpha < 0:
        out -= -alpha * value
    else:
        out += alpha * value
    return out


def plus_product(contract, base, spec, a, b, alpha=1.0):
    """base + alpha * einsum(spec, a, b) as a new array (base is not changed)."""
    into = getattr(contract, "into", None)
    if into is not None:
        out = contract.copy(base)
        if into(spec, a, b, out, alpha):
            return out
    value = contract(spec, a, b)
    if alpha == 1.0:
        return base + value
    if alpha == -1.0:
        return base - value
    if alpha < 0:
        return base - -alpha * value
    return base + alpha * value


def copy_of(contract, x):
    """A copy of x (on all threads with the compiled CPU path)."""
    copier = getattr(contract, "copy", None)
    return x.copy() if copier is None else copier(x)


def spin_pair(contract, x):
    """2 x - x.swapaxes(2, 3)."""
    combine = getattr(contract, "combine", None)
    if combine is not None:
        value = combine(x, x.swapaxes(2, 3), 2.0, -1.0)
        if value is not None:
            return value
    return 2 * x - x.swapaxes(2, 3)


@dataclass
class Intermediates:
    vv: np.ndarray
    oo: np.ndarray
    oooo: np.ndarray
    ovvo: np.ndarray
    ovov: np.ndarray


def lower_residuals(f, g, t2, o, v, contract=contract, vvvv_ladder=None):
    """Singles and unsymmetrized doubles before triples feedback.

    `vvvv_ladder(g_vvvv, t2)` may replace the particle-particle ladder
    0.5 sum_cd g(abcd) t2(ijcd) (the CPU path passes a pair-sector GEMM).
    """
    ov = f[o, v]
    oovv = g[o, o, v, v]
    spin_g = spin_pair(contract, oovv)
    c2 = spin_pair(contract, t2)
    vv = plus_product(contract, f[v, v], "kldc,kldb->bc", spin_g, t2, -1.0)
    oo = plus_product(contract, f[o, o], "lkcd,ljcd->kj", spin_g, t2)
    oooo = plus_product(contract, g[o, o, o, o], "klcd,ijcd->klij", oovv, t2)
    ovvo = 2 * g[o, v, v, o] - g[o, v, o, v].swapaxes(2, 3)
    ovvo = add_product(contract, ovvo, "mled,miea->ladi", spin_g, c2, 0.5)
    ovov = plus_product(contract, g[o, v, o, v], "mlde,imea->laid", oovv, t2, -0.5)
    r1 = f[v, o].T.copy()
    r1 = add_product(contract, r1, "kc,ikac->ia", ov, c2)
    r1 = add_product(contract, r1, "akcd,ikcd->ia", g[v, o, v, v], c2)
    r1 = add_product(contract, r1, "klic,klac->ia", g[o, o, o, v], c2, -1.0)
    r2 = 0.5 * g[v, v, o, o].transpose(2, 3, 0, 1)
    if isinstance(r2, np.ndarray):
        # C order: 0.5 * view keeps the view's transposed order, which every
        # update below would then write through, and the symmetry-blocked T3
        # feedback of the CPU path addresses r2 by rows.
        r2 = np.ascontiguousarray(r2)
    r2 = add_product(contract, r2, "bc,ijac->ijab", vv, t2)
    r2 = add_product(contract, r2, "kj,ikab->ijab", oo, t2, -1.0)
    if vvvv_ladder is None:
        r2 = add_product(contract, r2, "abcd,ijcd->ijab", g[v, v, v, v], t2, 0.5)
    else:
        r2 += vvvv_ladder(g[v, v, v, v], t2)
    r2 = add_product(contract, r2, "klij,klab->ijab", oooo, t2, 0.5)
    r2 = add_product(contract, r2, "kaci,kjcb->ijab", ovvo, c2, 0.5)
    r2 = add_product(contract, r2, "kaic,jkcb->ijab", ovov, t2, -0.5)
    r2 = add_product(contract, r2, "kbic,jkca->ijab", ovov, t2, -1.0)
    return r1, r2, Intermediates(vv, oo, oooo, ovvo, ovov), spin_g, c2


def triples_sources(f, g, t2, c2, o, v, contract=contract):
    """Construct W used for the T2-driven part of the triples residual."""
    oovo = g[o, o, v, o]
    ooov = g[o, o, o, v]
    ovvv = g[o, v, v, v]
    vooo = copy_of(contract, g[v, o, o, o])
    vooo = add_product(contract, vooo, "ld,ijad->alij", f[o, v], t2)
    vooo = add_product(contract, vooo, "mldj,mida->alij", oovo - 0.5 * ooov.swapaxes(2, 3), c2)
    vooo = add_product(contract, vooo, "mljd,imda->alij", ooov, t2, -0.5)
    vooo = add_product(contract, vooo, "mlid,jmda->alij", ooov, t2, -1.0)
    vooo = add_product(contract, vooo, "alde,ijde->alij", g[v, o, v, v], t2)
    vvvo = copy_of(contract, g[v, v, v, o])
    vvvo = add_product(contract, vvvo, "laed,ljeb->abdj", spin_pair(contract, ovvv), c2, 0.5)
    vvvo = add_product(contract, vvvo, "lade,jleb->abdj", ovvv, t2, -0.5)
    vvvo = add_product(contract, vvvo, "lbde,jlea->abdj", ovvv, t2, -1.0)
    vvvo = add_product(contract, vvvo, "lmdj,lmab->abdj", oovo, t2)
    return vooo, vvvo


def spin_sum(t3):
    """2 T(abc) - T(bac) - T(cba), with any leading occupied axes."""
    q = 2 * t3
    q -= t3.swapaxes(-3, -2)
    q -= t3.swapaxes(-3, -1)
    return q


def triples_feedback(r1, r2, vooo, vvvo, f, g, layout, t3, o, v, *,
                     contract=contract, xp=np, feedback=None):
    """Add all T3 terms to R1/R2 and to the triples sources."""
    no = layout.no
    for m in range(no):
        # Plane has fixed first occupied index. A single gathered plane is
        # shared by all five feedback contractions.
        q = layout.spin_plane(t3, m, xp=xp)
        p = layout.feedback_sum(q)
        r1 += 0.5 * contract("jbc,ijcab->ia", g[o, m, v, v], p)
        del p
        r2 += 0.5 * contract("c,ijcab->ijab", f[m, v], q)
        doubles = None if feedback is None else feedback.doubles(g[v, m, v, v], q, m)
        r2 += contract("bcd,ijdac->ijab", g[v, m, v, v], q) if doubles is None else doubles
        del doubles
        r2 -= contract("kjc,ikcab->ijab", g[o, m, o, v], q)
        oovv = g[o, m, v, v]
        vooo += contract("lde,ijead->alij", oovv, q)
        source = None if feedback is None else feedback.source(oovv, q, m)
        vvvo -= contract("lde,jleba->abdj", oovv, q) if source is None else source
        del source
        del q
