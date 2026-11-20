#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Yu Jin <yjin@flatironinstitute.org>


"""T1-dressed closed-shell contractions adapted from PySCF 2.14.0.

The contractions from rccsdt.py, rccsdt_highm.py and rccsdtq.py use
device-local occupied tiles and the independent IterCC driver. Arrays are
allocated with the solver's array module (`mycc.xp`, CuPy or NumPy), and
the kernels below run in the backend of their operands.
"""

import numpy as np

from . import _cpu_quadruples, _quadruples_cpu, _quadruples_cuda
from ._quadruples_cuda import (
    _accumulate_t4_, _unpack_t4_, _unpack_t4_triples_, prange,
)
from . import _runtime


# R4 reads W_vvoooo and W_vvvvoo one tile (occupied indices) at a time, and
# the T4 pass adds to them one occupied pair at a time: on the CPU they are
# stored with their occupied axes outermost (same index order).
_W_VVOOOO_ORDER = (3, 4, 5, 2, 0, 1)
_W_VVVVOO_ORDER = (4, 5, 0, 1, 2, 3)


def _empty_stored(xp, shape, order):
    """xp.empty(shape) stored with the axes `order` outermost on the CPU."""
    if xp is not np:
        return xp.empty(shape)
    return np.empty(tuple(shape[axis] for axis in order)).transpose(np.argsort(order))


def _kernels(array):
    """The kernel module for an array: NumPy arrays use the CPU backend (its
    compiled kernels when the core is available, else plain NumPy)."""
    if isinstance(array, np.ndarray):
        return _cpu_quadruples if _cpu_quadruples.available() else _quadruples_cpu
    return _quadruples_cuda


def einsum(spec, *operands, out=None, alpha=1.0, beta=0.0):
    """out = beta * out + alpha * einsum(spec, *operands)."""
    return _kernels(operands[0] if out is None else out).einsum(
        spec, *operands, out=out, alpha=alpha, beta=beta)


def t3_spin_summation(a, b, slots, nvir, pattern, alpha=1.0, beta=0.0):
    return _kernels(a).t3_spin_summation(a, b, slots, nvir, pattern, alpha, beta)


def t4_spin_summation(source, target, nvir, pattern, alpha=1.0, beta=0.0):
    return _kernels(source).t4_spin_summation(source, target, nvir, pattern, alpha, beta)


def _tile_einsum(mycc, bounds, **ranges):
    """Contraction for one R4 tile: symmetry-blocked when the solver has labels.

    `bounds` are (i0, i1, j0, j1, k0, k1, l0, l1); the tile letters i, j, k, l
    take the irreps of those occupied ranges, and each letter=(x0, x1) of
    `ranges` those of its range.
    """
    blocks = getattr(mycc, "_tile_blocks", None)
    if blocks is None:
        return einsum
    occupied = mycc.layout.occupied_irreps
    labels = {c: occupied[x0:x1] for c, x0, x1 in zip("ijkl", bounds[::2], bounds[1::2])}
    labels.update({c: occupied[x0:x1] for c, (x0, x1) in ranges.items()})

    def contract(spec, a, b, out=None, alpha=1.0, beta=0.0):
        return blocks.einsum(spec, a, b, out, alpha, beta, labels=labels)
    return contract


def _block_einsum(mycc):
    """Contraction for gathered T4 blocks: symmetry-blocked when the solver has labels.

    The returned function takes einsum(spec, a, b, out, alpha, beta, labels=...,
    irreps=...) with `labels` for letters bound to occupied subranges and
    `irreps` the total irreps of the operands.
    """
    blocks = getattr(mycc, "_block_blocks", None)

    def contract(spec, a, b, out=None, alpha=1.0, beta=0.0, labels=None, irreps=(0, 0)):
        if blocks is None:
            return einsum(spec, a, b, out=out, alpha=alpha, beta=beta)
        return blocks.einsum(spec, a, b, out, alpha, beta, labels=labels, irreps=irreps)
    return contract


def _t3_products(mycc, t3):
    """Contraction for the T3-level terms (compute_r3 and the T3 parts of the
    intermediates). On the CPU with orbital labels it is the symmetry-blocked
    executor (`_block_einsum`): those products otherwise ran on the one-irrep
    dense engine through every symmetry-forbidden block (7x slower for
    W_vvvv x T3 in C2v); products it cannot block run dense. The CUDA path
    keeps the dense products."""
    if isinstance(t3, np.ndarray) and getattr(mycc, "_block_blocks", None) is not None:
        return _block_einsum(mycc)
    return einsum


def intermediates_t1t2(mycc, imds, t2):
    '''Intermediates for the T1 and T2 residual equation.'''
    nocc = mycc.nocc

    t1_fock, t1_eris = imds.t1_fock, imds.t1_eris

    F_vv = t1_fock[nocc:, nocc:].copy()
    einsum('kldc,kldb->bc', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=F_vv, alpha=-2.0, beta=1.0)
    einsum('klcd,kldb->bc', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=F_vv, alpha=1.0, beta=1.0)
    F_oo = t1_fock[:nocc, :nocc].copy()
    einsum('lkcd,ljcd->kj', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=F_oo, alpha=2.0, beta=1.0)
    einsum('lkdc,ljcd->kj', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=F_oo, alpha=-1.0, beta=1.0)
    W_oooo = t1_eris[:nocc, :nocc, :nocc, :nocc].copy()
    einsum('klcd,ijcd->klij', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_oooo, alpha=1.0, beta=1.0)
    c_t2 = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
    W_ovvo = 2.0 * t1_eris[:nocc, nocc:, nocc:, :nocc] - t1_eris[:nocc, nocc:, :nocc, nocc:].transpose(0, 1, 3, 2)
    einsum('mled,miea->ladi', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t2, out=W_ovvo, alpha=1.0, beta=1.0)
    einsum('mlde,miea->ladi', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t2, out=W_ovvo, alpha=-0.5, beta=1.0)
    c_t2 = None
    W_ovov = t1_eris[:nocc, nocc:, :nocc, nocc:].copy()
    einsum('mlde,imea->laid', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_ovov, alpha=-0.5, beta=1.0)
    imds.F_vv, imds.F_oo, imds.W_oooo, imds.W_ovvo, imds.W_ovov = F_vv, F_oo, W_oooo, W_ovvo, W_ovov
    return imds


def compute_r1r2(mycc, imds, t2):
    '''Compute r1 and r2 without the contributions from T3 amplitudes.
    r2 will require a symmetry restoration step afterward.
    '''
    nocc = mycc.nocc

    t1_fock, t1_eris = imds.t1_fock, imds.t1_eris
    F_oo, F_vv, W_oooo, W_ovvo, W_ovov = imds.F_oo, imds.F_vv, imds.W_oooo, imds.W_ovvo, imds.W_ovov

    c_t2 = 2.0 * t2 - t2.transpose(0, 1, 3, 2)

    r1 = t1_fock[nocc:, :nocc].T.copy()
    einsum('kc,ikac->ia', t1_fock[:nocc, nocc:], c_t2, out=r1, alpha=1.0, beta=1.0)
    einsum('akcd,ikcd->ia', t1_eris[nocc:, :nocc, nocc:, nocc:], c_t2, out=r1, alpha=1.0, beta=1.0)
    einsum('klic,klac->ia', t1_eris[:nocc, :nocc, :nocc, nocc:], c_t2, out=r1, alpha=-1.0, beta=1.0)

    r2 = 0.5 * t1_eris[nocc:, nocc:, :nocc, :nocc].T
    einsum("bc,ijac->ijab", F_vv, t2, out=r2, alpha=1.0, beta=1.0)
    einsum("kj,ikab->ijab", F_oo, t2, out=r2, alpha=-1.0, beta=1.0)
    einsum("abcd,ijcd->ijab", t1_eris[nocc:, nocc:, nocc:, nocc:], t2, out=r2, alpha=0.5, beta=1.0)
    einsum("klij,klab->ijab", W_oooo, t2, out=r2, alpha=0.5, beta=1.0)
    einsum("kaci,kjcb->ijab", W_ovvo, c_t2, out=r2, alpha=0.5, beta=1.0)
    einsum("kaic,jkcb->ijab", W_ovov, t2, out=r2, alpha=-0.5, beta=1.0)
    einsum("kbic,jkca->ijab", W_ovov, t2, out=r2, alpha=-1.0, beta=1.0)
    c_t2 = None
    return r1, r2


def intermediates_t3(mycc, imds, t2):
    '''Intermediates for the T3 residual equation (excluding T3 contributions).'''
    nocc = mycc.nocc

    t1_fock, t1_eris = imds.t1_fock, imds.t1_eris

    W_vvvv = t1_eris[nocc:, nocc:, nocc:, nocc:].copy()
    einsum('lmde,lmab->abde', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=W_vvvv, alpha=1.0, beta=1.0)

    W_vooo = t1_eris[nocc:, :nocc, :nocc, :nocc].copy()
    einsum('ld,ijad->alij', t1_fock[:nocc, nocc:], t2, out=W_vooo, alpha=1.0, beta=1.0)
    c_t2 = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
    einsum('mldj,mida->alij', t1_eris[:nocc, :nocc, nocc:, :nocc], c_t2, out=W_vooo, alpha=1.0, beta=1.0)
    einsum('mljd,mida->alij', t1_eris[:nocc, :nocc, :nocc, nocc:], c_t2, out=W_vooo, alpha=-0.5, beta=1.0)
    einsum('mljd,imda->alij', t1_eris[:nocc, :nocc, :nocc, nocc:], t2, out=W_vooo, alpha=-0.5, beta=1.0)
    einsum('mlid,jmda->alij', t1_eris[:nocc, :nocc, :nocc, nocc:], t2, out=W_vooo, alpha=-1.0, beta=1.0)
    einsum('alde,ijde->alij', t1_eris[nocc:, :nocc, nocc:, nocc:], t2, out=W_vooo, alpha=1.0, beta=1.0)

    W_vvvo = t1_eris[nocc:, nocc:, nocc:, :nocc].copy()
    einsum('laed,ljeb->abdj', t1_eris[:nocc, nocc:, nocc:, nocc:], c_t2, out=W_vvvo, alpha=1.0, beta=1.0)
    einsum('lade,ljeb->abdj', t1_eris[:nocc, nocc:, nocc:, nocc:], c_t2, out=W_vvvo, alpha=-0.5, beta=1.0)
    einsum('lade,jleb->abdj', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_vvvo, alpha=-0.5, beta=1.0)
    einsum('lbde,jlea->abdj', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_vvvo, alpha=-1.0, beta=1.0)
    einsum('lmdj,lmab->abdj', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_vvvo, alpha=1.0, beta=1.0)

    einsum('mled,miea->ladi', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t2, out=imds.W_ovvo, alpha=1.0, beta=1.0)
    einsum('mlde,miea->ladi', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t2, out=imds.W_ovvo, alpha=-0.5, beta=1.0)
    c_t2 = None

    einsum('mlde,imea->laid', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=imds.W_ovov, alpha=-0.5, beta=1.0)
    imds.W_vooo, imds.W_vvvo, imds.W_vvvv = W_vooo, W_vvvo, W_vvvv
    return imds


def _spin_summed_t3(imds, t3):
    """P3_201 of T3, formed once per evaluation: r1r2_add_t3_ keeps it on
    `imds` for the T3 intermediates and r3, and intermediates_t4_tri, its
    last reader before the large T4 pass, drops it (compute_r4_tri forms it
    again then, so the peak memory is that of separate copies)."""
    kept = getattr(imds, "spin_summed_t3", None)
    return kept[1] if kept is not None and kept[0] is t3 else None


def r1r2_add_t3_(mycc, imds, r1, r2, t3):
    '''Add the T3 contributions to r1 and r2. T3 amplitudes are stored in full form.'''
    einsum = _t3_products(mycc, t3)
    nocc, nmo = mycc.nocc, mycc.nmo
    xp = mycc.xp
    nvir = nmo - nocc

    t1_fock, t1_eris = imds.t1_fock, imds.t1_eris

    c_t3 = xp.empty_like(t3)
    t3_spin_summation(t3, c_t3, nocc**3, nvir, "P3_422", 1.0, 0.0)
    einsum('jkbc,kijcab->ia', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t3, out=r1, alpha=0.5, beta=1.0)

    t3_spin_summation(t3, c_t3, nocc**3, nvir, "P3_201", 1.0, 0.0)
    imds.spin_summed_t3 = (t3, c_t3)
    einsum("kc,kijcab->ijab", t1_fock[:nocc, nocc:], c_t3, out=r2, alpha=0.5, beta=1.0)
    einsum("bkcd,kijdac->ijab", t1_eris[nocc:, :nocc, nocc:, nocc:], c_t3, out=r2, alpha=1.0, beta=1.0)
    einsum("kljc,likcab->ijab", t1_eris[:nocc, :nocc, :nocc, nocc:], c_t3, out=r2, alpha=-1.0, beta=1.0)
    c_t3 = None
    return r1, r2


def intermediates_t3_add_t3(mycc, imds, t3):
    '''Add the T3-dependent contributions to the T3 intermediates, with T3 stored in full form.'''
    einsum = _t3_products(mycc, t3)
    nocc, nmo = mycc.nocc, mycc.nmo
    xp = mycc.xp
    nvir = nmo - nocc

    t1_eris, W_vooo, W_vvvo = imds.t1_eris, imds.W_vooo, imds.W_vvvo

    c_t3 = _spin_summed_t3(imds, t3)
    if c_t3 is None:
        c_t3 = xp.empty_like(t3)
        t3_spin_summation(t3, c_t3, nocc**3, nvir, "P3_201", 1.0, 0.0)
    einsum('lmde,mijead->alij', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t3, out=W_vooo, alpha=1.0, beta=1.0)
    einsum('lmde,mjleba->abdj', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t3, out=W_vvvo, alpha=-1.0, beta=1.0)
    c_t3 = None
    return imds


def compute_r3(mycc, imds, t2, t3):
    '''Compute r3 with full T3 amplitudes; r3 is returned in full form as well.
    r3 will require a symmetry restoration step afterward.
    '''
    einsum = _t3_products(mycc, t3)

    nocc, nmo = mycc.nocc, mycc.nmo
    xp = mycc.xp
    nvir = nmo - nocc

    F_oo, F_vv = imds.F_oo, imds.F_vv
    W_oooo, W_ovvo, W_ovov, W_vvvv = imds.W_oooo, imds.W_ovvo, imds.W_ovov, imds.W_vvvv
    W_vooo, W_vvvo = imds.W_vooo, imds.W_vvvo

    c_t3 = _spin_summed_t3(imds, t3)
    if c_t3 is None:
        c_t3 = xp.empty_like(t3)
        t3_spin_summation(t3, c_t3, nocc**3, nvir, "P3_201", 1.0, 0.0)

    r3 = xp.empty_like(t3)
    einsum('abdj,ikdc->ijkabc', W_vvvo, t2, out=r3, alpha=1.0, beta=0.0)
    einsum('alij,lkbc->ijkabc', W_vooo, t2, out=r3, alpha=-1.0, beta=1.0)
    einsum('ad,ijkdbc->ijkabc', F_vv, t3, out=r3, alpha=0.5, beta=1.0)

    einsum('li,ljkabc->ijkabc', F_oo, t3, out=r3, alpha=-0.5, beta=1.0)
    einsum('ladi,ljkdbc->ijkabc', W_ovvo, c_t3, out=r3, alpha=0.25, beta=1.0)
    c_t3 = None

    einsum('laid,jlkdbc->ijkabc', W_ovov, t3, out=r3, alpha=-0.5, beta=1.0)
    einsum('lbid,jlkdac->ijkabc', W_ovov, t3, out=r3, alpha=-1.0, beta=1.0)

    einsum('lmij,lmkabc->ijkabc', W_oooo, t3, out=r3, alpha=0.5, beta=1.0)

    einsum('abde,ijkdec->ijkabc', W_vvvv, t3, out=r3, alpha=0.5, beta=1.0)
    return r3


def intermediates_t4_tri(mycc, imds, t2, t3):
    '''T2 and T3 parts of the intermediates for the T4 residual equation.
    In place modification of W_vvvo. `t4_into_lower_ranks_` adds the T4
    parts and `finish_intermediates_t4` symmetrizes.
    '''
    einsum = _t3_products(mycc, t3)
    nocc, nmo = mycc.nocc, mycc.nmo
    xp = mycc.xp
    nvir = nmo - nocc

    t1_fock, t1_eris = imds.t1_fock, imds.t1_eris
    W_vvvo, W_oooo, W_ovov, W_vvvv = imds.W_vvvo, imds.W_oooo, imds.W_ovov, imds.W_vvvv

    einsum('me,mjab->abej', t1_fock[:nocc, nocc:], t2, out=W_vvvo, alpha=-1.0, beta=1.0)

    W_oovvvo = xp.empty((nocc,) * 2 + (nvir,) * 3 + (nocc,))
    einsum('maef,jibf->ijeabm', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_oovvvo, alpha=2.0, beta=0.0)
    einsum('mafe,jibf->ijeabm', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_oovvvo, alpha=-1.0, beta=1.0)
    einsum('mnei,njab->ijeabm', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_oovvvo, alpha=-2.0, beta=1.0)
    einsum('nmei,njab->ijeabm', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_oovvvo, alpha=1.0, beta=1.0)
    c_t3 = _spin_summed_t3(imds, t3)
    imds.spin_summed_t3 = None
    if c_t3 is None:
        c_t3 = xp.empty_like(t3)
        t3_spin_summation(t3, c_t3, nocc**3, nvir, "P3_201", 1.0, 0.0)
    einsum('nmfe,nijfab->ijeabm', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t3, out=W_oovvvo, alpha=0.5, beta=1.0)
    einsum('mnfe,nijfab->ijeabm', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t3, out=W_oovvvo, alpha=-0.25, beta=1.0)
    c_t3 = None

    W_ovovvo = xp.empty((nocc, nvir, nocc, nvir, nvir, nocc))
    einsum('mafe,jibf->iejabm', t1_eris[:nocc, nocc:, nocc:, nocc:], t2, out=W_ovovvo, alpha=1.0, beta=0.0)
    einsum('mnie,njab->iejabm', t1_eris[:nocc, :nocc, :nocc, nocc:], t2, out=W_ovovvo, alpha=-1.0, beta=1.0)
    einsum('nmef,injfab->iejabm', t1_eris[:nocc, :nocc, nocc:, nocc:], t3, out=W_ovovvo, alpha=-0.5, beta=1.0)

    W_ooooov = xp.empty((nocc,) * 5 + (nvir,))
    einsum('mnek,ijae->kjinma', t1_eris[:nocc, :nocc, nocc:, :nocc], t2, out=W_ooooov, alpha=1.0, beta=0.0)
    einsum('mnef,ijkaef->kjinma', t1_eris[:nocc, :nocc, nocc:, nocc:], t3, out=W_ooooov, alpha=0.5, beta=1.0)
    _kernels(W_ooooov).add_transpose_(W_ooooov, (1, 0, 2, 4, 3, 5))

    W_vvoooo = _empty_stored(xp, (nvir,) * 2 + (nocc,) * 4, _W_VVOOOO_ORDER)
    einsum('amef,ijkebf->abmijk', t1_eris[nocc:, :nocc, nocc:, nocc:], t3, out=W_vvoooo, alpha=1.0, beta=0.0)
    tmp_ovvo = t1_eris[:nocc, nocc:, nocc:, :nocc].copy()
    c_t2 = 2.0 * t2 - t2.transpose(0, 1, 3, 2)
    einsum('nmfe,nifa->maei', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t2, out=tmp_ovvo, alpha=1.0, beta=1.0)
    einsum('mnfe,nifa->maei', t1_eris[:nocc, :nocc, nocc:, nocc:], c_t2, out=tmp_ovvo, alpha=-0.5, beta=1.0)
    c_t2 = None
    einsum('nmef,infa->maei', t1_eris[:nocc, :nocc, nocc:, nocc:], t2, out=tmp_ovvo, alpha=-0.5, beta=1.0)
    einsum('maei,jkbe->abmijk', tmp_ovvo, t2, out=W_vvoooo, alpha=1.0, beta=1.0)
    tmp_ovvo = None
    einsum('make,jibe->abmijk', W_ovov, t2, out=W_vvoooo, alpha=1.0, beta=1.0)
    einsum('mnki,njab->abmijk', W_oooo, t2, out=W_vvoooo, alpha=-0.5, beta=1.0)

    W_vvvvoo = _empty_stored(xp, (nvir,) * 4 + (nocc,) * 2, _W_VVVVOO_ORDER)
    einsum('abef,jkfc->abcejk', W_vvvv, t2, out=W_vvvvoo, alpha=0.5, beta=0.0)

    imds.W_oovvvo, imds.W_ovovvo, imds.W_ooooov = W_oovvvo, W_ovovvo, W_ooooov
    imds.W_vvoooo, imds.W_vvvvoo = W_vvoooo, W_vvvvoo
    return imds


def t4_into_lower_ranks_(mycc, imds, r2, r3, t4, *, budget=2 * 2**30):
    '''Add every T4 contribution to r2, r3, W_vvoooo and W_vvvvoo.

    All of them contract T4 over its first one or two occupied indices, so one
    pass over the blocks T4[m, n, p, q] (all m, n; one ordered pair p, q) feeds
    them all: the P4_442 spin sum gives r2[p, q], and the P4_201 spin sum (read
    as T4[m, i, j, k] with j, k = p, q) gives r3 and the W intermediates. The
    first index is split into chunks of at most `budget` bytes; every
    contraction sums over it. Both spin sums are gathered straight from the
    packed amplitudes (allowed entries only), so no plain block is unpacked.
    '''
    nocc, nvir = mycc.nocc, mycc.nvir
    xp = mycc.xp
    t1_fock, t1_eris = imds.t1_fock, imds.t1_eris
    W_vvoooo, W_vvvvoo = imds.W_vvoooo, imds.W_vvvvoo
    o, v = slice(None, nocc), slice(nocc, None)
    eris_oovv, eris_vovv = t1_eris[o, o, v, v], t1_eris[v, o, v, v]
    fock_ov = t1_fock[o, v]
    # ITERCC_LOWER_CHUNK (diagnostic) fixes the number of m per chunk.
    chunk = _runtime.integer("ITERCC_LOWER_CHUNK") or max(1, min(nocc, budget // (8 * nocc * nvir**4)))
    if hasattr(mycc.layout, "add_oooo_pairs") and mycc.layout.symmetric:
        # The compiled CPU layout: packed blocks and box-level plans, the
        # budget for a chunk's packed rows (no symmetry zeros: 1/N of the
        # dense size; at 13o/63v two m per chunk with B1-subgroup labels and
        # five with C2v, instead of one).
        from . import _cpu_packed
        from ._runtime import cpu_threads

        packed = 8 * nocc * int(max(mycc.layout._slot_sizes()))
        chunk = _runtime.integer("ITERCC_LOWER_CHUNK") or max(1, min(nocc, budget // packed))
        _cpu_packed.t4_into_lower_ranks(mycc.layout, r2, r3, t4, eris_oovv, eris_vovv, fock_ov,
                                        t1_eris[o, o, v, o], W_vvoooo, W_vvvvoo, chunk, cpu_threads(None),
                                        single=getattr(mycc, "precision", "double") == "single")
        return r2, r3
    # With blocked contractions (allowed entries only) the gathers skip the
    # symmetry zeros of the block; it starts at zero, so the rest stays finite.
    allowed = getattr(mycc, "_block_blocks", None) is not None
    summed = xp.zeros((chunk, nocc, 1, 1) + (nvir,) * 4, dtype=t4.dtype)
    contract = _block_einsum(mycc)
    occupied = mycc.layout.occupied_irreps
    for p in range(nocc):
        for q in range(nocc):
            for m0, m1 in prange(0, nocc, chunk):
                ring = summed[:m1 - m0]
                part = occupied[m0:m1]
                first, second = occupied[p:p + 1], occupied[q:q + 1]
                # Spin-summed blocks P(T4[m0:m1, :, p, q]) straight from the packed amplitudes.
                bounds = (m0, m1, 0, nocc, p, p + 1, q, q + 1)
                mycc.layout.gather(t4, ring, bounds, spin="P4_442_r2", allowed_only=allowed)
                contract('mnef,mnijefab->ijab', eris_oovv[m0:m1], ring, out=r2[p:p + 1, q:q + 1],
                         alpha=0.25, beta=1.0, labels=dict(m=part, i=first, j=second))
                mycc.layout.gather(t4, ring, bounds, spin="P4_201", allowed_only=allowed)
                pair = dict(m=part, j=first, k=second)
                contract('me,mijkeabc->ijkabc', fock_ov[m0:m1], ring,
                         out=r3[:, p:p + 1, q:q + 1], alpha=1.0 / 6.0, beta=1.0, labels=pair)
                contract('amef,mijkfebc->ijkabc', eris_vovv[:, m0:m1], ring,
                         out=r3[:, p:p + 1, q:q + 1], alpha=0.5, beta=1.0, labels=pair)
                contract('mjen,mijkeabc->inkabc', t1_eris[m0:m1, p:p + 1, v, o], ring,
                         out=r3[:, :, q:q + 1], alpha=-0.5, beta=1.0, labels=pair)
                pair = dict(n=part, j=first, k=second)
                contract('mnef,nijkfabe->abmijk', eris_oovv[:, m0:m1], ring,
                         out=W_vvoooo[..., p:p + 1, q:q + 1], alpha=0.5, beta=1.0, labels=pair)
                contract('inef,nijkfabc->abcejk', eris_oovv[:, m0:m1], ring,
                         out=W_vvvvoo[..., p:p + 1, q:q + 1], alpha=-0.5, beta=1.0, labels=pair)
    return r2, r3


def finish_intermediates_t4(imds):
    '''Symmetrize the T4-equation intermediates once all their parts are in.'''
    kernels = _kernels(imds.W_oovvvo)
    kernels.add_transpose_(imds.W_oovvvo, (1, 0, 2, 4, 3, 5))
    kernels.add_transpose_(imds.W_vvoooo, (1, 0, 2, 4, 3, 5))
    kernels.add_transpose_(imds.W_vvvvoo, (0, 2, 1, 3, 5, 4))
    return imds


def _pair_ordered(letters, index, output):
    """T3 letters and occupied index with the (occupied, virtual) pairs
    reordered: summed virtuals first, then the free ones in `output` order.
    T3 is symmetric under permutations of its pairs (to rounding, from
    `project_triples`), and in this order a pair-blocked copy serves the
    product with contiguous blocks (`_cpu_tile_program`)."""
    pairs = sorted(zip(letters[:3], letters[3:], index), key=lambda p: (p[1] in output, output.find(p[1])))
    return "".join(p[0] for p in pairs) + "".join(p[1] for p in pairs), tuple(p[2] for p in pairs)


def _t3_term(tile, t3, pair_order):
    """tile(spec, w, t3[index], ...) of a T3 term, with `_pair_ordered` T3
    pairs when `pair_order`."""
    def term(spec, w, index, out, alpha, beta):
        operands, output = spec.split("->")
        left, right = operands.split(",")
        if pair_order:
            right, index = _pair_ordered(right, index, output)
        tile(f"{left},{right}->{output}", w, t3[index], out=out, alpha=alpha, beta=beta)
    return term


def compute_r4_tri(mycc, imds, t2, t3, t4):
    '''Compute r4 with triangular-stored T4 amplitudes; r4 is returned in triangular form as well.
    r4 will require a symmetry restoration step afterward.
    '''

    nocc, nmo, blksize = mycc.nocc, mycc.nmo, mycc.blksize
    xp = mycc.xp
    nvir = nmo - nocc

    F_oo, F_vv = imds.F_oo, imds.F_vv
    W_oooo, W_ovvo, W_ovov = imds.W_oooo, imds.W_ovvo, imds.W_ovov
    W_vvvo, W_vooo, W_vvvv = imds.W_vvvo, imds.W_vooo, imds.W_vvvv
    W_oovvvo, W_ovovvo, W_ooooov = imds.W_oovvvo, imds.W_ovovvo, imds.W_ooooov
    W_vvoooo, W_vvvvoo = imds.W_vvoooo, imds.W_vvvvoo

    W_voov = xp.ascontiguousarray(W_ovvo.transpose(1, 0, 3, 2))

    c_t3 = xp.empty_like(t3)
    t3_spin_summation(t3, c_t3, nocc**3, nvir, "P3_201", 1.0, 0.0)

    program = getattr(mycc, "_tile_program", None)
    # The compiled T3 pass writes every entry (beta = 0): a recycled R4 needs
    # no zeroing first, unless the Python tile loop below runs instead.
    r4 = mycc.new_quadruples_residual(t4, zeroed=program is None)
    # With pair-blocked T3 copies (compiled CPU tile programs, unless
    # ITERCC_TILE_BLOCKED_T3=0) the T3 views of the W_vooo, W_ovovvo and W_ooooov terms take
    # their index pairs in the order those copies serve (`_t3_term`).
    pair_order = program is not None and program.blocked_enabled()
    def t3_terms(tile, i0, i1, j0, j1, k0, k1, l0, l1, out):
        # The R4 tile terms with T3: one call per term of one tile.
        t3_term = _t3_term(tile, t3, pair_order)

        tile("abej,iklecd->ijklabcd", W_vvvo[..., j0:j1], t3[i0:i1, k0:k1, l0:l1],
            out=out, alpha=1.0, beta=0.0)
        tile("acek,ijlebd->ijklabcd", W_vvvo[..., k0:k1], t3[i0:i1, j0:j1, l0:l1],
            out=out, alpha=1.0, beta=1.0)
        tile("adel,ijkebc->ijklabcd", W_vvvo[..., l0:l1], t3[i0:i1, j0:j1, k0:k1],
            out=out, alpha=1.0, beta=1.0)
        tile("baei,jklecd->ijklabcd", W_vvvo[..., i0:i1], t3[j0:j1, k0:k1, l0:l1],
            out=out, alpha=1.0, beta=1.0)
        tile("caei,kjlebd->ijklabcd", W_vvvo[..., i0:i1], t3[k0:k1, j0:j1, l0:l1],
            out=out, alpha=1.0, beta=1.0)
        tile("daei,ljkebc->ijklabcd", W_vvvo[..., i0:i1], t3[l0:l1, j0:j1, k0:k1],
            out=out, alpha=1.0, beta=1.0)
        tile("bcek,jilead->ijklabcd", W_vvvo[..., k0:k1], t3[j0:j1, i0:i1, l0:l1],
            out=out, alpha=1.0, beta=1.0)
        tile("bdel,jikeac->ijklabcd", W_vvvo[..., l0:l1], t3[j0:j1, i0:i1, k0:k1],
            out=out, alpha=1.0, beta=1.0)
        tile("cbej,kilead->ijklabcd", W_vvvo[..., j0:j1], t3[k0:k1, i0:i1, l0:l1],
            out=out, alpha=1.0, beta=1.0)
        tile("dbej,likeac->ijklabcd", W_vvvo[..., j0:j1], t3[l0:l1, i0:i1, k0:k1],
            out=out, alpha=1.0, beta=1.0)
        tile("cdel,kijeab->ijklabcd", W_vvvo[..., l0:l1], t3[k0:k1, i0:i1, j0:j1],
            out=out, alpha=1.0, beta=1.0)
        tile("dcek,lijeab->ijklabcd", W_vvvo[..., k0:k1], t3[l0:l1, i0:i1, j0:j1],
            out=out, alpha=1.0, beta=1.0)

        t3_term("amij,mklbcd->ijklabcd", W_vooo[:, :, i0:i1, j0:j1],
            np.s_[:, k0:k1, l0:l1], out=out, alpha=-1.0, beta=1.0)
        t3_term("amik,mjlcbd->ijklabcd", W_vooo[:, :, i0:i1, k0:k1],
            np.s_[:, j0:j1, l0:l1], out=out, alpha=-1.0, beta=1.0)
        t3_term("amil,mjkdbc->ijklabcd", W_vooo[:, :, i0:i1, l0:l1],
            np.s_[:, j0:j1, k0:k1], out=out, alpha=-1.0, beta=1.0)
        t3_term("bmji,mklacd->ijklabcd", W_vooo[:, :, j0:j1, i0:i1],
            np.s_[:, k0:k1, l0:l1], out=out, alpha=-1.0, beta=1.0)
        t3_term("cmki,mjlabd->ijklabcd", W_vooo[:, :, k0:k1, i0:i1],
            np.s_[:, j0:j1, l0:l1], out=out, alpha=-1.0, beta=1.0)
        t3_term("dmli,mjkabc->ijklabcd", W_vooo[:, :, l0:l1, i0:i1],
            np.s_[:, j0:j1, k0:k1], out=out, alpha=-1.0, beta=1.0)
        t3_term("bmjk,milcad->ijklabcd", W_vooo[:, :, j0:j1, k0:k1],
            np.s_[:, i0:i1, l0:l1], out=out, alpha=-1.0, beta=1.0)
        t3_term("bmjl,mikdac->ijklabcd", W_vooo[:, :, j0:j1, l0:l1],
            np.s_[:, i0:i1, k0:k1], out=out, alpha=-1.0, beta=1.0)
        t3_term("cmkj,milbad->ijklabcd", W_vooo[:, :, k0:k1, j0:j1],
            np.s_[:, i0:i1, l0:l1], out=out, alpha=-1.0, beta=1.0)
        t3_term("dmlj,mikbac->ijklabcd", W_vooo[:, :, l0:l1, j0:j1],
            np.s_[:, i0:i1, k0:k1], out=out, alpha=-1.0, beta=1.0)
        t3_term("cmkl,mijdab->ijklabcd", W_vooo[:, :, k0:k1, l0:l1],
            np.s_[:, i0:i1, j0:j1], out=out, alpha=-1.0, beta=1.0)
        t3_term("dmlk,mijcab->ijklabcd", W_vooo[:, :, l0:l1, k0:k1],
            np.s_[:, i0:i1, j0:j1], out=out, alpha=-1.0, beta=1.0)

        tile("ijeabm,mklecd->ijklabcd", W_oovvvo[i0:i1, j0:j1],
            c_t3[:, k0:k1, l0:l1], out=out, alpha=0.5, beta=1.0)
        tile("ikeacm,mjlebd->ijklabcd", W_oovvvo[i0:i1, k0:k1],
            c_t3[:, j0:j1, l0:l1], out=out, alpha=0.5, beta=1.0)
        tile("ileadm,mjkebc->ijklabcd", W_oovvvo[i0:i1, l0:l1],
            c_t3[:, j0:j1, k0:k1], out=out, alpha=0.5, beta=1.0)
        tile("jkebcm,milead->ijklabcd", W_oovvvo[j0:j1, k0:k1],
            c_t3[:, i0:i1, l0:l1], out=out, alpha=0.5, beta=1.0)
        tile("jlebdm,mikeac->ijklabcd", W_oovvvo[j0:j1, l0:l1],
            c_t3[:, i0:i1, k0:k1], out=out, alpha=0.5, beta=1.0)
        tile("klecdm,mijeab->ijklabcd", W_oovvvo[k0:k1, l0:l1],
            c_t3[:, i0:i1, j0:j1], out=out, alpha=0.5, beta=1.0)

        t3_term("iejcbm,mklaed->ijklabcd", W_ovovvo[i0:i1, :, j0:j1],
            np.s_[:, k0:k1, l0:l1], out=out, alpha=-1.0, beta=1.0)
        t3_term("iejdbm,mlkaec->ijklabcd", W_ovovvo[i0:i1, :, j0:j1],
            np.s_[:, l0:l1, k0:k1], out=out, alpha=-1.0, beta=1.0)
        t3_term("iekbcm,mjlaed->ijklabcd", W_ovovvo[i0:i1, :, k0:k1],
            np.s_[:, j0:j1, l0:l1], out=out, alpha=-1.0, beta=1.0)
        t3_term("ielbdm,mjkaec->ijklabcd", W_ovovvo[i0:i1, :, l0:l1],
            np.s_[:, j0:j1, k0:k1], out=out, alpha=-1.0, beta=1.0)
        t3_term("iekdcm,mljaeb->ijklabcd", W_ovovvo[i0:i1, :, k0:k1],
            np.s_[:, l0:l1, j0:j1], out=out, alpha=-1.0, beta=1.0)
        t3_term("ielcdm,mkjaeb->ijklabcd", W_ovovvo[i0:i1, :, l0:l1],
            np.s_[:, k0:k1, j0:j1], out=out, alpha=-1.0, beta=1.0)
        t3_term("jeicam,mklbed->ijklabcd", W_ovovvo[j0:j1, :, i0:i1],
            np.s_[:, k0:k1, l0:l1], out=out, alpha=-1.0, beta=1.0)
        t3_term("jeidam,mlkbec->ijklabcd", W_ovovvo[j0:j1, :, i0:i1],
            np.s_[:, l0:l1, k0:k1], out=out, alpha=-1.0, beta=1.0)
        t3_term("keibam,mjlced->ijklabcd", W_ovovvo[k0:k1, :, i0:i1],
            np.s_[:, j0:j1, l0:l1], out=out, alpha=-1.0, beta=1.0)
        t3_term("leibam,mjkdec->ijklabcd", W_ovovvo[l0:l1, :, i0:i1],
            np.s_[:, j0:j1, k0:k1], out=out, alpha=-1.0, beta=1.0)
        t3_term("keidam,mljceb->ijklabcd", W_ovovvo[k0:k1, :, i0:i1],
            np.s_[:, l0:l1, j0:j1], out=out, alpha=-1.0, beta=1.0)
        t3_term("leicam,mkjdeb->ijklabcd", W_ovovvo[l0:l1, :, i0:i1],
            np.s_[:, k0:k1, j0:j1], out=out, alpha=-1.0, beta=1.0)
        t3_term("jekacm,milbed->ijklabcd", W_ovovvo[j0:j1, :, k0:k1],
            np.s_[:, i0:i1, l0:l1], out=out, alpha=-1.0, beta=1.0)
        t3_term("jeladm,mikbec->ijklabcd", W_ovovvo[j0:j1, :, l0:l1],
            np.s_[:, i0:i1, k0:k1], out=out, alpha=-1.0, beta=1.0)
        t3_term("kejabm,milced->ijklabcd", W_ovovvo[k0:k1, :, j0:j1],
            np.s_[:, i0:i1, l0:l1], out=out, alpha=-1.0, beta=1.0)
        t3_term("lejabm,mikdec->ijklabcd", W_ovovvo[l0:l1, :, j0:j1],
            np.s_[:, i0:i1, k0:k1], out=out, alpha=-1.0, beta=1.0)
        t3_term("keladm,mijceb->ijklabcd", W_ovovvo[k0:k1, :, l0:l1],
            np.s_[:, i0:i1, j0:j1], out=out, alpha=-1.0, beta=1.0)
        t3_term("lekacm,mijdeb->ijklabcd", W_ovovvo[l0:l1, :, k0:k1],
            np.s_[:, i0:i1, j0:j1], out=out, alpha=-1.0, beta=1.0)
        t3_term("jekdcm,mlibea->ijklabcd", W_ovovvo[j0:j1, :, k0:k1],
            np.s_[:, l0:l1, i0:i1], out=out, alpha=-1.0, beta=1.0)
        t3_term("jelcdm,mkibea->ijklabcd", W_ovovvo[j0:j1, :, l0:l1],
            np.s_[:, k0:k1, i0:i1], out=out, alpha=-1.0, beta=1.0)
        t3_term("kejdbm,mlicea->ijklabcd", W_ovovvo[k0:k1, :, j0:j1],
            np.s_[:, l0:l1, i0:i1], out=out, alpha=-1.0, beta=1.0)
        t3_term("lejcbm,mkidea->ijklabcd", W_ovovvo[l0:l1, :, j0:j1],
            np.s_[:, k0:k1, i0:i1], out=out, alpha=-1.0, beta=1.0)
        t3_term("kelbdm,mjicea->ijklabcd", W_ovovvo[k0:k1, :, l0:l1],
            np.s_[:, j0:j1, i0:i1], out=out, alpha=-1.0, beta=1.0)
        t3_term("lekbcm,mjidea->ijklabcd", W_ovovvo[l0:l1, :, k0:k1],
            np.s_[:, j0:j1, i0:i1], out=out, alpha=-1.0, beta=1.0)

    # The compiled CPU backend runs the tile loop slot-parallel in the core.
    bases = dict(W_vvvo=W_vvvo, W_vooo=W_vooo, W_oovvvo=W_oovvvo, W_ovovvo=W_ovovvo, t3=t3,
                 c_t3=c_t3)
    # Single precision: the tile programs read T3 and its spin sum as floats.
    single = program is not None and getattr(mycc, "precision", "double") == "single"
    narrow = dict(t3=_cpu_quadruples.to_float32(t3), c_t3=_cpu_quadruples.to_float32(c_t3)) if single else None
    # Pair-blocked copies of T3 and its spin sum (T3 itself is unchanged below).
    blocked = program.blocked(dict(t3=t3, c_t3=c_t3), single=single) if pair_order else None
    key = "t3 terms, pair order" if pair_order else "t3 terms"
    if program is None or not program.run(key, t3_terms, bases, r4, 0.0, narrow=narrow, blocked=blocked):
        if program is not None:
            mycc._scale_(r4, 0.0)
        r4_tmp = xp.empty((blksize,) * 4 + (nvir,) * 4, dtype=t4.dtype)
        for l0, l1 in prange(0, nocc, blksize):
            bl = l1 - l0
            for k0, k1 in prange(0, l1, blksize):
                bk = k1 - k0
                for j0, j1 in prange(0, k1, blksize):
                    bj = j1 - j0
                    for i0, i1 in prange(0, j1, blksize):
                        bi = i1 - i0
                        bounds = (i0, i1, j0, j1, k0, k1, l0, l1)
                        t3_terms(_tile_einsum(mycc, bounds), *bounds, r4_tmp[:bi, :bj, :bk, :bl])
                        _accumulate_t4_(mycc, r4, r4_tmp, *bounds)
        r4_tmp = None
    c_t3 = None
    W_vvvo = imds.W_vvvo = None
    W_vooo = imds.W_vooo = None
    W_oovvvo = imds.W_oovvvo = None

    c_t3 = _kernels(t3).add_transposed(t3, (0, 1, 2, 4, 5, 3))
    _kernels(W_ovovvo).add_transpose_(W_ovovvo, (2, 1, 0, 4, 3, 5))
    if xp is np:
        # Every tile fixes the occupied indices of W_vvoooo and W_vvvvoo, which are
        # innermost: on the CPU each read would take its own cache line of a large
        # array. They are stored with the occupied indices outermost (no copy
        # when the intermediates were made so, `_empty_stored`).
        W_vvoooo = _kernels(W_vvoooo).relayout(W_vvoooo, _W_VVOOOO_ORDER)
        W_vvvvoo = _kernels(W_vvvvoo).relayout(W_vvvvoo, _W_VVVVOO_ORDER)
        imds.W_vvoooo, imds.W_vvvvoo = W_vvoooo, W_vvvvoo

    def t4_terms(tile, i0, i1, j0, j1, k0, k1, l0, l1, out, t4_tile):
        # The R4 tile terms with T3, T2 and the tile's own T4 block.
        t3_term = _t3_term(tile, t3, pair_order)

        tile("iejabm,mklced->ijklabcd", W_ovovvo[i0:i1, :, j0:j1],
            c_t3[:, k0:k1, l0:l1], out=out, alpha=-0.5, beta=0.0)
        tile("iekacm,mjlbed->ijklabcd", W_ovovvo[i0:i1, :, k0:k1],
            c_t3[:, j0:j1, l0:l1], out=out, alpha=-0.5, beta=1.0)
        tile("ieladm,mjkbec->ijklabcd", W_ovovvo[i0:i1, :, l0:l1],
            c_t3[:, j0:j1, k0:k1], out=out, alpha=-0.5, beta=1.0)
        tile("jekbcm,milaed->ijklabcd", W_ovovvo[j0:j1, :, k0:k1],
            c_t3[:, i0:i1, l0:l1], out=out, alpha=-0.5, beta=1.0)
        tile("jelbdm,mikaec->ijklabcd", W_ovovvo[j0:j1, :, l0:l1],
            c_t3[:, i0:i1, k0:k1], out=out, alpha=-0.5, beta=1.0)
        tile("kelcdm,mijaeb->ijklabcd", W_ovovvo[k0:k1, :, l0:l1],
            c_t3[:, i0:i1, j0:j1], out=out, alpha=-0.5, beta=1.0)

        t3_term("kjinma,mnlbcd->ijklabcd", W_ooooov[k0:k1, j0:j1, i0:i1],
            np.s_[:, :, l0:l1], out=out, alpha=1.0, beta=1.0)
        t3_term("ljinma,mnkbdc->ijklabcd", W_ooooov[l0:l1, j0:j1, i0:i1],
            np.s_[:, :, k0:k1,], out=out, alpha=1.0, beta=1.0)
        t3_term("lkinma,mnjcdb->ijklabcd", W_ooooov[l0:l1, k0:k1, i0:i1],
            np.s_[:, :, j0:j1], out=out, alpha=1.0, beta=1.0)
        t3_term("kijnmb,mnlacd->ijklabcd", W_ooooov[k0:k1, i0:i1, j0:j1],
            np.s_[:, :, l0:l1], out=out, alpha=1.0, beta=1.0)
        t3_term("lijnmb,mnkadc->ijklabcd", W_ooooov[l0:l1, i0:i1, j0:j1],
            np.s_[:, :, k0:k1], out=out, alpha=1.0, beta=1.0)
        t3_term("lkjnmb,mnicda->ijklabcd", W_ooooov[l0:l1, k0:k1, j0:j1],
            np.s_[:, :, i0:i1], out=out, alpha=1.0, beta=1.0)
        t3_term("jiknmc,mnlabd->ijklabcd", W_ooooov[j0:j1, i0:i1, k0:k1],
            np.s_[:, :, l0:l1], out=out, alpha=1.0, beta=1.0)
        t3_term("liknmc,mnjadb->ijklabcd", W_ooooov[l0:l1, i0:i1, k0:k1],
            np.s_[:, :, j0:j1], out=out, alpha=1.0, beta=1.0)
        t3_term("ljknmc,mnibda->ijklabcd", W_ooooov[l0:l1, j0:j1, k0:k1],
            np.s_[:, :, i0:i1], out=out, alpha=1.0, beta=1.0)
        t3_term("jilnmd,mnkabc->ijklabcd", W_ooooov[j0:j1, i0:i1, l0:l1],
            np.s_[:, :, k0:k1], out=out, alpha=1.0, beta=1.0)
        t3_term("kilnmd,mnjacb->ijklabcd", W_ooooov[k0:k1, i0:i1, l0:l1],
            np.s_[:, :, j0:j1], out=out, alpha=1.0, beta=1.0)
        t3_term("kjlnmd,mnibca->ijklabcd", W_ooooov[k0:k1, j0:j1, l0:l1],
            np.s_[:, :, i0:i1], out=out, alpha=1.0, beta=1.0)

        tile("mlcd,abmijk->ijklabcd", t2[:, l0:l1], W_vvoooo[..., i0:i1, j0:j1, k0:k1],
            out=out, alpha=-1.0, beta=1.0)
        tile("mkdc,abmijl->ijklabcd", t2[:, k0:k1], W_vvoooo[..., i0:i1, j0:j1, l0:l1],
            out=out, alpha=-1.0, beta=1.0)
        tile("mlbd,acmikj->ijklabcd", t2[:, l0:l1], W_vvoooo[..., i0:i1, k0:k1, j0:j1],
            out=out, alpha=-1.0, beta=1.0)
        tile("mkbc,admilj->ijklabcd", t2[:, k0:k1], W_vvoooo[..., i0:i1, l0:l1, j0:j1],
            out=out, alpha=-1.0, beta=1.0)
        tile("mjdb,acmikl->ijklabcd", t2[:, j0:j1], W_vvoooo[..., i0:i1, k0:k1, l0:l1],
            out=out, alpha=-1.0, beta=1.0)
        tile("mjcb,admilk->ijklabcd", t2[:, j0:j1], W_vvoooo[..., i0:i1, l0:l1, k0:k1],
            out=out, alpha=-1.0, beta=1.0)
        tile("mlad,bcmjki->ijklabcd", t2[:, l0:l1], W_vvoooo[..., j0:j1, k0:k1, i0:i1],
            out=out, alpha=-1.0, beta=1.0)
        tile("mkac,bdmjli->ijklabcd", t2[:, k0:k1], W_vvoooo[..., j0:j1, l0:l1, i0:i1],
            out=out, alpha=-1.0, beta=1.0)
        tile("mjab,cdmkli->ijklabcd", t2[:, j0:j1], W_vvoooo[..., k0:k1, l0:l1, i0:i1],
            out=out, alpha=-1.0, beta=1.0)
        tile("mida,bcmjkl->ijklabcd", t2[:, i0:i1], W_vvoooo[..., j0:j1, k0:k1, l0:l1],
            out=out, alpha=-1.0, beta=1.0)
        tile("mica,bdmjlk->ijklabcd", t2[:, i0:i1], W_vvoooo[..., j0:j1, l0:l1, k0:k1],
            out=out, alpha=-1.0, beta=1.0)
        tile("miba,cdmklj->ijklabcd", t2[:, i0:i1], W_vvoooo[..., k0:k1, l0:l1, j0:j1],
            out=out, alpha=-1.0, beta=1.0)

        tile("iled,abcejk->ijklabcd", t2[i0:i1, l0:l1], W_vvvvoo[..., j0:j1, k0:k1],
            out=out, alpha=1.0, beta=1.0)
        tile("ikec,abdejl->ijklabcd", t2[i0:i1, k0:k1], W_vvvvoo[..., j0:j1, l0:l1],
            out=out, alpha=1.0, beta=1.0)
        tile("ijeb,acdekl->ijklabcd", t2[i0:i1, j0:j1], W_vvvvoo[..., k0:k1, l0:l1],
            out=out, alpha=1.0, beta=1.0)
        tile("jled,baceik->ijklabcd", t2[j0:j1, l0:l1], W_vvvvoo[..., i0:i1, k0:k1],
            out=out, alpha=1.0, beta=1.0)
        tile("jkec,badeil->ijklabcd", t2[j0:j1, k0:k1], W_vvvvoo[..., i0:i1, l0:l1],
            out=out, alpha=1.0, beta=1.0)
        tile("kled,cabeij->ijklabcd", t2[k0:k1, l0:l1], W_vvvvoo[..., i0:i1, j0:j1],
            out=out, alpha=1.0, beta=1.0)
        tile("lkec,dabeij->ijklabcd", t2[l0:l1, k0:k1], W_vvvvoo[..., i0:i1, j0:j1],
            out=out, alpha=1.0, beta=1.0)
        tile("kjeb,cadeil->ijklabcd", t2[k0:k1, j0:j1], W_vvvvoo[..., i0:i1, l0:l1],
            out=out, alpha=1.0, beta=1.0)
        tile("ljeb,daceik->ijklabcd", t2[l0:l1, j0:j1], W_vvvvoo[..., i0:i1, k0:k1],
            out=out, alpha=1.0, beta=1.0)
        tile("jiea,bcdekl->ijklabcd", t2[j0:j1, i0:i1], W_vvvvoo[..., k0:k1, l0:l1],
            out=out, alpha=1.0, beta=1.0)
        tile("kiea,cbdejl->ijklabcd", t2[k0:k1, i0:i1], W_vvvvoo[..., j0:j1, l0:l1],
            out=out, alpha=1.0, beta=1.0)
        tile("liea,dbcejk->ijklabcd", t2[l0:l1, i0:i1], W_vvvvoo[..., j0:j1, k0:k1],
            out=out, alpha=1.0, beta=1.0)

        tile("ae,ijklebcd->ijklabcd", F_vv, t4_tile,
            out=out, alpha=1.0, beta=1.0)
        tile("be,ijklaecd->ijklabcd", F_vv, t4_tile,
            out=out, alpha=1.0, beta=1.0)
        tile("ce,ijklabed->ijklabcd", F_vv, t4_tile,
            out=out, alpha=1.0, beta=1.0)
        tile("de,ijklabce->ijklabcd", F_vv, t4_tile,
            out=out, alpha=1.0, beta=1.0)

        tile("abef,ijklefcd->ijklabcd", W_vvvv, t4_tile,
            out=out, alpha=1.0, beta=1.0)
        tile("acef,ijklebfd->ijklabcd", W_vvvv, t4_tile,
            out=out, alpha=1.0, beta=1.0)
        tile("adef,ijklebcf->ijklabcd", W_vvvv, t4_tile,
            out=out, alpha=1.0, beta=1.0)
        tile("bcef,ijklaefd->ijklabcd", W_vvvv, t4_tile,
            out=out, alpha=1.0, beta=1.0)
        tile("bdef,ijklaecf->ijklabcd", W_vvvv, t4_tile,
            out=out, alpha=1.0, beta=1.0)
        tile("cdef,ijklabef->ijklabcd", W_vvvv, t4_tile,
            out=out, alpha=1.0, beta=1.0)

    bases = dict(W_ovovvo=W_ovovvo, W_ooooov=W_ooooov, W_vvoooo=W_vvoooo, W_vvvvoo=W_vvvvoo,
                 F_vv=F_vv, W_vvvv=W_vvvv, t2=t2, t3=t3, c_t3=c_t3)
    if single:
        narrow["c_t3"] = _cpu_quadruples.to_float32(c_t3)
    # The spin sum of this pass is another array: only T3 stays blocked.
    blocked = None if blocked is None else dict(t3=blocked["t3"])
    key = "t4 terms, pair order" if pair_order else "t4 terms"
    if program is None or not program.run(key, t4_terms, bases, r4, 1.0, t4=t4, narrow=narrow,
                                          blocked=blocked):
        t4_tmp = xp.empty((blksize,) * 4 + (nvir,) * 4, dtype=t4.dtype)
        r4_tmp = xp.empty((blksize,) * 4 + (nvir,) * 4, dtype=t4.dtype)
        for l0, l1 in prange(0, nocc, blksize):
            bl = l1 - l0
            for k0, k1 in prange(0, l1, blksize):
                bk = k1 - k0
                for j0, j1 in prange(0, k1, blksize):
                    bj = j1 - j0
                    for i0, i1 in prange(0, j1, blksize):
                        bi = i1 - i0
                        bounds = (i0, i1, j0, j1, k0, k1, l0, l1)
                        _unpack_t4_(mycc, t4, t4_tmp, *bounds)
                        t4_terms(_tile_einsum(mycc, bounds), *bounds, r4_tmp[:bi, :bj, :bk, :bl],
                                 t4_tmp[:bi, :bj, :bk, :bl])
                        _accumulate_t4_(mycc, r4, r4_tmp, *bounds, beta=1.0)
        t4_tmp = r4_tmp = None
    c_t3 = narrow = blocked = None
    F_vv = imds.F_vv = None
    W_vvvv = imds.W_vvvv = None
    W_ovovvo = imds.W_ovovvo = None
    W_ooooov = imds.W_ooooov = None
    W_vvoooo = imds.W_vvoooo = None
    W_vvvvoo = imds.W_vvvvoo = None


    if isinstance(r4, xp.ndarray):     # not a host-staged R4 of a device T4
        r4_add_rings_triples_(mycc, r4, t4, F_oo, W_ovov, W_voov)
    else:
        r4_add_rings_tiles_(mycc, r4, t4, F_oo, W_ovov, W_voov)
    F_oo = imds.F_oo = None
    W_ovvo = imds.W_ovvo = None
    W_ovov = imds.W_ovov = None

    if isinstance(r4, xp.ndarray):     # not a host-staged R4 of a device T4
        r4_add_oooo_pairs_(mycc, r4, t4, W_oooo)
    else:
        r4_add_oooo_tiles_(mycc, r4, t4, W_oooo)
    W_oooo = imds.W_oooo = None
    return r4


def r4_add_oooo_pairs_(mycc, r4, t4, W_oooo, *, budget=2**30):
    '''Add the six W_oooo terms of R4 one spectator pair at a time.

    With the pair symmetry of T4, every term is
        R4(o; v) += sum_mn W[m, n, o_P] T4[m, n, o_Q; v_P, v_Q]
    for one split of the occupied positions into a target pair P and a
    spectator pair Q. For each spectator pair (p, q) the unpacked block
    T4[:, :, p, q] is contracted once with W as a GEMM, and each row of the
    product is added to every stored slot it belongs to.
    '''
    nocc, nvir = mycc.nocc, mycc.nvir
    xp = mycc.xp
    layout = mycc.layout
    if hasattr(layout, "add_oooo_pairs") and layout.symmetric:
        # The compiled CPU layout: packed blocks and one dense GEMM per row class.
        return layout.add_oooo_pairs(r4, t4, W_oooo)
    block = xp.empty((nocc, nocc, 1, 1) + (nvir,) * 4, dtype=t4.dtype)
    if getattr(mycc, "_block_blocks", None) is not None:
        block[...] = 0.0     # the gathers below write allowed entries only
        # Symmetry-blocked: Y[r, o; v] = sum_mn W[m, n, r, o] T4[m, n, p, q; v] for a
        # chunk of r at a time. Only the allowed (mn, o, v) blocks are touched, about
        # nirrep**2 fewer flops than the dense GEMM below.
        occupied = layout.occupied_irreps
        plain = block.reshape((nocc, nocc) + (nvir,) * 4)
        chunk = int(max(1, min(nocc, budget // (nocc * nvir**4 * block.itemsize))))
        rows = xp.empty((chunk, nocc) + (nvir,) * 4, dtype=t4.dtype)
        contract = _block_einsum(mycc)
        for p in range(nocc):
            for q in range(p, nocc):
                _unpack_t4_(mycc, t4, block, 0, nocc, 0, nocc, p, p + 1, q, q + 1, allowed_only=True)
                spectators = (0, int(occupied[p] ^ occupied[q]))
                for r0 in range(0, nocc, chunk):
                    r1 = min(nocc, r0 + chunk)
                    contract("mnix,mnabcd->ixabcd", W_oooo[:, :, r0:r1], plain, out=rows[:r1 - r0],
                             alpha=1.0, beta=0.0, labels=dict(i=occupied[r0:r1]), irreps=spectators)
                    for r in range(r0, r1):
                        layout.add_rows(r4, rows[r - r0], layout.pair_entries(p, q, r))
        return
    pairs = xp.ascontiguousarray(W_oooo.reshape(nocc * nocc, nocc * nocc))
    rows = xp.empty((nocc,) + (nvir,) * 4, dtype=t4.dtype)
    flat = block.reshape(nocc * nocc, -1)
    for p in range(nocc):
        for q in range(p, nocc):
            _unpack_t4_(mycc, t4, block, 0, nocc, 0, nocc, p, p + 1, q, q + 1)
            for r in range(nocc):
                xp.matmul(pairs[:, r * nocc:(r + 1) * nocc].T, flat, out=rows.reshape(nocc, -1))
                layout.add_rows(r4, rows, layout.pair_entries(p, q, r))


def r4_add_oooo_tiles_(mycc, r4, t4, W_oooo):
    '''Tile-by-tile W_oooo terms of R4 for a host-staged residual.'''
    nocc, nmo, blksize = mycc.nocc, mycc.nmo, mycc.blksize
    xp = mycc.xp
    nvir = nmo - nocc

    t4_tmp = xp.empty((blksize,) * 3 + (nocc,) + (nvir,) * 4, dtype=t4.dtype)
    r4_tmp = xp.empty((blksize,) * 4 + (nvir,) * 4, dtype=t4.dtype)
    for l0, l1 in prange(0, nocc, blksize):
        bl = l1 - l0
        for k0, k1 in prange(0, l1, blksize):
            bk = k1 - k0
            for j0, j1 in prange(0, k1, blksize):
                bj = j1 - j0
                for i0, i1 in prange(0, j1, blksize):
                    bi = i1 - i0
                    for m0, m1 in prange(0, nocc, blksize):
                        bm = m1 - m0
                        contract = _tile_einsum(mycc, (i0, i1, j0, j1, k0, k1, l0, l1),
                                                m=(m0, m1))

                        # Sum every m block into the tile, then scatter once.
                        _unpack_t4_(mycc, t4, t4_tmp, k0, k1, l0, l1,  m0, m1, 0, nocc, blksize, blksize, blksize, nocc)
                        contract("mnij,klmncdab->ijklabcd", W_oooo[m0:m1, :, i0:i1, j0:j1],
                            t4_tmp[:bk, :bl, :bm], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0,
                            beta=0.0 if m0 == 0 else 1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, j0, j1, l0, l1, m0, m1, 0, nocc, blksize, blksize, blksize, nocc)
                        contract("mnik,jlmnbdac->ijklabcd", W_oooo[m0:m1, :, i0:i1, k0:k1],
                            t4_tmp[:bj, :bl, :bm], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, j0, j1, k0, k1, m0, m1, 0, nocc, blksize, blksize, blksize, nocc)
                        contract("mnil,jkmnbcad->ijklabcd", W_oooo[m0:m1, :, i0:i1, l0:l1],
                            t4_tmp[:bj, :bk, :bm], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, i0, i1, l0, l1, m0, m1, 0, nocc, blksize, blksize, blksize, nocc)
                        contract("mnjk,ilmnadbc->ijklabcd", W_oooo[m0:m1, :, j0:j1, k0:k1],
                            t4_tmp[:bi, :bl, :bm], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, i0, i1, k0, k1, m0, m1, 0, nocc, blksize, blksize, blksize, nocc)
                        contract("mnjl,ikmnacbd->ijklabcd", W_oooo[m0:m1, :, j0:j1, l0:l1],
                            t4_tmp[:bi, :bk, :bm], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)
                        _unpack_t4_(mycc, t4, t4_tmp, i0, i1, j0, j1, m0, m1, 0, nocc, blksize, blksize, blksize, nocc)
                        contract("mnkl,ijmnabcd->ijklabcd", W_oooo[m0:m1, :, k0:k1, l0:l1],
                            t4_tmp[:bi, :bj, :bm], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=1.0, beta=1.0)

                    _accumulate_t4_(mycc, r4, r4_tmp, i0, i1, j0, j1, k0, k1, l0, l1, beta=1.0)

    t4_tmp = None
    r4_tmp = None


def r4_add_rings_triples_(mycc, r4, t4, F_oo, W_ovov, W_voov):
    '''Add the F_oo, W_ovov and W_voov terms of R4 one spectator triple at a time.

    For a target position t and the occupied triple (p, q, r) in the other
    positions, every term has the form
        R4(o; v) += A[o_t; v_t, v_rest],  A[x] from T4[m, p, q, r] contracted over m (and e),
    so each sorted triple is unpacked once and A is added to the slots of all
    four target positions.
    '''
    nocc, nvir = mycc.nocc, mycc.nvir
    xp = mycc.xp
    layout = mycc.layout
    if hasattr(layout, "add_oooo_pairs") and layout.symmetric:
        # The compiled CPU layout: packed blocks and box-level plans.
        from . import _cpu_packed
        from ._runtime import cpu_threads

        return _cpu_packed.add_rings(layout, r4, t4, F_oo, W_ovov, W_voov, cpu_threads(None))
    shape = (nocc,) + (nvir,) * 4
    # With blocked contractions (allowed entries only) the gathers skip the
    # symmetry zeros of the blocks; they start at zero, so the rest stays finite
    # (the dense spin pass below maps allowed entries onto allowed ones).
    allowed = getattr(mycc, "_block_blocks", None) is not None
    block = xp.zeros((nocc, 1, 1, 1) + (nvir,) * 4, dtype=t4.dtype)
    ring = xp.zeros_like(block)
    rows = xp.empty(shape, dtype=t4.dtype)
    plain, summed = block.reshape(shape), ring.reshape(shape)
    contract = _block_einsum(mycc)
    occupied = mycc.layout.occupied_irreps
    for p in range(nocc):
        for q in range(p, nocc):
            for r in range(q, nocc):
                spectators = (0, int(occupied[p] ^ occupied[q] ^ occupied[r]))
                _unpack_t4_(mycc, t4, block, 0, nocc, p, p + 1, q, q + 1, r, r + 1,
                            allowed_only=allowed)
                _unpack_t4_triples_(mycc, t4, ring, 0, nocc, p, p + 1, q, q + 1, r, r + 1,
                                    allowed_only=allowed)
                contract("mx,mabcd->xabcd", F_oo, plain, out=rows, alpha=-1.0, beta=0.0,
                         irreps=spectators)
                contract("mbxe,maecd->xabcd", W_ovov, plain, out=rows, alpha=-1.0, beta=1.0,
                         irreps=spectators)
                contract("mcxe,mabed->xabcd", W_ovov, plain, out=rows, alpha=-1.0, beta=1.0,
                         irreps=spectators)
                contract("mdxe,mabce->xabcd", W_ovov, plain, out=rows, alpha=-1.0, beta=1.0,
                         irreps=spectators)
                contract("maxe,mbecd->xabcd", W_ovov, summed, out=rows, alpha=-0.5, beta=1.0,
                         irreps=spectators)
                # The plain block is at hand: a dense spin pass beats regathering (A100).
                t4_spin_summation(block, ring, nvir, "P4_201")
                contract("amxe,mebcd->xabcd", W_voov, summed, out=rows, alpha=0.5, beta=1.0,
                         irreps=spectators)
                layout.add_rows(r4, rows, layout.triple_entries((p, q, r)))


def r4_add_rings_tiles_(mycc, r4, t4, F_oo, W_ovov, W_voov):
    '''Tile-by-tile F_oo, W_ovov and W_voov terms of R4 for a host-staged residual.'''
    nocc, nmo, blksize = mycc.nocc, mycc.nmo, mycc.blksize
    xp = mycc.xp
    nvir = nmo - nocc

    t4_tmp = xp.empty((nocc,) + (blksize,) * 3 + (nvir,) * 4, dtype=t4.dtype)
    r4_tmp = xp.empty((blksize,) * 4 + (nvir,) * 4, dtype=t4.dtype)
    for l0, l1 in prange(0, nocc, blksize):
        bl = l1 - l0
        for k0, k1 in prange(0, l1, blksize):
            bk = k1 - k0
            for j0, j1 in prange(0, k1, blksize):
                bj = j1 - j0
                for i0, i1 in prange(0, j1, blksize):
                    bi = i1 - i0
                    contract = _tile_einsum(mycc, (i0, i1, j0, j1, k0, k1, l0, l1))

                    _unpack_t4_(mycc, t4, t4_tmp, 0, nocc, j0, j1, k0, k1, l0, l1, nocc, blksize, blksize, blksize)
                    contract("mi,mjklabcd->ijklabcd", F_oo[:, i0:i1], t4_tmp[:, :bj, :bk, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=0.0)
                    contract("mbie,mjklaecd->ijklabcd", W_ovov[:, :, i0:i1, :], t4_tmp[:, :bj, :bk, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    contract("mcie,mjklabed->ijklabcd", W_ovov[:, :, i0:i1, :], t4_tmp[:, :bj, :bk, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    contract("mdie,mjklabce->ijklabcd", W_ovov[:, :, i0:i1, :], t4_tmp[:, :bj, :bk, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    mycc.layout.gather(t4, t4_tmp, (0, nocc, j0, j1, k0, k1, l0, l1), spin="P4_201")
                    contract("amie,mjklebcd->ijklabcd", W_voov[:, :, i0:i1, :], t4_tmp[:, :bj, :bk, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.5, beta=1.0)

                    _unpack_t4_(mycc, t4, t4_tmp, 0, nocc, i0, i1, k0, k1, l0, l1, nocc, blksize, blksize, blksize)
                    contract("mj,miklbacd->ijklabcd", F_oo[:, j0:j1], t4_tmp[:, :bi, :bk, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    contract("maje,miklbecd->ijklabcd", W_ovov[:, :, j0:j1, :], t4_tmp[:, :bi, :bk, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    contract("mcje,miklbaed->ijklabcd", W_ovov[:, :, j0:j1, :], t4_tmp[:, :bi, :bk, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    contract("mdje,miklbace->ijklabcd", W_ovov[:, :, j0:j1, :], t4_tmp[:, :bi, :bk, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    mycc.layout.gather(t4, t4_tmp, (0, nocc, i0, i1, k0, k1, l0, l1), spin="P4_201")
                    contract("bmje,mikleacd->ijklabcd", W_voov[:, :, j0:j1, :], t4_tmp[:, :bi, :bk, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.5, beta=1.0)

                    _unpack_t4_(mycc, t4, t4_tmp, 0, nocc, i0, i1, j0, j1, l0, l1, nocc, blksize, blksize, blksize)
                    contract("mk,mijlcabd->ijklabcd", F_oo[:, k0:k1], t4_tmp[:, :bi, :bj, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    contract("make,mijlcebd->ijklabcd", W_ovov[:, :, k0:k1, :], t4_tmp[:, :bi, :bj, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    contract("mbke,mijlcaed->ijklabcd", W_ovov[:, :, k0:k1, :], t4_tmp[:, :bi, :bj, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    contract("mdke,mijlcabe->ijklabcd", W_ovov[:, :, k0:k1, :], t4_tmp[:, :bi, :bj, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    mycc.layout.gather(t4, t4_tmp, (0, nocc, i0, i1, j0, j1, l0, l1), spin="P4_201")
                    contract("cmke,mijleabd->ijklabcd", W_voov[:, :, k0:k1, :], t4_tmp[:, :bi, :bj, :bl],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.5, beta=1.0)

                    _unpack_t4_(mycc, t4, t4_tmp, 0, nocc, i0, i1, j0, j1, k0, k1, nocc, blksize, blksize, blksize)
                    contract("ml,mijkdabc->ijklabcd", F_oo[:, l0:l1], t4_tmp[:, :bi, :bj, :bk],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    contract("male,mijkdebc->ijklabcd", W_ovov[:, :, l0:l1, :], t4_tmp[:, :bi, :bj, :bk],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    contract("mble,mijkdaec->ijklabcd", W_ovov[:, :, l0:l1, :], t4_tmp[:, :bi, :bj, :bk],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    contract("mcle,mijkdabe->ijklabcd", W_ovov[:, :, l0:l1, :], t4_tmp[:, :bi, :bj, :bk],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-1.0, beta=1.0)
                    mycc.layout.gather(t4, t4_tmp, (0, nocc, i0, i1, j0, j1, k0, k1), spin="P4_201")
                    contract("dmle,mijkeabc->ijklabcd", W_voov[:, :, l0:l1, :], t4_tmp[:, :bi, :bj, :bk],
                        out=r4_tmp[:bi, :bj, :bk, :bl], alpha=0.5, beta=1.0)

                    _unpack_t4_triples_(mycc, t4, t4_tmp, 0, nocc, j0, j1, k0, k1, l0, l1,
                                        nocc, blksize, blksize, blksize)
                    contract("maie,mjklbecd->ijklabcd", W_ovov[:, :, i0:i1, :],
                        t4_tmp[:, :bj, :bk, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    _unpack_t4_triples_(mycc, t4, t4_tmp, 0, nocc, i0, i1, k0, k1, l0, l1,
                                        nocc, blksize, blksize, blksize)
                    contract("mbje,miklaecd->ijklabcd", W_ovov[:, :, j0:j1, :],
                        t4_tmp[:, :bi, :bk, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    _unpack_t4_triples_(mycc, t4, t4_tmp, 0, nocc, i0, i1, j0, j1, l0, l1,
                                        nocc, blksize, blksize, blksize)
                    contract("mcke,mijlaebd->ijklabcd", W_ovov[:, :, k0:k1, :],
                        t4_tmp[:, :bi, :bj, :bl], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)
                    _unpack_t4_triples_(mycc, t4, t4_tmp, 0, nocc, i0, i1, j0, j1, k0, k1,
                                        nocc, blksize, blksize, blksize)
                    contract("mdle,mijkaebc->ijklabcd", W_ovov[:, :, l0:l1, :],
                        t4_tmp[:, :bi, :bj, :bk], out=r4_tmp[:bi, :bj, :bk, :bl], alpha=-0.5, beta=1.0)

                    _accumulate_t4_(mycc, r4, r4_tmp, i0, i1, j0, j1, k0, k1, l0, l1, beta=1.0)
    t4_tmp = None
    r4_tmp = None
