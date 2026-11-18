"""CPU Jacobian actions of the restricted CCSDT triples terms (EOM and response).

The EOM and response engines differentiate `RestrictedCC._evaluate` with dual
numbers (`_directional.Directional`). Two pieces dominate and both have a
simple structure:

* the triples residual is a T2-driven part, bilinear in (t2, vooo/vvvo),
  plus a part linear in T3 whose operators are built from the intermediates
  and the ladder matrix W;
* the T3 feedback is bilinear in T3 and the dressed (f, g).

So their directional derivatives are, by the product rule, calls of the
compiled CPU functions of `_cpu_triples` on plain arrays (two for the
tangents, one more for the feedback's ground value), instead of one pass of
dual-number arithmetic through the per-(j,k) loop. Each call checks
block-diagonality before using symmetry blocks, so non-totally-symmetric
directions (excited states of any irrep) stay exact.

An EOM engine linearizes at one ground state: the dressed (f, g), T2 and T3
values are the same arrays in every action. With a `ground` cache (a dict the
engine owns) the parts that depend on them alone are computed once: the
feedback's ground increments and the ladder matrix without its one-body term.
"""

from dataclasses import fields
import weakref

import numpy as np

from . import _cpu_kernels, _runtime
from . import _cpu_triples as _cpu
from ._restricted_equations import Intermediates


def _parts(x):
    """(value, tangent) of a Directional or plain array; tangent None means constant."""
    return getattr(x, "value", x), getattr(x, "tangent", None)


def _zero_if_none(value, tangent):
    return np.zeros_like(value) if tangent is None else tangent


def _cached(ground, name, inputs, build):
    """build() once per set of `inputs` (compared by identity) in `ground`.
    The first entry (the engine's ground state, which every Jacobian action
    uses) stays; one more slot serves other points (response's mixed
    derivatives at shifted amplitudes) without evicting it."""
    if ground is None:
        return build()
    entries = ground.setdefault(name, [])
    for stored, value in entries:
        if len(stored) == len(inputs) and all(a is b for a, b in zip(stored, inputs)):
            return value
    value = build()
    if len(entries) >= 2:
        entries.pop()
    entries.append((inputs, value))
    return value


def feedback(r1, r2, vooo, vvvo, f, g, layout, t3, o, v, *, threads=None, ground=None, proxy=None,
             labels=None, irrep=None):
    """Directional `triples_feedback` in place: values get the ground terms,
    tangents d(feedback) = feedback(dT3; f, g) + feedback(T3; df, dg).

    With `labels` (the caller has checked that f and g are exactly totally
    symmetric and df and dg exactly of the direction's irrep `irrep`: projected
    operators, a projected ground T1 and a direction T1 of that irrep), each
    term whose T3 is exactly of its irrep runs on blocked T3 with
    symmetry-blocked products (`_blocked_residual.feedback`): dT3 in the
    labels shifted by the irrep (the copy `triples` reuses), T3 in the ground
    labels."""
    outputs = (r1, r2, vooo, vvvo)
    for out in outputs:
        if out.tangent is None:
            out.tangent = np.zeros_like(out.value)
    t3_0, dt3 = _parts(t3)
    f0, df = _parts(f)
    g0, dg = _parts(g)
    irrep = int(irrep or 0)
    workers = _runtime.cpu_threads(threads)

    def blocked_t3(array, shift, key):
        """(layout, tables, blocked array) of a T3 exactly of irrep `shift`, else None."""
        plan = _blocked_layout(proxy, labels, ground, shift) if labels is not None else None
        if plan is None or not len(array):
            return None
        packed = (_blocked_direction(plan[0], ground, array, shift, workers) if key is None else
                  _cached(ground, key, (array,), lambda: _packed(plan[0], array, workers)))
        return None if packed is None else (plan[0], plan[1], packed)

    with _cpu.ThreadPool(workers) as pool:
        def increments():
            parts = tuple(np.zeros_like(x.value) for x in outputs)
            _cpu.feedback(*parts, f0, g0, layout, t3_0, o, v, pool)
            return parts
        for out, part in zip(outputs, _cached(ground, "feedback", (f0, g0, t3_0, layout), increments)):
            out.value += part
        tangents = tuple(x.tangent for x in outputs)
        if dt3 is not None:
            found = blocked_t3(dt3, irrep, None)
            if found is not None:
                from . import _blocked_residual

                _blocked_residual.feedback(found[0], found[1], *tangents, f0, g0, found[2], o, v,
                                           workers, shift=irrep)
            else:
                _cpu.feedback(*tangents, f0, g0, layout, dt3, o, v, pool)
        if df is not None or dg is not None:
            df, dg = _zero_if_none(f0, df), _zero_if_none(g0, dg)
            found = blocked_t3(t3_0, 0, "blocked ground")
            if found is not None:
                from . import _blocked_residual

                _blocked_residual.feedback(found[0], found[1], *tangents, df, dg, found[2], o, v,
                                           workers, operator_irrep=irrep)
            else:
                _cpu.feedback(*tangents, df, dg, layout, t3_0, o, v, pool)


def _blocked_layout(proxy, labels, ground, irrep=0):
    """(blocked layout, plane tables, T2 forbidden mask) of the engine's orbital
    labels with every occupied label XORed with `irrep`, built once per engine
    and irrep, or None (no labels, no cache, no core, or an earlier operator
    that coupled irreps).

    Shifting the occupied labels by the direction's irrep G moves every T3
    slot's irrep by G (three occupied indices) and leaves pair irreps alone
    (two): in the shifted labels the direction's T3, the derivative triples
    sources (three or one occupied index) and every ground-state operator
    are totally symmetric, so the blocked residual serves them unchanged."""
    if (proxy is None or labels is None or ground is None or ground.get(("blocked off", irrep))
            or _cpu_kernels.library() is None or not proxy.layout.shape[0]):
        return None
    from ._blocked import BlockedTriples
    from ._blocked_residual import PlaneTables

    no = proxy.layout.no
    occupied, virtual = np.asarray(labels[:no]) ^ int(irrep), np.asarray(labels[no:])

    def build():
        layout = BlockedTriples(proxy.layout, occupied, virtual)
        occ2, vir2 = occupied[:, None] ^ occupied, virtual[:, None] ^ virtual
        return layout, PlaneTables(layout, occupied), occ2[:, :, None, None] != vir2[None, None]
    return _cached(ground, f"blocked layout {int(irrep)}", (proxy.layout, labels), build)


def _packed(layout, t3, threads):
    """Blocked copy of a dense-slot T3 that is exactly zero in every
    symmetry-forbidden entry, else None."""
    t3 = np.ascontiguousarray(t3, dtype=np.float64)
    if layout.leakage(t3, threads=threads) != 0.0:
        return None
    return layout.pack(t3, threads=threads)


def _blocked_direction(layout, ground, dt3, irrep, threads):
    """The direction's T3 in the blocked layout of its irrep (None when it is
    not exactly of that irrep), packed once per direction. The cache refers
    to the dense T3 weakly and drops the packed copy with it, so a finished
    action's direction (a full T3) is not kept alive until the next one."""
    hit = ground.get("blocked direction")
    if hit is None or hit[0]() is not dt3 or hit[1] != irrep:
        entry = []

        def forget(_):
            if ground.get("blocked direction") is entry:
                del ground["blocked direction"]
        entry.extend((weakref.ref(dt3, forget), irrep, _packed(layout, dt3, threads)))
        ground["blocked direction"] = hit = entry
    return hit[2]


def _blocked_operands(proxy, labels, t2_0, t3_0, dt2, dt3, ground, threads, irrep=0):
    """(blocked layout, plane tables, blocked ground T3, blocked dT3) for the
    blocked residual (`_blocked_residual.triples`), else None. It needs orbital
    labels, a ground cache, a ground T2/T3 exactly zero in every
    symmetry-forbidden entry (a projected ground state) and a direction T3
    exactly of irrep `irrep`; for a totally symmetric direction also its T2.
    The ground T3 is packed once per engine (only irrep 0 uses it)."""
    plan = _blocked_layout(proxy, labels, ground, irrep)
    if plan is None or not len(t3_0):
        return None
    layout, tables, t2_forbidden = plan
    workers = _runtime.cpu_threads(threads)
    if np.any(t2_0[t2_forbidden]) or (not irrep and np.any(dt2[t2_forbidden])):
        return None
    t3_0b = None
    if not irrep:
        t3_0b = _cached(ground, "blocked ground", (t3_0,), lambda: _packed(layout, t3_0, workers))
        if t3_0b is None:
            return None
    dt3b = _blocked_direction(layout, ground, dt3, irrep, workers)
    if dt3b is None:
        return None
    return layout, tables, t3_0b, dt3b


def triples(proxy, t2, t3, have_t3, im, vooo, vvvo, g, *, threads=None, labels=None, ground=None,
            irrep=None):
    """Directional derivative of the triples residual (a plain array), as
    `_eom_triples.triples_tangent`:
    dR3 = R3(dt2, dT3; im, vooo, vvvo, W) + R3(t2, T3; dim, dvooo, dvvvo, dW).

    The triples residual is S(t2; sources) + L(operators; T3), so dR3 also
    equals R3(t2, dT3; im, dvooo, dvvvo, W) + R3(dt2, T3; dim, vooo, vvvo, dW)
    (the seeds and the T3 terms paired the other way). For a direction of
    irrep G (`irrep`) of a projected ground state, the first of these is
    totally symmetric in the occupied labels shifted by G and runs on blocked
    T3 (`_blocked_layout`); for G = 0 both run there. Otherwise, and for the
    second when G != 0, the dense-slot path runs with symmetry-blocked GEMMs
    where its operators are exactly block-diagonal."""
    o, v, nv = proxy.o, proxy.v, proxy.layout.nv
    t2_0, dt2 = _parts(t2)
    t3_0, dt3 = _parts(t3)
    g0, dg = _parts(g)
    vooo_0, dvooo = _parts(vooo)
    vvvo_0, dvvvo = _parts(vvvo)
    names = [field.name for field in fields(Intermediates)]
    im0 = Intermediates(*(_parts(getattr(im, name))[0] for name in names))
    dim = Intermediates(*(_zero_if_none(*_parts(getattr(im, name))) for name in names))
    dt2, dt3 = _zero_if_none(t2_0, dt2), _zero_if_none(t3_0, dt3)
    dg = _zero_if_none(g0, dg)
    irrep = int(irrep or 0)
    blocked = _blocked_operands(proxy, labels, t2_0, t3_0, dt2, dt3, ground, threads, irrep)
    # W = g(oovv) t2 + g(vvvv) + 0.5 (vv x 1 + 1 x vv) and its derivative.
    base = _cached(ground, "ladder", (g0, t2_0),
                   lambda: _cpu._ladder_base(g0, o, v, t2_0, nv, threads))
    w0 = _cpu.CompiledContract(threads).copy(base) if ground is not None else base
    _cpu.add_one_body(w0, im0.vv)
    # dW by the product rule, on the core executor when compiled.
    dw = _cpu._ladder_matrix(dg, o, v, t2_0, dim, nv, threads)
    if not _cpu.product_into("lmde,lmab->abde", g0[o, o, v, v], dt2, dw, 1.0, 1.0, threads):
        dw += np.einsum("lmde,lmab->abde", g0[o, o, v, v], dt2, optimize=True)
    dvooo, dvvvo = _zero_if_none(vooo_0, dvooo), _zero_if_none(vvvo_0, dvvvo)
    shifted = None
    if blocked is not None:
        from . import _blocked_residual

        layout, tables, t3_0b, dt3b = blocked
        workers = _runtime.cpu_threads(threads)
        try:
            # Ground operators on dT3 with the derivative sources' seeds.
            shifted = _blocked_residual.triples(layout, tables, t2_0, dt3b, True, im0, dvooo, dvvvo,
                                                g0, o, v, workers, wvvvv=w0)
            if not irrep:
                # Derivative operators on T3 with dT2's seeds: totally symmetric too.
                shifted += _blocked_residual.triples(layout, tables, dt2, t3_0b, have_t3, dim,
                                                     vooo_0, vvvo_0, dg, o, v, workers, wvvvv=dw)
                return layout.unpack(shifted, threads=workers)
            shifted = layout.unpack(shifted, threads=workers)
        except _blocked_residual.NotBlockable:
            ground[("blocked off", irrep)] = True       # an operator couples irreps: stay dense
            shifted = None
    options = dict(ph_irreps=proxy._ph_irreps, virtual_irreps=proxy._virtual_irreps)
    if labels is not None:
        labels = np.asarray(labels)
        occupied, virtual = labels[:proxy.layout.no], labels[proxy.layout.no:]
        options = dict(ph_irreps=(occupied[:, None] ^ virtual[None, :]).reshape(-1),
                       virtual_irreps=virtual)
    with _cpu.ThreadPool(_runtime.cpu_threads(threads)) as pool:
        # dT2's seeds with the derivative operators on T3 (the remaining part
        # of a non-A1 direction, or everything without blocked T3).
        r3 = _cpu.triples(proxy.layout, dt2, t3_0, have_t3, dim, vooo_0, vvvo_0, dg, o, v, pool,
                          wvvvv=dw, **options)
        if shifted is not None:
            r3 += shifted
        else:
            r3 += _cpu.triples(proxy.layout, t2_0, dt3, True, im0, dvooo, dvvvo, g0, o, v, pool,
                               wvvvv=w0, **options)
    return r3
