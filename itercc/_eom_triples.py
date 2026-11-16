"""Tangent-only assembly of the restricted triples response."""
from functools import partial

from . import _restricted_equations as eq
from ._cuda_packed import PackedTriples
from ._directional import DirectionalCudaLayout
from .restricted import _pair_ladder


def triples_tangent(proxy, xp, t2, t3, have_t3, im, vooo, vvvo, g, *, batch=None, projected=None, irrep=None, particle_hole=None, compiled_seed=True, fused_seed_fold=None, occupied_plan=None, stream_particle_hole=True):
    layout = proxy.layout
    fused = isinstance(layout, DirectionalCudaLayout)
    if fused:
        layout = layout.base
    crossed_update = layout.subtract_crossed_fused if fused else layout.subtract_crossed
    contract = partial(eq.contract, xp=xp)
    no, nv = layout.no, layout.nv
    if batch is None:
        batch = min(4, max(1, (48 << 20) // (no * nv**3 * 8))) if fused else 1

    if fused_seed_fold is None:
        # A one-line seed is already contiguous and does not need repacking.
        fused_seed_fold = min(batch, no) > 1

    def response(spec, left, right):
        return (contract(spec, left.tangent, right.value)
                + contract(spec, left.value, right.tangent))

    def matrices(oooo, oo, ovvo, ovov):
        occupied = oooo * 0.5
        for j in range(no):
            occupied[:, j, :, j] -= 0.5 * oo
        occupied = xp.ascontiguousarray(occupied.transpose(2, 3, 0, 1)).reshape(no**2, no**2)
        direct = xp.ascontiguousarray(ovvo.transpose(3, 1, 0, 2)).reshape(no * nv, no * nv)
        crossed = xp.ascontiguousarray(ovov.transpose(2, 1, 0, 3)).reshape(no * nv, no * nv)
        return occupied, direct * 0.25, crossed

    occupied, direct, crossed = matrices(im.oooo.value, im.oo.value, im.ovvo.value, im.ovov.value)
    doccupied, ddirect, dcrossed = matrices(im.oooo.tangent, im.oo.tangent,
                                           im.ovvo.tangent, im.ovov.tangent)
    actions = None if particle_hole is None else particle_hole.prepare_tangent(
        direct, crossed, ddirect, dcrossed, irrep)
    assembler = None
    if fused and compiled_seed:
        from ._cuda_seed import CudaTriplesSeed
        assembler = CudaTriplesSeed(xp, no, nv, min(batch, no))
    occupied_action = None
    if assembler is not None and occupied_plan is not None:
        occupied_action = occupied_plan.prepare(occupied, doccupied)
    if isinstance(t3.tangent, PackedTriples):
        # A packed direction is exactly in the target irrep; its image is
        # accumulated only in that irrep through the compact occupied fold
        # and the projected ladder.
        if (occupied_action is None or projected is None or not fused
                or t3.tangent.irrep != irrep):
            raise ValueError("Packed triples need the projected CUDA tangent path")
        residual = t3.tangent.zeros_like()
    else:
        residual = xp.zeros(layout.shape)
    for k in range(no):
        plane = layout.plane(t3.value, k, axis=2, xp=xp)
        dplane = layout.plane(t3.tangent, k, axis=2, xp=xp)
        if not fused:
            occ = occupied @ dplane.reshape(no**2, nv**3)
            occ += doccupied @ plane.reshape(no**2, nv**3)
            occ = occ.reshape(plane.shape)
        compact_occ = None if occupied_action is None else occupied_action.plane(plane, dplane, k)
        if assembler is not None:
            doubles = assembler.prepare_doubles(t2, k)
        # At large one-line tiles, keep only one particle-hole group live.
        groups = (actions.owner.groups[k] if actions is not None and compact_occ is not None
                  and batch == 1 and stream_particle_hole else (None,))
        for group in groups:
            if actions is not None:
                actions.prepare_plane(plane, dplane, k, group=group)
            starts = range(0, no, batch) if group is None else group
            for start in starts:
                stop = min(start + batch, no)
                width = stop - start
                selection = slice(start, stop)
                left, dleft = plane[:, selection], dplane[:, selection]
                if assembler is not None:
                    seed = assembler.assemble(selection, doubles, vvvo, vooo, im.vv,
                                              plane, dplane, occupied, doccupied,
                                              include_occupied=compact_occ is None,
                                              include_virtual=False)
                else:
                    seed = response("abdj,idc->ijabc", vvvo[:, :, :, selection], t2[:, k])
                    seed -= response("alij,lbc->ijabc", vooo[:, :, :, selection], t2[:, k])
                    seed += 0.5 * (contract("ad,ijdbc->ijabc", im.vv.value, dleft)
                                   + contract("ad,ijdbc->ijabc", im.vv.tangent, left))
                if actions is None:
                    rhs_shape = no * nv, width * nv**2
                    image_shape = no, nv, width, nv, nv
                    ds = layout.spin_sum(dleft).transpose(0, 2, 1, 3, 4).reshape(rhs_shape)
                    tile = direct @ ds
                    del ds
                    source = layout.spin_sum(left).transpose(0, 2, 1, 3, 4).reshape(rhs_shape)
                    tile += ddirect @ source
                    del source
                    seed += tile.reshape(image_shape).transpose(0, 2, 1, 3, 4)
                    tile = crossed @ dplane[selection].transpose(1, 2, 0, 3, 4).reshape(rhs_shape)
                    tile += dcrossed @ plane[selection].transpose(1, 2, 0, 3, 4).reshape(rhs_shape)
                    crossed_tiles = tile.reshape(image_shape).transpose(0, 2, 1, 3, 4)
                    del tile
                if assembler is None and fused:
                    # Evaluate the occupied ladder for the seed being folded.
                    # A full output plane and its second product are unnecessary.
                    shape = no * width, no**2
                    rows = occupied.reshape(no, no, no**2)[:, selection].reshape(shape)
                    drows = doccupied.reshape(no, no, no**2)[:, selection].reshape(shape)
                    tile = rows @ dplane.reshape(no**2, nv**3)
                    seed += tile.reshape(seed.shape)
                    del tile
                    tile = drows @ plane.reshape(no**2, nv**3)
                    seed += tile.reshape(seed.shape)
                    del tile
                elif assembler is None:
                    seed += occ[:, selection]
                for j in range(width):
                    if actions is not None and assembler is not None and fused_seed_fold and compact_occ is None:
                        actions.fold(residual, seed, j, start + j, k, layout)
                    else:
                        if actions is None:
                            crossed_update(seed[:, j], crossed_tiles[:, j])
                        else:
                            actions.add(seed[:, j], start + j)
                        if compact_occ is None:
                            layout.add_seed(residual, seed[:, j], start + j, k)
                        else:
                            layout.add_seed(residual, seed[:, j], start + j, k, occupied=compact_occ)
                del seed, left, dleft
                if actions is None:
                    del crossed_tiles
        del plane, dplane, compact_occ
        if actions is not None:
            actions.release_plane()

    fold_virtual = assembler is not None
    del assembler, occupied_action
    o, v = proxy.o, proxy.v
    oovv = g[o, o, v, v]
    def add_ladder(source, matrix, source_irrep=None):
        if projected is None:
            _pair_ladder(residual, source, matrix, proxy.ladder_batch, xp=xp, fuse_cuda=fused)
        else:
            # The compact products per batch are far smaller than a dense
            # tile; wider batches give the pair GEMMs a longer N dimension.
            ladder_batch = max(proxy.ladder_batch, min(48, (128 << 20) // (nv**3 * 8)))
            projected.add(residual, source, matrix, irrep, ladder_batch, source_irrep=source_irrep)

    # The compiled seed omits the virtual one-body term 1/2 vv(a,d) T(i j k, d b c).
    # Folded over the six (ia, jb, kc) orderings it equals the pair operator
    # 1/2 (vv x 1 + 1 x vv) added to W in each of the three ladder orientations,
    # as the occupied one-body term is folded into the occupied ladder above.
    def one_body(matrix, vv):
        if fold_virtual:
            diagonal = xp.arange(nv)
            half = 0.5 * vv
            matrix[:, diagonal, :, diagonal] += half
            matrix[diagonal, :, diagonal, :] += half
        return matrix

    w = contract("lmde,lmab->abde", oovv.value, t2.value)
    w += g.value[v, v, v, v]
    add_ladder(t3.tangent, one_body(w, im.vv.value), irrep)
    del w
    dw = contract("lmde,lmab->abde", oovv.tangent, t2.value)
    dw += contract("lmde,lmab->abde", oovv.value, t2.tangent)
    dw += g.tangent[v, v, v, v]
    add_ladder(t3.value, one_body(dw, im.vv.tangent))
    layout.purify(residual, xp=xp)
    return residual
