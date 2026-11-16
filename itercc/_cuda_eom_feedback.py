"""Tiled CUDA contractions for feedback from frozen triples amplitudes."""

from functools import cache, partial

import numpy as np

from ._restricted_equations import contract


@cache
def _kernels():
    from cupy import ElementwiseKernel

    gather = ElementwiseKernel(
        "raw float64 plane, raw int32 rows, raw int32 columns, int32 width, "
        "int32 no, int32 nv, int32 nv3, bool source", "float64 packed",
        r"""
        const int row = rows[i / width], col = columns[i % width];
        const int u = row / nv, v = row % nv;
        long long index;
        if (source) {
            const int ab = col / no, j = col % no;
            index = (static_cast<long long>(j) * no + u) * nv3
                  + (v * nv + ab % nv) * nv + ab / nv;
        } else {
            const int ij = col / nv, a = col % nv;
            index = static_cast<long long>(ij) * nv3 + (v * nv + a) * nv + u;
        }
        packed = plane[index];
        """, "itercc_eom_feedback_pack")
    scatter = ElementwiseKernel(
        "float64 value, raw int32 rows, raw int32 columns, int32 width, "
        "int32 no, int32 nv, bool source", "raw float64 result",
        r"""
        const int row = rows[i / width], col = columns[i % width];
        if (source) {
            const long long offset = (static_cast<long long>(col / no) * nv + row) * no + col % no;
            result[offset] -= value;
        } else {
            result[static_cast<long long>(col) * nv + row] += value;
        }
        """, "itercc_eom_feedback_add")
    return gather, scatter


def _singles(operator, plane, xp):
    """0.5 * einsum('jbc,ijcab->ia', operator, 2 plane - plane.swapaxes(-2, -1)).

    Two batched GEMVs read the plane in place; the exchange-summed plane and
    the einsum transposes (each a full plane copy) are never materialized.
    """
    no, nv = plane.shape[0], plane.shape[2]
    lines = plane.reshape(no, no * nv, nv, nv)
    weights = xp.ascontiguousarray(operator.transpose(0, 2, 1)).reshape(no * nv, nv)
    direct = xp.matmul(lines, weights[:, :, None])[..., 0].sum(axis=1)
    exchange = xp.matmul(weights[:, None, :], lines)[..., 0, :].sum(axis=1)
    return direct - 0.5 * exchange


def _fock_doubles(vector, plane, xp):
    """einsum('c,ijcab->ijab', vector, plane) as one batched GEMV."""
    no, nv = plane.shape[0], plane.shape[2]
    return xp.matmul(vector[None, :], plane.reshape(no * no, nv, nv * nv)).reshape(no, no, nv, nv)


def _occupied_doubles(operator, plane, xp):
    """einsum('kjc,ikcab->ijab', operator, plane) without transposing the plane."""
    no, nv = plane.shape[0], plane.shape[2]
    rows = xp.ascontiguousarray(operator.transpose(1, 0, 2)).reshape(no, no * nv)
    return xp.matmul(rows, plane.reshape(no, no * nv, nv * nv)).reshape(no, no, nv, nv)


class CudaTangentFeedback:
    """Differentiate the feedback while reusing its frozen source terms."""

    def __init__(self, occupied, virtual, xp):
        self.xp = xp
        self.no, self.nv = len(occupied), len(virtual)
        self.occupied = occupied
        pairs = virtual[:, None] ^ virtual
        ov = occupied[:, None] ^ virtual
        oov = occupied[:, None, None] ^ occupied[None, :, None] ^ virtual
        vvo = pairs[:, :, None] ^ occupied
        self.virtual_rows = self._sectors(virtual)
        self.pair_rows, self.ov_rows = self._sectors(pairs), self._sectors(ov)
        self.doubles_columns = self._sectors(oov)
        self.source_columns = self._sectors(vvo)
        self.all_virtual = xp.arange(self.nv, dtype=xp.int32)
        self.all_pairs = xp.arange(self.nv**2, dtype=xp.int32)
        self.all_ov = xp.arange(self.no*self.nv, dtype=xp.int32)
        self.primal_sources = None

    def _sectors(self, labels):
        flat = labels.ravel()
        return tuple(self.xp.asarray(np.flatnonzero(flat == sector), dtype=self.xp.int32)
                     for sector in range(8))

    def _products(self, out, ground, tangent, operator, derivative, groups, all_rows,
                  scratch, *, source):
        xp, no, nv = self.xp, self.no, self.nv
        gather, scatter = _kernels()
        operator = xp.ascontiguousarray(operator).reshape(nv, -1)
        derivative = xp.ascontiguousarray(derivative).reshape(nv, -1)
        # One reusable tile holds the unrestricted input rows when required.
        chunk = scratch.size // len(all_rows)
        for rows, inner, columns in groups:
            if not len(rows) or not len(columns):
                continue
            direct = operator[rows[:, None], inner]
            deriv = derivative[rows]
            for start in range(0, len(columns), chunk):
                selection = columns[start:start+chunk]
                width = len(selection)
                packed = scratch[:len(all_rows)*width].reshape(len(all_rows), width)
                gather(ground, all_rows, selection, width, no, nv, nv**3, source, packed)
                value = deriv @ packed
                if len(inner):
                    packed = scratch[:len(inner)*width].reshape(len(inner), width)
                    gather(tangent, inner, selection, width, no, nv, nv**3, source, packed)
                    value += direct @ packed
                scatter(value, rows, selection, width, no, nv, source, out)

    def apply(self, r1, r2, vooo, vvvo, f, g, layout, t3, o, v, *, irrep):
        xp = self.xp
        no, nv = self.no, self.nv
        ct = partial(contract, xp=xp)
        scratch = xp.empty(32 * 2**20 // 8)
        original = layout.base
        make_primal = self.primal_sources is None
        if not make_primal:
            # Reuse the complete frozen sources rather than retaining a
            # second copy of the feedback increments during seed assembly.
            vooo.value, vvvo.value = self.primal_sources
        for m in range(no):
            ground = original.spin_plane(t3.value, m)
            tangent = original.spin_plane(t3.tangent, m)
            oovv = g[o, m, v, v]
            r1.tangent += _singles(oovv.value, tangent, xp)
            r1.tangent += _singles(oovv.tangent, ground, xp)
            fov = f[m, v]
            r2.tangent += 0.5 * _fock_doubles(fov.value, tangent, xp)
            r2.tangent += 0.5 * _fock_doubles(fov.tangent, ground, xp)
            ooov = g[o, m, o, v]
            r2.tangent -= _occupied_doubles(ooov.value, tangent, xp)
            r2.tangent -= _occupied_doubles(ooov.tangent, ground, xp)
            vooo.tangent += ct("lde,ijead->alij", oovv.value, tangent)
            vooo.tangent += ct("lde,ijead->alij", oovv.tangent, ground)
            if make_primal:
                vooo.value += ct("lde,ijead->alij", oovv.value, ground)
                vvvo.value -= ct("lde,jleba->abdj", oovv.value, ground)
            # Doubles are subsequently packed in the target irrep. Both
            # tangent source terms keep all output rows d for VVVO.
            label_m = int(self.occupied[m])
            doubles = tuple((rows, self.pair_rows[sector ^ label_m],
                             self.doubles_columns[sector ^ irrep])
                            for sector, rows in enumerate(self.virtual_rows))
            vovv = g[v, m, v, v]
            self._products(r2.tangent, ground, tangent, vovv.value, vovv.tangent,
                           doubles, self.all_pairs, scratch, source=False)
            sources = tuple((self.all_virtual, self.ov_rows[sector ^ label_m ^ irrep], columns)
                            for sector, columns in enumerate(self.source_columns))
            self._products(vvvo.tangent, ground, tangent, oovv.value.transpose(1, 0, 2),
                           oovv.tangent.transpose(1, 0, 2), sources, self.all_ov,
                           scratch, source=True)
            del ground, tangent
        if make_primal:
            self.primal_sources = vooo.value, vvvo.value
