"""Device-local occupied tiling and permutation kernels for RCCSDTQ.

T4 and R4 live in one flat array. Each occupied tuple i <= j <= k <= l (a
slot) owns a contiguous segment with the virtual quadruples (a, b, c, d) that
Abelian orbital symmetry allows: irrep(a)^irrep(b)^irrep(c)^irrep(d) equals
irrep(i)^irrep(j)^irrep(k)^irrep(l). Inside a segment the entries are ordered
by irrep block (g_a, g_b, g_c) and row-major within each block, so virtual
orbitals must be sorted by irrep. Without labels every slot holds all nv**4
entries in row-major order, which is the dense packed layout flattened.

CuPy is imported on first use and the kernels are built on first launch, so
the layout planning (and the NumPy backend in `_quadruples_cpu`, which
shares it) works without CuPy.
"""
from functools import lru_cache
from itertools import combinations_with_replacement, permutations, product

import numpy as np

from .gemm_contract import einsum_into


class _CuPy:
    """The cupy module, imported on first attribute access."""

    def __getattr__(self, name):
        import cupy

        value = getattr(cupy, name)
        setattr(self, name, value)
        return value


cp = _CuPy()


def _lazy(kind, *args, **kwargs):
    """A CuPy kernel (`cp.<kind>(*args, **kwargs)`) built on its first launch."""
    built = []

    def launch(*call_args, **call_kwargs):
        if not built:
            built.append(getattr(cp, kind)(*args, **kwargs))
        return built[0](*call_args, **call_kwargs)
    return launch


_T3_DIVIDE = _lazy("ElementwiseKernel", "raw float64 eia, int64 no, int64 nv", "float64 r", """
    const long long nv2 = nv * nv, nv3 = nv2 * nv, ijk = i / nv3, abc = i - ijk * nv3;
    const long long a = abc / nv2, b = (abc / nv) % nv, c = abc % nv;
    const long long p = ijk / (no * no), q = (ijk / no) % no, s = ijk % no;
    r = r / ((eia[p * nv + a] + eia[q * nv + b]) + eia[s * nv + c]);
""", "itercc_t3_divide")


def divide_triples(r3, eia):
    """r3 /= (eia[i,a] + eia[j,b]) + eia[k,c] on the device, as dividing by the
    dense d3 (same sums, IEEE division) without forming it."""
    no, nv = eia.shape
    if r3.flags.c_contiguous and r3.shape == (no,) * 3 + (nv,) * 3 and r3.dtype == np.float64:
        _T3_DIVIDE(cp.ascontiguousarray(eia, dtype=cp.float64), no, nv, r3)
    else:
        d2 = eia[:, None, :, None] + eia[None, :, None, :]
        r3 /= d2[:, :, None, :, :, None] + eia[None, None, :, None, None, :]
    return r3


def prange(start, stop, block):
    for first in range(start, stop, block):
        yield first, min(first + block, stop)


def einsum(spec, *operands, out=None, alpha=1.0, beta=0.0):
    """out = beta * out + alpha * einsum(spec, *operands).

    Two-operand products into `out` take the faster of direct GEMMs and
    einsum for their shape; other contractions use einsum and an update.
    """
    if out is not None and len(operands) == 2:
        return einsum_into(spec, *operands, out, alpha, beta)
    value = cp.einsum(spec, *operands, optimize=True)
    if alpha != 1.0:
        # Some unary contractions return views of their inputs.
        value = value * alpha
    if out is None:
        return value
    if beta == 0.0:
        out[...] = value
    else:
        if beta != 1.0:
            out *= beta
        out += value
    return out


_DIGITS = """
    const int d = i % nv;
    const int c = (i / nv) % nv;
    const int b = (i / nv / nv) % nv;
    const int a = (i / nv / nv / nv) % nv;
"""
# Offset of (a, b, c, d) inside a slot segment of total irrep `irrep`, or -1.
_OFFSET = """
__device__ __forceinline__ long long quad_offset(
        const int* virr, const int* vloc, const int* vsize, const long long* boff,
        int nirrep, int irrep, int a, int b, int c, int d) {
    const int ga = virr[a], gb = virr[b], gc = virr[c], gd = virr[d];
    if ((ga ^ gb ^ gc ^ gd) != irrep) return -1;
    return boff[((irrep * nirrep + ga) * nirrep + gb) * nirrep + gc]
        + (((long long)vloc[a] * vsize[gb] + vloc[b]) * vsize[gc] + vloc[c]) * vsize[gd] + vloc[d];
}
"""
_TABLES = ("raw int32 virr, raw int32 vloc, raw int32 vsize, raw int64 boff, "
           "raw int64 soff, raw int32 sirr, int32 nirrep")
_AT = "&virr[0], &vloc[0], &vsize[0], &boff[0], nirrep"
_GATHER = _lazy("ElementwiseKernel",
    "raw float64 packed, raw int32 plan, " + _TABLES + ", int32 nv, int32 nv4, bool triples",
    "float64 tile",
    _DIGITS + f"""
    const long long row = (i / nv4) * 5;
    const int slot = plan[row];
    tile = 0.0;
    if (slot >= 0) {{
        const long long base = soff[slot];
        const int g = sirr[slot];
        const int p[4] = {{plan[row+1], plan[row+2], plan[row+3], plan[row+4]}};
        const int x[4] = {{a, b, c, d}};
        const long long q = quad_offset({_AT}, g, x[p[0]], x[p[1]], x[p[2]], x[p[3]]);
        if (q >= 0) {{
            tile = packed[base + q];
            if (triples) {{
                const int y[4] = {{c, a, b, d}};
                const int z[4] = {{d, a, c, b}};
                tile += packed[base + quad_offset({_AT}, g, y[p[0]], y[p[1]], y[p[2]], y[p[3]])]
                      + packed[base + quad_offset({_AT}, g, z[p[0]], z[p[1]], z[p[2]], z[p[3]])];
            }}
        }}
    }}
    """, "itercc_quad_gather", preamble=_OFFSET,
)
# Tile rows from packed slots, line box by line box: with the virtuals sorted by
# irrep, the entries of one tile row with virtual irreps (ga, gb, gc) on a, b, c
# and any d form contiguous runs of nc * nv doubles (one per a, b); the d of the
# allowed irrep gd come from one dense block of the packed slot, read with the
# row's occupied permutation applied to the axes, and the other d are zero.
# A row is (tile offset, packed slot offset, template); a template holds, for
# every (ga, gb, gc), the box fields relative to the row: tile offset, na, nb,
# nc, d start, nd, then per term the packed block offset and the packed strides
# of the tile axes a, b, c, d. The entry is the weighted sum of those reads (a
# spin summation, say), or the plain sum of three for the ring terms. Templates
# depend only on the row's occupied permutation and slot irrep; rows outside the
# bounds use a template that only writes zeros.
_GATHER_BOXES = _lazy("RawKernel", r"""
extern "C" __global__ void itercc_quad_gather_boxes(
        const double* __restrict__ packed, double* __restrict__ tile,
        const long long* __restrict__ rows, const long long* __restrict__ templates,
        const long long nbox, const int per_row, const int terms, const int nv,
        const int ring, const double* __restrict__ weights) {
    const int width = 6 + 5 * terms;
    for (long long x = blockIdx.z; x < nbox; x += gridDim.z) {
        const long long* row = rows + 3 * (x / per_row);
        const long long* box = templates + (row[2] * per_row + x % per_row) * width;
        const int na = (int)box[1], nb = (int)box[2], nc = (int)box[3];
        const int d0 = (int)box[4], nd = (int)box[5];
        const long long toff = row[0] + box[0];
        const double* src = packed + row[1];
        const long long* q = box + 6;
        const int nab = na * nb, ncv = nc * nv;
        for (int ab = blockIdx.y * blockDim.y + threadIdx.y; ab < nab; ab += gridDim.y * blockDim.y) {
            const int a = ab / nb, b = ab - a * nb;
            double* line = tile + toff + ((long long)a * nv + b) * nv * nv;
            for (int cd = blockIdx.x * blockDim.x + threadIdx.x; cd < ncv; cd += gridDim.x * blockDim.x) {
                const int c = cd / nv, d = cd - c * nv - d0;
                double v = 0.0;
                if (d >= 0 && d < nd) {
                    if (ring) {   // same association as the per-entry kernel
                        v = src[q[0] + a * q[1] + b * q[2] + c * q[3] + d * q[4]]
                          + (src[q[5] + a * q[6] + b * q[7] + c * q[8] + d * q[9]]
                             + src[q[10] + a * q[11] + b * q[12] + c * q[13] + d * q[14]]);
                    } else {
                        for (int t = 0; t < terms; ++t) {
                            const long long* r = q + 5 * t;
                            v += weights[t] * src[r[0] + a * r[1] + b * r[2] + c * r[3] + d * r[4]];
                        }
                    }
                }
                line[cd] = v;
            }
        }
    }
}
""", "itercc_quad_gather_boxes")
# Virtual permutations summed by the ring terms: (a, b, c, d), (c, a, b, d), (d, a, c, b).
_RING_TERMS = ((0, 1, 2, 3), (2, 0, 1, 3), (3, 0, 2, 1))


@lru_cache(maxsize=None)
def _gather_terms(kind):
    """(virtual permutation, weight) pairs read by a tile gather of `kind`."""
    if kind == "plain":
        return (((0, 1, 2, 3), 1.0),)
    if kind == "triples":
        return tuple((perm, 1.0) for perm in _RING_TERMS)
    perms, weights, divisor = _SPIN[kind]
    return tuple((perm, w / divisor) for perm, w in zip(perms, weights) if w)


@lru_cache(maxsize=None)
def _gather_weights(kind, device):
    with cp.cuda.Device(device):
        # The cached array may next be used on a different stream.
        return cp.asarray([w for _, w in _gather_terms(kind)], dtype=cp.float64, blocking=True)
_SCATTER = _lazy("ElementwiseKernel",
    "raw float64 tile, raw int32 plan, " + _TABLES + ", int32 nv, int32 nv4, float64 alpha, float64 beta",
    "raw float64 packed",
    _DIGITS + f"""
    const int slot = plan[(i / nv4) * 5];
    if (slot >= 0) {{
        const long long q = quad_offset({_AT}, sirr[slot], a, b, c, d);
        if (q >= 0) {{
            const long long pos = soff[slot] + q;
            const double old = beta == 0.0 ? 0.0 : beta * packed[pos];
            packed[pos] = old + alpha * tile[i];
        }}
    }}
    """, "itercc_quad_scatter", preamble=_OFFSET,
)
# Copy the allowed entries of selected tile rows into consecutive slot segments.
_ROWS = _lazy("ElementwiseKernel",
    "raw float64 tile, raw int32 rows, raw int32 rslot, raw int64 roff, " + _TABLES
    + ", int32 nv, int32 nv4, float64 alpha",
    "raw float64 buffer",
    f"""
    const long long r = i / nv4;
    const int e = i % nv4;
    const int d = e % nv;
    const int c = (e / nv) % nv;
    const int b = (e / nv / nv) % nv;
    const int a = e / nv / nv / nv;
    const int slot = rslot[r];
    const long long q = quad_offset({_AT}, sirr[slot], a, b, c, d);
    if (q >= 0) {{
        buffer[roff[r] + q] = alpha * tile[(long long)rows[r] * nv4 + e];
    }}
    """, "itercc_quad_rows", preamble=_OFFSET,
)
_EXPAND = _lazy("ElementwiseKernel",
    "raw float64 packed, " + _TABLES + ", int32 nv, int32 nv4, int32 first, int64 shift",
    "float64 dense",
    _DIGITS + f"""
    const int slot = first + (int)(i / nv4);
    const long long q = quad_offset({_AT}, sirr[slot], a, b, c, d);
    dense = q >= 0 ? packed[soff[slot] - shift + q] : 0.0;
    """, "itercc_quad_expand", preamble=_OFFSET,
)
_COMPRESS = _lazy("ElementwiseKernel",
    "raw float64 dense, " + _TABLES + ", int32 nv, int32 nv4, int32 first, int64 shift",
    "raw float64 packed",
    _DIGITS + f"""
    const int slot = first + (int)(i / nv4);
    const long long q = quad_offset({_AT}, sirr[slot], a, b, c, d);
    if (q >= 0) {{
        packed[soff[slot] - shift + q] = dense[i];
    }}
    """, "itercc_quad_compress", preamble=_OFFSET,
)
_PURIFY = _lazy("ElementwiseKernel",
    "raw bool invalid, int32 nv, int32 nv4", "float64 value",
    _DIGITS + """
    if (invalid[i / nv4] || (a == b && (b == c || b == d))
        || (c == d && (a == c || b == c))) {
        value = 0.0;
    }
    """, "itercc_quad_purify",
)
_STEP = _lazy("ElementwiseKernel",
    "raw float64 eia, raw int32 slots, int32 nv, int32 nv4",
    "float64 residual",
    _DIGITS + """
    const long long row = 4 * (i / nv4);
    const double denom = eia[slots[row] * nv + a] + eia[slots[row+1] * nv + b]
                       + eia[slots[row+2] * nv + c] + eia[slots[row+3] * nv + d];
    residual /= denom;
    """, "itercc_quad_jacobi",
)


# R4(o; v) += Y[row, v[m0], v[m1], v[m2], v[m3]] for entries (slot of o, row, m0, m1, m2, m3).
_ROW_ADD = _lazy("ElementwiseKernel",
    "raw float64 rows, raw int32 entries, " + _TABLES + ", int32 nv, int32 nv4",
    "raw float64 packed",
    f"""
    const long long e = i / nv4;
    const int v = i % nv4;
    const int x[4] = {{v / nv / nv / nv, (v / nv / nv) % nv, (v / nv) % nv, v % nv}};
    const int* entry = &entries[e * 6];
    const int slot = entry[0];
    const long long q = quad_offset({_AT}, sirr[slot], x[0], x[1], x[2], x[3]);
    if (q >= 0) {{
        const long long source = (long long)entry[1] * nv4
            + ((((long long)x[entry[2]] * nv + x[entry[3]]) * nv + x[entry[4]]) * nv + x[entry[5]]);
        packed[soff[slot] + q] += rows[source];
    }}
    """, "itercc_quad_row_add", preamble=_OFFSET,
)
# The six ways to split occupied positions into a target pair P and a spectator pair Q.
_SPLITS = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2), (2, 3, 0, 1))


def _segment(source, window):
    """Device copy of the flattened elements `window` of a device, host or on-disk array."""
    if isinstance(source, cp.ndarray):
        return source.reshape(-1)[window]
    if hasattr(source, "read"):
        return cp.asarray(source.read(window.start, window.stop))
    return cp.asarray(np.ascontiguousarray(np.asarray(source).reshape(-1)[window]))


def _labels(values, count, name):
    labels = np.zeros(count, dtype=np.int64) if values is None else np.asarray(values)
    if labels.shape != (count,) or labels.dtype.kind not in "iu" or np.any(labels < 0) or np.any(labels > 7):
        raise ValueError(f"Expected one Abelian label 0..7 per {name} orbital")
    return labels.astype(np.int64)


class QuadruplesLayout:
    """Symmetry-compressed storage of T4 or R4 with i <= j <= k <= l slots."""

    def __init__(self, nocc, nvir, occupied_irreps=None, virtual_irreps=None):
        occupied = _labels(occupied_irreps, nocc, "occupied")
        virtual = _labels(virtual_irreps, nvir, "virtual")
        if np.any(np.diff(virtual) < 0):
            raise ValueError("Virtual orbitals must be sorted by irrep")
        nirrep = 1
        while nirrep <= max(int(occupied.max(initial=0)), int(virtual.max(initial=0))):
            nirrep *= 2
        self.nocc = nocc
        self.nvir = nvir
        self.nv4 = nvir**4
        self.nirrep = nirrep
        self.symmetric = nirrep > 1
        self.occupied_irreps = occupied
        self.virtual_irreps = virtual
        self.indices = tuple(combinations_with_replacement(range(nocc), 4))
        self._slot = {key: idx for idx, key in enumerate(self.indices)}
        self.dense_shape = (len(self.indices),) + (nvir,) * 4

        sizes = np.bincount(virtual, minlength=nirrep).astype(np.int64)
        first = np.r_[0, np.cumsum(sizes)[:-1]]
        local = np.arange(nvir) - first[virtual]
        blocks = np.zeros((nirrep,) * 4, dtype=np.int64)
        slot_size = np.zeros(nirrep, dtype=np.int64)
        for irrep in range(nirrep):
            offset = 0
            for g0, g1, g2 in product(range(nirrep), repeat=3):
                blocks[irrep, g0, g1, g2] = offset
                offset += sizes[g0] * sizes[g1] * sizes[g2] * sizes[irrep ^ g0 ^ g1 ^ g2]
            slot_size[irrep] = offset
        keys = np.asarray(self.indices, dtype=np.int64).reshape(-1, 4)
        slot_irrep = np.bitwise_xor.reduce(occupied[keys], axis=1) if len(keys) else keys[:, 0]
        self.slot_irreps = slot_irrep
        self.offsets = np.r_[0, np.cumsum(slot_size[slot_irrep])].astype(np.int64)
        self.size = int(self.offsets[-1])
        self.shape = (self.size,)
        self._box_tables = (sizes, first, blocks.ravel())
        array = self._array
        self._tables = (array(virtual.astype(np.int32)), array(local.astype(np.int32)),
                        array(sizes.astype(np.int32)), array(blocks.ravel()),
                        array(self.offsets), array(slot_irrep.astype(np.int32)), nirrep)

        self._plans = {}
        self._host_scatter_plans = {}
        self._slots = array(np.asarray(self.indices, dtype=np.int32))
        self._invalid = array(np.asarray([key[0] == key[2] or key[1] == key[3]
                                          for key in self.indices], dtype=np.bool_))
        self._host_ties = tuple(np.asarray([idx for idx, key in enumerate(self.indices)
                                            if key[p] == key[p + 1]], dtype=np.int32)
                                for p in range(3))

    @staticmethod
    def _array(host):
        """An index table where the layout's kernels run (the device here)."""
        return cp.asarray(host)

    @property
    def transfer_slots(self):
        return max(1, (256 * 2**20) // (8 * self.nv4))

    def batches(self):
        """Slot ranges whose dense form fits the transfer buffer."""
        return prange(0, len(self.indices), self.transfer_slots)

    def segment(self, first, stop):
        return slice(int(self.offsets[first]), int(self.offsets[stop]))

    def zeros(self, *, host=False):
        return np.zeros(self.size) if host else cp.zeros(self.size)

    def _plan(self, bounds, tile_shape, scatter):
        key = (bounds, tile_shape, scatter)
        if key not in self._plans:
            start, stop = bounds[::2], bounds[1::2]
            rows = np.full((int(np.prod(tile_shape)), 5), -1, dtype=np.int32)
            identity = (0, 1, 2, 3)
            for row, local in enumerate(np.ndindex(tile_shape)):
                actual = tuple(origin + index for origin, index in zip(start, local))
                if any(index >= edge for index, edge in zip(actual, stop)):
                    continue
                order = tuple(sorted(identity, key=actual.__getitem__))
                if not scatter or order == identity:
                    rows[row] = (self._slot[tuple(actual[k] for k in order)], *order)
            self._plans[key] = self._array(rows)
        return self._plans[key]

    def _templates(self, kind):
        """Box templates of one term kind: (device table, boxes per row, grid y).

        Template `perm_index * nirrep + irrep` serves rows whose occupied
        permutation is `_PERMS4[perm_index]` and whose slot irrep is `irrep`; the
        last one writes a zero row."""
        key = ("templates", kind)
        if key not in self._plans:
            terms = [np.asarray(perm) for perm, _ in _gather_terms(kind)]
            width = 6 + 5 * len(terms)
            sizes, first, blocks = self._box_tables
            nirrep, nv = self.nirrep, self.nvir
            perm = np.repeat(np.asarray(_PERMS4, dtype=np.int64), nirrep, axis=0)
            irrep = np.tile(np.arange(nirrep, dtype=np.int64), len(_PERMS4))
            triple = np.array(list(product(range(nirrep), repeat=3)), dtype=np.int64)
            # Tile irreps of axes a, b, c, d for every template and box.
            tile_irreps = np.empty((len(perm), len(triple), 4), dtype=np.int64)
            tile_irreps[..., :3] = triple
            tile_irreps[..., 3] = irrep[:, None] ^ np.bitwise_xor.reduce(triple, axis=1)
            dims = sizes[tile_irreps]
            start = first[tile_irreps]
            fields = [((start[..., 0] * nv + start[..., 1]) * nv + start[..., 2]) * nv,
                      dims[..., 0], dims[..., 1], dims[..., 2], start[..., 3], dims[..., 3]]
            for term in terms:
                # Packed position k holds tile axis axes[k] (occupied order, then the term).
                axes = np.broadcast_to(term[perm][:, None, :], tile_irreps.shape)
                packed_irreps = np.take_along_axis(tile_irreps, axes, axis=2)
                s = sizes[packed_irreps]
                packed_strides = np.stack([s[..., 1] * s[..., 2] * s[..., 3], s[..., 2] * s[..., 3],
                                           s[..., 3], np.ones_like(s[..., 3])], axis=-1)
                strides = np.empty_like(packed_strides)
                np.put_along_axis(strides, axes, packed_strides, axis=2)
                block = ((irrep[:, None] * nirrep + packed_irreps[..., 0]) * nirrep
                         + packed_irreps[..., 1]) * nirrep + packed_irreps[..., 2]
                fields += [blocks[block], *np.moveaxis(strides, -1, 0)]
            table = np.stack(fields, axis=-1)
            # Lines with an empty a, b or c block write nothing; an empty d block writes zeros.
            table[~np.all(dims[..., :3] > 0, axis=2), 1:4] = 0
            zero = np.zeros((1, len(triple), width), dtype=np.int64)
            zero[0, 0, 1:4] = nv
            table = np.concatenate([table, zero])
            nab = int(np.max(sizes)) ** 2
            self._plans[key] = (self._array(np.ascontiguousarray(table)), len(triple),
                                min(-(-nab // 8), 65535))
        return self._plans[key]

    def _boxes(self, bounds, tile_shape, kind):
        """Cache the grid and invariant kernel arguments for one tile and term kind."""
        key = ("boxes", bounds, tile_shape, kind)
        if key not in self._plans:
            templates, per_row, grid_y = self._templates(kind)
            plan = self._plan(bounds, tile_shape, False)
            plan = (plan if isinstance(plan, np.ndarray) else cp.asnumpy(plan)).astype(np.int64)
            slot = plan[:, 0]
            valid = slot >= 0
            index = {perm: i for i, perm in enumerate(_PERMS4)}
            perm_index = np.array([index[tuple(p)] if ok else 0 for p, ok in zip(plan[:, 1:].tolist(), valid)],
                                  dtype=np.int64)
            safe = np.maximum(slot, 0)
            rows = np.empty((len(plan), 3), dtype=np.int64)
            rows[:, 0] = np.arange(len(plan)) * self.nv4
            rows[:, 1] = np.where(valid, self.offsets[safe], 0)
            rows[:, 2] = np.where(valid, perm_index * self.nirrep + self.slot_irreps[safe],
                                  len(_PERMS4) * self.nirrep)
            nbox = len(plan) * per_row
            grid = (1, grid_y, min(nbox, 65535))
            arguments = (self._array(rows), templates, np.int64(nbox), np.int32(per_row),
                         np.int32(len(_gather_terms(kind))), np.int32(self.nvir),
                         np.int32(kind == "triples"), _gather_weights(kind, cp.cuda.Device().id))
            self._plans[key] = (grid, arguments)
        return self._plans[key]

    def gather(self, packed, tile, bounds, *, triples=False, spin=None, allowed_only=False):
        """Unpack the occupied tile `bounds` into dense `tile` (rows, nv, nv, nv, nv).

        With `triples`, each entry is the sum over the virtual permutations
        (a, b, c, d), (c, a, b, d) and (d, a, c, b) used by the ring terms; with
        `spin` (a pattern of `_SPIN`), the tile is the spin summation P(T4) of
        the plain tile, read straight from the packed amplitudes. Symmetry-
        forbidden entries and rows outside the bounds are zero; every entry is
        written once, line box by line box (see `_GATHER_BOXES`).
        """
        if not tile.flags.c_contiguous:
            raise ValueError("gather needs a C-contiguous tile")
        kind = spin or ("triples" if triples else "plain")
        grid, arguments = self._boxes(bounds, tile.shape[:4], kind)
        _GATHER_BOXES(grid, (32, 8, 1), (packed, tile, *arguments))

    def scatter(self, packed, tile, bounds, *, alpha, beta):
        plan = self._plan(bounds, tile.shape[:4], True)
        if isinstance(packed, cp.ndarray):
            _SCATTER(tile, plan, *self._tables, self.nvir, self.nv4, alpha, beta, packed,
                     size=tile.size)
            return
        key = (bounds, tile.shape[:4])
        if key not in self._host_scatter_plans:
            slots = plan[:, 0].get()
            rows = np.flatnonzero(slots >= 0)
            chunks = []
            for first, stop in prange(0, rows.size, self.transfer_slots):
                chosen = slots[rows[first:stop]]
                sizes = np.diff(self.offsets)[chosen]
                local = np.r_[0, np.cumsum(sizes)]
                chunks.append((cp.asarray(rows[first:stop], dtype=cp.int32),
                               cp.asarray(chosen, dtype=cp.int32), cp.asarray(local[:-1]),
                               int(local[-1]), [self.segment(s, s + 1) for s in chosen],
                               [slice(int(a), int(b)) for a, b in zip(local[:-1], local[1:])]))
            self._host_scatter_plans[key] = chunks
        from . import _cpu_quadruples

        for rows, rslot, roff, total, targets, sources in self._host_scatter_plans[key]:
            buffer = cp.empty(total)
            _ROWS(tile, rows, rslot, roff, *self._tables, self.nvir, self.nv4, alpha, buffer,
                  size=rows.size * self.nv4)
            values = buffer.get(out=self._pinned(total))      # page-locked: a fast copy
            del buffer
            # Each slot on all threads (compiled), as packed = beta packed + values.
            for target, source in zip(targets, sources):
                _cpu_quadruples.permuted_add(packed[target], None if beta == 0 else packed[target],
                                             values[source], float(beta), 1.0)

    def _pinned(self, count):
        """A reusable page-locked host buffer of `count` doubles (host-staged R4)."""
        buffer = getattr(self, "_pinned_buffer", None)
        if buffer is None or buffer.size < count:
            import cupyx

            buffer = self._pinned_buffer = cupyx.empty_pinned((count,), dtype=np.float64)
        return buffer[:count]

    def _row_entries(self, key, placements):
        """Device entry lists (slot, row, map[4]) grouped so that no list writes a slot twice."""
        if key not in self._plans:
            lists = []
            for group in placements:
                entries = [(self._slot[o], row, *mapping) for o, row, mapping in group
                           if list(o) == sorted(o)]
                if entries:
                    lists.append(self._array(np.asarray(entries, dtype=np.int32)))
            self._plans[key] = lists
        return self._plans[key]

    def add_rows(self, packed, rows, lists):
        """R4(o; v) += rows[row, v[map0], v[map1], v[map2], v[map3]] for every entry."""
        for entries in lists:
            _ROW_ADD(rows, entries, *self._tables, self.nvir, self.nv4, packed,
                     size=entries.shape[0] * self.nv4)

    def pair_entries(self, p, q, r):
        """Entries for rows Y[(r, s)] of the spectator pair (p, q): for every split
        (P, Q) and s >= r, the slot o with o[P] = (r, s) and o[Q] = (p, q)."""
        def placements():
            for split in _SPLITS:
                group = []
                for s in range(r, self.nocc):
                    o = [0] * 4
                    o[split[0]], o[split[1]], o[split[2]], o[split[3]] = r, s, p, q
                    group.append((tuple(o), s, split))
                yield group
        return self._row_entries(("pairs", p, q, r), placements())

    def triple_entries(self, triple):
        """Entries for rows A[x] of the spectator triple (p, q, r): for every target
        position t, the slot o with o[t] = x and the triple in the other positions."""
        def placements():
            for target in range(4):
                rest = tuple(axis for axis in range(4) if axis != target)
                group = []
                for x in range(self.nocc):
                    o = [0] * 4
                    o[target] = x
                    for axis, value in zip(rest, triple):
                        o[axis] = value
                    group.append((tuple(o), x, (target, *rest)))
                yield group
        return self._row_entries(("triples",) + tuple(triple), placements())

    def expand(self, segment, first, stop):
        """Dense (stop-first, nv, nv, nv, nv) device copy of a slot range."""
        shape = (stop - first,) + (self.nvir,) * 4
        if not self.symmetric and isinstance(segment, cp.ndarray):
            return segment.reshape(shape)
        dense = cp.empty(shape)
        _EXPAND(cp.asarray(segment), *self._tables, self.nvir, self.nv4, first,
                int(self.offsets[first]), dense.reshape(-1))
        return dense

    def compress(self, dense, segment, first):
        """Write the allowed entries of a dense slot range into its device segment."""
        if not self.symmetric:
            if not (dense.data.ptr == segment.data.ptr and dense.size == segment.size):
                segment[...] = dense.reshape(-1)
            return
        _COMPRESS(dense.reshape(-1), *self._tables, self.nvir, self.nv4, first,
                  int(self.offsets[first]), segment, size=dense.size)

    def project(self, residual):
        """Symmetrize occupied ties, spin-adapt, and purify every slot."""
        host = not isinstance(residual, cp.ndarray)
        for first, stop in self.batches():
            segment = residual[self.segment(first, stop)]
            device = cp.asarray(segment) if host else segment
            dense = self.expand(device, first, stop)
            ties = tuple(cp.asarray(tied[(tied >= first) & (tied < stop)] - first)
                         for tied in self._host_ties)
            self._project_device(dense, ties, self._invalid[first:stop])
            self.compress(dense, device, first)
            if host:
                device.get(out=segment)

    def _project_device(self, residual, ties, invalid):
        block = _slot_batch(self.nv4)
        for axis, tied in enumerate(ties, start=1):
            for start, stop in prange(0, tied.size, block):
                idx = tied[start:stop]
                original = residual[idx]
                residual[idx] = 0.5 * (original + original.swapaxes(axis, axis + 1))
        _spin_inplace(residual, residual.shape[0], self.nvir, "project", 1.0, 0.0)
        _PURIFY(invalid, self.nvir, self.nv4, residual.reshape(-1))

    def divide(self, residual, eia):
        for first, stop in self.batches():
            segment = residual[self.segment(first, stop)]
            dense = self.expand(segment, first, stop)
            _STEP(eia, self._slots[first:stop], self.nvir, self.nv4, dense.reshape(-1))
            self.compress(dense, segment, first)

    def update_from_host(self, t4, residual, eia, scale, *, keep_step=False):
        """t4 += scale * D^-1 residual for a host residual, batch by batch; with
        `keep_step`, the residual is overwritten by that step."""
        for first, stop in self.batches():
            window = self.segment(first, stop)
            update = cp.asarray(residual[window])
            dense = self.expand(update, first, stop)
            _STEP(eia, self._slots[first:stop], self.nvir, self.nv4, dense.reshape(-1))
            self.compress(dense, update, first)
            if scale != 1:
                update *= scale
            t4[window] += update
            if keep_step:
                residual[window] = update.get()

    def from_dense(self, source, *, order=None, out=None):
        """Compress dense packed slots, optionally reordering virtual axes.

        `order[k]` is the source virtual index stored at internal position k.
        Returns the device array and the largest discarded magnitude.
        """
        if source.shape != self.dense_shape:
            raise ValueError(f"Expected dense packed quadruples with shape {self.dense_shape}")
        out = cp.empty(self.size) if out is None else out
        leakage = 0.0
        for first, stop in self.batches():
            dense = cp.array(source[first:stop], dtype=cp.float64)
            if order is not None:
                for axis in range(1, 5):
                    dense = cp.take(dense, order, axis=axis)
            segment = out[self.segment(first, stop)]
            self.compress(dense, segment, first)
            if self.symmetric:
                kept = self.expand(segment, first, stop)
                gap = float(cp.abs(dense - kept).max())
                if leakage == leakage and not gap <= leakage:   # a NaN stays
                    leakage = gap
        return out, leakage

    def to_dense(self, packed, *, order=None):
        """Host dense packed copy; `order` as in from_dense restores source order."""
        result = np.empty(self.dense_shape)
        inverse = None if order is None else np.argsort(order)
        for first, stop in self.batches():
            dense = self.expand(cp.asarray(packed[self.segment(first, stop)]), first, stop)
            if inverse is not None:
                for axis in range(1, 5):
                    dense = cp.take(dense, cp.asarray(inverse), axis=axis)
            result[first:stop] = cp.asnumpy(dense)
        return result

    def accumulate_from(self, out, weight, source, source_layout, perm=None):
        """out += weight * `source` re-laid into this layout, one slot batch at a time.

        `source` (device, host, memory-mapped or `SavedArray`) is packed in
        `source_layout`, which must have the same occupied tuples and virtual
        count but may use other symmetry labels. `perm[k]` is the source's
        internal virtual index at this layout's internal position k (None when
        the orders agree). Entries this layout's symmetry forbids are dropped.
        """
        if source_layout.dense_shape != self.dense_shape:
            raise ValueError("The source quadruples use another occupied or virtual space")
        if (perm is None
                and np.array_equal(self.occupied_irreps, source_layout.occupied_irreps)
                and np.array_equal(self.virtual_irreps, source_layout.virtual_irreps)):
            for first, stop in self.batches():
                window = self.segment(first, stop)
                out[window] += weight * _segment(source, window)
            return
        index = None if perm is None else cp.asarray(perm)
        for first, stop in self.batches():
            dense = source_layout.expand(_segment(source, source_layout.segment(first, stop)),
                                         first, stop)
            if index is not None:
                for axis in range(1, 5):
                    dense = cp.take(dense, index, axis=axis)
            target = out[self.segment(first, stop)]
            part = cp.empty_like(target)
            self.compress(dense, part, first)
            target += weight * part

    def max_abs(self, packed):
        if isinstance(packed, cp.ndarray):
            return max(float(packed.max()), -float(packed.min())) if packed.size else 0.0
        from . import _cpu_quadruples

        if _cpu_quadruples.available() and packed.dtype == np.float64 and packed.flags.c_contiguous:
            value = _cpu_quadruples.max_abs(packed)         # one pass on all threads
            return value if np.isfinite(value) else float("nan")
        value = 0.0
        for first, stop in self.batches():
            window = packed[self.segment(first, stop)]
            if window.size:
                high, low = float(window.max()), float(window.min())
                if not np.isfinite((high, low)).all():
                    return float("nan")
                value = max(value, high, -low)
        return value


def _unpack_t4_(mycc, packed, tile, *bounds_and_sizes, allowed_only=False):
    mycc.layout.gather(packed, tile, bounds_and_sizes[:8], allowed_only=allowed_only)


def _unpack_t4_triples_(mycc, packed, tile, *bounds_and_sizes, allowed_only=False):
    mycc.layout.gather(packed, tile, bounds_and_sizes[:8], triples=True, allowed_only=allowed_only)


def _accumulate_t4_(mycc, packed, tile, *bounds_and_sizes, alpha=1.0, beta=0.0):
    mycc.layout.scatter(packed, tile, bounds_and_sizes[:8], alpha=alpha, beta=beta)


_PERMS3 = tuple(permutations(range(3)))
_PERMS4 = tuple(permutations(range(4)))
# Coefficients in lexicographic permutation order, verified against PySCF 2.14.0.
_SPIN = {
    "P3_201": (_PERMS3, (2, 0, -1, 0, 0, -1), 1),
    "P3_422": (_PERMS3, (4, -2, -2, 1, 1, -2), 1),
    "P3_full": (_PERMS3, (1,) * 6, 1),
    "P4_201": (_PERMS4, (2, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
                        0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0), 1),
    "P4_442": (_PERMS4, (4, 0, -2, 0, 0, -2, -2, 0, 1, 0, 0, 1,
                        1, 0, -2, 0, 1, 0, 0, 1, 0, -2, 0, 1), 1),
    "project": (_PERMS4, (14, -4, -4, -1, -1, -4, -4, 2, -1, 2, 2, -1,
                         -1, 2, -4, -1, 2, 2, 2, -1, -1, -4, 2, 2), 24),
    # P4_442 for the T4 -> R2 term only (sum_mnef <mn|ef> P(T)_mnij^efab, then
    # R2 + R2^T(ij,ab)). <mn|ef> = <nm|fe>, the pair symmetry of T4 and the
    # final transpose make terms conjugate under swapping positions (0, 1) or
    # (2, 3) contribute equally; summing each class onto one representative
    # with the innermost axis in place leaves 5 terms, one strided, instead of 12.
    "P4_442_r2": (_PERMS4, (4, 0, -8, 0, 0, 0, -2, 0, 4, 0, 0, 0,
                            0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0), 1),
}


@lru_cache(maxsize=None)
def _spin_kernel(pattern, nvir):
    """target = alpha * P(source) / divisor + beta * target over the trailing
    `rank` axes of length `nvir`, slot by slot. `nvir` is compiled in, so the
    digit extraction divides by constants (multiply and shift)."""
    perms, weights, divisor = _SPIN[pattern]
    rank = len(perms[0])
    dims = "abcd"[:rank]
    lines = [f"const long long slot = i / {nvir**rank}LL;",
             f"unsigned int rest = (unsigned int)(i - slot * {nvir**rank}LL);"]
    for c in reversed(dims):
        lines.append(f"const unsigned int {c} = rest % {nvir}u; rest /= {nvir}u;")
    terms = []
    for perm, weight in zip(perms, weights):
        if weight:
            offset = "0u"
            for axis in perm:
                offset = f"({offset} * {nvir}u + {dims[axis]})"
            terms.append(f"({weight}.0 * row[{offset}])")
    value = " + ".join(terms)
    lines.append(f"const double* row = &source[slot * {nvir**rank}LL];")
    return cp.ElementwiseKernel(
        "raw float64 source, float64 alpha, float64 beta",
        "float64 target",
        "\n".join(lines) + f"\ntarget = alpha * ({value}) / {divisor}.0"
        " + (beta == 0.0 ? 0.0 : beta * target);",
        f"itercc_quad_spin_{pattern}_{nvir}",
    )


_TILE_THREADS = 256


@lru_cache(maxsize=None)
def _spin_tiled(pattern, nvir):
    """Rank-four spin summation through shared-memory planes, or None.

    A term whose source has output axis L (not d) innermost reads the source
    with a stride along d. Such terms are grouped by L; the pass for a group
    loads, for each fixed pair of the other two axes, the (L, d) plane of
    every term coalesced along L into shared memory and reads it transposed.
    Terms with d innermost join the first pass and read the source directly.
    Each thread keeps its plane entries in registers, and every pass writes
    the output coalesced along d (the first pass applies alpha and beta, later
    ones add).
    """
    perms, weights, divisor = _SPIN[pattern]
    if len(perms[0]) != 4 or nvir > 64:
        return None
    nv, threads = nvir, _TILE_THREADS
    row = nv if nv % 2 else nv + 1        # odd stride: conflict-free double reads
    per_thread = -(-nv * nv // threads)
    terms = [(p, w) for p, w in zip(perms, weights) if w]
    groups = {}
    for perm, weight in terms:
        if perm[3] != 3:
            groups.setdefault(perm[3], []).append((perm, weight))
    direct = [(p, w) for p, w in terms if p[3] == 3]
    # Worth it (measured) for one strided group whose source has d outermost,
    # so the plain kernel's reads jump by nvir**3 (P4_201); every extra pass
    # would reread and rewrite the output.
    if len(groups) != 1 or any(p[0] != 3 for group in groups.values() for p, _ in group):
        return None
    passes = sorted(groups) or [2]

    def offset(perm, names):
        return " + ".join(f"{names[axis]} * {nv**(3 - k)}" for k, axis in enumerate(perm))

    kernels = []
    for number, axis in enumerate(passes):
        fixed = [a for a in range(3) if a != axis]
        out_names = {axis: "x", 3: "d", fixed[0]: "y", fixed[1]: "z"}
        load_names = {axis: "xx", 3: "dd", fixed[0]: "y", fixed[1]: "z"}
        body = []
        for perm, weight in (direct if number == 0 else []):
            body.append(f"for (int k = 0; k < {per_thread}; ++k) {{ const int idx = threadIdx.x + k * {threads};"
                        f" if (idx < {nv * nv}) {{ const int x = idx / {nv}, d = idx % {nv};"
                        f" acc[k] += {weight}.0 * src[{offset(perm, out_names)}]; }} }}")
        for perm, weight in groups.get(axis, []):
            body.append("__syncthreads();")
            body.append(f"for (int idx = threadIdx.x; idx < {nv * nv}; idx += {threads}) {{"
                        f" const int dd = idx / {nv}, xx = idx % {nv};"
                        f" tile[dd * {row} + xx] = src[{offset(perm, load_names)}]; }}")
            body.append("__syncthreads();")
            body.append(f"for (int k = 0; k < {per_thread}; ++k) {{ const int idx = threadIdx.x + k * {threads};"
                        f" if (idx < {nv * nv}) {{ const int x = idx / {nv}, d = idx % {nv};"
                        f" acc[k] += {weight}.0 * tile[d * {row} + x]; }} }}")
        store = ("out[o] = alpha * acc[k] / {div}.0 + (beta == 0.0 ? 0.0 : beta * out[o]);" if number == 0
                 else "out[o] += alpha * acc[k] / {div}.0;").format(div=divisor)
        name = f"itercc_quad_spin_tiled_{pattern}_{nv}_{number}"
        source = f"""
extern "C" __global__ void {name}(const double* __restrict__ source, double* __restrict__ target,
                                  const long long slots, const double alpha, const double beta) {{
    __shared__ double tile[{nv * row}];
    const int y = blockIdx.x / {nv}, z = blockIdx.x % {nv};
    for (long long slot = blockIdx.y; slot < slots; slot += gridDim.y) {{
        const double* src = source + slot * {nv**4}LL;
        double* out = target + slot * {nv**4}LL;
        double acc[{per_thread}];
        for (int k = 0; k < {per_thread}; ++k) acc[k] = 0.0;
        {chr(10).join(body)}
        for (int k = 0; k < {per_thread}; ++k) {{
            const int idx = threadIdx.x + k * {threads};
            if (idx < {nv * nv}) {{
                const int x = idx / {nv}, d = idx % {nv};
                const long long o = {offset((0, 1, 2, 3), out_names)};
                {store}
            }}
        }}
        __syncthreads();
    }}
}}
"""
        kernels.append(cp.RawKernel(source, name))

    def run(source, target, slots, alpha, beta):
        grid = (nv * nv, int(min(slots, 65535)))
        for kernel in kernels:
            kernel(grid, (threads,), (source, target, np.int64(slots), np.float64(alpha), np.float64(beta)))
    return run


def _spin4(source, target, nvir, pattern, alpha, beta):
    """target = alpha * P(source) + beta * target on C-contiguous rank-four rows
    of nvir**4 entries; source and target must not overlap."""
    slots = source.size // nvir**4
    tiled = _spin_tiled(pattern, nvir)
    if tiled is None:
        _spin_kernel(pattern, nvir)(source, alpha, beta, target.reshape(-1))
    elif slots:
        tiled(source, target, slots, alpha, beta)


def _slot_batch(stride):
    return max(1, (64 * 2**20) // (8 * stride))


def _spin_inplace(a, slots, nvir, pattern, alpha, beta):
    rows = a.reshape(slots, -1)
    stride = rows.shape[1]
    rank = len(_SPIN[pattern][0][0])
    if stride != nvir ** rank:
        raise ValueError(f"{pattern} acts on rows of {nvir}**rank entries, not {stride}")
    for start, stop in prange(0, slots, _slot_batch(stride)):
        target = rows[start:stop].reshape(-1)
        if rank == 4:
            _spin4(target.copy(), target, nvir, pattern, alpha, beta)
        else:
            _spin_kernel(pattern, nvir)(target.copy(), alpha, beta, target)


def t3_spin_summation(a, b, slots, nvir, pattern, alpha=1.0, beta=0.0):
    _spin_kernel(pattern, nvir)(a, alpha, beta, b.reshape(-1))


def t4_spin_summation(source, target, nvir, pattern, alpha=1.0, beta=0.0):
    """target = alpha * P(source) + beta * target over the last four (virtual) axes.

    Out of place, so it needs no copy of
    the source; both arrays must be C-contiguous with the same shape.
    """
    if source.shape != target.shape or not (source.flags.c_contiguous and target.flags.c_contiguous):
        raise ValueError("t4_spin_summation needs C-contiguous arrays of one shape")
    if cp.may_share_memory(source, target):     # bounds only; shares_memory is O(size)
        raise ValueError("t4_spin_summation needs separate source and target arrays")
    _spin4(source, target, nvir, pattern, alpha, beta)


def add_transpose_(x, perm):
    """x += x.transpose(perm) in place."""
    x += x.transpose(perm)
    return x


def add_transposed(a, perm):
    """a + a.transpose(perm) as a new array."""
    return a + a.transpose(perm)


def relayout(array, order):
    """The same array stored with its axes `order` outermost first in memory."""
    return cp.ascontiguousarray(array.transpose(order)).transpose([int(i) for i in np.argsort(order)])


def project_triples(r3):
    source = r3.copy()
    for perm in _PERMS3[1:]:
        r3 += source.transpose(perm + tuple(i + 3 for i in perm))
    nocc, nvir = r3.shape[0], r3.shape[3]
    _spin_inplace(r3, nocc**3, nvir, "P3_full", -1.0 / 6.0, 1.0)
    occupied = cp.arange(nocc)
    virtual = cp.arange(nvir)
    r3[occupied, occupied, occupied, ...] = 0.0
    r3[..., virtual, virtual, virtual] = 0.0
