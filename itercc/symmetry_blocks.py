"""Abelian-symmetry-blocked two-operand contractions on the GPU.

For tensors whose indices carry irrep labels (products by XOR) and which are
totally symmetric, ``out = beta * out + alpha * einsum(spec, a, b)`` only
couples symmetry-allowed blocks. `SymmetryBlocks.einsum` evaluates it as a
direct-product decomposition:

* labelled "combination" indices (here: occupied indices of a tile, a handful
  of values each) are enumerated explicitly;
* the other free indices of each operand are merged into one composite row
  (or column) index and sorted by its irrep, and the summed indices into one
  composite contracted index sorted the same way;
* for every pair of row and column combinations and every irrep kappa of the
  contracted index, the allowed rows and columns form one dense GEMM. All of
  them run in a single cuBLAS grouped-batched call (cuBLAS 12.5 or newer).

Only the operand blocks some GEMM reads are gathered (about 1/nirrep of each
totally symmetric operand), densely, and the products are added back into
``out`` (any strides) with precomputed offsets. Compared with a dense einsum,
the GEMMs skip every forbidden block: about nirrep**2 fewer flops for
rank-four outputs, and only allowed output entries are written. Contractions
the decomposition does not cover (batch indices, one-sided sums, non-float64)
fall back to `fallback` (dense einsum by default).

Shallow or skinny GEMMs (a short contracted index, or a handful of rows or
columns, as in most coupled-cluster tile terms) are memory-latency bound, not
flop bound: cuBLAS spends its time on padded tiles, and the gather, product
and scatter passes each cross memory. Those plans run as one fused kernel
instead, reading the operands in place through the same offset tables and
adding the products straight into ``out`` (see `_FUSED_DEPTH`).
"""
import hashlib
import ctypes
import glob
import os
from functools import reduce
from itertools import product

import numpy as np
from . import _runtime

_GROUPED = None
_KERNELS = {}

# Both kernels walk a list of dense blocks on a (x: inner/column, y: middle/row,
# z: block) grid, so no thread divides a flat index. Each block table is
# fields x blocks int64, one row per field.
_SOURCE = r"""
extern "C" __global__ void itercc_sym_gather(
        const double* __restrict__ src, double* __restrict__ dst, const int blocks,
        const long long* __restrict__ table, const long long* __restrict__ middle,
        const long long* __restrict__ inner) {
    // Fields: source base, middle start, rows, inner start, columns, destination start.
    // dst[start + m * columns + i] = src[base + middle[m0 + m] + inner[i0 + i]]
    for (int b = blockIdx.z; b < blocks; b += gridDim.z) {
        const long long base = table[b], m0 = table[blocks + b], rows = table[2 * blocks + b];
        const long long i0 = table[3 * blocks + b], cols = table[4 * blocks + b];
        double* target = dst + table[5 * blocks + b];
        for (long long m = blockIdx.y * (long long)blockDim.y + threadIdx.y; m < rows;
             m += (long long)gridDim.y * blockDim.y) {
            const double* row = src + base + middle[m0 + m];
            const long long* columns = inner + i0;
            double* line = target + m * cols;
            for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < cols;
                 i += (long long)gridDim.x * blockDim.x)
                line[i] = row[columns[i]];
        }
    }
}

extern "C" __global__ void itercc_sym_scatter(
        const double* __restrict__ x, double* __restrict__ out, const long long count,
        const int blocks, const long long* __restrict__ table, const long long* __restrict__ rpos,
        const long long* __restrict__ cpos, const int rowfast, const double alpha) {
    // Fields: x start, rows, columns, row start, column start, output base, slices.
    // out[obase + rpos[r0 + r] + cpos[c0 + c]] += alpha * sum_s x[s * count + start + r * columns + c],
    // or x[... + c * rows + r] with rowfast (products stored column by column; the lanes
    // then run along rows).
    for (int b = blockIdx.z; b < blocks; b += gridDim.z) {
        const long long start = table[b], rows = table[blocks + b], cols = table[2 * blocks + b];
        const long long r0 = table[3 * blocks + b], c0 = table[4 * blocks + b];
        const long long splits = table[6 * blocks + b];
        double* target = out + table[5 * blocks + b];
        const long long outer_n = rowfast ? cols : rows, inner_n = rowfast ? rows : cols;
        const long long* outer_pos = rowfast ? cpos + c0 : rpos + r0;
        const long long* inner_pos = rowfast ? rpos + r0 : cpos + c0;
        for (long long o = blockIdx.y * (long long)blockDim.y + threadIdx.y; o < outer_n;
             o += (long long)gridDim.y * blockDim.y) {
            const double* value = x + start + o * inner_n;
            double* line = target + outer_pos[o];
            for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < inner_n;
                 i += (long long)gridDim.x * blockDim.x) {
                double sum = value[i];
                for (long long s = 1; s < splits; ++s) sum += value[i + s * count];
                line[inner_pos[i]] += alpha * sum;
            }
        }
    }
}

#define FUSED_KC 16
// Up to eight entries of one operand chunk, base[fast_tab[fast] + slow_tab[slow]] for
// slow = slow0 + i * slow_step, into shared memory at fast * fast_pitch + slow * slow_pitch.
// The thread keeps one index of the operand's faster dimension (`fast`), so its loads
// are independent and all in flight together.
__device__ __forceinline__ void itercc_fused_stage(
        double* tile, const int fast_pitch, const int slow_pitch, const double* base,
        const long long* fast_tab, const long long* slow_tab, const int fast, const int fast_n,
        const int slow0, const int slow_step, const int slow_n, const int count) {
    if (fast >= fast_n) return;
    const double* p = base + fast_tab[fast];
    double v[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int s = slow0 + i * slow_step;
        if (i < count && s < slow_n) v[i] = p[slow_tab[s]];
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int s = slow0 + i * slow_step;
        if (i < count && s < slow_n) tile[fast * fast_pitch + s * slow_pitch] = v[i];
    }
}

extern "C" __global__ void __launch_bounds__(256, 2) itercc_sym_fused(
        const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out,
        const int blocks, const long long* __restrict__ table,
        const long long* __restrict__ a_rows, const long long* __restrict__ a_sum,
        const long long* __restrict__ b_sum, const long long* __restrict__ b_cols,
        const long long* __restrict__ rpos, const long long* __restrict__ cpos,
        const double alpha, const int lr, const int rfast, const int afast, const int bfast) {
    // Fields: rows, columns, depth, row start, column start, depth start, A base, B base,
    // output base. For every block (one GEMM of the plan):
    //   out[obase + rpos[r0 + r] + cpos[c0 + c]] += alpha * sum_k
    //       a[abase + a_rows[r0 + r] + a_sum[k0 + k]] * b[bbase + b_sum[k0 + k] + b_cols[c0 + c]]
    // A thread block computes a tr x tc tile of outputs (tr = 2**lr, tr * tc = 1024, four
    // per thread) with A and B staged in shared memory ([k][row] and [k][column], padded)
    // in chunks of FUSED_KC summed entries. Lanes run along the output's faster index
    // (rows when rfast) so the accumulation coalesces; loads run along each operand's
    // faster index (rows of A when afast, summed entries of B when bfast).
    extern __shared__ double smem[];
    const int lc = 10 - lr, tr = 1 << lr, tc = 1 << lc;
    const int pa = tr + 1, pb = tc + 1;
    double* sa = smem;                    // FUSED_KC x pa
    double* sb = smem + FUSED_KC * pa;    // FUSED_KC x pb
    const int t = threadIdx.x;
    const int lane = rfast ? (t & (tr - 1)) : (t & (tc - 1));
    const int first = rfast ? (t >> lr) : (t >> lc);
    const int step = rfast ? (256 >> lr) : (256 >> lc);
    // Staging maps: the thread's fast index, first slow index, slow step, entries per thread.
    const int na = tr >= 16 ? tr >> 4 : 1, nb = tc >= 16 ? tc >> 4 : 1;
    const int a_fast = afast ? (t & (tr - 1)) : (t & 15);
    const int a_slow = afast ? (t >> lr) : (t >> 4);
    const int a_step = afast ? (256 >> lr) : 16;
    const int b_fast = bfast ? (t & 15) : (t & (tc - 1));
    const int b_slow = bfast ? (t >> 4) : (t >> lc);
    const int b_step = bfast ? 16 : (256 >> lc);
    // Shared offsets: A is [k][row], B is [k][column].
    const int a_fp = afast ? 1 : pa, a_sp = afast ? pa : 1;
    const int b_fp = bfast ? pb : 1, b_sp = bfast ? 1 : pb;
    for (int g = blockIdx.z; g < blocks; g += gridDim.z) {
        const int rows = (int)table[g], cols = (int)table[blocks + g], depth = (int)table[2 * blocks + g];
        const long long r0 = table[3 * blocks + g], c0 = table[4 * blocks + g];
        const long long k0 = table[5 * blocks + g];
        const double* ab = a + table[6 * blocks + g];
        const double* bb = b + table[7 * blocks + g];
        double* ob = out + table[8 * blocks + g];
        const int tiles_c = (cols + tc - 1) >> lc;
        const int tiles = ((rows + tr - 1) >> lr) * tiles_c;
        for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
            const int rt = (tile / tiles_c) << lr, ct = (tile % tiles_c) << lc;
            const int rn = min(rows - rt, tr), cn = min(cols - ct, tc);
            double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
            for (int kc = 0; kc < depth; kc += FUSED_KC) {
                const int kn = min(FUSED_KC, depth - kc);
                __syncthreads();
                itercc_fused_stage(sa, a_fp, a_sp, ab,
                                   afast ? a_rows + r0 + rt : a_sum + k0 + kc,
                                   afast ? a_sum + k0 + kc : a_rows + r0 + rt,
                                   a_fast, afast ? rn : kn, a_slow, a_step, afast ? kn : rn, na);
                itercc_fused_stage(sb, b_fp, b_sp, bb,
                                   bfast ? b_sum + k0 + kc : b_cols + c0 + ct,
                                   bfast ? b_cols + c0 + ct : b_sum + k0 + kc,
                                   b_fast, bfast ? kn : cn, b_slow, b_step, bfast ? cn : kn, nb);
                __syncthreads();
                // Rows or columns past the block hold stale values: their outputs are dropped.
                if (rfast) {
                    const double* x = sa + lane;
                    const double* y = sb + first;
                    for (int k = 0; k < kn; ++k, x += pa, y += pb) {
                        const double u = *x;
                        acc0 += u * y[0]; acc1 += u * y[step]; acc2 += u * y[2 * step]; acc3 += u * y[3 * step];
                    }
                } else {
                    const double* x = sa + first;
                    const double* y = sb + lane;
                    for (int k = 0; k < kn; ++k, x += pa, y += pb) {
                        const double u = *y;
                        acc0 += x[0] * u; acc1 += x[step] * u; acc2 += x[2 * step] * u; acc3 += x[3 * step] * u;
                    }
                }
            }
            // Read-modify-write the (up to) four outputs: addresses first, then loads, then stores.
            const double accs[4] = {acc0, acc1, acc2, acc3};
            long long at[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int r = rfast ? lane : first + j * step, c = rfast ? first + j * step : lane;
                at[j] = (r < rn && c < cn) ? rpos[r0 + rt + r] + cpos[c0 + ct + c] : -1;
            }
            double old[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) old[j] = at[j] >= 0 ? ob[at[j]] : 0.0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) if (at[j] >= 0) ob[at[j]] = old[j] + alpha * accs[j];
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) itercc_sym_direct(
        const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out,
        const int blocks, const long long* __restrict__ table,
        const long long* __restrict__ a_rows, const long long* __restrict__ a_sum,
        const long long* __restrict__ b_sum, const long long* __restrict__ b_cols,
        const long long* __restrict__ rpos, const long long* __restrict__ cpos,
        const double alpha, const int lr, const int rfast, const int afast, const int bfast) {
    // Same contraction and tiles as itercc_sym_fused, without shared memory: each thread
    // reads its row of A and its columns of B straight from the operands (lanes that
    // share a column or row read the same address, served once per warp), which keeps
    // registers and latency low for shallow sums.
    const int lc = 10 - lr, tr = 1 << lr, tc = 1 << lc;
    const int t = threadIdx.x;
    const int lane = rfast ? (t & (tr - 1)) : (t & (tc - 1));
    const int first = rfast ? (t >> lr) : (t >> lc);
    const int step = rfast ? (256 >> lr) : (256 >> lc);
    for (int g = blockIdx.z; g < blocks; g += gridDim.z) {
        const int rows = (int)table[g], cols = (int)table[blocks + g], depth = (int)table[2 * blocks + g];
        const long long r0 = table[3 * blocks + g], c0 = table[4 * blocks + g];
        const long long k0 = table[5 * blocks + g];
        const double* ab = a + table[6 * blocks + g];
        const double* bb = b + table[7 * blocks + g];
        double* ob = out + table[8 * blocks + g];
        const long long* ks_a = a_sum + k0;
        const long long* ks_b = b_sum + k0;
        const int tiles_c = (cols + tc - 1) >> lc;
        const int tiles = ((rows + tr - 1) >> lr) * tiles_c;
        for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
            const int rt = (tile / tiles_c) << lr, ct = (tile % tiles_c) << lc;
            const int rn = min(rows - rt, tr), cn = min(cols - ct, tc);
            // The thread's single index (row when rfast, else column) and its four others.
            const int own_n = rfast ? rn : cn;
            if (lane >= own_n) continue;
            const int other_n = rfast ? cn : rn;
            const double* own = rfast ? ab + a_rows[r0 + rt + lane] : bb + b_cols[c0 + ct + lane];
            const long long* own_k = rfast ? ks_a : ks_b;
            const long long* other_k = rfast ? ks_b : ks_a;
            const double* other = rfast ? bb : ab;
            long long o[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int x = first + j * step;
                o[j] = x < other_n ? (rfast ? b_cols[c0 + ct + x] : a_rows[r0 + rt + x]) : -1;
            }
            double acc[4] = {0.0, 0.0, 0.0, 0.0};
            #pragma unroll 4
            for (int k = 0; k < depth; ++k) {
                const double u = own[own_k[k]];
                const double* p = other + other_k[k];
                #pragma unroll
                for (int j = 0; j < 4; ++j) if (o[j] >= 0) acc[j] += u * p[o[j]];
            }
            long long at[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int x = first + j * step;
                at[j] = o[j] >= 0 ? (rfast ? rpos[r0 + rt + lane] + cpos[c0 + ct + x]
                                           : rpos[r0 + rt + x] + cpos[c0 + ct + lane]) : -1;
            }
            double old[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) old[j] = at[j] >= 0 ? ob[at[j]] : 0.0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) if (at[j] >= 0) ob[at[j]] = old[j] + alpha * acc[j];
        }
    }
}
"""

# Route of a plan (measured per plan on a 29-virtual CuF evaluation, A100):
# * every GEMM at most _DIRECT_DEPTH deep: `itercc_sym_direct`, operands read in
#   place through the offset tables, products added straight into `out`;
# * at most _FUSED_DEPTH deep with skinny products (mean rows or columns at most
#   _FUSED_SKINNY, where cuBLAS tiles are mostly padding): `itercc_sym_fused`,
#   the same with operand tiles staged in shared memory;
# * otherwise gathers, cuBLAS grouped GEMM and a scatter.
# ITERCC_SYMMETRY_FUSED_DEPTH=0 puts every plan on cuBLAS.
_FUSED_DEPTH = _runtime.integer("ITERCC_SYMMETRY_FUSED_DEPTH")
_FUSED_SKINNY = _runtime.integer("ITERCC_SYMMETRY_FUSED_SKINNY")
_DIRECT_DEPTH = _runtime.integer("ITERCC_SYMMETRY_DIRECT_DEPTH")


def _kernel(name):
    import cupy as cp

    if name not in _KERNELS:
        _KERNELS[name] = cp.RawKernel(_SOURCE, name)
    return _KERNELS[name]


def _shape(outer, middle, inner):
    """Grid and block for an (outer, middle, inner) loop nest of 256-thread blocks:
    x spans `inner` (a power of two up to 256 threads), y `middle`, z `outer`."""
    tx = 1 << max(0, min(8, (int(inner) - 1).bit_length()))
    ty = 256 // tx
    grid = (int(min(-(-inner // tx), 2**31 - 1)), int(min(-(-middle // ty), 65535)),
            int(min(outer, 65535)))
    return grid, (tx, ty, 1)


_pointer = np.uint64     # a device pointer passed by value to a kernel


def _launch(name, shape, *args):
    if 0 in shape[0]:
        return
    _kernel(name)(shape[0], shape[1], args)


def _grouped_gemm():
    """cublasDgemmGroupedBatched from the libcublas CuPy uses, or None."""
    global _GROUPED
    if _GROUPED is not None:
        return _GROUPED or None
    candidates = ["libcublas.so.12"]
    try:
        import nvidia.cublas as package
        for root in package.__path__:
            candidates += sorted(glob.glob(os.path.join(root, "lib", "libcublas.so*")))
    except ImportError:
        pass
    import cupy
    candidates += sorted(glob.glob(os.path.join(os.path.dirname(cupy.__file__), "..", "nvidia",
                                                "cublas", "lib", "libcublas.so*")))
    for name in candidates:
        try:
            library = ctypes.CDLL(name)
        except OSError:
            continue
        function = getattr(library, "cublasDgemmGroupedBatched", None)
        if function is not None:
            function.restype = ctypes.c_int
            _GROUPED = function
            return function
    _GROUPED = False
    return None


def available():
    """Whether cuBLAS grouped GEMM (12.5 or later) is loadable."""
    return _grouped_gemm() is not None


def engine_available():
    """Whether the CUDA engine can run: its fused and direct plans need only
    CuPy; a plan of grouped GEMMs without cuBLAS 12.5 runs that product on
    the fallback instead (`SymmetryBlocks.einsum`)."""
    try:
        import cupy  # noqa: F401
    except ImportError:
        return False
    return True


def _parse(spec):
    inputs, output = spec.replace(" ", "").split("->")
    first, second = inputs.split(",")
    return first, second, output


def summed_pairs(spec, ordered):
    """How many constraints of `ordered` link two summed letters of `spec`:
    each restricts the sum to x < y (an exchangeable pair, whose sum over both
    orders is twice that; the caller doubles alpha), so a fallback product
    over the whole sum is scaled by 1/2 per pair."""
    sa, sb, so = _parse(spec)
    summed = set(sa) & set(sb) - set(so)
    return sum(1 for x, y, _ in ordered if x in summed and y in summed)


def _composite(letters, extent, labels, nirrep, ordered=()):
    """Sorted composite index over `letters`: permutation, per-irrep starts and irreps.

    Returns (order, starts) where `order[s]` is the row-major flat index (over
    `letters`) of the s-th element in irrep-sorted order and `starts[g]` is the
    first sorted position with irrep g (length nirrep + 1). `ordered` holds
    (i, j, strict) for positions i, j of `letters`: only tuples whose value at
    i is below (strict) or at most the value at j are kept, so a product
    writes only the entries its reader takes (antisymmetric pairs, say).
    """
    if not letters:
        irreps = np.zeros(1, dtype=np.int64)
    else:
        grids = np.meshgrid(*[np.asarray(labels[c], dtype=np.int64) for c in letters], indexing="ij")
        irreps = reduce(np.bitwise_xor, grids).reshape(-1)
    flat = np.arange(irreps.size)
    if ordered:
        values = np.indices([extent[c] for c in letters]).reshape(len(letters), -1)
        keep = np.ones(irreps.size, dtype=bool)
        for i, j, strict in ordered:
            keep &= values[i] < values[j] if strict else values[i] <= values[j]
        flat = np.flatnonzero(keep)
    order = flat[np.argsort(irreps[flat], kind="stable")]
    counts = np.bincount(irreps[flat], minlength=nirrep)
    return order, np.r_[0, np.cumsum(counts)]


def _offsets(letters, extent, strides, flat, positions=None):
    """Element offsets (with `strides` per letter) of row-major flat indices
    over `letters`; a letter with `positions` stores index i at positions[i],
    and the members (group, j) of an `OffsetGroup` (all of them in `letters`)
    together add table[their flat index, member j at place j] * stride."""
    if not letters:
        return np.zeros(len(flat), dtype=np.int64)
    positions = positions or {}
    digits = dict(zip(letters, np.unravel_index(flat, [extent[c] for c in letters])))
    total = np.zeros(len(flat), dtype=np.int64)
    groups = {}
    for c in letters:
        p = positions.get(c)
        if isinstance(p, tuple):
            groups.setdefault(id(p[0]), (p[0], []))[1].append((p[1], c))
        elif p is not None:
            total += p[digits[c]].astype(np.int64) * strides[c]
        else:
            total += digits[c].astype(np.int64) * strides[c]
    for group, members in groups.values():
        if len(members) != len(group.extents):
            raise ValueError("An offset group is split between composites")
        members.sort()
        index = np.ravel_multi_index(tuple(digits[c] for _, c in members), group.extents)
        rank = group.table[index]
        if np.any(rank < 0):
            raise ValueError("A composite holds tuples outside its offset group")
        total += rank * group.stride
    return total


class OffsetGroup:
    """Axes stored together as one: their tuples kept in `table` (a rank for
    each flat tuple over `extents`, member j at place j, -1 for tuples never
    stored) times `stride`. An operand's axis k is member j of a group when
    its positions[k] is (group, j). A slot program keeps the letters of an
    intermediate or tile that every reader takes in order this way, each kept
    tuple once (b < c < d: a sixth of the dense entries)."""

    def __init__(self, extents, table, stride):
        self.extents = tuple(int(n) for n in extents)
        self.table, self.stride = np.ascontiguousarray(table, dtype=np.int64), int(stride)
        # Plans are cached by content: a group freed with its program must not
        # lend its key to another one at the same address.
        self.key = (self.extents, self.stride, hashlib.blake2b(self.table.tobytes(), digest_size=16).digest())


def _position_key(p):
    """A cache key of one axis's positions, by content (an object's identity
    can be reused once it is freed, while plans outlive it)."""
    if p is None:
        return None
    if isinstance(p, tuple):
        return p[0].key, p[1]
    p = np.asarray(p)
    return p.dtype.str, p.tobytes()


class PositionedView(np.ndarray):
    """An operand whose axes store their indices in another order: axis k
    keeps index i at positions[k][i] (None: at i), or belongs to an
    `OffsetGroup` (axes that share one stored axis). The composites and every
    sum are those of the plain operand; only its offsets differ (a slot
    program keeps the virtual axes of its intermediates and tiles in irrep
    order, so a symmetry class is a contiguous run)."""

    def __new__(cls, array, positions):
        view = np.asarray(array).view(cls)
        view.positions = tuple(positions)
        return view

    def __array_finalize__(self, obj):
        self.positions = getattr(obj, "positions", None)


def positions_key(array):
    """A cache key of an operand's positions (by content), or None for a
    plain operand."""
    positions = getattr(array, "positions", None)
    if not positions:
        return None
    return tuple(_position_key(p) for p in positions)


_POINTERS = None
_PROCESSORS = {}


def _processors():
    import cupy as cp

    device = cp.cuda.Device().id
    if device not in _PROCESSORS:
        _PROCESSORS[device] = cp.cuda.runtime.getDeviceProperties(device)["multiProcessorCount"]
    return _PROCESSORS[device]


def _fill_pointers(offsets, bases, count):
    """Device array of absolute pointers: three runs of `count` offsets on three bases."""
    import cupy as cp

    global _POINTERS
    if _POINTERS is None:
        _POINTERS = cp.ElementwiseKernel(
            "int64 offset, int64 b0, int64 b1, int64 b2, int64 count", "int64 pointer",
            "pointer = offset + (i < count ? b0 : (i < 2 * count ? b1 : b2))",
            "itercc_sym_pointers")
    return _POINTERS(offsets, *map(np.int64, bases), np.int64(count))


class SymmetryBlocks:
    """Symmetry-blocked `einsum` for one set of index labels.

    `labels` maps each index letter to the irreps (XOR group 0..nirrep-1) of
    the index values it spans in the operands passed to `einsum`; letters in
    `combinations` (for example the occupied indices of a tile) are enumerated
    instead of merged into sorted composite indices when they are free indices,
    and their labels may be replaced per call (summed letters always use the
    labels given here). Every operand must be totally symmetric (its entries
    vanish unless the irreps of all its indices XOR to zero); `out` then only
    receives allowed entries. Free letters of extent one (a fixed occupied
    index, say) are treated as combination letters too, so their labels only
    select GEMMs. Plans are cached: the sorted composite indices per spec,
    shape and strides (their offset tables shared between specs), and the GEMM
    list per irrep pattern of the combination indices.
    """

    def __init__(self, labels, combinations, nirrep, *, fallback=None):
        self.labels = {c: np.asarray(v, dtype=np.int64) for c, v in labels.items()}
        self.combinations = set(combinations)
        self.nirrep = int(nirrep)
        self.fallback = fallback
        self._structures = {}
        self._groups = {}
        self._composites = {}
        self._tables = {}
        self._calls = {}
        self._workspace = None

    def release_workspace(self):
        """Return the operand and product workspace to the memory pool (it is
        reallocated, and the GEMM pointers rebuilt, on the next call)."""
        self._workspace = None

    def _labels_key(self, values):
        """A cache key of one letter's labels (its extent with one irrep)."""
        return len(values) if self.nirrep == 1 else values.tobytes()

    def _sorted(self, letters, extent, labels, ordered=()):
        """Cached `_composite` over `letters` (keyed by extents, labels and the
        ordering constraints among `letters`: (letter, letter, strict))."""
        local = tuple((letters.index(x), letters.index(y), bool(strict)) for x, y, strict in ordered
                      if x in letters and y in letters)
        key = (tuple((extent[c], self._labels_key(labels[c])) for c in letters), local)
        hit = self._composites.get(key)
        if hit is None:
            hit = self._composites[key] = (key, *_composite(letters, extent, labels, self.nirrep, local))
        return hit

    def _offsets(self, letters, extent, strides, composite, positions=None):
        """Device offsets of a sorted composite, shared between structures."""
        import cupy as cp

        positions = {c: p for c, p in (positions or {}).items() if c in letters and p is not None}
        key = (composite[0], tuple(strides[c] for c in letters),
               tuple(sorted((c, _position_key(p)) for c, p in positions.items())))
        hit = self._tables.get(key)
        if hit is None:
            hit = self._tables[key] = cp.asarray(_offsets(letters, extent, strides, composite[1], positions))
        return hit

    def _structure(self, spec, a, b, out, labels, ordered=()):
        sa, sb, so = _parse(spec)
        rows = [c for c in so if c in sa and c not in sb]
        cols = [c for c in so if c in sb and c not in sa]
        summed = [c for c in sa if c in sb and c not in so]
        if (len(set(sa)) != len(sa) or len(set(sb)) != len(sb) or set(so) - set(rows) - set(cols)
                or set(sa) - set(rows) - set(summed) or set(sb) - set(cols) - set(summed)):
            return None
        extent = {}
        for letters, array in ((sa, a), (sb, b), (so, out)):
            for c, n in zip(letters, array.shape):
                if extent.setdefault(c, n) != n:
                    return None
        # Summed letters always join the sorted contracted index.
        if any((c not in self.combinations or c in summed) and len(labels.get(c, ())) != extent[c]
               for c in extent):
            return None
        sta, stb, sto = (dict(zip(s, (x // 8 for x in arr.strides)))
                         for s, arr in ((sa, a), (sb, b), (so, out)))
        # Positioned operands (PositionedView): offsets by stored positions.
        pa, pb, po = ({c: p for c, p in zip(s, getattr(arr, "positions", None) or ()) if p is not None}
                      for s, arr in ((sa, a), (sb, b), (so, out)))
        enumerated = self.combinations | {c for c in so if extent[c] == 1}
        orow = [c for c in rows if c in enumerated]
        vrow = [c for c in rows if c not in enumerated]
        ocol = [c for c in cols if c in enumerated]
        vcol = [c for c in cols if c not in enumerated]

        def size(letters):
            return int(np.prod([extent[c] for c in letters])) if letters else 1

        def arrange(letters, strides):
            # Smallest stride last: the sorted composite (stable within each irrep,
            # and virtual labels come sorted) then reads the operand that carries
            # most of its traffic in contiguous runs.
            return sorted(letters, key=lambda c: (extent[c] > 1, -strides[c]))

        def last(letters, strides):
            moving = [c for c in letters if extent[c] > 1]
            return strides[moving[-1]] if moving else np.inf

        n_a = size(orow) * size(vrow) * size(summed)
        n_b = size(ocol) * size(summed) * size(vcol)
        n_o = size(orow) * size(ocol) * size(vrow) * size(vcol)
        vrow = arrange(vrow, sta if n_a >= n_o else sto)
        vcol = arrange(vcol, stb if n_b >= n_o else sto)
        summed = arrange(summed, sta if n_a >= n_b else stb)
        # Sorted operand layouts: A is [combination][row][k], or [combination][k][row]
        # when its rows vary faster in memory; B is [combination][k][column] or
        # [combination][column][k]. cuBLAS reads either through its transpose flags.
        ta = last(vrow, sta) < last(summed, sta)
        tb = last(summed, stb) < last(vcol, stb)
        # Ordered free letters of one operand: only those output entries are formed.
        r_sorted = self._sorted(vrow, extent, labels, ordered)
        c_sorted = self._sorted(vcol, extent, labels, ordered)
        # Ordered summed letters: the sum runs over those tuples only.
        k_sorted = self._sorted(summed, extent, labels, ordered)
        (_, _, r_start), (_, _, c_start), (_, _, k_start) = r_sorted, c_sorted, k_sorted
        combos_r = list(product(*[range(extent[c]) for c in orow]))
        combos_c = list(product(*[range(extent[c]) for c in ocol]))

        def occ(letters, combo, strides, positions):
            return sum(int(positions[c][i] if c in positions else i) * strides[c] for i, c in zip(combo, letters))

        a_rows = self._offsets(vrow, extent, sta, r_sorted, pa)
        a_sum = self._offsets(summed, extent, sta, k_sorted, pa)
        b_sum = self._offsets(summed, extent, stb, k_sorted, pb)
        b_cols = self._offsets(vcol, extent, stb, c_sorted, pb)
        return dict(
            orow=orow, ocol=ocol, ta=ta, tb=tb, rfast=last(vrow, sto) < last(vcol, sto),
            a_rows=a_rows, a_sum=a_sum, b_sum=b_sum, b_cols=b_cols,
            r_start=r_start, c_start=c_start, k_start=k_start, combos_r=combos_r, combos_c=combos_c,
            obase=[[occ(orow, o_r, sto, po) + occ(ocol, o_c, sto, po) for o_c in combos_c] for o_r in combos_r],
            a_outer=[occ(orow, o, sta, pa) for o in combos_r],
            a_middle=a_sum if ta else a_rows, a_inner=a_rows if ta else a_sum,
            b_outer=[occ(ocol, o, stb, pb) for o in combos_c],
            b_middle=b_cols if tb else b_sum, b_inner=b_sum if tb else b_cols,
            rpos=self._offsets(vrow, extent, sto, r_sorted, po),
            cpos=self._offsets(vcol, extent, sto, c_sorted, po))

    def _group_plan(self, structure, labels, irreps):
        """GEMMs for one irrep pattern of the combination indices and operand irreps.

        Only the symmetry-allowed blocks of each operand are gathered: block
        (combination, kappa) of A holds the rows of irrep rho and the summed
        entries of irrep kappa, densely ([k][row] when `ta`, else [row][k]), and
        likewise for B ([column][k] when `tb`, else [k][column]).
        """
        import cupy as cp

        def irrep(letters, combo):
            return reduce(np.bitwise_xor, [int(labels[c][i]) for i, c in zip(combo, letters)], 0)

        ta, tb = structure["ta"], structure["tb"]
        r_start, c_start, k_start = structure["r_start"], structure["c_start"], structure["k_start"]
        blocks, fused, x_start = [], [], 0
        a_blocks, b_blocks = {}, {}
        sizes = [0, 0]

        def operand(store, key, which, rows, cols):
            if key not in store:
                store[key] = sizes[which]
                sizes[which] += rows * cols
            return store[key]

        for i_r, o_r in enumerate(structure["combos_r"]):
            g_r = irrep(structure["orow"], o_r)
            for i_c, o_c in enumerate(structure["combos_c"]):
                g_c = irrep(structure["ocol"], o_c)
                for kappa in range(self.nirrep):
                    rho, sigma = kappa ^ g_r ^ irreps[0], kappa ^ g_c ^ irreps[1]
                    rows = int(r_start[rho + 1] - r_start[rho])
                    cols = int(c_start[sigma + 1] - c_start[sigma])
                    depth = int(k_start[kappa + 1] - k_start[kappa])
                    if not (rows and cols and depth):
                        continue
                    r0, c0 = int(r_start[rho]), int(c_start[sigma])
                    a0 = operand(a_blocks, (i_r, kappa, r0, rows), 0, rows, depth)
                    b0 = operand(b_blocks, (i_c, kappa, c0, cols), 1, cols, depth)
                    blocks.append((rows, cols, depth, x_start, a0, b0, r0, c0,
                                   structure["obase"][i_r][i_c]))
                    fused.append((rows, cols, depth, r0, c0, int(k_start[kappa]),
                                   structure["a_outer"][i_r], structure["b_outer"][i_c],
                                   structure["obase"][i_r][i_c]))
                    x_start += rows * cols
        if not blocks:
            return dict(count=0)
        deepest = max(depth for _, _, depth, *_ in blocks)
        skinny = min(np.mean([b[0] for b in blocks]), np.mean([b[1] for b in blocks])) <= _FUSED_SKINNY
        if deepest <= min(_DIRECT_DEPTH, _FUSED_DEPTH) or (deepest <= _FUSED_DEPTH and skinny):
            return self._fused_plan(fused)
        # Split the contracted index when the output blocks are too few to fill the
        # device (vector-like products such as T4 into R2): each slice writes its
        # own partial product and the scatter adds them up. A block shallower
        # than `splits` gets one slice per contracted entry.
        tiles = sum(-(-rows // 64) * -(-cols // 64) for rows, cols, *_ in blocks)
        splits = int(max(1, min(32, -(-2 * _processors() // tiles), deepest // 512)))
        # Products go to the workspace in the output's faster order: [row][column]
        # (X^T = B^T A^T in cuBLAS's column-major view) unless the output runs along
        # rows (rfast), then [column][row] (X = A B), so the scatter adds them back
        # along contiguous runs of `out`.
        rfast = bool(structure["rfast"])
        gemms, slices = [], []
        for rows, cols, depth, start, a0, b0, *_ in blocks:
            parts = min(splits, depth)
            slices.append(parts)
            for s in range(parts):
                k1, k2 = depth * s // parts, depth * (s + 1) // parts
                b_offset, ld_b = (b0 + k1, depth) if tb else (b0 + k1 * cols, cols)
                a_offset, ld_a = (a0 + k1 * rows, rows) if ta else (a0 + k1, depth)
                if rfast:
                    gemms.append((rows, cols, k2 - k1, a_offset, b_offset, s * x_start + start, ld_a, ld_b))
                else:
                    gemms.append((cols, rows, k2 - k1, b_offset, a_offset, s * x_start + start, ld_b, ld_a))
        gemm = np.array(gemms, dtype=np.int64)
        count = len(gemms)

        def ints(values):
            return (ctypes.c_int * count)(*map(int, values))

        def gather(store, outer, rows_middle):
            # Block (combination, kappa, start, length) -> base, middle start, rows,
            # inner start, columns, destination; the summed index is the inner one
            # when the free index is the middle one.
            entries = []
            for (i, kappa, f0, length), start in store.items():
                k0, depth = int(k_start[kappa]), int(k_start[kappa + 1] - k_start[kappa])
                entries.append((outer[i], f0, length, k0, depth, start) if rows_middle
                               else (outer[i], k0, depth, f0, length, start))
            table = np.array(entries, dtype=np.int64)
            return (cp.asarray(np.ascontiguousarray(table.T)), len(entries),
                    _shape(len(entries), table[:, 2].max(), table[:, 4].max()))

        table = np.column_stack([np.array(blocks, dtype=np.int64), slices])
        return dict(
            count=count, blocks=len(blocks), x_size=x_start, splits=splits,
            a_size=sizes[0], b_size=sizes[1], rfast=rfast,
            # Workspace: sorted A, sorted B, then `splits` slices of the products.
            a_at=0, b_at=sizes[0], x_at=sizes[0] + sizes[1], work=sizes[0] + sizes[1] + splits * x_start,
            # A is [row][k] (rows in the middle) unless ta; B is [column][k] when tb.
            a_gather=gather(a_blocks, structure["a_outer"], not ta),
            b_gather=gather(b_blocks, structure["b_outer"], tb),
            scatter=(cp.asarray(np.ascontiguousarray(table[:, [3, 0, 1, 6, 7, 8, 9]].T)), len(blocks),
                     _shape(len(blocks), *((table[:, 1].max(), table[:, 0].max()) if rfast
                                           else (table[:, 0].max(), table[:, 1].max())))),
            offsets=cp.asarray(np.concatenate([gemm[:, 3], gemm[:, 4], gemm[:, 5]]) * 8),
            m=ints(gemm[:, 0]), n=ints(gemm[:, 1]), k=ints(gemm[:, 2]),
            lda=ints(gemm[:, 6]), ldb=ints(gemm[:, 7]), ldc=ints(gemm[:, 0]),
            transa=ints([int(not ta) if rfast else int(tb)] * count),
            transb=ints([int(not tb) if rfast else int(ta)] * count),
            size=ints([1] * count),
            one=(ctypes.c_double * count)(*[1.0] * count), zero=(ctypes.c_double * count)(*[0.0] * count))

    @staticmethod
    def _fused_plan(fused):
        """One fused launch for shallow GEMMs (see `itercc_sym_fused`): the tile shape
        (tr rows by 1024 / tr columns) that pads the blocks least, and the block table."""
        import cupy as cp

        table = np.array(fused, dtype=np.int64)
        rows, cols = table[:, 0], table[:, 1]

        def padded(tr):
            tc = 1024 // tr
            return int(np.sum(-(-rows // tr) * tr * (-(-cols // tc) * tc)))
        tr = min((8, 16, 32, 64, 128), key=padded)
        tc = 1024 // tr
        tiles = int(np.max(-(-rows // tr) * -(-cols // tc)))
        grid = (min(tiles, 2**31 - 1), 1, min(len(fused), 65535))
        shallow = int(np.max(table[:, 2])) <= _DIRECT_DEPTH
        return dict(count=len(fused), fused=True, tr=tr, blocks=len(fused),
                    kernel="itercc_sym_direct" if shallow else "itercc_sym_fused",
                    shared=0 if shallow else 16 * (tr + tc + 2) * 8,
                    table=cp.asarray(np.ascontiguousarray(table.T)), shape=(grid, (256, 1, 1)))

    def einsum(self, spec, a, b, out, alpha=1.0, beta=0.0, labels=None, irreps=(0, 0), ordered=()):
        """out = beta * out + alpha * einsum(spec, a, b) (with `ordered`, only
        the output entries those constraints keep, and sums over only the
        summed tuples they keep; see `_composite` and `summed_pairs`).

        `labels` replaces the labels of any letters for this call; `irreps`
        gives the total irreps of `a` and `b` (zero for totally symmetric
        operands, for example Gamma_p ^ Gamma_q ^ Gamma_r for a block of T4 with
        three occupied indices fixed).
        """
        import cupy as cp

        # One lookup per call: shapes, strides, dtypes, per-call labels and irreps.
        call = (spec, a.shape, a.strides, a.dtype.char, b.shape, b.strides, b.dtype.char,
                out.shape, out.strides, out.dtype.char, int(irreps[0]), int(irreps[1]),
                () if not labels else tuple((c, np.asarray(v, dtype=np.int64).tobytes())
                                            for c, v in labels.items()), tuple(ordered),
                positions_key(a), positions_key(b), positions_key(out))
        hit = self._calls.get(call)
        if hit is None:
            hit = self._calls[call] = self._resolve(spec, a, b, out, labels, irreps, ordered)
        structure, plan = hit
        if not structure or (plan["count"] and not plan.get("fused") and _grouped_gemm() is None):
            if self.fallback is None:
                raise ValueError(f"No symmetry-blocked plan for {spec}")
            return self.fallback(spec, a, b, out=out, alpha=alpha * 0.5 ** summed_pairs(spec, ordered), beta=beta)
        if beta == 0.0:
            out[...] = 0.0
        elif beta != 1.0:
            out *= beta
        if not plan["count"] or alpha == 0.0:
            return out
        if plan.get("fused"):
            (grid, block), arguments = plan["fused_launch"]
            _kernel(plan["kernel"])(grid, block, (
                _pointer(a.data.ptr), _pointer(b.data.ptr), _pointer(out.data.ptr), *arguments[:8],
                np.float64(alpha), *arguments[8:]), shared_mem=plan["shared"])
            return out
        # Operands and products live in one workspace reused between calls (the
        # stream orders them), so each plan's GEMM pointers are built once per
        # workspace allocation.
        if self._workspace is None or self._workspace.size < plan["work"]:
            self._workspace = None
            self._workspace = cp.empty(plan["work"])
        space = self._workspace
        base = space.data.ptr
        count = plan["count"]
        if plan.get("base") != base:
            first, second = ("a_at", "b_at") if plan["rfast"] else ("b_at", "a_at")
            pointers = _fill_pointers(plan["offsets"], (base + 8 * plan[first], base + 8 * plan[second],
                                                        base + 8 * plan["x_at"]), count)
            start = pointers.data.ptr
            plan["pointers"], plan["base"] = pointers, base
            plan["arguments"] = (
                plan["transa"], plan["transb"], plan["m"], plan["n"], plan["k"], plan["one"],
                ctypes.c_void_p(start), plan["lda"], ctypes.c_void_p(start + 8 * count), plan["ldb"],
                plan["zero"], ctypes.c_void_p(start + 16 * count), plan["ldc"], ctypes.c_int(count),
                plan["size"])
        # Kernel arguments go in as raw device pointers (CuPy's per-array
        # argument handling costs more host time than the launch).
        for source, side in ((a, "a"), (b, "b")):
            shape, arguments = plan[side + "_launch"]
            _launch("itercc_sym_gather", shape, _pointer(source.data.ptr),
                    _pointer(base + 8 * plan[side + "_at"]), *arguments)
        handle = cp.cuda.device.get_cublas_handle()
        from cupy_backends.cuda.libs import cublas
        cublas.setStream(handle, cp.cuda.get_current_stream().ptr)
        # Column-major view: X^T (cols x rows) = B^T (cols x k) @ A^T (k x rows).
        status = _grouped_gemm()(ctypes.c_void_p(handle), *plan["arguments"])
        if status != 0:
            raise RuntimeError(f"cublasDgemmGroupedBatched failed with status {status}")
        shape, arguments = plan["scatter_launch"]
        _launch("itercc_sym_scatter", shape, _pointer(base + 8 * plan["x_at"]), _pointer(out.data.ptr),
                *arguments, np.float64(alpha))
        return out

    def _supported(self, a, b, out):
        """Whether the operands can use a blocked plan (else the fallback runs)."""
        import cupy as cp

        return all(isinstance(x, cp.ndarray) and x.dtype == cp.float64 for x in (a, b, out))

    def _plan(self, structure, labels, irreps):
        """The executable plan for one irrep pattern (GEMM list and launch arguments)."""
        plan = self._group_plan(structure, labels, irreps)
        if plan.get("fused"):
            plan["fused_launch"] = (plan["shape"], (
                np.int32(plan["blocks"]), _pointer(plan["table"].data.ptr),
                *(_pointer(structure[name].data.ptr)
                  for name in ("a_rows", "a_sum", "b_sum", "b_cols", "rpos", "cpos")),
                np.int32(plan["tr"].bit_length() - 1), np.int32(structure["rfast"]),
                np.int32(structure["ta"]),
                np.int32(structure["tb"])))
        elif plan["count"]:
            for side in "ab":
                table, n, shape = plan[side + "_gather"]
                plan[side + "_launch"] = (shape, (
                    np.int32(n), _pointer(table.data.ptr), _pointer(structure[side + "_middle"].data.ptr),
                    _pointer(structure[side + "_inner"].data.ptr)))
            table, n, shape = plan["scatter"]
            plan["scatter_launch"] = (shape, (
                np.int64(plan["x_size"]), np.int32(n), _pointer(table.data.ptr),
                _pointer(structure["rpos"].data.ptr), _pointer(structure["cpos"].data.ptr),
                np.int32(plan["rfast"])))
        return plan

    def _resolve(self, spec, a, b, out, labels, irreps, ordered=()):
        """(structure, plan) for one call signature, or (False, None) for the fallback.
        `ordered`: (letter, letter, strict) constraints on free output letters
        (see `_composite`), whose other output entries are not written, and on
        summed letters, whose other summed tuples are skipped."""
        current = self.labels if not labels else {**self.labels,
                                                   **{c: np.asarray(v, dtype=np.int64)
                                                      for c, v in labels.items()}}
        # Labels of enumerated letters (combinations and unit free letters) only
        # select GEMMs; the others shape the sorted composites.
        output = _parse(spec)[2]
        enumerated = self.combinations.intersection(output) | {
            c for c, n in zip(output, out.shape) if n == 1}
        fixed = () if not labels else tuple(sorted(
            (c, self._labels_key(current[c])) for c in labels
            if c not in enumerated))
        key = (spec, a.shape, a.strides, b.shape, b.strides, out.shape, out.strides, fixed, tuple(ordered),
               positions_key(a), positions_key(b), positions_key(out))
        structure = self._structures.get(key)
        if structure is None:
            ok = self._supported(a, b, out)
            structure = (self._structure(spec, a, b, out, current, ordered) if ok else None) or False
            self._structures[key] = structure
        if not structure:
            return False, None
        pattern = tuple(self._labels_key(current[c]) for c in structure["orow"] + structure["ocol"])
        irreps = (int(irreps[0]), int(irreps[1]))
        plan = self._groups.get((key, pattern, irreps))
        if plan is None:
            plan = self._groups[key, pattern, irreps] = self._plan(structure, current, irreps)
        return structure, plan
