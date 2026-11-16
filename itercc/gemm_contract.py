"""Two-operand tensor contractions as cuBLAS GEMMs that accumulate into views.

``einsum_into(spec, a, b, out, alpha=1.0, beta=0.0)`` evaluates

    out = beta * out + alpha * einsum(spec, a, b)

for float64 CuPy arrays, including strided views. High-rank coupled-cluster
tiles spend most of their time moving data: einsum materializes its product
in GEMM order and then adds it, element by element, into a permuted output.
When the strides allow, the direct route instead runs GEMMs that read the
operands in place and accumulate straight into ``out`` (cuBLAS beta = 1).

Indices are classified as batch (in a, b and out), row (a and out), column
(b and out) and summed (a and b). The longest chains of row, column and
summed indices that merge into uniformly strided matrix dimensions form one
GEMM; the other free indices become a batch (strided or pointer-array GEMM)
and the other summed indices a sequence of accumulating GEMMs. An operand
whose chains cannot form a matrix with one unit stride is copied once into
GEMM order. Which route is faster depends on the shapes and the GPU, so
``einsum_into`` times both on the first contraction of each shape and stride
pattern and keeps the winner (ITERCC_GEMM_AUTOTUNE=0 uses a static
data-movement estimate). ``contract`` runs only the direct route and returns
False, with ``out`` untouched, when there is no direct plan (other dtypes,
one-sided sums, repeated indices, an output whose unit-stride index is a
batch index, or many small GEMMs that would move more data than einsum).
"""
from dataclasses import dataclass
import os
from itertools import product
from math import prod

import numpy as np

_INT32 = np.iinfo(np.int32).max
_plans = {}
_launch = {}


@dataclass(frozen=True)
class _Plan:
    swap: bool                 # operands exchanged so that out's unit index is a column
    copy_a: tuple | None       # axes for a contiguous transposed copy of (swapped) a
    copy_b: tuple | None
    m: int
    n: int
    k: int
    trans_a: int               # cuBLAS op on the column-major view (see _direct)
    trans_b: int
    lda: int
    ldb: int
    ldc: int
    batch: tuple               # (extents, a, b, c strides) of batched free indices
    serial: tuple              # (extents, a, b strides) of sequentially summed indices


def _parse(spec):
    inputs, output = spec.replace(" ", "").split("->")
    inputs = inputs.split(",")
    return inputs, output


def _strides(array):
    return tuple(s // array.itemsize for s in array.strides)


def _chain(candidates, extent, stride, *, first=None):
    """Longest run of candidates forming one uniformly strided dimension.

    Indices are taken by increasing stride, starting with `first` when given.
    Returns the chain from outer to inner index.
    """
    pool = sorted(candidates, key=lambda c: stride[c])
    if first is not None:
        pool.remove(first)
        pool.insert(0, first)
    chain = []
    for c in pool:
        if not chain or stride[c] == stride[chain[-1]] * extent[chain[-1]]:
            chain.append(c)
        else:
            break
    return tuple(reversed(chain))


def _merges(chain, extent, stride):
    return all(stride[o] == stride[i] * extent[i] for o, i in zip(chain, chain[1:]))


def _matrix(rows, cols, extent, stride):
    """('r', ld) for a row-major or ('c', ld) for a column-major operand, else None."""
    if not (_merges(rows, extent, stride) and _merges(cols, extent, stride)):
        return None
    nr, nc = prod(extent[c] for c in rows), prod(extent[c] for c in cols)
    sr = stride[rows[-1]] if rows else None
    sc = stride[cols[-1]] if cols else None
    if nc == 1 or sc == 1:
        ld = sr if nr > 1 else max(1, nc)
        if ld >= max(1, nc):
            return "r", ld
    if nr == 1 or sr == 1:
        ld = sc if nc > 1 else max(1, nr)
        if ld >= max(1, nr):
            return "c", ld
    return None


def _copy_axes(subscripts, leading, matrix):
    """Axis order of a GEMM-ready copy: extent-one and batch axes, then the matrix."""
    tail = [subscripts.index(c) for c in matrix]
    head = [subscripts.index(c) for c in leading if c in subscripts]
    rest = [axis for axis in range(len(subscripts)) if axis not in head + tail]
    return tuple(rest + head + tail)


def _contiguous(subscripts, axes, extent):
    stride, result = 1, {}
    for axis in reversed(axes):
        c = subscripts[axis]
        result[c] = stride
        stride *= extent[c]
    return result


def _plan(spec, a, b, out, *, cost=True):
    (sa, sb), so = _parse(spec)
    if (len(set(sa)) != len(sa) or len(set(sb)) != len(sb) or len(set(so)) != len(so)
            or not set(so) <= set(sa) | set(sb) or set(sa) - set(sb) - set(so)
            or set(sb) - set(sa) - set(so)):
        return None
    extent = {}
    for subscripts, array in ((sa, a), (sb, b), (so, out)):
        if len(subscripts) != array.ndim:
            return None
        for c, size in zip(subscripts, array.shape):
            if size == 0 or extent.setdefault(c, size) != size:
                return None
    st_a, st_b, st_c = (dict(zip(s, _strides(x))) for s, x in ((sa, a), (sb, b), (so, out)))
    big_a = a.size >= b.size
    # Extent-one indices never move an address.
    live = [c for c in so if extent[c] > 1]
    unit = [c for c in live if st_c[c] == 1]
    if not unit:
        return None
    swap = unit[0] in sa and unit[0] not in sb
    if swap:
        sa, sb, st_a, st_b, big_a = sb, sa, st_b, st_a, not big_a
    if unit[0] in sa:
        return None
    rows_c = [c for c in live if c in sa and c not in sb]
    cols_c = [c for c in live if c in sb and c not in sa]
    summed = [c for c in sa if c in sb and c not in so and extent[c] > 1]
    batch = [c for c in live if c in sa and c in sb]
    cols = _chain(cols_c, extent, st_c, first=unit[0])
    rows = _chain(rows_c, extent, st_c)
    while rows and not _merges(rows, extent, st_a):
        rows = rows[1:]
    while len(cols) > 1 and not _merges(cols, extent, st_b):
        cols = cols[1:]
    batch += [c for c in rows_c if c not in rows] + [c for c in cols_c if c not in cols]
    inner = _chain(summed, extent, st_a if big_a else st_b)
    serial = [c for c in summed if c not in inner]
    copy_a = copy_b = None
    layout_a = _matrix(rows, inner, extent, st_a)
    layout_b = _matrix(inner, cols, extent, st_b)
    if layout_a is None:
        copy_a = _copy_axes(sa, batch + serial, rows + inner)
        st_a = _contiguous(sa, copy_a, extent)
        layout_a = _matrix(rows, inner, extent, st_a)
    if layout_b is None:
        copy_b = _copy_axes(sb, batch + serial, inner + cols)
        st_b = _contiguous(sb, copy_b, extent)
        layout_b = _matrix(inner, cols, extent, st_b)
    if layout_a is None or layout_b is None:
        return None
    m, n, k = (prod(extent[c] for c in group) for group in (rows, cols, inner))
    ldc = st_c[rows[-1]] if rows and m > 1 else max(1, n)
    if ldc < max(1, n) or max(m, n, k, layout_a[1], layout_b[1], ldc) > _INT32:
        return None
    # Words moved: every GEMM reads its A and B blocks and updates its C block.
    # The einsum route transposes both operands, writes the product and adds it
    # into the permuted output; that strided update runs at a fraction of the
    # bandwidth of a contiguous pass (measured about 2x), hence its weight.
    calls = prod(extent[c] for c in batch + serial)
    direct = calls * (m * k + k * n + 2 * m * n)
    size_a, size_b = (b.size, a.size) if swap else (a.size, b.size)
    direct += 2 * size_a * (copy_a is not None) + 2 * size_b * (copy_b is not None)
    if cost and direct > 2 * (a.size + b.size) + 8 * out.size:
        return None
    # Column-major view: C^T (n x m) = op(B^T) (n x k) @ op(A^T) (k x m).
    trans_b = 0 if layout_b[0] == "r" else 1
    trans_a = 0 if layout_a[0] == "r" else 1
    batch_info = (tuple(extent[c] for c in batch), tuple(st_a.get(c, 0) for c in batch),
                  tuple(st_b.get(c, 0) for c in batch), tuple(st_c[c] for c in batch))
    serial_info = (tuple(extent[c] for c in serial), tuple(st_a[c] for c in serial),
                   tuple(st_b[c] for c in serial))
    return _Plan(swap, copy_a, copy_b, m, n, k, trans_a, trans_b, layout_a[1], layout_b[1], ldc,
                 batch_info, serial_info)


def _offsets(extents, *strides):
    if not extents:
        return [np.zeros(1, dtype=np.int64) for _ in strides]
    grid = np.indices(extents, dtype=np.int64).reshape(len(extents), -1)
    return [np.asarray(s, dtype=np.int64) @ grid for s in strides]


def _uniform(extents, *strides):
    """Merge batch indices into one strided batch if every operand allows it."""
    order = sorted(range(len(extents)), key=lambda i: strides[-1][i])
    for outer, inner in zip(order[1:], order):
        if any(s[outer] != s[inner] * extents[inner] for s in strides):
            return None
    return tuple(s[order[0]] for s in strides) if extents else (0,) * len(strides)


def _key(spec, a, b, out):
    return (out.device.id, spec, a.shape, a.strides, b.shape, b.strides, out.shape, out.strides)


def contract(spec, a, b, out, alpha=1.0, beta=0.0):
    """Run the direct GEMM plan; False (with `out` untouched) when there is none."""
    import cupy as cp

    if not all(isinstance(x, cp.ndarray) and x.dtype == cp.float64 for x in (a, b, out)):
        return False
    if out.size == 0:
        return True
    key = _key(spec, a, b, out)
    plan = _plans.get((key, True))
    if plan is None:
        plan = _plan(spec, a, b, out)
        _plans[key, True] = plan if plan is not None else False
    if not plan:
        return False
    _direct(key, plan, a, b, out, alpha, beta)
    return True


def _direct(key, plan, a, b, out, alpha, beta):
    import cupy as cp
    from cupy_backends.cuda.libs import cublas

    if plan.swap:
        a, b = b, a
    if plan.copy_a is not None:
        a = cp.ascontiguousarray(a.transpose(plan.copy_a))
    if plan.copy_b is not None:
        b = cp.ascontiguousarray(b.transpose(plan.copy_b))
    handle = cp.cuda.device.get_cublas_handle()
    cublas.setStream(handle, cp.cuda.get_current_stream().ptr)
    one = np.array(1.0)
    scale = np.array(float(alpha))
    extents, sa, sb, sc = plan.batch
    serial_extents, ka, kb = plan.serial
    count = prod(extents)
    launch_key = (key, plan)
    launch = _launch.get(launch_key)
    if launch is None:
        uniform = _uniform(extents, sa, sb, sc)
        offsets = (None if uniform is not None or count == 1
                   else tuple(cp.asarray(o * 8) for o in _offsets(extents, sb, sa, sc)))
        launch = _launch[launch_key] = (uniform, offsets)
    uniform, offsets = launch
    for step, index in enumerate(product(*map(range, serial_extents))):
        shift_a = sum(i * s for i, s in zip(index, ka)) * 8
        shift_b = sum(i * s for i, s in zip(index, kb)) * 8
        weight = np.array(float(beta)) if step == 0 else one
        pa, pb, pc = a.data.ptr + shift_a, b.data.ptr + shift_b, out.data.ptr
        if count == 1:
            cublas.dgemm(handle, plan.trans_b, plan.trans_a, plan.n, plan.m, plan.k,
                         scale.ctypes.data, pb, plan.ldb, pa, plan.lda,
                         weight.ctypes.data, pc, plan.ldc)
        elif uniform is not None:
            ua, ub, uc = uniform
            cublas.dgemmStridedBatched(handle, plan.trans_b, plan.trans_a, plan.n, plan.m, plan.k,
                                       scale.ctypes.data, pb, plan.ldb, ub, pa, plan.lda, ua,
                                       weight.ctypes.data, pc, plan.ldc, uc, count)
        else:
            arrays = [o + p for o, p in zip(offsets, (pb, pa, pc))]
            cublas.dgemmBatched(handle, plan.trans_b, plan.trans_a, plan.n, plan.m, plan.k,
                                scale.ctypes.data, arrays[0].data.ptr, plan.ldb,
                                arrays[1].data.ptr, plan.lda, weight.ctypes.data,
                                arrays[2].data.ptr, plan.ldc, count)
            del arrays


def _einsum_update(spec, a, b, out, alpha, beta):
    import cupy as cp

    value = cp.einsum(spec, a, b, optimize=True)
    if alpha != 1.0:
        # Some contractions return views of their inputs.
        value = value * alpha
    if beta == 0.0:
        out[...] = value
    else:
        if beta != 1.0:
            out *= beta
        out += value


# Autotuned route per contraction shape: True for direct GEMMs, False for einsum.
AUTOTUNE = os.environ.get("ITERCC_GEMM_AUTOTUNE", "1") != "0"
_route = {}


def _seconds(function):
    import cupy as cp

    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    function()
    stop.record()
    stop.synchronize()
    return cp.cuda.get_elapsed_time(start, stop) / 1000


def einsum_into(spec, a, b, out, alpha=1.0, beta=0.0, *, cost_limit=False):
    """out = beta * out + alpha * einsum(spec, a, b) by the faster route.

    The first contraction of each shape and stride pattern runs both direct
    GEMMs and einsum on the real data (restoring `out` in between) and keeps
    the faster for that shape (`routes` exports the choices, and
    `restore_routes` reuses them in another process). Set
    ITERCC_GEMM_AUTOTUNE=0 to use the static data-movement estimate instead.
    With `cost_limit`, discard expensive direct plans before timing them.
    """
    import cupy as cp

    if not all(isinstance(x, cp.ndarray) and x.dtype == cp.float64 for x in (a, b, out)):
        _einsum_update(spec, a, b, out, alpha, beta)
        return out
    if out.size == 0:
        return out
    key = _key(spec, a, b, out)
    plan_key = (key, cost_limit or not AUTOTUNE)
    route_key = (key, cost_limit, AUTOTUNE)
    route = _route.get(route_key)
    if route is None:
        plan = _plans.get(plan_key)
        if plan is None:
            plan = _plan(spec, a, b, out, cost=plan_key[1])
            _plans[plan_key] = plan if plan is not None else False
        if not plan:
            route = _route[route_key] = False
        elif not AUTOTUNE:
            route = _route[route_key] = True
        else:
            saved = out.copy() if beta else None
            direct = _seconds(lambda: _direct(key, plan, a, b, out, alpha, beta))
            if saved is not None:
                out[...] = saved
            einsum = _seconds(lambda: _einsum_update(spec, a, b, out, alpha, beta))
            _route[route_key] = direct < einsum
            if direct < einsum:
                if saved is not None:
                    out[...] = saved
                _direct(key, plan, a, b, out, alpha, beta)
            return out
    plan = _plans.get(plan_key) if route else False
    if plan is None:                    # a route restored from another process
        plan = _plan(spec, a, b, out, cost=plan_key[1])
        _plans[plan_key] = plan if plan is not None else False
    if plan:
        _direct(key, plan, a, b, out, alpha, beta)
    else:
        _einsum_update(spec, a, b, out, alpha, beta)
    return out


def _device_name(device):
    import cupy as cp

    name = cp.cuda.runtime.getDeviceProperties(device)["name"]
    return name.decode() if isinstance(name, bytes) else str(name)


def routes():
    """The routes this process timed, as JSON-ready rows (see `restore_routes`)."""
    rows = []
    for (key, cost_limit, tuned), direct in list(_route.items()):
        if tuned:                       # static routes are the same in every process
            device, spec, *layout = key
            rows.append(dict(gpu=_device_name(device), spec=spec, cost_limit=bool(cost_limit),
                             layout=[list(part) for part in layout], direct=bool(direct)))
    return rows


def restore_routes(rows):
    """Take over the routes another process timed, so that a resumed solve
    rounds exactly as the saved one did (the direct and einsum routes round
    differently, and timing picks either). Rows from another GPU model are
    skipped: cuBLAS itself rounds differently there, so it retunes instead."""
    import cupy as cp

    devices = {}
    for device in range(cp.cuda.runtime.getDeviceCount()):
        devices.setdefault(_device_name(device), []).append(device)
    for row in rows:
        layout = tuple(tuple(part) for part in row["layout"])
        for device in devices.get(row["gpu"], ()):
            _route[(device, row["spec"], *layout), row["cost_limit"], True] = row["direct"]

