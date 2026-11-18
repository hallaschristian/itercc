"""Memory-bounded tensor contractions for NumPy and CuPy arrays.

`einsum` in both libraries materializes the full result and, on CuPy, a
transposed copy of each operand. For many-index tensors (triples or
quadruples amplitudes, any basis or augmentation) those temporaries dominate
memory. These helpers run a contraction in slabs of one index so that every
temporary stays under a byte budget, without changing what is computed:

* `einsum(xp, spec, *operands)`: small results; a contracted index shared by
  every operand is sliced and the partial results are summed.
* `products_into(xp, targets, spec, terms)`: view += c * sum_t phase_t *
  einsum(spec, *operands_t) for any number of output-shaped, possibly strided
  target views (for example transposed views of one block). An output index
  is sliced, so neither the full product nor a transposed copy exists.
* `einsum_into` / `einsum_new`: the single-product forms of the above.

Output-index slabs leave every element's arithmetic unchanged; contracted
-index slabs change only the summation order of the sliced index. The budget
is `SLAB_BYTES` (environment variable ITERCC_SLAB_MIB, default 256 MiB)
unless passed per call. Accumulation uses one fused kernel with FMA
contraction disabled, the same arithmetic as `previous + coefficient*product`.
"""
from functools import lru_cache
from math import ceil

import numpy as np

from .gemm_contract import einsum_into as gemm_into
from . import _runtime

SLAB_BYTES = int(float(_runtime.switch("ITERCC_SLAB_MIB")) * 2**20)
CUDA_GEMM = _runtime.enabled("ITERCC_BOUNDED_GEMM")


def split(spec):
    """('ab,bc->ac') -> (['ab', 'bc'], 'ac')."""
    inputs, output = spec.replace(" ", "").split("->")
    return inputs.split(","), output


def plan(spec, operands, output_shape, *, output_index, itemsize=8, budget=None):
    """Return (index, extent, slab count) or None when no slab is needed.

    With `output_index` the slab runs over an output index (results are
    accumulated slab by slab); otherwise over a contracted index shared by
    every operand (partial results are summed). The index chosen is the
    longest one present in every operand or result larger than the budget.
    """
    budget = SLAB_BYTES if budget is None else budget
    subscripts, output = split(spec)
    extents = {}
    for subscript, operand in zip(subscripts, operands):
        extents.update(zip(subscript, operand.shape))
    large = [s for s, a in zip(subscripts, operands) if a.nbytes > budget]
    output_bytes = itemsize * int(np.prod(output_shape, dtype=np.int64))
    biggest = max([a.nbytes for a in operands] + [output_bytes])
    if output_index and output_bytes > budget:
        large.append(output)
    if not large:
        return None
    pool = set(output) if output_index else set.intersection(*map(set, subscripts)) - set(output)
    candidates = [c for c in pool if all(c in s for s in large)]
    if not candidates:
        return None
    char = max(candidates, key=lambda c: extents[c])
    count = min(extents[char], ceil(biggest / budget))
    return (char, extents[char], count) if count > 1 else None


def take(subscript, operand, char, piece):
    """Slice `operand` along the axis labelled `char`, if it has one."""
    if char not in subscript:
        return operand
    index = [slice(None)] * operand.ndim
    index[subscript.index(char)] = piece
    return operand[tuple(index)]


def pieces(extent, count):
    step = ceil(extent / count)
    return [slice(start, min(start + step, extent)) for start in range(0, extent, step)]


def _host_accumulate(previous, product, coefficient, result):
    result[...] = previous + coefficient * product


@lru_cache(maxsize=None)
def accumulator(xp):
    """result = previous + coefficient * product, elementwise and in place."""
    if xp is np:
        return _host_accumulate
    return xp.ElementwiseKernel("T previous, T product, T coefficient", "T result",
                                "result = previous + coefficient * product",
                                "itercc_bounded_accumulate", options=("--fmad=false",))


def einsum(xp, spec, *operands, optimize=True, budget=None):
    """einsum(spec, *operands) with contracted-index slabs for large operands."""
    subscripts, output = split(spec)
    extents = {}
    for subscript, operand in zip(subscripts, operands):
        extents.update(zip(subscript, operand.shape))
    shape = [extents[c] for c in output]
    chosen = plan(spec, operands, shape, output_index=False, budget=budget,
                  itemsize=np.result_type(*operands).itemsize)
    if xp is np and len(operands) == 2 and optimize is True:
        # The CPU core's GEMM executor, as for products into a target.
        from ._cpu_triples import product_into

        result = np.empty(shape)
        slabs = [None] if chosen is None else pieces(chosen[1], chosen[2])
        for n, piece in enumerate(slabs):
            parts = operands if piece is None else (take(s, a, chosen[0], piece) for s, a in zip(subscripts, operands))
            if not product_into(spec, *parts, result, 1.0, 0.0 if n == 0 else 1.0):
                break           # not applicable (dtype, no core): the same for every slab
        else:
            return result
    if chosen is None:
        return xp.einsum(spec, *operands, optimize=optimize)
    char, extent, count = chosen
    result = None
    for piece in pieces(extent, count):
        part = xp.einsum(spec, *(take(s, a, char, piece) for s, a in zip(subscripts, operands)),
                         optimize=optimize)
        if result is None:
            result = part
        else:
            result += part
        del part
    return result


def products_into(xp, targets, spec, terms, *, optimize=True, budget=None):
    """view += coefficient * sum_t phase_t * einsum(spec, *operands_t) for every target.

    `targets` are (coefficient, view) pairs of output-shaped arrays that may be
    strided views; `terms` are (phase, operands) sharing `spec`. Each output
    slab of the summed product is formed once and added to every target.
    """
    targets, terms = list(targets), [(float(p), tuple(ops)) for p, ops in terms]
    if not targets or not terms:
        return
    accumulate = accumulator(xp)
    subscripts, output = split(spec)
    first = targets[0][1]
    chosen = plan(spec, terms[0][1], first.shape, output_index=True, itemsize=first.dtype.itemsize,
                  budget=budget)
    for piece in [None] if chosen is None else pieces(chosen[1], chosen[2]):
        def cut(subscript, array):
            return array if piece is None else take(subscript, array, chosen[0], piece)
        if len(terms) == 1:
            # One product: accumulate it directly (it is only read).
            phase, operands = terms[0]
            operands = tuple(cut(s, a) for s, a in zip(subscripts, operands))
            if xp is np and len(operands) == 2 and len(targets) == 1 and optimize is True:
                # The CPU core's GEMM executor adds the product in place (False: not applicable).
                from ._cpu_triples import product_into

                coefficient, view = targets[0]
                if product_into(spec, *operands, cut(output, view), coefficient * phase, 1.0):
                    continue
            if CUDA_GEMM and xp is not np and len(operands) == 2 and len(targets) == 1 and optimize is True:
                coefficient, view = targets[0]
                view = cut(output, view)
                if (view.dtype == operands[0].dtype == operands[1].dtype == np.float64
                        and not any(xp.may_share_memory(view, operand) for operand in operands)):
                    gemm_into(spec, *operands, view, alpha=coefficient * phase, beta=1.0,
                              cost_limit=True)
                    continue
            total = xp.einsum(spec, *operands, optimize=optimize)
            scale = phase
        else:
            total, scale = None, 1.
            for phase, operands in terms:
                value = xp.einsum(spec, *(cut(s, a) for s, a in zip(subscripts, operands)), optimize=optimize)
                if total is None:
                    # einsum may return a view of an operand (pure transposes).
                    total = xp.multiply(value, phase)
                else:
                    accumulate(total, value, total.dtype.type(phase), total)
                del value
        for coefficient, view in targets:
            view = cut(output, view)
            accumulate(view, total, view.dtype.type(coefficient * scale), view)
        del total


def einsum_into(xp, target, coefficient, spec, *operands, optimize=True, budget=None):
    """target += coefficient * einsum(spec, *operands), bounded by output slabs."""
    products_into(xp, [(coefficient, target)], spec, [(1., operands)], optimize=optimize, budget=budget)
    return target


def einsum_new(xp, coefficient, spec, *operands, optimize=True, budget=None):
    """coefficient * einsum(spec, *operands) assembled slab by slab into a new array."""
    subscripts, output = split(spec)
    extents = {}
    for subscript, operand in zip(subscripts, operands):
        extents.update(zip(subscript, operand.shape))
    target = xp.zeros(tuple(extents[c] for c in output), dtype=np.result_type(*operands))
    return einsum_into(xp, target, coefficient, spec, *operands, optimize=optimize, budget=budget)
