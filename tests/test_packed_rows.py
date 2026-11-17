"""The CPU blocked engine reads signed packed rows in place, exactly as their dense gather."""
import numpy as np
import pytest

from itercc import _cpu_kernels
from itercc._cpu_symmetry_blocks import CpuSymmetryBlocks
from itercc._tensor_scatter import SignedRows

pytestmark = pytest.mark.skipif(_cpu_kernels.library() is None, reason="compiled core unavailable")


def signed_rows(rng, lengths, entries):
    """Rows (lengths) of `entries` values each, laid out as entries..., rows... like packed T4."""
    rows, width = int(np.prod(lengths)), int(np.prod(entries))
    window = 3 * width
    vector = rng.normal(size=rows * window)
    bases, maps = [], []
    for r in range(rows):
        if r % 5 == 3:                                  # a zero row (a repeated hole)
            bases.append(0)
            maps.append(None)
            continue
        bases.append((-1 if r % 2 else 1) * (r * window + 1))
        mapping = rng.integers(-window, window + 1, size=width).astype(np.int32)
        mapping[rng.random(width) < 0.3] = 0
        maps.append(mapping)
    axes = tuple(range(len(lengths), len(lengths) + len(entries))) + tuple(range(len(lengths)))
    return SignedRows(vector, bases, maps, width, lengths, entries, axes)


# (spec, rows letter extent p, entries (a, b, c), row lengths (k, l)): B is abckl.
CASES = {
    "loops": ("pckl,abckl->pab", 2, (3, 3, 2), (2, 2)),
    "shallow": ("pckl,abckl->pab", 40, (30, 30, 2), (2, 2)),
    "skinny, long B": ("pckl,abckl->pab", 6, (40, 40, 2), (2, 3)),
    "skinny, short B": ("pckl,abckl->pab", 1100, (3, 4, 5), (2, 2)),
    "gemm": ("pckl,abckl->pab", 48, (24, 24, 6), (2, 3)),
    "gemm, rows fastest": ("pckl,abckl->abp", 48, (24, 24, 6), (2, 3)),
    "split sum": ("pckl,abckl->pab", 8, (4, 4, 128), (4, 5)),
    "row letter in columns": ("pck,abckl->pabl", 32, (12, 12, 4), (3, 4)),
    "packed first": ("abckl,pckl->pab", 48, (24, 24, 6), (2, 3)),
}


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("case", sorted(CASES))
def test_packed_rows_equal_dense_gather(case, threads):
    spec, p, entries, lengths = CASES[case]
    rng = np.random.default_rng(sum(map(ord, case)))
    rows = signed_rows(rng, lengths, entries)
    dense = rows.dense()
    extents = dict(zip("abckl", dense.shape), p=p)
    (first, second), output = spec.split("->")[0].split(","), spec.split("->")[1]
    small = first if "p" in first else second
    a = rng.normal(size=tuple(extents[c] for c in small))
    start = rng.normal(size=tuple(extents[c] for c in output))
    engine = CpuSymmetryBlocks({c: np.zeros(n, dtype=np.int64) for c, n in extents.items()}, "", 1,
                               fallback=None, threads=threads)
    for alpha, beta in ((1.0, 0.0), (-0.5, 1.0)):
        expected, actual = start.copy(), start.copy()
        operands = (a, dense) if small == first else (dense, a)
        engine.einsum(spec, *operands, expected, alpha, beta)
        operands = (a, rows) if small == first else (rows, a)
        engine.einsum(spec, *operands, actual, alpha, beta)
        np.testing.assert_array_equal(actual, expected)
        reference = beta * start + alpha * np.einsum(spec, *(a, dense) if small == first else (dense, a))
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)


def test_signed_rows_describe_their_dense_tensor():
    rows = signed_rows(np.random.default_rng(5), (2, 3), (4, 5))
    dense = rows.dense()
    assert rows.shape == dense.shape == (4, 5, 2, 3)
    assert rows.strides == dense.strides and rows.ndim == 4 and rows.dtype == np.float64
