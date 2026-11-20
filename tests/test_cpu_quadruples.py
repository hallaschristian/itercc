"""The compiled CPU T4 kernels agree with the NumPy layout (the reference)."""
import numpy as np
import pytest

from itercc import _cpu_quadruples
from itercc._quadruples_cpu import NumpyQuadruplesLayout, _spin_sum
from itercc._quadruples_cuda import _SPIN

pytestmark = pytest.mark.skipif(not _cpu_quadruples.available(), reason="compiled core unavailable")

OCC = np.array([0, 3, 1, 2, 0])
VIR = np.sort(np.array([0, 0, 1, 2, 2, 3, 0, 1, 3]))
TOL = dict(rtol=1e-13, atol=1e-13)


@pytest.fixture(params=["symmetric", "plain"])
def layouts(request):
    occ, vir = (OCC, VIR) if request.param == "symmetric" else (None, None)
    nocc, nvir = len(OCC), len(VIR)
    return (NumpyQuadruplesLayout(nocc, nvir, occ, vir),
            _cpu_quadruples.CompiledQuadruplesLayout(nocc, nvir, occ, vir, threads=3))


def packed(layout, seed):
    return np.random.default_rng(seed).normal(size=layout.size)


BOUNDS = [(0, 2, 1, 3, 0, 3, 2, 4), (1, 3, 1, 3, 1, 3, 1, 3), (3, 5, 0, 2, 2, 5, 4, 5)]


@pytest.mark.parametrize("kind", ["plain", "triples", "P4_201", "P4_442", "P4_442_r2", "project"])
@pytest.mark.parametrize("bounds", BOUNDS)
def test_gather(layouts, kind, bounds):
    reference, compiled = layouts
    t4 = packed(reference, 1)
    shape = (2, 2, 3, 2) + (len(VIR),) * 4          # wider than some bounds: zero rows
    options = dict(triples=kind == "triples", spin=None if kind in ("plain", "triples") else kind)
    expected, actual = np.full(shape, np.nan), np.full(shape, np.nan)
    reference.gather(t4, expected, bounds, **options)
    compiled.gather(t4, actual, bounds, **options)
    np.testing.assert_allclose(actual, expected, **TOL)


@pytest.mark.parametrize("alpha, beta", [(1.0, 0.0), (0.5, 1.0), (-2.0, 0.25)])
@pytest.mark.parametrize("bounds", BOUNDS)
def test_scatter(layouts, bounds, alpha, beta):
    reference, compiled = layouts
    tile = np.random.default_rng(2).normal(size=(2, 2, 3, 2) + (len(VIR),) * 4)
    expected, actual = packed(reference, 3), packed(reference, 3)
    reference.scatter(expected, tile, bounds, alpha=alpha, beta=beta)
    compiled.scatter(actual, tile, bounds, alpha=alpha, beta=beta)
    np.testing.assert_allclose(actual, expected, **TOL)


def test_add_rows(layouts):
    reference, compiled = layouts
    rng = np.random.default_rng(4)
    nocc, nv = len(OCC), len(VIR)
    for kind, args in (("pair", (1, 3, 0)), ("pair", (2, 2, 1)), ("triple", ((0, 2, 4),)),
                       ("triple", ((1, 1, 3),))):
        rows = rng.normal(size=(nocc,) + (nv,) * 4)
        expected, actual = packed(reference, 5), packed(reference, 5)
        method = "pair_entries" if kind == "pair" else "triple_entries"
        reference.add_rows(expected, rows, getattr(reference, method)(*args))
        compiled.add_rows(actual, rows, getattr(compiled, method)(*args))
        np.testing.assert_allclose(actual, expected, **TOL)


def test_expand_compress_project_divide(layouts):
    reference, compiled = layouts
    t4 = packed(reference, 6)
    for first, stop in ((0, 7), (3, len(reference.indices))):
        segment = t4[reference.segment(first, stop)]
        dense = reference.expand(segment, first, stop)
        np.testing.assert_array_equal(compiled.expand(segment, first, stop), dense)
        expected, actual = np.zeros_like(segment), np.zeros_like(segment)
        reference.compress(dense * 2, expected, first)
        compiled.compress(np.ascontiguousarray(dense * 2), actual, first)
        np.testing.assert_array_equal(actual, expected)
    expected, actual, dense = t4.copy(), t4.copy(), t4.copy()
    reference.project(expected)
    compiled.project(actual)
    np.testing.assert_allclose(actual, expected, **TOL)
    # The packed projection has the arithmetic of the dense compiled kernels.
    NumpyQuadruplesLayout.project(compiled, dense)
    np.testing.assert_array_equal(actual, dense)
    eia = -1.0 - np.random.default_rng(7).random((len(OCC), len(VIR)))
    reference.divide(expected, eia)
    compiled.divide(actual, eia)
    np.testing.assert_allclose(actual, expected, **TOL)
    assert compiled.max_abs(actual) == reference.max_abs(actual)
    actual[5] = np.nan
    assert np.isnan(compiled.max_abs(actual))


@pytest.mark.parametrize("pattern", sorted(_SPIN))
@pytest.mark.parametrize("alpha, beta", [(1.0, 0.0), (-0.5, 1.5)])
def test_spin_summation(pattern, alpha, beta):
    rng = np.random.default_rng(8)
    rank = len(_SPIN[pattern][0][0])
    source = rng.normal(size=(3,) + (7,) * rank)
    target = rng.normal(size=source.shape)
    expected = target.copy()
    _spin_sum(source, expected, pattern, alpha, beta)
    _cpu_quadruples.spin_summation(source, target, pattern, alpha, beta, threads=4)
    np.testing.assert_allclose(target, expected, rtol=1e-13, atol=1e-12)
    # In place (the source is copied first).
    inplace = source.copy()
    _cpu_quadruples.spin_summation(inplace, inplace, pattern, alpha, 0.0)
    expected = np.empty_like(source)
    _spin_sum(source, expected, pattern, alpha, 0.0)
    np.testing.assert_allclose(inplace, expected, rtol=1e-13, atol=1e-12)


def test_array_helpers_match_numpy():
    """The compiled whole-array operations compute the NumPy expressions exactly."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=(3, 3, 3, 5, 5, 5))
    for perm in ((1, 0, 2, 4, 3, 5), (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4), (0, 2, 1, 3, 5, 4)):
        np.testing.assert_array_equal(_cpu_quadruples.add_transpose_(x.copy(), perm),
                                      x + x.transpose(perm))
    # A permutation that is not its own inverse takes NumPy's path (same result).
    np.testing.assert_array_equal(_cpu_quadruples.add_transpose_(x.copy(), (1, 2, 0, 3, 4, 5)),
                                  x + x.transpose(1, 2, 0, 3, 4, 5))
    t3 = rng.normal(size=(3, 3, 3, 5, 5, 5))
    np.testing.assert_array_equal(_cpu_quadruples.add_transposed(t3, (0, 1, 2, 4, 5, 3)),
                                  t3 + t3.transpose(0, 1, 2, 4, 5, 3))
    out = rng.normal(size=t3.shape)
    expected = 0.5 * out - 2.0 * t3.transpose(1, 0, 2, 4, 3, 5)
    np.testing.assert_array_equal(
        _cpu_quadruples.permuted_add(out, out, t3.transpose(1, 0, 2, 4, 3, 5), 0.5, -2.0), expected)
    stored = _cpu_quadruples.relayout(x, (4, 5, 0, 1, 2, 3))
    np.testing.assert_array_equal(stored, x)
    assert stored.strides[4] > stored.strides[5] > stored.strides[0] > stored.strides[3]
    # Already stored so: the array itself; and pairwise symmetrization through
    # the transposed storage (W_vvvvoo is made occupied-outermost on the CPU).
    assert _cpu_quadruples.relayout(stored, (4, 5, 0, 1, 2, 3)) is stored
    for perm in ((0, 2, 1, 3, 5, 4), (1, 0, 2, 4, 3, 5)):
        through = _cpu_quadruples.relayout(x, (4, 5, 0, 1, 2, 3))
        assert _cpu_quadruples.add_transpose_(through, perm) is through
        np.testing.assert_array_equal(through, x + x.transpose(perm))
    eia = -1.0 - rng.random((3, 5))
    d2 = eia[:, None, :, None] + eia[None, :, None, :]
    r3 = rng.normal(size=t3.shape)
    expected = r3 / (d2[:, :, None, :, :, None] + eia[None, None, :, None, None, :])
    np.testing.assert_array_equal(_cpu_quadruples.divide_triples(r3.copy(), eia), expected)
    y, update = rng.normal(size=1000), rng.normal(size=1000)
    np.testing.assert_array_equal(_cpu_quadruples.add_(y.copy(), update), y + update)
    np.testing.assert_array_equal(_cpu_quadruples.scale_(y.copy(), 0.8), y * 0.8)
    assert _cpu_quadruples.max_abs(y) == max(y.max(), -y.min())


def test_compiled_project_triples_matches_numpy():
    from itercc._quadruples_cpu import project_triples as reference

    r3 = np.random.default_rng(12).normal(size=(4, 4, 4, 6, 6, 6))
    expected, actual = r3.copy(), r3.copy()
    reference(expected)
    _cpu_quadruples.project_triples(actual)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("kind", ["P4_442_r2", "P4_201", "ring"])
def test_gather_permute_matches_gather_then_permute(kind):
    """The one-pass spin (or ring) sum of packed block rows equals gathering the
    rows and permuting them, bit for bit."""
    from itercc import _cpu_packed

    layout = _cpu_quadruples.CompiledQuadruplesLayout(len(OCC), len(VIR), OCC, VIR, threads=3)
    t4 = packed(layout, 9)
    occ = np.asarray(layout.occupied_irreps, dtype=np.int64)
    for p, q in ((0, 0), (1, 3), (4, 2)):
        rows = _cpu_packed.PackedRows(layout, [occ, occ], int(occ[p] ^ occ[q]))
        slots, orders = _cpu_packed._row_slots(
            layout, [(m, n, p, q) for m in range(len(OCC)) for n in range(len(OCC))])
        offsets = np.ascontiguousarray(rows.offset.reshape(-1), dtype=np.int64)
        plain, expected, actual = np.zeros(rows.size), np.zeros(rows.size), np.zeros(rows.size)
        layout._kernels.quad_gather_rows(layout._layout, t4.ctypes.data, len(slots), slots.ctypes.data,
                                         orders.ctypes.data, offsets.ctypes.data, plain.ctypes.data, 3)
        _cpu_packed._permute(layout, plain, expected, rows, kind, 3)
        _cpu_packed._gather_permute(layout, t4, actual, rows, slots, orders, kind, 3)
        np.testing.assert_array_equal(actual, expected)
        assert np.abs(expected).max() > 0


def test_diis_dots_and_combine(monkeypatch):
    """The compiled DIIS passes over T4-sized vectors: inner products and linear
    combinations agree with NumPy and are the same bits on any thread count."""
    rng = np.random.default_rng(10)
    n = 3 * (1 << 14) + 123
    x = rng.normal(size=n)
    others = [rng.normal(size=n) for _ in range(4)]
    weights = rng.normal(size=4)
    results = []
    for threads in ("1", "3", "8"):
        monkeypatch.setenv("ITERCC_CPU_THREADS", threads)
        dots = _cpu_quadruples.dots(x, others)
        out = _cpu_quadruples.combine(weights, others, np.empty(n))
        copied = _cpu_quadruples.copy(x)
        np.testing.assert_allclose(dots, [x @ y for y in others] + [x @ x], rtol=1e-12)
        np.testing.assert_allclose(out, sum(w * y for w, y in zip(weights, others)), rtol=1e-12,
                                   atol=1e-13)
        np.testing.assert_array_equal(copied, x)
        results.append((dots, out))
    for dots, out in results[1:]:
        np.testing.assert_array_equal(dots, results[0][0])
        np.testing.assert_array_equal(out, results[0][1])
