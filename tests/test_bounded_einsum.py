"""Slab-bounded contractions equal plain einsum for any budget, backend and dtype."""
import os

import numpy as np
import pytest

from itercc import bounded_einsum as bounded


def backends():
    yield np
    if os.environ.get("ITERCC_TEST_CUDA") == "1":
        import cupy
        yield cupy


def random(xp, rng, shape, dtype):
    value = rng.standard_normal(shape)
    if np.issubdtype(dtype, np.complexfloating):
        value = value + 1j * rng.standard_normal(shape)
    return xp.asarray(value.astype(dtype))


@pytest.mark.parametrize("xp", list(backends()), ids=lambda m: m.__name__)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("budget", [None, 64, 1 << 10])
def test_products_into_matches_einsum_for_strided_targets(xp, dtype, budget):
    rng = np.random.default_rng(1)
    f = random(xp, rng, (5, 4), dtype)
    t = random(xp, rng, (4, 3, 5, 2, 2, 2), dtype)
    r = random(xp, rng, (4, 3, 5, 2, 2, 2), dtype)
    spec = "ae,ebcijk->abcijk"
    out = xp.zeros((5, 3, 5, 2, 2, 2), dtype=dtype)
    expected = xp.zeros_like(out)
    product = xp.einsum(spec, f, t) - 2 * xp.einsum(spec, f, r)
    targets = []
    for coefficient, axes in ((1., (0, 1, 2, 3, 4, 5)), (-.5, (2, 1, 0, 3, 4, 5)), (.25, (0, 1, 2, 5, 4, 3))):
        expected += coefficient * product.transpose(axes)
        targets.append((coefficient, out.transpose(tuple(np.argsort(axes)))))
    bounded.products_into(xp, targets, spec, [(1., (f, t)), (-2., (f, r))], budget=budget)
    np.testing.assert_allclose(out.get() if xp is not np else out,
                               expected.get() if xp is not np else expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("xp", list(backends()), ids=lambda m: m.__name__)
@pytest.mark.parametrize("budget", [None, 64])
def test_three_operand_and_contracted_slabs(xp, budget):
    rng = np.random.default_rng(2)
    a = random(xp, rng, (6, 4, 3), np.float64)
    b = random(xp, rng, (4, 3, 5), np.float64)
    c = random(xp, rng, (5, 6), np.float64)
    spec = "pqr,qrs,sp->p"
    np.testing.assert_allclose(*(v.get() if xp is not np else v for v in (
        bounded.einsum(xp, spec, a, b, c, budget=budget), xp.einsum(spec, a, b, c))), rtol=1e-13, atol=1e-12)
    target = random(xp, rng, (6, 4, 5), np.float64)
    expected = target + 3. * xp.einsum("pqr,qrs->pqs", a, b)
    bounded.einsum_into(xp, target, 3., "pqr,qrs->pqs", a, b, budget=budget)
    made = bounded.einsum_new(xp, 3., "pqr,qrs->pqs", a, b, budget=budget)
    for value in (target, made + (expected - 3. * xp.einsum("pqr,qrs->pqs", a, b))):
        np.testing.assert_allclose(*(v.get() if xp is not np else v for v in (value, expected)), rtol=0, atol=1e-12)


def test_view_returning_einsum_never_modifies_operands():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((3, 4))
    keep = a.copy()
    out = np.zeros((4, 3))
    bounded.products_into(np, [(2., out)], "ab->ba", [(-1., (a,)), (3., (a,))], budget=8)
    np.testing.assert_array_equal(a, keep)
    np.testing.assert_allclose(out, 4. * a.T, rtol=0, atol=1e-14)


def test_plan_respects_budget_and_skips_small_problems():
    a, b = np.zeros((40, 30, 20)), np.zeros((30, 20, 10))
    assert bounded.plan("pqr,qrs->ps", (a, b), (40, 10), output_index=False, budget=1 << 30) is None
    char, extent, count = bounded.plan("pqr,qrs->ps", (a, b), (40, 10), output_index=False, budget=a.nbytes // 4)
    assert char in "qr" and count >= 4
    char, extent, count = bounded.plan("pqr,qrs->pqs", (a, b), (40, 30, 10), output_index=True, budget=a.nbytes // 5)
    assert char in "pqs" and char in "pq" and count >= 5


def test_cuda_direct_gemm_preserves_aliased_operand(monkeypatch):
    if os.environ.get("ITERCC_TEST_CUDA") != "1":
        pytest.skip("CUDA test is opt-in")
    import cupy as cp

    monkeypatch.setattr(bounded, "CUDA_GEMM", True)
    target = cp.arange(16, dtype=cp.float64).reshape(4, 4)
    other = cp.arange(16, dtype=cp.float64).reshape(4, 4).T.copy()
    expected = target + target @ other
    bounded.einsum_into(cp, target, 1.0, "ij,jk->ik", target, other)
    np.testing.assert_array_equal(target.get(), expected.get())
    bounded.einsum_into(cp, target, 1.0, "ij,jk->ik", cp.empty((4, 0)), cp.empty((0, 4)))
    np.testing.assert_array_equal(target.get(), expected.get())
