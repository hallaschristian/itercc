"""GEMM contractions into strided views agree with einsum."""
import zlib

import numpy as np
import pytest

from itercc import gemm_contract
from itercc._quadruples_equations import __file__ as EQUATIONS


@pytest.fixture(scope="module")
def cuda():
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        count = 0
    if not count:
        pytest.skip("CUDA GPU required")
    return cp


def equation_specs():
    import re
    text = open(EQUATIONS).read()
    specs = set(re.findall(r"(?:einsum|tile)\(\s*['\"]([a-z]+,[a-z]+->[a-z]+)['\"]", text))
    return sorted(specs)


def strided(cp, rng, shape, pad):
    """A view of a larger array, sliced so every axis has a nontrivial stride."""
    big = cp.asarray(rng.normal(size=tuple(n + p for n, p in zip(shape, pad))))
    return big[tuple(slice(p // 2, p // 2 + n) for n, p in zip(shape, pad))]


@pytest.mark.parametrize("spec", equation_specs() + [
    "ia,jb->ijab", "ij,jk->ik", "ijk,kj->i", "ab,b->a", "pqr,qrs->ps", "xab,xbc->xac"])
@pytest.mark.parametrize("views", [False, True])
def test_contract_matches_einsum(cuda, spec, views):
    cp = cuda
    rng = np.random.default_rng(zlib.crc32(spec.encode()))
    letters = sorted(set(spec) - set(",->"))
    extent = dict(zip(letters, rng.integers(1, 5, size=len(letters))))
    (sa, sb), so = [s.split(",") for s in spec.split("->")][0], spec.split("->")[1]
    shapes = [tuple(int(extent[c]) for c in s) for s in (sa, sb, so)]
    if views:
        a, b, out = (strided(cp, rng, shape, rng.integers(0, 3, size=len(shape))) for shape in shapes)
    else:
        a, b, out = (cp.asarray(rng.normal(size=shape)) for shape in shapes)
    for alpha, beta in ((1.0, 0.0), (-0.5, 1.0), (0.25, -2.0)):
        expected = beta * out + alpha * cp.einsum(spec, a, b)
        target = out.copy() if not views else out
        before = target.copy()
        if not gemm_contract.contract(spec, a, b, target, alpha, beta):
            pytest.skip("not a GEMM contraction")
        np.testing.assert_allclose(cp.asnumpy(target), cp.asnumpy(expected), rtol=1e-12, atol=1e-12,
                                   err_msg=f"{spec} alpha={alpha} beta={beta}")
        if views:
            target[...] = before


@pytest.mark.parametrize("spec,extents", [
    ("abef,ijklefcd->ijklabcd", dict(i=2, j=2, k=2, l=2, a=9, b=9, c=9, d=9, e=9, f=9)),
    ("mnij,klmncdab->ijklabcd", dict(m=2, n=5, i=2, j=2, k=2, l=2, a=9, b=9, c=9, d=9)),
    ("mbie,mjklaecd->ijklabcd", dict(m=5, i=2, j=2, k=2, l=2, a=9, b=9, c=9, d=9, e=9)),
    ("ae,ijklebcd->ijklabcd", dict(i=2, j=2, k=2, l=2, a=9, b=9, c=9, d=9, e=9)),
])
def test_tile_products_have_direct_gemm_plans(cuda, spec, extents):
    """Products whose output order allows it can accumulate without einsum temporaries."""
    cp = cuda
    (sa, sb), so = spec.split("->")[0].split(","), spec.split("->")[1]
    a, b, out = (cp.ones(tuple(extents[c] for c in s)) for s in (sa, sb, so))
    assert gemm_contract._plan(spec, a, b, out, cost=False) is not None


@pytest.mark.parametrize("spec", ["mnij,klmncdab->ijklabcd", "acef,ijklebfd->ijklabcd",
                                  "iejcbm,mklaed->ijklabcd", "kldc,kldb->bc"])
def test_autotuned_route_is_exact_and_reused(cuda, spec):
    cp = cuda
    rng = np.random.default_rng(11)
    letters = sorted(set(spec) - set(",->"))
    extent = {c: (2 if c in "ijklmn" else 6) for c in letters}
    (sa, sb), so = spec.split("->")[0].split(","), spec.split("->")[1]
    a, b, out = (cp.asarray(rng.normal(size=tuple(extent[c] for c in s))) for s in (sa, sb, so))
    for alpha, beta in ((0.5, 1.0), (-1.0, 0.0), (2.0, -0.5)):
        expected = beta * out + alpha * cp.einsum(spec, a, b)
        gemm_contract.einsum_into(spec, a, b, out, alpha, beta)
        np.testing.assert_allclose(cp.asnumpy(out), cp.asnumpy(expected), rtol=1e-12, atol=1e-12)
    assert (gemm_contract._key(spec, a, b, out), False,
            gemm_contract.AUTOTUNE) in gemm_contract._route

