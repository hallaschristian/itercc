"""Symmetry-blocked contractions agree with dense einsum on symmetric tensors."""
import re

import numpy as np
import pytest

from itercc import symmetry_blocks
from itercc._quadruples_equations import __file__ as EQUATIONS
from test_quadruples import cuda  # noqa: F401  (fixture)

OCC = np.array([0, 3, 1, 2, 0, 0, 3])          # labels of seven occupied orbitals
VIR = np.sort(np.array([0, 0, 1, 2, 2, 3, 0, 1, 3]))


def tile_specs():
    text = open(EQUATIONS).read()
    specs = set(re.findall(r"(?:einsum|tile)\(\s*['\"]([a-z]+,[a-z]+->ijklabcd)['\"]", text))
    return sorted(specs)


def symmetric(cp, rng, letters, labels):
    shape = tuple(len(labels[c]) for c in letters)
    value = rng.normal(size=shape)
    grids = np.meshgrid(*[labels[c] for c in letters], indexing="ij")
    allowed = np.bitwise_xor.reduce(np.stack(grids), axis=0) == 0
    return cp.asarray(np.where(allowed, value, 0.0))


@pytest.fixture(params=["direct", "fused", "gemm"])
def route(request, monkeypatch):
    """Run a test with every plan on one route: the direct kernel, the shared-memory
    fused kernel (any depth) or gathers + cuBLAS + scatter."""
    monkeypatch.setattr(symmetry_blocks, "_FUSED_DEPTH", 0 if request.param == "gemm" else 10**9)
    monkeypatch.setattr(symmetry_blocks, "_DIRECT_DEPTH", 10**9 if request.param == "direct" else 0)
    monkeypatch.setattr(symmetry_blocks, "_FUSED_SKINNY", 10**9)
    return request.param


@pytest.mark.parametrize("spec", tile_specs()[:200])
def test_tile_terms_match_dense_einsum(cuda, route, spec):
    cp = cuda
    if not symmetry_blocks.available():
        pytest.skip("cuBLAS grouped GEMM unavailable")
    rng = np.random.default_rng(len(spec) * 31 + sum(map(ord, spec)))
    ranges = dict(i=(0, 2), j=(2, 4), k=(3, 5), l=(5, 7))
    labels = {c: OCC[slice(*r)] for c, r in ranges.items()}
    labels.update(m=OCC, n=OCC)
    labels.update({c: VIR for c in "abcdef"})
    (sa, sb), so = spec.split("->")[0].split(","), spec.split("->")[1]
    a, b = symmetric(cp, rng, sa, labels), symmetric(cp, rng, sb, labels)
    out = symmetric(cp, rng, so, labels)
    blocks = symmetry_blocks.SymmetryBlocks(labels, "ijklmn", 4)
    for alpha, beta in ((0.5, 1.0), (-1.0, 0.0)):
        expected = beta * out + alpha * cp.einsum(spec, a, b)
        blocks.einsum(spec, a, b, out, alpha, beta)
        np.testing.assert_allclose(cp.asnumpy(out), cp.asnumpy(expected), rtol=1e-12, atol=1e-12,
                                   err_msg=spec)


def test_strided_views_and_fallback(cuda, route):
    cp = cuda
    if not symmetry_blocks.available():
        pytest.skip("cuBLAS grouped GEMM unavailable")
    rng = np.random.default_rng(2)
    labels = dict(i=OCC[1:3], m=OCC, a=VIR, b=VIR, e=VIR)
    big = symmetric(cp, rng, "mibae", dict(m=OCC, i=OCC, b=VIR, a=VIR, e=VIR))
    w = big[:, 1:3]                                   # strided view over i
    t = symmetric(cp, rng, "me", dict(m=OCC, e=VIR))
    out = cp.zeros((2, len(VIR), len(VIR)))
    calls = []
    blocks = symmetry_blocks.SymmetryBlocks(labels, "i", 4, fallback=lambda *x, **k: calls.append(x))
    blocks.einsum("mibae,me->iab", w, t, out, 1.0, 0.0)
    np.testing.assert_allclose(cp.asnumpy(out), cp.asnumpy(cp.einsum("mibae,me->iab", w, t)),
                               rtol=1e-12, atol=1e-12)
    blocks.einsum("ma,ma->a", t, t, cp.zeros(len(VIR)), 1.0, 0.0)   # batch index: falls back
    assert len(calls) == 1


def test_combination_labels_change_per_call(cuda, route):
    cp = cuda
    if not symmetry_blocks.available():
        pytest.skip("cuBLAS grouped GEMM unavailable")
    rng = np.random.default_rng(9)
    spec = "abej,iklecd->ijklabcd"
    base = dict(m=OCC, n=OCC, **{c: VIR for c in "abcdef"})
    blocks = symmetry_blocks.SymmetryBlocks(dict(base, i=OCC[:2], j=OCC[:2], k=OCC[:2], l=OCC[:2]),
                                            "ijklmn", 4)
    for tile in ((0, 2, 4, 5), (1, 3, 2, 5), (4, 0, 1, 3)):
        labels = {c: OCC[t:t + 2] for c, t in zip("ijkl", tile)}
        full = dict(base, **labels)
        a = symmetric(cp, rng, "abej", full)
        b = symmetric(cp, rng, "iklecd", full)
        out = cp.zeros((2, 2, 2, 2) + (len(VIR),) * 4)
        blocks.einsum(spec, a, b, out, 1.0, 0.0, labels=labels)
        np.testing.assert_allclose(cp.asnumpy(out), cp.asnumpy(cp.einsum(spec, a, b)),
                                   rtol=1e-12, atol=1e-12)
    assert len(blocks._structures) == 1 and len(blocks._groups) == 3


def test_operand_irreps_and_per_call_labels_for_any_letter(cuda, route):
    """Blocks of T4 with fixed occupied indices: operand irreps and chunked summed indices."""
    cp = cuda
    if not symmetry_blocks.available():
        pytest.skip("cuBLAS grouped GEMM unavailable")
    rng = np.random.default_rng(13)
    base = dict(x=OCC, m=OCC, **{c: VIR for c in "abcde"})
    blocks = symmetry_blocks.SymmetryBlocks(base, "", 4)
    for g_block in range(4):
        # block[m, a, e, c, d] has total irrep g_block (a fixed spectator triple).
        full = dict(base)
        shape = tuple(len(full[c]) for c in "maecd")
        grids = np.meshgrid(*[full[c] for c in "maecd"], indexing="ij")
        allowed = np.bitwise_xor.reduce(np.stack(grids), axis=0) == g_block
        block = cp.asarray(np.where(allowed, rng.normal(size=shape), 0.0))
        w = symmetric(cp, rng, "mbxe", full)
        out = cp.zeros((len(OCC),) + (len(VIR),) * 4)
        blocks.einsum("mbxe,maecd->xabcd", w, block, out, 1.0, 0.0, irreps=(0, g_block))
        np.testing.assert_allclose(cp.asnumpy(out), cp.asnumpy(cp.einsum("mbxe,maecd->xabcd", w, block)),
                                   rtol=1e-12, atol=1e-12)
    # A chunk of the summed occupied index m.
    w = symmetric(cp, rng, "mbxe", base)
    t = symmetric(cp, rng, "maecd", dict(base))
    out = cp.zeros((len(OCC),) + (len(VIR),) * 4)
    blocks.einsum("mbxe,maecd->xabcd", w[2:5], t[2:5], out, 1.0, 0.0, labels=dict(m=OCC[2:5]))
    np.testing.assert_allclose(cp.asnumpy(out), cp.asnumpy(cp.einsum("mbxe,maecd->xabcd", w[2:5], t[2:5])),
                               rtol=1e-12, atol=1e-12)
