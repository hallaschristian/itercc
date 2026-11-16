"""The compiled CPU symmetry-blocked einsum agrees with dense einsum on symmetric tensors."""
import re
from pathlib import Path

import numpy as np
import pytest

import itercc
from itercc import _cpu_kernels
from itercc._cpu_symmetry_blocks import CpuSymmetryBlocks

pytestmark = pytest.mark.skipif(_cpu_kernels.library() is None, reason="compiled core unavailable")

OCC = np.array([0, 3, 1, 2, 0, 0, 3])          # labels of seven occupied orbitals
VIR = np.sort(np.array([0, 0, 1, 2, 2, 3, 0, 1, 3]))
EQUATIONS = Path(itercc.__file__).with_name("_quadruples_equations.py")


def tile_specs():
    text = EQUATIONS.read_text()
    return sorted(set(re.findall(r"(?:einsum|tile)\(\s*['\"]([a-z]+,[a-z]+->ijklabcd)['\"]", text)))


def symmetric(rng, letters, labels, irrep=0):
    shape = tuple(len(labels[c]) for c in letters)
    grids = np.meshgrid(*[labels[c] for c in letters], indexing="ij")
    allowed = np.bitwise_xor.reduce(np.stack(grids), axis=0) == irrep
    return np.where(allowed, rng.normal(size=shape), 0.0)


def sorted_labels(n, seed):
    return np.sort(np.random.default_rng(seed).integers(0, 4, size=n))


@pytest.mark.parametrize("spec", tile_specs()[:200])
def test_tile_terms_match_dense_einsum(spec):
    rng = np.random.default_rng(len(spec) * 31 + sum(map(ord, spec)))
    ranges = dict(i=(0, 2), j=(2, 4), k=(3, 5), l=(5, 7))
    labels = {c: OCC[slice(*r)] for c, r in ranges.items()}
    labels.update(m=OCC, n=OCC)
    labels.update({c: VIR for c in "abcdef"})
    (sa, sb), so = spec.split("->")[0].split(","), spec.split("->")[1]
    a, b = symmetric(rng, sa, labels), symmetric(rng, sb, labels)
    out = symmetric(rng, so, labels)
    blocks = CpuSymmetryBlocks(labels, "ijklmn", 4, fallback=None)
    for alpha, beta in ((0.5, 1.0), (-1.0, 0.0), (2.0, -0.5)):
        expected = beta * out + alpha * np.einsum(spec, a, b)
        blocks.einsum(spec, a, b, out, alpha, beta)
        np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12, err_msg=spec)


ROUTES = {
    # Plain loops: blocks of at most a few thousand multiply-adds.
    "loops": ("abef,efcd->abcd", dict(zip("abcdef", [VIR] * 6))),
    # Single-threaded GEMMs spread over the threads (about 36^3 per block).
    "threaded": ("abef,efcd->abcd", dict(zip("abcdef", [sorted_labels(12, 1)] * 6))),
    # Large GEMMs on the BLAS threads (about 144^3 per block).
    "large": ("abef,efcd->abcd", dict(zip("abcdef", [sorted_labels(24, 2)] * 6))),
    # Skinny: a short side against a long one, the long operand read in place.
    "skinny_rows": ("xme,meabcd->xabcd", dict(x=OCC[:3], m=OCC, e=VIR, a=VIR, b=VIR, c=VIR, d=VIR)),
    "skinny_cols": ("abcdme,mex->abcdx", dict(x=OCC[:3], m=OCC, e=VIR, a=VIR, b=VIR, c=VIR, d=VIR)),
    # A small output with a deep sum, split into slices.
    "split": ("mnefa,mnefb->ab", dict(m=sorted_labels(12, 3), n=sorted_labels(12, 4),
                                      e=sorted_labels(16, 5), f=sorted_labels(16, 6),
                                      a=sorted_labels(8, 7), b=sorted_labels(8, 8))),
}


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_routes_match_dense_einsum_and_ignore_thread_count(route):
    spec, labels = ROUTES[route]
    rng = np.random.default_rng(5)
    (sa, sb), so = spec.split("->")[0].split(","), spec.split("->")[1]
    a, b = symmetric(rng, sa, labels), symmetric(rng, sb, labels)
    expected = np.einsum(spec, a, b)
    results = []
    for threads in (1, 3, 8):
        blocks = CpuSymmetryBlocks(labels, "", 4, fallback=None, threads=threads)
        out = np.full(expected.shape, np.nan)
        blocks.einsum(spec, a, b, out, 1.0, 0.0)
        np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-11)
        results.append(out)
    for out in results[1:]:
        np.testing.assert_array_equal(out, results[0])


@pytest.mark.parametrize("order", ["c", "a_transposed", "b_transposed", "out_transposed"])
def test_operand_memory_orders(order):
    """Gathers and the large-block GEMM follow the operands' memory order."""
    spec, labels = ROUTES["large"]
    rng = np.random.default_rng(6)
    a, b = symmetric(rng, "abef", labels), symmetric(rng, "efcd", labels)
    out = np.zeros((24,) * 4)
    if order == "a_transposed":
        a = np.ascontiguousarray(a.transpose(3, 2, 1, 0)).transpose(3, 2, 1, 0)
    elif order == "b_transposed":
        b = np.ascontiguousarray(b.transpose(3, 2, 1, 0)).transpose(3, 2, 1, 0)
    elif order == "out_transposed":
        out = out.T
    blocks = CpuSymmetryBlocks(labels, "", 4, fallback=None)
    blocks.einsum(spec, a, b, out, 1.0, 0.0)
    np.testing.assert_allclose(out, np.einsum(spec, a, b), rtol=1e-12, atol=1e-11)


def test_strided_views_and_fallback():
    rng = np.random.default_rng(2)
    labels = dict(i=OCC[1:3], m=OCC, a=VIR, b=VIR, e=VIR)
    big = symmetric(rng, "mibae", dict(m=OCC, i=OCC, b=VIR, a=VIR, e=VIR))
    w = big[:, 1:3]                                   # strided view over i
    t = symmetric(rng, "me", dict(m=OCC, e=VIR))
    out = np.zeros((2, len(VIR), len(VIR)))
    calls = []
    blocks = CpuSymmetryBlocks(labels, "i", 4, fallback=lambda *x, **k: calls.append(x))
    blocks.einsum("mibae,me->iab", w, t, out, 1.0, 0.0)
    np.testing.assert_allclose(out, np.einsum("mibae,me->iab", w, t), rtol=1e-12, atol=1e-12)
    blocks.einsum("ma,ma->a", t, t, np.zeros(len(VIR)), 1.0, 0.0)   # batch index: falls back
    assert len(calls) == 1
    # Strided output: zeroed and scaled in NumPy, blocks written through the strides.
    out = np.ones((2, len(VIR), 2 * len(VIR)))[:, :, ::2]
    blocks.einsum("mibae,me->iab", w, t, out, 1.0, 0.0)
    np.testing.assert_allclose(out, np.einsum("mibae,me->iab", w, t), rtol=1e-12, atol=1e-12)


def test_combination_labels_change_per_call():
    rng = np.random.default_rng(9)
    spec = "abej,iklecd->ijklabcd"
    base = dict(m=OCC, n=OCC, **{c: VIR for c in "abcdef"})
    blocks = CpuSymmetryBlocks(dict(base, i=OCC[:2], j=OCC[:2], k=OCC[:2], l=OCC[:2]), "ijklmn", 4,
                               fallback=None)
    for tile in ((0, 2, 4, 5), (1, 3, 2, 5), (4, 0, 1, 3)):
        labels = {c: OCC[t:t + 2] for c, t in zip("ijkl", tile)}
        full = dict(base, **labels)
        a = symmetric(rng, "abej", full)
        b = symmetric(rng, "iklecd", full)
        out = np.zeros((2, 2, 2, 2) + (len(VIR),) * 4)
        blocks.einsum(spec, a, b, out, 1.0, 0.0, labels=labels)
        np.testing.assert_allclose(out, np.einsum(spec, a, b), rtol=1e-12, atol=1e-12)
    assert len(blocks._structures) == 1 and len(blocks._groups) == 3


def test_operand_irreps_and_per_call_labels_for_any_letter():
    """Blocks of T4 with fixed occupied indices: operand irreps and chunked summed indices."""
    rng = np.random.default_rng(13)
    base = dict(x=OCC, m=OCC, **{c: VIR for c in "abcde"})
    blocks = CpuSymmetryBlocks(base, "", 4, fallback=None)
    for g_block in range(4):
        block = symmetric(rng, "maecd", base, irrep=g_block)
        w = symmetric(rng, "mbxe", base)
        out = np.zeros((len(OCC),) + (len(VIR),) * 4)
        blocks.einsum("mbxe,maecd->xabcd", w, block, out, 1.0, 0.0, irreps=(0, g_block))
        np.testing.assert_allclose(out, np.einsum("mbxe,maecd->xabcd", w, block),
                                   rtol=1e-12, atol=1e-12)
    w = symmetric(rng, "mbxe", base)
    t = symmetric(rng, "maecd", base)
    out = np.zeros((len(OCC),) + (len(VIR),) * 4)
    blocks.einsum("mbxe,maecd->xabcd", w[2:5], t[2:5], out, 1.0, 0.0, labels=dict(m=OCC[2:5]))
    np.testing.assert_allclose(out, np.einsum("mbxe,maecd->xabcd", w[2:5], t[2:5]),
                               rtol=1e-12, atol=1e-12)


def test_without_compiled_core_everything_falls_back(monkeypatch):
    rng = np.random.default_rng(3)
    labels = dict(zip("abcdef", [VIR] * 6))
    a, b = symmetric(rng, "abef", labels), symmetric(rng, "efcd", labels)
    blocks = CpuSymmetryBlocks(labels, "", 4)
    monkeypatch.setattr(blocks, "_library", None)
    out = np.zeros((len(VIR),) * 4)
    blocks.einsum("abef,efcd->abcd", a, b, out, 2.0, 0.0)
    np.testing.assert_allclose(out, 2.0 * np.einsum("abef,efcd->abcd", a, b), rtol=1e-12, atol=1e-12)
