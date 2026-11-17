"""The compiled setup kernels reproduce the NumPy and PySCF constructions."""
import numpy as np
import pytest

from itercc import _cpu_kernels
from itercc import spatial_symmetry

pytestmark = pytest.mark.skipif(_cpu_kernels.library() is None, reason="compiled core unavailable")

LABELS = np.array([0, 1, 2, 3, 0, 1, 0, 2, 3])


def leaky_integrals(rng, size):
    n = len(LABELS)
    eri = rng.standard_normal((n,) * 4)
    total = LABELS[:, None, None, None] ^ LABELS[None, :, None, None] ^ LABELS[None, None, :, None] \
        ^ LABELS[None, None, None, :]
    return np.where(total == 0, eri, size * eri)


def test_compiled_projection_matches_the_numpy_projection():
    rng = np.random.default_rng(4)
    eri = leaky_integrals(rng, 1e-10)
    projected, audit = spatial_symmetry._project_eri(eri, LABELS, 1e-8)
    reference, expected = spatial_symmetry._project(
        eri, spatial_symmetry._eri_forbidden(LABELS[:, None] ^ LABELS), 1e-8)
    np.testing.assert_array_equal(projected, reference)
    assert audit.count == expected.count
    assert audit.maximum == expected.maximum
    assert audit.norm == pytest.approx(expected.norm, rel=1e-12)
    with pytest.raises(ValueError, match="leakage"):
        spatial_symmetry._project_eri(leaky_integrals(rng, 1e-3), LABELS, 1e-8)
    nan = eri.copy()
    nan[0, 0, 0, 1] = np.nan                                  # forbidden entry (0 ^ 1 != 0)
    with pytest.raises(ValueError):
        spatial_symmetry._project_eri(nan, LABELS, 1e-8)


def test_compiled_unpack_matches_ao2mo_restore():
    pytest.importorskip("pyscf")
    from pyscf import ao2mo

    from itercc.pyscf_interface import _full_eri

    rng = np.random.default_rng(6)
    n = 11
    npair = n * (n + 1) // 2
    packed = rng.standard_normal((npair, npair))
    packed = packed + packed.T
    np.testing.assert_array_equal(_full_eri(packed, n), ao2mo.restore(1, packed, n))


@pytest.mark.parametrize("no", [0, 4, 13])
def test_compiled_dressing_derivative_matches_the_leg_by_leg_sum(no):
    library = _cpu_kernels.library()
    if not hasattr(library, "dress_derivative"):
        pytest.skip("core without dress_derivative")
    rng = np.random.default_rng(8)
    n = 13
    g = rng.standard_normal((n,) * 4)
    d = np.ascontiguousarray(rng.standard_normal((no, n - no)))
    o, v = slice(0, no), slice(no, n)
    reference = np.zeros_like(g)
    for axis in (0, 1):                                       # creation legs
        np.moveaxis(reference, axis, 0)[v] -= np.tensordot(d.T, np.moveaxis(g, axis, 0)[o], axes=1)
    for axis in (2, 3):                                       # annihilation legs
        np.moveaxis(reference, axis, 0)[o] += np.tensordot(d, np.moveaxis(g, axis, 0)[v], axes=1)
    results = []
    for threads in (1, 3):
        out = np.full_like(g, np.nan)                         # the kernel needs no initialized output
        library.dress_derivative(out.ctypes.data, g.ctypes.data, n, no, d.ctypes.data, threads)
        results.append(out)
    np.testing.assert_allclose(results[0], reference, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(results[0], results[1])     # fixed sums at any thread count




def _pair_symmetric(rng, n, no):
    g = rng.standard_normal((n,) * 4)
    g = g + g.transpose(1, 0, 3, 2)                          # pair symmetry
    t2 = rng.standard_normal((no, no, n - no, n - no))
    return g, t2 + t2.transpose(1, 0, 3, 2), 0.2 * rng.standard_normal((no, n - no))


@pytest.mark.parametrize("transposed", [False, True])
def test_ccsd_dressing_matches_the_full_dressing_where_ccsd_reads(transposed):
    from itercc._cpu_triples import dressed_integrals

    if not hasattr(_cpu_kernels.library(), "dress_integrals_ccsd"):
        pytest.skip("core without dress_integrals_ccsd")
    rng = np.random.default_rng(9)
    n, no = 11, 3
    g0, _, t1 = _pair_symmetric(rng, n, no)
    if transposed:
        g0 = g0.transpose(0, 2, 1, 3)                         # read through strides
    full = dressed_integrals(g0, t1, 3)
    ccsd = dressed_integrals(g0, t1, 3, ccsd=True, out=np.full((n,) * 4, np.nan))
    o, v = slice(0, no), slice(no, n)
    np.testing.assert_array_equal(ccsd[o], full[o])           # an occupied first or second index
    np.testing.assert_array_equal(ccsd[v, o], full[v, o])
    left_out = np.einsum("ie,jf,abef->abij", t1, t1, np.asarray(g0)[v, v, v, v])
    np.testing.assert_allclose(ccsd[v, v, o, o] + left_out, full[v, v, o, o], rtol=1e-12, atol=1e-12)
    assert np.isnan(ccsd[v, v, v]).all() and np.isnan(ccsd[v, v, o, v]).all()   # never written


def test_bare_ladder_adds_what_the_ccsd_dressing_leaves_out():
    from itercc._cpu_triples import BareLadder, doubles_ladder, dressed_integrals

    rng = np.random.default_rng(10)
    n, no = 12, 3
    g, t2, t1 = _pair_symmetric(rng, n, no)
    dressed = dressed_integrals(g, t1, 2)
    o, v = slice(0, no), slice(no, n)
    ladder = 0.5 * np.einsum("abcd,ijcd->ijab", dressed[v, v, v, v], t2)
    np.testing.assert_allclose(doubles_ladder(dressed[v, v, v, v], t2), ladder, rtol=1e-12, atol=1e-12)
    # What the CCSD residual takes from both: 0.5 g~(abij) + the ladder.
    reference = 0.5 * dressed[v, v, o, o].transpose(2, 3, 0, 1) + ladder
    ccsd = dressed_integrals(g, t1, 2, ccsd=True)
    total = 0.5 * ccsd[v, v, o, o].transpose(2, 3, 0, 1) + BareLadder(g, no, threads=2)(t1, t2)
    np.testing.assert_allclose(total, reference, rtol=1e-11, atol=1e-11)
