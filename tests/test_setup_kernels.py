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
def test_ccsd_dressing_matches_the_full_dressing_where_ccsd_reads(transposed, monkeypatch):
    from itercc._cpu_triples import dressed_integrals

    if not hasattr(_cpu_kernels.library(), "dress_integrals_ccsd"):
        pytest.skip("core without dress_integrals_ccsd")
    rng = np.random.default_rng(9)
    n, no = 11, 3
    g0, _, t1 = _pair_symmetric(rng, n, no)
    if transposed:
        g0 = g0.transpose(0, 2, 1, 3)                         # read through strides
    full = dressed_integrals(g0, t1, 3)
    schedules = []
    for slabs in ("1", "0"):                                  # slab by slab, plane by plane
        monkeypatch.setenv("ITERCC_DRESS_SLABS", slabs)
        schedules.append(dressed_integrals(g0, t1, 3, ccsd=True, out=np.full((n,) * 4, np.nan)))
    np.testing.assert_array_equal(schedules[0], schedules[1])
    ccsd = schedules[0]
    o, v = slice(0, no), slice(no, n)
    np.testing.assert_array_equal(ccsd[o], full[o])           # an occupied first or second index
    np.testing.assert_array_equal(ccsd[v, o], full[v, o])
    left_out = np.einsum("ie,jf,abef->abij", t1, t1, np.asarray(g0)[v, v, v, v])
    np.testing.assert_allclose(ccsd[v, v, o, o] + left_out, full[v, v, o, o], rtol=1e-12, atol=1e-12)
    assert np.isnan(ccsd[v, v, v]).all() and np.isnan(ccsd[v, v, o, v]).all()   # never written


@pytest.mark.parametrize("transposed", [False, True])
def test_open_dressing_matches_the_full_dressing_outside_vvvv(transposed, monkeypatch):
    from itercc._cpu_triples import dressed_integrals

    if not hasattr(_cpu_kernels.library(), "dress_integrals_open"):
        pytest.skip("core without dress_integrals_open")
    rng = np.random.default_rng(12)
    n, no = 13, 4
    g0, _, t1 = _pair_symmetric(rng, n, no)
    if transposed:
        g0 = g0.transpose(0, 2, 1, 3)                         # read through strides
    full = dressed_integrals(g0, t1, 3)
    o, v = slice(0, no), slice(no, n)
    for slabs in ("1", "0"):                                  # slab by slab, plane by plane
        for threads in (1, 3):
            monkeypatch.setenv("ITERCC_DRESS_SLABS", slabs)
            part = dressed_integrals(g0, t1, threads, open=True, out=np.full((n,) * 4, np.nan))
            np.testing.assert_array_equal(part[o], full[o])       # the same values exactly
            np.testing.assert_array_equal(part[v, o], full[v, o])
            np.testing.assert_array_equal(part[v, v, o], full[v, v, o])
            np.testing.assert_array_equal(part[v, v, v, o], full[v, v, v, o])
            assert np.isnan(part[v, v, v, v]).all()              # never written


@pytest.mark.parametrize("labelled", [False, True])
def test_dressed_ladder_sectors_from_the_bare_ones(labelled):
    from itercc._blocked import irrep_groups
    from itercc._cpu_triples import LadderSectors, dressed_integrals

    if not hasattr(_cpu_kernels.library(), "ladder_sectors_t1"):
        pytest.skip("core without ladder_sectors_t1")
    rng = np.random.default_rng(13)
    labels = np.array([0, 1, 0, 2, 3, 0, 1, 2, 0, 3, 1, 0]) if labelled else np.zeros(12, dtype=np.int64)
    n, no = len(labels), 3
    g, _, t1 = _pair_symmetric(rng, n, no)
    total = labels[:, None, None, None] ^ labels[None, :, None, None] ^ labels[None, None, :, None]         ^ labels[None, None, None, :]
    g[total != 0] = 0.0                                       # symmetric integrals and T1
    t1[labels[:no, None] != labels[None, no:]] = 0.0
    g = g.transpose(0, 2, 1, 3).copy().transpose(0, 2, 1, 3)  # a strided view
    structure = LadderSectors(labels[no:], irrep_groups(labels))
    v = slice(no, n)
    bare = structure.build(g[v, v, v, v])
    expected = structure.build(dressed_integrals(g, t1, 2)[v, v, v, v])
    for threads in (1, 3):
        found = structure.dressed(bare, g, no, t1, threads=threads)
        for a, b in zip(found, expected):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("labelled", [False, True])
def test_open_dressing_from_the_ladder_sectors(labelled, monkeypatch):
    from itercc._blocked import irrep_groups
    from itercc._cpu_triples import LadderSectors, dressed_integrals

    if not hasattr(_cpu_kernels.library(), "sectors_contract_t1"):
        pytest.skip("core without sectors_contract_t1")
    rng = np.random.default_rng(14)
    labels = np.array([0, 1, 0, 2, 3, 0, 1, 2, 0, 3, 1, 0, 2]) if labelled else np.zeros(13, dtype=np.int64)
    n, no = len(labels), 4
    g, _, t1 = _pair_symmetric(rng, n, no)
    total = labels[:, None, None, None] ^ labels[None, :, None, None] ^ labels[None, None, :, None]         ^ labels[None, None, None, :]
    g[total != 0] = 0.0
    t1[labels[:no, None] != labels[None, no:]] = 0.0
    g = g.transpose(0, 2, 1, 3).copy().transpose(0, 2, 1, 3)  # a strided view
    structure = LadderSectors(labels[no:], irrep_groups(labels))
    o, v = slice(0, no), slice(no, n)
    contracted = structure.contract_t1(structure.build(g[v, v, v, v]), t1, threads=3)
    np.testing.assert_allclose(contracted, np.einsum("abcd,jd->abcj", g[v, v, v, v], t1), rtol=1e-12, atol=1e-12)
    full = dressed_integrals(g, t1, 3)
    for slabs in ("1", "0"):
        monkeypatch.setenv("ITERCC_DRESS_SLABS", slabs)
        part = dressed_integrals(g, t1, 3, open=True, contracted=contracted, out=np.full((n,) * 4, np.nan))
        for block in ((o,), (v, o), (v, v, o), (v, v, v, o)):
            np.testing.assert_allclose(part[block], full[block], rtol=1e-12, atol=1e-12)
        assert np.isnan(part[v, v, v, v]).all()


def test_ccsdt_with_and_without_the_open_dressing(monkeypatch):
    pytest.importorskip("pyscf")
    from pyscf import gto, scf

    from itercc import RestrictedCC
    from itercc.pyscf_interface import from_rhf

    mol = gto.M(atom="O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469", basis="6-31g", symmetry=True,
                verbose=0)
    m = from_rhf(scf.RHF(mol).run(conv_tol=1e-12), frozen=1)
    results = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("ITERCC_OPEN_DRESSING", flag)
        cc = RestrictedCC(m.h, m.eri, m.nocc, m.constant, orbital_irreps=m.orbital_irreps,
                          symmetry_tolerance=1e-8, rank=3)
        t1 = 0.01 * np.random.default_rng(3).standard_normal(cc.shapes[0])
        t1[m.orbital_irreps[:m.nocc, None] != m.orbital_irreps[None, m.nocc:]] = 0.0
        opened = cc._open_dressing(t1)
        assert opened == (flag == "1")
        results[flag] = cc.solve(energy_tol=1e-10, residual_tol=1e-8, max_iter=60)
        assert results[flag].converged
        assert (cc._ladder_bare is not None) == opened            # the bare sectors were used
    assert abs(results["0"].energy - results["1"].energy) < 1e-10
    for a, b in zip(results["0"].amplitudes, results["1"].amplitudes):
        np.testing.assert_allclose(a, b, atol=1e-9)


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


@pytest.mark.parametrize("no", [0, 4])
def test_ccsd_dressing_derivative_matches_the_full_one_where_ccsd_reads(no):
    library = _cpu_kernels.library()
    if not hasattr(library, "dress_derivative_ccsd"):
        pytest.skip("core without dress_derivative_ccsd")
    rng = np.random.default_rng(11)
    n = 13
    g = rng.standard_normal((n,) * 4)
    d = np.ascontiguousarray(rng.standard_normal((no, n - no)))
    full = np.empty_like(g)
    library.dress_derivative(full.ctypes.data, g.ctypes.data, n, no, d.ctypes.data, 2)
    results = []
    for threads in (1, 3):
        out = np.full_like(g, np.nan)
        library.dress_derivative_ccsd(out.ctypes.data, g.ctypes.data, n, no, d.ctypes.data, threads)
        results.append(out)
    ccsd = results[0]
    o, v = slice(0, no), slice(no, n)
    np.testing.assert_array_equal(ccsd[o], full[o])
    np.testing.assert_array_equal(ccsd[v, o], full[v, o])
    np.testing.assert_array_equal(ccsd[v, v, o, o], full[v, v, o, o])
    assert np.isnan(ccsd[v, v, v]).all() and np.isnan(ccsd[v, v, o, v]).all()   # never written
    np.testing.assert_array_equal(results[0], results[1])


def _symmetric_model(rng, labels, no):
    """Pair-symmetric g without symmetry-forbidden entries, A1 t1 and t2."""
    n = len(labels)
    g, t2, t1 = _pair_symmetric(rng, n, no)
    total = labels[:, None, None, None] ^ labels[None, :, None, None] ^ labels[None, None, :, None] \
        ^ labels[None, None, None, :]
    occupied, virtual = labels[:no], labels[no:]
    pair = occupied[:, None, None, None] ^ occupied[None, :, None, None] ^ virtual[None, None, :, None] \
        ^ virtual[None, None, None, :]
    return np.where(total == 0, g, 0.0), np.where(pair == 0, t2, 0.0), \
        np.where(occupied[:, None] == virtual[None, :], t1, 0.0)


@pytest.mark.parametrize("tau", [True, False])
@pytest.mark.parametrize("symmetry", [None, "sectors", "scan"])
def test_bare_ladder_tangent_is_its_derivative(tau, symmetry):
    from itercc._blocked import irrep_groups
    from itercc._cpu_triples import BareLadder, LadderSectors

    rng = np.random.default_rng(12)
    no, labels = 3, np.array([0, 1, 0, 0, 2, 1, 3, 0, 2, 3, 1, 0])
    g, t2, t1 = _symmetric_model(rng, labels, no)
    # A direction of irrep 2: the derivative of a non-A1 perturbation.
    occupied, virtual = labels[:no], labels[no:]
    dt1 = np.where((occupied[:, None] ^ virtual[None, :]) == 2, rng.standard_normal(t1.shape), 0.0)
    dt2 = rng.standard_normal(t2.shape)
    dt2 = dt2 + dt2.transpose(1, 0, 3, 2)
    pair = occupied[:, None, None, None] ^ occupied[None, :, None, None] ^ virtual[None, None, :, None] \
        ^ virtual[None, None, None, :]
    dt2 = np.where(pair == 2, dt2, 0.0)
    options = {}
    if symmetry == "sectors":
        options = dict(structure=LadderSectors(virtual, irrep_groups(labels)))
    elif symmetry == "scan":
        options = dict(virtual_irreps=virtual)
    bare = BareLadder(g, no, threads=2, **options)

    def along(step):
        return bare(t1 + step * dt1, t2 + step * dt2, tau=tau)

    # The ladder is cubic in the step: two central differences give its slope exactly.
    h = 0.1
    wide, narrow = (along(h) - along(-h)) / (2 * h), (along(h / 2) - along(-h / 2)) / h
    slope = (4 * narrow - wide) / 3
    tangent = bare.tangent(t1, t2, dt1, dt2, tau=tau)
    np.testing.assert_allclose(tangent, slope, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(bare.tangent(t1, t2, dt1, dt2, tau=tau), tangent)   # cached point


def test_eom_actions_agree_with_and_without_the_bare_ladder(monkeypatch):
    pytest.importorskip("pyscf")
    from pyscf import gto, scf

    from itercc import EOMSinglet, RestrictedCC
    from itercc.pyscf_interface import from_rhf

    mol = gto.M(atom="O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469", basis="6-31g", symmetry=True,
                verbose=0)
    m = from_rhf(scf.RHF(mol).run(conv_tol=1e-12), frozen=1)
    for rank in (2, 3):
        cc = RestrictedCC(m.h, m.eri, m.nocc, m.constant, orbital_irreps=m.orbital_irreps,
                          symmetry_tolerance=1e-8, rank=rank)
        ground = cc.solve(energy_tol=1e-12, residual_tol=1e-10, max_iter=100)
        images = []
        for flag in ("0", "1"):
            monkeypatch.setenv("ITERCC_BARE_LADDER", flag)
            for irrep in (0, 2):
                eom = EOMSinglet(cc, ground.amplitudes, orbital_irreps=m.orbital_irreps, irrep=irrep,
                                 method="ccsd" if rank == 2 else "ccsdt")
                assert (eom.proxy._tangent_ladder is not None) == (flag == "1")
                vector = np.random.default_rng(irrep).standard_normal(eom.space.dimension
                                                                       if hasattr(eom.space, "dimension")
                                                                       else len(eom.diagonal))
                images.append(eom.sigma(vector))
        for with_bare, without in zip(images[2:], images[:2]):
            np.testing.assert_allclose(with_bare, without, rtol=1e-11, atol=1e-11)
    monkeypatch.delenv("ITERCC_BARE_LADDER")                  # 8 virtuals: the dense ladder
    eom = EOMSinglet(cc, ground.amplitudes, orbital_irreps=m.orbital_irreps, method="ccsdt")
    assert eom.proxy._tangent_ladder is None


@pytest.mark.parametrize("columns", [1, 4, 10, 16])
def test_narrow_gemm_matches_numpy(columns):
    from itercc._cpu_triples import _narrow, matmul

    if not hasattr(_cpu_kernels.library(), "gemm_narrow"):
        pytest.skip("core without gemm_narrow")
    rng = np.random.default_rng(columns)
    big = rng.standard_normal((300, 420))
    a = big[:, 7:407]                                        # row stride beyond the row length
    b = np.ascontiguousarray(rng.standard_normal((400, columns)))
    assert _narrow(a, b, np.empty((300, columns)))
    reference = a @ b
    np.testing.assert_allclose(matmul(a, b, np.empty((300, columns))), reference, rtol=1e-12, atol=1e-12)
    out = rng.standard_normal((300, columns))
    expected = 0.5 * out - 2.0 * reference
    np.testing.assert_allclose(matmul(a, b, out, alpha=-2.0, beta=0.5), expected, rtol=1e-12, atol=1e-12)
    one = matmul(a, b, np.empty((300, columns)), threads=1)
    np.testing.assert_array_equal(matmul(a, b, np.empty((300, columns)), threads=5), one)


def test_allocator_thresholds_are_set_once_unless_opted_out(monkeypatch):
    import platform
    import sys

    from itercc import _core

    for name in ("MALLOC_MMAP_THRESHOLD_", "MALLOC_TRIM_THRESHOLD_"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delitem(_core._state, "malloc", raising=False)
    monkeypatch.setenv("ITERCC_MALLOC", "0")
    assert _core.keep_freed_memory() is False
    assert _core.keep_freed_memory() is False                 # decided once per process
    monkeypatch.delitem(_core._state, "malloc")
    monkeypatch.delenv("ITERCC_MALLOC")
    glibc = sys.platform.startswith("linux") and platform.libc_ver()[0] == "glibc"
    assert _core.keep_freed_memory() is glibc


def test_heap_scope_nests_and_restores_mapping(monkeypatch):
    from itercc import _core

    calls = []
    if not _core.keep_freed_memory():
        with _core.heap_scope() as scope:                     # nothing to change here
            assert scope.active is False
        return
    monkeypatch.setitem(_core._state, "libc", (lambda option, value: calls.append((option, value)) or 1,
                                               lambda pad: calls.append(("trim", pad)) or 1))
    with _core.heap_scope():
        with _core.heap_scope():
            assert calls == [(-4, 0)]                          # M_MMAP_MAX 0 once
        assert calls == [(-4, 0)]
    assert calls == [(-4, 0), (-4, 65536), ("trim", 0)]        # restored by the outermost scope
    with pytest.raises(RuntimeError):
        with _core.heap_scope():
            raise RuntimeError
    assert calls[-2:] == [(-4, 65536), ("trim", 0)] and _core._heap["depth"] == 0


@pytest.mark.parametrize("no", [0, 3])
def test_spin_orbital_dressing_derivative_takes_the_compiled_pass(no, monkeypatch):
    """The triplet actions' T1-dressing derivative (`dress_direction`) runs
    the singlet actions' one-pass kernel on the CPU."""
    if _cpu_kernels.library() is None:
        pytest.skip("no compiled core")
    from itercc._directional import dress_direction

    rng = np.random.default_rng(11 + no)
    n = 9
    g = rng.standard_normal((n,) * 4)
    d = rng.standard_normal((n - no, no))
    compiled = dress_direction(g, d, no, np)
    monkeypatch.setenv("ITERCC_CPU_KERNELS", "0")
    reference = dress_direction(g, d, no, np)
    np.testing.assert_allclose(compiled, reference, rtol=1e-13, atol=1e-12)
