"""The thread-parallel CPU triples assembly against the generic per-(j, k) loop."""

import ctypes
import os

import numpy as np
import pytest

from itercc import RestrictedCC, _cpu_kernels, _cpu_triples
from itercc.restricted import _pair_ladder
from test_restricted import make_model, trial_amplitudes

LABELS = np.array([0, 1, 2, 0, 1, 2, 3, 0, 3])


def labelled_model(no, nv, labels):
    h, eri, _, constant = make_model(no, nv)
    pair = labels[:, None] ^ labels
    h[pair != 0] = 0
    eri[pair[:, :, None, None] != pair] = 0
    return h, eri, no, constant


def symmetric_amplitudes(solver, labels, with_t3):
    no = solver.layout.no
    t1, t2, t3 = trial_amplitudes(solver, 0.05, with_t3)
    occ, vir = labels[:no], labels[no:]
    t1[occ[:, None] != vir] = 0
    t2[(occ[:, None] ^ occ)[:, :, None, None] != (vir[:, None] ^ vir)] = 0
    vir3 = vir[:, None, None] ^ vir[None, :, None] ^ vir[None, None, :]
    for tile, (i, j, k) in zip(t3, solver.triples_indices):
        tile[vir3 != (occ[i] ^ occ[j] ^ occ[k])] = 0
    return t1, t2, t3


@pytest.fixture(params=["compiled", "numpy"])
def cpu_path(request, monkeypatch):
    """Run with the OpenMP kernels (skipped without a C compiler) or the NumPy pool."""
    if request.param == "numpy":
        monkeypatch.setenv("ITERCC_CPU_KERNELS", "0")
    elif _cpu_kernels.library() is None:
        pytest.skip("no C compiler with OpenMP for the CPU kernels")
    return request.param


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("with_t3", [False, True])
def test_threaded_cpu_residual_matches_generic_loop(symmetric, with_t3, cpu_path):
    no, nv = 4, 5
    model = labelled_model(no, nv, LABELS)
    options = dict(orbital_irreps=LABELS) if symmetric else {}
    reference = RestrictedCC(*model, **options)
    reference._cpu_fast = False
    amplitudes = symmetric_amplitudes(reference, LABELS, with_t3)
    expected = reference.residuals(amplitudes)
    results = []
    for threads in (1, 3):
        solver = RestrictedCC(*model, cpu_threads=threads, **options)
        results.append(solver.residuals(amplitudes))
        for actual, wanted in zip(results[-1], expected):
            np.testing.assert_allclose(actual, wanted, rtol=0, atol=1e-15)
    # Each output entry is summed in the same order for any worker count.
    for serial, threaded in zip(*results):
        np.testing.assert_array_equal(serial, threaded)


@pytest.mark.parametrize("symmetric", [False, True])
def test_chunked_cpu_ladder_matches_generic_ladder(symmetric, cpu_path):
    rng = np.random.default_rng(8)
    labels = np.array([0, 1, 1, 0, 2, 3])
    size = len(labels)
    irreps = labels[:, None] ^ labels
    matrix = rng.normal(scale=0.1, size=(size,) * 4)
    matrix += matrix.transpose(1, 0, 3, 2).copy()
    if symmetric:
        matrix[irreps[:, :, None, None] != irreps] = 0.0
    t3 = rng.normal(scale=0.03, size=(11, size, size, size))
    virtual = labels if symmetric else None
    expected = np.zeros_like(t3)
    _pair_ladder(expected, t3, matrix, 3, virtual_irreps=virtual)
    actual = np.zeros_like(t3)
    pairs = size * size                    # symmetric plus antisymmetric pairs
    budget = 4 * 2 * 3 * size * pairs      # four slots per chunk: three chunks, one partial
    with _cpu_triples.ThreadPool(3) as pool:
        if cpu_path == "compiled":
            _cpu_triples._ladder_compiled(_cpu_kernels.library(), actual, t3, matrix, 3,
                                          virtual, budget)
        else:
            _cpu_triples.pair_ladder(actual, t3, matrix, pool, virtual_irreps=virtual,
                                     budget=budget)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)


def test_compiled_solver_steps_match_numpy():
    lib = _cpu_kernels.library()
    if lib is None:
        pytest.skip("no C compiler with OpenMP for the CPU kernels")
    ptr = _cpu_kernels.pointer
    rng = np.random.default_rng(4)
    r3 = rng.normal(size=(7, 3, 3, 3))
    occupied = rng.normal(size=7) - 3.0
    virtual = rng.normal(size=(3, 3, 3)) + 3.0
    expected = r3.copy()
    for tile, e in zip(expected, occupied):
        tile *= 0.8
        tile /= e - virtual
    lib.triples_step(ptr(r3), 7, 27, ptr(occupied), ptr(virtual), 0.8, 3)
    np.testing.assert_array_equal(r3, expected)
    assert lib.max_abs(ptr(r3), r3.size, 3) == np.max(np.abs(r3))
    # The DIIS vector operations (shared with the CCSDTQ host DIIS).
    from itercc import _cpu_quadruples

    a, b = rng.normal(size=100003), rng.normal(size=100003)
    dots = []
    for threads in (1, 4):
        with _cpu_triples.thread_scope(threads):
            dots.append(_cpu_quadruples.dots(a, [b])[0])
        assert abs(dots[-1] - a @ b) < 1e-10
    assert dots[0] == dots[1]
    out = _cpu_quadruples.permuted_add(np.empty_like(a), a, b, 1.0, 1.0)
    np.testing.assert_array_equal(out, a + b)
    _cpu_quadruples.combine([0.25, -1.5], [a, b], out)
    np.testing.assert_allclose(out, 0.25 * a - 1.5 * b, rtol=0, atol=1e-15)


def test_compiled_solve_matches_numpy_solve(monkeypatch):
    if _cpu_kernels.library() is None:
        pytest.skip("no C compiler with OpenMP for the CPU kernels")
    from itercc.restricted import _BlockDIIS

    monkeypatch.setattr(_BlockDIIS, "LARGE", 1)      # threaded DIIS on every rank
    model = labelled_model(3, 4, LABELS[:7])
    controls = dict(energy_tol=1e-12, residual_tol=1e-10, diis_size=3, diis_triples=True,
                    damping=0.1)
    compiled = RestrictedCC(*model, orbital_irreps=LABELS[:7], cpu_threads=3).solve(**controls)
    monkeypatch.setenv("ITERCC_CPU_KERNELS", "0")
    numpy = RestrictedCC(*model, orbital_irreps=LABELS[:7]).solve(**controls)
    assert compiled.converged and numpy.converged
    assert len(compiled.iterations) == len(numpy.iterations)
    assert compiled.energy == pytest.approx(numpy.energy, abs=1e-12, rel=0)
    for left, right in zip(compiled.amplitudes, numpy.amplitudes):
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-10)


def test_thread_pool_reports_task_errors():
    def task(start, stop):
        if start <= 5 < stop:
            raise RuntimeError("task failed")

    with _cpu_triples.ThreadPool(3) as pool:
        with pytest.raises(RuntimeError, match="task failed"):
            pool.run(task, 10)
        with pytest.raises(RuntimeError, match="task failed"):
            pool.join(pool.stage(lambda: pool.run(task, 10)))


def test_cpu_thread_count(monkeypatch):
    with pytest.raises(ValueError, match="cpu_threads"):
        RestrictedCC(*make_model(2, 2), cpu_threads=0)
    assert _cpu_triples.cpu_threads(5) == 5
    monkeypatch.setenv("ITERCC_CPU_THREADS", "3")
    assert _cpu_triples.cpu_threads() == 3
    monkeypatch.delenv("ITERCC_CPU_THREADS")
    # Compiled kernels default to one thread per physical core (SMT siblings
    # once), a solver's scope wins over the environment; the NumPy pool
    # defaults to at most 2.
    if _cpu_kernels.library() is not None:
        cores = _cpu_triples.cpu_threads()
        assert cores == _cpu_triples._PROCESS_CORES
        assert cores == _cpu_triples._cores(frozenset(_cpu_triples._PROCESS_CPU_SET))
        assert 1 <= cores <= _cpu_triples._available_cpus()
        monkeypatch.setenv("ITERCC_CPU_THREADS", "3")
        with _cpu_triples.thread_scope(2):
            assert _cpu_triples.cpu_threads() == 2 and _cpu_triples.cpu_threads(4) == 4
        assert _cpu_triples.cpu_threads() == 3
        monkeypatch.delenv("ITERCC_CPU_THREADS")
    monkeypatch.setenv("ITERCC_CPU_KERNELS", "0")
    assert 1 <= _cpu_triples.cpu_threads() <= 2


@pytest.mark.parametrize("threads", [1, 3])
def test_compiled_dressing_matches_numpy(threads):
    from itercc.solver import _dress_axis

    if _cpu_kernels.library() is None:
        pytest.skip("compiled CPU kernels unavailable")
    rng = np.random.default_rng(5)
    no, nv = 3, 5
    n = no + nv
    g0 = rng.standard_normal((n,) * 4)
    t1 = 0.1 * rng.standard_normal((no, nv))
    o, v = slice(None, no), slice(no, None)
    expected = g0.copy()
    for axis in (0, 1):
        _dress_axis(expected, axis, v, o, -t1.T, np)
    for axis in (2, 3):
        _dress_axis(expected, axis, o, v, t1, np)
    actual = _cpu_triples.dressed_integrals(g0, t1, threads)
    assert actual.flags.c_contiguous and not np.shares_memory(actual, g0)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    # Every thread count gives the same values.
    np.testing.assert_array_equal(actual, _cpu_triples.dressed_integrals(g0, t1, 1))


def test_subtract_vecmat_matches_numpy(monkeypatch):
    if _cpu_kernels.library() is None:
        pytest.skip("compiled CPU kernels unavailable")
    rng = np.random.default_rng(7)
    q = rng.standard_normal((3, 5000, 7))         # several blocks of the sum
    w = rng.standard_normal(5000)
    start = rng.standard_normal((3, 7))
    expected = start - 0.5 * np.matmul(w, q)
    results = []
    for threads in ("1", "4"):
        monkeypatch.setenv("ITERCC_CPU_THREADS", threads)
        out = start.copy()
        _cpu_triples.subtract_vecmat(out, w, q, 0.5)
        results.append(out)
    np.testing.assert_allclose(results[0], expected, rtol=0, atol=1e-11)
    np.testing.assert_array_equal(results[0], results[1])


def test_compiled_ladder_sectors_match_numpy(monkeypatch):
    if _cpu_kernels.library() is None:
        pytest.skip("compiled CPU kernels unavailable")
    rng = np.random.default_rng(3)
    nv = 7
    labels = np.array([0, 1, 0, 2, 3, 1, 0])
    a, b, d, e = np.ix_(labels, labels, labels, labels)
    w = rng.standard_normal((nv,) * 4) * ((a ^ b) == (d ^ e))
    compiled = _cpu_triples._ladder_sectors(w, labels)
    monkeypatch.setattr(_cpu_kernels, "library", lambda: None)
    reference = _cpu_triples._ladder_sectors(w, labels)
    for x, y in zip(compiled[0], reference[0]):
        np.testing.assert_array_equal(x, y)
    for mine, theirs in zip(compiled[1:], reference[1:]):
        assert len(mine) == len(theirs)
        for (rows, cols, block), (rows_ref, cols_ref, block_ref) in zip(mine, theirs):
            assert rows == rows_ref == cols == cols_ref         # block-diagonal: no shift
            np.testing.assert_array_equal(block, block_ref)


def test_shifted_sector_blocks_and_products():
    """A matrix coupling sector s only to s ^ G (a derivative operator of a
    direction of irrep G) splits into rectangular blocks; the blocked product
    equals the dense one, with zero rows where the partner sector is absent."""
    from itercc._pair_symmetry import PairSectors

    rng = np.random.default_rng(9)
    labels = np.array([0, 1, 0, 2, 1, 0, 2, 2])     # no sector 3: sector 1 ^ 2 has no partner
    sectors = PairSectors(labels)
    grouped_labels = labels[sectors.order]
    for shift in (0, 2, 1):
        matrix = rng.normal(size=(8, 8)) * ((grouped_labels[:, None] ^ grouped_labels[None, :]) == shift)
        blocks = _cpu_triples._sector_blocks(sectors, matrix)
        assert blocks is not None
        rhs = rng.normal(size=(8, 5))
        out = np.full((8, 5), np.nan)
        np.testing.assert_allclose(_cpu_triples._product(blocks, rhs, out), matrix @ rhs, atol=1e-13)
    mixed = rng.normal(size=(8, 8))                  # couples sectors with several shifts
    assert _cpu_triples._sector_blocks(sectors, mixed) is None


def test_lower_ranks_on_the_executor_match_numpy():
    """CCSD/CCSDT lower ranks and triples sources on the core's GEMM executor
    (`CompiledContract`) agree with NumPy and do not depend on the thread count."""
    from dataclasses import astuple

    from itercc import _restricted_equations as eq

    if _cpu_kernels.library() is None:
        pytest.skip("compiled CPU kernels unavailable")
    solver = RestrictedCC(*labelled_model(4, 5, LABELS))
    t1, t2, _ = trial_amplitudes(solver, 0.05, False)
    f, g = solver._dressed(t1)
    o, v = solver.o, solver.v

    def evaluate(contract):
        r1, r2, im, spin_g, c2 = eq.lower_residuals(f, g, t2, o, v, contract)
        return (r1, r2, *astuple(im), spin_g, c2, *eq.triples_sources(f, g, t2, c2, o, v, contract))

    expected = evaluate(eq.contract)
    serial, threaded = (evaluate(_cpu_triples.CompiledContract(threads)) for threads in (1, 3))
    for actual, wanted in zip(serial, expected):
        assert actual.flags.c_contiguous
        np.testing.assert_allclose(actual, wanted, rtol=0, atol=1e-14)
    for a, b in zip(serial, threaded):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("alpha", [1.0, -1.0, 0.5, -0.5])
def test_product_helpers_keep_the_numpy_arithmetic(alpha):
    """Without `contract.into` the helpers evaluate the expressions they replace."""
    from itercc import _restricted_equations as eq

    rng = np.random.default_rng(2)
    a, b, base = rng.standard_normal((3, 4)), rng.standard_normal((4, 5)), rng.standard_normal((3, 5))
    value = eq.contract("ik,kj->ij", a, b)
    scaled = value if abs(alpha) == 1 else abs(alpha) * value
    wanted = base + scaled if alpha > 0 else base - scaled
    np.testing.assert_array_equal(eq.plus_product(eq.contract, base, "ik,kj->ij", a, b, alpha), wanted)
    out = base.copy()
    assert eq.add_product(eq.contract, out, "ik,kj->ij", a, b, alpha) is out
    np.testing.assert_array_equal(out, wanted)
    x = rng.standard_normal((2, 3, 4, 4))
    np.testing.assert_array_equal(eq.spin_pair(eq.contract, x), 2 * x - x.swapaxes(2, 3))


def test_compiled_copy_and_spin_pair_are_exact():
    """Whole-array copies (of strided views too) and 2 x - x.swapaxes(2, 3) on all
    threads give NumPy's values bit for bit."""
    from itercc import _cpu_quadruples
    from itercc import _restricted_equations as eq

    if _cpu_kernels.library() is None:
        pytest.skip("compiled CPU kernels unavailable")
    rng = np.random.default_rng(4)
    x = rng.standard_normal((4, 4, 64, 64))           # above CompiledContract.LARGE
    contract = _cpu_triples.CompiledContract(3)
    np.testing.assert_array_equal(eq.spin_pair(contract, x), 2 * x - x.swapaxes(2, 3))
    view = x.transpose(0, 2, 1, 3)
    copied = _cpu_quadruples.copy(view)
    assert copied.flags.c_contiguous and not np.shares_memory(copied, x)
    np.testing.assert_array_equal(copied, view)


def test_host_memory_limit(monkeypatch):
    """The host limit is the least of physical memory, cgroup, Slurm and NUMA
    limits; SLURM_MEM_PER_NODE=0 (the whole node) imposes none."""
    physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    monkeypatch.delenv("SLURM_MEM_PER_CPU", raising=False)
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "0")
    unlimited = _cpu_triples.host_memory_limit()
    assert 0 < unlimited <= physical
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "1024")
    assert _cpu_triples.host_memory_limit() == min(unlimited, 2**30)
    monkeypatch.delenv("SLURM_MEM_PER_NODE")
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "1")
    assert _cpu_triples.host_memory_limit() == min(unlimited, 2**20 * _cpu_triples._PROCESS_CPUS)
    assert _cpu_triples._id_list("0-2,5,7-8") == {0, 1, 2, 5, 7, 8}
