"""CCSDTQ checks (NumPy and CUDA backends) against PySCF RCCSDTQ and four-electron FCI."""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from itercc import RestrictedCCSDTQ
from test_restricted import make_model, trial_amplitudes


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


# The NumPy backend always runs; the CUDA one with ITERCC_TEST_CUDA=1.
CUDA = pytest.mark.skipif(os.environ.get("ITERCC_TEST_CUDA") != "1",
                          reason="set ITERCC_TEST_CUDA=1 to run CUDA checks")
BACKENDS = ["cpu", pytest.param("cuda", marks=CUDA)]


@pytest.fixture(params=BACKENDS)
def backend(request):
    if request.param == "cuda":
        request.getfixturevalue("cuda")
    return request.param


def host(array):
    """A NumPy copy or view of a NumPy or CuPy array."""
    return np.asarray(array) if isinstance(array, np.ndarray) else array.get()


def array_module(backend):
    if backend == "cpu":
        return np
    import cupy

    return cupy


def pyscf_reference(solver, block):
    rccsdtq = pytest.importorskip("pyscf.cc.rccsdtq")
    rccsdt = rccsdtq.rccsdt
    no, nv = solver.nocc, solver.nvir
    ref = solver._reference
    cc = SimpleNamespace(nocc=no, nmo=no+nv, cc_order=4, blksize=block,
                         einsum_backend="numpy", level_shift=0.0,
                         stdout=sys.stdout, verbose=0)
    cc.tri2block_map, cc.tri2block_mask, cc.tri2block_tp = rccsdt.setup_tri2block_rhf(cc)
    eris = rccsdt._PhysicistsERIs()
    eris.nocc = no
    eris.fock = ref.fock
    eris.pppp = np.ascontiguousarray(ref.g)
    eris.mo_energy = ref.fock.diagonal().copy()
    return rccsdtq, cc, eris


@pytest.mark.parametrize("no,nv,block", [(2, 3, 1), (3, 4, 2)])
def test_raw_residuals_match_pyscf(backend, no, nv, block):
    solver = RestrictedCCSDTQ(*make_model(no, nv), block_size=block, backend=backend)
    rccsdtq, cc, eris = pyscf_reference(solver, block)
    rccsdt = rccsdtq.rccsdt
    ref = solver._reference
    t1, t2, packed = trial_amplitudes(ref, 0.017, True)
    t3 = solver.expand_triples(packed)
    rccsdt.t3_spin_summation_inplace_(t3, no**3, nv, "P3_full", -1 / 6, 1)
    t4 = np.random.default_rng(430).normal(scale=0.009, size=solver.layout.dense_shape)
    rccsdt.symmetrize_tamps_tri_(t4, no)
    rccsdtq.t4_project_1_minus_p4_p31_inplace_(t4, t4.shape[0], nv)
    rccsdt.purify_tamps_tri_(t4, no)
    initial = (t1, t2, t3, t4)

    cpu = [t.copy() for t in initial]
    rccsdtq.update_amps_rccsdtq_tri_(cc, cpu, eris)
    gaps = eris.mo_energy[:no, None] - eris.mo_energy[None, no:]
    d2 = gaps[:, None, :, None] + gaps[None, :, None, :]
    d3 = d2[:, :, None, :, :, None] + gaps[None, None, :, None, None, :]
    occ = np.asarray(solver.layout.indices).T
    gi, gj, gk, gl = (gaps[ix] for ix in occ)
    d4 = (gi[:, :, None, None, None] + gj[:, None, :, None, None]
          + gk[:, None, None, :, None] + gl[:, None, None, None, :])
    for rank, (gpu, after, before, denom) in enumerate(zip(
            solver.residuals(initial), cpu, initial, (gaps, d2, d3, d4)), 1):
        expected = (after - before) * denom
        np.testing.assert_allclose(host(gpu).reshape(expected.shape), expected,
                                   atol=5e-13, rtol=5e-12, err_msg=f"rank {rank}")


FCI_CASES = [(False, False), (True, False), (False, True), ("host", False), ("host", True)]


@pytest.mark.parametrize("backend,diis_quadruples,host_r4",
                         [("cpu", d, h) for d, h in FCI_CASES if not h]
                         + [pytest.param("cuda", d, h, marks=CUDA) for d, h in FCI_CASES])
def test_four_electron_energy_matches_fci(backend, diis_quadruples, host_r4):
    fci = pytest.importorskip("pyscf.fci.direct_spin1")
    h, eri, no, constant = make_model(2, 3)
    solver = RestrictedCCSDTQ(h, eri, no, constant, block_size=2, host_r4=host_r4, backend=backend)
    result = solver.solve(max_iter=100, damping=0.2,
                          energy_tol=1e-12, residual_tol=1e-10,
                          diis_size=3 if diis_quadruples is True else 6,
                          diis_quadruples=diis_quadruples)
    exact, _ = fci.kernel(h, eri, h.shape[0], (2, 2), ecore=constant, tol=1e-13)
    assert result.converged
    assert abs(result.energy - exact) < 5e-11
    assert max(result.residual_max) < 1e-10
    assert float(np.abs(host(result.amplitudes[3])).max()) > 1e-6


def test_cpu_backend_options():
    model = make_model(2, 3)
    with pytest.raises(ValueError, match="backend"):
        RestrictedCCSDTQ(*model, backend="opencl")
    with pytest.raises(ValueError, match="host_r4"):
        RestrictedCCSDTQ(*model, backend="cpu", host_r4=True)
    solver = RestrictedCCSDTQ(*model, backend="cpu")
    with pytest.raises(ValueError, match="host_r4"):
        solver.host_r4 = True
    result = solver.solve(max_iter=2, damping=0.2)
    assert all(isinstance(a, np.ndarray) for a in result.amplitudes)


def test_cpu_backend_runs_without_cupy():
    """The NumPy backend never imports CuPy: rerun a symmetric solve with CuPy blocked."""
    script = """
import sys
sys.modules["cupy"] = None
import numpy as np
from itercc import RestrictedCCSDTQ
from test_quadruples import LABELS, symmetric_model
solver = RestrictedCCSDTQ(*symmetric_model(), block_size=2, orbital_irreps=LABELS, backend="cpu")
for diis in (False, True, "host"):
    result = solver.solve(max_iter=3, damping=0.2, diis_size=3, diis_quadruples=diis)
    print(repr(result.energy))
state = []
solver.solve(max_iter=2, damping=0.2, diis_size=3, checkpoint=state.append)
print(repr(solver.solve(resume=state[-1], max_iter=1, damping=0.2, diis_size=3).energy))
assert sys.modules["cupy"] is None
"""
    tests = Path(__file__).resolve().parent
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(tests.parent), str(tests), os.environ.get("PYTHONPATH", "")]))
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env,
                         cwd=tests)
    assert run.returncode == 0, run.stderr
    energies = [float(line) for line in run.stdout.split()]
    solver = RestrictedCCSDTQ(*symmetric_model(), block_size=2, orbital_irreps=LABELS, backend="cpu")
    expected = [solver.solve(max_iter=3, damping=0.2, diis_size=3, diis_quadruples=diis).energy
                for diis in (False, True, "host")]
    expected.append(solver.solve(max_iter=2, damping=0.2, diis_size=3).energy)
    np.testing.assert_allclose(energies, expected, rtol=0, atol=1e-11)


def test_host_quadruples_residual_and_updates(cuda, monkeypatch, tmp_path):
    from itercc._quadruples_cuda import QuadruplesLayout

    # Exercise different occupied offsets and padded edge tiles.
    monkeypatch.setattr(QuadruplesLayout, "transfer_slots", 2)
    model = make_model(3, 4)
    device = RestrictedCCSDTQ(*model, block_size=2)
    host = RestrictedCCSDTQ(*model, block_size=2, host_r4=True)
    settings = dict(max_iter=4, damping=0.2, diis_size=0)
    result_device = device.solve(**settings)
    result_host = host.solve(**settings)
    for reference, actual in zip(result_device.amplitudes, result_host.amplitudes):
        np.testing.assert_allclose(cuda.asnumpy(actual), cuda.asnumpy(reference),
                                   atol=2e-13, rtol=2e-12)
    a = result_device.amplitudes
    device_residual = device.residuals(a)[3]
    host_residual = host.residuals(a)[3]
    assert isinstance(host_residual, np.ndarray)
    np.testing.assert_allclose(host_residual, cuda.asnumpy(device_residual),
                               atol=2e-13, rtol=2e-12)
    np.save(tmp_path / "t4.npy", cuda.asnumpy(a[3]).reshape(device.layout.dense_shape))
    initial = (*a[:3], np.load(tmp_path / "t4.npy", mmap_mode="r"))
    result_host = host.solve(initial=initial, max_iter=2, damping=0.2, diis_size=0)
    result_device = device.solve(initial=a, max_iter=2, damping=0.2, diis_size=0)
    for reference, actual in zip(result_device.amplitudes, result_host.amplitudes):
        np.testing.assert_allclose(cuda.asnumpy(actual), cuda.asnumpy(reference),
                                   atol=2e-13, rtol=2e-12)


def test_six_electron_energy_matches_pyscf(backend):
    h, eri, no, constant = make_model(3, 3)
    solver = RestrictedCCSDTQ(h, 0.35 * eri, no, constant, block_size=2, backend=backend)
    rccsdtq, cc, eris = pyscf_reference(solver, 2)
    gaps = solver._eia(0.0)
    d2, _ = solver._denominators(gaps)
    cpu = [host(t).copy() for t in solver._initial(None, gaps, d2)]
    cpu[3] = cpu[3].reshape(solver.layout.dense_shape)
    for _ in range(100):
        norms = rccsdtq.update_amps_rccsdtq_tri_(cc, cpu, eris)
        if max(norms) < 1e-12:
            break
    else:
        pytest.fail("PySCF reference did not converge")
    result = solver.solve(energy_tol=1e-12, residual_tol=1e-11)
    assert result.converged
    assert max(result.residual_max) < 1e-11
    assert abs(result.correlation_energy - solver._reference._energy(*cpu[:2])) < 5e-12


# Occupied labels fixed, virtual labels deliberately unsorted (C2v-like XOR group).
LABELS = np.array([0, 2, 3, 1, 0, 3, 2, 0, 1])


def symmetric_model(labels=LABELS, no=3, seed=137):
    h, eri, no, constant = make_model(no, len(labels) - no, seed)
    pairs = labels[:, None] ^ labels
    h = np.where(pairs == 0, h, 0.0)
    eri = np.where(pairs[:, :, None, None] == pairs[None, None], eri, 0.0)
    return h, 0.6 * eri, no, constant


def warm_amplitudes(model, evaluations=3, backend="cuda"):
    solver = RestrictedCCSDTQ(*model, block_size=2, backend=backend)
    result = solver.solve(max_iter=evaluations, damping=0.2, diis_size=0)
    return solver, result.amplitudes


def test_symmetric_layout_round_trip(backend):
    model = symmetric_model()
    dense_solver, amplitudes = warm_amplitudes(model, backend=backend)
    solver = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=LABELS, backend=backend)
    assert solver.layout.symmetric and solver.layout.size < dense_solver.layout.size / 2
    dense = host(amplitudes[3]).reshape(dense_solver.layout.dense_shape)
    packed, leakage = solver.layout.from_dense(dense, order=solver._virtual_order)
    assert leakage < 1e-15
    np.testing.assert_array_equal(solver.quadruples_to_dense(packed), dense)
    noisy = dense + 1e-6
    with pytest.raises(ValueError, match="leakage"):
        solver._quadruples(noisy, False)


@pytest.mark.parametrize("triples", [False, True])
@pytest.mark.parametrize("irreps", [True, False])
def test_box_gather_matches_elementwise_gather(cuda, triples, irreps):
    """The box-by-box tile gather equals the per-entry kernel for every tile shape used."""
    from itercc import _quadruples_cuda as qc

    cp = cuda
    nocc, nvir = 5, 7
    occupied = np.array([0, 3, 1, 2, 0]) if irreps else None
    virtual = np.array([0, 0, 1, 1, 2, 3, 3]) if irreps else None
    layout = qc.QuadruplesLayout(nocc, nvir, occupied, virtual)
    packed = cp.asarray(np.random.default_rng(3).normal(size=layout.size))
    cases = [((0, nocc, 1, 2, 2, 3, 4, 5), (nocc, 1, 1, 1)),
             ((0, nocc, 0, nocc, 1, 2, 3, 4), (nocc, nocc, 1, 1)),
             ((0, 2, 2, 4, 2, 4, 4, 5), (2, 2, 2, 2)),
             ((0, nocc, 1, 3, 2, 4, 4, 5), (nocc, 2, 2, 2))]
    for bounds, shape in cases:
        tile = cp.full(shape + (nvir,) * 4, np.nan)
        layout.gather(packed, tile, bounds, triples=triples)
        plan = layout._plan(bounds, shape, False)
        expected = cp.empty_like(tile)
        qc._GATHER(packed, plan, *layout._tables, nvir, layout.nv4, triples, expected.reshape(-1))
        np.testing.assert_array_equal(cp.asnumpy(tile), cp.asnumpy(expected), err_msg=str(bounds))
        if triples:
            continue
        # Spin-summed gathers equal a plain gather followed by the spin summation.
        for pattern in ("P4_201", "P4_442_r2"):
            summed = cp.full_like(tile, np.nan)
            layout.gather(packed, summed, bounds, spin=pattern)
            reference = cp.empty_like(tile)
            qc.t4_spin_summation(expected, reference, nvir, pattern)
            np.testing.assert_allclose(cp.asnumpy(summed), cp.asnumpy(reference), rtol=1e-14, atol=1e-14,
                                       err_msg=f"{pattern} {bounds}")


@CUDA
@pytest.mark.parametrize("irreps", [True, False])
def test_numpy_layout_matches_cuda_layout(cuda, irreps):
    """Every packed-layout primitive of the NumPy backend equals the CUDA kernel."""
    from itercc import _quadruples_cpu as qn
    from itercc import _quadruples_cuda as qc

    cp = cuda
    nocc, nvir = 5, 7
    occupied = np.array([0, 3, 1, 2, 0]) if irreps else None
    virtual = np.array([0, 0, 1, 1, 2, 3, 3]) if irreps else None
    gpu = qc.QuadruplesLayout(nocc, nvir, occupied, virtual)
    cpu = qn.NumpyQuadruplesLayout(nocc, nvir, occupied, virtual)
    assert cpu.size == gpu.size and np.array_equal(cpu.offsets, gpu.offsets)
    rng = np.random.default_rng(5)
    packed = rng.normal(size=gpu.size)
    cases = [((0, nocc, 1, 2, 2, 3, 4, 5), (nocc, 1, 1, 1)),
             ((0, nocc, 0, nocc, 1, 2, 3, 4), (nocc, nocc, 1, 1)),
             ((0, 2, 2, 4, 2, 4, 4, 5), (2, 2, 2, 2)),
             ((0, nocc, 1, 3, 2, 4, 4, 5), (nocc, 2, 2, 2))]
    for bounds, shape in cases:
        for kind in (dict(), dict(triples=True), dict(spin="P4_201"), dict(spin="P4_442_r2")):
            expected = cp.full(shape + (nvir,) * 4, np.nan)
            gpu.gather(cp.asarray(packed), expected, bounds, **kind)
            actual = np.full(expected.shape, np.nan)
            cpu.gather(packed, actual, bounds, **kind)
            np.testing.assert_allclose(actual, expected.get(), rtol=1e-14, atol=1e-14,
                                       err_msg=f"{kind} {bounds}")
        tile = rng.normal(size=shape + (nvir,) * 4)
        expected, actual = cp.asarray(packed), packed.copy()
        gpu.scatter(expected, cp.asarray(tile), bounds, alpha=0.5, beta=-2.0)
        cpu.scatter(actual, tile, bounds, alpha=0.5, beta=-2.0)
        np.testing.assert_allclose(actual, expected.get(), rtol=1e-14, atol=1e-14, err_msg=str(bounds))
    rows = rng.normal(size=(nocc,) + (nvir,) * 4)
    for lists in (gpu.pair_entries(0, 2, 1), gpu.triple_entries((1, 1, 3))):
        expected = cp.asarray(packed)
        gpu.add_rows(expected, cp.asarray(rows), lists)
        actual = packed.copy()
        cpu.add_rows(actual, rows, [entries.get() for entries in lists])
        np.testing.assert_allclose(actual, expected.get(), rtol=1e-14, atol=1e-14)
    eia = -1.0 - rng.random((nocc, nvir))
    expected, actual = cp.asarray(packed), packed.copy()
    gpu.project(expected)
    cpu.project(actual)
    np.testing.assert_allclose(actual, expected.get(), rtol=1e-13, atol=1e-14)
    gpu.divide(expected, cp.asarray(eia))
    cpu.divide(actual, eia)
    np.testing.assert_allclose(actual, expected.get(), rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(cpu.to_dense(packed), gpu.to_dense(cp.asarray(packed)))


@pytest.mark.parametrize("backend,host_r4", [("cpu", False), pytest.param("cuda", False, marks=CUDA),
                                             pytest.param("cuda", True, marks=CUDA)])
def test_symmetric_residuals_match_dense(backend, host_r4):
    model = symmetric_model()
    dense_solver, amplitudes = warm_amplitudes(model, backend=backend)
    solver = RestrictedCCSDTQ(*model, block_size=2, host_r4=host_r4, orbital_irreps=LABELS,
                              backend=backend)
    dense_t4 = host(amplitudes[3]).reshape(dense_solver.layout.dense_shape)
    expected = dense_solver.residuals(amplitudes)
    actual = solver.residuals((*amplitudes[:3], dense_t4))
    for rank in range(3):
        np.testing.assert_allclose(host(actual[rank]), host(expected[rank]),
                                   atol=1e-14, rtol=1e-12, err_msg=f"rank {rank + 1}")
    np.testing.assert_allclose(solver.quadruples_to_dense(actual[3]),
                               host(expected[3]).reshape(dense_t4.shape),
                               atol=1e-14, rtol=1e-12)


@pytest.mark.parametrize("fused,blocked", [("1", "1"), ("1", "0"), ("0", "0")])
def test_cpu_tile_program_matches_tile_loop(monkeypatch, fused, blocked):
    """One-slot tiles (the CPU default) run the R4 tile loops in the compiled tile
    program, fused into partition GEMMs or one pass per term, on pair-blocked T3
    (the default) or the dense T3: same residuals as the Python tile loop, bit
    for bit the same for any thread count or teams."""
    from itercc import _cpu_quadruples

    if not _cpu_quadruples.available():
        pytest.skip("compiled CPU core unavailable")
    monkeypatch.setenv("ITERCC_TILE_FUSED", fused)
    monkeypatch.setenv("ITERCC_TILE_BLOCKED_T3", blocked)
    order = ", pair order" if blocked == "1" else ""
    model = symmetric_model()
    dense_solver, amplitudes = warm_amplitudes(model, backend="cpu")
    dense_t4 = host(amplitudes[3]).reshape(dense_solver.layout.dense_shape)
    loop = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=LABELS, backend="cpu")
    assert loop._tile_program is None
    expected = loop.residuals((*amplitudes[:3], dense_t4))
    results = []
    for threads, team in (("1", "1"), ("4", "1"), ("4", "2"), ("3", "3")):
        monkeypatch.setenv("ITERCC_CPU_THREADS", threads)
        monkeypatch.setenv("ITERCC_TILE_TEAM", team)    # slot-parallel, then teams
        solver = RestrictedCCSDTQ(*model, orbital_irreps=LABELS, backend="cpu")
        assert solver.blksize == 1 and solver._tile_program is not None
        actual = solver.residuals((*amplitudes[:3], dense_t4))
        recorded = solver._tile_program._recorded
        assert sorted(recorded) == ["t3 terms" + order, "t4 terms" + order]
        assert not solver._tile_program._failed
        assert all(program.fused for program in recorded.values())
        for rank in range(4):
            np.testing.assert_allclose(actual[rank], expected[rank], atol=1e-14, rtol=1e-12,
                                       err_msg=f"rank {rank + 1}")
        results.append(actual)
    for other in results[1:]:
        for first, second in zip(results[0], other):
            np.testing.assert_array_equal(first, second)


def test_lower_rank_pass_matches_reference():
    """T4 into lower ranks on packed rows gives the residuals of the generic tile loop."""
    from itercc import _cpu_quadruples

    if not _cpu_quadruples.available():
        pytest.skip("compiled CPU core unavailable")
    model = symmetric_model()
    dense_solver, amplitudes = warm_amplitudes(model, backend="cpu")
    dense_t4 = host(amplitudes[3]).reshape(dense_solver.layout.dense_shape)
    expected = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=LABELS,
                                backend="cpu").residuals((*amplitudes[:3], dense_t4))
    solver = RestrictedCCSDTQ(*model, orbital_irreps=LABELS, backend="cpu")
    actual = solver.residuals((*amplitudes[:3], dense_t4))
    for rank in range(4):
        np.testing.assert_allclose(actual[rank], expected[rank], atol=1e-14, rtol=1e-12,
                                   err_msg=f"rank {rank + 1}")


def test_concurrent_lower_rank_pairs(monkeypatch):
    """T4 into lower ranks (pairs of disjoint indices) and the ring pass (triples
    sharing at most one index) on concurrent thread teams: the same bits for
    any team count, and the residuals of the loop."""
    from collections import Counter
    import itertools

    from itercc import _cpu_packed, _cpu_quadruples

    if not _cpu_quadruples.available():
        pytest.skip("compiled CPU core unavailable")
    for n in (1, 2, 3, 6, 13):
        rounds = _cpu_packed._pair_rounds(n)
        pairs = [pair for round_ in rounds for pair in round_]
        assert sorted(pairs) == [(p, q) for p in range(n) for q in range(p, n)]
        for round_ in rounds:
            indices = [x for p, q in round_ for x in {p, q}]
            assert len(indices) == len(set(indices))
        triples = _cpu_packed._triple_rounds(n)
        assert sorted(t for round_ in triples for t in round_) == [
            (p, q, r) for p in range(n) for q in range(p, n) for r in range(q, n)]
        for round_ in triples:          # at most one shared index: disjoint target slots
            for a, b in itertools.combinations(round_, 2):
                assert sum((Counter(a) & Counter(b)).values()) <= 1
    model = symmetric_model()
    dense_solver, amplitudes = warm_amplitudes(model, backend="cpu")
    dense_t4 = host(amplitudes[3]).reshape(dense_solver.layout.dense_shape)
    expected = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=LABELS,
                                backend="cpu").residuals((*amplitudes[:3], dense_t4))
    monkeypatch.setenv("ITERCC_CPU_THREADS", "6")
    results = []
    for teams in ("1", "3"):
        monkeypatch.setenv("ITERCC_LOWER_TEAMS", teams)
        solver = RestrictedCCSDTQ(*model, orbital_irreps=LABELS, backend="cpu")
        results.append(solver.residuals((*amplitudes[:3], dense_t4)))
    for a, b in zip(*results):
        np.testing.assert_array_equal(a, b)
    for rank, (actual, wanted) in enumerate(zip(results[1], expected), 1):
        np.testing.assert_allclose(actual, wanted, atol=1e-14, rtol=1e-12, err_msg=f"rank {rank}")


def test_single_precision_operands(monkeypatch):
    """precision="single": the R4 passes read float32 T3 copies and float32
    spin-summed T4 rows, with products and sums in double. The residual moves
    by about float32 rounding, the converged energy by far less than 1e-10."""
    from itercc import _cpu_quadruples

    if not _cpu_quadruples.available():
        pytest.skip("compiled CPU core unavailable")
    with pytest.raises(ValueError, match="CPU backend"):
        RestrictedCCSDTQ(*symmetric_model(), orbital_irreps=LABELS, precision="single")
    model = symmetric_model()
    dense_solver, amplitudes = warm_amplitudes(model, backend="cpu")
    dense_t4 = host(amplitudes[3]).reshape(dense_solver.layout.dense_shape)
    runs = {}
    for precision in ("double", "single"):
        solver = RestrictedCCSDTQ(*model, orbital_irreps=LABELS, backend="cpu", precision=precision)
        runs[precision] = solver.residuals((*amplitudes[:3], dense_t4))
    for rank, (a, b) in enumerate(zip(runs["double"], runs["single"]), 1):
        scale = max(float(np.max(np.abs(a))), 1e-300)
        difference = float(np.max(np.abs(a - b)))
        assert difference <= 1e-6 * scale, (rank, difference, scale)
    assert any(not np.array_equal(a, b) for a, b in zip(runs["double"], runs["single"]))
    settings = dict(max_iter=80, damping=0.1, energy_tol=1e-12, residual_tol=1e-9, diis_size=6,
                    diis_quadruples="host")
    energies = [RestrictedCCSDTQ(*model, orbital_irreps=LABELS, backend="cpu",
                                 precision=precision).solve(**settings) for precision in ("double", "single")]
    assert all(result.converged for result in energies)
    assert abs(energies[0].energy - energies[1].energy) < 1e-10


@pytest.mark.parametrize("precision", ["double", "single"])
def test_pair_blocked_t3_tile_programs(monkeypatch, precision):
    """Pair-blocked T3 (the default; ITERCC_TILE_BLOCKED_T3=0 turns it off):
    the T3 terms of the tile programs take their
    T3 pairs in blocked order (T3 is pair symmetric to rounding) and read
    pair-blocked copies of T3 and its spin sum. The copies change only
    addresses (the same values in the same order into the same GEMMs): bit for
    bit the pair-ordered terms on the dense arrays."""
    from itercc import _cpu_quadruples
    from itercc._cpu_packed import PairBlocks
    from itercc._cpu_tile_program import TileProgram

    if not _cpu_quadruples.available():
        pytest.skip("compiled CPU core unavailable")
    model = symmetric_model()
    dense_solver, amplitudes = warm_amplitudes(model, backend="cpu")
    dense_t4 = host(amplitudes[3]).reshape(dense_solver.layout.dense_shape)
    runs, programs = {}, {}
    for mode in ("0", "1", "ordered"):
        with monkeypatch.context() as patch:
            patch.setenv("ITERCC_TILE_BLOCKED_T3", "0" if mode == "0" else "1")
            if mode == "ordered":        # pair order, dense arrays
                patch.setattr(TileProgram, "blocked", lambda self, arrays, single=False: None)
            solver = RestrictedCCSDTQ(*model, orbital_irreps=LABELS, backend="cpu", precision=precision)
            runs[mode] = solver.residuals((*amplitudes[:3], dense_t4))
            programs[mode] = solver._tile_program
    for a, b, c in zip(runs["0"], runs["1"], runs["ordered"]):
        np.testing.assert_array_equal(b, c)
        assert np.max(np.abs(a - b)) <= 1e-12 * max(np.max(np.abs(a)), 1e-300)
    program = programs["1"]
    assert program is not None and len(program._recorded) == 2
    remapped = 0
    for recorded in program._recorded.values():
        swapped = next(key for key in recorded.fused if key)
        assert recorded.fused[swapped]
        names = [name for name, *_ in recorded.signature]
        remapped += len(program._blocked_sides(recorded, names, swapped))
    assert remapped == 66                  # every T3 term but the six with the second spin sum

    # The pair permutation itself: block by block, C order inside.
    labels = np.array([0, 0, 1, 2, 2, 2, 3])
    pairs = PairBlocks(labels, 4)
    x = np.random.default_rng(3).standard_normal((2, 3, 7, 7))
    y = pairs.permute(x).reshape(6, -1)
    assert np.array_equal(y[:, pairs.position], x.reshape(6, -1))
    first = pairs.base[0, 2]
    np.testing.assert_array_equal(y[4, first:first + 2 * 3], x[1, 1, 0:2, 3:6].reshape(-1))
    table = np.array([5 * 49 + 3 * 7 + 4, 2 * 343, 6 * 7 + 1])
    np.testing.assert_array_equal(pairs.remap(table), [5 * 49 + pairs.position[25], 2 * 343,
                                                       pairs.position[43]])


@pytest.mark.parametrize("packed_spin,direct", [(False, False), (True, False), (True, True)])
def test_cpu_triplet_eom_ccsdtq_solve_matches_fci(packed_spin, direct):
    """Spin-flip EOM-CCSDTQ (`TripletEOMCCSDTQ.solve`) on the CPU: four
    electrons, so the lowest triplet root is the FCI singlet-triplet gap."""
    from itercc import TripletEOMCCSDTQ
    if direct:
        from itercc import _cpu_quadruples
        if not _cpu_quadruples.available():
            pytest.skip("compiled CPU core unavailable")

    fci = pytest.importorskip("pyscf.fci")
    h, eri, no, constant = make_model(2, 3 if packed_spin else 2)
    eri = eri * 0.45
    solver = RestrictedCCSDTQ(h, eri, no, constant, backend="cpu")
    ground = solver.solve(max_iter=100, energy_tol=1e-13, residual_tol=1e-11, damping=0.2,
                          diis_quadruples="host", diis_size=3)
    assert ground.converged
    n = len(h)
    ndets = fci.cistring.num_strings(n, no)
    address, matrix = fci.direct_spin1.pspace(h, eri, n, (no, no), np=ndets**2)
    values, vectors = np.linalg.eigh(matrix)
    spins = {}
    for value, vector in zip(values, vectors.T):
        full = np.zeros(ndets**2)
        full[address] = vector
        s2 = fci.spin_op.spin_square(full.reshape(ndets, ndets), n, (no, no))[0]
        spins.setdefault(int(round(s2)), []).append(value + constant)
    assert abs(spins[0][0] - ground.energy) < 1e-11
    engine = TripletEOMCCSDTQ(solver, ground.amplitudes, backend="cpu",
                             occupied_tiles=packed_spin, packed_spin=packed_spin,
                             packed_contractions=direct)
    result = engine.solve(1, max_space=16, max_iter=60, residual_tol=1e-8,
                          memory_budget_gib=0.02 if packed_spin else None)
    assert result.converged
    assert abs(result.excitation_energies[0] - (spins[2][0] - spins[0][0])) < 2e-8


def test_cpu_triplet_occupied_tiles_use_the_executor(monkeypatch):
    """Triplet EOM-CCSDTQ with occupied tiles on the CPU: the symmetry-blocked
    tile products run on the compiled executor and match the dense action."""
    from itercc import TripletEOMCCSDTQ, _cpu_quadruples

    if not _cpu_quadruples.available():
        pytest.skip("compiled CPU core unavailable")
    labels = LABELS[:7]
    model = symmetric_model(labels, no=2)
    solver = RestrictedCCSDTQ(*model, orbital_irreps=labels, backend="cpu")
    ground = solver.solve(max_iter=80, energy_tol=1e-12, residual_tol=1e-10, damping=0.1,
                          diis_quadruples="host", diis_size=4)
    assert ground.converged
    actions = []
    compiled_specs = set()
    for tiles in (False, True):
        engine = TripletEOMCCSDTQ(solver, ground.amplitudes, labels, 1, backend="cpu",
                                  occupied_tiles=tiles)
        assert (engine._symmetry_engine is not None) == tiles
        if tiles:
            # Rank four's executor, and that of ranks two and three (their
            # occupied letters inside the GEMMs).
            for executor in {id(x): x for x in (engine._symmetry_engine, engine._lower_engine)}.values():
                def checked(spec, a, b, out, executor=executor, original=executor.einsum, **kwargs):
                    plan, _ = executor.plan_for(spec, a, b, out, kwargs["labels"], kwargs["irreps"])
                    if plan:
                        compiled_specs.add(spec)
                    return original(spec, a, b, out, **kwargs)

                monkeypatch.setattr(executor, "einsum", checked)
        vector = engine.space.project(np.random.default_rng(5).normal(size=engine.space.size))
        actions.append(engine.sigma(vector))
    assert "klcd,abcdijkl->abij" in compiled_specs
    assert "alde,bcdeijkl->abcijk" in compiled_specs
    np.testing.assert_allclose(actions[1], actions[0], rtol=0,
                               atol=1e-11 * max(1.0, float(np.max(np.abs(actions[0])))))


@pytest.mark.parametrize("direct", [False, True])
def test_packed_triplet_probe_preserves_lower_rank_feedback(monkeypatch, direct):
    from itercc import TripletEOMCCSDTQ, _spin_quadruples_tiles
    if direct:
        from itercc import _cpu_quadruples
        if not _cpu_quadruples.available():
            pytest.skip("compiled CPU core unavailable")

    # Force slicing of free occupied indices even for a small reference.
    monkeypatch.setattr(_spin_quadruples_tiles, "_PACKED_TILE_BYTES", 8 * 3**4 * 3)
    labels = np.array([0, 1, 0, 1, 1])
    solver = RestrictedCCSDTQ(*symmetric_model(labels, no=2), orbital_irreps=labels, backend="cpu")
    ground = solver.solve(max_iter=80, energy_tol=1e-12, residual_tol=1e-10, damping=0.1,
                          diis_quadruples="host", diis_size=4)
    assert ground.converged
    reference = TripletEOMCCSDTQ(solver, ground.amplitudes, labels, 1)
    packed = TripletEOMCCSDTQ(solver, ground.amplitudes, labels, 1,
                             occupied_tiles=True, action_only=True, packed_contractions=direct,
                             intermediate_cache_gib=0.0001)
    production = TripletEOMCCSDTQ(solver, ground.amplitudes, labels, 1,
                                 occupied_tiles=True, packed_spin=True, packed_contractions=direct,
                                 intermediate_cache_gib=0.0001)
    tail = reference.space.slices[3]
    rows = {tuple(row): i for i, row in enumerate(reference.space.indices[3].T)}
    layout = packed._direction_t4
    order = [rows[(*particles, *holes)] for holes, (key, _) in layout.slots.items()
             for particles in layout.groups[key]]
    assert order
    vector = np.random.default_rng(72).normal(size=reference.space.size)
    probe_vector = np.concatenate((vector[:tail.start], vector[tail][order]))
    expected = reference.sigma(vector)
    without_r4 = vector.copy()
    without_r4[tail] = 0
    assert np.linalg.norm(expected[:tail.start] - reference.sigma(without_r4)[:tail.start]) > 1e-9
    np.testing.assert_allclose(packed.probe_lower_ranks(probe_vector), expected[:tail.start], atol=1e-11)
    np.testing.assert_allclose(packed.probe_rank_four(probe_vector, len(layout.slots)),
                               expected[tail][order], atol=1e-11)
    np.testing.assert_allclose(production.sigma(probe_vector),
                               np.r_[expected[:tail.start], expected[tail][order]], atol=1e-11)
    for active in (reference.space.slices[0], tail):
        sparse = np.zeros_like(vector)
        sparse[active] = vector[active]
        dense = reference.sigma(sparse)
        packed_sparse = np.r_[sparse[:tail.start], sparse[tail][order]]
        np.testing.assert_allclose(production.sigma(packed_sparse),
                                   np.r_[dense[:tail.start], dense[tail][order]], atol=1e-11)
        if active == reference.space.slices[0]:
            def unexpected_expansion(*args, **kwargs):
                pytest.fail("Singles-only doubles projection expanded ground-state T4")

            with monkeypatch.context() as patch:
                patch.setattr(production._ground_t4, "expand", unexpected_expansion)
                np.testing.assert_allclose(production._action(packed_sparse, ranks=(2,)),
                                           dense[reference.space.slices[1]], atol=1e-11)
    np.testing.assert_allclose(production.diagonal,
                               np.r_[reference.diagonal[:tail.start], reference.diagonal[tail][order]])


def test_packed_rank_four_spin_projector_matches_dense():
    from itercc.eom_spin_quadruples import SpinFlipSpace
    from itercc._spin_quadruples_storage import PackedSpinQuadruples

    no, nv, irrep = 4, 5, 1
    labels = np.array([0, 0, 1, 2, 0, 1, 2, 3, 3])
    layout = PackedSpinQuadruples(labels[:no], labels[no:], irrep, 1)
    dense = SpinFlipSpace(no, nv, labels, irrep, np)
    packed = SpinFlipSpace(no, nv, labels, irrep, np, packed=layout)
    tail = dense.slices[3]
    rows = {tuple(row): i for i, row in enumerate(dense.indices[3].T)}
    order = [rows[(*particles, *holes)] for holes, (key, _) in layout.slots.items()
             for particles in layout.groups[key]]
    take = np.r_[np.arange(tail.start), tail.start + np.asarray(order)]
    vector = np.random.default_rng(730).normal(size=dense.size)
    expected = dense.project(vector)
    assert np.linalg.norm(expected[tail] - vector[tail]) > 1e-3
    projected = packed.project(vector[take])
    np.testing.assert_allclose(projected, expected[take], rtol=0, atol=1e-13)
    np.testing.assert_allclose(packed.project(projected), projected, rtol=0, atol=1e-13)


@pytest.mark.parametrize("occupied_slab", [None, 1, 2])
def test_cpu_direct_spin_quadruples_six_electrons_match_dense(occupied_slab):
    from itercc import TripletEOMCCSDTQ, _cpu_quadruples
    if not _cpu_quadruples.available():
        pytest.skip("compiled CPU core unavailable")
    labels = np.array([0, 1, 2, 0, 1, 2, 3])
    solver, amplitudes = warm_amplitudes(symmetric_model(labels, no=3), evaluations=2, backend="cpu")
    dense = TripletEOMCCSDTQ(solver, amplitudes, labels, 2)
    direct = TripletEOMCCSDTQ(solver, amplitudes, labels, 2, occupied_tiles=True,
                             action_only=True, packed_contractions=True, occupied_slab=occupied_slab)
    vector = np.random.default_rng(715).normal(size=dense.space.size)
    tail = dense.space.slices[3]
    rows = {tuple(row): i for i, row in enumerate(dense.space.indices[3].T)}
    layout = direct._direction_t4
    order = [rows[(*particles, *holes)] for holes, (key, _) in layout.slots.items()
             for particles in layout.groups[key]]
    expected = dense._action(vector)
    measured = direct._action(np.r_[vector[:tail.start], vector[tail][order]])
    np.testing.assert_allclose(measured, np.r_[expected[:tail.start], expected[tail][order]],
                               rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("occupied_slab", [0, -1, 1.5, True])
def test_triplet_rejects_invalid_occupied_slab(occupied_slab):
    from types import SimpleNamespace
    from itercc import TripletEOMCCSDTQ
    with pytest.raises(ValueError, match="Occupied slab"):
        TripletEOMCCSDTQ(SimpleNamespace(nocc=1, nvir=1), (), occupied_slab=occupied_slab)


def test_cpu_finite_difference_eom_refuses_single_precision():
    from itercc import _cpu_quadruples
    from itercc.eom_ccsdtq import FiniteDifferenceEOM

    if not _cpu_quadruples.available():
        pytest.skip("compiled CPU core unavailable")
    h, eri, no, constant = make_model(2, 2)
    solver = RestrictedCCSDTQ(h, eri, no, constant, backend="cpu", precision="single")
    with pytest.raises(ValueError, match="precision='double'"):
        FiniteDifferenceEOM(solver, None, np.zeros(len(h), dtype=int), 0)   # refused before use


def test_cpu_one_sided_singlet_eom_takes_single_precision():
    """A one-sided action's sector entries are O(eps) term by term, so on a
    precision='single' solver (float32 copies of T3 and of the spin-summed T4
    rows) it stays within their rounding of the double action."""
    from itercc import _cpu_quadruples
    from itercc.eom_ccsdtq import FiniteDifferenceEOM
    from itercc.spatial_symmetry import sector_labels

    if not _cpu_quadruples.available():
        pytest.skip("compiled CPU core unavailable")
    model = symmetric_model()
    dense = RestrictedCCSDTQ(*model, backend="cpu")
    ground = dense.solve(max_iter=3, damping=0.2, diis_size=0).amplitudes
    images = []
    for precision in ("double", "single"):
        solver = RestrictedCCSDTQ(*model, backend="cpu", precision=precision,
                                  orbital_irreps=sector_labels(LABELS, 2))
        engine = FiniteDifferenceEOM(solver, solver.combine_states([(1.0, ground, None)]), LABELS, 2)
        assert engine.one_sided
        images.append(engine.sigma(np.random.default_rng(37).normal(size=engine.space.size)))
    difference = np.linalg.norm(images[1] - images[0]) / np.linalg.norm(images[0])
    assert 0 < difference < 1e-6


def test_cpu_finite_difference_eom_ccsdtq_matches_fci():
    """EOM-CCSDTQ on the compiled CPU solver: four electrons, so CCSDTQ is FCI
    and the lowest singlet excitation equals the FCI gap."""
    from itercc import _cpu_quadruples
    from itercc.eom_ccsdtq import FiniteDifferenceEOM

    fci = pytest.importorskip("pyscf.fci")
    if not _cpu_quadruples.available():
        pytest.skip("compiled CPU core unavailable")
    h, eri, no, constant = make_model(2, 2)
    eri = eri * 0.45
    solver = RestrictedCCSDTQ(h, eri, no, constant, backend="cpu")
    ground = solver.solve(max_iter=100, energy_tol=1e-13, residual_tol=1e-11, damping=0.2,
                          diis_quadruples="host", diis_size=3)
    assert ground.converged
    n = len(h)
    ndets = fci.cistring.num_strings(n, no)
    address, matrix = fci.direct_spin1.pspace(h, eri, n, (no, no), np=ndets**2)
    values, vectors = np.linalg.eigh(matrix)
    singlets = []
    for value, vector in zip(values, vectors.T):
        full = np.zeros(ndets**2)
        full[address] = vector
        if abs(fci.spin_op.spin_square(full.reshape(ndets, ndets), n, (no, no))[0]) < 1e-7:
            singlets.append(value + constant)
    assert abs(singlets[0] - ground.energy) < 1e-11
    engine = FiniteDifferenceEOM(solver, ground.amplitudes, np.zeros(n, dtype=int), 0)
    assert engine.xp is np
    first = engine.solve_one(max_space=12, max_iter=80, residual_tol=1e-8)
    assert first.converged
    assert abs(first.excitation_energies[0] - (singlets[1] - singlets[0])) < 2e-8


def test_sector_labels_keep_the_target_totally_symmetric():
    from itercc.spatial_symmetry import sector_labels

    c2v = np.arange(4)
    np.testing.assert_array_equal(sector_labels(c2v, 0), c2v)
    np.testing.assert_array_equal(sector_labels(c2v, 1), [0, 0, 1, 1])
    np.testing.assert_array_equal(sector_labels(c2v, 2), [0, 1, 0, 1])
    np.testing.assert_array_equal(sector_labels(c2v, 3), [0, 1, 1, 0])
    d2h = np.arange(8)
    for irrep in range(1, 8):
        kept = sector_labels(d2h, irrep)
        assert kept.max() == 3 and len(set(kept.tolist())) == 4
        assert all(kept[x ^ y] == kept[x] ^ kept[y] for x in range(8) for y in range(8))
        assert all(kept[x ^ irrep] == kept[x] for x in range(8))


@pytest.mark.parametrize("irrep", [0, 1, 2])
def test_cpu_singlet_eom_sector_paths_agree(irrep):
    """Singlet EOM-CCSDTQ actions of a C2v model: on the dense layout with the
    central difference (no labels), with one residual per action (nonzero
    irreps: the sector part of Res(T + eps R) is the central difference, bit
    for bit), and with a solver in the subgroup that keeps the target totally
    symmetric (symmetry-blocked GEMMs, compressed T4, internal virtual order)."""
    from itercc import _cpu_quadruples
    from itercc.eom_ccsdtq import FiniteDifferenceEOM
    from itercc.spatial_symmetry import sector_labels

    if not _cpu_quadruples.available():
        pytest.skip("compiled CPU core unavailable")
    model = symmetric_model()
    dense = RestrictedCCSDTQ(*model, backend="cpu")
    ground = dense.solve(max_iter=3, damping=0.2, diis_size=0).amplitudes
    reference = FiniteDifferenceEOM(dense, ground, LABELS, irrep, one_sided=False)
    vector = np.random.default_rng(13 + irrep).normal(size=reference.space.size)
    base = reference.sigma(vector)
    scale = np.abs(base).max()
    if irrep:
        one_sided = FiniteDifferenceEOM(dense, ground, LABELS, irrep)
        assert one_sided.one_sided
        np.testing.assert_array_equal(one_sided.sigma(vector), base)
    solver = RestrictedCCSDTQ(*model, backend="cpu", orbital_irreps=sector_labels(LABELS, irrep))
    assert solver.layout.symmetric and solver.layout.size < dense.layout.size
    converted = solver.combine_states([(1.0, ground, None)])
    engine = FiniteDifferenceEOM(solver, converted, LABELS, irrep)
    assert engine.one_sided == bool(irrep)
    np.testing.assert_array_equal(engine.diagonal, reference.diagonal)
    np.testing.assert_allclose(engine.sigma(vector), base, rtol=0, atol=(1e-14 if irrep else 1e-10) * scale)
    omega = float(vector @ base) / float(vector @ vector)
    assert engine.full_residual_norm(vector, omega) == pytest.approx(
        reference.full_residual_norm(vector, omega), rel=1e-9)
    # The rank-four positions exist only when asked for.
    assert engine.space._indices is None
    np.testing.assert_array_equal(np.concatenate(engine.space.indices), np.concatenate(reference.space.indices))


def test_cpu_singlet_eom_sector_outputs_keep_the_action(monkeypatch):
    """One-sided actions form only the target sector's R4 entries in the tile
    programs (each fused block split by the classes of its rows and columns)
    and the ring terms (each GEMM block split likewise): the same action bit
    for bit."""
    from itercc import _cpu_packed, _cpu_quadruples
    from itercc.eom_ccsdtq import FiniteDifferenceEOM
    from itercc.spatial_symmetry import sector_labels

    if not _cpu_quadruples.available():
        pytest.skip("compiled CPU core unavailable")
    model = symmetric_model()
    dense = RestrictedCCSDTQ(*model, backend="cpu")
    ground = dense.solve(max_iter=3, damping=0.2, diis_size=0).amplitudes
    solver = RestrictedCCSDTQ(*model, backend="cpu", orbital_irreps=sector_labels(LABELS, 2))
    engine = FiniteDifferenceEOM(solver, solver.combine_states([(1.0, ground, None)]), LABELS, 2)
    assert engine._sector is not None
    vector = np.random.default_rng(29).normal(size=engine.space.size)
    monkeypatch.setenv("ITERCC_EOM_SECTOR_OUTPUTS", "0")
    every = engine.sigma(vector)
    monkeypatch.setenv("ITERCC_EOM_SECTOR_OUTPUTS", "1")
    np.testing.assert_array_equal(engine.sigma(vector), every)
    fused = [f for recorded in solver._tile_program._recorded.values() for f in recorded.fused.values() if f]
    assert fused and all(f.sectors for f in fused)
    # Ring plans of the sector (shared by layouts of one symmetry).
    sector = engine._sector.key
    plans = [cache for key, cache in _cpu_packed._SHARED.items() if "packed rings" in str(key[0])]
    assert any(sector in key for cache in plans for key in cache if isinstance(key, tuple))
    assert solver.layout.output_sector is None                 # only within the action


def test_cpu_packed_plan_sector_forms_the_sector_entries():
    """`build_plan` with a sector forms exactly the output entries whose fine
    labels XOR to the target, with the unfiltered plan's values bit for bit,
    and leaves the others unwritten."""
    from itercc import _cpu_kernels, _cpu_packed, _cpu_quadruples

    if not _cpu_quadruples.available():
        pytest.skip("compiled CPU core unavailable")
    rng = np.random.default_rng(5)
    fine = dict(i=np.array([0, 1, 2, 3, 1]), a=np.array([0, 0, 1, 1, 2, 2, 3, 3, 1]),
                e=np.array([0, 1, 2, 3, 0, 1, 2]))
    coarse = {c: np.sort(v % 2) for c, v in fine.items()}       # the subgroup labels mod 2, sorted
    fine = {c: v[np.argsort(v % 2, kind="stable")] for c, v in fine.items()}
    labels = dict(i=coarse["i"], a=coarse["a"], b=coarse["a"], e=coarse["e"])
    fines = dict(i=fine["i"], a=fine["a"], b=fine["a"], e=fine["e"])
    A = rng.normal(size=(5, 7))                                  # i e
    B = rng.normal(size=(7, 9, 9))                               # e a b
    library = _cpu_kernels.library()
    outs = []
    for sector in (None, (fines, 2)):                            # B1 of C2v: coarse 0
        out = np.full((5, 9, 9), 7.0)
        plan = _cpu_packed.build_plan("ie,eab->iab", labels, _cpu_packed.Dense("ie", A),
                                      _cpu_packed.Dense("eab", B), _cpu_packed.Dense("iab", out), 2, sector)
        plan.run(library, A.ctypes.data, B.ctypes.data, out.ctypes.data, 1.0, 1)
        outs.append(out)
    allowed = (labels["i"][:, None, None] ^ labels["a"][None, :, None] ^ labels["a"][None, None, :]) \
        == 0
    inside = (fines["i"][:, None, None] ^ fines["a"][None, :, None] ^ fines["a"][None, None, :]) == 2
    np.testing.assert_array_equal(outs[0][~allowed], 7.0)
    np.testing.assert_array_equal(outs[1][inside], outs[0][inside])
    np.testing.assert_array_equal(outs[1][~inside], 7.0)
    assert inside.any() and (allowed & ~inside).any()


def test_cpu_singlet_eom_shifts_the_ground_amplitudes_in_place(tmp_path):
    """A one-sided action shifts the engine's ground amplitudes in place: the
    same image as a shifted copy, bit for bit, with the totally symmetric
    entries untouched and the sector's zero afterwards; read-only amplitudes
    (memory maps) are copied once."""
    from itercc import _cpu_quadruples
    from itercc.eom_ccsdtq import FiniteDifferenceEOM
    from itercc.spatial_symmetry import sector_labels

    if not _cpu_quadruples.available():
        pytest.skip("compiled CPU core unavailable")
    model = symmetric_model()
    dense = RestrictedCCSDTQ(*model, backend="cpu")
    ground = dense.solve(max_iter=3, damping=0.2, diis_size=0).amplitudes
    solver = RestrictedCCSDTQ(*model, backend="cpu", orbital_irreps=sector_labels(LABELS, 2))
    start = solver.combine_states([(1.0, ground, None)])
    engine = FiniteDifferenceEOM(solver, start, LABELS, 2)
    before = tuple(a.copy() for a in engine.amplitudes)
    vector = np.random.default_rng(31).normal(size=engine.space.size)
    shifted = tuple(a.copy() for a in before)
    engine.space.shift(shifted, vector, engine.step, assign=True)
    solver.layout.output_sector = engine._sector
    expected = engine.space.pack(solver._residuals(shifted)) / engine.step
    solver.layout.output_sector = None
    np.testing.assert_array_equal(engine.sigma(vector), expected)
    marker = tuple(np.zeros(a.shape) for a in before)
    engine.space.shift(marker, np.ones(engine.space.size), 1.0, assign=True)
    for after, previous, inside in zip(engine.amplitudes, before, marker):
        inside = inside != 0
        assert inside.any() and not after[inside].any()
        np.testing.assert_array_equal(after[~inside], previous[~inside])
    # A read-only memory map is copied, not written.
    np.save(tmp_path / "t4.npy", start[3])
    mapped = np.load(tmp_path / "t4.npy", mmap_mode="r")
    other = FiniteDifferenceEOM(solver, (*start[:3], mapped), LABELS, 2)
    assert other.amplitudes[3] is not mapped and type(other.amplitudes[3]) is np.ndarray
    np.testing.assert_array_equal(other.sigma(vector), expected)


def test_singlet_eom_benchmark_variants_agree(tmp_path):
    """benchmarks.singlet_eom_ccsdtq on a small synthetic model: every variant
    reports actions that agree with the dense central difference."""
    import json

    from itercc import _cpu_quadruples

    if not _cpu_quadruples.available():
        pytest.skip("compiled CPU core unavailable")
    output = tmp_path / "records.jsonl"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-m", "benchmarks.singlet_eom_ccsdtq", "--synthetic", "3", "9",
                    "--irrep", "1", "--actions", "1", "--ground-iterations", "1", "--threads", "2",
                    "--output", str(output)], check=True, cwd=root, capture_output=True,
                   env={**os.environ, "PYTHONPATH": str(root)})
    records = [json.loads(line) for line in output.read_text().splitlines()]
    actions = [r for r in records if r["event"] == "action"]
    assert [r["variant"] for r in actions] == ["dense-central", "dense-one-sided", "sector", "sector-every",
                                               "sector-single", "sector-central"]
    # float32 copies of T3 and the summed T4 rows move the action by their rounding only.
    assert all(r["relative_difference"] < (1e-6 if r["variant"] == "sector-single" else 1e-12)
               for r in actions[1:])


def test_cpu_singlet_eom_refuses_labels_that_split_the_sector():
    from itercc import _cpu_quadruples
    from itercc.eom_ccsdtq import FiniteDifferenceEOM

    if not _cpu_quadruples.available():
        pytest.skip("compiled CPU core unavailable")
    model = symmetric_model()
    solver = RestrictedCCSDTQ(*model, backend="cpu", orbital_irreps=LABELS)
    with pytest.raises(ValueError, match="sector_labels"):
        FiniteDifferenceEOM(solver, None, LABELS, 2)


def test_fused_gemm_orientations_agree():
    """Fused tile GEMMs give the same bits whichever orientation they take (the
    core picks the one that fills the register tiles better per product)."""
    from itercc import _cpu_kernels, _cpu_quadruples

    if not _cpu_quadruples.available():
        pytest.skip("compiled CPU core unavailable")
    model = symmetric_model()
    dense_solver, amplitudes = warm_amplitudes(model, backend="cpu")
    amplitudes = (*amplitudes[:3], host(amplitudes[3]).reshape(dense_solver.layout.dense_shape))
    solver = RestrictedCCSDTQ(*model, orbital_irreps=LABELS, backend="cpu")
    library = _cpu_kernels.library()
    results = []
    try:
        for mode in (0, 1, 2):
            library.fused_orientation(mode)
            results.append(solver.residuals(amplitudes))
    finally:
        library.fused_orientation(0)
    assert all(recorded.fused for recorded in solver._tile_program._recorded.values())
    for other in results[1:]:
        for first, second in zip(results[0], other):
            np.testing.assert_array_equal(first, second)



def test_symmetric_solve_matches_dense(backend):
    model = symmetric_model()
    settings = dict(max_iter=80, damping=0.1, energy_tol=1e-12, residual_tol=1e-10, diis_size=6)
    dense = RestrictedCCSDTQ(*model, block_size=2, backend=backend).solve(**settings)
    symmetric = RestrictedCCSDTQ(*model, block_size=3, orbital_irreps=LABELS,
                                 backend=backend).solve(**settings)
    assert dense.converged and symmetric.converged
    assert abs(dense.energy - symmetric.energy) < 1e-11
    for a, b in zip(dense.amplitudes[:3], symmetric.amplitudes[:3]):
        np.testing.assert_allclose(host(b), host(a), atol=1e-9)


def test_packed_triples_diis_matches_full_arrays(backend):
    from itercc.quadruples import _PackedTriplesDIIS
    from itercc.solver import _DIIS

    model = symmetric_model()
    xp = array_module(backend)
    solver, _ = warm_amplitudes(model, 1, backend)
    full, packed = _DIIS(4, xp), _PackedTriplesDIIS(4, xp, solver.nocc)
    for step in range(6):
        _, amplitudes = warm_amplitudes(model, 2 + step, backend)
        residuals = solver.residuals(amplitudes)[:3]
        expected = full.update(tuple(a.copy() for a in amplitudes[:3]), residuals)
        actual = packed.update(tuple(a.copy() for a in amplitudes[:3]), residuals)
        for a, b in zip(expected, actual):
            np.testing.assert_allclose(host(b), host(a), atol=1e-13, rtol=1e-11)
    np.testing.assert_allclose(packed.gram, full.gram, rtol=1e-12)


def test_triples_diis_drops_a_stale_history(backend):
    """A converged CCSDT start (T1–T3 error ~0 at T4 = 0) must not pin the extrapolation."""
    from itercc.quadruples import _PackedTriplesDIIS

    model = symmetric_model()
    solver, _ = warm_amplitudes(model, 1, backend)
    diis = _PackedTriplesDIIS(4, array_module(backend), solver.nocc)
    _, amplitudes = warm_amplitudes(model, 2, backend)
    tiny = tuple(1e-12 * r for r in solver.residuals(amplitudes)[:3])
    diis.update(tuple(a.copy() for a in amplitudes[:3]), tiny)
    _, moved = warm_amplitudes(model, 3, backend)
    steps = solver.residuals(moved)[:3]
    out = diis.update(tuple(a.copy() for a in moved[:3]), steps)
    assert len(diis.history) == 1                      # the stale entry is gone
    for a, b in zip(out, moved[:3]):
        np.testing.assert_array_equal(host(a), host(b))


@pytest.mark.parametrize("orbital_irreps", [None, LABELS])
def test_saved_state_resumes_exactly(backend, tmp_path, orbital_irreps):
    model = symmetric_model()
    settings = dict(damping=0.2, diis_size=3)
    straight = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=orbital_irreps,
                                backend=backend)
    reference = straight.solve(max_iter=7, **settings)

    first = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=orbital_irreps,
                             backend=backend)
    first.solve(max_iter=4, checkpoint=lambda state: first.save_state(tmp_path / "state", state,
                                                                      dict(note="test")), **settings)
    second = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=orbital_irreps,
                              backend=backend)
    state, metadata = second.load_state(tmp_path / "state")
    assert metadata == dict(note="test") and state.evaluations == 3
    numbers = []
    resumed = second.solve(resume=state, max_iter=4, callback=lambda n, info: numbers.append(n),
                           **settings)
    assert numbers == [4, 5, 6, 7]
    assert len(resumed.iterations) == 7
    assert resumed.energy == reference.energy
    for a, b in zip(reference.amplitudes, resumed.amplitudes):
        np.testing.assert_array_equal(host(b), host(a))
    with pytest.raises(ValueError, match="state used"):
        second.solve(resume=state, max_iter=1, damping=0.3, diis_size=3)


@CUDA
def test_resume_in_a_new_process_keeps_the_gemm_routes(tmp_path, monkeypatch):
    """A saved CUDA state carries the autotuned GEMM routes: a process whose
    timings would pick the other routes still continues bit for bit."""
    from itercc import gemm_contract

    def timed(direct_wins):
        def seconds(function):
            function()
            seconds.calls += 1
            return float((seconds.calls % 2 == 1) != direct_wins)   # direct timed first
        seconds.calls = 0
        return seconds

    monkeypatch.setattr(gemm_contract, "AUTOTUNE", True)
    monkeypatch.setattr(gemm_contract, "_route", {})
    monkeypatch.setattr(gemm_contract, "_seconds", timed(True))
    model = symmetric_model()
    settings = dict(damping=0.2, diis_size=3)
    reference = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=LABELS,
                                 backend="cuda").solve(max_iter=6, **settings)
    first = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=LABELS, backend="cuda")
    first.solve(max_iter=3, checkpoint=lambda state: first.save_state(tmp_path / "state", state),
                **settings)
    assert any(gemm_contract._route.values())
    monkeypatch.setattr(gemm_contract, "_route", {})                # a new process
    monkeypatch.setattr(gemm_contract, "_seconds", timed(False))
    second = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=LABELS, backend="cuda")
    state, _ = second.load_state(tmp_path / "state")
    resumed = second.solve(resume=state, max_iter=4, **settings)
    assert resumed.energy == reference.energy
    for a, b in zip(reference.amplitudes, resumed.amplitudes):
        np.testing.assert_array_equal(host(b), host(a))



@pytest.mark.parametrize("backend,host_r4,chunk", [("cpu", False, None), ("cpu", False, 1000)] + [
    pytest.param("cuda", *case, marks=CUDA) for case in [(False, None), (True, None), (False, 1000)]])
def test_host_t4_diis_matches_device_t4_diis(backend, monkeypatch, host_r4, chunk):
    """T4 history in host memory gives the same iterates as the device history."""
    from itercc.quadruples import _HostQuadruplesDIIS

    monkeypatch.setenv("ITERCC_DIIS_COMPACT", "0")      # the float64 history
    if chunk:                           # many host chunks per pass
        monkeypatch.setattr(_HostQuadruplesDIIS, "CHUNK", chunk)
    model = symmetric_model()
    settings = dict(max_iter=8, damping=0.2, diis_size=4)
    device = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=LABELS, backend=backend).solve(
        diis_quadruples=True, **settings)
    solver = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=LABELS, host_r4=host_r4,
                              backend=backend)
    staged = solver.solve(diis_quadruples="host", **settings)
    for a, b in zip(device.iterations, staged.iterations):
        assert abs(a.energy - b.energy) < 1e-12
        np.testing.assert_allclose(b.residual_max, a.residual_max, rtol=1e-8, atol=1e-14)
    for a, b in zip(device.amplitudes, staged.amplitudes):
        np.testing.assert_allclose(host(b), host(a), rtol=1e-9, atol=1e-13)


def test_compact_t4_diis_matches_float64_history(monkeypatch):
    """The float32 T4 history of the compiled CPU backend follows the float64
    one to rounding relative to the steps and converges to the same solution."""
    from itercc import _cpu_quadruples

    if not _cpu_quadruples.available():
        pytest.skip("compiled CPU kernels unavailable")
    model = symmetric_model()
    settings = dict(damping=0.2, diis_size=4, diis_quadruples="host", energy_tol=1e-12,
                    residual_tol=1e-10, max_iter=80)
    runs = {}
    for compact in ("0", "1"):
        monkeypatch.setenv("ITERCC_DIIS_COMPACT", compact)
        solver = RestrictedCCSDTQ(*model, block_size=1, orbital_irreps=LABELS, backend="cpu")
        runs[compact] = solver.solve(**settings)
    exact, compact = runs["0"], runs["1"]
    for a, b in zip(exact.iterations[:8], compact.iterations[:8]):
        assert abs(a.energy - b.energy) < 1e-12
        np.testing.assert_allclose(b.residual_max, a.residual_max, rtol=1e-6, atol=1e-13)
    assert exact.converged and compact.converged
    assert abs(len(compact.iterations) - len(exact.iterations)) <= 2
    assert abs(compact.energy - exact.energy) < 1e-11
    for a, b in zip(exact.amplitudes, compact.amplitudes):
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-9)


def test_compact_diis_kernels_match_numpy():
    from itercc import _cpu_kernels
    from itercc.quadruples import _pointers

    library = _cpu_kernels.library()
    if library is None:
        pytest.skip("compiled CPU kernels unavailable")
    rng = np.random.default_rng(6)
    n, k = 70001, 3
    step = rng.standard_normal(n)
    errors = [rng.standard_normal(n).astype(np.float32) for _ in range(k)]
    dots = np.zeros(k + 1)
    library.dots_f32(n, k, step.ctypes.data, _pointers(errors), dots.ctypes.data, 3)
    np.testing.assert_allclose(dots, [step @ e.astype(np.float64) for e in errors] + [step @ step],
                               rtol=1e-12)
    for weights in (np.array([0.2, -0.1, 0.4, 0.5]), None):
        trial = rng.standard_normal(n)
        deltas = [rng.standard_normal(n).astype(np.float32) for _ in range(k)]
        if weights is None:
            change = step
        else:
            change = sum(w * d.astype(np.float64) for w, d in zip(weights, deltas)) + weights[-1] * step
        wanted_x = trial + (change - step)
        wanted = [(d.astype(np.float64) - change).astype(np.float32) for d in deltas]
        x, new, error = trial.copy(), np.empty(n, np.float32), np.empty(n, np.float32)
        library.diis_rebase(n, k, None if weights is None else weights.ctypes.data, _pointers(deltas),
                            new.ctypes.data, error.ctypes.data, x.ctypes.data, step.ctypes.data, 4)
        np.testing.assert_allclose(x, wanted_x, rtol=0, atol=1e-13)
        for d, w in zip(deltas, wanted):
            np.testing.assert_allclose(d, w, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(error, step.astype(np.float32))
        np.testing.assert_allclose(new, (step - change).astype(np.float32), rtol=1e-6, atol=1e-6)


def test_host_t4_diis_state_resumes(backend, tmp_path):
    """A host-DIIS state resumes exactly with its history and keeps converging without it."""
    model = symmetric_model()
    settings = dict(damping=0.2, diis_size=3, diis_quadruples="host")
    reference = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=LABELS,
                                 backend=backend).solve(max_iter=7, **settings)
    first = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=LABELS, backend=backend)
    saved = []
    first.solve(max_iter=4, checkpoint=saved.append, **settings)
    assert saved[-1].history_bytes() > 0
    first.save_state(tmp_path / "full", saved[-1])
    first.save_state(tmp_path / "bare", saved[-1].without_history())
    second = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=LABELS, backend=backend)
    state, _ = second.load_state(tmp_path / "full")
    resumed = second.solve(resume=state, max_iter=4, **settings)
    assert resumed.energy == reference.energy
    bare, _ = second.load_state(tmp_path / "bare")
    assert bare.diis_history == [] and bare.evaluations == 3
    rebuilt = second.solve(resume=bare, max_iter=60, energy_tol=1e-12, residual_tol=1e-10, **settings)
    assert rebuilt.converged and len(rebuilt.iterations) > 7


@CUDA
@pytest.mark.parametrize("diis_quadruples", [False, "host"])
def test_cuda_state_resumes_on_the_numpy_backend(cuda, tmp_path, diis_quadruples):
    """Both backends share the packed T4 layout and the DIIS history of a saved state."""
    model = symmetric_model()
    settings = dict(damping=0.2, diis_size=3, diis_quadruples=diis_quadruples)
    reference = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=LABELS).solve(max_iter=7, **settings)
    first = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=LABELS)
    first.solve(max_iter=4, checkpoint=lambda state: first.save_state(tmp_path / "state", state),
                **settings)
    second = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=LABELS, backend="cpu")
    state, _ = second.load_state(tmp_path / "state")
    assert state.diis_history
    resumed = second.solve(resume=state, max_iter=4, **settings)
    assert abs(resumed.energy - reference.energy) < 1e-12
    for a, b in zip(reference.amplitudes, resumed.amplitudes):
        np.testing.assert_allclose(b, host(a), rtol=1e-9, atol=1e-13)


def test_array_directory_streams_device_arrays_both_ways(cuda, tmp_path):
    from itercc.checkpoint_io import read_array_directory, to_device, write_array_directory

    rng = np.random.default_rng(8)
    device = cuda.asarray(rng.normal(size=(7, 11, 13)))
    host = rng.normal(size=(5, 3))
    empty = cuda.zeros((0, 4))
    write_array_directory(tmp_path / "state", dict(device=device, host=host, empty=empty),
                          dict(note="x"), chunk_bytes=8 * 37)
    arrays, meta = read_array_directory(tmp_path / "state")
    assert meta == dict(note="x") and arrays["empty"].shape == (0, 4)
    for source, expected in ((arrays["device"], device), (arrays["host"], cuda.asarray(host))):
        for chunk in (8 * 37, 2**20):
            copied = to_device(source, chunk_bytes=chunk)
            np.testing.assert_array_equal(cuda.asnumpy(copied), cuda.asnumpy(expected))
    sliced = to_device(arrays["device"][2:5], chunk_bytes=8 * 37)
    np.testing.assert_array_equal(cuda.asnumpy(sliced), cuda.asnumpy(device[2:5]))


def test_checkpoint_upload_preserves_mapped_generation_and_strides(cuda, tmp_path):
    from itercc.checkpoint_io import read_array_directory, to_device, write_array_directory

    path = tmp_path / "state"
    old = np.arange(60.0).reshape(5, 12)[:, ::2]
    write_array_directory(path, {"test": cuda.asarray(old.T).T}, {}, chunk_bytes=64)
    mapped = read_array_directory(path)[0]["test"]
    write_array_directory(path, {"test": old * 2}, {}, chunk_bytes=64)
    out = cuda.empty((6, 5)).T
    assert to_device(mapped, out=out, chunk_bytes=64) is out
    np.testing.assert_array_equal(out.get(), old)
    to_device(old[:, ::-1], out=out, chunk_bytes=64)
    np.testing.assert_array_equal(out.get(), old[:, ::-1])
    with pytest.raises(ValueError, match="shapes differ"):
        to_device(old, out=cuda.empty(old.size))


def test_saved_ccsdtq_state_rejects_another_hamiltonian(tmp_path):
    h, eri, no, constant = symmetric_model()
    solver = RestrictedCCSDTQ(h, eri, no, constant, backend="cpu")
    path = tmp_path / "state"
    solver.solve(max_iter=2, checkpoint=lambda state: solver.save_state(path, state))
    different = h.copy()
    different[no, no] += 0.02
    receiver = RestrictedCCSDTQ(different, eri, no, constant, backend="cpu")
    with pytest.raises(ValueError, match="belongs to another Hamiltonian"):
        receiver.load_state(path)


def test_numpy_quadruples_sector_handles_strided_residuals_and_updates():
    from itercc.eom_ccsdtq import QuadruplesSector

    solver = RestrictedCCSDTQ(*symmetric_model(), backend="cpu")
    sector = QuadruplesSector(solver, LABELS, 0)
    arrays = tuple(np.arange(np.prod(shape), dtype=float).reshape(shape)[..., ::-1]
                   for shape in solver.shapes)
    expected = np.concatenate([np.asarray(array).ravel()[indices]
                               for array, indices in zip(arrays, sector.indices)])
    np.testing.assert_array_equal(sector.pack(arrays), expected)
    update = np.linspace(-1, 1, sector.size)
    sector.shift(arrays, update, 0.25)
    np.testing.assert_array_equal(sector.pack(arrays), expected + 0.25 * update)


def test_background_state_writes_resume_exactly(backend, tmp_path):
    from itercc.checkpoint_io import BackgroundArrayWriter

    model = symmetric_model()
    settings = dict(damping=0.2, diis_size=3)
    reference = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=LABELS, backend=backend).solve(
        max_iter=6, **settings)
    first = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=LABELS, backend=backend)
    writer = BackgroundArrayWriter()
    saved = []

    def checkpoint(state):
        first.save_state(tmp_path / "state", state, writer=writer)
        saved.append(state.evaluations)

    first.solve(max_iter=3, checkpoint=checkpoint, **settings)
    writer.wait()
    assert saved == [1, 2] and writer.last_seconds is not None
    second = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=LABELS, backend=backend)
    state, _ = second.load_state(tmp_path / "state")
    resumed = second.solve(resume=state, max_iter=4, **settings)
    assert resumed.energy == reference.energy
    for a, b in zip(reference.amplitudes, resumed.amplitudes):
        np.testing.assert_array_equal(host(b), host(a))


def test_state_directory_can_be_replaced_while_resumed_from(cuda, tmp_path):
    """A resumed solve rewrites the directory it read; leftovers never block it."""
    from itercc.checkpoint_io import SavedArray, read_array_directory, to_device, write_array_directory

    target = tmp_path / "state"
    first = cuda.arange(10.0)
    write_array_directory(target, dict(a=first), dict(n=1))
    arrays, _ = read_array_directory(target, lazy=True)
    assert isinstance(arrays["a"], SavedArray) and arrays["a"].shape == (10,)
    (tmp_path / ".state.previous-stale").mkdir()
    (tmp_path / ".state.previous-stale" / ".nfs000123").write_text("held open elsewhere")
    for n in (2, 3):
        write_array_directory(target, dict(a=first * n), dict(n=n))
    np.testing.assert_array_equal(cuda.asnumpy(to_device(arrays["a"])), np.arange(10.0) * 3)
    np.testing.assert_array_equal(np.asarray(read_array_directory(target)[0]["a"]), np.arange(10.0) * 3)


def test_failed_checkpoint_write_preserves_recoverable_state(tmp_path, monkeypatch):
    from itercc import checkpoint_io

    target = tmp_path / "state"
    fallback = tmp_path / ".state.previous-interrupted"
    first = np.arange(10.0)
    checkpoint_io.write_array_directory(target, dict(a=first), dict(n=1))
    target.rename(fallback)

    def fail(*args):
        raise OSError("injected checkpoint write failure")

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint_io, "_write_npy", fail)
        with pytest.raises(OSError, match="injected checkpoint"):
            checkpoint_io.write_array_directory(target, dict(a=first * 2), dict(n=2))

    arrays, meta = checkpoint_io.read_array_directory(target, lazy=True)
    np.testing.assert_array_equal(np.asarray(arrays["a"]), first)
    assert meta == dict(n=1)
    checkpoint_io.write_array_directory(target, dict(a=first * 2), dict(n=2))
    assert checkpoint_io.read_array_directory(target)[1] == dict(n=2)
    assert not fallback.exists()


def test_host_staged_r4_norm_detects_nan(cuda):
    from itercc._quadruples_cuda import QuadruplesLayout

    layout = QuadruplesLayout(2, 3)
    r4 = layout.zeros(host=True)
    r4[-1] = np.nan
    assert np.isnan(layout.max_abs(r4))


def test_numpy_layout_norm_detects_nan():
    from itercc._quadruples_cpu import NumpyQuadruplesLayout

    layout = NumpyQuadruplesLayout(2, 3)
    r4 = layout.zeros()
    r4[-1] = np.nan
    assert np.isnan(layout.max_abs(r4))


@pytest.mark.parametrize("symmetric", [False, True])
def test_t4_pass_gives_the_same_residuals_in_chunks(backend, monkeypatch, symmetric):
    from itercc import _quadruples_equations as eq

    model = symmetric_model()
    dense, amplitudes = warm_amplitudes(model, backend=backend)
    solver = (RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=LABELS, backend=backend)
              if symmetric else dense)
    if symmetric:
        amplitudes = (*amplitudes[:3], host(amplitudes[3]).reshape(dense.layout.dense_shape))
    expected = solver.residuals(amplitudes)
    monkeypatch.setitem(eq.t4_into_lower_ranks_.__kwdefaults__, "budget", 1)
    actual = solver.residuals(amplitudes)
    for rank, (a, b) in enumerate(zip(actual, expected), 1):
        np.testing.assert_allclose(host(a), host(b), rtol=1e-12, atol=1e-14,
                                   err_msg=f"rank {rank}")


def test_solve_can_iterate_initial_amplitudes_in_place(backend):
    model = symmetric_model()
    dense_solver, amplitudes = warm_amplitudes(model, evaluations=2, backend=backend)
    solver = RestrictedCCSDTQ(*model, block_size=2, orbital_irreps=LABELS, backend=backend)
    dense = host(amplitudes[3]).reshape(dense_solver.layout.dense_shape)
    start = solver.combine_states([(1.0, (*amplitudes[:3], dense), None)])
    settings = dict(max_iter=4, damping=0.2, diis_size=3)
    copied = solver.solve(initial=start, **settings)
    assert not array_module(backend).shares_memory(copied.amplitudes[3], start[3])
    owned = solver.solve(initial=start, copy_initial=False, **settings)
    assert owned.amplitudes[3] is start[3]
    assert owned.energy == copied.energy
    for a, b in zip(owned.amplitudes, copied.amplitudes):
        np.testing.assert_array_equal(host(a), host(b))


@pytest.mark.parametrize("host_r4", [False, True])
@pytest.mark.parametrize("symmetric", [False, True])
def test_device_estimate_covers_the_measured_peak(cuda, symmetric, host_r4):
    from benchmarks.ccsdtq_profile import synthetic

    # Large enough for T4, the T3-sized intermediates and the tile buffers to
    # dominate, as they do in production (tiny systems are all overhead).
    h, eri, no, constant, labels = synthetic(4, 20, 5)
    pool = cuda.get_default_memory_pool()
    pool.free_all_blocks()
    base = pool.used_bytes()
    peak = [0]

    def malloc(size):     # live bytes after every allocation, not the cached pool
        memory = pool.malloc(size)
        peak[0] = max(peak[0], pool.used_bytes() - base)
        return memory

    cuda.cuda.set_allocator(malloc)
    try:
        solver = RestrictedCCSDTQ(h, eri, no, constant, block_size=2, host_r4=host_r4,
                                  orbital_irreps=labels if symmetric else None)
        solver.solve(max_iter=3, damping=0.2, diis_size=6)
    finally:
        cuda.cuda.set_allocator(pool.malloc)
    estimate = solver.device_bytes(diis_size=6)
    # Covers the peak (with the allowance for fragmentation) without being vacuous.
    assert peak[0] <= estimate <= 2 * peak[0]
