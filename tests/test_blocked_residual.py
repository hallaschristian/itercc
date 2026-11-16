"""Residual terms on symmetry-blocked T3 against the dense-slot compiled path."""

import numpy as np
import pytest

from itercc import _blocked_residual, _cpu_kernels, _cpu_triples
from itercc._blocked import BlockedTriples
from itercc._triples_layout import TriplesLayout

OCCUPIED = np.array([0, 1, 2, 0, 3])
VIRTUAL = np.array([1, 2, 3, 0, 3, 0, 1, 2])


@pytest.fixture
def library():
    found = _cpu_kernels.library()
    if found is None:
        pytest.skip("needs the compiled core")
    return found


def symmetric_case(seed):
    rng = np.random.default_rng(seed)
    layout = TriplesLayout(len(OCCUPIED), len(VIRTUAL))
    blocked = BlockedTriples(layout, OCCUPIED, VIRTUAL)
    v = VIRTUAL
    triple = v[:, None, None] ^ v[None, :, None] ^ v[None, None, :]
    t3 = rng.normal(size=layout.shape)
    for tile, gs in zip(t3, blocked.slot_irrep):
        tile[triple != gs] = 0.0
    pair = v[:, None] ^ v[None, :]
    w = rng.normal(scale=0.1, size=(len(v),) * 4)
    w += w.transpose(1, 0, 3, 2).copy()             # pair interchange
    w[pair[:, :, None, None] != pair[None, None]] = 0.0
    return blocked, t3, w


@pytest.mark.parametrize("budget", [2**25, 3000])   # one chunk, several chunks
def test_blocked_ladder_and_purify_match_dense(library, budget):
    blocked, t3, w = symmetric_case(11)
    expected = np.zeros(blocked.layout.shape)
    _cpu_triples._ladder_compiled(library, expected, t3, w, 3, VIRTUAL, budget)
    library.purify(expected.ctypes.data, len(expected), len(VIRTUAL), 3)
    actual = np.zeros(blocked.size)
    _blocked_residual.ladder(blocked, actual, blocked.pack(t3), w, 3, budget=budget)
    _blocked_residual.purify(blocked, actual, 3)
    np.testing.assert_allclose(blocked.unpack(actual), expected, rtol=0, atol=1e-14)
    assert blocked.leakage(expected) < 1e-15


@pytest.mark.parametrize("with_t3", [False, True])
def test_blocked_triples_and_feedback_match_dense(library, monkeypatch, with_t3):
    from itercc import RestrictedCC
    from itercc._blocked_residual import PlaneTables
    from test_cpu_triples import LABELS, labelled_model, symmetric_amplitudes

    no = 4
    monkeypatch.setenv("ITERCC_BLOCKED", "0")          # capture the dense-slot calls
    solver = RestrictedCC(*labelled_model(no, len(LABELS) - no, LABELS), orbital_irreps=LABELS,
                          cpu_threads=3)
    amplitudes = symmetric_amplitudes(solver, LABELS, with_t3)
    calls = {}

    def spy(name):
        original = getattr(_cpu_triples, name)

        def wrapper(*args, **kwargs):
            before = [a.copy() if isinstance(a, np.ndarray) else a for a in args]
            result = original(*args, **kwargs)
            calls[name] = (before, result, args)
            return result
        monkeypatch.setattr(_cpu_triples, name, wrapper)

    spy("triples")
    spy("feedback")
    solver.residuals(amplitudes)
    blocked = BlockedTriples(solver.layout, LABELS[:no], LABELS[no:])
    tables = PlaneTables(blocked, LABELS[:no])
    before, expected, _ = calls["triples"]
    _, t2, t3, have_t3, im, vooo, vvvo, g, o, v, _ = before
    actual = _blocked_residual.triples(blocked, tables, t2, blocked.pack(t3), have_t3, im, vooo,
                                       vvvo, g, o, v, 3, ladder_budget=3000)
    np.testing.assert_allclose(blocked.unpack(actual), expected, rtol=0, atol=1e-14)
    if with_t3:
        before, _, after = calls["feedback"]
        r1, r2, vooo, vvvo, f, g, _, t3, o, v, _ = before
        _blocked_residual.feedback(blocked, tables, r1, r2, vooo, vvvo, f, g, blocked.pack(t3),
                                   o, v, 3)
        for mine, theirs in zip((r1, r2, vooo, vvvo), after[:4]):
            np.testing.assert_allclose(mine, theirs, rtol=0, atol=1e-14)


def test_solver_uses_blocked_path_only_for_symmetric_amplitudes(library, monkeypatch):
    from itercc import RestrictedCC
    from test_cpu_triples import LABELS, labelled_model, symmetric_amplitudes

    no = 4
    model = labelled_model(no, len(LABELS) - no, LABELS)
    solver = RestrictedCC(*model, orbital_irreps=LABELS, cpu_threads=3)
    amplitudes = symmetric_amplitudes(solver, LABELS, True)
    used = []
    original = _blocked_residual.triples
    monkeypatch.setattr(_blocked_residual, "triples",
                        lambda *a, **k: used.append(True) or original(*a, **k))
    monkeypatch.setenv("ITERCC_BLOCKED", "0")
    dense = RestrictedCC(*model, orbital_irreps=LABELS, cpu_threads=3).residuals(amplitudes)
    monkeypatch.delenv("ITERCC_BLOCKED")
    blocked = solver.residuals(amplitudes)
    assert used == [True]
    for left, right in zip(blocked, dense):
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-14)
    # One forbidden T3 entry: the dense path must run, keeping it.
    t3 = amplitudes[2].copy()
    v = LABELS[no:]
    triple = v[:, None, None] ^ v[None, :, None] ^ v[None, None, :]
    s = 0
    occ = LABELS[:no]
    i, j, k = solver.triples_indices[s]
    index = np.argwhere(triple != (occ[i] ^ occ[j] ^ occ[k]))[0]
    t3[(s, *index)] = 1e-9
    if i == j:
        t3[(s, index[1], index[0], index[2])] = 1e-9
    elif j == k:
        t3[(s, index[0], index[2], index[1])] = 1e-9
    solver.residuals((amplitudes[0], amplitudes[1], t3))
    assert used == [True]


def test_solve_on_blocked_path_matches_dense(library, monkeypatch):
    from itercc import RestrictedCC
    from test_cpu_triples import LABELS, labelled_model

    model = labelled_model(3, 4, LABELS[:7])
    controls = dict(energy_tol=1e-12, residual_tol=1e-10, diis_size=3, diis_triples=True)
    blocked = RestrictedCC(*model, orbital_irreps=LABELS[:7], cpu_threads=3).solve(**controls)
    monkeypatch.setenv("ITERCC_BLOCKED", "0")
    dense = RestrictedCC(*model, orbital_irreps=LABELS[:7], cpu_threads=3).solve(**controls)
    assert blocked.converged and dense.converged
    assert len(blocked.iterations) == len(dense.iterations)
    assert blocked.energy == pytest.approx(dense.energy, abs=1e-12, rel=0)


def test_blocked_triples_preserves_raw_symmetry_breaking_seeds(library, monkeypatch):
    from itercc import RestrictedCC
    from test_cpu_triples import LABELS, labelled_model, symmetric_amplitudes

    no = 4
    h, eri, _, constant = labelled_model(no, len(LABELS) - no, LABELS)
    # A symmetry-breaking OVVV operator can enter R3 directly through T2
    # when T3 is zero, without going through any checked ladder intermediate.
    v = slice(no, None)
    perturbation = np.zeros_like(eri)
    perturbation[0, v, v, v] = np.random.default_rng(880).normal(scale=0.002, size=eri[v, v, v, 0].shape)
    perturbation = perturbation + perturbation.swapaxes(0, 1)
    perturbation = perturbation + perturbation.swapaxes(2, 3)
    perturbation = perturbation + perturbation.transpose(2, 3, 0, 1)
    options = dict(orbital_irreps=LABELS, cpu_threads=2)
    raw = RestrictedCC(h, eri + perturbation, no, constant, **options)
    amplitudes = symmetric_amplitudes(raw, LABELS, False)
    actual = raw.residuals(amplitudes)
    monkeypatch.setenv("ITERCC_BLOCKED", "0")
    reference = RestrictedCC(h, eri + perturbation, no, constant, **options)
    for left, right in zip(actual, reference.residuals(amplitudes)):
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-14)


def test_blocked_ladder_is_thread_independent(library):
    blocked, t3, w = symmetric_case(12)
    packed = blocked.pack(t3)
    results = []
    for threads in (1, 4):
        out = np.zeros(blocked.size)
        _blocked_residual.ladder(blocked, out, packed, w, threads, budget=3000)
        _blocked_residual.purify(blocked, out, threads)
        results.append(out)
    np.testing.assert_array_equal(*results)
