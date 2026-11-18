from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("active", ["all", "singles", "quadruples"])
def test_native_triplet_quadruples_agree_with_numpy(active):
    from itercc import TripletEOMCCSDTQ
    from test_quadruples import warm_amplitudes, symmetric_model
    labels = np.array([0, 1, 2, 0, 1, 2, 3])
    solver, amplitudes = warm_amplitudes(symmetric_model(labels, no=3), evaluations=2, backend="cpu")
    engine = TripletEOMCCSDTQ(solver, amplitudes, labels, 2, occupied_tiles=True,
                             action_only=True, intermediate_cache_gib=0.0001)
    vector = np.random.default_rng(729).normal(size=engine.space.size)
    if active != "all":
        part = engine.space.slices[0 if active == "singles" else 3]
        keep = vector[part].copy()
        vector[:] = 0
        vector[part] = keep
    for limit in (5, len(engine._direction_t4.slots)):
        engine._ground_t4.native_expansion = engine._direction_t4.native_expansion = False
        engine.native_projection = False
        reference = engine.probe_rank_four(vector, limit)
        engine._ground_t4.native_expansion = engine._direction_t4.native_expansion = True
        engine.native_projection = True
        actual = engine.probe_rank_four(vector, limit)
        np.testing.assert_allclose(actual, reference, atol=2e-12, rtol=2e-12)


def test_packed_t4_rows_in_the_gemms_change_nothing(monkeypatch):
    """The blocked GEMMs read packed T4 rows in place: the same action, bit for bit."""
    from itercc import TripletEOMCCSDTQ
    from itercc._cpu_symmetry_blocks import CpuSymmetryBlocks
    from test_quadruples import warm_amplitudes, symmetric_model
    labels = np.array([0, 1, 2, 0, 1, 2, 3])
    solver, amplitudes = warm_amplitudes(symmetric_model(labels, no=3), evaluations=2, backend="cpu")
    engine = TripletEOMCCSDTQ(solver, amplitudes, labels, 2, occupied_tiles=True, packed_spin=True,
                             intermediate_cache_gib=0.0001)
    if not isinstance(engine._symmetry_engine, CpuSymmetryBlocks):
        pytest.skip("compiled core unavailable")
    vector = np.random.default_rng(731).normal(size=engine.space.size)
    engine.fused_tiles = False         # the compiled slots then match the loop bit for bit
    monkeypatch.setattr(CpuSymmetryBlocks, "packed_operands", False)
    dense = engine.sigma(vector)
    monkeypatch.setattr(CpuSymmetryBlocks, "packed_operands", True)
    np.testing.assert_array_equal(engine.sigma(vector), dense)


def _triplet_engine(**options):
    from itercc import TripletEOMCCSDTQ
    from itercc._cpu_symmetry_blocks import CpuSymmetryBlocks
    from test_quadruples import warm_amplitudes, symmetric_model
    labels = np.array([0, 1, 2, 0, 1, 2, 3])
    solver, amplitudes = warm_amplitudes(symmetric_model(labels, no=3), evaluations=2, backend="cpu")
    engine = TripletEOMCCSDTQ(solver, amplitudes, labels, 2, occupied_tiles=True, **options)
    if not isinstance(engine._symmetry_engine, CpuSymmetryBlocks):
        pytest.skip("compiled core unavailable")
    return engine


@pytest.mark.parametrize("active", ["all", "no singles", "no quadruples"])
def test_compiled_slots_match_the_loop(active):
    """The rank-four slots as one compiled program: unfused, the loop's action
    bit for bit, also with no R1 (bare integrals) or no R4 (no T4 direction),
    and on any number of threads; with fused tile products the same action to
    round-off, again independent of the thread count."""
    from itercc._cpu_triples import thread_scope
    engine = _triplet_engine(packed_spin=True, intermediate_cache_gib=0.0001)
    vector = np.random.default_rng(733).normal(size=engine.space.size)
    if active != "all":
        vector[engine.space.slices[0 if active == "no singles" else 3]] = 0
    engine.slot_program = False
    loop = engine.sigma(vector)
    engine.slot_program = True
    engine.fused_tiles = False
    for threads in (1, 3):
        with thread_scope(threads):
            np.testing.assert_array_equal(engine._action(vector), loop)
    engine.share_work = 1              # every intermediate in pieces that waiting tiles run too
    for threads in (2, 4):
        with thread_scope(threads):
            np.testing.assert_array_equal(engine._action(vector), loop)
    assert engine.last_intermediate_cache["shared"] > 0
    engine.share_work = None
    engine.fused_tiles = True
    with thread_scope(1):
        fused = engine._action(vector)
    with thread_scope(3):
        np.testing.assert_array_equal(engine._action(vector), fused)
    np.testing.assert_allclose(fused, loop, rtol=0, atol=1e-13 * np.abs(loop).max())
    assert engine._slot_programs and all(p is not None for p in engine._slot_programs.values())


def test_compiled_slots_report_progress():
    """The compiled program reports the slots done after each wave, as the
    loop reports its milestones (the CuF drivers print them)."""
    engine = _triplet_engine(action_only=True, intermediate_cache_gib=0.0001)
    vector = np.random.default_rng(741).normal(size=engine.space.size)
    reports = []
    engine._action(vector, ranks=(4,), slot_limit=5,
                   slot_finished=lambda done, total, seconds: reports.append((done, total)))
    assert reports and reports[-1] == (5, len(engine._direction_t4.slots))
    assert engine._slot_programs and all(p is not None for p in engine._slot_programs.values())


def test_compiled_slots_probe_a_prefix():
    engine = _triplet_engine(action_only=True, intermediate_cache_gib=0.0001)
    vector = np.random.default_rng(737).normal(size=engine.space.size)
    engine.fused_tiles = False
    for limit in (1, 5):
        engine.slot_program = False
        loop = engine.probe_rank_four(vector, limit)
        engine.slot_program = True
        np.testing.assert_array_equal(engine.probe_rank_four(vector, limit), loop)
    assert np.any(loop)


def test_dense_direct_terms_match_the_sparse_lookups(monkeypatch):
    """Rank-four direct packed terms by contexts (each context's T4 values
    gathered once, then dense dot products) add the same nonzero products in
    the same order as the per-row sparse lookups: the same action bit for bit,
    on any number of threads."""
    from itercc._cpu_triples import thread_scope
    engine = _triplet_engine(packed_spin=True, intermediate_cache_gib=0.0001)
    vector = np.random.default_rng(739).normal(size=engine.space.size)
    monkeypatch.setenv("ITERCC_DIRECT_DENSE", "0")
    sparse = engine.sigma(vector)
    monkeypatch.setenv("ITERCC_DIRECT_DENSE", "1")
    for threads in (1, 3):
        with thread_scope(threads):
            np.testing.assert_array_equal(engine._action(vector), sparse)


def _split_engine():
    """The engine of the split-action tests (a slot worker's builder)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    return _triplet_engine(packed_spin=True, intermediate_cache_gib=0.0001)


@pytest.mark.parametrize("program", [True, False])
def test_rank_four_slot_parts_add_up_to_the_action(program):
    """Each slot writes its own rows with values that do not depend on the
    other slots of the process: the parts of a split action add up to the
    whole action bit for bit (compiled program and Python loop)."""
    from itercc._slot_split import split_slots

    engine = _split_engine()
    engine.slot_program = program
    vector = np.random.default_rng(745).normal(size=engine.space.size)
    whole = engine.sigma(vector)
    total = len(engine._direction_t4.slots)
    parts = [split_slots(total, k, 3, block=2) for k in range(3)]
    assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(total))
    summed = np.zeros(engine.space.slices[3].stop - engine.space.slices[3].start)
    for part in parts:
        engine.rank_four_slots = part
        summed += engine._action(vector, ranks=(4,))
    engine.rank_four_slots = None
    assert np.array_equal(summed, whole[engine.space.slices[3]])


def test_split_action_with_a_worker_process(tmp_path):
    """`Partners` runs part of every action in a worker process that builds
    the same engine; the solver's sigma is the one-process action."""
    from itercc._slot_split import Partners

    engine = _split_engine()
    vectors = np.random.default_rng(747).normal(size=(2, engine.space.size))
    whole = [engine.sigma(v) for v in vectors]
    partners = Partners(engine, f"{Path(__file__).resolve()}:_split_engine", {}, parts=2,
                        exchange=tmp_path / "exchange", bind=False, threads=2, block=2)
    try:
        split = [engine.sigma(v) for v in vectors]
        assert len(engine.rank_four_slots) < len(engine._direction_t4.slots)
    finally:
        partners.close()
    assert engine.rank_four_partner is None and not (tmp_path / "exchange").exists()
    for one, two in zip(whole, split):
        np.testing.assert_allclose(two, one, rtol=1e-12, atol=1e-12 * np.abs(one).max())


def test_ordered_tiles_and_wave_sizes_keep_the_action(monkeypatch):
    """Tile products form only the entries the projection reads (ordered
    letters of antisymmetric subsets): the action equals the one with every
    tile entry formed, and does not depend on how many partial sums a wave
    holds."""
    from itercc import _spin_quadruples_program, _spin_quadruples_tiles

    engine = _split_engine()
    vector = np.random.default_rng(749).normal(size=engine.space.size)
    ordered = engine.sigma(vector)
    engine.partial_sums_bytes = 1                  # one slot per wave
    assert np.array_equal(engine.sigma(vector), ordered)
    engine.partial_sums_bytes = None
    for module in (_spin_quadruples_program, _spin_quadruples_tiles):
        monkeypatch.setattr(module, "ordered_letters", lambda *args: ())
    engine._slot_programs = {}                     # recorded again, with dense tiles
    dense = engine.sigma(vector)
    np.testing.assert_allclose(ordered, dense, rtol=1e-13, atol=1e-13 * np.abs(dense).max())
    engine.slot_program = False                    # the Python tile loop, dense tiles
    np.testing.assert_allclose(engine.sigma(vector), dense, rtol=1e-13, atol=1e-13 * np.abs(dense).max())
