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
    gathered once; a symmetry class's contexts in one GEMM with its
    coefficient block) give the per-row sparse lookups' action to round-off,
    the same on any number of threads."""
    from itercc._cpu_triples import thread_scope
    engine = _triplet_engine(packed_spin=True, intermediate_cache_gib=0.0001)
    vector = np.random.default_rng(739).normal(size=engine.space.size)
    monkeypatch.setenv("ITERCC_DIRECT_OPENED", "0")
    monkeypatch.setenv("ITERCC_DIRECT_DENSE", "0")
    sparse = engine.sigma(vector)
    monkeypatch.setenv("ITERCC_DIRECT_DENSE", "1")
    images = []
    for threads in (1, 3):
        with thread_scope(threads):
            images.append(engine._action(vector))
    np.testing.assert_array_equal(images[0], images[1])
    np.testing.assert_allclose(images[0], sparse, rtol=1e-12, atol=1e-13 * np.abs(sparse).max())


@pytest.mark.parametrize("irrep", [0, 2])
def test_opened_direct_terms_match_the_contexts(monkeypatch, irrep):
    """Rank-four direct packed terms by opened letters (per hole tuple J and
    class, the amplitude block gathered once and one GEMM with the coefficient
    block; itercc_spin_quad_opened) give the contexts' action to round-off,
    bit for bit on any number of threads, and per set of target slots."""
    from itercc import TripletEOMCCSDTQ
    from itercc._cpu_symmetry_blocks import CpuSymmetryBlocks
    from itercc._cpu_triples import thread_scope
    from test_quadruples import warm_amplitudes, symmetric_model
    labels = np.array([0, 1, 2, 0, 1, 2, 3])
    solver, amplitudes = warm_amplitudes(symmetric_model(labels, no=3), evaluations=2, backend="cpu")
    engine = TripletEOMCCSDTQ(solver, amplitudes, labels, irrep, occupied_tiles=True, packed_spin=True,
                             intermediate_cache_gib=0.0001)
    if not isinstance(engine._symmetry_engine, CpuSymmetryBlocks):
        pytest.skip("compiled core unavailable")
    vector = np.random.default_rng(755).normal(size=engine.space.size)
    monkeypatch.setenv("ITERCC_DIRECT_OPENED", "0")
    contexts = engine.sigma(vector)
    monkeypatch.setenv("ITERCC_DIRECT_OPENED", "1")
    images = []
    for threads in (1, 3):
        with thread_scope(threads):
            images.append(engine._action(vector))
    np.testing.assert_array_equal(images[0], images[1])
    np.testing.assert_allclose(images[0], contexts, rtol=1e-12, atol=1e-13 * np.abs(contexts).max())
    monkeypatch.setenv("ITERCC_DIRECT_OPENED", "0")
    lookups = engine._action(vector, ranks=(2, 3))
    monkeypatch.setenv("ITERCC_DIRECT_OPENED", "1")
    lower = engine._action(vector, ranks=(2, 3))         # ranks two and three by opened letters too
    np.testing.assert_allclose(lower, lookups, rtol=1e-12, atol=1e-13 * np.abs(lookups).max())
    with thread_scope(3):
        np.testing.assert_array_equal(engine._action(vector, ranks=(2, 3)), lower)
    total = len(engine._direction_t4.slots)
    rows = engine.space.slices[3]
    summed = np.zeros(rows.stop - rows.start)
    for part in (np.arange(0, total, 2), np.arange(1, total, 2)):
        engine.rank_four_slots = part
        summed += engine._action(vector, ranks=(4,))
    engine.rank_four_slots = None
    assert np.array_equal(summed, images[0][rows])


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


def test_lower_rank_tile_parts_add_up_to_the_action():
    """Ranks two and three dealt out by tiles (`lower_part`): part 0 also
    takes the direct packed terms; the parts add up to the whole to rounding."""
    engine = _split_engine()
    vector = np.random.default_rng(753).normal(size=engine.space.size)
    whole = engine._action(vector, ranks=(2, 3))
    summed = np.zeros_like(whole)
    for part in range(3):
        engine.lower_part = (part, 3)
        summed += engine._action(vector, ranks=(2, 3))
    engine.lower_part = None
    np.testing.assert_allclose(summed, whole, rtol=1e-12, atol=1e-13 * np.abs(whole).max())


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


def test_partners_clean_up_after_a_failed_start(tmp_path):
    """A worker that cannot start leaves no exchange files and no split engine."""
    from itercc._slot_split import Partners

    engine = _split_engine()
    with pytest.raises(OSError):
        Partners(engine, f"{Path(__file__).resolve()}:_split_engine", {}, parts=2,
                 exchange=tmp_path / "exchange", bind=False, python=str(tmp_path / "no-python"))
    assert engine.rank_four_partner is None and engine.lower_part is None
    assert not (tmp_path / "exchange").exists()


def test_slot_budget_leaves_room_for_the_davidson_vectors():
    """A default slot-program cache and its partial sums take only what the
    memory budget leaves beside the fixed working set and a solve's Davidson
    vectors; an explicit cache counts in the Davidson working set."""
    engine = _triplet_engine(packed_spin=True, memory_budget_gib=8)
    vectors = (2 * 7 + 6) * engine.space.size * 8
    assert engine.davidson_working_bytes(7) == engine._fixed_bytes + vectors
    cache, partial = engine._slot_budget()
    assert engine._fixed_bytes + vectors + cache + partial <= engine.memory_budget_bytes
    explicit = _triplet_engine(packed_spin=True, memory_budget_gib=8, intermediate_cache_gib=1)
    assert explicit._slot_budget() == (2**30, 2**30)
    assert explicit.davidson_working_bytes(7) == explicit._fixed_bytes + vectors + 2 * 2**30


def test_complex_vectors_run_the_compiled_program():
    """The real and imaginary parts of a complex vector are strided views: the
    compiled slot program gets contiguous copies."""
    engine = _split_engine()
    rng = np.random.default_rng(769)
    v, w = rng.normal(size=(2, engine.space.size))
    expected = engine.sigma(v) + 1e-9j * engine.sigma(w)
    np.testing.assert_allclose(engine.sigma(v + 1e-9j * w), expected, rtol=1e-12, atol=1e-13)


def test_recording_switches_select_another_program(monkeypatch):
    """A switch read while the slot program is recorded keys the engine's cached
    programs: changing it records another program, not the stale one."""
    engine = _split_engine()
    vector = np.random.default_rng(771).normal(size=engine.space.size)
    compact = engine.sigma(vector)
    monkeypatch.setenv("ITERCC_COMPACT_TILES", "0")
    dense = engine.sigma(vector)
    programs = [p for p in engine._slot_programs.values() if p is not None]
    assert len(programs) == 2
    assert any(p._tile_layouts for p in programs) and not all(p._tile_layouts for p in programs)
    np.testing.assert_array_equal(dense, compact)


def test_grouped_ordered_lower_ranks_keep_the_action(monkeypatch):
    """Ranks two and three by projection groups, each tile formed only where
    its particle and hole letters are ordered: the action of one group with
    every diagram, dense tiles and every permutation."""
    from itercc import eom_spin_quadruples
    from itercc._spin_quadruples_diagrams import DIAGRAMS

    engine = _split_engine()
    vector = np.random.default_rng(751).normal(size=engine.space.size)
    grouped = engine._action(vector, ranks=range(1, 4))
    monkeypatch.setattr(eom_spin_quadruples, "projection_diagrams",
                        lambda rank: ((((), ()), 1, DIAGRAMS[rank]),))
    engine._lower_projection = {}
    whole = engine._action(vector, ranks=range(1, 4))
    np.testing.assert_allclose(grouped, whole, rtol=1e-13, atol=1e-13 * np.abs(whole).max())


def test_intermediates_are_formed_only_where_read(monkeypatch):
    """Slot-program intermediates keep the ordering constraints every product
    reading them shares (`SlotProgram._order_intermediates`): some are
    formed in part, and the action is the one with whole intermediates."""
    from itercc import _spin_quadruples_program as sp

    engine = _split_engine()
    vector = np.random.default_rng(757).normal(size=engine.space.size)
    monkeypatch.setenv("ITERCC_ORDERED_INTERMEDIATES", "0")
    engine._slot_programs = {}
    whole = engine.sigma(vector)
    monkeypatch.setenv("ITERCC_ORDERED_INTERMEDIATES", "1")
    engine._slot_programs = {}
    ordered = engine.sigma(vector)
    np.testing.assert_allclose(ordered, whole, rtol=1e-13, atol=1e-13 * np.abs(whole).max())
    program = next(p for p in engine._slot_programs.values() if p is not None)
    producers = [program._contraction_info[kind.contracts[x[3]][0]] for kind in program._kinds
                 for x in kind.code if x[0] == sp.CONTRACT and x[6][0] == sp.INTER]
    assert any(info[5] for info in producers)


def test_compact_intermediates_keep_the_action(monkeypatch):
    """Ordered intermediates store only their kept entries (each chain of
    ordered letters one axis of ranks, `OffsetGroup`): smaller buffers, the
    action of whole buffers bit for bit."""
    engine = _split_engine()
    vector = np.random.default_rng(761).normal(size=engine.space.size)
    monkeypatch.setenv("ITERCC_COMPACT_INTERMEDIATES", "0")
    engine._slot_programs = {}
    whole = engine.sigma(vector)
    program = next(p for p in engine._slot_programs.values() if p is not None)
    dense = program.intermediates["size"].sum()
    monkeypatch.setenv("ITERCC_COMPACT_INTERMEDIATES", "1")
    engine._slot_programs = {}
    np.testing.assert_array_equal(engine.sigma(vector), whole)
    program = next(p for p in engine._slot_programs.values() if p is not None)
    assert program._offset_groups and program.intermediates["size"].sum() < dense


def test_holes_first_rank_three_keeps_the_action(monkeypatch):
    """Dense T3 and R3 stored with the occupied axes outermost (a tile fixes
    two of them): the same values through other strides, the action to
    rounding."""
    monkeypatch.setenv("ITERCC_HOLES_FIRST", "0")
    plain_engine = _split_engine()
    vector = np.random.default_rng(765).normal(size=plain_engine.space.size)
    plain = plain_engine.sigma(vector)
    monkeypatch.setenv("ITERCC_HOLES_FIRST", "1")
    engine = _split_engine()
    t3 = engine.amplitudes[2]
    for signature, block in t3.blocks.items():
        assert block.strides[3] > block.strides[0]            # occupied axes outermost
        np.testing.assert_array_equal(block, plain_engine.amplitudes[2][signature])
    np.testing.assert_allclose(engine.sigma(vector), plain, rtol=1e-12, atol=1e-13 * np.abs(plain).max())


def test_spin_blocked_t3_keeps_the_action(monkeypatch):
    """T3 kept as its nonzero spin blocks (20 of 64): the blocks of the dense
    tensor, the same action bit for bit, under a third of the bytes."""
    from itercc._spin_quadruples_tiles import SpinBlocks, spin_blocks

    monkeypatch.setenv("ITERCC_SPIN_BLOCKED_T3", "0")
    dense_engine = _split_engine()
    vector = np.random.default_rng(773).normal(size=dense_engine.space.size)
    dense = dense_engine.sigma(vector)
    monkeypatch.setenv("ITERCC_SPIN_BLOCKED_T3", "1")
    engine = _split_engine()
    t3 = engine.amplitudes[2]
    assert isinstance(t3, SpinBlocks) and len(t3.blocks) == 20
    assert 3 * t3.nbytes < dense_engine.amplitudes[2].nbytes
    views = spin_blocks(dense_engine.amplitudes[2])
    for signature, view in views.items():
        np.testing.assert_array_equal(t3[signature], view)
    np.testing.assert_array_equal(engine.sigma(vector), dense)
    engine.slot_program = False                      # the Python loop takes the blocks too
    np.testing.assert_allclose(engine.sigma(vector), dense, rtol=1e-12, atol=1e-13 * np.abs(dense).max())


def test_exchangeable_pairs_are_summed_once(monkeypatch):
    """A pair of summed letters that every operand holding one holds both of,
    in one antisymmetric half, is summed once: the (beta, alpha) route is
    dropped, equal spins sum over x < y, and every kept route counts twice.
    The action moves at rounding only."""
    from itercc._spin_quadruples_tiles import exchange_pairs, paired_routes, routes
    from itercc.symmetry_blocks import summed_pairs
    assert exchange_pairs("klcd,abcdijkl->abij", ("goovv", "t4")) == (("k", "l"), ("c", "d"))
    assert exchange_pairs("lmde,ablm,cdeijk->abcijk", ("goovv", "t2", "t3")) == (("l", "m"), ("d", "e"))
    assert exchange_pairs("akci,bcjk->abij", ("gvovo", "t2")) == ()
    assert exchange_pairs("me,abim,cdejkl->abcdijkl", ("fov", "t2", "t3")) == ()
    spec, target = "mnef,bcdfklmn->bcdekl", (0, 0, 0, 0, 0, 1)
    kept = paired_routes(spec, ("goovv", "t4"), target, 1)
    every = routes(spec, target, 1)
    assert 0 < len(kept) < len(every) and all(factor == 2 for _, factor, _ in kept)
    assert summed_pairs("mnef,abmn->abef", (("m", "n", True), ("a", "b", True))) == 1
    engine = _split_engine()
    vector = np.random.default_rng(767).normal(size=engine.space.size)
    monkeypatch.setenv("ITERCC_PAIR_SUMS", "0")
    engine._slot_programs = {}
    both = engine.sigma(vector)
    monkeypatch.setenv("ITERCC_PAIR_SUMS", "1")
    engine._slot_programs = {}
    np.testing.assert_allclose(engine.sigma(vector), both, rtol=1e-12, atol=1e-13 * np.abs(both).max())
    program = next(p for p in engine._slot_programs.values() if p is not None)
    assert any(c[0] not in info[0].split("->")[1] for info in program._contraction_info for c in info[5])


def test_compact_tiles_keep_the_action(monkeypatch):
    """Tiles store only the entries their kind's products form and its
    gathers read (each chain of ordered particle letters one axis of ranks):
    the action of dense tiles bit for bit."""
    engine = _split_engine()
    vector = np.random.default_rng(763).normal(size=engine.space.size)
    monkeypatch.setenv("ITERCC_COMPACT_TILES", "0")
    engine._slot_programs = {}
    dense = engine.sigma(vector)
    monkeypatch.setenv("ITERCC_COMPACT_TILES", "1")
    engine._slot_programs = {}
    np.testing.assert_array_equal(engine.sigma(vector), dense)
    program = next(p for p in engine._slot_programs.values() if p is not None)
    assert program._tile_layouts and program._arrays["tile_sizes"].min() < engine.nvir**4


def test_irrep_ordered_program_arrays_keep_the_action(monkeypatch):
    """Intermediates and tiles with virtual axes in irrep order (positions),
    tiles cleared only where symmetry allows entries: the action of the plain
    layout bit for bit (only addresses change)."""
    from itercc import TripletEOMCCSDTQ
    from itercc._cpu_symmetry_blocks import CpuSymmetryBlocks
    from test_quadruples import warm_amplitudes, symmetric_model
    labels = np.array([0, 1, 2, 3, 0, 2, 1, 3, 0])       # virtual irreps interleaved
    solver, amplitudes = warm_amplitudes(symmetric_model(labels, no=3), evaluations=2, backend="cpu")
    engine = TripletEOMCCSDTQ(solver, amplitudes, labels, 2, occupied_tiles=True, packed_spin=True,
                             intermediate_cache_gib=0.0001)
    if not isinstance(engine._symmetry_engine, CpuSymmetryBlocks):
        pytest.skip("compiled core unavailable")
    vector = np.random.default_rng(759).normal(size=engine.space.size)
    monkeypatch.setenv("ITERCC_SLOT_POSITIONS", "0")
    engine._slot_programs = {}
    plain = engine.sigma(vector)
    program = next(p for p in engine._slot_programs.values() if p is not None)
    assert program._vpos is None and not program._arrays["tile_axes"]["naxes"].any()
    engine.slot_program = False                      # the Python loop: the same action
    np.testing.assert_allclose(engine.sigma(vector), plain, rtol=1e-12, atol=1e-13 * np.abs(plain).max())
    engine.slot_program = True
    monkeypatch.setenv("ITERCC_SLOT_POSITIONS", "1")
    engine._slot_programs = {}
    np.testing.assert_array_equal(engine.sigma(vector), plain)
    program = next(p for p in engine._slot_programs.values() if p is not None)
    assert program._vpos is not None and program._arrays["tile_axes"]["naxes"].all()


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
