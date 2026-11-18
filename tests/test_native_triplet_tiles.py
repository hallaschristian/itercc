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
