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
