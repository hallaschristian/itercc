"""Verify tiled frozen feedback against the original directional equations."""
import os

import numpy as np
import pytest


@pytest.mark.parametrize("irrep", [0, 2])
def test_tiled_feedback_preserves_target_and_all_source_components(irrep):
    if os.environ.get("ITERCC_TEST_CUDA") != "1":
        pytest.skip("CUDA check")
    import cupy as cp
    from itercc._cuda_eom_feedback import CudaTangentFeedback
    from itercc._cuda_layout import CudaTriplesLayout
    from itercc._directional import Directional, DirectionalBackend, DirectionalCudaLayout
    from itercc._restricted_equations import contract, triples_feedback
    from functools import partial

    occupied, virtual = np.array([0, 2, 3]), np.array([0, 2, 2, 3, 3])
    no, nv = len(occupied), len(virtual)
    rng = np.random.default_rng(2389)
    layout = DirectionalCudaLayout(CudaTriplesLayout(no, nv, cp))
    ad = DirectionalBackend(cp)
    o, v = slice(0, no), slice(no, None)
    shape = (no + nv,)
    f = Directional(*[cp.asarray(rng.normal(size=shape*2)) for _ in range(2)])
    g = Directional(*[cp.asarray(rng.normal(size=shape*4)) for _ in range(2)])
    ground = cp.asarray(rng.normal(size=layout.shape))
    occupied_triples = np.bitwise_xor.reduce(occupied[np.asarray(layout.indices)], axis=1)
    mask = (occupied_triples[:, None, None, None] ^ virtual[:, None, None]
            ^ virtual[None, :, None] ^ virtual) == irrep
    doubles_mask = (occupied[:, None, None, None] ^ occupied[None, :, None, None]
                    ^ virtual[:, None] ^ virtual) == irrep
    owner = CudaTangentFeedback(occupied, virtual, cp)
    for _ in range(2):
        direction = cp.asarray(np.where(mask, rng.normal(size=layout.shape), 0.))
        t3 = Directional(ground, direction)
        sizes = ((no, nv), (no, no, nv, nv), (nv, no, no, no), (nv, nv, nv, no))
        actual = [Directional(cp.zeros(s), cp.zeros(s)) for s in sizes]
        expected = [item.copy() for item in actual]
        triples_feedback(*expected, f, g, layout, t3, o, v, contract=partial(contract, xp=ad), xp=ad)
        owner.apply(*actual, f, g, layout, t3, o, v, irrep=irrep)
        for i, (a, e) in enumerate(zip(actual, expected)):
            at, et = a.tangent.get(), e.tangent.get()
            if i == 1:
                at, et = at[doubles_mask], et[doubles_mask]
            np.testing.assert_allclose(at, et, atol=2e-12, rtol=0)
            if i >= 2:
                np.testing.assert_allclose(a.value.get(), e.value.get(), atol=2e-12, rtol=0)
