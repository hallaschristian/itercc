"""Check the projected CUDA ladder against the dense ladder for arbitrary leakage."""
import os

import numpy as np
import pytest

from itercc._triples_layout import TriplesLayout
from itercc.restricted import _pair_ladder


@pytest.mark.parametrize("target,source_sector", [(0, 0), (2, 2), (2, 0), (2, None)])
def test_projected_virtual_ladder_preserves_target_component(target, source_sector):
    if os.environ.get("ITERCC_TEST_CUDA") != "1":
        pytest.skip("CUDA check")
    import cupy as cp
    from itercc._cuda_projected_ladder import CudaProjectedLadder

    no, nv = 3, 4
    layout = TriplesLayout(no, nv)
    labels = np.array([0, 2, 3, 0, 2, 2, 3])
    vir = labels[no:]
    occupied = np.bitwise_xor.reduce(labels[np.asarray(layout.indices)], axis=1)
    virtual = vir[:, None, None] ^ vir[None, :, None] ^ vir[None, None, :]
    sectors = occupied[:, None, None, None] ^ virtual
    rng = np.random.default_rng(789)
    source = rng.normal(size=layout.shape)
    if source_sector is not None:
        source[sectors != source_sector] = 0.
    w = rng.normal(size=(nv,) * 4)
    w = .5 * (w + w.transpose(1, 0, 3, 2))
    source, w = cp.asarray(source), cp.asarray(w)
    dense, projected = cp.zeros(layout.shape), cp.zeros(layout.shape)
    _pair_ladder(dense, source, w, 2, xp=cp, fuse_cuda=True)
    operator = CudaProjectedLadder(layout, labels, cp)
    operator.add(projected, source, w, target, 2, source_irrep=source_sector)
    expected = cp.where(cp.asarray(sectors == target), dense, 0.)
    np.testing.assert_allclose(projected.get(), expected.get(), atol=5e-14, rtol=0)
