"""Exercise projected particle-hole tangent products without a CC solve."""
import os

import numpy as np
import pytest

from itercc._triples_layout import TriplesLayout


@pytest.mark.parametrize("irrep", [0, 2])
def test_projected_particle_hole_preserves_target_component(irrep):
    if os.environ.get("ITERCC_TEST_CUDA") != "1":
        pytest.skip("CUDA check")
    import cupy as cp
    from itercc._cuda_projected_ph import CudaProjectedParticleHole

    no, nv = 3, 4
    occupied = np.array([0, 2, 3])
    virtual = np.array([0, 2, 2, 3])
    rng = np.random.default_rng(7349)
    plane = rng.normal(size=(no, no, nv, nv, nv))
    direction = rng.normal(size=plane.shape)
    shape = no * nv, no * nv
    direct, crossed, ddirect, dcrossed = (rng.normal(size=shape) for _ in range(4))
    row = occupied[:, None] ^ virtual
    column = virtual[:, None] ^ virtual
    owner = CudaProjectedParticleHole(occupied, virtual, cp)
    action = owner.prepare_tangent(*(cp.asarray(x) for x in (direct, crossed, ddirect, dcrossed)), irrep)
    for k in range(no):
        sectors = (row[:, None, :, None, None] ^ column ^ occupied[None, :, None, None, None]
                   ^ occupied[k])
        tangent = np.where(sectors == irrep, direction, 0.)
        action.prepare_plane(cp.asarray(plane), cp.asarray(tangent), k)
        for j in range(no):
            a = TriplesLayout.spin_sum(tangent[:, j]).reshape(no*nv, nv**2)
            b = TriplesLayout.spin_sum(plane[:, j]).reshape(no*nv, nv**2)
            d = (direct @ a + ddirect @ b).reshape(no, nv, nv, nv)
            x = (crossed @ tangent[j].reshape(no*nv, nv**2)
                 + dcrossed @ plane[j].reshape(no*nv, nv**2)).reshape(d.shape)
            initial = rng.normal(size=d.shape)
            seed = cp.asarray(initial)
            action.add(seed, j)
            expected = initial + np.where(sectors[:, j] == irrep, d - x.swapaxes(1, 2) - .5*x, 0.)
            np.testing.assert_allclose(seed.get(), expected, atol=5e-14, rtol=0)


def test_fused_particle_hole_fold_matches_two_step_projection():
    if os.environ.get("ITERCC_TEST_CUDA") != "1":
        pytest.skip("CUDA check")
    import cupy as cp
    from itercc._cuda_layout import CudaTriplesLayout
    from itercc._cuda_projected_ph import CudaProjectedParticleHole

    no, nv, irrep = 3, 4, 2
    occupied = np.array([0, 2, 3])
    virtual = np.array([0, 2, 2, 3])
    rng = np.random.default_rng(7349)
    owner = CudaProjectedParticleHole(occupied, virtual, cp)
    layout = CudaTriplesLayout(no, nv, cp)
    matrices = [cp.asarray(rng.normal(size=(no*nv, no*nv))) for _ in range(4)]
    action = owner.prepare_tangent(*matrices, irrep)
    plane = cp.asarray(rng.normal(size=(no, no, nv, nv, nv)))
    labels = (occupied[:, None, None, None, None] ^ occupied[None, :, None, None, None]
              ^ virtual[:, None, None] ^ virtual[None, :, None] ^ virtual)
    previous, fused = cp.zeros(layout.shape), cp.zeros(layout.shape)
    for k in range(no):
        tangent = cp.where(cp.asarray((labels ^ occupied[k]) == irrep), plane, 0.)
        action.prepare_plane(plane, tangent, k)
        for start, width in ((0, 2), (2, 1)):
            seed = cp.asarray(rng.normal(size=(no, width, nv, nv, nv)))
            for line in range(width):
                j = start + line
                expected = seed[:, line].copy()
                action.add(expected, j)
                layout.add_seed(previous, expected, j, k)
                action.fold(fused, seed, line, j, k, layout)
    occ = np.bitwise_xor.reduce(occupied[np.asarray(layout.indices)], axis=1)
    mask = occ[:, None, None, None] ^ virtual[:, None, None] ^ virtual[None, :, None] ^ virtual
    expected = cp.where(cp.asarray(mask == irrep), previous, 0.)
    np.testing.assert_array_equal(fused.get(), expected.get())


def test_streamed_particle_hole_groups_equal_whole_plane():
    if os.environ.get("ITERCC_TEST_CUDA") != "1":
        pytest.skip("CUDA check")
    import cupy as cp
    from itercc._cuda_projected_ph import CudaProjectedParticleHole

    occupied = np.array([0, 2, 2, 3])
    virtual = np.array([0, 2, 2, 3, 3])
    no, nv = len(occupied), len(virtual)
    rng = np.random.default_rng(223)
    owner = CudaProjectedParticleHole(occupied, virtual, cp)
    matrices = [cp.asarray(rng.normal(size=(no*nv, no*nv))) for _ in range(4)]
    whole = owner.prepare_tangent(*matrices, 2)
    streamed = owner.prepare_tangent(*matrices, 2)
    ground, tangent = (cp.asarray(rng.normal(size=(no, no, nv, nv, nv))) for _ in range(2))
    for k in range(no):
        whole.prepare_plane(ground, tangent, k)
        for group in owner.groups[k]:
            streamed.prepare_plane(ground, tangent, k, group=group)
            assert set(streamed.lines) == set(group)
            for j in group:
                seed = cp.asarray(rng.normal(size=(no, nv, nv, nv)))
                actual = seed.copy()
                whole.add(seed, j)
                streamed.add(actual, j)
                np.testing.assert_array_equal(actual.get(), seed.get())
