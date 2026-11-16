"""Packed CUDA triples directions/images must reproduce the dense projected action."""
import os

import numpy as np
import pytest

from itercc import RestrictedCC
from itercc.eom import EOMSinglet, SingletSpace
from test_restricted import make_model, trial_amplitudes


def symmetric_model(no, nv, labels, leakage):
    """Random Abelian-symmetric Hamiltonian plus small symmetry-forbidden entries."""
    h, eri, _, constant = make_model(no, nv)
    pairs = labels[:, None] ^ labels
    rng = np.random.default_rng(41)
    h = np.where(pairs == 0, h, leakage * rng.normal(size=h.shape))
    h = 0.5 * (h + h.T)
    forbidden = (pairs[:, :, None, None] ^ pairs[None, None]) != 0
    noise = leakage * rng.normal(size=eri.shape)
    noise = noise + noise.transpose(1, 0, 2, 3)
    noise = noise + noise.transpose(0, 1, 3, 2)
    noise = noise + noise.transpose(2, 3, 0, 1)
    eri = np.where(forbidden, noise / 8, eri)
    return h, eri, no, constant


@pytest.mark.parametrize("irrep", [0, 1, 3])
@pytest.mark.parametrize("tiled", [True, False])
def test_packed_triples_action_matches_dense_and_cpu(irrep, tiled):
    if os.environ.get("ITERCC_TEST_CUDA") != "1":
        pytest.skip("CUDA check")
    cp = pytest.importorskip("cupy")
    no, nv = 3, 5
    labels = np.array([0, 1, 3, 0, 1, 2, 3, 2])
    h, eri, no, constant = symmetric_model(no, nv, labels, 1e-10)
    solver = RestrictedCC(h, eri, no, constant)
    # Ground amplitudes keep real off-sector components (no projection).
    amplitudes = trial_amplitudes(solver, .035, True)
    options = dict(orbital_irreps=labels, irrep=irrep, ground_residual_tol=1e3,
                   projected_occupied=True, tangent_feedback=tiled)
    packed = EOMSinglet(solver, amplitudes, backend="cuda", **options)
    dense = EOMSinglet(solver, amplitudes, backend="cuda", packed_triples=False, **options)
    cpu = EOMSinglet(solver, amplitudes, **options)
    assert packed._packed_path(irrep) and not dense._packed_path(irrep)
    rng = np.random.default_rng(97 + irrep)
    vector = rng.normal(size=packed.space.size)
    expected = cpu.sigma(cpu.space.project(vector))
    for engine in (packed, dense):
        actual = engine.sigma(cp.asarray(vector)).get()
        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=0)
    # Packed and dense GPU paths follow the same arithmetic up to summation order.
    np.testing.assert_allclose(packed.sigma(cp.asarray(vector)).get(),
                               dense.sigma(cp.asarray(vector)).get(), atol=1e-13, rtol=0)


@pytest.mark.parametrize("irrep", [0, 2])
def test_packed_projection_matches_dense_projection(irrep):
    if os.environ.get("ITERCC_TEST_CUDA") != "1":
        pytest.skip("CUDA check")
    cp = pytest.importorskip("cupy")
    no, nv = 4, 5
    labels = np.array([0, 2, 3, 2, 0, 1, 2, 2, 3])
    solver = RestrictedCC(*make_model(no, nv))
    space = SingletSpace(solver, labels, irrep, xp=cp)
    cpu = SingletSpace(solver, labels, irrep)
    vector = np.random.default_rng(5).normal(size=space.size)
    packed = space.unpack(cp.asarray(vector), packed=True)
    dense = space.unpack(cp.asarray(vector))
    np.testing.assert_array_equal(packed[2].data.get(), dense[2].reshape(-1)[space.indices[2]].get())
    np.testing.assert_allclose(space.project(cp.asarray(vector)).get(), cpu.project(vector),
                               atol=1e-15, rtol=0)
    # Idempotent and exactly equal to the dense projection on the device.
    projected = space.project(cp.asarray(vector))
    np.testing.assert_array_equal(space.project(projected).get(),
                                  space.pack(space.unpack(projected)).get())
    # Purification of a packed image equals the dense purification, restricted.
    from itercc._cuda_layout import CudaTriplesLayout
    layout = CudaTriplesLayout(no, nv, cp)
    image = cp.asarray(np.random.default_rng(6).normal(size=layout.shape))
    image_packed = space.unpack(cp.zeros(space.size), packed=True)[2]
    image_packed.data[...] = image.reshape(-1)[space.indices[2]]
    layout.purify(image)
    layout.purify(image_packed)
    np.testing.assert_array_equal(image_packed.data.get(), image.reshape(-1)[space.indices[2]].get())
    # Plane gathers from packed storage equal gathers of the dense zero-filled array.
    for k in range(no):
        for axis, spin in ((2, False), (0, True)):
            if spin:
                actual = layout.spin_plane(packed[2], k)
                expected = layout.spin_plane(dense[2], k)
            else:
                actual = layout.plane(packed[2], k, axis=axis)
                expected = layout.plane(dense[2], k, axis=axis)
            np.testing.assert_array_equal(actual.get(), expected.get())
