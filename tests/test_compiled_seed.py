"""Check in-place seed assembly against the independent dense formulas."""
import os
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("no,nv,batch", [(2, 3, 1), (3, 4, 2), (4, 2, 4)])
def test_compiled_seed_reuses_workspace_without_changing_values(no, nv, batch):
    if os.environ.get("ITERCC_TEST_CUDA") != "1":
        pytest.skip("CUDA check")
    import cupy as cp
    from itercc._cuda_seed import CudaTriplesSeed

    rng = np.random.default_rng(1295)

    def pair(*shape):
        return SimpleNamespace(value=rng.normal(size=shape), tangent=rng.normal(size=shape))

    t2 = pair(no, no, nv, nv)
    vvvo = pair(nv, nv, nv, no)
    vooo = pair(nv, no, no, no)
    vv = pair(nv, nv)
    occupied, doccupied = rng.normal(size=(2, no**2, no**2))
    plane, tangent = rng.normal(size=(2, no, no, nv, nv, nv))
    device = lambda a: SimpleNamespace(value=cp.asarray(a.value), tangent=cp.asarray(a.tangent))
    with cp.cuda.Stream(non_blocking=True):
        assembly = CudaTriplesSeed(cp, no, nv, batch)
        a, b, u, du = map(cp.asarray, (occupied, doccupied, plane, tangent))
        tensors = tuple(map(device, (t2, vvvo, vooo, vv)))
        for k in range(no):
            doubles = assembly.prepare_doubles(tensors[0], k)
            for start in range(0, no, batch):
                selection = slice(start, min(start+batch, no))
                actual = assembly.assemble(selection, doubles, *tensors[1:], u, du, a, b)
                width = selection.stop-selection.start
                expected = np.einsum("abdj,idc->ijabc", vvvo.tangent[..., selection], t2.value[:, k])
                expected += np.einsum("abdj,idc->ijabc", vvvo.value[..., selection], t2.tangent[:, k])
                expected -= np.einsum("alij,lbc->ijabc", vooo.tangent[..., selection], t2.value[:, k])
                expected -= np.einsum("alij,lbc->ijabc", vooo.value[..., selection], t2.tangent[:, k])
                expected += .5*np.einsum("ad,ijdbc->ijabc", vv.value, tangent[:, selection])
                expected += .5*np.einsum("ad,ijdbc->ijabc", vv.tangent, plane[:, selection])
                shape = no * width, no**2
                expected += (occupied.reshape(no, no, no**2)[:, selection].reshape(shape)
                             @ tangent.reshape(no**2, nv**3)).reshape(expected.shape)
                expected += (doccupied.reshape(no, no, no**2)[:, selection].reshape(shape)
                             @ plane.reshape(no**2, nv**3)).reshape(expected.shape)
                assert actual.data.ptr == assembly.workspace.data.ptr
                np.testing.assert_allclose(actual.get(), expected, atol=2e-13, rtol=0)
