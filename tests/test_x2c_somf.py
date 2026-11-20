import numpy as np
import pytest

from itercc.x2c_somf import two_electron_x2c_somf


def test_blocked_somf_against_dense_derivative_tensor():
    gto = pytest.importorskip("pyscf.gto")
    mol = gto.M(atom="O 0 0 0; H .8 .3 .3; H -.7 .5 .3",basis="sto-3g",verbose=0)
    n = mol.nao
    # Asymmetric test matrices expose exchanged transition-density indices.
    ll,ls,ss = np.random.default_rng(67).normal(size=(3,n,n))
    raw = mol.intor("int2e_ip1ip2_sph",comp=9).reshape(3,3,n,n,n,n)
    cross = raw[[1,2,0],[2,0,1]]-raw[[2,0,1],[1,2,0]]
    expected = (-2*np.einsum("ipmqn,pq->imn",cross,ss),
                -np.einsum("impqr,pq->imr",cross,ls)-np.einsum("ipmqr,pq->imr",cross,ls),
                -2*(np.einsum("irspq,pq->irs",cross,ll)+np.einsum("irsqp,pq->irs",cross,ll)
                    -np.einsum("irpsq,pq->irs",cross,ll)))
    for mib in (.001,64):
        for got,want in zip(two_electron_x2c_somf(mol,ll,ls,ss,block_mib=mib),expected):
            np.testing.assert_allclose(got,want,rtol=1e-12,atol=1e-12)
