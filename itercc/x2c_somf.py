"""Shell-blocked derivative integrals for the two-electron X2C spin–orbit mean field."""
import numpy as np


def _shell_ranges(locations, target):
    start = 0
    while start < len(locations)-1:
        end = max(start+1, np.searchsorted(locations, locations[start]+target, side="right")-1)
        yield start, end, slice(locations[start], locations[end])
        start = end


def two_electron_x2c_somf(mol, gamma_ll, gamma_ls, gamma_ss, *, block_mib=64):
    """Contract LL, LS and SS densities in the spherical PySCF X2C AO basis.

    The output follows the convention of get_x2c_hso2e in block2. Slices of
    the first two shell indices are sized for the integral and a single
    Cartesian cross-product component. Individual shells are indivisible.
    The input small/large transition density need not be symmetric.
    """
    loc = mol.ao_loc_nr(cart=False)
    n = int(loc[-1])
    if not np.isfinite(block_mib) or block_mib <= 0:
        raise ValueError("The integral block size must be positive")
    if any(np.shape(d) != (n,n) for d in (gamma_ll,gamma_ls,gamma_ss)):
        raise ValueError("Densities must be square matrices in the spherical X2C AO basis")
    target = max(1, int(np.sqrt(block_mib*2**20/(10*8))/n))
    ranges = list(_shell_ranges(loc,target))
    dtype = np.result_type(gamma_ll,gamma_ls,gamma_ss,np.float64)
    ll,ls,ss = (np.zeros((3,n,n),dtype=dtype) for _ in range(3))
    sym_ll = gamma_ll + gamma_ll.T
    for i0,i1,a in ranges:
        for j0,j1,b in ranges:
            shell = (i0,i1,j0,j1,0,mol.nbas,0,mol.nbas)
            raw = mol.intor("int2e_ip1ip2_sph",comp=9,aosym="s1",shls_slice=shell)
            for axis,(u,v) in enumerate(((1,2),(2,0),(0,1))):
                k = raw[3*u+v] - raw[3*v+u]
                ll[axis,b,:] -= 2*np.einsum("abcd,ac->bd",k,gamma_ss[a,:])
                ls[axis,a,:] -= np.einsum("abcd,bc->ad",k,gamma_ls[b,:])
                ls[axis,b,:] -= np.einsum("abcd,ac->bd",k,gamma_ls[a,:])
                ss[axis,a,b] -= 2*np.einsum("abcd,cd->ab",k,sym_ll)
                ss[axis,a,:] += 2*np.einsum("abcd,bd->ac",k,gamma_ll[b,:])
    return ll,ls,ss
