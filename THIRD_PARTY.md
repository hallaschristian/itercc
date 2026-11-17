# Third-party source

`itercc/_equations.py` adapts tensor contractions from the [pdaggerq CCSDT example](https://github.com/edeprince3/pdaggerq/blob/7676be740eebeeba9735da8bcd9c3715aa40159b/examples/full_cc_codes/ccsdt.py) at commit `7676be740eebeeba9735da8bcd9c3715aa40159b`. The pdaggerq project is by the DePrince group and contributors. Its Apache License 2.0 is included in `LICENSE.pdaggerq`.

Changes include extracting the T1-free terms for use with an exactly T1-dressed Hamiltonian, using transposed views for permutations, adding pair-ladder and array-backend hooks, and factoring the triples diagrams before final antisymmetrization. The dressing, spin-integral preparation, solver, and benchmarks live in separate modules.

To regenerate the equations, obtain the pinned source example, run `python tools/generate_equations.py upstream_ccsdt.py itercc/_equations.py`, then run `python tools/factor_triples.py itercc/_equations.py itercc/_equations.py`.

`itercc/_spin_quadruples_diagrams.py` derives its spin-orbital contractions
from the [CCSDTQ example at the same pdaggerq revision](https://github.com/edeprince3/pdaggerq/blob/7676be740eebeeba9735da8bcd9c3715aa40159b/examples/full_cc_codes/ccsdtq.py).
Regenerate with `python tools/generate_spin_quadruples.py upstream_ccsdtq.py itercc/_spin_quadruples_diagrams.py`.
The generator removes explicit T1 terms for exact one-body dressing and
combines antisymmetry-equivalent contractions before differentiation.

`tests/oracle` holds the small determinant-space CC prototype IterCC grew from, kept unchanged as an independent reference.

The spatial RCCSDT contraction expressions in `itercc/_restricted_equations.py` and
`itercc/restricted.py` adapt the Apache-2.0 PySCF 2.14.0 equations by Yu Jin and
Huanchen Zhai from [rccsdt.py](https://github.com/pyscf/pyscf/blob/v2.14.0/pyscf/cc/rccsdt.py)
and [rccsdt_highm.py](https://github.com/pyscf/pyscf/blob/v2.14.0/pyscf/cc/rccsdt_highm.py).
The accompanying `LICENSE.pyscf` and `NOTICE.pyscf` retain PySCF's license and notices.
IterCC implements its own triangular storage, plane contractions, virtual pair
ladder, exact T1 dressing, Jacobi iteration, and DIIS. The restricted solver
requires only NumPy. Validation against the separate spin-orbital implementation
checks the restricted equations and the packing schedule independently.

The CCSDTQ contractions in `itercc/_quadruples_equations.py` also
adapt the PySCF 2.14.0 [rccsdtq.py](https://github.com/pyscf/pyscf/blob/v2.14.0/pyscf/cc/rccsdtq.py)
equations by Yu Jin and Huanchen Zhai. IterCC evaluates them with its own
packed T4 layout, occupied-tile gather and scatter, spin sums, projectors and
iteration, on CuPy/CUDA or on the compiled CPU core (`core/`). The spin permutation coefficients were checked against the
upstream kernels. The installed solver does not import PySCF.

`itercc/basis_sets` holds Gaussian basis sets in NWChem format, copied unchanged
from the PySCF 2.14.0 distribution (`pyscf/gto/basis`, Apache License 2.0, see
`LICENSE.pyscf` and `NOTICE.pyscf`), which took them from the Basis Set
Exchange (https://www.basissetexchange.org). Each file's header lists the
original publication of every element's basis; please cite those, and
B. P. Pritchard, D. Altarawy, B. Didier, T. D. Gibson and T. L. Windus,
J. Chem. Inf. Model. 59, 4814 (2019), for the Basis Set Exchange.
