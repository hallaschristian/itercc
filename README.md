# IterCC

IterCC is an iterative closed-shell coupled-cluster package for high-accuracy
molecular calculations: full CCSD, CCSDT and CCSDTQ ground states,
EOM-CCSDT singlet excitation energies, and CCSDT linear-response dipole
polarizabilities (static, frequency-dependent and damped). It is built for
large calculations on CPU clusters and GPUs:

- **A compiled CPU core** (C++17/OpenMP, its own GEMM micro-kernels, no BLAS
  on the hot paths) runs every product and element-wise pass on all cores.
  It is built automatically on first use, or ahead of time per instruction-set
  level for clusters.
- **Abelian point-group symmetry** (D2h and subgroups) blocks every product
  of the ground state, the Jacobian and the response, so CCSDT and CCSDTQ
  skip all symmetry-forbidden work.
- **A CUDA backend** (CuPy, cuBLAS and IterCC's kernels) for CCSDT, CCSDTQ,
  EOM and response.
- **Reproducible numbers**: results do not depend on the thread count or the
  instruction set, and CCSDTQ solves resume from saved states bit for bit.

On one 56-core Sapphire Rapids socket, a CCSDT residual evaluation with 13
occupied and 131 virtual orbitals takes 9.2 s (an A100 takes 26.6 s), and a
CCSDTQ evaluation with 13 occupied and 47 virtual orbitals takes 85 s in C2v
symmetry. See [performance](docs/performance.md).

## Features

| Task | Entry point |
| --- | --- |
| CCSD and full CCSDT ground state | `itercc.RestrictedCC` (`rank=2` for CCSD) |
| Full CCSDTQ ground state (CPU or CUDA) | `itercc.RestrictedCCSDTQ` |
| Singlet EOM-EE-CCSDT (or EOM-CCSD) excitation energies | `itercc.EOMSinglet` |
| Static, frequency-dependent and damped CCSDT dipole polarizability | `itercc.DipoleResponse` |
| Spin-orbit state interaction of scalar singlet and triplet states | `itercc.spin_orbit_state_interaction` |
| Experimental triplet (spin-flip) EOM-CCSDTQ | `itercc.TripletEOMCCSDTQ` |
| Independent spin-orbital CCSD/CCSDT reference implementation | `itercc.TensorCC` |
| MO integrals from a PySCF RHF or scalar-X2C RHF calculation | `itercc.pyscf_interface.from_rhf` |

All excitation ranks iterate in full (no perturbative triples or
quadruples). Open-shell references, perturbative CCSD(T), analytic
gradients and orbital relaxation are outside the scope.

## Installation

Python 3.10 or newer on Linux (or WSL) with a C++17 compiler with OpenMP
(GCC 8 or newer):

```sh
git clone <this repository> itercc
cd itercc
python -m pip install .                 # NumPy and SciPy only
python -m pip install ".[pyscf]"        # plus PySCF for integrals and the examples
python -m pip install ".[test]"         # plus pytest and the pinned PySCF reference
```

The compiled core needs no separate step: on first use, IterCC compiles
`core/` with the system compiler (`ITERCC_CXX`, else `c++`) for the running
CPU into `~/.cache/itercc` (about ten seconds) and reuses that build
afterwards. To build it ahead of time instead:

```sh
cmake -S core -B core/build && cmake --build core/build -j    # this machine (native)
bash core/build_levels.sh                                     # portable AVX2 and AVX-512 builds into itercc/
ITERCC_CUDA=ON bash core/build_levels.sh                      # plus the CUDA library
```

A cluster installation with the portable builds serves every node type: each
process loads the best level its CPU supports. Without a compiler the
solvers fall back to NumPy with a warning; they are then many times slower.

**GPU**: install a CuPy build matching your CUDA driver (for example
`pip install cupy-cuda12x`) and check that `cupy.cuda.runtime.getDeviceCount()`
finds the device. The CUDA paths are selected per solver (below).

## Quick start

With PySCF (`examples/water_ccsdt.py` is the complete script):

```python
import numpy as np
from pyscf import gto, scf
from pyscf.scf import hf_symm

from itercc import DipoleResponse, EOMSinglet, RestrictedCC
from itercc.pyscf_interface import from_rhf

mol = gto.M(atom="O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469",
            basis="cc-pvdz", symmetry=True)
mf = scf.RHF(mol).run(conv_tol=1e-12)
model = from_rhf(mf, frozen=1)                                  # freeze O 1s
labels = np.asarray(hf_symm.get_orbsym(mol, mf.mo_coeff))[1:]   # C2v irreps of the correlated MOs

cc = RestrictedCC(model.h, model.eri, model.nocc, model.constant,
                  orbital_irreps=labels, symmetry_tolerance=1e-8)
ground = cc.solve(energy_tol=1e-12, residual_tol=1e-10)
print(ground.energy)                                            # -76.2411862596 Eh

eom = EOMSinglet(cc, ground.amplitudes, orbital_irreps=labels, irrep=3)
print(eom.solve(nroots=2).excitation_energies)                  # B2 roots, hartree
```

Without PySCF, any real Hamiltonian works (`examples/minimal.py`):

```python
import numpy as np
from itercc import RestrictedCC

h = np.diag([-1.0, 0.5])            # one occupied, one virtual spatial orbital
eri = np.zeros((2, 2, 2, 2))        # (pq|rs)
result = RestrictedCC(h, eri, nocc=1).solve()
print(result.energy, result.converged)
```

## Conventions

- Orbitals are orthonormal real spatial MOs with the **correlated occupied
  orbitals first**; `nocc` counts correlated occupied spatial orbitals.
- `h[p, q]` is the one-electron Hamiltonian and `eri[p, q, r, s] = (pq|rs)`
  in chemists' notation. `constant` holds the nuclear repulsion plus the
  frozen-core energy (`from_rhf` returns all four).
- `orbital_irreps` are Abelian labels 0..7 (XOR products; PySCF's D2h-subgroup
  irrep ids), one per correlated orbital. With `symmetry_tolerance > 0`
  (for example 1e-8), integrals whose symmetry-forbidden entries stay within
  it are projected, so the symmetry-blocked kernels apply.
- Amplitudes: T1 `[i, a]`, T2 `[i, j, a, b]`, T3 `[slot, a, b, c]` over the
  occupied triples `cc.triples_indices`. Energies are in hartree,
  polarizabilities in atomic units.
- A result counts only when its `converged` flag is set: ground states test
  both the energy change and the raw residual of every rank, and EOM roots
  are accepted only after a fresh full-space residual check.

## Methods

**CCSD and CCSDT** (`RestrictedCC`). DIIS extrapolates T1/T2 by default;
`diis_triples=True` includes T3. On the CPU with orbital labels, T3 is stored
and contracted in symmetry blocks. On a GPU pass `residual_backend="cuda"`
together with the fused CUDA paths:

```python
cc = RestrictedCC(h, eri, nocc, constant, residual_backend="cuda",
                  cuda_pair_fusion=True, cuda_layout_fusion=True, cuda_crossed_fusion=True,
                  cuda_pool_cache=True)
```

**CCSDTQ** (`RestrictedCCSDTQ`, `backend="cpu"` or `"cuda"`): packed T4 over
ordered occupied quadruples, compiled tile programs on the CPU, optional T4
DIIS with a compact host history, and resumable saved states. See
[CCSDTQ](docs/ccsdtq.md) and `examples/ccsdtq.py`.

**EOM-CCSDT** (`EOMSinglet`): exact analytic Jacobian actions (product rules
on the residual equations), a restarted non-Hermitian Davidson solver, one
Abelian sector per solve. See [EOM](docs/eom.md).

**Linear response** (`DipoleResponse`): CCSDT Lagrangian response with
shifted GMRES solves; converged solutions and their Jacobian images are
reused across wavelengths. See [response](docs/response.md).

**Spin-orbit state interaction**: first-order SOC between scalar singlet and
triplet states from supplied transition densities; a PySCF adapter builds
the X2C SOC operator. See [spin-orbit](docs/spin-orbit.md).

## Performance

Run one calculation per CPU socket with its memory bound to it
(`numactl --cpunodebind=N --membind=N`), and set `ITERCC_CPU_THREADS` to the
cores you were allocated (the kernels ignore `OMP_NUM_THREADS`).
[Performance](docs/performance.md) lists measured timings, the core's build
and loading rules and every tuning switch; the [architecture
notes](docs/architecture.md) explain the design and record what was tried.

## Examples and benchmarks

| Script | What it does |
| --- | --- |
| `examples/minimal.py` | CCSDT, one EOM root and a polarizability of a two-orbital model |
| `examples/water_ccsdt.py` | Water: CCSDT, EOM-CCSDT in two symmetry sectors, static and 532 nm polarizability |
| `examples/ccsdtq.py` | Water: CCSDT, then CCSDTQ (`--cuda` for the GPU) |
| `benchmarks/prepare_input.py` | Hamiltonian archive (and CCSDT amplitudes) for the benchmarks, from PySCF |
| `benchmarks/ccsdt_backends.py` | CCSDT residual timings on the CPU and GPU |
| `benchmarks/ccsdtq_profile.py` | CCSDTQ evaluations profiled by phase and kernel |
| `benchmarks/ccsdtq_scaling.py` | CCSDTQ thread scaling, phase by phase |

## Tests

```sh
python -m pip install ".[test]"
python -m pytest -q                     # CPU
ITERCC_TEST_CUDA=1 python -m pytest -q  # also the CUDA tests
```

The tests compare residuals and energies against PySCF's RCCSDT and
RCCSDTQ, exact diagonalization and an independent spin-orbital
implementation; EOM and response against determinant-space references; and
the compiled kernels against the NumPy paths, bit for bit where the
arithmetic is the same (thread counts, instruction-set levels, team
schedules).

## Repository layout

```
itercc/       Python package (solvers, EOM, response, layouts, GPU kernels)
core/         C++/OpenMP CPU core and CUDA library (CMake), loaded through a C ABI
docs/         method guides, performance and architecture notes
examples/     runnable examples (PySCF for the molecular ones)
benchmarks/   timing and profiling scripts
tests/        pytest suite (tests/oracle: determinant-space reference)
tools/        generators of the equation modules from pdaggerq output
```

## License

IterCC is licensed under the Apache License 2.0 ([LICENSE](LICENSE)). Parts
of the equations are adapted from [pdaggerq](https://github.com/edeprince3/pdaggerq)
and [PySCF](https://github.com/pyscf/pyscf), both Apache-2.0; see
[THIRD_PARTY.md](THIRD_PARTY.md), `LICENSE.pdaggerq`, `LICENSE.pyscf` and
`NOTICE.pyscf`.
