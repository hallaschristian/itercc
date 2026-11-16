# Ground-state CCSDTQ

`RestrictedCCSDTQ(h, eri, nocc, constant=0.0)` takes the same real spatial
MO Hamiltonian as `RestrictedCC`: chemist-order integrals, occupied orbitals
first, `constant` with nuclear repulsion and the frozen-core energy.
`solve()` returns the total energy, the projected raw residual maxima of all
four ranks and the four amplitudes. Convergence requires both the energy
change and all four residuals to meet their tolerances.

```python
from itercc import RestrictedCC, RestrictedCCSDTQ

cc = RestrictedCC(h, eri, nocc, constant, orbital_irreps=labels, symmetry_tolerance=1e-8)
ccsdt = cc.solve(energy_tol=1e-12, residual_tol=1e-10)

solver = RestrictedCCSDTQ(h, eri, nocc, constant, orbital_irreps=labels, backend="cpu")
t1, t2, t3 = ccsdt.amplitudes
result = solver.solve(initial=(t1, t2, solver.expand_triples(t3)),   # T4 starts at zero
                      energy_tol=1e-10, residual_tol=1e-7, damping=0.2,
                      diis_size=8, diis_quadruples="host")
print(result.energy, result.converged)
```

## Backends

`backend="cpu"` keeps every array in host memory and runs the T4 layout,
spin summations and contractions in the compiled core: OpenMP kernels, an
indirect GEMM executor that reads operands in place through offset tables,
and compiled tile programs that evaluate the R4 terms slot by slot. With
`orbital_irreps` every product is symmetry-blocked. Without the compiled core
it runs the same equations in NumPy (much slower).

`backend="cuda"` (the default) uses CuPy, cuBLAS and IterCC's CUDA kernels.
Two-operand tile products run either as direct GEMMs into strided views or
through einsum, whichever is faster for each shape on the first call
(`ITERCC_GEMM_AUTOTUNE=0` uses a static estimate instead). When T4 and the
contraction tiles fit on the GPU but its residual does not, use
`RestrictedCCSDTQ(..., block_size=2, host_r4=True)`: R4 stays in host RAM and
occupied slices are staged through the GPU. `solver.device_bytes(...)`
estimates the peak device memory of a solve.

`precision="single"` (CPU only) lets the memory-bound R4 passes read float32
copies of T3 and the spin-summed T4 rows; every product and sum is formed in
double and T1-T4 stay double. It is about 3% faster, moves converged energies
by about 1e-11 Eh, and puts residual tolerances below about 1e-9 out of reach.

## Storage and iteration

T3 is expanded. T4 stores the lexicographically ordered occupied tuples
i <= j <= k <= l, each with its symmetry-allowed virtual quadruples (four
dense virtual axes without labels). DIIS extrapolates T1-T3 by default, and
T4 takes damped Jacobi steps. Systems with a stiff T4 mode (diffuse basis
sets: a T4 residual that grows while the energy settles) need T4 in DIIS:

- `diis_quadruples=True` keeps the history on the device (two packed T4 per
  entry);
- `diis_quadruples="host"` keeps it in host memory. On the CPU backend the
  history is compact: each entry is one float32 packed T4 holding the trial
  minus the current amplitudes, which a step re-bases in one pass, so the
  fixed point and convergence are those of the float64 history
  (`ITERCC_DIIS_COMPACT=0` selects the float64 history).

On the development system (13 occupied, 29 virtuals) DIIS 10 with damping
0.2 converged in 38 evaluations against 50 for DIIS 6.

## Checkpoints

`solve(checkpoint=callback)` passes a `CCSDTQState` to the callback after
every update. `solver.save_state(path, state)` writes it atomically (with a
`checkpoint_io.BackgroundArrayWriter`, while the solve continues), and
`solver.load_state(path)` reads it back for this Hamiltonian and layout;
`solve(resume=state)` then continues the iteration exactly, including the
DIIS history. On CUDA the state carries the autotuned GEMM routes, so a
resume in a new process on the same GPU model rounds as the saved run would
have. States resume on either backend.

## Validation

`tests/test_quadruples.py` checks all four residual ranks against PySCF
RCCSDTQ and converged energies against four-electron FCI and six-electron
RCCSDTQ, for both backends, with and without orbital labels; compiled and
Python tile loops, thread counts and team schedules are compared bit for bit
where the arithmetic is the same.

## Performance

One Sapphire Rapids socket (56 cores, its memory bound with `numactl`),
13 occupied and 47 virtual orbitals, seconds per residual evaluation:

| Symmetry | Seconds per evaluation |
| --- | --- |
| C2v (4 irreps) | 85 |
| Cs (2 irreps) | 227 |

Run one point per socket: a socket with local memory streams faster than a
whole two-socket node with pages on both, so two independent calculations
per node nearly double the throughput. See [performance](performance.md).
