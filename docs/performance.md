# Performance and tuning

## Compiled core

The CPU paths run on `libitercc_core`, a C++17/OpenMP library with its own
GEMM micro-kernels (no BLAS calls on the hot paths). It is found, in order:

1. `ITERCC_CORE_DIR` (one or more directories, `:`-separated);
2. a `native` build in a source checkout's `core/build`;
3. per-level builds installed in the package directory
   (`libitercc_core.x86-64-v4.so`, `...-v3.so`, `...-v2.so`), the best one the
   CPU supports;
4. otherwise the sources are compiled on first use with the system C++
   compiler (`ITERCC_CXX`, else `c++`/`g++`) into `ITERCC_KERNEL_CACHE`
   (default `~/.cache/itercc`), keyed by sources, flags, compiler and CPU
   model, so different node types sharing a home directory keep separate
   builds. Installed packages carry the sources for this (`itercc/_core_src`).

Without a compiler the solvers fall back to NumPy, which is many times
slower, with a warning (`ITERCC_CPU_KERNELS=0` selects that path silently).
A library must report the ABI number the Python side expects, and a `native`
build is skipped on a CPU that lacks one of the instruction-set extensions it
was compiled for.

All builds use `-ffp-contract=off` (and `--fmad=false` for CUDA), so results
are bitwise identical across instruction-set levels. Every reduction uses a
fixed blocking, so CPU results do not depend on the thread count either.

For a cluster, build the portable levels once and install them next to the
package:

```sh
bash core/build_levels.sh                      # x86-64-v3 and v4 into itercc/
ITERCC_CUDA=ON bash core/build_levels.sh       # plus libitercc_cuda.so
```

## Threads and memory placement

- The kernels use one thread per physical core of the process's CPU affinity
  by default. Set `ITERCC_CPU_THREADS` (or `cpu_threads=` on `RestrictedCC`)
  to the cores you were allocated. The kernels do not read
  `OMP_NUM_THREADS`; on SMT machines, hyperthreads make them slower.
- Do not set `OMP_PROC_BIND`: libgomp then binds the main thread to one core
  and the affinity the solver sees shrinks to it.
- On multi-socket nodes, run one calculation per socket with its memory bound
  to it (`numactl --cpunodebind=N --membind=N`). A socket with local memory
  streams faster than the whole node with pages spread over both sockets;
  a second socket gives a single CCSDT or CCSDTQ calculation little or
  nothing.
- The CCSDTQ T4 passes run thread teams concurrently on independent pairs and
  triples of occupied indices (`ITERCC_LOWER_TEAMS`, default
  min(7, threads // 8)); results are identical for every team count.

## Measured

Sapphire Rapids 8480CL, one socket (56 cores, local memory), double precision,
seconds per residual evaluation, 13 correlated occupied orbitals:

| Method | Virtuals | Symmetry | CPU socket | A100 (same code, CUDA) |
| --- | --- | --- | --- | --- |
| CCSDT | 88 | C2v | 2.5 | 7.9 |
| CCSDT | 131 | C2v | 9.2 | 26.6 |
| CCSDTQ | 47 | C2v | 85 | |
| CCSDTQ | 47 | Cs | 227 | |

A singlet EOM-CCSDT or response Jacobian action costs about one to two
ground-state residual evaluations (13o/29v on 12 cores: 0.88 s for a totally
symmetric direction).

The benchmark scripts reproduce such measurements for any molecule:

```sh
python -m benchmarks.prepare_input "O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469" \
    aug-cc-pvtz water.npz --frozen 1 --ground water_ccsdt.npz
ITERCC_CPU_THREADS=56 numactl --cpunodebind=0 --membind=0 \
    python -m benchmarks.ccsdt_backends water.npz --ground water_ccsdt.npz \
    --variant cpu-sym --fixed 3 --threads 56 --cpu-threads 56
ITERCC_CPU_THREADS=56 python -m benchmarks.ccsdtq_profile --backend cpu \
    --input water.npz --ground water_ccsdt.npz profile.json
```

## Environment switches

| Variable | Effect |
| --- | --- |
| `ITERCC_CPU_THREADS` | Kernel threads (default: one per physical core in the affinity mask) |
| `ITERCC_CORE_DIR` | Directories searched first for prebuilt core libraries |
| `ITERCC_KERNEL_CACHE` | Cache directory of first-use builds (default `~/.cache/itercc`) |
| `ITERCC_CXX` | Compiler for first-use builds |
| `ITERCC_CPU_KERNELS=0` | Use the NumPy path (no compiled kernels) |
| `ITERCC_BLOCKED=0` | CCSDT on dense T3 instead of symmetry-blocked T3 |
| `ITERCC_CPU_TANGENT=0` | EOM/response actions on the generic dual-number path |
| `ITERCC_TILE_BLOCKED_T3=0` | CCSDTQ tile programs read dense instead of pair-blocked T3 |
| `ITERCC_TILE_FUSED=0` | CCSDTQ tile programs run one pass per term instead of fused GEMMs |
| `ITERCC_TILE_MEMORY`, `ITERCC_TILE_TEAM` | Memory budget (bytes) and fixed team size of the CCSDTQ tile programs |
| `ITERCC_LOWER_TEAMS` | Concurrent thread teams in the CCSDTQ T4 passes |
| `ITERCC_TEAM_SCHEDULE=rounds` | Run the T4-pass teams in rounds instead of dependency order |
| `ITERCC_DIIS_COMPACT=0` | Float64 instead of compact float32 host T4 DIIS history (CPU) |
| `ITERCC_GEMM_AUTOTUNE=0` | CUDA: static GEMM route choice instead of timing both routes |
| `ITERCC_SYMMETRY_FUSED_DEPTH`, `ITERCC_SYMMETRY_FUSED_SKINNY`, `ITERCC_SYMMETRY_DIRECT_DEPTH` | CUDA: thresholds between the fused symmetry-block kernels and cuBLAS grouped GEMMs |
| `ITERCC_SLAB_MIB`, `ITERCC_BOUNDED_GEMM=1` | CUDA: slab budget and GEMM route of the memory-bounded einsum |

The [architecture notes](architecture.md) record how each path was designed
and measured, including the ideas that were tried and lost.
