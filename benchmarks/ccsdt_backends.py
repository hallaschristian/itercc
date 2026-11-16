"""Time restricted CCSDT on the CPU and the GPU for one Hamiltonian archive
(`benchmarks.prepare_input`).

Every variant solves the same symmetry-projected Hamiltonian. `--fixed N`
times N residual evaluations at the supplied (projected) amplitudes after
one warm-up evaluation; `--solve` runs a full solve with tight production
controls from the MP2 start. `--profile` prints the costliest calls of one CPU evaluation.
Run each variant in its own process so the memory peaks are separate:

    python -m benchmarks.ccsdt_backends zero.npz --ground ground.npz \\
        --variant cpu-sym --threads 12 --fixed 3 --solve
"""

import argparse
import cProfile
import json
import pstats
import resource
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from itercc import RestrictedCC
from itercc.spatial_symmetry import project_amplitudes, project_integrals

# The production settings of the CUDA CCSDT solver (all fused CUDA paths on).
PRODUCTION_GPU = dict(residual_backend="cuda", ladder_batch=16,
                      cuda_pair_fusion=True, cuda_layout_fusion=True, cuda_ph_columns=True,
                      cuda_ladder_columns=True, cuda_occupied_columns=True, cuda_t1_dressing=True,
                      cuda_jacobi=True, cuda_crossed_fusion=True, cuda_feedback_columns=True,
                      cuda_assembly_columns=True, cuda_ph_fusion=True, cuda_pool_cache=True,
                      host_diis_columns=True)
# name: (pass orbital symmetry labels, solver options). The "-reference"
# variants use the generic per-(j, k) triples loop instead of _cpu_triples.
VARIANTS = {
    "cpu": (False, dict(ladder_batch=4)),
    "cpu-sym": (True, dict(ladder_batch=4)),
    "cpu-reference": (False, dict(ladder_batch=4)),
    "cpu-sym-reference": (True, dict(ladder_batch=4)),
    "gpu-dense": (False, dict(residual_backend="cuda", ladder_batch=4, cuda_pair_fusion=True,
                              cuda_layout_fusion=True, cuda_crossed_fusion=True,
                              cuda_pool_cache=True)),
    "gpu": (True, PRODUCTION_GPU),
}
CONTROLS = dict(energy_tol=1e-12, residual_tol=1e-10, diis_size=3, diis_triples=True)


def maxima(residuals):
    return [float(np.max(np.abs(r), initial=0.0)) for r in residuals]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="zero-field MO archive (h, eri, nocc, ...)")
    parser.add_argument("--ground", type=Path, help="CCSDT amplitudes for --fixed and --profile")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--threads", type=int, default=4, help="BLAS threads")
    parser.add_argument("--cpu-threads", type=int, default=None,
                        help="element-wise CPU workers (default: the solver's default)")
    parser.add_argument("--rank", type=int, choices=(2, 3), default=3,
                        help="2 for restricted CCSD (a ground T3 is ignored), 3 for CCSDT")
    parser.add_argument("--fixed", type=int, default=0, metavar="N")
    parser.add_argument("--solve", action="store_true")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--save-residual", type=Path)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as model:
        h, eri = model["h"], model["eri"]
        nocc, constant = int(model["nocc"]), float(model["constant"])
        labels = model["orbital_irreps"]
    h, eri, _ = project_integrals(h, eri, labels, tolerance=2e-9)
    symmetric, options = VARIANTS[args.variant]
    cuda = options.get("residual_backend") == "cuda"
    if not cuda:
        options = dict(options, cpu_threads=args.cpu_threads)
    row = dict(variant=args.variant, rank=args.rank, threads=args.threads, nocc=nocc,
               nvir=len(h) - nocc)
    if args.rank == 2:
        if cuda:
            # CCSD has no triples layout or compressed triples contractions.
            options = {key: options[key] for key in (
                "residual_backend", "ladder_batch", "cuda_pool_cache", "cuda_t1_dressing")
                       if key in options}
        options = dict(options, rank=2)
    with threadpool_limits(limits=args.threads, user_api="blas"):
        start = perf_counter()
        solver = RestrictedCC(h, eri, nocc, constant,
                              orbital_irreps=labels if symmetric else None, **options)
        if args.variant.endswith("-reference"):
            solver._cpu_fast = False
        elif not cuda:
            from itercc._cpu_kernels import library
            from itercc._cpu_triples import cpu_threads
            row.update(cpu_threads=cpu_threads(args.cpu_threads),
                       cpu_kernels=library() is not None)
        row["setup_seconds"] = perf_counter() - start
        if cuda:
            import cupy as cp

            row["gpu"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()

        if args.fixed or args.profile:
            with np.load(args.ground, allow_pickle=False) as ground:
                amplitudes = tuple(ground[k] for k in ("t1", "t2", "t3")[:args.rank])
            amplitudes, _ = project_amplitudes(amplitudes, labels, tolerance=1e-8)
            if args.rank == 2:
                amplitudes = (*amplitudes, np.zeros(solver.layout.shape))
            start = perf_counter()
            residuals = solver.residuals(amplitudes)
            row["first_evaluation_seconds"] = perf_counter() - start
            # Timed as in solve: without the input checks of `residuals`
            # (about 0.6 s at 88 virtuals).
            amplitudes = solver._validated(amplitudes)
            times = []
            for _ in range(args.fixed):
                start = perf_counter()
                residuals = solver._residuals(amplitudes)
                times.append(perf_counter() - start)
            row.update(evaluation_seconds=times,
                       median_evaluation_seconds=float(np.median(times)) if times else None,
                       residual_max=maxima(residuals))
            if args.save_residual:
                np.savez(args.save_residual, r1=residuals[0], r2=residuals[1], r3=residuals[2])
            if args.profile:
                profile = cProfile.Profile()
                profile.enable()
                solver._residuals(amplitudes)
                profile.disable()
                pstats.Stats(profile).sort_stats("tottime").print_stats(25)
            del residuals

        if args.solve:
            stamps = []
            start = perf_counter()
            result = solver.solve(max_iter=args.max_iter, **CONTROLS,
                                  callback=lambda info: stamps.append(perf_counter() - start))
            wall = perf_counter() - start
            row.update(solve_seconds=wall, iterations=len(result.iterations),
                       converged=bool(result.converged), total_Eh=result.energy,
                       final_residual_max=list(result.iterations[-1].residual_max),
                       seconds_per_iteration=wall / len(result.iterations),
                       steady_seconds_per_iteration=float(np.median(np.diff(stamps)))
                       if len(stamps) > 2 else None)
        if cuda:
            row["device_pool_GiB"] = solver.device_pool_reserved_bytes / 2**30
    row["peak_host_rss_GiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
