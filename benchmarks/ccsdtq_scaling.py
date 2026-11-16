"""Thread scaling of CPU CCSDTQ, phase by phase, in steady state.

Run once per thread count (ITERCC_CPU_THREADS sets the compiled kernels;
they ignore OMP_NUM_THREADS):

    ITERCC_CPU_THREADS=56 python -m benchmarks.ccsdtq_scaling --synthetic 7 29 out.json
    ITERCC_CPU_THREADS=56 python -m benchmarks.ccsdtq_scaling --input zero.npz --ground ccsdt.npz out.json

After warm-up evaluations (plans, recordings), the residual's equation phases,
the T4 layout operations and the preconditioner are timed (exclusive times)
over --evaluations steady-state evaluations; "untimed" is the rest of the
solve loop. A memory-bandwidth probe (y += x over arrays of
--probe-gib, first touched on the same threads) gives the bandwidth the
kernels reach at this thread count.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np


def bandwidth(gib, threads):
    """GB/s of y += x (3 streams) with arrays first touched by the kernels."""
    from itercc import _cpu_quadruples

    n = int(gib * 2**30 / 8)
    x = _cpu_quadruples.permuted_add(np.empty(n), None, np.zeros(n), 0.0, 1.0)
    y = _cpu_quadruples.permuted_add(np.empty(n), None, np.zeros(n), 0.0, 1.0)
    best = float("inf")
    for _ in range(4):
        start = perf_counter()
        _cpu_quadruples.add_(y, x)
        best = min(best, perf_counter() - start)
    return 3 * 8 * n / best / 1e9


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--synthetic", nargs=2, type=int, metavar=("NOCC", "NVIR"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--ground", type=Path)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--evaluations", type=int, default=2)
    parser.add_argument("--probe-gib", type=float, default=2.0)
    args = parser.parse_args()

    from itercc import RestrictedCCSDTQ
    from itercc import _quadruples_equations as eq
    from itercc._cpu_triples import cpu_threads
    from benchmarks.ccsdtq_profile import synthetic

    initial = None
    if args.synthetic:
        h, eri, no, constant, labels = synthetic(*args.synthetic, 137)
    else:
        with np.load(args.input, allow_pickle=False) as model:
            h, eri, no = model["h"], model["eri"], int(model["nocc"])
            constant, labels = float(model["constant"]), model["orbital_irreps"]
    solver = RestrictedCCSDTQ(h, eri, no, constant, orbital_irreps=labels, backend="cpu")
    if args.ground is not None:
        with np.load(args.ground, allow_pickle=False) as ground:
            initial = (ground["t1"], ground["t2"], solver.expand_triples(ground["t3"]))
    from types import SimpleNamespace

    from benchmarks.ccsdtq_profile import PhaseTimer

    # Inclusive and exclusive phase times (phases nest), no device to synchronize.
    timer = PhaseTimer(SimpleNamespace(cuda=SimpleNamespace(
        Device=lambda: SimpleNamespace(synchronize=lambda: None))))
    for name, function in inspect.getmembers(eq, inspect.isfunction):
        if function.__module__ == eq.__name__ and not name.startswith("_"):
            setattr(eq, name, timer.wrap(name, function))
    for name in ("project", "divide", "max_abs", "add_oooo_pairs"):
        if hasattr(solver.layout, name):
            setattr(solver.layout, name, timer.wrap("layout." + name, getattr(solver.layout, name)))
    solver._project_triples = timer.wrap("project_triples", solver._project_triples)
    solver._residuals = timer.wrap("residual", solver._residuals)
    solver._precondition = timer.wrap("precondition", solver._precondition)

    result = solver.solve(initial=initial, max_iter=args.warmup, damping=0.2)
    for table in (timer.inclusive, timer.children, timer.calls):
        table.clear()
    start = perf_counter()
    result = solver.solve(initial=(*result.amplitudes,), max_iter=args.evaluations, damping=0.2)
    wall = perf_counter() - start
    profile = timer.report()
    exclusive = {k: v["exclusive_seconds"] / args.evaluations for k, v in profile.items()}
    timed = sum(v["exclusive_seconds"] for k, v in profile.items())
    report = dict(threads=cpu_threads(), nocc=solver.nocc, nvir=solver.nvir,
                  evaluations=args.evaluations, seconds_per_evaluation=wall / args.evaluations,
                  phases=exclusive, untimed=(wall - timed) / args.evaluations,
                  bandwidth_GBps=bandwidth(args.probe_gib, cpu_threads()),
                  job_id=os.environ.get("SLURM_JOB_ID"))
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in report.items()
                      if k != "phases"}), flush=True)
    print(json.dumps({k: round(v, 3) for k, v in report["phases"].items()}), flush=True)


if __name__ == "__main__":
    main()
