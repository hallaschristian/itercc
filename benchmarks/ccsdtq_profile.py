"""Time the phases and kernel classes of RestrictedCCSDTQ residual evaluations.

Every instrumented call synchronizes the device, so the totals attribute
device time to the Python call that issued it. Inclusive times are reported
for the equation phases and exclusive times for the leaf kernels. The
synchronization adds a small overhead; use the plain wall time of an
uninstrumented run for absolute comparisons.

    python -m benchmarks.ccsdtq_profile --synthetic 4 12 out.json
    python -m benchmarks.ccsdtq_profile --input zero.npz --ground ground.npz out.json
    python -m benchmarks.ccsdtq_profile --backend cpu --input zero.npz --symmetry out.json

With `--backend cpu` there is no device to synchronize, the symmetry-block
kernels are not split out, and memory is the host peak RSS; set the threads
of the compiled kernels with ITERCC_CPU_THREADS (they ignore OMP_NUM_THREADS).
"""
import argparse
from collections import Counter, defaultdict
import inspect
import json
import os
from pathlib import Path
import resource
from threading import Event, Thread
from time import perf_counter

import numpy as np


class PhaseTimer:
    def __init__(self, cp):
        self.cp = cp
        self.inclusive = defaultdict(float)
        self.children = defaultdict(float)
        self.calls = Counter()
        self.stack = []

    def wrap(self, label, function, key=None):
        def timed(*args, **kwargs):
            name = label if key is None else f"{label}:{key(args, kwargs)}"
            self.cp.cuda.Device().synchronize()
            start = perf_counter()
            self.stack.append(name)
            try:
                return function(*args, **kwargs)
            finally:
                self.cp.cuda.Device().synchronize()
                elapsed = perf_counter() - start
                self.stack.pop()
                self.inclusive[name] += elapsed
                self.calls[name] += 1
                if self.stack:
                    self.children[self.stack[-1]] += elapsed
        timed.__wrapped__ = function
        return timed

    def report(self):
        rows = {name: dict(calls=self.calls[name], inclusive_seconds=total,
                           exclusive_seconds=total - self.children[name])
                for name, total in self.inclusive.items()}
        return dict(sorted(rows.items(), key=lambda item: -item[1]["exclusive_seconds"]))


def instrument(cp, solver, timer):
    """Wrap the equation phases, the T4 layout and the shared kernels."""
    from itercc import _quadruples_equations as eq

    for name, function in inspect.getmembers(eq, inspect.isfunction):
        if function.__module__ != eq.__name__:
            continue
        setattr(eq, name, timer.wrap(f"phase {name}", function))
    leaves = {"einsum": lambda a, k: a[0]}
    for name in ("einsum", "_unpack_t4_", "_unpack_t4_triples_", "_accumulate_t4_",
                 "t3_spin_summation", "t4_spin_summation"):
        if hasattr(eq, name):
            setattr(eq, name, timer.wrap(name, getattr(eq, name), leaves.get(name)))
    if getattr(solver, "backend", "cuda") == "cuda":
        blocks_engine(timer)
    layout = solver.layout
    for name in ("project", "divide", "update_from_host", "gather", "scatter"):
        if hasattr(layout, name):
            setattr(layout, name, timer.wrap(f"layout.{name}", getattr(layout, name)))
    solver._project_triples = timer.wrap("project_triples", solver._project_triples)
    solver._dressing._dressed = timer.wrap("t1_dressing", solver._dressing._dressed)
    solver.new_quadruples_residual = timer.wrap("new_quadruples_residual",
                                                solver.new_quadruples_residual)
    solver._residuals = timer.wrap("residual evaluation", solver._residuals)


def blocks_engine(timer):
    """Time the symmetry-blocked contractions: whole calls per spec (exclusive
    time is Python and memsets), and their gathers, grouped GEMMs and scatters."""
    from itercc import symmetry_blocks

    if getattr(symmetry_blocks, "_profiled", False):
        return
    symmetry_blocks._profiled = True
    current = {"spec": ""}
    einsum = symmetry_blocks.SymmetryBlocks.einsum

    def tagged(self, spec, *args, **kwargs):
        current["spec"] = spec
        return einsum(self, spec, *args, **kwargs)
    symmetry_blocks.SymmetryBlocks.einsum = timer.wrap("blocks", tagged, lambda a, k: a[1])
    symmetry_blocks._launch = timer.wrap("blocks kernel", symmetry_blocks._launch,
                                         lambda a, k: f"{a[0]}:{current['spec']}")
    grouped = symmetry_blocks._grouped_gemm

    def timed_grouped():
        function = grouped()
        if function is None:
            return None
        return timer.wrap("blocks gemm", function, lambda a, k: current["spec"])
    symmetry_blocks._grouped_gemm = timed_grouped


def synthetic(no, nv, seed, nirrep=4):
    """A real model with Abelian orbital labels and a gapped Fock diagonal."""
    rng = np.random.default_rng(seed)
    n = no + nv
    labels = rng.integers(0, nirrep, size=n)
    allowed = (labels[:, None] ^ labels[None, :]) == 0
    h = np.diag(np.r_[np.linspace(-1.6, -0.8, no), np.linspace(0.25, 1.2, nv)])
    delta = rng.normal(scale=0.03, size=(n, n)) * allowed
    h += delta + delta.T
    factor = rng.normal(scale=0.3 / np.sqrt(n), size=(n + 3, n, n))
    factor = factor + factor.transpose(0, 2, 1)
    channel = rng.integers(0, nirrep, size=n + 3)
    factor *= (labels[:, None] ^ labels[None, :])[None] == channel[:, None, None]
    eri = np.einsum("xpq,xrs->pqrs", factor, factor)
    return h, eri, no, 0.4, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--synthetic", nargs=2, type=int, metavar=("NOCC", "NVIR"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--ground", type=Path)
    parser.add_argument("--block-size", type=int, default=None,
                        help="R4 tile size (default: the solver's, 1 on the CPU, where only 1 runs the compiled tile programs)")
    parser.add_argument("--evaluations", type=int, default=2)
    parser.add_argument("--host-r4", action="store_true")
    parser.add_argument("--symmetry", action="store_true",
                        help="pass the model's orbital_irreps to the solver")
    parser.add_argument("--pool-gib", type=float, default=None)
    parser.add_argument("--damping", type=float, default=0.2)
    parser.add_argument("--diis-size", type=int, default=0)
    parser.add_argument("--no-instrument", action="store_true")
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    from types import SimpleNamespace

    from itercc import RestrictedCCSDTQ
    from itercc._cpu_triples import cpu_threads

    if args.backend == "cuda":
        import cupy as cp
    else:      # no device: synchronization is a no-op for the timer
        cp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda: SimpleNamespace(synchronize=lambda: None)))

    start = perf_counter()
    labels = None
    if args.synthetic:
        h, eri, no, constant, labels = synthetic(*args.synthetic, args.seed)
        initial = None
    else:
        with np.load(args.input, allow_pickle=False) as model:
            h, eri = model["h"], model["eri"]
            no, constant = int(model["nocc"]), float(model["constant"])
            labels = model["orbital_irreps"] if "orbital_irreps" in model else None
        initial = None
        if args.ground is not None:
            with np.load(args.ground, allow_pickle=False) as ground:
                initial = tuple(ground[key] for key in ("t1", "t2", "t3"))
    cuda = args.backend == "cuda"
    if cuda:
        pool = cp.get_default_memory_pool()
        if args.pool_gib:
            pool.set_limit(size=int(args.pool_gib * 2**30))
        device = cp.cuda.Device()
    options = dict(host_r4=args.host_r4, backend=args.backend)
    if args.block_size is not None:
        options["block_size"] = args.block_size
    if args.symmetry:
        options["orbital_irreps"] = labels
    solver = RestrictedCCSDTQ(h, eri, no, constant, **options)
    if initial is not None:
        initial = (*initial[:2], solver.expand_triples(initial[2]))
    timer = PhaseTimer(cp)
    if not args.no_instrument:
        instrument(cp, solver, timer)
    peak = dict(device_bytes=0, pool_used_bytes=0, pool_reserved_bytes=0)
    stop = Event()

    def sample():
        if not cuda:
            return
        with device:
            while not stop.is_set():
                free, total = device.mem_info
                peak["device_bytes"] = max(peak["device_bytes"], total - free)
                peak["pool_used_bytes"] = max(peak["pool_used_bytes"], pool.used_bytes())
                peak["pool_reserved_bytes"] = max(peak["pool_reserved_bytes"], pool.total_bytes())
                stop.wait(0.1)

    report = dict(nocc=solver.nocc, nvir=solver.nvir, block_size=solver.blksize,
                  host_r4=args.host_r4, symmetry=bool(args.symmetry), backend=args.backend,
                  gpu=cp.cuda.runtime.getDeviceProperties(device.id)["name"].decode() if cuda else None,
                  threads=cpu_threads(),
                  job_id=os.environ.get("SLURM_JOB_ID"), evaluations=[])
    last = perf_counter()

    def callback(number, info):
        nonlocal last
        now = perf_counter()
        report["evaluations"].append(dict(evaluation=number, energy_Eh=info.energy,
                                          raw_residual_max_Eh=info.residual_max,
                                          seconds=now - last))
        last = now
        report["profile"] = timer.report()
        report["sampled_peaks"] = dict(peak)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["evaluations"][-1]), flush=True)

    sampler = Thread(target=sample, daemon=True)
    sampler.start()
    try:
        solver.solve(initial=initial, max_iter=args.evaluations, damping=args.damping,
                     diis_size=args.diis_size, callback=callback)
    finally:
        stop.set()
        sampler.join()
        report["wall_seconds"] = perf_counter() - start
        report["profile"] = timer.report()
        report["sampled_peaks"] = dict(peak)
        report["host_peak_rss_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
