"""Time singlet EOM-CCSDTQ Jacobian actions (`FiniteDifferenceEOM`) by variant.

Every variant applies the same Jacobian of one symmetry sector to the same
vector; the first variant is the reference the others are compared with:

- `dense-central`: a solver without orbital labels (C1 GEMMs on dense T4) and
  the central difference (two residuals per action);
- `dense-one-sided`: the same solver, one residual per action (nonzero irrep);
- `sector`: a solver labelled with `spatial_symmetry.sector_labels` (the full
  group for irrep 0, the half-order subgroup otherwise) and the engine's
  defaults (one residual per action, forming only the target sector's
  residual entries, for a nonzero irrep; the central difference for irrep 0);
- `sector-every`: as `sector`, with every allowed residual entry formed
  (ITERCC_EOM_SECTOR_OUTPUTS=0);
- `sector-single`: as `sector` on a `precision="single"` solver (float32
  copies of T3 and of the spin-summed T4 rows; one-sided actions only);
- `sector-central`: as `sector`, with the central difference.

A synthetic C2v model takes its ground amplitudes from a few iterations of
the solver without labels; an archive takes a saved CCSDTQ state (a
directory with t1..t4 and the `layout` signature in its metadata), re-laid
into each solver's layout by `combine_states`:

    python -m benchmarks.singlet_eom_ccsdtq --synthetic 5 36 --irrep 2 --threads 12
    python -m benchmarks.singlet_eom_ccsdtq --input zero.npz --state final_0 --irrep 2 \\
        --variants sector --threads 56 --memory-trace --output b1.jsonl

The dense variants need about four times the memory of a C2v sector one; run
large cases one variant per process (the references then come from the
`norm` and `rayleigh_Eh` fields). `--phases` adds each action's seconds in the
residual's phases (R4 tile programs, T4 into ranks two and three, ring and
oooo terms, the R4 projection).
"""
import argparse
import json
import os
import resource
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

VARIANTS = ("dense-central", "dense-one-sided", "sector", "sector-every", "sector-single", "sector-central")


def _emit(output):
    def emit(**row):
        row.setdefault("peak_RSS_GiB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2))
        line = json.dumps(row)
        print(line, flush=True)
        if output is not None:
            with output.open("a") as stream:
                stream.write(line + "\n")
    return emit


def _phase_timers():
    """Seconds by residual phase, accumulated into the returned dict."""
    from itercc import _cpu_quadruples, _quadruples_equations

    phases = defaultdict(float)

    def wrap(owner, name, label):
        original = getattr(owner, name)

        def timed(*a, **k):
            start = time.perf_counter()
            try:
                return original(*a, **k)
            finally:
                phases[label] += time.perf_counter() - start
        setattr(owner, name, timed)

    for name, label in (("t4_into_lower_ranks_", "lower"), ("r4_add_rings_triples_", "rings"),
                        ("r4_add_rings_tiles_", "rings"), ("r4_add_oooo_pairs_", "oooo")):
        wrap(_quadruples_equations, name, label)
    wrap(_cpu_quadruples.CompiledQuadruplesLayout, "project", "project")
    return phases


def _hamiltonian(args):
    """(h, eri, nocc, constant, labels, ground source (amplitudes, signature) or None)."""
    from itercc.spatial_symmetry import project_integrals

    if args.synthetic:
        from .ccsdtq_profile import synthetic

        h, eri, no, constant, labels = synthetic(*args.synthetic, args.seed)
        return h, eri, no, constant, np.asarray(labels, dtype=np.int64), None
    with np.load(args.input) as model:
        labels = np.asarray(model["orbital_irreps"], dtype=np.int64)
        no, constant = int(model["nocc"]), float(model["constant"])
        h, eri, _ = project_integrals(model["h"], model["eri"], labels, tolerance=args.symmetry_tolerance)
    source = None
    if args.state is not None:
        from itercc.checkpoint_io import read_array_directory

        arrays, meta = read_array_directory(args.state, lazy=True)
        source = (tuple(arrays[f"t{rank}"] for rank in range(1, 5)), meta["layout"])
    return h, eri, no, constant, labels, source


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic", nargs=2, type=int, metavar=("NOCC", "NVIR"))
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--input", type=Path, help="Hamiltonian archive (h, eri, nocc, constant, orbital_irreps)")
    parser.add_argument("--state", type=Path, help="saved CCSDTQ state directory (t1..t4, layout signature)")
    parser.add_argument("--symmetry-tolerance", type=float, default=1e-9)
    parser.add_argument("--irrep", type=int, default=2)
    parser.add_argument("--variants", default=None,
                        help=f"comma-separated, from {', '.join(VARIANTS)} (default: all that apply)")
    parser.add_argument("--actions", type=int, default=2, help="actions per variant (the first plans)")
    parser.add_argument("--ground-iterations", type=int, default=3,
                        help="iterations of the synthetic ground state (from MP2)")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--memory-trace", action="store_true",
                        help="resident memory by phase (benchmarks._memory.MemoryTrace)")
    parser.add_argument("--phases", action="store_true",
                        help="each action's seconds in the residual's phases")
    parser.add_argument("--output", type=Path, help="also append the JSON records here")
    args = parser.parse_args()
    if bool(args.synthetic) == bool(args.input):
        parser.error("Give --synthetic or --input")
    if args.input and args.state is None:
        parser.error("An archive needs --state (a saved CCSDTQ ground state)")

    from itercc import RestrictedCCSDTQ, _cpu_tile_program
    from itercc._runtime import thread_scope
    from itercc.eom_ccsdtq import FiniteDifferenceEOM
    from itercc.spatial_symmetry import sector_labels

    emit = _emit(args.output)
    variants = (args.variants.split(",") if args.variants else
                [v for v in VARIANTS if args.irrep or v in ("dense-central", "sector")])
    if not args.irrep and "sector-single" in variants:
        parser.error("sector-single needs a nonzero irrep (one-sided actions)")
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        parser.error(f"Unknown variants {sorted(unknown)}")
    trace = None
    if args.memory_trace:
        from ._memory import MemoryTrace

        trace = MemoryTrace(emit)
    tile_seconds = []
    original = _cpu_tile_program.TileProgram.run

    def timed(self, *a, **k):
        start = time.perf_counter()
        try:
            return original(self, *a, **k)
        finally:
            tile_seconds.append(time.perf_counter() - start)
    _cpu_tile_program.TileProgram.run = timed
    phases = _phase_timers() if args.phases else None

    h, eri, no, constant, labels, source = _hamiltonian(args)
    options = dict(backend="cpu", block_size=1, cpu_threads=args.threads)
    reference = None
    with thread_scope(args.threads):
        if source is None:
            start = time.perf_counter()
            dense = RestrictedCCSDTQ(h, eri, no, constant, **options)
            source = (dense.solve(max_iter=args.ground_iterations, damping=0.2, diis_size=0).amplitudes, None)
            emit(event="ground", iterations=args.ground_iterations, seconds=round(time.perf_counter() - start, 2))
        emit(event="model", nocc=no, nvir=len(h) - no, irrep=args.irrep, threads=args.threads,
             labels=np.bincount(labels, minlength=4).tolist())
        for variant in variants:
            sector = variant.startswith("sector")
            start = time.perf_counter()
            solver = RestrictedCCSDTQ(h, eri, no, constant, **options,
                                      precision="single" if variant == "sector-single" else "double",
                                      orbital_irreps=sector_labels(labels, args.irrep) if sector else None)
            ground = solver.combine_states([(1.0, *source)])
            one_sided = {"dense-central": False, "dense-one-sided": True, "sector-central": False}.get(variant)
            engine = FiniteDifferenceEOM(solver, ground, labels, args.irrep, one_sided=one_sided)
            del ground
            emit(event="engine", variant=variant, seconds=round(time.perf_counter() - start, 2),
                 nirrep=int(solver.layout.nirrep), t4_GiB=round(solver.layout.size * 8 / 2**30, 3),
                 size=int(engine.space.size), one_sided=engine.one_sided)
            vector = np.random.default_rng(args.seed).standard_normal(engine.space.size)
            vector /= np.linalg.norm(vector)
            os.environ["ITERCC_EOM_SECTOR_OUTPUTS"] = "0" if variant == "sector-every" else "1"
            for number in range(1, args.actions + 1):
                tile_seconds.clear()
                if phases is not None:
                    phases.clear()
                if trace is not None:
                    trace.mark(f"{variant} before action {number}")
                start = time.perf_counter()
                image = engine.sigma(vector)
                seconds = time.perf_counter() - start
                if trace is not None:
                    trace.mark(f"{variant} action {number}")
                row = dict(event="action", variant=variant, number=number, seconds=round(seconds, 3),
                           tile_seconds=round(sum(tile_seconds), 3), norm=float(np.linalg.norm(image)),
                           rayleigh_Eh=float(vector @ image))
                if phases is not None:
                    row["phases"] = {k: round(v, 3) for k, v in phases.items()}
                if reference is not None:
                    row["relative_difference"] = float(np.linalg.norm(image - reference) / np.linalg.norm(reference))
                emit(**row)
                if number < args.actions:
                    del image            # not held through the next action's peak
            if reference is None:
                reference = image
            del engine, solver, image
            os.environ.pop("ITERCC_EOM_SECTOR_OUTPUTS", None)
    if trace is not None:
        emit(event="memory_peak", peak_GiB=trace.overall_peak_GiB)


if __name__ == "__main__":
    main()
