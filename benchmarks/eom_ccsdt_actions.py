"""Time EOM-CCSDT Jacobian actions (`EOMSinglet` on the compiled CPU tangent)
by target irrep.

For each irrep the same engine applies its Jacobian to the same random vector
of the sector, by variant (the first is the reference of the others):

- `blocked`: the defaults. Both halves of the triples tangent run on
  symmetry-blocked T3; for a nonzero irrep the derivative operators read the
  ground T3 blocked in the orbital labels
  (`_blocked_residual.triples_shifted`).
- `dense-half`: for a nonzero irrep, the derivative-operator half on
  dense-slot ground T3 (ITERCC_SHIFTED_TANGENT=0); for irrep 0 the same as
  `blocked`.

A synthetic model (random labels of four irreps) takes its ground amplitudes
from a few iterations of the labelled solver; an archive (h, eri, nocc,
constant, orbital_irreps) with a saved ground state (t1, t2, t3; a converged
one, the engine audits its residual) gives a real one:

    python -m benchmarks.eom_ccsdt_actions --synthetic 8 56 --threads 12
    python -m benchmarks.eom_ccsdt_actions --input zero.npz --ground ground.npz --threads 12

Each `action` row has the seconds of one action (after one warm-up action
that builds the engine's cached ground parts) and its difference from the
reference relative to the reference's largest entry; `--phases` adds the
seconds of the blocked, shifted and dense-slot triples and of the feedback.
"""
import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .singlet_eom_ccsdtq import _emit

VARIANTS = ("blocked", "dense-half")


def _phase_timers():
    """Seconds of the triples and feedback paths, accumulated into the returned dict."""
    from itercc import _blocked_residual, _cpu_triples

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

    wrap(_blocked_residual, "triples", "triples blocked")
    wrap(_blocked_residual, "triples_shifted", "triples shifted")
    wrap(_cpu_triples, "triples", "triples dense-slot")
    wrap(_blocked_residual, "feedback", "feedback blocked")
    wrap(_cpu_triples, "feedback", "feedback dense-slot")
    return phases


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic", nargs=2, type=int, metavar=("NOCC", "NVIR"))
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--ground-iterations", type=int, default=3,
                        help="iterations of the synthetic ground state")
    parser.add_argument("--input", type=Path, help="Hamiltonian archive (h, eri, nocc, constant, orbital_irreps)")
    parser.add_argument("--ground", type=Path, help="saved CCSDT ground state (t1, t2, t3)")
    parser.add_argument("--irreps", default=None, help="comma-separated target irreps (default: all)")
    parser.add_argument("--variants", default=",".join(VARIANTS),
                        help=f"comma-separated, from {', '.join(VARIANTS)}")
    parser.add_argument("--actions", type=int, default=2, help="timed actions per variant")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--phases", action="store_true", help="seconds of the triples and feedback paths")
    parser.add_argument("--output", type=Path, help="also append the JSON records here")
    args = parser.parse_args()
    if bool(args.synthetic) == bool(args.input):
        parser.error("Give --synthetic or --input")
    if args.input and args.ground is None:
        parser.error("An archive needs --ground (a saved CCSDT ground state)")
    variants = args.variants.split(",")
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        parser.error(f"Unknown variants {sorted(unknown)}")

    from itercc import EOMSinglet, RestrictedCC
    from itercc._runtime import thread_scope

    emit = _emit(args.output)
    phases = _phase_timers() if args.phases else None
    with thread_scope(args.threads):
        if args.synthetic:
            from .ccsdtq_profile import synthetic

            h, eri, no, constant, labels = synthetic(*args.synthetic, args.seed)
            solver = RestrictedCC(h, eri, no, constant, orbital_irreps=labels, cpu_threads=args.threads)
            start = time.perf_counter()
            amplitudes = solver.solve(max_iter=args.ground_iterations, energy_tol=1e-30, residual_tol=1e-30,
                                      diis_size=0).amplitudes
            emit(event="ground", iterations=args.ground_iterations, seconds=round(time.perf_counter() - start, 2))
            tolerance = 1e9                  # a few iterations: not converged
        else:
            with np.load(args.input, allow_pickle=False) as model:
                h, eri, labels = model["h"], model["eri"], model["orbital_irreps"]
                no, constant = int(model["nocc"]), float(model["constant"])
            with np.load(args.ground, allow_pickle=False) as state:
                amplitudes = tuple(state[k] for k in ("t1", "t2", "t3"))
            solver = RestrictedCC(h, eri, no, constant, cpu_threads=args.threads)
            tolerance = 1e-6
        labels = np.asarray(labels, dtype=np.int64)
        groups = 1 << int(labels.max()).bit_length() if labels.max() else 1
        emit(event="model", nocc=no, nvir=len(h) - no, threads=args.threads,
             labels=np.bincount(labels, minlength=groups).tolist(),
             occupied=np.bincount(labels[:no], minlength=groups).tolist())
        irreps = [int(x) for x in args.irreps.split(",")] if args.irreps else list(range(groups))
        rng = np.random.default_rng(args.seed)
        for irrep in irreps:
            vector = reference = None
            for variant in variants:
                os.environ["ITERCC_SHIFTED_TANGENT"] = "0" if variant == "dense-half" else "1"
                start = time.perf_counter()
                engine = EOMSinglet(solver, amplitudes, orbital_irreps=labels, irrep=irrep,
                                    ground_residual_tol=tolerance)
                if vector is None:
                    vector = engine.space.project(rng.standard_normal(engine.space.size))
                    vector /= np.linalg.norm(vector)
                engine.sigma(vector)              # the cached ground parts
                emit(event="engine", irrep=irrep, variant=variant, size=int(engine.space.size),
                     symmetry_projected=bool(engine.symmetry_projected),
                     seconds=round(time.perf_counter() - start, 2))
                for number in range(1, args.actions + 1):
                    if phases is not None:
                        phases.clear()
                    start = time.perf_counter()
                    image = engine.sigma(vector)
                    row = dict(event="action", irrep=irrep, variant=variant, number=number,
                               seconds=round(time.perf_counter() - start, 3), norm=float(np.linalg.norm(image)))
                    if phases is not None:
                        row["phases"] = {k: round(v, 3) for k, v in phases.items()}
                    if reference is not None:
                        row["relative_difference"] = float(np.max(np.abs(image - reference))
                                                           / np.max(np.abs(reference)))
                    emit(**row)
                if reference is None:
                    reference = image
                del engine
            os.environ.pop("ITERCC_SHIFTED_TANGENT", None)


if __name__ == "__main__":
    main()
