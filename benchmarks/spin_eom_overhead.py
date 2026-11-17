"""Measure the full CuF DZ spin projector without an eigensolve."""
import argparse
import cProfile
import json
from pathlib import Path
import pstats
import resource
from time import perf_counter

import numpy as np

from itercc.eom_spin_quadruples import SpinFlipSpace
from itercc._spin_quadruples_storage import PackedSpinQuadruples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    with np.load(args.model) as model:
        no, labels = int(model["nocc"]), model["orbital_irreps"]
    nv = len(labels) - no
    layout = PackedSpinQuadruples(labels[:no], labels[no:], 2, 1)
    space = SpinFlipSpace(no, nv, labels, 2, np, packed=layout)
    vector = np.random.default_rng(717).standard_normal(space.size)
    vector /= np.linalg.norm(vector)
    print(json.dumps(dict(event="ready", size=space.size, setup_seconds=perf_counter()-start)), flush=True)
    profiler = cProfile.Profile()
    start = perf_counter()
    profiler.enable()
    projected = space.project(vector)
    profiler.disable()
    elapsed = perf_counter()-start
    defect = float(np.linalg.norm(space.project(projected)-projected))
    if not np.isfinite(defect) or defect > 1e-10:
        raise RuntimeError(f"Spin projector idempotence failed: {defect}")
    result = dict(project_seconds=elapsed, spin_defect=defect,
                  peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    with args.output.with_suffix(".profile.txt").open("w") as stream:
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(30)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
