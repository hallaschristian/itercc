"""Matched packed/tiled triplet actions on a saved CuF CCSDTQ checkpoint.

The triples probe retains all virtuals and summed occupied indices, selecting
canonical outputs whose holes lie in the first --holes spatial orbitals.
The rank-four probe retains all virtuals in the first --slots canonical slots.
These are bounded action benchmarks, not an eigensolve or a full-action timing.
Run from a frozen source snapshot, with identical CPU/thread allocation.
"""
import argparse
import cProfile
from hashlib import sha256
import json
from pathlib import Path
import resource
import pstats
from time import perf_counter

import numpy as np

from itercc import RestrictedCCSDTQ, TripletEOMCCSDTQ
from itercc import eom_spin_quadruples as action


def main():
    worker_started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground", type=Path)
    parser.add_argument("--part", choices=("doubles", "triples", "quadruples"), required=True)
    parser.add_argument("--version", choices=("tiled", "packed"), required=True)
    parser.add_argument("--holes", type=int, default=3)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--cache-gib", type=float, default=0)
    parser.add_argument("--occupied-slab", type=int,
                        help="Maximum summed occupied slab width; default uses the tile byte budget")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--singles-only", action="store_true")
    parser.add_argument("--numpy-projection", action="store_true")
    parser.add_argument("--numpy-tile-expansion", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--numpy-unpack", action="store_true", help="Use the pre-optimization expansion for A/B timing")
    args = parser.parse_args()
    if args.numpy_unpack:
        from itercc import _tensor_scatter
        _tensor_scatter.expand_antisymmetric = lambda *args, **kwargs: None
    ground = json.loads((args.ground / "ground.json").read_text())
    model_path = args.ground / "zero.npz"
    digest = sha256()
    with model_path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b""):
            digest.update(block)
    if (digest.hexdigest() != ground["input_sha256"] or not ground["converged"]
            or max(ground["residual_max_Eh"]) > 3e-11):
        raise ValueError("Hamiltonian or ground-state convergence audit failed")
    model = np.load(model_path)
    if (int(model["nocc"]), len(model["h"]) - int(model["nocc"])) != (13, 29):
        raise ValueError("Expected the full 26-electron, 29-virtual CuF DZ checkpoint")
    if not 1 <= args.holes <= 13 or args.slots < 1:
        raise ValueError("Invalid output sample")
    solver = RestrictedCCSDTQ(model["h"], model["eri"], int(model["nocc"]),
                              float(model["constant"]), backend="cpu")
    amplitudes = tuple(np.load(args.ground / "checkpoint" / f"t{r}.npy", mmap_mode="r")
                       for r in range(1, 5))
    start = perf_counter()
    engine = TripletEOMCCSDTQ(solver, amplitudes, model["orbital_irreps"], 2,
                              occupied_tiles=True, action_only=True, memory_budget_gib=300,
                              packed_contractions=args.version == "packed",
                              occupied_slab=args.occupied_slab,
                              intermediate_cache_gib=args.cache_gib if args.version == "packed" else 0)
    engine.native_projection = not args.numpy_projection
    engine._ground_t4.native_expansion = not args.numpy_tile_expansion
    engine._direction_t4.native_expansion = not args.numpy_tile_expansion
    if not ground["converged"] or abs(engine.ground_energy - ground["energy_Eh"]) > 2e-10:
        raise ValueError("Ground checkpoint energy audit failed")
    if args.slots > len(engine._direction_t4.slots):
        raise ValueError("Requested more canonical slots than the layout contains")
    vector = np.random.default_rng(714).standard_normal(engine.space.size)
    if args.singles_only:
        vector[engine.space.slices[0].stop:] = 0
    vector /= np.linalg.norm(vector)
    if args.part == "triples":
        axes = engine.space.indices[2]
        engine.space.indices[2] = axes[:, np.all(axes[3:] < 2 * args.holes, axis=0)]
        unpack = engine.space.unpack

        def unpack_full(*positional, **options):
            selected = engine.space.indices[2]
            engine.space.indices[2] = axes
            try:
                return iter(tuple(unpack(*positional, **options)))
            finally:
                engine.space.indices[2] = selected

        engine.space.unpack = unpack_full
        original = action.contract_tile

        def selected_holes(xp, rank, raw, cuts, *positional, **options):
            return original(xp, rank, raw, (slice(0, args.holes),) * rank,
                            *positional, **options)

        action.contract_tile = selected_holes
    print(json.dumps(dict(event="ready", part=args.part, version=args.version,
                          input_sha256=digest.hexdigest(), singles_only=args.singles_only,
                          numpy_unpack=args.numpy_unpack, numpy_projection=args.numpy_projection,
                          numpy_tile_expansion=args.numpy_tile_expansion,
                          occupied_slab=engine.occupied_slab,
                          setup_seconds=perf_counter() - start)), flush=True)
    start = perf_counter()
    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()
    output = (engine._action(vector, ranks=(2 if args.part == "doubles" else 3,)) if args.part != "quadruples" else
              engine.probe_rank_four(vector, args.slots)[:list(engine._direction_t4.slots.values())[
                  args.slots - 1][1].stop])
    elapsed = perf_counter() - start
    if profiler:
        profiler.disable()
        with args.output.with_suffix(".profile.txt").open("w") as stream:
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(45)
    if not np.all(np.isfinite(output)) or not np.linalg.norm(output):
        raise RuntimeError("Invalid action result")
    np.save(args.output, output, allow_pickle=False)
    print(json.dumps(dict(event="measured", part=args.part, version=args.version,
                          seconds=elapsed, norm=float(np.linalg.norm(output)), size=output.size,
                          worker_wall_seconds=perf_counter() - worker_started,
                          intermediate_cache=engine.last_intermediate_cache,
                          peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)), flush=True)


if __name__ == "__main__":
    main()
