"""Run matched double-zeta CuF triplet CCSDTQ pilots on the CPU."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np

from itercc import RestrictedCCSDTQ, TripletEOMCCSDTQ
from itercc.checkpoint_io import write_array_archive


GROUND_SHA256 = "bbd8146f152d056d50e4cca62042e6f6f6dcc9cf6132c039c1d447aacba67fde"


def digest(path):
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 2**20), b""):
            value.update(chunk)
    return value.hexdigest()


class MonitoredTriplet(TripletEOMCCSDTQ):
    """Measure the time and peak memory of individual full Jacobian actions."""

    def _action(self, vector, **options):
        action = self.matvec_count + 1

        def rank_finished(rank, elapsed):
            print(json.dumps(dict(event="rank_finished", action=action, rank=rank,
                                  wall_seconds=elapsed,
                                  peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),
                  flush=True)

        def slot_finished(done, total, elapsed):
            print(json.dumps(dict(event="rank4_progress", action=action, slots_done=done,
                                  slots_total=total, wall_seconds=elapsed)), flush=True)

        return super()._action(vector, rank_finished=rank_finished, slot_finished=slot_finished,
                               **options)

    def sigma(self, vector):
        start = perf_counter()
        action = self.matvec_count + 1
        print(json.dumps(dict(event="action_started", action=action)), flush=True)
        image = super().sigma(vector)
        print(json.dumps(dict(event="action_finished", action=action,
                              wall_seconds=perf_counter() - start,
                              intermediate_cache=self.last_intermediate_cache,
                              peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),
              flush=True)
        return image


def run_solve(engine, args, signature):
    identity = np.array(json.dumps(signature, sort_keys=True))
    guesses = None
    if args.output.exists():
        with np.load(args.output, allow_pickle=False) as saved:
            if str(saved["identity"]) != str(identity):
                raise ValueError("Restart belongs to a different sector or Hamiltonian")
            guesses = tuple(saved[f"vector_{root}"] for root in range(args.roots))

    def checkpoint(info, vectors):
        if info["iteration"] != 1 and info["iteration"] % 3 != 0:
            return
        arrays = {f"vector_{root}": vector for root, vector in enumerate(vectors)}
        write_array_archive(args.output, dict(identity=identity, **arrays))

    result = engine.solve(
        args.roots, residual_tol=1e-6, max_iter=args.max_iter, max_space=2 * args.roots + 5,
        guesses=guesses, callback=lambda info: print(json.dumps(info), flush=True),
        checkpoint_callback=checkpoint)
    spin_defects = [float(np.linalg.norm(vector - engine.space.project(vector)))
                    for vector in result.vectors] if result.converged else []
    if any(not (0 <= defect <= 1e-9) for defect in spin_defects):
        raise RuntimeError("A converged root failed the triplet spin audit")
    arrays = {f"vector_{root}": vector for root, vector in enumerate(result.vectors)}
    write_array_archive(args.output, dict(identity=identity, **arrays,
                                         excitation_Eh=result.excitation_energies,
                                         residual_Eh=result.residual_norms,
                                         spin_projection_defect=spin_defects,
                                         converged=np.array(result.converged)))
    print(json.dumps(dict(event="finished_solve", converged=result.converged,
                          excitation_Eh=result.excitation_energies.tolist(),
                          residual_Eh=result.residual_norms, spin_projection_defect=spin_defects,
                          matvecs=result.matvecs)),
          flush=True)
    if not result.converged:
        raise RuntimeError("The triplet roots did not converge within the iteration limit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground", type=Path)
    parser.add_argument("--memory-gib", type=float, default=100)
    parser.add_argument("--cache-gib", type=float, default=0,
                        help="Byte budget in GiB for action-local products; 8 is a cluster starting point")
    parser.add_argument("--occupied-slab", type=int,
                        help="Maximum summed occupied slab width; default uses the tile byte budget")
    parser.add_argument("--irrep", type=int, choices=range(4), default=2)
    parser.add_argument("--mode", choices=("feedback", "solve"), default="feedback")
    parser.add_argument("--roots", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    solve = args.mode == "solve"
    if solve and (args.output is None or args.roots < 1 or args.max_iter < 1):
        parser.error("An eigensolve requires an output checkpoint and positive root and iteration counts")
    receipt = args.ground / "ground.json"
    model_path = args.ground / "zero.npz"
    if digest(receipt) != GROUND_SHA256:
        raise ValueError("Expected the audited bare ANO-RCC-VDZP ground state")
    ground = json.loads(receipt.read_text())
    if (not ground["converged"] or max(ground["residual_max_Eh"]) > 3e-11
            or digest(model_path) != ground["input_sha256"]):
        raise ValueError("Ground state or Hamiltonian audit failed")
    with np.load(model_path, allow_pickle=False) as model:
        h, eri = model["h"], model["eri"]
        nocc, constant = int(model["nocc"]), float(model["constant"])
        labels = model["orbital_irreps"]
    if (nocc, len(h) - nocc) != (13, 29):
        raise ValueError("Expected 26 correlated electrons and 29 virtual orbitals")

    start = perf_counter()
    solver = RestrictedCCSDTQ(h, eri, nocc, constant, backend="cpu")
    amplitudes = tuple(np.load(args.ground / "checkpoint" / f"t{rank}.npy",
                               mmap_mode="r", allow_pickle=False) for rank in range(1, 5))
    engine = MonitoredTriplet(solver, amplitudes, labels, args.irrep, backend="cpu",
                              occupied_tiles=True, action_only=not solve, packed_spin=solve,
                              memory_budget_gib=args.memory_gib, intermediate_cache_gib=args.cache_gib,
                              occupied_slab=args.occupied_slab)
    if abs(engine.ground_energy - ground["energy_Eh"]) > 2e-10:
        raise ValueError("Ground-state energy changed on reload")
    if engine._symmetry_engine is None:
        raise RuntimeError("The CuF pilot requires the compiled symmetry executor")
    dimensions = [part.stop - part.start for part in engine.space.slices]
    signature = dict(basis=ground["basis"], input_sha256=ground["input_sha256"],
                     ground_sha256=GROUND_SHA256, irrep=args.irrep, dimensions=dimensions,
                     roots=args.roots)
    print(json.dumps(dict(event="ready", **signature, setup_seconds=perf_counter() - start,
                          packed_contractions=engine.packed_contractions, cache_gib=args.cache_gib,
                          occupied_slab=engine.occupied_slab)),
          flush=True)
    if solve:
        run_solve(engine, args, signature)
        return

    vector = np.random.default_rng(714).standard_normal(engine.space.size)
    vector /= np.linalg.norm(vector)
    start = perf_counter()
    image = engine.probe_lower_ranks(vector)
    norm = float(np.linalg.norm(image))
    if image.size != sum(dimensions[:3]) or not np.isfinite(norm) or norm == 0:
        raise FloatingPointError("The feedback probe did not yield a finite nonzero image")
    print(json.dumps(dict(event="feedback_measured", **signature, norm_Eh=norm,
                          wall_seconds=perf_counter() - start,
                          peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),
          flush=True)


if __name__ == "__main__":
    main()
