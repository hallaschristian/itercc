"""Run matched double-zeta CuF triplet CCSDTQ pilots on the CPU."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
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
                              slot_phases=slot_phases(),
                              peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),
              flush=True)
        return image


def slot_phases():
    """The slot program's thread-seconds by phase since the last call
    (ITERCC_SLOT_PROFILE=1; else None)."""
    from itercc import _runtime, _cpu_kernels

    if not _runtime.enabled("ITERCC_SLOT_PROFILE"):
        return None
    seconds = np.zeros(9)
    _cpu_kernels.library().triplet_profile(seconds.ctypes.data, 1)
    names = ("zero", "tile_products", "intermediates", "t4_rows", "gathers", "cache_waits", "slot_sums",
             "allocation", "wave_barrier")
    return dict(zip(names, (round(float(x), 1) for x in seconds)))


def memory_trace():
    """Resident memory by phase (`benchmarks._memory.MemoryTrace`): around the
    slot program, the rank-four direct terms and the lower-rank tiles of
    every action; Davidson iterations are marked by `run_solve`."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _memory import MemoryTrace
    from itercc import _spin_quadruples_cpu, _spin_quadruples_program

    trace = MemoryTrace(lambda **row: print(json.dumps(row), flush=True))
    trace.wrap(_spin_quadruples_program.SlotProgram, "run", "slot program")
    trace.wrap(_spin_quadruples_cpu.PackedBinaryAction, "add_rank_four", "rank-4 direct terms")
    trace.wrap(TripletEOMCCSDTQ, "_project_lower_tiles", "lower tiles")
    return trace


def run_solve(engine, args, signature, trace=None):
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

    partners = None
    if args.sockets > 1:
        # One process per socket shares every action's rank-four slots: this
        # one (bound to node 0 by its launcher) and workers bound to nodes 1...
        from itercc._slot_split import Partners

        kwargs = dict(ground=str(args.ground), memory_gib=args.memory_gib, cache_gib=args.cache_gib,
                      occupied_slab=args.occupied_slab, irrep=args.irrep, monitored=False,
                      intermediate_precision=args.intermediate_precision)
        partners = Partners(engine, f"{Path(__file__).resolve()}:build_engine", kwargs, parts=args.sockets,
                            threads=args.socket_threads,
                            log=open(args.output.with_name(args.output.stem + "-workers.log"), "a"))
        print(json.dumps(dict(event="split_action", sockets=args.sockets,
                              slots_here=len(engine.rank_four_slots),
                              slots_total=len(engine._direction_t4.slots))), flush=True)
    def progress(info):
        print(json.dumps(info), flush=True)
        if trace is not None:
            trace.mark(f"davidson iteration {info['iteration']}")

    result = engine.solve(
        args.roots, residual_tol=1e-6, max_iter=args.max_iter, max_space=2 * args.roots + 5,
        guesses=guesses, callback=progress, checkpoint_callback=checkpoint)
    if partners is not None:
        partners.close()
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


def build_engine(ground, memory_gib, cache_gib, occupied_slab, irrep, *, solve=True, monitored=True,
                 intermediate_precision="double"):
    """The audited CuF triplet engine (also a slot worker's builder, see
    itercc._slot_split): (engine, ground receipt, setup seconds)."""
    ground_dir = Path(ground)
    receipt = ground_dir / "ground.json"
    model_path = ground_dir / "zero.npz"
    if digest(receipt) != GROUND_SHA256:
        raise ValueError("Expected the audited bare ANO-RCC-VDZP ground state")
    receipt_values = json.loads(receipt.read_text())
    if (not receipt_values["converged"] or max(receipt_values["residual_max_Eh"]) > 3e-11
            or digest(model_path) != receipt_values["input_sha256"]):
        raise ValueError("Ground state or Hamiltonian audit failed")
    with np.load(model_path, allow_pickle=False) as model:
        h, eri = model["h"], model["eri"]
        nocc, constant = int(model["nocc"]), float(model["constant"])
        labels = model["orbital_irreps"]
    if (nocc, len(h) - nocc) != (13, 29):
        raise ValueError("Expected 26 correlated electrons and 29 virtual orbitals")
    start = perf_counter()
    solver = RestrictedCCSDTQ(h, eri, nocc, constant, backend="cpu")
    amplitudes = tuple(np.load(ground_dir / "checkpoint" / f"t{rank}.npy",
                               mmap_mode="r", allow_pickle=False) for rank in range(1, 5))
    engine = (MonitoredTriplet if monitored else TripletEOMCCSDTQ)(
        solver, amplitudes, labels, irrep, backend="cpu", occupied_tiles=True, action_only=not solve,
        packed_spin=solve, memory_budget_gib=memory_gib, intermediate_cache_gib=cache_gib,
        occupied_slab=occupied_slab, intermediate_precision=intermediate_precision)
    if abs(engine.ground_energy - receipt_values["energy_Eh"]) > 2e-10:
        raise ValueError("Ground-state energy changed on reload")
    if engine._symmetry_engine is None:
        raise RuntimeError("The CuF pilot requires the compiled symmetry executor")
    return engine if not monitored else (engine, receipt_values, perf_counter() - start)


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
    parser.add_argument("--sockets", type=int, default=1,
                        help="Processes (one per NUMA node) that share each action's rank-four slots")
    parser.add_argument("--socket-threads", type=int,
                        help="Threads of each worker process (default: ITERCC_CPU_THREADS)")
    parser.add_argument("--intermediate-precision", choices=("double", "single"), default="double",
                        help="single: rank-four slot-program intermediates rounded to float32 (opt-in)")
    parser.add_argument("--memory-trace", action="store_true",
                        help="resident memory by phase of every action and Davidson iteration")
    args = parser.parse_args()
    solve = args.mode == "solve"
    if solve and (args.output is None or args.roots < 1 or args.max_iter < 1):
        parser.error("An eigensolve requires an output checkpoint and positive root and iteration counts")
    if args.sockets < 1 or (args.sockets > 1 and not solve):
        parser.error("--sockets splits the actions of an eigensolve")
    trace = memory_trace() if args.memory_trace else None
    engine, ground, setup_seconds = build_engine(args.ground, args.memory_gib, args.cache_gib,
                                                 args.occupied_slab, args.irrep, solve=solve,
                                                 intermediate_precision=args.intermediate_precision)
    dimensions = [part.stop - part.start for part in engine.space.slices]
    signature = dict(basis=ground["basis"], input_sha256=ground["input_sha256"],
                     ground_sha256=GROUND_SHA256, irrep=args.irrep, dimensions=dimensions,
                     roots=args.roots)
    print(json.dumps(dict(event="ready", **signature, setup_seconds=setup_seconds,
                          packed_contractions=engine.packed_contractions, cache_gib=args.cache_gib,
                          occupied_slab=engine.occupied_slab,
                          intermediate_precision=engine.intermediate_precision)),
          flush=True)
    if trace is not None:
        trace.mark("engine")
    if solve:
        run_solve(engine, args, signature, trace)
        if trace is not None:
            print(json.dumps(dict(event="memory_peak", peak_GiB=trace.overall_peak_GiB)), flush=True)
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
