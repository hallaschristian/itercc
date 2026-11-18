"""One triplet EOM-CCSDTQ action split over processes (sockets or nodes).

Each slot of the rank-four action writes only its own rows, and the values
it writes do not depend on which other slots run in the same process. So the
slots can be divided between processes and the images added: the sum is the
one-process action bit for bit. The tiles of ranks two and three are dealt
out too (`lower_part`; their sums then differ by rounding). On one node a
process per socket, each with its socket's cores and memory, nearly halves
an action without any thread crossing a socket.

* `split_slots` deals the slots out in blocks.
* `Partners` belongs to the process that runs the eigensolver. It starts the
  workers, and the engine's `sigma` sends it each vector (`start`) before
  computing its own share, then adds the workers' rows (`finish`). Vectors and
  images pass through files in shared memory.
* `serve` is a worker's loop. A worker builds the same engine from a builder,
  "module:function" or "path/to/file.py:function", called with JSON keyword
  arguments: `python -m itercc._slot_split --builder ... --kwargs ...`.
"""

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

BLOCK = 16          # consecutive slots kept together (they share intermediates)
_POLL = 0.02


def split_slots(total, part, parts, block=BLOCK):
    """The slots of process `part` of `parts`: blocks of `block` consecutive
    slots dealt out in turn, which evens out costs that change along the slot
    order (later CuF slots cost more)."""
    if not 0 <= part < parts:
        raise ValueError("part must lie in 0..parts-1")
    index = np.arange(total, dtype=np.int64)
    return index[(index // block) % parts == part]


def _builder(spec):
    target, _, name = spec.rpartition(":")
    if not target or not name:
        raise ValueError("A builder is 'module:function' or 'path/file.py:function'")
    if target.endswith(".py"):
        module_spec = importlib.util.spec_from_file_location(Path(target).stem, target)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return getattr(module, name)


class Partners:
    """The worker processes of an engine's action (see the module docstring).

    `parts` processes share the slots: this one takes part 0, workers 1 to
    parts-1 are started here. `bind=True` binds worker k to NUMA node k with
    numactl (the caller binds itself to node 0); `threads` sets each worker's
    ITERCC_CPU_THREADS. Workers build their engines while this process works,
    so the first action waits for them. `close` (also at exit) stops them.
    """

    def __init__(self, engine, builder, kwargs, parts=2, *, exchange=None, bind=True, threads=None,
                 python=sys.executable, log=None, block=BLOCK):
        if parts < 2:
            raise ValueError("Splitting needs at least two parts")
        self.engine, self.parts = engine, int(parts)
        total = len(engine._direction_t4.slots)
        self.exchange = Path(exchange) if exchange is not None else Path(
            "/dev/shm" if Path("/dev/shm").is_dir() else ".") / f"itercc-split-{os.getpid()}"
        self.exchange.mkdir(parents=True)
        rows = engine.space.size - engine.space.slices[1].start        # ranks two to four
        self.vector = np.lib.format.open_memmap(self.exchange / "vector.npy", mode="w+", dtype=np.float64,
                                                shape=(engine.space.size,))
        self.images = [np.lib.format.open_memmap(self.exchange / f"image-{k}.npy", mode="w+",
                                                 dtype=np.float64, shape=(rows,))
                       for k in range(1, self.parts)]
        self.processes = []
        environment = dict(os.environ)
        if threads is not None:
            environment["ITERCC_CPU_THREADS"] = str(int(threads))
        for k in range(1, self.parts):
            command = [python, "-m", "itercc._slot_split", "--builder", builder, "--kwargs", json.dumps(kwargs),
                       "--exchange", str(self.exchange), "--part", str(k), "--parts", str(self.parts),
                       "--block", str(int(block))]
            if bind:
                command = ["numactl", f"--cpunodebind={k}", f"--membind={k}", *command]
            self.processes.append(subprocess.Popen(command, env=environment, stdout=log,
                                                   stderr=subprocess.STDOUT if log is not None else None))
        self.action = 0
        self.seconds = []           # each action's wait for the workers after this share
        engine.rank_four_slots = split_slots(total, 0, self.parts, block)
        engine.lower_part = (0, self.parts)
        engine.rank_four_partner = self
        import atexit
        atexit.register(self.close)

    def _check(self):
        for k, process in enumerate(self.processes, 1):
            error = self.exchange / f"error-{k}"
            if error.exists():
                raise RuntimeError(f"Slot worker {k} failed:\n{error.read_text()}")
            if process.poll() is not None:
                raise RuntimeError(f"Slot worker {k} exited with status {process.returncode}")

    def start(self, vector):
        """Send `vector` (real) to the workers."""
        self._check()
        self.action += 1
        np.copyto(self.vector, vector)
        self.vector.flush()
        signal = self.exchange / f"go-{self.action}"
        signal.with_suffix(".tmp").write_text("")
        os.replace(signal.with_suffix(".tmp"), signal)

    def finish(self):
        """The workers' rows of ranks two to four for the last vector, summed."""
        started = time.perf_counter()
        pending = set(range(1, self.parts))
        while pending:
            pending = {k for k in pending if not (self.exchange / f"done-{self.action}-{k}").exists()}
            if pending:
                self._check()
                time.sleep(_POLL)
        self.seconds.append(time.perf_counter() - started)
        (self.exchange / f"go-{self.action}").unlink(missing_ok=True)
        for k in range(1, self.parts):
            (self.exchange / f"done-{self.action}-{k}").unlink(missing_ok=True)
        if len(self.images) == 1:
            return self.images[0]
        total = np.array(self.images[0])
        for image in self.images[1:]:
            total += image
        return total

    def close(self):
        """Stop the workers and remove the exchange files."""
        if not self.processes:
            return
        if self.exchange.exists():
            (self.exchange / "stop").write_text("")
        for process in self.processes:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
        self.processes = []
        self.engine.rank_four_partner = None
        self.engine.rank_four_slots = None
        self.engine.lower_part = None
        shutil.rmtree(self.exchange, ignore_errors=True)


def serve(engine, exchange, part, parts, block=BLOCK):
    """A worker's loop: for every vector the solver sends, this part's rows of
    ranks two to four (its lower tiles and rank-four slots)."""
    from . import _runtime

    exchange = Path(exchange)
    engine.rank_four_slots = split_slots(len(engine._direction_t4.slots), part, parts, block)
    engine.lower_part = (part, parts)
    vector = np.load(exchange / "vector.npy", mmap_mode="r")
    image = np.load(exchange / f"image-{part}.npy", mmap_mode="r+")
    print(json.dumps(dict(event="slot_worker_ready", part=part, parts=parts,
                          slots=len(engine.rank_four_slots))), flush=True)
    action = 0
    while True:
        if (exchange / "stop").exists() or not exchange.exists():
            return
        if not (exchange / f"go-{action + 1}").exists():
            time.sleep(_POLL)
            continue
        action += 1
        started = time.perf_counter()
        with _runtime.thread_scope(engine.cpu_threads):
            rows = engine._action(np.asarray(vector), ranks=(2, 3, 4))
        np.copyto(image, rows)
        image.flush()
        (exchange / f"done-{action}-{part}").write_text("")
        record = dict(event="slot_worker_action", part=part, action=action,
                      wall_seconds=time.perf_counter() - started, intermediate_cache=engine.last_intermediate_cache)
        if _runtime.enabled("ITERCC_SLOT_PROFILE"):         # the slot program's thread-seconds by phase
            from . import _cpu_kernels

            seconds = np.zeros(9)
            _cpu_kernels.library().triplet_profile(seconds.ctypes.data, 1)
            record["slot_phases"] = [round(float(x), 1) for x in seconds]
        print(json.dumps(record), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="A slot worker of a split triplet EOM-CCSDTQ action")
    parser.add_argument("--builder", required=True)
    parser.add_argument("--kwargs", default="{}")
    parser.add_argument("--exchange", required=True)
    parser.add_argument("--part", type=int, required=True)
    parser.add_argument("--parts", type=int, required=True)
    parser.add_argument("--block", type=int, default=BLOCK)
    args = parser.parse_args(argv)
    exchange = Path(args.exchange)
    try:
        engine = _builder(args.builder)(**json.loads(args.kwargs))
        serve(engine, exchange, args.part, args.parts, args.block)
    except BaseException:
        if exchange.exists():
            (exchange / f"error-{args.part}").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
