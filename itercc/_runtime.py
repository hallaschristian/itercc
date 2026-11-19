"""What every method shares at run time: threads, host memory and switches.

This is the one place the package decides these, so that every solver, engine
and kernel wrapper behaves the same way:

* **Threads.** `cpu_threads()` is the thread count of every compiled kernel:
  an explicit count, else the enclosing `thread_scope`, else
  ITERCC_CPU_THREADS, else one per physical core of the affinity mask. Every
  solver and excited-state engine takes a `cpu_threads` option (checked by
  `thread_option`; engines built on a ground-state solver inherit its count)
  and runs its public calls inside `thread_scope(self.cpu_threads)`, so the
  kernels below call `cpu_threads(None)` and never read the environment.
* **Host memory.** `host_memory_limit()` (physical memory, cgroup, Slurm
  allocation, NUMA nodes) and `resident_bytes()`. Budgets are in bytes;
  options that take GiB end in `_gib`.
* **Switches.** Every ITERCC_* environment variable is registered in
  `SWITCHES` with its default, when it is read and what it does, and read
  through `switch`, `enabled` or `integer`. The table in docs/guide.md is
  `switch_table()` (a test keeps them equal, and checks that the package and
  the core read no unregistered ITERCC_* variable).

The compiled CPU core is reached through `_cpu_kernels.library()` (None
without it); `_core` loads the libraries and checks their ABI.
"""

from contextlib import contextmanager
from functools import lru_cache, wraps
from numbers import Integral
import os
from pathlib import Path
import threading
from typing import NamedTuple


# Environment switches ------------------------------------------------------

class Switch(NamedTuple):
    default: str        # the value when unset or empty
    read: str           # when it is read: "call", "construction", "import", "load" or "core" (C++)
    effect: str         # one line for docs/guide.md


SWITCHES = {
    # Threads, the compiled core and inputs
    "ITERCC_CPU_THREADS": Switch(
        "", "call", "kernel threads when no `cpu_threads` option is given (default: physical cores "
        "in the affinity mask)"),
    "ITERCC_CPU_KERNELS": Switch("1", "call", "`0`: NumPy path only (no compiled core, no warning)"),
    "ITERCC_CORE_DIR": Switch("", "load", "directories searched first for a prebuilt core"),
    "ITERCC_KERNEL_CACHE": Switch("", "load", "where first-use core builds go (default `~/.cache/itercc`)"),
    "ITERCC_CXX": Switch("", "load", "the compiler of first-use core builds (default `c++`, then `g++`)"),
    "ITERCC_MALLOC": Switch("1", "load", "`0`: leave glibc's malloc thresholds alone"),
    "ITERCC_BASIS_PATH": Switch("", "call", "extra directories of basis-set files"),
    "ITERCC_NATIVE_TRANSFORM": Switch(
        "1", "call", "`0`: PySCF's integral transformation in `from_rhf` instead of the core's"),
    # CCSD and CCSDT
    "ITERCC_BLOCKED": Switch("1", "construction", "`0`: CCSDT on dense instead of symmetry-blocked T3"),
    "ITERCC_BARE_LADDER": Switch(
        "", "construction", "`0` or `1`: particle-particle ladder without or with the cached bare "
        "sectors (default: always in ground states, from 30 virtuals in EOM/response actions)"),
    "ITERCC_OPEN_DRESSING": Switch(
        "1", "call", "`0`: CCSDT T1 dressing writes the dressed g(vvvv) instead of ladder sectors"),
    "ITERCC_DRESS_SLABS": Switch(
        "", "core", "`1` or `0`: CCSD T1 dressing by slabs or by plane tasks (default: plane tasks "
        "with more than 3 threads per occupied orbital)"),
    # EOM and response
    "ITERCC_CPU_TANGENT": Switch("1", "construction", "`0`: EOM/response actions on the generic path"),
    # CCSDTQ
    "ITERCC_TILE_BLOCKED_T3": Switch("1", "call", "`0`: CCSDTQ tile programs read dense T3"),
    "ITERCC_TILE_FUSED": Switch("1", "call", "`0`: CCSDTQ tile programs make one pass per term"),
    "ITERCC_TILE_MEMORY": Switch("", "call", "CCSDTQ tile memory budget in bytes (default: host memory / 8)"),
    "ITERCC_TILE_TEAM": Switch("", "call", "CCSDTQ threads per tile-program team"),
    "ITERCC_LOWER_TEAMS": Switch("", "call", "CCSDTQ concurrent teams of the T4-into-lower-rank passes"),
    "ITERCC_TEAM_SCHEDULE": Switch("", "call", "`rounds`: CCSDTQ teams in fixed rounds (diagnostic)"),
    "ITERCC_LOWER_CHUNK": Switch(
        "", "call", "occupied rows per chunk of the CCSDTQ T4-into-lower-rank pass (default: 2 GiB)"),
    "ITERCC_DIIS_COMPACT": Switch("1", "call", "`0`: float64 instead of compact float32 host T4 DIIS history"),
    # Triplet EOM-CCSDTQ
    "ITERCC_TRIPLET_PROGRAM": Switch(
        "1", "call", "`0`: triplet EOM-CCSDTQ rank-four slots in the Python loop instead of the "
        "compiled program"),
    "ITERCC_TRIPLET_FUSED": Switch(
        "1", "call", "`0`: triplet slot program adds products into a tile one by one (the loop's "
        "action bit for bit)"),
    "ITERCC_DIRECT_DENSE": Switch(
        "1", "call", "`0`: triplet direct packed T4 terms by per-row sparse lookups instead of contexts"),
    "ITERCC_SLOT_POSITIONS": Switch(
        "1", "call", "`0`: triplet slot-program intermediates and tiles keep virtual axes in orbital "
        "order, not irrep order"),
    "ITERCC_LOWER_TILE_REUSE": Switch(
        "1", "call", "`0`: the triplet engine's rank-two and rank-three tiles are fresh arrays in letter order, not "
        "one reused tile with its occupied axes outermost"),
    "ITERCC_SPIN_BLOCKED_T3": Switch(
        "1", "construction", "`0`: the triplet engine keeps T3 and each action's R3 as dense spin-orbital "
        "tensors, not their nonzero spin blocks"),
    "ITERCC_HOLES_FIRST": Switch(
        "1", "call", "`0`: triplet dense rank-three tensors with the occupied axes innermost, not outermost"),
    "ITERCC_PAIR_SUMS": Switch(
        "1", "call", "`0`: triplet tile products sum both orders of an exchangeable pair of summed "
        "letters instead of x < y twice"),
    "ITERCC_COMPACT_TILES": Switch(
        "1", "call", "`0`: triplet slot-program tiles stored whole (nv^4), not only their ordered entries"),
    "ITERCC_TILE_CHAINS_INNER": Switch(
        "1", "call", "`0`: compact slot-program tiles store their axes in letter order, not chains innermost"),
    "ITERCC_INTER_CHAINS_INNER": Switch(
        "1", "call", "`0`: compact slot-program intermediates store their axes in letter order, not chains "
        "innermost"),
    "ITERCC_COMPACT_INTERMEDIATES": Switch(
        "1", "call", "`0`: triplet slot-program intermediates stored whole, not only their ordered entries"),
    "ITERCC_ORDERED_INTERMEDIATES": Switch(
        "1", "call", "`0`: triplet slot-program intermediates formed at every entry, not only where "
        "their readers read"),
    "ITERCC_DIRECT_OPENED": Switch(
        "1", "call", "`0`: triplet rank-four direct packed T4 terms by contexts per output slot "
        "(`ITERCC_DIRECT_DENSE`) instead of one GEMM per hole tuple and class"),
    "ITERCC_SLOT_SHARE_WORK": Switch(
        "", "call", "multiply-adds from which a slot-program intermediate is computed in shared "
        "pieces (default 2^22)"),
    "ITERCC_SLOT_PROFILE": Switch("", "core", "`1`: slot-program phase timers (`itercc_triplet_profile`)"),
    "ITERCC_SLOT_SORT": Switch("", "core", "`0`: slot-program tiles in slot order (diagnostic)"),
    "ITERCC_TILE_RUNS": Switch("", "core", "`0`: slot-program tiles (dense or compact) cleared whole, not only "
                               "where symmetry allows entries (diagnostic)"),
    # The CPU GEMM executor (read once by the core)
    "ITERCC_SYM_MIN_SHARE": Switch("", "core", "GEMM executor: least work per thread of a shared block"),
    "ITERCC_SYM_PARALLEL_WORK": Switch(
        "", "core", "GEMM executor: work from which one block runs on all threads"),
    "ITERCC_GEMM_NC": Switch("", "core", "`auto`: narrower GEMM column blocks for deep products (experiment)"),
    "ITERCC_GEMM_KNOCKOUT": Switch("", "core", "GEMM micro-kernel stages to skip (timing diagnostic only)"),
    # CUDA
    "ITERCC_GEMM_AUTOTUNE": Switch("1", "import", "`0`: CUDA GEMM routes chosen statically instead of timed"),
    "ITERCC_BOUNDED_GEMM": Switch("0", "import", "`1`: CUDA slab einsum products through the GEMM router"),
    "ITERCC_SLAB_MIB": Switch("256", "import", "`bounded_einsum`: MiB per slab intermediate"),
    "ITERCC_SYMMETRY_FUSED_DEPTH": Switch("128", "import", "CUDA symmetry blocks: deepest fused-kernel sum"),
    "ITERCC_SYMMETRY_FUSED_SKINNY": Switch("16", "import", "CUDA symmetry blocks: widest skinny fused product"),
    "ITERCC_SYMMETRY_DIRECT_DEPTH": Switch("16", "import", "CUDA symmetry blocks: deepest direct-kernel sum"),
}


def switch(name):
    """The value of the registered switch `name` (its default when unset or empty)."""
    default = SWITCHES[name].default
    return os.environ.get(name) or default


def enabled(name):
    """A registered on/off switch: on unless "0" when it defaults on, off
    unless "1" when it defaults off."""
    value = switch(name)
    return value != "0" if SWITCHES[name].default == "1" else value == "1"


def integer(name):
    """A registered integer switch, or None when unset."""
    value = switch(name)
    return int(float(value)) if value else None


def switch_table():
    """The Markdown table of every switch (docs/guide.md carries it verbatim)."""
    rows = ["| Environment variable | Read | Effect |", "| --- | --- | --- |"]
    rows += [f"| `{name}` | {entry.read} | {entry.effect} |" for name, entry in SWITCHES.items()]
    return "\n".join(rows) + "\n"


# CPUs and cores ------------------------------------------------------------

def _cpu_set():
    try:
        return set(os.sched_getaffinity(0))
    except AttributeError:
        return set(range(os.cpu_count() or 1))


def _process_cpus():
    return len(_cpu_set())


# The CPUs of the process, read at import (with OMP_PROC_BIND set, the OpenMP
# runtime later binds the calling thread to a single core, and an affinity
# read then would see one CPU).
_PROCESS_CPU_SET = _cpu_set()
_PROCESS_CPUS = len(_PROCESS_CPU_SET)


def _available_cpus():
    return max(_process_cpus(), _PROCESS_CPUS)


def _id_list(text):
    """The integers of a Linux list such as "0-3,8,10-11"."""
    values = set()
    for part in text.strip().split(","):
        if part:
            first, _, last = part.partition("-")
            values.update(range(int(first), int(last or first) + 1))
    return values


@lru_cache(maxsize=8)
def _cores(cpus):
    """Physical cores among the CPUs `cpus` (SMT siblings count once), or
    len(cpus) where the topology is not exposed."""
    cores = set()
    for cpu in cpus:
        try:
            siblings = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text()
            cores.add(min(_id_list(siblings)))
        except (OSError, ValueError):
            return len(cpus)
    return max(1, len(cores))


_PROCESS_CORES = _cores(frozenset(_PROCESS_CPU_SET))


def _available_cores():
    # Fewer CPUs than at import: bound to one by the OpenMP runtime.
    current = _cpu_set()
    return _PROCESS_CORES if len(current) <= _PROCESS_CPUS else _cores(frozenset(current))


# Host memory ---------------------------------------------------------------

def _cgroup_memory_limit():
    """The smallest memory limit of this process's cgroup and its ancestors
    (cgroup v2 memory.max or v1 memory.limit_in_bytes), or None."""
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    limits = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        if parts[1] == "":
            base, name = Path("/sys/fs/cgroup"), "memory.max"
        elif "memory" in parts[1].split(","):
            base, name = Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes"
        else:
            continue
        group = Path(parts[2])
        while True:
            try:
                text = (base / group.relative_to("/") / name).read_text().strip()
            except (OSError, ValueError):
                text = ""
            if text.isdigit() and int(text) < 2**60:
                limits.append(int(text))
            if group == group.parent:
                break
            group = group.parent
    return min(limits, default=None)


def _numa_memory():
    """Memory of the NUMA nodes this process allocates on, or None: the nodes
    of Mems_allowed_list whose CPUs it may run on. numactl --cpunodebind
    narrows the affinity (read at import), while --membind leaves
    Mems_allowed_list (the cpuset) unchanged, so the CPUs decide."""
    try:
        status = Path("/proc/self/status").read_text()
        allowed = _id_list(next(line.split(":", 1)[1] for line in status.splitlines()
                                if line.startswith("Mems_allowed_list")))
        nodes = {node for node in allowed
                 if _id_list(Path(f"/sys/devices/system/node/node{node}/cpulist").read_text())
                 & _PROCESS_CPU_SET} or allowed
        total = 0
        for node in nodes:
            text = Path(f"/sys/devices/system/node/node{node}/meminfo").read_text()
            total += next(int(line.split()[-2]) for line in text.splitlines() if "MemTotal" in line) * 1024
        return total or None
    except (OSError, StopIteration, ValueError):
        return None


def host_memory_limit():
    """Bytes of host memory this process may use: the least of physical
    memory, the cgroup limit, the Slurm allocation (SLURM_MEM_PER_NODE, or
    SLURM_MEM_PER_CPU per allowed CPU; 0 means the whole node) and the memory
    of the NUMA nodes it runs on (a point per socket)."""
    limits = []
    try:
        physical = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        pass
    else:
        if physical > 0:
            limits.append(physical)
    for value in (_cgroup_memory_limit(), _numa_memory()):
        if value is not None:
            limits.append(value)
    per_node, per_cpu = os.environ.get("SLURM_MEM_PER_NODE"), os.environ.get("SLURM_MEM_PER_CPU")
    if per_node and per_node.isdigit() and int(per_node) > 0:
        limits.append(int(per_node) * 2**20)
    elif per_cpu and per_cpu.isdigit() and int(per_cpu) > 0:
        limits.append(int(per_cpu) * 2**20 * _PROCESS_CPUS)
    return min(limits, default=None)


def resident_bytes():
    """Resident host memory of this process in bytes (0 where unknown)."""
    try:
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError, AttributeError):
        return 0


# Threads -------------------------------------------------------------------

_SCOPE = threading.local()


def thread_option(value):
    """A solver's `cpu_threads` option, checked: None or a positive integer."""
    if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value < 1):
        raise ValueError("cpu_threads must be a positive integer")
    return None if value is None else int(value)


def threaded(method):
    """A solver's or engine's public method, run inside
    `thread_scope(self.cpu_threads)` (None: the count in effect at the call)."""
    @wraps(method)
    def call(self, *args, **kwargs):
        with thread_scope(getattr(self, "cpu_threads", None)):
            return method(self, *args, **kwargs)
    return call


@contextmanager
def thread_scope(threads):
    """Inside the block, `cpu_threads()` without an explicit count returns
    `threads` in this Python thread: a solver's thread option reaches every
    kernel it calls. None pins the count in effect when the block starts, so
    the kernels of the block skip the environment lookup (one triplet
    EOM-CCSDTQ action makes 300k calls of `cpu_threads`). The scope is per
    Python thread: worker pools pass explicit counts."""
    previous = getattr(_SCOPE, "threads", None)
    _SCOPE.threads = int(threads) if threads is not None else cpu_threads()
    try:
        yield
    finally:
        _SCOPE.threads = previous


def cpu_threads(requested=None):
    """Threads of the compiled kernels: `requested`, the enclosing
    `thread_scope`, ITERCC_CPU_THREADS, or one per physical core of the
    affinity mask (the hot paths call no BLAS, so NumPy's OpenBLAS count,
    capped at 64, does not bound them). SMT siblings add no arithmetic, and a
    team on every logical CPU of a shared machine waits at each barrier for
    CPUs the host is using: in a 24-CPU WSL VM on 12 cores, 24 threads made a
    CCSDT Jacobian action 4.6x slower than 12. The NumPy fallback is limited
    by GIL hand-offs between Python threads and overlaps BLAS, so it defaults
    to 2 workers.
    """
    if requested is None:
        requested = getattr(_SCOPE, "threads", None)
    if requested is None:
        requested = integer("ITERCC_CPU_THREADS")
    if requested is None:
        from . import _cpu_kernels

        requested = (_available_cores() if _cpu_kernels.library() is not None
                     else min(2, _available_cpus()))
    return max(1, int(requested))
