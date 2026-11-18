"""Load the compiled IterCC core libraries built from `core/` with CMake.

`libitercc_core.<target>.so` holds the CPU kernels; `libitercc_cuda.so` the
CUDA backend. The loader searches `ITERCC_CORE_DIR` (one or more directories),
the package directory and `core/build` of a source checkout. For the CPU
library it prefers a `native` build from `core/build`, then the best
instruction-set level the running CPU supports (x86-64-v4, v3, v2). A
library must report the ABI number this module expects, so a stale build is
never used, and its build string's `isa=` extensions must all be present on
this CPU, so a `native` build from another node type is skipped.
"""

import ctypes
import os
from pathlib import Path
import sys
import threading

CORE_ABI = 30
CUDA_ABI = 2
_P, _I, _D, _T = ctypes.c_void_p, ctypes.c_int64, ctypes.c_double, ctypes.c_int
# name (without the itercc_ prefix): (argument types, result type)
CPU_FUNCTIONS = {
    "unpack_s8_s4": ([_P, _I, _P, _T], None),
    "unpack_pairs": ([_P, _I, _I, _I, _P, _T], None),
    "pack_pairs": ([_P, _I, _I, _P, _I, _T], None),
    "scatter_pairs": ([_P, _I, _I, _I, _P, _P, _T], None),
    "jk_s8": ([_P, _I, _P, _P, _P, _T], None),
    "int1e": ([_T, _I, _P, _P, _P, _P, _P, _P, _P, _P, _I, _P, _P, _P, _P, _I, _P, _T], None),
    "int2e_s8": ([_I, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _I, _D, _P, _T], None),
    "signed_gather_rows": ([_I, _I, _P, _P, _P, _P, _T], None),
    "permuted_gather_add": ([_I, _T, _T, _P, _P, _P, _P, _P, _P, _T], None),
    "sector_unpack": ([_I, _I, _P, _P, _P, _T], None),
    "spin_unpack": ([_I, _T, _I, _I, _P, _P, _T, _P, _P, _P, _T], None),
    "spin_quad_contract": ([_I, _T, _P, _I, _I, _P, _P, _P, _P, _P,
                            _T, _P, _P, _P, _P, _P, _P, _D, _P, _T], None),
    "spin_quad_contract_terms": ([_I, _T, _P, _I, _I, _I, _P, _P, _T], None),
    "triples_plane": ([_P, _I, _I, _P, _P, _P, _P, _P, _P, _T], None),
    "triples_seeds": ([_I, _I, _P, _T, _P, _P, _P, _P, _P, _T], None),
    "triples_scatter": ([_I, _I, _P, _P, _P, _P, _P, _T], None),
    "feedback_plane": ([_P, _I, _I, _P, _P, _P, _P, _T], None),
    "ladder_gather": ([_P, _I, _I, _I, _P, _P, _I, _P, _P, _P, _P, _P, _T], None),
    "ladder_scatter": ([_I, _I, _P, _P, _P, _P, _P, _P, _P, _T], None),
    "purify": ([_P, _I, _I, _T], None),
    "triples_step": ([_P, _I, _I, _P, _P, _D, _T], None),
    "max_abs": ([_P, _I, _T], _D),
    "t3_pack": ([_P, _I, _I, _I, _P, _P, _P, _P, _P, _P, _T], None),
    "t3_unpack": ([_P, _I, _I, _I, _P, _P, _P, _P, _P, _P, _T], None),
    "t3_leakage": ([_P, _I, _I, _P, _P, _T], _D),
    "scale": ([_P, _I, _D, _T], None),
    "scratch_release": ([_T], None),
    "sym_contract": ([_I, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _D, _T, _T], _T),
    "sym_contract_packed": ([_I, _P, _P, _P, _P, _P, _I, _P, _P, _P, _P, _P, _P, _P, _D, _T, _T], _T),
    "triplet_slots": ([_P, _I, _I, _P, _P, _P, _P, _I, _P, _T, _T, _I], _T),
    "triplet_profile": ([_P, _T], None),
    "triplet_contraction_profile": ([_I, _P, _P, _T], None),
    "quad_gather_boxes": ([_P, _P, _P, _P, _I, _I, _I, _I, _T, _P, _T], None),
    "quad_scatter": ([_P, _P, _P, _I, _D, _D, _P, _T], None),
    "quad_expand": ([_P, _P, _I, _I, _P, _T], None),
    "quad_compress": ([_P, _P, _I, _I, _P, _T], None),
    "quad_place": ([_P, _P, _I, _I, _P, _T], None),
    "quad_gather_rows": ([_P, _P, _I, _P, _P, _P, _P, _T], None),
    "quad_row_add_packed": ([_P, _P, _P, _P, _I, _P, _T], None),
    "quad_rows_permute": ([_P, _P, _P, _I, _P, _P, _I, _P, _P, _T, _T], None),
    "quad_gather_permute": ([_P, _P, _I, _P, _P, _P, _I, _P, _P, _T, _P, _T], None),
    "quad_gather_permute_f32": ([_P, _P, _I, _P, _P, _P, _I, _P, _P, _T, _P, _T], None),
    "quad_purify": ([_P, _I, _I, _P, _T], None),
    "quad_row_add": ([_P, _P, _P, _I, _P, _T], None),
    "quad_spin": ([_P, _P, _I, _I, _I, _I, _P, _P, _D, _D, _D, _T], None),
    "quad_tie": ([_P, _P, _I, _I, _I, _T], None),
    "quad_project": ([_P, _P, _I, _I, _P, _P, _I, _P, _P, _D, _D, _T], None),
    "tile_program": ([_P, _I, _P, _P, _P, _I, _P, _P, _P, _D, _T, _T], None),
    "fused_orientation": ([_T], None),
    "fused_contract": ([_I, _P, _P, _P, _P, _P, _D, _T], None),
    "fused_tile_program": ([_P, _I, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _D, _T, _T], None),
    "quad_divide": ([_P, _P, _P, _P, _I, _I, _T], None),
    "permuted_add": ([_P, _P, _P, _I, _P, _P, _D, _D, _T], _T),
    "dress_axis": ([_P, _I, _T, _I, _I, _I, _I, _P, _T], None),
    "dress_axis_from": ([_P, _P, _I, _T, _I, _I, _I, _I, _P, _T], None),
    "stacked_vecmat": ([_I, _I, _I, _P, _P, _D, _P, _T], None),
    "pair_matrices": ([_P, _I, _I, _P, _P, _I, _P, _P, _P, _P, _T], None),
    "dress_integrals": ([_P, _P, _P, _I, _I, _P, _T], None),
    "dress_integrals_ccsd": ([_P, _P, _P, _I, _I, _P, _T], None),
    "dress_integrals_open": ([_P, _P, _P, _I, _I, _P, _P, _T], None),
    "sectors_contract_t1": ([_I, _I, _P, _I, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _T],
                            None),
    "project_symmetric": ([_P, _P, _I, _P, _D, _P, _T], _T),
    "dress_derivative": ([_P, _P, _I, _I, _P, _T], None),
    "dress_derivative_ccsd": ([_P, _P, _I, _I, _P, _T], None),
    "gemm_narrow": ([_P, _I, _P, _I, _P, _I, _I, _I, _I, _D, _D, _T], None),
    "unpack_s4": ([_P, _P, _I, _T], None),
    "ladder_sectors": ([_P, _P, _P, _I, _I, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _T, _T], None),
    "ladder_sectors_t1": ([_P, _P, _I, _I, _P, _I, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _P, _T],
                          None),
    "symmetrize_pairs": ([_P, _I, _P, _P, _T], _T),
    "axpy": ([_P, _P, _D, _I, _T], None),
    "dots": ([_I, _I, _P, _P, _P, _T], None),
    "combine": ([_I, _I, _P, _P, _P, _T], None),
    "dots_f32": ([_I, _I, _P, _P, _P, _T], None),
    "to_f32": ([_P, _P, _I, _T], None),
    "permute_columns": ([_P, _I, _I, _P, _P, _T], None),
    "permute_columns_f32": ([_P, _I, _I, _P, _P, _T], None),
    "diis_rebase": ([_I, _I, _P, _P, _P, _P, _P, _P, _T], None),
    "t3_divide": ([_P, _I, _I, _P, _T], None),
    "bt3_ladder_gather": ([_P, _P, _P, _P, _P, _T], None),
    "bt3_ladder_scatter": ([_P, _P, _P, _P, _P, _T], None),
    "bt3_purify": ([_P, _P, _T], None),
    "bt3_plane": ([_P, _P, _P, _P, _P, _P, _T], None),
    "bt3_seeds": ([_P, _P, _T, _P, _P, _P, _P, _P, _T], None),
    "bt3_scatter": ([_P, _P, _I, _P, _P, _P, _P, _P, _T], None),
    "bt3_feedback_plane": ([_P, _I, _P, _P, _P, _P, _P, _T], None),
}


class BlockedPlane(ctypes.Structure):
    """itercc_blocked_plane of core.h."""
    _fields_ = [("no", ctypes.c_int64), ("k", ctypes.c_int64)] + [
        (name, ctypes.c_void_p) for name in (
            "occ_irrep", "tile_slot", "tile_perm", "pair_order", "pair_start", "pair_row",
            "pair_base", "ph_row", "ph_base", "ph_ncols", "jcol", "poff", "ntriple",
            "a_row", "a_col", "a_base", "a_ncols", "b_row", "b_base", "b_ncols")]


class BlockedT3(ctypes.Structure):
    """itercc_blocked_t3 of core.h."""
    _fields_ = [(name, ctypes.c_int64) for name in ("nv", "groups", "nslots", "size")] + [
        (name, ctypes.c_void_p) for name in ("slot_irrep", "slot_offset", "irrep", "rank",
                                              "count", "start", "order", "toff")]


class BlockedLadder(ctypes.Structure):
    """itercc_blocked_ladder of core.h."""
    _fields_ = [("first_slot", ctypes.c_int64), ("nslots", ctypes.c_int64)] + [
        (name, ctypes.c_void_p) for name in (
            "even_p", "even_q", "even_start", "odd_r", "odd_s", "odd_start", "even_row",
            "odd_row", "odd_sign", "colstart", "ncols", "sym_base", "anti_base")]


class QuadLayout(ctypes.Structure):
    """itercc_quad_layout of core.h."""
    _fields_ = [("nv", ctypes.c_int64), ("nirrep", ctypes.c_int64)] + [
        (name, ctypes.c_void_p) for name in ("virr", "vloc", "vsize", "boff", "soff", "sirr")]


class TileTerm(ctypes.Structure):
    """itercc_tile_term of core.h."""
    _fields_ = ([("a", ctypes.c_void_p), ("b", ctypes.c_void_p),
                 ("a_coef", ctypes.c_int64 * 5), ("b_coef", ctypes.c_int64 * 5)]
                + [(name, ctypes.c_void_p) for name in (
                    "a_rows", "a_sum", "b_sum", "b_cols", "rpos", "cpos",
                    "plan_start", "plan_count", "blocks")]
                + [("alpha", ctypes.c_double), ("flags", ctypes.c_int64)])


class FusedSegment(ctypes.Structure):
    """itercc_fused_segment of core.h."""
    _fields_ = ([("x_base", ctypes.c_int64), ("y_base", ctypes.c_int64),
                 ("x_coef", ctypes.c_int64 * 5), ("y_coef", ctypes.c_int64 * 5)]
                + [(name, ctypes.c_void_p) for name in ("x_rows", "x_sum", "y_sum", "y_cols")]
                + [("depth", ctypes.c_int64), ("alpha", ctypes.c_double), ("flags", ctypes.c_int64)])


class FusedBlock(ctypes.Structure):
    """itercc_fused_block of core.h."""
    _fields_ = [("rows", ctypes.c_int64), ("cols", ctypes.c_int64), ("o_base", ctypes.c_int64),
                ("rpos", ctypes.c_void_p),
                ("cpos", ctypes.c_void_p), ("seg_start", ctypes.c_int64),
                ("seg_count", ctypes.c_int64), ("tag", ctypes.c_int64)]


def structure(kind, scalars, arrays):
    """A ctypes structure whose pointer fields refer to `arrays` (kept alive on it)."""
    value = kind(**scalars, **{name: array.ctypes.data for name, array in arrays.items()})
    value._keep = arrays
    return value


CUDA_FUNCTIONS = {
    "device_count": ([ctypes.POINTER(ctypes.c_int)], _T),
    "gemm_selftest": ([_I, ctypes.POINTER(ctypes.c_double)], _T),
    "t3_pack": ([_P, _I, _I, _I, _P, _P, _P, _P, _P, _P, _P], _T),
    "t3_unpack": ([_P, _I, _I, _I, _P, _P, _P, _P, _P, _P, _P], _T),
    "gemm": ([_T, _T, _I, _I, _I, _D, _P, _I, _P, _I, _D, _P, _I, _P], _T),
    "grouped_gemm": ([_I, _P, _P, _P, _P, _P, _P], _T),
}
# x86-64 microarchitecture levels and the /proc/cpuinfo flags they need.
_V2 = {"sse4_1", "sse4_2", "ssse3", "popcnt"}
_V3 = _V2 | {"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "movbe", "abm"}
_V4 = _V3 | {"avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"}
LEVELS = (("x86-64-v4", _V4), ("x86-64-v3", _V3), ("x86-64-v2", _V2))

_lock = threading.Lock()
_state = {}


class Kernels:
    """The C functions of one loaded library, without their name prefix."""

    def __init__(self, library, functions, path, prefix="itercc_"):
        self.path = path
        self.handle = library
        for name, (arguments, result) in functions.items():
            function = getattr(library, prefix + name)
            function.argtypes, function.restype = arguments, result
            setattr(self, name, function)

    def __repr__(self):
        return f"<itercc kernels {self.path}>"


def _cpu_flags():
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as info:
            for line in info:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def supported_levels():
    flags = _cpu_flags()
    return [name for name, needed in LEVELS if needed <= flags]


def _directories():
    """(explicit ITERCC_CORE_DIR entries, package directory, source-checkout build)."""
    explicit = [Path(p) for p in os.environ.get("ITERCC_CORE_DIR", "").split(os.pathsep) if p]
    package = Path(__file__).resolve().parent
    return explicit, package, package.parent / "core" / "build"


def keep_freed_memory():
    """Let glibc keep freed memory for reuse (once per process, Linux only).

    By default glibc maps blocks above a threshold that it moves at run time
    and returns freed memory at the top of the heap to the kernel; depending on
    the order of frees, every iteration then faults its temporaries in again,
    zero-filled (water CCSDT aug-cc-pVDZ without symmetry, 12 threads: 9000
    faults and a quarter of each iteration in some processes, none in others).
    Fixed thresholds: blocks up to 32 MiB come from the heap, and up to 1 GiB
    of free heap stays mapped. `ITERCC_MALLOC=0`, or setting glibc's own
    `MALLOC_MMAP_THRESHOLD_` or `MALLOC_TRIM_THRESHOLD_`, leaves the allocator
    alone, also within `heap_scope`. Returns whether the thresholds were set."""
    done = _state.get("malloc")
    if done is not None:
        return done
    with _lock:
        if "malloc" in _state:
            return _state["malloc"]
        _state["malloc"] = False
        if (not sys.platform.startswith("linux") or os.environ.get("ITERCC_MALLOC", "1") == "0"
                or any(name in os.environ for name in _GLIBC_SETTINGS)):
            return False
        try:
            libc = ctypes.CDLL(None)
            mallopt, trim = libc.mallopt, libc.malloc_trim       # glibc (not on musl)
        except (OSError, AttributeError):
            return False
        mallopt.argtypes, mallopt.restype = [ctypes.c_int, ctypes.c_int], ctypes.c_int
        trim.argtypes, trim.restype = [ctypes.c_size_t], ctypes.c_int
        M_TRIM_THRESHOLD, M_MMAP_THRESHOLD = -1, -3
        _state["malloc"] = bool(mallopt(M_MMAP_THRESHOLD, 32 << 20) and mallopt(M_TRIM_THRESHOLD, 1 << 30))
        if _state["malloc"]:
            _state["libc"] = (mallopt, trim)
        return _state["malloc"]


_GLIBC_SETTINGS = ("MALLOC_MMAP_THRESHOLD_", "MALLOC_TRIM_THRESHOLD_", "MALLOC_MMAP_MAX_")
_heap = {"depth": 0}


class heap_scope:
    """Within: every block from the heap, none mapped on its own (glibc's
    M_MMAP_MAX 0); on leaving the outermost scope, mapping is allowed again and
    the free heap pages go back to the kernel (`malloc_trim`).

    CCSDT solves allocate their T3-sized temporaries anew every iteration;
    above 32 MiB glibc maps each one and unmaps it when freed, and the next
    iteration faults it in again, zero-filled (water on one 56-core socket,
    cc-pVQZ: 24000 faults per iteration, 0.36 instead of 0.29 s). Not used for
    CCSD, EOM or CCSDTQ: no gain there, and in CCSDTQ calloc then clears
    large arrays on one thread (2-7% slower), in dense CCSD the heap
    fragments (aug-cc-pVQZ: 23 instead of 17 GiB). No effect where
    `keep_freed_memory` does not apply."""

    def __enter__(self):
        if not keep_freed_memory() or "libc" not in _state:
            self.active = False
            return self
        self.active = True
        with _lock:
            _heap["depth"] += 1
            if _heap["depth"] == 1:
                _state["libc"][0](-4, 0)               # M_MMAP_MAX
        return self

    def __exit__(self, *exc):
        if self.active:
            with _lock:
                _heap["depth"] -= 1
                if _heap["depth"] == 0:
                    mallopt, trim = _state["libc"]
                    mallopt(-4, 65536)                 # glibc's default
                    trim(0)
        return False



def _load(path, abi_function, abi, functions, prefix="itercc_"):
    try:
        library = ctypes.CDLL(str(path))
    except OSError:
        return None
    probe = getattr(library, abi_function, None)
    if probe is None:
        return None
    probe.restype = ctypes.c_int
    if probe() != abi:
        return None
    try:
        kernels = Kernels(library, functions, path, prefix)
    except AttributeError:          # a function is missing: not this ABI after all
        return None
    build_info = getattr(library, prefix.rstrip("_") + "_core_build", None)
    if build_info is not None:
        build_info.restype = ctypes.c_char_p
        kernels.build = build_info().decode()
    else:
        kernels.build = None
    return kernels


def build_isa(build):
    """The instruction-set extensions a core build string names (`isa=...`)."""
    for part in (build or "").split():
        if part.startswith("isa="):
            return set(part[4:].split(","))
    return set()


def runs_here(kernels):
    """Whether this CPU has every extension the build was compiled for (a
    `native` build made on another node type would crash or underuse it)."""
    flags = _cpu_flags()
    return not flags or build_isa(getattr(kernels, "build", None)) <= flags


def _find_cpu():
    explicit, package, build = _directories()
    # A native build is only trusted where it was made for this machine: an
    # explicit directory or this checkout's build tree, never the package.
    candidates = [d / "libitercc_core.native.so" for d in (*explicit, build)]
    for level in supported_levels():
        candidates += [d / f"libitercc_core.{level}.so" for d in (*explicit, package, build)]
    for path in candidates:
        if path.exists():
            kernels = _load(path, "itercc_core_abi", CORE_ABI, CPU_FUNCTIONS)
            if kernels is not None and runs_here(kernels):
                return kernels
    return None


def library():
    """The prebuilt CPU core (Kernels), or None when none is installed or usable."""
    with _lock:
        if "cpu" not in _state:
            _state["cpu"] = _find_cpu()
        return _state["cpu"]


def cuda():
    """The prebuilt CUDA backend (Kernels), or None."""
    with _lock:
        if "cuda" not in _state:
            _state["cuda"] = None
            explicit, package, build = _directories()
            for directory in (*explicit, package, build):
                path = directory / "libitercc_cuda.so"
                if path.exists():
                    kernels = _load(path, "itercc_cuda_abi", CUDA_ABI, CUDA_FUNCTIONS,
                                    prefix="itercc_cuda_")
                    if kernels is not None:
                        _state["cuda"] = kernels
                        break
        return _state["cuda"]


def reset():
    """Forget loaded libraries (for tests that change the search path)."""
    with _lock:
        _state.clear()
