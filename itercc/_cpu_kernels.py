"""The compiled CPU kernels: the prebuilt core, or a build of its sources on first use.

`library()` returns the kernels of `libitercc_core` (see `_core.py` and
`core/CMakeLists.txt`). Without a prebuilt library, the core sources (a
checkout's `core/`, or the copy an installed package carries in
`itercc/_core_src`, see setup.py) are compiled once with the system C++
compiler (`ITERCC_CXX`, else `c++` or `g++`) into `ITERCC_KERNEL_CACHE`
(default `~/.cache/itercc`), keyed by sources, flags, compiler and CPU
model, so nodes of different types sharing a home directory keep separate
`-march=native` builds. With neither, it warns once and returns None, and
the much slower NumPy thread-pool path is used; `ITERCC_CPU_KERNELS=0`
chooses that path silently.
"""

import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import threading
import warnings

from . import _core, _runtime

_PACKAGE = Path(__file__).resolve().parent
# The core sources: a source checkout's core/, else an installed package's copy.
CORE = next((path for path in (_PACKAGE.parent / "core", _PACKAGE / "_core_src")
             if (path / "include" / "itercc" / "core.h").exists()), _PACKAGE.parent / "core")
SOURCES = ("src/version.cpp", "src/cpu/triples.cpp",
           "src/cpu/blocked.cpp", "src/cpu/blocked_residual.cpp",
           "src/cpu/symmetry_blocks.cpp", "src/cpu/quadruples.cpp", "src/cpu/gemm_kernel.cpp",
           "src/cpu/tile_program.cpp", "src/cpu/triplet_program.cpp", "src/cpu/array_ops.cpp",
           "src/cpu/tensor_scatter.cpp", "src/cpu/spin_quadruples.cpp",
           "src/cpu/integrals.cpp", "src/cpu/gto_integrals.cpp")
HEADERS = ("include/itercc/core.h", "src/cpu/scratch.h", "src/cpu/gemm_kernel.h", "src/cpu/contract.h")
FLAGS = ("-std=c++17", "-O3", "-march=native", "-ffp-contract=off", "-fno-math-errno",
         "-fopenmp", "-fPIC", "-shared", '-DITERCC_BUILD_TARGET="runtime-native"')

_lock = threading.Lock()
_state = {}


def _compiler():
    for name in (_runtime.switch("ITERCC_CXX"), "c++", "g++"):
        if name and shutil.which(name):
            return shutil.which(name)
    return None


def _cpu_identity():
    """CPU model and feature flags, which `-march=native` builds depend on."""
    identity = platform.machine()
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as info:
            for line in info:
                if line.startswith(("model name", "flags")):
                    identity += line
                if line.startswith("flags"):
                    break
    except OSError:
        pass
    return identity


def _stat_key(compiler, files):
    """A digest of the sizes and modification times of the sources and the
    compiler, with the flags and CPU: a cheap stand-in for their contents."""
    parts = [" ".join(FLAGS), _cpu_identity(), compiler]
    for path in (Path(compiler), *files):
        info = path.stat()
        parts.append(f"{path}:{info.st_size}:{info.st_mtime_ns}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:20]


def _runtime_build(compiler):
    """Compile the core sources of this checkout into the kernel cache and load them.

    A process that finds the sources and compiler unchanged since an earlier
    one (`_stat_key`) loads that build without reading the sources and running
    the compiler for its version (about 80 ms, more on network file systems)."""
    files = [CORE / name for name in (*SOURCES, *HEADERS)]
    if not all(path.exists() for path in files):
        _state["error"] = f"the core sources are missing ({CORE})"
        return None
    cache = Path(_runtime.switch("ITERCC_KERNEL_CACHE") or Path.home() / ".cache" / "itercc")
    known = cache / f"known-{_stat_key(compiler, files)}"
    try:
        name = known.read_text(encoding="utf-8").strip()
        if name.startswith("libitercc_core.runtime-") and (cache / name).exists():
            kernels = _core._load(cache / name, "itercc_core_abi", _core.CORE_ABI, _core.CPU_FUNCTIONS)
            if kernels is not None:
                return kernels
    except OSError:
        pass
    kernels = _full_build(compiler, files, cache)
    if kernels is not None:
        note = None
        try:
            with tempfile.NamedTemporaryFile("w", dir=cache, delete=False, encoding="utf-8",
                                             prefix=".known-") as handle:
                note = handle.name
                handle.write(Path(kernels.path).name)
            os.replace(note, known)         # atomic for concurrent first loads
        except OSError:
            if note is not None and os.path.exists(note):
                os.unlink(note)
    return kernels


def _full_build(compiler, files, cache):
    """The build keyed by the contents of the sources, compiler version and CPU."""
    version = subprocess.run([compiler, "--version"], capture_output=True, text=True).stdout
    digest = hashlib.sha256()
    for part in (*(path.read_bytes() for path in files), " ".join(FLAGS).encode(),
                 version.encode(), _cpu_identity().encode()):
        digest.update(part)
    target = cache / f"libitercc_core.runtime-{digest.hexdigest()[:20]}.so"
    if not target.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as scratch:
            built = Path(scratch) / target.name
            subprocess.run([compiler, *FLAGS, f"-I{CORE / 'include'}",
                            *(str(CORE / name) for name in SOURCES), "-o", str(built)],
                           check=True, capture_output=True, text=True)
            os.replace(built, target)       # atomic for concurrent first builds
    return _core._load(target, "itercc_core_abi", _core.CORE_ABI, _core.CPU_FUNCTIONS)


def library():
    """The CPU kernels, or None when disabled or neither prebuilt nor buildable."""
    if not _runtime.enabled("ITERCC_CPU_KERNELS"):
        return None
    prebuilt = _core.library()
    if prebuilt is not None:
        _core.keep_freed_memory()
        return prebuilt
    with _lock:
        if "library" not in _state:
            compiler = _compiler()
            try:
                if compiler is None:
                    _state["error"] = "no C++ compiler (set ITERCC_CXX)"
                    _state["library"] = None
                else:
                    _state["library"] = _runtime_build(compiler)
                    if _state["library"] is None and "error" not in _state:
                        _state["error"] = "the runtime build does not load (ABI mismatch?)"
            except (OSError, subprocess.CalledProcessError) as error:
                _state["library"] = None
                detail = getattr(error, "stderr", None) or ""
                _state["error"] = f"{error}: {detail.strip()[-2000:]}" if detail else str(error)
            if _state["library"] is None:
                warnings.warn("IterCC: no compiled CPU kernels (" + _state["error"] + "); the "
                              "solvers fall back to NumPy, which is many times slower. Build "
                              "the core (README) or set ITERCC_CPU_KERNELS=0 to silence this.",
                              RuntimeWarning, stacklevel=2)
        built = _state["library"]
    if built is not None:
        _core.keep_freed_memory()
    return built


def pointer(array, dtype=None):
    """Address of a C-contiguous array (checked), for a kernel argument."""
    if not array.flags.c_contiguous or (dtype is not None and array.dtype != dtype):
        raise ValueError("Kernel arguments must be C-contiguous arrays of the expected type")
    return array.ctypes.data
