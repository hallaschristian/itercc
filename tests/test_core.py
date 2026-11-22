"""The compiled core libraries: loading, instruction-set selection and agreement."""

import ctypes
import shutil
import subprocess

import numpy as np
import pytest

from itercc import RestrictedCC, _core, _cpu_kernels
from test_cpu_triples import LABELS, labelled_model, symmetric_amplitudes


@pytest.fixture
def core_dir(tmp_path, monkeypatch):
    """An empty ITERCC_CORE_DIR; the loader cache is reset around the test."""
    monkeypatch.setenv("ITERCC_CORE_DIR", str(tmp_path))
    _core.reset()
    yield tmp_path
    monkeypatch.delenv("ITERCC_CORE_DIR")
    _core.reset()


def built_core():
    """A loadable core library for this machine: prebuilt or a runtime build."""
    library = _cpu_kernels.library()
    if library is None:
        pytest.skip("no compiled core and no C++ compiler with OpenMP")
    return library


def test_loader_prefers_the_best_supported_level(core_dir):
    source = built_core().path
    levels = _core.supported_levels()
    if not levels:
        pytest.skip("no x86-64 level flags on this CPU")
    for level in ("x86-64-v2", levels[0]):
        shutil.copy(source, core_dir / f"libitercc_core.{level}.so")
    _core.reset()                   # built_core() may have cached "none found"
    library = _core.library()
    assert library is not None
    assert library.path in (core_dir / f"libitercc_core.{levels[0]}.so",
                            _core._directories()[2] / "libitercc_core.native.so")


def test_loader_rejects_another_abi(core_dir):
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("no C compiler")
    source = core_dir / "stale.c"
    source.write_text("int itercc_core_abi(void) { return 999; }\n")
    for level in _core.supported_levels():
        subprocess.run([compiler, "-shared", "-fPIC", str(source), "-o",
                        str(core_dir / f"libitercc_core.{level}.so")], check=True)
    assert _core._load(core_dir / f"libitercc_core.{_core.supported_levels()[0]}.so",
                       "itercc_core_abi", _core.CORE_ABI, _core.CPU_FUNCTIONS) is None


def test_core_reports_its_build():
    library = built_core()
    assert library.handle.itercc_core_abi() == _core.CORE_ABI
    library.handle.itercc_core_build.restype = ctypes.c_char_p
    assert b"target=" in library.handle.itercc_core_build()


def test_loader_skips_a_build_for_missing_extensions(core_dir, monkeypatch):
    """A native build names its instruction-set extensions; a CPU without one
    of them (another node type sharing the checkout) does not load it."""
    library = built_core()
    isa = _core.build_isa(library.build)
    assert "sse2" in isa
    shutil.copy(library.path, core_dir / "libitercc_core.native.so")
    flags = _core._cpu_flags()
    if not flags:
        pytest.skip("no /proc/cpuinfo flags")
    for missing, expected in ((set(), True), (isa - {"sse2"}, False)):
        if expected is False and not missing:
            continue
        monkeypatch.setattr(_core, "_cpu_flags", lambda missing=missing: flags - missing)
        monkeypatch.setattr(_core, "_directories", lambda: ([core_dir], core_dir / "none",
                                                            core_dir / "none"))
        _core.reset()
        found = _core.library()
        assert (found is not None and found.path == core_dir / "libitercc_core.native.so") is expected


def test_prebuilt_and_runtime_builds_agree_bitwise(monkeypatch):
    prebuilt = _core.library()
    compiler = _cpu_kernels._compiler()
    if prebuilt is None or compiler is None:
        pytest.skip("needs a prebuilt core and a C++ compiler")
    runtime = _cpu_kernels._runtime_build(compiler)
    model = labelled_model(4, 5, LABELS)
    reference = RestrictedCC(*model, orbital_irreps=LABELS)
    amplitudes = symmetric_amplitudes(reference, LABELS, True)
    results = []
    for library in (prebuilt, runtime):
        monkeypatch.setattr(_cpu_kernels, "library", lambda library=library: library)
        results.append(RestrictedCC(*model, orbital_irreps=LABELS, cpu_threads=3)
                       .residuals(amplitudes))
    # No FMA contraction: the instruction set does not change any rounding.
    for left, right in zip(*results):
        np.testing.assert_array_equal(left, right)


def test_runtime_build_install_never_replaces_a_library(tmp_path, monkeypatch):
    # A second build of the same key must not rename over the first: on a
    # network file system that deletes the file under processes mapping it.
    target = tmp_path / "libitercc_core.runtime-0.so"
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    _cpu_kernels._install(first, target)
    inode = target.stat().st_ino
    _cpu_kernels._install(second, target)
    assert target.read_bytes() == b"first" and target.stat().st_ino == inode

    def no_links(source, name):
        raise PermissionError("no hard links")
    monkeypatch.setattr(_cpu_kernels.os, "link", no_links)
    _cpu_kernels._install(second, target)
    assert target.read_bytes() == b"first"
    fresh = tmp_path / "libitercc_core.runtime-1.so"
    _cpu_kernels._install(second, fresh)
    assert fresh.read_bytes() == b"second"


def test_cuda_backend_selftest():
    backend = _core.cuda()
    if backend is None:
        pytest.skip("libitercc_cuda is not built")
    count = ctypes.c_int()
    assert backend.device_count(ctypes.byref(count)) == 0
    if count.value == 0:
        pytest.skip("no CUDA device")
    error = ctypes.c_double()
    assert backend.gemm_selftest(96, ctypes.byref(error)) == 0
    assert error.value < 1e-10


def test_cuda_grouped_gemms_keep_the_callers_stream():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    backend = _core.cuda()
    if backend is None:
        pytest.skip("libitercc_cuda is not built")
    cp = pytest.importorskip("cupy")
    count = ctypes.c_int()
    if backend.device_count(ctypes.byref(count)) != 0 or not count.value:
        pytest.skip("no CUDA device")

    size, groups = 32, 48
    descriptor = np.tile([0, 0, size, size, size, 0, size, 0, size, 0, size],
                         (groups, 1)).astype(np.int64)
    coefficients = np.tile([1., 0.], (groups, 1))
    ready = Barrier(2, timeout=20)

    def launch(base):
        with cp.cuda.Device(0), cp.cuda.Stream(non_blocking=True) as stream:
            a, b, c = (cp.empty((size, size)) for _ in range(3))
            b.fill(1.)
            ready.wait()
            for step in range(12):
                value = base + step
                a.fill(value)
                c.fill(float("nan"))
                status = backend.grouped_gemm(groups, descriptor.ctypes.data, coefficients.ctypes.data,
                                              a.data.ptr, b.data.ptr, c.data.ptr, stream.ptr)
                assert status == 0
                np.testing.assert_array_equal(c.get(), value*size)

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = [workers.submit(launch, base) for base in (1, 101)]
        for result in results:
            result.result()
