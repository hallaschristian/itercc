"""The runtime layer every method shares (`itercc._runtime`): one registry of
ITERCC_* switches that the package, the core and the guide agree on, and one
`cpu_threads` option that reaches every kernel of a solver or engine."""
from pathlib import Path
import re

import numpy as np
import pytest

from itercc import _runtime

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "itercc"
CORE = ROOT / "core" / "src"
GUIDE = ROOT / "docs" / "guide.md"
NOT_SWITCHES = {"ITERCC_BUILD_TARGET"}          # a compile-time definition of the runtime build


def _python_sources():
    return {path: path.read_text(encoding="utf-8") for path in PACKAGE.glob("*.py")}


def _core_reads():
    if not CORE.is_dir():
        pytest.skip("no core sources")
    names = set()
    for path in CORE.rglob("*.cpp"):
        names |= set(re.findall(r'(?:getenv|tuned)\("(ITERCC_[A-Z0-9_]+)"', path.read_text(encoding="utf-8")))
    return names


def test_every_switch_the_package_or_core_reads_is_registered():
    named = set()
    for path, text in _python_sources().items():
        named |= set(re.findall(r"ITERCC_[A-Z0-9_]+", text))
        if path.name != "_runtime.py":
            # Only the registry reads the environment for an ITERCC_* switch.
            assert not re.search(r"""(environ|getenv)[^\n]*["']ITERCC_""", text), path.name
    assert named - NOT_SWITCHES <= set(_runtime.SWITCHES)
    core = _core_reads()
    assert core <= set(_runtime.SWITCHES)
    assert {name for name, entry in _runtime.SWITCHES.items() if entry.read == "core"} == core


def test_every_registered_switch_is_read_somewhere():
    sources = "\n".join(text for path, text in _python_sources().items() if path.name != "_runtime.py")
    core = _core_reads()
    for name, entry in _runtime.SWITCHES.items():
        assert name in (core if entry.read == "core" else sources), name


def test_the_guide_carries_the_switch_table():
    if not GUIDE.exists():
        pytest.skip("no docs")
    assert _runtime.switch_table() in GUIDE.read_text(encoding="utf-8")


def test_switch_values(monkeypatch):
    monkeypatch.delenv("ITERCC_BLOCKED", raising=False)
    assert _runtime.enabled("ITERCC_BLOCKED")
    monkeypatch.setenv("ITERCC_BLOCKED", "")             # empty: the default
    assert _runtime.enabled("ITERCC_BLOCKED")
    monkeypatch.setenv("ITERCC_BLOCKED", "0")
    assert not _runtime.enabled("ITERCC_BLOCKED")
    monkeypatch.delenv("ITERCC_BOUNDED_GEMM", raising=False)
    assert not _runtime.enabled("ITERCC_BOUNDED_GEMM")   # off unless "1"
    monkeypatch.setenv("ITERCC_BOUNDED_GEMM", "1")
    assert _runtime.enabled("ITERCC_BOUNDED_GEMM")
    monkeypatch.setenv("ITERCC_TILE_TEAM", "6")
    assert _runtime.integer("ITERCC_TILE_TEAM") == 6
    monkeypatch.delenv("ITERCC_TILE_TEAM")
    assert _runtime.integer("ITERCC_TILE_TEAM") is None
    with pytest.raises(KeyError):
        _runtime.switch("ITERCC_NOT_A_SWITCH")


def test_threaded_methods_run_in_their_thread_option(monkeypatch):
    monkeypatch.setenv("ITERCC_CPU_THREADS", "5")

    class Engine:
        cpu_threads = None

        @_runtime.threaded
        def run(self):
            return _runtime.cpu_threads()

    engine = Engine()
    assert engine.run() == 5                             # None: the count at the call
    engine.cpu_threads = 3
    assert engine.run() == 3
    assert _runtime.cpu_threads() == 5                   # the scope ends with the call
    for bad in (0, -1, 2.0, True, "4"):
        with pytest.raises(ValueError):
            _runtime.thread_option(bad)
    assert _runtime.thread_option(np.int64(4)) == 4


def test_solvers_take_and_pass_on_the_thread_option():
    pytest.importorskip("pyscf")
    from pyscf import gto, scf

    from itercc import DipoleResponse, EOMSinglet, RestrictedCC, RestrictedCCSDTQ
    from itercc.pyscf_interface import dipole_integrals, from_rhf

    mol = gto.M(atom="O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469", basis="sto-3g", symmetry=True,
                verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    m = from_rhf(mf)
    with pytest.raises(ValueError):
        RestrictedCC(m.h, m.eri, m.nocc, m.constant, cpu_threads=0)
    with pytest.raises(ValueError):
        RestrictedCCSDTQ(m.h, m.eri, m.nocc, m.constant, backend="cpu", cpu_threads=0)
    assert RestrictedCCSDTQ(m.h, m.eri, m.nocc, m.constant, backend="cpu", cpu_threads=2).cpu_threads == 2
    cc = RestrictedCC(m.h, m.eri, m.nocc, m.constant, orbital_irreps=m.orbital_irreps, cpu_threads=2)
    ground = cc.solve(energy_tol=1e-12, residual_tol=1e-10)
    eom = EOMSinglet(cc, ground.amplitudes, orbital_irreps=m.orbital_irreps)
    assert eom.cpu_threads == 2 and eom._primal.cpu_threads == 2       # inherited
    assert EOMSinglet(cc, ground.amplitudes, orbital_irreps=m.orbital_irreps, cpu_threads=1).cpu_threads == 1
    dipoles, dipole_irreps = dipole_integrals(m, mol)
    response = DipoleResponse(cc, ground.amplitudes, dipoles, orbital_irreps=m.orbital_irreps,
                              dipole_irreps=dipole_irreps, cpu_threads=1)
    assert response.cpu_threads == response.engine.cpu_threads == 1


@pytest.mark.parametrize("labelled", [True, False])
def test_cuda_options_are_the_production_solver(labelled):
    """`cuda_options()` names options RestrictedCC takes; on a GPU its solve
    agrees with the CPU solver."""
    import inspect

    from itercc import RestrictedCC
    from itercc.restricted import cuda_options

    options = cuda_options(labelled=labelled)
    assert set(options) <= set(inspect.signature(RestrictedCC).parameters)
    cupy = pytest.importorskip("cupy")
    try:
        cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("no CUDA device")
    pytest.importorskip("pyscf")
    from pyscf import gto, scf

    from itercc.pyscf_interface import from_rhf

    mol = gto.M(atom="O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469", basis="6-31g", symmetry=True,
                verbose=0)
    m = from_rhf(scf.RHF(mol).run(conv_tol=1e-12), frozen=1)
    labels = m.orbital_irreps if labelled else None
    controls = dict(energy_tol=1e-11, residual_tol=1e-9, max_iter=60)
    cpu = RestrictedCC(m.h, m.eri, m.nocc, m.constant, orbital_irreps=labels).solve(**controls)
    gpu = RestrictedCC(m.h, m.eri, m.nocc, m.constant, orbital_irreps=labels, **options).solve(**controls)
    assert abs(gpu.energy - cpu.energy) < 1e-9
