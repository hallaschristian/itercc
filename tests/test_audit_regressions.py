import os
import gc
from unittest.mock import patch

import numpy as np
import pytest

from itercc import RestrictedCC
from itercc.checkpoint_io import write_array_archive
from itercc.orbital_rotation import rotate_amplitudes
from itercc.spatial_symmetry import project_amplitudes, project_integrals
from test_restricted import make_model, trial_amplitudes


def test_nonfinite_late_triples_tile_cannot_converge():
    solver = RestrictedCC(*make_model(3, 2))
    residuals = tuple(np.zeros(shape) for shape in solver.shapes)
    residuals[2][-1, 0, 0, 0] = np.nan
    with patch.object(solver, "_residuals", return_value=residuals):
        with pytest.raises(FloatingPointError, match="Nonfinite"):
            solver.solve(max_iter=2)


def test_rotation_rejects_nonfinite_checkpoint():
    solver = RestrictedCC(*make_model(2, 2))
    values = trial_amplitudes(solver, 0.01, True)
    values[2][-1, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        rotate_amplitudes(values, np.eye(2), np.eye(2))


def test_failed_checkpoint_preserves_prior_archive(tmp_path):
    target = tmp_path / "state.npz"
    write_array_archive(target, {"amplitudes": np.arange(3)})
    original = target.read_bytes()

    def fail(stream, **unused):
        stream.write(b"incomplete")
        raise OSError("simulated write failure")

    with patch("numpy.savez", fail), pytest.raises(OSError):
        write_array_archive(target, {"amplitudes": np.arange(5)})
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.skipif(os.environ.get("ITERCC_TEST_CUDA") != "1", reason="CUDA test")
def test_cached_allocations_preserve_solution_and_release_after_failure():
    pytest.importorskip("cupy")
    model = make_model(3, 3)
    settings = dict(energy_tol=1e-12, residual_tol=1e-10, diis_size=3, diis_triples=True)
    expected = RestrictedCC(*model, residual_backend="cuda").solve(**settings)
    candidate = RestrictedCC(*model, residual_backend="cuda", cuda_pool_cache=True)
    actual = candidate.solve(**settings)
    assert actual.converged and expected.converged
    assert abs(actual.energy - expected.energy) < 1e-12
    assert candidate._pool.total_bytes() > 0
    candidate.release_device_memory()
    assert candidate._pool.total_bytes() == 0
    def fail(*unused):
        raise RuntimeError("failed evaluation")

    with patch.object(candidate, "_evaluate", fail):
        with pytest.raises(RuntimeError):
            candidate.residuals(actual.amplitudes)
    gc.collect()
    candidate.release_device_memory()
    assert candidate._pool.total_bytes() == 0


@pytest.mark.skipif(os.environ.get("ITERCC_TEST_CUDA") != "1", reason="CUDA test")
def test_axial_rejects_unsupported_irrep_encoding():
    pytest.importorskip("cupy")
    with pytest.raises(ValueError, match="C2v"):
        RestrictedCC(*make_model(2, 2), orbital_irreps=np.array([0, 4, 0, 4]),
                     residual_backend="cuda", cuda_pair_fusion=True, cuda_ladder_columns=True,
                     cuda_axial=(np.zeros(2, dtype=int), np.arange(2)))


@pytest.mark.skipif(os.environ.get("ITERCC_TEST_CUDA") != "1", reason="CUDA test")
def test_caller_label_mutation_cannot_change_existing_solver():
    pytest.importorskip("cupy")
    h, eri, no, constant = make_model(3, 3)
    labels = np.array([0, 1, 1, 0, 1, 1])
    h, eri, _ = project_integrals(h, eri, labels, tolerance=1)
    baseline = RestrictedCC(h, eri, no, constant)
    initial, _ = project_amplitudes(trial_amplitudes(baseline, 0.02, True), labels, tolerance=1)
    candidate = RestrictedCC(h, eri, no, constant, orbital_irreps=labels,
                            residual_backend="cuda", cuda_layout_fusion=True,
                            cuda_feedback_columns=True, cuda_assembly_columns=True)
    labels[:] = 0
    for actual, expected in zip(candidate.residuals(initial), baseline.residuals(initial)):
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)


def test_diis_falls_back_to_its_newest_entries():
    """A dependent history (here a repeated error) extrapolates from the
    newest entries, the older ones weighted zero, instead of a plain step."""
    from itercc.solver import _DIIS

    diis = _DIIS(4, np)
    rng = np.random.default_rng(2)
    first, second = rng.normal(size=6), rng.normal(size=6)
    for error in (first, first.copy(), second):
        weights = diis._append(np.zeros(6), error, np.dot)
    assert weights is not None and weights[0] == 0.0
    assert abs(weights.sum() - 1.0) < 1e-12
    full = _DIIS(4, np)
    for error in (first, second):
        expected = full._append(np.zeros(6), error, np.dot)
    np.testing.assert_allclose(weights[1:], expected, rtol=1e-12)


def test_state_swap_removes_a_linked_state_but_not_its_target(tmp_path):
    from itercc.checkpoint_io import read_array_directory, write_array_directory

    target = tmp_path / "scratch" / "state_0"
    write_array_directory(target, dict(x=np.arange(3.0)), dict(n=1))
    link = tmp_path / "run" / "state_0"
    link.parent.mkdir()
    link.symlink_to(target, target_is_directory=True)
    write_array_directory(link, dict(x=np.arange(4.0)), dict(n=2))
    assert not link.is_symlink() and not list(link.parent.glob(".state_0.previous*"))
    assert read_array_directory(target)[1] == dict(n=1)          # the old data stays
    assert read_array_directory(link)[1] == dict(n=2)
