"""Finite-field polarizabilities against analytic CCSDT response and four-electron FCI."""
import numpy as np
import pytest

from itercc import DipoleResponse, Hamiltonian, RestrictedCC, RestrictedCCSDTQ
from itercc.finite_field import finite_field_polarizability
from test_quadruples import symmetric_model
from test_response import model as response_model
from test_restricted import make_model
from test_tensor_oracle import determinant_problem

TIGHT = dict(energy_tol=1e-13, residual_tol=1e-12, diis_triples=True)


def ccsdt(eri, nocc, constant):
    return lambda h, labels: RestrictedCC(h, eri, nocc, constant, orbital_irreps=labels)


def symmetric_dipoles(labels, irreps, seed=11):
    rng = np.random.default_rng(seed)
    pairs = labels[:, None] ^ labels
    out = []
    for irrep in irreps:
        value = rng.normal(scale=0.4, size=pairs.shape)
        out.append(np.where(pairs == irrep, value + value.T, 0.0))
    return np.asarray(out)


def exact(h, eri, nocc, constant, dipoles):
    """Sum-over-states static polarizability and dipole <0|mu|0> of the full CI."""
    matrix, _ = determinant_problem(Hamiltonian.restricted(h, eri, nocc, constant), nocc, nocc)
    energies, states = np.linalg.eigh(matrix)
    operators = [determinant_problem(Hamiltonian.restricted(d, np.zeros_like(eri), nocc), nocc, nocc)[0]
                 for d in dipoles]
    moments = np.asarray([states[:, 1:].T @ op @ states[:, 0] for op in operators])
    gaps = energies[1:] - energies[0]
    return (moments * (2 / gaps)) @ moments.T, np.asarray([states[:, 0] @ op @ states[:, 0] for op in operators])


def test_ccsdt_tensor_matches_analytic_response():
    h, eri, constant, dipoles = response_model(2, 2)
    solver = RestrictedCC(h, eri, 2, constant)
    ground = solver.solve(**TIGHT)
    analytic = DipoleResponse(solver, ground.amplitudes, dipoles).polarizability(np.inf, tol=1e-11).tensor_au
    found = finite_field_polarizability(ccsdt(eri, 2, constant), h, dipoles, fields=(1e-3, 2e-3),
                                        solve_options=TIGHT)
    assert found.converged and len(found.points) == 1 + 6 * 4            # zero, 3 axes, 3 pair lines
    assert abs(found.energy_Eh - ground.energy) < 1e-12
    np.testing.assert_allclose(found.tensor_au, analytic, atol=2e-8, rtol=0)
    np.testing.assert_allclose(found.tensor_au, found.tensor_au.T, atol=0, rtol=0)
    # The extrapolation removes an F**2 term that the centered values still carry.
    assert np.max(np.abs(found.centered_au[1] - analytic)) > 10 * np.max(np.abs(found.tensor_au - analytic))


@pytest.mark.parametrize("labelled", [False, True])
def test_ccsdtq_matches_four_electron_fci(labelled):
    labels = np.array([0, 1, 0, 2, 1])
    h, eri, no, constant = symmetric_model(labels, no=2)
    dipoles = symmetric_dipoles(labels, (1, 2, 0))
    alpha, mu = exact(h, eri, no, constant, dipoles)

    def factory(h, orbital_irreps):
        return RestrictedCCSDTQ(h, eri, no, constant, block_size=2, backend="cpu", orbital_irreps=orbital_irreps)

    seen = []
    found = finite_field_polarizability(factory, h, dipoles, orbital_irreps=labels if labelled else None,
                                        fields=(1e-3, 2e-3), callback=seen.append,
                                        solve_options=dict(damping=0.2, energy_tol=1e-13, residual_tol=1e-11))
    assert found.converged and seen == list(found.points)
    np.testing.assert_allclose(found.tensor_au, alpha, atol=2e-7, rtol=0)
    np.testing.assert_allclose(found.dipole_au, mu, atol=1e-9, rtol=0)
    if labelled:
        # x (irrep 1) and y (irrep 2) need +F only; z (irrep 0) both signs; no pair lines.
        assert [p.direction for p in found.points] == ["zero", "x", "x", "y", "y", "z", "z", "z", "z"]
        assert found.tensor_au[0, 1] == found.tensor_au[0, 2] == found.tensor_au[1, 2] == 0.0
        assert found.dipole_au[0] == found.dipole_au[1] == 0.0
    else:
        assert len(found.points) == 25


def test_symmetry_and_warm_starts_keep_the_tensor():
    labels = np.array([0, 2, 3, 1, 0, 3, 2, 0, 1])
    h, eri, no, constant = symmetric_model(labels)
    dipoles = symmetric_dipoles(labels, (1, 1, 0))
    results = {}
    for name, options in {"dense": dict(), "labelled": dict(orbital_irreps=labels),
                          "cold": dict(orbital_irreps=labels, warm_start=False)}.items():
        results[name] = finite_field_polarizability(ccsdt(eri, no, constant), h, dipoles,
                                                    solve_options=TIGHT, **options)
    dense, labelled, cold = results["dense"], results["labelled"], results["cold"]
    # Each point stops at its own iterate, and the energy errors are divided by F**2.
    np.testing.assert_allclose(labelled.tensor_au, dense.tensor_au, atol=5e-7, rtol=0)
    np.testing.assert_allclose(cold.tensor_au, labelled.tensor_au, atol=5e-7, rtol=0)
    assert labelled.tensor_au[0, 1] != 0.0                               # x and y share irrep 1
    assert labelled.tensor_au[0, 2] == labelled.tensor_au[1, 2] == 0.0
    assert [p.direction for p in labelled.points].count("xy") == 2      # +F only
    assert len(labelled.points) == 1 + 2 + 2 + 4 + 2 < len(dense.points) == 25
    field_evaluations = [sum(p.evaluations for p in r.points[1:]) for r in (labelled, cold)]
    assert field_evaluations[0] < 0.8 * field_evaluations[1]


def test_requested_components_and_failures():
    h, eri, constant, dipoles = response_model(2, 2)
    found = finite_field_polarizability(ccsdt(eri, 2, constant), h, dipoles, components=("zx",),
                                        fields=(1e-3,), solve_options=TIGHT)
    assert sorted({p.direction for p in found.points}) == ["x", "xz", "z", "zero"]
    assert np.isnan(found.tensor_au[1]).all() and np.isnan(found.dipole_au[1])
    assert np.isfinite(found.tensor_au[[0, 0, 2, 2], [0, 2, 0, 2]]).all()
    np.testing.assert_array_equal(found.extrapolation_au[[0, 2], [0, 2]], 0.0)    # one field
    for bad in (dict(fields=()), dict(fields=(1e-3, 1e-3)), dict(fields=(-1e-3,)), dict(components=("xw",)),
                dict(solve_options=dict(initial=None)), dict(dipole_irreps=(0, 1), orbital_irreps=[0] * 4)):
        with pytest.raises(ValueError):
            finite_field_polarizability(ccsdt(eri, 2, constant), h, dipoles, **bad)
    with pytest.raises(RuntimeError, match="did not converge"):
        finite_field_polarizability(ccsdt(eri, 2, constant), h, dipoles, solve_options=dict(max_iter=2))
