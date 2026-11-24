import numpy as np
import pytest

from itercc import RestrictedCCSDTQ
from itercc.response import HC_EH_NM
from test_quadruples import LABELS, symmetric_model
from test_restricted import make_model


def _compiled():
    from itercc import _cpu_quadruples
    if not _cpu_quadruples.available():
        pytest.skip("compiled CPU core unavailable")


def _dipoles(n, labels=None, irreps=(0, 0, 0), seed=11):
    rng = np.random.default_rng(seed)
    out = []
    for irrep in irreps:
        m = rng.normal(scale=0.3, size=(n, n))
        m = m + m.T
        if labels is not None:
            m = np.where((labels[:, None] ^ labels) == irrep, m, 0.0)
        out.append(m)
    return np.array(out)


def _ground(solver):
    result = solver.solve(max_iter=200, energy_tol=1e-13, residual_tol=1e-11, damping=0.2,
                          diis_quadruples="host", diis_size=4)
    assert result.converged
    return result


def _fci_polarizability(h, eri, no, dipoles, omega):
    """Exact (FCI) dipole response of the lowest state of a closed-shell model."""
    fci = pytest.importorskip("pyscf.fci")
    n = len(h)
    nd = fci.cistring.num_strings(n, no)
    address, matrix = fci.direct_spin1.pspace(h, eri, n, (no, no), np=nd**2)
    full = np.zeros((nd**2, nd**2))
    full[np.ix_(address, address)] = matrix
    values, vectors = np.linalg.eigh(full)
    e0, c0 = values[0], vectors[:, 0]
    projected = full - e0 * np.eye(nd**2) + np.outer(c0, c0)      # nonsingular, |0> -> |0>
    v = []
    for mu in dipoles:
        x = fci.direct_spin1.contract_1e(mu, c0.reshape(nd, nd), n, (no, no)).reshape(-1)
        v.append(x - c0 * (c0 @ x))
    tensor = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            plus = np.linalg.solve(projected - omega * np.eye(nd**2), v[j])
            minus = np.linalg.solve(projected + omega * np.eye(nd**2), v[i])
            tensor[i, j] = v[i] @ plus + v[j] @ minus
    return tensor


@pytest.mark.parametrize("omega", [0.0, 0.21])
def test_lr_ccsdtq_is_exact_for_four_electrons(omega):
    """Four electrons: CCSDTQ is FCI, so its linear response is the exact response."""
    from itercc.response_ccsdtq import QuadruplesDipoleResponse

    _compiled()
    h, eri, no, constant = make_model(2, 4)
    eri = eri * 0.45
    solver = RestrictedCCSDTQ(h, eri, no, constant, backend="cpu")
    ground = _ground(solver)
    dipoles = _dipoles(len(h))
    response = QuadruplesDipoleResponse(solver, ground.amplitudes, dipoles)
    wavelength = np.inf if omega == 0 else HC_EH_NM / omega
    tensor = np.real(response.polarizability(wavelength, tol=1e-10).tensor_au)
    expected = _fci_polarizability(h, eri, no, dipoles, omega)
    # (LR-CCSDT misses the FCI tensor of this model by 1.6e-5.)
    np.testing.assert_allclose(tensor, expected, rtol=1e-8, atol=1e-8)


def test_lr_ccsdtq_static_matches_finite_field():
    """Six electrons (CCSDTQ is not FCI): the static response is the field
    derivative of the fixed-orbital CCSDTQ energy."""
    from itercc.response_ccsdtq import QuadruplesDipoleResponse

    _compiled()
    h, eri, no, constant = make_model(3, 4, seed=5)
    eri = eri * 0.5
    dipoles = _dipoles(len(h), seed=3)
    solver = RestrictedCCSDTQ(h, eri, no, constant, backend="cpu")
    ground = _ground(solver)
    response = QuadruplesDipoleResponse(solver, ground.amplitudes, dipoles, axes=(2,))
    zz = np.real(response.polarizability(np.inf, tol=1e-10).tensor_au)[2, 2]

    def energy(field):
        shifted = RestrictedCCSDTQ(h - field * dipoles[2], eri, no, constant, backend="cpu")
        return _ground(shifted).energy

    e0 = ground.energy
    curvature = [-(energy(f) + energy(-f) - 2 * e0) / f**2 for f in (2e-3, 1e-3)]
    expected = (4 * curvature[1] - curvature[0]) / 3
    # Agreement 1.3e-9 au; quadruples move this zz by 1e-5 au (LR-CCSDT).
    assert abs(zz - expected) < 2e-8 * abs(expected)


def test_lr_ccsdtq_symmetry_sectors_match_dense():
    """C2v-like labels: each dipole solved in its own sector (one-sided actions
    on a subgroup solver, property solves on the full-group solver) gives the
    dense tensor at a finite frequency."""
    from itercc.response_ccsdtq import QuadruplesDipoleResponse
    from itercc.spatial_symmetry import sector_labels

    _compiled()
    model = symmetric_model()
    irreps = (2, 3, 0)
    dipoles = _dipoles(len(LABELS), LABELS, irreps, seed=21)
    dense = RestrictedCCSDTQ(*model, backend="cpu")
    ground = _ground(dense)
    wavelength = HC_EH_NM / 0.15
    expected = np.real(QuadruplesDipoleResponse(dense, ground.amplitudes, dipoles)
                       .polarizability(wavelength, tol=1e-10).tensor_au)
    solvers = {k: RestrictedCCSDTQ(*model, backend="cpu", orbital_irreps=sector_labels(LABELS, k))
               for k in (0, 2, 3)}
    amplitudes = {k: s.combine_states([(1.0, ground.amplitudes, None)]) for k, s in solvers.items()}
    response = QuadruplesDipoleResponse(solvers, amplitudes, dipoles, LABELS, dipole_irreps=irreps)
    tensor = np.real(response.polarizability(wavelength, tol=1e-10).tensor_au)
    np.testing.assert_allclose(np.diag(tensor), np.diag(expected), rtol=1e-7)
    assert np.max(np.abs(tensor - np.diag(np.diag(tensor)))) == 0
    assert np.max(np.abs(expected - np.diag(np.diag(expected)))) < 1e-8
    # Only some axes: the others stay zero and need no solver.
    only_x = QuadruplesDipoleResponse({k: solvers[k] for k in (0, 2)}, amplitudes, dipoles, LABELS,
                                      dipole_irreps=irreps, axes=(0,))
    xx = np.real(only_x.polarizability(wavelength, tol=1e-10).tensor_au)
    assert abs(xx[0, 0] - tensor[0, 0]) < 1e-9 * abs(tensor[0, 0]) and not xx[1, 1] and not xx[2, 2]


def test_lr_ccsdtq_rejects_damping_and_wrong_sector_solvers():
    from itercc.response_ccsdtq import QuadruplesDipoleResponse

    _compiled()
    model = symmetric_model()
    irreps = (2, 3, 0)
    dipoles = _dipoles(len(LABELS), LABELS, irreps)
    solver = RestrictedCCSDTQ(*model, backend="cpu", orbital_irreps=LABELS)
    ground = solver.solve(max_iter=3, damping=0.2, diis_size=0).amplitudes
    with pytest.raises(ValueError, match="No solver for the dipole sectors"):
        QuadruplesDipoleResponse(solver, ground, dipoles, LABELS, dipole_irreps=irreps)
    with pytest.raises(ValueError, match="sector_labels|not a subgroup|separate"):
        QuadruplesDipoleResponse({0: solver, 2: solver}, ground, dipoles, LABELS, dipole_irreps=irreps,
                                 axes=(0,))
    response = QuadruplesDipoleResponse(solver, ground, dipoles, LABELS, dipole_irreps=irreps, axes=(2,))
    with pytest.raises(ValueError, match="real frequencies"):
        response.polarizability(500.0, damping_Eh=1e-3)
