"""Selected CI against independent full CI: the spin-orbital determinant oracle and PySCF."""
from itertools import combinations

import numpy as np
import pytest

from itercc import Hamiltonian
from itercc._cpu_kernels import library
from itercc.response import HC_EH_NM
from itercc.selected_ci import SelectedCI, SelectedCIResponse, extrapolate_gap, rotate_integrals
from test_quadruples import LABELS, symmetric_model
from test_tensor_oracle import determinant_problem

pytestmark = pytest.mark.skipif(library() is None, reason="selected CI needs the compiled core")

LOOSE = dict(epsilons=(1e-14,), new_fraction=0.0, residual_tol=1e-10)


@pytest.fixture(scope="module")
def model():
    labels = LABELS[:7]
    h, eri, no, constant = symmetric_model(labels=labels, no=2)
    return h, eri, no, constant, labels


def oracle_spectrum(model, irrep):
    """Eigenvalues of the M_S = 0 determinants of one irrep (oracle convention)."""
    h, eri, no, constant, labels = model
    matrix, masks = determinant_problem(Hamiltonian.restricted(h, eri, no, constant), no, no)
    keep = [i for i, mask in enumerate(masks)
            if np.bitwise_xor.reduce([labels[p // 2] for p in range(2 * len(h)) if mask >> p & 1]) == irrep]
    return np.linalg.eigvalsh(matrix[np.ix_(keep, keep)])


def solver(model, **options):
    h, eri, no, constant, labels = model
    return SelectedCI(h, eri, 2 * no, constant, orbital_irreps=labels, **options)


def references(sci, irrep):
    if irrep == 0:
        return sci.aufbau()[None]
    return sci.single_excitations(sci.aufbau(), irrep=irrep)[:2]


@pytest.mark.parametrize("irrep", [0, 3])
def test_full_space_limit_equals_full_ci(model, irrep):
    sci = solver(model)
    exact = oracle_spectrum(model, irrep)
    refs = references(sci, irrep)
    every = sci.solve(4, references=refs, **LOOSE)
    np.testing.assert_allclose(every.energies, exact[:4], atol=1e-10)
    found = []
    for parity, s2 in ((1, (0.0, 6.0)), (-1, (2.0, 12.0))):
        res = sci.solve(2, references=refs, spin_parity=parity, **LOOSE)
        assert all(min(abs(x - s) for s in s2) < 1e-8 for x in res.stages[-1].s2)
        for e in res.energies:
            assert np.min(np.abs(exact - e)) < 1e-10
        found.extend(res.energies)
    # The lowest state of each parity is among the lowest of the irrep.
    assert np.min(found) == pytest.approx(exact[0], abs=1e-10)


def test_results_do_not_depend_on_threads(model):
    one = solver(model, cpu_threads=1).solve(2, spin_parity=1, epsilons=(1e-2, 1e-3), pt2_epsilon=1e-6)
    many = solver(model, cpu_threads=3).solve(2, spin_parity=1, epsilons=(1e-2, 1e-3), pt2_epsilon=1e-6)
    np.testing.assert_array_equal(one.determinants, many.determinants)
    for a, b in zip(one.stages, many.stages):
        np.testing.assert_array_equal(a.energies, b.energies)
        np.testing.assert_array_equal(a.pt2, b.pt2)


def test_pt2_equals_brute_force_and_batches(model):
    h, eri, no, constant, labels = model
    sci = solver(model)
    small = sci.solve(2, spin_parity=1, epsilons=(3e-2,), pt2_epsilon=0.0)
    stage = small.stages[-1]
    full = sci.solve(1, spin_parity=1, **LOOSE).determinants
    index = {bytes(row): i for i, row in enumerate(full)}
    position = np.array([index[bytes(row)] for row in small.determinants])
    space = sci._space(full)
    ham = sci.kernels.sci_hamiltonian(space, 0.0, 0, 2)
    try:
        diagonal = np.empty(len(full))
        sci.kernels.sci_ham_diagonal(ham, diagonal.ctypes.data)
        outside = np.ones(len(full), bool)
        outside[position] = False
        for k in range(2):
            c = np.zeros(len(full))
            c[position] = small.vectors[k]
            y = np.empty(len(full))
            sci.kernels.sci_matvec(ham, c.ctypes.data, y.ctypes.data, 2)
            e = stage.energies[k] - constant
            assert stage.pt2[k] == pytest.approx(np.sum(y[outside] ** 2 / (e - diagonal[outside])), abs=1e-14)
            norm = np.sum((y[outside] / (e - diagonal[outside])) ** 2)
            assert stage.pt2_norm[k] == pytest.approx(norm, abs=1e-13)
            assert stage.renormalized_pt2[k] == pytest.approx(stage.pt2[k] / (1 + norm), abs=1e-14)
    finally:
        sci.kernels.sci_ham_free(ham)
        sci.kernels.sci_space_free(space)
    batched = sci.solve(2, spin_parity=1, epsilons=(3e-2,), pt2_epsilon=0.0, pt2_batches=3)
    np.testing.assert_array_equal(batched.stages[-1].pt2, stage.pt2)


def test_semistochastic_pt2_is_unbiased_and_seeded(model):
    sci = solver(model)
    options = dict(spin_parity=1, epsilons=(3e-2,))
    exact = sci.solve(2, pt2_epsilon=0.0, **options).stages[-1]
    runs = [sci.solve(2, pt2_epsilon=0.0, pt2_deterministic=5e-3, pt2_samples=30, pt2_sample_batches=40,
                      seed=seed, **options).stages[-1] for seed in (1, 1, 2)]
    np.testing.assert_array_equal(runs[0].pt2, runs[1].pt2)
    for stage in runs[1:]:
        assert np.all(stage.pt2_error > 0)
        assert np.all(np.abs(stage.pt2 - exact.pt2) < 5 * stage.pt2_error + 1e-12)
        assert np.all(np.abs(stage.pt2_norm - exact.pt2_norm) < 0.2 * exact.pt2_norm)


def test_extended_hamiltonian_equals_a_fresh_build(model):
    sci = solver(model)
    dets = sci.solve(1, spin_parity=1, **LOOSE).determinants
    part = len(dets) // 3
    rng = np.random.default_rng(5)
    x = rng.normal(size=len(dets))
    products = []
    small = sci._space(dets[:part])
    first = sci.kernels.sci_hamiltonian(small, 1e-15, 0, 2)
    space = sci._space(dets)
    try:
        for ham in (sci.kernels.sci_hamiltonian_extend(space, first, part, 1e-15, 0, 2),
                    sci.kernels.sci_hamiltonian(space, 1e-15, 0, 2)):
            y = np.empty(len(dets))
            sci.kernels.sci_matvec(ham, x.ctypes.data, y.ctypes.data, 2)
            products.append((y, int(sci.kernels.sci_ham_nnz(ham))))
            sci.kernels.sci_ham_free(ham)
    finally:
        sci.kernels.sci_ham_free(first)
        sci.kernels.sci_space_free(small)
        sci.kernels.sci_space_free(space)
    assert products[0][1] == products[1][1]
    np.testing.assert_array_equal(products[0][0], products[1][0])


def test_single_precision_and_block_products(model):
    sci = solver(model)
    dets = sci.solve(1, spin_parity=1, **LOOSE).determinants
    n = len(dets)
    rng = np.random.default_rng(7)
    block = np.ascontiguousarray(rng.normal(size=(n, 5)))
    space = sci._space(dets)
    double = sci.kernels.sci_hamiltonian(space, 1e-15, 0, 3)
    small = sci._space(dets[:n // 2])
    half = sci.kernels.sci_hamiltonian(small, 1e-15, 0, 3)
    single = sci.kernels.sci_hamiltonian_extend(space, half, n // 2, 1e-15, 1, 3)   # double rows -> float
    try:
        def product(ham, x):
            x = np.ascontiguousarray(x)
            y = np.empty_like(x)
            k = 1 if x.ndim == 1 else x.shape[1]
            sci.kernels.sci_matmat(ham, k, x.ctypes.data, y.ctypes.data, 3)
            return y
        reference = np.stack([product(double, block[:, r]) for r in range(5)], axis=1)
        for k in (1, 2, 3, 4, 5):     # one pass for k vectors equals k single products
            np.testing.assert_array_equal(product(double, block[:, :k]), reference[:, :k])
        np.testing.assert_allclose(product(single, block), reference, rtol=0, atol=1e-6 * np.abs(reference).max())
        assert sci.kernels.sci_ham_bytes(single) < sci.kernels.sci_ham_bytes(double)
    finally:
        for ham in (double, half, single):
            sci.kernels.sci_ham_free(ham)
        sci.kernels.sci_space_free(space)
        sci.kernels.sci_space_free(small)
    exact = sci.solve(1, spin_parity=1, **LOOSE).energies[0]
    fast = solver(model, hamiltonian_precision="single").solve(1, spin_parity=1, **LOOSE).energies[0]
    assert abs(fast - exact) < 1e-7


def test_resume_from_a_checkpoint_reproduces_the_series(model):
    sci = solver(model)
    options = dict(spin_parity=1, pt2_epsilon=1e-6)
    whole = sci.solve(2, epsilons=(3e-2, 1e-2, 3e-3), **options)
    saved = {}
    sci.solve(2, epsilons=(3e-2,), checkpoint=lambda st, d, v: saved.update(dets=d.copy(), vectors=v.copy()),
              **options)
    rest = sci.solve(2, epsilons=(1e-2, 3e-3), initial=(saved["dets"], saved["vectors"]), **options)
    np.testing.assert_array_equal(rest.determinants, whole.determinants)
    for a, b in zip(rest.stages, whole.stages[1:]):
        np.testing.assert_array_equal(a.energies, b.energies)
        np.testing.assert_array_equal(a.pt2, b.pt2)


def test_determinant_cap_ends_the_series(model):
    sci = solver(model)
    free = sci.solve(1, spin_parity=1, epsilons=(3e-2, 1e-3))
    cap = free.stages[0].determinants + 1
    capped = sci.solve(1, spin_parity=1, epsilons=(3e-2, 1e-3, 1e-4), max_determinants=cap)
    assert len(capped.stages) == 1 and len(capped.determinants) <= cap


def test_density_matrix_and_natural_orbitals(model):
    h, eri, no, constant, labels = model
    sci = solver(model)
    res = sci.solve(1, spin_parity=1, **LOOSE)
    rdm = sci.density_matrix(res)
    assert np.trace(rdm) == pytest.approx(2 * no, abs=1e-10)
    np.testing.assert_allclose(rdm, rdm.T, atol=1e-14)
    assert np.all(np.abs(rdm[labels[:, None] != labels[None]]) < 1e-14)
    occupations, coefficients, new_labels = sci.natural_orbitals(res)
    assert np.all(np.diff(occupations) <= 1e-12) and occupations.sum() == pytest.approx(2 * no)
    h2, eri2 = rotate_integrals(h, eri, coefficients)
    rotated = SelectedCI(h2, eri2, 2 * no, constant, orbital_irreps=new_labels)
    again = rotated.solve(1, spin_parity=1, **LOOSE)
    assert again.energies[0] == pytest.approx(res.energies[0], abs=1e-10)


def test_extrapolation_approaches_full_ci(model):
    sci = solver(model)
    exact = oracle_spectrum(model, 0)
    ground = sci.solve(1, spin_parity=1, epsilons=(2e-2, 1e-2, 5e-3), pt2_epsilon=0.0)
    variational = ground.energies[0] - exact[0]
    estimate, _ = ground.extrapolate(0)
    assert abs(estimate - exact[0]) < abs(variational)
    excited = sci.solve(1, references=references(sci, 3), spin_parity=1, epsilons=(2e-2, 1e-2, 5e-3),
                        pt2_epsilon=0.0)
    gap, _ = extrapolate_gap(ground, excited)
    assert np.isfinite(gap)


def sum_over_states(model, dipoles, frequency):
    """Full-CI alpha_ij(z) and <0|mu|0> from the oracle's eigenstates (z = w + i gamma)."""
    h, eri, no, constant, labels = model
    matrix, _ = determinant_problem(Hamiltonian.restricted(h, eri, no, constant), no, no)
    energies, states = np.linalg.eigh(matrix)
    operators = [determinant_problem(Hamiltonian.restricted(d, np.zeros_like(eri), no), no, no)[0] for d in dipoles]
    moments = np.asarray([states[:, 1:].T @ op @ states[:, 0] for op in operators])
    gaps = energies[1:] - energies[0]
    weights = 1 / (gaps - frequency) + 1 / (gaps + frequency)
    return (moments * weights) @ moments.T, np.asarray([states[:, 0] @ op @ states[:, 0] for op in operators])


def test_polarizability_full_space_equals_full_ci(model):
    from test_finite_field import symmetric_dipoles

    h, eri, no, constant, labels = model
    dipoles = symmetric_dipoles(labels, (3, 2, 0))
    sci = solver(model)
    results = sci.polarizability(dipoles, (np.inf, HC_EH_NM / 0.05), epsilons=(1e-2, 1e-14), tol=1e-11)
    final = results[-1]
    assert results[0].determinants < final.determinants
    for k, w in enumerate((0.0, 0.05)):
        exact, moment = sum_over_states(model, dipoles, w)
        assert final.polarizabilities[k].frequency_Eh == pytest.approx(w)
        np.testing.assert_allclose(final.tensors_au[k], exact, atol=1e-8)
        np.testing.assert_allclose(final.dipole_au, moment, atol=1e-9)
    # components of different irreps couple through no state
    assert final.tensors_au[0][0, 1] == 0 and final.tensors_au[0][0, 2] == 0


def test_polarizability_series_continues_from_a_space(model):
    from test_finite_field import symmetric_dipoles

    h, eri, no, constant, labels = model
    dipoles = symmetric_dipoles(labels, (3, 2, 0))
    sci = solver(model)
    first = sci.polarizability(dipoles, epsilons=(3e-2,))[-1]
    more = sci.polarizability(dipoles, epsilons=(1e-14,), initial=first.space)[-1]
    np.testing.assert_array_equal(more.space[:first.determinants], first.space)
    exact, _ = sum_over_states(model, dipoles, 0.0)
    np.testing.assert_allclose(more.tensors_au[0], exact, atol=1e-7)


def test_damped_response_in_a_fixed_space_equals_full_ci(model):
    from test_finite_field import symmetric_dipoles

    h, eri, no, constant, labels = model
    dipoles = symmetric_dipoles(labels, (3, 2, 0))
    sci = solver(model)
    # the full space of the irreps the dipoles reach from the ground state
    full = sci.polarizability(dipoles, epsilons=(1e-14,))[-1].space
    response = SelectedCIResponse(sci, dipoles, full, spin_parity=1, residual_tol=1e-11)
    try:
        for w, gamma in ((0.3, 0.02), (0.0, 0.01)):
            exact, _ = sum_over_states(model, dipoles, complex(w, gamma))
            found = response.polarizability(HC_EH_NM / w if w else np.inf, damping_Eh=gamma, tol=1e-11)
            np.testing.assert_allclose(found.tensor_au, exact, atol=1e-8)
            assert all(r["relative_residual"] <= 1e-11 for r in found.solves)
        # a second space with the first as its prefix extends its Hamiltonian
        small = sci.polarizability(dipoles, epsilons=(3e-2,))[0].space
        first = SelectedCIResponse(sci, dipoles, small, spin_parity=1)
        rest = full[~np.isin(full.view([("w", np.uint64, (full.shape[1],))]).ravel(),
                             small.view([("w", np.uint64, (small.shape[1],))]).ravel())]
        grown = SelectedCIResponse(sci, dipoles, np.concatenate([small, rest]), spin_parity=1, previous=first,
                                   residual_tol=1e-11)
        assert first._ham is None
        exact, _ = sum_over_states(model, dipoles, 0.0)
        np.testing.assert_allclose(grown.polarizability(np.inf, tol=1e-11).tensor_au.real, exact, atol=1e-8)
        grown.close()
    finally:
        response.close()


def test_fixed_space_finite_field_matches_linear_response(model):
    from itercc.finite_field import finite_field_polarizability
    from test_finite_field import symmetric_dipoles

    h, eri, no, constant, labels = model
    dipoles = symmetric_dipoles(labels, (3, 2, 0))
    sci = solver(model)
    space = sci.polarizability(dipoles, epsilons=(3e-2,))[0].space
    response = sci.polarizability(dipoles, spaces=[space], tol=1e-12)[0]
    found = finite_field_polarizability(sci.field_solver_factory(space, residual_tol=1e-11), h, dipoles,
                                        orbital_irreps=labels, fields=(1e-3, 2e-3))
    np.testing.assert_allclose(found.tensor_au, response.tensors_au[0], atol=2e-6)
    # E_var + E_PT2 at each field: the space's PT2-corrected tensor, several times closer to full CI
    corrected = finite_field_polarizability(sci.field_solver_factory(space, residual_tol=1e-11, pt2_epsilon=0.0),
                                            h, dipoles, orbital_irreps=labels, fields=(1e-3, 2e-3),
                                            components=("xx", "zz"))
    exact, _ = sum_over_states(model, dipoles, 0.0)
    for k in (0, 2):
        assert abs(corrected.tensor_au[k, k] - exact[k, k]) < 0.25 * abs(response.tensors_au[0][k, k] - exact[k, k])
    # screened PT2 (pt2_epsilon > 0): a fixed screen keeps the same terms at every field, so
    # the centered second differences at two strengths agree (the field-following screen jumps)
    from itercc.selected_ci import SelectedCIResponse
    fixed = SelectedCIResponse(sci, dipoles, space, spin_parity=1)
    screen = fixed.field_screen(4e-3)
    fixed.close()
    smooth = finite_field_polarizability(
        sci.field_solver_factory(space, residual_tol=1e-11, pt2_epsilon=2e-3, pt2_screen=screen), h, dipoles,
        orbital_irreps=labels, fields=(2e-3, 4e-3), components=("xx", "zz"))
    for k in (0, 2):
        assert abs(smooth.centered_au[0][k, k] - smooth.centered_au[1][k, k]) < 1e-4


def test_water_against_pyscf_full_ci():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import fci, gto, scf
    from pyscf.fci import direct_spin1_symm
    from itercc.pyscf_interface import from_rhf

    mol = gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", basis="6-31g",
                symmetry="c2v", verbose=0)
    mf = scf.RHF(mol).run()
    model = from_rhf(mf, frozen=1)
    n, ne = len(model.h), 2 * model.nocc
    sci = SelectedCI(model.h, model.eri, ne, model.constant, orbital_irreps=model.orbital_irreps)
    for irrep, parity, ss in ((0, 1, 0.0), (2, -1, 2.0)):
        exact = direct_spin1_symm.FCI(mol)
        exact.orbsym, exact.wfnsym, exact.nroots, exact.conv_tol = model.orbital_irreps, irrep, 2, 1e-12
        exact.max_cycle = 400
        fci.addons.fix_spin_(exact, ss=ss, shift=0.5)
        energies, vectors = exact.kernel(model.h, model.eri, n, ne)
        refs = None if irrep == 0 else sci.single_excitations(sci.aufbau(), irrep=irrep)[:4]
        res = sci.solve(1, references=refs, spin_parity=parity, epsilons=(1e-4, 1e-14), new_fraction=0.0,
                        residual_tol=1e-10)
        assert res.energies[0] - model.constant == pytest.approx(np.min(energies), abs=1e-9)
        if irrep == 0:
            ground = vectors[int(np.argmin(energies))]
            rdm = sci.density_matrix(res)
            np.testing.assert_allclose(rdm, fci.direct_spin1.make_rdm1(ground, n, ne), atol=1e-7)
