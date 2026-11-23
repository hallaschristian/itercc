"""Axial (Lz-adapted) orbitals: pairing, real integrals and full-CI energies equal to real orbitals."""
import numpy as np
import pytest

from itercc import scf
from itercc._cpu_kernels import library
from itercc.axial import axial_dipoles, axial_integrals, axial_natural_orbitals, axial_orbitals
from itercc.molecule import Molecule
from itercc.selected_ci import SelectedCI, _Pairs, rotate_integrals

pytestmark = pytest.mark.skipif(library() is None, reason="selected CI needs the compiled core")

FULL = dict(epsilons=(1e-14,), new_fraction=0.0, residual_tol=1e-10)


@pytest.fixture(scope="module")
def lih():
    """LiH cc-pVDZ (19 orbitals: sigma, pi and Li delta), real and axial integrals."""
    molecule = Molecule("Li 0 0 0; H 0 0 1.6", "cc-pvdz")
    result, hcore, eri_s8 = scf.run(molecule)
    model = scf.from_scf(result, hcore, eri_s8, molecule.energy_nuc())
    axial = axial_orbitals(molecule.basis, model.mo_coeff)
    labels = np.asarray(model.orbital_irreps)
    h, eri = rotate_integrals(model.h, model.eri, axial.real, orbital_irreps=labels)
    hc, eric = axial_integrals(h, eri, axial)
    return dict(molecule=molecule, model=model, axial=axial, labels=labels, real=(h, eri), complex=(hc, eric))


def test_orbitals_pair_into_lz_eigenfunctions(lih):
    axial, labels = lih["axial"], lih["labels"]
    lz, mirror = axial.lz, axial.mirror
    assert sorted(set(np.abs(lz))) == [0, 1, 2]
    np.testing.assert_array_equal(mirror[mirror], np.arange(len(lz)))
    np.testing.assert_array_equal(lz[mirror], -lz)
    # The aligned real orbitals keep their C2v irreps: each cos/sin pair's
    # irreps multiply to A2 (B1 x B2, A1 x A2), the same nonzero label.
    products = {int(labels[i] ^ labels[mirror[i]]) for i in np.flatnonzero(lz > 0)}
    assert len(products) == 1 and 0 not in products
    np.testing.assert_allclose(axial.real.T @ axial.real, np.eye(len(lz)), atol=1e-12)
    t = axial.transform
    np.testing.assert_allclose(t.conj().T @ t, np.eye(len(lz)), atol=1e-12)
    hc, eric = lih["complex"]
    # Real, M_L-conserving, 4-fold symmetric but not 8-fold.
    np.testing.assert_allclose(eric, eric.transpose(2, 3, 0, 1), atol=1e-12)
    np.testing.assert_allclose(eric, eric.transpose(1, 0, 3, 2), atol=1e-12)
    assert np.abs(eric - eric.transpose(1, 0, 2, 3)).max() > 1e-3
    np.testing.assert_allclose(hc[np.ix_(mirror, mirror)], hc, atol=1e-12)


def test_full_ci_energies_equal_real_orbitals(lih):
    h, eri = lih["real"]
    hc, eric = lih["complex"]
    model, labels, lz = lih["model"], lih["labels"], lih["axial"].lz
    real = SelectedCI(h, eri, 4, model.constant, orbital_irreps=labels)
    cplx = SelectedCI(hc, eric, 4, model.constant, orbital_lz=lz)
    aufbau = cplx.aufbau()
    assert cplx.irrep(aufbau) == 0

    def lowest(sci, target, count):
        refs = sci.single_excitations(sci.aufbau(), irrep=target)[:4] if target else sci.aufbau()[None]
        return sci.solve(count, references=refs, spin_parity=1, **FULL).energies

    # Irrep labels of the real orbitals (the native numbering): a pi orbital's, and A2.
    pi = np.flatnonzero(lz == 1)[0]
    b1, a2 = int(labels[pi]), int(labels[pi] ^ labels[lih["axial"].mirror[pi]])
    # X 1Sigma+ (A1 / M_L = 0), the lowest 1Pi (B1 / M_L = 1).
    np.testing.assert_allclose(lowest(cplx, 0, 1), lowest(real, 0, 1), atol=1e-9)
    np.testing.assert_allclose(lowest(cplx, 1, 1), lowest(real, b1, 1), atol=1e-9)
    # The lowest 1Delta (M_L = 2) is among the A2 states (Delta and Sigma-).
    delta = lowest(cplx, 2, 1)[0]
    assert np.min(np.abs(lowest(real, a2, 3) - delta)) < 1e-9
    # M_L = 0 holds no Delta: fewer determinants than A1 in the full space.
    full_real = real.solve(1, spin_parity=1, **FULL).determinants
    full_cplx = cplx.solve(1, spin_parity=1, **FULL).determinants
    assert len(full_cplx) < len(full_real)


def test_natural_orbitals_keep_the_pairing(lih):
    hc, eric = lih["complex"]
    h, eri = lih["real"]
    model, labels, axial = lih["model"], lih["labels"], lih["axial"]
    cplx = SelectedCI(hc, eric, 4, model.constant, orbital_lz=axial.lz)
    pi = cplx.solve(1, references=cplx.single_excitations(cplx.aufbau(), irrep=1)[:4], spin_parity=1,
                    epsilons=(1e-3,))
    density = cplx.density_matrix(pi)
    rotation, occupations = axial_natural_orbitals(density, axial.lz, axial.mirror)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(len(occupations)), atol=1e-12)
    np.testing.assert_allclose(occupations[axial.mirror], occupations, atol=1e-12)
    assert abs(np.trace(density) - 4.0) < 1e-10
    # The same rotation in the axial and the aligned real orbitals: equal full-CI energies.
    hn, erin = rotate_integrals(hc, eric, rotation, axial=True)
    rotated = SelectedCI(hn, erin, 4, model.constant, orbital_lz=axial.lz)
    hr, erir = rotate_integrals(h, eri, rotation, orbital_irreps=labels)
    real = SelectedCI(hr, erir, 4, model.constant, orbital_irreps=labels)
    a = rotated.solve(1, spin_parity=1, **FULL).energies
    b = real.solve(1, spin_parity=1, **FULL).energies
    np.testing.assert_allclose(a, b, atol=1e-9)
    np.testing.assert_allclose(a, cplx.solve(1, spin_parity=1, **FULL).energies, atol=1e-9)


def test_axial_orbitals_reject_off_axis_molecules():
    molecule = Molecule("H 0 0 0; H 0.3 0 0.7", "sto-3g", symmetry=False)
    with pytest.raises(ValueError, match="z axis"):
        axial_orbitals(molecule.basis, np.eye(molecule.nao))


def test_reflection_orbits_halve_sigma_states(lih):
    """Sigma+ (mirror parity +1) roots equal the M_L = 0 roots without the
    reflection, on about half the rows and nonzeros; Sigma- (-1) roots are
    among the real orbitals' A2 states."""
    hc, eric = lih["complex"]
    h, eri = lih["real"]
    model, labels, axial = lih["model"], lih["labels"], lih["axial"]
    plain = SelectedCI(hc, eric, 4, model.constant, orbital_lz=axial.lz)
    mirror = SelectedCI(hc, eric, 4, model.constant, orbital_lz=axial.lz, orbital_mirror=axial.mirror)
    options = dict(spin_parity=1, epsilons=(1e-2, 1e-3, 1e-4), pt2_epsilon=1e-7)
    a = plain.solve(2, **options)
    b = mirror.solve(2, mirror_parity=1, **options)
    np.testing.assert_array_equal(np.sort(a.determinants.view("V16").ravel()),
                                  np.sort(b.determinants.view("V16").ravel()))
    for x, y in zip(a.stages, b.stages):
        np.testing.assert_allclose(x.energies, y.energies, atol=1e-10)
        np.testing.assert_allclose(x.pt2, y.pt2, atol=1e-12)
    # LiH's four electrons leave many determinants their own reflection
    # (0.77 here); CuF's spaces lose about half their nonzeros.
    assert b.stages[-1].nonzeros < 0.8 * a.stages[-1].nonzeros
    # Reflections: an involution that negates M_L (LiH's M_L = 0 singles are
    # sigma -> sigma, their own images: Sigma- needs pi or delta pairs).
    dets = b.determinants
    np.testing.assert_array_equal(mirror.mirrored(mirror.mirrored(dets)), dets)
    pi_single = mirror.single_excitations(mirror.aufbau(), irrep=1)[0]
    assert mirror.irrep(mirror.mirrored(pi_single[None])[0]) == -1
    # In the full M_L = 0 singlet space the Sigma+ and Sigma- functions
    # split the swap pairs and their spectra together are the pairs' spectrum.
    full = plain.solve(1, spin_parity=1, **FULL).determinants

    def spectrum(sci, t, m):
        space = sci._space(full)
        try:
            ham = sci.kernels.sci_hamiltonian_pairs(space, t, m, 0.0, 0, 2)
            count = _Pairs(sci, space, len(full), t, m).count
            dense = np.empty((count, count))
            sci.kernels.sci_ham_dense(ham, dense.ctypes.data)
            sci.kernels.sci_ham_free(ham)
        finally:
            sci.kernels.sci_space_free(space)
        return np.linalg.eigvalsh(dense) + sci.constant

    pairs = spectrum(plain, 1.0, 0.0)
    plus, minus = spectrum(mirror, 1.0, 1.0), spectrum(mirror, 1.0, -1.0)
    assert len(plus) + len(minus) == len(pairs) and len(minus) > 0
    np.testing.assert_allclose(np.sort(np.concatenate([plus, minus])), pairs, atol=1e-9)
    assert minus[0] > plus[0]
    # A Sigma- series finds the lowest Sigma- state.
    sigma_minus = mirror.solve(1, spin_parity=1, mirror_parity=-1, **FULL).energies[0]
    assert sigma_minus == pytest.approx(minus[0], abs=1e-9)


def test_axial_polarizability_equals_real_orbitals(lih):
    """Full-space polarizabilities of LiH in axial orbitals (x and z; the
    response spaces hold M_L = 0 and +-1; alpha_yy = alpha_xx), with and
    without the reflection, equal those in real orbitals."""
    h, eri = lih["real"]
    hc, eric = lih["complex"]
    model, labels, axial, molecule = lih["model"], lih["labels"], lih["axial"], lih["molecule"]
    dipoles, _ = scf.dipole_integrals(molecule, model)
    dipoles = np.array([axial.real.T @ d @ axial.real for d in dipoles])
    options = dict(epsilons=(1e-14,), tol=1e-9)
    real = SelectedCI(h, eri, 4, model.constant, orbital_irreps=labels).polarizability(dipoles, **options)[-1]
    ad = axial_dipoles(dipoles, axial)
    assert not np.any(ad[1])
    plain = SelectedCI(hc, eric, 4, model.constant, orbital_lz=axial.lz).polarizability(ad, **options)[-1]
    mirror = SelectedCI(hc, eric, 4, model.constant, orbital_lz=axial.lz, orbital_mirror=axial.mirror)
    reflected = mirror.polarizability(ad, mirror_parity=1, **options)[-1]
    expected = np.real(real.tensors_au[0])
    assert expected[0, 0] == pytest.approx(expected[1, 1], abs=1e-7)
    for result in (plain, reflected):
        np.testing.assert_allclose(np.real(result.tensors_au[0]), expected, atol=1e-7)
        assert result.energy == pytest.approx(real.energy, abs=1e-9)


def test_reflection_pt2_matches_and_samples_without_bias(lih):
    """PT2 over swap x reflection orbits equals the determinant sums; its
    semistochastic remainder (several images of one external reached from
    one internal added together) is unbiased."""
    hc, eric = lih["complex"]
    model, axial = lih["model"], lih["axial"]
    mirror = SelectedCI(hc, eric, 4, model.constant, orbital_lz=axial.lz, orbital_mirror=axial.mirror)
    options = dict(spin_parity=1, mirror_parity=1, epsilons=(3e-3,), pt2_epsilon=0.0)
    exact = mirror.solve(2, **options)
    full = SelectedCI(hc, eric, 4, model.constant, orbital_lz=axial.lz, spin_pairs=False)
    space = full._space(exact.determinants)
    try:
        pt2, norm, stats = np.zeros(2), np.zeros(2), np.zeros(2, dtype=np.int64)
        vectors = np.ascontiguousarray(exact.vectors)
        electronic = np.ascontiguousarray(exact.energies - full.constant)
        full.kernels.sci_pt2(space, vectors.ctypes.data, 2, electronic.ctypes.data, 0.0, 1, pt2.ctypes.data,
                             norm.ctypes.data, stats.ctypes.data, 2)
    finally:
        full.kernels.sci_space_free(space)
    stage = exact.stages[-1]
    np.testing.assert_allclose(stage.pt2, pt2, atol=1e-13)
    # Externals whose orbit has no Sigma+ (singlet) part carry no numerator and are not stored.
    assert 0.9 * stats[0] < stage.pt2_externals <= stats[0]
    runs = [mirror.solve(2, pt2_deterministic=1e-3, pt2_samples=40, pt2_sample_batches=400, seed=seed,
                         **options).stages[-1] for seed in (5, 5)]
    np.testing.assert_array_equal(runs[0].pt2, runs[1].pt2)
    assert np.all(runs[0].pt2_error > 0)
    assert np.all(np.abs(runs[0].pt2 - stage.pt2) < 3 * runs[0].pt2_error + 1e-12)


def test_automatic_pt2_batches_equal_one_pass(lih):
    """With 0 batches and a memory limit the PT2 probes 1/32 of the
    externals and takes the fewest passes that fit; the fixed-point sums are
    the same bits as one pass, over determinants and over orbits."""
    hc, eric = lih["complex"]
    model, axial = lih["model"], lih["axial"]
    sci = SelectedCI(hc, eric, 4, model.constant, orbital_lz=axial.lz, orbital_mirror=axial.mirror)
    exact = sci.solve(2, spin_parity=1, mirror_parity=1, epsilons=(1e-3,))
    vectors = np.ascontiguousarray(exact.vectors)
    electronic = np.ascontiguousarray(exact.energies - sci.constant)
    space = sci._space(exact.determinants)
    try:
        def run(batches, memory, group):
            sci.kernels.sci_set_pt2_memory(sci._ctx, memory)
            pt2, norm, stats = np.zeros(2), np.zeros(2), np.zeros(2, dtype=np.int64)
            if group is None:
                sci.kernels.sci_pt2(space, vectors.ctypes.data, 2, electronic.ctypes.data, 0.0, batches,
                                    pt2.ctypes.data, norm.ctypes.data, stats.ctypes.data, 3)
            else:
                sci.kernels.sci_pt2_orbits(space, vectors.ctypes.data, 2, electronic.ctypes.data, 0.0, batches,
                                           *group, pt2.ctypes.data, norm.ctypes.data, stats.ctypes.data, 3)
            return pt2, norm, stats
        for group in (None, (1.0, 0.0), (1.0, 1.0)):
            one = run(1, 0, group)
            for batches, memory in ((0, 2 << 20), (0, 0), (5, 0)):       # 2 MiB forces many passes
                other = run(batches, memory, group)
                np.testing.assert_array_equal(other[0], one[0])
                np.testing.assert_array_equal(other[1], one[1])
                assert other[2][0] == one[2][0]
    finally:
        sci.kernels.sci_space_free(space)


def test_selection_closes_the_group_in_the_kernel(lih):
    """Selection from one member per orbit, closed under the swap and the
    reflection by the kernel, finds the space the whole space finds."""
    hc, eric = lih["complex"]
    model, axial = lih["model"], lih["axial"]
    sci = SelectedCI(hc, eric, 4, model.constant, orbital_lz=axial.lz, orbital_mirror=axial.mirror)
    ground = sci.solve(2, spin_parity=1, mirror_parity=1, epsilons=(3e-3,))
    dets = ground.determinants
    cmax = np.ascontiguousarray(np.max(np.abs(ground.vectors), axis=0))
    space = sci._space(dets)
    try:
        def select(c, flags):
            c = np.ascontiguousarray(c)
            return sci._take(sci.kernels.sci_select(space, c.ctypes.data, 3e-4, flags, 3))
        reference = sci._close(select(cmax, 1), True)
        reps = sci._representatives(space, len(dets), (1, 1))
        assert 0 < reps.sum() < len(dets)
        closed = select(np.where(reps, cmax, 0.0), 3)
    finally:
        sci.kernels.sci_space_free(space)
    def as_set(rows):
        return {bytes(r) for r in np.ascontiguousarray(rows)}
    assert len(closed) == len(reference) > 0
    assert as_set(closed) == as_set(reference)
    np.testing.assert_array_equal(sci._unique(closed), closed)            # sorted and distinct
    assert as_set(sci._close(closed, True)) == as_set(closed)               # closed under the reflection


def test_natural_virtuals_keep_axial_pairs(lih):
    """Frozen natural virtuals of a selected-CI density in axial orbitals:
    pairs kept together, labels and reflection consistent; keeping every
    virtual is a rotation (same full-CI energy), fewer give a higher one."""
    from itercc.active_space import natural_virtuals

    hc, eric = lih["complex"]
    model, axial = lih["model"], lih["axial"]
    sci = SelectedCI(hc, eric, 4, model.constant, orbital_lz=axial.lz, orbital_mirror=axial.mirror)
    ground = sci.solve(1, spin_parity=1, mirror_parity=1, epsilons=(1e-3,))
    density = sci.density_matrix(ground)
    nocc, n = model.nocc, len(hc)
    occ = np.arange(nocc)
    fock = hc + 2 * eric[:, :, occ, occ].sum(-1) - np.einsum("piiq->pq", eric[:, :nocc, :nocc, :])
    energies = {}
    for count in (n - nocc, 10):
        c, lz, mirror, discarded = natural_virtuals(density, nocc, count, fock=fock, orbital_lz=axial.lz,
                                                    mirror=axial.mirror)
        assert c.shape == (n, nocc + count)
        np.testing.assert_allclose(c.T @ c, np.eye(nocc + count), atol=1e-12)
        np.testing.assert_array_equal(lz[mirror], -lz)
        np.testing.assert_array_equal(mirror[mirror], np.arange(nocc + count))
        h2, eri2 = rotate_integrals(hc, eric, c, axial=True)
        # The reference energy is unchanged (occupied orbitals untouched).
        a = np.arange(nocc)
        e_ref = model.constant + 2 * np.trace(h2[np.ix_(a, a)]) + 2 * np.einsum("iijj->", eri2[np.ix_(a, a, a, a)]) \
            - np.einsum("ijji->", eri2[np.ix_(a, a, a, a)])
        assert e_ref == pytest.approx(lih["model"].constant + 2 * np.trace(hc[np.ix_(a, a)])
                                      + 2 * np.einsum("iijj->", eric[np.ix_(a, a, a, a)])
                                      - np.einsum("ijji->", eric[np.ix_(a, a, a, a)]), abs=1e-10)
        small = SelectedCI(h2, eri2, 4, model.constant, orbital_lz=lz, orbital_mirror=mirror)
        energies[count] = small.solve(1, spin_parity=1, mirror_parity=1, **FULL).energies[0]
        if count == n - nocc:
            assert discarded == pytest.approx(0.0, abs=1e-12)
        else:
            assert discarded > 0
    full = sci.solve(1, spin_parity=1, mirror_parity=1, **FULL).energies[0]
    assert energies[n - nocc] == pytest.approx(full, abs=1e-9)
    assert energies[10] > full


def test_natural_virtuals_with_c2v_labels(lih):
    """Real orbitals: natural virtuals within each C2v irrep; keeping all is a rotation."""
    from itercc.active_space import natural_virtuals

    h, eri = lih["real"]
    model, labels = lih["model"], lih["labels"]
    sci = SelectedCI(h, eri, 4, model.constant, orbital_irreps=labels)
    density = sci.density_matrix(sci.solve(1, spin_parity=1, epsilons=(1e-3,)))
    nocc, n = model.nocc, len(h)
    c, new_labels, mirror, discarded = natural_virtuals(density, nocc, n - nocc, orbital_irreps=labels)
    assert mirror is None and discarded == pytest.approx(0.0, abs=1e-12)
    for k in range(n):
        assert np.all(labels[np.abs(c[:, k]) > 1e-12] == new_labels[k])
    h2, eri2 = rotate_integrals(h, eri, c, orbital_irreps=new_labels)
    a = SelectedCI(h2, eri2, 4, model.constant, orbital_irreps=new_labels).solve(1, spin_parity=1, **FULL)
    b = sci.solve(1, spin_parity=1, **FULL)
    assert a.energies[0] == pytest.approx(b.energies[0], abs=1e-9)
