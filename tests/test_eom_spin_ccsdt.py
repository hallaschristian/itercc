"""Spin-integrated EOM-CCSDT (singlet and triplet, right and left)."""
from itertools import permutations, product

import numpy as np
import pytest

from itercc import EOMSinglet, Hamiltonian, RestrictedCC
from itercc._spin_ccsdt_terms import SAME_SPIN, _amplitude, jacobian_terms, parity, residual_terms
from itercc.eom_spin_ccsdt import SpinEOMCCSDT, dense_triples, ground_blocks
from itercc.eom_spin_quadruples import _restricted_amplitude
from itercc.solver import TensorCC
from test_quadruples import symmetric_model

A, B = slice(0, None, 2), slice(1, None, 2)
LABELS = np.array([0, 2, 3, 1, 0, 3, 2, 0, 1])


def antisymmetrize(x, block):
    for group in SAME_SPIN[block]:
        total = np.zeros_like(x)
        perms = list(permutations(group))
        for p in perms:
            axes = list(range(x.ndim))
            for src, dst in zip(group, p):
                axes[src] = dst
            total += parity([group.index(q) for q in p]) * x.transpose(axes)
        x = total / len(perms)
    return x


def apply_terms(terms, arrays, no, n, p):
    """Reference NumPy evaluation of block terms (direction factors scaled by
    the parity when reached by a spin flip)."""
    o, v = slice(no), slice(no, n)
    out = {}
    for t in terms:
        operands = []
        for f in t.factors:
            x = arrays[f.tensor][f.block]
            if f.tensor in ("f", "g"):
                x = x[tuple(o if c == "o" else v for c in f.space)]
            if f.flipped and f.tensor.startswith("r"):
                x = p * x
            operands.append(x)
        spec = ",".join(f.letters for f in t.factors) + "->" + t.letters
        value = t.coefficient * np.einsum(spec, *operands, optimize=True)
        out[t.output] = out[t.output] + value if t.output in out else value
    return {k: antisymmetrize(x, k.split(":")[1]) for k, x in out.items()}


@pytest.fixture(scope="module")
def model():
    h, eri, no, constant = symmetric_model(LABELS, no=3)
    solver = RestrictedCC(h, eri, no, constant)
    result = solver.solve(energy_tol=1e-12, residual_tol=1e-11, diis_triples=True)
    assert result.converged
    return h, eri, no, constant, solver, result.amplitudes


def spin_orbital(model):
    h, eri, no, constant, solver, amps = model
    nv = len(h) - no
    cc = TensorCC(Hamiltonian.restricted(h, eri, no, constant), rank=3)
    t = (_restricted_amplitude(np, amps[0].T, no, nv, 1),
         _restricted_amplitude(np, amps[1].transpose(2, 3, 0, 1), no, nv, 2),
         _restricted_amplitude(np, dense_triples(solver, amps[2]), no, nv, 3))
    return cc, t


def test_ground_blocks_and_residual_terms(model):
    h, eri, no, constant, solver, amps = model
    cc, t = spin_orbital(model)
    ground = ground_blocks(solver, amps)
    f, g = cc._dressed(t[0])
    np.testing.assert_allclose(ground["f"]["a"], f[A, A], atol=1e-14, rtol=0)
    np.testing.assert_allclose(ground["g"]["aaaa"], g[A, A, A, A], atol=1e-14, rtol=0)
    np.testing.assert_allclose(ground["g"]["abab"], g[A, B, A, B], atol=1e-14, rtol=0)
    np.testing.assert_allclose(ground["t3"]["aab"], t[2][A, A, B, A, A, B], atol=1e-15, rtol=0)
    # Perturbed amplitudes: the spin-integrated residual equals the dense one.
    rng = np.random.default_rng(4)
    shifted = tuple(x + 0.05 * _restricted_amplitude(np, s, no, len(h) - no, r) for x, s, r in zip(
        t, (rng.normal(size=a.shape) for a in (amps[0].T,)), (1,)))
    t1 = shifted[0]
    f, g = cc._dressed(t1)
    arrays = {"f": {"a": f[A, A]}, "g": {"aaaa": g[A, A, A, A], "abab": g[A, B, A, B]},
              "t2": {"aa": t[1][A, A, A, A], "ab": t[1][A, B, A, B]},
              "t3": {"aaa": t[2][A, A, A, A, A, A], "aab": t[2][A, A, B, A, A, B]}}
    got = apply_terms(residual_terms(), arrays, no, len(h), 1)
    r1, r2, r3 = cc._residuals((t1, t[1], t[2]))
    want = {"1:a": r1[A, A], "2:aa": r2[A, A, A, A], "2:ab": r2[A, B, A, B],
            "3:aaa": r3[A, A, A, A, A, A], "3:aab": r3[A, A, B, A, A, B]}
    for key in want:
        np.testing.assert_allclose(got[key], want[key], atol=1e-13, rtol=0)


def expand_spin_orbital(blocks, rank, p, nv, no):
    letters = "abc"[:rank] + "ijk"[:rank]
    spaces = {c: ("v" if c in "abc" else "o") for c in letters}
    dense = np.zeros((2 * nv,) * rank + (2 * no,) * rank)
    for spins in product((0, 1), repeat=2 * rank):
        mapped = _amplitude("r", letters, dict(zip(letters, spins)), spaces)
        if mapped is not None:
            sign, f = mapped
            dense[tuple(slice(s, None, 2) for s in spins)] = (
                sign * (p if f.flipped else 1) * np.einsum(f.letters + "->" + letters, blocks[f.block]))
    return dense


@pytest.mark.parametrize("p", [1, -1])
def test_jacobian_terms_match_finite_differences(model, p):
    h, eri, no, constant, solver, amps = model
    nv, n = len(h) - no, len(h)
    cc, t = spin_orbital(model)
    rng = np.random.default_rng(7)
    ab = rng.normal(size=(nv, nv, no, no))
    r = {"r1": {"a": rng.normal(size=(nv, no))},
         "r2": {"aa": antisymmetrize(rng.normal(size=(nv, nv, no, no)), "aa"),
                "ab": 0.5 * (ab + p * ab.transpose(1, 0, 3, 2))},
         "r3": {"aaa": antisymmetrize(rng.normal(size=(nv,) * 3 + (no,) * 3), "aaa"),
                "aab": antisymmetrize(rng.normal(size=(nv,) * 3 + (no,) * 3), "aab")}}
    R = [expand_spin_orbital(r[f"r{k}"], k, p, nv, no) for k in (1, 2, 3)]

    def central(eps):
        plus = cc._residuals(tuple(x + eps * y for x, y in zip(t, R)))
        minus = cc._residuals(tuple(x - eps * y for x, y in zip(t, R)))
        return [(a - b) / (2 * eps) for a, b in zip(plus, minus)]
    coarse, fine = central(1e-3), central(5e-4)
    J = [(4 * y - x) / 3 for x, y in zip(coarse, fine)]
    f, g = cc._dressed(t[0])
    arrays = dict(r, f={"a": f[A, A]}, g={"aaaa": g[A, A, A, A], "abab": g[A, B, A, B]},
                  t2={"aa": t[1][A, A, A, A], "ab": t[1][A, B, A, B]},
                  t3={"aaa": t[2][A, A, A, A, A, A], "aab": t[2][A, A, B, A, A, B]})
    got = apply_terms(jacobian_terms(), arrays, no, n, p)
    want = {"1:a": J[0][A, A], "2:aa": J[1][A, A, A, A], "2:ab": J[1][A, B, A, B],
            "3:aaa": J[2][A, A, A, A, A, A], "3:aab": J[2][A, A, B, A, A, B]}
    for key in want:
        np.testing.assert_allclose(got[key], want[key], atol=1e-9, rtol=0)


@pytest.mark.parametrize("irrep", [0, 2])
def test_left_is_the_transpose_and_singlets_match(model, irrep):
    h, eri, no, constant, solver, amps = model
    reference = EOMSinglet(solver, amps, orbital_irreps=LABELS, irrep=irrep).solve(2, residual_tol=1e-9)
    for spin in ("singlet", "triplet"):
        engine = SpinEOMCCSDT(solver, amps, orbital_irreps=LABELS, irrep=irrep, spin=spin)
        rng = np.random.default_rng(3)
        x, y = (engine.space.project(rng.normal(size=engine.space.size)) for _ in range(2))
        assert abs(y @ engine.sigma(x) - engine.sigma_left(y) @ x) < 1e-12 * abs(y @ engine.sigma(x))
        right = engine.solve(2, residual_tol=1e-9)
        left = engine.solve(2, left=True, residual_tol=1e-9, guesses=list(right.vectors))
        assert right.converged and left.converged
        np.testing.assert_allclose(left.excitation_energies.real, right.excitation_energies.real, atol=1e-9)
        if spin == "singlet":
            np.testing.assert_allclose(right.excitation_energies.real, reference.excitation_energies.real,
                                       atol=1e-9)
        assert abs(left.vectors[0].real @ right.vectors[1].real) < 1e-8


def test_dense_engine_matches_symmetric_and_lambda_source(model):
    h, eri, no, constant, solver, amps = model
    labelled = SpinEOMCCSDT(solver, amps, orbital_irreps=LABELS, irrep=0, spin="singlet")
    dense = SpinEOMCCSDT(solver, amps, spin="singlet")
    full = dense.space.size
    rng = np.random.default_rng(5)
    x = labelled.space.project(rng.normal(size=labelled.space.size))
    # Embed the labelled vector into the dense coordinates.
    index = {}
    for key, block in dense.space.blocks.items():
        coords = list(zip(*block.coordinates))
        index[key] = {c: i for i, c in enumerate(coords)}
    embedded = np.zeros(full)
    for (key, block), part, dpart in zip(labelled.space.blocks.items(), labelled.space.slices, dense.space.slices):
        rows = [dpart.start + index[key][c] for c in zip(*block.coordinates)]
        embedded[rows] = x[part]
        y_rows = rows
    jl = labelled.sigma(x)
    jd = dense.sigma(embedded)
    back = np.concatenate([jd[[dpart.start + index[key][c] for c in zip(*block.coordinates)]]
                           for (key, block), dpart in zip(labelled.space.blocks.items(), dense.space.slices)])
    np.testing.assert_allclose(jl, back, atol=1e-10, rtol=0)
    # dE/dT along a packed singlet direction, against the spin-orbital energy.
    cc, t = spin_orbital(model)
    nv = len(h) - no
    blocks = dense.space.unpack(embedded)
    R = [expand_spin_orbital(blocks[f"r{k}"], k, 1, nv, no) for k in (1, 2)]
    eps = 1e-4
    plus = cc._correlation_energy((t[0] + eps * R[0], t[1] + eps * R[1], t[2]))
    minus = cc._correlation_energy((t[0] - eps * R[0], t[1] - eps * R[1], t[2]))
    assert abs((plus - minus) / (2 * eps) - dense.lambda_source() @ embedded) < 1e-8


@pytest.mark.parametrize("irrep", [0, 2])
def test_singlet_vectors_convert_to_spin_eigenvectors(model, irrep):
    from itercc.eom import SingletSpace
    from itercc.eom_spin_ccsdt import amplitude_blocks

    h, eri, no, constant, solver, amps = model
    reference = EOMSinglet(solver, amps, orbital_irreps=LABELS, irrep=irrep).solve(1, residual_tol=1e-10)
    space = SingletSpace(solver, LABELS, irrep)
    blocks = amplitude_blocks(solver, space.unpack(reference.vectors[0].real))
    engine = SpinEOMCCSDT(solver, amps, orbital_irreps=LABELS, irrep=irrep, spin="singlet")
    vector = engine.space.project(engine.space.pack(blocks))
    vector /= np.linalg.norm(vector)
    omega = reference.excitation_energies[0].real
    assert engine.full_residual_norm(vector, omega) < 1e-8


def test_shared_engine_matches_fresh(model):
    h, eri, no, constant, solver, amps = model
    first = SpinEOMCCSDT(solver, amps, orbital_irreps=LABELS, irrep=2, spin="triplet")
    shared = SpinEOMCCSDT(solver, amps, orbital_irreps=LABELS, irrep=1, spin="singlet", share=first)
    fresh = SpinEOMCCSDT(solver, amps, orbital_irreps=LABELS, irrep=1, spin="singlet")
    assert shared._executor.cache is first._executor.cache
    x = fresh.space.project(np.random.default_rng(9).normal(size=fresh.space.size))
    np.testing.assert_allclose(shared.sigma(x), fresh.sigma(x), atol=1e-13, rtol=0)
    np.testing.assert_allclose(shared.sigma_left(x), fresh.sigma_left(x), atol=1e-13, rtol=0)
