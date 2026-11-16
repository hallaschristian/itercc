"""First-order spin-orbit state interaction for scalar singlets and triplets."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SpinOrbitResult:
    """Energies in hartree and eigenvectors in the reported spin-free basis."""

    energies: Any
    eigenvectors: Any
    hamiltonian: Any
    basis: tuple[tuple[int, int, int], ...]
    max_antihermitian_Eh: float


def _contract(xp, operator, real_operator, densities, shape, name):
    storage = xp if isinstance(densities, xp.ndarray) else np
    densities = storage.asarray(densities)
    if densities.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {densities.shape}")
    if densities.dtype.kind not in "fiuc":
        raise ValueError(f"{name} must be numeric")
    left_count, right_count, orbitals, _ = shape
    pairs = left_count * right_count
    values = xp.empty((left_count, right_count, 3), dtype=xp.complex128)
    flat_values = values.reshape(pairs, 3)
    # Bound any dtype/layout conversion, including when densities are strided.
    tile = max(1, (8 * 1024**2) // (16 * orbitals**2))
    for start in range(0, pairs, tile):
        end = min(start + tile, pairs)
        if densities.flags.c_contiguous:
            block = densities.reshape(pairs, orbitals**2)[start:end]
        else:
            indices = storage.arange(start, end)
            block = densities[indices // right_count, indices % right_count]
            block = block.reshape(end - start, orbitals**2)
        block = xp.asarray(block)
        if block.dtype.kind == "c":
            flat_values[start:end] = block @ operator
        else:
            products = block @ real_operator
            target = flat_values[start:end]
            target.real = products[:, :3]
            target.imag = products[:, 3:]
    return values


def spin_orbit_state_interaction(
    singlet_energies,
    triplet_energies,
    pauli,
    singlet_triplet,
    triplet_singlet,
    triplet_triplet,
    *,
    backend="cpu",
):
    """Mix scalar states using a common-orbital one-electron SOC operator.

    The energy arrays are one-dimensional, in hartree, with a common zero.
    The three Hermitian orbital matrices in pauli multiply sigma_x, sigma_y
    and sigma_z (the Pauli matrices, with no extra factor of 1/2).
    A spin-orbit mean-field operator can be supplied in this convention.

    Each directional density has shape (bra roots, ket roots, nmo, nmo)
    and stores D_alpha[p,q] - D_beta[p,q], where D[p,q] is the transition
    matrix element of a_q^dagger a_p. Use M=0 for singlet-triplet densities
    and M=+1 for triplet-triplet densities, with |1,+1> = S+|1,0>/sqrt(2).
    For EOM-CC supply consistently normalized biorthogonal left/right
    transition densities in both directions. Their SOC matrix is averaged
    with its adjoint before diagonalization. The reported defect is the
    maximum absolute element of the raw matrix minus its adjoint.

    Basis labels are (spin, zero-based root within that spin, M). CUDA
    returns device arrays and stages host-resident densities in tiles.
    Both backends assemble and diagonalize the small state-interaction matrix on CPU.
    """
    if backend == "cpu":
        xp = np
    elif backend == "cuda":
        import cupy as xp
    else:
        raise ValueError("backend must be 'cpu' or 'cuda'")

    es, et = (xp.asarray(e) for e in (singlet_energies, triplet_energies))
    for energy in (es, et):
        if energy.ndim != 1 or energy.dtype.kind not in "fiu":
            raise ValueError("energies must be finite real vectors")
    ns, nt = es.size, et.size
    if not ns + nt:
        raise ValueError("at least one scalar root is required")

    pauli = xp.asarray(pauli, dtype=xp.complex128)
    if pauli.ndim != 3 or pauli.shape[0] != 3 or pauli.shape[1] != pauli.shape[2] or not pauli.shape[1]:
        raise ValueError("pauli must have shape (3, nmo, nmo) with nmo > 0")
    pauli_error = xp.max(abs(pauli - pauli.conj().swapaxes(1, 2)))
    nmo = pauli.shape[1]
    operator = xp.ascontiguousarray(pauli.transpose(2, 1, 0)).reshape(nmo**2, 3)
    real_operator = xp.concatenate((operator.real, operator.imag), axis=1)
    st = _contract(xp, operator, real_operator, singlet_triplet, (ns, nt, nmo, nmo), "singlet_triplet")
    ts = _contract(xp, operator, real_operator, triplet_singlet, (nt, ns, nmo, nmo), "triplet_singlet")
    tt = _contract(xp, operator, real_operator, triplet_triplet, (nt, nt, nmo, nmo), "triplet_triplet")
    # Copy the reduced blocks in one transfer. Assembly and LAPACK work
    # on the small state matrix without separate CUDA validation syncs.
    packet = xp.concatenate((es, et, pauli_error.reshape(1), st.ravel(), ts.ravel(), tt.ravel()))
    if xp is not np:
        packet = xp.asnumpy(packet)
    offsets = np.cumsum((ns + nt, 1, st.size, ts.size))
    scalar, error, st_flat, ts_flat, tt_flat = np.split(packet, offsets)
    if not np.isfinite(error[0]) or error[0].real > 1e-10:
        raise ValueError("pauli must be finite and Hermitian")
    if not np.isfinite(scalar).all():
        raise ValueError("energies must be finite real vectors")
    if not np.isfinite(packet[offsets[1]:]).all():
        raise ValueError("transition-density contractions must be finite")
    st, ts, tt = st_flat.reshape(st.shape), ts_flat.reshape(ts.shape), tt_flat.reshape(tt.shape)
    return _assemble(scalar[:ns].real, scalar[ns:].real, st, ts, tt, xp)


def spin_orbit_from_components(singlet_energies, triplet_energies, st, ts, tt):
    """Mix spin-free states from already contracted Cartesian SOC elements.

    The three arrays are ordered (bra root, ket root, x/y/z). Use the M=0
    components for the two singlet-triplet directions and M=+1 for the
    triplet-triplet direction. This accepts matrix elements evaluated
    directly from the similarity-transformed operators.
    """
    es, et = map(np.asarray, (singlet_energies, triplet_energies))
    for energy in (es, et):
        if energy.ndim != 1 or energy.dtype.kind not in "fiu" or not np.isfinite(energy).all():
            raise ValueError("energies must be finite real vectors")
    ns, nt = es.size, et.size
    if not ns + nt:
        raise ValueError("at least one scalar root is required")
    st, ts, tt = (np.asarray(block, dtype=np.complex128) for block in (st, ts, tt))
    for block, shape in zip((st, ts, tt), ((ns, nt, 3), (nt, ns, 3), (nt, nt, 3))):
        if block.shape != shape or not np.isfinite(block).all():
            raise ValueError(f"expected finite Cartesian components of shape {shape}")
    return _assemble(es, et, st, ts, tt, np)


def _assemble(es, et, st, ts, tt, xp):
    ns, nt = es.size, et.size
    raw = np.zeros((ns + 3 * nt, ns + 3 * nt), dtype=np.complex128)
    inv_sqrt2 = 2**-0.5
    for values, block, phase in (
        (st, raw[:ns, ns:].reshape(ns, nt, 3), 1j),
        (ts, raw[ns:, :ns].reshape(nt, 3, ns).transpose(0, 2, 1), -1j),
    ):
        x, y, z = (values[..., i] for i in range(3))
        block[..., 0] = -(x + phase * y) * inv_sqrt2
        block[..., 1] = z
        block[..., 2] = (x - phase * y) * inv_sqrt2

    block = raw[ns:, ns:].reshape(nt, 3, nt, 3)
    x, y, z = (tt[..., i] for i in range(3))
    block[:, 0, :, 0] = z
    block[:, 2, :, 2] = -z
    block[:, 0, :, 1] = block[:, 1, :, 2] = (x - 1j * y) * inv_sqrt2
    block[:, 1, :, 0] = block[:, 2, :, 1] = (x + 1j * y) * inv_sqrt2

    adjoint = raw.conj().T
    defect = float(np.max(abs(raw - adjoint)))
    hamiltonian = 0.5 * (raw + adjoint)
    diagonal = np.arange(ns + 3 * nt)
    hamiltonian[diagonal, diagonal] += np.concatenate((es, np.repeat(et, 3)))
    energies, eigenvectors = np.linalg.eigh(hamiltonian)
    basis = tuple((0, i, 0) for i in range(ns)) + tuple(
        (1, i, m) for i in range(nt) for m in (1, 0, -1)
    )
    return SpinOrbitResult(xp.asarray(energies), xp.asarray(eigenvectors), xp.asarray(hamiltonian), basis, defect)
