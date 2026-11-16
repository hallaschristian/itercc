"""Analytic spin-flip EOM-EE-CCSDTQ in virtual-first spin orbitals."""
from itertools import combinations, product
from time import perf_counter

import numpy as np

from ._directional import Directional, dress_direction, product_directions
from ._spin_quadruples_diagrams import DIAGRAMS
from ._spin_quadruples_storage import PackedSpinQuadruples, signed_permutations as _permutations
from ._spin_quadruples_tiles import contract_tile, external_routes, spin_blocks
from .bounded_einsum import einsum_into
from .eom import EOMSinglet
from .hamiltonian import Hamiltonian
from .solver import TensorCC
from .spatial_symmetry import orbital_labels, project_integrals


def _antisymmetrize(xp, tensor):
    rank = tensor.ndim // 2
    identity = tuple(range(rank))
    order = _permutations(rank)
    for side in (0, 1):
        projected = xp.zeros_like(tensor)
        for axes, sign in order:
            perm = (axes + tuple(range(rank, 2 * rank)) if side == 0 else
                    identity + tuple(rank + axis for axis in axes))
            view = tensor.transpose(perm)
            if sign > 0:
                projected += view
            else:
                projected -= view
        projected /= len(order)
        tensor = projected
    return tensor


def _restricted_amplitude(xp, spatial, nocc, nvir, rank):
    """Lift simultaneous-pair-symmetric spatial amplitudes to spin orbitals."""
    result = xp.zeros((2 * nvir,) * rank + (2 * nocc,) * rank)
    slots = (slice(0, None, 2), slice(1, None, 2))
    for holes in product(range(2), repeat=rank):
        for order, sign in _permutations(rank):
            particles = tuple(holes[p] for p in order)
            axes = order + tuple(range(rank, 2 * rank))
            target = result[tuple(slots[s] for s in particles + holes)]
            if sign > 0:
                target += spatial.transpose(axes)
            else:
                target -= spatial.transpose(axes)
    return result


def _ground_amplitudes(solver, amplitudes, xp, packed_layout=None):
    no, nv = solver.nocc, solver.nvir
    lower = []
    for rank, a in enumerate(amplitudes[:3], 1):
        if xp is np and hasattr(a, "get"):
            a = a.get()
        lower.append(xp.asarray(a).transpose((*range(rank, 2 * rank), *range(rank))))
    result = [_restricted_amplitude(xp, a, no, nv, rank)
              for rank, a in enumerate(lower, 1)]
    if packed_layout is not None:
        return (*result, packed_layout.from_restricted(solver, amplitudes[3], xp))
    slots = solver.layout.indices
    packed = solver.quadruples_to_dense(amplitudes[3])
    spatial = xp.zeros((nv,) * 4 + (no,) * 4)
    for row, occupied in enumerate(slots):
        tile = xp.asarray(packed[row])
        for order, _ in _permutations(4):
            spatial[(slice(None),) * 4 + tuple(occupied[p] for p in order)] = tile.transpose(order)
    result.append(_restricted_amplitude(xp, spatial, no, nv, 4))
    return tuple(result)


def _raise_spin(mask, spatial_orbital):
    # The annihilation/creation indices are adjacent in the interleaved order.
    bit = 1 << (2 * spatial_orbital)
    return mask ^ (3 * bit) if mask & (3 * bit) == 2 * bit else None


class SpinFlipSpace:
    """Canonical M_S=+1 coordinates in one spatial irrep, projected onto S=1."""

    def __init__(self, nocc, nvir, labels, irrep, xp, *, packed=None, project_spin=True):
        self.xp, self.irrep = xp, irrep
        self.shapes = tuple((2 * nvir,) * r + (2 * nocc,) * r for r in range(1, 5))
        self.slices, self.indices = [], []
        self.size = 0
        masks, phases = [], []
        reference = (1 << (2 * nocc)) - 1
        for rank in range(1, 5 if packed is None else 4):
            holes = list(combinations(range(2 * nocc), rank))
            by_spin_and_irrep = {}
            for hole in holes:
                sector = int(np.bitwise_xor.reduce(labels[np.array(hole) // 2]))
                by_spin_and_irrep.setdefault((sum(h % 2 == 0 for h in hole), sector), []).append(hole)
            rows = []
            for part in combinations(range(2 * nvir), rank):
                sector = int(np.bitwise_xor.reduce(labels[nocc + np.array(part) // 2]))
                for hole in by_spin_and_irrep.get((sum(p % 2 == 0 for p in part) - 1, sector ^ irrep), ()):
                    rows.append(part + hole)
                    if project_spin:
                        masks.append(reference ^ sum(1 << i for i in hole)
                                     ^ sum(1 << (2 * nocc + a) for a in part))
                        phases.append((-1) ** (sum(hole) - rank * (rank - 1) // 2))
            self.slices.append(slice(self.size, self.size + len(rows)))
            self.size += len(rows)
            self.indices.append(xp.asarray(np.array(rows, dtype=np.int32).reshape(-1, 2 * rank).T))
        if packed is not None:
            self.slices.append(slice(self.size, self.size + packed.size))
            self.size += packed.size
        if not self.size:
            raise ValueError("The requested spin-flip symmetry sector is empty")
        source, target, raised = [], [], {}
        for i, mask in enumerate(masks):
            for p in range(nocc + nvir):
                moved = _raise_spin(mask, p)
                if moved is not None:
                    source.append(i)
                    target.append(raised.setdefault(moved, len(raised)))
        self._raised_size = len(raised)
        self._source, self._target = map(xp.asarray, (np.array(source, dtype=np.int32),
                                                    np.array(target, dtype=np.int32)))
        self._phase = xp.asarray(np.array(phases, dtype=np.float64))[self._source]

    def unpack(self, vector, *, max_rank=4):
        for rank, (shape, indices, section) in enumerate(
                zip(self.shapes[:max_rank], self.indices, self.slices), 1):
            full = self.xp.zeros(shape, dtype=vector.dtype)
            values = vector[section]
            for (p, sp), (h, sh) in product(_permutations(rank), repeat=2):
                axes = (*p, *(rank + j for j in h))
                full[tuple(indices[j] for j in axes)] = sp * sh * values
            yield full

    def pack(self, tensors):
        return self.xp.concatenate([array[tuple(indices)]
                                    for array, indices in zip(tensors, self.indices)])

    def project(self, vector):
        # S- S+ has eigenvalues 0, 4, 10 and 18 for S=1..4 at M_S=1.
        if not self._raised_size:
            return vector
        xp = self.xp
        vector = vector.copy()
        for eigenvalue in (4., 10., 18.):
            raised = xp.zeros(self._raised_size, dtype=vector.dtype)
            xp.add.at(raised, self._target, self._phase * vector[self._source])
            xp.add.at(vector, self._source, (-self._phase / eigenvalue) * raised[self._target])
        return vector


class TripletEOMCCSDTQ:
    """Full T1–T4 spin-flip Jacobian with a dense reference tensor backend.

    The dense backend is a small-model reference. The occupied-tile option
    bounds output intermediates and, with orbital labels, runs the tile
    products on the shared symmetry-blocked GEMMs (cuBLAS grouped GEMMs on the
    GPU, the compiled executor on the CPU), with the same contractions and
    spin conventions.
    """

    method = "ccsdtq"

    def __init__(self, solver, amplitudes, orbital_irreps=None, irrep=0, *,
                 backend="cpu", memory_budget_gib=4, occupied_tiles=False, action_only=False):
        no, nv = solver.nocc, solver.nvir
        self.estimated_dense_bytes = 8 * (5 * (4 * no * nv)**4 + 6 * (2 * (no + nv))**4)
        if action_only:
            # The rank-four probe keeps T3 and R3 dense, a bounded virtual
            # tile, and the compact spin/irrep T4, R4, and image vectors.
            self.estimated_dense_bytes = int(8 * (2.5 * (4 * no * nv)**3
                                                 + 6 * (2 * (no + nv))**4
                                                 + 128 * nv**4))
        elif occupied_tiles:
            self.estimated_dense_bytes = 8 * (10 * (4 * no * nv)**3 +
                                              4 * no**3 * nv**4 + 6 * (2 * (no + nv))**4)
        if self.estimated_dense_bytes > memory_budget_gib * 2**30:
            raise MemoryError(f"Spin-flip tensor estimate is {self.estimated_dense_bytes / 2**30:.1f} GiB")
        ref = solver._reference
        # Integrals are stored internally in irrep order for packed ground states.
        order = np.r_[np.arange(no), no + (np.arange(nv) if solver._virtual_inverse is None
                                           else solver._virtual_inverse)]
        h = ref.h[np.ix_(order, order)]
        g = ref.g[np.ix_(order, order, order, order)].transpose(0, 2, 1, 3)
        # The spatial reference energy includes the input scalar constant.
        o = slice(no)
        bare_reference = (2 * np.trace(h[o, o])
                          + 2 * np.einsum("ijij->", ref.g[o, o, o, o])
                          - np.einsum("ijji->", ref.g[o, o, o, o]))
        labels = (np.zeros(no + nv, dtype=np.int32) if orbital_irreps is None else
                  orbital_labels(orbital_irreps, no + nv))
        if np.any(labels):
            h, g, _ = project_integrals(h, g, labels, tolerance=1e-9)
        model = Hamiltonian.restricted(h, g, no, solver.reference_energy - bare_reference)
        self.proxy = TensorCC(model, backend=backend)
        self.xp = xp = self.proxy.xp
        self.nocc, self.nvir = no, nv
        self._spatial_labels = {"o": labels[:no], "v": labels[no:]}
        self._symmetry_engine = None
        if occupied_tiles and np.any(labels):
            # Symmetry-blocked tile products: cuBLAS grouped GEMMs on the GPU,
            # the compiled executor on the CPU.
            nirrep = 1 << int(labels.max()).bit_length()
            by_letter = {c: labels[no:] for c in "abcdef"}
            by_letter.update({c: labels[:no] for c in "ijklmn"})
            if xp is not np:
                from . import symmetry_blocks
                from ._quadruples_cuda import einsum as cuda_einsum
                if symmetry_blocks.available():
                    self._symmetry_engine = symmetry_blocks.SymmetryBlocks(
                        by_letter, "ijkl", nirrep, fallback=cuda_einsum)
            else:
                from . import _cpu_quadruples
                if _cpu_quadruples.available():
                    self._symmetry_engine = _cpu_quadruples.blocked_einsum(by_letter, "ijkl", nirrep)
        self.occupied_tiles = occupied_tiles
        if occupied_tiles:
            self._ground_t4 = PackedSpinQuadruples(labels[:no], labels[no:], 0, 0)
            self._direction_t4 = PackedSpinQuadruples(labels[:no], labels[no:], irrep, 1)
        if action_only and not occupied_tiles:
            raise ValueError("The action-only benchmark requires occupied tiling")
        self.action_only = action_only
        self.space = SpinFlipSpace(no, nv, labels, irrep, xp,
                                   packed=self._direction_t4 if action_only else None,
                                   project_spin=not action_only)
        self.amplitudes = _ground_amplitudes(solver, amplitudes, xp,
                                            self._ground_t4 if occupied_tiles else None)
        t1 = self.amplitudes[0]
        self.fock, self.integrals, self.one_body = self.proxy._dressed(t1, one_body=True)
        self.ground_energy = self.proxy.reference_energy + self.proxy._correlation_energy(self.amplitudes)
        eo, ev = xp.asarray(np.diag(self.proxy.fock)[:2 * no]), xp.asarray(np.diag(self.proxy.fock)[2 * no:])
        self.diagonal = (None if action_only else xp.concatenate(
            [sum(ev[x] for x in axes[:rank]) - sum(eo[x] for x in axes[rank:])
             for rank, axes in enumerate(self.space.indices, 1)]))
        self.matvec_count, self.matvec_seconds = 0, 0.
        if occupied_tiles and not action_only:
            coordinates = self.space.indices[3]
            self._tiled_r4_order = xp.empty(coordinates.shape[1], dtype=np.int32)
            # The dense reference coordinates are particle-major. The storage
            # backend groups amplitudes by occupied slot for contiguous gathers.
            host = coordinates if xp is np else coordinates.get()
            order = {tuple(row): i for i, row in enumerate(host.T)}
            for holes, (key, window) in self._direction_t4.slots.items():
                values = [order[(*p, *holes)] for p in self._direction_t4.groups[key]]
                self._tiled_r4_order[window] = xp.asarray(values, dtype=np.int32)

    def _tiled_action(self, blocks, rank, packed_direction, *, slot_limit=None):
        xp = self.xp
        no, nv = self.nocc, self.nvir
        primal = {name: spin_blocks(pair.value) for name, pair in blocks.items()}
        tangent = {name: spin_blocks(pair.tangent) for name, pair in blocks.items()}
        for tensors, layout, vector in ((primal, self._ground_t4, self.amplitudes[3]),
                                        (tangent, self._direction_t4, packed_direction)):
            def view(signature, layout=layout, vector=vector):
                def fetch(selector):
                    return layout.expand(vector, signature, selector[4:], xp)[selector[:4]]
                return fetch
            tensors["t4"] = {signature: view(signature) for signature in product((0, 1), repeat=8)}
        space = self.space
        if rank == 4 and self.action_only:
            result = xp.zeros(self._direction_t4.size)
            for index, (holes, (key, rows)) in enumerate(self._direction_t4.slots.items()):
                if slot_limit is not None and index >= slot_limit:
                    break
                particles = self._direction_t4.groups[key]
                # A canonical particle list can have several spin signatures.
                for pspin in product((0, 1), repeat=4):
                    mask = np.flatnonzero(np.all(particles % 2 == pspin, axis=1))
                    if not len(mask):
                        continue
                    signature = pspin + tuple(h % 2 for h in holes)
                    result[rows][xp.asarray(mask)] = self._project_tile(
                        rank, signature, tuple(h // 2 for h in holes),
                        xp.asarray(particles[mask].T // 2), primal, tangent)
            yield None, result
            return
        indices = space.indices[rank - 1]
        host = indices if xp is np else indices.get()
        if rank < 4:
            for raw in product((0, 1), repeat=2 * rank):
                if sum(raw[rank:]) - sum(raw[:rank]) != 1:
                    continue
                cuts = (slice(0, no),) * rank
                yield raw, contract_tile(xp, rank, raw, cuts, primal, tangent, no, nv)
            return
        groups = {}
        for index, values in enumerate(host.T):
            signature = tuple(int(v % 2) for v in values)
            occupied = tuple(int(v // 2) for v in values[rank:])
            groups.setdefault((signature, occupied), []).append(index)
        result = xp.zeros(host.shape[1])
        for (signature, occupied), rows in groups.items():
            rows = xp.asarray(rows, dtype=np.int32)
            particles = indices[:rank, rows] // 2
            result[rows] = self._project_tile(rank, signature, occupied, particles, primal, tangent)
        yield None, result

    def _project_tile(self, rank, signature, occupied, particles, primal, tangent):
        xp, no, nv = self.xp, self.nocc, self.nvir
        subtotal = xp.zeros(particles.shape[1])
        for raw, placements in external_routes(rank, signature):
            by_occupied = {}
            for p, h, sign in placements:
                by_occupied.setdefault(tuple(occupied[i] for i in h), []).append((p, sign))
            for fixed, permutations in by_occupied.items():
                cuts = tuple(slice(i, i + 1) for i in fixed)
                tile = contract_tile(xp, rank, raw, cuts, primal, tangent, no, nv,
                                     irrep=self.space.irrep, labels=self._spatial_labels,
                                     engine=self._symmetry_engine).reshape((nv,) * rank)
                for p, sign in permutations:
                    subtotal += sign * tile[tuple(particles[i] for i in p)]
        return subtotal / len(_permutations(rank))**2

    def _action(self, vector, *, slot_limit=None):
        xp = self.xp
        no = self.proxy.hamiltonian.nocc
        r1, *direction = self.space.unpack(vector, max_rank=3 if self.occupied_tiles else 4)
        dg = dress_direction(self.integrals, r1, no, xp)
        df = dress_direction(self.one_body, r1, no, xp) + xp.einsum("piqi->pq", dg[:, :no, :, :no])
        blocks = {"f" + key: Directional(self.fock[ix], df[ix]) for key, ix in self.proxy._fblocks.items()}
        blocks.update({"g" + key: Directional(self.integrals[ix], dg[ix])
                       for key, ix in self.proxy._blocks.items()})
        blocks.update({f"t{rank}": Directional(t, r) for rank, (t, r) in
                       enumerate(zip(self.amplitudes[1:], direction), 2)})
        outputs = []
        ranks = (4,) if slot_limit is not None else range(1, 5)
        for rank in ranks:
            shape = self.space.shapes[rank - 1]
            if self.occupied_tiles and rank > 1:
                r4 = vector[self.space.slices[3]]
                if not self.action_only:
                    r4 = r4[self._tiled_r4_order]
                tiles = self._tiled_action(blocks, rank, r4, slot_limit=slot_limit)
                if rank == 4:
                    outputs.append(next(tiles)[1])
                else:
                    result = xp.zeros(shape)
                    for raw, tile in tiles:
                        result[tuple(slice(s, None, 2) for s in raw)] = tile
                    outputs.append(_antisymmetrize(xp, result)[tuple(self.space.indices[rank - 1])])
                continue
            result = xp.zeros(shape)
            for coefficient, spec, names in DIAGRAMS[rank]:
                for operands in product_directions(*(blocks[name] for name in names)):
                    einsum_into(xp, result, coefficient, spec, *operands)
            outputs.append(_antisymmetrize(xp, result)[tuple(self.space.indices[rank - 1])])
        return xp.concatenate(outputs)

    def probe_rank_four(self, vector, slots):
        """Measure a bounded prefix of occupied rank-four slots."""
        if not self.action_only or slots < 1:
            raise ValueError("Rank-four probing needs the action-only packed layout and positive slots")
        return self._action(vector, slot_limit=slots)

    def solve(self, *args, **options):
        if self.action_only:
            raise ValueError("The action-only layout has no large-system triplet spin projector")
        return EOMSinglet.solve(self, *args, **options)

    def sigma(self, vector):
        if self.action_only:
            raise ValueError("Use probe_rank_four for the action-only layout")
        if self.xp.iscomplexobj(vector):
            return self.sigma(vector.real) + 1j * self.sigma(vector.imag)
        start = perf_counter()
        result = self._action(vector)
        if not bool(self.xp.isfinite(result).all()):
            raise FloatingPointError("Nonfinite spin-flip Jacobian action")
        self.matvec_count += 1
        self.matvec_seconds += perf_counter() - start
        return result

    def full_residual_norm(self, vector, energy):
        return float(self.xp.linalg.norm(self.sigma(vector) - energy * vector))
