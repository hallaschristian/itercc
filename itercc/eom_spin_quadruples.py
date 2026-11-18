"""Analytic spin-flip EOM-EE-CCSDTQ in virtual-first spin orbitals."""
from itertools import combinations, product
from time import perf_counter

import numpy as np

from ._directional import Directional, dress_direction, product_directions
from ._spin_quadruples_diagrams import DIAGRAMS
from ._spin_quadruples_storage import (
    PackedSpinProjector, PackedSpinQuadruples,
)
from ._tensor_symmetry import signed_permutations as _permutations
from ._tensor_scatter import unpack_antisymmetric
from ._spin_quadruples_tiles import contract_tile, external_routes, projection_diagrams, spin_blocks
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
        self._packed_projector = (PackedSpinProjector(packed, xp)
                                  if packed is not None and packed.size and project_spin else None)
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

    def unpack(self, vector, *, max_rank=4, nonzero=None):
        for rank, (shape, indices, section) in enumerate(
                zip(self.shapes[:max_rank], self.indices, self.slices), 1):
            if nonzero is not None and not nonzero[rank - 1]:
                yield None
                continue
            yield unpack_antisymmetric(vector[section], indices, shape, xp=self.xp)

    def pack(self, tensors):
        return self.xp.concatenate([array[tuple(indices)]
                                    for array, indices in zip(tensors, self.indices)])

    def project(self, vector):
        # S- S+ has eigenvalues 0, 4, 10 and 18 for S=1..4 at M_S=1.
        if not self._raised_size and self._packed_projector is None:
            return vector
        xp = self.xp
        vector = vector.copy()
        if self._raised_size:
            for eigenvalue in (4., 10., 18.):
                raised = xp.zeros(self._raised_size, dtype=vector.dtype)
                xp.add.at(raised, self._target, self._phase * vector[self._source])
                xp.add.at(vector, self._source, (-self._phase / eigenvalue) * raised[self._target])
        if self._packed_projector is not None:
            tail = self.slices[3]
            vector[tail] = self._packed_projector.project(vector[tail])
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
                 backend="cpu", memory_budget_gib=4, occupied_tiles=False, action_only=False,
                 packed_spin=False, packed_contractions=None, intermediate_cache_gib=0,
                 occupied_slab=None):
        no, nv = solver.nocc, solver.nvir
        if occupied_slab is not None and (isinstance(occupied_slab, (bool, np.bool_)) or
                not isinstance(occupied_slab, (int, np.integer)) or occupied_slab < 1):
            raise ValueError("Occupied slab must be a positive integer or None")
        self.occupied_slab = None if occupied_slab is None else int(occupied_slab)
        # Rank-four slots as one compiled program on the CPU (see _compiled_slots).
        self.slot_program = True
        self.fused_tiles = True
        self.share_work = None
        self._slot_programs = {}
        if not np.isfinite(intermediate_cache_gib) or intermediate_cache_gib < 0:
            raise ValueError("Intermediate cache budget must be finite and nonnegative")
        self.intermediate_cache_bytes = int(intermediate_cache_gib * 2**30)
        self._packed_direction = packed_spin or action_only
        if self._packed_direction and not occupied_tiles:
            raise ValueError("The packed spin layout requires occupied tiling")
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
        self.estimated_dense_bytes += self.intermediate_cache_bytes
        if occupied_tiles and backend == "cpu":
            # Shared signed virtual maps: <=16 spin signatures x 8 irreps,
            # each with nv**4 int32 entries. No second copy for the R4 sector.
            self.estimated_dense_bytes += 512 * nv**4
        self.memory_budget_bytes = int(memory_budget_gib * 2**30)
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
        if packed_contractions is None:
            from . import _cpu_quadruples
            packed_contractions = (xp is np and occupied_tiles and self._packed_direction
                                   and _cpu_quadruples.available())
        self.packed_contractions = bool(packed_contractions)
        self._packed_diagrams = {}
        if self.packed_contractions:
            from . import _cpu_quadruples
            if xp is not np or not occupied_tiles or not self._packed_direction or not _cpu_quadruples.available():
                raise ValueError("Direct packed contractions require the compiled CPU and packed occupied tiles")
            from ._spin_quadruples_cpu import supported_diagrams
            self._packed_diagrams = {r: supported_diagrams(r) for r in (2, 3, 4)}
        if occupied_tiles:
            self._ground_t4 = PackedSpinQuadruples(labels[:no], labels[no:], 0, 0)
            self._direction_t4 = PackedSpinQuadruples(labels[:no], labels[no:], irrep, 1)
            self._direction_t4.share_virtual_maps(self._ground_t4)
        if occupied_tiles and self._packed_direction and self.packed_contractions:
            # The compiled rank-four slot program (_compiled_slots): a raw tile
            # per thread, the dense direct terms' per-slot buffers (72 terms and
            # 16 contexts per row at most), and the waves' partial sums, as
            # large as an explicit intermediate cache (a default cache takes
            # half of what remains, the partial sums the other half).
            from ._cpu_triples import cpu_threads

            rows = max((len(p) for p in self._direction_t4.groups.values()), default=0)
            self.estimated_dense_bytes += (cpu_threads() * (8 * nv**4 + rows * (72 * 8 + 16 * 24))
                                           + self.intermediate_cache_bytes)
            if self.estimated_dense_bytes > self.memory_budget_bytes:
                raise MemoryError(f"Spin-flip tensor estimate is {self.estimated_dense_bytes / 2**30:.1f} GiB")
        self.action_only = action_only
        self._lower_projection = {}
        self.native_projection = True
        self._gather_plans = {}
        self.space = SpinFlipSpace(no, nv, labels, irrep, xp,
                                   packed=self._direction_t4 if self._packed_direction else None,
                                   project_spin=not action_only)
        self.amplitudes = _ground_amplitudes(solver, amplitudes, xp,
                                            self._ground_t4 if occupied_tiles else None)
        t1 = self.amplitudes[0]
        self.fock, self.integrals, self.one_body = self.proxy._dressed(t1, one_body=True)
        self.ground_energy = self.proxy.reference_energy + self.proxy._correlation_energy(self.amplitudes)
        self.diagonal = None if action_only else self._orbital_diagonal()
        self.matvec_count, self.matvec_seconds = 0, 0.
        if occupied_tiles and not self._packed_direction:
            coordinates = self.space.indices[3]
            self._tiled_r4_order = xp.empty(coordinates.shape[1], dtype=np.int32)
            # The dense reference coordinates are particle-major. The storage
            # backend groups amplitudes by occupied slot for contiguous gathers.
            host = coordinates if xp is np else coordinates.get()
            order = {tuple(row): i for i, row in enumerate(host.T)}
            for holes, (key, window) in self._direction_t4.slots.items():
                values = [order[(*p, *holes)] for p in self._direction_t4.groups[key]]
                self._tiled_r4_order[window] = xp.asarray(values, dtype=np.int32)

    def _orbital_diagonal(self):
        xp, space = self.xp, self.space
        occupied, virtual = np.split(np.diag(self.proxy.fock), [2 * self.nocc])
        eo, ev = xp.asarray(occupied), xp.asarray(virtual)
        diagonal = xp.empty(space.size)
        for rank, (axes, part) in enumerate(zip(space.indices, space.slices), 1):
            diagonal[part] = sum(ev[x] for x in axes[:rank]) - sum(eo[x] for x in axes[rank:])
        if self._packed_direction:
            layout = self._direction_t4
            particle_energy = {key: xp.asarray(virtual[rows].sum(axis=1))
                               for key, rows in layout.groups.items()}
            tail = diagonal[space.slices[3]]
            for holes, (key, part) in layout.slots.items():
                tail[part] = particle_energy[key] - occupied[list(holes)].sum()
        return diagonal

    def _tiled_action(self, blocks, rank, packed_direction, *, slot_limit=None, slot_finished=None,
                      packed_action=None, intermediates=None):
        xp = self.xp
        no, nv = self.nocc, self.nvir
        primal = {name: spin_blocks(pair.value) for name, pair in blocks.items()}
        tangent = {name: spin_blocks(pair.tangent) for name, pair in blocks.items()}
        for tensors, layout, vector in ((primal, self._ground_t4, self.amplitudes[3]),
                                        (tangent, self._direction_t4, packed_direction)):
            if vector is None:
                tensors["t4"] = None
                continue
            def view(signature, layout=layout, vector=vector):
                described = {}                  # signed rows of this action, by occupied cuts

                def fetch(selector, packed=False):
                    # packed: the tile as signed rows for the CPU blocked engine
                    if packed and all(s == slice(None) for s in selector[:4]):
                        key = tuple((s.start, s.stop, s.step) for s in selector[4:])
                        rows = described.get(key)
                        if rows is None:
                            rows = layout.signed_rows(vector, signature, selector[4:], xp)
                            if rows is not None:
                                if len(described) >= 4096:
                                    described.clear()
                                described[key] = rows
                        if rows is not None:
                            return rows
                    return layout.expand(vector, signature, selector[4:], xp)[selector[:4]]
                return fetch
            tensors["t4"] = {signature: view(signature) for signature in product((0, 1), repeat=8)}
        space = self.space
        if rank == 4 and self._packed_direction:
            result = xp.zeros(self._direction_t4.size)
            if self._compiled_slots(blocks, packed_direction, result, slot_limit, slot_finished, packed_action):
                yield None, result
                return
            pending = []

            def flush():
                if pending:
                    start, stop = pending[0][0], pending[-1][1]
                    coordinates = (pending[0][2] if len(pending) == 1 else
                                   np.concatenate([c for _, _, c in pending]))
                    pending.clear()
                    packed_action.add(rank, coordinates, result[start:stop])

            spin_groups = {}
            started = perf_counter() if slot_finished is not None else 0.
            total = len(self._direction_t4.slots)
            for index, (holes, (key, rows)) in enumerate(self._direction_t4.slots.items()):
                if slot_limit is not None and index >= slot_limit:
                    break
                if key not in spin_groups:
                    particles = self._direction_t4.groups[key]
                    signatures = particles % 2
                    groups = []
                    for pspin in product((0, 1), repeat=4):
                        mask = np.flatnonzero(np.all(signatures == pspin, axis=1))
                        if mask.size:
                            groups.append((pspin, xp.asarray(mask), xp.asarray(particles[mask].T // 2)))
                    spin_groups[key] = groups
                self._project_packed_tile(result[rows], holes, spin_groups[key], primal, tangent, intermediates)
                if packed_action is not None:
                    # Direct packed terms of consecutive slots go in one call:
                    # each output row gets the same terms in the same order.
                    if pending and pending[-1][1] != rows.start:
                        flush()
                    particles = self._direction_t4.groups[key]
                    coordinates = np.empty((len(particles), 8), dtype=np.int32)
                    coordinates[:, :4], coordinates[:, 4:] = particles, holes
                    pending.append((rows.start, rows.stop, coordinates))
                    if rows.stop - pending[0][0] >= 1 << 22:
                        flush()
                done = index + 1
                if slot_finished is not None and (done in (1, 4, 16, total) or done % 64 == 0):
                    flush()
                    if xp is not np:
                        xp.cuda.get_current_stream().synchronize()
                    slot_finished(done, total, perf_counter() - started)
            flush()
            yield None, result
            return
        indices = space.indices[rank - 1]
        host = indices if xp is np else indices.get()
        if rank < 4:
            diagrams = tuple(d for d in DIAGRAMS[rank] if d not in self._packed_diagrams.get(rank, ()))
            for raw in product((0, 1), repeat=2 * rank):
                if sum(raw[rank:]) - sum(raw[:rank]) != 1:
                    continue
                cuts = (slice(0, no),) * rank
                yield raw, contract_tile(xp, rank, raw, cuts, primal, tangent, no, nv,
                                         irrep=space.irrep, labels=self._spatial_labels,
                                         engine=self._symmetry_engine, diagrams=diagrams,
                                         occupied_slab=self.occupied_slab,
                                         intermediates=intermediates, compiled_products=self.packed_contractions)
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

    def _compiled_slots(self, blocks, direction, result, slot_limit, slot_finished, packed_action):
        """Run the rank-four slots as one compiled program (`_spin_quadruples_program`):
        the same result as the slot loop, bit for bit. False when it does not
        apply (GPU, no symmetry engine, no direct packed terms, the reference
        expansion or projection, a product without a blocked plan, or
        ITERCC_TRIPLET_PROGRAM=0): the loop runs."""
        import os

        if (not self.slot_program or self.xp is not np or not self.packed_contractions
                or not self.native_projection or not self._direction_t4.native_expansion
                or not self._ground_t4.native_expansion
                or not getattr(self._symmetry_engine, "packed_operands", False)
                or os.environ.get("ITERCC_TRIPLET_PROGRAM", "1") == "0"):
            return False
        from ._cpu_triples import cpu_threads
        from ._spin_quadruples_program import SlotProgram, Unsupported

        presence = {name: pair.tangent is not None for name, pair in blocks.items()}
        presence["t4"] = direction is not None
        key = tuple(sorted(presence.items())), SlotProgram._signature(blocks)
        if key not in self._slot_programs:
            groups = [(equivalent, multiplicity, tuple(d for d in diagrams if d not in self._packed_diagrams[4]))
                      for equivalent, multiplicity, diagrams in projection_diagrams(4)]
            try:
                program = SlotProgram(self, blocks, groups, presence)
            except Unsupported:
                program = None
            self._slot_programs[key] = program
        program = self._slot_programs[key]
        if program is None or (self.amplitudes[3] is None and program.reads_ground):
            return False
        started = perf_counter()
        layout = self._direction_t4
        total = len(layout.slots)
        done = total if slot_limit is None else min(int(slot_limit), total)
        # Shared intermediates: the configured budget, else half of what the
        # memory budget leaves beside the estimate (which counts the program's
        # tiles and direct-term buffers): the waves' partial sums take as much
        # as the cache. At least 256 MiB.
        cache = self.intermediate_cache_bytes
        if not cache:
            cache = (self.memory_budget_bytes - self.estimated_dense_bytes) // 2
        cache = max(cache, 2**28)
        # Fused tile products (other rounding); ITERCC_TRIPLET_FUSED=0 keeps
        # the loop's products, bit for bit.
        fused = self.fused_tiles and os.environ.get("ITERCC_TRIPLET_FUSED", "1") != "0"
        # Intermediates of at least this many multiply-adds are computed in
        # pieces that waiting tiles help with (0: the core's default, 2^22).
        share = int(os.environ.get("ITERCC_SLOT_SHARE_WORK", self.share_work or 0))
        report = None
        if slot_finished is not None:
            def report(slots):
                slot_finished(slots, total, perf_counter() - started)
        self._program_cache = program.run(blocks, self.amplitudes[3], direction, result, cache_bytes=cache,
                                          threads=cpu_threads(), slot_limit=done, fused=fused, share_work=share,
                                          progress=report)
        if packed_action is not None and done:
            # The direct packed terms of consecutive slots (windows follow each
            # other), at least 2^22 rows per call: the dense kernel runs a
            # call's slots in parallel (2^18 rows were 12 CuF slots for 56
            # threads), and each row's sums do not depend on the chunks.
            slots = list(layout.slots.items())[:done]
            start = 0
            while start < done:
                stop, count = start, 0
                while stop < done and count < 1 << 22:
                    count += len(layout.groups[slots[stop][1][0]])
                    stop += 1
                parts = []
                for holes, (key, _) in slots[start:stop]:
                    particles = layout.groups[key]
                    coordinates = np.empty((len(particles), 8), dtype=np.int32)
                    coordinates[:, :4], coordinates[:, 4:] = particles, holes
                    parts.append(coordinates)
                first, last = slots[start][1][1].start, slots[stop - 1][1][1].stop
                packed_action.add(4, parts[0] if len(parts) == 1 else np.concatenate(parts), result[first:last])
                start = stop
        return True

    def _project_lower_tiles(self, rank, tiles):
        """Accumulate antisymmetry directly in canonical coordinates for ranks 2 and 3."""
        xp = self.xp
        indices = self.space.indices[rank - 1]
        if rank not in self._lower_projection:
            signatures = xp.zeros(indices.shape[1], dtype=xp.uint8)
            for row in indices:
                signatures <<= 1
                signatures |= (row & 1).astype(xp.uint8)
            by_raw = {}
            for code in xp.unique(signatures).tolist():
                bits = tuple((code >> bit) & 1 for bit in reversed(range(2 * rank)))
                rows = xp.flatnonzero(signatures == code)
                positions = tuple(indices[:, rows] // 2)
                for raw, placements in external_routes(rank, bits):
                    # The raw tile axes are those specified by the external permutation.
                    by_raw.setdefault(raw, []).append((rows, positions, placements))
            self._lower_projection[rank] = by_raw
        by_raw = self._lower_projection[rank]
        result = xp.zeros(indices.shape[1])
        for raw, tile in tiles:
            for rows, positions, placements in by_raw.get(raw, ()):
                permutations = tuple((p + tuple(rank + i for i in h), sign) for p, h, sign in placements)
                self._accumulate_projection(result, rows, positions, permutations, tile, 1)
        return result / len(_permutations(rank))**2

    def _project_packed_tile(self, result, holes, groups, primal, tangent, intermediates=None):
        """Contract each raw tile once and scatter to every compatible particle spin pattern."""
        rank = 4
        occupied = tuple(h // 2 for h in holes)
        hole_spins = tuple(h % 2 for h in holes)
        for equivalent, multiplicity, diagrams in projection_diagrams(rank):
            if self.packed_contractions:
                diagrams = tuple(d for d in diagrams if d not in self._packed_diagrams[rank])
                if not diagrams:
                    continue
            scatter = {}
            for particle_spins, rows, particles in groups:
                signature = particle_spins + hole_spins
                for raw, placements in external_routes(rank, signature, equivalent):
                    for p, h, sign in placements:
                        fixed = tuple(occupied[i] for i in h)
                        by_group = scatter.setdefault((raw, fixed), {})
                        by_group.setdefault(particle_spins, (rows, particles, []))[2].append((p, sign))
            for (raw, fixed), by_group in scatter.items():
                cuts = tuple(slice(i, i + 1) for i in fixed)
                tile = contract_tile(self.xp, rank, raw, cuts, primal, tangent, self.nocc, self.nvir,
                                     irrep=self.space.irrep, labels=self._spatial_labels,
                                     engine=self._symmetry_engine, diagrams=diagrams,
                                     occupied_slab=self.occupied_slab,
                                     intermediates=intermediates,
                                     compiled_products=self.packed_contractions,
                                     out=self._zeroed_tile()).reshape((self.nvir,) * rank)
                for rows, particles, permutations in by_group.values():
                    self._accumulate_projection(result, rows, particles, permutations, tile, multiplicity)
        result /= len(_permutations(rank))**2

    def _zeroed_tile(self):
        """A zeroed rank-four raw tile for one occupied tuple, reused: each tile
        is projected before the next one is formed (None on the GPU). Cleared
        on the calling thread, which runs the tile's small products: a team
        would leave the tile in the other cores' caches."""
        if self.xp is not np:
            return None
        tile = getattr(self, "_raw_tile", None)
        if tile is None:
            tile = self._raw_tile = np.empty((self.nvir,) * 4 + (1,) * 4)
        tile.fill(0.0)
        return tile

    def _accumulate_projection(self, destination, rows, particles, permutations, tile, multiplicity):
        if self.xp is np and self.packed_contractions and self.native_projection:
            from ._tensor_scatter import PermutedGather
            key = id(rows), id(particles)
            if key not in self._gather_plans:
                # Keep owners alive: object identities cannot be recycled during this action.
                self._gather_plans[key] = (rows, particles, PermutedGather(rows, particles))
            plan = self._gather_plans[key][2]
            if plan.add(tile, destination, tuple((p, multiplicity * sign) for p, sign in permutations)):
                return
        subtotal = self.xp.zeros(len(rows))
        for p, sign in permutations:
            subtotal += (multiplicity * sign) * tile[tuple(particles[i] for i in p)]
        destination[rows] += subtotal

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
                                     engine=self._symmetry_engine,
                                     occupied_slab=self.occupied_slab).reshape((nv,) * rank)
                for p, sign in permutations:
                    subtotal += sign * tile[tuple(particles[i] for i in p)]
        return subtotal / len(_permutations(rank))**2

    def _action(self, vector, *, ranks=range(1, 5), slot_limit=None, rank_finished=None, slot_finished=None):
        xp = self.xp
        self._gather_plans = {}
        no = self.proxy.hamiltonian.nocc
        nonzero = (tuple(bool(xp.any(vector[section])) for section in self.space.slices)
                   if self.occupied_tiles else None)
        r1, *direction = self.space.unpack(vector, max_rank=3 if self.occupied_tiles else 4, nonzero=nonzero)
        dg = df = None
        if r1 is not None:
            dg = dress_direction(self.integrals, r1, no, xp)
            df = dress_direction(self.one_body, r1, no, xp) + xp.einsum("piqi->pq", dg[:, :no, :, :no])
        blocks = {"f" + key: Directional(self.fock[ix], None if df is None else df[ix])
                  for key, ix in self.proxy._fblocks.items()}
        # T1 dressing changes virtual creator and occupied annihilator legs.
        # The oovv block has neither, so avoid expanding T4 for a zero product.
        for key, ix in self.proxy._blocks.items():
            tangent = dg[ix] if dg is not None and (not self.occupied_tiles or key != "oovv") else None
            blocks["g" + key] = Directional(self.integrals[ix], tangent)
        blocks.update({f"t{rank}": Directional(t, r) for rank, (t, r) in
                       enumerate(zip(self.amplitudes[1:], direction), 2)})
        packed_action = None
        if self.packed_contractions:
            from ._spin_quadruples_cpu import PackedBinaryAction
            packed_action = PackedBinaryAction(self, blocks,
                                              vector[self.space.slices[3]] if nonzero[3] else None)
        outputs = []
        self._program_cache = None
        from ._intermediate_cache import IntermediateCache
        intermediates = IntermediateCache(self.intermediate_cache_bytes) if self.intermediate_cache_bytes else None
        for rank in ranks:
            start = perf_counter() if rank_finished is not None else 0.
            shape = self.space.shapes[rank - 1]
            if self.occupied_tiles and rank > 1:
                r4 = vector[self.space.slices[3]] if nonzero[3] else None
                if r4 is not None and not self._packed_direction:
                    r4 = r4[self._tiled_r4_order]
                tiles = self._tiled_action(blocks, rank, r4, slot_limit=slot_limit,
                                           slot_finished=slot_finished, packed_action=packed_action,
                                           intermediates=intermediates)
                if rank == 4:
                    outputs.append(next(tiles)[1])
                else:
                    outputs.append(self._project_lower_tiles(rank, tiles))
            else:
                result = xp.zeros(shape)
                for coefficient, spec, names in DIAGRAMS[rank]:
                    for operands in product_directions(*(blocks[name] for name in names)):
                        einsum_into(xp, result, coefficient, spec, *operands)
                outputs.append(_antisymmetrize(xp, result)[tuple(self.space.indices[rank - 1])])
            if packed_action is not None and 1 < rank < 4:
                packed_action.add(rank, self.space.indices[rank - 1].T, outputs[-1])
            if rank_finished is not None:
                if xp is not np:
                    xp.cuda.get_current_stream().synchronize()
                rank_finished(rank, perf_counter() - start)
        self.last_intermediate_cache = (self._program_cache if self._program_cache is not None else
                                        None if intermediates is None else
                                        dict(hits=intermediates.hits, misses=intermediates.misses,
                                             peak_bytes=intermediates.peak_bytes))
        self._gather_plans = {}
        return xp.concatenate(outputs)

    def probe_rank_four(self, vector, slots):
        """Measure a bounded prefix of occupied rank-four slots."""
        if not self.action_only or slots < 1:
            raise ValueError("Rank-four probing needs the action-only packed layout and positive slots")
        return self._action(vector, ranks=(4,), slot_limit=slots)

    def probe_lower_ranks(self, vector):
        """Apply the full Jacobian into ranks one through three, including R4 feedback.

        The input uses the action-only layout with a packed rank-four tail.
        The output uses the canonical coordinates of ranks one through three.
        """
        if not self.action_only:
            raise ValueError("The feedback probe needs the action-only packed layout")
        return self._action(vector, ranks=range(1, 4))

    def solve(self, *args, **options):
        if self.action_only:
            raise ValueError("The action-only layout has no large-system triplet spin projector")
        return EOMSinglet.solve(self, *args, **options)

    def davidson_working_bytes(self, max_space):
        """Conservative tensor workspace plus packed Davidson vectors.

        The logical rank-four shape is not a dense allocation on the packed
        path. Reuse the constructor's tiled estimate, including its cache.
        """
        estimate = int(self.estimated_dense_bytes) + (2 * max_space + 6) * self.space.size * 8
        if not self._packed_direction:
            estimate += 7 * sum(int(np.prod(shape)) for shape in self.space.shapes) * 8
        return estimate

    def sigma(self, vector):
        if self.action_only:
            raise ValueError("Use probe_rank_four for the action-only layout")
        if self.xp.iscomplexobj(vector):
            return self.sigma(vector.real) + 1j * self.sigma(vector.imag)
        start = perf_counter()
        from ._cpu_triples import thread_scope

        with thread_scope(None):                 # one thread count for the whole action
            result = self._action(vector)
        if not bool(self.xp.isfinite(result).all()):
            raise FloatingPointError("Nonfinite spin-flip Jacobian action")
        self.matvec_count += 1
        self.matvec_seconds += perf_counter() - start
        return result

    def full_residual_norm(self, vector, energy):
        return float(self.xp.linalg.norm(self.sigma(vector) - energy * vector))
