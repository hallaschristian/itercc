"""CUDA kernels for the occupied-triangular triples layout."""

from functools import cache

import numpy as np

from ._cuda_packed import PackedTriples
from ._triples_layout import TriplesLayout


@cache
def _crossed_kernel():
    from cupy import ElementwiseKernel

    return ElementwiseKernel("float64 seed, float64 crossed, float64 swapped", "float64 updated",
                             "updated = (seed - swapped) - 0.5 * crossed",
                             "itercc_subtract_crossed", options=("--fmad=false",))


@cache
def _feedback_sum_kernel():
    from cupy import ElementwiseKernel

    return ElementwiseKernel("float64 direct, float64 exchange", "float64 result",
                             "result = 2.0 * direct - exchange",
                             "itercc_triples_feedback_sum", options=("--fmad=false",))


@cache
def _kernels():
    from cupy import ElementwiseKernel

    plane = ElementwiseKernel(
        "raw float64 source, raw int32 plan, int32 nv, int32 nv3, bool spin",
        "float64 result",
        """
        const int row = (i / nv3) * 4;
        const int slot = plan[row];
        if (slot < 0) {
            result = 0;
        } else {
            const int rem = i % nv3;
            const int a = rem / (nv * nv);
            const int b = (rem / nv) % nv;
            const int c = rem % nv;
            const int sa = plan[row + 1], sb = plan[row + 2], sc = plan[row + 3];
            const long long base = static_cast<long long>(slot) * nv3;
            const double abc = source[base + a * sa + b * sb + c * sc];
            if (spin) {
                const double bac = source[base + b * sa + a * sb + c * sc];
                const double cba = source[base + c * sa + b * sb + a * sc];
                result = (2.0 * abc - bac) - cba;
            } else {
                result = abc;
            }
        }
        """, "itercc_unpack_triples_plane")
    add = ElementwiseKernel(
        "raw float64 seed, raw int32 plan, int32 nv, int32 nv3", "raw float64 target",
        """
        const int row = (i / nv3) * 5;
        const int slot = plan[row];
        if (slot >= 0) {
            const int rem = i % nv3;
            const int a = rem / (nv * nv);
            const int b = (rem / nv) % nv;
            const int c = rem % nv;
            const int sa = plan[row + 1], sb = plan[row + 2], sc = plan[row + 3];
            const int swap = plan[row + 4];
            const long long base = (i / nv3) * nv3;
            const long long index = static_cast<long long>(slot) * nv3 + rem;
            double value = target[index] + seed[base + a * sa + b * sb + c * sc];
            if (swap == 1) value += seed[base + b * sa + a * sb + c * sc];
            if (swap == 2) value += seed[base + a * sa + c * sb + b * sc];
            target[index] = value;
        }
        """, "itercc_add_triples_seed")
    spin = ElementwiseKernel(
        "float64 abc, float64 bac, float64 cba", "float64 result",
        "result = (2.0 * abc - bac) - cba", "itercc_triples_spin_sum")
    purify = ElementwiseKernel(
        "float64 abc, float64 bac, float64 cba, float64 acb, float64 cab, float64 bca, int32 nv",
        "float64 result",
        """
        const int a = (i / (nv * nv)) % nv;
        const int b = (i / nv) % nv;
        const int c = i % nv;
        const double left = (abc + bac) + cba;
        const double right = (acb + cab) + bca;
        result = a == b && b == c ? 0.0 : abc - (left + right) / 6.0;
        """, "itercc_purify_triples")
    return plane, add, spin, purify


class CudaTriplesLayout(TriplesLayout):
    """Gather planes and fold each seed with one device launch."""

    def __init__(self, no, nv, xp):
        super().__init__(no, nv)
        self.xp = xp
        strides = np.array([nv**2, nv, 1], dtype=np.int32)
        plan = np.empty((2, no, no, no, 4), dtype=np.int32)
        fold = np.empty((no, no, no, 5), dtype=np.int32)
        for i, j, k in np.ndindex((no,) * 3):
            key = i, j, k
            slot = self.slots[key]
            gather = (slot, *strides[self.from_sorted[key]])
            plan[0, i, j, k] = plan[1, k, i, j] = gather
            repeat = 0
            if slot >= 0:
                u, v, w = self.indices[slot]
                repeat = 1 if u == v else 2 if v == w else 0
            fold[j, k, i] = (slot, *strides[self.to_sorted[key]], repeat)
        self._plane_plan = xp.asarray(plan)
        self._fold_plan = xp.asarray(fold)

    def _plane(self, data, fixed, axis, spin):
        if isinstance(data, PackedTriples):
            return data.plane(self._plane_plan[axis // 2, fixed], spin)
        plane, _, _, _ = _kernels()
        nv = self.nv
        shape = self.no, self.no, nv, nv, nv
        plan = self._plane_plan[axis // 2, fixed]
        value = plane(self.xp.ascontiguousarray(data), plan, nv, nv**3, spin,
                      size=self.no**2 * nv**3)
        return value.reshape(shape)

    def plane(self, data, fixed, *, axis, xp=np):
        return self._plane(data, fixed, axis, False)

    def spin_plane(self, data, fixed, *, xp=np):
        return self._plane(data, fixed, 0, True)

    def spin_sum(self, data):
        _, _, spin, _ = _kernels()
        return spin(data, data.swapaxes(-3, -2), data.swapaxes(-3, -1))

    @staticmethod
    def feedback_sum(plane):
        return _feedback_sum_kernel()(plane, plane.swapaxes(-2, -1))

    @staticmethod
    def subtract_crossed_fused(seed, tile):
        _crossed_kernel()(seed, tile, tile.swapaxes(1, 2), seed)

    def add_seed(self, target, seed, j, k, *, occupied=None):
        seed = self.xp.ascontiguousarray(seed)
        plan = self._fold_plan[j, k]
        if occupied is not None:
            occupied.fold(target, seed, plan, j, self.nv)
            return
        if isinstance(target, PackedTriples):
            raise ValueError("Packed triples images require the compact occupied fold")
        _, add, _, _ = _kernels()
        add(seed, plan, self.nv, self.nv**3, target, size=seed.size)

    def purify(self, triples, *, xp=np):
        if isinstance(triples, PackedTriples):
            triples.purify()
            return
        _, _, _, purify = _kernels()
        for start in range(0, len(triples), 16):
            tiles = triples[start:start + 16]
            tiles[...] = purify(tiles, tiles.transpose(0, 2, 1, 3), tiles.transpose(0, 3, 2, 1),
                                tiles.transpose(0, 1, 3, 2), tiles.transpose(0, 2, 3, 1),
                                tiles.transpose(0, 3, 1, 2), self.nv)


@cache
def _projection_kernel():
    from cupy import ElementwiseKernel

    return ElementwiseKernel(
        "T abc, T bac, T cba, T acb, T cab, T bca, raw int8 repeats, int32 nv",
        "T projected",
        """
        const int a = (i / (nv * nv)) % nv;
        const int b = (i / nv) % nv;
        const int c = i % nv;
        const int repeat = repeats[i / (nv * nv * nv)];
        T paired = abc;
        if (repeat == 1) paired = (abc + bac) * 0.5;
        if (repeat == 2) paired = (abc + acb) * 0.5;
        const T symmetric = ((abc + bac) + cba) + ((acb + cab) + bca);
        projected = a == b && b == c ? T(0) : paired - symmetric / 6.0;
        """, "itercc_project_eom_triples", options=("--fmad=false",))


def project_eom_triples(triples, repeats, nv):
    """Project repeated occupied pairs and the null virtual component together."""
    project = _projection_kernel()
    for start in range(0, len(triples), 16):
        tiles = triples[start:start + 16]
        # Allocate a temporary output so each thread reads the unmodified tile.
        tiles[...] = project(tiles, tiles.transpose(0, 2, 1, 3),
                             tiles.transpose(0, 3, 2, 1), tiles.transpose(0, 1, 3, 2),
                             tiles.transpose(0, 3, 1, 2), tiles.transpose(0, 2, 3, 1),
                             repeats[start:start + 16], nv)
