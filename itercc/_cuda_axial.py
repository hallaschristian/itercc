"""Real virtual-pair blocks for the axial angular-momentum representation."""

from functools import cache
from typing import NamedTuple

import numpy as np

from ._cuda_pairs import CudaPairPacking


class _Transform(NamedTuple):
    forward: tuple
    reverse: tuple
    norms: object
    groups: object


def _table(rows):
    width = max(map(len, rows), default=1)
    indices = np.zeros((len(rows), width), dtype=np.int32)
    values = np.zeros_like(indices, dtype=float)
    for i, row in enumerate(rows):
        for j, (index, value) in enumerate(row):
            indices[i, j], values[i, j] = index, value
    return indices, values


def real_pair_transform(momentum, irreps, partners, parity):
    """Build an orthonormal real basis from conjugate signed-m pair states."""
    nv = len(momentum)
    if (partners.dtype.kind not in "iu" or momentum.dtype.kind not in "iu"
            or len(partners) != nv or len(irreps) != nv
            or np.any(partners < 0) or np.any(partners >= nv)
            or not np.array_equal(partners[partners], np.arange(nv))
            or not np.array_equal(momentum[partners], -momentum)
            or np.any((momentum == 0) != (partners == np.arange(nv)))):
        raise ValueError("Expected matched x/y partners for every nonzero angular momentum")
    left, right = np.triu_indices(nv, int(parity == -1))
    pairs = tuple(zip(left, right))
    index = {pair: i for i, pair in enumerate(pairs)}
    labels = irreps[left] ^ irreps[right]
    half = np.sqrt(0.5)
    orbitals = []
    for a, m in enumerate(momentum):
        if m == 0:
            orbitals.append(((a, 1),))
        elif m > 0:
            orbitals.append(((a, half), (partners[a], 1j * half)))
        else:
            orbitals.append(((partners[a], half), (a, -1j * half)))
    columns, groups = [], []
    for p, q in pairs:
        conjugate = tuple(sorted((partners[p], partners[q])))
        if (p, q) > conjugate:
            continue
        vector = {}
        for a, x in orbitals[p]:
            for b, y in orbitals[q]:
                if a == b and parity == -1:
                    continue
                key = index[min(a, b), max(a, b)]
                sign = parity if a > b else 1
                scale = 1 if (p == q) == (a == b) else (half if p == q else 1 / half)
                vector[key] = vector.get(key, 0.0) + sign * scale * x * y
        scale = 1 if (p, q) == conjugate else 1 / half
        for part in (np.real, np.imag):
            column = [(k, float(part(v) * scale)) for k, v in vector.items() if abs(part(v)) > 1e-13]
            if column:
                sectors = {int(labels[k]) for k, _ in column}
                if len(sectors) != 1:
                    raise ValueError("Angular partners do not share a consistent C2v representation")
                columns.append(column)
                groups.append(4 * abs(int(momentum[p] + momentum[q])) + sectors.pop())
    if len(columns) != len(pairs):
        raise RuntimeError("Angular transform did not span the real pair space")
    order = np.argsort(groups, kind="stable")
    groups = np.asarray(groups, dtype=np.int32)[order]
    columns = [columns[i] for i in order]
    reverse = [[] for _ in pairs]
    for j, column in enumerate(columns):
        for i, value in column:
            reverse[i].append((j, value))
    return _Transform(_table(columns), _table(reverse), np.where(left == right, 1.0, 1 / half), groups)


@cache
def _kernels():
    from cupy import ElementwiseKernel

    right = ElementwiseKernel(
        "raw float64 a, raw int32 indices, raw float64 coeff, raw float64 norms, int32 n, int32 width",
        "float64 result",
        r"""
        const int row = i / n, col = i % n;
        double sum = 0.0;
        for (int p = 0; p < width; ++p) {
            const int entry = col * width + p;
            const int k = indices[entry];
            if (coeff[entry] != 0) sum += a[static_cast<long long>(row) * n + k] * coeff[entry] / norms[k];
        }
        result = sum * norms[row];
        """, "itercc_axial_right")
    left = ElementwiseKernel(
        "raw float64 a, raw int32 indices, raw float64 coeff, int32 n, int32 width",
        "float64 result",
        r"""
        const int row = i / n, col = i % n;
        double sum = 0.0;
        for (int p = 0; p < width; ++p) {
            const int entry = row * width + p;
            if (coeff[entry] != 0) sum += coeff[entry] * a[static_cast<long long>(indices[entry]) * n + col];
        }
        result = sum;
        """, "itercc_axial_left")
    pack = ElementwiseKernel(
        "raw float64 source, raw int32 pairs, raw int32 indices, raw float64 coeff, "
        "raw int32 columns, int32 n, int32 width, int32 first, int32 nv, "
        "int32 sa, int32 sb, int32 sc, int32 parity", "float64 result",
        r"""
        const int row = first + i / width;
        const int col = columns[i % width];
        const long long base = static_cast<long long>(col / nv) * nv * nv * nv + (col % nv) * sc;
        double sum = 0.0;
        for (int p = 0; p < n; ++p) {
            const int entry = row * n + p;
            if (coeff[entry] == 0.0) continue;
            const int pair = indices[entry];
            const int a = pairs[2 * pair], b = pairs[2 * pair + 1];
            const double ab = source[base + a * sa + b * sb];
            const double ba = source[base + b * sa + a * sb];
            sum += coeff[entry] * (parity == 1 ? ab + ba : ab - ba);
        }
        result = 0.5 * sum;
        """, "itercc_pack_axial_ladder")
    restore = ElementwiseKernel(
        "float64 current, raw float64 packed, raw int64 bases, raw int32 rows, "
        "raw int32 indices, raw float64 coeff, raw uint8 pair_labels, raw uint8 columns, "
        "raw int32 ranks, int32 n, int32 nv, int32 nv3, int32 orientation, int32 parity",
        "float64 result",
        r"""
        const int rem = i % nv3;
        const int v0 = rem / (nv * nv), v1 = (rem / nv) % nv, v2 = rem % nv;
        const int a = orientation == 2 ? v1 : v0;
        const int b = orientation == 0 ? v1 : v2;
        const int c = orientation == 0 ? v2 : (orientation == 1 ? v1 : v0);
        const int pair = a * nv + b;
        const int col = (i / nv3) * nv + c;
        result = current;
        if ((parity == 1 || a != b) && pair_labels[pair] == columns[col]) {
            const int row = rows[pair];
            const int rank = ranks[col];
            double sum = 0.0;
            for (int p = 0; p < n; ++p) {
                const int entry = row * n + p;
                if (coeff[entry] != 0.0) sum += coeff[entry] * packed[bases[indices[entry]] + rank];
            }
            result += (parity == -1 && a > b) ? -sum : sum;
        }
        """, "itercc_restore_axial_ladder")
    return right, left, pack, restore


class CudaAxialLadder:
    """Only the virtual ladder changes gauge. Other CCSDT terms stay real and unchanged."""

    def __init__(self, momentum, irreps, partners, xp):
        self.xp = xp
        self.nv = len(momentum)
        self.irreps = irreps.copy()
        self.pair_labels = xp.asarray(irreps[:, None] ^ irreps, dtype=xp.uint8)
        self.packing = CudaPairPacking(self.nv, np.triu_indices(self.nv),
                                       np.triu_indices(self.nv, 1), xp)
        self.plans = []
        for parity in (1, -1):
            forward, reverse, norms, groups = real_pair_transform(momentum, irreps, partners, parity)
            index, values = forward
            rev_index, rev_values = reverse
            ids, counts = np.unique(groups, return_counts=True)
            ends = np.r_[0, np.cumsum(counts)]
            self.plans.append(dict(
                forward=(xp.asarray(index), xp.asarray(values)),
                scaled=(xp.asarray(index), xp.asarray(values * norms[index])),
                reverse=(xp.asarray(rev_index), xp.asarray(rev_values / norms[:, None])),
                norms=xp.asarray(norms), groups=xp.asarray(groups),
                blocks=tuple((int(g % 4), slice(int(a), int(b))) for g, a, b in zip(ids, ends, ends[1:]))))

    def _blocks(self, matrix, plan):
        xp = self.xp
        right, left, _, _ = _kernels()
        n = len(matrix)
        if not n:
            return [], 0.0
        indices, coeff = plan["forward"]
        width = indices.shape[1]
        first = right(matrix, indices, coeff, plan["norms"], n, width, size=n*n)
        rotated = left(first, indices, coeff, n, width, size=n*n).reshape(n, n)
        del first
        groups = plan["groups"]
        maxima = [xp.max(xp.where(groups[start:start + 128, None] == groups, 0.0,
                                   xp.abs(rotated[start:start + 128])))
                  for start in range(0, n, 128)]
        error = float(xp.max(xp.stack(maxima)).item())
        return [(label, band, xp.ascontiguousarray(rotated[band, band]))
                for label, band in plan["blocks"]], error

    def apply(self, target, source, w, occupied_labels, batch, *, tolerance):
        """Apply the audited axial blocks, or return False before touching the target."""
        xp, nv = self.xp, self.nv
        matrices = self.packing.matrices(w)
        transformed = [self._blocks(matrix, plan) for matrix, plan in zip(matrices, self.plans)]
        del matrices
        self.leakage = max(value for _, value in transformed)
        if not np.isfinite(self.leakage) or self.leakage > tolerance:
            return False
        _, _, pack, restore = _kernels()
        for start in range(0, len(source), batch):
            stop = start + batch
            tile = xp.ascontiguousarray(source[start:stop])
            destination = target[start:stop]
            labels = (occupied_labels[start:stop, None] ^ self.irreps).ravel()
            columns = [np.flatnonzero(labels == j).astype(np.int32) for j in range(4)]
            ranks = np.empty(len(labels), dtype=np.int32)
            for selected in columns:
                ranks[selected] = np.arange(len(selected), dtype=np.int32)
            dev_labels, dev_ranks = xp.asarray(labels, dtype=xp.uint8), xp.asarray(ranks)
            dev_columns = [xp.asarray(indices) for indices in columns]
            for parity_index, ((blocks, _), plan) in enumerate(zip(transformed, self.plans)):
                if not blocks:
                    continue
                sign = 1 - 2 * parity_index
                bases = np.empty(len(plan["groups"]), dtype=np.int64)
                products = []
                end = 0
                for label, band, matrix in blocks:
                    width = len(columns[label])
                    rows = len(matrix)
                    bases[band] = end + np.arange(rows, dtype=np.int64) * width
                    if width:
                        end += rows * width
                        products.append((band.start, dev_columns[label], matrix, slice(end - rows*width, end)))
                bases = xp.asarray(bases)
                result = xp.empty(end, dtype=xp.float64)
                indices, coeff = plan["scaled"]
                rev_indices, rev_values = plan["reverse"]
                for orientation, permutation in enumerate(((0, 1, 2), (0, 2, 1), (1, 2, 0))):
                    strides = tuple(nv ** (2 - a) for a in permutation)
                    for first, chosen, matrix, output in products:
                        rows, width = len(matrix), len(chosen)
                        rhs = pack(tile, self.packing.indices[parity_index], indices, coeff, chosen,
                                   indices.shape[1], width, first, nv, *strides, sign,
                                   size=rows * width).reshape(rows, width)
                        xp.matmul(matrix, rhs, out=result[output].reshape(rows, width))
                    restore(destination, result, bases, self.packing.rows[parity_index],
                            rev_indices, rev_values, self.pair_labels, dev_labels, dev_ranks,
                            rev_indices.shape[1], nv, nv**3, orientation, sign, destination)
        return True