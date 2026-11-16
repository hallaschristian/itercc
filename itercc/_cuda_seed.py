"""Build a triples seed tile with in-place FP64 cuBLAS contractions."""

import numpy as np


class CudaTriplesSeed:
    def __init__(self, xp, no, nv, batch):
        from cupy.cuda import cublas, device

        self.xp, self.cublas = xp, cublas
        self.handle = device.get_cublas_handle()
        self.no, self.nv = no, nv
        self.workspace = xp.empty(no * batch * nv**3)
        self.scalars = np.array([0., 1., -1., .5], dtype=np.float64)
        self.zero, self.one, self.minus_one, self.half = (
            self.scalars.ctypes.data + i * self.scalars.itemsize for i in range(len(self.scalars))
        )
        self._buffer = None

    def _gemm(self, output, left, right, rows, columns, inner, alpha, beta):
        # Row-major C = A B is column-major C^T = B^T A^T.
        blas = self.cublas
        blas.dgemm(self.handle, blas.CUBLAS_OP_N, blas.CUBLAS_OP_N, columns, rows, inner,
                   alpha, right.data.ptr, columns, left.data.ptr, inner, beta,
                   output.data.ptr, columns)

    def _batched(self, output, left, right, rows, columns, inner, alpha, beta):
        blas = self.cublas
        blas.dgemmBatched(self.handle, blas.CUBLAS_OP_N, blas.CUBLAS_OP_N, columns, rows, inner,
                          alpha, right.data.ptr, columns, left.data.ptr, inner, beta,
                          output.data.ptr, columns, len(output))

    def prepare_doubles(self, t2, k):
        xp, no, nv = self.xp, self.no, self.nv
        value, tangent = (xp.ascontiguousarray(a[:, k]) for a in (t2.value, t2.tangent))
        # [T2(i,c,d) | dT2(i,c,d)] is the left factor of the stacked VVVO
        # product; [T2 ; dT2](l, bc) is the right factor of the VOOO product.
        stacked = xp.empty((no * nv, 2 * nv))
        stacked[:, :nv] = value.transpose(0, 2, 1).reshape(no * nv, nv)
        stacked[:, nv:] = tangent.transpose(0, 2, 1).reshape(no * nv, nv)
        lines = xp.concatenate((value.reshape(no, nv**2), tangent.reshape(no, nv**2)))
        return value, tangent, stacked, lines

    def _scratch(self, size):
        if self._buffer is None or self._buffer.size < size:
            self._buffer = self.xp.empty(size)
        return self._buffer[:size]

    def assemble(self, selection, doubles, vvvo, vooo, vv, plane, tangent, occupied, doccupied, *,
                 include_occupied=True, include_virtual=True):
        """Seed tile for occupied lines `selection` at fixed k.

        With include_virtual=False the caller must add the virtual one-body
        term elsewhere (the tangent assembly folds it into the pair ladder).
        """
        xp, blas, no, nv = self.xp, self.cublas, self.no, self.nv
        width = selection.stop - selection.start
        seed = self.workspace[:no * width * nv**3].reshape(no, width, nv, nv, nv)
        t2, dt2, stacked_t2, stacked_lines = doubles
        mode = blas.getPointerMode(self.handle)
        blas.setStream(self.handle, xp.cuda.get_current_stream().ptr)
        if mode != blas.CUBLAS_POINTER_MODE_HOST:
            blas.setPointerMode(self.handle, blas.CUBLAS_POINTER_MODE_HOST)
        try:
            # VVVO: one (i,c) x (a,b) GEMM per line with K = 2 nv stacks the
            # value/tangent pair, [T2|dT2](ic,d') [dVVVO;VVVO](d',ab). This
            # replaces 2 no thin (ab x c x d) GEMMs; the transpose is cheap.
            product = self._scratch(no * nv**3).reshape(no * nv, nv**2)
            right = xp.empty((2 * nv, nv**2))
            for line, j in enumerate(range(selection.start, selection.stop)):
                right[:nv] = vvvo.tangent[:, :, :, j].transpose(2, 0, 1).reshape(nv, nv**2)
                right[nv:] = vvvo.value[:, :, :, j].transpose(2, 0, 1).reshape(nv, nv**2)
                self._gemm(product, stacked_t2, right, no * nv, nv**2, 2 * nv, self.one, self.zero)
                seed[:, line] = product.reshape(no, nv, nv, nv).transpose(0, 2, 3, 1)
            del right
            # VOOO: one GEMM with K = 2 no, [dVOOO|VOOO](i j a, l) [T2;dT2](l, bc).
            left = xp.empty((no, width, nv, 2 * no))
            left[..., :no] = vooo.tangent.transpose(2, 3, 0, 1)[:, selection]
            left[..., no:] = vooo.value.transpose(2, 3, 0, 1)[:, selection]
            self._gemm(seed, left, stacked_lines, no * width * nv, nv**2, 2 * no,
                       self.minus_one, self.one)
            del left
            if include_virtual:
                # out(a, i bc) = 1/2 [vv | dvv](a, d') [dP ; P](d', i bc) as one
                # wide GEMM per line, then a transposed accumulation.
                stacked = xp.empty((2 * nv, no, nv**2))
                matrices = xp.empty((nv, 2 * nv))
                matrices[:, :nv], matrices[:, nv:] = vv.value, vv.tangent
                product = product.reshape(nv, no * nv**2)
                for line, j in enumerate(range(selection.start, selection.stop)):
                    stacked[:nv] = tangent[:, j].reshape(no, nv, nv**2).transpose(1, 0, 2)
                    stacked[nv:] = plane[:, j].reshape(no, nv, nv**2).transpose(1, 0, 2)
                    self._gemm(product, matrices, stacked, nv, no * nv**2, 2 * nv,
                               self.half, self.zero)
                    seed[:, line] += product.reshape(nv, no, nv, nv).transpose(1, 0, 2, 3)
                del stacked, matrices
            if include_occupied:
                for matrix, source in ((occupied, tangent), (doccupied, plane)):
                    left = xp.ascontiguousarray(matrix.reshape(no, no, no**2)[:, selection])
                    self._gemm(seed, left, source, no * width, nv**3, no**2, self.one, self.one)
        finally:
            if mode != blas.CUBLAS_POINTER_MODE_HOST:
                blas.setPointerMode(self.handle, mode)
        return seed
