"""Blocked T2 source and virtual one-body actions in triples assembly."""

from ._cuda_feedback import _Product


class CudaAssembly:
    def __init__(self, occupied, virtual, xp):
        self.xp = xp
        self.occupied = occupied
        self.no, self.nv = len(occupied), len(virtual)
        pairs = virtual[:, None] ^ virtual
        ov = occupied[:, None] ^ virtual
        ovv = occupied[:, None, None] ^ pairs
        self.seed_product = _Product(pairs.ravel(), virtual, ov.ravel(), xp)
        self.virtual_product = _Product(virtual, virtual, ovv.ravel(), xp)
        self.forbidden = xp.asarray((occupied[:, None, None, None] ^ occupied[None, :, None, None]
                                     ^ pairs) != 0)

    def prepare(self, t2, source, matrix):
        seeds = [None] * self.no
        if not bool(self.xp.any(t2[self.forbidden])):
            seeds = [self.seed_product.prepare(source[:, :, :, j], int(label))
                     for j, label in enumerate(self.occupied)]
        virtual = self.virtual_product.prepare(matrix, 0)
        return _Actions(self, seeds, virtual)


class _Actions:
    def __init__(self, owner, seeds, virtual):
        self.owner, self.seeds, self.virtual = owner, seeds, virtual

    def seed(self, t2, j, k):
        if self.seeds[j] is None:
            return None
        owner = self.owner
        shift = int(owner.occupied[j] ^ owner.occupied[k])
        result = owner.seed_product.multiply(self.seeds[j], t2[:, k].transpose(1, 0, 2), shift)
        return result.reshape(owner.nv, owner.nv, owner.no, owner.nv).transpose(2, 0, 1, 3)

    def one_body(self, plane, j, k):
        if self.virtual is None:
            return None
        owner = self.owner
        shift = int(owner.occupied[j] ^ owner.occupied[k])
        result = owner.virtual_product.multiply(self.virtual, plane[:, j].transpose(1, 0, 2, 3), shift)
        return result.reshape(owner.nv, owner.no, owner.nv, owner.nv).transpose(1, 0, 2, 3)
