"""Analytic forward directional derivatives of NumPy/CuPy tensor operations.

No finite differencing or autodiff dependency. A missing tangent denotes an
exact constant, avoiding zero contractions for fixed Hamiltonian operands.
"""
import numpy as np


class Directional:
    __array_priority__ = 1000

    def __init__(self, value, tangent=None):
        self.value, self.tangent = value, tangent

    @property
    def shape(self):
        return self.value.shape

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def size(self):
        return self.value.size

    @property
    def T(self):
        return self.transpose()

    def __len__(self):
        return len(self.value)

    def __getitem__(self, key):
        return Directional(self.value[key], None if self.tangent is None else self.tangent[key])

    def __setitem__(self, key, other):
        other = lift(other)
        self.value[key] = other.value
        if other.tangent is not None and self.tangent is None:
            # All mutable allocations are initialized with tangent storage by
            # the backend; this branch is for non-view constant temporaries.
            self.tangent = self.value * 0
        if self.tangent is not None:
            self.tangent[key] = 0 if other.tangent is None else other.tangent

    def copy(self):
        return Directional(self.value.copy(), None if self.tangent is None else self.tangent.copy())

    def reshape(self, *shape):
        return Directional(self.value.reshape(*shape),
                           None if self.tangent is None else self.tangent.reshape(*shape))

    def transpose(self, *axes):
        return Directional(self.value.transpose(*axes),
                           None if self.tangent is None else self.tangent.transpose(*axes))

    def swapaxes(self, a, b):
        return Directional(self.value.swapaxes(a, b),
                           None if self.tangent is None else self.tangent.swapaxes(a, b))

    def fill(self, value):
        self.value.fill(value)
        if self.tangent is not None:
            self.tangent.fill(0)

    def __neg__(self):
        return Directional(-self.value, None if self.tangent is None else -self.tangent)

    def __add__(self, other):
        other = lift(other)
        d = other.tangent if self.tangent is None else self.tangent
        if d is not None:
            d = d.copy()
        if self.tangent is not None and other.tangent is not None:
            d = self.tangent + other.tangent
        return Directional(self.value + other.value, d)

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-lift(other))

    def __rsub__(self, other):
        return lift(other) + (-self)

    def __mul__(self, other):
        other = lift(other)
        d = None if self.tangent is None else self.tangent * other.value
        if other.tangent is not None:
            term = self.value * other.tangent
            d = term if d is None else d + term
        return Directional(self.value * other.value, d)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Directional):
            raise TypeError("Variable denominators are not part of CC residuals")
        return self * (1 / other)

    def __matmul__(self, other):
        return product(lambda a, b: a @ b, self, other)

    def __iadd__(self, other):
        other = lift(other)
        if other.tangent is not None:
            if self.tangent is None:
                self.tangent = self.value * 0
            self.tangent += other.tangent
        self.value += other.value
        return self

    def __isub__(self, other):
        return self.__iadd__(-lift(other))

    def __imul__(self, other):
        result = self * other
        self.value[...] = result.value
        if result.tangent is not None:
            if self.tangent is None:
                self.tangent = result.tangent
            else:
                self.tangent[...] = result.tangent
        return self


def lift(value):
    return value if isinstance(value, Directional) else Directional(value)


def product(operation, *operands):
    operands = tuple(map(lift, operands))
    result = operation(*(a.value for a in operands))
    derivative = None
    for args in product_directions(*operands):
        term = operation(*args)
        derivative = term if derivative is None else derivative + term
    return Directional(result, derivative)


def product_directions(*operands):
    """Yield the operand tuples in the product rule, without evaluating the primal."""
    values = tuple(operand.value for operand in operands)
    for i, operand in enumerate(operands):
        if operand.tangent is not None:
            yield (*values[:i], operand.tangent, *values[i + 1:])


def dress_direction(value, direction, nocc, xp):
    """Differentiate the exact T1 similarity transform on one or two body legs."""
    from .solver import _dress_axis

    occupied, virtual = slice(nocc), slice(nocc, None)
    result = xp.zeros_like(value)
    creators = value.ndim // 2
    for axis in range(value.ndim):
        target, source = (virtual, occupied) if axis < creators else (occupied, virtual)
        rotation = -direction if axis < creators else direction.T
        _dress_axis(result, axis, target, source, rotation, xp, operand=value)
    return result


class DirectionalBackend:
    ndarray = Directional

    def __init__(self, xp):
        self.xp = xp
        # NumPy: two-operand products of primal and tangent parts on the
        # compiled GEMM executor (`_cpu_triples.CompiledContract`, the engine
        # of the ground-state solvers), NumPy's einsum otherwise.
        self._product2 = None
        if xp is np:
            from . import _cpu_kernels

            if _cpu_kernels.library() is not None:
                from ._cpu_triples import CompiledContract

                self._product2 = CompiledContract()

    def array(self, value, **kwargs):
        value = lift(value)
        # Mutable constant arrays need an allocated zero tangent so that all
        # subsequent slices share derivative storage with their parent.
        primal = self.xp.array(value.value, **kwargs)
        tangent = (self.xp.zeros_like(primal) if value.tangent is None else
                   self.xp.array(value.tangent, **kwargs))
        return Directional(primal, tangent)

    def asarray(self, value):
        value = lift(value)
        return Directional(self.xp.asarray(value.value), None if value.tangent is None
                           else self.xp.asarray(value.tangent))

    def ascontiguousarray(self, value):
        value = lift(value)
        return Directional(self.xp.ascontiguousarray(value.value), None if value.tangent is None
                           else self.xp.ascontiguousarray(value.tangent))

    def empty(self, shape):
        return Directional(self.xp.empty(shape), self.xp.zeros(shape))

    def zeros(self, shape):
        return Directional(self.xp.zeros(shape), self.xp.zeros(shape))

    def empty_like(self, value):
        return self.empty(value.shape)

    def zeros_like(self, value):
        return self.zeros(value.shape)

    def moveaxis(self, value, source, target):
        value = lift(value)
        return Directional(self.xp.moveaxis(value.value, source, target),
                           None if value.tangent is None else
                           self.xp.moveaxis(value.tangent, source, target))

    def tensordot(self, a, b, axes=1):
        return product(lambda x, y: self.xp.tensordot(x, y, axes=axes), a, b)

    def einsum(self, spec, *operands, **kwargs):
        if self._product2 is not None and len(operands) == 2:
            return product(lambda a, b: self._product2(spec, a, b), *operands)
        return product(lambda *args: self.xp.einsum(spec, *args, **kwargs), *operands)

    def triu_indices(self, *args):
        return self.xp.triu_indices(*args)

    def arange(self, *args):
        return self.xp.arange(*args)


class DirectionalCudaLayout:
    """Apply the existing linear CUDA layout kernels to primal and tangent."""
    def __init__(self, layout):
        self.base = layout

    def __getattr__(self, name):
        return getattr(self.base, name)

    def plane(self, data, fixed, *, axis, xp=None):
        return Directional(self.base.plane(data.value, fixed, axis=axis),
            None if data.tangent is None else self.base.plane(data.tangent, fixed, axis=axis))

    def spin_plane(self, data, fixed, *, xp=None):
        return Directional(self.base.spin_plane(data.value, fixed),
            None if data.tangent is None else self.base.spin_plane(data.tangent, fixed))

    def spin_sum(self, data):
        return Directional(self.base.spin_sum(data.value), None if data.tangent is None
                           else self.base.spin_sum(data.tangent))

    def feedback_sum(self, plane):
        return Directional(self.base.feedback_sum(plane.value),
            None if plane.tangent is None else self.base.feedback_sum(plane.tangent))

    def add_seed(self, target, seed, j, k):
        self.base.add_seed(target.value, seed.value, j, k)
        if seed.tangent is not None:
            self.base.add_seed(target.tangent, seed.tangent, j, k)

    def purify(self, triples, *, xp=None):
        self.base.purify(triples.value)
        if triples.tangent is not None:
            self.base.purify(triples.tangent)

    def subtract_crossed(self, seed, tile):
        self.base.subtract_crossed_fused(seed.value, tile.value)
        if tile.tangent is not None:
            self.base.subtract_crossed_fused(seed.tangent, tile.tangent)
