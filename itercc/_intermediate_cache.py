"""Byte-bounded tensor intermediates shared by CPU and device actions."""
from collections import OrderedDict
from operator import index


class IntermediateCache:
    """Action-local LRU cache of immutable NumPy/CuPy products.

    Create a fresh cache for each residual/Jacobian evaluation. Keys must
    identify the expression AND all operands, including tiles/spin/irreps;
    never reuse it across a change of amplitudes or direction. Values must
    own their allocation (a view's nbytes can undercount retained storage).
    The budget covers retained products, not live contraction workspace.
    """

    def __init__(self, budget):
        self.budget = index(budget)
        if self.budget < 0:
            raise ValueError("Intermediate cache budget must be nonnegative")
        self.entries = OrderedDict()
        self.bytes = self.peak_bytes = self.hits = self.misses = 0

    def get(self, key):
        value = self.entries.get(key)
        if value is None:
            self.misses += 1
        else:
            self.entries.move_to_end(key)
            self.hits += 1
        return value

    def release(self):
        """Drop every retained product (the counters stay)."""
        self.entries.clear()
        self.bytes = 0

    def put(self, key, value):
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.bytes -= previous.nbytes
        if value.nbytes > self.budget:
            return
        while self.bytes + value.nbytes > self.budget and self.entries:
            _, removed = self.entries.popitem(last=False)
            self.bytes -= removed.nbytes
        self.entries[key] = value
        self.bytes += value.nbytes
        self.peak_bytes = max(self.bytes, self.peak_bytes)
