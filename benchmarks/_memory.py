"""Resident-memory tracing for the benchmark drivers (Linux).

`MemoryTrace` samples the resident set every millisecond on a thread and, at
each `mark(phase)`, reports the peak since the previous mark, the current
resident set, and glibc's split of its heap into blocks in use and free
blocks it keeps (with `malloc_info`, which every glibc has) besides the
blocks it maps on their own. Freed heap blocks stay resident until the
allocator trims them, so this split tells allocator retention from live
data. `wrap(owner, name, label)` marks before and after a method.

    trace = MemoryTrace(emit)            # emit(**row) prints a JSON record
    trace.wrap(SlotProgram, "run", "slot program")
    ...
    trace.mark("solve end")
"""
import ctypes
import os
import re
import threading
import time

GiB = 2.0**30


def _resident_bytes():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


class _Glibc:
    """Heap totals from glibc's malloc_info: (in use, free kept, mapped) bytes."""

    def __init__(self):
        libc = ctypes.CDLL("libc.so.6")
        libc.open_memstream.restype = ctypes.c_void_p
        libc.open_memstream.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t)]
        libc.malloc_info.argtypes = [ctypes.c_int, ctypes.c_void_p]
        libc.fclose.argtypes = [ctypes.c_void_p]
        libc.free.argtypes = [ctypes.c_void_p]
        self.libc = libc

    def split(self):
        libc = self.libc
        buffer, length = ctypes.c_char_p(), ctypes.c_size_t()
        stream = libc.open_memstream(ctypes.byref(buffer), ctypes.byref(length))
        libc.malloc_info(0, stream)
        libc.fclose(stream)
        text = ctypes.string_at(buffer, length.value).decode()
        libc.free(ctypes.cast(buffer, ctypes.c_void_p))
        tail = text[text.rfind("</heap>"):]

        def total(kind):
            found = re.search(rf'<total type="{kind}" count="\d+" size="(\d+)"', tail)
            return int(found.group(1)) if found else 0
        found = re.search(r'<system type="current" size="(\d+)"', tail)
        current = int(found.group(1)) if found else 0
        free = total("fast") + total("rest")
        return current - free, free, total("mmap")


class MemoryTrace:
    """Peak resident memory by phase (see the module docstring); a no-op
    off Linux."""

    def __init__(self, emit, interval=0.001):
        self.emit = emit
        self.enabled = os.path.exists("/proc/self/statm")
        self.peak = self.top = _resident_bytes() if self.enabled else 0
        self._glibc = None
        if self.enabled:
            try:
                self._glibc = _Glibc()
            except (OSError, AttributeError):
                self._glibc = None
            thread = threading.Thread(target=self._sample, args=(interval,), daemon=True)
            thread.start()

    def _sample(self, interval):
        while True:
            value = _resident_bytes()
            if value > self.peak:
                self.peak = value
            time.sleep(interval)

    def mark(self, phase):
        if not self.enabled:
            return None
        now = _resident_bytes()
        row = dict(event="memory", phase=phase, peak_GiB=round(self.peak / GiB, 2), now_GiB=round(now / GiB, 2))
        if self._glibc is not None:
            used, free, mapped = self._glibc.split()
            row.update(heap_used_GiB=round(used / GiB, 2), heap_free_GiB=round(free / GiB, 2),
                       mapped_GiB=round(mapped / GiB, 2))
        self.top = max(self.top, self.peak)
        self.peak = now
        self.emit(**row)
        return row

    def wrap(self, owner, name, label):
        """Mark before and after every call of owner.name."""
        original = getattr(owner, name)

        def wrapped(*args, **kwargs):
            self.mark(f"before {label}")
            try:
                return original(*args, **kwargs)
            finally:
                self.mark(label)
        setattr(owner, name, wrapped)

    @property
    def overall_peak_GiB(self):
        return round(max(self.top, self.peak) / GiB, 2)
