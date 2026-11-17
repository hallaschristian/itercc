"""Atomic array archives, with opt-in lossless compression for large saved states."""

from pathlib import Path
import os
import tempfile
import time
import numpy as np


def _shared_mode(path):
    """Give a mkstemp file (mode 0600) the mode open() would: 0666 less the
    umask, so group members can read checkpoints on a shared drive."""
    mask = None
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("Umask:"):
                mask = int(line.split()[1], 8)
                break
    except (OSError, ValueError, IndexError):
        pass
    if mask is None:        # os.umask can only be read by setting it
        mask = os.umask(0o022)
        os.umask(mask)
    try:
        os.chmod(path, 0o666 & ~mask)
    except OSError:
        pass


def write_array_archive(path, arrays, *, compressed=False):
    """Replace an archive only after its complete contents reach the filesystem.

    Compression is lossless and deliberately optional: frequent iterative
    checkpoints usually favor speed; sparse final CI vectors favor disk space.
    Caller-owned writer locks are still required to prevent competing results.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    _shared_mode(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            saver = np.savez_compressed if compressed else np.savez
            saver(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if hasattr(os, "O_DIRECTORY"):
            directory = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _fsync_directory(path):
    if hasattr(os, "O_DIRECTORY"):
        descriptor = os.open(path, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _staging(count, dtype, device):
    """A reusable host buffer; page-locked when it exchanges data with a GPU."""
    if device:
        import cupyx

        return cupyx.empty_pinned((count,), dtype=dtype)
    return np.empty(count, dtype=dtype)


def _write_npy(target, array, chunk_bytes):
    """Write a NumPy or CuPy array as .npy with sequential buffered writes."""
    header = {"descr": np.lib.format.dtype_to_descr(array.dtype), "fortran_order": False,
              "shape": tuple(array.shape)}
    device = hasattr(array, "get")
    flat = array.reshape(-1) if array.flags.c_contiguous else array.flat
    step = max(1, chunk_bytes // array.itemsize)
    with open(target, "wb") as stream:
        try:
            np.lib.format.write_array_header_1_0(stream, header)
        except ValueError:
            np.lib.format.write_array_header_2_0(stream, header)
        buffer = _staging(min(step, array.size), array.dtype, True) if device else None
        for first in range(0, array.size, step):
            piece = flat[first:first + step]
            if device:
                host = buffer[:piece.size]
                piece.get(out=host)
            else:
                host = np.ascontiguousarray(piece)
            stream.write(memoryview(host).cast("B"))
        stream.flush()
        os.fsync(stream.fileno())


def write_array(path, array, *, chunk_bytes=128 * 2**20):
    """Atomically save one NumPy or CuPy array as a streamed .npy file."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                             dir=path.parent)
    os.close(descriptor)
    _shared_mode(temporary)
    try:
        _write_npy(temporary, array, chunk_bytes)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_array_directory(path, arrays, metadata, *, chunk_bytes=128 * 2**20):
    """Atomically replace a directory of .npy arrays plus metadata.json.

    Arrays may be NumPy or CuPy. Device arrays go through one page-locked
    buffer of `chunk_bytes` with sequential writes, so no host copy of a whole
    array is made and file systems see large streaming writes. The replaced
    directory is renamed to a unique `.NAME.previous-*` name before it is
    removed, so files that another process still holds open (which network
    file systems keep as hidden files) never block a later write;
    `read_array_directory` falls back to the newest one after an interrupted
    swap, and later successful writes remove leftovers. The caller must
    serialize writers for the same path.
    """
    import json
    import shutil

    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{path.name}.tmp-{os.getpid()}"
    for stale in parent.glob(f".{path.name}.tmp-*"):
        shutil.rmtree(stale, ignore_errors=True)
    temporary.mkdir()
    for name, array in arrays.items():
        _write_npy(temporary / f"{name}.npy", array, chunk_bytes)
    with (temporary / "metadata.json").open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(temporary)
    if path.exists():
        previous = parent / f".{path.name}.previous-{os.getpid()}-{time.time_ns()}"
        os.replace(path, previous)
    os.replace(temporary, path)
    _fsync_directory(parent)
    for stale in parent.glob(f".{path.name}.previous*"):
        if stale.is_symlink():          # a linked state (e.g. on scratch): drop the link only
            stale.unlink(missing_ok=True)
        else:
            shutil.rmtree(stale, ignore_errors=True)


def _host_copy(array):
    """A host copy of a NumPy array; float64 arrays are copied on all threads
    when the compiled CPU core is present (a packed T4 is tens of GB, and the
    copy holds up the solve)."""
    array = np.asarray(array)
    if array.dtype == np.float64 and array.flags.c_contiguous and array.size > 2**20:
        from . import _cpu_quadruples

        if _cpu_quadruples.available():
            return _cpu_quadruples.copy(array)
    return np.array(array)


class BackgroundArrayWriter:
    """Write array directories from a thread so computation can continue.

    `submit` copies every array to host memory (a device-to-host transfer,
    seconds even for tens of GB) and returns; a thread then writes the copy
    with `write_array_directory`. One write is in flight at a time: `submit`
    and `wait` first finish the previous one and re-raise its error.
    `last_seconds` is the duration of the last completed write.
    """

    def __init__(self):
        self._thread = None
        self._error = None
        self.last_seconds = None

    def submit(self, path, arrays, metadata, **options):
        import threading
        from time import perf_counter

        self.wait()
        host = {name: array.get() if hasattr(array, "get") else _host_copy(array)
                for name, array in arrays.items()}

        def run():
            begin = perf_counter()
            try:
                write_array_directory(path, host, metadata, **options)
            except BaseException as error:      # re-raised by wait()
                self._error = error
            self.last_seconds = perf_counter() - begin

        self._thread = threading.Thread(target=run, name="itercc-array-writer")
        self._thread.start()

    @property
    def busy(self):
        return self._thread is not None and self._thread.is_alive()

    def wait(self):
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            error, self._error = self._error, None
            raise error


class SavedArray:
    """A .npy file that is read on demand and holds no open file.

    `read_array_directory(..., lazy=True)` returns these, so a process can
    replace the directory it resumed from (an open memory map would pin the
    old files on network file systems). `to_device` streams them to a GPU;
    `np.asarray` loads them into host memory.
    """

    def __init__(self, filename):
        self.filename = str(filename)
        with open(self.filename, "rb") as stream:
            version = np.lib.format.read_magic(stream)
            read = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                    else np.lib.format.read_array_header_2_0)
            shape, fortran, dtype = read(stream)
            self.offset = stream.tell()
        if fortran:
            raise ValueError(f"{filename} is stored in Fortran order")
        self.shape, self.dtype = tuple(shape), np.dtype(dtype)
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape, dtype=np.int64))
        self.itemsize = self.dtype.itemsize
        self.nbytes = self.size * self.itemsize

    def __array__(self, dtype=None, copy=None):
        value = np.load(self.filename, allow_pickle=False)
        return value if dtype is None else value.astype(dtype, copy=False)

    def __len__(self):
        return self.shape[0]

    def read(self, start, stop):
        """Elements [start, stop) of the flattened array, read from the file."""
        start, stop = max(0, int(start)), min(self.size, int(stop))
        count = max(0, stop - start)
        with open(self.filename, "rb") as stream:
            stream.seek(self.offset + start * self.itemsize)
            data = np.fromfile(stream, dtype=self.dtype, count=count)
        if data.size != count:
            raise EOFError(f"{self.filename} ended early")
        return data


def to_device(source, out=None, *, chunk_bytes=128 * 2**20):
    """Copy an array into a same-shaped CuPy array in bounded chunks.

    A SavedArray streams its file into a page-locked buffer. NumPy memmaps
    are read through their mapping so an atomic file replacement cannot
    silently change the source generation.
    """
    import cupy as cp

    if out is not None and out.shape != source.shape:
        raise ValueError("Source and destination shapes differ")
    if isinstance(source, cp.ndarray):
        if out is None:
            return source.copy()
        out[...] = source
        return out
    out = cp.empty(source.shape, dtype=source.dtype) if out is None else out
    flat_out = out.reshape(-1) if out.flags.c_contiguous else None
    count = out.size
    step = max(1, chunk_bytes // out.itemsize)
    if count == 0:
        return out
    buffer = _staging(min(step, count), out.dtype, True)

    def upload(first, host):
        part = slice(first, first + host.size)
        if flat_out is None:
            out.flat[part] = cp.asarray(host)
        else:
            flat_out[part].set(host)
        # Do not refill page-locked memory while an upload is still in flight.
        cp.cuda.get_current_stream().synchronize()

    if isinstance(source, SavedArray) and source.dtype == out.dtype:
        with open(source.filename, "rb") as stream:
            stream.seek(source.offset)
            for first in range(0, count, step):
                host = buffer[:min(step, count - first)]
                view = memoryview(host).cast("B")
                filled = 0
                while filled < len(view):
                    read = stream.readinto(view[filled:])
                    if not read:
                        raise EOFError(f"{source.filename} ended early")
                    filled += read
                upload(first, host)
        return out
    if isinstance(source, SavedArray):
        read_part = source.read
    else:
        array = np.asarray(source)
        flat = array.reshape(-1) if array.flags.c_contiguous else array.flat
        read_part = lambda first, stop: flat[first:stop]
    for first in range(0, count, step):
        host = buffer[:min(step, count - first)]
        host[...] = read_part(first, first + host.size)
        upload(first, host)
    return out


def read_array_directory(path, *, lazy=False):
    """Return ({name: array}, metadata) from write_array_directory.

    Arrays are read-only memory maps, or `SavedArray` objects that hold no
    open file when `lazy` is true.
    """
    import json

    path = Path(path)
    if not (path / "metadata.json").exists():
        candidates = sorted((item for item in path.parent.glob(f".{path.name}.previous*")
                             if (item / "metadata.json").exists()),
                            key=lambda item: (item / "metadata.json").stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No complete array directory at {path}")
        path = candidates[-1]
    metadata = json.loads((path / "metadata.json").read_text())
    load = SavedArray if lazy else (lambda item: np.load(item, mmap_mode="r"))
    arrays = {item.stem: load(item) for item in sorted(path.glob("*.npy"))}
    return arrays, metadata
