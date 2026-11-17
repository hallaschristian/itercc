"""Matched shared-sector expansion microbenchmark (not a CC solve).

Run with the same ITERCC_CPU_THREADS and CPU affinity for both paths.
Includes output allocation, zeroing and scatter; comparisons are untimed.
"""
import argparse
import json
from pathlib import Path
import resource
from statistics import median
from time import perf_counter

import numpy as np

from itercc._cpu_kernels import library
from itercc._cpu_triples import cpu_threads
from itercc._tensor_scatter import unpack_sector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=16_000_000)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if library() is None:
        raise RuntimeError("Benchmark requires the compiled core")
    indices = np.arange(0, args.size, args.stride, dtype=np.int64)
    values = np.random.default_rng(721).normal(size=len(indices))

    def reference():
        out = np.zeros(args.size)
        out[indices] = values
        return out

    functions = dict(numpy=reference, compiled=lambda: unpack_sector(values, indices, (args.size,)))
    expected = reference()
    for function in functions.values():
        np.testing.assert_array_equal(function(), expected)
    seconds = {name: [] for name in functions}
    for repeat in range(args.repeat):
        # Alternate order to reduce cache/frequency ordering bias.
        for name in (tuple(functions) if repeat % 2 else tuple(reversed(functions))):
            start = perf_counter()
            actual = functions[name]()
            seconds[name].append(perf_counter() - start)
            np.testing.assert_array_equal(actual, expected)
            del actual
    medians = {name: median(times) for name, times in seconds.items()}
    result = dict(size=args.size, nonzero=len(indices), threads=cpu_threads(None),
                  seconds=seconds, median_seconds=medians,
                  speedup=medians["numpy"] / medians["compiled"], max_abs_error=0,
                  process_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                  note="Combined process peak includes reference/comparison arrays; not solver peak memory")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
