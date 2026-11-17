#!/bin/bash
# IterCC's thread scaling on one workstation: seconds per
# iteration of CCSD, CCSDT and CCSDTQ on water with 1 to 12 threads, a few
# iterations each. Plot with
#   python benchmarks/plot_compare.py RESULTS docs/images --threads OUT.jsonl
#
# usage: bash benchmarks/thread_scaling.sh OUT.jsonl
set -u
OUT=$(realpath "$1")
HERE=$(cd "$(dirname "$0")" && pwd)
for threads in 1 2 4 6 8 12; do
    run() { python -P "$HERE/compare.py" run --out "$OUT" --threads "$threads" --code itercc "$@"; }
    run --method ccsd --basis aug-cc-pvtz --max-iter 6
    run --method ccsdt --basis aug-cc-pvtz --max-iter 4
    run --method ccsdtq --basis aug-cc-pvdz --max-iter 3
done
