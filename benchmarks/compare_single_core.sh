#!/bin/bash
# Single-core comparison, including ORCA (whose Windows build has no parallel
# AUTOCI): every program on one core (NWChem: one MPI rank). ORCA needs `orca`
# on PATH or ORCA_EXE; for a Windows ORCA driven from WSL also set
# ORCA_SCRATCH to a directory under /mnt/c.
#
# usage: bash benchmarks/compare_single_core.sh OUT.jsonl
set -u
OUT=$(realpath "$1")
HERE=$(cd "$(dirname "$0")" && pwd)
run() { python -P "$HERE/compare.py" run --out "$OUT" --threads 1 "$@"; }

for basis in 6-31g cc-pvdz aug-cc-pvdz cc-pvtz; do
    for code in itercc pyscf-highm pyscf ccpy nwchem orca; do
        run --code $code --method ccsdt --basis $basis --timeout 7200
    done
done
for code in itercc pyscf-highm; do
    run --code $code --method ccsdtq --basis 6-31g --timeout 3600
done
for code in pyscf ccpy orca; do
    run --code $code --method ccsdtq --basis 6-31g --max-iter 2 --timeout 5400
done
