#!/bin/bash
# The comparison runs behind docs/benchmarks.md: water (O 1s frozen) on one
# 12-core machine, each program with its own default algorithms and 12 threads
# (NWChem: 12 MPI ranks). Slow program/basis pairs time a few iterations
# (--max-iter) instead of converging. Append to OUT, then plot with
#   python benchmarks/plot_compare.py OUT docs/images
#
# usage: bash benchmarks/compare_campaign.sh OUT.jsonl [threads]
# needs: itercc, pyscf, coupled-cluster-py (ccpy), psutil, and nwchem/mpirun on
# PATH with NWCHEM_BASIS_LIBRARY set.
set -u
OUT=$(realpath "$1")
THREADS=${2:-12}
HERE=$(cd "$(dirname "$0")" && pwd)
run() { python -P "$HERE/compare.py" run --out "$OUT" --threads "$THREADS" "$@"; }

# CCSDT ground state, converged everywhere (converging ccpy in aug-cc-pVTZ passes the
# 13 GiB memory limit; its time per iteration comes from this two-iteration run)
run --code ccpy --method ccsdt --basis aug-cc-pvtz --max-iter 2 --timeout 5400
for basis in 6-31g cc-pvdz aug-cc-pvdz cc-pvtz aug-cc-pvtz; do
    for code in itercc itercc-nosym pyscf-highm pyscf ccpy nwchem; do
        run --code $code --method ccsdt --basis $basis --timeout 5400
    done
done

# CCSDTQ ground state
for basis in 6-31g 6-31g*; do
    for code in itercc itercc-nosym pyscf-highm pyscf ccpy; do
        run --code $code --method ccsdtq --basis $basis --timeout 3600
    done
done
run --code itercc --method ccsdtq --basis cc-pvdz
run --code itercc-nosym --method ccsdtq --basis cc-pvdz
run --code pyscf-highm --method ccsdtq --basis cc-pvdz --timeout 3600
run --code pyscf --method ccsdtq --basis cc-pvdz --max-iter 3 --timeout 3600
run --code ccpy --method ccsdtq --basis cc-pvdz --max-iter 2 --timeout 5400
run --code itercc --method ccsdtq --basis aug-cc-pvdz
run --code itercc-nosym --method ccsdtq --basis aug-cc-pvdz --max-iter 3 --timeout 3600
run --code pyscf-highm --method ccsdtq --basis aug-cc-pvdz --max-iter 2 --timeout 3600

# EOM-CCSDT: the three lowest singlet roots of B1 (PySCF irrep 2; NWChem's axes call it b2)
for basis in cc-pvdz aug-cc-pvdz; do
    run --code itercc --method eom-ccsdt --basis $basis --roots 3 --irrep 2
    run --code ccpy --method eom-ccsdt --basis $basis --roots 3 --irrep 2 --timeout 5400
    run --code nwchem --method eom-ccsdt --basis $basis --roots 3 --irrep 2 --irrep-name b2 --timeout 5400
done
