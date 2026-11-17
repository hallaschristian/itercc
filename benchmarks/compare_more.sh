#!/bin/bash
# The second set of comparison runs behind docs/benchmarks.md, on water: CCSD
# (six programs), polarizabilities (CCSD and CCSDT linear response) and
# NWChem's CCSDTQ. Same
# machine and rules as compare_campaign.sh; slow program/basis pairs time a few
# iterations (--max-iter).
#
# usage: bash benchmarks/compare_more.sh OUT.jsonl [threads] [section ...]
# sections: ccsd response ccsdtq itercc (default: all but itercc, which
# reruns IterCC's own points). Needs NWCHEM_EXE for a NWChem built with CCSDTQ=y,
# CCSDTLR=y and NWCHEM_MODULES="tce property" (the response and ccsdtq sections;
# conda-forge's build has neither), PSI4_PYTHON, ORCA_EXE and
# ORCA_SCRATCH as described in compare.py.
set -u
OUT=$(realpath "$1")
THREADS=${2:-12}
shift $(( $# >= 2 ? 2 : $# ))
SECTIONS=${*:-"ccsd response ccsdtq"}
HERE=$(cd "$(dirname "$0")" && pwd)
run() { python -P "$HERE/compare.py" run --out "$OUT" --threads "$THREADS" "$@"; }
want() { [[ " $SECTIONS " == *" $1 "* ]]; }

if want ccsd; then
    for basis in 6-31g cc-pvdz aug-cc-pvdz cc-pvtz aug-cc-pvtz; do
        for code in pyscf psi4 nwchem orca ccpy; do
            run --code $code --method ccsd --basis $basis --timeout 3600
        done
    done
fi

if want response; then
    for basis in cc-pvdz aug-cc-pvdz cc-pvtz; do
        for code in psi4 nwchem; do
            run --code $code --method response-ccsd --basis $basis --timeout 5400
        done
    done
    for basis in cc-pvdz aug-cc-pvdz; do
        run --code nwchem --method response --basis $basis --timeout 7200
    done
fi

if want ccsdtq; then
    for basis in 6-31g 6-31g*; do
        run --code nwchem --method ccsdtq --basis $basis --timeout 7200
    done
    run --code nwchem --method ccsdtq --basis cc-pvdz --max-iter 3 --timeout 7200
fi

if want itercc; then
    for code in itercc itercc-nosym; do
        for basis in 6-31g cc-pvdz aug-cc-pvdz cc-pvtz aug-cc-pvtz; do
            run --code $code --method ccsd --basis $basis
        done
    done
    for basis in cc-pvdz aug-cc-pvdz cc-pvtz; do
        run --code itercc --method response-ccsd --basis $basis
    done
    for basis in cc-pvdz aug-cc-pvdz; do
        run --code itercc --method response --basis $basis
    done
fi
