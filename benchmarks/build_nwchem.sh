#!/bin/bash
# Build NWChem 7.3.1 with what the comparisons need and the conda-forge package
# lacks: TCE CCSDTQ (CCSDTQ=y), TCE CCSDT linear response (CCSDTLR=y, with the
# property module for the dipole integrals), plus the CCSD module.
#
# usage: bash benchmarks/build_nwchem.sh PREFIX SOURCE [JOBS]
#   PREFIX  conda environment for the toolchain (created if missing; needs
#           conda or mamba on PATH)
#   SOURCE  checkout of NWChem (cloned at v7.3.1-release if missing)
# The binary is SOURCE/bin/LINUX64/nwchem; point compare.py at it with
# NWCHEM_EXE, and set NWCHEM_BASIS_LIBRARY=SOURCE/src/basis/libraries/.
set -euo pipefail
PREFIX=$1
SOURCE=$2
JOBS=${3:-$(nproc)}
CONDA=$(command -v mamba || command -v conda)
if [ ! -x "$PREFIX/bin/mpifort" ]; then
    "$CONDA" create -y -q -p "$PREFIX" -c conda-forge --override-channels \
        fortran-compiler c-compiler cxx-compiler openmpi openmpi-mpifort openblas make perl python=3.12
fi
eval "$(conda shell.bash hook)"
conda activate "$PREFIX"
if [ ! -d "$SOURCE/src" ]; then
    git clone -q --depth 1 --branch v7.3.1-release https://github.com/nwchemgit/nwchem.git "$SOURCE"
fi
export NWCHEM_TOP=$SOURCE NWCHEM_TARGET=LINUX64 USE_MPI=y ARMCI_NETWORK=MPI-PR
export NWCHEM_MODULES="tce ccsd property" CCSDTQ=y CCSDTLR=y
export BLAS_SIZE=4 USE_64TO32=y BLASOPT="-L$PREFIX/lib -lopenblas" LAPACK_LIB="-L$PREFIX/lib -lopenblas"
export USE_OPENMP=n USE_SERIALEIGENSOLVERS=y FC=gfortran CC=gcc
ln -sf "$(command -v x86_64-conda-linux-gnu-gfortran)" "$PREFIX/bin/gfortran"
ln -sf "$(command -v x86_64-conda-linux-gnu-gcc)" "$PREFIX/bin/gcc"
cd "$NWCHEM_TOP/src"
make nwchem_config > "$SOURCE/build-config.log" 2>&1
make 64_to_32 > "$SOURCE/build-64to32.log" 2>&1
rm -f "$NWCHEM_TOP/bin/LINUX64/nwchem"
make -j "$JOBS" > "$SOURCE/build-make.log" 2>&1 || true
test -x "$NWCHEM_TOP/bin/LINUX64/nwchem" || { tail -30 "$SOURCE/build-make.log"; exit 1; }
echo "built $NWCHEM_TOP/bin/LINUX64/nwchem"
