#!/bin/bash
# Build portable IterCC cores and install them next to the Python package:
#   * libitercc_core.x86-64-v3.so (AVX2)
#   * libitercc_core.x86-64-v4.so (AVX-512)
#   * libitercc_cuda.so with ITERCC_CUDA=ON (A100 sm_80, H100/H200 sm_90, plus PTX)
# The loader (itercc/_core.py) picks the best level each machine supports, so
# one installation serves a cluster with mixed node types.
#
#   bash core/build_levels.sh [INSTALL_DIR]       # default: <source>/itercc
#   CXX=g++-12 ITERCC_CUDA=ON BUILD_JOBS=16 bash core/build_levels.sh
#
# Build with the compiler whose libstdc++ and OpenMP runtime the compute
# nodes load at run time (often the system GCC). Compilers that do not know
# the x86-64-v3/v4 names (GCC before 11) get -march=haswell and
# -march=skylake-avx512 instead.
set -euo pipefail
source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
install=${1:-$source/itercc}
cuda=${ITERCC_CUDA:-OFF}
build=$(mktemp -d)
trap 'rm -rf -- "$build"' EXIT
configure() {
    local name=$1 march=$2 with_cuda=$3
    cmake -S "$source/core" -B "$build/$name" -DITERCC_TARGET="$name" -DITERCC_MARCH="$march" \
        -DITERCC_CUDA="$with_cuda" ${CXX:+-DCMAKE_CUDA_HOST_COMPILER="$CXX"} >/dev/null
    cmake --build "$build/$name" -j "${BUILD_JOBS:-8}"
    cmake --install "$build/$name" --prefix "$install" >/dev/null
}
if ! echo 'int main(){}' | "${CXX:-c++}" -march=x86-64-v3 -x c++ -o /dev/null - 2>/dev/null; then
    configure x86-64-v3 haswell OFF
    configure x86-64-v4 skylake-avx512 "$cuda"
else
    configure x86-64-v3 x86-64-v3 OFF
    configure x86-64-v4 x86-64-v4 "$cuda"
fi
ls -l "$install"/libitercc_*.so
