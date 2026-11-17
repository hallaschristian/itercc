# IterCC compiled core.
#
#   cmake -S core -B core/build [-DITERCC_TARGET=x86-64-v4] [-DITERCC_CUDA=ON]
#   cmake --build core/build -j
#   cmake --install core/build --prefix itercc      # copies the libraries next to the package
#
# libitercc_core (CPU, OpenMP) is named libitercc_core.<target>.so, so one
# installation can hold builds for several instruction-set levels; the Python
# loader (itercc/_core.py) picks the best one the running CPU supports.
# `native` builds are for the build machine only. libitercc_cuda is separate
# so CPU-only nodes never load CUDA libraries.
cmake_minimum_required(VERSION 3.18)
project(itercc_core LANGUAGES CXX)

set(ITERCC_TARGET "native" CACHE STRING
    "Library target name: native, x86-64-v2, x86-64-v3 (AVX2) or x86-64-v4 (AVX-512)")
set(ITERCC_MARCH "${ITERCC_TARGET}" CACHE STRING
    "-march value; compilers before GCC 11 need haswell for x86-64-v3, skylake-avx512 for v4")
option(ITERCC_CUDA "Build libitercc_cuda" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

add_library(itercc_core SHARED src/version.cpp src/cpu/triples.cpp
            src/cpu/blocked.cpp src/cpu/blocked_residual.cpp
            src/cpu/symmetry_blocks.cpp src/cpu/quadruples.cpp src/cpu/gemm_kernel.cpp
            src/cpu/tile_program.cpp src/cpu/array_ops.cpp src/cpu/tensor_scatter.cpp src/cpu/spin_quadruples.cpp
            src/cpu/integrals.cpp src/cpu/gto_integrals.cpp)
target_include_directories(itercc_core PUBLIC include)
# No FMA contraction: results do not depend on the target instruction set.
target_compile_options(itercc_core PRIVATE -O3 -march=${ITERCC_MARCH} -ffp-contract=off
                       -fno-math-errno)
target_compile_definitions(itercc_core PRIVATE ITERCC_BUILD_TARGET="${ITERCC_TARGET}")
target_link_libraries(itercc_core PRIVATE OpenMP::OpenMP_CXX)
set_target_properties(itercc_core PROPERTIES OUTPUT_NAME "itercc_core.${ITERCC_TARGET}"
                      PREFIX "lib" SUFFIX ".so")
install(TARGETS itercc_core LIBRARY DESTINATION .)

if(ITERCC_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    # A100, H100/H200, and PTX for newer GPUs.
    set(CMAKE_CUDA_ARCHITECTURES "80-real;90-real;90-virtual")
  endif()
  set(CMAKE_CUDA_STANDARD 17)
  add_library(itercc_cuda SHARED src/cuda/device.cu src/cuda/blocked.cu src/cuda/blas.cu)
  target_include_directories(itercc_cuda PUBLIC include)
  target_compile_options(itercc_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --fmad=false>)
  target_link_libraries(itercc_cuda PRIVATE CUDA::cudart_static CUDA::cublas)
  # Find cuBLAS where it was found at build time (e.g. the cluster CUDA module).
  set_target_properties(itercc_cuda PROPERTIES PREFIX "lib" SUFFIX ".so"
                        INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"
                        BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}")
  install(TARGETS itercc_cuda LIBRARY DESTINATION .)
endif()
