#include "itercc/core.h"

#ifndef ITERCC_BUILD_TARGET
#define ITERCC_BUILD_TARGET "unspecified"
#endif

#define ITERCC_STRING2(x) #x
#define ITERCC_STRING(x) ITERCC_STRING2(x)

// Non-default builds (another GEMM sum chunk: other rounding; profile builds:
// counters and knockouts) say so in the build string.
#ifdef ITERCC_GEMM_KC
#define ITERCC_OPTION_KC " kc=" ITERCC_STRING(ITERCC_GEMM_KC)
#else
#define ITERCC_OPTION_KC ""
#endif
#ifdef ITERCC_PROFILE_SYM
#define ITERCC_OPTION_PROFILE " profile"
#else
#define ITERCC_OPTION_PROFILE ""
#endif
#define ITERCC_OPTIONS ITERCC_OPTION_KC ITERCC_OPTION_PROFILE

// The instruction-set extensions the compiler was allowed to use, as
// /proc/cpuinfo flag names: the loader refuses a `native` build on a CPU
// that lacks one of them (a checkout shared by different node types).
#ifdef __AVX__
#define ITERCC_ISA_AVX ",avx"
#else
#define ITERCC_ISA_AVX ""
#endif
#ifdef __AVX2__
#define ITERCC_ISA_AVX2 ",avx2"
#else
#define ITERCC_ISA_AVX2 ""
#endif
#ifdef __FMA__
#define ITERCC_ISA_FMA ",fma"
#else
#define ITERCC_ISA_FMA ""
#endif
#ifdef __BMI2__
#define ITERCC_ISA_BMI2 ",bmi2"
#else
#define ITERCC_ISA_BMI2 ""
#endif
#ifdef __AVX512F__
#define ITERCC_ISA_512F ",avx512f"
#else
#define ITERCC_ISA_512F ""
#endif
#ifdef __AVX512BW__
#define ITERCC_ISA_512BW ",avx512bw"
#else
#define ITERCC_ISA_512BW ""
#endif
#ifdef __AVX512DQ__
#define ITERCC_ISA_512DQ ",avx512dq"
#else
#define ITERCC_ISA_512DQ ""
#endif
#ifdef __AVX512VL__
#define ITERCC_ISA_512VL ",avx512vl"
#else
#define ITERCC_ISA_512VL ""
#endif
#ifdef __AVX512FP16__
#define ITERCC_ISA_512FP16 ",avx512_fp16"
#else
#define ITERCC_ISA_512FP16 ""
#endif
#define ITERCC_ISA " isa=sse2" ITERCC_ISA_AVX ITERCC_ISA_AVX2 ITERCC_ISA_FMA ITERCC_ISA_BMI2 \
    ITERCC_ISA_512F ITERCC_ISA_512BW ITERCC_ISA_512DQ ITERCC_ISA_512VL ITERCC_ISA_512FP16

extern "C" int itercc_core_abi(void) { return ITERCC_CORE_ABI; }

extern "C" const char *itercc_core_build(void)
{
#if defined(__clang__)
    return "clang " __clang_version__ " target=" ITERCC_BUILD_TARGET ITERCC_OPTIONS ITERCC_ISA;
#elif defined(__GNUC__)
    return "gcc " ITERCC_STRING(__GNUC__) "." ITERCC_STRING(__GNUC_MINOR__) "."
           ITERCC_STRING(__GNUC_PATCHLEVEL__) " target=" ITERCC_BUILD_TARGET ITERCC_OPTIONS ITERCC_ISA;
#else
    return "unknown compiler target=" ITERCC_BUILD_TARGET ITERCC_OPTIONS ITERCC_ISA;
#endif
}
