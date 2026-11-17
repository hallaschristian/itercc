// Per-thread scratch buffers of the CPU kernels. A buffer keeps its pages
// between calls (no zero fill, and no page faults after its first use) until
// itercc_scratch_release, which the solvers call between phases.
#ifndef ITERCC_SCRATCH_H
#define ITERCC_SCRATCH_H

#include <cstddef>
#include <memory>

namespace itercc {

struct Scratch {
    std::unique_ptr<double[]> data;
    size_t size = 0;

    double *get(size_t n)
    {
        if (n > size) {
            data.reset();
            data.reset(new double[n]);
            size = n;
        }
        return data.get();
    }
    void release()
    {
        data.reset();
        size = 0;
    }
};

constexpr int kScratchSlots = 7;   // 0-3 contraction routes, 4-5 tile programs, 6 packed B panels

// Slot `which` (0 .. kScratchSlots - 1) of the calling thread.
Scratch &scratch(int which);

}  // namespace itercc

#endif
