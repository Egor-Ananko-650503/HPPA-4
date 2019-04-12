#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

typedef short data_t;

__global__ void mirror_rotate_kernel(uint *src, uint *dst,
                                     size_t rows, size_t cols);

__host__ void launch_kernel(dim3 threads_per_block,
                            dim3 num_blocks,
                            uint* in,
                            uint* out,
                            size_t rows,
                            size_t cols);
