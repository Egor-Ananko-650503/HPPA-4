#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_kernel.cuh"

__global__ void mirror_rotate_kernel(uint* src, uint* dst,
                                     size_t rows, size_t cols) {
    // Allocate shared memory for block
    __shared__ uint smem[32 * 32 * 8];
    // Shift to shared memory for current warp
    uint* smem_ = &smem[32 * 32 * threadIdx.y];

    // x, y for current thread on global memory relatively blocks
    dim3 src_offset{blockIdx.x * 32,
                    blockIdx.y * 32 * 8 + threadIdx.y * 32};

    size_t length = rows * (cols / 2);
    size_t src_idx = src_offset.y * (cols / 2) + src_offset.x;

    if (src_idx < length) {
        // Shift to global memoty for current thread
        uint* src_ = &src[src_offset.y * (cols / 2) + src_offset.x];

        // Load to shared memory
        for (size_t i = 0; i < 32; i++) {
            smem_[threadIdx.x + i * 32] = src_[threadIdx.x + i * (cols / 2)];
        }

        // __syncthreads();

        // Swap 2 shorts on uint [2B_1 2B_2] -> ]2B_2 2B_1]
        for (size_t i = 0; i < 32; i++) {
            data_t* pair = (data_t*)&smem_[threadIdx.x + i * 32];
            data_t tmp = pair[0];
            pair[0] = pair[1];
            pair[1] = tmp;
        }

        uint* dst_ = &dst[src_offset.y * (cols / 2) + src_offset.x];

        for (size_t i = 0; i < 32; i++) {
            dst_[threadIdx.x + i * (cols / 2)] = smem_[threadIdx.x + i * 32];
        }

        // __syncthreads();
    }
}

__host__ void launch_kernel(dim3 threads_per_block,
                            dim3 num_blocks,
                            uint* in,
                            uint* out,
                            size_t rows,
                            size_t cols) {
    mirror_rotate_kernel<<<num_blocks, threads_per_block>>>(in, out, rows, cols);
}
