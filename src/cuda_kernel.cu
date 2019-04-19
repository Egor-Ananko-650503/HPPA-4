#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_kernel.cuh"

__global__ void mirror_rotate_kernel(uint* src, uint* dst,
                                     size_t rows, size_t cols) {
    // Allocate shared memory for block
    __shared__ uint smem[32 * 32 * 8];
    // Shift to shared memory for current warp
    uint* smem_w = &smem[32 * 32 * threadIdx.y];

    // x, y for current warp on global memory (in) relatively blocks
    uint2 src_offset{blockIdx.x * 32,
                     blockIdx.y * 32 * 8 + threadIdx.y * 32};

    // size_t length = (rows - 1) * (cols / 2);
    size_t src_idx = src_offset.y * (cols / 2) + src_offset.x;

    // if (src_idx < length) {
    // Shift to global memory (in) for current warp
    uint* src_w = &src[src_idx];

    // Load to shared memory
    for (size_t i = 0; i < 32; i++) {
        smem_w[threadIdx.x + i * 32] = src_w[threadIdx.x + i * (cols / 2)];
    }

    __syncthreads();

    // "Unpack" array of uint to array of data_t
    data_t* smem_d = (data_t*)&smem[0];

    // x, y for current warp on shared memory
    // relatively 32x64 (cols x rows) blocks (for transactions)
    // dim3 -> uint2
    uint2 smem_d_offset{32 * (threadIdx.y % 2),
                        32 * 2 * (threadIdx.y / 2)};
    // Shift to shared memory for current warp
    data_t* smem_d_w = &smem_d[smem_d_offset.y * 32 * 2 + smem_d_offset.x];

    // x, y for current warp on global memory (out) relatively blocks
    uint2 dst_offset{(uint)rows / 2 - 1 - blockIdx.y * 32 * (8 / 2) - (threadIdx.y / 2) * (32 * 2 / 2),
                     (uint)cols - 1 - blockIdx.x * 32 * 2 - (threadIdx.y % 2) * 32};
    // Shift to global memory (out) for current warp
    uint* dst_w = &dst[dst_offset.y * (rows / 2) + dst_offset.x];

    for (size_t i = 0; i < 32; i++) {
        uint tmp;
        data_t* tmp_ptr = (data_t*)&tmp;
        tmp_ptr[0] = smem_d_w[(32 * 2 - 1 - threadIdx.x * 2) * 32 * 2 + i];
        tmp_ptr[1] = smem_d_w[(32 * 2 - 1 - (threadIdx.x * 2 + 1)) * 32 * 2 + i];
        *(dst_w - (32 - 1 - threadIdx.x) - i * (rows / 2)) = tmp;
    }
    // }
}

__host__ void launch_kernel(dim3 threads_per_block,
                            dim3 num_blocks,
                            uint* in,
                            uint* out,
                            size_t rows,
                            size_t cols) {
    mirror_rotate_kernel<<<num_blocks, threads_per_block>>>(in, out, rows, cols);
}
