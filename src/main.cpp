#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <memory>

#include "cpu_transform.hpp"
#include "cuda_kernel.cuh"
#include "cuda_utils.cuh"

typedef short data_t;

int main(int argc, char const *argv[]) {
    data_t *d_data_in;
    data_t *h_data_in;
    data_t *d_data_out;
    data_t *h_data_out;
    size_t rows = 1024;
    size_t cols = 1024;

    {
        size_t rows_tmp;
        size_t cols_tmp;
        if (argc > 2) {
            try {
                rows_tmp = std::stoul(argv[1]);
                cols_tmp = std::stoul(argv[2]);
                rows = rows_tmp;
                cols = cols_tmp;
            } catch (const std::invalid_argument &e) {
                std::cerr << e.what() << ": invalid argument" << '\n';
            } catch (const std::out_of_range &e) {
                std::cerr << e.what() << ": out of range" << '\n';
            }
        }
    }

    std::cout << "Rows: " << rows << std::endl
              << "Cols: " << cols << std::endl;

    size_t len = rows * cols;
    size_t size_in_bytes = len * sizeof(data_t);

    std::cout << "len: " << len << std::endl;
    std::cout << "sib: " << size_in_bytes << std::endl;

    h_data_in = (data_t *)calloc(len, sizeof(data_t));
    h_data_out = (data_t *)calloc(len, sizeof(data_t));
    cudaMalloc(&d_data_in, size_in_bytes);
    random_init(d_data_in, size_in_bytes);
    cudaMalloc(&d_data_out, size_in_bytes);

    cudaEvent_t e_start;
    cudaEvent_t e_stop;
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);

    dim3 threads_per_block{32 * 8};
    dim3 num_blocks{(uint)(cols / 2 / 32),
                    (uint)(rows / 2 / 32 / 8)};  // TODO: change for all dims

    cudaEventRecord(e_start, 0);
    launch_kernel(threads_per_block, num_blocks, (uint *)d_data_in, (uint *)d_data_out, rows, cols);
    cudaEventRecord(e_stop, 0);

    cudaEventSynchronize(e_stop);
    float elapsedTimeGPU;
    cudaEventElapsedTime(&elapsedTimeGPU, e_start, e_stop);

    std::cout << "[GPU] Elapsed time: " << elapsedTimeGPU << " ms" << std::endl;

    cudaMemcpy(h_data_in, d_data_in, size_in_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data_out, d_data_out, size_in_bytes, cudaMemcpyDeviceToHost);

    auto c_start = std::chrono::steady_clock::now();
    auto cpu_data = cpu_transform(h_data_in, rows, cols);
    auto c_stop = std::chrono::steady_clock::now();
    auto elapsedTimeCPU = std::chrono::duration<float, std::milli>(c_stop - c_start).count();

    std::cout << "[CPU] Elapsed time: " << elapsedTimeCPU << " ms" << std::endl;

    /* for (size_t i = 0; i < len; i++) {
        std::cout << cpu_data[i] << std::endl;
    } */

    free(h_data_in);
    free(h_data_out);
    cudaFree(d_data_in);
    cudaFree(d_data_out);
    cudaEventDestroy(e_start);
    cudaEventDestroy(e_stop);
    cudaDeviceReset();
    return 0;
}
