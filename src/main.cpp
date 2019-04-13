#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

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

    dim3 threads_per_block{32, 8};
    dim3 num_blocks{(uint)((cols / 2 + 31) / 32),
                    (uint)((rows + (32 * 8 - 1)) / (32 * 8))};  // TODO: change for all dims

    std::cout << "Threads per block: "
              << threads_per_block.x << ' '
              << threads_per_block.y << ' '
              << threads_per_block.z << std::endl
              << "Blocks: "
              << num_blocks.x << ' '
              << num_blocks.y << ' '
              << num_blocks.z << std::endl;

    cudaEventRecord(e_start, 0);
    launch_kernel(threads_per_block, num_blocks, (uint *)d_data_in, (uint *)d_data_out, rows, cols);
    cudaEventRecord(e_stop, 0);

    cudaEventSynchronize(e_stop);
    float elapsedTimeGPU;
    cudaEventElapsedTime(&elapsedTimeGPU, e_start, e_stop);

    std::cout << "[GPU] Elapsed time: " << elapsedTimeGPU << " ms" << std::endl;

    cudaMemcpy(h_data_in, d_data_in, size_in_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data_out, d_data_out, size_in_bytes, cudaMemcpyDeviceToHost);

    // Testing
    {
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j += 2) {
                data_t tmp = std::move(h_data_in[i * cols + j]);
                h_data_in[i * cols + j] = std::move(h_data_in[i * cols + j + 1]);
                h_data_in[i * cols + j + 1] = std::move(tmp);
            }
        }

        // std::vector<std::pair<size_t, size_t>> v_er;
        size_t errors_count = 0;
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                if (h_data_in[i * cols + j] != h_data_out[i * cols + j]) {
                    errors_count++;
                    // v_er.push_back(std::move(std::make_pair(i, j)));
                }
            }
        }
        std::cout << "Errors: " << errors_count << std::endl;

        // for (size_t i = 0; i < 100; i++) {
        //     auto i_ = v_er.at(i).first;
        //     auto j_ = v_er.at(i).second;
        //     std::cout << i_ << ' ' << j_ << " - "
        //               << h_data_in[i_ * cols + j_] << ' '
        //               << h_data_out[i_ * cols + j_] << std::endl;
        // }

        // for (size_t i = 0; i < 0 + 32; i++) {
        //     for (size_t j = 0; j < 32; j++) {
        //         std::cout << std::setw(6) << h_data_in[i * cols + j] << ' ';
        //     }
        //     std::cout << std::endl;
        //     for (size_t j = 0; j < 32; j++) {
        //         std::cout << std::setw(6) << h_data_out[i * cols + j] << ' ';
        //     }
        //     std::cout << "\n\n";
        // }
    }

    auto c_start = std::chrono::steady_clock::now();
    auto cpu_data = cpu_transform(h_data_in, rows, cols);
    auto c_stop = std::chrono::steady_clock::now();
    auto elapsedTimeCPU = std::chrono::duration<float, std::milli>(c_stop - c_start).count();

    std::cout << "[CPU] Elapsed time: " << elapsedTimeCPU << " ms" << std::endl;

    // size_t errors_count = 0;
    // for (size_t i = 0; i < rows; i++) {
    //     for (size_t j = 0; j < cols; j++) {
    //         if (cpu_data[i * cols + j] != h_data_out[i * cols + j]) {
    //             errors_count++;
    //         }
    //     }
    // }

    // std::cout << "Errors: " << errors_count << std::endl;

    free(h_data_in);
    free(h_data_out);
    cudaFree(d_data_in);
    cudaFree(d_data_out);
    cudaEventDestroy(e_start);
    cudaEventDestroy(e_stop);
    cudaDeviceReset();
    return 0;
}
