#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <memory>

#include "cpu_transform.hpp"
#include "cuda_kernel.cuh"
#include "cuda_utils.cuh"

typedef short data_t;

int main(int argc, char const *argv[]) {
    data_t *d_data;
    data_t *h_data;
    size_t rows = 10;
    size_t cols = 10;
    size_t len = rows * cols;
    size_t size_in_bytes = len * sizeof(data_t);

    h_data = (data_t *)calloc(len, sizeof(data_t));
    cudaMalloc(&d_data, size_in_bytes);
    random_init(d_data, size_in_bytes);

    // cudaEvent_t start;
    // cudaEvent_t stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudaEventRecord(start, 0);

    // cudaEventRecord(stop, 0);

    // cudaEventSynchronize(stop);
    // float elapsedTime;
    // cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(h_data, d_data, size_in_bytes, cudaMemcpyDeviceToHost);
    /* for (size_t i = 0; i < len; i++) {
        std::cout << h_data[i] << std::endl;
    } */

    free(h_data);
    cudaFree(d_data);
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    return 0;
}
