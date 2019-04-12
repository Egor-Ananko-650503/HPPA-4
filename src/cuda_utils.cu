#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

#include "cuda_utils.cuh"

__host__ void random_init(data_t* data,
                          size_t size_in_bytes,
                          unsigned long long seed) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerate(gen, (unsigned int*)data,
                   size_in_bytes / sizeof(unsigned int));
    curandDestroyGenerator(gen);
}
