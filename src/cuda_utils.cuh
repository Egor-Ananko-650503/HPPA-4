#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

typedef short data_t;

// data must be allocated on device
// _in_ _out_ data
// _in_ size_in_bytes
// _in_ seed
__host__ void random_init(data_t *data,
                          size_t size_in_bytes,
                          unsigned long long seed = 1234ULL);
