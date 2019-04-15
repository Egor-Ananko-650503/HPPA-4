#include "cpu_transform.hpp"
#include <omp.h>

std::shared_ptr<data_t[]> cpu_transform(const data_t *in,
                                        const size_t &rows,
                                        const size_t &cols) {
    auto out = std::shared_ptr<data_t[]>(new data_t[rows * cols]);
#pragma omp parallel for
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            out[(cols - 1 - j) * rows + (rows - 1 - i)] = in[i * cols + j];
        }
    }

    return std::move(out);
}
