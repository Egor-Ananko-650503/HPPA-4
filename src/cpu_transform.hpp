#pragma once

#include <memory>

typedef short data_t;

std::shared_ptr<data_t[]> cpu_transform(const data_t *in,
                                        const size_t &rows,
                                        const size_t &cols);
