#pragma once

#include <cstdint>
#include <ctime>

namespace utils {
auto init_matrices(double* A, double* B, double* C, std::size_t size) -> void;

auto elapsed_seconds(struct timespec const& start,
                     struct timespec const& stop)-> double;
}
