#include "utils.hpp"

#include <cstdlib>

namespace utils {
auto init_matrices(double* A, double* B, double* C, std::size_t size) -> void {
    srand(2023);
    for (std::size_t i = 0; i < size; i++) {
        for (std::size_t j = 0; j < size; j++) {
            A[i * size + j] = drand48();
            B[i * size + j] = drand48();
            C[i * size + j] = drand48();
        }
    }
}

auto elapsed_seconds(
    struct timespec const& start,
    struct timespec const& stop
) -> double {
    return (double)(stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) * 1.0E-9;
}
} // namespace utils
