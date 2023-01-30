#pragma once

#define BLOCK_SIZE 32

namespace kernels {
namespace host {
auto dgemm(
    std::size_t n, std::size_t m, std::size_t k,
    double alpha,
    double* A, std::size_t lda,
    double* B, std::size_t ldb,
    double beta,
    double* C, std::size_t ldc
) -> void;
} // namespace host

namespace device {
__global__
auto dgemm(
    std::size_t n, std::size_t m, std::size_t k,
    double alpha,
    double* A, std::size_t lda,
    double* B, std::size_t ldb,
    double beta,
    double* C, std::size_t ldc
) -> void;

__global__
auto dgemm_shared(
    std::size_t n, std::size_t m, std::size_t k,
    double alpha,
    double* A, std::size_t lda,
    double* B, std::size_t ldb,
    double beta,
    double* C, std::size_t ldc
) -> void;
} // namespace
} // namespace kernels
