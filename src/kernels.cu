#include "kernels.hpp"

namespace kernels {
namespace host {
auto dgemm(
    std::size_t m, std::size_t n, std::size_t k,
    double alpha,
    double* A, std::size_t lda,
    double* B, std::size_t ldb,
    double beta,
    double* C, std::size_t ldc
) -> void {
    #pragma omp parallel for
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double tmp = C[i * ldc + j] * beta;
            for (std::size_t l = 0; l < k; ++l) {
                tmp += A[i * lda + l] * B[l * ldb + j];
            }
            C[i * ldc + j] += alpha * tmp;
        }
    }
}
} // namespace host

namespace device {
__global__
auto dgemm(
    std::size_t m, std::size_t n, std::size_t k,
    double alpha,
    double* A, std::size_t lda,
    double* B, std::size_t ldb,
    double beta,
    double* C, std::size_t ldc
) -> void {
    std::size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    double tmp = C[i * ldc + j] * beta;
    for (std::size_t l = 0; l < k; ++l) {
        tmp += A[i * lda + l] * B[l * ldb + j];
    }
    C[i * ldc + j] += alpha * tmp;
}

__global__
auto dgemm_shared(
    std::size_t m, std::size_t n, std::size_t k,
    double alpha,
    double* A, std::size_t lda,
    double* B, std::size_t ldb,
    double beta,
    double* C, std::size_t ldc
) -> void {
    size_t blockRow = blockIdx.y;
    size_t blockCol = blockIdx.x;
    double* Csub = &C[blockRow * ldc * BLOCK_SIZE + blockCol * BLOCK_SIZE];
    double tmp = *Csub * beta;
    size_t row = threadIdx.y;
    size_t col = threadIdx.x;
    for (size_t l = 0; l < (k / BLOCK_SIZE); ++l) {
        double* Asub = &A[blockRow * lda * BLOCK_SIZE + l * BLOCK_SIZE];
        double* Bsub = &B[l * ldb * BLOCK_SIZE + blockCol * BLOCK_SIZE];

        // Shared memory used to store Asub and Bsub respectively
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        As[row][col] = Asub[row * lda + col];
        Bs[row][col] = Bsub[row * ldb + col];

        // Synchronize to make sure the sub-matrices are loaded before starting the computation
        __syncthreads();
        for (size_t o = 0; o < BLOCK_SIZE; ++o) {
            tmp += As[row][o] * Bs[o][col];
        }
        // Synchronize to make sure that the preceding computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    Csub[row * ldc + col] += alpha * tmp;
}
} // namespace
} // namespace kernels
