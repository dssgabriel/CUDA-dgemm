#include "drivers.hpp"
#include "kernels.hpp"
#include "utils.hpp"

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <cuda.h>

namespace drivers {
namespace host {
auto dgemm(
    std::size_t m, std::size_t n, std::size_t k,
    double alpha,
    double const* A,
    double const* B,
    double beta,
    double const* C
) -> int32_t {
    int32_t status_code = 0;
    double duration = 0.0;

    double* h_A = (double*)(malloc(m * k * sizeof(double)));
    if (!h_A) {
        fprintf(stderr, "error: failed to allocate `h_A`\n");
        status_code = -1;
    }

    double* h_B = (double*)(malloc(k * n * sizeof(double)));
    if (!h_B) {
        fprintf(stderr, "error: failed to allocate `h_B`\n");
        status_code = -1;
    }

    double* h_C = (double*)(malloc(m * n * sizeof(double)));
    if (!h_C) {
        fprintf(stderr, "error: failed to allocate `h_C`\n");
        status_code = -1;
    }

    if (status_code == -1) { goto failure; }

    memcpy(h_A, A, m * k * sizeof(double));
    if (!h_A) {
        fprintf(stderr, "error: failed to copy memory from `A` to `h_A`\n");
        status_code = -1;
        goto failure;
    }

    memcpy(h_B, B, k * n * sizeof(double));
    if (!h_B) {
        fprintf(stderr, "error: failed to copy memory from `B` to `h_B`\n");
        status_code = -1;
        goto failure;
    }

    memcpy(h_C, C, m * n * sizeof(double));
    if (!h_C) {
        fprintf(stderr, "error: failed to copy memory from `C` to `h_C`\n");
        status_code = -1;
        goto failure;
    }

    struct timespec start, stop;
    // Invoke kernel
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    kernels::host::dgemm(m, n, k, alpha, h_A, k, h_B, n, beta, h_C, n);
    clock_gettime(CLOCK_MONOTONIC_RAW, &stop);

    duration = utils::elapsed_seconds(start, stop);
    #if defined(DEBUG)
    printf("h_C(0, 0) = %lf\n", h_C[0]);
    #endif
    printf(
        "  Host time: %.6lf ms, GFlop/s: %.3lf\n",
        duration * 1.0E-3,
        m * n * k * 1.0E-9 / duration
    );

failure:
    if (h_A) { free(h_A); }
    if (h_B) { free(h_B); }
    if (h_C) { free(h_C); }

    return status_code;
}
} // namespace host

namespace device {
auto dgemm(
    DgemmKind kind,
    std::size_t m, std::size_t n, std::size_t k,
    double alpha,
    double const* A,
    double const* B,
    double beta,
    double const* C
) -> int32_t {
    cudaError_t cuda_status = cudaSuccess;
    double duration = 0.0;
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dim_grid(k / dim_block.x, k / dim_block.y);

    double* d_A;
    cuda_status = cudaMalloc((void**)&d_A, m * k * sizeof(double));
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "error: failed to allocate `d_A` (%s)\n", cudaGetErrorString(cuda_status));
    }

    double* d_B;
    cuda_status = cudaMalloc((void**)&d_B, k * n * sizeof(double));
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "error: failed to allocate `d_B` (%s)\n", cudaGetErrorString(cuda_status));
    }

    double* d_C;
    cuda_status = cudaMalloc((void**)&d_C, m * n * sizeof(double));
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "error: failed to allocate `d_C` (%s)\n", cudaGetErrorString(cuda_status));
    }

    double* result = (double*)(malloc(m * n * sizeof(double)));
    if (!result) {
        fprintf(stderr, "error: failed to allocate `result`\n");
        cuda_status = cudaErrorMemoryAllocation;
    }

    if (cuda_status != cudaSuccess) { goto failure; }
    
    cuda_status = cudaMemcpy(d_A, A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        fprintf(
            stderr,
            "error: failed to copy memory from `A` to `d_A` (%s)\n",
            cudaGetErrorString(cuda_status)
        );
        goto failure;
    }

    cuda_status = cudaMemcpy(d_B, B, m * k * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        fprintf(
            stderr,
            "error: failed to copy memory from `B` to `d_B` (%s)\n",
            cudaGetErrorString(cuda_status)
        );
        goto failure;
    }

    cuda_status = cudaMemcpy(d_C, C, m * n * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        fprintf(
            stderr,
            "error: failed to copy memory from `C` to `d_C` (%s)\n",
            cudaGetErrorString(cuda_status)
        );
        goto failure;
    }

    struct timespec start, stop;
    // Invoke kernel
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    if (kind == DgemmKind::Naive) {
        kernels::device::dgemm<<<dim_grid, dim_block>>>(
            m, n, k,
            alpha,
            d_A, k,
            d_B, n,
            beta,
            d_C, n
        );
    } else if (kind == DgemmKind::Shared) {
        kernels::device::dgemm_shared<<<dim_grid, dim_block>>>(
            m, n, k,
            alpha,
            d_A, k,
            d_B, n,
            beta,
            d_C, n
        );
    }
    
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(
            stderr,
            "error: failed to launch kernel (%s)\n",
            cudaGetErrorString(cuda_status)
        );
        goto failure;
    }

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &stop);
    duration = utils::elapsed_seconds(start, stop);

    cuda_status = cudaMemcpy(result, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        fprintf(
            stderr,
            "error: failed to copy memory from `d_C` to `result` (%s)\n",
            cudaGetErrorString(cuda_status)
        );
        goto failure;
    }

    #if defined(DEBUG)
    printf("d_C(0, 0) = %lf\n", result[0]);
    #endif
    printf(
        "Device time: %.6lf ms, GFlop/s: %.3lf\n",
        duration * 1.0E-3,
        m * n * k * 1.0E-9 / duration
    );

failure:
    if (d_A) { cudaFree(d_A); }
    if (d_B) { cudaFree(d_B); }
    if (d_C) { cudaFree(d_C); }
    if (result) { free(result); }

    return cuda_status;
}
} // namespace device
} // namespace drivers
