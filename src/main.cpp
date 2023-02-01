#include "drivers.hpp"
#include "utils.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define DEFAULT_SIZE 1024

auto main(int32_t argc, char** argv) -> int32_t {
    std::size_t n = (argc < 2) ? DEFAULT_SIZE : (std::size_t)(atoi(argv[1]));
    printf("DGEMM (%zux%zu):\n", n, n);

    // Allocations
    double* A = (double*)(malloc(n * n * sizeof(double)));
    if (!A) {
        fprintf(stderr, "error: failed to allocate matrix `A`\n");
        return EXIT_FAILURE;
    }

    double* B = (double*)(malloc(n * n * sizeof(double)));
    if (!B) {
        fprintf(stderr, "error: failed to allocate matrix `B`\n");
        free(A);
        return EXIT_FAILURE;
    }

    double* C = (double*)(malloc(n * n * sizeof(double)));
    if (!C) {
        fprintf(stderr, "error: failed to allocate matrix `C`\n");
        free(A);
        free(B);
        return EXIT_FAILURE;
    }

    // Initializations
    double alpha = drand48();
    double beta = drand48();
    utils::init_matrices(A, B, C, n);

    // Invoke kernels
    int32_t status_code;
    status_code = drivers::host::dgemm(n, n, n, alpha, A, B, beta, C);
    if (status_code != EXIT_SUCCESS) {
        fprintf(stderr, "error: something went wrong running host `dgemm`, aborting\n");
        goto failure;
    }

    status_code = drivers::device::dgemm(
        drivers::device::DgemmKind::Naive,
        n, n, n,
        alpha,
        A,
        B,
        beta,
        C
    );
    if (status_code != 0) {
        fprintf(stderr, "error: something went wrong running naive device `dgemm`, aborting\n");
        goto failure;
    }

    status_code = drivers::device::dgemm(
        drivers::device::DgemmKind::Shared,
        n, n, n,
        alpha,
        A,
        B,
        beta,
        C
    );
    if (status_code != 0) {
        fprintf(stderr, "error: something went wrong running shared device `dgemm`, aborting\n");
        goto failure;
    }

failure:
    if (A) { free(A); }
    if (B) { free(B); }
    if (C) { free(C); }

    return status_code;
}