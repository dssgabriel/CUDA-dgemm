#include "drivers.hpp"
#include "utils.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

auto main(int32_t argc, char** argv) -> int32_t {
    std::size_t size = (argc < 2) ? 1024 : (std::size_t)(atoi(argv[1]));
    printf("Double-precision general matrix multiplication benchmark (%zux%zu)\n", size, size);

    int32_t status_code = 0;
    double alpha = drand48();
    double beta = drand48();

    double* A = (double*)(malloc(size * size * sizeof(double)));
    if (!A) {
        fprintf(stderr, "error: failed to allocate matrix `A`\n");
        status_code = -1;
    }
    double* B = (double*)(malloc(size * size * sizeof(double)));
    if (!B) {
        fprintf(stderr, "error: failed to allocate matrix `B`\n");
        status_code = -1;
    }
    double* C = (double*)(malloc(size * size * sizeof(double)));
    if (!C) {
        fprintf(stderr, "error: failed to allocate matrix `C`\n");
        status_code = -1;
    }

    if (status_code != 0) { goto failure; }
    utils::init_matrices(A, B, C, size);
    status_code = drivers::host::dgemm(size, size, size, alpha, A, B, beta, C);
    if (status_code != 0) {
        fprintf(stderr, "error: something went wrong running host `dgemm`, aborting\n");
        goto exit;
    }

    status_code = drivers::device::dgemm(
        drivers::device::DgemmKind::Naive,
        size, size, size,
        alpha,
        A,
        B,
        beta,
        C
    );
    if (status_code != 0) {
        fprintf(stderr, "error: something went wrong running naive device `dgemm`, aborting\n");
        goto exit;
    }

    status_code = drivers::device::dgemm(
        drivers::device::DgemmKind::Shared,
        size, size, size,
        alpha,
        A,
        B,
        beta,
        C
    );
    if (status_code != 0) {
        fprintf(stderr, "error: something went wrong running shared device `dgemm`, aborting\n");
        goto exit;
    }

failure:
    if (A) { free(A); }
    if (B) { free(B); }
    if (C) { free(C); }

exit:
    return status_code;
}