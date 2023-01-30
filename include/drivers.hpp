#pragma once

#include <cstdint>

namespace drivers {
namespace host {
auto dgemm(
    std::size_t m, std::size_t n, std::size_t k,
    double alpha,
    double const* A,
    double const* B,
    double beta,
    double const* C
) -> int32_t;
} // namespace host

namespace device {
enum class DgemmKind {
    Naive,
    Shared
};

auto dgemm(
    DgemmKind kind,
    std::size_t m, std::size_t n, std::size_t k,
    double alpha,
    double const* A,
    double const* B,
    double beta,
    double const* C
) -> int32_t;
} // namespace device
} // namespace drivers
