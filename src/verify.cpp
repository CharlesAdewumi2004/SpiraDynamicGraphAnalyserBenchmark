#include "verify.h"

#include <cmath>
#include <cstdio>

double l1_norm_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        sum += std::fabs(a[i] - b[i]);
    }
    // If sizes differ, count missing elements as full errors.
    if (a.size() > n) {
        for (size_t i = n; i < a.size(); ++i) sum += std::fabs(a[i]);
    }
    if (b.size() > n) {
        for (size_t i = n; i < b.size(); ++i) sum += std::fabs(b[i]);
    }
    return sum;
}

bool verify_phase(size_t actual_nnz,
                  const std::vector<double>& actual_rank,
                  const PhaseReference& reference,
                  double tolerance)
{
    bool ok = true;

    if (actual_nnz != reference.expected_nnz) {
        std::fprintf(stderr, "  NNZ mismatch: got %zu, expected %zu\n",
                     actual_nnz, reference.expected_nnz);
        ok = false;
    }

    double l1 = l1_norm_diff(actual_rank, reference.expected_rank);
    if (l1 >= tolerance) {
        std::fprintf(stderr, "  Rank vector L1 norm = %.3e (tolerance %.3e)\n",
                     l1, tolerance);
        ok = false;
    }

    return ok;
}
