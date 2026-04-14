#pragma once

#include <cstdint>
#include <vector>

struct PhaseReference {
    size_t expected_nnz;
    std::vector<double> expected_rank;
};

// Check structural and numerical correctness against reference values.
// Returns true if nnz matches exactly and rank L1 norm < tolerance.
bool verify_phase(size_t actual_nnz,
                  const std::vector<double>& actual_rank,
                  const PhaseReference& reference,
                  double tolerance = 1e-10);

// Compute L1 norm of difference between two vectors.
double l1_norm_diff(const std::vector<double>& a, const std::vector<double>& b);
