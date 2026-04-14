#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mutation.h"

// Abstract interface that each sparse-matrix library implements.
class SpMVProvider {
public:
    virtual ~SpMVProvider() = default;

    virtual std::string name() const = 0;

    // Load the initial edge set (called once, untimed).
    virtual void bulk_load(const std::vector<std::pair<uint32_t, uint32_t>>& edges,
                           uint32_t num_nodes) = 0;

    // Apply a batch of mutations (insertions and deletions).
    virtual void apply_mutations(const MutationBatch& batch) = 0;

    // Sparse matrix-vector multiply: y = A * x.
    virtual void spmv(const double* x, double* y, uint32_t n) const = 0;

    // Current number of structural nonzeros.
    virtual size_t nnz() const = 0;
};
