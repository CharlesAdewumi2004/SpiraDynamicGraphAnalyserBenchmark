#pragma once

#include <cstdint>
#include <vector>

#include "provider.h"

struct PageRankState {
    std::vector<double> rank;
    std::vector<uint32_t> out_degree;
    uint32_t num_nodes;
    uint32_t dangling_count;

    // Persistent scratch vectors reused across all run_pagerank calls.
    std::vector<double> scratch_x;
    std::vector<double> scratch_y;

    explicit PageRankState(uint32_t n);

    // Initialise out_degree from initial edge list.
    void init_degrees(const std::vector<std::pair<uint32_t, uint32_t>>& edges);

    // Update out_degree for a mutation batch.
    void update_degrees(const MutationBatch& batch);
};

// Run `iterations` of PageRank SpMV (fixed iteration count, no convergence check).
// Damping factor d = 0.85.
void run_pagerank(SpMVProvider& provider, PageRankState& state, int iterations);
