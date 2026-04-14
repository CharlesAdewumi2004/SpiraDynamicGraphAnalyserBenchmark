#pragma once

#include <cstdint>
#include <vector>
#include <random>

#include "graph.h"
#include "rmat.h"

struct Mutation {
    uint32_t row;       // destination node
    uint32_t col;       // source node
    bool     is_insert; // true = insert 1.0, false = remove
};

struct MutationBatch {
    std::vector<Mutation> mutations;
};

// Pre-generate all 50 mutation batches.
// B = number of insertions per batch; deletions = 0.3 * B per batch.
// Mutates current_edges in place to track the live edge set.
std::vector<MutationBatch> generate_all_batches(
    int num_phases,
    int insertions_per_batch,
    EdgeSet& current_edges,
    std::vector<std::pair<uint32_t, uint32_t>>& insert_pool,
    const std::vector<uint32_t>& out_degree,
    uint32_t num_nodes,
    int scale,
    const RmatParams& rmat_params,
    std::mt19937_64& rng);
