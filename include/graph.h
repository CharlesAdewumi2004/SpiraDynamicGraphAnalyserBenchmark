#pragma once

#include <cstdint>
#include <utility>
#include <vector>
#include <random>
#include <unordered_set>

struct EdgeHash {
    size_t operator()(const std::pair<uint32_t, uint32_t>& e) const {
        return std::hash<uint64_t>{}(
            (static_cast<uint64_t>(e.first) << 32) | e.second);
    }
};

using EdgeSet = std::unordered_set<std::pair<uint32_t, uint32_t>, EdgeHash>;

struct GraphData {
    uint32_t num_nodes;
    std::vector<std::pair<uint32_t, uint32_t>> initial_edges;  // 80%
    std::vector<std::pair<uint32_t, uint32_t>> insert_pool;    // 20%
};

// Remove self-loops and duplicate edges, shuffle, and split 80/20.
GraphData process_edges(
    std::vector<std::pair<uint32_t, uint32_t>>& raw_edges,
    uint32_t num_nodes,
    std::mt19937_64& rng);
