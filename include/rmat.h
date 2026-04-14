#pragma once

#include <cstdint>
#include <utility>
#include <vector>
#include <random>

struct RmatParams {
    double a = 0.57;
    double b = 0.19;
    double c = 0.19;
    double d = 0.05;
    int    edge_factor = 16;
};

// Generate raw R-MAT edges (may contain duplicates and self-loops).
std::vector<std::pair<uint32_t, uint32_t>>
rmat_generate(int scale, const RmatParams& params, std::mt19937_64& rng);

// Generate a single R-MAT edge.
std::pair<uint32_t, uint32_t>
rmat_single_edge(int scale, const RmatParams& params, std::mt19937_64& rng);
