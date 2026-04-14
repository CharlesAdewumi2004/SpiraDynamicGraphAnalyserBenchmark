#include "rmat.h"

#include <algorithm>

std::pair<uint32_t, uint32_t>
rmat_single_edge(int scale, const RmatParams& params, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    uint32_t u = 0, v = 0;
    for (int level = 0; level < scale; ++level) {
        double p = dist(rng);
        uint32_t half = 1u << (scale - 1 - level);

        if (p < params.a) {
            // top-left: u and v stay
        } else if (p < params.a + params.b) {
            v += half; // top-right
        } else if (p < params.a + params.b + params.c) {
            u += half; // bottom-left
        } else {
            u += half;
            v += half; // bottom-right
        }
    }
    return {u, v};
}

std::vector<std::pair<uint32_t, uint32_t>>
rmat_generate(int scale, const RmatParams& params, std::mt19937_64& rng) {
    uint64_t num_nodes = 1ull << scale;
    uint64_t target_edges = static_cast<uint64_t>(params.edge_factor) * num_nodes;

    std::vector<std::pair<uint32_t, uint32_t>> edges;
    edges.reserve(target_edges);

    for (uint64_t i = 0; i < target_edges; ++i) {
        edges.push_back(rmat_single_edge(scale, params, rng));
    }

    return edges;
}
