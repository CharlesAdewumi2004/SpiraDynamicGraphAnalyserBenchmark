#include "graph.h"

#include <algorithm>

GraphData process_edges(
    std::vector<std::pair<uint32_t, uint32_t>>& raw_edges,
    uint32_t num_nodes,
    std::mt19937_64& rng)
{
    // Remove self-loops.
    raw_edges.erase(
        std::remove_if(raw_edges.begin(), raw_edges.end(),
                       [](const auto& e) { return e.first == e.second; }),
        raw_edges.end());

    // Remove duplicates.
    EdgeSet seen;
    seen.reserve(raw_edges.size());
    std::vector<std::pair<uint32_t, uint32_t>> unique;
    unique.reserve(raw_edges.size());

    for (auto& e : raw_edges) {
        if (seen.insert(e).second) {
            unique.push_back(e);
        }
    }

    // Fisher-Yates shuffle.
    for (size_t i = unique.size() - 1; i > 0; --i) {
        std::uniform_int_distribution<size_t> dist(0, i);
        size_t j = dist(rng);
        std::swap(unique[i], unique[j]);
    }

    // 80/20 split.
    size_t split_point = (unique.size() * 80) / 100;

    GraphData data;
    data.num_nodes = num_nodes;
    data.initial_edges.assign(unique.begin(), unique.begin() + split_point);
    data.insert_pool.assign(unique.begin() + split_point, unique.end());

    return data;
}
