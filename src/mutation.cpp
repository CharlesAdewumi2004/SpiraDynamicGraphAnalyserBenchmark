#include "mutation.h"

#include <algorithm>
#include <numeric>
#include <unordered_map>

using SrcAdjacency = std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>>;

// Pick a deletion edge: 50% uniform random, 50% hub-biased.
static std::pair<uint32_t, uint32_t> pick_deletion(
    const EdgeSet& current_edges,
    const std::vector<uint32_t>& out_degree,
    uint32_t num_nodes,
    bool hub_biased,
    std::mt19937_64& rng,
    SrcAdjacency* src_adj_cache)
{
    if (!hub_biased) {
        // Uniform random deletion: pick a random edge from the set.
        std::uniform_int_distribution<size_t> dist(0, current_edges.size() - 1);
        auto it = current_edges.begin();
        std::advance(it, dist(rng));
        return *it;
    }

    // Hub-biased: pick source node proportional to out_degree, then pick one of
    // its edges uniformly. Use pre-built src_adj_cache (O(1) amortised).
    uint64_t total_degree = 0;
    for (uint32_t i = 0; i < num_nodes; ++i) {
        total_degree += out_degree[i];
    }

    if (total_degree == 0 || src_adj_cache->empty()) {
        // Fallback to uniform.
        std::uniform_int_distribution<size_t> dist(0, current_edges.size() - 1);
        auto it = current_edges.begin();
        std::advance(it, dist(rng));
        return *it;
    }

    std::uniform_int_distribution<uint64_t> ddist(0, total_degree - 1);
    uint64_t target = ddist(rng);
    uint64_t cumulative = 0;
    uint32_t chosen_src = 0;

    for (uint32_t i = 0; i < num_nodes; ++i) {
        cumulative += out_degree[i];
        if (cumulative > target) {
            chosen_src = i;
            break;
        }
    }

    // Look up edges from chosen_src in the cache (O(1) amortised).
    auto it = src_adj_cache->find(chosen_src);
    if (it == src_adj_cache->end() || it->second.empty()) {
        // Fallback to uniform if no edges found for this source.
        std::uniform_int_distribution<size_t> dist(0, current_edges.size() - 1);
        auto edge_it = current_edges.begin();
        std::advance(edge_it, dist(rng));
        return *edge_it;
    }

    const auto& src_edges = it->second;
    std::uniform_int_distribution<size_t> edist(0, src_edges.size() - 1);
    return src_edges[edist(rng)];
}

std::vector<MutationBatch> generate_all_batches(
    int num_phases,
    int insertions_per_batch,
    EdgeSet& current_edges,
    std::vector<std::pair<uint32_t, uint32_t>>& insert_pool,
    const std::vector<uint32_t>& out_degree_in,
    uint32_t num_nodes,
    int scale,
    const RmatParams& rmat_params,
    std::mt19937_64& rng)
{
    std::vector<uint32_t> out_degree = out_degree_in;
    int deletions_per_batch = static_cast<int>(0.3 * insertions_per_batch);
    size_t pool_idx = 0;

    // Build source adjacency cache once: O(nnz), reused for all hub-biased deletions.
    // Without this, hub-biased deletion scans all nnz edges per deletion: O(nnz * deletions).
    SrcAdjacency src_adj_cache;
    for (const auto& [dst, src] : current_edges) {
        src_adj_cache[src].push_back({dst, src});
    }

    std::vector<MutationBatch> batches;
    batches.reserve(num_phases);

    for (int phase = 0; phase < num_phases; ++phase) {
        MutationBatch batch;
        batch.mutations.reserve(insertions_per_batch + deletions_per_batch);

        // ── Insertions ──────────────────────────────────────────────────
        int inserted = 0;
        while (inserted < insertions_per_batch) {
            std::pair<uint32_t, uint32_t> edge;

            if (pool_idx < insert_pool.size()) {
                edge = insert_pool[pool_idx++];
            } else {
                // Pool exhausted: generate fresh R-MAT edges.
                edge = rmat_single_edge(scale, rmat_params, rng);
                // Skip self-loops.
                if (edge.first == edge.second) continue;
            }

            // De-duplicate against current matrix state.
            if (current_edges.count(edge)) continue;

            current_edges.insert(edge);
            out_degree[edge.second]++;
            src_adj_cache[edge.second].push_back(edge);
            batch.mutations.push_back({edge.first, edge.second, true});
            ++inserted;
        }

        // ── Deletions ───────────────────────────────────────────────────
        for (int d = 0; d < deletions_per_batch && !current_edges.empty(); ++d) {
            bool hub_biased = (d % 2 == 1); // 50/50 split
            auto edge = pick_deletion(current_edges, out_degree, num_nodes,
                                      hub_biased, rng, &src_adj_cache);

            current_edges.erase(edge);
            out_degree[edge.second]--;

            // Update src_adj_cache: remove this edge from the source's list.
            auto& src_edges = src_adj_cache[edge.second];
            auto it = std::find(src_edges.begin(), src_edges.end(), edge);
            if (it != src_edges.end()) {
                src_edges.erase(it);
            }

            batch.mutations.push_back({edge.first, edge.second, false});
        }

        batches.push_back(std::move(batch));
    }

    return batches;
}
