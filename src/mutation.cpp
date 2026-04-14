#include "mutation.h"

#include <algorithm>
#include <numeric>

// Pick a deletion edge: 50% uniform random, 50% hub-biased.
static std::pair<uint32_t, uint32_t> pick_deletion(
    const EdgeSet& current_edges,
    const std::vector<uint32_t>& out_degree,
    uint32_t num_nodes,
    bool hub_biased,
    std::mt19937_64& rng)
{
    if (!hub_biased) {
        // Uniform random deletion: pick a random edge from the set.
        std::uniform_int_distribution<size_t> dist(0, current_edges.size() - 1);
        auto it = current_edges.begin();
        std::advance(it, dist(rng));
        return *it;
    }

    // Hub-biased: pick source node proportional to out_degree, then pick one of
    // its edges uniformly.
    // Build a CDF over out_degrees.
    uint64_t total_degree = 0;
    for (uint32_t i = 0; i < num_nodes; ++i) {
        total_degree += out_degree[i];
    }

    if (total_degree == 0) {
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

    // Collect edges from chosen_src and pick one.
    std::vector<std::pair<uint32_t, uint32_t>> src_edges;
    for (const auto& e : current_edges) {
        // Edge (row, col) = (dst, src): col is source.
        if (e.second == chosen_src) {
            src_edges.push_back(e);
        }
    }

    if (src_edges.empty()) {
        // Fallback to uniform.
        std::uniform_int_distribution<size_t> dist(0, current_edges.size() - 1);
        auto it = current_edges.begin();
        std::advance(it, dist(rng));
        return *it;
    }

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
            batch.mutations.push_back({edge.first, edge.second, true});
            ++inserted;
        }

        // ── Deletions ───────────────────────────────────────────────────
        for (int d = 0; d < deletions_per_batch && !current_edges.empty(); ++d) {
            bool hub_biased = (d % 2 == 1); // 50/50 split
            auto edge = pick_deletion(current_edges, out_degree, num_nodes,
                                      hub_biased, rng);

            current_edges.erase(edge);
            out_degree[edge.second]--;
            batch.mutations.push_back({edge.first, edge.second, false});
        }

        batches.push_back(std::move(batch));
    }

    return batches;
}
