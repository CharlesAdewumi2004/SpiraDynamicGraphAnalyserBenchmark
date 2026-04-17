#include "mutation.h"

#include <cstdint>
#include <unordered_map>

namespace {

using Edge = std::pair<uint32_t, uint32_t>;

struct EdgeLoc {
    uint32_t live_idx;  // position in live_edges
    uint32_t src_idx;   // position in edges_by_src[src]
};

// Fenwick (binary-indexed) tree over per-source out-degree for O(log N)
// weighted sampling and O(log N) degree updates.  Replaces the O(N) full scan
// that the old implementation ran for every hub-biased deletion.
class FenwickTree {
public:
    explicit FenwickTree(uint32_t n) : n_(n), bit_(n + 1, 0) {
        log_step_ = 1;
        while (log_step_ * 2 <= n_) log_step_ *= 2;
    }

    void update(uint32_t pos, int64_t delta) {
        for (uint32_t i = pos + 1; i <= n_; i += i & -i) bit_[i] += delta;
    }

    int64_t total() const {
        int64_t sum = 0;
        for (uint32_t i = n_; i > 0; i -= i & -i) sum += bit_[i];
        return sum;
    }

    // Smallest 0-indexed pos whose cumulative weight is > target.
    // Requires target in [0, total()).
    uint32_t find(int64_t target) const {
        uint32_t pos = 0;
        for (uint32_t step = log_step_; step > 0; step >>= 1) {
            uint32_t next = pos + step;
            if (next <= n_ && bit_[next] <= target) {
                pos = next;
                target -= bit_[pos];
            }
        }
        return pos;
    }

private:
    uint32_t n_;
    uint32_t log_step_;
    std::vector<int64_t> bit_;
};

// Live-edge index with O(1) uniform pick/remove and O(log N) hub-biased pick.
class MutationState {
public:
    MutationState(uint32_t num_nodes, const EdgeSet& initial,
                  const std::vector<uint32_t>& out_degree)
        : edges_by_src_(num_nodes),
          degree_tree_(num_nodes)
    {
        live_edges_.reserve(initial.size());
        edge_loc_.reserve(initial.size());

        for (const auto& e : initial) {
            const uint32_t src = e.second;
            EdgeLoc loc{
                static_cast<uint32_t>(live_edges_.size()),
                static_cast<uint32_t>(edges_by_src_[src].size())
            };
            edge_loc_.emplace(e, loc);
            live_edges_.push_back(e);
            edges_by_src_[src].push_back(e);
        }

        for (uint32_t i = 0; i < num_nodes; ++i) {
            if (out_degree[i] != 0)
                degree_tree_.update(i, static_cast<int64_t>(out_degree[i]));
        }
    }

    bool contains(const Edge& e) const { return edge_loc_.count(e) != 0; }
    size_t size() const { return live_edges_.size(); }

    void insert(const Edge& e) {
        const uint32_t src = e.second;
        EdgeLoc loc{
            static_cast<uint32_t>(live_edges_.size()),
            static_cast<uint32_t>(edges_by_src_[src].size())
        };
        edge_loc_.emplace(e, loc);
        live_edges_.push_back(e);
        edges_by_src_[src].push_back(e);
        degree_tree_.update(src, 1);
    }

    void remove(const Edge& e) {
        auto it = edge_loc_.find(e);
        const EdgeLoc loc = it->second;
        const uint32_t src = e.second;

        // Swap-with-back in live_edges_.
        const uint32_t last_live = static_cast<uint32_t>(live_edges_.size() - 1);
        if (loc.live_idx != last_live) {
            Edge moved = live_edges_.back();
            live_edges_[loc.live_idx] = moved;
            edge_loc_[moved].live_idx = loc.live_idx;
        }
        live_edges_.pop_back();

        // Swap-with-back in edges_by_src_[src].
        auto& bucket = edges_by_src_[src];
        const uint32_t last_src = static_cast<uint32_t>(bucket.size() - 1);
        if (loc.src_idx != last_src) {
            Edge moved = bucket.back();
            bucket[loc.src_idx] = moved;
            edge_loc_[moved].src_idx = loc.src_idx;
        }
        bucket.pop_back();

        edge_loc_.erase(it);
        degree_tree_.update(src, -1);
    }

    Edge pick_uniform(std::mt19937_64& rng) const {
        std::uniform_int_distribution<uint32_t> dist(
            0, static_cast<uint32_t>(live_edges_.size() - 1));
        return live_edges_[dist(rng)];
    }

    Edge pick_hub_biased(std::mt19937_64& rng) const {
        const int64_t total = degree_tree_.total();
        if (total <= 0) return pick_uniform(rng);

        std::uniform_int_distribution<int64_t> sdist(0, total - 1);
        const uint32_t src = degree_tree_.find(sdist(rng));

        const auto& bucket = edges_by_src_[src];
        std::uniform_int_distribution<uint32_t> edist(
            0, static_cast<uint32_t>(bucket.size() - 1));
        return bucket[edist(rng)];
    }

private:
    std::vector<Edge> live_edges_;
    std::unordered_map<Edge, EdgeLoc, EdgeHash> edge_loc_;
    std::vector<std::vector<Edge>> edges_by_src_;
    FenwickTree degree_tree_;
};

} // namespace

std::vector<MutationBatch> generate_all_batches(
    int num_phases,
    int insertions_per_batch,
    EdgeSet& current_edges,
    std::vector<std::pair<uint32_t, uint32_t>>& insert_pool,
    const std::vector<uint32_t>& out_degree,
    uint32_t num_nodes,
    int scale,
    const RmatParams& rmat_params,
    std::mt19937_64& rng)
{
    MutationState state(num_nodes, current_edges, out_degree);

    const int deletions_per_batch = static_cast<int>(0.3 * insertions_per_batch);
    size_t pool_idx = 0;

    std::vector<MutationBatch> batches;
    batches.reserve(num_phases);

    for (int phase = 0; phase < num_phases; ++phase) {
        MutationBatch batch;
        batch.mutations.reserve(insertions_per_batch + deletions_per_batch);

        // ── Insertions ──────────────────────────────────────────────────
        int inserted = 0;
        while (inserted < insertions_per_batch) {
            Edge edge;
            if (pool_idx < insert_pool.size()) {
                edge = insert_pool[pool_idx++];
            } else {
                edge = rmat_single_edge(scale, rmat_params, rng);
                if (edge.first == edge.second) continue;
            }
            if (state.contains(edge)) continue;

            state.insert(edge);
            batch.mutations.push_back({edge.first, edge.second, true});
            ++inserted;
        }

        // ── Deletions ───────────────────────────────────────────────────
        for (int d = 0; d < deletions_per_batch && state.size() > 0; ++d) {
            const bool hub_biased = (d % 2 == 1);
            const Edge edge = hub_biased ? state.pick_hub_biased(rng)
                                         : state.pick_uniform(rng);
            state.remove(edge);
            batch.mutations.push_back({edge.first, edge.second, false});
        }

        batches.push_back(std::move(batch));
    }

    return batches;
}
