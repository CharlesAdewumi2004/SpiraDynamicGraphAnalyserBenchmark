#include "pagerank.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <omp.h>

static constexpr double DAMPING = 0.85;

PageRankState::PageRankState(uint32_t n)
    : rank(n, 1.0 / n)
    , out_degree(n, 0)
    , num_nodes(n)
    , dangling_count(0)
    , scratch_x(n)
    , scratch_y(n)
{}

void PageRankState::init_degrees(
    const std::vector<std::pair<uint32_t, uint32_t>>& edges)
{
    std::fill(out_degree.begin(), out_degree.end(), 0);
    for (const auto& [dst, src] : edges) {
        out_degree[src]++;
    }
    dangling_count = 0;
    for (uint32_t i = 0; i < num_nodes; ++i) {
        if (out_degree[i] == 0) dangling_count++;
    }
}

void PageRankState::update_degrees(const MutationBatch& batch) {
    for (const auto& m : batch.mutations) {
        if (m.is_insert) {
            if (out_degree[m.col] == 0) dangling_count--;
            out_degree[m.col]++;
        } else {
            out_degree[m.col]--;
            if (out_degree[m.col] == 0) dangling_count++;
        }
    }
}

void run_pagerank(SpMVProvider& provider, PageRankState& state, int iterations) {
    const uint32_t n = state.num_nodes;
    const double inv_n = 1.0 / n;
    const double base = (1.0 - DAMPING) * inv_n;

    // Reuse persistent scratch vectors — no allocation in the hot path.
    double* __restrict__ x = state.scratch_x.data();
    double* __restrict__ y = state.scratch_y.data();

    for (int iter = 0; iter < iterations; ++iter) {
        // Step 1: Scale input vector — x[j] = r[j] / out_degree[j].
        // For dangling nodes (out_degree == 0), x[j] = 0; their rank is
        // redistributed uniformly via dangling_sum.
        double dangling_sum = 0.0;

        #pragma omp parallel for reduction(+:dangling_sum) schedule(static)
        for (uint32_t j = 0; j < n; ++j) {
            if (state.out_degree[j] > 0) {
                x[j] = state.rank[j] / state.out_degree[j];
            } else {
                x[j] = 0.0;
                dangling_sum += state.rank[j];
            }
        }

        // Step 2: SpMV — y = A * x.
        // Each provider is responsible for its own parallelisation here.
        // Use spmv_vec so providers with native vector APIs (Spira) avoid
        // memcpy overhead.  Other providers delegate to raw-pointer spmv.
        std::fill(state.scratch_y.begin(), state.scratch_y.end(), 0.0);
        provider.spmv_vec(state.scratch_x, state.scratch_y, n);

        // Step 3: Apply damping and dangling node redistribution.
        double dangling_contrib = DAMPING * dangling_sum * inv_n;

        #pragma omp parallel for schedule(static)
        for (uint32_t i = 0; i < n; ++i) {
            state.rank[i] = base + DAMPING * y[i] + dangling_contrib;
        }
    }
}
