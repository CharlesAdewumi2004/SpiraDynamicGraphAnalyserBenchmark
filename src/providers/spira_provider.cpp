#ifdef SPIRA_HAS_SPIRA

#include "provider.h"
#include "spira_provider.h"

#include <spira/spira.hpp>
#include <spira/parallel/parallel.hpp>

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

// Use the parallel_matrix with SOA layout + uint32_t indices + double values
// to hit Spira's SIMD SpMV codepath (AVX2/AVX-512 sparse_dot_double).
//
// Lock policy: compact_preserve — keeps per-row buffers after CSR build so
// that open() is O(1).  This matches the streaming update-then-compute cycle
// where we alternate between mutations and SpMV every phase.
//
// Insert policy: staged — accumulates inserts in L1/L2-friendly staging
// buffers before flushing to row buffers, reducing random-access cache misses
// from the shuffled mutation order.

using SpiraMatrix = spira::parallel::parallel_matrix<
    spira::layout::tags::soa_tag,
    uint32_t,
    double,
    spira::buffer::tags::array_buffer<spira::layout::tags::soa_tag>,
    64,
    spira::config::lock_policy::compact_preserve,
    spira::config::insert_policy::staged,
    256>;

class SpiraProvider : public SpMVProvider {
public:
    explicit SpiraProvider(std::size_t n_threads)
        : n_threads_(n_threads) {}

    std::string name() const override { return "Spira"; }

    void bulk_load(const std::vector<std::pair<uint32_t, uint32_t>>& edges,
                   uint32_t num_nodes) override
    {
        n_ = num_nodes;

        // Construct the parallel matrix (starts in open mode).
        mat_.reset();
        mat_ = std::make_unique<SpiraMatrix>(n_, n_, n_threads_);

        // Use parallel_fill for efficient bulk loading — each worker thread
        // writes directly into its own partition with zero routing overhead.
        // Pre-bucket edges by their destination row (which determines partition
        // ownership) so each thread only touches its own rows.
        mat_->parallel_fill(
            [&edges](auto& rows, std::size_t row_start, std::size_t row_end,
                     std::size_t /*thread_id*/) {
                for (const auto& [dst, src] : edges) {
                    if (dst >= row_start && dst < row_end) {
                        rows[dst - row_start].insert(src, 1.0);
                    }
                }
            });

        mat_->lock();

        // Pre-allocate persistent SpMV vectors to avoid per-call heap allocs.
        xv_.resize(n_);
        yv_.resize(n_);
    }

    void apply_mutations(const MutationBatch& batch) override {
        mat_->open();

        for (const auto& m : batch.mutations) {
            if (m.is_insert) {
                mat_->insert(m.row, m.col, 1.0);
            } else {
                // Spira removes entries by inserting a zero value; the
                // zero-filter in lock() strips them from the CSR.
                mat_->insert(m.row, m.col, 0.0);
            }
        }

        mat_->lock();
    }

    void spmv(const double* x, double* y, uint32_t n) const override {
        // Copy input into persistent vector (Spira's API requires std::vector).
        // Use memcpy — no construction overhead, just a flat 8*N byte copy.
        std::memcpy(xv_.data(), x, sizeof(double) * n);

        spira::parallel::algorithms::spmv(
            const_cast<SpiraMatrix&>(*mat_), xv_, yv_);

        std::memcpy(y, yv_.data(), sizeof(double) * n);
    }

    size_t nnz() const override {
        return mat_->nnz();
    }

private:
    std::size_t n_threads_;
    uint32_t n_ = 0;
    std::unique_ptr<SpiraMatrix> mat_;

    // Persistent vectors reused across all SpMV calls to avoid heap allocation.
    // Mutable because spmv() is const but we need to write into these buffers.
    mutable std::vector<double> xv_;
    mutable std::vector<double> yv_;
};

std::unique_ptr<SpMVProvider> make_spira_provider(std::size_t n_threads) {
    return std::make_unique<SpiraProvider>(n_threads);
}

#endif // SPIRA_HAS_SPIRA
