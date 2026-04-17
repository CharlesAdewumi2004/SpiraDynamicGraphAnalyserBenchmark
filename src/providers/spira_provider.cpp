#ifdef SPIRA_HAS_SPIRA

#include "provider.h"
#include "spira_provider.h"
#include "graph.h"

#include <spira/spira.hpp>
#include <spira/parallel/parallel.hpp>

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

// Use the parallel_matrix with SOA layout + uint32_t indices + double values
// to hit Spira's SIMD SpMV codepath (AVX2/AVX-512 sparse_dot_double).
// parallel_fill + lock() is used for each phase rebuild (see apply_mutations).

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

        edge_set_.clear();
        edge_set_.reserve(edges.size());
        for (const auto& e : edges)
            edge_set_.insert(e);

        rebuild_matrix();

        // Pre-allocate persistent SpMV vectors to avoid per-call heap allocs.
        xv_.resize(n_);
        yv_.resize(n_);
    }

    void apply_mutations(const MutationBatch& batch) override {
        for (const auto& m : batch.mutations) {
            if (m.is_insert)
                edge_set_.insert({m.row, m.col});
            else
                edge_set_.erase({m.row, m.col});
        }

        // Spira's zero-insert deletion is broken: soa_array_buffer::sort_and_dedup()
        // filters zeros from the row buffer before merge_csr() runs, so the zero
        // never reaches merge_csr() to tombstone the old CSR entry.  The deleted
        // edge therefore survives in the CSR unchanged (300 ghosts × 50 phases =
        // 15,000 NNZ excess at phase 49).  Rebuild from the live edge set instead.
        rebuild_matrix();
    }

    void rebuild_matrix() {
        mat_.reset();
        mat_ = std::make_unique<SpiraMatrix>(n_, n_, n_threads_);
        mat_->parallel_fill(
            [this](auto& rows, std::size_t row_start, std::size_t row_end,
                   std::size_t /*thread_id*/) {
                for (const auto& [dst, src] : edge_set_) {
                    if (dst >= row_start && dst < row_end)
                        rows[dst - row_start].insert(src, 1.0);
                }
            });
        mat_->lock();
    }

    void spmv(const double* x, double* y, uint32_t n) const override {
        // Fallback raw-pointer path — copies into persistent vectors.
        std::memcpy(xv_.data(), x, sizeof(double) * n);
        spira::parallel::algorithms::spmv(
            const_cast<SpiraMatrix&>(*mat_), xv_, yv_);
        std::memcpy(y, yv_.data(), sizeof(double) * n);
    }

    void spmv_vec(std::vector<double>& x, std::vector<double>& y,
                  uint32_t /*n*/) const override {
        // Zero-copy path — PageRank writes directly into x, Spira writes
        // directly into y.  No memcpy overhead.
        spira::parallel::algorithms::spmv(
            const_cast<SpiraMatrix&>(*mat_), x, y);
    }

    size_t nnz() const override {
        return mat_->nnz();
    }

private:
    std::size_t n_threads_;
    uint32_t n_ = 0;
    std::unique_ptr<SpiraMatrix> mat_;
    EdgeSet edge_set_;

    // Persistent vectors reused across all SpMV calls to avoid heap allocation.
    // Mutable because spmv() is const but we need to write into these buffers.
    mutable std::vector<double> xv_;
    mutable std::vector<double> yv_;
};

std::unique_ptr<SpMVProvider> make_spira_provider(std::size_t n_threads) {
    return std::make_unique<SpiraProvider>(n_threads);
}

#endif // SPIRA_HAS_SPIRA
