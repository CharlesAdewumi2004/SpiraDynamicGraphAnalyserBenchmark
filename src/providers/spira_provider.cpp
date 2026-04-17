#ifdef SPIRA_HAS_SPIRA

#include "provider.h"
#include "spira_provider.h"
#include "graph.h"

#include <spira/spira.hpp>
#include <spira/parallel/parallel.hpp>

#include <cstring>
#include <vector>

// Use the parallel_matrix with SOA layout + uint32_t indices + double values
// to hit Spira's SIMD SpMV codepath (AVX2/AVX-512 sparse_dot_double).
// parallel_fill + lock() is used for bulk load; incremental mutations stage
// into the buffer (zero = tombstone for compact_preserve) and flush via lock().

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
        mat_ = std::make_unique<SpiraMatrix>(n_, n_, n_threads_);
        mat_->parallel_fill(
            [&edges](auto& rows, std::size_t row_start, std::size_t row_end,
                     std::size_t /*thread_id*/) {
                for (const auto& [dst, src] : edges) {
                    if (dst >= row_start && dst < row_end)
                        rows[dst - row_start].insert(src, 1.0);
                }
            });
        mat_->lock();

        xv_.resize(n_);
        yv_.resize(n_);
    }

    void apply_mutations(const MutationBatch& batch) override {
        // Insertions stage value 1.0; deletions stage 0.0.
        // With compact_preserve lock policy, lock_for_compact() calls
        // sort_and_dedup_keep_zeros() so zeros survive to merge_csr(), which
        // then suppresses the old CSR entry — correct incremental deletion.
        mat_->open();
        mat_->parallel_fill(
            [&batch](auto& rows, std::size_t row_start, std::size_t row_end,
                     std::size_t /*thread_id*/) {
                for (const auto& m : batch.mutations) {
                    if (m.row >= row_start && m.row < row_end)
                        rows[m.row - row_start].insert(m.col,
                                                       m.is_insert ? 1.0 : 0.0);
                }
            });
        mat_->lock();
    }

    void spmv(const double* x, double* y, uint32_t n) const override {
        std::memcpy(xv_.data(), x, sizeof(double) * n);
        spira::parallel::algorithms::spmv(
            const_cast<SpiraMatrix&>(*mat_), xv_, yv_);
        std::memcpy(y, yv_.data(), sizeof(double) * n);
    }

    void spmv_vec(std::vector<double>& x, std::vector<double>& y,
                  uint32_t /*n*/) const override {
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

    mutable std::vector<double> xv_;
    mutable std::vector<double> yv_;
};

std::unique_ptr<SpMVProvider> make_spira_provider(std::size_t n_threads) {
    return std::make_unique<SpiraProvider>(n_threads);
}

#endif // SPIRA_HAS_SPIRA
