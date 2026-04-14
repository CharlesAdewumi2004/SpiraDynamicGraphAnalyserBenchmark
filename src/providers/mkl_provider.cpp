#ifdef SPIRA_HAS_MKL

#include "provider.h"
#include "mkl_provider.h"

#include <mkl.h>
#include <mkl_spblas.h>
#include <algorithm>
#include <numeric>

class MKLProvider : public SpMVProvider {
public:
    ~MKLProvider() override { destroy_handle(); }

    std::string name() const override { return "MKL"; }

    void bulk_load(const std::vector<std::pair<uint32_t, uint32_t>>& edges,
                   uint32_t num_nodes) override
    {
        n_ = num_nodes;
        edge_set_.clear();
        edge_set_.reserve(edges.size());

        for (const auto& [dst, src] : edges) {
            edge_set_.insert({dst, src});
        }

        rebuild_csr_and_handle();
    }

    void apply_mutations(const MutationBatch& batch) override {
        for (const auto& m : batch.mutations) {
            if (m.is_insert) {
                edge_set_.insert({m.row, m.col});
            } else {
                edge_set_.erase({m.row, m.col});
            }
        }

        rebuild_csr_and_handle();
    }

    void spmv(const double* x, double* y, uint32_t /*n*/) const override {
        // MKL handles threading internally via its own OpenMP runtime.
        double alpha = 1.0, beta = 0.0;
        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;

        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                        alpha, handle_, descr, x, beta, y);
    }

    size_t nnz() const override { return edge_set_.size(); }

private:
    void destroy_handle() {
        if (handle_valid_) {
            mkl_sparse_destroy(handle_);
            handle_valid_ = false;
        }
    }

    void rebuild_csr_and_handle() {
        destroy_handle();

        size_t nnz_count = edge_set_.size();

        // Build COO sorted by row.
        std::vector<std::pair<uint32_t, uint32_t>> coo(edge_set_.begin(),
                                                        edge_set_.end());
        std::sort(coo.begin(), coo.end());

        // Build standard CSR arrays with MKL_INT indexing.
        // MKL's 3-array CSR: row_ptr[n+1], col_idx[nnz], values[nnz].
        row_ptr_.assign(static_cast<size_t>(n_) + 1, 0);
        col_idx_.resize(nnz_count);
        values_.assign(nnz_count, 1.0);

        for (size_t k = 0; k < nnz_count; ++k) {
            row_ptr_[coo[k].first + 1]++;
        }
        for (MKL_INT i = 0; i < static_cast<MKL_INT>(n_); ++i) {
            row_ptr_[i + 1] += row_ptr_[i];
        }
        for (size_t k = 0; k < nnz_count; ++k) {
            col_idx_[k] = static_cast<MKL_INT>(coo[k].second);
        }

        // Create CSR handle using the 3-array variant (row_ptr has n+1 entries).
        // Convert to the 4-array form MKL expects (rows_start, rows_end).
        rows_start_.resize(n_);
        rows_end_.resize(n_);
        for (MKL_INT i = 0; i < static_cast<MKL_INT>(n_); ++i) {
            rows_start_[i] = row_ptr_[i];
            rows_end_[i]   = row_ptr_[i + 1];
        }

        mkl_sparse_d_create_csr(&handle_, SPARSE_INDEX_BASE_ZERO,
                                 static_cast<MKL_INT>(n_),
                                 static_cast<MKL_INT>(n_),
                                 rows_start_.data(), rows_end_.data(),
                                 col_idx_.data(), values_.data());

        // Inspector-executor: tell MKL we'll do non-transposed SpMV repeatedly
        // so it can build optimised internal structures (e.g. BSR tiling).
        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        mkl_sparse_set_mv_hint(handle_, SPARSE_OPERATION_NON_TRANSPOSE,
                               descr, PAGERANK_EXPECTED_CALLS);
        mkl_sparse_optimize(handle_);

        handle_valid_ = true;
    }

    static constexpr MKL_INT PAGERANK_EXPECTED_CALLS = 20; // matches iteration count

    uint32_t n_ = 0;
    EdgeSet edge_set_;

    std::vector<MKL_INT> row_ptr_;
    std::vector<MKL_INT> rows_start_;
    std::vector<MKL_INT> rows_end_;
    std::vector<MKL_INT> col_idx_;
    std::vector<double> values_;

    sparse_matrix_t handle_ = nullptr;
    bool handle_valid_ = false;
};

std::unique_ptr<SpMVProvider> make_mkl_provider() {
    return std::make_unique<MKLProvider>();
}

#endif // SPIRA_HAS_MKL
