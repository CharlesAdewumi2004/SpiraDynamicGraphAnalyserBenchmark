#include "provider.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>
#include <vector>

#include <omp.h>

// CSR-rebuild provider: mimics what Eigen/MKL/Armadillo must do.
// On every apply_mutations call, it rebuilds the full CSR from the live edge
// set.  SpMV runs on the compressed CSR — fast compute, expensive updates.
//
// The rebuild uses a counting-sort scatter (O(nnz + N)) rather than a full
// comparison sort (O(nnz log nnz)).  This is the same algorithmic approach
// that Eigen's setFromTriplets and MKL's COO-to-CSR conversion use internally.

class CSRReferenceProvider : public SpMVProvider {
public:
    std::string name() const override { return "CSR-Reference"; }

    void bulk_load(const std::vector<std::pair<uint32_t, uint32_t>>& edges,
                   uint32_t num_nodes) override
    {
        n_ = num_nodes;
        edge_set_.clear();
        edge_set_.reserve(edges.size());

        for (const auto& [dst, src] : edges) {
            edge_set_.insert({dst, src});
        }

        rebuild_csr();
    }

    void apply_mutations(const MutationBatch& batch) override {
        for (const auto& m : batch.mutations) {
            if (m.is_insert) {
                edge_set_.insert({m.row, m.col});
            } else {
                edge_set_.erase({m.row, m.col});
            }
        }

        rebuild_csr();
    }

    void spmv(const double* x, double* y, uint32_t n) const override {
        // y = A * x using CSR.  Each row is independent → trivially parallel.
        // dynamic scheduling handles the power-law row-length variance.
        #pragma omp parallel for schedule(dynamic, 512)
        for (uint32_t i = 0; i < n; ++i) {
            double sum = 0.0;
            for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
                sum += x[col_idx_[k]];
            }
            y[i] = sum;
        }
    }

    size_t nnz() const override { return edge_set_.size(); }

private:
    void rebuild_csr() {
        size_t nnz_count = edge_set_.size();

        // Counting-sort scatter: O(nnz + N).
        // Pass 1: count entries per row.
        row_ptr_.assign(static_cast<size_t>(n_) + 1, 0);
        for (const auto& [dst, src] : edge_set_) {
            row_ptr_[dst + 1]++;
        }

        // Prefix sum.
        for (uint32_t i = 0; i < n_; ++i) {
            row_ptr_[i + 1] += row_ptr_[i];
        }

        // Pass 2: scatter column indices into position using write cursors.
        col_idx_.resize(nnz_count);
        write_pos_.resize(static_cast<size_t>(n_) + 1);
        std::memcpy(write_pos_.data(), row_ptr_.data(),
                    (static_cast<size_t>(n_) + 1) * sizeof(size_t));

        for (const auto& [dst, src] : edge_set_) {
            col_idx_[write_pos_[dst]++] = src;
        }

        // Optional: sort column indices within each row for better SpMV
        // cache locality (gather pattern hits sequential cache lines).
        #pragma omp parallel for schedule(dynamic, 1024)
        for (uint32_t i = 0; i < n_; ++i) {
            if (row_ptr_[i + 1] - row_ptr_[i] > 1) {
                std::sort(col_idx_.begin() + row_ptr_[i],
                          col_idx_.begin() + row_ptr_[i + 1]);
            }
        }
    }

    uint32_t n_ = 0;
    EdgeSet edge_set_;

    // CSR arrays (values are all 1.0 so we skip the values array).
    std::vector<size_t> row_ptr_;
    std::vector<size_t> write_pos_;  // reusable scratch for scatter
    std::vector<uint32_t> col_idx_;
};

#include "csr_reference.h"

std::unique_ptr<SpMVProvider> make_csr_reference() {
    return std::make_unique<CSRReferenceProvider>();
}
