#ifdef SPIRA_HAS_EIGEN

#include "provider.h"
#include "eigen_provider.h"

#include <Eigen/Sparse>

// Eigen's SpMV (mat_ * xv) is automatically parallelised via OpenMP when
// Eigen is compiled with -fopenmp (which our CMake config ensures).
//
// The design doc (§4.1) specifies two strategies:
//   Option A — rebuild via setFromTriplets each phase (documented best practice)
//   Option B — incremental via coeffRef + prune + makeCompressed
//
// We implement Option A as the primary path.  Option A is the fairest
// comparison because it is what Eigen's own docs recommend for batch updates,
// and it avoids pathological O(nnz)-per-insert costs.

class EigenProvider : public SpMVProvider {
public:
    std::string name() const override { return "Eigen"; }

    void bulk_load(const std::vector<std::pair<uint32_t, uint32_t>>& edges,
                   uint32_t num_nodes) override
    {
        n_ = num_nodes;
        edge_set_.clear();
        edge_set_.reserve(edges.size());

        for (const auto& [dst, src] : edges) {
            edge_set_.insert({dst, src});
        }

        rebuild();
    }

    void apply_mutations(const MutationBatch& batch) override {
        for (const auto& m : batch.mutations) {
            if (m.is_insert) {
                edge_set_.insert({m.row, m.col});
            } else {
                edge_set_.erase({m.row, m.col});
            }
        }

        rebuild();
    }

    void spmv(const double* x, double* y, uint32_t n) const override {
        // Eigen parallelises this SpMV internally via OpenMP.
        Eigen::Map<const Eigen::VectorXd> xv(x, n);
        Eigen::Map<Eigen::VectorXd> yv(y, n);
        yv.noalias() = mat_ * xv;
    }

    size_t nnz() const override { return edge_set_.size(); }

private:
    void rebuild() {
        // Option A: full rebuild via setFromTriplets — the documented best
        // practice for batch construction in Eigen.
        // Reuse the triplets vector to avoid re-allocation.
        triplets_.clear();
        triplets_.reserve(edge_set_.size());
        for (const auto& [dst, src] : edge_set_) {
            triplets_.emplace_back(dst, src, 1.0);
        }

        mat_.resize(n_, n_);
        mat_.setFromTriplets(triplets_.begin(), triplets_.end());
        mat_.makeCompressed();
    }

    uint32_t n_ = 0;
    Eigen::SparseMatrix<double> mat_;
    std::vector<Eigen::Triplet<double>> triplets_;
    EdgeSet edge_set_;
};

std::unique_ptr<SpMVProvider> make_eigen_provider() {
    return std::make_unique<EigenProvider>();
}

#endif // SPIRA_HAS_EIGEN
