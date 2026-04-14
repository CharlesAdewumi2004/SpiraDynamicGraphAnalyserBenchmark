#ifdef SPIRA_HAS_ARMADILLO

#include "provider.h"
#include "armadillo_provider.h"

#include <armadillo>
#include <algorithm>

// Armadillo's sp_mat uses CSC internally.  Its SpMV is delegated to whatever
// BLAS backend it's linked against (OpenBLAS, MKL, or its own fallback),
// which will use threading natively.
//
// Rebuild strategy (§4.3 Option A): construct from locations + values matrices
// each phase.  This is the batch-construction path Armadillo recommends and
// is comparable to Eigen's setFromTriplets.

class ArmadilloProvider : public SpMVProvider {
public:
    std::string name() const override { return "Armadillo"; }

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
        // Wrap raw pointers as non-owning Armadillo vectors (no copy).
        arma::vec xv(const_cast<double*>(x), n, /*copy_aux_mem=*/false,
                     /*strict=*/true);
        arma::vec yv(y, n, /*copy_aux_mem=*/false, /*strict=*/true);
        // Armadillo delegates to BLAS which uses its own threading.
        yv = mat_ * xv;
    }

    size_t nnz() const override { return edge_set_.size(); }

private:
    void rebuild() {
        // Build from locations (2×nnz umat) and values (vec of 1.0s).
        // Armadillo sorts the locations internally to build CSC.
        size_t nnz_count = edge_set_.size();
        arma::umat locations(2, nnz_count);
        arma::vec values(nnz_count, arma::fill::ones);

        size_t k = 0;
        for (const auto& [dst, src] : edge_set_) {
            locations(0, k) = dst;
            locations(1, k) = src;
            ++k;
        }

        mat_ = arma::sp_mat(/*sort_locations=*/true, locations, values,
                             n_, n_);
    }

    uint32_t n_ = 0;
    arma::sp_mat mat_;
    EdgeSet edge_set_;
};

std::unique_ptr<SpMVProvider> make_armadillo_provider() {
    return std::make_unique<ArmadilloProvider>();
}

#endif // SPIRA_HAS_ARMADILLO
