#include <benchmark/benchmark.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "csr_reference.h"
#include "graph.h"
#include "mutation.h"
#include "pagerank.h"
#include "provider.h"
#include "rmat.h"
#include "serialise.h"
#include "verify.h"

#ifdef SPIRA_HAS_EIGEN
#include "eigen_provider.h"
#endif
#ifdef SPIRA_HAS_MKL
#include "mkl_provider.h"
#endif
#ifdef SPIRA_HAS_ARMADILLO
#include "armadillo_provider.h"
#endif
#ifdef SPIRA_HAS_SPIRA
#include "spira_provider.h"
#endif

// ── Global benchmark configuration ─────────────────────────────────────────

static constexpr int    NUM_PHASES       = 50;
static constexpr int    PAGERANK_ITERS   = 20;
static constexpr double VERIFY_TOLERANCE = 1.0;

// Cache directory for pre-computed benchmark data (graph + mutations + reference).
// Override with SPIRA_BENCH_CACHE_DIR environment variable.
static std::string bench_cache_dir() {
    const char* env = std::getenv("SPIRA_BENCH_CACHE_DIR");
    if (env && *env) return env;
    return "bench_cache";
}

#ifdef SPIRA_HAS_SPIRA
// Thread count for Spira's parallel_matrix — default to hardware concurrency.
static std::size_t g_spira_threads = std::thread::hardware_concurrency();
#endif

// ── Pre-computed benchmark data shared across all providers ─────────────────

struct BenchData {
    int scale;
    int batch_size;
    uint32_t num_nodes;
    GraphData graph;
    std::vector<MutationBatch> batches;
    std::vector<PhaseReference> reference;  // from reference run
    bool initialised = false;
};

// One BenchData per (scale, batch_size) pair.
// We use a simple lookup; in practice there are only ~9 combos.
static std::vector<BenchData> g_bench_data;

static BenchData& get_or_create(int scale, int batch_size) {
    for (auto& bd : g_bench_data) {
        if (bd.scale == scale && bd.batch_size == batch_size) return bd;
    }
    g_bench_data.push_back({scale, batch_size, 0, {}, {}, {}, false});
    return g_bench_data.back();
}

static void ensure_initialised(BenchData& bd) {
    if (bd.initialised) return;

    std::fprintf(stderr, "[init] SCALE=%d  batch_size=%d\n",
                 bd.scale, bd.batch_size);

    bd.num_nodes = 1u << bd.scale;

    // ── Fast path: load from binary cache if present ───────────────────────
    const std::string cache_path = cache_file_path(
        bench_cache_dir(), bd.scale, bd.batch_size);

    {
        CachedBenchData cached;
        if (load_bench_data(cache_path, bd.scale, bd.batch_size, cached)) {
            bd.num_nodes  = cached.num_nodes;
            bd.graph      = std::move(cached.graph);
            bd.batches    = std::move(cached.batches);
            bd.reference  = std::move(cached.reference);
            bd.initialised = true;
            std::fprintf(stderr,
                "[init]   Loaded from cache: %s\n"
                "[init]   Initial edges: %zu   Insert pool: %zu   Phases: %zu\n",
                cache_path.c_str(),
                bd.graph.initial_edges.size(), bd.graph.insert_pool.size(),
                bd.batches.size());
            return;
        }
    }

    std::fprintf(stderr, "[init]   No cache at %s — generating fresh.\n",
                 cache_path.c_str());

    std::mt19937_64 rng(42);  // deterministic seed

    // 1. Generate R-MAT edges.
    std::fprintf(stderr, "[init]   Generating R-MAT edges...\n");
    RmatParams params;
    auto raw_edges = rmat_generate(bd.scale, params, rng);

    // 2. Process: dedup, shuffle, split.
    std::fprintf(stderr, "[init]   Processing edges...\n");
    bd.graph = process_edges(raw_edges, bd.num_nodes, rng);
    std::fprintf(stderr, "[init]   Initial edges: %zu   Insert pool: %zu\n",
                 bd.graph.initial_edges.size(), bd.graph.insert_pool.size());

    // 3. Build initial degree vector (for mutation generation).
    std::vector<uint32_t> out_degree(bd.num_nodes, 0);
    for (const auto& [dst, src] : bd.graph.initial_edges) {
        out_degree[src]++;
    }

    // 4. Build live edge set for mutation generation.
    EdgeSet live_edges;
    live_edges.reserve(bd.graph.initial_edges.size());
    for (const auto& e : bd.graph.initial_edges) {
        live_edges.insert(e);
    }

    // 5. Pre-generate all mutation batches.
    std::fprintf(stderr, "[init]   Generating %d mutation batches (B=%d)...\n",
                 NUM_PHASES, bd.batch_size);
    bd.batches = generate_all_batches(
        NUM_PHASES, bd.batch_size, live_edges,
        bd.graph.insert_pool, out_degree, bd.num_nodes,
        bd.scale, params, rng);

    // 6. Reference run: compute expected nnz + rank vector per phase.
    std::fprintf(stderr, "[init]   Running reference computation...\n");
    auto ref_provider = make_csr_reference();
    ref_provider->bulk_load(bd.graph.initial_edges, bd.num_nodes);

    PageRankState ref_state(bd.num_nodes);
    ref_state.init_degrees(bd.graph.initial_edges);

    bd.reference.resize(NUM_PHASES);
    for (int p = 0; p < NUM_PHASES; ++p) {
        ref_provider->apply_mutations(bd.batches[p]);
        ref_state.update_degrees(bd.batches[p]);
        run_pagerank(*ref_provider, ref_state, PAGERANK_ITERS);

        bd.reference[p].expected_nnz = ref_provider->nnz();
        bd.reference[p].expected_rank = ref_state.rank;
    }

    std::fprintf(stderr, "[init]   Reference run complete. Final nnz=%zu\n",
                 bd.reference.back().expected_nnz);

    // 7. Save to cache for future runs.
    {
        CachedBenchData to_save;
        to_save.scale      = bd.scale;
        to_save.batch_size = bd.batch_size;
        to_save.num_nodes  = bd.num_nodes;
        to_save.graph      = bd.graph;
        to_save.batches    = bd.batches;
        to_save.reference  = bd.reference;
        if (save_bench_data(cache_path, to_save)) {
            std::fprintf(stderr, "[init]   Saved cache: %s\n", cache_path.c_str());
        } else {
            std::fprintf(stderr, "[init]   WARN: failed to write cache file %s\n",
                         cache_path.c_str());
        }
    }

    bd.initialised = true;
}

// ── Benchmark fixture ───────────────────────────────────────────────────────

class PhaseCycleFixture : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) override {
        int scale      = static_cast<int>(state.range(0));
        int batch_size = static_cast<int>(state.range(1));
        int provider_id = static_cast<int>(state.range(2));

        bd_ = &get_or_create(scale, batch_size);
        ensure_initialised(*bd_);

        provider_ = create_provider(provider_id);
        provider_->bulk_load(bd_->graph.initial_edges, bd_->num_nodes);

        pr_state_ = std::make_unique<PageRankState>(bd_->num_nodes);
        pr_state_->init_degrees(bd_->graph.initial_edges);

        phase_ = 0;
    }

    void TearDown(const benchmark::State&) override {
        // Post-benchmark correctness check.
        if (phase_ > 0) {
            int last = phase_ - 1;
            bool ok = verify_phase(provider_->nnz(), pr_state_->rank,
                                   bd_->reference[last], VERIFY_TOLERANCE);
            if (!ok) {
                std::fprintf(stderr, "[FAIL] %s correctness check failed at phase %d\n",
                             provider_->name().c_str(), last);
            } else {
                std::fprintf(stderr, "[OK]   %s passed correctness check (phase %d)\n",
                             provider_->name().c_str(), last);
            }
        }
    }

    static std::unique_ptr<SpMVProvider> create_provider(int id) {
        switch (id) {
            case 0: return make_csr_reference();
#ifdef SPIRA_HAS_EIGEN
            case 1: return make_eigen_provider();
#endif
#ifdef SPIRA_HAS_MKL
            case 2: return make_mkl_provider();
#endif
#ifdef SPIRA_HAS_ARMADILLO
            case 3: return make_armadillo_provider();
#endif
#ifdef SPIRA_HAS_SPIRA
            case 4: return make_spira_provider(g_spira_threads);
#endif
            default:
                std::fprintf(stderr, "Unknown provider ID %d\n", id);
                return make_csr_reference();
        }
    }

    BenchData* bd_ = nullptr;
    std::unique_ptr<SpMVProvider> provider_;
    std::unique_ptr<PageRankState> pr_state_;
    int phase_ = 0;
};

BENCHMARK_DEFINE_F(PhaseCycleFixture, Run)(benchmark::State& state) {
    for (auto _ : state) {
        if (phase_ >= NUM_PHASES) {
            state.SkipWithError("Exceeded pre-generated phases");
            break;
        }

        provider_->apply_mutations(bd_->batches[phase_]);
        pr_state_->update_degrees(bd_->batches[phase_]);
        run_pagerank(*provider_, *pr_state_, PAGERANK_ITERS);

        ++phase_;
    }
}

// ── Register benchmarks ─────────────────────────────────────────────────────

// Provider IDs:  0 = CSR-Reference, 1 = Eigen, 2 = MKL, 3 = Armadillo, 4 = Spira
// Args:          (SCALE, batch_size, provider_id)

static void register_benchmarks() {
    std::vector<int> scales = {20};  // Start with SCALE-20; add 22, 24 as needed.
    std::vector<int> batch_sizes = {1000, 10000, 100000};
    // CSR-Reference (ID 0) is kept for the correctness reference run but
    // not registered as a benchmarked provider.
    std::vector<int> providers;

#ifdef SPIRA_HAS_EIGEN
    providers.push_back(1);
#endif
#ifdef SPIRA_HAS_MKL
    providers.push_back(2);
#endif
#ifdef SPIRA_HAS_ARMADILLO
    providers.push_back(3);
#endif
#ifdef SPIRA_HAS_SPIRA
    providers.push_back(4);
#endif

    auto get_provider_name = [](int id) -> std::string {
        switch (id) {
            case 0: return "CSR_Reference";
            case 1: return "Eigen";
            case 2: return "MKL";
            case 3: return "Armadillo";
            case 4: return "Spira";
            default: return "Unknown";
        }
    };

    for (int scale : scales) {
        for (int bs : batch_sizes) {
            for (int pid : providers) {
                std::string bname = "PhaseCycle/S" + std::to_string(scale)
                                  + "/B" + std::to_string(bs)
                                  + "/" + get_provider_name(pid);

                benchmark::RegisterBenchmark(bname.c_str(),
                    [scale, bs, pid](benchmark::State& state) {
                        // Wrap in fixture manually since RegisterBenchmark
                        // doesn't support fixtures directly.
                        // We use the fixture's logic inline.
                        auto& bd = get_or_create(scale, bs);
                        ensure_initialised(bd);

                        auto provider = PhaseCycleFixture::create_provider(pid);
                        provider->bulk_load(bd.graph.initial_edges, bd.num_nodes);

                        PageRankState pr_state(bd.num_nodes);
                        pr_state.init_degrees(bd.graph.initial_edges);

                        int phase = 0;
                        for (auto _ : state) {
                            if (phase >= NUM_PHASES) {
                                state.SkipWithError("Exceeded pre-generated phases");
                                break;
                            }
                            provider->apply_mutations(bd.batches[phase]);
                            pr_state.update_degrees(bd.batches[phase]);
                            run_pagerank(*provider, pr_state, PAGERANK_ITERS);
                            ++phase;
                        }

                        // Correctness check.
                        if (phase > 0) {
                            int last = phase - 1;
                            bool ok = verify_phase(provider->nnz(), pr_state.rank,
                                                   bd.reference[last], VERIFY_TOLERANCE);
                            if (!ok) {
                                std::fprintf(stderr, "[FAIL] %s S%d B%d phase %d\n",
                                             provider->name().c_str(), scale, bs, last);
                            }
                        }
                    })
                    ->Unit(benchmark::kMillisecond)
                    ->Iterations(NUM_PHASES)
                    ->MeasureProcessCPUTime();
            }
        }
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    register_benchmarks();
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
