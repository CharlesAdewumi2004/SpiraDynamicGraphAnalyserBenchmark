# Spira Dynamic Graph Analyser Benchmark — Design Document

## 1. Purpose

This benchmark measures the end-to-end performance of sparse matrix libraries under a **streaming graph analytics** workload: repeated cycles of batch edge mutations followed by PageRank computation via sparse matrix-vector multiply (SpMV). The goal is to evaluate how well each library handles the full update-then-compute pipeline on realistic power-law graphs, not just isolated SpMV throughput.

### Libraries Under Test

| ID | Library | Version / Branch | Role |
|----|---------|-----------------|------|
| 0 | CSR-Reference | Internal | Correctness oracle only (not timed) |
| 1 | Eigen | 3.x (system) | Widely-used C++ sparse matrix library |
| 2 | Intel MKL | oneAPI | Industry-standard HPC sparse BLAS |
| 3 | Armadillo | System + OpenBLAS | High-level C++ linear algebra library |
| 4 | Spira | stage4/MutilThreaded | Mutable sparse matrix with native thread pool + SIMD |

---

## 2. Workload Design

### 2.1 Graph Generation (R-MAT)

Graphs are generated using the **Recursive MATrix (R-MAT)** model with Graph500 parameters:

- **a = 0.57, b = 0.19, c = 0.19, d = 0.05** (skewed toward top-left quadrant)
- **Edge factor = 16** (target edges = 16 * 2^SCALE)
- **SCALE = 20** by default (1,048,576 nodes, ~13M unique edges after dedup)

R-MAT produces power-law degree distributions that are representative of real-world graphs (social networks, web graphs, citation networks). The skewed parameters ensure a small number of hub nodes with very high degree, which stresses both the mutation pipeline (hub-biased deletions) and the SpMV kernel (load imbalance across rows).

**Post-processing:**
1. Self-loops removed
2. Duplicate edges removed via hash set
3. Fisher-Yates shuffle (deterministic seed = 42)
4. 80/20 split: 80% initial graph, 20% reserved as insertion pool

### 2.2 Mutation Batches

Each benchmark runs **50 phases**. Before each phase, a pre-generated mutation batch is applied:

- **B insertions** per batch (B = 1000, 10000, or 100000)
- **0.3B deletions** per batch
- Net growth of 0.7B edges per phase

**Insertion sources:**
1. Draw from the 20% reserved pool (maintains R-MAT distribution)
2. When pool exhausted, generate fresh R-MAT edges on-the-fly
3. Deduplicated against current live edge set

**Deletion strategy (50/50 split):**
- 50% **uniform random**: pick any live edge uniformly
- 50% **hub-biased**: pick source node proportional to out-degree, then pick one of its edges uniformly

This mixed deletion strategy ensures both random and structurally-targeted mutations, reflecting real workloads where high-degree nodes are more likely to have connections churned.

### 2.3 PageRank Computation

Each phase runs **20 fixed iterations** of PageRank (no convergence check):

```
r(t+1) = (1-d)/N + d * A * D^{-1} * r(t) + d * dangling_sum/N
```

Where:
- **d = 0.85** (damping factor)
- **A** = adjacency matrix (binary, unweighted)
- **D** = diagonal out-degree matrix
- **dangling_sum** = sum of rank for nodes with out-degree 0

Fixed iteration count ensures deterministic, comparable compute work across all providers. The rank vector is carried forward across phases (not reset), so the computation reflects a continuously-evolving graph state.

---

## 3. Provider Implementations

Each library implements the `SpMVProvider` interface:

```cpp
class SpMVProvider {
    virtual void bulk_load(edges, num_nodes) = 0;  // untimed
    virtual void apply_mutations(batch) = 0;        // timed
    virtual void spmv(x, y, n) = 0;                 // timed
    virtual void spmv_vec(x, y, n);                  // zero-copy override
    virtual size_t nnz() = 0;
};
```

### 3.1 Eigen

**Strategy:** Full rebuild via `setFromTriplets` each phase (Option A from Eigen documentation).

**Rationale:** `setFromTriplets` is Eigen's documented best practice for batch construction of sparse matrices. It uses an internal O(nnz + N) algorithm to build compressed column storage. Incremental updates via `coeffRef` have pathological O(nnz) per-insert cost due to element shifting in compressed storage.

**Implementation details:**
- Persistent `std::vector<Eigen::Triplet<double>>` avoids reallocation across phases
- Persistent `EdgeSet` tracks live edges; iterated to rebuild triplets each phase
- SpMV via `yv.noalias() = mat_ * xv` with `Eigen::Map<>` zero-copy wrappers
- Threading: Eigen parallelises SpMV internally via OpenMP (`-fopenmp`)

### 3.2 Intel MKL

**Strategy:** Manual CSR construction via counting-sort scatter + inspector-executor optimisation.

**Rationale:** MKL's inspector-executor model (`mkl_sparse_set_mv_hint` + `mkl_sparse_optimize`) allows MKL to build optimised internal structures (e.g. BSR tiling, NUMA-aware partitioning) when it knows the SpMV will be called repeatedly. We hint 20 calls per matrix lifetime (matching PageRank iterations).

**Implementation details:**
- CSR built via counting-sort scatter: O(nnz + N), no temporary COO allocation
- Persistent CSR arrays (`row_ptr_`, `col_idx_`, `values_`) avoid reallocation
- 4-array CSR creation (`rows_start`/`rows_end` split) as required by `mkl_sparse_d_create_csr`
- SpMV via `mkl_sparse_d_mv` with raw pointers (zero-copy)
- Threading: MKL manages its own OpenMP thread pool internally

### 3.3 Armadillo

**Strategy:** Rebuild `sp_mat` from sorted locations + values each phase.

**Rationale:** Armadillo's `sp_mat` uses CSC internally. The batch construction path via `arma::umat` locations + `arma::vec` values with `sort_locations=true` is the documented approach for building sparse matrices from edge data.

**Implementation details:**
- `arma::umat locations(2, nnz)` and `arma::vec values(nnz, fill::ones)` allocated each rebuild — this is the natural Armadillo API usage pattern
- SpMV via `yv = mat_ * xv` with non-owning `arma::vec` wrappers (zero-copy, `copy_aux_mem=false`)
- Threading: Armadillo delegates SpMV to its BLAS backend (OpenBLAS), which manages its own thread pool

### 3.4 Spira

**Strategy:** Native incremental mutations via `open()` / `lock()` cycling.

**Rationale:** Spira's `parallel_matrix` is designed specifically for mutable sparse matrices. Instead of rebuilding from scratch, it supports incremental mutations:
1. `open()` — O(1) transition to mutable mode (row buffers preserved via `compact_preserve`)
2. Insert/delete mutations via `insert(row, col, val)` — staged inserts accumulate in L1/L2-friendly buffers
3. `lock()` — compress to CSR, zero-filter removes deletions (inserted as value 0.0)

**Implementation details:**
- Template parameters: SOA layout, `uint32_t` indices, `double` values, 64-element array buffers, `compact_preserve` lock policy, `staged` insert policy with 256-element staging buffers
- Bulk load via `parallel_fill` — each thread writes directly into its own row partition
- SpMV via `spmv_vec()` override — zero-copy, reads/writes `std::vector<double>&` directly. Spira's SIMD kernels (AVX2/AVX-512 `sparse_dot_double`) operate on the vectors without any pointer-to-vector translation
- Threading: Spira's own thread pool (configured to `hardware_concurrency()` threads)

---

## 4. Fairness Measures

### 4.1 Identical Workload

All providers process the **exact same** pre-generated data:
- Same R-MAT graph (deterministic seed = 42)
- Same 80/20 edge split
- Same 50 mutation batches in the same order
- Same 20 PageRank iterations with the same damping factor
- Same initial rank vector (uniform 1/N)

Data is generated once during initialisation (untimed) and shared across all provider runs for each (SCALE, batch_size) configuration.

### 4.2 What Is Timed

Each timed iteration measures one complete phase cycle:

```
apply_mutations(batch[phase])  →  update_degrees(batch[phase])  →  run_pagerank(provider, state, 20)
```

This captures the **full update-then-compute pipeline**, which is the real-world workload. Libraries that can perform incremental mutations (Spira) benefit naturally. Libraries that must rebuild from scratch (Eigen, MKL, Armadillo) pay the rebuild cost as part of the timed region. This is intentional — rebuild cost is a real cost that users pay.

### 4.3 What Is NOT Timed

- Graph generation (R-MAT)
- Edge deduplication, shuffle, and split
- Mutation batch pre-generation
- CSR-Reference correctness run
- Initial `bulk_load()` for each provider
- PageRankState initialisation

### 4.4 Zero-Copy SpMV

The `SpMVProvider` interface provides two SpMV entry points:

```cpp
virtual void spmv(const double* x, double* y, uint32_t n) const = 0;
virtual void spmv_vec(std::vector<double>& x, std::vector<double>& y, uint32_t n) const;
```

The `spmv_vec` default delegates to the raw-pointer `spmv`. This design ensures:

- **Eigen**: uses `Eigen::Map<>` — zero-copy wrapping of raw pointers as Eigen vectors
- **MKL**: takes `const double*` and `double*` directly — zero overhead
- **Armadillo**: uses `arma::vec(ptr, n, copy_aux_mem=false)` — non-owning wrapper
- **Spira**: overrides `spmv_vec` — passes `std::vector<double>&` directly to its SIMD kernel with no memcpy

No provider pays a data-copy tax that isn't intrinsic to its API design.

### 4.5 Threading Parity

All providers use multi-threading, but each uses its **native** threading mechanism:

| Provider | Threading Model | Thread Count |
|----------|----------------|-------------|
| Eigen | OpenMP (via `-fopenmp`) | OMP_NUM_THREADS (default: all cores) |
| MKL | Internal OpenMP runtime | MKL_NUM_THREADS (default: all cores) |
| Armadillo | OpenBLAS thread pool | OPENBLAS_NUM_THREADS (default: all cores) |
| Spira | Own thread pool | `hardware_concurrency()` |

The benchmark script pins execution to physical cores via `taskset` to prevent thread migration and ensure all providers see the same core count.

PageRank vector operations (degree scaling, dangling sum reduction, damping application) are parallelised with OpenMP `#pragma omp parallel for` and run identically for all providers.

### 4.6 Compiler and Optimisation Parity

All code is compiled with:
```
-O3 -march=native -fopenmp
```

`-march=native` enables AVX2/AVX-512 instruction generation for all providers equally. Spira's SIMD kernels, Eigen's vectorised operations, MKL's optimised routines, and OpenBLAS's BLAS kernels all benefit from native ISA targeting.

C++ standard: C++23 for all providers (required by Spira's template metaprogramming).

### 4.7 Memory Allocation Discipline

Hot-path allocations are minimised across all providers:

- **PageRank**: Persistent `scratch_x` and `scratch_y` vectors in `PageRankState` — no per-iteration allocation
- **Eigen**: Persistent `triplets_` vector — `clear()` + `reserve()` reuses capacity
- **MKL**: Persistent CSR arrays (`row_ptr_`, `col_idx_`, `values_`, `write_pos_`) — `resize()` reuses capacity when nnz doesn't shrink
- **Armadillo**: Fresh `arma::umat` and `arma::vec` per rebuild — this is unavoidable with Armadillo's batch construction API and represents realistic usage
- **Spira**: No allocation in timed path — `open()` preserves row buffers, `lock()` compresses in-place

### 4.8 Rebuild Strategy: Each Library's Best Practice

Each provider uses the **documented recommended approach** for its library:

| Provider | Rebuild Method | Complexity | Source |
|----------|---------------|-----------|--------|
| Eigen | `setFromTriplets` | O(nnz + N) | Eigen documentation: "the recommended way" |
| MKL | Counting-sort scatter + inspector-executor | O(nnz + N) + optimise | MKL developer reference |
| Armadillo | `sp_mat(sort_locations, locs, vals, n, n)` | O(nnz log nnz) | Armadillo documentation |
| Spira | `open()` / mutations / `lock()` | O(mutations + nnz) | Native incremental path |

No provider is handicapped by a suboptimal algorithm that a competent developer would avoid.

### 4.9 Hardware Isolation (Benchmark Script)

The `run_benchmarks.sh` script applies system-level isolation:

1. **CPU governor** set to `performance` (no frequency scaling)
2. **Turbo boost disabled** (eliminates thermal-dependent frequency variation)
3. **ASLR disabled** (deterministic memory layout)
4. **Filesystem caches dropped** before benchmark start
5. **Core pinning** via `taskset -c 0-7` (physical cores only, no hyperthreading interference)
6. **Sequential execution** — one benchmark at a time, no contention

### 4.10 Correctness Verification

After each provider's timed run, correctness is verified against the CSR-Reference oracle:

1. **Structural check**: `provider.nnz()` must match reference NNZ exactly
2. **Numerical check**: L1 norm of `|rank_provider - rank_reference|` must be < 1e-10

Any provider that fails verification has its results flagged. This ensures timing results are only meaningful for correct implementations — a fast but wrong SpMV is caught immediately.

---

## 5. Benchmark Configuration

### 5.1 Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| SCALE | 20 | 1M nodes, ~13M edges — fits in L3 cache boundary, representative of medium-scale graphs |
| Batch sizes | 1000, 10000, 100000 | Spans 2 orders of magnitude: small incremental updates to large batch mutations |
| Phases | 50 | Enough iterations for Google Benchmark to compute stable statistics |
| PageRank iterations | 20 | Fixed count, no convergence — ensures identical compute work |
| Damping factor | 0.85 | Standard PageRank parameter |
| RNG seed | 42 | Deterministic reproducibility |
| Verification tolerance | 1e-10 | Catches floating-point divergence from incorrect SpMV |

### 5.2 Measurement

- **Timer**: Google Benchmark with `MeasureProcessCPUTime()` — measures CPU time, not wall time, to eliminate I/O and scheduling noise
- **Units**: Milliseconds per phase cycle
- **Iterations**: 50 (one per pre-generated phase)
- **Output**: JSON format, one file per benchmark configuration

### 5.3 What the Results Show

Each benchmark entry reports the **total time for one phase cycle** (mutations + PageRank). This captures:

- **Mutation overhead**: EdgeSet updates + library-specific rebuild (Eigen/MKL/Armadillo) or incremental update (Spira)
- **SpMV compute**: 20 iterations of matrix-vector multiply
- **PageRank overhead**: Degree scaling, dangling sum reduction, damping (identical across providers)

Libraries that are fast at SpMV but slow at rebuilding will show competitive times at small batch sizes (rebuild cost amortised over 20 SpMV iterations) but diverge at large batch sizes. Libraries with efficient incremental updates (Spira) should show more stable scaling.

---

## 6. File Structure

```
SpiraDynamicGraphAnalyserBenchmark/
  CMakeLists.txt              — Build system with optional provider flags
  include/
    provider.h                — SpMVProvider abstract interface
    pagerank.h                — PageRankState + run_pagerank declaration
    graph.h                   — EdgeSet, GraphData, process_edges
    mutation.h                — Mutation, MutationBatch, generate_all_batches
    rmat.h                    — RmatParams, rmat_generate, rmat_single_edge
    verify.h                  — PhaseReference, verify_phase, l1_norm_diff
    csr_reference.h           — make_csr_reference factory
    eigen_provider.h          — make_eigen_provider factory
    mkl_provider.h            — make_mkl_provider factory
    armadillo_provider.h      — make_armadillo_provider factory
    spira_provider.h          — make_spira_provider factory
  src/
    main.cpp                  — Google Benchmark harness + registration
    pagerank.cpp              — OpenMP PageRank implementation
    rmat.cpp                  — R-MAT graph generator
    graph.cpp                 — Edge processing (dedup, shuffle, split)
    mutation.cpp              — Mutation batch pre-generation
    verify.cpp                — Correctness verification
    providers/
      csr_reference.cpp       — Internal correctness oracle
      eigen_provider.cpp      — Eigen setFromTriplets provider
      mkl_provider.cpp        — MKL inspector-executor provider
      armadillo_provider.cpp  — Armadillo locations+values provider
      spira_provider.cpp      — Spira parallel_matrix provider
  scripts/
    run_benchmarks.sh         — Self-contained AWS benchmark runner (auto-tmux)
    analyse.py                — Result analysis + plot generation
  docs/
    benchmark_design.md       — This document
```

---

## 7. Running the Benchmark

```bash
# Clone and run (on a fresh AWS instance):
git clone <repo-url> && cd SpiraDynamicGraphAnalyserBenchmark
./scripts/run_benchmarks.sh

# The script:
# 1. Auto-launches inside tmux (session "bench")
# 2. Installs all system dependencies
# 3. Applies hardware isolation
# 4. Detects available libraries and builds
# 5. Runs all benchmarks sequentially with core pinning
# 6. Merges results and generates plots
#
# Detach: Ctrl-B D
# Reattach: tmux attach -t bench
```

Results are saved to `results/<timestamp>/`:
- `all_results.json` — merged benchmark data
- `system_info.txt` — hardware and compiler info
- `stderr.log` — init output and correctness check results
- `plots/` — PNG charts
- `PhaseCycle_*.json` — per-run raw data
