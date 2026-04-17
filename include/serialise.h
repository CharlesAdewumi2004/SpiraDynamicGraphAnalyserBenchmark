#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "graph.h"
#include "mutation.h"
#include "verify.h"

// Binary cache for pre-generated benchmark data (graph + mutations + reference).
//
// The full init pipeline (R-MAT generation, mutation batches with hub-biased
// deletions, and the 50-phase CSR reference run) is deterministic under a
// fixed RNG seed but can take hours for large batch sizes.  Caching lets us
// pay that cost once and reuse it across every benchmark invocation.
//
// File format (little-endian, no compression):
//   magic:     4 bytes "SPBC"
//   version:   uint32
//   scale:     int32
//   batch_size:int32
//   num_nodes: uint32
//   num_initial_edges:  uint64
//   initial_edges[]:    (uint32, uint32)
//   num_insert_pool:    uint64
//   insert_pool[]:      (uint32, uint32)
//   num_phases:         uint32
//   for each phase:
//     num_mutations:    uint64
//     mutations[]:      (uint32, uint32, uint8)   // is_insert as 0/1
//     expected_nnz:     uint64
//     rank_size:        uint64
//     rank[]:           double

struct CachedBenchData {
    int scale;
    int batch_size;
    uint32_t num_nodes;
    GraphData graph;
    std::vector<MutationBatch> batches;
    std::vector<PhaseReference> reference;
};

// Compute cache file path for a given (scale, batch_size) combination.
// The path lives under cache_dir/S{scale}_B{batch_size}.bin.
std::string cache_file_path(const std::string& cache_dir,
                            int scale, int batch_size);

// Save benchmark data to a binary file.  Returns true on success.
bool save_bench_data(const std::string& path, const CachedBenchData& data);

// Load benchmark data from a binary file.  Returns true on success.
// Validates magic, version, scale, and batch_size fields against the
// expected_* arguments and returns false if they don't match.
bool load_bench_data(const std::string& path,
                     int expected_scale, int expected_batch_size,
                     CachedBenchData& out);
