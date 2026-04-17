#include "serialise.h"

#include <cstdio>
#include <cstring>
#include <filesystem>

namespace {

constexpr uint32_t MAGIC   = 0x43425053u; // "SPBC" little-endian
constexpr uint32_t VERSION = 1;

template <typename T>
bool write_pod(FILE* f, const T& v) {
    return std::fwrite(&v, sizeof(T), 1, f) == 1;
}

template <typename T>
bool read_pod(FILE* f, T& v) {
    return std::fread(&v, sizeof(T), 1, f) == 1;
}

bool write_edges(FILE* f, const std::vector<std::pair<uint32_t, uint32_t>>& edges) {
    uint64_t n = edges.size();
    if (!write_pod(f, n)) return false;
    if (n == 0) return true;
    return std::fwrite(edges.data(), sizeof(std::pair<uint32_t, uint32_t>),
                       n, f) == n;
}

bool read_edges(FILE* f, std::vector<std::pair<uint32_t, uint32_t>>& edges) {
    uint64_t n = 0;
    if (!read_pod(f, n)) return false;
    edges.resize(n);
    if (n == 0) return true;
    return std::fread(edges.data(), sizeof(std::pair<uint32_t, uint32_t>),
                      n, f) == n;
}

bool write_mutations(FILE* f, const std::vector<Mutation>& muts) {
    uint64_t n = muts.size();
    if (!write_pod(f, n)) return false;
    for (const auto& m : muts) {
        uint8_t flag = m.is_insert ? 1 : 0;
        if (!write_pod(f, m.row)) return false;
        if (!write_pod(f, m.col)) return false;
        if (!write_pod(f, flag)) return false;
    }
    return true;
}

bool read_mutations(FILE* f, std::vector<Mutation>& muts) {
    uint64_t n = 0;
    if (!read_pod(f, n)) return false;
    muts.resize(n);
    for (uint64_t i = 0; i < n; ++i) {
        uint8_t flag = 0;
        if (!read_pod(f, muts[i].row)) return false;
        if (!read_pod(f, muts[i].col)) return false;
        if (!read_pod(f, flag)) return false;
        muts[i].is_insert = (flag != 0);
    }
    return true;
}

bool write_doubles(FILE* f, const std::vector<double>& v) {
    uint64_t n = v.size();
    if (!write_pod(f, n)) return false;
    if (n == 0) return true;
    return std::fwrite(v.data(), sizeof(double), n, f) == n;
}

bool read_doubles(FILE* f, std::vector<double>& v) {
    uint64_t n = 0;
    if (!read_pod(f, n)) return false;
    v.resize(n);
    if (n == 0) return true;
    return std::fread(v.data(), sizeof(double), n, f) == n;
}

} // namespace

std::string cache_file_path(const std::string& cache_dir,
                            int scale, int batch_size) {
    return cache_dir + "/S" + std::to_string(scale) +
           "_B" + std::to_string(batch_size) + ".bin";
}

bool save_bench_data(const std::string& path, const CachedBenchData& data) {
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path());

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;

    bool ok = true;
    ok = ok && write_pod(f, MAGIC);
    ok = ok && write_pod(f, VERSION);
    ok = ok && write_pod(f, data.scale);
    ok = ok && write_pod(f, data.batch_size);
    ok = ok && write_pod(f, data.num_nodes);
    ok = ok && write_edges(f, data.graph.initial_edges);
    ok = ok && write_edges(f, data.graph.insert_pool);

    uint32_t num_phases = static_cast<uint32_t>(data.batches.size());
    ok = ok && write_pod(f, num_phases);

    for (uint32_t p = 0; p < num_phases && ok; ++p) {
        ok = ok && write_mutations(f, data.batches[p].mutations);
        uint64_t nnz64 = data.reference[p].expected_nnz;
        ok = ok && write_pod(f, nnz64);
        ok = ok && write_doubles(f, data.reference[p].expected_rank);
    }

    std::fclose(f);

    if (!ok) {
        std::filesystem::remove(path);
    }
    return ok;
}

bool load_bench_data(const std::string& path,
                     int expected_scale, int expected_batch_size,
                     CachedBenchData& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;

    bool ok = true;
    uint32_t magic = 0, version = 0;
    ok = ok && read_pod(f, magic);
    ok = ok && read_pod(f, version);
    if (!ok || magic != MAGIC || version != VERSION) {
        std::fclose(f);
        return false;
    }

    ok = ok && read_pod(f, out.scale);
    ok = ok && read_pod(f, out.batch_size);
    ok = ok && read_pod(f, out.num_nodes);
    if (!ok || out.scale != expected_scale || out.batch_size != expected_batch_size) {
        std::fclose(f);
        return false;
    }

    out.graph.num_nodes = out.num_nodes;
    ok = ok && read_edges(f, out.graph.initial_edges);
    ok = ok && read_edges(f, out.graph.insert_pool);

    uint32_t num_phases = 0;
    ok = ok && read_pod(f, num_phases);
    out.batches.resize(num_phases);
    out.reference.resize(num_phases);

    for (uint32_t p = 0; p < num_phases && ok; ++p) {
        ok = ok && read_mutations(f, out.batches[p].mutations);
        uint64_t nnz64 = 0;
        ok = ok && read_pod(f, nnz64);
        out.reference[p].expected_nnz = static_cast<size_t>(nnz64);
        ok = ok && read_doubles(f, out.reference[p].expected_rank);
    }

    std::fclose(f);
    return ok;
}
