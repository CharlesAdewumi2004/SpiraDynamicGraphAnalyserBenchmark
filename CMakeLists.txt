cmake_minimum_required(VERSION 3.22)
project(SpiraBenchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Options ──────────────────────────────────────────────────────────────────
option(ENABLE_SPIRA     "Build with Spira provider (stage4)"  ON)
option(ENABLE_EIGEN     "Build with Eigen provider"           OFF)
option(ENABLE_MKL       "Build with Intel MKL provider"       OFF)
option(ENABLE_ARMADILLO "Build with Armadillo provider"       OFF)

# ── Compiler flags ───────────────────────────────────────────────────────────
add_compile_options(-O3 -march=native -Wall -Wextra -Wpedantic)

find_package(Threads REQUIRED)

# ── OpenMP (used by CSR-Reference SpMV, PageRank vector ops, and as the
#    native threading backend for Eigen / MKL / Armadillo) ────────────────────
find_package(OpenMP REQUIRED)

# ── FetchContent dependencies ────────────────────────────────────────────────
include(FetchContent)

# Google Benchmark
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.1
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# ── Spira (stage4/MutilThreaded — parallel_matrix with thread pool + SIMD) ──
if(ENABLE_SPIRA)
    # Disable Spira's own tests and benchmarks when consumed as a dependency.
    set(SPIRA_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
    set(SPIRA_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(SPIRA_BUILD_PARALLEL   ON  CACHE BOOL "" FORCE)
    set(SPIRA_ENABLE_SIMD      ON  CACHE BOOL "" FORCE)

    FetchContent_Declare(
        spira
        GIT_REPOSITORY https://github.com/CharlesAdewumi2004/Spira.git
        GIT_TAG        stage4/MutilThreaded
    )
    FetchContent_MakeAvailable(spira)
endif()

# ── Core library (everything except main) ────────────────────────────────────
set(CORE_SOURCES
    src/rmat.cpp
    src/graph.cpp
    src/mutation.cpp
    src/pagerank.cpp
    src/verify.cpp
    src/serialise.cpp
    src/providers/csr_reference.cpp
)

if(ENABLE_SPIRA)
    list(APPEND CORE_SOURCES src/providers/spira_provider.cpp)
endif()

if(ENABLE_EIGEN)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
    list(APPEND CORE_SOURCES src/providers/eigen_provider.cpp)
endif()

if(ENABLE_MKL)
    # oneAPI MKL provides MKLConfig.cmake — requires `source /opt/intel/oneapi/setvars.sh`
    # before running cmake so that CMAKE_PREFIX_PATH includes the MKL directory.
    find_package(MKL CONFIG REQUIRED)
    list(APPEND CORE_SOURCES src/providers/mkl_provider.cpp)
endif()

if(ENABLE_ARMADILLO)
    find_package(Armadillo REQUIRED)
    list(APPEND CORE_SOURCES src/providers/armadillo_provider.cpp)
endif()

add_library(spira_core STATIC ${CORE_SOURCES})
target_include_directories(spira_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(spira_core PUBLIC Threads::Threads OpenMP::OpenMP_CXX)

if(ENABLE_SPIRA)
    target_compile_definitions(spira_core PUBLIC SPIRA_HAS_SPIRA)
    target_link_libraries(spira_core PUBLIC spira)
    # Spira provider needs C++23; ensure it propagates.
    target_compile_features(spira_core PUBLIC cxx_std_23)
endif()

if(ENABLE_EIGEN)
    target_compile_definitions(spira_core PUBLIC SPIRA_HAS_EIGEN)
    target_link_libraries(spira_core PUBLIC Eigen3::Eigen)
endif()

if(ENABLE_MKL)
    target_compile_definitions(spira_core PUBLIC SPIRA_HAS_MKL)
    target_link_libraries(spira_core PUBLIC MKL::MKL)
endif()

if(ENABLE_ARMADILLO)
    target_compile_definitions(spira_core PUBLIC SPIRA_HAS_ARMADILLO)
    target_include_directories(spira_core PUBLIC ${ARMADILLO_INCLUDE_DIRS})
    target_link_libraries(spira_core PUBLIC ${ARMADILLO_LIBRARIES})
endif()

# ── Main benchmark executable ────────────────────────────────────────────────
add_executable(spira-bench src/main.cpp)
target_link_libraries(spira-bench PRIVATE spira_core benchmark::benchmark)
