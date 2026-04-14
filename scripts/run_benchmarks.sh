#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_benchmarks.sh — Fully self-contained benchmark runner.
#
# Designed for a fresh AWS r8i.4xlarge (Ubuntu 22.04/24.04). Does everything:
#   1. Installs system dependencies (cmake, g++, eigen, armadillo, python, etc.)
#   2. Applies hardware isolation (perf governor, turbo off, core pinning)
#   3. Builds with every available library enabled
#   4. Runs each (SCALE, batch_size, provider) benchmark sequentially
#   5. Merges JSON results and generates plots
#
# Usage:
#   # On a fresh instance — clone the repo, then:
#   tmux new-session -s bench './scripts/run_benchmarks.sh'
#
#   # Detach: Ctrl-B D     Reattach: tmux attach -t bench
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$ROOT_DIR/results/$TIMESTAMP"
BUILD_DIR="$ROOT_DIR/build"
BENCH_BIN="$BUILD_DIR/spira-bench"

# Physical cores to pin to. r8i.4xlarge has 8 physical / 16 logical.
# Adjust if your instance is different.
TASKSET_CORES="0-7"

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${CYAN}[bench]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
ok()   { echo -e "${GREEN}[ ok ]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }
banner() {
    echo ""
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  $*${NC}"
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo ""
}

# ═════════════════════════════════════════════════════════════════════════════
# 1. INSTALL DEPENDENCIES
# ═════════════════════════════════════════════════════════════════════════════
install_deps() {
    banner "Installing Dependencies"

    # Core build tools.
    local pkgs=(
        build-essential
        cmake
        git
        g++-13
        libomp-dev          # OpenMP runtime
        linux-tools-common  # perf, cpufreq-set
        cpufrequtils
        util-linux          # taskset
    )

    # Eigen (header-only, no linking needed beyond cmake find_package).
    pkgs+=( libeigen3-dev )

    # Armadillo + a fast BLAS backend.
    pkgs+=( libarmadillo-dev libopenblas-dev )

    # Python + plotting.
    pkgs+=( python3 python3-pip python3-venv )

    log "Updating apt..."
    sudo apt-get update -qq

    log "Installing packages: ${pkgs[*]}"
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${pkgs[@]}"

    # Ensure a C++23-capable compiler is the default.
    if command -v g++-13 &>/dev/null; then
        sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 2>/dev/null || true
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 2>/dev/null || true
    fi

    # Python packages for analysis.
    log "Installing Python analysis packages..."
    pip3 install --quiet --break-system-packages matplotlib numpy 2>/dev/null \
        || pip3 install --quiet matplotlib numpy 2>/dev/null \
        || warn "pip install failed — plots may not generate"

    ok "All dependencies installed"
}

# ═════════════════════════════════════════════════════════════════════════════
# 2. HARDWARE ISOLATION
# ═════════════════════════════════════════════════════════════════════════════
apply_hw_isolation() {
    banner "Applying Hardware Isolation"

    # CPU performance governor.
    if command -v cpufreq-set &>/dev/null; then
        for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
            local id
            id=$(basename "$cpu" | tr -dc '0-9')
            sudo cpufreq-set -c "$id" -g performance 2>/dev/null || true
        done
        ok "CPU governor → performance"
    elif [ -d /sys/devices/system/cpu/cpu0/cpufreq ]; then
        for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            echo performance | sudo tee "$gov" >/dev/null 2>&1 || true
        done
        ok "CPU governor → performance (sysfs)"
    else
        warn "Could not set CPU governor"
    fi

    # Disable turbo boost.
    if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
        echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null 2>&1 || true
        ok "Turbo boost → disabled (intel_pstate)"
    elif [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
        echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost >/dev/null 2>&1 || true
        ok "Turbo boost → disabled (cpufreq)"
    else
        warn "Turbo boost control not found"
    fi

    # Disable ASLR.
    if [ -f /proc/sys/kernel/randomize_va_space ]; then
        echo 0 | sudo tee /proc/sys/kernel/randomize_va_space >/dev/null 2>&1 || true
        ok "ASLR → disabled"
    fi

    # Drop filesystem caches.
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
    ok "Filesystem caches → dropped"
}

# ═════════════════════════════════════════════════════════════════════════════
# 3. DETECT LIBRARIES AND BUILD
# ═════════════════════════════════════════════════════════════════════════════
build_benchmark() {
    banner "Building Benchmark"

    # Auto-detect which libraries are available.
    local cmake_flags=(
        -DCMAKE_BUILD_TYPE=Release
        -DENABLE_SPIRA=ON
    )

    # Eigen
    if dpkg -s libeigen3-dev &>/dev/null 2>&1; then
        cmake_flags+=( -DENABLE_EIGEN=ON )
        ok "Eigen3 → detected, enabling"
    else
        cmake_flags+=( -DENABLE_EIGEN=OFF )
        warn "Eigen3 not found, skipping"
    fi

    # Armadillo
    if dpkg -s libarmadillo-dev &>/dev/null 2>&1; then
        cmake_flags+=( -DENABLE_ARMADILLO=ON )
        ok "Armadillo → detected, enabling"
    else
        cmake_flags+=( -DENABLE_ARMADILLO=OFF )
        warn "Armadillo not found, skipping"
    fi

    # MKL (optional — not in apt, needs Intel repo)
    if pkg-config --exists mkl-dynamic-lp64-seq 2>/dev/null || [ -d /opt/intel/oneapi/mkl ]; then
        cmake_flags+=( -DENABLE_MKL=ON )
        ok "Intel MKL → detected, enabling"
    else
        cmake_flags+=( -DENABLE_MKL=OFF )
        warn "Intel MKL not found, skipping (install via Intel oneAPI for MKL support)"
    fi

    log "CMake flags: ${cmake_flags[*]}"

    # Clean build to ensure flags take effect.
    rm -rf "$BUILD_DIR"

    log "Configuring..."
    cmake -B "$BUILD_DIR" -S "$ROOT_DIR" "${cmake_flags[@]}" 2>&1 | tail -10

    log "Compiling ($(nproc) threads)..."
    cmake --build "$BUILD_DIR" -j"$(nproc)" 2>&1 | tail -5

    if [ ! -x "$BENCH_BIN" ]; then
        fail "Build failed — $BENCH_BIN not found"
    fi
    ok "Build complete: $BENCH_BIN"

    # Show which providers are compiled in.
    log "Registered benchmarks:"
    "$BENCH_BIN" --benchmark_list_tests 2>/dev/null | sed 's/^/  /'
}

# ═════════════════════════════════════════════════════════════════════════════
# 4. SAVE SYSTEM INFO
# ═════════════════════════════════════════════════════════════════════════════
save_system_info() {
    {
        echo "══════════════════════════════════════"
        echo "  Benchmark System Report"
        echo "══════════════════════════════════════"
        echo ""
        echo "Date:          $(date -Iseconds)"
        echo "Hostname:      $(hostname)"
        echo "Instance:      ${INSTANCE_TYPE:-unknown}"
        echo "Kernel:        $(uname -r)"
        echo "CPU:           $(lscpu | grep 'Model name' | sed 's/.*: *//')"
        echo "Physical cores: $(lscpu | grep '^Core(s) per socket' | awk '{print $NF}')"
        echo "Sockets:       $(lscpu | grep '^Socket(s)' | awk '{print $NF}')"
        echo "Threads/core:  $(lscpu | grep '^Thread(s) per core' | awk '{print $NF}')"
        echo "Total logical: $(nproc)"
        echo "RAM:           $(free -h | awk '/Mem:/{print $2}')"
        echo "L3 cache:      $(lscpu | grep 'L3 cache' | sed 's/.*: *//')"
        echo ""
        echo "Taskset cores: $TASKSET_CORES"
        echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-$(nproc)}"
        echo "Compiler:      $(g++ --version | head -1)"
        echo "CMake:         $(cmake --version | head -1)"
        echo ""
        echo "══ Enabled providers ══"
        "$BENCH_BIN" --benchmark_list_tests 2>/dev/null \
            | sed 's|.*/.*/.*/ *||; s|/.*||' | sort -u | sed 's/^/  /'
        echo ""
        echo "══ Full lscpu ══"
        lscpu
        echo ""
        echo "══ /proc/cpuinfo flags ══"
        grep -m1 'flags' /proc/cpuinfo | tr ' ' '\n' | grep -E 'avx|sse|fma' | sort | tr '\n' ' '
        echo ""
    } > "$RESULTS_DIR/system_info.txt" 2>&1
    ok "System info → $RESULTS_DIR/system_info.txt"
}

# ═════════════════════════════════════════════════════════════════════════════
# 5. RUN BENCHMARKS
# ═════════════════════════════════════════════════════════════════════════════
run_single() {
    local filter="$1"
    local outfile="$2"
    local label="$3"
    local count="$4"
    local total="$5"

    log "[$count/$total] $label"

    local start_time
    start_time=$(date +%s)

    # Pin to physical cores, redirect stderr to log file.
    taskset -c "$TASKSET_CORES" \
        "$BENCH_BIN" \
        --benchmark_filter="$filter" \
        --benchmark_format=json \
        --benchmark_out="$outfile" \
        --benchmark_counters_tabular=true \
        2>> "$RESULTS_DIR/stderr.log"

    local elapsed=$(( $(date +%s) - start_time ))

    # Quick sanity check: does the JSON have benchmark entries?
    local n_entries
    n_entries=$(python3 -c "import json; print(len(json.load(open('$outfile')).get('benchmarks',[])))" 2>/dev/null || echo "?")

    ok "  ${elapsed}s — ${n_entries} entries → $(basename "$outfile")"
}

run_all() {
    banner "Running Benchmarks"

    # Discover all registered benchmarks.
    local benchmarks
    benchmarks=$("$BENCH_BIN" --benchmark_list_tests 2>/dev/null)

    if [ -z "$benchmarks" ]; then
        fail "No benchmarks registered — check build configuration"
    fi

    local total
    total=$(echo "$benchmarks" | wc -l)
    log "Found $total benchmark configurations"
    echo ""

    local count=0
    local overall_start
    overall_start=$(date +%s)

    while IFS= read -r bname; do
        count=$((count + 1))

        # "PhaseCycle/S20/B1000/Spira/iterations:50/process_time"
        # → "PhaseCycle_S20_B1000_Spira"
        local safe_name
        safe_name=$(echo "$bname" | sed 's|/iterations.*||' | tr '/' '_')
        local outfile="$RESULTS_DIR/${safe_name}.json"

        run_single "^$(echo "$bname" | sed 's/[.[\*^$()+?{|]/\\&/g')$" \
                   "$outfile" "$bname" "$count" "$total"

    done <<< "$benchmarks"

    local total_elapsed=$(( $(date +%s) - overall_start ))
    echo ""
    ok "All $total benchmarks complete in ${total_elapsed}s"
}

# ═════════════════════════════════════════════════════════════════════════════
# 6. MERGE AND ANALYSE
# ═════════════════════════════════════════════════════════════════════════════
merge_and_analyse() {
    banner "Merging Results & Generating Plots"

    # Merge all per-run JSONs into one file.
    python3 - "$RESULTS_DIR" <<'PYEOF'
import json, sys, glob, os

results_dir = sys.argv[1]
merged = {"context": {}, "benchmarks": []}

for f in sorted(glob.glob(os.path.join(results_dir, "PhaseCycle_*.json"))):
    with open(f) as fh:
        data = json.load(fh)
    if not merged["context"] and "context" in data:
        merged["context"] = data["context"]
    merged["benchmarks"].extend(data.get("benchmarks", []))

out = os.path.join(results_dir, "all_results.json")
with open(out, "w") as fh:
    json.dump(merged, fh, indent=2)
print(f"  Merged {len(merged['benchmarks'])} entries → all_results.json")
PYEOF
    ok "Merged JSON → $RESULTS_DIR/all_results.json"

    # Generate plots.
    if python3 -c "import matplotlib" 2>/dev/null; then
        log "Generating plots..."
        python3 "$SCRIPT_DIR/analyse.py" \
            "$RESULTS_DIR/all_results.json" \
            -o "$RESULTS_DIR/plots" 2>&1 | sed 's/^/  /'
        ok "Plots → $RESULTS_DIR/plots/"
    else
        warn "matplotlib missing — skipping plots"
    fi

    # Print summary table.
    echo ""
    python3 - "$RESULTS_DIR/all_results.json" <<'PYEOF'
import json, sys, re
from collections import defaultdict

with open(sys.argv[1]) as f:
    data = json.load(f)

results = defaultdict(dict)
for b in data.get("benchmarks", []):
    m = re.match(r"PhaseCycle/S(\d+)/B(\d+)/(.+?)(?:/|$)", b["name"])
    if not m:
        continue
    scale, bs, provider = int(m[1]), int(m[2]), m[3]
    results[(scale, provider)][bs] = b.get("real_time", 0)

if not results:
    print("  No results to summarise.")
    sys.exit(0)

scales = sorted({k[0] for k in results})
providers = sorted({k[1] for k in results})
batch_sizes = sorted({bs for v in results.values() for bs in v})

print("  ┌─────────────────────┬" + "┬".join("────────────" for _ in batch_sizes) + "┐")
print("  │ Provider            │" + "│".join(f" B={bs:<9d}" for bs in batch_sizes) + "│")
print("  ├─────────────────────┼" + "┼".join("────────────" for _ in batch_sizes) + "┤")

for scale in scales:
    for prov in providers:
        key = (scale, prov)
        if key not in results:
            continue
        label = f"S{scale}/{prov}"
        vals = []
        for bs in batch_sizes:
            if bs in results[key]:
                vals.append(f" {results[key][bs]:>8.1f} ms")
            else:
                vals.append("          — ")
        print(f"  │ {label:<19s} │" + "│".join(vals) + "│")

print("  └─────────────────────┴" + "┴".join("────────────" for _ in batch_sizes) + "┘")
PYEOF
}

# ═════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ═════════════════════════════════════════════════════════════════════════════
main() {
    banner "Spira Dynamic Graph Analyser Benchmark"
    log "$(date)"
    log "Results will be saved to: $RESULTS_DIR"

    # Detect instance type if on EC2.
    INSTANCE_TYPE=$(curl -sf --connect-timeout 2 \
        http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "not-ec2")
    log "Instance type: $INSTANCE_TYPE"

    mkdir -p "$RESULTS_DIR"

    install_deps
    apply_hw_isolation
    build_benchmark
    save_system_info
    run_all
    merge_and_analyse

    banner "Done"
    ok "Results directory: $RESULTS_DIR"
    echo ""
    echo "  all_results.json  — merged benchmark data"
    echo "  system_info.txt   — hardware / compiler info"
    echo "  stderr.log        — init + correctness check output"
    echo "  plots/            — PNG charts (if matplotlib available)"
    echo "  PhaseCycle_*.json — per-run raw data"
    echo ""
    log "To re-generate plots later:"
    log "  python3 scripts/analyse.py results/$TIMESTAMP/all_results.json -o results/$TIMESTAMP/plots"
}

main "$@"
