#!/usr/bin/env python3
"""
Analyse Google Benchmark JSON output and generate plots.

Usage:
    python3 analyse.py results/*.json -o plots/
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_benchmark_json(filepath: str) -> list[dict]:
    """Parse a Google Benchmark JSON file and return benchmark entries."""
    with open(filepath) as f:
        data = json.load(f)
    return data.get("benchmarks", [])


def extract_config(name: str) -> dict | None:
    """Extract (scale, batch_size, provider) from a benchmark name.

    Expected format: PhaseCycle/S20/B1000/CSR_Reference
    """
    m = re.match(r"PhaseCycle/S(\d+)/B(\d+)/(.+)", name)
    if not m:
        return None
    return {
        "scale": int(m.group(1)),
        "batch_size": int(m.group(2)),
        "provider": m.group(3),
    }


def load_all_results(json_files: list[str]) -> list[dict]:
    """Load and tag all benchmark results from multiple JSON files."""
    results = []
    for fpath in json_files:
        for entry in parse_benchmark_json(fpath):
            config = extract_config(entry["name"])
            if config is None:
                continue
            config["real_time_ms"] = entry.get("real_time", 0.0)
            config["cpu_time_ms"] = entry.get("cpu_time", 0.0)
            config["iterations"] = entry.get("iterations", 0)
            results.append(config)
    return results


def compute_statistics(results: list[dict]) -> dict:
    """Group results by (scale, batch_size, provider) and compute median/IQR."""
    grouped = defaultdict(list)
    for r in results:
        key = (r["scale"], r["batch_size"], r["provider"])
        grouped[key].append(r["real_time_ms"])

    stats = {}
    for key, times in grouped.items():
        arr = np.array(times)
        stats[key] = {
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "n": len(arr),
        }
    return stats


def plot_cycle_time_vs_batch(stats: dict, output_dir: str):
    """Plot 1: Total cycle time vs batch size (one line per provider, per scale)."""
    # Group by scale.
    scales = sorted({k[0] for k in stats})

    for scale in scales:
        fig, ax = plt.subplots(figsize=(10, 6))

        providers = sorted({k[2] for k in stats if k[0] == scale})
        batch_sizes = sorted({k[1] for k in stats if k[0] == scale})

        for provider in providers:
            medians = []
            q25s = []
            q75s = []
            bs_vals = []

            for bs in batch_sizes:
                key = (scale, bs, provider)
                if key in stats:
                    s = stats[key]
                    medians.append(s["median"])
                    q25s.append(s["q25"])
                    q75s.append(s["q75"])
                    bs_vals.append(bs)

            if not bs_vals:
                continue

            ax.plot(bs_vals, medians, "o-", label=provider, linewidth=2,
                    markersize=8)
            ax.fill_between(bs_vals, q25s, q75s, alpha=0.2)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Batch Size (mutations per phase)", fontsize=12)
        ax.set_ylabel("Phase Cycle Time (ms)", fontsize=12)
        ax.set_title(f"Total Cycle Time vs Batch Size — SCALE-{scale}",
                     fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xticks(batch_sizes)
        ax.set_xticklabels([str(b) for b in batch_sizes])

        outpath = os.path.join(output_dir, f"cycle_time_vs_batch_S{scale}.png")
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {outpath}")


def plot_phase_timeseries(results: list[dict], output_dir: str):
    """Plot 2: Phase cycle time over 50 phases (time series per provider).

    This requires per-iteration timing data. If Google Benchmark was run with
    --benchmark_repetitions, each repetition is a separate entry. Otherwise
    we plot what we have.
    """
    # Group by (scale, batch_size, provider) preserving order.
    grouped = defaultdict(list)
    for r in results:
        key = (r["scale"], r["batch_size"], r["provider"])
        grouped[key].append(r["real_time_ms"])

    scales = sorted({k[0] for k in grouped})
    batch_sizes = sorted({k[1] for k in grouped})

    for scale in scales:
        for bs in batch_sizes:
            providers_data = {}
            for key, times in grouped.items():
                if key[0] == scale and key[1] == bs:
                    providers_data[key[2]] = times

            if not providers_data:
                continue

            fig, ax = plt.subplots(figsize=(12, 5))
            for provider, times in sorted(providers_data.items()):
                ax.plot(range(1, len(times) + 1), times, ".-",
                        label=provider, linewidth=1.5)

            ax.set_xlabel("Phase", fontsize=12)
            ax.set_ylabel("Cycle Time (ms)", fontsize=12)
            ax.set_title(
                f"Phase Cycle Time Over Phases — S{scale} B{bs}", fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)

            outpath = os.path.join(
                output_dir, f"phase_timeseries_S{scale}_B{bs}.png")
            fig.savefig(outpath, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {outpath}")


def print_summary_table(stats: dict):
    """Print a text summary table of median cycle times."""
    print("\n" + "=" * 72)
    print("BENCHMARK RESULTS SUMMARY (median cycle time in ms)")
    print("=" * 72)

    scales = sorted({k[0] for k in stats})
    for scale in scales:
        print(f"\n  SCALE-{scale}")
        print(f"  {'Provider':<20} {'B=1000':>10} {'B=10000':>10} {'B=100000':>10}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

        providers = sorted({k[2] for k in stats if k[0] == scale})
        for prov in providers:
            vals = []
            for bs in [1000, 10000, 100000]:
                key = (scale, bs, prov)
                if key in stats:
                    vals.append(f"{stats[key]['median']:.1f}")
                else:
                    vals.append("—")
            print(f"  {prov:<20} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyse Spira benchmark results and generate plots.")
    parser.add_argument("json_files", nargs="+",
                        help="Google Benchmark JSON output files")
    parser.add_argument("-o", "--output-dir", default="plots",
                        help="Directory for output plots (default: plots/)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading results...")
    results = load_all_results(args.json_files)
    if not results:
        print("No valid benchmark results found.", file=sys.stderr)
        sys.exit(1)

    print(f"  Loaded {len(results)} benchmark entries.")

    stats = compute_statistics(results)
    print_summary_table(stats)

    print("Generating plots...")
    plot_cycle_time_vs_batch(stats, args.output_dir)
    plot_phase_timeseries(results, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
