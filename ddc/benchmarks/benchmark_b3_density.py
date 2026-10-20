"""Same-machine B.4-versus-B.3 Density_Calc benchmark and MCMC profile."""

from __future__ import annotations

import ast
import cProfile
import csv
import io
import json
from pathlib import Path
import pstats
import statistics
import time
import tracemalloc
from unittest.mock import patch

import numpy as np

import ddc.mcmc as mcmc
import ddc.trajectories as trajectories
from benchmarks.benchmark_b4_mcmc import execute
from benchmarks.fixtures import environment


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results"
STAGE_A_SHA = "a54496a4c08591fee307314ef95a2046b8c4c4e59c710ec913549691a1441279"
B2_SHA = "61c19e3af0a368b5e28fc65449e5c065cd50841478d6242d47a9d087cd3c42e8"
B4_SHA = "cb4aeca40f8b7cfcb62c6e544796568c6dac6554609c773b35bdbbe61c7a9eb8"


def load_b4_density():
    """Load the accepted nested reference from the equivalence test only."""

    path = ROOT / "tests/test_b3_equivalence.py"
    tree = ast.parse(path.read_text())
    wanted = {"_b4_raw", "_b4_density"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = dict(trajectories.__dict__)
    exec(compile(ast.Module(nodes, []), str(path), "exec"), namespace)
    return namespace["_b4_density"]


def fixture(n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260819 + n)
    locations = rng.normal(0.0, 30.0, size=(n, 3)).astype(np.float64)
    frames = np.sort(rng.integers(1, max(5, n // 2), size=n)).astype(np.float64)
    return locations, frames


def timed(function, repetitions: int) -> dict[str, float | int]:
    function()
    values = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        values.append(time.perf_counter() - start)
    return {
        "repetitions": repetitions,
        "minimum_seconds": min(values),
        "median_seconds": statistics.median(values),
        "mean_seconds": statistics.mean(values),
    }


def peak_memory(function) -> int:
    tracemalloc.start()
    function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def profile_post_b3() -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    execute(mcmc._eliminate_blinking_de_loc15, 200, 20261019, 15)
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs()
    stream.write("TOP 30 BY CUMULATIVE TIME\n")
    stats.sort_stats("cumulative").print_stats(30)
    stream.write("\nTOP 30 BY SELF TIME\n")
    stats.sort_stats("tottime").print_stats(30)
    return stream.getvalue()


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    baseline = load_b4_density()
    optimized = trajectories.density_calc
    rows = []

    for n in (25, 50, 100, 200, 400, 800, 1600):
        locations, frames = fixture(n)
        before_call = lambda l=locations, f=frames: baseline(l, f, 10.0, 2.0)
        after_call = lambda l=locations, f=frames: optimized(l, f, 10.0, 2.0)
        before_repetitions = 3 if n <= 800 else 1
        after_repetitions = 20 if n <= 400 else (10 if n <= 800 else 5)
        before = timed(before_call, before_repetitions)
        after = timed(after_call, after_repetitions)
        rows.append(
            {
                "n_localizations": n,
                "b4": before,
                "b3": after,
                "speedup": before["median_seconds"] / after["median_seconds"],
                "b4_peak_tracemalloc_bytes": peak_memory(before_call),
                "b3_peak_tracemalloc_bytes": peak_memory(after_call),
                "estimated_max_block_temporary_bytes": (
                    min(trajectories._DENSITY_BLOCK_SIZE, n) * n * 3 * 8
                    + 2 * min(trajectories._DENSITY_BLOCK_SIZE, n) * n * 8
                ),
            }
        )

    block_size_rows = []
    for n in (400, 800):
        locations, frames = fixture(n)
        for block_size in (32, 64, 128, 256, 512):
            with patch.object(trajectories, "_DENSITY_BLOCK_SIZE", block_size):
                measurement = timed(
                    lambda: optimized(locations, frames, 10.0, 2.0), 10
                )
                memory = peak_memory(lambda: optimized(locations, frames, 10.0, 2.0))
            block_size_rows.append(
                {
                    "n_localizations": n,
                    "block_size": block_size,
                    "median_seconds": measurement["median_seconds"],
                    "peak_tracemalloc_bytes": memory,
                }
            )

    scaling_rows = [row for row in rows if 50 <= row["n_localizations"] <= 800]
    b4_exponent = float(
        np.polyfit(
            np.log([row["n_localizations"] for row in scaling_rows]),
            np.log([row["b4"]["median_seconds"] for row in scaling_rows]),
            1,
        )[0]
    )
    b3_exponent = float(
        np.polyfit(
            np.log([row["n_localizations"] for row in scaling_rows]),
            np.log([row["b3"]["median_seconds"] for row in scaling_rows]),
            1,
        )[0]
    )

    mcmc_rows = []
    for n in (23, 50, 100, 200):
        samples = {}
        for label, density in (("B.4", baseline), ("B.3", optimized)):
            with patch.object(mcmc, "density_calc", density):
                measured = [
                    execute(
                        mcmc._eliminate_blinking_de_loc15,
                        n,
                        20260819 + n,
                        15,
                        memory=(repetition == 0),
                    )
                    for repetition in range(5)
                ]
            samples[label] = {
                "median_total_seconds": statistics.median(
                    sample["total_seconds"] for sample in measured
                ),
                "median_density_seconds": statistics.median(
                    sample["component_seconds"]["density"] for sample in measured
                ),
                "median_eliminator_seconds": statistics.median(
                    sample["component_seconds"]["eliminator"] for sample in measured
                ),
                "median_frame_cross_seconds": statistics.median(
                    sample["component_seconds"]["frame_cross"] for sample in measured
                ),
                "median_pairwise_seconds": statistics.median(
                    sample["component_seconds"]["pairwise_coordinate"]
                    for sample in measured
                ),
                "median_likelihood_seconds": statistics.median(
                    sample["component_seconds"]["likelihood"] for sample in measured
                ),
                "peak_tracemalloc_bytes": measured[0]["peak_tracemalloc_bytes"],
                "representative_counts": measured[1]["counts"],
            }
        mcmc_rows.append(
            {
                "n_localizations": n,
                "B.4": samples["B.4"],
                "B.3": samples["B.3"],
                "whole_mcmc_speedup": samples["B.4"]["median_total_seconds"]
                / samples["B.3"]["median_total_seconds"],
            }
        )

    profile = profile_post_b3()
    (RESULTS / "stage_b3_profile_top.txt").write_text(profile)
    result = {
        "environment": environment(),
        "references": {
            "frozen_stage_a_sha256": STAGE_A_SHA,
            "accepted_b2_review_sha256": B2_SHA,
            "accepted_b4_review_sha256": B4_SHA,
        },
        "methodology": {
            "same_machine_before_after": True,
            "warmups_per_measurement": 1,
            "selected_block_size": trajectories._DENSITY_BLOCK_SIZE,
            "temporary_memory_complexity": "O(block_size * n * d)",
            "arithmetic_complexity": "O(n^2 * d)",
        },
        "density_benchmarks": rows,
        "block_size_benchmarks": block_size_rows,
        "empirical_scaling_exponents_50_to_800": {
            "B.4": b4_exponent,
            "B.3": b3_exponent,
        },
        "mcmc_benchmarks": mcmc_rows,
        "ckdtree_used": False,
        "total_benchmark_runtime_seconds": time.perf_counter() - started,
    }
    (RESULTS / "stage_b3_density.json").write_text(json.dumps(result, indent=2) + "\n")
    with (RESULTS / "stage_b3_density.csv").open("w", newline="") as handle:
        keys = [
            "n_localizations",
            "b4_median_seconds",
            "b3_median_seconds",
            "speedup",
            "b4_peak_tracemalloc_bytes",
            "b3_peak_tracemalloc_bytes",
            "estimated_max_block_temporary_bytes",
        ]
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row.get(key) for key in keys},
                    "b4_median_seconds": row["b4"]["median_seconds"],
                    "b3_median_seconds": row["b3"]["median_seconds"],
                }
            )
    print(
        json.dumps(
            {
                "density": [
                    {
                        "n": row["n_localizations"],
                        "speedup": row["speedup"],
                    }
                    for row in rows
                ],
                "scaling": result["empirical_scaling_exponents_50_to_800"],
                "mcmc": [
                    {
                        "n": row["n_localizations"],
                        "speedup": row["whole_mcmc_speedup"],
                    }
                    for row in mcmc_rows
                ],
                "seconds": result["total_benchmark_runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
