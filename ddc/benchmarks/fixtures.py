"""Deterministic, supported floating-frame fixtures for Stage B.1."""

from __future__ import annotations

from dataclasses import dataclass
import os
import platform
from pathlib import Path
import sys

import numpy as np
import scipy

from ddc.io import DDCDataset, load_dataset


SEED = 20260819
TIERS = {"tiny": 25, "small": 100, "medium": 500, "large": 1000}


@dataclass(frozen=True)
class Fixture:
    name: str
    locations: np.ndarray
    frames: np.ndarray


def fixture(n: int, dimensions: int = 2, *, seed: int = SEED) -> Fixture:
    """Return clustered coordinates and float64 frames without timing setup."""
    rng = np.random.default_rng(seed + n + dimensions)
    centers = rng.normal(0.0, 250.0, size=(max(3, n // 20), dimensions))
    locations = centers[np.arange(n) % centers.shape[0]] + rng.normal(
        0.0, 18.0, size=(n, dimensions)
    )
    # Repeated, sorted float frames exercise same-frame and multi-lag behavior.
    frames = (np.arange(n) // 4 + 1).astype(np.float64)
    return Fixture(f"synthetic-n{n}-{dimensions}d", locations, frames)


def pipeline_fixture() -> Fixture:
    """The known supported Stage-A end-to-end fixture (23 localizations)."""
    frames = np.concatenate((np.ones(11), [2.0], np.full(11, 4.0)))
    x = np.concatenate((np.arange(11.0) * 10, [0.2], np.arange(11.0) * 10 + 0.5))
    return Fixture("pipeline-supported-23-2d", np.column_stack((x, np.zeros(x.size))), frames)


def dataset_from_fixture(value: Fixture) -> DDCDataset:
    n, d = value.locations.shape
    return DDCDataset(
        localizations=[value.locations.copy()],
        frames=[value.frames.copy()],
        true_localizations=[np.empty((0, d))],
        photons=[np.ones(n)],
        addon=np.zeros((1, 1)),
        source=Path("PERFORMANCE_ONLY_IN_MEMORY.mat"),
    )


def repository_surrogate(max_n: int = 100) -> Fixture | None:
    """Real coordinates plus copied float frames; performance use only."""
    candidate = Path(__file__).resolve().parents[1] / (
        "Main_DDC_Folder/Figure_1_and_2_data_and_analyzed_data/"
        "Split_Simulation_2_dark_state_sparse_Clusters_10per_3_per_ROI.mat"
    )
    if not candidate.exists():
        return None
    data = load_dataset(candidate)
    locations = np.asarray(data.localizations[0])[:max_n].copy()
    frames = np.asarray(data.frames[0])[:max_n].astype(np.float64, copy=True)
    return Fixture("PERFORMANCE-ONLY SUPPORTED SURROGATE", locations, frames)


def environment() -> dict[str, object]:
    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    ram = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                ram = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    return {
        "python": sys.version.split()[0], "numpy": np.__version__,
        "scipy": scipy.__version__, "platform": platform.platform(),
        "architecture": platform.machine(), "cpu_model": cpu,
        "logical_cores": os.cpu_count(), "total_ram_bytes": ram,
        "environment_variables": {k: os.environ.get(k) for k in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }
