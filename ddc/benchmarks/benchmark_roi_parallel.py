"""Time fixed-work ROI processing using a saved shared distribution.

Run from a terminal: python benchmarks/benchmark_roi_parallel.py INPUT RESULT
This excludes distribution estimation and checkpoint I/O, and never overwrites
either MAT file. RESULT supplies only bins/blink/X_overall/M_mat, not MCMC state.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import h5py
import numpy as np
from scipy import sparse

import ddc.pipeline as pipeline
from ddc.cli import _prepare_operational_dataset, _rngs
from ddc.io import load_dataset
from ddc.pipeline import DDCConfig, run_ddc
from ddc._acceleration import native_available


def assert_equal(left, right):
    if is_dataclass(left):
        for field in fields(left):
            assert_equal(getattr(left, field.name), getattr(right, field.name))
    elif sparse.issparse(left):
        np.testing.assert_array_equal(left.toarray(), right.toarray())
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right, strict=True)
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


def load_baseline(folder):
    modules = {}
    for name in ("distances", "probabilities", "mcmc"):
        alias = f"ddc._before_{name}"
        spec = importlib.util.spec_from_file_location(alias, folder / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
        modules[name] = module
    modules["mcmc"].make_distance_matrix_individual = modules["distances"].make_distance_matrix_individual
    modules["mcmc"].determine_deviation_in_probability8 = modules["probabilities"].determine_deviation_in_probability8
    return modules["mcmc"].run_mcmc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--rois", type=int, default=8)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.rois < 1 or args.steps < 11:
        parser.error("use at least one ROI and at least 11 steps")
    data = load_dataset(args.input)
    data.localizations = data.localizations[:args.rois]
    data.frames = data.frames[:args.rois]
    data.true_localizations = data.true_localizations[:args.rois]
    if data.photons is not None:
        data.photons = data.photons[:args.rois]
    _prepare_operational_dataset(data, photon_weighted_correction=1)
    with h5py.File(args.result) as source:
        read = lambda name: source[name][()].T.copy()
        bins = read("bins").ravel()
        blink = read("Distribution_for_Blink").ravel()
        x_overall = read("X_overall").ravel()
        m_mat = read("M_mat")
        resolution = float(read("Resolution").item())
        n_f = int(read("N_f").item())
    settings = DDCConfig(resolution=resolution, n_f=n_f, stepper=args.steps, timer=float("inf"))
    shared = (bins, blink, np.empty((0, blink.size)), resolution, x_overall, m_mat)
    records = []
    expected = None
    runs = [("optimized", n) for n in args.workers]
    if args.baseline_dir:
        runs.insert(0, ("before", 1))
    for version, workers in runs:
        mcmc = load_baseline(args.baseline_dir) if version == "before" else pipeline.run_mcmc
        with patch.object(pipeline, "determine_blinking_distribution", return_value=shared), patch.object(pipeline, "run_mcmc", mcmc):
            start = perf_counter()
            result = run_ddc(data, settings, rngs=_rngs(123, data.n_images), workers=workers)
            elapsed = perf_counter() - start
        if expected is None:
            expected = result
        else:
            assert_equal(expected, result)
        record = dict(version=version, workers=workers, seconds=elapsed, rois=data.n_images,
                      native_kernels=version != "before" and native_available(),
                      localizations=[len(c) for c in data.localizations], final_steps=result.steps.tolist(),
                      equal_to_first_run=True)
        records.append(record)
        print(json.dumps(record), flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
