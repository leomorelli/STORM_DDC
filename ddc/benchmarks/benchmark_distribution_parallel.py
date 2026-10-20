"""Compare ordered shared-distribution outputs and elapsed time by worker count."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from ddc.cli import _prepare_operational_dataset
from ddc.distributions import determine_blinking_distribution
from ddc.io import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--rois", type=int, default=4)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.rois < 1:
        parser.error("use at least one ROI")
    data = load_dataset(args.input)
    _prepare_operational_dataset(data, photon_weighted_correction=1)
    locations, frames = data.localizations[:args.rois], data.frames[:args.rois]
    runs = [("optimized", workers, determine_blinking_distribution) for workers in args.workers]
    if args.baseline:
        spec = importlib.util.spec_from_file_location("ddc._before_distributions", args.baseline)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        runs.insert(0, ("before", 1, module.determine_blinking_distribution))
    expected = None
    records = []
    for version, workers, function in runs:
        start = perf_counter()
        kwargs = {} if version == "before" else {"workers": workers}
        actual = function(locations, frames, 200, 80, **kwargs)
        elapsed = perf_counter() - start
        if expected is None:
            expected = actual
        else:
            for left, right in zip(expected, actual):
                np.testing.assert_array_equal(left, right, strict=True)
        record = dict(version=version, workers=workers, seconds=elapsed,
                      rois=len(locations), equal_to_first_run=True)
        records.append(record)
        print(json.dumps(record), flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
