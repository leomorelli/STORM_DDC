"""Command-line boundary for the faithful optimized Stage-B package."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re
import sys
from typing import Sequence

import h5py
import numpy as np

from .io import DDCDataset, MatFileError, load_dataset
from .combine_io import load_combined_images
from .tool_cli import ALIASES, register_tools
from .pipeline import DDCConfig, DDCPipelineResult, run_ddc
from .serialization import (
    CheckpointWriter,
    DDCCheckpoint,
    load_checkpoint,
    make_analyzed_filename,
    resume_ddc,
    save_v73_checkpoint,
)


def _version() -> str:
    try:
        return version("python-ddc")
    except PackageNotFoundError:
        return "0.1.0"


def _rngs(seed: int | None, n_images: int) -> list[np.random.Generator]:
    sequence = np.random.SeedSequence(seed)
    return [np.random.default_rng(child) for child in sequence.spawn(n_images)]


def _output_path(input_path: Path, requested: str | None) -> Path:
    if requested is not None:
        return Path(requested)
    return input_path.parent / make_analyzed_filename(input_path.name)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _prepare_operational_dataset(
    dataset: DDCDataset, *, photon_weighted_correction: int
) -> None:
    dataset.frames = [
        np.asarray(values, dtype=np.float64)
        if np.issubdtype(np.asarray(values).dtype, np.integer)
        else values
        for values in dataset.frames
    ]
    if photon_weighted_correction and dataset.photons is not None:
        dataset.photons = [
            np.asarray(values, dtype=np.float64)
            if np.issubdtype(np.asarray(values).dtype, np.integer)
            else values
            for values in dataset.photons
        ]


def _run(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    dataset = load_dataset(input_path)
    photon_weighted_correction = (
        0 if args.no_photon_weighted_correction else 1
    )
    if not args.preserve_source_dtypes:
        _prepare_operational_dataset(
            dataset,
            photon_weighted_correction=photon_weighted_correction,
        )
    config = DDCConfig(
        resolution=args.resolution,
        stepper=args.stepper,
        n_f=args.n_f,
        timer=args.timer,
        photon_weighted_correction=photon_weighted_correction,
    )
    output = _output_path(input_path, args.output)
    _ensure_parent(output)
    writer = CheckpointWriter(output, dataset, config)
    run_ddc(
        dataset,
        config,
        rngs=_rngs(args.seed, dataset.n_images),
        checkpoint_callback=writer,
        workers=args.workers,
        verbose=not args.quiet,
    )
    print(output)
    return 0


def _resume(args: argparse.Namespace) -> int:
    input_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(input_path)
    output = Path(args.output) if args.output is not None else input_path
    _ensure_parent(output)
    writer = CheckpointWriter(
        output,
        checkpoint.dataset,
        checkpoint.config,
        analyzed_filename=checkpoint.analyzed_filename,
        source_metadata=checkpoint.source_metadata,
    )
    result = resume_ddc(
        checkpoint,
        rngs=_rngs(args.seed, checkpoint.dataset.n_images),
        checkpoint_callback=writer,
        workers=args.workers,
        verbose=not args.quiet,
    )
    # A completed checkpoint emits no outer callback. Preserve the requested
    # output behavior through the approved serializer without rerunning MCMC.
    if writer.last_checkpoint is None:
        final_checkpoint = DDCCheckpoint.from_result(
            result,
            checkpoint.dataset,
            checkpoint.config,
            checkpoint.analyzed_filename,
            source_filename=checkpoint.source_filename,
            source_metadata=checkpoint.source_metadata,
            phase="outer",
        )
        save_v73_checkpoint(final_checkpoint, output)
        if not args.quiet:
            print("[DDC] Completed checkpoint saved.", file=sys.stderr, flush=True)
    print(output)
    return 0


def _counts(dataset: DDCDataset) -> list[int]:
    return [int(np.asarray(cell).shape[0]) for cell in dataset.localizations]


def _dataset_summary(dataset: DDCDataset) -> list[str]:
    dimensions = [
        int(np.asarray(cell).shape[1]) if np.asarray(cell).ndim == 2 else -1
        for cell in dataset.localizations
    ]
    photon_cells = dataset.photons or []
    truth_present = (
        [bool(np.asarray(cell).size) for cell in dataset.true_localizations]
        if dataset.true_localizations
        else [False] * dataset.n_images
    )
    return [
        f"images: {dataset.n_images}",
        f"localizations: {_counts(dataset)}",
        f"localization_dimensions: {dimensions}",
        f"frame_dtypes: {[str(np.asarray(cell).dtype) for cell in dataset.frames]}",
        f"photons_present: {bool(photon_cells)}",
        f"photon_dtypes: {[str(np.asarray(cell).dtype) for cell in photon_cells]}",
        f"truth_present: {truth_present}",
    ]


def _inspect(args: argparse.Namespace) -> int:
    path = Path(args.input)
    try:
        checkpoint = load_checkpoint(path)
    except (KeyError, ValueError, TypeError, OSError):
        try:
            dataset = load_dataset(path)
        except MatFileError:
            dataset = load_combined_images(path)
            print("type: combined corrected DDC MAT dataset")
        else:
            print("type: raw DDC MAT dataset")
        print("\n".join(_dataset_summary(dataset)))
        return 0

    print("type: Python-generated DDC checkpoint")
    print(f"format: {'MATLAB v7.3' if h5py.is_hdf5(path) else 'MATLAB Level-5'}")
    print(f"phase: {checkpoint.phase}")
    print("\n".join(_dataset_summary(checkpoint.dataset)))
    print(f"Step: {checkpoint.result.steps.tolist()}")
    print(f"stepper: {checkpoint.config.stepper}")
    print(f"complete: {bool(np.all(checkpoint.result.steps >= checkpoint.config.stepper))}")
    return 0


def _positive_workers(value: str) -> int:
    try:
        workers = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("workers must be a positive integer") from None
    if workers < 1:
        raise argparse.ArgumentTypeError("workers must be a positive integer")
    return workers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ddc",
        description="Faithful optimized Stage-B Python port of DDC.",
    )
    parser.add_argument("--debug", action="store_true", help="show tracebacks")
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="run DDC on an input MAT dataset")
    run_parser.add_argument("input")
    run_parser.add_argument("--output")
    run_parser.add_argument("--resolution", type=float, default=80.0)
    run_parser.add_argument("--stepper", type=int, default=100)
    run_parser.add_argument("--n-f", type=int, default=200)
    run_parser.add_argument("--timer", type=float, default=720.0)
    run_parser.add_argument("--seed", type=int)
    run_parser.add_argument("--quiet", action="store_true", help="hide progress messages")
    run_parser.add_argument(
        "--workers", type=_positive_workers, default=1,
        help="image worker processes (default: 1; capped by eligible image count)",
    )
    run_parser.add_argument("--no-photon-weighted-correction", action="store_true")
    run_parser.add_argument(
        "--preserve-source-dtypes",
        action="store_true",
        help="retain source integer dtypes and source-faithful failures",
    )
    run_parser.set_defaults(handler=_run)

    resume_parser = commands.add_parser(
        "resume", help="continue a checkpoint with corrected Python resume"
    )
    resume_parser.add_argument("checkpoint")
    resume_parser.add_argument("--output")
    resume_parser.add_argument("--seed", type=int)
    resume_parser.add_argument("--quiet", action="store_true", help="hide progress messages")
    resume_parser.add_argument(
        "--workers", type=_positive_workers, default=1,
        help="image worker processes (default: 1; capped by eligible image count)",
    )
    resume_parser.set_defaults(handler=_resume)

    inspect_parser = commands.add_parser("inspect", help="inspect MAT metadata only")
    inspect_parser.add_argument("input")
    inspect_parser.set_defaults(handler=_inspect)
    register_tools(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    for i, value in enumerate(argv):
        if value == "--debug":
            continue
        argv[i] = ALIASES.get(value, value)
        break
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as error:
        if args.debug:
            raise
        match = re.search(r"DDC-SOURCE-BUG-\d{3}", str(error))
        if match:
            print(f"error [{match.group(0)}]: {error}", file=sys.stderr)
        else:
            print(f"error: {error}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "main"]
