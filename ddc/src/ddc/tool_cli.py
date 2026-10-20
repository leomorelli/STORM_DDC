"""CLI registration for the Python-only DDC preprocessing commands."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from .io import load_dataset, save_dataset, save_tool_mat
from .combine_io import load_corrected_regions, save_combined_images
from .combining import combine_images
from .tools import determine_n, determine_resolution, inspect_resolution, split_images


ALIASES = {"-det_N": "determine-n", "-det_Res": "determine-resolution",
           "-det_res": "determine-resolution", "-split": "split", "-combine": "combine"}


def _progress(message):
    print(f"[DDC tools] {message}", file=sys.stderr, flush=True)


def _tool(args):
    if args.output and len(args.inputs) != 1:
        raise ValueError("--output requires a single input; use --output-dir for multiple files")
    # Reject conflicting output names before processing any input.
    prefixes = {"split": "Split_", "determine-n": "Determine_N_",
                "determine-resolution": "Determine_Res_", "inspect-resolution": "Inspect_Res_",
                "combine": "Combined_Final_"}
    outputs = [Path(args.output) if args.output else
               (Path(args.output_dir) if args.output_dir else Path(p).parent) /
               (prefixes[args.command] + Path(p).stem + ".mat") for p in args.inputs]
    all_inputs = {Path(p).resolve() for p in args.inputs}
    if any(p.suffix.lower() != ".mat" for p in outputs):
        raise ValueError("--output must have a .mat extension; CSV and PNG names are derived from it")
    if len({p.resolve() for p in outputs}) != len(outputs):
        raise ValueError("inputs would have the same output name; process them separately")
    if any(p.resolve() in all_inputs for p in outputs):
        raise ValueError("preprocessing output must not replace an input file")
    progress = None if args.quiet else _progress
    for filename, output in zip(args.inputs, outputs):
        if args.command == "combine":
            regions = load_corrected_regions(filename, allow_incomplete=args.allow_incomplete,
                                              progress=progress)
            result = combine_images(regions, boundary_mode=args.boundary_mode, progress=progress)
            save_combined_images(result, output)
            if progress:
                progress(f"Saved {result.n_images} combined images to {output}")
            print(output)
            continue
        dataset = load_dataset(filename)
        if progress:
            progress(f"Reading {filename}: {dataset.n_images} images")
        if args.command == "split":
            result = split_images(dataset, min_localizations=args.min_localizations,
                                  buffer=args.buffer, search_iterations=args.search_iterations,
                                  max_trials=args.max_trials, rng=np.random.default_rng(args.seed),
                                  progress=progress)
            save_dataset(result, output)
            if progress:
                progress(f"Saved {result.n_images} regions to {output}")
            if args.plot:
                from .tool_plots import plot_split
                figure = plot_split(result)
        else:
            common = dict(block_size=args.block_size, progress=progress)
            if args.command == "determine-n":
                result = determine_n(dataset, max_frame=args.max_frame, gap=args.gap,
                                     max_localizations=args.max_localizations, **common)
            elif args.command == "determine-resolution":
                result = determine_resolution(dataset, n_f=args.n_f,
                                              max_resolution=args.max_resolution,
                                              min_resolution=args.min_resolution, step=args.step,
                                              **common)
            else:
                result = inspect_resolution(dataset, n_f=args.n_f, resolution=args.resolution,
                                            image_fraction=args.image_fraction, **common)
            save_tool_mat(output, result.matlab_variables())
            if args.command == "determine-n":
                np.savetxt(output.with_suffix(".csv"),
                           np.column_stack((result.frame_differences, result.z, result.pair_counts)),
                           delimiter=",", header="frame_difference,Z,pair_count", comments="",
                           fmt=["%d", "%.17g", "%d"])
                if not np.isfinite(result.z).any():
                    _progress("No finite Z curve: lag 1 has no pairs. Check frame units/sampling.")
                elif np.any(result.pair_counts == 0):
                    _progress("Some lags have no pairs and appear as gaps; they are not a plateau.")
                if progress:
                    progress("Choose N_f from the sustained plateau in Z; pass it to --n-f.")
            elif args.command == "determine-resolution":
                print(f"Resolution: {result.resolution:g} nm")
                if result.reached_minimum:
                    _progress("Reached the minimum tested bin width; inspect the plot before using it.")
            if not args.no_plot:
                from .tool_plots import plot_n, plot_resolution
                figure = plot_n(result) if args.command == "determine-n" else plot_resolution(result)
        if (args.command == "split" and args.plot) or (args.command != "split" and not args.no_plot):
            import matplotlib.pyplot as plt
            figure.savefig(output.with_suffix(".png"), dpi=160)
            plt.close(figure)
        print(output)
    return 0


def register_tools(commands):
    for name, description in (
        ("split", "split localization images into overlapping regions (alias: -split)"),
        ("determine-n", "compute the N_f diagnostic curve (alias: -det_N)"),
        ("determine-resolution", "estimate the DDC bin width (alias: -det_Res)"),
        ("inspect-resolution", "inspect pair distributions at a chosen bin width"),
        ("combine", "reconstruct images from corrected split regions (alias: -combine)"),
    ):
        parser = commands.add_parser(name, help=description)
        parser.add_argument("inputs", nargs="+", help="one or more input MAT datasets")
        output = parser.add_mutually_exclusive_group()
        output.add_argument("--output", help="output MAT path for a single input")
        output.add_argument("--output-dir", help="destination folder for all outputs")
        parser.add_argument("--quiet", action="store_true", help="hide progress")
        if name == "split":
            parser.add_argument("--min-localizations", "--min-loc", type=int, default=5000)
            parser.add_argument("--buffer", type=float, default=250, help="initial overlap buffer in nm")
            parser.add_argument("--search-iterations", type=int, default=300)
            parser.add_argument("--max-trials", type=int, default=10000)
            parser.add_argument("--seed", type=int)
            parser.add_argument("--plot", action="store_true", help="save an xy preview of the first image")
        elif name == "combine":
            parser.add_argument("--boundary-mode", choices=("partition", "matlab"), default="partition",
                                help="partition shared boundaries per image (default), or reproduce MATLAB's boundary rules")
            parser.add_argument("--allow-incomplete", action="store_true",
                                help="combine current corrected results before all regions reach stepper")
        else:
            parser.add_argument("--block-size", type=int, default=512, help="pairwise distance block size")
            parser.add_argument("--no-plot", action="store_true", help="save numerical results without PNG")
            if name == "determine-n":
                parser.add_argument("--max-frame", "--nftp", type=int, default=1000)
                parser.add_argument("--gap", type=int, default=10)
                parser.add_argument("--max-localizations", type=int, default=6000,
                                    help="exclude images with this many rows or more (MATLAB default: 6000)")
            else:
                parser.add_argument("--n-f", type=float, required=True,
                                    help="frame difference chosen from determine-n")
                if name == "determine-resolution":
                    parser.add_argument("--max-resolution", type=int, default=150)
                    parser.add_argument("--min-resolution", type=int, default=10)
                    parser.add_argument("--step", type=int, default=10)
                else:
                    parser.add_argument("--resolution", type=float, default=80, help="bin width in nm")
                    parser.add_argument("--image-fraction", type=float, default=.1)
        parser.set_defaults(handler=_tool)
