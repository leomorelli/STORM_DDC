# Python DDC

DDC corrects repeat localizations in single-molecule localization microscopy
(SMLM) data. This package is the final optimized Stage-B implementation of the
repository's MATLAB computational pipeline. Its conservative fast paths retain
the accepted Stage-A numerical behavior, including documented source defects;
the frozen Stage-A archive remains the immutable unoptimized reference.

## Installation

```bash
conda create -n ddc python=3.12.13
conda activate ddc

cd ddc
python -m pip install ".[test]"
```

Python 3.10 or newer is required. Runtime dependencies are NumPy, SciPy, h5py,
hdf5storage, and Matplotlib (for preprocessing diagnostic plots).

To use this edited workspace copy instead of a previously installed copy,
activate your DDC environment, open a terminal in this folder, and run:

```bash
python -m pip install -e ".[test]"
```

### Compiled C acceleration

The package builds optional Cython C kernels during installation when a C
compiler is available. These kernels are already built and enabled in the local
`ddc-parallel` environment. NumPy and SciPy versions were kept unchanged.
The runtime does not require Cython once the extension has been built.

The native kernels handle distance-bin construction and the MCMC/probability
trajectory-linking loops for supported float64 data. Integer or unusual inputs
use the existing Python/NumPy path. Ambiguous distance-bin interactions cause
the linking pass to rerun with the original NumPy calculations.
Compiler options disable fast-math and fused multiply-add contraction; mean
probability reductions retain NumPy's original pair order and reduction.

Check availability in a fresh interpreter:

```python
from ddc._acceleration import native_available
print(native_available())
```

`run` and `resume` print `Compiled C kernels: enabled` when available. Restart an
already-open notebook kernel after installing or rebuilding the extension.
To compare with the Python kernels, set `DDC_DISABLE_NATIVE=1` before running;
remove that environment variable to restore compiled kernels. Both paths use
the same scientific settings and random draws.

On a new machine, installing this package builds the extension with Cython and
the local C compiler (Visual Studio C++ Build Tools on Windows). If compilation
is unavailable, the optional extension is omitted and Python kernels remain
usable. The compiled binary is specific to the operating system, architecture,
and Python version; rebuild it when changing those.

## Command line

### Python preprocessing tools — no MATLAB required

The localization-image splitting, parameter diagnostics, and combining from MATLAB's
`DDC_Tools` folder are now available directly in this package:

```bash
ddc -split input.mat --seed 123
ddc -det_N Split_input.mat
ddc -det_Res Split_input.mat --n-f 200
ddc run Split_input.mat --n-f 200 --resolution 80 --workers 4 --seed 123 --output analyzed.mat
ddc -combine analyzed.mat
```

**200 and 80 above are examples, not values inferred for your data.**
`-det_N` saves a Z-versus-frame plot, CSV, and MAT file. Choose N_f where the
populated curve reaches a sustained plateau, then pass it to `-det_Res`.
The resolution command prints its recommended bin width and saves its plot
and MAT results. Pass the chosen values explicitly to `ddc run`.
DDC resolution here means **distance histogram bin width in nm**.

You can pass multiple files; each is processed independently:

```bash
ddc -det_N "file1.mat" "file2.mat" --output-dir results/parameters
ddc -det_N "file1.mat" --max-frame 2000 --gap 10
ddc inspect-resolution "file1.mat" --n-f 200 --resolution 80
```

`-combine` trims the overlap buffers and reconstructs the original images,
writing `Combined_Final_analyzed.mat`. Its default assigns shared boundaries
once per source image; use `--boundary-mode matlab` to reproduce the original
MATLAB boundary rules for development comparisons. It accepts both Level-5
and v7.3 analyzed files, including Python checkpoints. Numerical exchange files
remain MAT for now; all calculations run in Python. NPZ/pickle support is deferred.

The conventional command names are `ddc split`, `ddc determine-n`,
`ddc determine-resolution`, and `ddc combine`. `python -m ddc` accepts the same commands and
aliases. Splitting operates on localization coordinates, with overlapping
regions and preserved frames/photons; it is not pixel-mask segmentation.
See [the preprocessing guide](docs/PREPROCESSING.md) for Python functions,
output names, options, and source-fidelity details.

### Run and resume

Run a new analysis:

```bash
ddc run example.mat --seed 123
```

The CLI operationally converts integer frame and photon arrays to floating
point before running, so repository MAT inputs can be processed directly. Use
`--preserve-source-dtypes` when you need the source-faithful integer dtype
behavior instead.

By default this writes
`Analyzed_TimeDD-Mmm-YYYY_example.mat` beside the input. Choose a path with
`--output`; missing parent directories are created. Source defaults are
`Resolution=80`, `stepper=100`, `N_f=200`, `timer=720`, and
`Photon_weighted_Correction=1`. They can be changed for a new run:

```bash
ddc run example.mat --output results/ddc.mat --resolution 80 \
  --stepper 100 --n-f 200 --timer 720 --seed 123 \
  --no-photon-weighted-correction
```

Continue a checkpoint using the corrected operational Python resume:

```bash
ddc resume results/ddc.mat --seed 123
```

The checkpoint's saved scientific configuration and state are authoritative;
resume provides no numerical override flags. Without `--output`, it atomically
updates the supplied checkpoint. Inspect a raw dataset or checkpoint without
running MCMC or changing the file:

```bash
ddc inspect example.mat
ddc inspect results/ddc.mat
python -m ddc --help
```

## Parallel processing of images / ROIs

Add `--workers` to process images in separate CPU processes:

```bash
python -m ddc run example.mat --workers 4 --seed 123 --output results/ddc.mat
python -m ddc resume results/ddc.mat --workers 4 --seed 123
```

The default is `--workers 1` (sequential). Worker count is capped by the number
of eligible unfinished images. A single eligible image runs locally. This
parallelizes images/ROIs already present as separate localization cells in the
input MAT file; it does not split one image into smaller ROIs.

The shared blinking distribution is calculated once for the whole dataset.
Per-image maxima, histograms, and mixture fits use the requested worker count;
the parent combines them in original image order. Each MCMC worker then receives
that shared result and processes an image's MCMC
chunk and photon-weighted correction. The parent collects results as they
finish, preserves image order in the saved arrays, and writes the checkpoint
after a complete outer pass. Workers never
write checkpoints. A worker error propagates to the caller, and the last
successfully written checkpoint remains available. Resume reuses the saved
distribution and may use a different worker count.

Each image retains its own NumPy random-generator state between chunks,
including when successive chunks run in different processes. For fixed work
(e.g. `--timer inf`), the same input, configuration, package version, and seed
produce matching sequential and parallel results. With a finite timer, process
scheduling can change chunk boundaries, so exact equality is not guaranteed.
As before, checkpoint files do **not** store RNG state: resuming from a file
does not promise the same trajectory as an uninterrupted run.

Start with 2-4 workers and check memory use: workers hold separate image data
and MCMC caches. Small jobs can be slower because of process startup and data
transfer. Existing NumPy/SciPy thread settings are left unchanged.

Terminal progress is enabled by default for both `run` and `resume`. It shows
total ROIs and localizations, shared-distribution preparation, actual worker
count, ROI completion as results arrive, skipped/already-complete ROIs, and
checkpoint saves. Each message includes elapsed seconds and is flushed
immediately. A finished processing pass may require another pass; only `DONE`
means the ROI reached the configured stopping threshold. Output arrays retain
their original ROI order even when completion messages arrive out of order.
Use `--quiet` to hide progress. Progress goes to stderr; the final output-file
path remains on stdout. The Python API is quiet by default; pass `verbose=True`
to `run_ddc()` or `resume_ddc()` to print the same messages.

### Python scripts and notebooks

Pass `workers=4` to `run_ddc()` or `resume_ddc()`. Parallel execution with multiple
eligible images requires a separate `numpy.random.Generator` per image; the
default generators and the `SeedSequence` example below satisfy this.
Custom RNG objects remain supported in sequential mode.

### Choosing worker count and measuring speed

For the local MATLAB comparison, `run_parallel_matlab_comparison.ps1` uses the
224-region input in MATLAB's `DDC_Tools` folder and the `ddc-parallel` environment:

```powershell
./run_parallel_matlab_comparison.ps1 -Workers 4
```

It sets OMP/OpenBLAS/MKL thread counts to one per process, restores those
environment variables afterward, and explicitly uses this source tree. Compare
2, 4, and 8 workers on your machine; memory and CPU contention can make additional
workers slower. Each worker needs its own ROI arrays and MCMC caches. The number
of workers cannot exceed the number of eligible ROIs. Large ROIs are submitted
first to reduce idle workers at the end of a processing pass.

The current optimization batches distance-bin appends and uses binary searches
for candidate frame pairs in the MCMC and probability eliminators. The searches
apply only to sorted, finite, integer-valued float64 frames in a safe numerical
range. Other frame inputs retain the original scans. Distance arithmetic,
candidate order, and random draws are preserved.

`benchmarks/benchmark_roi_parallel.py INPUT.mat ANALYZED.mat --workers 1 2 4`
measures ROI processing with fresh MCMC state and a saved shared distribution.
It uses the same seed, no timer interruptions, and verifies that all workers
produce identical outputs. This measurement excludes shared-distribution
estimation and checkpoint I/O. Distribution estimation now uses image workers;
its final reductions and checkpoint I/O run in the parent process. A full-run
MATLAB/Python comparison must use identical input regions and scientific
settings. The validation notebook only loads and plots existing results; it
does not run DDC or choose the worker count.

For a Python **script**, protect the entry point so spawned Windows processes
can import it safely:

```python
import numpy as np
from ddc import DDCConfig, CheckpointWriter, load_dataset, run_ddc


def main():
    dataset = load_dataset("example.mat")
    # The Python API preserves source dtypes; use floats for operational runs.
    dataset.frames = [np.asarray(f, dtype=float) for f in dataset.frames]
    if dataset.photons is not None:
        dataset.photons = [np.asarray(p, dtype=float) for p in dataset.photons]
    config = DDCConfig()
    rngs = [np.random.default_rng(child)
            for child in np.random.SeedSequence(123).spawn(dataset.n_images)]
    writer = CheckpointWriter("Analyzed_example.mat", dataset, config)
    run_ddc(dataset, config, rngs=rngs, workers=4, checkpoint_callback=writer)


if __name__ == "__main__":
    main()
```

In a **notebook**, the CLI can be launched with the notebook's Python interpreter
after installing this copy in that environment:

```python
import subprocess
import sys

subprocess.run([
    sys.executable, "-m", "ddc", "run", r"C:\path\input.mat",
    "--workers", "4", "--seed", "123",
    "--output", r"C:\path\results\ddc.mat",
], check=True)
```

## Python API

For real data, create a dataset with coordinates and frames. Ground truth is
optional: omit `true_localizations` (or pass `None` or `[]`). It is used only
for simulation diagnostics; DDC estimates the correction from the measured
localizations and frames.

```python
import numpy as np
from ddc import DDCDataset, DDCConfig, run_ddc

dataset = DDCDataset(
    localizations=[np.asarray(locations, dtype=float)],  # shape (n, 2) or (n, 3)
    frames=[np.asarray(frames, dtype=float)],            # one frame per row
)
result = run_ddc(dataset, DDCConfig())
```

For multiple images or ROIs, supply one array per image in each list. Supply
`photons=[...]` when photon weights are available. Input MAT files can likewise
omit `TrueLocalizations`.

To run a MAT file and save checkpoints from Python:

```python
import numpy as np
from ddc import DDCConfig, CheckpointWriter, load_dataset, run_ddc

dataset = load_dataset("example.mat")
config = DDCConfig()
children = np.random.SeedSequence(123).spawn(dataset.n_images)
rngs = [np.random.default_rng(child) for child in children]
writer = CheckpointWriter("Analyzed_example.mat", dataset, config)
result = run_ddc(dataset, config, rngs=rngs, checkpoint_callback=writer)
```

`resume_ddc()` is a corrected operational continuation using the modern Python
MCMC contract. `resume_finish_runs_source()` is instead a source-faithful
diagnostic representation of the broken, stale MATLAB `Finish_Runs.m` output
assignment (DDC-SOURCE-BUG-007). They are intentionally different APIs.

## Input and checkpoint contract

Input MAT files contain parallel MATLAB cells for `LocalizationsFinal` and
`Frame_Information`; optional supported cells include `TrueLocalizations` and
`Photons`, with split-image metadata retained when present. Localization rows
must correspond to frame and photon entries image by image. Use `ddc inspect`
to see counts, dimensions and dtypes, which are scientifically relevant.

The first callback writes a MATLAB Level-5 checkpoint after shared-distribution
initialization and before MCMC. Complete outer passes atomically replace it with
a true MATLAB-compatible v7.3 checkpoint. Sparse probability-cache fields,
cell orientation, dtypes, NaN/Inf, histories and continuation state are
preserved. Preprocessing tools generate Python diagnostic plots; the core
MCMC runner does not reproduce MATLAB's interactive figures.

## Reproducibility and source fidelity

Workspace parallel update validation (2026-10-02, Windows, Python 3.12.13):
all 15 new spawned-process tests passed, including complete result/RNG equality,
multiple outer passes, checkpoint reload/resume, and worker-failure recovery.
The complete suite reported **347 passed, 19 skipped, 5 failed**. The same five
failures also occur in the unchanged baseline (332 passed): four input-file
tests require absent `Main_DDC_Folder` MAT fixtures and one requires absent
MATLAB source files. No new regression failures were introduced. The historical
Stage-B release counts below and in `FINAL_MANIFEST.txt` refer to the original
release environment, not this workspace validation.

`--seed` uses `SeedSequence(seed).spawn(n_images)` to give each image an
independent deterministic NumPy generator. The same package version, input,
configuration and seed reproduce the Python execution to the extent guaranteed
by the accepted Stage-A contract. No identity with MATLAB RNG streams is claimed, and checkpoints do
not pretend to contain unsaved NumPy RNG state.

Stage A preserves confirmed upstream MATLAB defects rather than silently
changing numerical semantics. DDC-SOURCE-BUG-001 through -007 are documented
in `KNOWN_DEFECTS.md`. The Python API retains source dtypes by default;
Constant2ft leakage is retained, and broken MATLAB Finish_Runs behavior is not
mislabeled as the working Python resume. The CLI is the operational entry
point: it explicitly converts integer frames and photons before running, while
`--preserve-source-dtypes` restores the faithful failure behavior for
repository `uint16` frames at BUG-002.

Static source semantics and Python regressions are complete. MATLAB runtime
oracle tests remain skipped where MATLAB is unavailable; this release does not
claim runtime confirmation for those probes. See `docs/STAGE_B_FINAL_NOTES.md`.
