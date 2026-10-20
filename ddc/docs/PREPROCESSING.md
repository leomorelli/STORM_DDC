# DDC preparation and combining in Python

All preprocessing commands execute locally in Python. There is no MATLAB,
MATLAB Engine, or MATLAB toolbox dependency. MAT is just the input/output file
format; NumPy, SciPy, and Matplotlib perform the calculations and plotting.
Numerical input/output files remain MAT during development. Switching exchange
and checkpoint storage to a Python format such as NPZ or pickle is deferred;
no format migration is performed by these commands.

Activate your existing environment before running the commands:

```powershell
conda activate ddc-parallel
cd C:\Users\maula\Desktop\Coding\ddc
```

For a fresh installation, run `python -m pip install -e ".[test]"` here.
`python -m ddc` can always replace `ddc` when the executable is not on PATH.

## Workflow

```powershell
# 1. Split large localization images into buffered regions.
ddc -split "input.mat" --min-loc 5000 --seed 123 --plot

# 2. Inspect Z versus frame difference and choose N_f at a sustained plateau.
ddc -det_N "Split_input.mat" --max-frame 1000 --gap 10

# 3. Substitute your selected N_f below; this example uses 200.
ddc -det_Res "Split_input.mat" --n-f 200

# 4. Optional: inspect per-region distributions for a chosen bin width.
ddc inspect-resolution "Split_input.mat" --n-f 200 --resolution 80

# 5. Substitute the selected N_f and recommended resolution below.
ddc run "Split_input.mat" --n-f 200 --resolution 80 --workers 4 --seed 123 --output analyzed.mat

# 6. Reconstruct original images from the corrected regions.
ddc -combine "analyzed.mat"
```

The bin width is a DDC analysis parameter in nanometers, not an independent
measurement of optical resolution or localization precision. All coordinate
arrays must already be in nm. Frame differences use the same units as the
input frame numbers.

| MATLAB script | Python command | Python function |
| --- | --- | --- |
| `Split_UP_Image_Multiple_Images.m` | `ddc -split` / `ddc split` | `split_images` |
| `Determine_N.m` | `ddc -det_N` / `ddc determine-n` | `determine_n` |
| `New_Determine_Res.m` | `ddc -det_Res` / `ddc determine-resolution` | `determine_resolution` |
| `Determine_Res.m` | `ddc inspect-resolution` | `inspect_resolution` |
| `Combine_Images.m` | `ddc -combine` / `ddc combine` | `combine_images` |

## Passing files and finding results

All five tool commands accept one or more MAT paths. Files are
analyzed **independently**, rather than pooling acquisitions that may have
different imaging conditions. Each file can itself contain multiple image cells.

```powershell
ddc -det_N "first.mat" "second.mat" --output-dir "results\parameters"
ddc -split "first.mat" "second.mat" --output-dir "results\split" --seed 123
ddc -combine "analyzed_first.mat" "analyzed_second.mat" --output-dir "results\combined"
```

Without `--output-dir`, derived files are saved beside each input:

| Command | Outputs for `input.mat` |
| --- | --- |
| `-split` | `Split_input.mat`; PNG preview with `--plot` |
| `-det_N` | `Determine_N_input.mat`, `.csv`, `.png` |
| `-det_Res` | `Determine_Res_input.mat`, `.png` |
| `inspect-resolution` | `Inspect_Res_input.mat`, `.png` |
| `-combine` (analyzed input) | `Combined_Final_input.mat` |

Use `--output path.mat` for one input, `--no-plot` on diagnostic commands to
omit PNGs, and `--quiet` to hide progress. Missing output directories are
created. The input cannot be used as a preprocessing output path. Repeated
runs replace derived results of the same name. `ddc -det_N --help` and the
other commands' help pages list all options.

The MAT results preserve MATLAB diagnostic variable names (`Framestore`, `Z`,
`Cum_Sum_Store`, `Res_Store`, `Resolution`) and add pair counts or search history.
You can read them from Python with `scipy.io.loadmat`; MATLAB is not needed.

## Choosing N_f

`Determine_N.m` does not contain an automatic N_f estimator. The Python port
also computes the diagnostic without inventing a threshold: it pools pair
distances at **exact** frame differences `1, 1+gap, ... <= max_frame`, forms
CDFs using edges `[0:250:5000, Inf]`, and computes the L1 difference from the
lag-1 CDF. Choose N_f where Z levels off over a sustained range supported by
enough pairs. The lower plot shows pair counts; empty lags are gaps/NaNs, not
zero-valued evidence of a plateau. If lag 1 has no pairs, Z is undefined.
The command reports this and still saves the diagnostic data.

Like the source, the default excludes regions with **6000 or more** rows.
Split large images first or raise `--max-localizations`. Increase
`--max-frame` if the curve is still changing at the end. A smaller `--gap`
gives more detail. No chosen N_f is automatically injected into subsequent runs.

## Automatic and manual resolution diagnostics

`determine-resolution` evaluates widths from 150 down to 10 nm in 10 nm
steps. It averages per-image short-lag (`lag < N_f`) and long-lag
(`lag > N_f`) probability histograms, scales their tails near 1000 nm,
subtracts and cleans the blinking distribution, and searches for the first
width whose peak is outside the first bin. It recommends that width + 10 nm,
as the source does. Equality at N_f is excluded. Histograms weight images
equally, not by their number of pairs.

The first candidate can recommend 160 nm. If none triggers the stopping rule,
the result is 10 nm and the command reports that it reached the search limit.
Inspect the plot, as the MATLAB comments also advise. Options
`--max-resolution`, `--min-resolution`, and `--step` change the search range.

If an image lacks short/long pairs, the tail denominator is zero, or no positive
blinking distribution remains, automatic estimation raises a clear error
instead of silently choosing from NaNs. Use `inspect-resolution` to inspect a
candidate width manually or revisit N_f and the acquisition's support.

`inspect-resolution` uses the first `ceil(image_fraction * number_of_images)`
images (default 10%), strict `lag > N_f`, and edges
`[0:resolution:D_max-resolution, Inf]`, as in `Determine_Res.m`. The optional
truth comparison preserves the source's exact **x-coordinate** matching rule;
it is not nearest-neighbor ground-truth matching. Experimental data need no truth.

## Split behavior and compatibility

The splitter searches random quantile grids with MATLAB/Hazen quantiles, a
default initial 250 nm overlap, and a target of 5000 localizations. It retains
the source score, sign-changing search counter, strict x/y and inclusive z
buffer predicates, and 10 nm buffer adjustments. The target is approximate:
the source's buffer cutoff can leave a region above or below the target.
The search has a configurable `--max-trials` safety limit; reaching it raises
an error rather than silently accepting an incomplete search.

Rows, frames, and photons remain aligned. Missing photons become unit weights.
2D data are padded to 3D with z=0. Split files retain `Came_from_image`
(**1-based**), `addonarray`, `Parameters_to_split`, and the three cut arrays;
these metadata survive the existing Python checkpoints. Input arrays are not
modified. The same Python seed is reproducible; MATLAB and NumPy seeds do not
produce the same random stream.

Operational fixes relative to MATLAB are explicit: images at or below the
target stay whole (avoiding the equality-case duplication), small truth arrays
are padded correctly, empty images are retained, malformed aligned data are
rejected, and integer frame differences are computed in floating point.
The older source-faithful MCMC API and its dtype policy are unchanged.

Distances are computed in blocks (default 512 rows) instead of allocating a
full pair matrix or recomputing it for every lag. Work is still quadratic in
the number of localizations per image. Use split regions for large datasets;
`--block-size` adjusts pairwise working memory. The new preprocessing commands
run sequentially; `--workers` remains an option of `run`/`resume`.

## Combining corrected regions

```powershell
ddc -combine "analyzed.mat"
ddc combine "analyzed.mat" --output "results\combined.mat"
ddc inspect "results\combined.mat"
```

Combining reads `Final_Loc_Blinking_Corrected` and
`Final_Frame_Blinking_Corrected`, removes the overlap buffers using the three
unbuffered cut arrays, and concatenates the retained rows according to
`Came_from_image`. The input must therefore be an analyzed **split** dataset,
with all region cells and contiguous source-image IDs starting at 1. Output
coordinates stay in the original global coordinate system. There is no
translation, frame sorting, or re-estimation of localizations.

Both MATLAB Level-5 and MATLAB v7.3/HDF5 analyzed files are supported, including
historical MATLAB results and current Python checkpoints. Only fields needed
for combining are read, avoiding large MCMC probability caches. Outputs are
Level-5 MAT files with the same two final-result cell names as MATLAB, one cell
per original image, plus boundary-mode and source-file provenance. Frames are
column vectors on disk. Empty images are retained. No stale region metadata
or MCMC continuation state is written into the combined result.

The default `--boundary-mode partition` gives shared boundary points to the
first region containing them, independently for each original image. It
handles 2D and 3D faces, edges, corners, and repeated quantile cuts. Distinct
input detections with identical coordinates are retained; the algorithm assigns
geometric ownership rather than deduplicating coordinate rows.

For exact source comparisons, use:

```powershell
ddc -combine "analyzed.mat" --boundary-mode matlab
```

This compatibility mode intentionally retains the source's global `Cutsstore`,
comparison of **sums** of corner-coordinate differences, inclusive z boundaries,
and replacement of z bounds with `[-1000,1000]` when the upper z cut is zero.
Those rules can lose boundary detections across separate images and can retain
duplicates across z boundaries. The default partitions actual core boxes and
avoids those source defects. Both modes preserve region order and row order
within each retained region.

When `Step` and `stepper` are available, unfinished analyses are rejected by
default. To intentionally export a current intermediate result, use
`--allow-incomplete`. This requires corrected data for each ordinary region;
missing corrected outputs are never silently replaced with raw data.
Completed regions that DDC skipped because they have at most 10 raw
localizations are retained unchanged, with a progress message. An explicit
empty corrected `(0,3)` array stays empty.

## Storage boundary for a later Python format

`combine_images` consumes and returns `DDCDataset` objects holding NumPy arrays;
it does no file I/O. The MAT adapters `load_corrected_regions`,
`save_combined_images`, and `load_combined_images` are separate, in
`src/ddc/combine_io.py`. A later NPZ/pickle adapter can exchange the same arrays
without changing the combining algorithm. That future migration must retain
cell/region order, variable-length arrays, coordinate/frame dtypes, and all
split metadata, with a separate contract for resumable MCMC checkpoints.

## Python API

```python
import numpy as np
from ddc import (
    load_dataset, save_dataset, split_images, determine_n,
    determine_resolution, inspect_resolution,
)
from ddc.tool_plots import plot_n, plot_resolution, plot_split

dataset = load_dataset("input.mat")
regions = split_images(dataset, rng=np.random.default_rng(123))
save_dataset(regions, "Split_input.mat")
plot_split(regions).savefig("split.png")

diagnostic = determine_n(regions, max_frame=1000, gap=10)
plot_n(diagnostic).savefig("choose_N_f.png")
# Inspect the curve before setting this example value.
n_f = 200
estimate = determine_resolution(regions, n_f=n_f)
print(estimate.resolution)
plot_resolution(estimate).savefig("resolution_search.png")
check = inspect_resolution(regions, n_f=n_f, resolution=estimate.resolution)
plot_resolution(check).savefig("resolution_check.png")
```

Combine an analyzed file and read the combined output entirely in Python:

```python
from ddc import (
    combine_images, load_corrected_regions, save_combined_images, load_combined_images,
)

corrected_regions = load_corrected_regions("analyzed.mat")
combined = combine_images(corrected_regions)
save_combined_images(combined, "combined.mat")
reloaded = load_combined_images("combined.mat")
locations = reloaded.localizations[0]
frames = reloaded.frames[0]
```

For an already completed in-memory `run_ddc` result with all corrected cells
present, construct `DDCDataset(result.final_localizations, result.final_frames,
came_from_image=regions.came_from_image, cuts=regions.cuts)` and pass it directly
to `combine_images`; no MAT round trip is required. The file adapter also handles
the skipped-region convention described above.

`load_dataset` retains the existing raw Level-5/pre-v7.3 MAT contract.
`save_dataset` writes that input format atomically; checkpoints continue to use
the existing checkpoint writer and `load_checkpoint`.

## Validation

The tests compare blocked calculations with independent full pair-distance
calculations, verify boundary predicates and aligned rows, cover 2D/3D and
small/empty regions, and exercise multiple-file CLI output and a real DDC run.
The developer-only `tests/matlab/generate_preprocessing_reference.m` evaluates
the computation sections of the original MATLAB scripts without interactive
dialogs. Saved references check N_f, both resolution calculations, and split
results using replayed MATLAB random draws. These checks do not make MATLAB
a runtime dependency of the Python tools.

Combining tests cover geometric boundary ownership, repeated coordinate rows,
separate images, empty regions, skipped/incomplete results, both MAT versions,
and the complete split -> run -> combine CLI sequence. The developer-only
`tests/matlab/generate_combine_reference.m` runs the original MATLAB combining
computation on a boundary-heavy fixture to verify the `matlab` mode.

Combining update validation on 2026-10-19: 69 focused tool/CLI tests passed,
including the MATLAB combining reference. Another 67 pipeline/serialization
tests passed, six were skipped, and the existing test that looks for MATLAB
source under the old `Main_DDC_Folder` path still failed. A real v7.3 smoke test
combined the bundled historical 30-region result into one image with 24,401
localizations using the default partition rule. Because that historical file
has unfinished regions, this development check explicitly used
`--allow-incomplete`; its snapshot is
`results/preprocessing_smoke/Combined_legacy_snapshot.mat`.

Validation on 2026-10-19: all 30 new preprocessing tests passed, including the
MATLAB R2025b numerical references and the split-to-DDC CLI smoke test.
The complete suite reported 386 passed, 19 skipped, and the five previously
documented fixture-path failures (old tests expect `Main_DDC_Folder` directly
under the project instead of under `DDC-main`). A separate smoke run on the
first 4000-localization image of the bundled 2D simulation produced the N_f
curve, an automatic 70 nm recommendation with the example N_f=200, and 18
regions with a 1000-localization target. Those values are validation results,
not recommendations for other datasets. Preview files are under
`results/preprocessing_smoke`.
