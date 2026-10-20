# Stage A.8 — in-memory pipeline integration

Stage A.8 represents the active numerical orchestration of
`run_script_DDC.m` in `src/ddc/pipeline.py`. It consumes an already loaded
`DDCDataset` and returns `DDCPipelineResult`; it performs no file discovery,
checkpoint writing, MAT serialization, plotting, UI, CLI, parallel execution,
or performance optimization.

## Source classification

### A. Active numerical/in-memory behavior ported now

- `TrueLocalizations=[]`, later expanded to one MATLAB `[]` value per image
  when no truth was loaded.
- `addonarray(500)=0`; loaded addon values are used when present, otherwise a
  private zero vector supplies the active image range. `addon` remains inactive
  inside A.7b.
- `Photons={[]}` and first-cell-gated unit-weight initialization.
- defaults `Resolution=80`, `stepper=100`, `N_f=200`,
  `Photon_weighted_Correction=1`, and `timer=3600*.2=720` seconds.
- source 2D-to-3D assignment controlled by `min(size(X))<3`.
- one shared `Determine_Blinking_Distribution5` call before MCMC processing.
- one persistent state record per image, sequential equivalent of the source
  `parfor` body, `while sum(Step<stepper)>0`, eligibility, continuation
  writeback, improved-maximum replacement, and optional photon collapse.

### B. Loading already represented by A.1

`Run_cluster.m` contributes no numerical transformation after `load`. MAT
decoding, MATLAB cell normalization, dtypes and image correspondence remain the
responsibility of `load_dataset`. The A.8 integration audit found that active
MATLAB permits a missing `TrueLocalizations` variable: it is preinitialized to
`[]` before `load`. A.1 previously required the variable. The minimal A.8
correction makes it optional while retaining strict requirements for
`LocalizationsFinal` and `Frame_Information`; missing truth becomes aligned
empty coordinate cells. At the pipeline boundary those empty normalized cells
are converted back to MATLAB `[]` so `DDC_MCMC`'s `length` test stays disabled.

Stage A.8b also restores `Photons` at this boundary. `DDCDataset.photons` is
either `None` when the variable is absent or one flattened 1-D array per
MATLAB cell in column-major cell order. Row and column vectors use the same
per-image policy as frame vectors; empty cells and original numeric classes
are preserved. No integer-to-float conversion and no localization-count check
occurs during loading. Real User Guide fixtures
`Example_3d_2darkstate_random_data.mat` and
`Split_Simulation_2_darkstate_random_data.mat` contain integer Photons;
the readable non-v7.3 example was observed as `uint8`. A present numeric empty
`TrueLocalizations=[]` is now accepted like absent truth, while a nonempty
non-cell truth array remains invalid.

### C. Deferred output behavior

Both `save([String])` and `save([String],'-v7.3')`, date-based output naming,
sparse checkpoint representation, HDF5/MAT compatibility and checkpoint
reloading are deferred to A.9. `DDCPipelineResult` retains every numerical
array required for that later serialization. The MATLAB `sparse` calls change
storage representation only; Python keeps `ProbabilityCache` dense without
changing values.

### D. Deferred interfaces and execution infrastructure

`cluster`, `pwd`, `addpath`, `parpool`, hardcoded cluster filenames,
`uigetfile`, cancellation handling, `parfor` scheduling, multiprocessing,
progress UI and CLI behavior are not ported. Stage A.8 processes images
sequentially; there are no intended cross-image numerical dependencies beyond
the shared distribution inputs.

### E. Display-only statements

Comments, warnings printed for `length(LocalizationsFinal{i})>8000`, and other
status text have no numerical effect. Python records the >8000 warning in
`diagnostics` but does not block the image.

## API and state mapping

`DDCConfig` exposes the five source defaults and makes them configurable.
`run_ddc(dataset, config=..., photons=..., rngs=...)` is the only pipeline
entry point. No path or file-picker argument exists.

Each `DDCImageState` maps:

| MATLAB | Python |
|---|---|
| `Step(i)` | `step` |
| `Constantf{i}` | `constantf` |
| `Constantf2{i}` / returned `Constant2` | `constant2ft` |
| `Orderf{i}` | `orderf` |
| `Lik{i}` | `likelihood` |
| `RelScore{i}` | `score` |
| `Numb_of_Loc{i}` | `number_of_localizations` |
| `Prob_dists{i}` | `probability_cache` |

All start empty and `Step=0`, exactly once. Every A.7b return writes all fields
back before the next outer pass. Thus a nonempty `Orderf` and populated cache
select A.7b continuation behavior rather than recomputation.

`DDCPipelineResult` contains final localization/frame/trajectory cells, all
per-image states and histories, padded working input, frames, truth, photons,
shared bins, blink distribution, returned resolution, `X_overall`, `M_mat`,
and diagnostics. It is an in-memory checkpoint model, not a serializer.

## Preprocessing and MATLAB length semantics

The literal assignment is:

```matlab
if min(size(X))<3
    X(:,3)=X(:,2)*0;
end
```

Therefore ordinary `n×2` input becomes `n×3` with zero z; ordinary `n×3` with
`n>=3` is unchanged. `1×3` and `2×3` also satisfy the condition and have their
existing third column overwritten with zero. `1×2`, `2×2`, and `0×2` become
three-column matrices. A matrix with fewer than two columns reaches the
source's invalid `(:,2)` access. These small-shape effects are source quirks,
not BUG-006, because repository-supported localization matrices have many rows
and two or three columns. Caller arrays are copied and never mutated.

Eligibility and the >8000 diagnostic use
`length(X)=max(size(X))`, not row count. Exactly 10 is ineligible and sets
`Step=stepper`; exactly 11 is analyzed. Consequently an unusual `2×11` matrix
is eligible even though it has only two localization rows. This is preserved.

## Shared distribution and uint16 failure order

`determine_blinking_distribution` executes exactly once after padding and
before state initialization/MCMC image calls. Its `bins`,
`Distribution_for_Blink`, returned `Resolution`, `X_overall`, and `M_mat` are
shared by every image and every continuation chunk.

Frames are not converted. The pipeline passes source-equivalent
`round(Frame_Information{i})`; NumPy preserves `float32`, `float64`, and integer
frame dtype here. Repository `uint16` therefore reaches
DDC-SOURCE-BUG-002 in shared distribution estimation before `run_mcmc` can
reach DDC-SOURCE-BUG-001. A pipeline regression asserts that `run_mcmc` was
never called. BUG-001 remains independently reachable through direct A.7b use.

## Outer loop, continuation and replacement

The sequential loop preserves `while sum(Step<stepper)>0`. An image runs only
when MATLAB length is greater than 10 and its step is below `stepper`; otherwise
it is permanently assigned `stepper`. No artificial iteration limit is added.
A nonprogressing mocked or source MCMC can therefore keep the outer loop alive,
as MATLAB can.

Immediately before a chunk, `maxtemp=max(old Lik)` or `-Inf` for an empty
history. After writeback, final output cells are replaced only when
`maxtemp < max(new LikHood)`. A first improvement stores output, a higher later
maximum replaces it, and a chunk whose maximum merely equals the old maximum
does not. Accepted downhill history entries alone do not trigger replacement.
NaN follows the literal floating comparison and prevents replacement.

The A.7b `steps=10` reset is not altered. If it leaves `Step<stepper`, the image
is revisited on the next outer pass with its complete written-back state.

## Photon semantics

An explicit non-`None` `photons=` argument is an intentional Python override.
Otherwise `run_ddc` uses `dataset.photons`; only when that field is `None` does
it enter the source empty-Photons path. If photons are absent, the collection
is empty, or only the first supplied/loaded cell is empty, the source
first-cell gate replaces every image with a column-equivalent `double` vector
of ones. If the first cell is nonempty, supplied later cells are used as-is;
missing cells fail rather than being silently filled. Values are not clipped or
validated, so negative, NaN and Inf weights propagate through source
arithmetic.

Photon correction runs only after a new pipeline-level maximum. It iterates
through sorted `unique(Traj2)`, uses the padded original `LocalizationsFinal`
and `Photons`, normalizes weights by their trajectory sum, and computes x/y/z
weighted sums. Frames use the unweighted arithmetic mean of original
`Frame_Information` and are not rounded.

MATLAB default `sum(integer)` returns `double`, but integer-array division by
that scalar returns the original integer class using nearest rounding with
ties away from zero and saturation. Consequently `uint8([1 3])` and
`uint16([1 3])` normalize to integer `[0 1]`; two unit `uint8` values normalize
to `[1 1]`, while three units normalize to `[0 0 0]`. Integer zero-sum all-zero
vectors become integer zeros because MATLAB integer conversion maps NaN to
zero. Floating zero sums retain NaN/Inf behavior. `single` normalization stays
single and `double` stays double.

Repository coordinates are normally `double`. A nonscalar integer-weight
vector cannot be multiplied by a nonscalar double-coordinate vector, and the
faithful path raises at that operation. A scalar double coordinate times an
integer scalar succeeds in the integer class, including rounding and
saturation before `sum` returns double; `single` coordinates do not qualify
for this scalar-double exception.

Floating mixed-class arithmetic is different: MATLAB supports `double .* single`
and `single .* double`, and both the product and default `sum(product)` are
`single`. Python therefore performs the elementwise product and reduction in
float32 whenever either floating operand is single, preserving intermediate
single-precision rounding. Double/double remains double throughout. This
distinction corrects the A.8b Python helper and does not define a new source
defect. The integer behavior establishes DDC-SOURCE-BUG-006:
repository `uint8` Photons do not undergo ordinary floating normalization and
can change weighting or make the subsequent source multiplication fail.
Static semantics are confirmed; the MATLAB runtime oracle remains pending.
When correction is disabled, photon normalization is bypassed and the raw
A.7b best coordinates and frames are stored.

## RNG policy

No global NumPy RNG is used. Callers may inject one RNG-like object per image.
If omitted, a fresh `SeedSequence` spawns independent child generators, so
image 2's stream never depends on how many draws image 1 consumes. Sequential
execution and child streams do not claim bit-identical MATLAB `parfor` RNG
behavior. Tests explicitly inject independent generators and verify absence of
cross-image draw coupling.

## Defect audit and validation scope

DDC-SOURCE-BUG-006 is assigned for repository integer-photon normalization and
its downstream mixed-class multiplication. Newly documented behaviors—small-shape
third-column overwrite, photon first-cell gating, zero-weight propagation,
empty-truth boundary normalization, and possible nonprogressing outer loops—are
faithful quirks or integration boundaries. BUG-001 through BUG-005 remain
unchanged.

The real-component smoke fixture uses one 23-localization, double-frame image
with frame groups 1, 2 and 4, `N_f=2`, `Resolution=10`, and a controlled
per-image RNG. It executes real `run_ddc`, shared distribution estimation,
A.4 probability initialization, A.7b MCMC, likelihood, state writeback and
photon collapse. It asserts completion at `Step=10`, cache/result shapes and no
input mutation; it does not claim MATLAB/NumPy stochastic trajectory identity.

`tests/matlab/generate_pipeline_reference.m` records deterministic padding,
truth/photon initialization, cell-state shapes, eligibility boundaries,
photon collapse, replacement comparisons and a supported double-frame shared
distribution. MATLAB runtime remains unavailable locally, so the generated
`.mat` oracle is intentionally absent and its test skips explicitly.
