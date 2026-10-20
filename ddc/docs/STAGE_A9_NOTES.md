# Stage A.9 — MATLAB checkpoint serialization and resume

Stage A.9 adds checkpoint/output I/O only. It does not add a CLI, file picker,
GUI, plotting, multiprocessing, `parfor` replacement, optimization, or Stage B
behavior. Approved A.1–A.8c numerical operations remain unchanged.

## Source trace and save timing

`run_script_DDC.m` constructs `String` at line 84, computes the shared blinking
distribution at lines 105–107, initializes all continuation cells and `Step=0`
at lines 117–138, and executes `save([String])` at line 140. Thus the first
checkpoint exists before any MCMC call but already contains `bins`,
`Distribution_for_Blink`, resolved `Resolution`, `X_overall`, and `M_mat`.

The main `while sum(Step<stepper)>0` begins at line 148. The `parfor` performs
complete per-image state and best-output writeback at lines 175–209. Only after
the whole `parfor` completes does line 217 execute `save([String],'-v7.3')`.
Every outer pass overwrites the same filename with a MATLAB v7.3 workspace.

Both `Scripts_Functions/Finish_Runs.m` and the cluster copy load an analyzed
checkpoint and do not recalculate the blinking distribution. However, they
cannot be described as a working continuation of the current source: both use
an obsolete eleven-output `DDC_MCMC` assignment. The noncluster copy later uses
a default `save([String])`; the cluster copy uses `-v7.3`. Stage A.9 follows the
active initial-run format sequence and can read both Python-generated v5 and
v7.3 checkpoints; input compatibility is distinct from Finish_Runs
executability.

## Stage A.9b — stale Finish_Runs interface (DDC-SOURCE-BUG-007)

Direct source tracing confirms that current `DDC_MCMC.m` declares twelve
outputs, beginning with the newly added `Traj2`. `run_script_DDC.m` requests all
twelve. Both `Finish_Runs.m` copies still request eleven and omit `Traj2`.
MATLAB assigns the first requested number of declared outputs in order, giving
this exact shift:

| stale Finish_Runs variable | actual current DDC_MCMC output |
|---|---|
| `Final_Localizations_Blinking_Corrected2` | `Traj2` |
| `Final_Frame_Blinking_Corrected2` | `Final_Localizations_Blinking_Corrected` |
| `LikHood` | `Final_Frame_Blinking_Corrected` |
| `Score` | `LikHood` |
| `Numb` | `Score` |
| `bestlik` | `Numb_of_Loc` |
| `steps` | `bestlik` |
| `Constant` | `steps` |
| `Order` | `Constantf` |
| `Prob_dists2` | `Orderf` |
| `Constant2` | `Prob_dists` |

Actual output 12, `Constant2ft`, is discarded. If an unfinished image has
MATLAB `length(LocalizationsFinal{i})>10` and `DDC_MCMC` returns, the first
observable post-return failure is dot-field access on numeric `Orderf` through
`Prob_dists2.Deviation_in_Probability`. MATLAB normally reports this class of
failure as dot indexing unsupported for that variable type; the exact runtime
identifier/message remains oracle-pending. Repository `uint16` frames can fail
earlier inside `DDC_MCMC` through BUG-001/002/003 (with BUG-004 also affecting
behavior), masking BUG-007. Controlled floating frames expose BUG-007.

The complete drift audit also found that both resume scripts fail to request or
update `Trajectory_of_Localizations`, omit the initial runner's photon-weighted
correction block, and warn above 7000 localizations whereas `run_script_DDC.m`
uses 8000. The local resume saves in the default MAT format while the cluster
copy saves v7.3. No additional modern `DDC_MCMC` input was missing: the fatal
drift is confined to the output contract and downstream writeback behavior.

Three concepts are intentionally separate:

1. The checkpoint contract contains the state variables read by Finish_Runs.
2. The current MATLAB Finish_Runs cannot successfully consume an eligible
   supported checkpoint through the modern `DDC_MCMC` output contract.
3. `resume_ddc` is a corrected operational Python continuation over saved state,
   using the current Stage-A MCMC API. It is not a literal port of Finish_Runs.

`resume_finish_runs_source` is the narrow source-faithful diagnostic API. It is
a no-op for completed checkpoints, applies MATLAB `length=max(size)` and marks
unfinished images of length at most ten complete, reports earlier integer-frame
defects separately, and raises `DDCSourceBug007` for an eligible floating-frame
path. It never substitutes corrected MCMC execution. The serialization formats,
sparse boundary conversion and atomic-write design are unchanged by A.9b.

## Workspace classification

Required continuation/scientific state comprises preprocessed localization,
frame, truth and photon cells; configuration; shared distribution arrays;
`Constantf`, `Constantf2`, `Orderf`, `Step`, histories and `Prob_dists`; and the
three best/final output cells. Final localization/frame cells and trajectories
are scientific outputs. Shared distribution and per-image fields are both
scientific intermediates and resume state.

MATLAB `load(full_filename)` imports every source MAT variable, and bare
`save([String])` saves the whole current workspace. Consequently split-image
metadata and arbitrary source variables survive even when DDC does not inspect
them. `DDCDataset.mat_passthrough` retains unrecognized variables without dtype
conversion, excluding SciPy `__header__`, `__version__`, `__globals__` and all
variables already represented by typed fields. Algorithms never depend on this
dictionary. Unsupported arbitrary objects cause serialization to fail rather
than being silently discarded.

`filename`, `pathname`, and `full_filename` are reconstructed only when
`dataset.source` exists or metadata is explicitly supplied. `cluster=0` is a
reconstructable source default. No fake `Location`/working directory is
invented. Loop indices, file-picker state, pool handles and UI/path setup are
transient source artifacts and are not required for resume.

## Typed checkpoint model

`DDCCheckpoint` contains a typed `DDCDataset`, `DDCConfig`, complete
`DDCPipelineResult`, analyzed/source filename metadata, phase, and optional
source metadata. It is not a giant untyped workspace dictionary. The disk
mapping is dictionary-like only at the MAT boundary.

| Python state | MATLAB variable |
|---|---|
| `result.working_localizations` | `LocalizationsFinal` |
| `result.frames` | `Frame_Information` |
| `result.true_localizations` | `TrueLocalizations` |
| `result.photons` | `Photons` |
| `config.resolution` / resolved result value | `Resolution` |
| `config.stepper` | `stepper` |
| `config.n_f` | `N_f` |
| `config.photon_weighted_correction` | `Photon_weighted_Correction` |
| `config.timer` | `timer` |
| `result.bins` | `bins` |
| `result.distribution_for_blink` | `Distribution_for_Blink` |
| `result.x_overall` | `X_overall` |
| `result.m_mat` | `M_mat` |
| `state.constantf` | `Constantf{i}` |
| `state.constant2ft` | `Constantf2{i}` |
| `state.orderf` | `Orderf{i}` |
| `state.step` | `Step(i)` |
| `state.likelihood` | `Lik{i}` |
| `state.score` | `RelScore{i}` |
| `state.number_of_localizations` | `Numb_of_Loc{i}` |
| `state.probability_cache` | `Prob_dists{i}` |
| `result.final_localizations` | `Final_Loc_Blinking_Corrected` |
| `result.final_frames` | `Final_Frame_Blinking_Corrected` |
| `result.trajectories` | `Trajectory_of_Localizations` |
| analyzed filename | `String` |
| source basename | `filename` |

`Prob_dists{i}` has exactly `Deviation_in_Probability`,
`Prob_Distributions`, and `Dscale_store`. Python cache field names never appear
as primary MATLAB names.

## Cell orientation and dtypes

The source creates its cells by assigning `{ijk}`, producing MATLAB 1×N cells.
MATLAB v7.3 reverses dimensions in HDF5, so the historical file exposes these
as N×1 reference datasets. Stage A.9 writes MATLAB 1×N cells and therefore
matches the historical N×1 HDF5 structure. Per-image frame, photon, state,
history, final-frame, and trajectory vectors are MATLAB rows, consistent with
the repository input and historical checkpoint. Localization matrices remain
n×dimensions.

Source numeric dtypes are preserved for frames, photons, localizations,
split-image metadata, and passthrough arrays. Python `Orderf` is stored as
MATLAB double because MATLAB `randperm`/index order in the source is double.
NaN and Inf are retained. Empty truth and output cells remain MATLAB empties;
trailing never-assigned output cells are not invented on disk, while loading
expands them to aligned Python `None` slots for continuation.

## File formats and dependencies

The initial save uses SciPy `savemat(format='5')`, an ordinary pre-v7.3
MATLAB Level-5 file readable by `scipy.io.loadmat`. It contains `Step==0`, empty
continuation histories/caches, initialized output cells, and the completed
shared blinking distribution.

Iterative checkpoints use `hdf5storage` with `fmt='7.3'`, Python metadata
disabled and MATLAB-incompatible values configured as errors. SciPy cannot
write v7.3. `hdf5storage` supplies the MATLAB cell/struct/char schema and HDF5
user block. It does not write nested SciPy sparse matrices, so the writer uses
`h5py` only after schema generation to replace the two source-declared sparse
fields with the exact historical MATLAB CSC group representation:
`data`, `ir`, `jc`, `MATLAB_class='double'`, and `MATLAB_sparse=<row count>`.
`Dscale_store` remains dense. Both dependencies are explicit because the code
directly imports each; neither is used for numerical computation.

The historical
`Analyzed_Time15-Nov-2020_Split_Example_3d_2darkstate_random_data.mat` was
inspected read-only. Its top-level names, MATLAB class attributes, N×1 HDF5
cell orientation, char storage, scalar shapes, struct fields and sparse groups
form the structural reference. It is never modified or packaged.

## Callback and atomic replacement

`run_ddc(..., checkpoint_callback=...)` emits a detached snapshot once at
phase `initial` after distribution/state initialization and once at phase
`outer` after every complete image pass. Callback mutation cannot touch live
pipeline state. Regression tests show identical seeded numerical output and
identical subsequent RNG draws with and without a callback.

`CheckpointWriter` writes v5 at `initial`, then atomically replaces the same
path with v7.3 at each `outer`. Each write uses a temporary sibling and
`os.replace`; successful writes leave no temporary file, and a simulated
writer failure preserves the previous checkpoint. This is Python I/O safety,
not a source numerical change.

## Reload and resume

`load_checkpoint` detects MATLAB v5 versus HDF5/v7.3, reconstructs typed input,
configuration, shared arrays, every `DDCImageState`, dense in-memory
`ProbabilityCache`, best outputs, photons, split metadata and passthrough. It
does not require the original input MAT. The loader's supported contract is
Python-generated Stage A.9 checkpoints; the historical MATLAB file is used for
structural testing rather than claimed as a fully generic v7.3 import.

`resume_ddc` supplies saved working inputs, distribution, states and outputs to
`run_ddc(resume_from=...)`. It never calls
`determine_blinking_distribution`, never resets `Step` or histories, and passes
saved probability caches into MCMC. A completed checkpoint performs no MCMC
work. A controlled interrupted-save-load-resume regression equals a continuous
run for Step, likelihood histories, best outputs, constants, ordering and
probability caches under equivalent Python RNG streams. No MATLAB RNG identity
is claimed or stored, matching the source checkpoint contract.

## Filename and compatibility status

`make_analyzed_filename` reproduces
`Analyzed_TimeDD-Mmm-YYYY_<original filename>` with English MATLAB month
abbreviations. For 2020-11-15 and `example.mat`, the result is
`Analyzed_Time15-Nov-2020_example.mat`; the original `.mat` is not duplicated.

`tests/matlab/probe_checkpoint_compatibility.m` loads a Python-generated v7.3
file, checks required variables/cells/struct fields, verifies the two sparse
fields and dense `Dscale_store`, and verifies that the complete Finish_Runs
input state exists without running MCMC. MATLAB runtime is unavailable in the
development environment, so runtime MATLAB readability remains explicitly
pending; HDF5/hdf5storage structural checks alone are not presented as runtime
proof.

DDC-SOURCE-BUG-007 is confirmed by static source tracing; its MATLAB runtime
oracle remains pending. Whole-workspace passthrough, date-dependent naming,
initial v5 followed by v7.3 overwrite, sparse boundary conversion and
path/UI artifacts are source behaviors or interface details, not newly
established MATLAB execution defects.
