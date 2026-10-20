# Known defects in the MATLAB reference implementation

## DDC-SOURCE-BUG-007 — Finish_Runs uses the obsolete pre-Traj2 output signature

**Status:** static source behavior confirmed; MATLAB runtime oracle pending.

The active `Scripts_Functions/DDC_MCMC.m` declares twelve outputs and now puts
`Traj2` first. `run_script_DDC.m` was updated to request all twelve, but both
the local and cluster `Finish_Runs.m` copies still request eleven outputs using
the old list and omit `Traj2`. MATLAB assigns the first eleven declared outputs
in order, so every stale variable shifts left by one: `Prob_dists2` receives
numeric `Orderf`, `Constant2` receives the probability struct, and the twelfth
output `Constant2ft` is never captured.

After a successful `DDC_MCMC` return, Finish_Runs immediately evaluates
`Prob_dists2.Deviation_in_Probability`. Dot-field access on numeric `Orderf`
therefore fails. The path requires an unfinished image with MATLAB
`length(LocalizationsFinal{i})>10`; ineligible images are simply marked
complete. Earlier documented integer-frame defects can fail inside DDC_MCMC
first for the repository fixtures and mask BUG-007, while supported floating
frames make this stale-signature defect independently reachable.

The Python `resume_ddc` API intentionally remains a corrected operational
continuation using the modern Stage-A interface. The separate
`resume_finish_runs_source` API represents the defective source control flow;
the distinction prevents a useful repair from being mislabeled as literal
MATLAB equivalence. The MATLAB source itself is not modified.

## DDC-SOURCE-BUG-001 — unsigned frame subtraction can address `DistanceMatrix{0}`

**Status:** confirmed source defect in the upstream MATLAB implementation; not
a defect introduced by the Python port.

All input MAT fixtures included in the DDC repository store each
`Frame_Information` cell as `uint16`. Every active MCMC entry point passes
`round(Frame_Information{ksu})` to `DDC_MCMC`; this preserves the integer class.
`DDC_MCMC` only sorts and reindexes the frame array before passing it to
`Make_Distance_Matrix_Individual`, so the function receives `uint16` unchanged.

After sorting, for two unequal ascending frames, MATLAB's saturating unsigned
subtraction makes `frame(i)-frame(jj)` equal zero while
`frame(jj)-frame(i)` is positive. The source then attempts:

```matlab
DistanceMatrix{abs(frame(i)-frame(jj))}(end+1) = ...
```

which becomes the invalid 1-based subscript `DistanceMatrix{0}`. External data
using `double` or a suitable signed frame type can avoid the defect, but this
does not change the effective path for the repository's own fixtures.

The Stage-A port deliberately preserves source dtype on load and reproduces
this failure. It must not be hidden through a global frame cast or a silent
change in bin semantics. A later compatibility policy may expose an explicit,
documented workaround, but that is outside the faithful Stage-A port.

## DDC-SOURCE-BUG-002 — `pdist` receives an unsupported `uint16` matrix

**Status:** confirmed by authoritative static MATLAB type and API semantics;
runtime oracle pending because MATLAB is unavailable in the porting environment.

`Determine_Blinking_Distribution5.m` constructs frame points three times as:

```matlab
[[1:length(Frame_Information{i})]'*0, Frame_Information{i}(:)]
```

The left column is `double`, while repository `Frame_Information` cells are
`uint16`. MATLAB's documented concatenation precedence converts a mixture of
integer and double values to the integer class, so this expression is a
two-column `uint16` matrix. Statistics and Machine Learning Toolbox documents
the `X` input of `pdist` as accepting only `single` or `double`. The 2020 source
contains no explicit conversion before any of these calls.

Consequently, the repository's included inputs can fail at the first frame
`pdist` call rather than produce `Z2`. This defect is distinct from
DDC-SOURCE-BUG-001: bug 001 is saturating scalar subtraction; bug 002 is an
unsupported input class at a toolbox boundary.

The faithful Stage-A Python path therefore rejects integer frame arrays at this
operation. It does not cast them to float. External datasets whose frame cells
are already `single` or `double` proceed normally. A minimal MATLAB oracle is
provided in `tests/matlab/probe_uint16_frame_pdist.m` to record `class(X)` and
the `pdist` outcome when a MATLAB runtime is available.

## DDC-SOURCE-BUG-003 — integer frame matrices reach unsupported `mtimes`

**Status:** confirmed from the active source expression and MATLAB's documented
`mtimes` restriction; runtime oracle pending because MATLAB is unavailable.

`Eliminate_Blinking_De_Loc15_MCMC.m` preserves the class of `frame` when it
constructs `AAAA=[Frame(iit), Frame(iit)*0]` and `BBBB`. With repository
`uint16` frames these are integer matrices. Line 79 evaluates
`AAAA*BBBB'`. MATLAB documents that if either `mtimes` operand is integer, the
other operand must be scalar; here the operands are nonscalar even for one
localization (1-by-2 and 2-by-1).

The full repository path reaches BUG-002 first in
`Determine_Deviation_in_Probability8`, so BUG-003 is masked there. BUG-003 is
independently reachable when the eliminator is invoked directly with
repository-typed frames, or if execution is made to continue past BUG-002
without converting frames. The Stage-A port raises `DDCSourceBug003` only when
the integer matrix-product expression is reached; it does not globally cast or
reject integer frames.

## DDC-SOURCE-BUG-004 — asymmetric density from unsigned frame subtraction

**Status:** active for the repository's supported `uint16` frame arrays;
runtime oracle pending because MATLAB is unavailable.

`Density_Calc.m` evaluates, for every ordered pair `(i,j)`:

```matlab
abs(Frame_Information(i)-Frame_Information(j)) > A
```

MATLAB performs the scalar subtraction in the `uint16` class and saturates a
negative result to zero before `abs` is called. `abs` preserves the integer
class and cannot recover the lost sign. Thus for ascending unequal frames the
`i,j` direction can report zero while `j,i` reports the positive difference.
Since the script counts neighbors into `density(i)`, spatially symmetric pairs
can contribute asymmetrically. This is distinct from BUG-001: the same unsigned
arithmetic root causes an invalid cell index there, but here it silently changes
a DDC_MCMC input. Stage A reproduces the ordered-pair asymmetry without casting
frames to floating point.

## DDC-SOURCE-BUG-005 — intermediate branch-3 state can leak through `Constant2ft`

**Status:** confirmed directly from the active `DDC_MCMC.m` assignments and
reachable with normal floating-frame data.

At the start of an outer MCMC iteration, MATLAB copies `Constant2ft` into the
working `Constant2`. Inside every `vvse==3` proposal attempt with `steps>1`, it
then executes:

```matlab
Constant2ft = Constant2;
```

before perturbing `Constant2`. MATLAB assignment has value semantics, so the
first branch-3 attempt normally snapshots the accepted value and is harmless.
If that attempt leaves `loc` unchanged, however, the enclosing
`while isequal(loc,loc2)` retries. A second branch-3 attempt first copies the
already modified intermediate `Constant2` into persistent `Constant2ft`. If a
later conformation is finally different but fails Metropolis acceptance,
`Constantf` and `Orderf` remain at their accepted values while `Constant2ft`
retains this intermediate, unaccepted snapshot. The next outer iteration
therefore starts `Constant2` from leaked proposal state.

A single branch-3 attempt followed by rejection does not leak its newly
perturbed value: its pre-perturbation snapshot still equals the iteration's
starting `Constant2ft`. The defect requires repeated inner proposal attempts,
with branch 3 selected after `Constant2` has already changed. Stage A.7b
preserves the pre-acceptance copy rather than enforcing rollback. A controlled
full-driver regression and `tests/matlab/probe_mcmc_constant2ft_leak.m`
demonstrate the behavior.

## DDC-SOURCE-BUG-006 — integer Photons retain integer normalization semantics

**Status:** confirmed from repository inputs and authoritative static MATLAB
numeric-class semantics; MATLAB runtime oracle pending.

Repository User Guide MAT inputs contain `Photons` cells stored as `uint8`.
The active source evaluates `weights=weights/sum(weights)` without conversion.
For integer input MATLAB's default `sum` returns a scalar `double`, while
division of an integer array by a scalar double returns the original integer
class, rounded to nearest with ties away from zero and saturated to the class
range. Thus `uint8([1 3])/sum(...)` is `uint8([0 1])`, not floating
`[0.25 0.75]`; `uint8([1 1])` becomes `[1 1]`, and three unit weights become
`[0 0 0]`. This can change the effective weighting compared with ordinary
floating normalization.

The following coordinate operation has another class consequence. Repository
coordinates are normally nonscalar `double` vectors, while the normalized
weights remain integer. MATLAB mixed integer/double arithmetic permits the
double operand only when it is scalar, so a multi-localization trajectory
can fail at `LocalizationsFinal(mask,column).*weights(:)`. For a one-element
trajectory the scalar operation succeeds in the integer class, including
rounding and saturation before the default `sum(integer)` returns double.
This restriction is integer-specific: floating double/single arithmetic is
supported and returns a single product and single sum.

Stage A reproduces these class-specific values and failures; it does not cast
integer Photons to floating point. The probe
`tests/matlab/probe_photon_integer_normalization.m` records both normalization
and coordinate multiplication when MATLAB becomes available.
