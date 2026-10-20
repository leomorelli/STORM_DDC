# Stage A.7 — `DDC_MCMC.m`

Stage A.7 ports the active MCMC driver to `src/ddc/mcmc.py`. It deliberately
does not port `run_script_DDC.m`, cluster/file orchestration, plotting, CLI
behavior, persistence, or Stage-B optimizations.

## Call contract

`run_mcmc` preserves the MATLAB argument order and adds keyword-only `rng` and
`clock` dependencies. `rng` accepts `numpy.random.Generator` or the same narrow
method protocol, allowing deterministic controlled tests. `clock` represents
`tic`/`toc` and defaults to a monotonic clock.

| MATLAB input | Python input | Active role |
|---|---|---|
| `ksu` | `ksu` | image identifier used only by the removed display message |
| `LocalizationsFinal` | `localizations_final` | `(n,d)` localization matrix |
| `Frame_Information` | `frame_information` | `n`-element frame vector; dtype preserved |
| `A` | `a` | frame horizon |
| `Resolution` | `resolution` | distance-bin width |
| `TrueLocalizations` | `true_localizations` | optional simulation truth and PALM diagnostic |
| `bins`, `Distribution_for_Blink` | same snake-case names | true/blink distribution construction |
| `steps`, `timer`, `step_total` | same | loop counter, wall-time limit, terminal counter |
| `Constantf`, `Orderf` | snake-case equivalents | accepted proposal parameters |
| `Constant2ft` | `constant2ft` | persistent frame-dependent state; normally accepted, but branch 3 can overwrite it before acceptance (BUG-005) |
| `LikHood`, `Score`, `Numb_of_Loc` | history arguments | histories from an earlier invocation |
| `Prob_dists` | `ProbabilityCache` | deviation, log-probability, and mixture arrays |
| `X_overall`, `M_mat` | snake-case equivalents | optional A.4 cached numerical inputs |
| `addon` | `addon` | inactive source input |

`MCMCResult` exposes all twelve MATLAB outputs and the final `Final_lik`,
`max_lik`, jittered coordinates, `D_Overall`, and aligned `Z2` for equivalence
inspection. MATLAB row/column vectors are normalized to one-dimensional Python
vectors. MATLAB `Orderf` remains externally 1-based; internal order indices are
zero-based.

If `steps >= step_total` on entry, the MATLAB `if` body never assigns several
required output variables. Python represents that source failure explicitly
with `ValueError` rather than manufacturing outputs.

## Top-to-bottom call order

1. Materialize the two potentially sparse arrays in `Prob_dists`.
2. Stable-sort frames and apply the same permutation to coordinates.
3. Call approved `make_distance_matrix_individual`; collect bins containing
   more than 100 distances and whose frame difference is greater than `A`.
4. Trim distribution bins, compute primary and paired-bin true distributions.
5. If `Orderf` is empty, call approved
   `determine_deviation_in_probability8`, then apply `log` to its probability
   matrix and cache all outputs. Otherwise reuse `Prob_dists` unchanged.
6. If MATLAB `length(TrueLocalizations)>1`, match truth by exact x-coordinate,
   construct its PALM image, and calculate the pre-jitter truth likelihood.
7. Calculate pre-jitter `D_Overall`/`Z2`, then recompute `Z2` and run the jitter
   uniqueness loop described below.
8. Compute the initial PALM squared-error denominator when truth is active.
9. Call approved `determine_locs_w_multi2`, then approved `density_calc`.
10. Draw `Traj_num = randperm(length(LocalizationsFinal))`.
11. Initialize a fresh chain, or reconstruct the continuation conformation.
12. Execute `while steps < step_total`, checking elapsed time before increment,
    generating proposals, calling the active eliminator, calling approved
    `calculate_score_8`, and applying acceptance/best-state/history updates.

The plotting-only `scatter`, `colormap`, `axis`, and `drawnow` statements are
omitted. Their position is documented but they neither consume RNG nor alter
the numerical state.

## Active Algorithm-1 dependency

The driver calls `Eliminate_Blinking_De_Loc15.m`, not
`Eliminate_Blinking_De_Loc15_MCMC.m`. These are materially different: the
active driver version accepts `Constants`, returns filtered coordinates,
frames and trajectory labels, and divides its mean probability by
`abs(1+Constants(target))`. A.4 had ported only the `_MCMC` variant. The active
Algorithm-1 loop is therefore represented privately in `mcmc.py`; A.4 code was
not changed or duplicated. Like the source, its frame-derived matrix product
preserves the established integer failure semantics.

## Random operation ordering

For an uncached run the numerical RNG order is:

1. the `randperm` inside A.4 probability initialization;
2. one `rand(size(LocalizationsFinal))` per jitter attempt;
3. `randperm(length(LocalizationsFinal))` for `Traj_num`;
4. per proposal attempt, `randi(3)`;
5. depending on that branch: index `randi`, `randn`, direction `rand`, or a
   repeated-frame selector `randi` followed by `randperm`;
6. only for a downhill Metropolis proposal, the acceptance `rand`.

Fresh runs skip item 1 when a probability cache is supplied. MATLAB and NumPy
are not expected or claimed to produce identical streams from equal integer
seeds. Tests use fixed NumPy generators for reproducibility and a controlled
RNG for exact branch/call-order assertions.

`randi(k)` maps to `integers(0,k)` only for array selection; `randi(3)` maps to
`integers(1,4)` to retain source branch labels. `randperm` is a no-replacement
permutation.

## Initial jitter and exact intersection invariant

Jitter runs before the main MCMC loop. Each attempt adds independent uniform
noise in `[0,.01)` to every coordinate, recomputes `pdist`, filters by `Z2<A`
and `Z2>0`, and accepts only when
`length(unique(D_Overall)) == length(D_Overall)`. The accepted jitter replaces
`LocalizationsFinal`; `D_Overall` corresponds exactly to it. No tolerance is
used and jitter is not moved into the loop. This is required by A.6's exact
`intersect(D_Overall,D)`.

The source destructively filters `Z2` inside the retry loop. If a failed first
attempt also shortens `Z2`, a later retry can encounter a logical-mask length
mismatch against a newly recomputed full `pdist` vector. Stage A preserves
this ordering; it is documented as a source quirk because random continuous
jitter makes a retry exceptional and no repository-supported occurrence was
established.

## Proposal construction

Every main-loop iteration copies `Constantf`, persistent `Constant2ft`, and
`Orderf`, increments `steps`, then repeatedly proposes until the resulting
`loc` differs exactly from the prior working `loc`.

- Branch 1 perturbs one density-dependent `Constant` element by `randn*.1`,
  randomly enforces monotonicity forward or backward, clamps below `-.9`, and
  mean-centers. At `steps<=1` it makes no parameter change.
- Branch 2 chooses a frame from `Frames_w_Multi` and permutes order values at
  that frame without replacement. If no such frames exist, branch 2 is
  converted to branch 1.
- Branch 3 first assigns `Constant2ft=Constant2` with MATLAB value semantics,
  then perturbs the frame-dependent `Constant2`, enforces its opposite
  monotonic direction, clamps, and mean-centers. At `steps<=1` neither the
  snapshot nor perturbation occurs.

`kappas=Constant+Constant2`, values below `-1` become `-.9999`, coordinates are
reordered, and active Algorithm 1 is called. Pair distances are not cached.
At the first step, data without a productive repeated-frame order proposal can
leave the source's inner `while isequal(loc,loc2)` unable to terminate. This is
retained as a source quirk rather than hidden behind an artificial retry cap.

## Accepted state versus best state

`Constantf` and `Orderf` are accepted-only parameter state. `Final_lik` is the
accepted likelihood. `Constant2ft` is persistent current state but is not
strictly accepted-only because of BUG-005. Best-ever state is
`Final_Localizations_Blinking_Corrected`,
`Final_Frame_Blinking_Corrected`, `Traj2`, and `max_lik`. Python represents
them separately in `MCMCState` and `MCMCResult`.

The exact acceptance condition is:

`new_lik >= Final_lik || log(rand) <= new_lik-Final_lik`.

The random draw is short-circuited for equal/uphill proposals. Therefore an
accepted likelihood may decrease and `LikHood` is not monotonic. Only
`max_lik` is non-decreasing. NaN proposals reject; infinities follow the
literal comparison/subtraction/log expression.

On acceptance, `Constantf`, `Orderf`, final `Constant2`, and `Final_lik`
update. An accepted nonmaximum does not overwrite best coordinate/frame/
trajectory snapshots. On a strict new maximum all best arrays update and the source assigns
`steps = 10` exactly. This can move the counter backward or forward. Both
accepted and rejected proposals have already performed the ordinary increment;
only accepted proposals append histories.

Between rejected iterations the MATLAB local `loc` variable still contains
the rejected working conformation. `Constantf` and `Orderf` retain their prior
accepted values. `Constant2ft` usually does too, but not after the repeated
branch-3 sequence below. Python exposes the final working conformation
separately as `MCMCResult.working_localizations`.

### DDC-SOURCE-BUG-005: branch-3 pre-acceptance leakage

On the first branch-3 attempt in an outer iteration,
`Constant2ft=Constant2` normally copies the starting persistent value. If the
resulting `loc` is unchanged, the inner `while isequal(loc,loc2)` continues
with the already perturbed working `Constant2`. If branch 3 is selected again,
the same assignment now copies that intermediate proposal into
`Constant2ft`. A later Metropolis rejection does not restore it. The next
outer iteration's `Constant2=Constant2ft` therefore starts from unaccepted
intermediate state. MATLAB assignment copies values; the Python port uses an
explicit array copy and does not create reference aliasing.

A single branch-3 attempt does not leak its new perturbation on rejection,
because its snapshot occurs before that perturbation. `Constantf` has no
corresponding assignment inside proposal generation, and `Orderf` changes only
inside the acceptance block. Full-driver controlled tests execute two branch-3
attempts, reject the resulting proposal, and observe the leaked value at the
start of the following real driver iteration.

On continuation, the source reconstructs from `Constantf`/`Constant2ft` and
sets `max_lik=max(LikHood)`, but later initializes `Final_lik=-Inf` rather than
restoring the last accepted likelihood. This makes the first finite proposal
of a resumed invocation pass the uphill half of the condition. It is preserved
as an observable chunk-boundary quirk; intent is not sufficiently established
to label it a new defect.

## PALM diagnostic

Truth scoring is diagnostic only and never enters acceptance. The initial full
localization error becomes `max_scorey`. On every accepted proposal the driver
computes `scorey11` with the approved PALM numerical core and records
`scorey11/max_scorey`. Lower is closer to truth. Division by zero preserves
MATLAB NaN/Inf behavior.

The source uses `length(TrueLocalizations)>1`, not row count. Consequently an
empty shaped `0x2` matrix has MATLAB length 2 and incorrectly enables the truth
path; plain MATLAB `[]` has length zero and disables it. This existing length
quirk is preserved and tested through the API contract.

## Source defects and repository dtype order

`DDC-SOURCE-BUG-005` is assigned to the reachable `Constant2ft` leakage above.
For repository `uint16` frames, the
first reachable failure remains `DDC-SOURCE-BUG-001` during the initial call to
`Make_Distance_Matrix_Individual`, before later `pdist`, Algorithm-1, or density
operations can run. Frames are not globally cast.

No approved A.1--A.6 implementation or test was modified for Stage A.7b.

## MATLAB oracle limitations

`generate_mcmc_reference.m` executes the unmodified driver on a small cached,
floating-frame fixture and stops before stochastic proposals using the source
timer check. It records sorting/jitter invariants and output classes/shapes.
`probe_mcmc_acceptance.m` records controlled acceptance edge cases and the
literal new-maximum step assignment. Generated `.mat` files are intentionally
absent and oracle tests skip until these scripts are run in MATLAB. Equal seeds
are not used to claim trajectory identity.

## Stage A.7b full-driver integration coverage

Three deterministic tests enter at least one actual `run_mcmc` proposal
iteration. Narrow patches control only preprocessing, Algorithm-1 outputs and
likelihood values; the real driver retains its inner proposal loop, parameter
assignments, RNG calls, acceptance block, histories, best-state update and
step transitions.

- `test_run_mcmc_main_loop_uphill_then_accepted_downhill_and_histories`
  executes a strict new maximum followed by an accepted downhill move.
- `test_run_mcmc_main_loop_rejected_downhill_preserves_constantf_orderf_only`
  verifies rejection, accepted-only histories, best-state preservation and the
  rejected working-`loc` remainder.
- `test_run_mcmc_main_loop_constant2ft_leaks_then_starts_next_iteration`
  executes two branch-3 attempts in one inner loop, rejects the result, and
  observes the leaked snapshot as the following iteration's starting value.

A fourth driver test reaches the real timer check with an injected positive
timer and deterministic clock, confirming that expiration occurs before the
ordinary step increment. The standalone `MCMCState` helper is now explicitly a
post-score transition model: its `constant2ft` field may already contain a
pre-acceptance branch-3 snapshot, and it no longer claims to model the complete
proposal-generation rollback behavior.
