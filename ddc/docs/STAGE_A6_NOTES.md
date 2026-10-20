# Stage A.6 — likelihood and image-scoring core

This stage ports only the active numerical statements of
`Calculate_Score_8.m` and `PALMplot.m`. It does not port `DDC_MCMC`, the
random-jitter loop, proposal generation, plotting, pipeline integration, or
performance changes.

## `Calculate_Score_8.m` workspace contract

The MATLAB file is a script. At entry it reads these caller variables:

| MATLAB | Python argument | Shape / role |
|---|---|---|
| `loc` | `loc` | `(n,d)` candidate localizations; floating point in repository data |
| `fram` | `fram` | MATLAB vector of `n` candidate frames |
| `A` | `a` | positive scalar frame horizon |
| `D_Overall` | `d_overall` | condensed pair-distance vector retained by the caller |
| `Z2` | `z2` | frame difference corresponding element-for-element to `D_Overall` |
| `Resolution` | `resolution` | positive distance-bin width |
| `Prob_Distributions` | `prob_distributions` | `(A,m)` probability/log-probability lookup matrix |

The script creates or overwrites `D`, `Z22`, `D_temp`, `Z2_temp`, `C`, `ia`,
`ib`, `lik`, `lik2`, and `iin`. Only `lik` and `lik2` are consumed afterward
by active `DDC_MCMC`. Python returns all final visible values in
`LikelihoodResult` so equivalence tests can inspect them. Inputs are copied;
the Python boundary does not mutate caller arrays.

`D` and `Z22` initially use condensed `pdist` pair order. MATLAB describes
this as lower-triangle column order and SciPy as upper-triangle row order;
for a symmetric distance matrix these enumerate the same unordered pair
sequence. `D` is restricted first to `Z22 < A` and then to `Z22 > 0`.
The second mask is deliberately not applied back to `Z22`, exactly as in the
source; `Z22` is dead after that statement.

Integer `fram` reaches the same `pdist` construction investigated in Stage
A.3. Therefore repository `uint16` frames explicitly reproduce
`DDC-SOURCE-BUG-002`; Stage A.6 does not silently cast them.

## Exact `intersect` behavior

`[C,ia,ib] = intersect(D_Overall,D)` uses modern MATLAB vector semantics:

- exact numeric equality, with no tolerance;
- sorted, unique `C`;
- first occurrence indices for duplicates;
- `C = D_Overall(ia)` and `C = D(ib)`;
- NaNs do not intersect;
- MATLAB indices are 1-based column vectors, while returned Python indices
  are zero-based 1-D `int64` arrays.

Only positions `ia` are deleted from `D_temp` and the aligned `Z2_temp`.
The identity of a localization pair is not compared. Thus duplicate equal
distances collapse to one intersection value and delete only the first
matching `D_Overall` entry. Active `DDC_MCMC` jitters coordinates until
`D_Overall` distances are unique before calling this script. That uniqueness
invariant is caller behavior and is documented here, but the jitter loop is
outside Stage A.6.

## Bins and likelihood sums

Both distance vectors are transformed with
`floor(distance/Resolution)`. These are zero-origin bin numbers only until
MATLAB indexes `Prob_Distributions` with `D+1`; Python indexes the same bin
with column `D`. Values above the available columns are clamped to the final
column. The source has no lower clamp: negative or NaN bins remain invalid
MATLAB subscripts and the faithful Python path raises `IndexError`. `Inf` is
captured by the upper clamp.

Retained candidate pairs use the final row:

`lik = sum(Prob_Distributions(end,D+1))`.

Every unmatched original pair uses the row equal to its 1-based frame
difference `iin`, for `iin = 1:A`. Empty selections add nothing. Zero, NaN,
and Inf lookup values are summed directly. `Calculate_Score_8.m` does not
take logarithms: active `DDC_MCMC` has already transformed the table before
the call.

## `PALMplot.m` numerical contract

The active function is not a localization matcher and defines no TP/FP/FN
or match radius. It rasterizes each inferred or true point as a truncated
`21 x 21` 2-D Gaussian. `coords(:,1:2)` are x/y in nm. Column 3 is localization
sigma in nm—not z—and defaults to 40 when the inferred input has fewer than
three columns. The truth coordinates define the shared image bounds and
coordinate origin. Inferred points outside those bounds are skipped by the
source bounds test; duplicates add another Gaussian to the same pixels.

The Stage-A scoring helper reproduces the active `DDC_MCMC` expression:
independently apply MATLAB-style `mat2gray` to the inferred and truth images,
then sum squared pixel differences. It returns raw `scorey11`; division by
`max_scorey` belongs to `DDC_MCMC` and is not included. Empty inferred input
produces a zero image. Empty truth input cannot define bounds and fails.
Inputs are converted to independent floating arrays and are not mutated.

## Oracle infrastructure

`tests/matlab/generate_likelihood_reference.m` constructs a minimal caller
workspace and executes the unmodified scoring script. Likewise,
`tests/matlab/generate_scoring_reference.m` calls the unmodified `PALMplot.m`
and records images, shapes, classes, and the raw score. The Python oracle
tests skip explicitly until those `.mat` files are generated by MATLAB.

No new repository-affecting MATLAB source defect was found in Stage A.6.
The exact-value intersection and duplicate-distance behavior are retained
source quirks whose active safety depends on the caller's jitter invariant.
