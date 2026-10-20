# Stage A.4 source trace and numerical contract

## Active call path

`DDC_MCMC.m` sorts `Frame_Information` and applies the same permutation to
`LocalizationsFinal` (lines 21--23), then calls
`Determine_Deviation_in_Probability8` at line 130. That function calls
`Eliminate_Blinking_De_Loc15_MCMC` at its line 123. Stage A.4 does not port any
trajectory, score, or MCMC routine.

## `Determine_Deviation_in_Probability8.m`

The inputs, in source order, are:

1. `True_Distribuiton` and `Distribution_for_Blink`: length-`q` floating
   probability rows used for the `(A,q)` outputs.
2. `bins`: no active use; its `histcounts` line is commented out.
3. `A`: positive integer frame horizon and row count.
4. `LocalizationsFinal`: `(n,d)` floating coordinates; `d` may be 2 or 3.
5. `Frame_Information`: length `n`. Its two-column `pdist` expression yields
   aligned differences for `single`/`double` and reaches BUG-002 for repository
   `uint16` data.
6. `loc`, `fram`: legacy inputs. `fram` is only in commented code; `loc` only
   controls a dead assignment to `Dscale_store5`.
7. `Resolution`: forwarded to the eliminator.
8. `True_Distribuiton2`, `Distribution_for_Blink2`, `bins2`: coarsened
   length-`q2` fitting rows and `q2+1` histogram edges.
9. `X_overall`: if nonempty, initial `Dscale_store`; otherwise every exact
   difference `w=1:A` is fit using the source one-parameter mixture.
10. `M_mat`: if nonempty, the initial and returned deviation matrix; otherwise
    the function calculates it from the primary distributions.

The three outputs are `Deviation_in_Probability (A,q)`,
`Prob_Distributions (A,q)`, and `Dscale_store (A,)`. Fit values above 1 are set
to 1, values below 0 to `1e-7`, and the final element is forced to 1. After the
eliminator estimates the retained count, the function subtracts its correction
from each observed histogram, clips values strictly below zero to zero,
renormalizes, and refits. Although `Combined` is recomputed after this pass, it
is not assigned back; the returned deviation matrix remains `M_mat` or the
first calculation.

The final probability matrix receives one global offset:
`min(Prob_Distributions(Prob_Distributions>0))/10`. This is not elementwise
clipping. Empty normalized histograms and zero normalizers produce NaN; fitting
an undefined residual remains an error. If the final matrix has no positive
entry, the source positive minimum is empty and cannot produce the declared
nonempty output.

## `Eliminate_Blinking_De_Loc15_MCMC.m`

Inputs are `(n,d)` coordinates, length-`n` frames, `Resolution`, integer `A`,
an `(A,q)` probability matrix, and trajectory labels. The active caller creates
`max(n,d)` unique labels, although only the first `n` are indexed. Outputs are
length-`max(n,d)` double vectors: `Loc` is 1 for retained points and 0 for
inferred blinks; `Probs` stores the accepted or best rejected mean probability.

MATLAB defines `length(LocalizationsFinal)` as `max(size(LocalizationsFinal))`,
not `size(LocalizationsFinal,1)`. It uses that value for `Linked_Loc`,
`Loc_is_Blink`, `Trajectory`, `Probs`, final `Loc`, the outer `randperm`, and
the `nb` numerator. Candidate/search loops instead use `length(Frame)` and
therefore follow the `n` localization rows. The port keeps separate `n_rows`
and `matlab_length=max(n,d)` values. For `n<d`, trailing work/output positions
remain at their defaults but contribute retained ones to `sum(loc)` and hence
change `nb` and the second-fit correction.

The loops investigate exact forward differences `i=1:round(A)`. The caller
sorts frames, but this function does not validate sorting. Acceptance requires
`Linked_Loc(ii)~=1`, a target not already marked as blink,
`max(Framer)<A`, and a mean probability strictly greater than `.5`. Equality is
rejected. Repeated frames are not direct candidates because `i` starts at 1.
Empty candidate sets retain the all-one mask and all-zero probabilities.

For every candidate, all cross-distances between its two current trajectory
groups are calculated. `DDt=floor(Distance/Resolution)+1` is a 1-based column,
capped only above at `q`; an exact boundary enters the next bin. `Framer` is the
1-based row. MATLAB column-major
`sub2ind([A,q],Framer,DDt)=Framer+(DDt-1)*A`; direct NumPy indexing at
`[Framer-1,DDt-1]` selects the same values in the same `(:)` sequence. Zero,
noninteger, NaN, Inf, or out-of-range subscripts are source indexing failures.

All work arrays have MATLAB's default double class except values derived from
the frame input. Inputs are not mutated. `Trajectory` is local shared state and
logical assignment merges complete groups. `randperm` supplies unique labels;
only equality is observed, so label values do not affect returned arrays.
The two scalar frame subtractions in the search loops retain the frame class;
integer subtraction therefore saturates at that class's endpoints. With the
active caller's ascending sort, forward differences are nonnegative. The port
still models saturation explicitly rather than relying on NumPy wraparound.

For integer frames, `AAAA` and `BBBB` remain integer and their nonscalar matrix
product is unsupported: DDC-SOURCE-BUG-003. The port represents this at the
expression site without a global frame cast.

This `length` behavior is a source quirk rather than a numbered defect because
all supported repository inputs satisfy `n>=d`. A static scan checked 927
nonempty localization matrices/cells with zero violations; the smallest
observed `n-d` was 950. Four `Analyzed_Time...` MAT v7.3 files are generated
analysis outputs, not inputs accepted by the Stage-A.1 loader.
