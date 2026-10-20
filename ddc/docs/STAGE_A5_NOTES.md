# Stage A.5 workspace trace and numerical contract

## Call site and execution order

Inside the active `DDC_MCMC.m`, the input frame vector has already been sorted
and the localization rows permuted with it (lines 21--23). After the random
coordinate jitter is accepted, `loc` and `fram` are assigned but neither Stage
A.5 script reads them. At lines 255 and 259 MATLAB executes, in order,
`Determine_Locs_w_Multi2` and `Density_Calc` in the shared function workspace.
The order is not a data dependency: neither reads the other's outputs.

## Workspace contract

| MATLAB name | Python name | Shape/class | Role after script |
|---|---|---|---|
| `LocalizationsFinal` | `localizations_final` | `(n,d)`, repository `double` | Read by both; never mutated. |
| `Frame_Information` | `frame_information` | vector length `n`, repository `uint16` | Read by both; never mutated. |
| `Resolution` | `resolution` | scalar numeric | Read by both. |
| `A` | `a` | scalar numeric | Read only by density. |
| `Frames_w_Multi` | function return | MATLAB `1×m double`; Python `(m,) float64` | Consumed at DDC_MCMC lines 339 and 450--452 to enable/permutate same-frame order moves. |
| `DistanceDens` | first density return | scalar `2*Resolution` | Script-visible but dead afterward. |
| `density` | second density return | MATLAB `L×1 double`; Python `(L,1) float64` | Sorted at line 356 to constrain the density-dependent κ updates. |

`ffs`, `inds42`, `Go_w_Multi`, `iosnei`, `iinoeubg`, `Datapoint`, `Distance`,
`nnnn2`, `isonepp3`, and `onerrrr` are overwritten loop temporaries and dead
after their scripts. No `Loc`, `Locs`, trajectory label, probability mask,
`find` result, `unique`, `intersect`, logical-index mutation, or linear indexing
is an active input/output of these two source files.
Because these are scripts, every listed output and temporary is created in or
overwrites the caller workspace. The later DDC_MCMC use of `inds42` assigns it
again before reading it, so the value left by `Determine_Locs_w_Multi2` is dead.

## `Determine_Locs_w_Multi2.m`

The outer loop follows `length(Frame_Information)=n`. For each occurrence it
uses equality and `find` to obtain 1-based same-frame row indices. Python uses
the equivalent zero-based indices from `flatnonzero`. If there are at least two
rows and the frame has not already been stored, the script examines ordered
distinct pairs. A frame is appended once when any distance is strictly
`<4*Resolution`; equality is excluded. Output order follows first occurrence,
so unsorted frames are permitted and not sorted by the script.

Frame values are only indexed and compared. There is no frame arithmetic,
concatenation, or conversion through another function. `Frames_w_Multi=[]`
starts as double, so appended repository `uint16` scalars become double. NaN
never equals itself and therefore produces no group; repeated Inf values can be
grouped. NaN distances fail the strict comparison. Coordinates are dimension
agnostic and both 2-D and 3-D follow the same sum-of-squares expression.

## `Density_Calc.m`

Let `L=length(LocalizationsFinal)`: MATLAB defines `L=0` for an empty matrix and
otherwise `max(n,d)`. The script allocates an `L×1 double` and performs both
loops over `1:L`; self-pairs and both ordered directions are explicitly tested.
For nonempty `n<d`, row `n+1` is out of bounds, so the source fails rather than
returning a padded vector. Stage A reproduces that failure.

Every coordinate pair uses Euclidean distance in all columns. Inclusion uses
`Distance <= 2*Resolution` and the frame condition is strictly `>A`; exact
spatial boundaries count, exact temporal boundaries do not. Self-pairs can only
count for unusual negative `A`. Inputs are not mutated.

For floating frames, subtraction and `abs` are symmetric. For repository
`uint16`, subtraction saturates before `abs`, producing the ordered asymmetry
recorded as DDC-SOURCE-BUG-004. No cast is applied.

Normalization is exactly
`(density-min(density))/(max(density)-min(density))`. Constant nonempty density
therefore yields all NaN, including the one-localization and all-zero cases.
The port suppresses only Python's runtime warning, not the NaN result, so the
suite can run with warnings treated as errors.
