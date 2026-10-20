# Stage A.3 — `Determine_Blinking_Distribution5` contract

This records the line-by-line numerical contract used for the faithful Python
port. MATLAB indices below are 1-based.

## Inputs and outputs

| Name | Shape/class on repository data | Meaning |
|---|---|---|
| `LocalizationsFinal{i}` | `(n_i, 2 or 3)`, `double` | Per-image coordinates in nm |
| `Frame_Information{i}` | row vector `(1, n_i)`, `uint16` | Frame associated with each coordinate row |
| `Pre_A` | positive scalar integer-valued `double` in callers | Frame horizon fitted at `w=1:Pre_A` |
| `Resolution` | positive scalar `double` in callers | Distance histogram step |
| `bins` | `(1, q+1)`, `double` | `[0:Resolution:D_maxf, Inf]` |
| `D_Counts3` | `(1, q)`, `double` | Cleaned blinking-distance probability |
| `Total_No_Blink` | `(n_images, q)`, `double` | Per-image histograms for `Pre_A < Δframe < 5*Pre_A` |
| returned `Resolution` | scalar, unchanged | Same input value |
| `X_overall` | `(1, Pre_A)` | Mean un-clipped fit across images, or the single fit row |
| `M_mat` | `(Pre_A, q)` | Mean deviation-in-probability across images |

## Intermediate arrays and source operations

| MATLAB lines | Intermediate | Shape/dtype and exact role |
|---|---|---|
| 13–30 | `D`, `D_max`, `D_maxf`, `D_maxm` | `pdist` condensed `double`; maximum per image, global maximum, and unused minimum-of-maxima. `D_maxm` has no output effect. |
| 35 | `bins` | MATLAB colon edges followed by `Inf`; `q = length(bins)-1`. |
| 50, 160, 246 | `Z2` | Intended condensed frame differences from `pdist([zeros, frame(:)])`. With repository `uint16` frames, concatenation yields `uint16`, which is unsupported by documented `pdist` input types: DDC-SOURCE-BUG-002. With `single`/`double` frames, output is floating and pair-aligned with `D`. |
| 52, 162, 248 | `D` | Condensed Euclidean coordinate distances, `double`, same pair order as `Z2`. |
| 54 | `D_Blink` | Distances with `Z2 < Pre_A`; this includes equal-frame pairs (`Z2=0`). |
| 56, 164–166, 250–251 | `D_No_Blink` | Distances with strict bounds `Z2 > Pre_A` and `Z2 < 5*Pre_A`. Line 164/250 is overwritten by the next line and has no effect. |
| 61–67 | `D_Counts`, `D_Counts2`, `Total_*` | `histcounts(...,'probability')`, hence `double` counts divided by input length; rows are concatenated by image. Empty inputs yield `0/0 = NaN`. |
| 77–87 | aggregate distributions, `D_Scale`, initial `D_Counts3` | Column means; tail ratio uses MATLAB bins `10:end`; subtraction and normalization are `double`. |
| 94–122 | cleaned `D_Counts3` | Stateful cleanup over bins `4:length-1`, negative clipping, two redundant normalizations, then optional zeroing of bins `8:end`. `ins` is assigned but never used. |
| 142–202 | `Dscale_store`, `Dscale_store2` | One `double` fit for every image and exact frame difference `w=1:Pre_A`; rows are stacked. |
| 168, 175 | `True_Distribuiton2`, `Temp_Distribution` | Probability histograms for no-blink distances and `Z2==w`. |
| 179–185 | `t`, `y`, `x` | `t` is `2×q`; model is `x*t(1,:) + (1-x)*t(2,:)`; unbounded `lsqcurvefit`, starting at `x0=1`, returns scalar `double`. |
| 207–215 | `X_overall`, `Dscale_store` | Multiple images: column means. One image: preserve the `1×Pre_A` row. |
| 225–229 | clipped `Dscale_store` | Values above 1 become 1; below 0 become `1e-7`; final element is forced to 1. `X_overall` remains the earlier un-clipped result. |
| 231–271 | `Deviation_in_Probabilityt{i}` | `(Pre_A,q)` `double`; mixture followed by elementwise division. Zero denominators produce MATLAB `NaN`/`Inf`. |
| 278–292 | `M_mat_t`, `M_mat` | Stack image rows for each `w`, then ordinary mean; NaNs propagate. One image returns its matrix directly. |
| 294–302 | `M_mat2` | Dead median calculation, neither returned nor used; intentionally omitted under the approved migration plan. |

## Pair ordering and dtype conclusions

MATLAB documents lower-triangle column order: `(2,1), (3,1), ..., (3,2), ...`.
SciPy documents `i<j` upper-triangle row order: `(0,1), (0,2), ..., (1,2), ...`.
For symmetric distances these are the same unordered pairs in the same sequence.
The code relies on this equivalence by masking coordinate distances `D` with a
separately computed frame vector `Z2`.

`DDC-SOURCE-BUG-001` does not describe these operations, because no scalar
subtraction is reached inside `pdist`. Instead, `DDC-SOURCE-BUG-002` occurs
first for repository `uint16` frames: mixed concatenation yields a `uint16`
matrix that the documented `pdist` API rejects. Stage A explicitly represents
that failure and permits only source inputs already typed `single` or `double`.

The fit is linear in one parameter. Stage A retains a SciPy trust-region fit as
the MATLAB-facing path. Independently, for `a=true`, `b=blink`, and `y=observed`,
the validation solution is:

```text
x = dot(a-b, y-b) / dot(a-b, a-b)
```

When the denominator is zero, `x` is unidentifiable; the analytical validator
returns NaN and the numerical solver retains the MATLAB/SciPy start value 1 for
the tested finite residual.
