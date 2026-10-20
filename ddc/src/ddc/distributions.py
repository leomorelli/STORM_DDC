"""Faithful Stage-A port of ``Determine_Blinking_Distribution5.m``.

Contract and shapes
-------------------
Inputs are parallel lists of per-image arrays. ``localizations_final[i]`` has
shape ``(n_i, d)`` (the real repository inputs are float64, with d=2 or 3) and
``frame_information[i]`` is a row/column/1D vector of length ``n_i`` (uint16 in
all included MAT inputs). ``pre_a`` is the positive integer frame horizon and
``resolution`` is the positive distance-bin width.

The six outputs match MATLAB in order:

``bins``
    Float64 edges ``[0:resolution:D_maxf, Inf]``, shape ``(n_bins + 1,)``.
``distribution_for_blink``
    Cleaned probability per distance bin, shape ``(n_bins,)``.
``total_no_blink``
    Per-image probability histograms for ``pre_a < Δframe < 5*pre_a``, shape
    ``(n_images, n_bins)``.
``resolution``
    The input scalar, returned unchanged by the MATLAB function.
``x_overall``
    Per-frame-difference mixture parameters. With multiple images this is the
    column mean of the un-clipped per-image fits; shape ``(pre_a,)``.
``m_mat``
    Mean per-image deviation-in-probability matrix, shape
    ``(pre_a, n_bins)``.

MATLAB numerical semantics
--------------------------
SciPy ``pdist`` is used directly for supported floating inputs. MATLAB documents
its condensed vector as lower-triangle column order ``(2,1), (3,1), ...``;
SciPy documents upper-triangle row order ``(0,1), (0,2), ...``. These enumerate
the same unordered pairs in the same sequence for a symmetric distance, so
coordinate distances and frame differences remain pair-aligned.

MATLAB constructs the frame input as ``[double_zeros, frame]``. Concatenation
with a repository `uint16` frame converts the full matrix to `uint16`, while
MATLAB `pdist` documents only `single` and `double` inputs. This is
DDC-SOURCE-BUG-002. The faithful path rejects integer frames here and does not
silently cast them. DDC-SOURCE-BUG-001 remains the separate scalar-subtraction
defect in ``Make_Distance_Matrix_Individual``.

``histcounts(..., 'Normalization','prob')`` is reproduced as counts divided by
the number of values counted. It is not density normalization. Bins are
left-closed/right-open except for the final bin, and the explicit ``Inf`` edge
captures every finite nonnegative distance. The one-parameter model fitted by
``lsqcurvefit`` is ``x*true + (1-x)*blink``. Stage A uses SciPy's trust-region
reflective least-squares solver from the MATLAB starting point x0=1, without
bounds. The analytical ordinary-least-squares solution is exposed only as an
independent validation aid; it does not replace the solver-facing result.

Relevant dtype propagation is explicit: for accepted floating inputs, both
MATLAB/SciPy ``pdist`` outputs,
histogram probabilities, means, fitted parameters, and `M_mat` are floating
point; comparisons create logical/boolean masks. Input coordinate and frame
arrays are not rewritten. ``resolution`` is returned unchanged, while its value
is converted to floating point only for constructing `bins`. Integer coordinate
inputs are outside the real repository contract; `pdist` itself produces
floating distances from them. Empty probability-normalized histograms are NaN
because MATLAB defines every bin as `count/N` with `N=0`.

Per-image calculations retain their frame loop and scientific operations, and
can be dispatched to spawned processes without changing reduction order. The dead
``M_mat2`` median calculation is omitted as authorized by the migration plan.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist


def _matlab_colon_zero_to_stop(step: Real, stop: Real) -> NDArray[np.float64]:
    """Reproduce the nonnegative ``0:step:stop`` used for histogram edges."""

    count = int(np.floor(float(stop) / float(step)))
    return np.arange(count + 1, dtype=np.float64) * float(step)


def _probability_histogram(
    values: NDArray[np.float64], bins: NDArray[np.float64]
) -> NDArray[np.float64]:
    """MATLAB ``histcounts(..., Normalization='prob')`` for DDC distances."""

    counts, _ = np.histogram(values, bins=bins)
    if values.size == 0:
        # MATLAB's documented probability normalization is count/N; for an
        # empty input N=0, so every 0/0 bin is NaN.
        return np.full(counts.shape, np.nan, dtype=np.float64)
    return counts.astype(np.float64) / values.size


def analytical_mixture_parameter(
    true_distribution: ArrayLike,
    blink_distribution: ArrayLike,
    observed_distribution: ArrayLike,
) -> float:
    """Independent OLS solution for ``x*true + (1-x)*blink``.

    Returns NaN when the model is degenerate because ``true == blink`` and the
    denominator is zero; in that case the fitted parameter is not identifiable.
    """

    true = np.asarray(true_distribution, dtype=np.float64)
    blink = np.asarray(blink_distribution, dtype=np.float64)
    observed = np.asarray(observed_distribution, dtype=np.float64)
    direction = true - blink
    denominator = float(np.dot(direction, direction))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(direction, observed - blink) / denominator)


def _fit_mixture_parameter(
    true_distribution: NDArray[np.float64],
    blink_distribution: NDArray[np.float64],
    observed_distribution: NDArray[np.float64],
) -> float:
    """Stage-A SciPy equivalent of the source ``lsqcurvefit`` call."""

    def residual(parameter: NDArray[np.float64]) -> NDArray[np.float64]:
        x = parameter[0]
        return (
            x * true_distribution
            + (1.0 - x) * blink_distribution
            - observed_distribution
        )

    fit = least_squares(residual, x0=np.array([1.0]), method="trf")
    return float(fit.x[0])


def _frame_pair_distances(frame: ArrayLike) -> NDArray[np.float64]:
    frames = np.asarray(frame)
    if frames.ndim > 2 or (frames.ndim == 2 and min(frames.shape, default=0) > 1):
        raise ValueError("each frame array must be a MATLAB-style vector")
    frames = frames.reshape(-1, order="F")
    if not (
        np.issubdtype(frames.dtype, np.floating)
        and frames.dtype.itemsize in (4, 8)
    ):
        raise TypeError(
            "DDC-SOURCE-BUG-002: MATLAB concatenation passes an integer "
            "frame matrix to pdist, whose documented X types are single/double"
        )
    # The zero column mirrors the exact two-column MATLAB expression. No dtype
    # cast is applied: supported single/double frame inputs retain their class.
    frame_points = np.column_stack(
        (np.zeros(frames.size, dtype=frames.dtype), frames)
    )
    return pdist(frame_points)


def determine_blinking_distribution(
    localizations_final: list[ArrayLike],
    frame_information: list[ArrayLike],
    pre_a: int,
    resolution: Real,
    *,
    workers: int = 1,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    Real,
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Determine the shared distribution; optionally compute images in processes.

    Per-image calculations are independent. Ordered collection and parent-side
    reductions retain the serial result and scientific weighting.
    """

    if isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 1:
        raise ValueError("workers must be a positive integer")

    if len(localizations_final) != len(frame_information):
        raise ValueError("localizations and frames must contain the same number of images")
    if not localizations_final:
        raise ValueError("at least one image is required")
    if not isinstance(pre_a, Integral) or pre_a < 1:
        raise ValueError("pre_a must be a positive integer")
    if not isinstance(resolution, Real) or not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be a finite positive number")

    locations: list[NDArray[Any]] = []
    frames: list[NDArray[Any]] = []
    for image_index, (image_locations, image_frames) in enumerate(
        zip(localizations_final, frame_information, strict=True)
    ):
        loc = np.asarray(image_locations)
        fram = np.asarray(image_frames)
        if loc.ndim != 2:
            raise ValueError(f"image {image_index}: localizations must be a 2D array")
        if fram.ndim > 2 or (fram.ndim == 2 and min(fram.shape, default=0) > 1):
            raise ValueError(f"image {image_index}: frames must be a vector")
        fram = fram.reshape(-1, order="F")
        if loc.shape[0] != fram.size:
            raise ValueError(f"image {image_index}: localization/frame length mismatch")
        if loc.shape[0] < 2:
            raise ValueError(
                "the MATLAB source cannot determine D_max from fewer than two localizations"
            )
        locations.append(loc)
        frames.append(fram)

    count = min(int(workers), len(locations))
    if count > 1:
        with ProcessPoolExecutor(max_workers=count, mp_context=get_context("spawn")) as pool:
            return _calculate_distribution(locations, frames, pre_a, resolution, pool.map)
    return _calculate_distribution(locations, frames, pre_a, resolution, map)


def _maximum_image_distance(loc):
    return float(np.max(pdist(loc)))


def _image_histograms(task):
    loc, fram, pre_a, bins = task
    z2 = _frame_pair_distances(fram)
    distances = pdist(loc)
    return (
        _probability_histogram(distances[z2 < pre_a], bins),
        _probability_histogram(distances[(z2 > pre_a) & (z2 < pre_a * 5)], bins),
    )


def _image_mixture_scales(task):
    loc, fram, pre_a, bins, distribution_for_blink, true_distribution = task
    z2 = _frame_pair_distances(fram)
    distances = pdist(loc)
    image_scales = np.empty(pre_a, dtype=np.float64)
    for w in range(1, pre_a + 1):
        temporary_distribution = _probability_histogram(distances[z2 == w], bins)
        image_scales[w - 1] = _fit_mixture_parameter(
            true_distribution, distribution_for_blink, temporary_distribution,
        )
    return image_scales


def _calculate_distribution(locations, frames, pre_a, resolution, map_images):
    d_maxf = 0.0
    for d_max in map_images(_maximum_image_distance, locations):
        if d_max > d_maxf:
            d_maxf = d_max

    bins = np.append(_matlab_colon_zero_to_stop(resolution, d_maxf), np.inf)

    total_blink_rows: list[NDArray[np.float64]] = []
    total_no_blink_rows: list[NDArray[np.float64]] = []
    tasks = ((loc, fram, pre_a, bins) for loc, fram in zip(locations, frames, strict=True))
    for blink_row, no_blink_row in map_images(_image_histograms, tasks):
        total_blink_rows.append(blink_row)
        total_no_blink_rows.append(no_blink_row)

    total_blink = np.vstack(total_blink_rows)
    total_no_blink = np.vstack(total_no_blink_rows)
    d_counts = np.mean(total_blink, axis=0)
    d_counts2 = np.mean(total_no_blink, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        d_scale = np.sum(d_counts[9:]) / np.sum(d_counts2[9:])
        distribution_for_blink = d_counts - d_counts2 * d_scale
        distribution_for_blink = distribution_for_blink / np.sum(distribution_for_blink)

    good = 1
    ins = 0
    # MATLAB: for i=4:length(D_Counts3)-1, translated from 1-based indexing.
    for matlab_i in range(4, distribution_for_blink.size):
        i = matlab_i - 1
        if (
            distribution_for_blink[i] > 0
            and distribution_for_blink[i + 1] < distribution_for_blink[i]
            and good == 1
        ):
            pass
        else:
            if ins == 0:
                ins = matlab_i
            good = 0
            distribution_for_blink[i] = 0

    distribution_for_blink[distribution_for_blink < 0] = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        distribution_for_blink = distribution_for_blink / np.sum(distribution_for_blink)
        distribution_for_blink = distribution_for_blink / np.sum(distribution_for_blink)

    # MATLAB D_Counts3(8:end) maps to Python [7:].
    if np.sum(distribution_for_blink[7:] > 0) > 1:
        distribution_for_blink[7:] = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            distribution_for_blink = distribution_for_blink / np.sum(
                distribution_for_blink
            )

    true_distributions = total_no_blink_rows
    tasks = (
        (loc, fram, pre_a, bins, distribution_for_blink, true)
        for loc, fram, true in zip(locations, frames, true_distributions, strict=True)
    )
    dscale_rows = list(map_images(_image_mixture_scales, tasks))

    dscale_store2 = np.vstack(dscale_rows)
    if len(locations) > 1:
        x_overall = np.mean(dscale_store2, axis=0)
        dscale_store = np.mean(dscale_store2, axis=0)
    else:
        x_overall = dscale_store2.copy()
        dscale_store = dscale_store2.copy()

    dscale_store = np.asarray(dscale_store).reshape(-1)
    dscale_store[dscale_store > 1] = 1
    dscale_store[dscale_store < 0] = 0.0000001
    dscale_store[-1] = 1

    deviation_by_image: list[NDArray[np.float64]] = []
    for true_distribution in true_distributions:
        deviation = np.empty((pre_a, distribution_for_blink.size), dtype=np.float64)
        for w in range(1, pre_a + 1):
            d_scale = dscale_store[w - 1]
            temporary_distribution2 = (
                distribution_for_blink * (1 - d_scale)
                + true_distribution * d_scale
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                combined = (
                    temporary_distribution2 - d_scale * true_distribution
                ) / temporary_distribution2
            deviation[w - 1, :] = combined
        deviation_by_image.append(deviation)

    if len(locations) > 1:
        m_mat = np.mean(np.stack(deviation_by_image, axis=0), axis=0)
    else:
        m_mat = deviation_by_image[0]

    return (
        bins,
        distribution_for_blink,
        total_no_blink,
        resolution,
        x_overall,
        m_mat,
    )


__all__ = [
    "analytical_mixture_parameter",
    "determine_blinking_distribution",
]
