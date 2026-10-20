"""Faithful numerical port of the active ``Calculate_Score_8.m`` script.

The MATLAB script reads ``loc``, ``fram``, ``A``, ``D_Overall``, ``Z2``,
``Resolution``, and ``Prob_Distributions`` from the caller workspace. It
overwrites/creates ``D``, ``Z22``, ``D_temp``, ``Z2_temp``, ``C``, ``ia``,
``ib``, ``lik``, ``lik2``, and loop variable ``iin``. Only ``lik`` and ``lik2``
are consumed by ``DDC_MCMC`` after the script; the result dataclass exposes the
other final intermediates for Stage-A equivalence tests.

``D_Overall`` and ``Z2`` are filtered pair vectors produced earlier by the
caller. The caller jitters coordinates until every ``D_Overall`` value is
unique, but that jitter loop is deliberately outside Stage A.6. Intersection
uses exact floating equality, sorted unique output, first duplicate indices,
and excludes NaN, matching modern MATLAB ``intersect``. Python indices are
zero-based 1-D arrays; MATLAB ``ia``/``ib`` are 1-based column vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import pdist

from .distributions import _frame_pair_distances


@dataclass(frozen=True)
class LikelihoodResult:
    """Final script-visible values, normalized to 1-D Python vectors."""

    lik: float
    lik2: float
    d: NDArray[np.float64]
    z22: NDArray[np.float64]
    c: NDArray[np.float64]
    ia: NDArray[np.int64]
    ib: NDArray[np.int64]
    d_temp: NDArray[np.float64]
    z2_temp: NDArray[np.float64]


def matlab_intersect_first(
    a: ArrayLike, b: ArrayLike
) -> tuple[NDArray[Any], NDArray[np.int64], NDArray[np.int64]]:
    """Modern MATLAB ``[C,ia,ib]=intersect(A,B)`` for numeric vectors.

    Values are deduplicated and sorted ascending. ``ia`` and ``ib`` point to
    the first occurrence in each input. NaNs never intersect. Signed zeros are
    equal; the value stored in ``C`` comes from the first matching occurrence
    in ``A``, consistent with ``C=A(ia)``.
    """

    left = np.asarray(a).reshape(-1, order="F")
    right = np.asarray(b).reshape(-1, order="F")
    if left.dtype == np.float64 and right.dtype == np.float64:
        return _intersect_float64_first(left, right)

    return _intersect_generic_first(left, right)


def _intersect_float64_first(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Fast exact path for the active DDC float64 pair vectors."""

    if left.size == 0 or right.size == 0:
        empty_indices = np.asarray([], dtype=np.int64)
        return np.asarray([], dtype=np.float64), empty_indices, empty_indices.copy()

    left_values, left_first = np.unique(left, return_index=True)
    right_values, right_first = np.unique(right, return_index=True)
    left_valid = ~np.isnan(left_values)
    right_valid = ~np.isnan(right_values)
    left_values, left_first = left_values[left_valid], left_first[left_valid]
    right_values, right_first = right_values[right_valid], right_first[right_valid]
    if left_values.size == 0 or right_values.size == 0:
        empty_indices = np.asarray([], dtype=np.int64)
        return np.asarray([], dtype=np.float64), empty_indices, empty_indices.copy()

    positions = np.searchsorted(right_values, left_values, side="left")
    in_bounds = positions < right_values.size
    matches = np.zeros(left_values.size, dtype=bool)
    matches[in_bounds] = right_values[positions[in_bounds]] == left_values[in_bounds]
    ia = np.asarray(left_first[matches], dtype=np.int64)
    ib = np.asarray(right_first[positions[matches]], dtype=np.int64)
    # Do not return left_values: indexing A is required to preserve the first
    # A occurrence's observable signed-zero representation.
    c = np.asarray(left[ia], dtype=np.float64)
    return c, ia, ib


def _intersect_generic_first(
    left: NDArray[Any], right: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[np.int64], NDArray[np.int64]]:
    """Frozen Stage-A scan retained for non-float64 supported inputs."""

    records: list[tuple[Any, int, int]] = []

    for left_index, value in enumerate(left):
        if np.issubdtype(left.dtype, np.inexact) and np.isnan(value):
            continue
        if any(value == existing[0] for existing in records):
            continue
        right_index = None
        for candidate_index, candidate in enumerate(right):
            if np.issubdtype(right.dtype, np.inexact) and np.isnan(candidate):
                continue
            if value == candidate:
                right_index = candidate_index
                break
        if right_index is not None:
            records.append((value, left_index, right_index))

    records.sort(key=lambda record: record[0])
    dtype = np.result_type(left.dtype, right.dtype)
    c = np.asarray([record[0] for record in records], dtype=dtype)
    ia = np.asarray([record[1] for record in records], dtype=np.int64)
    ib = np.asarray([record[2] for record in records], dtype=np.int64)
    return c, ia, ib


def _probability_columns(
    binned_distances: NDArray[np.float64], width: int
) -> NDArray[np.int64]:
    """Translate MATLAB 1-based ``D+1`` to zero-based probability columns."""

    if (
        np.any(~np.isfinite(binned_distances))
        or np.any(binned_distances != np.floor(binned_distances))
        or np.any(binned_distances < 0)
    ):
        raise IndexError("MATLAB probability column D+1 is not a valid subscript")
    columns = binned_distances.astype(np.int64)
    if np.any(columns >= width):
        raise IndexError("MATLAB probability column exceeds matrix width")
    return columns


def calculate_score_8(
    loc: ArrayLike,
    fram: ArrayLike,
    a: int,
    d_overall: ArrayLike,
    z2: ArrayLike,
    resolution: Real,
    prob_distributions: ArrayLike,
) -> LikelihoodResult:
    """Execute the active statements of ``Calculate_Score_8.m``.

    ``Prob_Distributions`` is summed exactly as supplied. In active DDC_MCMC it
    already contains logarithms; this function does not apply ``log`` itself.
    """

    locations = np.asarray(loc)
    frames = np.asarray(fram)
    probabilities = np.asarray(prob_distributions)
    overall = np.asarray(d_overall, dtype=np.float64).reshape(-1, order="F")
    overall_frames = np.asarray(z2, dtype=np.float64).reshape(-1, order="F")
    if locations.ndim != 2:
        raise ValueError("loc must be a 2D localization matrix")
    if frames.ndim > 2 or (frames.ndim == 2 and min(frames.shape, default=0) > 1):
        raise ValueError("fram must be a MATLAB-style vector")
    frames = frames.reshape(-1, order="F")
    if locations.shape[0] != frames.size:
        raise ValueError("loc rows and fram length must match")
    if overall.size != overall_frames.size:
        raise ValueError("D_Overall and Z2 lengths must match")
    if not isinstance(a, Integral) or a < 1:
        raise ValueError("A must be a positive integer")
    if not isinstance(resolution, Real) or resolution <= 0:
        raise ValueError("Resolution must be positive")
    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        raise ValueError("Prob_Distributions must be a nonempty 2D matrix")

    d_physical = pdist(locations)
    z22 = _frame_pair_distances(frames)
    under_horizon = z22 < a
    d_physical = d_physical[under_horizon]
    z22 = z22[under_horizon]
    # The source does not apply this mask back to Z22; Z22 is dead afterward.
    d_physical = d_physical[z22 > 0]

    d_temp_physical = overall.copy()
    z2_temp = overall_frames.copy()
    c, ia, ib = matlab_intersect_first(overall, d_physical)
    keep = np.ones(d_temp_physical.size, dtype=bool)
    keep[ia] = False
    d_temp_physical = d_temp_physical[keep]
    z2_temp = z2_temp[keep]

    d = np.floor(d_physical / resolution)
    d_temp = np.floor(d_temp_physical / resolution)
    width = probabilities.shape[1]
    d[d + 1 > width] = width - 1
    d_temp[d_temp + 1 > width] = width - 1

    d_columns = _probability_columns(d, width)
    lik = float(np.sum(probabilities[-1, d_columns]))
    lik2 = 0.0
    for matlab_row in range(1, a + 1):
        selected = d_temp[z2_temp == matlab_row]
        if selected.size != 0:
            if matlab_row > probabilities.shape[0]:
                raise IndexError("MATLAB probability row exceeds matrix height")
            columns = _probability_columns(selected, width)
            lik2 += float(np.sum(probabilities[matlab_row - 1, columns]))

    return LikelihoodResult(lik, lik2, d, z22, c, ia, ib, d_temp, z2_temp)
