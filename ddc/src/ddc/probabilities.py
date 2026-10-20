"""Faithful Stage-A port of the active DDC probability component.

This module ports ``Determine_Deviation_in_Probability8.m`` and
``Eliminate_Blinking_De_Loc15_MCMC.m`` with a conservative candidate-search fast path. MATLAB row ``w``
and distance-bin column ``DDt`` map to Python indices ``w-1`` and ``DDt-1``.
The latter is ``floor(distance / Resolution) + 1`` and is capped only above.

Repository ``uint16`` frames reach DDC-SOURCE-BUG-002 in the outer function's
``pdist`` call. Independently, the eliminator reaches DDC-SOURCE-BUG-003 when
its integer ``AAAA*BBBB'`` matrix product is evaluated; MATLAB permits an
integer operand to ``mtimes`` only when the other operand is scalar. No frame
cast is applied here.

The input coordinate and frame arrays are not mutated. MATLAB's local
``Trajectory`` array is shared only within a call. NaN means fail both source
comparisons, Inf means pass the strict ``>.5`` threshold, and empty candidate
sets leave the default retained mask and zero probabilities. MATLAB sizes
several work/output vectors with ``length(LocalizationsFinal)=max(n,d)`` even
though candidate loops follow the length-``n`` frame vector; the port preserves
that distinction.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import pdist
from ._acceleration import try_eliminate

from .distributions import (
    _fit_mixture_parameter,
    _frame_pair_distances,
    _probability_histogram,
)


class DDCSourceBug003(TypeError):
    """MATLAB integer ``mtimes`` failure in the active eliminator."""


def _matlab_matrix_length(array: NDArray[Any]) -> int:
    """MATLAB ``length(X)`` for a 2-D array: ``max(size(X))``."""

    if array.size == 0:
        return 0
    return max(array.shape)


def _matlab_frame_subtract(left: Any, right: Any, dtype: np.dtype[Any]) -> Any:
    """Scalar MATLAB subtraction, including saturating integer arithmetic."""

    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        return min(max(int(left) - int(right), limits.min), limits.max)
    return left - right


def _can_search_frame_candidates(frames: NDArray[Any], last_difference: int) -> bool:
    """Use exact binary searches only for sorted double integer frames."""

    return bool(
        frames.dtype == np.dtype(np.float64)
        and np.all(np.isfinite(frames))
        and np.all(frames == np.floor(frames))
        and np.all(frames[1:] >= frames[:-1])
        and np.all(np.abs(frames) < 2**50)
        and 0 <= last_difference < 2**50
    )


def _matlab_pairwise_expression(
    left: NDArray[Any], right: NDArray[Any]
) -> NDArray[np.float64]:
    """Source Euclidean cross-distance expression from line 71."""

    left_squared = np.sum(left**2, axis=1, keepdims=True)
    right_squared = np.sum(right**2, axis=1, keepdims=True).T
    with np.errstate(invalid="ignore"):
        return np.sqrt(left_squared + right_squared - 2 * (left @ right.T))


def _frame_cross_distances(
    left_frames: NDArray[Any], right_frames: NDArray[Any]
) -> NDArray[np.float64]:
    """Source lines 74--79, including the integer ``mtimes`` failure."""

    left = left_frames.reshape(-1, order="F")
    right = right_frames.reshape(-1, order="F")
    aaaa = np.column_stack((left, left * 0))
    bbbb = np.column_stack((right, right * 0))
    if np.issubdtype(aaaa.dtype, np.integer) or np.issubdtype(
        bbbb.dtype, np.integer
    ):
        raise DDCSourceBug003(
            "DDC-SOURCE-BUG-003: MATLAB mtimes requires the other operand "
            "to be scalar when either operand is an integer"
        )
    return _matlab_pairwise_expression(aaaa, bbbb)


def eliminate_blinking_de_loc15_mcmc(
    localizations_final: ArrayLike,
    frame: ArrayLike,
    resolution: Real,
    a: int,
    deviation_in_probability: ArrayLike,
    traj_num: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Port ``Eliminate_Blinking_De_Loc15_MCMC.m`` exactly by source loops."""

    locations = np.asarray(localizations_final)
    frames = np.asarray(frame)
    deviations = np.asarray(deviation_in_probability)
    labels = np.asarray(traj_num).reshape(-1, order="F")
    if locations.ndim != 2:
        raise ValueError("localizations_final must be a 2D array")
    if frames.ndim > 2 or (frames.ndim == 2 and min(frames.shape, default=0) > 1):
        raise ValueError("frame must be a MATLAB-style vector")
    frames = frames.reshape(-1, order="F")
    if locations.shape[0] != frames.size or labels.size < frames.size:
        raise ValueError(
            "frame length must match localization rows and traj_num must cover them"
        )
    if not isinstance(a, Integral) or a < 1:
        raise ValueError("A must be a positive integer")
    if not isinstance(resolution, Real) or resolution <= 0 or not np.isfinite(resolution):
        raise ValueError("Resolution must be finite and positive")
    if deviations.ndim != 2 or deviations.shape[0] < a or deviations.shape[1] == 0:
        raise ValueError("deviation_in_probability must have A rows and bins")

    native = try_eliminate(locations, frames, np.empty(0), resolution, a, deviations,
        labels[:frames.size], probability_mode=True)
    if native is not None:
        keep, probabilities, _ = native
        return keep.astype(np.float64), probabilities

    n_rows = locations.shape[0]
    matlab_length = _matlab_matrix_length(locations)
    linked = np.zeros(matlab_length, dtype=np.float64)
    is_blink = np.zeros(matlab_length, dtype=np.float64)
    trajectory = np.zeros(matlab_length, dtype=np.float64)
    probabilities = np.zeros(matlab_length, dtype=np.float64)

    search_frames = _can_search_frame_candidates(frames, a)
    for frame_difference in range(1, a + 1):
        starts = (
            np.searchsorted(frames, frames + frame_difference, side="left")
            if search_frames else None
        )
        # MATLAB uses length(Frame) for this search, not length(locations).
        for source in range(n_rows):
            investigate = int(starts[source]) - source if starts is not None else 0
            if starts is None:
                while _matlab_frame_subtract(
                    frames[source + investigate], frames[source], frames.dtype
                ) < frame_difference:
                    investigate += 1
                    if source + investigate >= frames.size:
                        break
            if source + investigate >= frames.size:
                break

            while _matlab_frame_subtract(
                frames[source + investigate], frames[source], frames.dtype
            ) == frame_difference:
                target = source + investigate
                if linked[source] != 1 and is_blink[target] != 1:
                    if trajectory[source] != 0:
                        source_group = np.flatnonzero(
                            trajectory == trajectory[source]
                        )
                    else:
                        source_group = np.array([source], dtype=np.intp)
                    if trajectory[target] != 0:
                        target_group = np.flatnonzero(
                            trajectory == trajectory[target]
                        )
                    else:
                        target_group = np.array([target], dtype=np.intp)

                    distance = _matlab_pairwise_expression(
                        locations[source_group, :], locations[target_group, :]
                    )
                    framer = _frame_cross_distances(
                        frames[source_group], frames[target_group]
                    )
                    distance_bins = (
                        np.floor(distance.reshape(-1, order="F") / resolution) + 1
                    )
                    distance_bins[distance_bins > deviations.shape[1]] = (
                        deviations.shape[1]
                    )

                    if np.max(framer) < a:
                        rows = framer.reshape(-1, order="F")
                        if (
                            np.any(~np.isfinite(rows))
                            or np.any(rows != np.floor(rows))
                            or np.any(rows < 1)
                            or np.any(rows > deviations.shape[0])
                            or np.any(distance_bins < 1)
                        ):
                            raise IndexError("MATLAB sub2ind received invalid subscripts")
                        # Equivalent to MATLAB column-major sub2ind and linear
                        # indexing, while retaining MATLAB's Distance(:) order.
                        selected = deviations[
                            rows.astype(np.intp) - 1,
                            distance_bins.astype(np.intp) - 1,
                        ]
                        mean_probability = float(np.mean(selected))
                        if mean_probability > 0.5:
                            is_blink[target] = 1
                            probabilities[target] = mean_probability
                            linked[source] = 1
                            if trajectory[source] == 0:
                                trajectory[source] = labels[source]
                            if trajectory[target] != 0:
                                target_members: NDArray[np.bool_] | int = (
                                    trajectory == trajectory[target]
                                )
                            else:
                                target_members = target
                            trajectory[target_members] = trajectory[source]
                        elif mean_probability > probabilities[target]:
                            probabilities[target] = mean_probability

                investigate += 1
                if source + investigate >= frames.size:
                    break

    retained = np.ones(matlab_length, dtype=np.float64)
    retained[is_blink > 0] = 0
    return retained, probabilities


def determine_deviation_in_probability8(
    true_distribution: ArrayLike,
    distribution_for_blink: ArrayLike,
    bins: ArrayLike,
    a: int,
    localizations_final: ArrayLike,
    frame_information: ArrayLike,
    loc: ArrayLike | None,
    fram: ArrayLike | None,
    resolution: Real,
    true_distribution2: ArrayLike,
    distribution_for_blink2: ArrayLike,
    bins2: ArrayLike,
    x_overall: ArrayLike | None,
    m_mat: ArrayLike | None,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Port ``Determine_Deviation_in_Probability8.m``.

    ``bins`` and ``fram`` have no active source use. ``loc`` only controls a
    dead assignment, so none of the three values affects the outputs.
    """

    del bins, loc, fram
    if not isinstance(a, Integral) or a < 1:
        raise ValueError("A must be a positive integer")
    locations = np.asarray(localizations_final)
    frames = np.asarray(frame_information)
    if locations.ndim != 2:
        raise ValueError("localizations_final must be a 2D array")
    if frames.ndim > 2 or (frames.ndim == 2 and min(frames.shape, default=0) > 1):
        raise ValueError("frame_information must be a MATLAB-style vector")
    frames = frames.reshape(-1, order="F")
    if locations.shape[0] != frames.size:
        raise ValueError("localization/frame length mismatch")

    true = np.asarray(true_distribution, dtype=np.float64).reshape(-1, order="F")
    blink = np.asarray(distribution_for_blink, dtype=np.float64).reshape(-1, order="F")
    true2 = np.asarray(true_distribution2, dtype=np.float64).reshape(-1, order="F")
    blink2 = np.asarray(distribution_for_blink2, dtype=np.float64).reshape(-1, order="F")
    fit_bins = np.asarray(bins2, dtype=np.float64).reshape(-1, order="F")
    if true.shape != blink.shape:
        raise ValueError("primary distributions must have the same shape")
    if true2.shape != blink2.shape or fit_bins.size != true2.size + 1:
        raise ValueError("fitting distributions and bins2 have incompatible shapes")

    distances = pdist(locations)
    z2 = _frame_pair_distances(frames)
    supplied_x = np.asarray([] if x_overall is None else x_overall, dtype=np.float64)
    if supplied_x.size == 0:
        dscale = np.empty(a, dtype=np.float64)
        for row in range(a):
            observed = _probability_histogram(
                distances[z2 == row + 1], fit_bins
            )
            dscale[row] = _fit_mixture_parameter(true2, blink2, observed)
    else:
        dscale = supplied_x.reshape(-1, order="F").copy()
    if dscale.size < a:
        raise IndexError("Dscale_store has fewer than A source elements")
    dscale[dscale > 1] = 1
    dscale[dscale < 0] = 1e-7
    dscale[-1] = 1

    supplied_m = np.asarray([] if m_mat is None else m_mat, dtype=np.float64)
    if supplied_m.size:
        deviation = supplied_m.copy()
    else:
        deviation = np.empty((a, true.size), dtype=np.float64)
        for row in range(a):
            mixture = blink * (1 - dscale[row]) + true * dscale[row]
            with np.errstate(divide="ignore", invalid="ignore"):
                deviation[row, :] = (mixture - dscale[row] * true) / mixture

    matlab_length = _matlab_matrix_length(locations)
    permutation = (
        np.random.permutation(matlab_length)
        if rng is None
        else rng.permutation(matlab_length)
    )
    retained, _ = eliminate_blinking_de_loc15_mcmc(
        locations, frames, resolution, a, deviation, permutation + 1
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        nb = (matlab_length - np.sum(retained)) / np.sum(retained)
        linked_fraction = (nb + nb + nb * nb) / (1 + nb + nb + nb * nb)

    for row in range(a):
        y = _probability_histogram(distances[z2 == row + 1], fit_bins)
        y = y - dscale[row] * (1 - linked_fraction) * true2
        y[y < 0] = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            y = y / np.sum(y)
        dscale[row] = _fit_mixture_parameter(true2, blink2, y)

    dscale[dscale > 1] = 1
    dscale[dscale < 0] = 1e-7
    dscale[-1] = 1
    probability_distributions = np.empty((a, true.size), dtype=np.float64)
    for row in range(a):
        mixture = blink * (1 - dscale[row]) + true * dscale[row]
        probability_distributions[row, :] = mixture
        with np.errstate(divide="ignore", invalid="ignore"):
            _ = (mixture - dscale[row] * true) / mixture

    positive = probability_distributions[probability_distributions > 0]
    if positive.size == 0:
        raise ValueError("MATLAB final positive-probability minimum is empty")
    probability_distributions += np.min(positive) / 10
    return deviation, probability_distributions, dscale
