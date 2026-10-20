"""Faithful Stage-A ports of the active DDC trajectory/density scripts.

``Determine_Locs_w_Multi2.m`` and ``Density_Calc.m`` are MATLAB scripts that
read variables from, and write variables into, the caller workspace. Their
Python boundary is represented by two narrow functions because the scripts do
not depend on each other's outputs:

``determine_locs_w_multi2``
    Reads ``LocalizationsFinal``, ``Frame_Information``, and ``Resolution``;
    produces the double row-vector ``Frames_w_Multi``. Its loop temporaries are
    dead after the script.
``density_calc``
    Reads those variables plus ``A``; produces scalar ``DistanceDens`` and the
    double column-vector ``density``. ``DDC_MCMC`` consumes only ``density``;
    ``DistanceDens`` is visible script state but otherwise dead.

The density implementation uses bounded block vectorization only for finite
``float32``/``float64`` coordinates and frames. All other inputs retain the
original ordered nested loops. MATLAB ``length(LocalizationsFinal)`` is zero
for an empty matrix and otherwise ``max(n,d)``. Therefore a nonempty ``n<d``
coordinate matrix reaches an out-of-range row access in the source loops; the
port reproduces that failure before selecting either calculation path.

Repository frames are ``uint16``. In ``Density_Calc``, scalar subtraction is
performed in that class and saturates before ``abs``. Consequently the ordered
pair with an earlier frame on the left can have difference zero while its
reverse has the positive difference. This is DDC-SOURCE-BUG-004 and is
preserved without a cast. ``Determine_Locs_w_Multi2`` only indexes and compares
frame values, so it has no corresponding integer-arithmetic failure.
"""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .probabilities import _matlab_frame_subtract, _matlab_matrix_length


def _validate_script_inputs(
    localizations_final: ArrayLike,
    frame_information: ArrayLike,
    resolution: Real,
) -> tuple[NDArray[Any], NDArray[Any]]:
    locations = np.asarray(localizations_final)
    frames = np.asarray(frame_information)
    if locations.ndim != 2:
        raise ValueError("LocalizationsFinal must be a 2D matrix")
    if frames.ndim > 2 or (frames.ndim == 2 and min(frames.shape, default=0) > 1):
        raise ValueError("Frame_Information must be a MATLAB-style vector")
    frames = frames.reshape(-1, order="F")
    if locations.shape[0] != frames.size:
        raise ValueError("localization rows and frame length must match")
    if not isinstance(resolution, Real):
        raise TypeError("Resolution must be a real scalar")
    return locations, frames


def determine_locs_w_multi2(
    localizations_final: ArrayLike,
    frame_information: ArrayLike,
    resolution: Real,
) -> NDArray[np.float64]:
    """Port ``Determine_Locs_w_Multi2.m`` in its original loop order.

    Returned values preserve first-frame-occurrence order. A frame is included
    once iff it occurs more than once and at least one distinct ordered pair on
    that frame has Euclidean distance strictly below ``4*Resolution``.
    """

    locations, frames = _validate_script_inputs(
        localizations_final, frame_information, resolution
    )
    frames_w_multi: list[float] = []

    for frame_index in range(frames.size):
        frame_value = frames[frame_index]
        same_frame = np.flatnonzero(frames == frame_value)
        already_stored = sum(value == frame_value for value in frames_w_multi)

        if same_frame.size > 1 and already_stored == 0:
            go_w_multi = 0
            for left_index in same_frame:
                for right_index in same_frame:
                    if right_index != left_index:
                        datapoint = locations[left_index, :] - locations[right_index, :]
                        with np.errstate(over="ignore", invalid="ignore"):
                            distance = np.sqrt(np.sum(datapoint**2))
                        if distance < resolution * 4:
                            go_w_multi = 1
                            break
            if go_w_multi == 1:
                # [] is a double MATLAB array, so indexed assignment converts
                # integer frame scalars to double.
                frames_w_multi.append(float(frame_value))

    return np.asarray(frames_w_multi, dtype=np.float64)


def _matlab_abs_frame_difference(
    left: Any, right: Any, dtype: np.dtype[Any]
) -> Any:
    difference = _matlab_frame_subtract(left, right, dtype)
    if np.issubdtype(dtype, np.integer):
        return min(abs(int(difference)), np.iinfo(dtype).max)
    return abs(difference)


_DENSITY_BLOCK_SIZE = 32


def _density_nested_counts(
    locations: NDArray[Any], frames: NDArray[Any], distance_dens: Real, a: Real
) -> NDArray[np.float64]:
    """Faithful fallback, including MATLAB integer-frame subtraction."""

    counts = np.zeros((locations.shape[0], 1), dtype=np.float64)
    for left_index in range(locations.shape[0]):
        for right_index in range(locations.shape[0]):
            datapoint = locations[left_index, :] - locations[right_index, :]
            with np.errstate(over="ignore", invalid="ignore"):
                distance = np.sqrt(np.sum(datapoint**2))
            frame_difference = _matlab_abs_frame_difference(
                frames[left_index], frames[right_index], frames.dtype
            )
            if distance <= distance_dens and frame_difference > a:
                counts[left_index, 0] += 1
    return counts


def _density_blocked_counts(
    locations: NDArray[Any], frames: NDArray[Any], distance_dens: Real, a: Real
) -> NDArray[np.float64]:
    """Exact floating-input pair predicates in bounded source-row blocks."""

    counts = np.zeros((locations.shape[0], 1), dtype=np.float64)
    for start in range(0, locations.shape[0], _DENSITY_BLOCK_SIZE):
        stop = min(start + _DENSITY_BLOCK_SIZE, locations.shape[0])
        with np.errstate(over="ignore", invalid="ignore"):
            delta = locations[start:stop, None, :] - locations[None, :, :]
            distance = np.sqrt(np.sum(delta**2, axis=2))
            frame_difference = np.abs(
                frames[start:stop, None] - frames[None, :]
            )
        mask = (distance <= distance_dens) & (frame_difference > a)
        counts[start:stop, 0] = np.count_nonzero(mask, axis=1)
    return counts


def density_calc(
    localizations_final: ArrayLike,
    frame_information: ArrayLike,
    resolution: Real,
    a: Real,
) -> tuple[Real, NDArray[np.float64]]:
    """Port ``Density_Calc.m`` with an exact blocked floating fast path.

    The return order makes both script-created variables explicit:
    ``(DistanceDens, density)``. ``density`` always has MATLAB column shape
    ``(length(LocalizationsFinal), 1)`` when the script completes.
    """

    locations, frames = _validate_script_inputs(
        localizations_final, frame_information, resolution
    )
    if not isinstance(a, Real):
        raise TypeError("A must be a real scalar")

    distance_dens = 2 * resolution
    matlab_length = _matlab_matrix_length(locations)
    density = np.zeros((matlab_length, 1), dtype=np.float64)

    # A nonempty n<d matrix makes the original source index a nonexistent row.
    if matlab_length > locations.shape[0]:
        raise IndexError(
            "MATLAB Density_Calc indexes beyond LocalizationsFinal rows because "
            "length(LocalizationsFinal)=max(n,d)"
        )

    floating_fast_path = (
        locations.dtype in (np.dtype(np.float32), np.dtype(np.float64))
        and frames.dtype in (np.dtype(np.float32), np.dtype(np.float64))
        and np.all(np.isfinite(locations))
        and np.all(np.isfinite(frames))
    )
    density = (
        _density_blocked_counts(locations, frames, distance_dens, a)
        if floating_fast_path
        else _density_nested_counts(locations, frames, distance_dens, a)
    )

    if density.size == 0:
        return distance_dens, density
    minimum = np.min(density)
    denominator = np.max(density) - minimum
    with np.errstate(divide="ignore", invalid="ignore"):
        density = (density - minimum) / denominator
    return distance_dens, density
