"""Faithful Stage-A port of ``Make_Distance_Matrix_Individual.m``.

MATLAB contract
---------------
Inputs
~~~~~~
``Localizations``
    Numeric array with shape ``(n_localizations, n_dimensions)``. The source
    operates row-wise and supports both 2D and 3D coordinates. All real DDC
    repository fixtures store localization coordinates as ``float64``. That is
    the supported Stage-A contract; integer-coordinate arrays are outside the
    real data contract because subtraction and squaring retain integer dtype
    semantics and can saturate or overflow before the square root.
``DistanceMatrix``
    MATLAB cell array whose cells are numeric vectors. Existing values are
    retained and new distances are appended. It must contain every cell that
    can be addressed by an accepted positive frame difference.
``frame``
    Row or column vector containing one frame value per localization.
``B``
    Inclusive maximum frame difference. The condition in MATLAB is
    ``abs(frame(i)-frame(jj)) < B+1``.

Output
~~~~~~
The returned list is the Python value equivalent of the updated MATLAB cell
array. Each element is a one-dimensional NumPy array. Inputs are not mutated,
matching MATLAB value semantics at the function boundary.

Index and bin semantics
-----------------------
MATLAB sorts ``frame`` and applies the same permutation to ``Localizations``.
Its loops are 1-based and use ``jj=i:length(frame)``. Pairs from the same frame
are excluded by ``(frame(jj)-frame(i)) > 0``. For every remaining pair, the
positive integer frame difference is used *directly* as a 1-based cell index:

===========================  =================  =============================
Frame difference             MATLAB cell        Python list element
===========================  =================  =============================
1                            DistanceMatrix{1}  distance_matrix[0]
2                            DistanceMatrix{2}  distance_matrix[1]
...                          ...                ...
B                            DistanceMatrix{B}  distance_matrix[B - 1]
===========================  =================  =============================

There is no bin for frame difference zero. A possible extra MATLAB cell
``DistanceMatrix{B+1}``, as allocated by the active caller, remains untouched.

Edge cases preserved
--------------------
Repeated frame values produce no distances. Unsorted frames are accepted and
sorted internally. Differences equal to ``B`` are included; differences equal
to ``B+1`` are excluded. A single localization and bins with no qualifying
pairs leave the corresponding cells unchanged. MATLAB cell indices must be
positive integers, so an accepted non-integral frame difference is invalid.
NaN frame differences fail the comparison and are ignored. Coordinate
arithmetic follows the input NumPy dtype. A small signed-integer fixture is
tested only to make this behavior explicit; callers should supply floating-
point coordinates as found in every real repository MAT input. MATLAB integer
arithmetic saturates: consequently, with an
unsigned frame dtype and ascending unequal frames,
``abs(frame(i)-frame(jj))`` becomes zero even though the opposite subtraction
used by the positivity check is nonzero, and the original function attempts
the invalid cell subscript ``DistanceMatrix{0}``. Python reproduces that error
rather than silently casting unsigned frames to a different arithmetic model.
Finite float32/float64 repository-style inputs use a conservative fast path
that vectorizes only frame-candidate discovery. Accepted coordinate distances
still use the exact scalar source expression and are collected in original
``i,jj`` order, then appended once per cell. Integer, nonfinite, and unusual inputs retain the nested-loop
fallback. Neither path uses ``pdist`` or spatial indexing.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from ._acceleration import try_distance_bins


def _matlab_scalar_subtract(left: np.generic, right: np.generic) -> Real:
    """Subtract scalars using MATLAB's saturating integer arithmetic."""

    dtype = np.result_type(left.dtype, right.dtype)
    left_value = left.item()
    right_value = right.item()
    difference = left_value - right_value
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        difference = min(max(difference, limits.min), limits.max)
    return difference


def _append_distance(
    result: list[NDArray[Any]],
    sorted_locations: NDArray[Any],
    i: int,
    jj: int,
    absolute_difference: Real,
) -> None:
    """Validate the MATLAB cell index and append the scalar-source distance."""

    if not isinstance(absolute_difference, Integral) and not float(
        absolute_difference
    ).is_integer():
        raise IndexError("an accepted frame difference is not a valid MATLAB cell index")

    matlab_cell_index = int(absolute_difference)
    if matlab_cell_index < 1:
        raise IndexError("DistanceMatrix{0} is invalid under MATLAB 1-based indexing")
    python_index = matlab_cell_index - 1
    if python_index >= len(result):
        raise IndexError(f"DistanceMatrix{{{matlab_cell_index}}} does not exist")

    datapoint = sorted_locations[i, :] - sorted_locations[jj, :]
    distance = np.sum(datapoint**2) ** 0.5
    result[python_index] = np.append(result[python_index], distance)


def _append_float_frame_candidates(
    result: list[NDArray[Any]],
    sorted_locations: NDArray[Any],
    sorted_frames: NDArray[Any],
    b: Real,
) -> None:
    """Discover finite floating-frame candidates without changing pair order."""

    # Grow Python lists, then concatenate once per cell. Repeated np.append
    # otherwise copies an ever-growing array for every accepted pair.
    additions: list[list[Any]] = [[] for _ in result]
    for i in range(sorted_locations.shape[0]):
        tail = sorted_frames[i:]
        backward_differences = sorted_frames[i] - tail
        forward_differences = tail - sorted_frames[i]
        absolute_differences = np.abs(backward_differences)
        candidates = np.flatnonzero(
            (absolute_differences < b + 1) & (forward_differences > 0)
        )
        for offset in candidates:
            jj = i + int(offset)
            difference = absolute_differences[offset]
            if not float(difference).is_integer():
                raise IndexError("an accepted frame difference is not a valid MATLAB cell index")
            index = int(difference) - 1
            if index >= len(result):
                raise IndexError(f"DistanceMatrix{{{index + 1}}} does not exist")
            datapoint = sorted_locations[i, :] - sorted_locations[jj, :]
            distance = np.sum(datapoint**2) ** 0.5
            additions[index].append(distance)
    for index, values in enumerate(additions):
        if values:
            result[index] = np.concatenate((result[index], np.asarray(values)))


def _append_nested_candidates(
    result: list[NDArray[Any]],
    sorted_locations: NDArray[Any],
    sorted_frames: NDArray[Any],
    b: Real,
) -> None:
    """Accepted Stage-A fallback including saturating integer subtraction."""

    for i in range(sorted_locations.shape[0]):
        for jj in range(i, sorted_frames.size):
            backward_difference = _matlab_scalar_subtract(
                sorted_frames[i], sorted_frames[jj]
            )
            forward_difference = _matlab_scalar_subtract(
                sorted_frames[jj], sorted_frames[i]
            )
            absolute_difference = abs(backward_difference)

            if absolute_difference < b + 1 and forward_difference > 0:
                _append_distance(
                    result, sorted_locations, i, jj, absolute_difference
                )


def make_distance_matrix_individual(
    localizations: ArrayLike,
    distance_matrix: Sequence[ArrayLike],
    frame: ArrayLike,
    b: Real,
) -> list[NDArray[Any]]:
    """Append Euclidean distances to bins indexed by positive frame difference."""

    locations = np.asarray(localizations)
    frames = np.asarray(frame)

    if locations.ndim != 2:
        raise ValueError("localizations must have shape (n_localizations, n_dimensions)")
    if frames.ndim > 2 or (frames.ndim == 2 and min(frames.shape, default=0) > 1):
        raise ValueError("frame must be a MATLAB-style row or column vector")

    frames = frames.reshape(-1, order="F")
    if locations.shape[0] != frames.size:
        raise ValueError("localizations and frame must contain the same number of rows")
    if not isinstance(b, Real) or not np.isfinite(b):
        raise ValueError("b must be a finite real number")

    # MATLAB `sort` is stable; use the explicit stable algorithm here.
    indices = np.argsort(frames, kind="stable")
    sorted_frames = frames[indices]
    sorted_locations = locations[indices, :]

    # MATLAB inputs are values. Copy each numeric cell before appending.
    result = [np.asarray(cell).reshape(-1, order="F").copy() for cell in distance_matrix]

    native = try_distance_bins(sorted_locations, sorted_frames, result, b)
    if native is not None:
        return native

    floating_fast_path = (
        frames.dtype in (np.dtype(np.float32), np.dtype(np.float64))
        and locations.dtype in (np.dtype(np.float32), np.dtype(np.float64))
        and np.all(np.isfinite(frames))
        and np.all(np.isfinite(locations))
    )
    if floating_fast_path:
        _append_float_frame_candidates(result, sorted_locations, sorted_frames, b)
    else:
        _append_nested_candidates(result, sorted_locations, sorted_frames, b)

    return result


__all__ = ["make_distance_matrix_individual"]
