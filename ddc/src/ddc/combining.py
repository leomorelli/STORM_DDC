"""Reconstruct images from corrected, buffered regions using NumPy arrays.

There is no file-format dependency here. ``DDCDataset`` holds the corrected
coordinates, their corresponding frames, and the split-region geometry.
MAT adapters currently live in ``combine_io``; other formats can reuse this
function without changing numerical behavior.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from .io import DDCDataset


def _corrected_cells(localizations, frames):
    if len(localizations) != len(frames):
        raise ValueError("corrected coordinate and frame cells must have equal counts")
    locations_out, frames_out = [], []
    for index, (loc, frame) in enumerate(zip(localizations, frames)):
        if loc is None or frame is None:
            raise ValueError(f"region {index + 1}: missing corrected coordinates or frames")
        loc, frame = np.asarray(loc), np.asarray(frame)
        if not loc.size and loc.shape in ((0,), (0, 0)):
            loc = np.empty((0, 3), dtype=loc.dtype)
        if loc.ndim != 2 or loc.shape[1] not in (2, 3):
            raise ValueError(f"region {index + 1}: coordinates must have shape (n, 2) or (n, 3)")
        if frame.ndim > 2 or (frame.ndim == 2 and min(frame.shape) > 1):
            raise ValueError(f"region {index + 1}: frames must be a vector")
        frame = frame.reshape(-1, order="F")
        if len(loc) != len(frame):
            raise ValueError(f"region {index + 1}: corrected coordinates and frames do not align")
        if not np.isrealobj(loc) or not np.isrealobj(frame) or not (
            np.isfinite(loc).all() and np.isfinite(frame).all()
        ):
            raise ValueError(f"region {index + 1}: coordinates and frames must be finite real numbers")
        if loc.shape[1] == 2:
            loc = np.column_stack((loc, np.zeros(len(loc), dtype=loc.dtype)))
        locations_out.append(loc)
        frames_out.append(frame)
    return locations_out, frames_out


def _geometry(regions: DDCDataset):
    n = regions.n_images
    if not n:
        raise ValueError("combining requires at least one region")
    if regions.came_from_image is None or len(regions.cuts) != 3 or any(
        axis is None for axis in regions.cuts
    ):
        raise ValueError("combining requires Came_from_image and all three cut arrays from splitting")
    origin = np.asarray(regions.came_from_image)
    if origin.ndim > 2 or (origin.ndim == 2 and min(origin.shape) > 1):
        raise ValueError("Came_from_image must be a vector")
    origin = origin.reshape(-1)
    if len(origin) != n or not np.isrealobj(origin) or not np.isfinite(origin).all() or (
        (origin < 1).any() or (origin != np.floor(origin)).any()
    ):
        raise ValueError("Came_from_image must have one positive integer image ID per region")
    # Both supported splitters preserve every source image, including empty
    # images. Requiring contiguous IDs catches damaged mappings before allocation.
    ids = np.unique(origin)
    if ids[-1] != len(ids):
        raise ValueError("Came_from_image IDs must be contiguous from 1")
    origin = origin.astype(np.int64)
    cuts = np.empty((n, 3, 2), dtype=float)
    for axis, values in enumerate(regions.cuts):
        if len(values) != n:
            raise ValueError(f"cut{axis + 1}array must have one interval per region")
        for i, interval in enumerate(values):
            interval = np.asarray(interval)
            if interval.size != 2 or not np.isrealobj(interval) or not np.isfinite(interval).all():
                raise ValueError(f"region {i + 1}: cut{axis + 1} must contain two finite endpoints")
            cuts[i, axis] = interval.reshape(-1)
    if (cuts[:, :, 0] > cuts[:, :, 1]).any():
        raise ValueError("cut intervals must have lower <= upper")
    return origin, cuts


def combine_images(
    regions: DDCDataset, *,
    boundary_mode: Literal["partition", "matlab"] = "partition",
    progress: Callable[[str], None] | None = None,
) -> DDCDataset:
    """Remove buffers and concatenate corrected regions by original image.

    ``regions.localizations`` and ``regions.frames`` must contain the corrected
    outputs, not the raw buffered inputs. Coordinates remain in their original
    global system (no translation). Input rows are never mutated or sorted.

    By default, inclusive core boxes are partitioned by region order, separately
    for each source image. A point on a shared face/edge/corner belongs to the
    first region containing it. This also handles repeated quantile cuts and
    planar data without deduplicating distinct localizations at identical
    coordinates. Empty images are preserved; output coordinates have 3 columns.

    ``boundary_mode='matlab'`` reproduces Combine_Images.m's corner-sum test,
    global Cutsstore, inclusive z boundaries, and cut3(2)==0 special case. That
    mode intentionally retains the source's boundary defects for comparison.
    """
    if boundary_mode not in ("partition", "matlab"):
        raise ValueError("boundary_mode must be 'partition' or 'matlab'")
    origins, cuts = _geometry(regions)
    locations, frames = _corrected_cells(regions.localizations, regions.frames)
    n_images = int(origins.max())
    location_parts = [[] for _ in range(n_images)]
    frame_parts = [[] for _ in range(n_images)]
    previous = [[] for _ in range(n_images)]
    corners = []
    for i, (loc, frame, origin) in enumerate(zip(locations, frames, origins)):
        image = int(origin) - 1
        box = cuts[i].copy()
        lower, upper = box[:, 0], box[:, 1]
        if boundary_mode == "matlab":
            if upper[2] == 0:
                box[2] = [-1000, 1000]
            lower, upper = box[:, 0], box[:, 1]
            lower_seen = upper_seen = False
            if corners:
                seen = np.asarray(corners)
                lower_seen = bool(np.any(np.sum(lower[:2] - seen, axis=1) == 0))
                upper_seen = bool(np.any(np.sum(upper[:2] - seen, axis=1) == 0))
            keep = ((loc >= lower) & (loc <= upper)).all(axis=1)
            if lower_seen:
                keep &= (loc[:, :2] > lower[:2]).all(axis=1)
            if upper_seen:
                keep &= (loc[:, :2] < upper[:2]).all(axis=1)
            corners.extend((upper[:2].copy(), lower[:2].copy()))
        else:
            keep = ((loc >= lower) & (loc <= upper)).all(axis=1)
            # Only boxes touching/overlapping this core can share its points.
            for prior_index in previous[image]:
                prior = cuts[prior_index]
                if np.all(lower <= prior[:, 1]) and np.all(upper >= prior[:, 0]):
                    keep &= ~((loc >= prior[:, 0]) & (loc <= prior[:, 1])).all(axis=1)
            previous[image].append(i)
        location_parts[image].append(loc[keep])
        frame_parts[image].append(frame[keep])
        if progress:
            progress(f"Combining region {i + 1}/{len(locations)} into image {origin}: "
                     f"kept {int(keep.sum())}/{len(loc)} corrected localizations")
    return DDCDataset(
        localizations=[np.concatenate(parts, axis=0) for parts in location_parts],
        frames=[np.concatenate(parts) for parts in frame_parts],
        source=regions.source,
        mat_passthrough={"Combination_Boundary_Mode": boundary_mode},
    )


__all__ = ["combine_images"]
