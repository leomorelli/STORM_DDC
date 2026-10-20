"""MAT file adapters for combining; the numerical algorithm is format-neutral."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import h5py
import hdf5storage
import numpy as np
from scipy.io import loadmat

from .combining import _corrected_cells, _geometry
from .io import DDCDataset, MatFileError, _cell_values, _mat_cell, save_tool_mat


_FINAL_LOC = "Final_Loc_Blinking_Corrected"
_FINAL_FRAME = "Final_Frame_Blinking_Corrected"
_CUTS = ("cut1array", "cut2array", "cut3array")


def _load_variables(path, names):
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if h5py.is_hdf5(source):
        # Load only combining inputs; large/sparse MCMC caches are not needed.
        return hdf5storage.loadmat(source, variable_names=names)
    return loadmat(source, variable_names=names, squeeze_me=False, struct_as_record=False)


def _required_cells(data, name):
    if name not in data:
        raise MatFileError(f"missing {name}; combining requires an analyzed split dataset")
    value = np.asarray(data[name])
    # Some MATLAB versions initialize unset output cell collections as [].
    if value.size == 0 and value.dtype != object:
        return []
    return _cell_values(value, name)


def load_corrected_regions(
    path: str | Path, *, allow_incomplete: bool = False,
    progress: Callable[[str], None] | None = None,
) -> DDCDataset:
    """Load corrected regions from a Level-5 or v7.3 analyzed MAT file.

    Both Python and historical MATLAB outputs are supported. When Step and
    stepper exist, unfinished runs require explicit allow_incomplete=True.
    Missing results are never replaced by raw data for ordinary regions.
    Confirmed complete regions skipped by DDC (<=10 raw localizations) pass
    through unchanged, with a progress message. Real empty (0,3) results remain
    empty; absent MATLAB [] placeholders are distinguished from them.
    """
    names = [_FINAL_LOC, _FINAL_FRAME, "Came_from_image", *_CUTS, "Step", "stepper"]
    data = _load_variables(path, names)
    if "Came_from_image" not in data or any(name not in data for name in _CUTS):
        raise MatFileError("combining requires Came_from_image and cut1array/cut2array/cut3array; "
                           "run DDC on the output of ddc -split")
    origins = np.asarray(data["Came_from_image"])
    n = origins.size
    loc = _required_cells(data, _FINAL_LOC)
    frames = _required_cells(data, _FINAL_FRAME)
    if len(loc) != len(frames) or len(loc) > n:
        raise MatFileError("corrected coordinate/frame cell counts do not match the split regions")
    if ("Step" in data) != ("stepper" in data):
        raise MatFileError("incomplete checkpoint metadata: Step and stepper must both be present")
    complete = np.zeros(n, dtype=bool)
    if "Step" in data:
        steps = np.asarray(data["Step"]).reshape(-1)
        target = np.asarray(data["stepper"]).reshape(-1)
        if (len(steps) != n or target.size != 1 or not np.isfinite(steps).all()
                or not np.isfinite(target).all() or target[0] <= 0):
            raise MatFileError("invalid Step/stepper completion metadata")
        complete = steps >= target[0]
        if not complete.all() and not allow_incomplete:
            raise MatFileError("DDC analysis is unfinished; resume it first, or use "
                               "--allow-incomplete to combine the available current results")
        if not complete.all() and progress:
            progress(f"Combining current results from an unfinished run ({int((~complete).sum())} regions unfinished)")
    loc.extend([None] * (n - len(loc)))
    frames.extend([None] * (n - len(frames)))
    missing = [i for i, cell in enumerate(loc)
               if cell is None or (cell.size == 0 and cell.shape not in ((0, 2), (0, 3)))]
    if missing:
        raw = _load_variables(path, ["LocalizationsFinal", "Frame_Information"])
        raw_loc = _required_cells(raw, "LocalizationsFinal")
        raw_frames = _required_cells(raw, "Frame_Information")
        if len(raw_loc) != n or len(raw_frames) != n:
            raise MatFileError("raw coordinate/frame cells do not match the split regions")
        for i in missing:
            if complete[i] and raw_loc[i].ndim == 2 and raw_loc[i].shape[0] <= 10:
                if frames[i] is not None and frames[i].size:
                    raise MatFileError(f"region {i + 1}: frames exist without corrected coordinates")
                loc[i], frames[i] = raw_loc[i], raw_frames[i]
                if progress:
                    progress(f"Region {i + 1}: retaining {len(loc[i])} raw localizations skipped by DDC")
            else:
                raise MatFileError(f"region {i + 1}: no corrected output; finish DDC before combining")
    loc, frames = _corrected_cells(loc, frames)
    result = DDCDataset(loc, frames, came_from_image=origins,
                        cuts=tuple(_required_cells(data, name) for name in _CUTS),
                        source=Path(path))
    _geometry(result)
    return result


def save_combined_images(dataset: DDCDataset, path: str | Path) -> Path:
    """Write MATLAB's two combined-result cells without MCMC/split state."""
    if dataset.came_from_image is not None or any(c is not None for c in dataset.cuts):
        raise ValueError("save_combined_images requires the result of combine_images, not split regions")
    loc, frames = _corrected_cells(dataset.localizations, dataset.frames)
    variables = {_FINAL_LOC: _mat_cell(loc),
                 _FINAL_FRAME: _mat_cell([f.reshape(-1, 1) for f in frames])}
    if "Combination_Boundary_Mode" in dataset.mat_passthrough:
        variables["Combination_Boundary_Mode"] = dataset.mat_passthrough["Combination_Boundary_Mode"]
    if dataset.source is not None:
        variables["Source_Analyzed_File"] = str(dataset.source)
    return save_tool_mat(path, variables)


def load_combined_images(path: str | Path) -> DDCDataset:
    """Read combined-result MAT files into plain coordinate/frame arrays."""
    data = _load_variables(path, [_FINAL_LOC, _FINAL_FRAME, "Came_from_image",
                                  "Combination_Boundary_Mode"])
    if "Came_from_image" in data:
        raise MatFileError("file still contains split regions; use load_corrected_regions and combine_images")
    loc, frames = _corrected_cells(_required_cells(data, _FINAL_LOC),
                                   _required_cells(data, _FINAL_FRAME))
    if not loc:
        raise MatFileError("combined output contains no image cells")
    return DDCDataset(loc, frames, source=Path(path),
                      mat_passthrough={k: data[k] for k in ("Combination_Boundary_Mode",) if k in data})


__all__ = ["load_corrected_regions", "load_combined_images", "save_combined_images"]
