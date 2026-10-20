"""Load and normalize input datasets produced by the MATLAB DDC pipeline.

Stage A.1 deliberately supports the repository's pre-v7.3 input MAT files.
Analyzed v7.3 files are a separate output format and are not normalized here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import tempfile
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat, savemat


class MatFileError(ValueError):
    """Raised when a MAT file cannot be interpreted as a DDC input dataset."""


@dataclass(slots=True)
class DDCDataset:
    """Normalized representation of a DDC input MAT file.

    MATLAB cell arrays are exposed as Python lists. Coordinate arrays retain
    their source dtype and have shape ``(n_localizations, n_dimensions)``;
    frame arrays retain their source dtype and are normalized to one dimension.
    Optional split-image bookkeeping remains aligned by image index.
    Ground truth is optional: omit ``true_localizations`` or pass ``None`` or
    ``[]`` for experimental data. It is used only for simulation diagnostics.
    """

    localizations: list[NDArray[Any]]
    frames: list[NDArray[Any]]
    true_localizations: list[NDArray[Any]] | None = None
    photons: list[NDArray[Any]] | None = None
    came_from_image: NDArray[Any] | None = None
    addon: NDArray[Any] | None = None
    cuts: tuple[
        list[NDArray[Any]] | None,
        list[NDArray[Any]] | None,
        list[NDArray[Any]] | None,
    ] = (None, None, None)
    parameters_to_split: list[NDArray[Any]] | None = None
    source: Path | None = None
    mat_passthrough: dict[str, Any] = field(default_factory=dict)

    @property
    def n_images(self) -> int:
        """Number of corresponding per-image records in the dataset."""

        return len(self.localizations)


def _cell_values(value: NDArray[Any], variable: str) -> list[NDArray[Any]]:
    if not isinstance(value, np.ndarray) or value.dtype != object:
        raise MatFileError(f"{variable} must be a MATLAB cell array")
    return [np.asarray(cell) for cell in value.ravel(order="F")]


def _coordinate_cells(
    value: NDArray[Any], variable: str, dimensions: int | None = None
) -> list[NDArray[Any]]:
    cells = _cell_values(value, variable)
    if dimensions is None:
        dimensions = next(
            (cell.shape[1] for cell in cells if cell.size and cell.ndim == 2),
            None,
        )
    if dimensions is None:
        raise MatFileError(f"{variable} contains no non-empty coordinate array")

    normalized: list[NDArray[Any]] = []
    for index, cell in enumerate(cells):
        if cell.size == 0:
            normalized.append(np.empty((0, dimensions), dtype=cell.dtype))
        elif cell.ndim == 2 and cell.shape[1] == dimensions:
            normalized.append(cell)
        else:
            raise MatFileError(
                f"{variable} cell {index} has shape {cell.shape}; "
                f"expected (n, {dimensions})"
            )
    return normalized


def _frame_cells(value: NDArray[Any]) -> list[NDArray[Any]]:
    cells = _cell_values(value, "Frame_Information")
    normalized: list[NDArray[Any]] = []
    for index, cell in enumerate(cells):
        if cell.ndim > 2 or (cell.ndim == 2 and min(cell.shape, default=0) > 1):
            raise MatFileError(
                f"Frame_Information cell {index} has shape {cell.shape}; "
                "expected a MATLAB row or column vector"
            )
        normalized.append(cell.reshape(-1, order="F"))
    return normalized


def _numeric_cell_vectors(
    value: NDArray[Any], variable: str
) -> list[NDArray[Any]]:
    """Normalize numeric MATLAB cell vectors without changing their dtype."""

    cells = _cell_values(value, variable)
    normalized: list[NDArray[Any]] = []
    for index, cell in enumerate(cells):
        if not np.issubdtype(cell.dtype, np.number):
            raise MatFileError(f"{variable} cell {index} must be numeric")
        if cell.ndim > 2 or (cell.ndim == 2 and min(cell.shape, default=0) > 1):
            raise MatFileError(
                f"{variable} cell {index} has shape {cell.shape}; "
                "expected a MATLAB row or column vector"
            )
        normalized.append(cell.reshape(-1, order="F"))
    return normalized


def _optional_vector(data: dict[str, Any], name: str) -> NDArray[Any] | None:
    if name not in data:
        return None
    value = np.asarray(data[name])
    if value.dtype == object or value.ndim > 2 or (
        value.ndim == 2 and min(value.shape, default=0) > 1
    ):
        raise MatFileError(f"{name} must be a MATLAB numeric vector")
    return value.reshape(-1, order="F")


def _optional_cells(
    data: dict[str, Any], name: str
) -> list[NDArray[Any]] | None:
    if name not in data:
        return None
    return [cell.reshape(-1, order="F") for cell in _cell_values(data[name], name)]


def _validate_correspondence(dataset: DDCDataset) -> None:
    n_images = dataset.n_images
    if len(dataset.frames) != n_images or (
        dataset.true_localizations is not None
        and len(dataset.true_localizations) not in (0, n_images)
    ):
        raise MatFileError(
            "LocalizationsFinal, Frame_Information, and TrueLocalizations "
            "must contain the same number of cells"
        )

    for index, (locations, frames) in enumerate(
        zip(dataset.localizations, dataset.frames, strict=True)
    ):
        if locations.shape[0] != frames.size:
            raise MatFileError(
                f"image {index}: {locations.shape[0]} localizations do not "
                f"match {frames.size} frame values"
            )

    aligned = {
        "Came_from_image": dataset.came_from_image,
        "addonarray": dataset.addon,
        "cut1array": dataset.cuts[0],
        "cut2array": dataset.cuts[1],
        "cut3array": dataset.cuts[2],
        "Parameters_to_split": dataset.parameters_to_split,
    }
    for name, values in aligned.items():
        if values is not None and len(values) != n_images:
            raise MatFileError(
                f"{name} has {len(values)} entries; expected {n_images}"
            )


def load_dataset(path: str | Path) -> DDCDataset:
    """Load a repository-style DDC input MAT file.

    The input must contain ``LocalizationsFinal`` and ``Frame_Information`` as
    MATLAB cell arrays. ``TrueLocalizations`` is optional because active
    ``run_script_DDC.m`` preinitializes it to ``[]`` before ``load`` and later
    expands a missing value to one empty cell per image. Pre-v7.3 MAT inputs are read
    through SciPy. MATLAB v7.3 analyzed outputs are detected and rejected with
    a targeted error because they use a different schema and are outside A.1.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    try:
        data = loadmat(source, squeeze_me=False, struct_as_record=False)
    except NotImplementedError as error:
        raise MatFileError(
            "MATLAB v7.3/HDF5 files are not DDC input files supported by "
            "Stage A.1; load a repository input MAT file instead"
        ) from error
    except (OSError, ValueError) as error:
        raise MatFileError(f"cannot read MAT file {source}") from error

    required = ("LocalizationsFinal", "Frame_Information")
    missing = [name for name in required if name not in data]
    if missing:
        raise MatFileError(f"missing required MAT variables: {', '.join(missing)}")

    localizations = _coordinate_cells(data["LocalizationsFinal"], "LocalizationsFinal")
    frames = _frame_cells(data["Frame_Information"])
    dimensions = localizations[0].shape[1]
    if "TrueLocalizations" in data:
        raw_truth = np.asarray(data["TrueLocalizations"])
        if raw_truth.dtype == object:
            true_localizations = _coordinate_cells(
                raw_truth, "TrueLocalizations", dimensions
            )
        elif raw_truth.size == 0 and np.issubdtype(raw_truth.dtype, np.number):
            # run_script_DDC initializes TrueLocalizations=[]; a loaded numeric
            # empty has the same isempty branch and becomes one empty per image.
            true_localizations = [
                np.empty((0, dimensions), dtype=raw_truth.dtype)
                for _ in localizations
            ]
        else:
            raise MatFileError(
                "TrueLocalizations must be a MATLAB cell array or numeric empty []"
            )
    else:
        true_localizations = [
            np.empty((0, dimensions), dtype=locations.dtype)
            for locations in localizations
        ]

    if localizations and true_localizations:
        if any(array.shape[1] != dimensions for array in true_localizations):
            raise MatFileError(
                "TrueLocalizations dimensionality does not match LocalizationsFinal"
            )

    represented = {
        "LocalizationsFinal",
        "Frame_Information",
        "TrueLocalizations",
        "Photons",
        "Came_from_image",
        "addonarray",
        "cut1array",
        "cut2array",
        "cut3array",
        "Parameters_to_split",
    }
    passthrough = {
        name: value
        for name, value in data.items()
        if not name.startswith("__") and name not in represented
    }

    dataset = DDCDataset(
        localizations=localizations,
        frames=frames,
        true_localizations=true_localizations,
        photons=(
            _numeric_cell_vectors(data["Photons"], "Photons")
            if "Photons" in data
            else None
        ),
        came_from_image=_optional_vector(data, "Came_from_image"),
        addon=_optional_vector(data, "addonarray"),
        cuts=(
            _optional_cells(data, "cut1array"),
            _optional_cells(data, "cut2array"),
            _optional_cells(data, "cut3array"),
        ),
        parameters_to_split=_optional_cells(data, "Parameters_to_split"),
        source=source,
        mat_passthrough=passthrough,
    )
    _validate_correspondence(dataset)
    return dataset


def _mat_cell(values) -> NDArray[Any]:
    cells = np.empty((1, len(values)), dtype=object)
    for i, value in enumerate(values):
        cells[0, i] = np.asarray(value)
    return cells


def save_tool_mat(path: str | Path, variables: dict[str, Any]) -> Path:
    """Atomically write a Level-5 MAT dataset or preprocessing diagnostic."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".mat", delete=False) as stream:
            temporary = Path(stream.name)
        savemat(temporary, variables, appendmat=False, do_compression=True, oned_as="row")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def save_dataset(dataset: DDCDataset, path: str | Path) -> Path:
    """Save a raw/split dataset readable by load_dataset and ``ddc run``.

    Retains coordinate/frame/photon dtypes, optional truth, and all split
    metadata. This is an input file writer, not a checkpoint writer.
    """
    _validate_correspondence(dataset)
    variables = dict(dataset.mat_passthrough)
    variables.update(LocalizationsFinal=_mat_cell(dataset.localizations),
                     Frame_Information=_mat_cell(dataset.frames))
    if dataset.true_localizations:
        variables["TrueLocalizations"] = _mat_cell(dataset.true_localizations)
    if dataset.photons is not None:
        if len(dataset.photons) != dataset.n_images or any(
            np.asarray(p).size != len(loc)
            for p, loc in zip(dataset.photons, dataset.localizations)
        ):
            raise MatFileError("Photons must align with localization cells and rows")
        variables["Photons"] = _mat_cell(dataset.photons)
    for name, values in (("Came_from_image", dataset.came_from_image),
                         ("addonarray", dataset.addon)):
        if values is not None:
            variables[name] = values
    for name, values in zip(("cut1array", "cut2array", "cut3array", "Parameters_to_split"),
                            (*dataset.cuts, dataset.parameters_to_split)):
        if values is not None:
            variables[name] = _mat_cell(values)
    return save_tool_mat(path, variables)
