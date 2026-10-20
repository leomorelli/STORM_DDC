from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from ddc.io import MatFileError, _coordinate_cells, load_dataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPOSITORY_ROOT / "Main_DDC_Folder" / "User_Guide_Files"
FIGURE_DATA = (
    REPOSITORY_ROOT
    / "Main_DDC_Folder"
    / "Figure_1_and_2_data_and_analyzed_data"
)


def test_load_real_2d_example_preserves_shapes_values_and_dtypes() -> None:
    path = EXAMPLES / "Simulation_2_dark_state_sparse_Clusters_10per_3_per_ROI.mat"
    dataset = load_dataset(path)

    assert dataset.n_images == 24
    assert dataset.localizations[0].shape == (4000, 2)
    assert dataset.frames[0].shape == (4000,)
    assert dataset.true_localizations[0].shape == (984, 2)
    assert dataset.localizations[0].dtype == np.float64
    assert dataset.frames[0].dtype == np.uint16
    assert dataset.true_localizations[0].dtype == np.float64
    np.testing.assert_array_equal(dataset.frames[0][:5], [461, 5316, 5320, 5329, 5332])
    np.testing.assert_allclose(
        dataset.localizations[0][0], [-900.7763894999252, 677.114011965691]
    )
    assert dataset.came_from_image is None
    assert dataset.addon is None
    assert dataset.cuts == (None, None, None)
    assert dataset.parameters_to_split is None
    assert dataset.photons is None


def test_real_repository_photons_preserve_cells_values_and_uint8_dtype() -> None:
    path = EXAMPLES / "Split_Example_3d_2darkstate_random_data.mat"
    if not path.exists():
        pytest.skip("repository photon MAT fixture is unavailable")
    raw = loadmat(path, squeeze_me=False, struct_as_record=False)
    assert "Photons" in raw
    raw_cells = raw["Photons"].ravel(order="F")

    dataset = load_dataset(path)

    assert dataset.photons is not None
    assert len(dataset.photons) == len(raw_cells) == 32
    for index in (0, 15, 31):
        expected = np.asarray(raw_cells[index]).reshape(-1, order="F")
        assert expected.size > 0
        np.testing.assert_array_equal(dataset.photons[index], expected)
        assert dataset.photons[index].dtype == expected.dtype == np.uint8


def test_load_real_split_3d_example_preserves_image_correspondence() -> None:
    path = FIGURE_DATA / "Split_Simulation_2_dark_state_sparse_Clusters_10per_3_per_ROI.mat"
    dataset = load_dataset(path)

    assert dataset.n_images == 24
    assert dataset.came_from_image is not None
    assert dataset.addon is not None
    np.testing.assert_array_equal(dataset.came_from_image[:6], [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(dataset.addon[:6], [250] * 6)
    assert dataset.came_from_image.dtype == np.uint8
    assert dataset.addon.dtype == np.uint8

    for locations, frames, truth in zip(
        dataset.localizations,
        dataset.frames,
        dataset.true_localizations,
        strict=True,
    ):
        assert locations.shape[0] == frames.size
        assert locations.shape[1] == truth.shape[1] == 3
        assert locations.dtype == truth.dtype == np.float64
        assert frames.dtype == np.uint16

    assert all(cut is not None and len(cut) == dataset.n_images for cut in dataset.cuts)
    assert dataset.parameters_to_split is not None
    assert len(dataset.parameters_to_split) == dataset.n_images
    np.testing.assert_allclose(
        dataset.cuts[0][0], [-1108.4783436674795, 1080.1478962860467]
    )
    np.testing.assert_array_equal(dataset.parameters_to_split[0], [1, 1, 1])


def test_normalized_values_match_raw_real_mat_cells_per_image() -> None:
    path = EXAMPLES / "Split_Example_3d_2darkstate_random_data.mat"
    raw = loadmat(path, squeeze_me=False, struct_as_record=False)
    dataset = load_dataset(path)

    raw_locations = raw["LocalizationsFinal"].ravel(order="F")
    raw_frames = raw["Frame_Information"].ravel(order="F")
    raw_truth = raw["TrueLocalizations"].ravel(order="F")
    assert dataset.n_images == 32

    for index in (0, 7, 15, 31):
        np.testing.assert_array_equal(dataset.localizations[index], raw_locations[index])
        np.testing.assert_array_equal(
            dataset.frames[index], raw_frames[index].reshape(-1, order="F")
        )
        np.testing.assert_array_equal(dataset.true_localizations[index], raw_truth[index])


def test_empty_matlab_cells_are_normalized_without_dtype_loss() -> None:
    cells = np.empty((1, 2), dtype=object)
    cells[0, 0] = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    cells[0, 1] = np.empty((0, 0), dtype=np.float64)

    normalized = _coordinate_cells(cells, "LocalizationsFinal")

    assert normalized[0].shape == (1, 3)
    assert normalized[0].dtype == np.float32
    assert normalized[1].shape == (0, 3)
    assert normalized[1].dtype == np.float64


def test_dataset_with_empty_true_localization_cells_keeps_image_alignment(
    tmp_path: Path,
) -> None:
    localizations = np.empty((1, 2), dtype=object)
    frames = np.empty((1, 2), dtype=object)
    truth = np.empty((1, 2), dtype=object)
    localizations[0, 0] = np.array([[1.0, 2.0], [3.0, 4.0]])
    localizations[0, 1] = np.array([[5.0, 6.0]])
    frames[0, 0] = np.array([[10, 11]], dtype=np.uint16)
    frames[0, 1] = np.array([[12]], dtype=np.uint16)
    truth[0, 0] = np.empty((0, 0))
    truth[0, 1] = np.empty((0, 0))
    path = tmp_path / "empty_truth.mat"
    savemat(
        path,
        {
            "LocalizationsFinal": localizations,
            "Frame_Information": frames,
            "TrueLocalizations": truth,
        },
    )

    dataset = load_dataset(path)

    assert dataset.n_images == 2
    assert [array.shape for array in dataset.true_localizations] == [(0, 2), (0, 2)]
    np.testing.assert_array_equal(dataset.frames[0], [10, 11])


def test_missing_true_localizations_matches_run_script_preinitialization(
    tmp_path: Path,
) -> None:
    localizations = np.empty((1, 2), dtype=object)
    frames = np.empty((1, 2), dtype=object)
    localizations[0, 0] = np.array([[1.0, 2.0], [3.0, 4.0]])
    localizations[0, 1] = np.array([[5.0, 6.0]])
    frames[0, 0] = np.array([[10, 11]], dtype=np.uint16)
    frames[0, 1] = np.array([[12]], dtype=np.uint16)
    path = tmp_path / "missing_truth.mat"
    savemat(
        path,
        {
            "LocalizationsFinal": localizations,
            "Frame_Information": frames,
        },
    )

    dataset = load_dataset(path)

    assert [array.shape for array in dataset.true_localizations] == [(0, 2), (0, 2)]
    assert all(array.dtype == np.float64 for array in dataset.true_localizations)


def test_photon_row_column_empty_cells_and_dtype_are_preserved(tmp_path: Path) -> None:
    locations = np.empty((1, 3), dtype=object)
    frames = np.empty((1, 3), dtype=object)
    photons = np.empty((1, 3), dtype=object)
    for index in range(3):
        locations[0, index] = np.array([[index, index + 1]], dtype=np.float64)
        frames[0, index] = np.array([[index + 1]], dtype=np.uint16)
    photons[0, 0] = np.array([[1, 2]], dtype=np.uint8)
    photons[0, 1] = np.array([[3], [4]], dtype=np.uint16)
    photons[0, 2] = np.empty((0, 0), dtype=np.int16)
    path = tmp_path / "photons.mat"
    savemat(path, {"LocalizationsFinal": locations, "Frame_Information": frames,
                   "Photons": photons})

    dataset = load_dataset(path)

    assert dataset.photons is not None
    np.testing.assert_array_equal(dataset.photons[0], [1, 2])
    np.testing.assert_array_equal(dataset.photons[1], [3, 4])
    assert [cell.dtype for cell in dataset.photons] == [np.uint8, np.uint16, np.int16]
    assert dataset.photons[2].shape == (0,)


def test_present_numeric_empty_truth_is_source_valid(tmp_path: Path) -> None:
    localizations = np.empty((1, 2), dtype=object)
    frames = np.empty((1, 2), dtype=object)
    localizations[0, 0] = np.array([[1.0, 2.0]])
    localizations[0, 1] = np.array([[3.0, 4.0]])
    frames[0, 0] = np.array([[1.0]])
    frames[0, 1] = np.array([[2.0]])
    path = tmp_path / "numeric_empty_truth.mat"
    savemat(path, {"LocalizationsFinal": localizations, "Frame_Information": frames,
                   "TrueLocalizations": np.empty((0, 0))})
    dataset = load_dataset(path)
    assert [cell.shape for cell in dataset.true_localizations] == [(0, 2), (0, 2)]


def test_nonempty_noncell_truth_remains_rejected(tmp_path: Path) -> None:
    localizations = np.empty((1, 1), dtype=object)
    frames = np.empty((1, 1), dtype=object)
    localizations[0, 0] = np.array([[1.0, 2.0]])
    frames[0, 0] = np.array([[1.0]])
    path = tmp_path / "bad_truth.mat"
    savemat(path, {"LocalizationsFinal": localizations, "Frame_Information": frames,
                   "TrueLocalizations": np.array([[1.0, 2.0]])})
    with pytest.raises(MatFileError, match="cell array or numeric empty"):
        load_dataset(path)


def test_unknown_mat_variables_are_preserved_as_passthrough(tmp_path: Path) -> None:
    localizations = np.empty((1, 1), dtype=object)
    frames = np.empty((1, 1), dtype=object)
    localizations[0, 0] = np.array([[1.0, 2.0]])
    frames[0, 0] = np.array([[3]], dtype=np.uint16)
    arbitrary = np.array([[1, 2], [3, 4]], dtype=np.float32)
    path = tmp_path / "passthrough.mat"
    savemat(path, {"LocalizationsFinal": localizations,
                   "Frame_Information": frames,
                   "ArbitraryUserVariable": arbitrary})

    dataset = load_dataset(path)

    assert set(dataset.mat_passthrough) == {"ArbitraryUserVariable"}
    np.testing.assert_array_equal(dataset.mat_passthrough["ArbitraryUserVariable"], arbitrary)
    assert dataset.mat_passthrough["ArbitraryUserVariable"].dtype == np.float32


def test_analyzed_v73_file_is_rejected_as_a_different_schema() -> None:
    path = EXAMPLES / "Analyzed_Time15-Nov-2020_Split_Example_3d_2darkstate_random_data.mat"

    with pytest.raises(MatFileError, match="v7.3/HDF5"):
        load_dataset(path)
