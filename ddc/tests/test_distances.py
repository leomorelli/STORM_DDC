from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from ddc.distances import make_distance_matrix_individual


def empty_bins(count: int, dtype: np.dtype = np.dtype(np.float64)) -> list[np.ndarray]:
    return [np.array([], dtype=dtype) for _ in range(count)]


def test_2d_multiple_differences_repeated_frames_and_empty_extra_bin() -> None:
    locations = np.array(
        [[0.0, 0.0], [3.0, 4.0], [0.0, 4.0], [0.0, 0.0]]
    )
    frames = np.array([1, 2, 4, 4])

    result = make_distance_matrix_individual(locations, empty_bins(4), frames, 3)

    np.testing.assert_allclose(result[0], [5.0])  # MATLAB DistanceMatrix{1}
    np.testing.assert_allclose(result[1], [3.0, 5.0])  # DistanceMatrix{2}
    np.testing.assert_allclose(result[2], [4.0, 0.0])  # DistanceMatrix{3}
    np.testing.assert_allclose(result[3], [])  # caller's extra {B+1} cell


def test_3d_unsorted_frames_are_sorted_with_corresponding_coordinates() -> None:
    locations = np.array(
        [
            [0.0, 0.0, 12.0],  # frame 5
            [0.0, 0.0, 0.0],  # frame 1
            [0.0, 4.0, 0.0],  # frame 3
        ]
    )
    frames = np.array([[5, 1, 3]])

    result = make_distance_matrix_individual(locations, empty_bins(4), frames, 4)

    np.testing.assert_allclose(result[0], [])
    np.testing.assert_allclose(result[1], [4.0, np.sqrt(160.0)])
    np.testing.assert_allclose(result[2], [])
    np.testing.assert_allclose(result[3], [12.0])


def test_boundary_difference_b_is_included_and_b_plus_one_is_excluded() -> None:
    locations = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 12.0]])
    frames = np.array([10, 13, 14])

    result = make_distance_matrix_individual(locations, empty_bins(4), frames, 3)

    np.testing.assert_allclose(result[0], [np.sqrt(73.0)])
    np.testing.assert_allclose(result[1], [])
    np.testing.assert_allclose(result[2], [5.0])
    np.testing.assert_allclose(result[3], [])


def test_single_localization_leaves_all_bins_empty() -> None:
    result = make_distance_matrix_individual(
        np.array([[1.0, 2.0, 3.0]]), empty_bins(3), np.array([7]), 2
    )

    for cell in result:
        np.testing.assert_allclose(cell, [])


def test_existing_bin_values_are_preserved_and_inputs_are_not_mutated() -> None:
    initial = [np.array([99.0]), np.array([], dtype=np.float64)]
    result = make_distance_matrix_individual(
        np.array([[0.0, 0.0], [0.0, 2.0]]), initial, np.array([1, 2]), 1
    )

    np.testing.assert_allclose(result[0], [99.0, 2.0])
    np.testing.assert_allclose(initial[0], [99.0])


@pytest.mark.parametrize("frame_dtype", [np.int64, np.float64])
@pytest.mark.parametrize("coordinate_dtype", [np.float32, np.float64])
def test_relevant_numeric_dtypes_produce_equivalent_distances(
    frame_dtype: type[np.generic], coordinate_dtype: type[np.generic]
) -> None:
    locations = np.array([[0, 0], [6, 8]], dtype=coordinate_dtype)
    frames = np.array([2, 4], dtype=frame_dtype)

    result = make_distance_matrix_individual(locations, empty_bins(2), frames, 2)

    np.testing.assert_allclose(result[0], [])
    np.testing.assert_allclose(result[1], [10.0])


def test_unsigned_frames_reproduce_matlab_saturating_subtraction_failure() -> None:
    with pytest.raises(IndexError, match=r"DistanceMatrix\{0\}"):
        make_distance_matrix_individual(
            np.array([[0.0, 0.0], [6.0, 8.0]]),
            empty_bins(2),
            np.array([2, 4], dtype=np.uint16),
            2,
        )


def test_small_signed_integer_coordinates_retain_integer_arithmetic() -> None:
    """Integer coordinates are outside the real float64 repository contract."""

    locations = np.array([[0, 0], [3, 4]], dtype=np.int64)
    result = make_distance_matrix_individual(
        locations, empty_bins(1), np.array([1, 2], dtype=np.int64), 1
    )

    np.testing.assert_allclose(result[0], [5.0])


def test_non_integral_accepted_frame_difference_matches_invalid_matlab_index() -> None:
    with pytest.raises(IndexError, match="valid MATLAB cell index"):
        make_distance_matrix_individual(
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            empty_bins(2),
            np.array([1.0, 1.5]),
            1,
        )


def test_matlab_generated_reference_when_available() -> None:
    reference_path = Path(__file__).parent / "matlab" / "distance_reference.mat"
    if not reference_path.exists():
        pytest.skip(
            "MATLAB Engine/runtime is unavailable; run "
            "tests/matlab/generate_distance_references.m with MATLAB to create oracle data"
        )

    reference = loadmat(reference_path, squeeze_me=False, struct_as_record=False)
    fixture_names = ("two_d", "three_d", "repeated", "single", "boundary")
    for name in fixture_names:
        locations = reference[f"{name}_locations"]
        frames = reference[f"{name}_frames"]
        b = reference[f"{name}_B"].item()
        matlab_cells = reference[f"{name}_expected"].ravel(order="F")
        python_cells = make_distance_matrix_individual(
            locations, empty_bins(len(matlab_cells)), frames, b
        )
        assert len(python_cells) == len(matlab_cells)
        for python_cell, matlab_cell in zip(python_cells, matlab_cells, strict=True):
            np.testing.assert_allclose(
                python_cell, np.asarray(matlab_cell).reshape(-1, order="F")
            )

    unsigned_succeeded = bool(reference["unsigned_succeeded"].item())
    unsigned_locations = reference["unsigned_locations"]
    unsigned_frames = reference["unsigned_frames"]
    unsigned_b = reference["unsigned_B"].item()
    if unsigned_succeeded:
        matlab_cells = reference["unsigned_expected"].ravel(order="F")
        python_cells = make_distance_matrix_individual(
            unsigned_locations,
            empty_bins(len(matlab_cells)),
            unsigned_frames,
            unsigned_b,
        )
        for python_cell, matlab_cell in zip(python_cells, matlab_cells, strict=True):
            np.testing.assert_allclose(
                python_cell, np.asarray(matlab_cell).reshape(-1, order="F")
            )
    else:
        with pytest.raises(IndexError, match=r"DistanceMatrix\{0\}"):
            make_distance_matrix_individual(
                unsigned_locations,
                empty_bins(unsigned_b),
                unsigned_frames,
                unsigned_b,
            )
