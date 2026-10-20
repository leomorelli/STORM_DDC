from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from ddc.trajectories import density_calc, determine_locs_w_multi2


def test_multi_frame_normal_2d_strict_distance_boundary_and_dtype() -> None:
    locations = np.array(
        [[0.0, 0.0], [3.9, 0.0], [10.0, 0.0], [14.0, 0.0]]
    )
    frames = np.array([1, 1, 2, 2], dtype=np.uint16)

    result = determine_locs_w_multi2(locations, frames, resolution=1.0)

    # Frame 1 has distance 3.9 < 4; frame 2 has distance exactly 4.
    np.testing.assert_array_equal(result, [1.0])
    assert result.shape == (1,)
    assert result.dtype == np.float64


def test_multi_frame_is_dimension_agnostic_and_preserves_first_occurrence_order() -> None:
    locations = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [10.0, 0.0, 1.0],
        ]
    )
    frames = np.array([5.0, 2.0, 5.0, 2.0])

    result = determine_locs_w_multi2(locations, frames, resolution=1.0)

    np.testing.assert_array_equal(result, [5.0, 2.0])


def test_multi_frame_empty_candidates_one_localization_and_nan_frame() -> None:
    cases = (
        (np.array([[0.0, 0.0]]), np.array([1.0])),
        (np.array([[0.0, 0.0], [0.1, 0.0]]), np.array([1.0, 2.0])),
        (np.array([[0.0, 0.0], [0.1, 0.0]]), np.array([np.nan, np.nan])),
    )
    for locations, frames in cases:
        result = determine_locs_w_multi2(locations, frames, 1.0)
        assert result.shape == (0,)
        assert result.dtype == np.float64


def test_multi_frame_repeated_inf_and_n_less_than_d_are_supported() -> None:
    result = determine_locs_w_multi2(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]]),
        np.array([np.inf, np.inf]),
        1.0,
    )

    np.testing.assert_array_equal(result, [np.inf])


def test_density_manual_float_fixture_boundary_a_plus_one_and_zero_point() -> None:
    locations = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
    frames = np.array([1.0, 3.0, 8.0])

    distance_dens, density = density_calc(locations, frames, 1.0, a=1)

    # Ordered pairs (1,2) and (2,1) both count: distance==2 and frame diff==A+1.
    # Point 3 has zero qualifying neighbors. Raw [1,1,0] is already normalized.
    assert distance_dens == 2.0
    np.testing.assert_array_equal(density, [[1.0], [1.0], [0.0]])
    assert density.shape == (3, 1)
    assert density.dtype == np.float64


def test_density_frame_difference_exactly_a_does_not_count_and_constant_is_nan() -> None:
    distance_dens, density = density_calc(
        np.array([[0.0, 0.0], [2.0, 0.0]]),
        np.array([1.0, 2.0]),
        1.0,
        a=1,
    )

    assert distance_dens == 2.0
    assert np.isnan(density).all()
    assert density.shape == (2, 1)


def test_density_one_2d_localization_reaches_matlab_length_row_failure() -> None:
    with pytest.raises(IndexError, match="indexes beyond LocalizationsFinal rows"):
        density_calc(
            np.array([[4.0, 7.0]]), np.array([3.0]), resolution=2.0, a=1
        )


def test_density_3d_distance_is_dimension_agnostic() -> None:
    _, density = density_calc(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [9.0, 0.0, 0.0]]),
        np.array([1.0, 3.0, 8.0]),
        resolution=1.0,
        a=1,
    )

    np.testing.assert_array_equal(density, [[1.0], [1.0], [0.0]])


def test_density_unsorted_floating_frames_are_not_reordered() -> None:
    locations = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
    _, density = density_calc(
        locations, np.array([8.0, 3.0, 1.0]), resolution=1.0, a=1
    )

    np.testing.assert_array_equal(density, [[1.0], [1.0], [0.0]])


def test_density_uint16_subtraction_reproduces_source_bug_004() -> None:
    locations = np.array([[0.0, 0.0], [0.0, 0.0]])

    _, integer_density = density_calc(
        locations, np.array([1, 3], dtype=np.uint16), 1.0, a=1
    )
    _, floating_density = density_calc(
        locations, np.array([1.0, 3.0]), 1.0, a=1
    )

    # uint16: 1-3 saturates to 0, but 3-1 is 2, so raw density is [0,1].
    np.testing.assert_array_equal(integer_density, [[0.0], [1.0]])
    # double: both ordered directions count, giving constant [1,1] -> NaN.
    assert np.isnan(floating_density).all()


def test_density_n_less_than_d_reproduces_matlab_row_index_failure() -> None:
    with pytest.raises(IndexError, match=r"length\(LocalizationsFinal\)=max\(n,d\)"):
        density_calc(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([1.0, 3.0]),
            1.0,
            a=1,
        )


def test_empty_zero_by_d_density_matches_matlab_empty_length() -> None:
    distance_dens, density = density_calc(
        np.empty((0, 3)), np.array([], dtype=np.uint16), 1.5, a=2
    )

    assert distance_dens == 3.0
    assert density.shape == (0, 1)
    assert density.dtype == np.float64


def test_stage_a5_functions_do_not_mutate_inputs() -> None:
    locations = np.array([[0.0, 0.0], [1.0, 0.0], [9.0, 0.0]])
    frames = np.array([1, 1, 4], dtype=np.uint16)
    locations_before = locations.copy()
    frames_before = frames.copy()

    determine_locs_w_multi2(locations, frames, 1.0)
    density_calc(locations, frames, 1.0, a=1)

    np.testing.assert_array_equal(locations, locations_before)
    np.testing.assert_array_equal(frames, frames_before)


def test_matlab_generated_trajectory_reference_when_available() -> None:
    reference_path = Path(__file__).parent / "matlab" / "trajectory_reference.mat"
    if not reference_path.exists():
        pytest.skip(
            "MATLAB runtime is unavailable; run "
            "tests/matlab/generate_trajectory_reference.m"
        )

    reference = loadmat(reference_path, squeeze_me=True, struct_as_record=False)
    frames_w_multi = determine_locs_w_multi2(
        reference["multi_locations"],
        reference["multi_frames"],
        float(reference["multi_resolution"]),
    )
    np.testing.assert_allclose(
        frames_w_multi, np.atleast_1d(reference["frames_w_multi_expected"])
    )

    distance_dens, density = density_calc(
        reference["density_locations"],
        reference["density_frames"],
        float(reference["density_resolution"]),
        float(reference["density_a"]),
    )
    np.testing.assert_allclose(distance_dens, reference["distance_dens_expected"])
    np.testing.assert_allclose(density, reference["density_expected"], equal_nan=True)
    assert str(reference["frames_w_multi_class"]) == "double"
    assert str(reference["density_class"]) == "double"
    _, constant_density = density_calc(
        reference["constant_locations"],
        reference["constant_frames"],
        float(reference["constant_resolution"]),
        float(reference["constant_a"]),
    )
    np.testing.assert_allclose(
        constant_density, reference["constant_density_expected"], equal_nan=True
    )
    assert not bool(reference["n_less_d_succeeded"])


def test_matlab_uint16_density_probe_when_available() -> None:
    probe_path = Path(__file__).parent / "matlab" / "uint16_density_probe.mat"
    if not probe_path.exists():
        pytest.skip(
            "MATLAB runtime is unavailable; run tests/matlab/probe_uint16_density.m"
        )

    probe = loadmat(probe_path, squeeze_me=True, struct_as_record=False)
    assert str(probe["frame_class"]) == "uint16"
    assert int(probe["forward_difference"]) == 0
    assert int(probe["reverse_difference"]) == 2
    _, density = density_calc(
        probe["LocalizationsFinal"],
        probe["Frame_Information"],
        float(probe["Resolution"]),
        float(probe["A"]),
    )
    np.testing.assert_allclose(density, probe["density_expected"], equal_nan=True)
