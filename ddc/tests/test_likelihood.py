from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat
from scipy.spatial.distance import pdist

from ddc.likelihood import calculate_score_8, matlab_intersect_first


@pytest.mark.parametrize(
    ("left", "right", "expected_c", "expected_ia", "expected_ib"),
    [
        ([3.0, 1.0, 2.0], [2.0, 3.0], [2.0, 3.0], [2, 0], [0, 1]),
        ([2.0, 2.0, 1.0], [2.0], [2.0], [0], [0]),
        ([2.0], [2.0, 2.0], [2.0], [0], [0]),
        ([2.0, 2.0], [2.0, 2.0], [2.0], [0], [0]),
        ([9.0, 1.0], [1.0, 9.0], [1.0, 9.0], [1, 0], [0, 1]),
        ([1.0], [2.0], [], [], []),
        ([2.0, 1.0], [1.0, 2.0], [1.0, 2.0], [1, 0], [0, 1]),
        ([1.0, np.nan], [np.nan, 1.0], [1.0], [0], [1]),
    ],
)
def test_matlab_intersect_sorted_unique_first_occurrences(
    left,
    right,
    expected_c,
    expected_ia,
    expected_ib,
) -> None:
    c, ia, ib = matlab_intersect_first(left, right)

    np.testing.assert_array_equal(c, expected_c)
    np.testing.assert_array_equal(ia, expected_ia)
    np.testing.assert_array_equal(ib, expected_ib)
    assert ia.dtype == ib.dtype == np.int64


def test_matlab_intersect_exact_not_approximate_float_equality() -> None:
    exact = np.float64(1.0)
    perturbed = np.nextafter(exact, np.float64(2.0))

    c_exact, _, _ = matlab_intersect_first([exact], [exact])
    c_near, _, _ = matlab_intersect_first([exact], [perturbed])

    np.testing.assert_array_equal(c_exact, [1.0])
    assert c_near.size == 0


def test_matlab_intersect_signed_zero_uses_first_left_representation() -> None:
    c, ia, ib = matlab_intersect_first([-0.0, 0.0], [0.0])

    assert c.size == 1 and np.signbit(c[0])
    np.testing.assert_array_equal(ia, [0])
    np.testing.assert_array_equal(ib, [0])


def _probabilities(rows: int = 4, columns: int = 4) -> np.ndarray:
    return np.vstack(
        [np.arange(columns, dtype=float) + 10 * (row + 1) for row in range(rows)]
    )


def test_all_retained_exact_intersection_and_pair_alignment_2d() -> None:
    locations = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    frames = np.array([1.0, 2.0, 4.0])
    overall = pdist(locations)
    overall_frames = np.array([1.0, 3.0, 2.0])

    result = calculate_score_8(
        locations, frames, 4, overall, overall_frames, 1.0, _probabilities()
    )

    np.testing.assert_array_equal(result.z22, [1.0, 3.0, 2.0])
    np.testing.assert_array_equal(result.d, [1.0, 3.0, 2.0])
    np.testing.assert_array_equal(result.c, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result.ia, [0, 2, 1])
    assert result.d_temp.size == result.z2_temp.size == 0
    assert result.lik == 41.0 + 43.0 + 42.0
    assert result.lik2 == 0.0


def test_subset_retained_uses_last_row_then_remaining_frame_rows() -> None:
    result = calculate_score_8(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([1.0, 2.0]),
        4,
        np.array([1.0, 3.0, 2.0]),
        np.array([1.0, 3.0, 2.0]),
        1.0,
        _probabilities(),
    )

    assert result.lik == 41.0
    assert result.lik2 == 33.0 + 22.0
    np.testing.assert_array_equal(result.d_temp, [3.0, 2.0])
    np.testing.assert_array_equal(result.z2_temp, [3.0, 2.0])


def test_one_retained_localization_and_empty_pair_vectors_sum_to_zero() -> None:
    result = calculate_score_8(
        np.array([[0.0, 0.0]]),
        np.array([1.0]),
        2,
        np.array([]),
        np.array([]),
        1.0,
        _probabilities(2, 2),
    )

    assert result.lik == result.lik2 == 0.0
    assert result.d.size == result.z22.size == 0


def test_repeated_frames_remove_zero_frame_pairs_from_retained_likelihood() -> None:
    result = calculate_score_8(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([2.0, 2.0]),
        3,
        np.array([]),
        np.array([]),
        1.0,
        _probabilities(3, 2),
    )

    assert result.d.size == 0
    np.testing.assert_array_equal(result.z22, [0.0])
    assert result.lik == result.lik2 == 0.0


@pytest.mark.parametrize(
    ("distance", "expected_column"),
    [
        (0.0, 0),
        (1.999999999, 0),
        (2.0, 1),
        (2.000000001, 1),
        (6.0, 3),
        (100.0, 3),
        (np.inf, 3),
    ],
)
def test_distance_binning_boundaries_last_bin_and_clamp(
    distance: float, expected_column: int
) -> None:
    probabilities = _probabilities(2, 4)
    result = calculate_score_8(
        np.array([[0.0, 0.0]]),
        np.array([1.0]),
        2,
        np.array([distance]),
        np.array([1.0]),
        2.0,
        probabilities,
    )

    assert result.lik == 0.0
    assert result.lik2 == probabilities[0, expected_column]


def test_negative_and_nan_distance_bins_reproduce_invalid_matlab_subscripts() -> None:
    for value in (-0.1, np.nan):
        with pytest.raises(IndexError, match="not a valid subscript"):
            calculate_score_8(
                [[0.0, 0.0]],
                [1.0],
                2,
                [value],
                [1.0],
                1.0,
                _probabilities(2, 2),
            )


def test_zero_nan_and_inf_probability_values_are_summed_directly() -> None:
    for probability in (0.0, np.nan, np.inf):
        matrix = np.array([[1.0], [probability]])
        result = calculate_score_8(
            [[0.0, 0.0], [0.0, 0.0]],
            [1.0, 2.0],
            2,
            [0.0],
            [1.0],
            1.0,
            matrix,
        )
        if np.isnan(probability):
            assert np.isnan(result.lik)
        else:
            assert result.lik == probability


def test_duplicate_intersection_deletes_only_first_overall_occurrence() -> None:
    result = calculate_score_8(
        [[0.0, 0.0], [1.0, 0.0]],
        [1.0, 2.0],
        3,
        [1.0, 1.0],
        [1.0, 2.0],
        1.0,
        _probabilities(3, 3),
    )

    np.testing.assert_array_equal(result.ia, [0])
    np.testing.assert_array_equal(result.d_temp, [1.0])
    np.testing.assert_array_equal(result.z2_temp, [2.0])


def test_near_distance_is_not_deleted_by_intersection() -> None:
    exact = np.float64(1.0)
    near = np.nextafter(exact, np.float64(2.0))
    result = calculate_score_8(
        [[0.0, 0.0], [1.0, 0.0]],
        [1.0, 2.0],
        3,
        [near],
        [1.0],
        1.0,
        _probabilities(3, 3),
    )

    assert result.c.size == 0
    assert result.d_temp.size == 1


def test_3d_coordinates_and_float32_float64_frames_match() -> None:
    locations = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    results = [
        calculate_score_8(
            locations,
            np.array([1, 2], dtype=dtype),
            3,
            [2.0],
            [1.0],
            1.0,
            _probabilities(3, 3),
        )
        for dtype in (np.float32, np.float64)
    ]

    assert results[0].lik == results[1].lik
    np.testing.assert_array_equal(results[0].d, results[1].d)


def test_uint16_frames_preserve_pdist_source_failure() -> None:
    with pytest.raises(TypeError, match="DDC-SOURCE-BUG-002"):
        calculate_score_8(
            [[0.0, 0.0], [1.0, 0.0]],
            np.array([1, 2], dtype=np.uint16),
            3,
            [1.0],
            [1.0],
            1.0,
            _probabilities(3, 2),
        )


def test_likelihood_inputs_are_not_mutated() -> None:
    loc = np.array([[0.0, 0.0], [1.0, 0.0]])
    fram = np.array([1.0, 2.0])
    overall = np.array([1.0])
    z2 = np.array([1.0])
    probabilities = _probabilities(3, 2)
    copies = [value.copy() for value in (loc, fram, overall, z2, probabilities)]

    calculate_score_8(loc, fram, 3, overall, z2, 1.0, probabilities)

    for actual, expected in zip(
        (loc, fram, overall, z2, probabilities), copies, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_matlab_generated_likelihood_reference_when_available() -> None:
    path = Path(__file__).parent / "matlab" / "likelihood_reference.mat"
    if not path.exists():
        pytest.skip(
            "MATLAB runtime is unavailable; run "
            "tests/matlab/generate_likelihood_reference.m"
        )
    reference = loadmat(path, squeeze_me=True, struct_as_record=False)
    result = calculate_score_8(
        reference["loc_input"],
        reference["fram_input"],
        int(reference["A_input"]),
        reference["D_Overall_input"],
        reference["Z2_input"],
        float(reference["Resolution_input"]),
        reference["Prob_Distributions_input"],
    )
    for actual, name in (
        (result.d, "D_expected"),
        (result.z22, "Z22_expected"),
        (result.c, "C_expected"),
        (result.ia + 1, "ia_expected"),
        (result.ib + 1, "ib_expected"),
        (result.d_temp, "D_temp_expected"),
        (result.z2_temp, "Z2_temp_expected"),
    ):
        np.testing.assert_allclose(actual, np.atleast_1d(reference[name]))
    np.testing.assert_allclose(result.lik, reference["lik_expected"])
    np.testing.assert_allclose(result.lik2, reference["lik2_expected"])

