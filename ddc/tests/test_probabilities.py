from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.io import loadmat

import ddc.probabilities as probabilities_module
from ddc.probabilities import (
    DDCSourceBug003,
    determine_deviation_in_probability8,
    eliminate_blinking_de_loc15_mcmc,
)


def test_binary_frame_search_matches_original_scan_with_repeated_frames():
    rng = np.random.default_rng(1082026)
    for _ in range(100):
        n = int(rng.integers(3, 30))
        locations = rng.normal(size=(n, 3))
        frames = np.sort(rng.integers(-3, 12, size=n)).astype(float)
        deviation = rng.random((4, 10))
        labels = rng.permutation(n) + 1
        with patch.object(probabilities_module, "_can_search_frame_candidates", return_value=False):
            expected = eliminate_blinking_de_loc15_mcmc(locations, frames, 1, 4, deviation, labels)
        actual = eliminate_blinking_de_loc15_mcmc(locations, frames, 1, 4, deviation, labels)
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right, strict=True)


def test_eliminator_uses_one_based_boundary_bin_and_first_frame_row() -> None:
    locations = np.array([[0.0, 0.0], [1.0, 0.0]])
    frames = np.array([1.0, 2.0])
    deviation = np.array([[0.2, 0.8], [0.99, 0.99]])

    retained, probabilities = eliminate_blinking_de_loc15_mcmc(
        locations, frames, 1.0, 2, deviation, np.array([2, 1])
    )

    # Distance exactly 1 maps to MATLAB DDt=floor(1/1)+1=2.
    np.testing.assert_array_equal(retained, [1.0, 0.0])
    np.testing.assert_allclose(probabilities, [0.0, 0.8])


def test_last_frame_difference_row_is_excluded_by_strict_max_comparison() -> None:
    retained, probabilities = eliminate_blinking_de_loc15_mcmc(
        np.array([[0.0, 0.0], [0.1, 0.0]]),
        np.array([1.0, 3.0]),
        1.0,
        2,
        np.ones((2, 1)),
        np.array([1, 2]),
    )

    np.testing.assert_array_equal(retained, [1.0, 1.0])
    np.testing.assert_array_equal(probabilities, [0.0, 0.0])


@pytest.mark.parametrize(
    ("probability", "expected_mask", "expected_probability"),
    [
        (0.0, [1.0, 1.0], 0.0),
        (0.5, [1.0, 1.0], 0.5),
        (np.nan, [1.0, 1.0], 0.0),
        (np.inf, [1.0, 0.0], np.inf),
    ],
)
def test_zero_threshold_nan_and_inf_comparisons(
    probability: float,
    expected_mask: list[float],
    expected_probability: float,
) -> None:
    retained, probabilities = eliminate_blinking_de_loc15_mcmc(
        np.array([[0.0, 0.0], [0.25, 0.0]]),
        np.array([1.0, 2.0]),
        1.0,
        2,
        np.array([[probability], [0.0]]),
        np.array([1, 2]),
    )

    np.testing.assert_array_equal(retained, expected_mask)
    np.testing.assert_allclose(
        probabilities[1], expected_probability, equal_nan=True
    )


def test_empty_candidates_and_repeated_frames_leave_defaults() -> None:
    for frames in (np.array([1.0, 10.0]), np.array([4.0, 4.0])):
        retained, probabilities = eliminate_blinking_de_loc15_mcmc(
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            frames,
            1.0,
            3,
            np.ones((3, 2)),
            np.array([2, 1]),
        )
        np.testing.assert_array_equal(retained, [1.0, 1.0])
        np.testing.assert_array_equal(probabilities, [0.0, 0.0])


def test_distance_bin_is_capped_at_last_column_for_3d_coordinates() -> None:
    retained, probabilities = eliminate_blinking_de_loc15_mcmc(
        np.array([[0.0, 0.0, 0.0], [20.0, 20.0, 20.0]]),
        np.array([1.0, 2.0], dtype=np.float32),
        1.0,
        2,
        np.array([[0.1, 0.9], [0.0, 0.0]]),
        np.array([1, 2]),
    )

    np.testing.assert_array_equal(retained, [1.0, 0.0, 1.0])
    np.testing.assert_allclose(probabilities, [0.0, 0.9, 0.0])


def test_uint16_eliminator_reaches_source_bug_003_only_at_candidate_expression() -> None:
    with pytest.raises(DDCSourceBug003, match="DDC-SOURCE-BUG-003"):
        eliminate_blinking_de_loc15_mcmc(
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            np.array([1, 2], dtype=np.uint16),
            1.0,
            2,
            np.ones((2, 2)),
            np.array([1, 2]),
        )

    retained, probabilities = eliminate_blinking_de_loc15_mcmc(
        np.array([[0.0, 0.0]]),
        np.array([1], dtype=np.uint16),
        1.0,
        2,
        np.ones((2, 1)),
        np.array([1]),
    )
    # MATLAB length(1-by-2 LocalizationsFinal) is 2, not one row.
    np.testing.assert_array_equal(retained, [1.0, 1.0])
    np.testing.assert_array_equal(probabilities, [0.0, 0.0])


def test_n_less_than_d_sizes_work_and_output_arrays_by_matlab_length() -> None:
    retained, probabilities = eliminate_blinking_de_loc15_mcmc(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([1.0, 2.0]),
        1.0,
        2,
        np.array([[0.8, 0.8], [0.0, 0.0]]),
        np.array([1, 2, 3]),
    )

    np.testing.assert_array_equal(retained, [1.0, 0.0, 1.0])
    np.testing.assert_allclose(probabilities, [0.0, 0.8, 0.0])


def test_n2_d3_matlab_length_changes_nb_and_second_fit_input() -> None:
    observed_fits: list[np.ndarray] = []
    original_fit = probabilities_module._fit_mixture_parameter

    def record_fit(
        true_distribution: np.ndarray,
        blink_distribution: np.ndarray,
        observed_distribution: np.ndarray,
    ) -> float:
        del true_distribution, blink_distribution
        observed_fits.append(np.asarray(observed_distribution).copy())
        return 0.5

    probabilities_module._fit_mixture_parameter = record_fit
    try:
        probabilities_module.determine_deviation_in_probability8(
            true_distribution=[0.0, 1.0],
            distribution_for_blink=[1.0, 0.0],
            bins=[0.0, 1.0, np.inf],
            a=2,
            localizations_final=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
            ),
            frame_information=np.array([1.0, 2.0]),
            loc=[],
            fram=[],
            resolution=1.0,
            true_distribution2=[1.0, -1.0],
            distribution_for_blink2=[0.0, 1.0],
            bins2=[0.0, 1.5, np.inf],
            x_overall=[0.5, 1.0],
            m_mat=[[0.8, 0.8], [0.0, 0.0]],
            rng=np.random.default_rng(2),
        )
    finally:
        probabilities_module._fit_mixture_parameter = original_fit

    # length=3 and retained=[1,0,1] give nb=(3-2)/2=.5. Therefore the
    # normalized row passed to the first second-stage fit is [7/9,2/9].
    np.testing.assert_allclose(observed_fits[0], [7 / 9, 2 / 9])
    # Replacing MATLAB length with n_rows would give nb=1 and [7/8,1/8].
    assert not np.allclose(observed_fits[0], [7 / 8, 1 / 8])


def test_matlab_uint16_eliminator_probe_when_available() -> None:
    probe_path = (
        Path(__file__).parent / "matlab" / "uint16_eliminator_mtimes_probe.mat"
    )
    if not probe_path.exists():
        pytest.skip(
            "MATLAB runtime is unavailable; run "
            "tests/matlab/probe_uint16_eliminator_mtimes.m"
        )

    probe = loadmat(probe_path, squeeze_me=True, struct_as_record=False)
    assert str(probe["AAAA_class"]) == "uint16"
    assert str(probe["BBBB_class"]) == "uint16"
    assert not bool(probe["mtimes_succeeded"])
    assert not bool(probe["function_succeeded"])


def _full_fixture(dimension: int = 2) -> tuple[np.ndarray, np.ndarray]:
    locations = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    if dimension == 3:
        locations = np.column_stack((locations, np.zeros(3)))
    return locations, np.array([1.0, 2.0, 3.0])


def _run_full(dimension: int = 2, bins: np.ndarray | None = None):
    locations, frames = _full_fixture(dimension)
    return determine_deviation_in_probability8(
        true_distribution=np.array([0.0, 1.0]),
        distribution_for_blink=np.array([1.0, 0.0]),
        bins=np.array([-100.0, 100.0]) if bins is None else bins,
        a=2,
        localizations_final=locations,
        frame_information=frames,
        loc=np.array([]),
        fram=np.array([]),
        resolution=1.0,
        true_distribution2=np.array([0.0, 1.0]),
        distribution_for_blink2=np.array([1.0, 0.0]),
        bins2=np.array([0.0, 1.5, np.inf]),
        x_overall=np.array([0.0, 1.0]),
        m_mat=np.array([[0.8, 0.8], [0.0, 0.0]]),
        rng=np.random.default_rng(4),
    )


def test_full_probability_component_finite_values_shapes_and_m_mat_identity() -> None:
    deviation, probability_distributions, dscale = _run_full()

    np.testing.assert_array_equal(deviation, [[0.8, 0.8], [0.0, 0.0]])
    np.testing.assert_allclose(dscale, [0.5, 1.0], atol=1e-8)
    np.testing.assert_allclose(
        probability_distributions, [[0.55, 0.55], [0.05, 1.05]], atol=1e-8
    )
    assert deviation.shape == probability_distributions.shape == (2, 2)
    assert dscale.shape == (2,)


def test_2d_and_3d_zero_z_coordinates_are_equivalent_and_bins_is_inactive() -> None:
    result_2d = _run_full(2, bins=np.array([0.0, 1.0, np.inf]))
    result_3d = _run_full(3, bins=np.array([999.0]))

    for actual, expected in zip(result_3d, result_2d, strict=True):
        np.testing.assert_allclose(actual, expected)


def test_outer_probability_component_preserves_uint16_pdist_source_failure() -> None:
    locations, frames = _full_fixture()
    with pytest.raises(TypeError, match="DDC-SOURCE-BUG-002"):
        determine_deviation_in_probability8(
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0, np.inf],
            2,
            locations,
            frames.astype(np.uint16),
            [],
            [],
            1.0,
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.5, np.inf],
            [0.0, 1.0],
            [[0.8, 0.8], [0.0, 0.0]],
        )


def test_zero_probabilities_propagate_nan_to_source_fit_failure() -> None:
    locations, frames = _full_fixture()
    with pytest.raises(ValueError, match="Residuals are not finite"):
        determine_deviation_in_probability8(
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0, np.inf],
            2,
            locations,
            frames,
            [],
            [],
            1.0,
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.5, np.inf],
            [0.0, 1.0],
            [[0.0, 0.0], [0.0, 0.0]],
            rng=np.random.default_rng(1),
        )


def test_matlab_generated_probability_reference_when_available() -> None:
    reference_path = Path(__file__).parent / "matlab" / "probability_reference.mat"
    if not reference_path.exists():
        pytest.skip(
            "MATLAB runtime is unavailable; run "
            "tests/matlab/generate_probability_reference.m"
        )

    reference = loadmat(reference_path, squeeze_me=False, struct_as_record=False)
    retained, probabilities = eliminate_blinking_de_loc15_mcmc(
        reference["locations"],
        reference["frames"],
        float(reference["resolution"].item()),
        int(reference["A"].item()),
        reference["deviation"],
        reference["traj_num"],
    )
    np.testing.assert_allclose(retained, reference["loc_expected"].squeeze())
    np.testing.assert_allclose(probabilities, reference["probs_expected"].squeeze())

    result = determine_deviation_in_probability8(
        reference["true_distribution"],
        reference["blink_distribution"],
        reference["bins"],
        int(reference["A"].item()),
        reference["locations"],
        reference["frames"],
        [],
        [],
        float(reference["resolution"].item()),
        reference["true_distribution2"],
        reference["blink_distribution2"],
        reference["bins2"],
        reference["x_overall"],
        reference["m_mat"],
        rng=np.random.default_rng(1),
    )
    for actual, name in zip(
        result,
        ("deviation_expected", "probability_expected", "dscale_expected"),
        strict=True,
    ):
        np.testing.assert_allclose(actual, reference[name].squeeze())
