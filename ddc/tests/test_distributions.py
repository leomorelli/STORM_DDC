from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from ddc.distributions import (
    _fit_mixture_parameter,
    _frame_pair_distances,
    _probability_histogram,
    analytical_mixture_parameter,
    determine_blinking_distribution,
)


def deterministic_fixture() -> tuple[list[np.ndarray], list[np.ndarray]]:
    index = np.arange(12)
    locations = np.column_stack((index * 2.0, (index % 3) * 1.0))
    frames = index % 6 + 1
    return [locations], [frames]


def test_frame_pdist_pair_sequence_for_double_frames() -> None:
    frames = np.array([4, 1, 3, 1], dtype=np.float64)
    expected = np.array([3.0, 1.0, 3.0, 2.0, 0.0, 2.0])

    result = _frame_pair_distances(frames)

    np.testing.assert_allclose(result, expected)


def test_uint16_frames_reproduce_documented_matlab_pdist_source_failure() -> None:
    with pytest.raises(TypeError, match="DDC-SOURCE-BUG-002"):
        _frame_pair_distances(np.array([1, 3, 7], dtype=np.uint16))


def test_matlab_uint16_pdist_probe_when_available() -> None:
    probe_path = Path(__file__).parent / "matlab" / "uint16_frame_pdist_probe.mat"
    if not probe_path.exists():
        pytest.skip(
            "MATLAB runtime is unavailable; run "
            "tests/matlab/probe_uint16_frame_pdist.m to record the oracle"
        )

    probe = loadmat(probe_path, squeeze_me=True, struct_as_record=False)
    assert str(probe["X_class"]) == "uint16"
    assert not bool(probe["pdist_succeeded"])
    with pytest.raises(TypeError, match="DDC-SOURCE-BUG-002"):
        _frame_pair_distances(np.asarray(probe["frame"], dtype=np.uint16))


def test_probability_histogram_normalization_inf_edge_and_boundaries() -> None:
    bins = np.array([0.0, 1.0, 2.0, np.inf])
    values = np.array([0.0, 0.999, 1.0, 1.999, 2.0, 100.0])

    result = _probability_histogram(values, bins)

    np.testing.assert_allclose(result, [2 / 6, 2 / 6, 2 / 6])
    np.testing.assert_allclose(result.sum(), 1.0)


def test_empty_probability_histogram_matches_zero_over_zero_semantics() -> None:
    result = _probability_histogram(
        np.array([], dtype=np.float64), np.array([0.0, 1.0, np.inf])
    )

    assert np.isnan(result).all()


@pytest.mark.parametrize("mixture", [-0.25, 0.0, 0.3, 1.0, 1.4])
def test_scipy_fit_matches_analytical_unbounded_linear_solution(mixture: float) -> None:
    true = np.array([0.1, 0.2, 0.7])
    blink = np.array([0.7, 0.2, 0.1])
    observed = mixture * true + (1.0 - mixture) * blink

    fitted = _fit_mixture_parameter(true, blink, observed)
    analytical = analytical_mixture_parameter(true, blink, observed)

    np.testing.assert_allclose(fitted, mixture, rtol=1e-7, atol=5e-9)
    np.testing.assert_allclose(fitted, analytical, rtol=1e-7, atol=5e-9)


def test_degenerate_mixture_is_unidentifiable_and_solver_retains_start() -> None:
    true = np.array([0.2, 0.3, 0.5])
    blink = true.copy()
    observed = np.array([0.1, 0.4, 0.5])

    fitted = _fit_mixture_parameter(true, blink, observed)
    analytical = analytical_mixture_parameter(true, blink, observed)

    assert np.isnan(analytical)
    np.testing.assert_allclose(fitted, 1.0)


def test_full_deterministic_single_image_shapes_and_values() -> None:
    locations, frames = deterministic_fixture()
    double_result = determine_blinking_distribution(
        locations, [frames[0].astype(np.float64)], pre_a=2, resolution=1.0
    )

    bins, blink, total_no_blink, resolution, x_overall, m_mat = double_result
    assert bins.shape == (24,)
    assert bins[-1] == np.inf
    np.testing.assert_allclose(bins[:-1], np.arange(23, dtype=np.float64))
    assert blink.shape == (23,)
    assert total_no_blink.shape == (1, 23)
    assert resolution == 1.0
    assert x_overall.shape == (1, 2)
    assert m_mat.shape == (2, 23)
    np.testing.assert_allclose(np.sum(blink), 1.0)



def test_full_function_rejects_repository_uint16_frame_dtype() -> None:
    locations, frames = deterministic_fixture()

    with pytest.raises(TypeError, match="DDC-SOURCE-BUG-002"):
        determine_blinking_distribution(
            locations, [frames[0].astype(np.uint16)], pre_a=2, resolution=1.0
        )


def test_multiple_images_mean_x_overall_and_m_matrix() -> None:
    locations, frames = deterministic_fixture()
    result = determine_blinking_distribution(
        [locations[0], locations[0] + 0.25],
        [frames[0].astype(float), frames[0].astype(float)],
        pre_a=2,
        resolution=1.0,
    )

    assert result[2].shape == (2, 23)
    assert result[4].shape == (2,)
    assert result[5].shape == (2, 23)


def test_matlab_generated_full_function_reference_when_available() -> None:
    reference_path = Path(__file__).parent / "matlab" / "distribution_reference.mat"
    if not reference_path.exists():
        pytest.skip(
            "MATLAB Engine/runtime is unavailable; run "
            "tests/matlab/generate_distribution_reference.m to create oracle data"
        )

    reference = loadmat(reference_path, squeeze_me=False, struct_as_record=False)
    locations = [cell for cell in reference["LocalizationsFinal"].ravel(order="F")]
    frames = [cell for cell in reference["Frame_Information"].ravel(order="F")]
    result = determine_blinking_distribution(
        locations,
        frames,
        int(reference["Pre_A"].item()),
        float(reference["Resolution_input"].item()),
    )
    expected_names = (
        "bins_expected",
        "blink_expected",
        "total_no_blink_expected",
        "resolution_expected",
        "x_overall_expected",
        "m_mat_expected",
    )
    for actual, name in zip(result, expected_names, strict=True):
        expected = reference[name]
        if np.isscalar(actual):
            np.testing.assert_allclose(actual, expected.item(), equal_nan=True)
        else:
            np.testing.assert_allclose(
                np.asarray(actual), np.asarray(expected).squeeze(), equal_nan=True
            )
