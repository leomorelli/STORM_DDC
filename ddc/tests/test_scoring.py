from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from ddc.scoring import palm_density_image, palm_image_squared_error


def test_perfect_inferred_truth_image_score_is_zero() -> None:
    truth = np.array([[0.0, 0.0], [100.0, 0.0]])
    assert palm_image_squared_error(truth, truth) == 0.0


def test_no_spatial_overlap_and_empty_inferred_have_positive_score() -> None:
    truth = np.array([[0.0, 0.0], [100.0, 0.0]])
    far = np.array([[1000.0, 1000.0]])

    assert palm_image_squared_error(far, truth) > 0
    assert palm_image_squared_error(np.empty((0, 2)), truth) > 0


def test_partial_set_and_duplicate_points_change_image_score() -> None:
    truth = np.array([[0.0, 0.0], [100.0, 0.0]])
    partial = truth[:1]
    duplicated = np.vstack((truth, truth[0]))

    assert palm_image_squared_error(partial, truth) > 0
    assert palm_image_squared_error(duplicated, truth) > 0


def test_empty_truth_is_unsupported_by_source_bounds_calculation() -> None:
    with pytest.raises(ValueError, match="empty coords2"):
        palm_density_image(np.empty((0, 2)), np.empty((0, 2)))


def test_truth_boundary_points_are_included_in_image() -> None:
    truth = np.array([[0.0, 0.0], [100.0, 100.0]])
    image = palm_density_image(truth, truth)

    assert image.shape == (50, 50)
    assert np.count_nonzero(image) > 0
    assert np.isfinite(image).all()


def test_third_column_is_sigma_not_a_spatial_z_coordinate() -> None:
    xy = np.array([[0.0, 0.0], [100.0, 0.0]])
    narrow = np.column_stack((xy, [10.0, 10.0]))
    wide = np.column_stack((xy, [80.0, 80.0]))

    narrow_image = palm_density_image(narrow, xy)
    wide_image = palm_density_image(wide, xy)

    assert narrow_image.shape == wide_image.shape
    assert not np.allclose(narrow_image, wide_image)


def test_palm_inputs_are_not_mutated() -> None:
    inferred = np.array([[0.0, 0.0], [50.0, 0.0]])
    truth = np.array([[0.0, 0.0], [100.0, 0.0]])
    inferred_before = inferred.copy()
    truth_before = truth.copy()

    palm_image_squared_error(inferred, truth)

    np.testing.assert_array_equal(inferred, inferred_before)
    np.testing.assert_array_equal(truth, truth_before)


def test_matlab_generated_scoring_reference_when_available() -> None:
    path = Path(__file__).parent / "matlab" / "scoring_reference.mat"
    if not path.exists():
        pytest.skip(
            "MATLAB runtime is unavailable; run tests/matlab/generate_scoring_reference.m"
        )
    reference = loadmat(path, squeeze_me=False, struct_as_record=False)
    inferred_image = palm_density_image(
        reference["inferred_coords"], reference["true_coords"]
    )
    true_image = palm_density_image(reference["true_coords"], reference["true_coords"])
    np.testing.assert_allclose(inferred_image, reference["inferred_image_expected"])
    np.testing.assert_allclose(true_image, reference["true_image_expected"])
    np.testing.assert_allclose(
        palm_image_squared_error(reference["inferred_coords"], reference["true_coords"]),
        reference["score_expected"].item(),
    )

