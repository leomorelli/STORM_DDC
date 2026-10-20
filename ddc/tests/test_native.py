"""Native-path coverage, including numerical fallbacks and mutation guards."""

from unittest.mock import patch

import numpy as np
import pytest

import ddc._acceleration as acceleration
from ddc.distances import make_distance_matrix_individual
from ddc.mcmc import _eliminate_blinking_de_loc15
from ddc.probabilities import eliminate_blinking_de_loc15_mcmc

pytestmark = pytest.mark.skipif(not acceleration.native_available(), reason="optional C extension unavailable")


def test_compiled_linker_executes_and_preserves_all_results_and_inputs():
    rng = np.random.default_rng(20261008)
    locations = rng.normal(size=(100, 3)) * 40
    frames = np.sort(rng.integers(1, 25, size=100)).astype(float)
    constants = rng.normal(size=100) * 0.3
    deviation = rng.random((10, 100))
    labels = rng.permutation(100).astype(float) + 1
    snapshots = [v.copy() for v in (locations, frames, constants, deviation, labels)]
    for probability_mode in (False, True):
        native = acceleration.try_eliminate(locations, frames, constants, 10, 10,
            deviation, labels, probability_mode=probability_mode)
        assert native is not None, "ordinary inputs must execute compiled code"
        with patch.object(acceleration, "native_available", return_value=False):
            if probability_mode:
                expected_keep, expected_probability = eliminate_blinking_de_loc15_mcmc(
                    locations, frames, 10, 10, deviation, labels)
                np.testing.assert_array_equal(native[0].astype(float), expected_keep, strict=True)
                np.testing.assert_array_equal(native[1], expected_probability, strict=True)
            else:
                expected = _eliminate_blinking_de_loc15(locations, frames, constants, 10, 10, deviation, labels)
                actual = locations[native[0]], frames[native[0]], native[2]
                for left, right in zip(expected, actual):
                    np.testing.assert_array_equal(left, right, strict=True)
    for original, saved in zip((locations, frames, constants, deviation, labels), snapshots):
        np.testing.assert_array_equal(original, saved, strict=True)


def test_distance_bin_boundary_uses_original_numerics():
    locations = np.array([[0., 0.], [10., 0.], [25., 0.]])
    frames = np.array([1., 2., 8.])
    deviation = np.full((4, 10), 0.9)
    labels = np.array([1., 2., 3.])
    assert acceleration.try_eliminate(locations, frames, np.zeros(3), 10, 4,
        deviation, labels, probability_mode=False) is None
    with patch.object(acceleration, "native_available", return_value=False):
        expected = _eliminate_blinking_de_loc15(locations, frames, np.zeros(3), 10, 4, deviation, labels)
    actual = _eliminate_blinking_de_loc15(locations, frames, np.zeros(3), 10, 4, deviation, labels)
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right, strict=True)


def test_native_distance_arrays_match_source_bit_patterns():
    rng = np.random.default_rng(1082026)
    for dimensions in (2, 3):
        locations = rng.normal(size=(150, dimensions)) * 100
        frames = rng.integers(1, 30, size=150).astype(float)
        cells = [np.array([float(i)]) for i in range(30)]
        with patch.object(acceleration, "native_available", return_value=False):
            expected = make_distance_matrix_individual(locations, cells, frames, 29)
        actual = make_distance_matrix_individual(locations, cells, frames, 29)
        assert [v.tobytes() for v in actual] == [v.tobytes() for v in expected]


def test_native_can_be_disabled_without_reinstalling(monkeypatch):
    monkeypatch.setenv("DDC_DISABLE_NATIVE", "1")
    assert not acceleration.native_available()
