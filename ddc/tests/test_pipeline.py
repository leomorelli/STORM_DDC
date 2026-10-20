from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from ddc.io import DDCDataset
from ddc.mcmc import ProbabilityCache
from ddc.pipeline import (
    _matlab_normalize_photon_weights,
    _matlab_weighted_coordinate_sum,
    DDCConfig,
    _matlab_round_frames,
    _pad_localizations_source,
    photon_weighted_collapse,
    run_ddc,
)


def _dataset(sizes=(11,), dtype=np.float64, dimensions=2, truth=True):
    locations = [
        np.column_stack(
            [np.arange(size, dtype=float) + column * 100 for column in range(dimensions)]
        )
        for size in sizes
    ]
    frames = [np.arange(1, size + 1, dtype=dtype) for size in sizes]
    truths = (
        [np.empty((0, dimensions)) for _ in sizes]
        if truth
        else []
    )
    return DDCDataset(locations, frames, truths)


def _distribution_result(config=DDCConfig()):
    return (
        np.array([0.0, 80.0, np.inf]),
        np.array([0.75, 0.25]),
        np.zeros((1, 2)),
        config.resolution,
        np.ones(config.n_f),
        np.ones((config.n_f, 2)),
    )


def _mcmc_result(
    n,
    *,
    steps,
    history,
    marker=1.0,
    trajectory=None,
    cache=None,
):
    labels = (
        np.arange(1, n + 1, dtype=float)
        if trajectory is None
        else np.asarray(trajectory, dtype=float)
    )
    probability_cache = cache or ProbabilityCache(
        np.ones((1, 2)), np.zeros((1, 2)), np.ones(1)
    )
    return SimpleNamespace(
        traj2=labels,
        final_localizations_blinking_corrected=np.full((1, 3), marker),
        final_frame_blinking_corrected=np.array([marker]),
        likelihood_history=np.asarray(history, dtype=float),
        score_history=np.asarray(history, dtype=float) * 0.1,
        number_of_localizations=np.full(len(history), n, dtype=float),
        steps=steps,
        constantf=np.full(n, marker),
        orderf=np.arange(1, n + 1, dtype=np.int64),
        probability_cache=probability_cache,
        constant2ft=np.full(n, -marker),
    )


def test_config_exposes_matlab_defaults() -> None:
    config = DDCConfig()
    assert config.resolution == 80
    assert config.stepper == 100
    assert config.n_f == 200
    assert config.timer == 720
    assert config.photon_weighted_correction == 1


def test_pipeline_frame_rounding_is_matlab_style_and_preserves_dtype() -> None:
    frames = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
    rounded = _matlab_round_frames(frames)
    np.testing.assert_array_equal(rounded, [-2, -1, 1, 2])
    assert rounded.dtype == np.float32
    integers = np.array([1, 2], dtype=np.uint16)
    integer_rounded = _matlab_round_frames(integers)
    np.testing.assert_array_equal(integer_rounded, integers)
    assert integer_rounded.dtype == np.uint16


@pytest.mark.parametrize(
    ("shape", "expected_shape", "third"),
    [
        ((4, 2), (4, 3), 0.0),
        ((4, 3), (4, 3), 9.0),
        ((1, 2), (1, 3), 0.0),
        ((2, 2), (2, 3), 0.0),
        ((1, 3), (1, 3), 0.0),
        ((2, 3), (2, 3), 0.0),
        ((0, 2), (0, 3), None),
    ],
)
def test_source_min_size_padding_semantics(shape, expected_shape, third) -> None:
    values = np.full(shape, 9.0)
    padded = _pad_localizations_source(values)
    assert padded.shape == expected_shape
    if third is not None:
        np.testing.assert_array_equal(padded[:, 2], third)
    np.testing.assert_array_equal(values, np.full(shape, 9.0))


def test_default_photons_empty_truth_and_no_caller_mutation() -> None:
    dataset = _dataset((10,), truth=False)
    before = dataset.localizations[0].copy()
    with patch("ddc.pipeline.determine_blinking_distribution", return_value=_distribution_result()):
        result = run_ddc(dataset, rngs=[np.random.default_rng(1)])
    np.testing.assert_array_equal(result.photons[0], np.ones(10))
    assert result.true_localizations[0].shape == (0,)
    assert result.working_localizations[0].shape == (10, 3)
    np.testing.assert_array_equal(dataset.localizations[0], before)
    assert result.states[0].step == 100


def test_first_empty_photon_cell_replaces_all_images_with_units() -> None:
    dataset = _dataset((10, 10))
    with patch("ddc.pipeline.determine_blinking_distribution", return_value=_distribution_result()):
        result = run_ddc(
            dataset,
            photons=[np.array([]), np.full(10, 7.0)],
            rngs=[np.random.default_rng(1), np.random.default_rng(2)],
        )
    np.testing.assert_array_equal(result.photons[0], 1)
    np.testing.assert_array_equal(result.photons[1], 1)


def test_dataset_photons_are_automatic_and_explicit_argument_overrides() -> None:
    dataset = _dataset((10,))
    dataset.photons = [np.full(10, 7, dtype=np.uint8)]
    with patch("ddc.pipeline.determine_blinking_distribution", return_value=_distribution_result()):
        automatic = run_ddc(dataset, rngs=[np.random.default_rng(1)])
        overridden = run_ddc(
            dataset, photons=[np.full(10, 3, dtype=np.uint16)],
            rngs=[np.random.default_rng(1)]
        )
    assert automatic.photons[0].dtype == np.uint8
    assert overridden.photons[0].dtype == np.uint16
    np.testing.assert_array_equal(automatic.photons[0], 7)
    np.testing.assert_array_equal(overridden.photons[0], 3)


def test_empty_first_loaded_photon_cell_replaces_all_but_later_empty_is_retained() -> None:
    dataset = _dataset((10, 10))
    dataset.photons = [np.array([], dtype=np.uint8), np.full(10, 9, dtype=np.uint8)]
    with patch("ddc.pipeline.determine_blinking_distribution", return_value=_distribution_result()):
        defaulted = run_ddc(dataset, rngs=[np.random.default_rng(1), np.random.default_rng(2)])
    assert all(np.array_equal(cell, np.ones(10)) for cell in defaulted.photons)

    dataset.photons = [np.ones(10, dtype=np.uint8), np.array([], dtype=np.uint8)]
    with patch("ddc.pipeline.determine_blinking_distribution", return_value=_distribution_result()):
        retained = run_ddc(dataset, rngs=[np.random.default_rng(1), np.random.default_rng(2)])
    assert retained.photons[1].size == 0


def test_uint16_distribution_bug_002_precedes_mcmc_bug_001() -> None:
    dataset = _dataset((11,), dtype=np.uint16)
    mcmc = Mock()
    with patch("ddc.pipeline.run_mcmc", mcmc):
        with pytest.raises(TypeError, match="DDC-SOURCE-BUG-002"):
            run_ddc(dataset, rngs=[np.random.default_rng(1)])
    assert mcmc.call_count == 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_supported_frame_dtype_reaches_mcmc_without_conversion(dtype) -> None:
    dataset = _dataset((11,), dtype=dtype)
    calls = []

    def fake_mcmc(*args, **kwargs):
        calls.append(args)
        return _mcmc_result(11, steps=1, history=[1.0])

    config = DDCConfig(stepper=1, photon_weighted_correction=0)
    with (
        patch("ddc.pipeline.determine_blinking_distribution", return_value=_distribution_result(config)),
        patch("ddc.pipeline.run_mcmc", side_effect=fake_mcmc),
    ):
        run_ddc(dataset, config, rngs=[np.random.default_rng(1)])
    assert calls[0][2].dtype == np.dtype(dtype)


def test_eligibility_exact_threshold_and_shared_distribution_once() -> None:
    dataset = _dataset((10, 11))
    distribution = Mock(return_value=_distribution_result(DDCConfig(stepper=1)))
    mcmc = Mock(return_value=_mcmc_result(11, steps=1, history=[1.0]))
    with (
        patch("ddc.pipeline.determine_blinking_distribution", distribution),
        patch("ddc.pipeline.run_mcmc", mcmc),
    ):
        result = run_ddc(
            dataset,
            DDCConfig(stepper=1, photon_weighted_correction=0),
            rngs=[np.random.default_rng(1), np.random.default_rng(2)],
        )
    assert distribution.call_count == 1
    assert mcmc.call_count == 1 and mcmc.call_args.args[0] == 2
    np.testing.assert_array_equal(result.steps, [1, 1])


def test_length_quirk_can_make_2x11_image_eligible() -> None:
    locations = np.arange(22.0).reshape(2, 11)
    dataset = DDCDataset([locations], [np.array([1.0, 2.0])], [np.empty((0, 11))])
    mcmc = Mock(return_value=_mcmc_result(2, steps=1, history=[1.0]))
    with (
        patch("ddc.pipeline.determine_blinking_distribution", return_value=_distribution_result(DDCConfig(stepper=1))),
        patch("ddc.pipeline.run_mcmc", mcmc),
    ):
        run_ddc(dataset, DDCConfig(stepper=1), rngs=[np.random.default_rng(1)])
    assert mcmc.call_count == 1


def test_more_than_8000_is_diagnostic_and_does_not_block_analysis() -> None:
    dataset = _dataset((8001,))
    mcmc = Mock(return_value=_mcmc_result(8001, steps=1, history=[1.0]))
    config = DDCConfig(stepper=1, photon_weighted_correction=0)
    with (
        patch("ddc.pipeline.determine_blinking_distribution", return_value=_distribution_result(config)),
        patch("ddc.pipeline.run_mcmc", mcmc),
    ):
        result = run_ddc(dataset, config, rngs=[np.random.default_rng(1)])
    assert mcmc.call_count == 1
    assert result.diagnostics == ("image 1: more than 8000 localizations",)


def test_continuation_writeback_best_replacement_and_cache_reuse() -> None:
    dataset = _dataset((11,))
    calls = []
    cache1 = ProbabilityCache(np.ones((1, 2)), np.ones((1, 2)), np.array([0.5]))

    def fake_mcmc(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            return _mcmc_result(11, steps=1, history=[5.0], marker=5, cache=cache1)
        if len(calls) == 2:
            return _mcmc_result(11, steps=2, history=[5.0, 7.0], marker=7, cache=cache1)
        return _mcmc_result(11, steps=3, history=[5.0, 7.0, 6.0], marker=99, cache=cache1)

    config = DDCConfig(stepper=3, photon_weighted_correction=0)
    with (
        patch("ddc.pipeline.determine_blinking_distribution", return_value=_distribution_result(config)),
        patch("ddc.pipeline.run_mcmc", side_effect=fake_mcmc),
    ):
        result = run_ddc(dataset, config, rngs=[np.random.default_rng(1)])

    assert len(calls) == 3
    assert calls[0][8] == 0 and calls[1][8] == 1 and calls[2][8] == 2
    np.testing.assert_array_equal(calls[1][12], [5.0])
    assert calls[1][15] is cache1 and calls[2][15] is cache1
    np.testing.assert_array_equal(result.final_localizations[0], [[7.0, 7.0, 7.0]])
    np.testing.assert_array_equal(result.states[0].likelihood, [5.0, 7.0, 6.0])
    np.testing.assert_array_equal(result.states[0].constantf, 99.0)


def test_multi_image_rng_objects_are_explicit_and_independent() -> None:
    dataset = _dataset((11, 11))
    rng1 = np.random.default_rng(10)
    rng2 = np.random.default_rng(20)
    observed = []

    def fake_mcmc(*args, **kwargs):
        rng = kwargs["rng"]
        if args[0] == 1:
            _ = rng.random(100)
        observed.append((args[0], float(rng.random())))
        return _mcmc_result(11, steps=1, history=[1.0])

    config = DDCConfig(stepper=1, photon_weighted_correction=0)
    with (
        patch("ddc.pipeline.determine_blinking_distribution", return_value=_distribution_result(config)),
        patch("ddc.pipeline.run_mcmc", side_effect=fake_mcmc),
    ):
        run_ddc(dataset, config, rngs=[rng1, rng2])
    expected_second = float(np.random.default_rng(20).random())
    assert observed[1] == (2, expected_second)


def test_photon_weighted_equal_unequal_multiple_and_unweighted_frames() -> None:
    locations = np.array([[0.0, 0.0, 1.0], [10.0, 20.0, 3.0], [30.0, 0.0, 5.0]])
    frames = np.array([1.0, 5.0, 9.0])
    labels = np.array([2.0, 2.0, 1.0])
    collapsed, collapsed_frames = photon_weighted_collapse(
        locations, frames, labels, [1.0, 3.0, 2.0]
    )
    # unique(Traj2) is sorted: trajectory 1 precedes trajectory 2.
    np.testing.assert_allclose(collapsed, [[30, 0, 5], [7.5, 15, 2.5]])
    np.testing.assert_allclose(collapsed_frames, [9, 3])


@pytest.mark.parametrize(
    ("values", "dtype", "expected"),
    [
        ([1, 3], np.float64, [0.25, 0.75]),
        ([1, 3], np.float32, [0.25, 0.75]),
        ([1, 1], np.uint8, [1, 1]),
        ([1, 1, 1], np.uint8, [0, 0, 0]),
        ([1, 3], np.uint8, [0, 1]),
        ([1, 3], np.uint16, [0, 1]),
        ([1, 3], np.int8, [0, 1]),
        ([1, 3], np.int16, [0, 1]),
    ],
)
def test_matlab_photon_normalization_dtype_and_values(values, dtype, expected) -> None:
    normalized = _matlab_normalize_photon_weights(np.array(values, dtype=dtype))
    assert normalized.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(normalized, np.asarray(expected, dtype=dtype))


def test_integer_zero_sum_and_float_zero_sum_match_matlab_classes() -> None:
    integer = _matlab_normalize_photon_weights(np.array([0, 0], dtype=np.uint8))
    floating = _matlab_normalize_photon_weights(np.array([0, 0], dtype=np.float64))
    np.testing.assert_array_equal(integer, [0, 0])
    assert integer.dtype == np.uint8
    assert np.isnan(floating).all() and floating.dtype == np.float64


def test_integer_photons_preserve_scalar_coordinate_arithmetic_and_vector_error() -> None:
    locations = np.array([[0.6, 1.4, 300.0], [10.0, 20.0, 30.0]])
    collapsed, frames = photon_weighted_collapse(
        locations, [1.0, 9.0], [1, 2], np.array([1, 1], dtype=np.uint8)
    )
    # Scalar double .* uint8 returns uint8: round and saturate before sum.
    np.testing.assert_array_equal(collapsed, [[1, 1, 255], [10, 20, 30]])
    np.testing.assert_array_equal(frames, [1, 9])
    with pytest.raises(TypeError, match="non-scalar double coordinate"):
        photon_weighted_collapse(
            locations, [1.0, 9.0], [1, 1], np.array([1, 3], dtype=np.uint8)
        )


@pytest.mark.parametrize(
    ("coordinate_dtype", "weight_dtype"),
    [
        (np.float64, np.float64),
        (np.float64, np.float32),
        (np.float32, np.float64),
        (np.float32, np.float32),
    ],
)
def test_floating_coordinate_weight_combinations_succeed(
    coordinate_dtype, weight_dtype
) -> None:
    result = _matlab_weighted_coordinate_sum(
        np.array([1, 2], dtype=coordinate_dtype),
        np.array([0.25, 0.75], dtype=weight_dtype),
    )
    assert result == 1.75


def test_double_single_uses_single_intermediate_multiplication_and_sum() -> None:
    coordinates = np.array([1.0e8, 1.0, -1.0e8], dtype=np.float64)
    weights = np.ones(3, dtype=np.float32)
    actual = _matlab_weighted_coordinate_sum(coordinates, weights)
    float64_then_cast = float(
        np.float32(np.sum(coordinates * weights.astype(np.float64)))
    )
    assert actual == 0.0
    assert float64_then_cast == 1.0


def test_float32_photon_collapse_with_double_coordinates_succeeds() -> None:
    collapsed, frames = photon_weighted_collapse(
        np.array([[0.0, 2.0, 4.0], [8.0, 10.0, 12.0]], dtype=np.float64),
        [1.0, 3.0], [1, 1], np.array([1, 3], dtype=np.float32)
    )
    np.testing.assert_allclose(collapsed, [[6.0, 8.0, 10.0]])
    np.testing.assert_array_equal(frames, [2.0])


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_nonscalar_double_coordinates_integer_weights_still_raise(dtype) -> None:
    with pytest.raises(TypeError, match="non-scalar double coordinate"):
        _matlab_weighted_coordinate_sum(
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([0, 1], dtype=dtype),
        )


def test_scalar_double_integer_weight_retains_integer_arithmetic() -> None:
    assert _matlab_weighted_coordinate_sum(
        np.array([300.0], dtype=np.float64), np.array([1], dtype=np.uint8)
    ) == 255.0


def test_single_coordinates_integer_weights_are_not_scalar_double_exception() -> None:
    with pytest.raises(TypeError, match="single coordinates"):
        _matlab_weighted_coordinate_sum(
            np.array([1.0], dtype=np.float32), np.array([1], dtype=np.uint8)
        )


def test_correction_disabled_bypasses_integer_photon_normalization() -> None:
    dataset = _dataset((11,))
    dataset.photons = [np.ones(11, dtype=np.uint8)]
    config = DDCConfig(stepper=1, photon_weighted_correction=0)
    mcmc = Mock(return_value=_mcmc_result(11, steps=1, history=[1.0], marker=8))
    with (
        patch("ddc.pipeline.determine_blinking_distribution", return_value=_distribution_result(config)),
        patch("ddc.pipeline.run_mcmc", mcmc),
        patch("ddc.pipeline._matlab_normalize_photon_weights") as normalize,
    ):
        result = run_ddc(dataset, config, rngs=[np.random.default_rng(1)])
    normalize.assert_not_called()
    np.testing.assert_array_equal(result.final_localizations[0], [[8, 8, 8]])


def test_loaded_photon_dtype_reaches_collapse_unchanged() -> None:
    dataset = _dataset((11,))
    dataset.photons = [np.ones(11, dtype=np.uint8)]
    config = DDCConfig(stepper=1)
    mcmc = Mock(return_value=_mcmc_result(11, steps=1, history=[1.0]))
    collapse = Mock(return_value=(np.ones((1, 3)), np.ones(1)))
    with (
        patch("ddc.pipeline.determine_blinking_distribution", return_value=_distribution_result(config)),
        patch("ddc.pipeline.run_mcmc", mcmc),
        patch("ddc.pipeline.photon_weighted_collapse", collapse),
    ):
        run_ddc(dataset, config, rngs=[np.random.default_rng(1)])
    assert collapse.call_args.args[3].dtype == np.uint8


def test_real_loaded_photons_flow_automatically_into_pipeline() -> None:
    from pathlib import Path
    from ddc.io import load_dataset
    path = (Path(__file__).resolve().parents[1] / "Main_DDC_Folder" /
            "User_Guide_Files" / "Example_3d_2darkstate_random_data.mat")
    if not path.exists():
        pytest.skip("repository photon MAT fixture is unavailable")
    dataset = load_dataset(path)
    assert dataset.photons is not None and dataset.photons[0].dtype == np.uint8
    config = DDCConfig(stepper=1, photon_weighted_correction=0)
    mcmc = Mock(return_value=_mcmc_result(
        dataset.frames[0].size, steps=1, history=[1.0]
    ))
    with (
        patch("ddc.pipeline.determine_blinking_distribution", return_value=_distribution_result(config)),
        patch("ddc.pipeline.run_mcmc", mcmc),
    ):
        result = run_ddc(dataset, config, rngs=[np.random.default_rng(1)])
    assert result.photons[0].dtype == np.uint8
    np.testing.assert_array_equal(result.photons[0], dataset.photons[0])


def test_matlab_photon_integer_normalization_oracle_when_available() -> None:
    from pathlib import Path
    from scipy.io import loadmat
    path = Path(__file__).parent / "matlab" / "photon_integer_normalization_reference.mat"
    if not path.exists():
        pytest.skip(
            "MATLAB runtime is unavailable; run "
            "tests/matlab/probe_photon_integer_normalization.m"
        )
    reference = loadmat(path, squeeze_me=True, struct_as_record=False)
    records = np.atleast_1d(reference["records"])
    expected = [
        (np.float64, [0.25, 0.75]),
        (np.float32, [0.25, 0.75]),
        (np.uint8, [1, 1]),
        (np.uint8, [0, 0, 0]),
        (np.uint8, [0, 1]),
        (np.uint16, [0, 1]),
    ]
    for record, (dtype, values) in zip(records[:6], expected, strict=True):
        actual = _matlab_normalize_photon_weights(
            np.asarray(record.original, dtype=dtype)
        )
        np.testing.assert_array_equal(actual, np.asarray(values, dtype=dtype))


def test_zero_photon_sum_propagates_nan_and_single_trajectory_shape() -> None:
    collapsed, frames = photon_weighted_collapse(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], [1.0, 3.0], [4, 4], [0.0, 0.0]
    )
    assert collapsed.shape == (1, 3) and np.isnan(collapsed).all()
    np.testing.assert_array_equal(frames, [2.0])


def test_pipeline_photon_correction_uses_padded_original_not_raw_mcmc_best() -> None:
    dataset = DDCDataset(
        [np.array([[0.0, 0.0], [10.0, 20.0]] + [[30.0, 40.0]] * 9)],
        [np.arange(1.0, 12.0)],
        [np.empty((0, 2))],
    )
    trajectory = np.array([1, 1] + list(range(2, 11)), dtype=float)
    mcmc = Mock(return_value=_mcmc_result(
        11, steps=1, history=[1.0], marker=999, trajectory=trajectory
    ))
    config = DDCConfig(stepper=1)
    with (
        patch("ddc.pipeline.determine_blinking_distribution", return_value=_distribution_result(config)),
        patch("ddc.pipeline.run_mcmc", mcmc),
    ):
        result = run_ddc(
            dataset, config, photons=[np.r_[1.0, 3.0, np.ones(9)]],
            rngs=[np.random.default_rng(1)],
        )
    np.testing.assert_allclose(result.final_localizations[0][0], [7.5, 15.0, 0.0])
    assert result.final_frames[0][0] == 1.5


def test_no_serialization_plotting_or_parallel_dependency_is_imported() -> None:
    import ddc.pipeline as pipeline
    source_names = set(pipeline.__dict__)
    assert not ({"matplotlib", "h5py", "multiprocessing", "concurrent"} & source_names)


class _SmokeRNG:
    def __init__(self):
        self.base = np.random.default_rng(42)

    def random(self, size=None):
        return self.base.random(size)

    def normal(self, size=None):
        return self.base.normal(size=size)

    def integers(self, low, high=None, size=None):
        if low == 1 and high == 4 and size is None:
            return 2
        return self.base.integers(low, high, size)

    def permutation(self, x):
        if isinstance(x, (int, np.integer)) and x == 11:
            return np.arange(10, -1, -1)
        return self.base.permutation(x)


@pytest.mark.parametrize(
    "truth_kwargs",
    [{}, {"true_localizations": None}, {"true_localizations": []},
     {"true_localizations": [np.empty((0, 2))]}],
    ids=["omitted", "none", "empty-list", "empty-cell"],
)
def test_real_component_floating_pipeline_smoke(truth_kwargs) -> None:
    frames = np.concatenate((np.ones(11), [2.0], np.full(11, 4.0)))
    x = np.concatenate((np.arange(11.0) * 10, [0.2], np.arange(11.0) * 10 + 0.5))
    locations = np.column_stack((x, np.zeros(x.size)))
    dataset = DDCDataset([locations], [frames], **truth_kwargs)
    before = locations.copy()

    result = run_ddc(
        dataset,
        DDCConfig(resolution=10.0, stepper=10, n_f=2, timer=30.0),
        rngs=[_SmokeRNG()],
    )

    assert result.steps[0] == 10
    assert result.states[0].probability_cache.prob_distributions.size > 0
    assert result.final_localizations[0] is not None
    assert result.trajectories[0].size == locations.shape[0]
    assert len(result.true_localizations) == 1
    assert result.true_localizations[0].size == 0
    np.testing.assert_array_equal(result.states[0].score, 0.0)
    np.testing.assert_array_equal(dataset.localizations[0], before)


def test_matlab_generated_pipeline_reference_when_available() -> None:
    from pathlib import Path
    from scipy.io import loadmat
    path = Path(__file__).parent / "matlab" / "pipeline_reference.mat"
    if not path.exists():
        pytest.skip(
            "MATLAB runtime is unavailable; run tests/matlab/generate_pipeline_reference.m"
        )
    reference = loadmat(path, squeeze_me=True)
    np.testing.assert_array_equal(reference["padded_2d_expected"][:, 2], 0)
    np.testing.assert_array_equal(reference["default_photons_expected"], 1)
    np.testing.assert_allclose(
        reference["weighted_locations_expected"], [[7.5, 15.0, 2.5]]
    )
