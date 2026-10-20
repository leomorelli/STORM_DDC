from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pytest
from scipy.io import loadmat

from ddc.io import DDCDataset
from ddc.mcmc import ProbabilityCache
from ddc.pipeline import DDCConfig, DDCImageState, DDCPipelineResult, run_ddc
from ddc.serialization import (
    CheckpointWriter,
    DDCCheckpoint,
    DDCSourceBug007,
    DDCSourceResumeBlockedByEarlierDefect,
    FINISH_RUNS_STALE_OUTPUT_MAPPING,
    load_checkpoint,
    make_analyzed_filename,
    resume_ddc,
    resume_finish_runs_source,
    save_initial_checkpoint,
    save_v73_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = (
    ROOT / "Main_DDC_Folder" / "User_Guide_Files" /
    "Analyzed_Time15-Nov-2020_Split_Example_3d_2darkstate_random_data.mat"
)

DDC_MCMC_SOURCE = ROOT / "Main_DDC_Folder" / "Scripts_Functions" / "DDC_MCMC.m"
FINISH_RUNS_SOURCES = (
    ROOT / "Main_DDC_Folder" / "Scripts_Functions" / "Finish_Runs.m",
    ROOT / "Main_DDC_Folder" / "Scripts_To_Use_on_Cluster" / "Finish_Runs.m",
)


def test_current_matlab_resume_contract_and_exact_stale_output_shift() -> None:
    declaration = DDC_MCMC_SOURCE.read_text().splitlines()[7]
    assert declaration.startswith("function [Traj2,")
    outputs = declaration.split("[", 1)[1].split("]", 1)[0].split(",")
    assert [value.strip() for value in outputs] == [
        "Traj2", "Final_Localizations_Blinking_Corrected",
        "Final_Frame_Blinking_Corrected", "LikHood", "Score", "Numb_of_Loc",
        "bestlik", "steps", "Constantf", "Orderf", "Prob_dists", "Constant2ft",
    ]
    assert len(outputs) == 12
    expected = (
        ("Final_Localizations_Blinking_Corrected2", "Traj2"),
        ("Final_Frame_Blinking_Corrected2", "Final_Localizations_Blinking_Corrected"),
        ("LikHood", "Final_Frame_Blinking_Corrected"),
        ("Score", "LikHood"), ("Numb", "Score"),
        ("bestlik", "Numb_of_Loc"), ("steps", "bestlik"),
        ("Constant", "steps"), ("Order", "Constantf"),
        ("Prob_dists2", "Orderf"), ("Constant2", "Prob_dists"),
    )
    assert FINISH_RUNS_STALE_OUTPUT_MAPPING == expected
    assert dict(expected)["Prob_dists2"] == "Orderf"
    for path in FINISH_RUNS_SOURCES:
        source = path.read_text()
        call = next(line for line in source.splitlines() if "]=DDC_MCMC(" in line)
        requested = call.split("[", 1)[1].split("]", 1)[0].split(",")
        assert len(requested) == 11
        assert "Traj2" not in requested


def _source_resume_checkpoint(*, width: int, integer_frames: bool = False) -> DDCCheckpoint:
    checkpoint = _checkpoint()
    checkpoint.result.working_localizations[0] = np.arange(
        2 * width, dtype=np.float64
    ).reshape(2, width)
    checkpoint.result.frames[0] = np.arange(
        checkpoint.result.working_localizations[0].shape[0],
        dtype=np.uint16 if integer_frames else np.float64,
    )
    checkpoint.result.states[0].step = 0
    checkpoint.result.states[1].step = checkpoint.config.stepper
    return checkpoint


def test_source_finish_runs_exposes_bug007_for_eligible_float_frames() -> None:
    with pytest.raises(DDCSourceBug007, match="Prob_dists2 receives numeric Orderf"):
        resume_finish_runs_source(_source_resume_checkpoint(width=11))


def test_source_finish_runs_completed_checkpoint_is_noop() -> None:
    checkpoint = _checkpoint(completed=True)
    resumed = resume_finish_runs_source(checkpoint)
    assert [state.step for state in resumed.states] == [3, 3]


def test_source_finish_runs_uses_matlab_length_and_marks_ineligible_complete() -> None:
    checkpoint = _source_resume_checkpoint(width=10)
    resumed = resume_finish_runs_source(checkpoint)
    assert resumed.states[0].step == checkpoint.config.stepper
    assert checkpoint.result.states[0].step == 0  # detached diagnostic result


def test_source_finish_runs_does_not_supersede_earlier_integer_defects() -> None:
    with pytest.raises(DDCSourceResumeBlockedByEarlierDefect, match="not yet observable"):
        resume_finish_runs_source(
            _source_resume_checkpoint(width=11, integer_frames=True)
        )


def test_corrected_resume_api_is_explicitly_not_literal_finish_runs() -> None:
    assert "corrected Python continuation" in (resume_ddc.__doc__ or "")
    assert "obsolete eleven-output assignment" in (resume_ddc.__doc__ or "")


def _cache(marker: float = 1.0) -> ProbabilityCache:
    return ProbabilityCache(
        np.array([[marker, 0.0], [np.nan, np.inf]]),
        np.array([[0.0, marker + 1.0]]),
        np.array([marker, np.nan, np.inf]),
    )


def _dataset() -> DDCDataset:
    return DDCDataset(
        localizations=[
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
        ],
        frames=[
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([9, 10], dtype=np.uint16)[:1],
        ],
        true_localizations=[np.empty((0, 3)), np.array([[7.0, 8.0, 9.0]])],
        photons=[np.array([1, 3], dtype=np.uint8), np.array([0.5], dtype=np.float32)],
        came_from_image=np.array([4, 5], dtype=np.uint8),
        addon=np.array([11, 12], dtype=np.uint16),
        cuts=(
            [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            [np.array([5.0, 6.0]), np.array([7.0, 8.0])],
            [np.array([9.0, 10.0]), np.array([11.0, 12.0])],
        ),
        parameters_to_split=[np.array([1, 2, 3]), np.array([4, 5, 6])],
        source=Path("/input/example.mat"),
        mat_passthrough={
            "ArbitraryUserVariable": np.array([[1.5, 2.5]], dtype=np.float32)
        },
    )


def _result(*, initial: bool = False, completed: bool = False) -> DDCPipelineResult:
    dataset = _dataset()
    state0 = DDCImageState()
    state1 = DDCImageState()
    if not initial:
        state0 = DDCImageState(
            step=3 if completed else 1,
            constantf=np.array([1.0, 2.0]),
            constant2ft=np.array([3.0, 4.0]),
            orderf=np.array([2, 1], dtype=np.int64),
            likelihood=np.array([-4.0, -3.0]),
            score=np.array([0.4, 0.3]),
            number_of_localizations=np.array([2.0, 1.0]),
            probability_cache=_cache(),
        )
        state1.step = 3
    return DDCPipelineResult(
        final_localizations=(
            [None, None] if initial
            else [np.array([[2.0, 3.0, 4.0]]), None]
        ),
        final_frames=[None, None] if initial else [np.array([1.5]), None],
        trajectories=[None, None] if initial else [np.array([1.0, 1.0]), None],
        states=[state0, state1],
        working_localizations=[value.copy() for value in dataset.localizations],
        frames=[value.copy() for value in dataset.frames],
        true_localizations=[value.copy() for value in dataset.true_localizations],
        photons=[value.copy() for value in dataset.photons or []],
        bins=np.array([0.0, 80.0, np.inf]),
        distribution_for_blink=np.array([0.75, 0.25]),
        resolution=80.0,
        x_overall=np.array([0.2, np.nan, np.inf]),
        m_mat=np.array([[1.0, 2.0], [3.0, 4.0]]),
        diagnostics=("diagnostic",),
    )


def _checkpoint(*, initial: bool = False, completed: bool = False) -> DDCCheckpoint:
    dataset = _dataset()
    config = DDCConfig(stepper=3, n_f=2, photon_weighted_correction=0, timer=5)
    return DDCCheckpoint.from_result(
        _result(initial=initial, completed=completed),
        dataset,
        config,
        "Analyzed_Time15-Nov-2020_example.mat",
        source_filename="example.mat",
        source_metadata={
            "pathname": "/input/",
            "full_filename": "/input/example.mat",
            "cluster": np.array([[0.0]]),
        },
        phase="initial" if initial else "outer",
    )


def _matlab_class(item: h5py.Dataset | h5py.Group) -> bytes:
    return bytes(item.attrs["MATLAB_class"])


def _assert_result_equal(left: DDCPipelineResult, right: DDCPipelineResult) -> None:
    np.testing.assert_array_equal(left.steps, right.steps)
    for a, b in zip(left.states, right.states, strict=True):
        np.testing.assert_allclose(a.constantf, b.constantf, equal_nan=True)
        np.testing.assert_allclose(a.constant2ft, b.constant2ft, equal_nan=True)
        np.testing.assert_array_equal(a.orderf, b.orderf)
        np.testing.assert_allclose(a.likelihood, b.likelihood, equal_nan=True)
        np.testing.assert_allclose(a.score, b.score, equal_nan=True)
        np.testing.assert_allclose(
            a.number_of_localizations, b.number_of_localizations, equal_nan=True
        )
        np.testing.assert_allclose(
            a.probability_cache.deviation_in_probability,
            b.probability_cache.deviation_in_probability,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            a.probability_cache.prob_distributions,
            b.probability_cache.prob_distributions,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            a.probability_cache.dscale_store,
            b.probability_cache.dscale_store,
            equal_nan=True,
        )
    for left_cells, right_cells in (
        (left.final_localizations, right.final_localizations),
        (left.final_frames, right.final_frames),
        (left.trajectories, right.trajectories),
    ):
        for a, b in zip(left_cells, right_cells, strict=True):
            assert (a is None) == (b is None)
            if a is not None:
                np.testing.assert_allclose(a, b, equal_nan=True)


def test_analyzed_filename_format_and_extension() -> None:
    assert make_analyzed_filename(
        "Split_Example_3d_2darkstate_random_data.mat",
        on_date=date(2020, 11, 15),
    ) == "Analyzed_Time15-Nov-2020_Split_Example_3d_2darkstate_random_data.mat"
    assert not make_analyzed_filename(
        "input.mat", on_date=date(2020, 11, 15)
    ).endswith(".mat.mat")


def test_initial_checkpoint_is_v5_roundtrips_and_has_pre_mcmc_state(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.mat"
    save_initial_checkpoint(_checkpoint(initial=True), path)

    assert not h5py.is_hdf5(path)
    assert path.read_bytes().startswith(b"MATLAB 5.0 MAT-file")
    raw = loadmat(path, squeeze_me=False, struct_as_record=False)
    np.testing.assert_array_equal(raw["Step"], [[0, 0]])
    assert raw["Lik"].shape == (1, 2)
    assert all(np.asarray(cell).size == 0 for cell in raw["Lik"].ravel(order="F"))
    np.testing.assert_allclose(raw["bins"].reshape(-1), [0, 80, np.inf])
    np.testing.assert_allclose(
        raw["Distribution_for_Blink"].reshape(-1), [0.75, 0.25]
    )
    np.testing.assert_allclose(raw["X_overall"].reshape(-1), [0.2, np.nan, np.inf], equal_nan=True)
    np.testing.assert_array_equal(raw["M_mat"], [[1, 2], [3, 4]])

    loaded = load_checkpoint(path)
    np.testing.assert_array_equal(loaded.result.steps, [0, 0])
    assert all(state.likelihood.size == 0 for state in loaded.result.states)
    assert loaded.result.final_localizations == [None, None]


def test_v73_structure_cells_sparse_cache_dtypes_and_names(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.mat"
    save_v73_checkpoint(_checkpoint(), path)
    assert h5py.is_hdf5(path)

    required = {
        "LocalizationsFinal", "Frame_Information", "TrueLocalizations", "Photons",
        "Final_Loc_Blinking_Corrected", "Final_Frame_Blinking_Corrected",
        "Trajectory_of_Localizations", "Constantf", "Constantf2", "Orderf",
        "Step", "Lik", "RelScore", "Numb_of_Loc", "Prob_dists", "bins",
        "Distribution_for_Blink", "X_overall", "M_mat", "Resolution", "stepper",
        "N_f", "Photon_weighted_Correction", "timer", "addonarray",
        "ArbitraryUserVariable", "Came_from_image", "Parameters_to_split",
        "cut1array", "cut2array", "cut3array", "filename", "pathname",
        "full_filename", "String", "cluster",
    }
    with h5py.File(path) as handle:
        assert required <= set(handle)
        for name in (
            "LocalizationsFinal", "Frame_Information", "TrueLocalizations", "Photons",
            "Final_Loc_Blinking_Corrected", "Final_Frame_Blinking_Corrected",
            "Trajectory_of_Localizations", "Constantf", "Constantf2", "Orderf",
            "Lik", "RelScore", "Numb_of_Loc", "Prob_dists",
        ):
            assert _matlab_class(handle[name]) == b"cell"
        # Source MATLAB cells are 1xN; v7.3 stores reversed HDF5 dimensions N x 1.
        assert handle["LocalizationsFinal"].shape == (2, 1)
        first_locations = handle[handle["LocalizationsFinal"][0, 0]]
        assert first_locations.shape == (3, 2)
        first_frame = handle[handle["Frame_Information"][0, 0]]
        second_frame = handle[handle["Frame_Information"][1, 0]]
        assert first_frame.dtype == np.float32
        assert second_frame.dtype == np.uint16
        first_photons = handle[handle["Photons"][0, 0]]
        second_photons = handle[handle["Photons"][1, 0]]
        assert first_photons.dtype == np.uint8
        assert second_photons.dtype == np.float32
        empty_truth = handle[handle["TrueLocalizations"][0, 0]]
        assert int(empty_truth.attrs["MATLAB_empty"]) == 1
        assert 0 in tuple(int(value) for value in empty_truth[:])

        prob = handle[handle["Prob_dists"][0, 0]]
        assert set(prob) == {
            "Deviation_in_Probability", "Prob_Distributions", "Dscale_store"
        }
        for name in ("Deviation_in_Probability", "Prob_Distributions"):
            group = prob[name]
            assert isinstance(group, h5py.Group)
            assert set(group) == {"data", "ir", "jc"}
            assert "MATLAB_sparse" in group.attrs
        assert isinstance(prob["Dscale_store"], h5py.Dataset)
        untouched = handle[handle["Prob_dists"][1, 0]]
        assert isinstance(untouched["Deviation_in_Probability"], h5py.Dataset)
        assert handle["Step"].shape == (2, 1)
        assert handle["Resolution"].shape == (1, 1)
        assert _matlab_class(handle["String"]) == b"char"


def test_v73_roundtrip_reconstructs_all_continuation_and_passthrough(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.mat"
    expected = _checkpoint()
    save_v73_checkpoint(expected, path)
    actual = load_checkpoint(path)

    assert actual.config == expected.config
    assert actual.analyzed_filename == expected.analyzed_filename
    assert actual.source_filename == "example.mat"
    _assert_result_equal(actual.result, expected.result)
    assert actual.dataset.frames[0].dtype == np.float32
    assert actual.dataset.frames[1].dtype == np.uint16
    assert actual.dataset.photons is not None
    assert actual.dataset.photons[0].dtype == np.uint8
    assert actual.dataset.photons[1].dtype == np.float32
    np.testing.assert_allclose(
        actual.dataset.mat_passthrough["ArbitraryUserVariable"],
        [[1.5, 2.5]],
    )
    assert actual.dataset.mat_passthrough["ArbitraryUserVariable"].dtype == np.float32
    np.testing.assert_array_equal(actual.dataset.came_from_image, [4, 5])
    assert actual.dataset.parameters_to_split is not None
    assert actual.dataset.cuts[0] is not None
    assert np.isnan(actual.result.x_overall[1])
    assert np.isinf(actual.result.x_overall[2])


def test_generated_v73_matches_historical_structural_contract(tmp_path: Path) -> None:
    if not HISTORICAL.exists():
        pytest.skip("historical repository analyzed v7.3 MAT fixture is unavailable")
    generated = tmp_path / "generated.mat"
    save_v73_checkpoint(_checkpoint(), generated)
    with h5py.File(HISTORICAL) as source, h5py.File(generated) as target:
        common_cells = {
            "LocalizationsFinal", "Frame_Information", "TrueLocalizations", "Photons",
            "Final_Loc_Blinking_Corrected", "Final_Frame_Blinking_Corrected",
            "Trajectory_of_Localizations", "Constantf", "Constantf2", "Orderf",
            "Lik", "RelScore", "Numb_of_Loc", "Prob_dists",
        }
        for name in common_cells:
            assert _matlab_class(source[name]) == _matlab_class(target[name]) == b"cell"
            assert source[name].shape[1] == target[name].shape[1] == 1
        source_prob = source[source["Prob_dists"][0, 0]]
        target_prob = target[target["Prob_dists"][0, 0]]
        assert set(source_prob) == set(target_prob)
        for name in ("Deviation_in_Probability", "Prob_Distributions"):
            assert set(source_prob[name]) == set(target_prob[name]) == {"data", "ir", "jc"}
            assert "MATLAB_sparse" in source_prob[name].attrs
            assert "MATLAB_sparse" in target_prob[name].attrs


def _distribution(config: DDCConfig):
    return (
        np.array([0.0, 80.0, np.inf]), np.array([0.75, 0.25]),
        np.zeros((1, 2)), config.resolution, np.ones(config.n_f),
        np.ones((config.n_f, 2)),
    )


def _mcmc_result(n: int, step: int, marker: float, previous: np.ndarray | None = None):
    old = np.empty(0) if previous is None else previous
    history = np.r_[old, marker]
    return SimpleNamespace(
        traj2=np.arange(1, n + 1, dtype=float),
        final_localizations_blinking_corrected=np.full((n, 3), marker),
        final_frame_blinking_corrected=np.full(n, marker),
        likelihood_history=history,
        score_history=history * 0.1,
        number_of_localizations=np.full(history.size, n, dtype=float),
        steps=step,
        constantf=np.full(n, marker),
        orderf=np.arange(1, n + 1, dtype=np.int64),
        probability_cache=ProbabilityCache(
            np.array([[marker]]), np.array([[marker + 1]]), np.array([marker])
        ),
        constant2ft=np.full(n, -marker),
    )


def _small_dataset() -> DDCDataset:
    n = 11
    return DDCDataset(
        [np.column_stack((np.arange(n, dtype=float), np.zeros(n)))],
        [np.arange(1, n + 1, dtype=np.float64)],
        [np.empty((0, 2))],
        photons=[np.ones(n, dtype=np.uint8)],
        addon=np.array([0.0]),
        mat_passthrough={"UserScalar": np.array([[7.0]])},
    )


def test_callback_timing_detachment_and_numerical_rng_invariance() -> None:
    dataset = _small_dataset()
    config = DDCConfig(stepper=1, n_f=2, photon_weighted_correction=0)

    def fake_mcmc(*args, **kwargs):
        marker = float(kwargs["rng"].random())
        return _mcmc_result(11, 1, marker)

    phases: list[str] = []
    observed_steps: list[np.ndarray] = []
    def callback(snapshot, phase):
        phases.append(phase)
        observed_steps.append(snapshot.steps.copy())
        snapshot.states[0].step = 999
        snapshot.working_localizations[0][:] = -999

    rng_without = np.random.default_rng(123)
    rng_with = np.random.default_rng(123)
    with (
        patch("ddc.pipeline.determine_blinking_distribution", return_value=_distribution(config)),
        patch("ddc.pipeline.run_mcmc", side_effect=fake_mcmc),
    ):
        without = run_ddc(dataset, config, rngs=[rng_without])
        with_callback = run_ddc(
            dataset, config, rngs=[rng_with], checkpoint_callback=callback
        )
    assert phases == ["initial", "outer"]
    np.testing.assert_array_equal(observed_steps[0], [0])
    np.testing.assert_array_equal(observed_steps[1], [1])
    _assert_result_equal(without, with_callback)
    assert float(rng_without.random()) == float(rng_with.random())
    assert with_callback.states[0].step == 1
    assert np.all(with_callback.working_localizations[0] != -999)


def test_initial_then_iterative_overwrite_and_real_mcmc_roundtrip(tmp_path: Path) -> None:
    frames = np.concatenate((np.ones(11), [2.0], np.full(11, 4.0)))
    x = np.concatenate((np.arange(11.0) * 10, [0.2], np.arange(11.0) * 10 + 0.5))
    locations = np.column_stack((x, np.zeros(x.size)))
    dataset = DDCDataset(
        [locations], [frames],
        photons=[np.ones(frames.size, dtype=np.uint8)], addon=np.array([0.0])
    )
    config = DDCConfig(
        resolution=10.0, stepper=10, n_f=2,
        photon_weighted_correction=0, timer=30.0
    )
    path = tmp_path / "analysis.mat"
    writer = CheckpointWriter(
        path, dataset, config,
        analyzed_filename="Analyzed_Time15-Nov-2020_fixture.mat"
    )
    formats: list[str] = []
    def observing_writer(snapshot, phase):
        writer(snapshot, phase)
        formats.append("v7.3" if h5py.is_hdf5(path) else "v5")
        if phase == "initial":
            raw = loadmat(path, squeeze_me=False, struct_as_record=False)
            np.testing.assert_array_equal(raw["Step"], [[0]])

    result = run_ddc(
        dataset, config, rngs=[np.random.default_rng(42)],
        checkpoint_callback=observing_writer
    )
    assert formats[0] == "v5" and formats[-1] == "v7.3"
    loaded = load_checkpoint(path)
    _assert_result_equal(loaded.result, result)
    assert loaded.dataset.photons is not None
    assert loaded.dataset.photons[0].dtype == np.uint8
    assert loaded.result.states[0].probability_cache.prob_distributions.size > 0


def test_resume_uses_saved_state_cache_and_never_recomputes_distribution() -> None:
    dataset = _small_dataset()
    config = DDCConfig(stepper=3, n_f=2, photon_weighted_correction=0)
    state = DDCImageState(
        step=1,
        constantf=np.ones(11),
        constant2ft=np.zeros(11),
        orderf=np.arange(1, 12, dtype=np.int64),
        likelihood=np.array([5.0]),
        score=np.array([0.5]),
        number_of_localizations=np.array([11.0]),
        probability_cache=_cache(),
    )
    saved = DDCPipelineResult(
        final_localizations=[np.ones((11, 3))],
        final_frames=[np.ones(11)],
        trajectories=[np.arange(1, 12, dtype=float)],
        states=[state],
        working_localizations=[np.column_stack((dataset.localizations[0], np.zeros(11)))],
        frames=[dataset.frames[0].copy()],
        true_localizations=[np.empty(0)],
        photons=[dataset.photons[0].copy()],
        bins=np.array([0.0, 80.0, np.inf]),
        distribution_for_blink=np.array([0.75, 0.25]),
        resolution=80.0,
        x_overall=np.ones(2),
        m_mat=np.ones((2, 2)),
        diagnostics=(),
    )
    checkpoint = DDCCheckpoint.from_result(
        saved, dataset, config, "partial.mat", phase="outer"
    )
    distribution = Mock()
    observed: dict[str, object] = {}

    def fake_mcmc(*args, **kwargs):
        observed["step"] = args[8]
        observed["history"] = args[12].copy()
        observed["cache"] = args[15]
        return _mcmc_result(11, 3, 6.0, args[12])

    with (
        patch("ddc.pipeline.determine_blinking_distribution", distribution),
        patch("ddc.pipeline.run_mcmc", side_effect=fake_mcmc),
    ):
        resumed = resume_ddc(checkpoint, rngs=[np.random.default_rng(1)])
    distribution.assert_not_called()
    assert observed["step"] == 1
    np.testing.assert_array_equal(observed["history"], [5.0])
    assert observed["cache"] is not checkpoint.result.states[0].probability_cache
    np.testing.assert_allclose(
        observed["cache"].prob_distributions,
        checkpoint.result.states[0].probability_cache.prob_distributions,
        equal_nan=True,
    )
    np.testing.assert_array_equal(resumed.states[0].likelihood, [5.0, 6.0])


def test_completed_checkpoint_resume_is_noop() -> None:
    checkpoint = _checkpoint(completed=True)
    distribution = Mock()
    mcmc = Mock()
    with (
        patch("ddc.pipeline.determine_blinking_distribution", distribution),
        patch("ddc.pipeline.run_mcmc", mcmc),
    ):
        result = resume_ddc(
            checkpoint, rngs=[np.random.default_rng(1), np.random.default_rng(2)]
        )
    distribution.assert_not_called()
    mcmc.assert_not_called()
    _assert_result_equal(result, checkpoint.result)


def test_continuous_equals_interrupted_saved_loaded_and_resumed(tmp_path: Path) -> None:
    dataset = _small_dataset()
    config = DDCConfig(stepper=2, n_f=2, photon_weighted_correction=0)

    def fake_mcmc(*args, **kwargs):
        marker = float(kwargs["rng"].random())
        return _mcmc_result(11, args[8] + 1, marker, args[12])

    continuous_rng = np.random.default_rng(91)
    with (
        patch("ddc.pipeline.determine_blinking_distribution", return_value=_distribution(config)),
        patch("ddc.pipeline.run_mcmc", side_effect=fake_mcmc),
    ):
        continuous = run_ddc(dataset, config, rngs=[continuous_rng])

    path = tmp_path / "interrupted.mat"
    writer = CheckpointWriter(path, dataset, config, analyzed_filename="interrupted.mat")
    interrupted_rng = np.random.default_rng(91)
    class Interrupted(RuntimeError):
        pass
    def interrupt_after_first(snapshot, phase):
        writer(snapshot, phase)
        if phase == "outer":
            raise Interrupted
    with (
        patch("ddc.pipeline.determine_blinking_distribution", return_value=_distribution(config)),
        patch("ddc.pipeline.run_mcmc", side_effect=fake_mcmc),
        pytest.raises(Interrupted),
    ):
        run_ddc(
            dataset, config, rngs=[interrupted_rng],
            checkpoint_callback=interrupt_after_first
        )

    loaded = load_checkpoint(path)
    distribution = Mock()
    with (
        patch("ddc.pipeline.determine_blinking_distribution", distribution),
        patch("ddc.pipeline.run_mcmc", side_effect=fake_mcmc),
    ):
        resumed = resume_ddc(loaded, rngs=[interrupted_rng])
    distribution.assert_not_called()
    _assert_result_equal(continuous, resumed)


def test_atomic_failure_preserves_previous_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.mat"
    path.write_bytes(b"previous-valid-checkpoint")
    with patch("ddc.serialization.hdf5storage.savemat", side_effect=OSError("boom")):
        with pytest.raises(OSError, match="boom"):
            save_v73_checkpoint(_checkpoint(), path)
    assert path.read_bytes() == b"previous-valid-checkpoint"
    assert list(tmp_path.iterdir()) == [path]


def test_no_plotting_cli_or_parallel_dependencies_are_introduced() -> None:
    import ddc.serialization as module
    names = set(module.__dict__)
    assert not ({"matplotlib", "argparse", "click", "typer", "multiprocessing"} & names)


def test_matlab_checkpoint_compatibility_probe_when_available() -> None:
    oracle = Path(__file__).parent / "matlab" / "checkpoint_compatibility_reference.mat"
    if not oracle.exists():
        pytest.skip(
            "MATLAB runtime is unavailable; run "
            "tests/matlab/probe_checkpoint_compatibility.m"
        )
    reference = loadmat(oracle, squeeze_me=True, struct_as_record=False)
    assert bool(reference["checkpoint_loaded"])
    assert bool(reference["finish_runs_state_complete"])
    assert bool(reference["finish_runs_input_state_complete"])
    assert not bool(reference["finish_runs_signature_compatible"])


def test_matlab_finish_runs_signature_probe_when_available() -> None:
    oracle = Path(__file__).parent / "matlab" / "finish_runs_signature_reference.mat"
    if not oracle.exists():
        pytest.skip(
            "MATLAB runtime is unavailable; run "
            "tests/matlab/probe_finish_runs_signature_mismatch.m"
        )
    reference = loadmat(oracle, squeeze_me=True, struct_as_record=False)
    assert int(reference["ddc_mcmc_nargout"]) == 12
    assert int(reference["finish_runs_requested_outputs"]) == 11
    assert bool(reference["finish_runs_omits_traj2"])
    assert str(reference["prob_dists2_actual_output_name"]) == "Orderf"
    assert not bool(reference["constant2ft_captured"])
