"""Frozen end-to-end acceptance tests for the unoptimized Stage-A baseline."""

from __future__ import annotations

import ast
from pathlib import Path
import re
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from scipy.io import savemat

import ddc
from ddc.io import load_dataset
from ddc.mcmc import run_mcmc as real_run_mcmc
from ddc.pipeline import DDCConfig, run_ddc
from ddc.serialization import CheckpointWriter, load_checkpoint, resume_ddc


def _cell(value: np.ndarray) -> np.ndarray:
    cell = np.empty((1, 1), dtype=object)
    cell[0, 0] = value
    return cell


def _write_supported_mat(path: Path, *, frame_dtype=np.float64, photon_dtype=np.float64) -> None:
    frames = np.concatenate((np.ones(11), [2.0], np.full(11, 4.0))).astype(frame_dtype)
    x = np.concatenate((np.arange(11.0) * 10, [0.2], np.arange(11.0) * 10 + 0.5))
    locations = np.column_stack((x, np.zeros(x.size)))
    savemat(path, {
        "LocalizationsFinal": _cell(locations),
        "Frame_Information": _cell(frames),
        "TrueLocalizations": _cell(np.empty((0, 2))),
        "Photons": _cell(np.ones(frames.size, dtype=photon_dtype)),
        "addonarray": np.zeros((1, 1)),
    })


CONFIG = DDCConfig(
    resolution=10.0, stepper=10, n_f=2,
    photon_weighted_correction=0, timer=30.0,
)


def test_supported_disk_run_checkpoint_reload_real_mcmc(tmp_path: Path) -> None:
    source = tmp_path / "supported.mat"
    output = tmp_path / "analyzed.mat"
    _write_supported_mat(source)
    dataset = load_dataset(source)
    writer = CheckpointWriter(output, dataset, CONFIG)
    result = run_ddc(
        dataset, CONFIG, rngs=[np.random.default_rng(4)],
        checkpoint_callback=writer,
    )
    assert result.steps.tolist() == [10]
    assert result.final_localizations[0] is not None
    assert result.states[0].probability_cache.prob_distributions.size
    assert h5py.is_hdf5(output)
    reloaded = load_checkpoint(output)
    assert reloaded.result.steps.tolist() == [10]
    np.testing.assert_allclose(
        reloaded.result.states[0].likelihood,
        result.states[0].likelihood,
    )


def test_interrupted_initial_checkpoint_resume_equals_continuous(tmp_path: Path) -> None:
    source = tmp_path / "supported.mat"
    checkpoint_path = tmp_path / "interrupted.mat"
    _write_supported_mat(source)
    dataset = load_dataset(source)
    continuous = run_ddc(dataset, CONFIG, rngs=[np.random.default_rng(9)])

    writer = CheckpointWriter(checkpoint_path, dataset, CONFIG)
    class Interrupted(RuntimeError):
        pass
    def stop_after_initial(result, phase):
        writer(result, phase)
        if phase == "initial":
            raise Interrupted
    with pytest.raises(Interrupted):
        run_ddc(
            dataset, CONFIG, rngs=[np.random.default_rng(9)],
            checkpoint_callback=stop_after_initial,
        )
    loaded = load_checkpoint(checkpoint_path)
    resumed = resume_ddc(loaded, rngs=[np.random.default_rng(9)])
    np.testing.assert_allclose(
        resumed.states[0].likelihood, continuous.states[0].likelihood
    )
    np.testing.assert_allclose(
        resumed.final_localizations[0], continuous.final_localizations[0]
    )
    np.testing.assert_array_equal(
        resumed.states[0].constant2ft, continuous.states[0].constant2ft
    )


class _OneIterationClock:
    """Give each real run_mcmc call exactly one timed outer iteration."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> float:
        value = (0.0, 0.0, 1.0)[self.calls % 3]
        self.calls += 1
        return value


def _real_mcmc_with_clock(controlled_clock: _OneIterationClock):
    """Forward the pipeline call to real run_mcmc with its injectable clock."""

    def invoke(*args, **kwargs):
        return real_run_mcmc(*args, **kwargs, clock=controlled_clock)

    return invoke


def test_resume_from_materialized_noninitial_mcmc_state(tmp_path: Path) -> None:
    source = tmp_path / "supported.mat"
    checkpoint_path = tmp_path / "noninitial.mat"
    _write_supported_mat(source)
    dataset = load_dataset(source)
    config = DDCConfig(
        resolution=10.0, stepper=12, n_f=2,
        photon_weighted_correction=0, timer=0.5,
    )

    continuous_rng = np.random.default_rng(21)
    continuous_clock = _OneIterationClock()
    with patch(
        "ddc.pipeline.run_mcmc", new=_real_mcmc_with_clock(continuous_clock)
    ):
        continuous = run_ddc(dataset, config, rngs=[continuous_rng])
    assert continuous_clock.calls > 0

    interrupted_rng = np.random.default_rng(21)
    writer = CheckpointWriter(checkpoint_path, dataset, config)
    captured_noninitial = False

    class InterruptedAfterMCMC(RuntimeError):
        pass

    def stop_after_noninitial(result, phase):
        nonlocal captured_noninitial
        writer(result, phase)
        state = result.states[0]
        if phase == "outer" and (
            state.step > 0
            and state.likelihood.size > 0
            and state.constantf.size > 0
            and state.constant2ft.size > 0
            and state.orderf.size > 0
            and state.probability_cache.prob_distributions.size > 0
        ):
            captured_noninitial = True
            raise InterruptedAfterMCMC

    interrupted_clock = _OneIterationClock()
    with patch(
        "ddc.pipeline.run_mcmc", new=_real_mcmc_with_clock(interrupted_clock)
    ):
        with pytest.raises(InterruptedAfterMCMC):
            run_ddc(
                dataset, config, rngs=[interrupted_rng],
                checkpoint_callback=stop_after_noninitial,
            )
        assert captured_noninitial
        loaded = load_checkpoint(checkpoint_path)
        assert loaded.result.states[0].step > 0
        assert loaded.result.states[0].likelihood.size > 0
        assert loaded.result.states[0].probability_cache.prob_distributions.size > 0
        resumed = resume_ddc(loaded, rngs=[interrupted_rng])
    assert interrupted_clock.calls > 0

    np.testing.assert_allclose(
        resumed.states[0].likelihood, continuous.states[0].likelihood
    )
    np.testing.assert_allclose(
        resumed.final_localizations[0], continuous.final_localizations[0]
    )
    np.testing.assert_array_equal(
        resumed.states[0].constantf, continuous.states[0].constantf
    )
    np.testing.assert_array_equal(
        resumed.states[0].constant2ft, continuous.states[0].constant2ft
    )
    np.testing.assert_array_equal(
        resumed.states[0].orderf, continuous.states[0].orderf
    )
    np.testing.assert_allclose(
        resumed.states[0].probability_cache.prob_distributions,
        continuous.states[0].probability_cache.prob_distributions,
    )


def test_repository_style_uint16_reaches_documented_bug002(tmp_path: Path) -> None:
    source = tmp_path / "uint16.mat"
    _write_supported_mat(source, frame_dtype=np.uint16)
    with pytest.raises(TypeError, match="DDC-SOURCE-BUG-002"):
        run_ddc(
            load_dataset(source), CONFIG,
            rngs=[np.random.default_rng(1)],
        )


def test_integer_photons_remain_integer_at_io_boundary(tmp_path: Path) -> None:
    source = tmp_path / "photons.mat"
    _write_supported_mat(source, photon_dtype=np.uint8)
    dataset = load_dataset(source)
    assert dataset.photons is not None
    assert dataset.photons[0].dtype == np.uint8
    result = run_ddc(dataset, CONFIG, rngs=[np.random.default_rng(4)])
    assert result.photons[0].dtype == np.uint8


def test_public_stage_a_boundaries_import() -> None:
    for name in (
        "DDCDataset", "load_dataset", "DDCConfig", "DDCPipelineResult",
        "run_ddc", "DDCCheckpoint", "CheckpointWriter", "load_checkpoint",
        "resume_ddc", "resume_finish_runs_source", "make_analyzed_filename",
        "DDCSourceBug007",
    ):
        assert getattr(ddc, name) is not None


def test_every_matlab_oracle_reference_exists() -> None:
    tests_directory = Path(__file__).parent
    references: set[str] = set()
    for test_path in tests_directory.glob("test_*.py"):
        tree = ast.parse(test_path.read_text(), filename=str(test_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                references.update(
                    re.findall(
                        r"tests/matlab/([A-Za-z0-9_.-]+\.m)", node.value
                    )
                )
    missing = sorted(
        name
        for name in references
        if not (tests_directory / "matlab" / name).is_file()
    )
    assert references
    assert missing == []
