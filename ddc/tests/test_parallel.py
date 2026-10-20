"""Real spawned-process coverage for image scheduling and checkpoint boundaries."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields, is_dataclass
from multiprocessing import get_context
import os
from unittest.mock import patch

import numpy as np
import pytest
from scipy import sparse

import ddc.pipeline as pipeline
from ddc.cli import _rngs, build_parser, main
from ddc.distributions import determine_blinking_distribution
from ddc.io import DDCDataset
from ddc.mcmc import run_mcmc as real_run_mcmc
from ddc.pipeline import DDCConfig, run_ddc
from ddc.serialization import CheckpointWriter, load_checkpoint, resume_ddc


def dataset():
    frame = np.r_[np.ones(11), 2.0, np.full(11, 4.0)]
    x = np.r_[np.arange(11.0) * 10, 0.2, np.arange(11.0) * 10 + 0.5]
    locations = np.column_stack((x, np.zeros(x.size)))
    return DDCDataset(
        [locations.copy(), locations * 1.05, locations[[0, 10, 11, 12, 22]].copy()],
        [frame.copy(), frame.copy(), frame[[0, 10, 11, 12, 22]].copy()],
        [np.empty((0, 2)) for _ in range(3)],
        photons=[np.linspace(1, 3, n) for n in (23, 23, 5)],
    )


def config(weighted=1):
    # Infinite timer fixes the amount of work regardless of process scheduling.
    return DDCConfig(resolution=10, stepper=12, n_f=2,
                     photon_weighted_correction=weighted, timer=float("inf"))


def test_parallel_shared_distribution_matches_every_serial_output():
    data = dataset()
    # Include different coordinate scales and a short image so the test
    # detects reordering or averaging with the wrong scientific weights.
    expected = determine_blinking_distribution(data.localizations, data.frames, 2, 10)
    actual = determine_blinking_distribution(data.localizations, data.frames, 2, 10, workers=2)
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right, strict=True)


def assert_equal(a, b):
    if is_dataclass(a):
        assert type(a) is type(b)
        for field in fields(a):
            assert_equal(getattr(a, field.name), getattr(b, field.name))
    elif sparse.issparse(a):
        assert_equal(a.toarray(), b.toarray())
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b, strict=True)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b, strict=True):
            assert_equal(left, right)
    else:
        assert a == b


@pytest.mark.parametrize("weighted", [0, 1])
def test_parallel_matches_every_field_and_rng_and_saves_in_parent(weighted):
    data, settings = dataset(), config(weighted)
    sequential_rngs, parallel_rngs = _rngs(123, 3), _rngs(123, 3)
    expected = run_ddc(data, settings, rngs=sequential_rngs)
    parent_pid = os.getpid()
    phases = []

    def callback(result, phase):
        assert os.getpid() == parent_pid
        phases.append(phase)
        if phase == "outer":
            assert_equal(result, expected)

    with (
        patch.object(pipeline, "determine_blinking_distribution",
                     wraps=pipeline.determine_blinking_distribution) as distribution,
        # A parent-side image call would fail; spawned interpreters load real code.
        patch.object(pipeline, "_advance_image",
                     side_effect=AssertionError("MCMC executed in parent")),
    ):
        actual = run_ddc(data, settings, workers=2, rngs=parallel_rngs,
                         checkpoint_callback=callback)
    distribution.assert_called_once()
    assert phases == ["initial", "outer"]
    assert_equal(actual, expected)
    assert all(value is not None for value in actual.final_localizations[:2])
    for sequential, parallel in zip(sequential_rngs, parallel_rngs):
        assert_equal(sequential.random(20), parallel.random(20))


def _one_iteration(*args, **kwargs):
    # Force deterministic chunk boundaries, avoiding timing-dependent tests.
    args = list(args)
    args[16] = min(args[16], args[8] + 1)
    return real_run_mcmc(*args, **kwargs)


def _initialize_short_chunks(shared):
    from ddc._parallel import _initialize_worker
    _initialize_worker(shared)
    pipeline.run_mcmc = _one_iteration


def _short_chunk_pool(workers, shared):
    return ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn"),
        initializer=_initialize_short_chunks, initargs=(shared,),
    )


def test_multiple_outer_passes_preserve_rng_and_image_state():
    data, settings = dataset(), config()
    seq_rngs, par_rngs = _rngs(18, 3), _rngs(18, 3)
    sequential, parallel = [], []
    with patch.object(pipeline, "run_mcmc", _one_iteration):
        expected = run_ddc(data, settings, rngs=seq_rngs,
                          checkpoint_callback=lambda r, p: sequential.append((r, p)))
    with patch("ddc._parallel.create_pool", _short_chunk_pool):
        actual = run_ddc(data, settings, workers=2, rngs=par_rngs,
                        checkpoint_callback=lambda r, p: parallel.append((r, p)))
    assert len(parallel) > 2
    assert_equal(parallel, sequential)
    assert_equal(actual, expected)
    for left, right in zip(seq_rngs, par_rngs):
        assert_equal(left.random(20), right.random(20))


def test_disk_checkpoint_resume_skips_distribution_and_matches_sequential(tmp_path):
    data, settings = dataset(), config()
    path = tmp_path / "partial.mat"
    writer = CheckpointWriter(path, data, settings)

    class Interrupted(Exception):
        pass

    def stop_after_pass(result, phase):
        writer(result, phase)
        if phase == "outer":
            raise Interrupted

    with patch("ddc._parallel.create_pool", _short_chunk_pool):
        with pytest.raises(Interrupted):
            run_ddc(data, settings, workers=2, rngs=_rngs(19, 3),
                    checkpoint_callback=stop_after_pass)
    loaded = load_checkpoint(path)
    assert loaded.result.steps.tolist() == [10, 10, 12]
    expected = resume_ddc(loaded, rngs=_rngs(20, 3))
    resumed_path = tmp_path / "resumed.mat"
    with patch.object(pipeline, "determine_blinking_distribution",
                      side_effect=AssertionError("distribution recalculated")):
        actual = resume_ddc(
            loaded, workers=2, rngs=_rngs(20, 3),
            checkpoint_callback=CheckpointWriter(resumed_path, data, settings),
        )
    assert_equal(actual, expected)
    assert_equal(load_checkpoint(resumed_path).result, actual)


def _fail_mcmc(*args, **kwargs):
    raise RuntimeError("injected worker failure")


def _initialize_failing_worker(shared):
    from ddc._parallel import _initialize_worker
    _initialize_worker(shared)
    pipeline.run_mcmc = _fail_mcmc


def _failing_pool(workers, shared):
    return ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn"),
        initializer=_initialize_failing_worker, initargs=(shared,),
    )


def test_worker_failure_preserves_initial_checkpoint(tmp_path):
    data, settings = dataset(), config()
    path = tmp_path / "recoverable.mat"
    writer = CheckpointWriter(path, data, settings)
    phases = []

    def callback(result, phase):
        writer(result, phase)
        phases.append(phase)

    with patch("ddc._parallel.create_pool", _failing_pool), pytest.raises(
        RuntimeError, match="injected worker failure"
    ):
        run_ddc(data, settings, workers=2, rngs=_rngs(123, 3),
                checkpoint_callback=callback)
    assert phases == ["initial"]
    saved = load_checkpoint(path)
    assert saved.phase == "initial"
    assert saved.result.steps.tolist() == [0, 0, 0]


@pytest.mark.parametrize("workers", [0, -1, 1.5, True, "2"])
def test_invalid_worker_count_rejected_before_distribution(workers):
    with patch.object(pipeline, "determine_blinking_distribution") as distribution:
        with pytest.raises(ValueError, match="positive integer"):
            run_ddc(dataset(), config(), workers=workers)
    distribution.assert_not_called()


def test_single_eligible_image_and_completed_resume_do_not_start_pool():
    data = dataset()
    data.localizations[1] = data.localizations[1][[0, 10, 11, 12, 22]]
    data.frames[1] = data.frames[1][[0, 10, 11, 12, 22]]
    data.photons[1] = data.photons[1][:5]
    with patch("ddc._parallel.create_pool", side_effect=AssertionError("pool started")):
        result = run_ddc(data, config(), workers=8, rngs=_rngs(10, 3))
        completed = run_ddc(data, config(), workers=8, resume_from=result)
    assert_equal(completed, result)


def test_parallel_rejects_shared_generator():
    generator = np.random.default_rng(1)
    with pytest.raises(ValueError, match="independent"):
        run_ddc(dataset(), config(), workers=2, rngs=[generator] * 3)


@pytest.mark.parametrize("command", ["run", "resume"])
def test_cli_worker_flag(command):
    parser = build_parser()
    assert parser.parse_args([command, "input.mat"]).workers == 1
    assert parser.parse_args([command, "input.mat", "--workers", "4"]).workers == 4
    for value in ["0", "-2", "1.5"]:
        with pytest.raises(SystemExit):
            parser.parse_args([command, "input.mat", "--workers", value])


def test_cli_run_and_resume_forward_workers(tmp_path, capsys):
    from scipy.io import savemat

    data = dataset()
    cells = lambda values: np.array(values, dtype=object).reshape(1, -1)
    source, output = tmp_path / "input.mat", tmp_path / "output.mat"
    savemat(source, {"LocalizationsFinal": cells(data.localizations),
                     "Frame_Information": cells(data.frames),
                     "Photons": cells(data.photons)})
    assert main(["run", str(source), "--output", str(output), "--workers", "2",
                 "--resolution", "10", "--n-f", "2", "--stepper", "12",
                 "--timer", "inf", "--seed", "5"]) == 0
    assert load_checkpoint(output).result.steps.tolist() == [12, 12, 12]
    messages = capsys.readouterr().err
    assert "Starting 3 ROIs" in messages
    assert "ROI 1/3: DONE" in messages
    assert "ROI 2/3: DONE" in messages
    assert "ROI 3/3: skipped" in messages
    assert "Checkpoint saved" in messages
    assert main(["resume", str(output), "--workers", "2", "--seed", "5"]) == 0
    messages = capsys.readouterr().err
    assert "Resuming 3 ROIs" in messages
    assert "already complete" in messages
    assert main(["resume", str(output), "--quiet"]) == 0
    assert capsys.readouterr().err == ""
