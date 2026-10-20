from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest

from ddc.cli import _rngs, build_parser, main
from ddc.io import DDCDataset
from ddc.pipeline import DDCConfig


def _dataset(dtype=np.float64, n=2) -> DDCDataset:
    return DDCDataset(
        [np.column_stack((np.arange(n, dtype=float), np.zeros(n)))],
        [np.arange(1, n + 1, dtype=dtype)],
        [np.empty((0, 2))],
        photons=[np.ones(n, dtype=np.uint8)],
    )


@pytest.mark.parametrize("argv", [["--help"], ["run", "--help"], ["resume", "--help"], ["inspect", "--help"]])
def test_help_routes(argv) -> None:
    with pytest.raises(SystemExit) as caught:
        main(argv)
    assert caught.value.code == 0


def test_invalid_subcommand_and_missing_input_fail() -> None:
    with pytest.raises(SystemExit):
        main(["unknown"])
    with pytest.raises(SystemExit):
        main(["run"])


def test_parser_source_defaults() -> None:
    args = build_parser().parse_args(["run", "input.mat"])
    assert (args.resolution, args.stepper, args.n_f, args.timer) == (80, 100, 200, 720)
    assert not args.no_photon_weighted_correction


def test_run_uses_approved_boundaries_and_all_overrides(tmp_path: Path) -> None:
    dataset = _dataset()
    writer = Mock()
    with (
        patch("ddc.cli.load_dataset", return_value=dataset) as loader,
        patch("ddc.cli.CheckpointWriter", return_value=writer) as writer_type,
        patch("ddc.cli.run_ddc") as runner,
    ):
        code = main([
            "run", str(tmp_path / "input.mat"), "--output", str(tmp_path / "x.mat"),
            "--resolution", "12.5", "--stepper", "7", "--n-f", "3",
            "--timer", "4", "--seed", "19", "--no-photon-weighted-correction",
        ])
    assert code == 0
    loader.assert_called_once()
    config = writer_type.call_args.args[2]
    assert config == DDCConfig(12.5, 7, 3, 0, 4.0)
    assert writer_type.call_args.args[0] == tmp_path / "x.mat"
    assert runner.call_args.kwargs["checkpoint_callback"] is writer
    assert len(runner.call_args.kwargs["rngs"]) == 1


def test_default_output_is_analyzed_name_in_input_directory(tmp_path: Path) -> None:
    dataset = _dataset()
    with (
        patch("ddc.cli.load_dataset", return_value=dataset),
        patch("ddc.cli.make_analyzed_filename", return_value="Analyzed_example.mat"),
        patch("ddc.cli.CheckpointWriter") as writer,
        patch("ddc.cli.run_ddc"),
    ):
        assert main(["run", str(tmp_path / "example.mat")]) == 0
    assert writer.call_args.args[0] == tmp_path / "Analyzed_example.mat"


def test_seed_streams_are_independent_and_reproducible() -> None:
    first = _rngs(123, 2)
    second = _rngs(123, 2)
    np.testing.assert_array_equal(first[0].random(5), second[0].random(5))
    assert not np.array_equal(_rngs(123, 2)[0].random(5), _rngs(124, 2)[0].random(5))
    assert not np.array_equal(_rngs(123, 2)[0].random(5), _rngs(123, 2)[1].random(5))


def test_known_source_failure_is_concise_and_nonzero(capsys) -> None:
    with patch("ddc.cli.load_dataset", side_effect=TypeError("DDC-SOURCE-BUG-002: integer pdist")):
        assert main(["run", "input.mat"]) == 2
    assert "DDC-SOURCE-BUG-002" in capsys.readouterr().err


def test_debug_reraises_known_source_failure() -> None:
    with patch("ddc.cli.load_dataset", side_effect=TypeError("DDC-SOURCE-BUG-002")):
        with pytest.raises(TypeError, match="BUG-002"):
            main(["--debug", "run", "input.mat"])


def test_run_converts_integer_input_dtypes_for_operational_cli(tmp_path: Path) -> None:
    dataset = _dataset(np.uint16, n=11)
    with (
        patch("ddc.cli.load_dataset", return_value=dataset),
        patch("ddc.cli.CheckpointWriter"),
        patch("ddc.cli.run_ddc"),
    ):
        assert main(["run", str(tmp_path / "input.mat")]) == 0
    assert dataset.frames[0].dtype == np.float64
    assert dataset.photons[0].dtype == np.float64


def test_preserve_source_dtypes_retains_known_failure(tmp_path: Path, capsys) -> None:
    dataset = _dataset(np.uint16, n=11)
    with patch("ddc.cli.load_dataset", return_value=dataset):
        code = main([
            "run",
            str(tmp_path / "input.mat"),
            "--stepper",
            "1",
            "--n-f",
            "2",
            "--preserve-source-dtypes",
        ])
    assert code == 2
    assert "DDC-SOURCE-BUG-002" in capsys.readouterr().err


def test_resume_uses_corrected_api_never_source_diagnostic(tmp_path: Path) -> None:
    checkpoint = Mock()
    checkpoint.dataset = _dataset()
    checkpoint.config = DDCConfig(stepper=1)
    checkpoint.analyzed_filename = "a.mat"
    checkpoint.source_metadata = {}
    checkpoint.source_filename = "input.mat"
    writer = Mock()
    writer.last_checkpoint = Mock()
    with (
        patch("ddc.cli.load_checkpoint", return_value=checkpoint) as loader,
        patch("ddc.cli.CheckpointWriter", return_value=writer),
        patch("ddc.cli.resume_ddc", return_value=Mock()) as corrected,
        patch("ddc.serialization.resume_finish_runs_source") as source_resume,
    ):
        assert main(["resume", str(tmp_path / "a.mat"), "--seed", "5"]) == 0
    loader.assert_called_once()
    corrected.assert_called_once()
    source_resume.assert_not_called()


def test_inspect_raw_dataset_is_nonmutating(tmp_path: Path, capsys) -> None:
    path = tmp_path / "input.mat"
    path.write_bytes(b"unchanged")
    before = path.read_bytes()
    with (
        patch("ddc.cli.load_checkpoint", side_effect=ValueError("not checkpoint")),
        patch("ddc.cli.load_dataset", return_value=_dataset()),
    ):
        assert main(["inspect", str(path)]) == 0
    output = capsys.readouterr().out
    assert "raw DDC MAT dataset" in output and "frame_dtypes" in output
    assert path.read_bytes() == before


def test_inspect_checkpoint_reports_steps(tmp_path: Path, capsys) -> None:
    checkpoint = Mock()
    checkpoint.phase = "outer"
    checkpoint.dataset = _dataset()
    checkpoint.config = DDCConfig(stepper=3)
    checkpoint.result.steps = np.array([3])
    with patch("ddc.cli.load_checkpoint", return_value=checkpoint), patch("ddc.cli.h5py.is_hdf5", return_value=True):
        assert main(["inspect", str(tmp_path / "checkpoint.mat")]) == 0
    output = capsys.readouterr().out
    assert "Python-generated DDC checkpoint" in output
    assert "Step: [3]" in output and "complete: True" in output


def test_python_module_route_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ddc", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "run" in result.stdout and "resume" in result.stdout and "inspect" in result.stdout


def test_cli_has_no_plotting_or_parallel_dependencies() -> None:
    import ddc.cli as module
    assert not ({"matplotlib", "multiprocessing", "click", "typer"} & set(module.__dict__))
