import numpy as np
import pytest
from scipy.io import loadmat

from ddc import DDCDataset, load_dataset, save_dataset, load_combined_images
from ddc.cli import main


def input_file(path):
    loc = np.column_stack((np.arange(30) * 90., np.zeros(30)))
    save_dataset(DDCDataset([loc], [np.arange(30, dtype=np.uint16)]), path)


def test_det_n_alias_multiple_files_writes_csv_mat_and_plots(tmp_path, capsys):
    paths = [tmp_path / "one.mat", tmp_path / "two.mat"]
    for path in paths:
        input_file(path)
    before = [p.read_bytes() for p in paths]
    assert main(["-det_N", *map(str, paths), "--max-frame", "20", "--gap", "1", "--quiet"]) == 0
    for path, original in zip(paths, before):
        stem = path.with_name("Determine_N_" + path.stem)
        assert stem.with_suffix(".png").read_bytes().startswith(b"\x89PNG")
        assert stem.with_suffix(".csv").read_text().startswith("frame_difference,Z,pair_count")
        assert loadmat(stem.with_suffix(".mat"))["Z"].size == 20
        assert path.read_bytes() == original


def test_split_command_output_works_with_inspect_and_run(tmp_path):
    path = tmp_path / "input.mat"
    frames = np.r_[np.ones(11), 2., np.full(11, 4.)].astype(np.uint16)
    x = np.r_[np.arange(11.) * 10, .2, np.arange(11.) * 10 + .5]
    save_dataset(DDCDataset([np.column_stack((x, np.zeros(x.size)))], [frames]), path)
    assert main(["-split", str(path), "--quiet"]) == 0
    output = tmp_path / "Split_input.mat"
    assert load_dataset(output).localizations[0].shape == (23, 3)
    assert main(["inspect", str(output)]) == 0
    # A small real pipeline run exercises serialization and the new metadata.
    assert main(["run", str(output), "--n-f", "2", "--resolution", "10", "--stepper", "1", "--quiet",
                 "--output", str(tmp_path / "analyzed.mat"), "--seed", "1"]) == 0
    assert main(["-combine", str(tmp_path / "analyzed.mat"), "--quiet"]) == 0
    combined = load_combined_images(tmp_path / "Combined_Final_analyzed.mat")
    assert combined.n_images == 1 and len(combined.localizations[0]) > 0
    assert len(combined.localizations[0]) == len(combined.frames[0])


def test_manual_resolution_command(tmp_path):
    path = tmp_path / "input.mat"
    input_file(path)
    assert main(["inspect-resolution", str(path), "--n-f", "3", "--resolution", "80",
                 "--no-plot", "--output-dir", str(tmp_path / "out"), "--quiet"]) == 0
    assert "Res_Store" in loadmat(tmp_path / "out" / "Inspect_Res_input.mat")


def test_preprocessing_cannot_overwrite_input(tmp_path, capsys):
    path = tmp_path / "input.mat"
    input_file(path)
    original = path.read_bytes()
    assert main(["-det_N", str(path), "--output", str(path)]) == 2
    assert "must not replace" in capsys.readouterr().err
    assert path.read_bytes() == original


@pytest.mark.parametrize("command", ["-det_N", "-det_Res", "-split", "inspect-resolution", "-combine"])
def test_tools_help(command):
    with pytest.raises(SystemExit) as caught:
        main([command, "--help"])
    assert caught.value.code == 0
