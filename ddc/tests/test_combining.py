from itertools import product
from pathlib import Path

import hdf5storage
import numpy as np
import pytest
from scipy.io import loadmat, savemat

from ddc import (DDCDataset, combine_images, load_combined_images,
                 load_corrected_regions, save_combined_images)
from ddc.cli import main
from ddc.io import MatFileError, _mat_cell


def grid_regions(dimensions=3, images=2):
    coordinates = np.array(list(product(*([[0., .5, 1., 1.5, 2.]] * dimensions))))
    # Repeated rows can be separate detections and must not be deduplicated.
    coordinates = np.concatenate((coordinates, coordinates[[0, 0]]))
    frames = np.arange(len(coordinates), dtype=np.uint16)
    intervals = [[(0., 1.), (1., 2.)] for _ in range(dimensions)]
    if dimensions == 2:
        intervals.append([(0., 0.)])
    boxes = list(product(*intervals))
    loc, frame, origin, cuts = [], [], [], ([], [], [])
    for image in range(1, images + 1):
        for box in boxes:
            loc.append(coordinates.copy())
            frame.append(frames.copy())
            origin.append(image)
            for axis in range(3):
                cuts[axis].append(np.array(box[axis]))
    return DDCDataset(loc, frame, came_from_image=np.array(origin), cuts=cuts), coordinates, frames


@pytest.mark.parametrize("dimensions", [2, 3])
def test_partition_keeps_every_boundary_row_once_for_each_image(dimensions):
    regions, coordinates, frames = grid_regions(dimensions)
    original = [x.copy() for x in regions.localizations]
    result = combine_images(regions)
    assert result.n_images == 2
    assert result.came_from_image is None and result.cuts == (None, None, None)
    xyz = np.column_stack((coordinates, np.zeros(len(coordinates)))) if dimensions == 2 else coordinates
    for loc, frame in zip(result.localizations, result.frames):
        order = np.argsort(frame)
        np.testing.assert_array_equal(frame[order], frames)
        np.testing.assert_array_equal(loc[order], xyz)
        assert frame.dtype == np.uint16
    for actual, before in zip(regions.localizations, original):
        np.testing.assert_array_equal(actual, before)


def test_buffer_trimming_and_frame_order():
    loc = np.array([[-1., 0.], [0., 0.], [1., .5], [2., .5], [3., 0.]])
    regions = DDCDataset([loc, loc], [np.array([50, 40, 30, 20, 10])] * 2,
                          came_from_image=np.array([1, 1]),
                          cuts=([np.array([0, 1]), np.array([1, 2])],
                                [np.array([0, 1])] * 2, [np.array([0, 0])] * 2))
    result = combine_images(regions)
    np.testing.assert_array_equal(result.localizations[0][:, :2], loc[1:4])
    np.testing.assert_array_equal(result.frames[0], [40, 30, 20])


def test_duplicate_boxes_and_empty_image():
    regions, coordinates, frames = grid_regions(2, images=1)
    regions.localizations.extend([regions.localizations[-1].copy(), np.empty((0, 3))])
    regions.frames.extend([regions.frames[-1].copy(), np.array([], dtype=np.uint16)])
    regions.came_from_image = np.r_[regions.came_from_image, 1, 2]
    for axis in regions.cuts:
        axis.extend([axis[-1].copy(), np.array([0., 0.])])
    result = combine_images(regions)
    assert len(result.localizations[0]) == len(coordinates)
    assert result.localizations[1].shape == (0, 3)
    assert result.frames[1].shape == (0,)


def test_partition_fixes_source_cross_image_boundary_loss():
    loc = np.array([[0., 0., 0.], [1., 1., 0.], [.5, .5, 0.]])
    regions = DDCDataset([loc, loc], [np.arange(3)] * 2, came_from_image=np.array([1, 2]),
                          cuts=([np.array([0, 1])] * 2, [np.array([0, 1])] * 2,
                                [np.array([0, 0])] * 2))
    safe = combine_images(regions)
    source = combine_images(regions, boundary_mode="matlab")
    assert [len(x) for x in safe.localizations] == [3, 3]
    assert [len(x) for x in source.localizations] == [3, 1]


@pytest.mark.parametrize("ids", [[0, 1], [1, 3], [1, 1.5], [1, np.nan]])
def test_invalid_image_mapping(ids):
    loc = np.zeros((1, 3))
    regions = DDCDataset([loc, loc], [np.array([1])] * 2, came_from_image=np.array(ids),
                          cuts=tuple([np.array([0, 1])] * 2 for _ in range(3)))
    with pytest.raises(ValueError, match="Came_from_image"):
        combine_images(regions)


def test_invalid_corrected_rows_and_metadata():
    regions, _, _ = grid_regions(2, 1)
    regions.frames[0] = regions.frames[0][:-1]
    with pytest.raises(ValueError, match="do not align"):
        combine_images(regions)
    regions, _, _ = grid_regions(2, 1)
    regions.cuts[0][0] = np.array([2, 1])
    with pytest.raises(ValueError, match="lower <= upper"):
        combine_images(regions)
    with pytest.raises(ValueError, match="boundary_mode"):
        combine_images(regions, boundary_mode="unknown")


def analyzed_variables(regions):
    return {"Final_Loc_Blinking_Corrected": _mat_cell(regions.localizations),
            "Final_Frame_Blinking_Corrected": _mat_cell(regions.frames),
            "Came_from_image": regions.came_from_image,
            **{f"cut{i + 1}array": _mat_cell(axis) for i, axis in enumerate(regions.cuts)},
            "Step": np.full(regions.n_images, 100.), "stepper": 100.}


@pytest.mark.parametrize("v73", [False, True])
def test_analyzed_input_and_combined_output_roundtrip(tmp_path, v73):
    regions, _, _ = grid_regions(3)
    path = tmp_path / "analyzed.mat"
    data = analyzed_variables(regions)
    if v73:
        hdf5storage.savemat(path, data, format="7.3", store_python_metadata=False)
    else:
        savemat(path, data)
    loaded = load_corrected_regions(path)
    actual = combine_images(loaded)
    expected = combine_images(regions)
    output = tmp_path / "nested" / "combined.mat"
    save_combined_images(actual, output)
    reloaded = load_combined_images(output)
    for loc, frame, ref_loc, ref_frame in zip(reloaded.localizations, reloaded.frames,
                                             expected.localizations, expected.frames):
        np.testing.assert_array_equal(loc, ref_loc)
        np.testing.assert_array_equal(frame, ref_frame)
        assert frame.dtype == np.uint16
    raw = loadmat(output)
    assert "Final_Loc_Blinking_Corrected" in raw
    assert "LocalizationsFinal" not in raw and "Came_from_image" not in raw
    assert raw["Final_Frame_Blinking_Corrected"][0, 0].shape[1] == 1


def test_completion_requires_explicit_override(tmp_path):
    regions, _, _ = grid_regions(2, 1)
    data = analyzed_variables(regions)
    data["Step"][0] = 25
    path = tmp_path / "partial.mat"
    savemat(path, data)
    with pytest.raises(MatFileError, match="unfinished"):
        load_corrected_regions(path)
    assert load_corrected_regions(path, allow_incomplete=True).n_images == regions.n_images


@pytest.mark.parametrize("raw_count,complete,expected", [(5, True, 5), (0, True, 0),
                                                          (11, True, None), (5, False, None)])
def test_missing_outputs_only_pass_through_confirmed_skipped_regions(tmp_path, raw_count, complete, expected):
    loc = np.zeros((raw_count, 3))
    regions = DDCDataset([loc], [np.arange(raw_count)], came_from_image=np.array([1]),
                          cuts=tuple([np.array([0, 1])] for _ in range(3)))
    data = analyzed_variables(regions)
    data["LocalizationsFinal"] = data["Final_Loc_Blinking_Corrected"]
    data["Frame_Information"] = data["Final_Frame_Blinking_Corrected"]
    data["Final_Loc_Blinking_Corrected"] = np.empty((0, 0), dtype=object)
    data["Final_Frame_Blinking_Corrected"] = np.empty((0, 0), dtype=object)
    data["Step"] = 100 if complete else 1
    path = tmp_path / "skipped.mat"
    savemat(path, data)
    if expected is None:
        with pytest.raises(MatFileError, match="no corrected output"):
            load_corrected_regions(path, allow_incomplete=True)
    else:
        result = load_corrected_regions(path)
        assert len(result.localizations[0]) == expected


def test_real_empty_correction_is_not_replaced_with_raw_data(tmp_path):
    regions = DDCDataset([np.empty((0, 3))], [np.empty(0)], came_from_image=np.array([1]),
                          cuts=tuple([np.array([0, 1])] for _ in range(3)))
    data = analyzed_variables(regions)
    data["LocalizationsFinal"] = _mat_cell([np.ones((4, 3))])
    data["Frame_Information"] = _mat_cell([np.arange(4)])
    path = tmp_path / "empty.mat"
    savemat(path, data)
    result = combine_images(load_corrected_regions(path))
    assert result.localizations[0].shape == (0, 3)
    save_combined_images(result, tmp_path / "out.mat")
    assert load_combined_images(tmp_path / "out.mat").localizations[0].shape == (0, 3)


def test_combine_cli_batch_alias_inspect_and_input_protection(tmp_path, capsys):
    regions, _, _ = grid_regions(2)
    paths = [tmp_path / "first.mat", tmp_path / "second.mat"]
    for p in paths:
        savemat(p, analyzed_variables(regions))
    original = [p.read_bytes() for p in paths]
    destination = tmp_path / "out"
    assert main(["-combine", *map(str, paths), "--output-dir", str(destination), "--quiet"]) == 0
    for p, before in zip(paths, original):
        output = destination / ("Combined_Final_" + p.name)
        assert load_combined_images(output).n_images == 2
        assert main(["inspect", str(output)]) == 0
        assert p.read_bytes() == before
    assert "combined corrected DDC MAT dataset" in capsys.readouterr().out
    assert main(["combine", str(paths[0]), "--output", str(paths[0])]) == 2
    assert paths[0].read_bytes() == original[0]


def test_matlab_combiner_oracle():
    folder = Path(__file__).parent / "matlab"
    source = folder / "combine_reference_input.mat"
    reference = folder / "combine_reference_expected.mat"
    if not reference.exists():
        pytest.skip("run tests/matlab/generate_combine_reference.m")
    result = combine_images(load_corrected_regions(source), boundary_mode="matlab")
    expected = load_combined_images(reference)
    for a, b in zip(result.localizations, expected.localizations):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(result.frames, expected.frames):
        np.testing.assert_array_equal(a, b)
