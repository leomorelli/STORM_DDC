from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat
from scipy.spatial.distance import pdist

from ddc import (DDCDataset, determine_n, determine_resolution, inspect_resolution,
                 load_dataset, save_dataset, split_images)
from ddc.tools import _axis_cuts, _mask


def fixture():
    rng = np.random.default_rng(6)
    centers = rng.uniform(0, 4000, (45, 2))
    loc = np.repeat(centers, 4, axis=0) + rng.normal(0, 45, (180, 2))
    frame = np.repeat(np.arange(45) * 20, 4) + np.tile(np.arange(4), 45)
    return DDCDataset([loc, loc * 1.1], [frame.astype(np.uint16), frame.astype(np.uint16)],
                      photons=[np.arange(180, dtype=np.uint16)] * 2)


@pytest.mark.parametrize("block", [1, 13, 512])
def test_n_matches_independent_full_pdist(block):
    dataset = fixture()
    result = determine_n(dataset, max_frame=81, gap=10, block_size=block)
    expected = []
    support = []
    for lag in np.arange(1, 82, 10):
        distances = np.concatenate([pdist(loc)[pdist(frame.astype(float)[:, None]) == lag]
                                    for loc, frame in zip(dataset.localizations, dataset.frames)])
        count = np.histogram(distances, result.bins)[0]
        expected.append(np.cumsum(count / len(distances)) if len(distances)
                        else np.full(count.shape, np.nan))
        support.append(len(distances))
    np.testing.assert_allclose(result.cdf, expected, rtol=1e-14)
    np.testing.assert_allclose(result.z, np.sum(np.abs(expected - expected[0]), axis=1), rtol=1e-14)
    np.testing.assert_array_equal(result.pair_counts, support)
    assert dataset.frames[0].dtype == np.uint16


def test_n_no_pairs_and_image_cutoff_are_explicit():
    dataset = DDCDataset([np.zeros((3, 2))], [np.array([0, 2, 4])])
    result = determine_n(dataset, gap=1, max_frame=5)
    assert np.isnan(result.z).all()
    np.testing.assert_array_equal(result.pair_counts, [0, 2, 0, 1, 0])
    with pytest.raises(ValueError, match="split"):
        determine_n(dataset, max_localizations=3)
    assert determine_n(dataset, max_localizations=None).image_indices.tolist() == [0]


def test_n_does_not_round_fractional_frame_lags():
    dataset = DDCDataset([np.zeros((3, 2))], [np.array([0, .9, 2])])
    result = determine_n(dataset, max_frame=2, gap=1)
    np.testing.assert_array_equal(result.pair_counts, [0, 1])


@pytest.mark.parametrize("block", [17, 512])
def test_manual_resolution_matches_source_histograms_and_truth(block):
    dataset = fixture()
    dataset.true_localizations = [x[::4].copy() for x in dataset.localizations]
    result = inspect_resolution(dataset, n_f=10, resolution=80, image_fraction=.5, block_size=block)
    maximum = max(pdist(x).max() for x in dataset.localizations)
    np.testing.assert_array_equal(result.bins, np.r_[np.arange(0, maximum - 80 + 1e-9, 80), np.inf])
    loc, frame = dataset.localizations[0], dataset.frames[0].astype(float)
    for selected, expected in [(np.ones(len(loc), bool), result.probabilities[0]),
                               (np.arange(len(loc)) % 4 == 0, result.true_probabilities[0])]:
        distances = pdist(loc[selected])
        dt = pdist(frame[selected, None])
        count = np.histogram(distances[dt > 10], result.bins)[0]
        np.testing.assert_allclose(expected, count / count.sum())


def test_auto_resolution_streaming_and_thresholds():
    dataset = fixture()
    result = determine_resolution(dataset, n_f=10, block_size=31)
    other = determine_resolution(dataset, n_f=10, block_size=512)
    assert result.resolution == other.resolution
    for trial, second in zip(result.trials, other.trials):
        np.testing.assert_array_equal(trial.blink_probability, second.blink_probability)
        short, long = [], []
        for loc, frame in zip(dataset.localizations, dataset.frames):
            distances, dt = pdist(loc), pdist(frame.astype(float)[:, None])
            for mask, rows in [(dt < 10, short), (dt > 10, long)]:
                counts = np.histogram(distances[mask], trial.bins)[0]
                rows.append(counts / counts.sum())
        np.testing.assert_allclose(trial.short_probability, np.mean(short, axis=0))
        np.testing.assert_allclose(trial.long_probability, np.mean(long, axis=0))
        np.testing.assert_allclose(trial.blink_probability.sum(), 1)
    assert result.resolution in range(10, 161, 10)


def test_auto_resolution_refuses_undefined_source_calculations():
    with pytest.raises(ValueError, match="tail support"):
        determine_resolution(DDCDataset([np.arange(40).reshape(20, 2)], [np.arange(20)]), n_f=3)
    dataset = fixture()
    dataset.frames = [np.zeros(180)] * 2
    with pytest.raises(ValueError, match="short- and long-lag"):
        determine_resolution(dataset)


@pytest.mark.parametrize("n", [1, 2, 8])
def test_split_small_and_equal_images_are_single_and_preserve_rows(n):
    loc = np.column_stack((np.arange(n), -np.arange(n)))
    dataset = DDCDataset([loc], [np.arange(n, dtype=np.uint16)], true_localizations=[loc[:1]])
    split = split_images(dataset, min_localizations=8)
    assert split.n_images == 1
    np.testing.assert_array_equal(split.localizations[0][:, :2], loc)
    assert split.localizations[0].shape == (n, 3)
    np.testing.assert_array_equal(split.photons[0], np.ones(n))
    assert split.true_localizations[0].shape == (1, 3)
    assert dataset.localizations[0].shape == (n, 2)


def test_split_hazen_quantiles():
    loc = np.column_stack((np.arange(1, 5), np.arange(1, 5), np.zeros(4)))
    cuts = _axis_cuts(loc, [.25, .5, 1])
    np.testing.assert_array_equal(cuts[0], [[1, 1.5], [1.5, 2.5], [2.5, 3.5], [3.5, 4]])


def test_split_seed_metadata_and_source_buffer_loops(tmp_path):
    dataset = fixture()
    result = split_images(dataset, min_localizations=60, search_iterations=15, rng=np.random.default_rng(4))
    again = split_images(dataset, min_localizations=60, search_iterations=15, rng=np.random.default_rng(4))
    assert result.n_images > dataset.n_images
    for i, (region, repeat) in enumerate(zip(result.localizations, again.localizations)):
        np.testing.assert_array_equal(region, repeat)
        original = int(result.came_from_image[i]) - 1
        loc = np.column_stack((dataset.localizations[original], np.zeros(180)))
        cuts = np.array([axis[i] for axis in result.cuts])
        # Independently run the literal MATLAB buffer-growth/shrink loops.
        addon, keep = 250, np.zeros(180, bool)
        while keep.sum() < 60:
            addon += 10
            keep = _mask(loc, cuts, addon)
        while keep.sum() > 60:
            addon -= 10
            keep = _mask(loc, cuts, addon)
            if addon < 150:
                break
        assert result.addon[i] == addon
        np.testing.assert_array_equal(region, loc[keep])
        np.testing.assert_array_equal(result.frames[i], dataset.frames[original][keep])
        np.testing.assert_array_equal(result.photons[i], dataset.photons[original][keep])
    path = tmp_path / "nested" / "split.mat"
    save_dataset(result, path)
    loaded = load_dataset(path)
    np.testing.assert_array_equal(loaded.came_from_image, result.came_from_image)
    for a, b in zip(loaded.localizations, result.localizations):
        np.testing.assert_array_equal(a, b)
    assert loaded.frames[0].dtype == np.uint16
    assert loaded.photons[0].dtype == np.uint16


def test_split_3d_and_empty_images():
    dataset = fixture()
    dataset.localizations = [np.column_stack((dataset.localizations[0], np.arange(180) * 40)),
                             np.empty((0, 3))]
    dataset.frames[1] = np.array([])
    dataset.photons[1] = np.array([])
    result = split_images(dataset, min_localizations=60, search_iterations=8, rng=np.random.default_rng(5))
    assert any(p[2] != 1 for p in result.parameters_to_split[:-1])
    assert result.localizations[-1].shape == (0, 3)
    assert result.came_from_image[-1] == 2


@pytest.mark.parametrize("operation,kwargs", [
    (determine_n, {"gap": 0}), (determine_n, {"block_size": 0}),
    (inspect_resolution, {"image_fraction": 1.1}),
    (determine_resolution, {"step": 0}), (split_images, {"min_localizations": 0}),
])
def test_invalid_configuration(operation, kwargs):
    with pytest.raises(ValueError):
        operation(fixture(), **kwargs)


def test_matlab_preprocessing_oracle():
    path = Path(__file__).parent / "matlab" / "preprocessing_reference.mat"
    if not path.exists():
        pytest.skip("run tests/matlab/generate_preprocessing_reference.m to generate the MATLAB oracle")
    data = loadmat(path, squeeze_me=False)
    dataset = load_dataset(path)
    n = determine_n(dataset, max_frame=81, gap=10)
    np.testing.assert_allclose(n.cdf, data["cdf_expected"], rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(n.z, data["z_expected"].ravel(), rtol=2e-14, atol=2e-15)
    diagnostic = inspect_resolution(dataset, n_f=10, resolution=80, image_fraction=1)
    np.testing.assert_allclose(diagnostic.probabilities, data["res_expected"], rtol=2e-14)
    auto = determine_resolution(dataset, n_f=10)
    assert auto.resolution == data["resolution_expected"].item()
    np.testing.assert_allclose(auto.trials[-1].blink_probability, data["blink_expected"].ravel(),
                               rtol=2e-13, atol=2e-15)


def test_matlab_split_oracle_with_replayed_random_draws():
    folder = Path(__file__).parent / "matlab"
    path = folder / "split_preprocessing_reference.mat"
    if not path.exists():
        pytest.skip("run generate_preprocessing_reference.m for the split oracle")
    expected = load_dataset(path)
    original = load_dataset(folder / "preprocessing_reference.mat")
    original.true_localizations = [loc[::4].copy() for loc in original.localizations]
    original.photons = [np.arange(1, 181)] * 2
    values = iter(loadmat(path)["random_values"].ravel())

    class Replay:
        def random(self, size=None):
            return next(values) if size is None else np.array([next(values) for _ in range(size)])

    result = split_images(original, min_localizations=60, search_iterations=15, rng=Replay())
    assert result.n_images == expected.n_images
    np.testing.assert_array_equal(result.addon, expected.addon)
    np.testing.assert_array_equal(result.came_from_image, expected.came_from_image)
    for actual, reference in zip(result.parameters_to_split, expected.parameters_to_split):
        np.testing.assert_array_equal(actual, reference)
    for a, b in zip(result.cuts, expected.cuts):
        np.testing.assert_allclose(a, b, rtol=2e-14, atol=2e-12)
    for name in ("localizations", "frames", "photons", "true_localizations"):
        for actual, reference in zip(getattr(result, name), getattr(expected, name)):
            np.testing.assert_array_equal(actual, reference)

