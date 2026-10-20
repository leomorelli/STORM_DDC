"""Python preprocessing counterparts of MATLAB's ``DDC_Tools`` scripts.

Coordinates and buffer/bin widths are in nm. These operational tools accept
integer frames without changing the input dataset. N_f remains a visual choice
from the Z curve, as in Determine_N.m; no automatic plateau rule is invented.
Pair distances are accumulated in blocks, never as a full image-sized matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from numbers import Integral
from typing import Callable, Iterator

import numpy as np
from scipy.spatial.distance import cdist, pdist

from .io import DDCDataset, _validate_correspondence
from .combining import combine_images

Progress = Callable[[str], None] | None


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _validate(dataset: DDCDataset) -> None:
    _validate_correspondence(dataset)
    if not dataset.n_images or not any(len(x) for x in dataset.localizations):
        raise ValueError("dataset must contain at least one localization")
    for i, (loc, frame) in enumerate(zip(dataset.localizations, dataset.frames)):
        loc, frame = np.asarray(loc), np.asarray(frame)
        if loc.ndim != 2 or loc.shape[1] not in (2, 3):
            raise ValueError(f"image {i + 1}: coordinates must have shape (n, 2) or (n, 3)")
        if not np.isrealobj(loc) or not np.isfinite(loc).all():
            raise ValueError(f"image {i + 1}: coordinates must be finite real numbers")
        if frame.ndim > 2 or (frame.ndim == 2 and min(frame.shape) > 1):
            raise ValueError(f"image {i + 1}: frames must be a vector")
        if not np.isrealobj(frame) or not np.isfinite(frame).all():
            raise ValueError(f"image {i + 1}: frames must be finite real numbers")


def _pairs(loc, frames, block_size: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield each unordered pair once, with aligned distance and frame lag."""
    loc = np.asarray(loc, dtype=float)
    frames = np.asarray(frames, dtype=float).reshape(-1)
    for start in range(0, len(loc), block_size):
        end = min(start + block_size, len(loc))
        yield pdist(loc[start:end]), pdist(frames[start:end, None])
        for other in range(end, len(loc), block_size):
            stop = min(other + block_size, len(loc))
            yield (
                cdist(loc[start:end], loc[other:stop]).ravel(),
                np.abs(frames[start:end, None] - frames[None, other:stop]).ravel(),
            )


def _max_distance(dataset: DDCDataset, block_size: int) -> float:
    maximum = 0.0
    for loc in dataset.localizations:
        loc = np.asarray(loc, dtype=float)
        for start in range(0, len(loc), block_size):
            a = loc[start:start + block_size]
            maximum = max(maximum, float(pdist(a).max(initial=0)))
            for other in range(start + block_size, len(loc), block_size):
                maximum = max(maximum, float(cdist(a, loc[other:other + block_size]).max()))
    return maximum


def _edges(width: float, stop: float) -> np.ndarray:
    if stop < 0:
        raise ValueError("resolution exceeds the maximum pair distance; choose a smaller bin width")
    return np.r_[np.arange(int(np.floor(stop / width)) + 1) * width, np.inf]


def _probability(counts: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return counts / counts.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class NDiagnostic:
    frame_differences: np.ndarray
    bins: np.ndarray
    cdf: np.ndarray
    z: np.ndarray
    pair_counts: np.ndarray
    image_indices: np.ndarray

    def matlab_variables(self) -> dict:
        return {"Framestore": self.frame_differences, "bins": self.bins,
                "Cum_Sum_Store": self.cdf, "Z": self.z,
                "Pair_Counts": self.pair_counts, "Image_Indices": self.image_indices + 1}


def determine_n(
    dataset: DDCDataset, *, max_frame: int = 1000, gap: int = 10,
    max_localizations: int | None = 6000, block_size: int = 512,
    progress: Progress = None,
) -> NDiagnostic:
    """Compute Determine_N.m's Z versus exact lag ``1:gap:max_frame``.

    Z is the L1 difference from the lag-1 cumulative distance histogram, with
    edges ``[0:250:5000, Inf]``. Images with >=6000 rows are excluded by default,
    just as in MATLAB. Empty lag distributions are NaN and are never interpreted
    as a plateau. Choose N_f where the populated Z curve levels off.
    """
    _validate(dataset)
    max_frame = _positive_int(max_frame, "max_frame")
    gap = _positive_int(gap, "gap")
    block_size = _positive_int(block_size, "block_size")
    if max_localizations is not None:
        max_localizations = _positive_int(max_localizations, "max_localizations")
    selected = np.array([i for i, loc in enumerate(dataset.localizations)
                         if max_localizations is None or len(loc) < max_localizations], dtype=int)
    if not selected.size:
        raise ValueError("no images below max_localizations; split the images first or increase the limit")
    lags = np.arange(1, max_frame + 1, gap)
    bins = np.r_[np.arange(0, 5001, 250), np.inf]
    counts = np.zeros((len(lags), len(bins) - 1), dtype=np.int64)
    for i in selected:
        if progress:
            progress(f"N_f diagnostic: image {i + 1}/{dataset.n_images}")
        for distances, dt in _pairs(dataset.localizations[i], dataset.frames[i], block_size):
            keep = (dt >= 1) & (dt <= lags[-1]) & ((dt - 1) % gap == 0)
            row = ((dt[keep] - 1) / gap).astype(np.int64)
            column = np.searchsorted(bins, distances[keep], side="right") - 1
            counts += np.bincount(row * counts.shape[1] + column,
                                  minlength=counts.size).reshape(counts.shape)
    cdf = np.cumsum(_probability(counts), axis=1)
    return NDiagnostic(lags, bins, cdf, np.sum(np.abs(cdf - cdf[0]), axis=1),
                       counts.sum(axis=1), selected)


@dataclass(frozen=True)
class ResolutionDiagnostic:
    resolution: float
    n_f: float
    bins: np.ndarray
    probabilities: np.ndarray
    true_probabilities: np.ndarray | None
    pair_counts: np.ndarray
    image_indices: np.ndarray

    def matlab_variables(self) -> dict:
        result = {"Res": self.resolution, "N_f": self.n_f, "bins": self.bins,
                  "Res_Store": self.probabilities, "Pair_Counts": self.pair_counts,
                  "Image_Indices": self.image_indices + 1}
        if self.true_probabilities is not None:
            result["Res_Store2"] = self.true_probabilities
        return result


def inspect_resolution(
    dataset: DDCDataset, *, resolution: float = 80, n_f: float = 200,
    image_fraction: float = 0.1, block_size: int = 512, progress: Progress = None,
) -> ResolutionDiagnostic:
    """Port Determine_Res.m's per-image true-pair distance distributions.

    Uses strict lag > N_f, the first ceil(image_fraction*n_images) images,
    and ``[0:resolution:D_max-resolution, Inf]``. Optional ground-truth matching
    deliberately uses x-coordinate equality, as the MATLAB diagnostic does.
    """
    _validate(dataset)
    resolution, n_f = _positive(resolution, "resolution"), _positive(n_f, "n_f")
    block_size = _positive_int(block_size, "block_size")
    if not np.isfinite(image_fraction) or not 0 < image_fraction <= 1:
        raise ValueError("image_fraction must be in (0, 1]")
    if progress:
        progress("Resolution diagnostic: finding maximum pair distance")
    bins = _edges(resolution, _max_distance(dataset, block_size) - resolution)
    selected = np.arange(int(np.ceil(dataset.n_images * image_fraction)))
    counts = np.zeros((len(selected), len(bins) - 1), dtype=np.int64)
    truth_counts = np.zeros_like(counts) if dataset.true_localizations else None
    for i in selected:
        if progress:
            progress(f"Resolution diagnostic: image {i + 1}/{len(selected)}")
        loc = np.asarray(dataset.localizations[i])
        frame = np.asarray(dataset.frames[i]).reshape(-1)
        for distances, dt in _pairs(loc, frame, block_size):
            counts[i] += np.histogram(distances[dt > n_f], bins)[0]
        if truth_counts is not None:
            truth = np.asarray(dataset.true_localizations[i])
            mask = np.isin(loc[:, 0], truth[:, 0]) if truth.size else np.zeros(len(loc), bool)
            for distances, dt in _pairs(loc[mask], frame[mask], block_size):
                truth_counts[i] += np.histogram(distances[dt > n_f], bins)[0]
    return ResolutionDiagnostic(resolution, n_f, bins, _probability(counts),
                                None if truth_counts is None else _probability(truth_counts),
                                counts.sum(axis=1), selected)


@dataclass(frozen=True)
class ResolutionTrial:
    resolution: float
    bins: np.ndarray
    short_probability: np.ndarray
    long_probability: np.ndarray
    blink_probability: np.ndarray
    tail_scale: float
    noise_trimmed: bool


@dataclass(frozen=True)
class ResolutionEstimate:
    resolution: float
    n_f: float
    trials: tuple[ResolutionTrial, ...]
    reached_minimum: bool

    def matlab_variables(self) -> dict:
        from .io import _mat_cell
        return {"Resolution": self.resolution, "N_f": self.n_f,
                "Tested_Resolutions": np.array([x.resolution for x in self.trials]),
                "Bins": _mat_cell([x.bins for x in self.trials]),
                "Blink_Probabilities": _mat_cell([x.blink_probability for x in self.trials]),
                "Short_Probabilities": _mat_cell([x.short_probability for x in self.trials]),
                "Long_Probabilities": _mat_cell([x.long_probability for x in self.trials]),
                "Tail_Scales": np.array([x.tail_scale for x in self.trials]),
                "Reached_Minimum": self.reached_minimum}


def _clean_blink(short, long, width) -> tuple[np.ndarray, float, bool]:
    # MATLAB lsne is 1-based: first lsne*Resolution > 1000.
    start = int(np.floor(1000 / width))
    denominator = long[start:].sum()
    if not np.isfinite(short).all() or not np.isfinite(long).all():
        raise ValueError("automatic resolution needs short- and long-lag pairs in every image; "
                         "check N_f or use inspect-resolution")
    if denominator <= 0:
        raise ValueError("automatic resolution has no long-lag tail support near 1000 nm; "
                         "use inspect-resolution to choose a bin width manually")
    scale = float(short[start:].sum() / denominator)
    blink = short - long * scale
    total = blink.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("automatic resolution cannot isolate a positive blinking distribution; "
                         "check N_f or use inspect-resolution")
    blink /= total
    good = True
    for i in range(3, len(blink) - 1):
        if not (good and blink[i] > 0 and blink[i + 1] < blink[i]):
            good = False
            blink[i] = 0
    blink[blink < 0] = 0
    if blink.sum() <= 0:
        raise ValueError("automatic resolution has no blinking probability after cleanup")
    blink /= blink.sum()
    blink /= blink.sum()
    trimmed = bool(np.count_nonzero(blink[7:] > 0) > 1)
    if trimmed:
        blink[7:] = 0
        if blink.sum() <= 0:
            raise ValueError("automatic resolution has only high-bin noise; use inspect-resolution")
        blink /= blink.sum()
    return blink, scale, trimmed


def determine_resolution(
    dataset: DDCDataset, *, n_f: float = 200, max_resolution: int = 150,
    min_resolution: int = 10, step: int = 10, block_size: int = 512,
    progress: Progress = None,
) -> ResolutionEstimate:
    """Port New_Determine_Res.m's descending bin-width search.

    Short/long pairs use lag < / > N_f, excluding equality. Probability
    histograms are averaged equally across images. The first candidate whose
    cleaned blinking peak is outside bin one recommends candidate + step
    (including 160 nm when the 150 nm candidate triggers). Undefined source
    calculations raise ValueError instead of returning an arbitrary resolution.
    """
    _validate(dataset)
    n_f = _positive(n_f, "n_f")
    max_resolution = _positive_int(max_resolution, "max_resolution")
    min_resolution = _positive_int(min_resolution, "min_resolution")
    step = _positive_int(step, "step")
    block_size = _positive_int(block_size, "block_size")
    if min_resolution > max_resolution or (max_resolution - min_resolution) % step:
        raise ValueError("resolution bounds must be ordered and separated by a multiple of step")
    if progress:
        progress("Automatic resolution: finding maximum pair distance")
    maximum = _max_distance(dataset, block_size)
    widths = list(range(max_resolution, min_resolution - 1, -step))
    edges = [_edges(width, maximum) for width in widths]
    short = [np.zeros((dataset.n_images, len(b) - 1), dtype=np.int64) for b in edges]
    long = [np.zeros_like(x) for x in short]
    for i, (loc, frame) in enumerate(zip(dataset.localizations, dataset.frames)):
        if progress:
            progress(f"Automatic resolution: image {i + 1}/{dataset.n_images}")
        for distances, dt in _pairs(loc, frame, block_size):
            close, far = distances[dt < n_f], distances[dt > n_f]
            for j, bins in enumerate(edges):
                short[j][i] += np.histogram(close, bins)[0]
                long[j][i] += np.histogram(far, bins)[0]
    trials = []
    for width, bins, a, b in zip(widths, edges, short, long):
        if progress:
            progress(f"Automatic resolution: checking {width} nm")
        a, b = _probability(a).mean(axis=0), _probability(b).mean(axis=0)
        blink, scale, trimmed = _clean_blink(a, b, width)
        trials.append(ResolutionTrial(width, bins, a, b, blink, scale, trimmed))
        if np.argmax(blink) != 0:
            return ResolutionEstimate(width + step, n_f, tuple(trials), False)
    return ResolutionEstimate(min_resolution, n_f, tuple(trials), True)


def _xyz(loc) -> np.ndarray:
    loc = np.asarray(loc)
    if not loc.size:
        return np.empty((0, 3), dtype=loc.dtype)
    if loc.ndim != 2 or loc.shape[1] not in (2, 3) or not np.isfinite(loc).all():
        raise ValueError("truth coordinates must be finite with 2 or 3 columns")
    return np.column_stack((loc, np.zeros(len(loc), dtype=loc.dtype))) if loc.shape[1] == 2 else loc.copy()


def _axis_cuts(loc, proportions):
    cuts = []
    for axis, p in enumerate(proportions):
        q = np.minimum(np.arange(int(np.ceil(1 / p)) + 1) * p, 1)
        # MATLAB quantile uses midpoint empirical probabilities (Hazen/type 5).
        bounds = np.quantile(loc[:, axis], q, method="hazen")
        cuts.append(np.column_stack((bounds[:-1], bounds[1:])))
    return cuts


def _mask(loc, cuts, addon, *, inclusive=False):
    lower, upper = cuts[:, 0] - addon, cuts[:, 1] + addon
    keep = (loc >= lower) & (loc <= upper)
    if not inclusive:
        keep[:, :2] = (loc[:, :2] > lower[:2]) & (loc[:, :2] < upper[:2])
    return keep.all(axis=1)


def _grid_score(loc, axes, buffer, target):
    membership = [(loc[:, a, None] >= c[:, 0] - buffer) &
                  (loc[:, a, None] <= c[:, 1] + buffer) for a, c in enumerate(axes)]
    counts = []
    for x, y in product(range(len(axes[0])), range(len(axes[1]))):
        xy = membership[0][:, x] & membership[1][:, y]
        counts.extend(np.count_nonzero(membership[2][xy], axis=0))
    return float(np.mean((np.asarray(counts, dtype=float) - target) ** 2))


def split_images(
    dataset: DDCDataset, *, min_localizations: int = 5000, buffer: float = 250,
    search_iterations: int = 300, max_trials: int = 10000,
    rng: np.random.Generator | None = None, progress: Progress = None,
) -> DDCDataset:
    """Split coordinate images into buffered quantile regions, as in MATLAB.

    The source search counter changes sign at each improvement. That rule is
    retained with a max_trials safety limit. Images at or below the target are
    kept whole (the source unnecessarily searches and duplicates equality cases).
    Empty images retain an empty region. Outputs are 3D, with z=0 for 2D data;
    Came_from_image is 1-based for MATLAB/checkpoint compatibility.
    """
    _validate(dataset)
    target = _positive_int(min_localizations, "min_localizations")
    search_iterations = _positive_int(search_iterations, "search_iterations")
    max_trials = _positive_int(max_trials, "max_trials")
    buffer = _positive(buffer, "buffer")
    rng = np.random.default_rng() if rng is None else rng
    photons = dataset.photons
    if photons is None or len(photons) == 0 or all(np.asarray(p).size == 0 for p in photons):
        photons = [np.ones(len(loc)) for loc in dataset.localizations]
    if len(photons) != dataset.n_images:
        raise ValueError("photons must contain one vector per image")
    for loc, p in zip(dataset.localizations, photons):
        if np.asarray(p).size != len(loc) or not np.isfinite(p).all():
            raise ValueError("photons must be finite and align with localization rows")
    truth = dataset.true_localizations or [np.empty((0, 3)) for _ in dataset.localizations]
    result = DDCDataset([], [], true_localizations=[], photons=[], cuts=([], [], []),
                        parameters_to_split=[], source=dataset.source,
                        mat_passthrough=dict(dataset.mat_passthrough))
    origins, buffers = [], []
    for image_index, (raw, frame, true, photon) in enumerate(
        zip(dataset.localizations, dataset.frames, truth, photons)
    ):
        loc, true = _xyz(raw), _xyz(true)
        frame, photon = np.asarray(frame).reshape(-1), np.asarray(photon).reshape(-1)
        if progress:
            progress(f"Splitting image {image_index + 1}/{dataset.n_images}: {len(loc)} localizations")
        best_p = np.ones(3)
        if len(loc) > target:
            best_score, counter = np.inf, 1
            for trial in range(max_trials):
                if counter >= search_iterations:
                    break
                counter += 1
                p = np.r_[rng.random(2) * .95 + .05,
                          rng.random() * .95 + .05 if np.max(loc[:, 2]) != 0 else 1.0]
                score = _grid_score(loc, _axis_cuts(loc, p), buffer, target)
                if score < best_score:
                    best_score, best_p = score, p
                    counter = -counter
                if progress and (trial + 1) % 100 == 0:
                    progress(f"Image {image_index + 1}: {trial + 1} split candidates evaluated")
            else:
                if counter < search_iterations:
                    raise ValueError("split search reached max_trials; increase max_trials or reduce search_iterations")
        axes = _axis_cuts(loc, best_p) if len(loc) else [np.array([[0., 0.]])] * 3
        for parts in product(*axes):
            cuts = np.asarray(parts)
            addon = buffer
            if len(loc) <= target:
                keep = np.ones(len(loc), dtype=bool)
            else:
                # Jump to the first possible 10-nm increment, then apply the
                # exact strict x/y and inclusive z predicates from the script.
                need = np.maximum(cuts[:, 0] - loc, loc - cuts[:, 1]).max(axis=1)
                kth = float(np.partition(need, target - 1)[target - 1])
                addon += 10 * max(1, int(np.ceil((kth - buffer) / 10)))
                keep = _mask(loc, cuts, addon)
                while keep.sum() < target:
                    addon += 10
                    keep = _mask(loc, cuts, addon)
                while keep.sum() > target:
                    addon -= 10
                    keep = _mask(loc, cuts, addon)
                    if addon < 150:
                        break
            true_keep = _mask(true, cuts, addon)
            result.localizations.append(loc[keep].copy())
            result.frames.append(frame[keep].copy())
            result.photons.append(photon[keep].copy())
            result.true_localizations.append(true[true_keep].copy())
            for axis in range(3):
                result.cuts[axis].append(cuts[axis].copy())
            result.parameters_to_split.append(best_p.copy())
            origins.append(image_index + 1)
            buffers.append(addon)
    result.came_from_image = np.asarray(origins, dtype=np.int64)
    result.addon = np.asarray(buffers, dtype=float)
    return result


__all__ = ["NDiagnostic", "ResolutionDiagnostic", "ResolutionTrial", "ResolutionEstimate",
           "determine_n", "inspect_resolution", "determine_resolution", "split_images", "combine_images"]
