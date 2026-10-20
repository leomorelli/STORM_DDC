"""DDC orchestration with optional spawned image workers and parent callbacks.

Distribution image calculations and MCMC can use spawned workers. Shared
reductions, orchestration, and checkpoint callbacks execute in the parent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import nullcontext
from numbers import Integral
import sys
from time import monotonic
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .distributions import determine_blinking_distribution
from ._acceleration import native_available
from .io import DDCDataset
from .mcmc import MCMCResult, ProbabilityCache, RNGLike, run_mcmc
from .probabilities import _matlab_matrix_length


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class DDCConfig:
    """Active numerical defaults from ``run_script_DDC.m``."""

    resolution: float = 80.0
    stepper: int = 100
    n_f: int = 200
    photon_weighted_correction: int = 1
    timer: float = 720.0


def _empty_cache() -> ProbabilityCache:
    return ProbabilityCache(
        np.empty((0, 0), dtype=np.float64),
        np.empty((0, 0), dtype=np.float64),
        np.empty(0, dtype=np.float64),
    )


@dataclass
class DDCImageState:
    """One image's persistent MATLAB cell-array state between MCMC chunks."""

    step: int = 0
    constantf: FloatArray = field(default_factory=lambda: np.empty(0))
    constant2ft: FloatArray = field(default_factory=lambda: np.empty(0))
    orderf: IntArray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    likelihood: FloatArray = field(default_factory=lambda: np.empty(0))
    score: FloatArray = field(default_factory=lambda: np.empty(0))
    number_of_localizations: FloatArray = field(default_factory=lambda: np.empty(0))
    probability_cache: ProbabilityCache = field(default_factory=_empty_cache)


@dataclass(frozen=True)
class DDCPipelineResult:
    """In-memory result and complete continuation state for future A.9 save."""

    final_localizations: list[FloatArray | None]
    final_frames: list[NDArray[Any] | None]
    trajectories: list[FloatArray | None]
    states: list[DDCImageState]
    working_localizations: list[NDArray[Any]]
    frames: list[NDArray[Any]]
    true_localizations: list[NDArray[Any]]
    photons: list[NDArray[Any]]
    bins: FloatArray
    distribution_for_blink: FloatArray
    resolution: float
    x_overall: FloatArray
    m_mat: FloatArray
    diagnostics: tuple[str, ...]

    @property
    def steps(self) -> IntArray:
        return np.asarray([state.step for state in self.states], dtype=np.int64)


CheckpointCallback = Callable[[DDCPipelineResult, str], None]


def _copy_cache(cache: ProbabilityCache) -> ProbabilityCache:
    return ProbabilityCache(
        cache.deviation_in_probability.copy(),
        cache.prob_distributions.copy(),
        cache.dscale_store.copy(),
    )


def _copy_state(state: DDCImageState) -> DDCImageState:
    return DDCImageState(
        step=int(state.step),
        constantf=state.constantf.copy(),
        constant2ft=state.constant2ft.copy(),
        orderf=state.orderf.copy(),
        likelihood=state.likelihood.copy(),
        score=state.score.copy(),
        number_of_localizations=state.number_of_localizations.copy(),
        probability_cache=_copy_cache(state.probability_cache),
    )


def _copy_optional_cells(
    values: Sequence[NDArray[Any] | None],
) -> list[NDArray[Any] | None]:
    return [None if value is None else value.copy() for value in values]


def _copy_pipeline_result(result: DDCPipelineResult) -> DDCPipelineResult:
    """Detached checkpoint snapshot; callbacks cannot mutate live state."""

    return DDCPipelineResult(
        final_localizations=_copy_optional_cells(result.final_localizations),
        final_frames=_copy_optional_cells(result.final_frames),
        trajectories=_copy_optional_cells(result.trajectories),
        states=[_copy_state(state) for state in result.states],
        working_localizations=[value.copy() for value in result.working_localizations],
        frames=[value.copy() for value in result.frames],
        true_localizations=[value.copy() for value in result.true_localizations],
        photons=[value.copy() for value in result.photons],
        bins=result.bins.copy(),
        distribution_for_blink=result.distribution_for_blink.copy(),
        resolution=float(result.resolution),
        x_overall=result.x_overall.copy(),
        m_mat=result.m_mat.copy(),
        diagnostics=tuple(result.diagnostics),
    )


def _pad_localizations_source(array: ArrayLike) -> NDArray[Any]:
    """Port ``if min(size(X))<3; X(:,3)=X(:,2)*0; end``."""

    locations = np.asarray(array).copy()
    if locations.ndim != 2:
        raise ValueError("each localization cell must be a 2D matrix")
    if min(locations.shape, default=0) < 3:
        if locations.shape[1] < 2:
            raise IndexError("MATLAB source indexes LocalizationsFinal(:,2)")
        zero_third = locations[:, 1] * 0
        if locations.shape[1] < 3:
            locations = np.column_stack((locations, zero_third))
        else:
            locations[:, 2] = zero_third
    return locations


def _normalize_truth(
    truth: Sequence[ArrayLike] | None, n_images: int, dimensions: int
) -> list[NDArray[Any]]:
    if truth is None or len(truth) == 0:
        return [np.empty(0, dtype=np.float64) for _ in range(n_images)]
    if len(truth) != n_images:
        raise ValueError("TrueLocalizations must align with images")
    normalized: list[NDArray[Any]] = []
    for value in truth:
        array = np.asarray(value)
        # A.1 keeps coordinate dimensionality for correspondence checks;
        # active run_script passes an empty MATLAB [] cell into DDC_MCMC.
        normalized.append(np.empty(0, dtype=array.dtype) if array.size == 0 else array.copy())
    return normalized


def _normalize_photons(
    photons: Sequence[ArrayLike] | None,
    frames: Sequence[NDArray[Any]],
) -> list[NDArray[Any]]:
    if photons is None or len(photons) == 0:
        supplied = [np.empty(0)]
    else:
        supplied = [np.asarray(value).reshape(-1, order="F").copy() for value in photons]
    # Source gates the default for every image solely on Photons{1}.
    if supplied[0].size == 0:
        return [np.ones(frame.size, dtype=np.float64) for frame in frames]
    if len(supplied) != len(frames):
        raise IndexError("MATLAB Photons cell does not cover every image")
    return supplied


def _round_away_from_zero(values: NDArray[Any]) -> NDArray[Any]:
    return np.copysign(np.floor(np.abs(values) + 0.5), values)


def _matlab_normalize_photon_weights(weights: ArrayLike) -> NDArray[Any]:
    """Reproduce ``weights/sum(weights)`` with MATLAB numeric classes.

    MATLAB ``sum`` returns double for integer inputs, but division of an
    integer array by that scalar double returns the original integer class,
    rounded to nearest with ties away from zero and saturated on conversion.
    ``single`` remains single; ``double`` remains double.
    """

    values = np.asarray(weights).reshape(-1, order="F")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("photon weights must be numeric")
    if np.issubdtype(values.dtype, np.integer):
        denominator = float(np.sum(values, dtype=np.float64))
        with np.errstate(divide="ignore", invalid="ignore"):
            quotient = values.astype(np.float64) / denominator
        info = np.iinfo(values.dtype)
        rounded = _round_away_from_zero(quotient)
        # MATLAB integer conversion maps NaN to zero and saturates infinities.
        rounded = np.nan_to_num(
            rounded, nan=0.0, posinf=float(info.max), neginf=float(info.min)
        )
        return np.clip(rounded, info.min, info.max).astype(values.dtype)
    if values.dtype == np.dtype(np.float32):
        denominator = np.sum(values, dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(values / denominator, dtype=np.float32)
    denominator = np.sum(values, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(values / denominator, dtype=np.float64)


def _matlab_weighted_coordinate_sum(
    coordinates: NDArray[Any], weights: NDArray[Any]
) -> float:
    """Port ``sum(double_coordinates .* typed_weights)`` class behavior."""

    coordinate_values = np.asarray(coordinates).reshape(-1, order="F")
    weight_values = np.asarray(weights).reshape(-1, order="F")
    if coordinate_values.size != weight_values.size:
        raise ValueError("coordinate and photon-weight lengths differ")

    coordinates_floating = np.issubdtype(coordinate_values.dtype, np.floating)
    weights_floating = np.issubdtype(weight_values.dtype, np.floating)
    if coordinates_floating and weights_floating:
        # MATLAB supports mixed single/double arithmetic. If either operand is
        # single, both product and default sum are single precision.
        if (
            coordinate_values.dtype == np.dtype(np.float32)
            or weight_values.dtype == np.dtype(np.float32)
        ):
            product = (
                coordinate_values.astype(np.float32)
                * weight_values.astype(np.float32)
            )
            return float(np.sum(product, dtype=np.float32))
        product = coordinate_values.astype(np.float64) * weight_values.astype(
            np.float64
        )
        return float(np.sum(product, dtype=np.float64))

    if np.issubdtype(weight_values.dtype, np.integer):
        # MATLAB integer arithmetic accepts a mixed floating operand only when
        # that operand is scalar double. single is not covered by this exception.
        if coordinate_values.dtype == np.dtype(np.float32):
            raise TypeError(
                "MATLAB source cannot multiply single coordinates by integer "
                "photon weights"
            )
        if (
            coordinate_values.dtype == np.dtype(np.float64)
            and coordinate_values.size != 1
        ):
            raise TypeError(
                "MATLAB source cannot multiply a non-scalar double coordinate "
                f"vector by {weight_values.dtype} photon weights"
            )
        product = coordinate_values.astype(np.float64) * weight_values.astype(np.float64)
        info = np.iinfo(weight_values.dtype)
        converted = np.clip(
            _round_away_from_zero(product), info.min, info.max
        ).astype(weight_values.dtype)
        # Default MATLAB sum(integer) returns double.
        return float(np.sum(converted, dtype=np.float64))

    raise TypeError(
        f"unsupported MATLAB coordinate/weight classes: "
        f"{coordinate_values.dtype} and {weight_values.dtype}"
    )


def photon_weighted_collapse(
    localizations: ArrayLike,
    frames: ArrayLike,
    trajectory: ArrayLike,
    photons: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Port lines 193--203, including zero-sum division behavior."""

    locations = np.asarray(localizations)
    frame_values = np.asarray(frames).reshape(-1, order="F")
    labels = np.asarray(trajectory).reshape(-1, order="F")
    photon_values = np.asarray(photons).reshape(-1, order="F")
    rows: list[list[float]] = []
    frame_rows: list[float] = []
    for label in np.unique(labels):
        selected = labels == label
        weights = _matlab_normalize_photon_weights(photon_values[selected])
        rows.append(
            [
                _matlab_weighted_coordinate_sum(locations[selected, column], weights)
                for column in range(3)
            ]
        )
        frame_rows.append(float(np.mean(frame_values[selected])))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 3), np.asarray(frame_rows)


def _independent_rngs(n_images: int) -> list[np.random.Generator]:
    seed_sequence = np.random.SeedSequence()
    return [np.random.default_rng(child) for child in seed_sequence.spawn(n_images)]


def _matlab_round_frames(frames: NDArray[Any]) -> NDArray[Any]:
    """MATLAB ``round`` ties away from zero while preserving numeric class."""

    if np.issubdtype(frames.dtype, np.integer):
        return frames.copy()
    rounded = np.copysign(np.floor(np.abs(frames) + 0.5), frames)
    return rounded.astype(frames.dtype, copy=False)


@dataclass
class _ImageTask:
    index: int
    locations: NDArray[Any]
    frames: NDArray[Any]
    truth: NDArray[Any]
    photons: NDArray[Any]
    state: DDCImageState
    settings: DDCConfig
    addon: Any
    rng: RNGLike


@dataclass
class _ImageUpdate:
    state: DDCImageState
    locations: FloatArray | None
    frames: NDArray[Any] | None
    trajectory: FloatArray | None
    rng_state: dict[str, Any] | None


def _advance_image(task: _ImageTask, shared: tuple) -> _ImageUpdate:
    """One MCMC chunk; no I/O or access to other images' mutable state."""
    bins, blink, resolution, x_overall, m_mat = shared
    state, settings = task.state, task.settings
    old_max = float(np.max(state.likelihood)) if state.likelihood.size else -np.inf
    result: MCMCResult = run_mcmc(
        task.index + 1, task.locations, _matlab_round_frames(task.frames),
        settings.n_f, resolution, task.truth, bins, blink, state.step,
        settings.timer, state.constantf, state.orderf, state.likelihood,
        state.score, state.number_of_localizations, state.probability_cache,
        settings.stepper, task.addon, state.constant2ft, x_overall, m_mat,
        rng=task.rng,
    )
    state.probability_cache = result.probability_cache
    state.constantf = result.constantf.copy()
    state.constant2ft = result.constant2ft.copy()
    state.orderf = result.orderf.copy()
    state.step = result.steps
    state.likelihood = result.likelihood_history.copy()
    state.score = result.score_history.copy()
    state.number_of_localizations = result.number_of_localizations.copy()
    locations = frames = trajectory = None
    new_max = float(np.max(state.likelihood)) if state.likelihood.size else None
    if new_max is not None and old_max < new_max:
        locations = result.final_localizations_blinking_corrected.copy()
        frames = result.final_frame_blinking_corrected.copy()
        if settings.photon_weighted_correction == 1:
            locations, frames = photon_weighted_collapse(
                task.locations, task.frames, result.traj2, task.photons,
            )
        trajectory = result.traj2.copy()
    rng_state = (
        task.rng.bit_generator.state if isinstance(task.rng, np.random.Generator)
        else None
    )
    return _ImageUpdate(state, locations, frames, trajectory, rng_state)


def run_ddc(
    dataset: DDCDataset,
    config: DDCConfig | None = None,
    *,
    photons: Sequence[ArrayLike] | None = None,
    rngs: Sequence[RNGLike] | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
    resume_from: DDCPipelineResult | None = None,
    workers: int = 1,
    verbose: bool = False,
) -> DDCPipelineResult:
    """Run or resume DDC, optionally processing images in separate processes.

    A callback receives a detached snapshot at ``"initial"`` before the first
    MCMC pass and at ``"outer"`` after every complete outer image pass. Resume
    skips shared-distribution calculation and does not emit a new initial save.
    ``workers=1`` executes locally. Larger values use spawned processes, capped
    by the number of eligible images, and require independent NumPy Generators.
    Shared-distribution reductions and checkpoint callbacks remain in the parent.
    Ground truth may be omitted from ``dataset`` for experimental data; only
    simulation diagnostics use it.
    ``verbose=True`` prints elapsed time and ROI progress to stderr, flushing
    each message immediately. Parallel ROIs are reported as they finish.
    """

    if isinstance(workers, bool) or not isinstance(workers, Integral) or workers < 1:
        raise ValueError("workers must be a positive integer")
    workers = int(workers)
    settings = DDCConfig() if config is None else config
    n_images = dataset.n_images
    if n_images == 0:
        raise ValueError("run_script_DDC requires at least one image")
    if len(dataset.frames) != n_images:
        raise ValueError("localization and frame cells must align")

    started = monotonic()

    def report(message: str) -> None:
        if verbose:
            print(f"[DDC {monotonic() - started:.1f}s] {message}",
                  file=sys.stderr, flush=True)

    report(f"{'Resuming' if resume_from is not None else 'Starting'} {n_images} ROIs "
           f"({sum(len(loc) for loc in dataset.localizations):,} localizations); "
           f"requested workers: {workers}.")
    report(f"Compiled C kernels: {'enabled' if native_available() else 'unavailable or disabled'}.")

    if resume_from is None:
        working = [_pad_localizations_source(value) for value in dataset.localizations]
        frame_cells = [
            np.asarray(value).reshape(-1, order="F").copy()
            for value in dataset.frames
        ]
    else:
        if len(resume_from.states) != n_images:
            raise ValueError("resume state does not align with dataset images")
        working = [value.copy() for value in resume_from.working_localizations]
        frame_cells = [value.copy() for value in resume_from.frames]
    for index, (locations, frame) in enumerate(zip(working, frame_cells, strict=True)):
        if locations.shape[0] != frame.size:
            raise ValueError(f"image {index}: localization/frame length mismatch")
    if resume_from is None:
        truth = _normalize_truth(
            dataset.true_localizations, n_images, working[0].shape[1]
        )
        # A keyword argument is an intentional caller override; otherwise use
        # MAT-file Photons, then source unit defaults.
        photon_source = photons if photons is not None else dataset.photons
        photon_cells = _normalize_photons(photon_source, frame_cells)
    else:
        truth = [value.copy() for value in resume_from.true_localizations]
        photon_cells = [value.copy() for value in resume_from.photons]

    if rngs is None:
        image_rngs: Sequence[RNGLike] = _independent_rngs(n_images)
    else:
        if len(rngs) != n_images:
            raise ValueError("rngs must provide one independent RNG per image")
        image_rngs = rngs

    if resume_from is None:
        # This shared calculation occurs exactly once, before any per-image MCMC.
        report("Calculating shared blinking distribution...")
        (
            bins,
            distribution_for_blink,
            _,
            resolution,
            x_overall,
            m_mat,
        ) = determine_blinking_distribution(
            working, frame_cells, settings.n_f, settings.resolution, workers=workers
        )
        report("Shared blinking distribution ready.")
        states = [DDCImageState() for _ in range(n_images)]
        final_localizations: list[FloatArray | None] = [None] * n_images
        final_frames: list[NDArray[Any] | None] = [None] * n_images
        trajectories: list[FloatArray | None] = [None] * n_images
        diagnostics: list[str] = []
    else:
        report("Reusing shared blinking distribution from checkpoint.")
        bins = resume_from.bins.copy()
        distribution_for_blink = resume_from.distribution_for_blink.copy()
        resolution = float(resume_from.resolution)
        x_overall = resume_from.x_overall.copy()
        m_mat = resume_from.m_mat.copy()
        states = [_copy_state(state) for state in resume_from.states]
        final_localizations = _copy_optional_cells(resume_from.final_localizations)
        final_frames = _copy_optional_cells(resume_from.final_frames)
        trajectories = _copy_optional_cells(resume_from.trajectories)
        diagnostics = list(resume_from.diagnostics)
    addon = (
        np.asarray(dataset.addon).reshape(-1, order="F")
        if dataset.addon is not None
        else np.zeros(500)
    )

    def current_result() -> DDCPipelineResult:
        return DDCPipelineResult(
            final_localizations=final_localizations,
            final_frames=final_frames,
            trajectories=trajectories,
            states=states,
            working_localizations=working,
            frames=frame_cells,
            true_localizations=truth,
            photons=photon_cells,
            bins=np.asarray(bins, dtype=np.float64),
            distribution_for_blink=np.asarray(
                distribution_for_blink, dtype=np.float64
            ),
            resolution=float(resolution),
            x_overall=np.asarray(x_overall, dtype=np.float64),
            m_mat=np.asarray(m_mat, dtype=np.float64),
            diagnostics=tuple(diagnostics),
        )

    if checkpoint_callback is not None and resume_from is None:
        report("Writing initial checkpoint...")
        checkpoint_callback(_copy_pipeline_result(current_result()), "initial")
        report("Initial checkpoint saved.")

    shared = (bins, distribution_for_blink, resolution, x_overall, m_mat)
    eligible_count = sum(
        _matlab_matrix_length(locations) > 10 and state.step < settings.stepper
        for locations, state in zip(working, states, strict=True)
    )
    worker_count = min(workers, eligible_count)
    pool_context = nullcontext(None)
    if worker_count > 1:
        if not all(isinstance(rng, np.random.Generator) for rng in image_rngs):
            raise TypeError("parallel execution requires one NumPy Generator per image")
        if len({id(rng.bit_generator) for rng in image_rngs}) != n_images:
            raise ValueError("parallel execution requires independent RNGs per image")
        from ._parallel import create_pool, completed_images
        pool_context = create_pool(worker_count, shared)

    report(f"{eligible_count} ROIs need processing; "
           f"using {worker_count} {'workers' if worker_count != 1 else 'worker'}.")
    for index, state in enumerate(states):
        if state.step >= settings.stepper:
            report(f"ROI {index + 1}/{n_images}: already complete.")

    pass_number = 0
    with pool_context as pool:
        while any(state.step < settings.stepper for state in states):
            pass_number += 1
            tasks = []
            for image_index in range(n_images):
                state = states[image_index]
                matlab_length = _matlab_matrix_length(working[image_index])
                if matlab_length > 8000:
                    diagnostics.append(
                        f"image {image_index + 1}: more than 8000 localizations"
                    )
                if matlab_length > 10 and state.step < settings.stepper:
                    if image_index >= addon.size:
                        raise IndexError("MATLAB addonarray does not cover image index")
                    tasks.append(_ImageTask(
                        image_index, working[image_index], frame_cells[image_index],
                        truth[image_index], photon_cells[image_index], state,
                        settings, addon[image_index], image_rngs[image_index],
                    ))
                else:
                    if state.step < settings.stepper:
                        report(f"ROI {image_index + 1}/{n_images}: skipped "
                               f"({len(working[image_index])} localizations; "
                               "too few for MCMC).")
                    state.step = settings.stepper

            report(f"Pass {pass_number}: processing {len(tasks)} ROIs...")

            def sequential_updates():
                for task in tasks:
                    report(f"ROI {task.index + 1}/{n_images}: processing "
                           f"{len(task.locations):,} localizations...")
                    yield task, _advance_image(task, shared)

            updates = (
                completed_images(pool, tasks)
                if pool is not None
                else sequential_updates()
            )
            finished_count = sum(state.step >= settings.stepper for state in states)
            # Completion order may vary; indexed storage preserves ROI order.
            # Only the parent commits state, reports progress, and saves.
            for task, update in updates:
                index = task.index
                states[index] = update.state
                if pool is not None:
                    image_rngs[index].bit_generator.state = update.rng_state
                if update.locations is not None:
                    final_localizations[index] = update.locations
                    final_frames[index] = update.frames
                    trajectories[index] = update.trajectory
                done = update.state.step >= settings.stepper
                finished_count += int(done)
                status = "DONE" if done else "pass finished; continuing"
                corrected = final_localizations[index]
                count_text = (
                    f"{len(corrected):,} corrected localizations"
                    if corrected is not None else "no corrected result yet"
                )
                report(f"ROI {index + 1}/{n_images}: {status} "
                       f"(step {update.state.step}/{settings.stepper}; {count_text}). "
                       f"Finished/skipped: {finished_count}/{n_images}.")

            if checkpoint_callback is not None:
                report(f"Writing checkpoint after pass {pass_number}...")
                checkpoint_callback(_copy_pipeline_result(current_result()), "outer")
                report("Checkpoint saved.")

    report(f"Finished: all {n_images} ROIs processed or skipped.")
    return current_result()


__all__ = [
    "DDCConfig",
    "DDCImageState",
    "DDCPipelineResult",
    "CheckpointCallback",
    "_matlab_normalize_photon_weights",
    "photon_weighted_collapse",
    "run_ddc",
]
