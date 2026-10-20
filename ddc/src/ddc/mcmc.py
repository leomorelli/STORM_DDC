"""Faithful Stage-A orchestration port of the active ``DDC_MCMC.m``.

The public boundary mirrors the MATLAB argument list through ``run_mcmc``.
Randomness is supplied explicitly and never uses NumPy's global RNG.  The
implementation intentionally retains the source's loop order, repeated
pair-distance calculations, exact jitter uniqueness test, Metropolis
``log(rand)`` expression, and ``steps = 10`` best-state reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from time import monotonic
from typing import Any, Callable, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.spatial.distance import pdist

from .distances import make_distance_matrix_individual
from ._acceleration import try_eliminate
from .distributions import _frame_pair_distances, _probability_histogram
from .likelihood import calculate_score_8
from .probabilities import (
    _can_search_frame_candidates,
    _frame_cross_distances,
    _matlab_matrix_length,
    _matlab_pairwise_expression,
    determine_deviation_in_probability8,
)
from .scoring import matlab_mat2gray, palm_density_image, palm_image_squared_error
from .trajectories import density_calc, determine_locs_w_multi2


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class RNGLike(Protocol):
    """Subset of ``numpy.random.Generator`` used in source operation order."""

    def random(self, size: Any = None) -> Any: ...
    def normal(self, size: Any = None) -> Any: ...
    def integers(self, low: int, high: int | None = None, size: Any = None) -> Any: ...
    def permutation(self, x: Any) -> Any: ...


@dataclass(frozen=True)
class ProbabilityCache:
    """Python equivalent of MATLAB ``Prob_dists``."""

    deviation_in_probability: FloatArray
    prob_distributions: FloatArray
    dscale_store: FloatArray


@dataclass(frozen=True)
class MCMCProposal:
    """One fully evaluated proposal, used by deterministic transition tests."""

    constant: FloatArray
    constant2: FloatArray
    order: IntArray
    localizations: FloatArray
    frames: NDArray[Any]
    trajectory: FloatArray
    likelihood: float
    score: float = 0.0


@dataclass
class MCMCState:
    """Post-score transition state and separate best-ever output snapshot.

    ``constant2ft`` may already contain a branch-3 pre-acceptance snapshot;
    this helper does not imply accepted-only semantics for that field.
    """

    constant: FloatArray
    constant2ft: FloatArray
    order: IntArray
    current_localizations: FloatArray
    current_frames: NDArray[Any]
    current_trajectory: FloatArray
    final_lik: float
    best_localizations: FloatArray
    best_frames: NDArray[Any]
    best_trajectory: FloatArray
    max_lik: float
    steps: int
    likelihood_history: list[float]
    score_history: list[float]
    number_history: list[int]


@dataclass(frozen=True)
class MCMCResult:
    """All MATLAB outputs plus traceable accepted/best-state diagnostics."""

    traj2: FloatArray
    final_localizations_blinking_corrected: FloatArray
    final_frame_blinking_corrected: NDArray[Any]
    likelihood_history: FloatArray
    score_history: FloatArray
    number_of_localizations: FloatArray
    bestlik: float
    steps: int
    constantf: FloatArray
    orderf: IntArray
    probability_cache: ProbabilityCache
    constant2ft: FloatArray
    final_lik: float
    max_lik: float
    jittered_localizations: FloatArray
    d_overall: FloatArray
    z2: FloatArray
    working_localizations: FloatArray


def metropolis_accept(new_lik: float, current_lik: float, random_value: float) -> bool:
    """Source acceptance expression, including NaN/Inf behavior."""

    if new_lik >= current_lik:
        return True
    with np.errstate(divide="ignore", invalid="ignore"):
        return bool(np.log(random_value) <= new_lik - current_lik)


def apply_proposal_transition(
    state: MCMCState,
    proposal: MCMCProposal,
    *,
    random_value: float,
) -> bool:
    """Apply lines 472--510 after the source's ordinary step increment.

    Returns whether the proposal was accepted. A new maximum resets ``steps``
    to exactly 10. Accepted nonmaxima update current state but not best state.
    """

    if not metropolis_accept(proposal.likelihood, state.final_lik, random_value):
        return False
    state.constant = proposal.constant.copy()
    state.constant2ft = proposal.constant2.copy()
    state.order = proposal.order.copy()
    state.current_localizations = proposal.localizations.copy()
    state.current_frames = proposal.frames.copy()
    state.current_trajectory = proposal.trajectory.copy()
    state.final_lik = float(proposal.likelihood)
    if proposal.likelihood > state.max_lik:
        state.best_localizations = proposal.localizations.copy()
        state.best_frames = proposal.frames.copy()
        state.best_trajectory = proposal.trajectory.copy()
        state.max_lik = float(proposal.likelihood)
        state.final_lik = float(proposal.likelihood)
        state.steps = 10
    state.likelihood_history.append(state.final_lik)
    state.score_history.append(float(proposal.score))
    state.number_history.append(_matlab_matrix_length(proposal.localizations))
    return True


def run_controlled_transitions(
    state: MCMCState,
    proposals: list[MCMCProposal],
    step_total: int,
    random_values: list[float],
) -> MCMCState:
    """Small deterministic harness with the source ``while steps<step_total``."""

    for proposal, value in zip(proposals, random_values):
        if not state.steps < step_total:
            break
        state.steps += 1
        apply_proposal_transition(state, proposal, random_value=value)
    return state


def jitter_unique_distances(
    localizations_final: ArrayLike,
    frame_information: ArrayLike,
    a: Real,
    rng: RNGLike,
) -> tuple[FloatArray, FloatArray, FloatArray, int]:
    """Port lines 194--220, before the main MCMC loop.

    Exact ``unique`` cardinality is used. As in MATLAB, ``Z2`` is destructively
    filtered inside each retry; no tolerance or retry optimization is added.
    """

    locations = np.asarray(localizations_final, dtype=np.float64)
    frames = np.asarray(frame_information).reshape(-1, order="F")
    z2 = _frame_pair_distances(frames)
    attempts = 0
    while True:
        attempts += 1
        candidate = locations + np.asarray(rng.random(locations.shape)) * 0.01
        distances = pdist(candidate)
        under = z2 < a
        distances = distances[under]
        z2 = z2[under]
        positive = z2 > 0
        distances = distances[positive]
        z2 = z2[positive]
        if np.unique(distances).size == distances.size:
            return candidate, distances, z2, attempts


def _full(value: Any) -> FloatArray:
    if sparse.issparse(value):
        return np.asarray(value.toarray(), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _matlab_length(value: NDArray[Any]) -> int:
    if value.ndim == 0:
        return 1
    return max(value.shape, default=0)


def _eliminate_blinking_de_loc15(
    localizations_final: NDArray[Any],
    frame: NDArray[Any],
    constants: FloatArray,
    resolution: Real,
    a: Real,
    deviation: FloatArray,
    traj_num: FloatArray,
) -> tuple[FloatArray, NDArray[Any], FloatArray]:
    """Active Algorithm-1 dependency called by DDC_MCMC (not the A.4 variant)."""

    locations = np.asarray(localizations_final)
    frames = np.asarray(frame).reshape(-1, order="F")
    matlab_length = _matlab_matrix_length(locations)
    if matlab_length > locations.shape[0]:
        raise IndexError("MATLAB length(LocalizationsFinal) exceeds its row count")
    native = try_eliminate(locations, frames, np.asarray(constants), resolution,
        int(np.floor(float(a) + 0.5)), np.asarray(deviation),
        np.asarray(traj_num).reshape(-1, order="F"), probability_mode=False)
    if native is not None:
        keep, _, trajectory = native
        return locations[keep, :], frames[keep], trajectory
    linked = np.zeros(matlab_length)
    is_blink = np.zeros(matlab_length)
    trajectory = np.asarray(traj_num, dtype=np.float64).reshape(-1, order="F").copy()
    singleton_trajectories = bool(
        np.all(trajectory != 0) and np.unique(trajectory).size == trajectory.size
    )
    loc = locations.copy()
    out_frames = frames.copy()
    # Rounded, sorted double frames are the operational pipeline's usual input.
    # Exact integer sums in this range let binary searches replace repeated
    # scans without changing source/target order or subtraction semantics.
    last_difference = int(np.floor(float(a) + 0.5))
    search_frames = _can_search_frame_candidates(frames, last_difference)
    for difference in range(1, int(np.floor(float(a) + 0.5)) + 1):
        starts = (
            np.searchsorted(frames, frames + difference, side="left")
            if search_frames else None
        )
        for source in range(frames.size):
            investigate = int(starts[source]) - source if starts is not None else 0
            if starts is None:
                while frames[source + investigate] - frames[source] < difference:
                    investigate += 1
                    if source + investigate >= frames.size:
                        break
            if source + investigate >= frames.size:
                break
            while frames[source + investigate] - frames[source] == difference:
                target = source + investigate
                if linked[source] != 1 and is_blink[target] != 1:
                    if singleton_trajectories:
                        source_group = np.array([source])
                    else:
                        source_group = (
                            np.flatnonzero(trajectory == trajectory[source])
                            if trajectory[source] != 0
                            else np.array([source])
                        )
                    if singleton_trajectories:
                        target_group = np.array([target])
                    else:
                        target_group = (
                            np.flatnonzero(trajectory == trajectory[target])
                            if trajectory[target] != 0
                            else np.array([target])
                        )
                    distance = _matlab_pairwise_expression(
                        loc[source_group, :], loc[target_group, :]
                    )
                    framer = _frame_cross_distances(
                        frames[source_group], frames[target_group]
                    )
                    bins = np.floor(distance.reshape(-1, order="F") / resolution) + 1
                    bins[bins > deviation.shape[1]] = deviation.shape[1]
                    if np.max(framer) < a:
                        rows = framer.reshape(-1, order="F")
                        if np.any(rows < 1) or np.any(rows != np.floor(rows)):
                            raise IndexError("MATLAB sub2ind received invalid subscripts")
                        values = deviation[
                            rows.astype(np.intp) - 1, bins.astype(np.intp) - 1
                        ]
                        probability = np.mean(values) / abs(1 + constants[target])
                        if probability > 0.5:
                            is_blink[target] = 1
                            linked[source] = 1
                            if trajectory[source] == 0:
                                trajectory[source] = traj_num[source]
                            members: NDArray[np.bool_] | int = (
                                trajectory == trajectory[target]
                                if trajectory[target] != 0
                                else target
                            )
                            trajectory[members] = trajectory[source]
                            singleton_trajectories = False
                investigate += 1
                if source + investigate >= frames.size:
                    break
    keep = is_blink <= 0
    return loc[keep, :], out_frames[keep], trajectory


def _prepare_distributions(
    distance_control: FloatArray,
    bins: FloatArray,
    blink: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    maximum = np.max(distance_control)
    mask = bins[1:] > maximum
    if np.sum(mask) > 1:
        blink = blink[~mask]
        blink = np.append(blink, 0.0)
        bins = bins[bins <= maximum]
        bins = np.append(bins, np.inf)
    true = _probability_histogram(distance_control, bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        true = true / np.sum(true)
    blink22 = np.asarray(
        [blink[index] + blink[index + 1] for index in range(0, blink.size - 1, 2)]
    )
    bins22 = np.asarray(
        [bins[index] for index in range(0, bins.size, 2) if blink.size >= index + 2]
        + [np.inf]
    )
    true22 = _probability_histogram(distance_control, bins22)
    with np.errstate(divide="ignore", invalid="ignore"):
        true22 = true22 / np.sum(true22)
    return bins, blink, true, np.asarray((blink22, bins22, true22), dtype=object)


def _proposal_parameters(
    constant: FloatArray,
    constant2: FloatArray,
    order: IntArray,
    constantf: FloatArray,
    persistent_constant2ft: FloatArray,
    frames_w_multi: FloatArray,
    frames: NDArray[Any],
    density: FloatArray,
    unsorted: IntArray,
    steps: int,
    rng: RNGLike,
) -> tuple[FloatArray, FloatArray, IntArray, FloatArray]:
    """Lines 334--455 for one proposal attempt, in RNG call order."""

    vvse = int(rng.integers(1, 4))
    if frames_w_multi.size == 0 and vvse == 2:
        vvse = 1
    if vvse == 1 and steps > 1:
        sorted_indices = np.argsort(density.reshape(-1), kind="stable")
        ranks = np.empty(constant.size, dtype=np.int64)
        ranks[sorted_indices] = unsorted
        original = constant.copy()
        while np.array_equal(constant, constantf) or np.array_equal(constant, original):
            position = int(rng.integers(0, constant.size))
            constant[position] += float(rng.normal()) * 0.1
            if float(rng.random()) < 0.5:
                for rank in range(1, constant.size):
                    if constant[ranks == rank - 1][0] > constant[ranks == rank][0]:
                        constant[ranks == rank] = constant[ranks == rank - 1][0]
            else:
                for rank in range(constant.size - 1, 0, -1):
                    if constant[ranks == rank - 1][0] > constant[ranks == rank][0]:
                        constant[ranks == rank - 1] = constant[ranks == rank][0]
        constant[constant < -0.9] = -0.9
        mean = np.mean(constant)
        constant = constant + abs(mean) if mean < 0 else constant - abs(mean)
    if vvse == 3 and steps > 1:
        # MATLAB line 410 mutates the persistent Constant2ft before the
        # proposal is scored. Assignment has value semantics, hence copy.
        persistent_constant2ft = constant2.copy()
        original2 = constant2.copy()
        while np.array_equal(constant2, original2):
            position = int(rng.integers(0, constant2.size))
            constant2[position] += float(rng.normal()) * 0.1
            if float(rng.random()) < 0.5:
                for index in range(1, constant2.size):
                    if constant2[index - 1] < constant2[index]:
                        constant2[index] = constant2[index - 1]
            else:
                for index in range(constant2.size - 1, 0, -1):
                    if constant2[index - 1] < constant2[index]:
                        constant2[index - 1] = constant2[index]
            constant2[constant2 < -0.9] = -0.9
            mean = np.mean(constant2)
            constant2 = constant2 + abs(mean) if mean < 0 else constant2 - abs(mean)
    if vvse == 2:
        selected = frames_w_multi[int(rng.integers(0, frames_w_multi.size))]
        indices = np.flatnonzero(frames == selected)
        shuffled = indices[np.asarray(rng.permutation(indices.size), dtype=np.intp)]
        order[indices] = order[shuffled]
    return constant, constant2, order, persistent_constant2ft


def run_mcmc(
    ksu: Any,
    localizations_final: ArrayLike,
    frame_information: ArrayLike,
    a: int,
    resolution: Real,
    true_localizations: ArrayLike,
    bins: ArrayLike,
    distribution_for_blink: ArrayLike,
    steps: int,
    timer: Real,
    constantf: ArrayLike,
    orderf: ArrayLike,
    likelihood_history: ArrayLike,
    score_history: ArrayLike,
    number_of_localizations: ArrayLike,
    probability_cache: ProbabilityCache,
    step_total: int,
    addon: Any,
    constant2ft: ArrayLike,
    x_overall: ArrayLike,
    m_mat: ArrayLike,
    *,
    rng: RNGLike | None = None,
    clock: Callable[[], float] = monotonic,
) -> MCMCResult:
    """Port ``DDC_MCMC.m`` without plotting or file/pipeline orchestration."""

    del addon  # Function input is inactive in the MATLAB source.
    generator: RNGLike = np.random.default_rng() if rng is None else rng
    locations = np.asarray(localizations_final).copy()
    frames = np.asarray(frame_information).reshape(-1, order="F").copy()
    truth = np.asarray(true_localizations)
    bin_edges = np.asarray(bins, dtype=np.float64).reshape(-1, order="F").copy()
    blink = np.asarray(distribution_for_blink, dtype=np.float64).reshape(-1, order="F").copy()
    if locations.ndim != 2 or locations.shape[0] != frames.size:
        raise ValueError("LocalizationsFinal rows must match Frame_Information")
    cache = ProbabilityCache(
        _full(probability_cache.deviation_in_probability),
        _full(probability_cache.prob_distributions),
        _full(probability_cache.dscale_store),
    )
    if not steps < step_total:
        raise ValueError("MATLAB leaves required outputs unassigned when steps>=step_total")

    indices = np.argsort(frames, kind="stable")
    frames = frames[indices]
    locations = locations[indices, :]

    maximum_frame = int(np.max(frames))
    distance_matrix = [np.asarray([], dtype=np.float64) for _ in range(maximum_frame + 1)]
    distance_matrix = make_distance_matrix_individual(
        locations, distance_matrix, frames, maximum_frame
    )
    controls: list[FloatArray] = []
    for matlab_bin in range(1, maximum_frame + 1):
        values = distance_matrix[matlab_bin - 1].reshape(-1, order="F")
        if values.size > 100 and matlab_bin > a:
            controls.append(values)
    distance_control = np.concatenate(controls) if controls else np.asarray([])
    bin_edges, blink, true_distribution, grouped = _prepare_distributions(
        distance_control, bin_edges, blink
    )
    blink22, bins22, true22 = grouped

    fresh = np.asarray(orderf).size == 0
    if fresh:
        deviation, probabilities, dscale = determine_deviation_in_probability8(
            true_distribution,
            blink,
            bin_edges,
            a,
            locations,
            frames,
            [],
            [],
            resolution,
            true22,
            blink22,
            bins22,
            x_overall,
            m_mat,
            rng=generator,  # randperm hidden in the helper occurs here.
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            probabilities = np.log(probabilities)
        cache = ProbabilityCache(deviation, probabilities, dscale)
    else:
        deviation = cache.deviation_in_probability.copy()
        probabilities = cache.prob_distributions.copy()

    truth_enabled = _matlab_length(truth) > 1
    true = np.empty((0, locations.shape[1]), dtype=locations.dtype)
    true_frames = np.asarray([], dtype=frames.dtype)
    if truth_enabled:
        mask = np.asarray(
            [np.any(row[0] == truth[:, 0]) for row in locations], dtype=bool
        )
        true = locations[mask, :]
        true_frames = frames[mask]
        truth_image = palm_density_image(true[:, :2], true[:, :2])

    initial_d = pdist(locations)
    initial_z = _frame_pair_distances(frames)
    keep = initial_z < a
    initial_d, initial_z = initial_d[keep], initial_z[keep]
    keep = initial_z > 0
    initial_d, initial_z = initial_d[keep], initial_z[keep]
    if truth_enabled:
        truth_lik = calculate_score_8(
            true, true_frames, a, initial_d, initial_z, resolution, probabilities
        )
        bestlik = truth_lik.lik + truth_lik.lik2
    else:
        bestlik = 0.0

    locations, d_overall, z2, _ = jitter_unique_distances(
        locations, frames, a, generator
    )
    scorey = 0.0
    max_scorey = 0.0
    if truth_enabled:
        initial_image = palm_density_image(locations[:, :2], true[:, :2])
        difference = matlab_mat2gray(initial_image) - matlab_mat2gray(truth_image)
        scorey = float(np.sum(np.abs(difference) ** 2))
        max_scorey = scorey

    frames_w_multi = determine_locs_w_multi2(locations, frames, resolution)
    _, density = density_calc(locations, frames, resolution, a)
    matlab_length = _matlab_matrix_length(locations)
    unsorted = np.arange(matlab_length, dtype=np.int64)
    traj_num = np.asarray(generator.permutation(matlab_length), dtype=np.float64) + 1
    traj2 = traj_num.copy()

    if fresh:
        order = np.arange(frames.size, dtype=np.int64)
        accepted_constant = np.zeros(frames.size)
        persistent_constant2ft = np.zeros(frames.size)
        histories: list[float] = []
        scores: list[float] = []
        numbers: list[int] = []
        max_lik = -np.inf
        best_locations = locations.copy()
        best_frames = frames.copy()
    else:
        order = np.asarray(orderf, dtype=np.int64).reshape(-1, order="F") - 1
        accepted_constant = np.asarray(constantf, dtype=np.float64).reshape(-1, order="F").copy()
        persistent_constant2ft = np.asarray(constant2ft, dtype=np.float64).reshape(-1, order="F").copy()
        histories = np.asarray(likelihood_history, dtype=float).reshape(-1).tolist()
        scores = np.asarray(score_history, dtype=float).reshape(-1).tolist()
        numbers = np.asarray(number_of_localizations, dtype=int).reshape(-1).tolist()
        max_lik = float(np.max(histories))
        reordered = locations[order, :]
        kappas = accepted_constant[order] + persistent_constant2ft[order]
        kappas[kappas < -1] = -0.9999
        best_locations, best_frames, traj2 = _eliminate_blinking_de_loc15(
            reordered, frames, kappas, resolution, a, deviation, traj_num
        )
        traj2 = traj2[order]

    current_locations = locations.copy()
    current_frames = frames.copy()
    current_traj = traj_num.copy()
    final_lik = -np.inf
    start = clock()
    while steps < step_total:
        if clock() - start > timer:
            break
        constant = accepted_constant.copy()
        constant2 = persistent_constant2ft.copy()
        proposal_order = order.copy()
        steps += 1
        previous_working_loc = current_locations.copy()
        while np.array_equal(current_locations, previous_working_loc):
            constant, constant2, proposal_order, persistent_constant2ft = _proposal_parameters(
                constant,
                constant2,
                proposal_order,
                accepted_constant,
                persistent_constant2ft,
                frames_w_multi,
                frames,
                density,
                unsorted,
                steps,
                generator,
            )
            kappas = constant + constant2
            kappas[kappas < -1] = -0.9999
            reordered = locations[proposal_order, :]
            proposed_loc, proposed_frames, proposed_traj = _eliminate_blinking_de_loc15(
                reordered,
                frames,
                kappas[proposal_order],
                resolution,
                a,
                deviation,
                traj_num,
            )
            proposed_traj = proposed_traj[proposal_order]
            current_locations = proposed_loc
        likelihood = calculate_score_8(
            proposed_loc,
            proposed_frames,
            a,
            d_overall,
            z2,
            resolution,
            probabilities,
        )
        proposed_lik = likelihood.lik + likelihood.lik2
        acceptance_draw = 1.0
        if proposed_lik < final_lik:
            acceptance_draw = float(generator.random())
        accepted = metropolis_accept(proposed_lik, final_lik, acceptance_draw)
        if accepted:
            accepted_constant = constant.copy()
            persistent_constant2ft = constant2.copy()
            order = proposal_order.copy()
            final_lik = proposed_lik
            current_frames = proposed_frames.copy()
            current_traj = proposed_traj.copy()
            if proposed_lik > max_lik:
                best_locations = proposed_loc.copy()
                best_frames = proposed_frames.copy()
                traj2[indices] = proposed_traj
                max_lik = proposed_lik
                final_lik = proposed_lik
                steps = 10
            if truth_enabled:
                score11 = palm_image_squared_error(proposed_loc[:, :2], true[:, :2])
                with np.errstate(divide="ignore", invalid="ignore"):
                    scorey = float(np.divide(score11, max_scorey))
            histories.append(final_lik)
            scores.append(scorey)
            numbers.append(_matlab_matrix_length(proposed_loc))

    return MCMCResult(
        traj2=np.asarray(traj2, dtype=np.float64),
        final_localizations_blinking_corrected=np.asarray(best_locations, dtype=np.float64),
        final_frame_blinking_corrected=np.asarray(best_frames),
        likelihood_history=np.asarray(histories, dtype=np.float64),
        score_history=np.asarray(scores, dtype=np.float64),
        number_of_localizations=np.asarray(numbers, dtype=np.float64),
        bestlik=float(bestlik),
        steps=int(steps),
        constantf=accepted_constant,
        orderf=order + 1,
        probability_cache=cache,
        constant2ft=persistent_constant2ft,
        final_lik=float(final_lik),
        max_lik=float(max_lik),
        jittered_localizations=locations,
        d_overall=d_overall,
        z2=z2,
        working_localizations=np.asarray(current_locations, dtype=np.float64),
    )
