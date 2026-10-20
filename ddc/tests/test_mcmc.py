from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from ddc.mcmc import (
    MCMCProposal,
    MCMCState,
    ProbabilityCache,
    _proposal_parameters,
    apply_proposal_transition,
    jitter_unique_distances,
    metropolis_accept,
    run_controlled_transitions,
    run_mcmc,
)


class ControlledRNG:
    def __init__(self, *, random=(), normal=(), integers=(), permutations=()):
        self.random_values = list(random)
        self.normal_values = list(normal)
        self.integer_values = list(integers)
        self.permutations = [np.asarray(value) for value in permutations]
        self.calls: list[str] = []

    def random(self, size=None):
        self.calls.append(f"random:{size}")
        value = self.random_values.pop(0)
        if size is None:
            return value
        array = np.asarray(value, dtype=float)
        return np.full(size, value) if array.ndim == 0 else array

    def normal(self, size=None):
        self.calls.append(f"normal:{size}")
        value = self.normal_values.pop(0)
        return value if size is None else np.full(size, value)

    def integers(self, low, high=None, size=None):
        self.calls.append(f"integers:{low}:{high}:{size}")
        value = self.integer_values.pop(0)
        return value if size is None else np.full(size, value)

    def permutation(self, x):
        self.calls.append(f"permutation:{x}")
        return self.permutations.pop(0)


def _state(*, final=0.0, maximum=0.0, steps=0) -> MCMCState:
    return MCMCState(
        constant=np.array([0.0, 0.0]),
        constant2ft=np.array([0.0, 0.0]),
        order=np.array([0, 1]),
        current_localizations=np.array([[0.0, 0.0], [1.0, 0.0]]),
        current_frames=np.array([1.0, 2.0]),
        current_trajectory=np.array([1.0, 2.0]),
        final_lik=final,
        best_localizations=np.array([[0.0, 0.0], [1.0, 0.0]]),
        best_frames=np.array([1.0, 2.0]),
        best_trajectory=np.array([1.0, 2.0]),
        max_lik=maximum,
        steps=steps,
        likelihood_history=[],
        score_history=[],
        number_history=[],
    )


def _proposal(likelihood: float, marker: float = 5.0) -> MCMCProposal:
    return MCMCProposal(
        constant=np.array([marker, 0.0]),
        constant2=np.array([0.0, marker]),
        order=np.array([1, 0]),
        localizations=np.array([[marker, 0.0]]),
        frames=np.array([2.0]),
        trajectory=np.array([2.0, 1.0]),
        likelihood=likelihood,
        score=0.25,
    )


@pytest.mark.parametrize(
    ("new", "current", "draw", "expected"),
    [
        (2.0, 1.0, 0.999, True),
        (1.0, 1.0, 0.999, True),
        (0.5, 1.0, 0.5, True),
        (0.0, 1.0, 0.9, False),
        (0.0, 1.0, 0.0, True),
        (np.nan, 1.0, 0.0, False),
        (np.inf, 1.0, 0.9, True),
        (-np.inf, 1.0, 0.0, True),
        (0.0, 0.0, 0.9, True),
    ],
)
def test_acceptance_expression_edges(new, current, draw, expected) -> None:
    assert metropolis_accept(new, current, draw) is expected


def test_accepted_downhill_can_reduce_final_lik_without_reducing_maximum() -> None:
    state = _state(final=10.0, maximum=12.0, steps=20)
    accepted = apply_proposal_transition(state, _proposal(9.5), random_value=0.5)

    assert accepted
    assert state.final_lik == 9.5
    assert state.max_lik == 12.0
    np.testing.assert_array_equal(state.best_localizations, [[0.0, 0.0], [1.0, 0.0]])
    np.testing.assert_array_equal(state.current_localizations, [[5.0, 0.0]])


def test_postscore_helper_rejection_leaves_supplied_state_unchanged() -> None:
    state = _state(final=10.0, maximum=12.0, steps=20)
    before = replace(
        state,
        constant=state.constant.copy(),
        current_localizations=state.current_localizations.copy(),
        best_localizations=state.best_localizations.copy(),
    )
    assert not apply_proposal_transition(state, _proposal(0.0), random_value=0.9)
    np.testing.assert_array_equal(state.constant, before.constant)
    np.testing.assert_array_equal(state.current_localizations, before.current_localizations)
    np.testing.assert_array_equal(state.best_localizations, before.best_localizations)
    assert state.final_lik == before.final_lik and state.max_lik == before.max_lik
    assert state.likelihood_history == []


def test_new_maximum_updates_best_and_resets_steps_exactly_to_ten() -> None:
    state = _state(final=1.0, maximum=1.0, steps=37)
    assert apply_proposal_transition(state, _proposal(2.0, 8.0), random_value=0.9)
    assert state.steps == 10
    assert state.final_lik == state.max_lik == 2.0
    np.testing.assert_array_equal(state.best_localizations, [[8.0, 0.0]])


def test_accepted_nonmaximum_does_not_overwrite_best_snapshot() -> None:
    state = _state(final=5.0, maximum=10.0, steps=20)
    apply_proposal_transition(state, _proposal(6.0, 9.0), random_value=0.9)
    assert state.final_lik == 6.0 and state.max_lik == 10.0
    np.testing.assert_array_equal(state.best_localizations, [[0.0, 0.0], [1.0, 0.0]])


def test_while_boundary_and_rejected_moves_increment_steps() -> None:
    state = _state(final=10.0, maximum=10.0, steps=2)
    proposals = [_proposal(0.0), _proposal(0.0), _proposal(0.0)]
    run_controlled_transitions(state, proposals, 4, [0.9, 0.9, 0.9])
    assert state.steps == 4
    assert state.likelihood_history == []


def test_new_maximum_reset_can_change_loop_progress_nonmonotonically() -> None:
    state = _state(final=1.0, maximum=1.0, steps=12)
    run_controlled_transitions(state, [_proposal(2.0), _proposal(1.5)], 14, [0.9, 0.5])
    assert state.steps == 11
    assert state.likelihood_history == [2.0, 1.5]
    assert state.max_lik == 2.0


def test_exact_jitter_uniqueness_retries_and_preserves_pair_order() -> None:
    first = np.zeros((3, 2))
    second = np.array([[0.0, 0.0], [0.001, 0.0], [0.0, 0.003]])
    rng = ControlledRNG(random=[first, second])
    locations, distances, z2, attempts = jitter_unique_distances(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], [1.0, 2.0, 3.0], 4, rng
    )
    assert attempts == 2
    assert np.unique(distances).size == distances.size
    np.testing.assert_array_equal(z2, [1.0, 2.0, 1.0])
    np.testing.assert_allclose(locations, np.array([[0, 0], [1.00001, 0], [2, 0.00003]]))
    assert rng.calls == ["random:(3, 2)", "random:(3, 2)"]


def test_seeded_jitter_is_deterministic_and_does_not_mutate_inputs() -> None:
    locations = np.array([[0.0, 0.0], [2.0, 0.0]])
    frames = np.array([1.0, 2.0], dtype=np.float32)
    before = locations.copy()
    one = jitter_unique_distances(locations, frames, 3, np.random.default_rng(7))
    two = jitter_unique_distances(locations, frames, 3, np.random.default_rng(7))
    np.testing.assert_allclose(one[0], two[0])
    np.testing.assert_allclose(one[1], two[1])
    np.testing.assert_array_equal(locations, before)


def test_proposal_order_branch_uses_randi_then_randperm_without_replacement() -> None:
    rng = ControlledRNG(integers=[2, 0], permutations=[[1, 0]])
    constant, constant2, order, constant2ft = _proposal_parameters(
        np.zeros(3),
        np.zeros(3),
        np.arange(3),
        np.zeros(3),
        np.zeros(3),
        np.array([1.0]),
        np.array([1.0, 1.0, 2.0]),
        np.arange(3.0)[:, None],
        np.arange(3),
        1,
        rng,
    )
    np.testing.assert_array_equal(order, [1, 0, 2])
    np.testing.assert_array_equal(np.sort(order), [0, 1, 2])
    np.testing.assert_array_equal(constant, 0)
    np.testing.assert_array_equal(constant2, 0)
    np.testing.assert_array_equal(constant2ft, 0)
    assert rng.calls == ["integers:1:4:None", "integers:0:1:None", "permutation:2"]


def test_empty_multi_frame_set_remaps_randi_two_to_one_without_more_draws() -> None:
    rng = ControlledRNG(integers=[2])
    _, _, order, _ = _proposal_parameters(
        np.zeros(1), np.zeros(1), np.array([0]), np.zeros(1), np.zeros(1),
        np.array([]), np.array([1.0]), np.array([[np.nan]]), np.array([0]), 1, rng
    )
    np.testing.assert_array_equal(order, [0])
    assert rng.calls == ["integers:1:4:None"]


def test_single_branch3_attempt_snapshots_preperturbation_constant2ft() -> None:
    rng = ControlledRNG(random=[0.4], normal=[1.0], integers=[3, 0])
    constant, proposed2, order, persistent2 = _proposal_parameters(
        np.zeros(2), np.zeros(2), np.arange(2), np.zeros(2), np.zeros(2),
        np.array([]), np.array([1.0, 2.0]), np.array([[0.0], [1.0]]),
        np.arange(2), 2, rng,
    )
    np.testing.assert_array_equal(constant, [0.0, 0.0])
    np.testing.assert_allclose(proposed2, [0.05, -0.05])
    np.testing.assert_array_equal(persistent2, [0.0, 0.0])
    np.testing.assert_array_equal(order, [0, 1])


def _run_full_loop_with_controlled_numerics(
    likelihoods,
    eliminated_locations,
    rng,
    *,
    step_total,
    timer=100.0,
    clock=lambda: 0.0,
    proposal_wrapper=None,
):
    locations = np.array([[0.0, 0.0], [1.0, 0.0]])
    frames = np.array([1.0, 1.0])
    cache = ProbabilityCache(np.ones((2, 2)), np.zeros((2, 2)), np.ones(2))
    eliminated = iter(eliminated_locations)
    scores = iter(likelihoods)

    def fake_eliminate(*args, **kwargs):
        loc = np.asarray(next(eliminated), dtype=float)
        return loc, np.ones(loc.shape[0]), np.array([1.0, 2.0])

    def fake_score(*args, **kwargs):
        return SimpleNamespace(lik=float(next(scores)), lik2=0.0)

    original_proposal = _proposal_parameters
    target_proposal = original_proposal if proposal_wrapper is None else proposal_wrapper
    with (
        patch("ddc.mcmc._prepare_distributions", return_value=(
            np.array([0.0, np.inf]), np.array([1.0]), np.array([1.0]),
            (np.array([1.0]), np.array([0.0, np.inf]), np.array([1.0])),
        )),
        patch("ddc.mcmc.jitter_unique_distances", return_value=(
            locations.copy(), np.array([]), np.array([]), 1,
        )),
        patch("ddc.mcmc.determine_locs_w_multi2", return_value=np.array([1.0])),
        patch("ddc.mcmc.density_calc", return_value=(2.0, np.array([[0.0], [1.0]]))),
        patch("ddc.mcmc._eliminate_blinking_de_loc15", side_effect=fake_eliminate),
        patch("ddc.mcmc.calculate_score_8", side_effect=fake_score),
        patch("ddc.mcmc._proposal_parameters", side_effect=target_proposal),
    ):
        return run_mcmc(
            1, locations, frames, 2, 1.0, np.array([]), [0.0, np.inf], [1.0],
            0, timer, np.zeros(2), np.array([1, 2]), [5.0], [], [], cache,
            step_total, [], np.zeros(2), [1.0, 1.0], np.ones((2, 2)),
            rng=rng, clock=clock,
        )


def test_run_mcmc_main_loop_uphill_then_accepted_downhill_and_histories() -> None:
    loc0 = np.array([[0.0, 0.0], [1.0, 0.0]])
    loc_best = np.array([[10.0, 0.0]])
    loc_downhill = np.array([[20.0, 0.0]])
    rng = ControlledRNG(
        random=[0.0], integers=[2, 0, 2, 0],
        permutations=[[0, 1], [1, 0], [1, 0]],
    )
    result = _run_full_loop_with_controlled_numerics(
        [10.0, 9.0], [loc0, loc_best, loc_downhill], rng, step_total=11
    )

    assert result.max_lik == 10.0 and result.final_lik == 9.0
    assert result.steps == 11
    np.testing.assert_array_equal(result.final_localizations_blinking_corrected, loc_best)
    np.testing.assert_array_equal(result.working_localizations, loc_downhill)
    np.testing.assert_array_equal(result.likelihood_history, [5.0, 10.0, 9.0])
    # MATLAB length(1x2) is 2, not the coordinate row count 1.
    np.testing.assert_array_equal(result.number_of_localizations, [2.0, 2.0])
    assert result.score_history.size == 2


def test_run_mcmc_main_loop_rejected_downhill_preserves_constantf_orderf_only() -> None:
    loc0 = np.array([[0.0, 0.0], [1.0, 0.0]])
    loc_best = np.array([[10.0, 0.0]])
    loc_rejected = np.array([[30.0, 0.0]])
    rng = ControlledRNG(
        random=[0.9], integers=[2, 0, 2, 0],
        permutations=[[0, 1], [1, 0], [1, 0]],
    )
    result = _run_full_loop_with_controlled_numerics(
        [10.0, 0.0], [loc0, loc_best, loc_rejected], rng, step_total=11
    )

    np.testing.assert_array_equal(result.likelihood_history, [5.0, 10.0])
    np.testing.assert_array_equal(result.final_localizations_blinking_corrected, loc_best)
    np.testing.assert_array_equal(result.working_localizations, loc_rejected)
    np.testing.assert_array_equal(result.constantf, [0.0, 0.0])
    np.testing.assert_array_equal(result.orderf, [2, 1])
    assert result.max_lik == result.final_lik == 10.0


def test_run_mcmc_main_loop_constant2ft_leaks_then_starts_next_iteration() -> None:
    loc0 = np.array([[0.0, 0.0], [1.0, 0.0]])
    loc_best = np.array([[10.0, 0.0]])
    loc_same = loc_best.copy()
    loc_rejected = np.array([[20.0, 0.0]])
    loc_next = np.array([[30.0, 0.0]])
    starts = []
    original = _proposal_parameters

    def recording_proposal(*args, **kwargs):
        starts.append(np.asarray(args[1]).copy())
        return original(*args, **kwargs)

    rng = ControlledRNG(
        random=[0.4, 0.4, 0.9, 0.9],
        normal=[1.0, 1.0],
        integers=[2, 0, 3, 0, 3, 1, 2, 0],
        permutations=[[0, 1], [1, 0], [1, 0]],
    )
    result = _run_full_loop_with_controlled_numerics(
        [10.0, 0.0, 0.0],
        [loc0, loc_best, loc_same, loc_rejected, loc_next],
        rng,
        step_total=12,
        proposal_wrapper=recording_proposal,
    )

    assert len(starts) == 4
    np.testing.assert_allclose(starts[2], [0.05, -0.05])
    np.testing.assert_allclose(starts[3], [0.05, -0.05])
    np.testing.assert_allclose(result.constant2ft, [0.05, -0.05])
    np.testing.assert_array_equal(result.constantf, [0.0, 0.0])
    np.testing.assert_array_equal(result.orderf, [2, 1])
    np.testing.assert_array_equal(result.working_localizations, loc_next)
    np.testing.assert_array_equal(result.likelihood_history, [5.0, 10.0])
    assert rng.calls == [
        "permutation:2",
        "integers:1:4:None", "integers:0:1:None", "permutation:2",
        "integers:1:4:None", "integers:0:2:None", "normal:None", "random:None",
        "integers:1:4:None", "integers:0:2:None", "normal:None", "random:None",
        "random:None",
        "integers:1:4:None", "integers:0:1:None", "permutation:2",
        "random:None",
    ]


def test_run_mcmc_timer_uses_injected_clock_before_step_increment() -> None:
    values = iter([0.0, 2.0])
    clock = lambda: next(values)
    rng = ControlledRNG(permutations=[[0, 1]])
    loc0 = np.array([[0.0, 0.0], [1.0, 0.0]])
    result = _run_full_loop_with_controlled_numerics(
        [], [loc0], rng, step_total=1, timer=1.0, clock=clock
    )
    assert result.steps == 0
    np.testing.assert_array_equal(result.likelihood_history, [5.0])
    assert rng.calls == ["permutation:2"]


def _driver_fixture(frame_dtype=np.float64):
    frames = np.concatenate((np.ones(11), np.array([2.0]), np.full(11, 3.0))).astype(frame_dtype)
    x = np.concatenate((np.arange(11.0) * 10, [0.2], np.arange(11.0) * 10 + 0.5))
    locations = np.column_stack((x, np.zeros(x.size)))
    bins = np.array([0.0, 20.0, 60.0, np.inf])
    blink = np.array([0.6, 0.3, 0.1])
    cache = ProbabilityCache(
        deviation_in_probability=np.full((1, 3), 0.9),
        prob_distributions=np.log(np.array([[0.6, 0.3, 0.1]])),
        dscale_store=np.array([1.0]),
    )
    return locations, frames, bins, blink, cache


def test_cached_driver_initialization_stable_sort_jitter_and_shapes() -> None:
    locations, frames, bins, blink, cache = _driver_fixture()
    permutation = np.r_[np.arange(11, 22), 11, np.arange(11)]
    unsorted_frames = frames[permutation]
    unsorted_locations = locations[permutation]
    before = unsorted_locations.copy()
    result = run_mcmc(
        1, unsorted_locations, unsorted_frames, 1, 10.0, np.array([]),
        bins, blink, 0, -1.0, np.zeros(23), np.arange(1, 24), [0.0], [], [],
        cache, 1, [], np.zeros(23), [1.0], np.full((1, 3), 0.9),
        rng=np.random.default_rng(4),
    )
    assert np.all(np.diff(result.final_frame_blinking_corrected) >= 0)
    assert result.jittered_localizations.shape == locations.shape
    assert np.unique(result.d_overall).size == result.d_overall.size
    assert result.constantf.shape == result.constant2ft.shape == (23,)
    assert result.orderf.dtype == np.int64
    np.testing.assert_array_equal(unsorted_locations, before)


def test_uncached_probability_initialization_path_is_deterministic() -> None:
    frames = np.concatenate((np.ones(11), [2.0], np.full(11, 4.0)))
    x = np.concatenate((np.arange(11.0) * 10, [0.2], np.arange(11.0) * 10 + 0.5))
    locations = np.column_stack((x, np.zeros(x.size)))
    bins = np.array([0.0, 20.0, 60.0, np.inf])
    blink = np.array([0.6, 0.3, 0.1])
    empty_cache = ProbabilityCache(np.empty((0, 0)), np.empty((0, 0)), np.array([]))
    kwargs = dict(
        ksu=1, localizations_final=locations, frame_information=frames, a=2,
        resolution=10.0, true_localizations=np.array([]), bins=bins,
        distribution_for_blink=blink, steps=0, timer=-1.0, constantf=[], orderf=[],
        likelihood_history=[], score_history=[], number_of_localizations=[],
        probability_cache=empty_cache, step_total=1, addon=[], constant2ft=[],
        x_overall=[0.5, 1.0], m_mat=np.full((2, 3), 0.9),
    )
    one = run_mcmc(**kwargs, rng=np.random.default_rng(9))
    two = run_mcmc(**kwargs, rng=np.random.default_rng(9))
    np.testing.assert_allclose(one.jittered_localizations, two.jittered_localizations)
    np.testing.assert_allclose(
        one.probability_cache.prob_distributions,
        two.probability_cache.prob_distributions,
    )


def test_truth_scoring_enabled_and_disabled_initialization() -> None:
    locations, frames, bins, blink, cache = _driver_fixture()
    common = dict(
        ksu=1, localizations_final=locations, frame_information=frames, a=1,
        resolution=10.0, bins=bins, distribution_for_blink=blink, steps=0,
        timer=-1.0, constantf=np.zeros(23), orderf=np.arange(1, 24),
        likelihood_history=[0.0], score_history=[], number_of_localizations=[],
        probability_cache=cache, step_total=1, addon=[], constant2ft=np.zeros(23),
        x_overall=[1.0], m_mat=np.full((1, 3), 0.9),
    )
    absent = run_mcmc(**common, true_localizations=np.array([]), rng=np.random.default_rng(2))
    enabled = run_mcmc(**common, true_localizations=locations[[0, 12]], rng=np.random.default_rng(2))
    assert absent.bestlik == 0.0
    assert np.isfinite(enabled.bestlik)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_frame_paths_reach_driver_initialization(dtype) -> None:
    locations, frames, bins, blink, cache = _driver_fixture(dtype)
    result = run_mcmc(
        1, locations, frames, 1, 10.0, np.array([]), bins, blink, 0, -1,
        np.zeros(23), np.arange(1, 24), [0.0], [], [], cache, 1, [],
        np.zeros(23), [1.0], np.full((1, 3), 0.9), rng=np.random.default_rng(3)
    )
    assert result.final_frame_blinking_corrected.dtype == np.dtype(dtype)


def test_repository_uint16_path_fails_first_at_source_bug_001() -> None:
    locations, frames, bins, blink, cache = _driver_fixture(np.uint16)
    with pytest.raises(IndexError, match=r"DistanceMatrix\{0\}"):
        run_mcmc(
            1, locations, frames, 1, 10.0, np.empty((0, 2)), bins, blink, 0, -1,
            np.zeros(23), np.arange(1, 24), [0.0], [], [], cache, 1, [],
            np.zeros(23), [1.0], np.full((1, 3), 0.9), rng=np.random.default_rng(3)
        )


def test_one_localization_and_empty_control_reproduce_source_max_failure() -> None:
    cache = ProbabilityCache(np.ones((1, 1)), np.zeros((1, 1)), np.ones(1))
    with pytest.raises(ValueError):
        run_mcmc(
            1, [[0.0, 0.0]], [1.0], 1, 1.0, np.empty((0, 2)), [0, np.inf],
            [1.0], 0, -1, [0.0], [1], [0.0], [], [], cache, 1, [], [0.0],
            [1.0], [[1.0]], rng=np.random.default_rng(1)
        )


def test_matlab_mcmc_oracles_when_available() -> None:
    from pathlib import Path
    from scipy.io import loadmat
    path = Path(__file__).parent / "matlab" / "mcmc_initialization_reference.mat"
    if not path.exists():
        pytest.skip(
            "MATLAB runtime is unavailable; run tests/matlab/generate_mcmc_reference.m"
        )
    reference = loadmat(path, squeeze_me=True, struct_as_record=False)
    assert bool(reference["frame_is_sorted"])
    assert bool(reference["d_overall_is_unique"])
    assert np.asarray(reference["Final_Localizations_Blinking_Corrected"]).shape == (23, 2)
    assert np.asarray(reference["Final_Frame_Blinking_Corrected"]).size == 23


def test_matlab_acceptance_probe_when_available() -> None:
    from pathlib import Path
    from scipy.io import loadmat
    path = Path(__file__).parent / "matlab" / "mcmc_acceptance_probe.mat"
    if not path.exists():
        pytest.skip(
            "MATLAB runtime is unavailable; run tests/matlab/probe_mcmc_acceptance.m"
        )
    reference = loadmat(path, squeeze_me=True)
    observed = [
        metropolis_accept(float(new), float(current), float(draw))
        for new, current, draw in zip(
            np.ravel(reference["new_lik"]),
            np.ravel(reference["current_lik"]),
            np.ravel(reference["controlled_rand"]),
        )
    ]
    np.testing.assert_array_equal(observed, np.ravel(reference["accepted"]))
    assert int(reference["steps_after_maximum"]) == 10


def test_matlab_constant2ft_leak_probe_when_available() -> None:
    from pathlib import Path
    from scipy.io import loadmat
    path = Path(__file__).parent / "matlab" / "mcmc_constant2ft_leak_probe.mat"
    if not path.exists():
        pytest.skip(
            "MATLAB runtime is unavailable; run "
            "tests/matlab/probe_mcmc_constant2ft_leak.m"
        )
    reference = loadmat(path, squeeze_me=True)
    np.testing.assert_array_equal(reference["first_snapshot"], [0.0, 0.0])
    np.testing.assert_array_equal(reference["second_snapshot"], [0.1, 0.0])
    np.testing.assert_array_equal(
        reference["constant2ft_after_rejection"], [0.1, 0.0]
    )
    np.testing.assert_array_equal(
        reference["Constant2_next_iteration"], [0.1, 0.0]
    )
