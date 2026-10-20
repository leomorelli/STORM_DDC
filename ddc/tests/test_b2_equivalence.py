"""Differential guards for the Stage-B.2 exact-intersection optimization."""

from __future__ import annotations

from dataclasses import fields
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.distance import pdist

import ddc.likelihood as likelihood_module
from ddc.distributions import determine_blinking_distribution
from ddc.io import DDCDataset
from ddc.likelihood import calculate_score_8, matlab_intersect_first
from ddc.mcmc import MCMCResult, ProbabilityCache, run_mcmc
from ddc.pipeline import DDCConfig, DDCPipelineResult, run_ddc


def _stage_a_intersect(a, b):
    """Test-only verbatim behavioral oracle from frozen Stage A."""
    left = np.asarray(a).reshape(-1, order="F")
    right = np.asarray(b).reshape(-1, order="F")
    records = []
    for left_index, value in enumerate(left):
        if np.issubdtype(left.dtype, np.inexact) and np.isnan(value):
            continue
        if any(value == existing[0] for existing in records):
            continue
        right_index = None
        for candidate_index, candidate in enumerate(right):
            if np.issubdtype(right.dtype, np.inexact) and np.isnan(candidate):
                continue
            if value == candidate:
                right_index = candidate_index
                break
        if right_index is not None:
            records.append((value, left_index, right_index))
    records.sort(key=lambda record: record[0])
    c = np.asarray([r[0] for r in records], dtype=np.result_type(left.dtype, right.dtype))
    return c, np.asarray([r[1] for r in records], dtype=np.int64), np.asarray(
        [r[2] for r in records], dtype=np.int64
    )


def _assert_intersection_equal(left, right):
    expected = _stage_a_intersect(left, right)
    actual = matlab_intersect_first(left, right)
    assert actual[0].dtype == expected[0].dtype
    np.testing.assert_array_equal(actual[0], expected[0])
    if np.issubdtype(actual[0].dtype, np.floating):
        np.testing.assert_array_equal(np.signbit(actual[0]), np.signbit(expected[0]))
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])


@pytest.mark.parametrize(("left", "right"), [
    ([], []), ([], [1.]), ([1.], []), ([3.,1.,2.], [2.,3.]),
    ([1.,2.], [3.,4.]), ([3.,1.,2.], [1.,2.,3.]),
    ([3.,1.,3.,2.], [3.,2.]), ([3.,1.], [3.,3.,1.]),
    ([2.,1.,2.], [2.,2.,1.]), ([7.,7.,7.], [7.,7.]),
    ([np.nan,1.,np.nan], [np.nan,1.]), ([np.inf,-np.inf], [-np.inf,np.inf]),
    ([-0.,0.], [0.]), ([0.,-0.], [-0.]),
    ([1.,np.nextafter(1.,np.inf)], [np.nextafter(1.,-np.inf),1.]),
])
def test_explicit_stage_a_intersection_semantics(left, right):
    _assert_intersection_equal(np.asarray(left,dtype=np.float64),np.asarray(right,dtype=np.float64))


@pytest.mark.parametrize("dtype", [np.float32,np.float64,np.int32,np.int64,np.uint16])
def test_randomized_same_dtype_differential(dtype):
    rng=np.random.default_rng(20260819+np.dtype(dtype).itemsize)
    for _ in range(80):
        left=rng.integers(0,20,size=rng.integers(0,45)).astype(dtype)
        right=rng.integers(0,20,size=rng.integers(0,45)).astype(dtype)
        if np.issubdtype(dtype,np.floating):
            specials=np.array([np.nan,np.inf,-np.inf,-0.,0.,1.,np.nextafter(1.,2.)],dtype=dtype)
            left=np.concatenate((left,specials[rng.integers(0,len(specials),size=4)]))
            right=np.concatenate((right,specials[rng.integers(0,len(specials),size=4)]))
        _assert_intersection_equal(left,right)


@pytest.mark.parametrize("dtypes", [(np.float64,np.float32),(np.float32,np.int32),
                                      (np.int64,np.uint16),(np.float64,np.int64)])
def test_mixed_dtype_generic_fallback(dtypes):
    left=np.array([-0.,1.,1.,2.,3.],dtype=dtypes[0])
    right=np.array([3.,2.,1.,0.],dtype=dtypes[1])
    _assert_intersection_equal(left,right)


def _fixture(n=100):
    rng=np.random.default_rng(900+n)
    loc=rng.normal(size=(n,2))*100
    frame=(np.arange(n)//4+1).astype(float)
    overall=pdist(loc); z2=pdist(np.column_stack((np.zeros(n),frame)))
    keep=(z2<2)&(z2>0)
    return loc,frame,overall[keep],z2[keep],np.log(np.full((2,200),1/200))


@pytest.mark.parametrize("fraction", [.25,.5,.75,1.0])
def test_calculate_score_complete_stage_a_equivalence(fraction):
    loc,frame,overall,z2,probabilities=_fixture(100)
    retained=max(2,int(len(frame)*fraction))
    with patch.object(likelihood_module,"matlab_intersect_first",_stage_a_intersect):
        expected=calculate_score_8(loc[:retained],frame[:retained],2,overall,z2,10.,probabilities)
    actual=calculate_score_8(loc[:retained],frame[:retained],2,overall,z2,10.,probabilities)
    for field in fields(actual):
        left=getattr(expected,field.name); right=getattr(actual,field.name)
        if isinstance(left,np.ndarray): np.testing.assert_array_equal(right,left)
        elif np.isnan(left): assert np.isnan(right)
        else: assert right == left


def _pipeline_fixture():
    frames=np.concatenate((np.ones(11),[2.],np.full(11,4.)))
    x=np.concatenate((np.arange(11.)*10,[.2],np.arange(11.)*10+.5))
    loc=np.column_stack((x,np.zeros(x.size)))
    data=DDCDataset([loc],[frames],[np.empty((0,2))],photons=[np.ones(23)],addon=np.zeros((1,1)))
    return loc,frames,data


def _mcmc(seed):
    loc,frame,_=_pipeline_fixture()
    bins,blink,_,resolution,x_overall,m_mat=determine_blinking_distribution([loc],[frame],2,10.)
    return run_mcmc(1,loc,frame,2,resolution,np.empty((0,0)),bins,blink,0,float("inf"),
        [],[],[],[],[],ProbabilityCache(np.array([]),np.array([]),np.array([])),20,0,[],
        x_overall,m_mat,rng=np.random.default_rng(seed))


def _assert_mcmc_equal(expected:MCMCResult,actual:MCMCResult):
    for field in fields(expected):
        left=getattr(expected,field.name); right=getattr(actual,field.name)
        if isinstance(left,ProbabilityCache):
            for cache_field in fields(left):
                np.testing.assert_array_equal(getattr(right,cache_field.name),getattr(left,cache_field.name))
        elif isinstance(left,np.ndarray): np.testing.assert_array_equal(right,left)
        else: assert right == left


def test_controlled_mcmc_trajectory_exact_equivalence():
    with patch.object(likelihood_module,"matlab_intersect_first",_stage_a_intersect):
        expected=_mcmc(51)
    actual=_mcmc(51)
    _assert_mcmc_equal(expected,actual)


def _assert_pipeline_equal(expected:DDCPipelineResult,actual:DDCPipelineResult):
    for name in ("bins","distribution_for_blink","x_overall","m_mat"):
        np.testing.assert_array_equal(getattr(actual,name),getattr(expected,name))
    assert actual.resolution==expected.resolution and actual.diagnostics==expected.diagnostics
    for name in ("final_localizations","final_frames","trajectories","working_localizations",
                 "frames","true_localizations","photons"):
        for left,right in zip(getattr(expected,name),getattr(actual,name),strict=True):
            if left is None: assert right is None
            else: np.testing.assert_array_equal(right,left)
    for left,right in zip(expected.states,actual.states,strict=True):
        assert left.step==right.step
        for name in ("constantf","constant2ft","orderf","likelihood","score","number_of_localizations"):
            np.testing.assert_array_equal(getattr(right,name),getattr(left,name))
        for name in ("deviation_in_probability","prob_distributions","dscale_store"):
            np.testing.assert_array_equal(getattr(right.probability_cache,name),getattr(left.probability_cache,name))


def test_controlled_pipeline_exact_equivalence():
    _,_,data=_pipeline_fixture(); config=DDCConfig(resolution=10.,stepper=10,n_f=2,timer=30.)
    with patch.object(likelihood_module,"matlab_intersect_first",_stage_a_intersect):
        expected=run_ddc(data,config,rngs=[np.random.default_rng(73)])
    actual=run_ddc(data,config,rngs=[np.random.default_rng(73)])
    _assert_pipeline_equal(expected,actual)
