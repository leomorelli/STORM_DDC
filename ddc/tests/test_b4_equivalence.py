"""Exact guards for Stage-B.4 trajectory-membership bookkeeping."""

from __future__ import annotations

from dataclasses import fields
from unittest.mock import patch

import numpy as np
import pytest

import ddc.mcmc as mcmc_module
from ddc.distributions import determine_blinking_distribution
from ddc.io import DDCDataset
from ddc.mcmc import MCMCResult, ProbabilityCache, run_mcmc
from ddc.pipeline import DDCConfig, run_ddc
from ddc.probabilities import _frame_cross_distances, _matlab_matrix_length, _matlab_pairwise_expression


def _b2_eliminator(localizations_final,frame,constants,resolution,a,deviation,traj_num):
    locations=np.asarray(localizations_final);frames=np.asarray(frame).reshape(-1,order="F")
    matlab_length=_matlab_matrix_length(locations)
    if matlab_length>locations.shape[0]:raise IndexError("MATLAB length(LocalizationsFinal) exceeds its row count")
    linked=np.zeros(matlab_length);is_blink=np.zeros(matlab_length)
    trajectory=np.asarray(traj_num,dtype=np.float64).reshape(-1,order="F").copy()
    loc=locations.copy();out_frames=frames.copy()
    for difference in range(1,int(np.floor(float(a)+.5))+1):
        for source in range(frames.size):
            investigate=0
            while frames[source+investigate]-frames[source]<difference:
                investigate+=1
                if source+investigate>=frames.size:break
            if source+investigate>=frames.size:break
            while frames[source+investigate]-frames[source]==difference:
                target=source+investigate
                if linked[source]!=1 and is_blink[target]!=1:
                    source_group=(np.flatnonzero(trajectory==trajectory[source]) if trajectory[source]!=0 else np.array([source]))
                    target_group=(np.flatnonzero(trajectory==trajectory[target]) if trajectory[target]!=0 else np.array([target]))
                    distance=_matlab_pairwise_expression(loc[source_group,:],loc[target_group,:])
                    framer=_frame_cross_distances(frames[source_group],frames[target_group])
                    bins=np.floor(distance.reshape(-1,order="F")/resolution)+1
                    bins[bins>deviation.shape[1]]=deviation.shape[1]
                    if np.max(framer)<a:
                        rows=framer.reshape(-1,order="F")
                        if np.any(rows<1) or np.any(rows!=np.floor(rows)):raise IndexError("MATLAB sub2ind received invalid subscripts")
                        values=deviation[rows.astype(np.intp)-1,bins.astype(np.intp)-1]
                        probability=np.mean(values)/abs(1+constants[target])
                        if probability>.5:
                            is_blink[target]=1;linked[source]=1
                            if trajectory[source]==0:trajectory[source]=traj_num[source]
                            members=(trajectory==trajectory[target] if trajectory[target]!=0 else target)
                            trajectory[members]=trajectory[source]
                investigate+=1
                if source+investigate>=frames.size:break
    keep=is_blink<=0
    return loc[keep,:],out_frames[keep],trajectory


def _compare_call(args):
    try:expected=_b2_eliminator(*args);expected_error=None
    except Exception as exc:expected_error=(type(exc),str(exc))
    try:actual=mcmc_module._eliminate_blinking_de_loc15(*args);actual_error=None
    except Exception as exc:actual_error=(type(exc),str(exc))
    assert actual_error==expected_error
    if expected_error is None:
        for left,right in zip(expected,actual,strict=True):np.testing.assert_array_equal(right,left)


EXPLICIT_SCENARIOS=[
 "no_links","one_link","probability_half","probability_above_half","multiple_targets",
 "repeated_frames","preexisting_groups","source_linked","target_blink","source_zero",
 "target_zero","both_nonzero","merge_groups","noncontiguous_labels","arbitrary_labels",
 "groups_larger_than_one","distance_boundary","below_boundary","above_boundary","max_frame_below_a",
 "max_frame_equal_a","nan_coordinates","inf_coordinates","coordinates_2d","coordinates_3d",
 "float32","float64","integer_bug003","n_lt_d","single_input"
]


@pytest.mark.parametrize("scenario",EXPLICIT_SCENARIOS)
def test_explicit_eliminator_matrix(scenario):
    seed=EXPLICIT_SCENARIOS.index(scenario)+1000;rng=np.random.default_rng(seed)
    n=1 if scenario=="single_input" else 8;d=3 if scenario in ("coordinates_3d","n_lt_d") else 2
    if scenario=="n_lt_d":n,d=2,3
    loc=rng.normal(size=(n,d))*5;frames=np.sort(rng.integers(1,5,size=n)).astype(float)
    if scenario=="repeated_frames":frames[:min(3,n)]=1
    if scenario=="nan_coordinates":loc[0,0]=np.nan
    if scenario=="inf_coordinates":loc[0,0]=np.inf
    if scenario=="distance_boundary" and n>1:loc[1]=loc[0]+np.r_[10.,np.zeros(d-1)]
    if scenario=="below_boundary" and n>1:loc[1]=loc[0]+np.r_[np.nextafter(10.,0.),np.zeros(d-1)]
    if scenario=="above_boundary" and n>1:loc[1]=loc[0]+np.r_[np.nextafter(10.,np.inf),np.zeros(d-1)]
    if scenario=="float32":loc=loc.astype(np.float32);frames=frames.astype(np.float32)
    if scenario=="integer_bug003":frames=frames.astype(np.uint16)
    traj=np.arange(1,n+1,dtype=float)
    if scenario in ("preexisting_groups","both_nonzero","merge_groups","groups_larger_than_one") and n>=4:traj[:2]=7;traj[2:4]=11
    if scenario in ("noncontiguous_labels","arbitrary_labels"):traj=np.resize(np.array([3.,11.,29.]),n)
    if scenario=="source_zero" and n:traj[0]=0
    if scenario=="target_zero" and n>1:traj[1]=0
    deviation=np.full((3,20),.9 if scenario!="probability_half" else .5)
    constants=np.zeros(n)
    args=(loc,frames,constants,10.,3,deviation,traj)
    if scenario=="nan_coordinates":
        with pytest.warns(RuntimeWarning):_compare_call(args)
    else:_compare_call(args)


def test_500_randomized_eliminator_differentials():
    rng=np.random.default_rng(20260819)
    for _ in range(500):
        n=int(rng.integers(1,13));d=int(rng.choice([2,3]));loc=rng.normal(size=(n,d))*20
        frames=np.sort(rng.integers(1,8,size=n)).astype(np.float64)
        a=int(rng.integers(1,4));width=30;deviation=rng.random((a,width))
        constants=rng.normal(0,.25,size=n)
        labels=np.array([0.,1.,3.,7.,11.,29.]);traj=labels[rng.integers(0,len(labels),size=n)]
        _compare_call((loc,frames,constants,10.,a,deviation,traj))


class TraceRNG:
    def __init__(self,seed):self.generator=np.random.default_rng(seed);self.trace=[]
    def _call(self,name,*args,**kwargs):
        value=getattr(self.generator,name)(*args,**kwargs);array=np.asarray(value)
        self.trace.append((name,repr(args),repr(kwargs),array.dtype.str,array.shape,array.tobytes()));return value
    def random(self,*a,**k):return self._call("random",*a,**k)
    def normal(self,*a,**k):return self._call("normal",*a,**k)
    def integers(self,*a,**k):return self._call("integers",*a,**k)
    def permutation(self,*a,**k):return self._call("permutation",*a,**k)


def _fixture(n=23):
    g=(n-1)//2;frames=np.r_[np.ones(g),2.,np.full(n-g-1,4.)]
    x=np.r_[np.arange(g)*10,.2,np.arange(n-g-1)*10+.5];loc=np.c_[x,np.zeros(n)]
    return loc,frames


def _run(seed,n=23):
    loc,frame=_fixture(n);bins,blink,_,resolution,xo,mm=determine_blinking_distribution([loc],[frame],2,10.)
    rng=TraceRNG(seed)
    result=run_mcmc(1,loc,frame,2,resolution,np.empty((0,0)),bins,blink,0,float("inf"),[],[],[],[],[],
        ProbabilityCache(np.array([]),np.array([]),np.array([])),15,0,[],xo,mm,rng=rng)
    return result,rng.trace


def _assert_mcmc(left:MCMCResult,right:MCMCResult):
    for field in fields(left):
        a=getattr(left,field.name);b=getattr(right,field.name)
        if isinstance(a,ProbabilityCache):
            for f in fields(a):np.testing.assert_array_equal(getattr(b,f.name),getattr(a,f.name))
        elif isinstance(a,np.ndarray):np.testing.assert_array_equal(b,a)
        else:assert b==a


@pytest.mark.parametrize("seed",range(10))
def test_multiseed_mcmc_and_rng_trace_exact(seed):
    with patch.object(mcmc_module,"_eliminate_blinking_de_loc15",_b2_eliminator):expected,trace1=_run(seed)
    actual,trace2=_run(seed);assert trace2==trace1;_assert_mcmc(expected,actual)


def test_larger_mcmc_and_rng_trace_exact():
    with patch.object(mcmc_module,"_eliminate_blinking_de_loc15",_b2_eliminator):expected,trace1=_run(91,50)
    actual,trace2=_run(91,50);assert trace2==trace1;_assert_mcmc(expected,actual)


@pytest.mark.parametrize("images",[1,2])
def test_pipeline_single_and_multi_image_exact(images):
    loc,frame=_fixture();data=DDCDataset([loc.copy() for _ in range(images)],[frame.copy() for _ in range(images)],
        [np.empty((0,2)) for _ in range(images)],photons=[np.ones(23) for _ in range(images)],addon=np.zeros((1,images)))
    config=DDCConfig(resolution=10.,stepper=10,n_f=2,timer=30.)
    with patch.object(mcmc_module,"_eliminate_blinking_de_loc15",_b2_eliminator):
        expected=run_ddc(data,config,rngs=[np.random.default_rng(300+i) for i in range(images)])
    actual=run_ddc(data,config,rngs=[np.random.default_rng(300+i) for i in range(images)])
    assert actual.steps.tolist()==expected.steps.tolist()
    for name in ("bins","distribution_for_blink","x_overall","m_mat"):np.testing.assert_array_equal(getattr(actual,name),getattr(expected,name))
    for old,new in zip(expected.states,actual.states,strict=True):
        for name in ("constantf","constant2ft","orderf","likelihood","score","number_of_localizations"):
            np.testing.assert_array_equal(getattr(new,name),getattr(old,name))
        for name in ("deviation_in_probability","prob_distributions","dscale_store"):
            np.testing.assert_array_equal(getattr(new.probability_cache,name),getattr(old.probability_cache,name))
    for name in ("final_localizations","final_frames","trajectories"):
        for old,new in zip(getattr(expected,name),getattr(actual,name),strict=True):np.testing.assert_array_equal(new,old)
