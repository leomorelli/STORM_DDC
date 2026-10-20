"""Exact equivalence guards for Stage-B.3 blocked Density_Calc."""

from __future__ import annotations
from dataclasses import fields
from unittest.mock import patch
import numpy as np
import pytest

import ddc.mcmc as mcmc_module
from ddc.distributions import determine_blinking_distribution
from ddc.io import DDCDataset
from ddc.mcmc import MCMCResult,ProbabilityCache,run_mcmc
from ddc.pipeline import DDCConfig,run_ddc
from ddc.trajectories import (_matlab_abs_frame_difference,_matlab_matrix_length,
    _density_blocked_counts,density_calc)


def _b4_raw(locations,frames,resolution,a):
    n=_matlab_matrix_length(locations);counts=np.zeros((n,1),dtype=np.float64)
    if n>locations.shape[0]:raise IndexError("MATLAB Density_Calc indexes beyond LocalizationsFinal rows because length(LocalizationsFinal)=max(n,d)")
    for i in range(n):
        for j in range(n):
            point=locations[i,:]-locations[j,:]
            with np.errstate(over="ignore",invalid="ignore"):distance=np.sqrt(np.sum(point**2))
            difference=_matlab_abs_frame_difference(frames[i],frames[j],frames.dtype)
            if distance<=2*resolution and difference>a:counts[i,0]+=1
    return counts


def _b4_density(localizations,frame_information,resolution,a):
    locations=np.asarray(localizations);frames=np.asarray(frame_information).reshape(-1,order="F")
    if locations.ndim!=2 or locations.shape[0]!=frames.size:raise ValueError
    distance=2*resolution;density=_b4_raw(locations,frames,resolution,a)
    if density.size==0:return distance,density
    minimum=np.min(density);denominator=np.max(density)-minimum
    with np.errstate(divide="ignore",invalid="ignore"):density=(density-minimum)/denominator
    return distance,density


def _compare(loc,frames,resolution,a):
    try:expected=_b4_density(np.asarray(loc),np.asarray(frames),resolution,a);error=None
    except Exception as exc:error=(type(exc),str(exc))
    try:actual=density_calc(loc,frames,resolution,a);actual_error=None
    except Exception as exc:actual_error=(type(exc),str(exc))
    assert actual_error==error
    if error is None:
        assert actual[0]==expected[0];assert actual[1].dtype==np.float64;assert actual[1].shape==expected[1].shape
        np.testing.assert_array_equal(actual[1],expected[1]);np.testing.assert_array_equal(np.isnan(actual[1]),np.isnan(expected[1]))
        raw=_b4_raw(np.asarray(loc),np.asarray(frames).reshape(-1,order="F"),resolution,a)
        if np.asarray(loc).dtype in (np.float32,np.float64) and np.asarray(frames).dtype in (np.float32,np.float64) and np.all(np.isfinite(loc)) and np.all(np.isfinite(frames)):
            np.testing.assert_array_equal(_density_blocked_counts(np.asarray(loc),np.asarray(frames),2*resolution,a),raw)


@pytest.mark.parametrize("dtype,dimensions",[(np.float64,2),(np.float64,3),(np.float32,2),(np.float32,3)])
def test_distance_and_frame_nextafter_boundaries(dtype,dimensions):
    resolution=dtype(5);boundary=dtype(10);below=np.nextafter(boundary,dtype(0),dtype=dtype);above=np.nextafter(boundary,dtype(np.inf),dtype=dtype)
    for distance in (below,boundary,above):
        loc=np.zeros((4,dimensions),dtype=dtype);loc[1,0]=distance;loc[2,0]=dtype(1);loc[3,0]=dtype(2)
        for frame_value in (np.nextafter(dtype(2),dtype(0),dtype=dtype),dtype(2),np.nextafter(dtype(2),dtype(np.inf),dtype=dtype),dtype(-2)):
            _compare(loc,np.array([0,frame_value,4,6],dtype=dtype),resolution,dtype(2))


@pytest.mark.parametrize("dtype",[np.uint8,np.uint16,np.uint32,np.int8,np.int16,np.int32])
def test_integer_frame_fallback_and_bug004(dtype):
    loc=np.array([[0.,0.],[1.,0.],[2.,0.]])
    frames=np.array([3,1,5],dtype=dtype);_compare(loc,frames,10.,1)
    raw=_b4_raw(loc,frames,10.,1)
    np.testing.assert_array_equal(raw,_b4_raw(loc,frames,10.,1))
    assert not np.array_equal(raw,raw[::-1]) or np.all(raw==raw[0])


def test_uint16_bug004_ordered_asymmetry_is_preserved():
    loc=np.array([[0.,0.],[1.,0.]])
    frames=np.array([1,3],dtype=np.uint16)
    raw=_b4_raw(loc,frames,10.,1)
    np.testing.assert_array_equal(raw,np.array([[0.],[1.]]))
    np.testing.assert_array_equal(_density_blocked_counts(loc,frames.astype(float),20.,1),np.array([[1.],[1.]]))
    _compare(loc,frames,10.,1)


def test_constant_empty_length_and_nonfinite_fallbacks():
    _compare(np.empty((0,2)),np.array([],dtype=float),1.,1)
    _compare(np.zeros((4,2)),np.arange(4,dtype=float),1.,99)
    _compare(np.zeros((4,2)),np.array([0.,2.,4.,6.]),1.,1)
    with pytest.raises(IndexError):density_calc(np.zeros((2,3)),np.array([1.,2.]),1.,1)
    for value in (np.nan,np.inf,-np.inf):
        loc=np.array([[0.,0.],[value,0.],[1.,0.]])
        with np.errstate(invalid="ignore"):_compare(loc,np.array([-0.,0.,3.]),1.,1)


@pytest.mark.parametrize("block_size",[1,2,3,7,32,64,128,256,512])
def test_internal_block_sizes_are_exact(block_size):
    rng=np.random.default_rng(441+block_size)
    loc=rng.normal(size=(37,3));frames=rng.integers(-3,9,size=37).astype(float)
    with patch("ddc.trajectories._DENSITY_BLOCK_SIZE",block_size):_compare(loc,frames,1.25,2.)


@pytest.mark.parametrize("dtype",[np.float32,np.float64])
def test_multicomponent_sum_near_distance_boundary(dtype):
    resolution=dtype(5);root=dtype(np.sqrt(dtype(50)))
    for component in (np.nextafter(root,dtype(0),dtype=dtype),root,np.nextafter(root,dtype(np.inf),dtype=dtype)):
        loc=np.array([[0,0,0],[component,component,0],[1,2,3]],dtype=dtype)
        _compare(loc,np.array([0,3,6],dtype=dtype),resolution,dtype(1))


def test_bit_pattern_stress_float32_float64():
    for dtype in (np.float32,np.float64):
        tiny=np.nextafter(dtype(0),dtype(1),dtype=dtype);large=np.finfo(dtype).max/4
        loc=np.array([[0,0],[tiny,-tiny],[1,-1],[large,0],[-large,0]],dtype=dtype)
        frames=np.array([-0.,0.,tiny,large,-large],dtype=dtype);_compare(loc,frames,dtype(1),dtype(0))


def test_1000_randomized_density_differentials():
    rng=np.random.default_rng(20260819)
    for _ in range(1000):
        n=int(rng.integers(1,22));d=int(rng.choice([2,3]));dtype=rng.choice([np.float32,np.float64])
        scale=float(rng.choice([.01,1.,100.]));loc=(rng.normal(size=(n,d))*scale).astype(dtype)
        frames=np.sort(rng.integers(-3,12,size=n)).astype(dtype)
        _compare(loc,frames,dtype(rng.choice([.1,1.,10.])),dtype(rng.integers(0,4)))


class TraceRNG:
    def __init__(self,seed):self.g=np.random.default_rng(seed);self.trace=[]
    def call(self,name,*a,**k):
        value=getattr(self.g,name)(*a,**k);x=np.asarray(value);self.trace.append((name,repr(a),repr(k),x.dtype.str,x.shape,x.tobytes()));return value
    def random(self,*a,**k):return self.call("random",*a,**k)
    def normal(self,*a,**k):return self.call("normal",*a,**k)
    def integers(self,*a,**k):return self.call("integers",*a,**k)
    def permutation(self,*a,**k):return self.call("permutation",*a,**k)


def _fixture(n):
    g=(n-1)//2;frames=np.r_[np.ones(g),2.,np.full(n-g-1,4.)];x=np.r_[np.arange(g)*10,.2,np.arange(n-g-1)*10+.5]
    return np.c_[x,np.zeros(n)],frames


def _run(seed,n):
    loc,frame=_fixture(n);bins,blink,_,resolution,xo,mm=determine_blinking_distribution([loc],[frame],2,10.);rng=TraceRNG(seed)
    result=run_mcmc(1,loc,frame,2,resolution,np.empty((0,0)),bins,blink,0,float("inf"),[],[],[],[],[],ProbabilityCache(np.array([]),np.array([]),np.array([])),15,0,[],xo,mm,rng=rng)
    return result,rng.trace


def _assert_mcmc(a:MCMCResult,b:MCMCResult):
    for field in fields(a):
        x=getattr(a,field.name);y=getattr(b,field.name)
        if isinstance(x,ProbabilityCache):
            for f in fields(x):np.testing.assert_array_equal(getattr(y,f.name),getattr(x,f.name))
        elif isinstance(x,np.ndarray):np.testing.assert_array_equal(y,x)
        else:assert y==x


@pytest.mark.parametrize("n,seeds",[(23,range(10)),(50,range(5)),(100,range(2))])
def test_mcmc_and_rng_exact_for_required_seeds(n,seeds):
    for seed in seeds:
        with patch.object(mcmc_module,"density_calc",_b4_density):expected,trace1=_run(seed+500,n)
        actual,trace2=_run(seed+500,n);assert trace2==trace1;_assert_mcmc(expected,actual)


@pytest.mark.parametrize("images",[1,2])
def test_pipeline_single_multi_exact(images):
    loc,frame=_fixture(23);data=DDCDataset([loc.copy() for _ in range(images)],[frame.copy() for _ in range(images)],[np.empty((0,2)) for _ in range(images)],photons=[np.ones(23) for _ in range(images)],addon=np.zeros((1,images)))
    config=DDCConfig(resolution=10.,stepper=10,n_f=2,timer=30.)
    with patch.object(mcmc_module,"density_calc",_b4_density):expected=run_ddc(data,config,rngs=[np.random.default_rng(900+i) for i in range(images)])
    actual=run_ddc(data,config,rngs=[np.random.default_rng(900+i) for i in range(images)])
    assert actual.steps.tolist()==expected.steps.tolist()
    for old,new in zip(expected.states,actual.states,strict=True):
        for name in ("constantf","constant2ft","orderf","likelihood","score","number_of_localizations"):np.testing.assert_array_equal(getattr(new,name),getattr(old,name))
        for name in ("deviation_in_probability","prob_distributions","dscale_store"):np.testing.assert_array_equal(getattr(new.probability_cache,name),getattr(old.probability_cache,name))
    for name in ("final_localizations","final_frames","trajectories"):
        for old,new in zip(getattr(expected,name),getattr(actual,name),strict=True):np.testing.assert_array_equal(new,old)
