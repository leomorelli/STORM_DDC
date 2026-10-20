"""Exact accepted-B.3 differential tests for the B.5 distance fast path."""

from __future__ import annotations

import numpy as np
import pytest

from ddc.distances import _matlab_scalar_subtract, make_distance_matrix_individual


def _b3_reference(localizations, distance_matrix, frame, b):
    locations=np.asarray(localizations);frames=np.asarray(frame)
    if locations.ndim!=2:raise ValueError("localizations must have shape (n_localizations, n_dimensions)")
    if frames.ndim>2 or (frames.ndim==2 and min(frames.shape,default=0)>1):raise ValueError("frame must be a MATLAB-style row or column vector")
    frames=frames.reshape(-1,order="F")
    if locations.shape[0]!=frames.size:raise ValueError("localizations and frame must contain the same number of rows")
    if not isinstance(b,(int,float,np.integer,np.floating)) or not np.isfinite(b):raise ValueError("b must be a finite real number")
    indices=np.argsort(frames,kind="stable");sf=frames[indices];sl=locations[indices,:]
    result=[np.asarray(cell).reshape(-1,order="F").copy() for cell in distance_matrix]
    for i in range(sl.shape[0]):
        for jj in range(i,sf.size):
            backward=_matlab_scalar_subtract(sf[i],sf[jj]);forward=_matlab_scalar_subtract(sf[jj],sf[i]);absolute=abs(backward)
            if absolute<b+1 and forward>0:
                if not isinstance(absolute,(int,np.integer)) and not float(absolute).is_integer():raise IndexError("an accepted frame difference is not a valid MATLAB cell index")
                cell=int(absolute)
                if cell<1:raise IndexError("DistanceMatrix{0} is invalid under MATLAB 1-based indexing")
                if cell>len(result):raise IndexError(f"DistanceMatrix{{{cell}}} does not exist")
                point=sl[i,:]-sl[jj,:];distance=np.sum(point**2)**0.5
                result[cell-1]=np.append(result[cell-1],distance)
    return result


def _compare(locations,frames,b,cells=None):
    if cells is None:cells=[np.array([],dtype=float) for _ in range(int(b)+1)]
    try:expected=_b3_reference(locations,cells,frames,b);expected_error=None
    except Exception as exc:expected_error=(type(exc),str(exc));expected=None
    try:actual=make_distance_matrix_individual(locations,cells,frames,b);actual_error=None
    except Exception as exc:actual_error=(type(exc),str(exc));actual=None
    assert actual_error==expected_error
    if expected_error is None:
        assert len(actual)==len(expected)
        for left,right in zip(actual,expected,strict=True):
            assert left.dtype==right.dtype and left.shape==right.shape
            np.testing.assert_array_equal(left,right)
            if np.issubdtype(left.dtype,np.inexact):np.testing.assert_array_equal(left.view(f"u{left.dtype.itemsize}"),right.view(f"u{right.dtype.itemsize}"))


@pytest.mark.parametrize("dtype",[np.float32,np.float64])
@pytest.mark.parametrize("dimensions",[2,3])
def test_boundaries_order_prepopulation_and_stable_sort(dtype,dimensions):
    loc=np.zeros((8,dimensions),dtype=dtype)
    loc[:,0]=np.array([7,0,3,3,-0.,9,7,2],dtype=dtype)
    frames=np.array([4,1,2,2,-0.,5,4,6],dtype=dtype)
    cells=[np.array([99.],dtype=dtype),np.array([],dtype=dtype),np.array([88.],dtype=dtype),np.array([],dtype=dtype)]
    _compare(loc,frames,3,cells)


def test_empty_single_nonintegral_and_nonfinite_fallbacks():
    _compare(np.empty((0,2)),np.array([],dtype=float),3)
    _compare(np.array([[1.,2.]]),np.array([-0.]),3)
    _compare(np.array([[0.,0.],[1.,0.]]),np.array([1.,1.5]),1)
    for value in (np.nan,np.inf,-np.inf):
        _compare(np.array([[0.,0.],[1.,0.]]),np.array([0.,value]),3)
        with np.errstate(invalid="ignore"):_compare(np.array([[0.,0.],[value,0.]]),np.array([0.,2.]),3)


@pytest.mark.parametrize("dtype",[np.uint16,np.int8,np.int16,np.int32,np.int64])
def test_integer_fallback_and_bug001(dtype):
    _compare(np.array([[0.,0.],[3.,4.]]),np.array([2,4],dtype=dtype),3)
    if dtype==np.uint16:
        with pytest.raises(IndexError,match=r"DistanceMatrix\{0\}"):
            make_distance_matrix_individual(np.array([[0.,0.],[3.,4.]]),[np.array([]) for _ in range(3)],np.array([2,4],dtype=dtype),3)


@pytest.mark.parametrize("dtype",[np.float32,np.float64])
def test_distance_bit_patterns_duplicated_coordinates(dtype):
    tiny=np.nextafter(dtype(0),dtype(1),dtype=dtype);large=np.finfo(dtype).max/4
    loc=np.array([[0,0,0],[-0.,tiny,-tiny],[3,4,0],[3,4,0],[large,0,0]],dtype=dtype)
    frames=np.array([0,1,2,3,4],dtype=dtype)
    with np.errstate(over="ignore",invalid="ignore"):_compare(loc,frames,4)


def test_1000_randomized_float_differentials():
    rng=np.random.default_rng(20260819)
    for _ in range(1000):
        n=int(rng.integers(0,35));dimensions=int(rng.choice([2,3]));dtype=rng.choice([np.float32,np.float64]);b=int(rng.integers(1,9))
        loc=(rng.normal(size=(n,dimensions))*rng.choice([.01,1.,100.])).astype(dtype)
        frames=rng.integers(-5,20,size=n).astype(dtype)
        if n and rng.random()<.5:frames=frames[rng.permutation(n)]
        cells=[rng.normal(size=int(rng.integers(0,3))).astype(dtype) for _ in range(b+1)]
        _compare(loc,frames,b,cells)
