"""Stage-A-vs-B.2 exact-pair-matching benchmark and residual profile."""

from __future__ import annotations

import cProfile
import csv
from dataclasses import asdict
import io
import json
from pathlib import Path
import pstats
import statistics
import time
from unittest.mock import patch

import numpy as np
import ddc.distributions as distributions_module
import ddc.likelihood as likelihood_module
from ddc.likelihood import calculate_score_8, matlab_intersect_first
from scipy.spatial.distance import pdist

from benchmarks.fixtures import environment, fixture

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/"benchmarks/results"
FROZEN_SHA="a54496a4c08591fee307314ef95a2046b8c4c4e59c710ec913549691a1441279"


def stage_a_intersect(a,b):
    left=np.asarray(a).reshape(-1,order="F"); right=np.asarray(b).reshape(-1,order="F")
    records=[]
    for li,value in enumerate(left):
        if np.issubdtype(left.dtype,np.inexact) and np.isnan(value): continue
        if any(value==record[0] for record in records): continue
        ri=None
        for candidate_index,candidate in enumerate(right):
            if np.issubdtype(right.dtype,np.inexact) and np.isnan(candidate): continue
            if value==candidate: ri=candidate_index; break
        if ri is not None: records.append((value,li,ri))
    records.sort(key=lambda record:record[0]); dtype=np.result_type(left.dtype,right.dtype)
    return (np.asarray([r[0] for r in records],dtype=dtype),
            np.asarray([r[1] for r in records],dtype=np.int64),
            np.asarray([r[2] for r in records],dtype=np.int64))


def timed(fn,repetitions):
    fn(); values=[]
    for _ in range(repetitions):
        start=time.perf_counter(); fn(); values.append(time.perf_counter()-start)
    return {"repetitions":repetitions,"min_seconds":min(values),
            "median_seconds":statistics.median(values),"mean_seconds":statistics.mean(values),
            "std_seconds":statistics.pstdev(values)}


def comparison(name,case,n,before_fn,after_fn,repetitions=3,extra=None):
    before=timed(before_fn,repetitions); after=timed(after_fn,50)
    row={"name":name,"case":case,"n_localizations":n,"stage_a":before,"stage_b2":after,
         "absolute_difference_seconds":after["median_seconds"]-before["median_seconds"],
         "speedup":before["median_seconds"]/after["median_seconds"]}
    if extra:row.update(extra)
    return row


def score_inputs(n,fraction):
    f=fixture(n); overall=pdist(f.locations)
    z2=pdist(np.column_stack((np.zeros(n),f.frames))); keep=(z2<2)&(z2>0)
    k=max(2,int(n*fraction)); probabilities=np.log(np.full((2,200),1/200))
    return f.locations[:k],f.frames[:k],overall[keep],z2[keep],probabilities


def score_call(args,intersect):
    with patch.object(likelihood_module,"matlab_intersect_first",intersect):
        return calculate_score_8(args[0],args[1],2,args[2],args[3],10.,args[4])


def profile_residual(args):
    real_pdist=pdist; pdist_time=0.; pdist_calls=0; intersect_time=0.; intersect_calls=0
    def counted_pdist(*call_args,**kwargs):
        nonlocal pdist_time,pdist_calls
        start=time.perf_counter(); result=real_pdist(*call_args,**kwargs)
        pdist_time+=time.perf_counter()-start; pdist_calls+=1; return result
    real_intersect=matlab_intersect_first
    def counted_intersect(*call_args,**kwargs):
        nonlocal intersect_time,intersect_calls
        start=time.perf_counter(); result=real_intersect(*call_args,**kwargs)
        intersect_time+=time.perf_counter()-start; intersect_calls+=1; return result
    profiler=cProfile.Profile(); start=time.perf_counter(); profiler.enable()
    with patch.object(likelihood_module,"pdist",counted_pdist), patch.object(
        distributions_module,"pdist",counted_pdist), patch.object(
        likelihood_module,"matlab_intersect_first",counted_intersect):
        calculate_score_8(args[0],args[1],2,args[2],args[3],10.,args[4])
    profiler.disable(); total=time.perf_counter()-start
    stream=io.StringIO(); stats=pstats.Stats(profiler,stream=stream).strip_dirs()
    stream.write("TOP 30 BY CUMULATIVE TIME\n");stats.sort_stats("cumulative").print_stats(30)
    stream.write("\nTOP 30 BY SELF TIME\n");stats.sort_stats("tottime").print_stats(30)
    return {"total_seconds":total,"pdist_calls":pdist_calls,"pdist_seconds":pdist_time,
            "pdist_fraction":pdist_time/total,"intersect_calls":intersect_calls,
            "intersect_seconds":intersect_time,"intersect_fraction":intersect_time/total,
            "indexing_deletion_probability_remainder_seconds":max(0.,total-pdist_time-intersect_time)},stream.getvalue()


def main():
    RESULTS.mkdir(parents=True,exist_ok=True); started=time.perf_counter(); rows=[]
    rng=np.random.default_rng(19)
    cases={"unique_partial":(np.arange(1000.),np.arange(0,1000.,2)),
           "duplicate_heavy":(rng.integers(0,200,1000).astype(float),rng.integers(0,200,500).astype(float)),
           "no_match":(np.arange(1000.),np.arange(1000.,1500.)),
           "high_match":(np.arange(1000.),np.arange(900.))}
    for label,(left,right) in cases.items():
        rows.append(comparison("matlab_intersect_first",label,len(left),
            lambda l=left,r=right:stage_a_intersect(l,r),
            lambda l=left,r=right:matlab_intersect_first(l,r),5,
            {"left_size":len(left),"right_size":len(right)}))
    intersection_scaling=[]
    for n in (100,250,500,1000,2000,4000):
        left=np.arange(float(n));right=np.arange(0.,float(n),2.)
        result=timed(lambda:matlab_intersect_first(left,right),50)
        intersection_scaling.append({"n":n,**result})
    full=[]
    for n in (50,100,200,400):
        for fraction in (.25,.5,.75,1.):
            args=score_inputs(n,fraction)
            row=comparison("calculate_score_8",f"retained_{fraction:.2f}",len(args[1]),
                lambda a=args:score_call(a,stage_a_intersect),lambda a=args:score_call(a,matlab_intersect_first),3,
                {"source_n":n,"retained_fraction":fraction})
            rows.append(row)
            if fraction==1.:full.append(row)
    args=score_inputs(800,1.)
    row=comparison("calculate_score_8","retained_1.00",800,
        lambda a=args:score_call(a,stage_a_intersect),lambda a=args:score_call(a,matlab_intersect_first),3,
        {"source_n":800,"retained_fraction":1.})
    rows.append(row);full.append(row)
    before_p=float(np.polyfit(np.log([r["n_localizations"] for r in full]),
                              np.log([r["stage_a"]["median_seconds"] for r in full]),1)[0])
    after_p=float(np.polyfit(np.log([r["n_localizations"] for r in full]),
                             np.log([r["stage_b2"]["median_seconds"] for r in full]),1)[0])
    intersect_p=float(np.polyfit(np.log([r["n"] for r in intersection_scaling]),
                                 np.log([r["median_seconds"] for r in intersection_scaling]),1)[0])
    residual,profile_text=profile_residual(score_inputs(400,1.))
    result={"environment":environment(),"frozen_stage_a_sha256":FROZEN_SHA,
            "methodology":{"warmup":1,"same_process_before_after":True,
                           "stage_a_score_repetitions":3,"stage_a_intersection_repetitions":5,
                           "stage_b2_repetitions":50},
            "comparisons":rows,"scaling":{"stage_a_score":before_p,"stage_b2_score":after_p,
                "stage_b2_intersection":intersect_p,"intersection_rows":intersection_scaling},
            "optimized_score_profile":residual,"total_benchmark_runtime_seconds":time.perf_counter()-started}
    (RESULTS/"stage_b2_likelihood.json").write_text(json.dumps(result,indent=2)+"\n")
    with (RESULTS/"stage_b2_likelihood.csv").open("w",newline="") as handle:
        keys=["name","case","source_n","n_localizations","retained_fraction","left_size","right_size",
              "stage_a_median_seconds","stage_b2_median_seconds","absolute_difference_seconds","speedup"]
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader()
        for row in rows:
            writer.writerow({**{k:row.get(k) for k in keys},
                "stage_a_median_seconds":row["stage_a"]["median_seconds"],
                "stage_b2_median_seconds":row["stage_b2"]["median_seconds"]})
    (RESULTS/"stage_b2_profile_top.txt").write_text(profile_text)
    print(json.dumps({"rows":len(rows),"scaling":result["scaling"],"profile":residual,
                      "seconds":result["total_benchmark_runtime_seconds"]},indent=2))


if __name__=="__main__":main()
