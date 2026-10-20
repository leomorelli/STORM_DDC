"""Gate-0 redundancy analysis and B.2-vs-B.4 MCMC benchmark."""

from __future__ import annotations

import ast,cProfile,csv,io,json,pstats,statistics,sys,time,tracemalloc
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import numpy as np

import ddc.mcmc as mcmc
from ddc.distributions import determine_blinking_distribution
from ddc.mcmc import ProbabilityCache,run_mcmc
from benchmarks.fixtures import environment

ROOT=Path(__file__).resolve().parents[1];RESULTS=ROOT/"benchmarks/results"
STAGE_A_SHA="a54496a4c08591fee307314ef95a2046b8c4c4e59c710ec913549691a1441279"
B2_SHA="61c19e3af0a368b5e28fc65449e5c065cd50841478d6242d47a9d087cd3c42e8"


def load_b2_reference():
    path=ROOT/"tests/test_b4_equivalence.py";tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_b2_eliminator")
    namespace=dict(mcmc.__dict__);exec(compile(ast.Module([node],[]),str(path),"exec"),namespace)
    return namespace["_b2_eliminator"]


def fixture(n):
    g=(n-1)//2;frames=np.r_[np.ones(g),2.,np.full(n-g-1,4.)]
    x=np.r_[np.arange(g)*10,.2,np.arange(n-g-1)*10+.5]
    return np.c_[x,np.zeros(n)],frames


def arguments(n,target=15):
    loc,frames=fixture(n);bins,blink,_,resolution,xo,mm=determine_blinking_distribution([loc],[frames],2,10.)
    return (1,loc,frames,2,resolution,np.empty((0,0)),bins,blink,0,float("inf"),[],[],[],[],[],
            ProbabilityCache(np.array([]),np.array([]),np.array([])),target,0,[],xo,mm)


class Instrumentation:
    def __init__(self,eliminator):
        self.eliminator=eliminator;self.counts={k:0 for k in
          ("eliminator","pairwise_coordinate","frame_cross","proposal","likelihood","density","membership_scans",
           "candidate_evaluations","probability_evaluations","trajectory_merges","repeated_coordinate_groups_local",
           "repeated_frame_groups_local")}
        self.seconds={k:0. for k in ("eliminator","pairwise_coordinate","frame_cross","proposal","likelihood","density")}
        self.group_sizes=[];self.group_counts=[];self._seen_coord=set();self._seen_frame=set();self._in_frame=False
        source=(ROOT/"tests/test_b4_equivalence.py").read_text().splitlines()
        self.lines={i+1:key for i,line in enumerate(source) for text,key in
          (("source_group=(np.flatnonzero","membership_scans"),("target_group=(np.flatnonzero","membership_scans"),
           ("target=source+investigate","candidate_evaluations"),("probability=np.mean","probability_evaluations"),
           ("is_blink[target]=1;linked[source]=1","trajectory_merges")) if text in line}
        self.reference_code=eliminator.__code__
    def trace(self,frame,event,arg):
        if event=="line" and frame.f_code is self.reference_code and frame.f_lineno in self.lines:
            self.counts[self.lines[frame.f_lineno]]+=1
        return self.trace
    def wrap_eliminator(self,*args,**kwargs):
        self.counts["eliminator"]+=1;self._seen_coord.clear();self._seen_frame.clear()
        trajectory=np.asarray(args[6]);self.group_counts.append(int(np.unique(trajectory[trajectory!=0]).size))
        start=time.perf_counter();out=self.eliminator(*args,**kwargs);self.seconds["eliminator"]+=time.perf_counter()-start
        self.group_counts.append(int(np.unique(out[2][out[2]!=0]).size));return out
    def wrap_pair(self,real):
        def call(left,right):
            key=(tuple(left.shape),left.tobytes(),tuple(right.shape),right.tobytes())
            if self._in_frame:
                self.counts["repeated_frame_groups_local"]+=key in self._seen_frame;self._seen_frame.add(key)
                return real(left,right)
            self.counts["pairwise_coordinate"]+=1;self.counts["repeated_coordinate_groups_local"]+=key in self._seen_coord
            self._seen_coord.add(key);self.group_sizes.extend((left.shape[0],right.shape[0]));start=time.perf_counter()
            out=real(left,right);self.seconds["pairwise_coordinate"]+=time.perf_counter()-start;return out
        return call
    def wrap_frame(self,real):
        def call(left,right):
            key=(tuple(left.shape),left.tobytes(),tuple(right.shape),right.tobytes())
            self.counts["frame_cross"]+=1
            self.counts["repeated_frame_groups_local"]+=key in self._seen_frame;self._seen_frame.add(key)
            start=time.perf_counter();self._in_frame=True
            try:return real(left,right)
            finally:self._in_frame=False;self.seconds["frame_cross"]+=time.perf_counter()-start
        return call
    def simple(self,module,name,key):
        real=getattr(module,name)
        def call(*args,**kwargs):
            self.counts[key]+=1;start=time.perf_counter()
            try:return real(*args,**kwargs)
            finally:self.seconds[key]+=time.perf_counter()-start
        return patch.object(module,name,call)


def execute(eliminator,n,seed,target=15,trace=False,memory=False):
    ins=Instrumentation(eliminator);real_pair=mcmc._matlab_pairwise_expression;real_frame=mcmc._frame_cross_distances
    pair_wrapper=ins.wrap_pair(real_pair);frame_wrapper=ins.wrap_frame(real_frame)
    function_globals=eliminator.__globals__
    saved_pair=function_globals["_matlab_pairwise_expression"]
    saved_frame=function_globals["_frame_cross_distances"]
    function_globals["_matlab_pairwise_expression"]=pair_wrapper
    function_globals["_frame_cross_distances"]=frame_wrapper
    with ExitStack() as stack:
        stack.enter_context(patch.object(mcmc,"_eliminate_blinking_de_loc15",ins.wrap_eliminator))
        stack.enter_context(ins.simple(mcmc,"_proposal_parameters","proposal"))
        stack.enter_context(ins.simple(mcmc,"calculate_score_8","likelihood"))
        stack.enter_context(ins.simple(mcmc,"density_calc","density"))
        if memory:tracemalloc.start()
        if trace:sys.settrace(ins.trace)
        start=time.perf_counter();result=run_mcmc(*arguments(n,target),rng=np.random.default_rng(seed));total=time.perf_counter()-start
        sys.settrace(None)
        peak=None
        if memory:_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    function_globals["_matlab_pairwise_expression"]=saved_pair
    function_globals["_frame_cross_distances"]=saved_frame
    outer=ins.counts["likelihood"];accepted=len(result.likelihood_history)
    return {"total_seconds":total,"outer_iterations":outer,"proposal_attempts":ins.counts["proposal"],
      "accepted_proposals":accepted,"rejected_proposals":outer-accepted,
      "strict_new_maxima":int(np.sum(result.likelihood_history>np.maximum.accumulate(np.r_[-np.inf,result.likelihood_history[:-1]]))),
      "counts":ins.counts,"component_seconds":ins.seconds,"peak_tracemalloc_bytes":peak,
      "median_group_size":statistics.median(ins.group_sizes) if ins.group_sizes else 0,
      "max_group_size":max(ins.group_sizes,default=0),"group_counts":ins.group_counts,
      "candidate_search_membership_remainder_seconds":max(0.,ins.seconds["eliminator"]-ins.seconds["pairwise_coordinate"]-ins.seconds["frame_cross"])}


def profile_b4(eliminator):
    profiler=cProfile.Profile();profiler.enable();execute(eliminator,23,150,60);profiler.disable()
    stream=io.StringIO();stats=pstats.Stats(profiler,stream=stream).strip_dirs()
    stream.write("TOP 30 BY CUMULATIVE TIME\n");stats.sort_stats("cumulative").print_stats(30)
    stream.write("\nTOP 30 BY SELF TIME\n");stats.sort_stats("tottime").print_stats(30)
    return stream.getvalue()


def main():
    RESULTS.mkdir(parents=True,exist_ok=True);start=time.perf_counter();reference=load_b2_reference();optimized=mcmc._eliminate_blinking_de_loc15
    gate0=execute(reference,23,150,60,trace=True,memory=True);rows=[]
    for n in (23,50,100):
        for label,fn in (("B.2",reference),("B.4",optimized)):
            samples=[execute(fn,n,700+n,15,memory=(rep==0)) for rep in range(7)]
            rows.append({"version":label,"n_localizations":n,"repetitions":7,
              "median_run_mcmc_seconds":statistics.median(x["total_seconds"] for x in samples),
              "median_eliminator_seconds":statistics.median(x["component_seconds"]["eliminator"] for x in samples),
              "median_eliminator_call_seconds":statistics.median(x["component_seconds"]["eliminator"]/max(x["counts"]["eliminator"],1) for x in samples),
              "peak_tracemalloc_bytes":samples[0]["peak_tracemalloc_bytes"],"representative":samples[1]})
    comparisons=[]
    for n in (23,50,100):
        old=next(r for r in rows if r["version"]=="B.2" and r["n_localizations"]==n)
        new=next(r for r in rows if r["version"]=="B.4" and r["n_localizations"]==n)
        comparisons.append({"n_localizations":n,"run_mcmc_speedup":old["median_run_mcmc_seconds"]/new["median_run_mcmc_seconds"],
          "eliminator_speedup":old["median_eliminator_seconds"]/new["median_eliminator_seconds"],
          "memory_ratio":new["peak_tracemalloc_bytes"]/old["peak_tracemalloc_bytes"]})
    profile=profile_b4(optimized);(RESULTS/"stage_b4_profile_top.txt").write_text(profile)
    result={"environment":environment(),"frozen_stage_a_sha256":STAGE_A_SHA,"accepted_b2_review_sha256":B2_SHA,
      "gate0_post_b2":gate0,"benchmarks":rows,"comparisons":comparisons,
      "accepted_optimization":"singleton-trajectory membership fast path with immediate original-scan fallback after any merge or for nonunique/zero labels",
      "rejected_candidates":{"general_membership_mapping":"eager and lazy mappings were neutral/slower on measured workloads",
        "coordinate_cache":"zero local hits in Gate 0","frame_cache":"2860 local repeats but made representative MCMC slower",
        "candidate_traversal":"higher risk; remainder did not justify semantic change","pairwise_arithmetic":"bitwise risk; unchanged"},
      "total_benchmark_runtime_seconds":time.perf_counter()-start}
    (RESULTS/"stage_b4_mcmc.json").write_text(json.dumps(result,indent=2)+"\n")
    with (RESULTS/"stage_b4_mcmc.csv").open("w",newline="") as handle:
        keys=["version","n_localizations","repetitions","median_run_mcmc_seconds","median_eliminator_seconds","median_eliminator_call_seconds","peak_tracemalloc_bytes"]
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows([{k:r[k] for k in keys} for r in rows])
    print(json.dumps({"gate0":gate0,"comparisons":comparisons,"seconds":result["total_benchmark_runtime_seconds"]},indent=2))


if __name__=="__main__":main()
