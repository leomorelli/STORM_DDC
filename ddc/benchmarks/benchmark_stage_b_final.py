"""Final same-machine frozen-Stage-A versus optimized-Stage-B benchmark."""

from __future__ import annotations
import cProfile,csv,gc,importlib,importlib.util,io,json,os,pstats,statistics,sys,tempfile,time,tracemalloc,zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.distance import pdist
from benchmarks.fixtures import Fixture,environment

ROOT=Path(__file__).resolve().parents[1];RESULTS=ROOT/"benchmarks/results"
STAGE_A=Path("/workspace/scratch/0ed803830bb9/DDC_PYTHON_STAGE_A_FINAL_UNOPTIMIZED.zip")

def fixture(n):
    """Fixed-work MCMC fixture used by the accepted B.3/B.4 benchmarks."""
    g=(n-1)//2;frames=np.r_[np.ones(g),2.,np.full(n-g-1,4.)]
    x=np.r_[np.arange(g)*10,.2,np.arange(n-g-1)*10+.5]
    return Fixture(f"fixed-work-n{n}",np.c_[x,np.zeros(n)],frames)

def load_stage_a(root):
    with zipfile.ZipFile(STAGE_A) as z:z.extractall(root)
    package=Path(root)/"src/ddc";spec=importlib.util.spec_from_file_location("stagea_ddc",package/"__init__.py",submodule_search_locations=[str(package)])
    module=importlib.util.module_from_spec(spec);sys.modules["stagea_ddc"]=module;spec.loader.exec_module(module)
    return {name:importlib.import_module(f"stagea_ddc.{name}") for name in ("distances","distributions","likelihood","trajectories","mcmc","pipeline","io","serialization")}

def load_stage_b():return {name:importlib.import_module(f"ddc.{name}") for name in ("distances","distributions","likelihood","trajectories","mcmc","pipeline","io","serialization")}

def measure(fn,repetitions=5,memory=False):
    fn();gc.collect();values=[];peak=None
    if memory:tracemalloc.start();fn();_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    for _ in range(repetitions):
        start=time.perf_counter();fn();values.append(time.perf_counter()-start)
    return {"minimum_seconds":min(values),"median_seconds":statistics.median(values),"mean_seconds":statistics.mean(values),"repetitions":repetitions,"peak_tracemalloc_bytes":peak}

def score_args(n):
    f=fixture(n);overall=pdist(f.locations);z=pdist(np.c_[np.zeros(n),f.frames]);keep=(z<2)&(z>0)
    return f.locations,f.frames,overall[keep],z[keep],np.log(np.full((2,300),1/300))

def mcmc_args(modules,n,target=15):
    f=fixture(n);bins,blink,_,resolution,xo,mm=modules["distributions"].determine_blinking_distribution([f.locations],[f.frames],2,10.)
    cache=modules["mcmc"].ProbabilityCache(np.array([]),np.array([]),np.array([]))
    return (1,f.locations,f.frames,2,resolution,np.empty((0,0)),bins,blink,0,float("inf"),[],[],[],[],[],cache,target,0,[],xo,mm)

def dataset_config(modules,n):
    f=fixture(n);data=modules["io"].DDCDataset([f.locations],[f.frames],[np.empty((0,2))],photons=[np.ones(n)],addon=np.zeros((1,1)))
    config=modules["pipeline"].DDCConfig(resolution=10.,stepper=10,n_f=2,timer=float("inf"))
    return data,config

def counted_pipeline(modules,n,seed):
    counts={"outer_mcmc_iterations":0,"proposal_attempts":0}
    real_score=modules["mcmc"].calculate_score_8;real_proposal=modules["mcmc"]._proposal_parameters
    def score(*a,**k):counts["outer_mcmc_iterations"]+=1;return real_score(*a,**k)
    def proposal(*a,**k):counts["proposal_attempts"]+=1;return real_proposal(*a,**k)
    data,config=dataset_config(modules,n);start=time.perf_counter()
    with patch.object(modules["mcmc"],"calculate_score_8",score),patch.object(modules["mcmc"],"_proposal_parameters",proposal):
        result=modules["pipeline"].run_ddc(data,config,rngs=[np.random.default_rng(seed)])
    return time.perf_counter()-start,counts,result

def eliminator_call(modules,n):
    f=fixture(n);constants=np.full(n,100.);deviation=np.ones((2,300));traj=np.arange(1,n+1,dtype=float)
    return lambda:modules["mcmc"]._eliminate_blinking_de_loc15(f.locations,f.frames,constants,10.,2.,deviation,traj)

def serialization_call(modules,n):
    _,_,result=counted_pipeline(modules,n,781+n);data,config=dataset_config(modules,n)
    checkpoint=modules["serialization"].DDCCheckpoint.from_result(result,data,config,"benchmark.mat")
    temporary=tempfile.TemporaryDirectory(prefix="ddc-final-");path=Path(temporary.name)/"state.mat"
    return temporary,path,lambda:modules["serialization"].save_v73_checkpoint(checkpoint,path),lambda:modules["serialization"].load_checkpoint(path)

def profile_final(modules):
    stream=io.StringIO();prepared=mcmc_args(modules,200)
    for label,fn in (("run_mcmc",lambda:modules["mcmc"].run_mcmc(*prepared,rng=np.random.default_rng(9200))),
                     ("run_ddc",lambda:counted_pipeline(modules,200,9400))):
        profiler=cProfile.Profile();profiler.enable();fn();profiler.disable();stats=pstats.Stats(profiler,stream=stream).strip_dirs()
        stream.write(f"\n{label} — TOP 30 BY CUMULATIVE TIME\n");stats.sort_stats("cumulative").print_stats(30)
        stream.write(f"\n{label} — TOP 30 BY SELF TIME\n");stats.sort_stats("tottime").print_stats(30)
    return stream.getvalue()

def main():
    RESULTS.mkdir(parents=True,exist_ok=True);started=time.perf_counter();rows=[];end_to_end=[]
    with tempfile.TemporaryDirectory(prefix="ddc-stage-a-") as extracted:
        versions={"Stage A":load_stage_a(extracted),"Stage B":load_stage_b()}
        component_specs=[]
        for label,modules in versions.items():
            f=fixture(200);cells=[np.array([]) for _ in range(51)]
            prepared_mcmc=mcmc_args(modules,100)
            component_specs.extend([
              (label,"make_distance_matrix_individual",200,lambda m=modules,x=f,c=cells:m["distances"].make_distance_matrix_individual(x.locations,c,x.frames,50),3),
              (label,"determine_blinking_distribution",200,lambda m=modules,x=f:m["distributions"].determine_blinking_distribution([x.locations],[x.frames],2,10.),3),
              (label,"calculate_score_8",200,lambda m=modules,a=score_args(200):m["likelihood"].calculate_score_8(a[0],a[1],2,a[2],a[3],10.,a[4]),3),
              (label,"density_calc",400,lambda m=modules,x=fixture(400):m["trajectories"].density_calc(x.locations,x.frames,10.,2.),3),
              (label,"_eliminate_blinking_de_loc15",200,eliminator_call(modules,200),3),
              (label,"run_mcmc",100,lambda m=modules,a=prepared_mcmc:m["mcmc"].run_mcmc(*a,rng=np.random.default_rng(8100)),3)])
        for label,name,n,fn,reps in component_specs:
            rows.append({"version":label,"component":name,"n_localizations":n,**measure(fn,reps,memory=name in ("make_distance_matrix_individual","density_calc"))})
        for label,modules in versions.items():
            tmp,path,save,load=serialization_call(modules,23);save()
            rows.append({"version":label,"component":"checkpoint_save_v73","n_localizations":23,**measure(save,3,memory=True)})
            rows.append({"version":label,"component":"checkpoint_load_v73","n_localizations":23,**measure(load,3,memory=True)})
            tmp.cleanup()
        for n in (23,50,100,200):
            records={}
            for label,modules in versions.items():
                repetitions=3 if n<=100 else 1
                samples=[counted_pipeline(modules,n,10000+n) for _ in range(repetitions)]
                records[label]={"minimum_seconds":min(x[0] for x in samples),"median_seconds":statistics.median(x[0] for x in samples),"mean_seconds":statistics.mean(x[0] for x in samples),"repetitions":repetitions,"outer_mcmc_iterations":samples[0][1]["outer_mcmc_iterations"],"proposal_attempts":samples[0][1]["proposal_attempts"]}
            end_to_end.append({"n_localizations":n,"Stage A":records["Stage A"],"Stage B":records["Stage B"],"speedup":records["Stage A"]["median_seconds"]/records["Stage B"]["median_seconds"]})
        # Explicit run_ddc component row is the representative n=100 end-to-end result.
        for label in versions:
            value=next(x for x in end_to_end if x["n_localizations"]==100)[label]
            rows.append({"version":label,"component":"run_ddc","n_localizations":100,**{k:value[k] for k in ("minimum_seconds","median_seconds","mean_seconds","repetitions")},"peak_tracemalloc_bytes":None})
        comparisons=[]
        for component in sorted(set(r["component"] for r in rows)):
            a=next(r for r in rows if r["version"]=="Stage A" and r["component"]==component);b=next(r for r in rows if r["version"]=="Stage B" and r["component"]==component)
            comparisons.append({"component":component,"stage_a_seconds":a["median_seconds"],"stage_b_seconds":b["median_seconds"],"speedup":a["median_seconds"]/b["median_seconds"],"absolute_saved_seconds":a["median_seconds"]-b["median_seconds"]})
        positive=sum(max(0,c["absolute_saved_seconds"]) for c in comparisons)
        for c in comparisons:c["contribution_of_positive_component_savings"]=(max(0,c["absolute_saved_seconds"])/positive if positive else 0.)
        profile=profile_final(versions["Stage B"]);(RESULTS/"stage_b_final_profile_top.txt").write_text(profile)
        result={"environment":environment(),"frozen_stage_a_sha256":"a54496a4c08591fee307314ef95a2046b8c4c4e59c710ec913549691a1441279","methodology":{"same_machine":True,"fixed_target_step_total":15,"wall_clock_not_terminating":True,"same_seed_per_version":True},"component_measurements":rows,"component_comparisons":comparisons,"end_to_end":end_to_end,"total_seconds":time.perf_counter()-started}
        (RESULTS/"stage_b_final.json").write_text(json.dumps(result,indent=2)+"\n")
        with (RESULTS/"stage_b_final.csv").open("w",newline="") as handle:
            keys=["component","stage_a_seconds","stage_b_seconds","speedup","absolute_saved_seconds","contribution_of_positive_component_savings"]
            writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(comparisons)
        print(json.dumps({"components":comparisons,"end_to_end":end_to_end,"seconds":result["total_seconds"]},indent=2))

if __name__=="__main__":main()
