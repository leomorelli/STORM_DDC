"""Focused accepted-B.3 versus candidate-B.5 distance benchmark."""

from __future__ import annotations
import ast,json,statistics,time,tracemalloc
from pathlib import Path
import numpy as np
import ddc.distances as distances
from benchmarks.fixtures import environment

ROOT=Path(__file__).resolve().parents[1];RESULTS=ROOT/"benchmarks/results"

def reference():
    path=ROOT/"tests/test_b5_distance_equivalence.py";tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_b3_reference")
    namespace=dict(distances.__dict__);exec(compile(ast.Module([node],[]),str(path),"exec"),namespace);return namespace["_b3_reference"]

def timed(fn,reps):
    fn();values=[]
    for _ in range(reps):start=time.perf_counter();fn();values.append(time.perf_counter()-start)
    return {"repetitions":reps,"minimum_seconds":min(values),"median_seconds":statistics.median(values),"mean_seconds":statistics.mean(values)}

def peak(fn):
    tracemalloc.start();fn();_,value=tracemalloc.get_traced_memory();tracemalloc.stop();return value

def fixture(n):
    rng=np.random.default_rng(5000+n);return rng.normal(size=(n,3)),rng.permutation(np.arange(n,dtype=float))

def main():
    RESULTS.mkdir(parents=True,exist_ok=True);old=reference();rows=[];started=time.perf_counter()
    for n in (25,50,100,200,400,800):
        loc,frames=fixture(n)
        for b in (2,10,50):
            cells=[np.array([],dtype=float) for _ in range(b+1)]
            before=lambda l=loc,f=frames,c=cells,w=b:old(l,c,f,w)
            after=lambda l=loc,f=frames,c=cells,w=b:distances.make_distance_matrix_individual(l,c,f,w)
            before_reps=3 if n<=400 else 1;after_reps=10 if n<=400 else 5
            a=timed(before,before_reps);z=timed(after,after_reps);output=after();accepted=sum(len(x) for x in output)-sum(len(x) for x in cells)
            rows.append({"n_localizations":n,"B":b,"total_pairs_considered":n*(n+1)//2,"accepted_pairs":accepted,"output_distances":sum(len(x) for x in output),"B.3":a,"B.5":z,"speedup":a["median_seconds"]/z["median_seconds"],"B.3_peak_tracemalloc_bytes":peak(before),"B.5_peak_tracemalloc_bytes":peak(after)})
    result={"environment":environment(),"algorithm":"vectorized finite floating-frame candidate discovery; unchanged scalar coordinate distance and append order","rows":rows,"total_seconds":time.perf_counter()-started}
    (RESULTS/"stage_b5_distance.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"rows":[{"n":r["n_localizations"],"B":r["B"],"speedup":r["speedup"]} for r in rows],"seconds":result["total_seconds"]},indent=2))

if __name__=="__main__":main()
