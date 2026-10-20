"""Execute and persist the complete Stage B.1 baseline characterization."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_components import run_component_benchmarks
from benchmarks.benchmark_mcmc import run_mcmc_benchmarks
from benchmarks.benchmark_pipeline import run_pipeline_benchmarks
from benchmarks.fixtures import TIERS, environment

FROZEN_SHA="a54496a4c08591fee307314ef95a2046b8c4c4e59c710ec913549691a1441279"
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/"benchmarks/results"


def main():
    RESULTS.mkdir(parents=True,exist_ok=True); start=time.perf_counter()
    components,scaling,pdist_rows=run_component_benchmarks()
    mcmc,profile_text=run_mcmc_benchmarks()
    pipeline=run_pipeline_benchmarks()
    all_rows=components+mcmc+pipeline
    ranked=[]
    for row in all_rows:
        value=row.get("median_seconds",row.get("total_seconds",0.))
        ranked.append({"component":row["name"],"fixture_n":row.get("n_localizations"),
                       "seconds":value})
    ranked=sorted(ranked,key=lambda x:x["seconds"],reverse=True)[:20]
    result={"environment":environment(),"frozen_stage_a_sha256":FROZEN_SHA,
            "fixture_tiers":TIERS,"methodology":{"warmup":1,
            "timing":"time.perf_counter; 5 repetitions cheap, 3 expensive",
            "memory":"tracemalloc peak; excludes some NumPy/SciPy native allocations"},
            "benchmarks":{"components":components,"run_mcmc":mcmc,"pipeline_and_serialization":pipeline},
            "scaling_exponents":scaling,"hotspots":ranked,
            "pdist_analysis":{"calculate_score_rows":pdist_rows,
                              "mcmc": [{"requested":r["requested_progress_after_reset"],
                                        "calls":r["counts"]["pdist"],
                                        "seconds":r["component_seconds"]["pdist"],
                                        "fraction":r["pdist_fraction_total"],
                                        "shapes":r["pdist_input_shapes"]} for r in mcmc]},
            "total_benchmark_runtime_seconds":time.perf_counter()-start}
    (RESULTS/"stage_b1_baseline.json").write_text(json.dumps(result,indent=2)+"\n")
    keys=["name","n_localizations","dimensions","repetitions","min_seconds",
          "median_seconds","mean_seconds","std_seconds","peak_tracemalloc_bytes",
          "total_seconds","retained_fraction","case","file_size_bytes"]
    with (RESULTS/"stage_b1_baseline.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=keys,extrasaction="ignore"); writer.writeheader()
        writer.writerows(all_rows)
    (RESULTS/"stage_b1_profile_top.txt").write_text(profile_text)
    print(json.dumps({"rows":len(all_rows),"seconds":result["total_benchmark_runtime_seconds"],
                      "json_sha256":hashlib.sha256((RESULTS/"stage_b1_baseline.json").read_bytes()).hexdigest(),
                      "scaling":scaling},indent=2))


if __name__=="__main__": main()
