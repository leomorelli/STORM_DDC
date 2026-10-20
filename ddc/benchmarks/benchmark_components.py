"""Component-level timing and memory benchmarks for the frozen Stage-A code."""

from __future__ import annotations

import statistics
import time
import tracemalloc
from unittest.mock import patch

import numpy as np
from scipy.spatial.distance import pdist
import ddc.distributions as distributions_module

from ddc.distances import make_distance_matrix_individual
from ddc.distributions import determine_blinking_distribution
from ddc.likelihood import calculate_score_8, matlab_intersect_first
from ddc.probabilities import determine_deviation_in_probability8, eliminate_blinking_de_loc15_mcmc
from ddc.trajectories import density_calc, determine_locs_w_multi2
from benchmarks.fixtures import fixture, repository_surrogate


def measure(name, fn, *, n, dimensions=2, repetitions=5, extra=None):
    fn()  # warm-up
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter(); fn(); samples.append(time.perf_counter() - start)
    # Tracing every scalar allocation makes the faithful Python loops hundreds
    # of times slower.  Take a separate peak sample only at practical sizes.
    peak = None
    if n <= 50:
        tracemalloc.start(); fn(); _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    row = {"name": name, "n_localizations": n, "dimensions": dimensions,
           "repetitions": repetitions, "min_seconds": min(samples),
           "median_seconds": statistics.median(samples),
           "mean_seconds": statistics.mean(samples),
           "std_seconds": statistics.pstdev(samples),
           "peak_tracemalloc_bytes": peak}
    if extra: row.update(extra)
    return row


def scaling(rows, name):
    selected = [r for r in rows if r["name"] == name]
    return float(np.polyfit(np.log([r["n_localizations"] for r in selected]),
                            np.log([r["median_seconds"] for r in selected]), 1)[0])


def run_component_benchmarks():
    rows = []
    for n in (25, 100, 250):
        f = fixture(n)
        b = int(np.max(f.frames))
        rows.append(measure("make_distance_matrix_individual",
            lambda f=f,b=b: make_distance_matrix_individual(f.locations,
                [np.array([]) for _ in range(b + 1)], f.frames, b), n=n,
                repetitions=3 if n == 250 else 5))
        rows.append(measure("determine_blinking_distribution",
            lambda f=f: determine_blinking_distribution([f.locations], [f.frames], 2, 10.0),
            n=n, repetitions=3 if n == 250 else 5))
    for n in (50, 100, 200, 400):
        f = fixture(n)
        rows.append(measure("density_calc", lambda f=f: density_calc(f.locations, f.frames, 10.0, 2),
                            n=n, repetitions=3 if n >= 400 else 5))
    for n in (25, 50, 100):
        f = fixture(n)
        rows.append(measure("determine_locs_w_multi2",
            lambda f=f: determine_locs_w_multi2(f.locations, f.frames, 10.0), n=n))

    # Direct probability paths with controlled, valid distributions.
    for n in (25, 50, 100):
        f = fixture(n); a = 2; width = 80
        true = np.full(width, 1 / width); blink = np.linspace(1, 2, width); blink /= blink.sum()
        bins = np.arange(width + 1, dtype=float) * 10
        deviation = np.vstack((np.linspace(.1,.9,width), np.linspace(.2,.8,width)))
        order = np.arange(1, n + 1)
        rows.append(measure("eliminate_blinking_de_loc15_mcmc",
            lambda f=f: eliminate_blinking_de_loc15_mcmc(f.locations, f.frames, 10., a, deviation, order), n=n))
        rows.append(measure("determine_deviation_in_probability8",
            lambda f=f: determine_deviation_in_probability8(true, blink, bins, a,
                f.locations, f.frames, [], [], 10., true, blink, bins,
                np.array([.5, 1.]), deviation, rng=np.random.default_rng(7)),
            n=n, repetitions=3))

    # Dimension sensitivity and one explicitly labelled real-coordinate surrogate.
    f3 = fixture(100, 3)
    rows.append(measure("density_calc_3d", lambda: density_calc(f3.locations, f3.frames, 10., 2),
                        n=100, dimensions=3, repetitions=3))
    surrogate = repository_surrogate(100)
    if surrogate is not None:
        rows.append(measure("density_calc_repository_surrogate",
            lambda: density_calc(surrogate.locations, surrogate.frames, 10., 2), n=100,
            dimensions=surrogate.locations.shape[1], repetitions=3,
            extra={"fixture":"PERFORMANCE-ONLY SUPPORTED SURROGATE"}))

    pdist_records = []
    real_pdist = pdist
    for n in (50, 100, 200, 400):
        whole = fixture(n); overall = real_pdist(whole.locations)
        z2 = real_pdist(np.column_stack((np.zeros(n), whole.frames)))
        keep = (z2 < 2) & (z2 > 0); overall, z2 = overall[keep], z2[keep]
        probabilities = np.log(np.full((2, 200), 1 / 200))
        for fraction in (.25, .5, .75, 1.0):
            k = max(2, int(n * fraction)); calls=[]
            def counted(x, *args, **kwargs):
                start=time.perf_counter(); out=real_pdist(x,*args,**kwargs)
                calls.append({"shape": list(np.asarray(x).shape), "seconds": time.perf_counter()-start}); return out
            def score():
                with patch("ddc.likelihood.pdist", counted), patch.object(
                    distributions_module, "pdist", counted
                ):
                    return calculate_score_8(whole.locations[:k], whole.frames[:k], 2,
                                             overall, z2, 10., probabilities)
            row=measure("calculate_score_8", score, n=k, repetitions=3,
                        extra={"source_n":n,"retained_fraction":fraction})
            row["pdist_call_count"] = len(calls)
            invocations = 1 + row["repetitions"] + (1 if k <= 50 else 0)
            row["pdist_calls_per_score"] = len(calls) / invocations
            row["pdist_cumulative_seconds"] = sum(x["seconds"] for x in calls)
            row["pdist_input_shapes"] = [x["shape"] for x in calls]
            rows.append(row); pdist_records.append(row)

    rng=np.random.default_rng(19)
    cases={"unique":(np.arange(1000.),np.arange(0,1000.,2)),
           "duplicates":(rng.integers(0,200,1000),rng.integers(0,200,500)),
           "no_match":(np.arange(1000.),np.arange(1000.,1500.)),
           "high_match":(np.arange(1000.),np.arange(900.))}
    for label,(a,b) in cases.items():
        rows.append(measure("matlab_intersect_first", lambda a=a,b=b: matlab_intersect_first(a,b),
                            n=len(a), repetitions=5, extra={"case":label,"right_size":len(b)}))
    scales={name:scaling(rows,name) for name in
            ("density_calc","make_distance_matrix_individual","determine_blinking_distribution")}
    score100=[r for r in rows if r["name"]=="calculate_score_8" and r["retained_fraction"]==1.0]
    scales["calculate_score_8"] = float(np.polyfit(np.log([r["n_localizations"] for r in score100]),
                                                    np.log([r["median_seconds"] for r in score100]),1)[0])
    return rows, scales, pdist_records
