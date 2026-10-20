"""Externally instrument the real Stage-A MCMC without changing it."""

from __future__ import annotations

import cProfile
import io
import pstats
import time
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np

import ddc.distributions as distributions
import ddc.likelihood as likelihood
import ddc.mcmc as mcmc
from ddc.distributions import determine_blinking_distribution
from ddc.mcmc import ProbabilityCache, run_mcmc
from benchmarks.fixtures import pipeline_fixture


def _arguments(target):
    f=pipeline_fixture()
    bins, blink, _, resolution, x_overall, m_mat = determine_blinking_distribution(
        [f.locations], [f.frames], 2, 10.)
    return (1, f.locations, f.frames, 2, resolution, np.empty((0,0)), bins, blink,
            0, float("inf"), np.array([]), np.array([]), np.array([]), np.array([]),
            np.array([]), ProbabilityCache(np.array([]),np.array([]),np.array([])),
            target, 0, np.array([]), x_overall, m_mat)


def run_one(target_iterations, *, profile=False):
    counters={k:0 for k in ("pdist","calculate_score_8","density_calc","proposal_generation","eliminator")}
    times={k:0.0 for k in counters}; pdist_shapes=[]
    def wrap(module, name, key):
        real=getattr(module,name)
        def called(*args,**kwargs):
            counters[key]+=1; start=time.perf_counter()
            try: return real(*args,**kwargs)
            finally:
                times[key]+=time.perf_counter()-start
                if key=="pdist": pdist_shapes.append(list(np.asarray(args[0]).shape))
        return patch.object(module,name,called)
    profiler=cProfile.Profile() if profile else None
    with ExitStack() as stack:
        stack.enter_context(wrap(mcmc,"pdist","pdist"))
        stack.enter_context(wrap(distributions,"pdist","pdist"))
        stack.enter_context(wrap(likelihood,"pdist","pdist"))
        stack.enter_context(wrap(mcmc,"calculate_score_8","calculate_score_8"))
        stack.enter_context(wrap(mcmc,"density_calc","density_calc"))
        stack.enter_context(wrap(mcmc,"_proposal_parameters","proposal_generation"))
        stack.enter_context(wrap(mcmc,"_eliminate_blinking_de_loc15","eliminator"))
        start=time.perf_counter()
        if profiler: profiler.enable()
        result=run_mcmc(*_arguments(10+target_iterations), rng=np.random.default_rng(100+target_iterations))
        if profiler: profiler.disable()
        total=time.perf_counter()-start
    history=result.likelihood_history
    new_maxima=int(np.sum(history > np.maximum.accumulate(np.r_[-np.inf,history[:-1]]))) if history.size else 0
    row={"name":"run_mcmc","requested_progress_after_reset":target_iterations,
         "n_localizations":23,"dimensions":2,"repetitions":1,"total_seconds":total,
         "outer_iterations":counters["calculate_score_8"],
         "accepted_proposals":int(history.size),
         "rejected_proposals":counters["calculate_score_8"]-int(history.size),
         "new_maxima":new_maxima,"proposal_attempts":counters["proposal_generation"],
         "seconds_per_outer_iteration":total/max(counters["calculate_score_8"],1),
         "seconds_per_accepted_state":total/max(history.size,1),"counts":counters,
         "component_seconds":times,"pdist_input_shapes":pdist_shapes,
         "pdist_fraction_total":times["pdist"]/total,
         "unattributed_bookkeeping_seconds":max(0.,total-sum(times.values()))}
    text=""
    if profiler:
        stream=io.StringIO(); stats=pstats.Stats(profiler,stream=stream).strip_dirs()
        stream.write("\nTOP 30 BY CUMULATIVE TIME\n"); stats.sort_stats("cumulative").print_stats(30)
        stream.write("\nTOP 30 BY SELF TIME\n"); stats.sort_stats("tottime").print_stats(30)
        text=stream.getvalue()
    return row,text


def run_mcmc_benchmarks():
    rows=[]; profile_text=""
    for requested in (1,10,50):
        row,text=run_one(requested,profile=requested==10)
        rows.append(row); profile_text += text
    return rows,profile_text
