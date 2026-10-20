"""End-to-end pipeline and checkpoint serialization baselines."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

import ddc.pipeline as pipeline
from ddc.pipeline import DDCConfig, run_ddc
from ddc.serialization import (CheckpointWriter, DDCCheckpoint, load_checkpoint,
                               save_initial_checkpoint, save_v73_checkpoint)
from benchmarks.fixtures import dataset_from_fixture, pipeline_fixture


def run_pipeline_benchmarks():
    data=dataset_from_fixture(pipeline_fixture())
    config=DDCConfig(resolution=10.,stepper=10,n_f=2,photon_weighted_correction=0,timer=30.)
    component={"blinking_distribution":0.,"mcmc":0.,"photon_collapse":0.}
    patches=[]
    for attr,key in (("determine_blinking_distribution","blinking_distribution"),
                     ("run_mcmc","mcmc"),("photon_weighted_collapse","photon_collapse")):
        real=getattr(pipeline,attr)
        def wrapper(*args,_real=real,_key=key,**kwargs):
            start=time.perf_counter()
            try:return _real(*args,**kwargs)
            finally:component[_key]+=time.perf_counter()-start
        patches.append(patch.object(pipeline,attr,wrapper))
    for p in patches:p.start()
    try:
        start=time.perf_counter(); result=run_ddc(data,config,rngs=[np.random.default_rng(33)])
        total=time.perf_counter()-start
    finally:
        for p in reversed(patches):p.stop()
    rows=[{"name":"run_ddc_no_checkpoint","n_localizations":23,"dimensions":2,
           "repetitions":1,"total_seconds":total,"component_seconds":component}]
    checkpoint=DDCCheckpoint.from_result(result,data,config,"benchmark.mat")
    with tempfile.TemporaryDirectory(prefix="ddc-b1-") as tmp:
        root=Path(tmp); v5=root/"initial.mat"; v73=root/"state.mat"
        for label,fn,path in (("save_initial_level5",save_initial_checkpoint,v5),
                              ("save_v73",save_v73_checkpoint,v73)):
            start=time.perf_counter(); fn(checkpoint,path); elapsed=time.perf_counter()-start
            rows.append({"name":label,"n_localizations":23,"dimensions":2,
                         "repetitions":1,"total_seconds":elapsed,"file_size_bytes":path.stat().st_size})
            start=time.perf_counter(); load_checkpoint(path); elapsed=time.perf_counter()-start
            rows.append({"name":"load_"+label,"n_localizations":23,"dimensions":2,
                         "repetitions":1,"total_seconds":elapsed,"file_size_bytes":path.stat().st_size})
        writer=CheckpointWriter(root/"callback.mat",data,config)
        start=time.perf_counter(); run_ddc(data,config,rngs=[np.random.default_rng(33)],checkpoint_callback=writer)
        rows.append({"name":"run_ddc_with_checkpoint","n_localizations":23,"dimensions":2,
                     "repetitions":1,"total_seconds":time.perf_counter()-start,
                     "file_size_bytes":(root/"callback.mat").stat().st_size})
    return rows
