# Windows DDC optimization, 8 October 2026

Environment: `ddc-parallel`, Python 3.12.15, NumPy 2.5.3,
Intel Core i7-1360P (16 logical processors), approximately 16 GB RAM.

Changes:

- Batch distance-bin appends while retaining scalar coordinate arithmetic and pair order.
- Replace repeated candidate scans with binary searches in both eliminators for
  sorted, finite, integer-valued float64 frames in a range where addition is exact.
  Retain source scans for other inputs.
- Submit larger MCMC regions first; retain per-image RNGs and indexed output storage.
- Dispatch shared-distribution maxima, histograms, and mixture fits by image to
  spawned processes. Collect in source order and average in the parent.
- Supply a local PowerShell launcher with four workers and one numerical-library
  thread per process. It explicitly selects the workspace source tree.

The ROI benchmark uses the first four regions of MATLAB's `DDC_Tools` split input
(988, 978, 991, and 999 localizations). It starts fresh MCMC states, uses the saved
MATLAB shared distribution, seed 123, a stopping threshold of 12, and an infinite
timer to avoid differences in chunk boundaries. The source resets the step value
to 10 on improvements; a threshold of 12 does not imply 12 proposals. All result
fields match exactly between runs. These timings exclude distribution estimation
and checkpoint I/O and are not a MATLAB/Python speed comparison.

| Implementation | Workers | Seconds |
|---|---:|---:|
| Before these changes | 1 | 178.13 |
| Updated | 1 | 101.11 |
| Updated | 2 | 49.11 |
| Updated | 4 | 25.81 |

Single measurements on the user's active computer: approximately 1.76x speedup
for the updated serial code and 6.90x for the updated code with four workers,
relative to the previous serial implementation. Do not extrapolate directly to
a full 224-region run with a stopping threshold of 100.

See `benchmarks/results/roi_parallel_windows.json` and
`benchmarks/benchmark_roi_parallel.py` for the recorded results and repeatable
worker comparison. Shared-distribution benchmarks are recorded separately in
`benchmarks/results/distribution_parallel_windows.json` (four regions) and
`benchmarks/results/distribution_224_parallel_windows.json` (all 224 regions).
For very small jobs, Windows process startup can outweigh the parallel benefit.

For all 224 regions, shared-distribution estimation took 79.00 seconds with one
worker and 27.43 seconds with four workers (approximately 2.88x), with all six
outputs exactly equal. The four-region distribution job was too short to
benefit: approximately 0.81 seconds serial versus 1.66 seconds with four workers.

Validation after all numerical and scheduling changes: 357 tests passed and
16 MATLAB-runtime oracle tests skipped. Differential tests include frame-candidate
searches, distance bins, seeded MCMC results, RNG traces, process scheduling,
checkpoints, resume, and ordered shared-distribution reductions. The temporary
pre-change source copies used for timing are in the user's local temporary folder.
The local MATLAB repository is exposed through a directory junction for reference
fixture tests; its files were not edited.
