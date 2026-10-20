# Stage B final notes — optimized faithful release

## Objective and immutable reference

Stage B removes measured Python overhead while preserving the accepted Stage-A
scientific contract exactly. The immutable Stage-A archive is
`DDC_PYTHON_STAGE_A_FINAL_UNOPTIMIZED.zip`, SHA-256
`a54496a4c08591fee307314ef95a2046b8c4c4e59c710ec913549691a1441279`.
It was never overwritten.

Accepted review chain:

- B.1 profiling: `7f2fcce3...b1ad7c`;
- B.2 exact likelihood/intersection: `61c19e3a...3c42e8`;
- B.4 conservative eliminator path: `cb4aeca4...a9eb8`;
- B.3 blocked density: `df3dcc11...04cac7`.

Production changes by stage were confined to B.2 `likelihood.py`, B.4
`mcmc.py`, B.3 `trajectories.py`, and B.5 `distances.py`. B.5 additionally
updates only CLI/README wording to identify the final optimized release.

## Gate 1: distance-matrix decision

The B.5 optimization was attempted and accepted. For finite float32/float64
frames and coordinates, stable sorting is unchanged and each source row uses a
vector expression only to discover valid `jj` candidates. Candidates retain
ascending `jj` order. Cell-index validation, scalar coordinate subtraction,
square/sum/square-root evaluation, `np.append`, bin order, and existing cell
contents are unchanged. Integer, nonfinite, and unusual inputs use the exact
accepted nested fallback; no integer cast was introduced.

Equivalence covers every cell's count, dtype, shape, order, values and value
bits across 1,000 randomized fixtures plus empty/single inputs, repeated and
unsorted frames, stable ties, bins 1/B/B+1, prepopulation, nonintegral indices,
NaN/Inf, signed zero, float32/64, signed integers, uint16 BUG-001, 2-D/3-D,
large and duplicated coordinates. All comparisons were exact.

Focused speedups ranged from 1.29× to 86.80×. At n=800 they were 86.80×,
17.62× and 5.01× for B=2, 10 and 50. Peak memory remained close to B.3:
59,441 versus 52,846 bytes at n=800/B=2, and 356,305 versus 355,630 bytes at
n=800/B=50. The fast path was retained.

## Integrated correctness

Final results are `339 passed, 16 skipped, 0 failed` both normally and with
`-W error`; the 16 skips are only unavailable MATLAB runtime oracles/probes.
The checkpoint/resume file passes all 7 tests, including materialized
noninitial MCMC continuation. Exact B.4/B.3 tests retain RNG draw count/order,
all histories, localizations, frames, trajectories, constants, probability
cache, Step and every `MCMCResult` field. Single- and multi-image pipelines
remain exact.

DDC-SOURCE-BUG-001 through BUG-007 remain unchanged: invalid uint cell 0,
unsupported uint16 `pdist`, integer `mtimes`, asymmetric uint density,
`Constant2ft` leakage, integer photon normalization, and stale `Finish_Runs`
signature. Stage B does not introduce a corrected scientific mode.

## Same-machine Stage A versus final Stage B

Representative component medians:

| Component | Stage A (s) | Stage B (s) | Speedup | Positive-savings share |
|---|---:|---:|---:|---:|
| Distance matrix, n=200/B=50 | 0.238890 | 0.121681 | 1.96× | 5.32% |
| Blinking distribution, n=200 | 0.002638 | 0.002786 | 0.95× | 0% |
| Likelihood score, n=200 | 0.009807 | 0.000430 | 22.81× | 0.43% |
| Density, n=400 | 1.754251 | 0.007626 | 230.03× | 79.22% |
| Eliminator, n=200 | 0.034079 | 0.019626 | 1.74× | 0.66% |
| `run_mcmc`, n=100 | 0.482490 | 0.330250 | 1.46× | 6.91% |
| `run_ddc`, n=100 | 0.316909 | 0.158854 | 1.99× | 7.17% |
| Checkpoint save | 0.022594 | 0.026455 | 0.85× | 0% |
| Checkpoint load | 0.029651 | 0.022859 | 1.30× | 0.31% |

Shares are derived from positive measured component deltas, not invented
percentages. Small regressions in unchanged distribution/save code represent
measurement variation and were not optimized.

Memory reflects the chosen trade-off: blocked NumPy density uses a bounded
764,880-byte traced peak at n=400 versus 11,432 bytes for scalar Stage A, while
being 230× faster. Distance peaks were 177,641 versus 173,813 bytes. Checkpoint
save/load memory was effectively unchanged (44,192/52,824 versus
46,440/52,910 bytes).

## End-to-end fixed-work benchmark

The same deterministic floating fixtures and RNG seed were used for both
versions. MCMC ended by `step_total=15`; wall-clock time was infinite and could
not terminate a run. Iteration and proposal counts matched exactly.

| n | Stage A median (s) | Stage B median (s) | Speedup | Repetitions | likelihood iterations / proposals |
|---:|---:|---:|---:|---:|---:|
| 23 | 0.025939 | 0.022913 | 1.13× | 3 | 1 / 2 |
| 50 | 0.064142 | 0.032240 | 1.99× | 3 | 1 / 1 |
| 100 | 0.316909 | 0.158854 | 1.99× | 3 | 1 / 8 |
| 200 | 0.969951 | 0.545674 | 1.78× | 1 | 1 / 5 |

## Final profile and deferred work

At n=200, final `run_mcmc` spent 0.642 s cumulatively in the eliminator,
0.275 s in frame-cross operations, 0.265 s in pairwise expressions, 0.231 s
in distance-matrix construction, and 0.158 s in
`determine_locs_w_multi2` under cProfile. `run_ddc` shows the same families.
The exact top-30 cumulative/self tables are stored in
`benchmarks/results/stage_b_final_profile_top.txt`.

Frame-cross arithmetic, pairwise arithmetic and general eliminator candidate
traversal were intentionally deferred because their semantic risk exceeds the
demonstrated release benefit. `determine_locs_w_multi2` was explicitly excluded
from B.3/B.5. This implementation is not claimed theoretically optimal.

## Installation and environment

The package built and installed successfully offline with `--no-deps` and
`--no-build-isolation` in a writable virtual environment. `import ddc` printed
`0.1.0`; `python -m ddc --help` and `ddc --help` both exited successfully.

Validation environment: Python 3.12.13, NumPy 2.3.5, SciPy 1.17.0, h5py
3.14.0, hdf5storage 0.2.2, pytest 9.1.1; Linux 6.18.35 x86_64, Intel Xeon
Platinum 8370C, 9 logical cores, 17,106,231,296 bytes RAM. OMP, OpenBLAS and
MKL thread variables were unset.

Stage A remains the immutable unoptimized behavioral reference. Stage B is the
final optimized faithful Python implementation.
