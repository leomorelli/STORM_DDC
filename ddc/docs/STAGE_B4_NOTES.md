# Stage B.4 — MCMC blink-eliminator optimization

## References and scope

The frozen Stage-A SHA-256 remains
`a54496a4c08591fee307314ef95a2046b8c4c4e59c710ec913549691a1441279`.
The accepted B.2 review is
`61c19e3af0a368b5e28fc65449e5c065cd50841478d6242d47a9d087cd3c42e8`;
its optimized `likelihood.py` remains byte-identical at SHA-256
`2524e8d7d2d98bfcace841129f329f9dbb5b7d3a284bf356532f93eb4d129546`.
B.4 changes only `_eliminate_blinking_de_loc15` in `src/ddc/mcmc.py`.

The remaining B.1 documentation value for
`make_distance_matrix_individual(n=250)` was corrected from `.559` to the
canonical `.588880` seconds. B.1 JSON/CSV were not changed.

## Gate 0: post-B.2 profile and redundancy

A deterministic 51-outer-iteration run made 143 proposal attempts and 143
eliminator calls. External instrumentation of the accepted B.2 function counted:

- 3,146 candidate source/target evaluations;
- 6,292 full `np.flatnonzero` trajectory-membership scans;
- 3,146 coordinate pair expressions and 3,146 frame-cross expressions;
- 1,573 actual probability evaluations;
- zero trajectory merges on this particular low-probability benchmark;
- zero repeated coordinate-group computations within one eliminator call;
- 2,860 repeated frame-group computations within one call;
- median/max group size 1/1 and 23 groups throughout this fixture.

Under deliberately intrusive tracing plus `tracemalloc`, the eliminator took
4.43 s of a 5.19 s run. Non-overlapping coordinate/frame arithmetic accounted
for 0.669/1.356 s; the approximate traversal, membership and bookkeeping
remainder was 2.406 s. These traced times characterize shares and redundancy,
not normal wall-clock performance.

## Candidates evaluated

1. **General trajectory-label mapping:** eager and lazy ordered mappings both
   preserved outputs but were neutral or slower because construction/update
   overhead offset the small-array scans. Rejected.
2. **Coordinate memoization:** Gate 0 found zero repeated ordered coordinate
   groups within an invocation. Rejected.
3. **Frame memoization:** despite 2,860 repeat keys, tuple/key/cache overhead made
   representative MCMC slower. Reverted.
4. **Candidate-frame traversal:** left unchanged because subtraction boundary
   semantics are high risk and the safer membership opportunity existed.
5. **Pairwise arithmetic:** left unchanged because alternative evaluation order
   could change bins, probabilities, trajectories, and acceptance.

## Accepted optimization

The source initializes `traj_num` as a nonzero permutation, so all trajectory
labels commonly begin unique. B.4 detects exactly that state once. While it
holds, the authoritative trajectory group for source or target is provably the
singleton original index, avoiding two full-vector equality/`flatnonzero`
scans per candidate. At the first accepted merge the flag is set false before
the next candidate, and all later lookups use the original Stage-B.2 scan.
Inputs containing zero, duplicate, noncontiguous, or arbitrary labels use the
original path from the start.

This changes common-case membership lookup from O(n) per source/target to O(1)
until the first merge. The trajectory array remains authoritative. No candidate
order, arithmetic, probability calculation, trajectory assignment, RNG call,
or result representation changed.

## Exact equivalence

The accepted B.2 eliminator is retained as a test-only oracle. Thirty explicit
scenarios cover no/one links, 0.5 boundaries, multiple targets, repeated frames,
preexisting and arbitrary groups, zero labels, group merging, distance
boundaries, A boundaries, NaN/Inf, 2D/3D, float32/float64, BUG-003 integer
frames, MATLAB n<d length failure, and small input. A further 500 deterministic
randomized supported fixtures compare complete locations, frames, trajectory,
or exact exception type/message.

Ten small-fixture seeds and one n=50 seed compare every `MCMCResult` field and
the complete RNG call trace—including method, arguments, result dtype, shape,
and bytes. All are exact. Single- and two-image `run_ddc` comparisons match all
deterministic outputs and continuation states. The existing branch-3
`Constant2ft` leakage regression remains green, preserving BUG-005. BUG-001
through BUG-007 are unchanged.

## B.2 versus B.4 performance

Seven repetitions per implementation used identical seeds/workloads. Timer
termination was disabled; actual outer/proposal counts were recorded.

| n | Outer / proposals | B.2 MCMC s | B.4 MCMC s | MCMC speedup | B.2 eliminator s | B.4 eliminator s | Eliminator speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 23 | 8 / 26 | .107596 | .091494 | 1.176x | .072631 | .065605 | 1.107x |
| 50 | 6 / 19 | .195829 | .183706 | 1.066x | .110100 | .106914 | 1.030x |
| 100 | 6 / 16 | .400440 | .391140 | 1.024x | .200871 | .172944 | 1.161x |

The gain is intentionally modest because the optimization stops applying after
a merge and does not alter expensive pair arithmetic. No measured scale became
materially slower.

## Memory

Separate first-run `tracemalloc` peaks were 48,396→47,196 bytes (n=23),
98,477→94,056 bytes (n=50), and 264,791→256,007 bytes (n=100), corresponding
to B.4/B.2 ratios 0.975, 0.955, and 0.967. There is no persistent cache or full
matrix. These figures undercount NumPy native allocations.

## New profile and recommendation

The post-B.4 cProfile still places `_eliminate_blinking_de_loc15` first:
approximately .553 cumulative seconds in a .652 s instrumented run. Next are
`_frame_cross_distances` and `_matlab_pairwise_expression` (.205 s cumulative),
then reductions and `column_stack`. `calculate_score_8` is only .026 s and
`density_calc` .012 s in this profile. The complete top-30 cumulative and self
tables are in `benchmarks/results/stage_b4_profile_top.txt`.

Density is not the new dominant component, so the data do not support B.3 as
the immediate next optimization. The narrowest next investigation would be a
separate, high-risk **B.4b** proof of bitwise-exact frame/pair arithmetic or
candidate traversal. No such work, B.3, or B.5 was begun here.
