# Stage B.1 — performance baseline

## Objective and reference

This stage measured the faithful, unoptimized Stage-A implementation. It did
not change numerical code or implement an optimization. The immutable reference
is `DDC_PYTHON_STAGE_A_FINAL_UNOPTIMIZED.zip`, SHA-256
`a54496a4c08591fee307314ef95a2046b8c4c4e59c710ec913549691a1441279`.

## Environment and method

Measurements ran on Python 3.12.13, NumPy 2.3.5 and SciPy 1.17.0, Linux
6.18.35 x86-64, Intel Xeon Platinum 8370C at 2.80 GHz, 9 visible logical
cores and 17,106,231,296 bytes RAM. `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` were unset and were not changed.

Fixtures use explicit seeds and float64 frames. Cheap component timings have
one warm-up and five repetitions; expensive cases use three. Reported values
are `perf_counter` min/median/mean/population-standard-deviation. Fixture setup
is outside the timed region. `tracemalloc` was sampled separately at practical
small sizes; it does not fully observe NumPy/SciPy native allocations and made
large faithful scalar loops pathologically slow, so large rows deliberately
contain null peak-memory values. Pipeline, MCMC, and serialization are single
representative runs and must not be treated as low-noise microbenchmarks.

## Fixtures

Synthetic clustered coordinates cover 25, 50, 100, 200, 250 and 400
localizations. The fixture catalog also defines medium/large tiers of 500/1000,
but the largest quadratic workload executed was 400 after larger pilot runs
proved impractical. Both 2D and 3D density paths ran. The pipeline/MCMC fixture
is the approved 23-localization floating-frame fixture. One 100-row benchmark
used repository coordinates with a copied float64 frame array and is labelled
**PERFORMANCE-ONLY SUPPORTED SURROGATE**; it is not an equivalence result and
does not alter the source MAT file.

## Runtime and scaling

Representative component medians (seconds):

| Component | Sizes / median seconds | Empirical exponent |
|---|---|---:|
| `density_calc` | 50/.025772, 100/.116652, 200/.457042, 400/1.891404 | 2.056 |
| `make_distance_matrix_individual` | 25/.005762, 100/.074119, 250/.588880 | 1.996 |
| `determine_blinking_distribution` | 25/.001978, 100/.001944, 250/.003655 | 0.244 |
| `calculate_score_8` at 100% retained | 50/.044524, 100/.154483, 200/.685409, 400/2.832125 | 2.012 |

The distribution exponent is not a general complexity claim: at these sizes,
fixed fitting overhead and optimized SciPy pair construction dominate. The
other three estimates are consistent with quadratic work. A 100-row 3D density
call measured .131 s and the real-coordinate surrogate .131 s.

Retained-fraction experiments show that score cost depends on both the retained
subset and the fixed `D_Overall`: for a source size of 400, medians were 1.21,
1.89, 2.40, and 2.83 s at 25%, 50%, 75%, and 100% retained.

## Exact intersection and repeated `pdist`

`matlab_intersect_first` at left size 1000 measured medians of .843340 s
(unique/partial match), .051161 s (duplicate-heavy), 1.027862 s (no match), and
1.117179 s (high match). Thus exact MATLAB-semantic matching, rather than SciPy's
distance kernel alone, is a major likelihood cost.

Every isolated `calculate_score_8` invocation made exactly two real `pdist`
calls: one for coordinates and one for the two-column frame expression. The
instrumented MCMC runs recorded:

| Requested progress after reset | Actual outer iterations | `pdist` calls | score calls | density calls | eliminator calls |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 14 | 2 | 1 | 3 |
| 10 | 11 | 32 | 11 | 1 | 52 |
| 50 | 51 | 112 | 51 | 1 | 143 |

The extra first outer iteration is the preserved `steps = 10` reset after the
strict initial maximum. The counts include initialization/probability work.
All wrappers delegated to the real functions. In the 51-iteration run, direct
`pdist` time was .00626 s (1.46% of .429 s); score evaluation was .0238 s,
while 143 real eliminator calls consumed .364 s. This small-n fixture therefore
exposes repeated proposal reconstruction more strongly than pair distance.
All proposals happened to be accepted for these fixed seeds; no RNG stream was
changed to manufacture rejection cases.

## MCMC profile and pipeline

The representative 11-outer-iteration cProfile run attributes approximately
77% of measured wall time to repeated `_eliminate_blinking_de_loc15` wrappers;
its internal cross-frame distance expressions and array construction lead the
self-time report. `density_calc` runs once, while proposal generation and score
evaluation repeat. The complete top-30 cumulative and self-time tables are in
`benchmarks/results/stage_b1_profile_top.txt`.

The supported 23-row `run_ddc` took .0186 s without checkpointing: .00185 s in
initial distribution work and .0165 s in MCMC; photon collapse was disabled.
With checkpoint callback it took .0983 s. Separate Level-5 write/load measured
.00250/.00119 s for a 6,136-byte file. v7.3 write/load measured .0383/.0400 s
for 32,408 bytes. At this deliberately tiny pipeline size, serialization is
large relative overhead, but this does not establish that it dominates real
scientific runs.

## Memory observations

Small-fixture traced peaks included 33,252 bytes for the 25-row distribution,
28,072 bytes for 25-row deviation calculation, 26,204 bytes for a 50-row full
score, and 6,776 bytes for the 25-row distance matrix. These are lower bounds:
native pair-distance arrays and HDF5 allocations may not be visible to
`tracemalloc`. Algorithmically, condensed pair arrays contain `n(n-1)/2`
values, so native memory risk still grows quadratically even where traced peaks
look modest.

## Evidence-based hotspot ranking

1. `calculate_score_8`, principally exact pair matching at larger `n`: 2.83 s
   at 400, exponent 2.012, called once per MCMC outer iteration.
2. `_eliminate_blinking_de_loc15` proposal reconstruction: .364 s of the .429 s
   51-iteration MCMC run, called 143 times.
3. `density_calc`: 1.89 s at 400, exponent 2.056, one call per MCMC invocation.
4. `matlab_intersect_first` alone: roughly .84–1.05 s for most 1000-element
   cases; duplicates substantially reduce the work.
5. `make_distance_matrix_individual`: .588880 s at 250, exponent 1.996.
6. `_frame_cross_distances` and `_matlab_pairwise_expression`, visible beneath
   the eliminator in cProfile.
7. MCMC proposal generation: .0166 s across 143 calls in the 51-iteration run.
8. Repeated `pdist`: 112 calls but only 1.46% direct time in the small MCMC;
   its produced arrays feed the more costly exact intersection.
9. Probability/deviation setup: .0492 s at 100 and once per fresh MCMC.
10. Checkpoint serialization: important for tiny runs (.0797 s added in the
    observed callback run), but not yet proven dominant at realistic MCMC cost.

Percentages are fixture-specific and wrapper timings overlap (for example,
score time contains its `pdist` calls); they must not be summed as disjoint CPU
percentages.

## Candidate Stage-B optimizations (not implemented)

- **B.2 pair/likelihood bookkeeping:** avoid recomputing pair order and exact
  intersection where an equivalence-safe index mapping can be proven. Expected
  benefit: remove repeated quadratic matching; risk: high because MATLAB exact
  equality, unique jitter, and ordering are numerical contracts.
- **B.4 trajectory/MCMC structure:** reduce repeated full eliminator work across
  unsuccessful inner proposals or update affected state only. Expected benefit:
  largest on the representative MCMC; risk: high due RNG consumption, proposal
  loop, trajectory labels, and `constant2ft` source behavior.
- **B.3 density:** vectorized/blockwise distance and frame masks or a carefully
  validated neighborhood structure. Expected benefit: lower Python-loop cost;
  risk: medium/high at distance boundaries and with source dtype semantics.
- Later, evaluate checkpoint frequency/format separately; it is operational,
  not a scientific hot path established by this baseline.

The evidence supports **B.2 → B.4 → B.3** rather than the provisional
B.2 → B.3 → B.4: likelihood/intersection scales badly, while repeated trajectory
reconstruction dominates the measured MCMC and density runs only once per MCMC
invocation. Larger realistic fixtures should be repeated before committing to
a B.4 design.

## Limitations

This cloud CPU was shared; no affinity or BLAS thread control was imposed.
MCMC counts and acceptance are seed/fixture-specific. The real-data surrogate
uses converted frames solely to bypass preserved BUG-002 for performance and
has no MATLAB-equivalence meaning. The xlarge tiers were not run because the
faithful quadratic Python paths were already long at 400. Native memory is
undercounted. No Stage-B optimization was performed in B.1.
