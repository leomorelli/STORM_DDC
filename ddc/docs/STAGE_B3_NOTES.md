# Stage B.3 — `Density_Calc` optimization

## Scope and reference chain

B.3 follows the accepted B.4 MCMC optimization because B.1 measured
`density_calc` with empirical exponent 2.056267 and B.4 still measured about
0.104 s of density work in the representative n=100 run. The immutable Stage-A
archive (`a54496...1279`), accepted B.2 archive (`61c19e...42e8`), and accepted
B.4 archive (`cb4aeca...9eb8`) remain unchanged. B.2 `likelihood.py` and B.4
`mcmc.py` retain their approved hashes.

The only production file changed is `src/ddc/trajectories.py`; neither
`determine_locs_w_multi2` nor another numerical component was optimized.

## Preserved semantics

The implementation retains ordered `(i,j)` pairs, self pairs,
`Distance <= 2*Resolution`, strict `abs(frame(i)-frame(j)) > A`, MATLAB matrix
`length=max(n,d)`, the n<d source failure, `(n,1)` float64 output, and NaN
normalization for constant counts. Floating distance arithmetic remains
subtract, square, sum, and square root before the exact comparison.

DDC-SOURCE-BUG-004 is unchanged. Integer frames are never routed through the
optimized subtraction because MATLAB saturates integer scalar subtraction and
can therefore produce asymmetric ordered counts. All integer frame types use
the accepted nested-loop fallback. The explicit uint16 regression obtains raw
counts `[0,1]`, while an intentionally floating comparison obtains `[1,1]`.
BUG-001 through BUG-007 are otherwise untouched.

## Fast path and fallback

The fast path requires all of the following:

- coordinate dtype exactly float32 or float64;
- frame dtype exactly float32 or float64;
- all coordinates and frames finite;
- a valid source shape (the n<d check runs before dispatch).

It processes 32 source rows at a time, broadcasts them against every target
row, evaluates the original predicates, counts true entries by source row, and
then performs the unchanged normalization. The selected block size is an
internal detail, not a scientific parameter. At n=800, blocks 32 and 64 had
nearly identical medians (0.031336 and 0.031426 s), while block 32 used about
half the traced peak memory (1,883,944 versus 3,752,744 bytes). Larger blocks
were slower and used more memory on this machine.

Unsupported dtypes, object/unusual data, integer frames, or any nonfinite
coordinate/frame value use the exact nested-loop fallback. Temporary memory
is O(block_size*n*d), not an unconditional O(n²*d) tensor. The estimated
principal block temporaries are 1,024,000 bytes at n=800 and 2,048,000 bytes at
n=1600 (3-D float64); measured total traced peaks were 1,883,944 and 3,765,544
bytes.

## Equivalence validation

The test-only accepted B.4 implementation was compared with B.3 for raw
counts and final values, with exact equality and exact NaN positions. Coverage
includes 2-D/3-D float32/float64, exact/nextafter distance and frame
boundaries, multi-component reduction boundaries, repeated and signed frames,
self and ordered pairs, signed zero, subnormals, large finite values, NaN/Inf,
empty and constant densities, n<d, integer fallback, nine internal block
sizes, and 1,000 randomized differentials.

Controlled MCMC comparisons ran 10 seeds at n=23, 5 at n=50, and 2 at n=100.
Every RNG call/value and every `MCMCResult` field was exact. Single- and
two-image `run_ddc` outputs/states were exact. The unchanged checkpoint/resume
acceptance suite also passes.

Final validation: `326 passed, 16 skipped, 0 failed` both normally and with
`-W error`. All 16 skips are existing MATLAB-runtime oracle/probe skips.

## Same-machine performance

Isolated density medians:

| n | B.4 (s) | B.3 (s) | speedup | B.4 peak | B.3 peak |
|---:|---:|---:|---:|---:|---:|
| 25 | 0.009575 | 0.0000819 | 116.9x | 2,040 B | 47,488 B |
| 50 | 0.033184 | 0.0001856 | 178.8x | 2,240 B | 133,672 B |
| 100 | 0.133696 | 0.0008018 | 166.8x | 3,344 B | 341,704 B |
| 200 | 0.545724 | 0.0018946 | 288.0x | 5,744 B | 546,504 B |
| 400 | 2.001274 | 0.0109262 | 183.2x | 10,544 B | 956,200 B |
| 800 | 8.014907 | 0.0356472 | 224.8x | 20,144 B | 1,883,944 B |
| 1600 | 31.559620 | 0.1619469 | 194.9x | 39,344 B | 3,765,544 B |

The small B.4 traced-memory values reflect scalar Python iteration, not lower
runtime allocation cost. B.3 deliberately trades bounded NumPy temporaries for
removing Python pair-loop overhead. Empirical timing exponents over n=50–800
were 1.974 (B.4) and 1.894 (B.3). Both algorithms remain O(n²*d); the finite
measurement does not establish a subquadratic algorithm.

Whole-MCMC medians on identical workloads/RNG seeds:

| n | B.4 (s) | B.3 (s) | speedup | B.3 density (s) |
|---:|---:|---:|---:|---:|
| 23 | 0.132736 | 0.135217 | 0.982x | 0.000138 |
| 50 | 0.225499 | 0.169597 | 1.330x | 0.000266 |
| 100 | 0.335902 | 0.250064 | 1.343x | 0.000696 |
| 200 | 1.180029 | 0.745274 | 1.583x | 0.002188 |

At n=23 the 1.8% variation is measurement noise after density becomes only
0.1 ms; gains become material as n grows. In the post-B.3 n=200 profile,
`make_distance_matrix_individual` (0.504 s cumulative), the B.4 eliminator
(0.472 s), frame-cross calculations (0.172–0.189 s including wrappers), and
pairwise expressions (0.169 s) dominate. Density no longer appears in the top
30.

## Gate-2 decision

cKDTree was not attempted or used. Block vectorization made density negligible
in the MCMC profile, so a tree would add floating-boundary risk without a
meaningful end-to-end opportunity. The remaining frame/pair/eliminator work is
substantial, but overlaps with high-risk B.4b semantics and
`make_distance_matrix_individual` is now the largest single cumulative item.
The recommended next stage is B.5 rather than B.4b, subject to B.5's defined
scope; B.4b is not justified solely by this profile.
