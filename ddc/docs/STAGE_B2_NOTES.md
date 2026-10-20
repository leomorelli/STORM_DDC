# Stage B.2 — likelihood / exact-pair-matching optimization

## Scope and motivation

The immutable Stage-A reference is SHA-256
`a54496a4c08591fee307314ef95a2046b8c4c4e59c710ec913549691a1441279`.
Stage B.1 measured `calculate_score_8` scaling as approximately n^2.012 and
showed that the Python nested-scan implementation of MATLAB `intersect` was a
major cost. B.2 changes only `src/ddc/likelihood.py`; density, MCMC proposal and
eliminator code are untouched.

Before the production change, `docs/STAGE_B1_NOTES.md` was synchronized with
the already-frozen JSON/CSV values. No B.1 machine-readable result was changed.

## Preserved semantics

The optimized helper still implements modern MATLAB numeric-vector
`[C,ia,ib]=intersect(A,B)` semantics used by Stage A:

- exact equality with no tolerance, rounding, or quantization;
- sorted unique `C`;
- first occurrence indices in A and B, returned zero-based in Python;
- NaNs excluded even when both inputs contain NaN;
- infinities matched exactly;
- `-0.0 == +0.0`, while `C=A(ia)` preserves the first A occurrence's sign bit;
- adjacent `nextafter` values do not match;
- generic Stage-A behavior and result dtype retained for non-float64 and mixed
  numeric inputs.

The Stage-A slow scan remains only as a test oracle. It is not a production API.

## Algorithm and complexity

When both flattened vectors are float64—the active DDC likelihood case—the new
path obtains sorted unique values and first indices with `np.unique`, excludes
NaNs, locates candidates with `np.searchsorted`, verifies exact equality, and
constructs `C` by indexing the original A. Complexity changes from a Python
nested scan, O(|A||B|) plus duplicate scans, to approximately
O(|A| log |A| + |B| log |B|). All other dtype combinations use the unchanged
generic Stage-A scan, limiting semantic risk.

No likelihood sum, deletion order, probability indexing, or `pdist` call was
changed. `D_temp` and `Z2_temp` are deleted using the same `ia` in their
original order.

## Differential validation

The tests execute 419 deterministic exact-intersection comparisons: 400
randomized cases across float32, float64, int32, int64, and uint16; 15 explicit
float64 edge cases; and four mixed-dtype cases. Every test compares `C`, dtype,
signed-zero sign bits, `ia`, and `ib` exactly against the frozen Stage-A scan.
A large duplicate-heavy benchmark also exercises first-index behavior.

Complete `LikelihoodResult` equality was checked at retained fractions 25%,
50%, 75%, and 100%, including `lik`, `lik2`, `d`, `z22`, `c`, `ia`, `ib`,
`d_temp`, and `z2_temp`. Existing probability boundary and preserved source-
error tests remain green.

With identical RNG seeds, the real MCMC was executed once with the Stage-A
oracle and once with B.2. Likelihood/score/number histories, final locations and
frames, trajectories, `Constantf`, `Constant2ft`, `Orderf`, probability cache,
and `Step` were exactly equal. The corresponding full `run_ddc` comparison also
matched every deterministic pipeline/state output exactly.

## Same-process before/after results

The benchmark used one warm-up, three Stage-A score repetitions, five Stage-A
intersection repetitions, and 50 optimized repetitions. Times are medians in
seconds.

| Standalone case, left n=1000 | Stage A | B.2 | Speedup |
|---|---:|---:|---:|
| unique / partial | 0.860248 | 0.0000772 | 11,145x |
| duplicate-heavy | 0.129532 | 0.0000851 | 1,522x |
| no match | 1.019565 | 0.0000676 | 15,091x |
| high match | 1.043954 | 0.0000780 | 13,392x |

| `calculate_score_8`, 100% retained | Stage A | B.2 | Speedup |
|---:|---:|---:|---:|
| n=50 | 0.050919 | 0.000200 | 255x |
| n=100 | 0.139954 | 0.000246 | 569x |
| n=200 | 0.633156 | 0.000450 | 1,408x |
| n=400 | 2.668841 | 0.005938 | 449x |
| n=800 | 12.123323 | 0.024608 | 493x |

For source n=400, B.2 medians at retained fractions 25%, 50%, 75%, and 100%
were 0.000470, 0.000656, 0.000949, and 0.005938 s, versus Stage-A medians
1.070407, 2.110527, 2.600791, and 2.668841 s. Every supported measured case was
materially faster.

The same-process full-retention exponent was 2.004 for the Stage-A oracle and
1.848 after B.2 when n=800 is included. The optimized standalone-intersection
exponent across vector sizes 100–4000 was 0.494 in this small, overhead-affected
range; it should not be interpreted as asymptotic sublinear complexity.

## Gate 2 residual profile

The optimized n=400 profile took 0.00312 s under instrumentation. Its two
unchanged `pdist` calls consumed 0.00131 s (42.16%), optimized intersection
0.000790 s (25.33%), and indexing/deletion/probability work plus instrumentation
remainder 0.00101 s (32.51%). The top-30 cumulative and self-time tables are in
`benchmarks/results/stage_b2_profile_top.txt`.

Repeated `pdist` is now a meaningful *fraction* but a small absolute cost.
Removing it safely would require tracking original localization/pair identity
through elimination; coordinate equality is not a valid identity mechanism.
That invasive pair bookkeeping was therefore deferred. Gate 2 performed
measurement only; no second optimization was attempted.

## Next stage

The B.1 MCMC profile remains decisive at workflow level: 143 eliminator calls
consumed 0.364 s of a 0.429 s run, whereas direct `pdist` consumed 0.00626 s.
With the likelihood nested scan removed, the recommended next stage remains
**B.4 MCMC/eliminator data structure**, followed by **B.3 density**. Neither was
started in B.2.
