# Stage A final notes — faithful unoptimized reference

Stage A is complete. This release integrates packaging, CLI, checkpointing and
validation around the approved numerical port. No Stage-B optimization,
vectorization, caching, compilation or parallelism was introduced.

## Source-defect audit

| ID | Source location | Short description | Repository reachability | Python Stage-A behavior | MATLAB runtime oracle |
|---|---|---|---|---|---|
| BUG-001 | `Make_Distance_Matrix_Individual.m` | `uint16` subtraction can index `DistanceMatrix{0}` | reachable after distribution if not masked | invalid index reproduced | pending |
| BUG-002 | `Determine_Blinking_Distribution5.m` | `pdist` receives unsupported `uint16` matrix | first full-pipeline failure for repository fixtures | explicit `DDC-SOURCE-BUG-002` failure | pending |
| BUG-003 | `Eliminate_Blinking_De_Loc15_MCMC.m` | nonscalar integer `mtimes` unsupported | masked by BUG-002 in full path; direct-call reachable | explicit failure at operation | pending |
| BUG-004 | `Density_Calc.m` | saturating unsigned subtraction makes density asymmetric | reachable in MCMC with integer frames | ordered asymmetry preserved | pending |
| BUG-005 | `DDC_MCMC.m` | rejected repeated branch-3 proposal can leak `Constant2ft` | reachable with floating frames | leak preserved | pending |
| BUG-006 | `run_script_DDC.m` | integer photon normalization/multiplication semantics | repository `uint8` photons; later than BUG-002 | dtype and values/failures preserved | pending |
| BUG-007 | both `Finish_Runs.m` copies | stale pre-`Traj2` eleven-output assignment | eligible floating-frame resume; earlier bugs may mask | diagnostic API raises BUG-007; operational resume corrected | static confirmed, runtime pending |

## MATLAB oracle/probe audit

All entries below remain real skips when MATLAB is unavailable. Static
confirmation is not a substitute for runtime oracle execution.

| Oracle/probe | Stage | Purpose | Status |
|---|---|---|---|
| `generate_distance_references.m` | A.2 | distance-bin equivalence | runtime pending |
| `probe_uint16_frame_pdist.m` | A.3b | BUG-002 class and `pdist` rejection | static confirmed; runtime pending |
| `generate_distribution_reference.m` | A.3 | distribution equivalence | runtime pending |
| `generate_likelihood_reference.m` | A.6 | likelihood/intersect equivalence | runtime pending |
| `generate_mcmc_reference.m` | A.7 | deterministic MCMC initialization contract | probe/oracle source present; MATLAB runtime pending |
| `probe_mcmc_acceptance.m` | A.7 | exact acceptance expression and maximum reset | probe/oracle source present; MATLAB runtime pending |
| `probe_mcmc_constant2ft_leak.m` | A.7b | BUG-005 leak | static confirmed; runtime pending |
| `probe_photon_integer_normalization.m` | A.8c | BUG-006 integer photon semantics | static confirmed; runtime pending |
| `generate_pipeline_reference.m` | A.8 | orchestration equivalence | runtime pending |
| `probe_uint16_eliminator_mtimes.m` | A.4b | BUG-003 integer `mtimes` | static confirmed; runtime pending |
| `generate_probability_reference.m` | A.4 | probability equivalence | runtime pending |
| `generate_scoring_reference.m` | A.6 | scoring equivalence | runtime pending |
| `probe_checkpoint_compatibility.m` | A.9/A.9b | MAT input-state/schema compatibility | runtime pending; not Finish_Runs executability proof |
| `probe_finish_runs_signature_mismatch.m` | A.9b | 12-versus-11 output contract | static confirmed; runtime pending |
| `generate_trajectory_reference.m` | A.5 | trajectory/density equivalence | runtime pending |
| `probe_uint16_density.m` | A.5 | BUG-004 unsigned density | static confirmed; runtime pending |

## User-facing execution contract

`ddc run` loads a raw MAT file, constructs `DDCConfig`, derives independent
per-image RNGs, runs the approved pipeline and uses `CheckpointWriter` for the
Level-5-to-v7.3 save sequence. `ddc resume` loads saved state and calls the
corrected `resume_ddc`; it never calls `resume_finish_runs_source`. `ddc inspect`
is read-only and displays compact per-image dtype/shape/state information.

The final acceptance fixtures execute real floating-frame distribution, MCMC,
checkpoint write/reload and corrected continuation. The strengthened resume
case saves only after materialized noninitial MCMC state exists (`Step>0`,
nonempty likelihood, initialized constants/order, and populated probability
cache), reloads it, resumes with the continued RNG stream, and matches an
equivalent uninterrupted run. A `uint16` fixture verifies that BUG-002 remains
the first full-pipeline source failure, and integer photons remain integer at
the I/O and pipeline boundaries. An AST-based inventory audit scans every
Python test reference to `tests/matlab/*.m`; the final result is zero missing
oracle/probe sources.
