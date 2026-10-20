# Compiled C acceleration, 8 October 2026

The optional `ddc._native` extension is implemented in `_native.pyx` and compiled
with Cython 3.3.0 and the installed Visual Studio C++ Build Tools. It is already
installed in the `ddc-parallel` environment. Python remains 3.12.15, NumPy 2.5.3,
and SciPy 1.18.1; the numerical-library versions were not changed.

The kernels handle distance-bin construction and trajectory linking in both
MCMC and probability initialization. The distance builder preallocates buffers,
retains source pair order, and releases the GIL while filling distances. The
linker runs frame-candidate and group-membership loops in C. It preserves candidate traversal and uses
NumPy's original mean reduction.

Native linking requires supported float64 locations, sorted integer-valued
float64 frames below 2^25 in magnitude, finite seed labels, and supported
coordinate and resolution ranges. Other cases dispatch to the existing Python
implementation. Numerical ambiguity near a distance-bin boundary or near zero
reruns the linking pass with the original NumPy implementation. Compilation
disables fast-math and floating-point contraction. No scientific parameter,
proposal rule, or RNG sequence is intentionally changed.

Validation: 361 tests passed, 16 MATLAB-runtime oracle tests skipped. This
includes exact native/reference distance values and bit patterns, boundary
fallbacks, input mutation guards, MCMC/RNG traces, spawned workers, checkpoints,
and resume. Real-data benchmark runs compare every saved result field exactly.

Benchmarks use the MATLAB `DDC_Tools` 224-region input, the saved shared
distribution, fresh MCMC state, seed 123, an infinite timer, and a stopping
threshold of 12. They exclude distribution estimation, checkpoint I/O, and
input loading. Process startup is included when worker processes are used.
The threshold does not fix the number of proposals: the source resets the
counter to 10 on improvements, and retries proposals until the reconstruction
changes. Consequently different regions can take very different amounts of work.

The initial four-region comparison took 129.73 seconds with the previous
optimized Python kernels and 3.13 seconds with the first native version, with
every field exactly equal. The earlier Python measurement of the same work was
101.11 seconds. These are single measurements on an active, memory-constrained
computer and should not be treated as precise full-dataset speedup estimates.

The final rebuilt version's four-region results are recorded in
`benchmarks/results/native_final_roi_windows.json`: 12.42 seconds with one
worker and 11.95 seconds with four, with identical outputs. Relative to the
previous 101.11-second Python measurement, the serial result is approximately
8.1x faster. The timing varied substantially as the user's available memory
and background workload changed; the 3.13-second initial result should not be
treated as a reliable prediction for a full run.

Experiments with per-interaction NumPy fallbacks and dictionary-cached group
membership were rejected after slower measurements. The final code retains
compiled member scans and reruns an ambiguous pass with NumPy. The retained
`native_16_roi_windows.json` records a scaling experiment for a rejected tuning
variant, not the final implementation. Four workers remain a practical starting
point for the full input; small jobs may finish faster with one worker.

To disable native dispatch for comparison, set `DDC_DISABLE_NATIVE=1`. Remove
that environment variable to restore it. Check `native_available()` in
`ddc._acceleration` or the CLI's `Compiled C kernels: enabled` progress message.
Restart notebooks after installing or rebuilding the binary.

`setup.py` builds the extension during installation. The extension is optional,
so missing compilers do not prevent use of the Python implementation. Rebuild
when changing Python version, architecture, or operating system. The source
distribution includes the `.pyx` file through `MANIFEST.in`.
