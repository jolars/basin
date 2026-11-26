# In-place stepping and borrowed stop reports

This experiment measures executor overhead on the small solves from issue #140
against `main` at `aa05436`. [Issue
#140](https://github.com/jolars/basin/issues/140) [Baseline
revision](https://github.com/jolars/basin/commit/aa05436feb6595365c13b30daab629fbf774a2d9)

## Reproduce

From the repository root, choose an available logical CPU and run the harness in
the Linux development environment.

```sh
python3 crates/basin/benches/step_overhead/run.py target/step-overhead-my-run --cpu 8
```

The runner tests and builds four copies of the pinned revision, each with its
own build directory. In-place stepping passes `&mut S` through the driver and
returns the completion flag and termination. The port covers LM, LM with pivoted
QR, unbounded L-BFGS, and the convergence wrapper used by the issue's L-BFGS
configuration. The driver chooses the implementation at compile time. The
borrowed-report change adds a manual step that moves the report into storage at
the first stop and returns a reference to it. The final result takes ownership.
The runner builds each change independently and then applies both for a combined
comparison. [In-place patch](in-place.patch) [Borrowed report
patch](borrowed-report.patch)

The timed build uses Rust 1.89.0, release mode with fat LTO and one codegen
unit, and only the `ndarray_v0_17` feature. All cases use `f64`. LM uses ndarray
0.17.2, the normal-equation trust-region path, and the starts, analytic
Jacobian, and tolerances from the issue. Unbounded L-BFGS uses `Vec`, an
analytic gradient, More–Thuente, and a history of 10. It runs both through
`Executor::run` and through manual stepping. The issue uses the full-run path
for L-BFGS.

Each timed solve includes executor construction, initialization, execution, and
teardown. Fixed inputs are constructed outside the timer. A zero iteration cap
measures initialization and the boundary stop. The default run uses 16
interleaved rounds with 20,000 fresh solves per batch, after 1,000 warm-up
solves. It writes the individual batches to `samples.csv`, summaries for every
cap to `summary.csv`, and saves the test logs and verification traces.

## Lifecycle checks

Before timing, the runner compares the full manual trajectory across all four
builds. This includes current and incumbent records, the raw counts stamped on
the incumbent at publication, and structured stop reports. Each build
independently reevaluates the final incumbent's objective, and compares manual
stepping and `Executor::run` with `InnerExecutor` on a shared problem that
already has charged evaluations. Resuming after initialization or after two
steps reproduces the remaining trajectory, including the stop report.

The repeat ran 11 test suites per build, with 211 to 213 passing tests. The
suites cover clean stops and hard failures during initialization and execution,
observer and checkpoint behavior, and LM's stopping safeguards. The in-place
tests exercise resumption after a continuing partial step and a hard error after
the solver mutates state. Failed callback attempts stay charged on both
standalone and shared inner runs. The failed standalone run publishes no
iteration or final observation and yields no checkpoint. The borrowed-report
test checks that repeated stops reference the stored report, that final
observation fires once, and that the criteria allocation moves into the final
result.

All four builds reproduced the following converged runs. Counts are residual and
Jacobian requests for LM and cost and gradient requests for L-BFGS. The short LM
stops after a completed step, the far LM start stops during a partial step, and
L-BFGS stops at a boundary.

  | Workload                                   | Completed iterations | Final counts | Incumbent iteration | Incumbent counts |
  | ------------------------------------------ | -------------------: | -----------: | ------------------: | ---------------: |
  | LM, 1 parameter and 9 residuals            |                    6 |        7 / 6 |                   6 |            7 / 6 |
  | LM, 1 parameter and 9 residuals, far start |                    9 |       10 / 9 |                   9 |           10 / 8 |
  | LM, 5 parameters and 21 residuals          |                   11 |       12 / 8 |                   7 |            8 / 7 |
  | L-BFGS, 4 parameters                       |                   12 |      15 / 15 |                  12 |          15 / 15 |

## Timing

Both sessions ran on an AMD Ryzen 9 7900 under Linux, pinned to logical CPU 8.
The first used 16 rounds and the repeat used 12. Times below include
construction, initialization, and teardown. A negative change means the solve
was faster. Each change is the median of the per-round percentage differences
against `main`.

### First session

  | Converged run                              | Main (µs) | In-place | Borrowed report | Both   |
  | ------------------------------------------ | --------: | -------: | --------------: | -----: |
  | LM, 1 parameter and 9 residuals            |     3.465 |    −3.8% |           +2.4% |  −3.2% |
  | LM, 1 parameter and 9 residuals, far start |     5.596 |    −3.6% |           +1.1% |  −4.4% |
  | LM, 5 parameters and 21 residuals          |    27.029 |    +3.3% |           +5.4% |  +0.2% |
  | L-BFGS, full run                           |     3.574 |   −13.8% |           +0.8% |  −9.4% |
  | L-BFGS, manual steps                       |     3.624 |   −12.1% |           −0.6% | −11.6% |

### Repeat

  | Converged run                              | Main (µs) | In-place | Borrowed report | Both   |
  | ------------------------------------------ | --------: | -------: | --------------: | -----: |
  | LM, 1 parameter and 9 residuals            |     3.455 |    −7.5% |           −1.8% |  −6.0% |
  | LM, 1 parameter and 9 residuals, far start |     5.617 |    −7.1% |           −2.0% |  −6.9% |
  | LM, 5 parameters and 21 residuals          |    25.748 |    +3.5% |           +4.9% |  −1.1% |
  | L-BFGS, full run                           |     3.493 |   −14.4% |           +3.2% | −12.2% |
  | L-BFGS, manual steps                       |     3.457 |   −11.6% |           +2.8% | −12.7% |

### Interpretation

First-session baseline batches ranged from 3.400 to 4.712 µs for the short LM
start and from 3.429 to 4.918 µs for full-run L-BFGS. Pairing by round, the
middle half of in-place changes ranged from −4.7% to −1.9% in the first session
and −9.0% to −6.6% in the repeat for the short LM start. For full-run L-BFGS,
those ranges were −16.6% to −12.4% and −15.1% to −12.4%. A four-iteration cap
gave about a 15% speedup for full-run L-BFGS in both sessions.

In-place stepping consistently sped up L-BFGS and one-parameter LM. The
five-parameter LM workload slowed by about 3% to 4%, including a slowdown at cap
zero, where the driver only initializes and stops at the boundary. Borrowing
removed a report clone. Its manual L-BFGS latency changed by −0.6% and +2.8%
across the two sessions. The original full-run path serves as a control for the
change to manual stop reports. Differences on that path and at cap zero show
that the optimized builds vary even when the altered operation is absent.

## Basin 4.0 adoption check

The in-place candidate used one mutable-state step path for standalone and
composed runs and kept owned step outcomes. The baseline was
`a026ab3`. Both builds used the same workloads, Rust 1.89.0,
`ndarray_v0_17` with ndarray 0.17.2, and isolated release builds with
fat LTO and one codegen unit. Both sessions ran on the Ryzen 9 7900, pinned
to logical CPU 8, with 16 rounds, alternating baseline and candidate order,
and 20,000 solves per batch.

The verification traces matched byte for byte. The harness also checked
equivalent results across manual stepping, full runs, and shared-problem
inner runs, and matched resumed trajectories after initialization and
after two steps.

| Converged workload | Session 1 median | Session 1 upper quartile | Session 2 median | Session 2 upper quartile |
| --- | ---: | ---: | ---: | ---: |
| LM, 1 parameter and 9 residuals | −5.63% | −4.98% | −5.09% | −2.93% |
| LM, 1 parameter and 9 residuals, far start | −5.15% | −4.34% | −4.68% | −1.49% |
| LM, 5 parameters and 21 residuals | −3.12% | +0.71% | −1.80% | −0.05% |
| L-BFGS, full run | −11.70% | −10.49% | −12.40% | −2.71% |
| L-BFGS, manual steps | −11.91% | −10.19% | −10.72% | +16.29% |

The gate required a median speedup of at least 5% on each L-BFGS path and
2% on each one-parameter LM case, with a speedup at the upper quartile in
both sessions. Five-parameter LM met its separate limit of a 4% median
slowdown. Manual L-BFGS missed the upper-quartile gate in session 2,
which prompted the follow-up below.

| Cap-zero control | Session 1 median | Session 2 median |
| --- | ---: | ---: |
| LM, 1 parameter and 9 residuals | +0.52% | +0.14% |
| LM, 1 parameter and 9 residuals, far start | −0.57% | −0.07% |
| LM, 5 parameters and 21 residuals | −3.50% | −2.48% |
| L-BFGS, full run | +1.65% | +1.16% |
| L-BFGS, manual steps | +0.79% | +1.37% |

### Follow-up on two physical cores

Two additional sessions reused the same binaries, with one session pinned
to logical CPU 8 and the other to CPU 9. Each used the same 16 rounds and
20,000 solves per batch. Verification traces matched again.

| Converged workload | CPU 8 median | CPU 8 upper quartile | CPU 9 median | CPU 9 upper quartile |
| --- | ---: | ---: | ---: | ---: |
| LM, 1 parameter and 9 residuals | −6.66% | −4.52% | −4.94% | −3.80% |
| LM, 1 parameter and 9 residuals, far start | −6.18% | −3.88% | −4.78% | −4.19% |
| LM, 5 parameters and 21 residuals | −1.21% | +2.45% | −2.45% | −1.87% |
| L-BFGS, full run | −12.27% | −8.53% | −12.36% | −12.06% |
| L-BFGS, manual steps | −11.51% | −9.68% | −11.96% | −11.43% |

Both reruns met the timing thresholds. Across the nonzero fixed caps
of 1, 2, 4, and 8, manual L-BFGS median gains ranged from 6.04% to
13.53%. At cap zero, its medians changed by +0.94% and −0.49%, while
its upper quartiles changed by +10.72% and +3.13%. The earlier
upper-quartile slowdown for a converged manual run did not recur.

Basin 4.0 adopts in-place stepping with owned stop reports. The final port
passed backend tests and checkpoint coverage, workspace Clippy, formatting,
the Rust 1.87 compatibility check, rustdoc, both WASM builds, and the web
checks. Its release verification trace matched the baseline after the
remaining caller fixes. The timing sessions used the earlier candidate
binary.
