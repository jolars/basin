//! Convergence-trace harness for the *solvers* benchmark axis: basin's
//! general optimizers (GD, NM, Bfgs, L-Bfgs, CMA-ES) on Rosenbrock from
//! multiple seeded starting points, each capped on wall-clock time. Powers
//! the `/benchmarks/solvers` page (see `web/scripts/collect-solvers.ts`).
//!
//! Where the competitors axis pits basin against argmin on a single classical
//! start with a fixed iteration cap, this axis compares solvers *against
//! each other* from several random starts sampled uniformly inside a per-
//! problem domain box. Each (solver, start) run is capped on a fixed
//! wall-clock `BUDGET` and stopped early on reaching subopt ≤ `TARGET`, so
//! a line that ends at the right edge of the chart never made it. Lines
//! within a panel share the same `f(x0)`, giving a direct head-to-head
//! read of "which family converges fastest from this start".
//!
//! Starts are seeded (not hand-picked) so the comparison scales without
//! curation when more problems are added: each problem only needs a domain
//! box and the existing seed list.
//!
//! Timing: solvers are deterministic on the cost sequence (CMA-ES too, since
//! its RNG is seeded per `(solver, start)`), so only timing jitters across
//! reps. `REPS` reps per (solver, start), median-elapsed-ns per iter index
//! paired with the rep-invariant cost, same as the competitor harness.
//!
//! Run: `cargo run -p competitor-bench --release --bin solver_compare > target/solver-traces.json`.

use std::time::{Duration, Instant};

use basin::problems::Rosenbrock;
use basin::{
    BasicSimplexState, BasicState, Bfgs, CmaEs, CmaEsState, CountsMirror,
    DenseMatrix, DenseQuasiNewtonState, Executor, GradientDescent, LbfgsState,
    Lbfgsb, MoreThuente, NelderMead, Solver, State as BasinState, StepOutcome,
};

/// Wall-clock budget per (solver, start, rep). 20 ms gives GD room to either
/// converge or visibly stall on hard starts; Bfgs and L-Bfgs converge in a few
/// hundred µs from easy starts and the line just ends there.
const BUDGET: Duration = Duration::from_millis(20);
/// Repetitions per (solver, start) for the per-iteration time median.
const REPS: usize = 11;
/// Rosenbrock's global minimum value.
const F_OPT: f64 = 0.0;
/// Suboptimality target. Solvers stop on reaching this so the line ends
/// where the solver actually finished, not where the budget ran out.
/// Also the chart's y-axis lower bound: every panel has at least one
/// solver that hits it, which lines the y-scales up across panels.
const TARGET: f64 = 1e-10;
/// Suboptimality floor so the log-scale y-axis stays well-defined. Equal
/// to `TARGET`: solvers can't drive subopt below the level they stop at.
const FLOOR: f64 = TARGET;
/// Base seed for CMA-ES, mixed with the start index so different starts
/// draw different populations while each `(solver, start)` is reproducible.
const CMAES_BASE_SEED: u64 = 42;
/// Starting-point seeds. Each seed produces one start sampled uniformly
/// from the problem's domain box (see [`sample_start`]). 1-indexed so the
/// `Seed N` titles on the web page match the actual seed used in the
/// RNG: no off-by-one between what's shown and what's reproducible. Six gives
/// the existing 3×2 panel layout; trim or extend without other changes.
const START_SEEDS: [u64; 6] = [1, 2, 3, 4, 5, 6];

// ---------------------------------------------------------------------
// problem configuration
// ---------------------------------------------------------------------
//
// One entry per problem the harness benchmarks. Each carries the
// human-readable name (matched on the web side), the dimensionality, and
// the per-dim domain box used to sample seeded starts. Today there's just
// Rosenbrock at n=10. Adding (say) Ackley is a one-line append plus a
// matching `run_<solver>` arm in `main`.

struct ProblemConfig {
    name: &'static str,
    n: usize,
    /// Lower bound, shared across all dimensions.
    domain_lo: f64,
    /// Upper bound, shared across all dimensions.
    domain_hi: f64,
}

const PROBLEMS: &[ProblemConfig] = &[ProblemConfig {
    name: "rosenbrock",
    // n=2 is essentially solved by any second-order method in microseconds and
    // doesn't separate the families. n=10 keeps everything tractable while
    // making GD work for it, exposing the Bfgs and L-Bfgs scaling gap, and
    // letting NM and CMA-ES actually breathe.
    n: 10,
    domain_lo: -2.0,
    domain_hi: 2.0,
}];

/// SplitMix64: small, deterministic, no external rand dep. Seeded per
/// (problem, start_seed); the sampled start is the next `n` outputs treated
/// as uniform `f64` in the per-dim domain.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_unit(&mut self) -> f64 {
        // 53-bit mantissa in [0, 1).
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Hash the (problem name, seed) pair into a SplitMix64 stream so different
/// problems draw different starts from the same seed list.
fn sample_start(p: &ProblemConfig, seed: u64) -> Vec<f64> {
    // Mix the problem name into the RNG seed via a FNV-like fold: keeps
    // problem-seed pairs uncorrelated without pulling in a hash dep.
    let mut h: u64 = 0xcbf29ce484222325;
    for b in p.name.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    let mut rng = SplitMix64(h.wrapping_add(seed));
    (0..p.n)
        .map(|_| p.domain_lo + (p.domain_hi - p.domain_lo) * rng.next_unit())
        .collect()
}

// ---------------------------------------------------------------------
// stepper + median plumbing (mirrors trace.rs, see note above)
// ---------------------------------------------------------------------

/// Step a basin solve until the wall-clock budget runs out, the solver
/// stops on its own, or `f - f*` drops below `TARGET`. Returns
/// `(elapsed_ns, cost)` at iter 0 and after every completed iteration.
fn basin_trace<P, S, So>(
    exec: Executor<P, S, So>,
    budget: Duration,
) -> Vec<(u128, f64)>
where
    S: BasinState<Float = f64> + CountsMirror,
    So: Solver<P, S>,
    So::Error: std::fmt::Debug,
{
    // The executor defaults to `max_iter = 1000`; bump it well past anything
    // reachable inside `budget` so the wall-clock guard is the only stop.
    let mut stepper = exec.max_iter(u64::MAX).into_stepper().unwrap();
    let mut pts = Vec::new();
    pts.push((0u128, stepper.state().cost()));
    let t0 = Instant::now();
    while t0.elapsed() < budget {
        if stepper.step().unwrap() != StepOutcome::Continue {
            break;
        }
        let cost = stepper.state().cost();
        pts.push((t0.elapsed().as_nanos(), cost));
        // Hand-rolled target-cost stop. basin's `CostTolerance` checks the
        // *change* `|f_k − f_{k-1}|`, not the absolute level, so it can't
        // serve here, and the framework has no `TargetCost` criterion yet.
        // Driving from the `Stepper` lets us cut as soon as `f − f*` ≤ TARGET.
        if cost - F_OPT <= TARGET {
            break;
        }
    }
    pts
}

/// Median elapsed-ns per iteration index across `REPS` reps. Trims to the
/// shortest rep: under a wall-clock budget the per-rep iteration count
/// can vary by one or two.
fn median_reps(mut run: impl FnMut() -> Vec<(u128, f64)>) -> Vec<(u128, f64)> {
    let runs: Vec<Vec<(u128, f64)>> = (0..REPS).map(|_| run()).collect();
    let len = runs.iter().map(Vec::len).min().unwrap_or(0);
    (0..len)
        .map(|i| {
            let mut times: Vec<u128> = runs.iter().map(|r| r[i].0).collect();
            times.sort_unstable();
            (times[REPS / 2], runs[0][i].1)
        })
        .collect()
}

/// Number of log-spaced grid points between `T_MIN_NS` and the budget. ~150
/// gives smooth log-log curves at the 380-pixel chart width without bloating
/// the committed JSON (NM does >150k iters in 20 ms: emitting them all
/// would produce a 50 MB file).
const GRID_POINTS: usize = 150;
/// Lower grid bound (1 µs). Anything finer than this isn't resolvable on the
/// chart; the `(0, f(x0))` anchor still shows the initial cost.
const T_MIN_NS: f64 = 1_000.0;

/// Downsample a dense trace onto a log-spaced time grid with best-so-far
/// suboptimality. Keeps the iter-0 anchor at `t = 0`, then emits one sample
/// per grid point carrying the minimum cost seen up to that time, but only
/// *up to the actual stop time of the trace*. The line ends where the solver
/// actually stopped (early-stop on hitting `TARGET`, or budget on running
/// out of time); no fake flat-line extension to the budget edge.
fn downsample(pts: &[(u128, f64)], budget_ns: u128) -> Vec<(u128, f64)> {
    let mut out = Vec::with_capacity(GRID_POINTS + 2);
    let Some(&first) = pts.first() else {
        return out;
    };
    out.push(first);
    if pts.len() <= 1 {
        return out;
    }
    let t_stop = pts.last().unwrap().0;

    let log_lo = T_MIN_NS.log10();
    let log_hi = (budget_ns as f64).log10();

    let mut grid_idx: usize = 0;
    let mut best_cost = first.1;
    let mut iter = pts.iter().skip(1);
    let mut next = iter.next();

    while grid_idx < GRID_POINTS {
        let g_log_t = log_lo
            + (log_hi - log_lo) * grid_idx as f64 / (GRID_POINTS as f64 - 1.0);
        let g_t = 10f64.powf(g_log_t) as u128;
        if g_t > t_stop {
            break;
        }
        while let Some(&(t, c)) = next {
            if t > g_t {
                break;
            }
            if c < best_cost {
                best_cost = c;
            }
            next = iter.next();
        }
        out.push((g_t, best_cost));
        grid_idx += 1;
    }

    // Anchor the line's right end at the true stop time, carrying the final
    // best cost.
    let final_best = pts
        .iter()
        .fold(f64::INFINITY, |acc, &(_, c)| acc.min(c))
        .min(best_cost);
    if out.last().map(|p| p.0) != Some(t_stop) {
        out.push((t_stop, final_best));
    }
    out
}

// ---------------------------------------------------------------------
// per-solver runners (Rosenbrock-specific for now; adding a new problem
// means duplicating these arms with the new problem type)
// ---------------------------------------------------------------------

fn run_gd(start: &[f64]) -> Vec<(u128, f64)> {
    median_reps(|| {
        basin_trace(
            Executor::new(
                Rosenbrock::<Vec<f64>>::default(),
                GradientDescent::with_line_search(MoreThuente::new()),
                BasicState::new(start.to_vec()),
            ),
            BUDGET,
        )
    })
}

fn run_nm(start: &[f64]) -> Vec<(u128, f64)> {
    median_reps(|| {
        basin_trace(
            Executor::new(
                Rosenbrock::<Vec<f64>>::default(),
                NelderMead::new(),
                BasicSimplexState::new(start.to_vec()),
            ),
            BUDGET,
        )
    })
}

fn run_bfgs(start: &[f64]) -> Vec<(u128, f64)> {
    median_reps(|| {
        basin_trace(
            Executor::new(
                Rosenbrock::<Vec<f64>>::default(),
                Bfgs::new(),
                DenseQuasiNewtonState::new(start.to_vec()),
            ),
            BUDGET,
        )
    })
}

fn run_lbfgs(start: &[f64]) -> Vec<(u128, f64)> {
    median_reps(|| {
        basin_trace(
            Executor::new(
                Rosenbrock::<Vec<f64>>::default(),
                Lbfgsb::new().unbounded(),
                LbfgsState::new(start.to_vec(), 10),
            ),
            BUDGET,
        )
    })
}

fn run_cmaes(start: &[f64], seed: u64) -> Vec<(u128, f64)> {
    let cma_seed = CMAES_BASE_SEED.wrapping_add(seed);
    median_reps(|| {
        basin_trace(
            Executor::new(
                Rosenbrock::<Vec<f64>>::default(),
                CmaEs::<Vec<f64>, DenseMatrix>::new(cma_seed),
                CmaEsState::<Vec<f64>, DenseMatrix>::new(start.to_vec(), 0.3),
            ),
            BUDGET,
        )
    })
}

// ---------------------------------------------------------------------
// JSON output (hand-rolled to match trace.rs convention; the collector
// adds env/generatedAt metadata)
// ---------------------------------------------------------------------

struct Trace {
    solver: &'static str,
    problem: &'static str,
    n: usize,
    seed: u64,
    start: Vec<f64>,
    points: Vec<(u128, f64)>,
}

fn fmt_array(xs: &[f64]) -> String {
    let parts: Vec<String> = xs.iter().map(|v| format!("{v}")).collect();
    format!("[{}]", parts.join(","))
}

fn print_traces(traces: &[Trace]) {
    let mut out = String::from("[\n");
    for (ti, t) in traces.iter().enumerate() {
        out.push_str(&format!(
            "  {{\"solver\":\"{}\",\"problem\":\"{}\",\"n\":{},\
             \"seed\":{},\"start\":{},\"budgetNs\":{},\"points\":[",
            t.solver,
            t.problem,
            t.n,
            t.seed,
            fmt_array(&t.start),
            BUDGET.as_nanos()
        ));
        let sampled = downsample(&t.points, BUDGET.as_nanos());
        let mut first = true;
        for &(t_ns, cost) in &sampled {
            let diff = cost - F_OPT;
            if !diff.is_finite() {
                continue;
            }
            let subopt = diff.max(FLOOR);
            if !first {
                out.push(',');
            }
            first = false;
            out.push_str(&format!("{{\"tNs\":{t_ns},\"subopt\":{subopt}}}"));
        }
        out.push(']');
        out.push('}');
        out.push_str(if ti + 1 < traces.len() { ",\n" } else { "\n" });
    }
    out.push(']');
    println!("{out}");
}

fn main() {
    let mut traces = Vec::with_capacity(PROBLEMS.len() * START_SEEDS.len() * 5);
    for p in PROBLEMS {
        for &seed in &START_SEEDS {
            let start = sample_start(p, seed);
            let push = |traces: &mut Vec<Trace>, solver, points| {
                traces.push(Trace {
                    solver,
                    problem: p.name,
                    n: p.n,
                    seed,
                    start: start.clone(),
                    points,
                });
            };
            push(&mut traces, "gd", run_gd(&start));
            push(&mut traces, "nm", run_nm(&start));
            push(&mut traces, "bfgs", run_bfgs(&start));
            push(&mut traces, "lbfgs", run_lbfgs(&start));
            push(&mut traces, "cmaes", run_cmaes(&start, seed));
        }
    }
    print_traces(&traces);
}
