//! Cross-validation of the public [`Bobyqa`](crate::solver::Bobyqa) solver
//! against PRIMA v0.7.2.
//!
//! PRIMA is the BSD-3 C and Fortran reference translation of Powell's solvers
//! (vendored at `tools/prima`). For each test problem we generate a fixture
//! with PRIMA's BOBYQA (`tests/fixtures/bobyqa_prima_driver.c` →
//! `tests/fixtures/bobyqa_<problem>_<n>d.tsv`; see that file and the fixtures
//! `README.md`) recording the locked inputs (including the box `xl`/`xu`), the
//! full evaluation trace, and the final `x`/`f`/`nf`. This module replays the
//! same bound-constrained problem through basin and asserts a tiered parity.
//!
//! Unlike NEWUOA's parity module (which drives a `pub(crate)` `minimize`
//! because NEWUOA has no public solver), BOBYQA already exposes
//! [`Bobyqa`](crate::solver::Bobyqa)/[`BobyqaState`](crate::BobyqaState), so
//! this drives the *public* surface through an [`Executor`]. The eval trace is
//! captured by a small [`Tracing`] problem wrapper that records every point its
//! `cost` is called on. The module lives in-crate (rather than under `tests/`)
//! only to sit beside `newuoa/parity.rs` and read its fixtures via
//! [`include_str!`].
//!
//! # What is and isn't asserted
//!
//! basin's BOBYQA is a port of PRIMA's `bobyqb`/`trustregion`/`geometry` but is
//! not a line-by-line FP-identical transcription, so the per-evaluation
//! arithmetic order differs and the eval sequences diverge once trust-region
//! iterations begin. The assertions are structured to be meaningful without
//! demanding FP-identical internal arithmetic:
//!
//! 1. **Objective equivalence (tight).** The C objective in the harness and the
//!    Rust objective here must be the same function; we recompute the Rust
//!    objective at *every* point PRIMA evaluated and require agreement to
//!    `1e-12` relative. This catches any coefficient/index/sign drift
//!    between the two definitions across the whole region PRIMA explored.
//! 2. **Initial design (tight, order-independent).** basin's first `npt`
//!    samples must equal PRIMA's first `npt` samples *as a set* (BOBYQA's
//!    initial design is bound-aware, so we compare against the fixture rather
//!    than reconstructing the coordinate cross analytically; PRIMA emits all
//!    `+` then all `−` per coordinate, basin interleaves).
//! 3. **Final output (loose).** basin must converge (ρ reached, not the eval
//!    budget) to the same minimizer PRIMA found: `f` to `1e-6·(1 + |f*|)` (i.e.
//!    relative when `|f*| ≫ 1`, but an absolute `~1e-6` floor when the optimum
//!    is near zero, as it is for these problems), `x` to `1e-4` in `‖·‖∞`, and
//!    the evaluation count `nf` within a margin (not exact: a non-transcription
//!    port will not match `nf` to the unit).
//!
//! # Why no RESCUE-path fixture
//!
//! These fixtures exercise the model core, TRSBOX, ALTMOV, and the driver, but
//! **not** RESCUE (§5): on clean problems RESCUE never fires. This is not a gap:
//! PRIMA's own `rescue.f90` header records that RESCUE "is never invoked on
//! CUTEst ... problems with at most 50 variables unless heavy noise is imposed",
//! and an attempt to provoke it here through the public driver (heavy
//! high-frequency noise, coarse objective quantization, tight `ρ_end`, up to 10
//! variables) never tripped the denominator-failure condition: it is a
//! rounding-damage path unreachable with clean `f64` arithmetic on
//! well-behaved problems. A bit-exact PRIMA *trajectory* parity on the RESCUE
//! path is therefore unattainable (and would in any case diverge: basin's
//! `rescue` uses the exact `β` of BOBYQA eq. 4.12, see `super::rescue`). RESCUE
//! is instead validated directly in `super::rescue`'s unit tests, which run it
//! on hand-built models and assert the rebuilt `H` equals `inv(W)` (KKT
//! identity) and the model interpolates `F` at every point. The fixtures here
//! double as a regression guard that the RESCUE wiring stays dormant and does
//! not perturb the clean-problem trajectory.

use std::cell::RefCell;
use std::rc::Rc;

use crate::core::constraint::BoxConstraints;
use crate::core::problem::CostFunction;
use crate::{Bobyqa, BobyqaState, Executor, MaxCostEvals, TerminationReason};

/// One PRIMA reference run, parsed from a `bobyqa_*.tsv` fixture.
struct Fixture {
    problem: String,
    n: usize,
    rho_beg: f64,
    rho_end: f64,
    max_fun: usize,
    npt: usize,
    x0: Vec<f64>,
    xl: Vec<f64>,
    xu: Vec<f64>,
    /// Every objective evaluation PRIMA made, in PRIMA's order: `(f, x)`.
    evals: Vec<(f64, Vec<f64>)>,
    final_nf: usize,
    final_f: f64,
    final_x: Vec<f64>,
}

/// `key=value` token → the value parsed as `T` (panics on a mismatched key).
fn kv<'a>(tok: &'a str, key: &str) -> &'a str {
    let (k, v) = tok
        .split_once('=')
        .unwrap_or_else(|| panic!("expected key=value, got {tok:?}"));
    assert_eq!(k, key, "expected key {key:?}, got {k:?}");
    v
}

fn parse_fixture(text: &str) -> Fixture {
    let mut problem = None;
    let mut n = None;
    let mut rho_beg = None;
    let mut rho_end = None;
    let mut max_fun = None;
    let mut npt = None;
    let mut x0 = None;
    let mut xl = None;
    let mut xu = None;
    let mut evals = Vec::new();
    let mut final_line = None;

    let floats = |rest: &str| -> Vec<f64> {
        rest.split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect::<Vec<f64>>()
    };

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("# config") {
            let t: Vec<&str> = rest.split_whitespace().collect();
            problem = Some(kv(t[0], "problem").to_string());
            n = Some(kv(t[1], "n").parse().unwrap());
            rho_beg = Some(kv(t[2], "rho_beg").parse().unwrap());
            rho_end = Some(kv(t[3], "rho_end").parse().unwrap());
            max_fun = Some(kv(t[4], "maxfun").parse().unwrap());
            npt = Some(kv(t[5], "npt").parse().unwrap());
        } else if let Some(rest) = line.strip_prefix("# x0") {
            x0 = Some(floats(rest));
        } else if let Some(rest) = line.strip_prefix("# xl") {
            xl = Some(floats(rest));
        } else if let Some(rest) = line.strip_prefix("# xu") {
            xu = Some(floats(rest));
        } else if let Some(rest) = line.strip_prefix("# final") {
            final_line = Some(rest.to_string());
        } else if line.starts_with('#') {
            continue; // any other comment line
        } else {
            // eval row: idx f x0 .. x(n-1)
            let mut t = line.split_whitespace();
            let _idx: usize = t.next().unwrap().parse().unwrap();
            let f: f64 = t.next().unwrap().parse().unwrap();
            let x: Vec<f64> = t.map(|s| s.parse().unwrap()).collect();
            evals.push((f, x));
        }
    }

    // `# final nf=<nf> rc=<rc> f=<f> x= <x0> .. <x(n-1)>`
    let fl = final_line.expect("fixture missing `# final` line");
    let mut t = fl.split_whitespace();
    let final_nf: usize = kv(t.next().unwrap(), "nf").parse().unwrap();
    let _rc: i32 = kv(t.next().unwrap(), "rc").parse().unwrap();
    let final_f: f64 = kv(t.next().unwrap(), "f").parse().unwrap();
    assert_eq!(t.next(), Some("x="), "expected `x=` before final x");
    let final_x: Vec<f64> = t.map(|s| s.parse().unwrap()).collect();

    Fixture {
        problem: problem.expect("problem"),
        n: n.expect("n"),
        rho_beg: rho_beg.expect("rho_beg"),
        rho_end: rho_end.expect("rho_end"),
        max_fun: max_fun.expect("maxfun"),
        npt: npt.expect("npt"),
        x0: x0.expect("x0"),
        xl: xl.expect("xl"),
        xu: xu.expect("xu"),
        evals,
        final_nf,
        final_f,
        final_x,
    }
}

/// The three test objectives, mirrored textually with the C harness
/// (`tests/fixtures/bobyqa_prima_driver.c`). Keep the summation order, the
/// coefficients, and the index base identical to that file.
fn objective(problem: &str, x: &[f64]) -> f64 {
    match problem {
        // basin coefficient form: sum 100 (x_{i+1}-x_i^2)^2 + (1-x_i)^2
        "rosenbrock" => {
            let mut s = 0.0;
            for i in 0..x.len() - 1 {
                let t = x[i + 1] - x[i] * x[i];
                s += 100.0 * t * t + (1.0 - x[i]) * (1.0 - x[i]);
            }
            s
        }
        // shifted sphere: sum (x_i - 3)^2 (min at 3, outside the box)
        "sphere" => {
            let mut s = 0.0;
            for &xi in x {
                let d = xi - 3.0;
                s += d * d;
            }
            s
        }
        // chrosen.f90 form: sum (x_i-1)^2 + 100 (x_{i+1}-x_i^2)^2
        "chrosen" => {
            let mut s = 0.0;
            for i in 0..x.len() - 1 {
                let a = x[i] - 1.0;
                let b = x[i + 1] - x[i] * x[i];
                s += a * a + 100.0 * b * b;
            }
            s
        }
        other => panic!("unknown problem {other:?}"),
    }
}

/// A box-constrained problem that records every point its `cost` is called on,
/// so the parity check can inspect basin's full evaluation trace.
struct Tracing<'a> {
    problem: &'a str,
    lower: Vec<f64>,
    upper: Vec<f64>,
    trace: Rc<RefCell<Vec<Vec<f64>>>>,
}

impl CostFunction for Tracing<'_> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        self.trace.borrow_mut().push(x.clone());
        Ok(objective(self.problem, x))
    }
}

impl BoxConstraints for Tracing<'_> {
    fn lower(&self) -> &Vec<f64> {
        &self.lower
    }
    fn upper(&self) -> &Vec<f64> {
        &self.upper
    }
}

/// `true` if `xs` contains a point within `tol` (∞-norm) of `want`.
fn contains(xs: &[Vec<f64>], want: &[f64], tol: f64) -> bool {
    xs.iter().any(|got| {
        got.len() == want.len()
            && got.iter().zip(want).all(|(a, b)| (a - b).abs() <= tol)
    })
}

/// Run the full tiered parity check for one fixture.
fn check_parity(text: &str) {
    let fx = parse_fixture(text);
    let n = fx.n;
    assert_eq!(fx.x0.len(), n);
    assert_eq!(fx.xl.len(), n);
    assert_eq!(fx.xu.len(), n);
    assert_eq!(fx.npt, 2 * n + 1, "fixtures use the recommended npt = 2n+1");

    // --- Tier 1: objective equivalence over PRIMA's whole explored trajectory.
    for (k, (f_prima, x)) in fx.evals.iter().enumerate() {
        let f_rust = objective(&fx.problem, x);
        let tol = 1e-12 * f_prima.abs().max(1.0);
        assert!(
            (f_rust - f_prima).abs() <= tol,
            "{} eval {}: objective mismatch C vs Rust: prima={:.17e} rust={:.17e} diff={:.3e}",
            fx.problem,
            k,
            f_prima,
            f_rust,
            (f_rust - f_prima).abs(),
        );
    }

    // Drive basin's public BOBYQA once, recording every evaluation it makes.
    // The trace is shared via `Rc` so it survives the problem being moved into
    // the executor.
    let trace = Rc::new(RefCell::new(Vec::new()));
    let problem = Tracing {
        problem: &fx.problem,
        lower: fx.xl.clone(),
        upper: fx.xu.clone(),
        trace: Rc::clone(&trace),
    };
    let result = Executor::new(
        problem,
        Bobyqa::new()
            .with_rho_beg(fx.rho_beg)
            .with_rho_end(fx.rho_end)
            .with_npt(fx.npt),
        BobyqaState::new(fx.x0.clone()),
    )
    .terminate_on(MaxCostEvals(fx.max_fun as u64))
    .run()
    .unwrap();
    let trace = trace.borrow();

    // --- Tier 2: basin's first npt samples equal PRIMA's first npt samples as
    // an (order-independent) set. BOBYQA's initial design is bound-aware, so we
    // compare against the fixture rather than reconstructing the cross.
    assert!(
        trace.len() >= fx.npt,
        "{}: basin made only {} evals, fewer than npt={}",
        fx.problem,
        trace.len(),
        fx.npt,
    );
    let basin_init = &trace[..fx.npt];
    for (f_prima, want) in &fx.evals[..fx.npt] {
        let _ = f_prima;
        assert!(
            contains(basin_init, want, 1e-12),
            "{}: initial design missing PRIMA point {:?}\n  basin init = {:?}",
            fx.problem,
            want,
            basin_init,
        );
    }

    // --- Tier 3: converged final output matches PRIMA's.
    assert_eq!(
        result.reason,
        TerminationReason::SolverConverged,
        "{}: basin stopped on the eval budget, not convergence",
        fx.problem,
    );

    // Mixed absolute and relative: ~1e-6 absolute when the optimum is near zero
    // (these problems), relative to |f*| when it is large.
    let f_tol = 1e-6 * (1.0 + fx.final_f.abs());
    assert!(
        (result.best_cost() - fx.final_f).abs() <= f_tol,
        "{}: final f mismatch: prima={:.17e} basin={:.17e} diff={:.3e} tol={:.3e}",
        fx.problem,
        fx.final_f,
        result.best_cost(),
        (result.best_cost() - fx.final_f).abs(),
        f_tol,
    );

    let x_inf = result
        .best_param()
        .iter()
        .zip(&fx.final_x)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        x_inf <= 1e-4,
        "{}: final x mismatch (||.||_inf = {:.3e} > 1e-4)\n  prima={:?}\n  basin={:?}",
        fx.problem,
        x_inf,
        fx.final_x,
        result.best_param(),
    );

    // nf is the weakest signal: a paper-derived port takes a different (but
    // valid) trajectory, so the eval count differs by a chunk while still
    // reaching the same minimizer. The margin is a same-ballpark sanity bound,
    // not a parity claim; see the module docs.
    let nf_margin = (0.25 * fx.final_nf as f64).max(10.0);
    let nf_diff = (result.cost_evals() as f64 - fx.final_nf as f64).abs();
    assert!(
        nf_diff <= nf_margin,
        "{}: nf out of margin: prima={} basin={} diff={} margin={:.1}",
        fx.problem,
        fx.final_nf,
        result.cost_evals(),
        nf_diff,
        nf_margin,
    );
}

macro_rules! parity_test {
    ($name:ident, $file:literal) => {
        #[test]
        fn $name() {
            check_parity(include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/",
                $file
            )));
        }
    };
}

parity_test!(rosenbrock_2d_matches_prima, "bobyqa_rosenbrock_2d.tsv");
parity_test!(sphere_2d_matches_prima, "bobyqa_sphere_2d.tsv");
parity_test!(chrosen_5d_matches_prima, "bobyqa_chrosen_5d.tsv");
