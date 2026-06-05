//! Iteration-wise parity test against Nocedal's L-Bfgs-B v3.0
//! Fortran reference.
//!
//! Reads `tests/fixtures/lbfgsb_rosenbrock_5d.tsv` (dumped from a
//! gfortran build of `references/lbfgsb-v3.0/` — see
//! `tests/fixtures/README.md`) and drives basin's `Lbfgsb` through
//! the same iterates, asserting per-step agreement on `x`, `f`, and
//! `g`.
//!
//! This is the load-bearing correctness signal for the port — the
//! point of mirroring the Fortran scalar arithmetic line-for-line is
//! that the trajectories must match. Tolerance is `~1e-10` per
//! component, which gives some slack for compiler-level
//! reordering of floating-point ops but is tight enough to catch
//! algorithmic divergence.

use basin::{BoxConstraints, CostFunction, Executor, Gradient, LbfgsState, Lbfgsb, MaxIter};
use std::fs;

/// Standard Rosenbrock 5D (basin's coefficient convention).
struct Rosen5D {
    l: Vec<f64>,
    u: Vec<f64>,
}

impl CostFunction for Rosen5D {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = std::convert::Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
        Ok({
            let n = x.len();
            let mut f = 0.0;
            for i in 0..n - 1 {
                let t = x[i + 1] - x[i] * x[i];
                f += 100.0 * t * t + (1.0 - x[i]).powi(2);
            }
            f
        })
    }
}

impl Gradient for Rosen5D {
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
        Ok({
            let n = x.len();
            let mut g = vec![0.0; n];
            for i in 0..n - 1 {
                let t = x[i + 1] - x[i] * x[i];
                g[i] += -400.0 * x[i] * t - 2.0 * (1.0 - x[i]);
                g[i + 1] += 200.0 * t;
            }
            g
        })
    }
}

impl BoxConstraints for Rosen5D {
    fn lower(&self) -> &Vec<f64> {
        &self.l
    }
    fn upper(&self) -> &Vec<f64> {
        &self.u
    }
}

#[derive(Debug)]
struct FortranIterate {
    iter: u64,
    f: f64,
    x: Vec<f64>,
    g: Vec<f64>,
}

/// Parse the Fortran-dumped TSV. Each non-empty line has columns
/// `iter f x(0..n) g(0..n)`; whitespace-separated, `es24.16` format.
fn load_fixture(path: &str, n: usize) -> Vec<FortranIterate> {
    let raw = fs::read_to_string(path).expect("fixture file not found");
    let mut out = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut tokens = line.split_whitespace();
        let iter: u64 = tokens.next().unwrap().parse().unwrap();
        let f: f64 = tokens.next().unwrap().parse().unwrap();
        let x: Vec<f64> = (&mut tokens).take(n).map(|s| s.parse().unwrap()).collect();
        let g: Vec<f64> = (&mut tokens).take(n).map(|s| s.parse().unwrap()).collect();
        assert_eq!(x.len(), n);
        assert_eq!(g.len(), n);
        out.push(FortranIterate { iter, f, x, g });
    }
    out
}

#[test]
fn rosenbrock_5d_matches_fortran_trajectory() {
    let n = 5;
    let fixture = load_fixture("tests/fixtures/lbfgsb_rosenbrock_5d.tsv", n);
    assert_eq!(fixture.len(), 31, "expected 31 iterates (iter 0..30)");

    let problem = Rosen5D {
        l: vec![0.0; n],
        u: vec![5.0; n],
    };
    // Start matching the Fortran driver: infeasible initial point gets
    // projected during `Lbfgsb::init`.
    let initial = vec![-1.0, 2.0, -1.0, 2.0, -1.0];
    let state = LbfgsState::new(initial, 5);

    // Match Fortran driver: `factr = 0`, `pgtol = 0` — disable both
    // convergence tolerances so the parity comparator runs all 30
    // iterations regardless of how small the projected gradient gets.
    let mut stepper = Executor::new(problem, Lbfgsb::new().with_tol_pg(0.0), state)
        .terminate_on(MaxIter(30))
        .into_stepper()
        .unwrap();

    // x_tol: variables can be at the boundary or in the interior;
    // either way the trajectory should agree to ~1e-10 absolute. The
    // line search uses `MoreThuente` with the same defaults as
    // Fortran's `lnsrlb` (ftol=1e-3, gtol=0.9, xtol=0.1), so the
    // selected step matches modulo final-step reordering.
    let x_tol = 1e-10;
    let f_tol = 1e-10;
    let g_tol = 1e-9;

    // iter 0: post-init state, before any step.
    let state0 = stepper.state();
    check_iterate(0, state0, &fixture[0], x_tol, f_tol, g_tol);

    // iters 1..=30: step then compare.
    for k in 1..=30 {
        let outcome = stepper.step().unwrap();
        match outcome {
            basin::StepOutcome::Continue => {}
            basin::StepOutcome::Stopped(reason) => {
                panic!("stepper halted at iter {} with {:?}", k, reason);
            }
        }
        let state_k = stepper.state();
        check_iterate(k, state_k, &fixture[k as usize], x_tol, f_tol, g_tol);
    }
}

fn check_iterate(
    k: u64,
    state: &LbfgsState<Vec<f64>>,
    expected: &FortranIterate,
    x_tol: f64,
    f_tol: f64,
    g_tol: f64,
) {
    use basin::{GradientState, State};
    assert_eq!(state.iter(), k, "basin iter counter at step {}", k);
    assert_eq!(expected.iter, k, "fixture iter counter at step {}", k);

    let x = state.param();
    for (i, (xi, ex)) in x.iter().zip(&expected.x).enumerate() {
        let diff = (xi - ex).abs();
        assert!(
            diff <= x_tol,
            "iter {} x[{}]: basin = {:.17e}, fortran = {:.17e}, diff = {:.3e}",
            k,
            i,
            xi,
            ex,
            diff
        );
    }
    let f_diff = (state.cost() - expected.f).abs();
    assert!(
        f_diff <= f_tol * expected.f.abs().max(1.0),
        "iter {} f: basin = {:.17e}, fortran = {:.17e}, diff = {:.3e}",
        k,
        state.cost(),
        expected.f,
        f_diff
    );
    let g = state.gradient().expect("gradient cached on state");
    for (i, (gi, eg)) in g.iter().zip(&expected.g).enumerate() {
        let diff = (gi - eg).abs();
        assert!(
            diff <= g_tol * eg.abs().max(1.0),
            "iter {} g[{}]: basin = {:.17e}, fortran = {:.17e}, diff = {:.3e}",
            k,
            i,
            gi,
            eg,
            diff
        );
    }
}
