//! One-shot convergence comparison (not a benchmark): runs the lm crate,
//! basin/nalgebra, and basin/faer on each problem once and prints where
//! each landed and how much work it took. Confirms the three solvers
//! agree on the optimum (and do comparable work) before the timings in
//! `benches/compare.rs` are meaningful.
//!
//! Run: `cargo run -p lm-bench --bin verify --release`.

use basin::problems::{ExponentialFit, PowellSingular};
use basin::{Executor, LevenbergMarquardt, NllsState};
use competitor_bench::{
    LM_DEFAULT_TOL, LmExponentialFit, LmPowellSingular, LmUnderDet, LmVarDim, UnderDet, VarDim,
    vardim_start,
};
use faer::Col;
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::DVector;

/// basin LM configured to match the lm crate's default stopping rules:
/// MINPACK gtol/ftol/xtol all at `30·ε`, absolute gradient test off.
fn basin_lm<V, M>() -> LevenbergMarquardt<V, M> {
    LevenbergMarquardt::new()
        .with_tol_grad(0.0)
        .with_tol_grad_rel(LM_DEFAULT_TOL)
        .with_tol_cost_rel(LM_DEFAULT_TOL)
        .with_tol_step_rel(LM_DEFAULT_TOL)
}

fn main() {
    println!("tol (ftol=xtol=gtol) = {LM_DEFAULT_TOL:e}\n");

    // ---- Exponential fit (n = 2, m = 10), poorly scaled ----
    println!("== exp_fit  a*exp(b*t),  optimum (1e5, -1) ==");

    let (prob, rep) = levenberg_marquardt::LevenbergMarquardt::new()
        .minimize(LmExponentialFit::sampled(1.0e5, -1.0, 10, 0.4, 5.0e4, -0.3));
    let p = prob.params();
    println!(
        "  lm-crate    {:>5} evals  cost={:.3e}  a={:.6}  b={:.6}  ({:?})",
        rep.number_of_evaluations, rep.objective_function, p[0], p[1], rep.termination
    );

    let r = Executor::new(
        ExponentialFit::<DVector<f64>>::sampled(1.0e5, -1.0, 10, 0.4),
        basin_lm(),
        NllsState::new(DVector::from_vec(vec![5.0e4, -0.3])),
    )
    .max_iter(200)
    .run()
    .unwrap();
    println!(
        "  basin/nalg  {:>5} iters  cost={:.3e}  a={:.6}  b={:.6}  ({} cost-evals, {:?})",
        r.iter(),
        r.cost(),
        r.param()[0],
        r.param()[1],
        r.cost_evals(),
        r.reason
    );

    let r = Executor::new(
        ExponentialFit::<Col<f64>>::sampled(1.0e5, -1.0, 10, 0.4),
        basin_lm(),
        NllsState::new(Col::from_fn(2, |i| if i == 0 { 5.0e4 } else { -0.3 })),
    )
    .max_iter(200)
    .run()
    .unwrap();
    println!(
        "  basin/faer  {:>5} iters  cost={:.3e}  a={:.6}  b={:.6}  ({} cost-evals, {:?})",
        r.iter(),
        r.cost(),
        r.param()[0],
        r.param()[1],
        r.cost_evals(),
        r.reason
    );

    // ---- Powell singular (n = 4, m = 4), rank-deficient at optimum ----
    println!("\n== powell  optimum 0 ==");

    let (prob, rep) = levenberg_marquardt::LevenbergMarquardt::new()
        .minimize(LmPowellSingular::new([3.0, -1.0, 0.0, 1.0]));
    let p = prob.params();
    println!(
        "  lm-crate    {:>5} evals  cost={:.3e}  x=[{:.2e}, {:.2e}, {:.2e}, {:.2e}]  ({:?})",
        rep.number_of_evaluations, rep.objective_function, p[0], p[1], p[2], p[3], rep.termination
    );

    let r = Executor::new(
        PowellSingular::<DVector<f64>>::new(),
        basin_lm(),
        NllsState::new(DVector::from_vec(vec![3.0, -1.0, 0.0, 1.0])),
    )
    .max_iter(200)
    .run()
    .unwrap();
    let p = r.param();
    println!(
        "  basin/nalg  {:>5} iters  cost={:.3e}  x=[{:.2e}, {:.2e}, {:.2e}, {:.2e}]  ({} cost-evals, {:?})",
        r.iter(),
        r.cost(),
        p[0],
        p[1],
        p[2],
        p[3],
        r.cost_evals(),
        r.reason
    );

    let r = Executor::new(
        PowellSingular::<Col<f64>>::new(),
        basin_lm(),
        NllsState::new(Col::from_fn(4, |i| [3.0, -1.0, 0.0, 1.0][i])),
    )
    .max_iter(200)
    .run()
    .unwrap();
    let p = r.param();
    println!(
        "  basin/faer  {:>5} iters  cost={:.3e}  x=[{:.2e}, {:.2e}, {:.2e}, {:.2e}]  ({} cost-evals, {:?})",
        r.iter(),
        r.cost(),
        p[0],
        p[1],
        p[2],
        p[3],
        r.cost_evals(),
        r.reason
    );

    // ---- Variably Dimensioned (well-conditioned, full-rank) at the
    //      eunoia regime n ∈ {10, 20, 30}. All three should converge in
    //      comparable iterations, so iteration count vs per-iter cost can
    //      be read off cleanly. ----
    for n in [10usize, 20, 30] {
        println!("\n== vardim  n={n}, m={}  optimum 0 ==", n + 2);
        let start = vardim_start(n);

        let (prob, rep) = levenberg_marquardt::LevenbergMarquardt::new().minimize(LmVarDim::new(n));
        println!(
            "  lm-crate    {:>5} evals  cost={:.3e}  ‖x−1‖∞={:.2e}  ({:?})",
            rep.number_of_evaluations,
            rep.objective_function,
            prob.params()
                .iter()
                .map(|&v| (v - 1.0).abs())
                .fold(0.0, f64::max),
            rep.termination
        );

        let r = Executor::new(
            VarDim::<DVector<f64>>::new(n),
            basin_lm(),
            NllsState::new(DVector::from_vec(start.clone())),
        )
        .max_iter(500)
        .run()
        .unwrap();
        println!(
            "  basin/nalg  {:>5} iters  cost={:.3e}  ‖x−1‖∞={:.2e}  ({} cost-evals, {:?})",
            r.iter(),
            r.cost(),
            r.param()
                .iter()
                .map(|&v| (v - 1.0).abs())
                .fold(0.0, f64::max),
            r.cost_evals(),
            r.reason
        );

        let r = Executor::new(
            VarDim::<Col<f64>>::new(n),
            basin_lm(),
            NllsState::new(Col::from_fn(n, |i| start[i])),
        )
        .max_iter(500)
        .run()
        .unwrap();
        println!(
            "  basin/faer  {:>5} iters  cost={:.3e}  ‖x−1‖∞={:.2e}  ({} cost-evals, {:?})",
            r.iter(),
            r.cost(),
            (0..n)
                .map(|i| (r.param()[i] - 1.0).abs())
                .fold(0.0, f64::max),
            r.cost_evals(),
            r.reason
        );
    }

    // ---- Underdetermined trigonometric (m < n, rank-deficient JᵀJ),
    //      issue #10's regime: infeasible so the solver iterates with
    //      rejected steps. eunoia's slow sizes are (n=15,m=7) and
    //      (n=20,m=15). The lm crate factors J once per outer iteration
    //      and reuses it across the inner λ-loop; basin's per-iteration
    //      cost on this regime is what #10 tracks. ----
    for (m, n) in [(7usize, 15usize), (15, 20), (12, 25)] {
        println!("\n== underdet  n={n}, m={m}  (m < n) ==");

        let (prob, rep) =
            levenberg_marquardt::LevenbergMarquardt::new().minimize(LmUnderDet::new(m, n));
        println!(
            "  lm-crate    {:>5} evals  cost={:.6e}  ‖x‖∞={:.2e}  ({:?})",
            rep.number_of_evaluations,
            rep.objective_function,
            prob.params().iter().map(|&v| v.abs()).fold(0.0, f64::max),
            rep.termination
        );

        let p = UnderDet::<DVector<f64>>::new(m, n);
        let x0 = DVector::from_vec(p.start());
        let r = Executor::new(p, basin_lm(), NllsState::new(x0))
            .max_iter(500)
            .run()
            .unwrap();
        println!(
            "  basin/nalg  {:>5} iters  cost={:.6e}  ‖x‖∞={:.2e}  ({} cost-evals, {:?})",
            r.iter(),
            r.cost(),
            r.param().iter().map(|&v| v.abs()).fold(0.0, f64::max),
            r.cost_evals(),
            r.reason
        );

        let p = UnderDet::<Col<f64>>::new(m, n);
        let start = p.start();
        let x0 = Col::from_fn(n, |i| start[i]);
        let r = Executor::new(p, basin_lm(), NllsState::new(x0))
            .max_iter(500)
            .run()
            .unwrap();
        println!(
            "  basin/faer  {:>5} iters  cost={:.6e}  ‖x‖∞={:.2e}  ({} cost-evals, {:?})",
            r.iter(),
            r.cost(),
            (0..n).map(|i| r.param()[i].abs()).fold(0.0, f64::max),
            r.cost_evals(),
            r.reason
        );
    }
}
