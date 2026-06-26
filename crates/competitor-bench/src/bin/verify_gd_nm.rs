//! One-shot GD and Nelder-Mead comparison (not a benchmark): basin vs
//! argmin on `Vec<f64>`, identical problem, start, configuration, and a
//! fixed iteration budget (no early stop), so the only thing that can
//! differ is the implementation. Prints iterations and the cost reached,
//! confirming the two reach comparable quality before the timings in
//! `benches/gd_nm.rs` are trusted.
//!
//! Run: `cargo run -p competitor-bench --bin verify_gd_nm --release`.

use argmin::core::Executor as ArgminExecutor;
use argmin::core::State;
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::neldermead::NelderMead as ArgminNelderMead;
use basin::problems::{Rosenbrock, rosenbrock, rosenbrock_gradient};
use basin::{
    BasicSimplexState, BasicState, Executor, GradientDescent, IntoInitialSimplex, MoreThuente,
    NelderMead,
};
use competitor_bench::ArgminProblem;

const MAX_ITERS: u64 = 200;

fn main() {
    let start = vec![-1.2, 1.0];
    println!("Rosenbrock 2D from {start:?}, fixed {MAX_ITERS} iters (no early stop)\n");

    // ---- Steepest descent + More-Thuente line search ----
    println!("== gradient descent (steepest + More-Thuente) ==");

    let r = Executor::new(
        Rosenbrock::<Vec<f64>>::default(),
        GradientDescent::with_line_search(MoreThuente::new()),
        BasicState::new(start.clone()),
    )
    .max_iter(MAX_ITERS)
    .run()
    .unwrap();
    println!("  basin   {:>4} iters  cost={:.6e}", r.iter(), r.cost());

    let ls: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> = MoreThuenteLineSearch::new();
    let res = ArgminExecutor::new(
        ArgminProblem::new(rosenbrock, rosenbrock_gradient),
        SteepestDescent::new(ls),
    )
    .configure(|s| s.param(start.clone()).max_iters(MAX_ITERS))
    .run()
    .unwrap();
    println!(
        "  argmin  {:>4} iters  cost={:.6e}",
        res.state().get_iter(),
        res.state().get_best_cost()
    );

    // ---- Nelder-Mead (standard coeffs, identical initial simplex) ----
    println!("\n== Nelder-Mead (standard: α=1, exp=2, con=0.5, shrink=0.5) ==");

    let r = Executor::new(
        Rosenbrock::<Vec<f64>>::default(),
        NelderMead::new(),
        BasicSimplexState::new(start.clone()),
    )
    .max_iter(MAX_ITERS)
    .run()
    .unwrap();
    println!("  basin   {:>4} iters  cost={:.6e}", r.iter(), r.cost());

    // Hand argmin the *exact* simplex basin builds (relative step 0.05),
    // and an sd-tolerance of 0 to disable its early stop so it runs the
    // full budget like basin.
    let simplex = IntoInitialSimplex::into_initial_simplex(start.clone(), 0.05);
    let nm = ArgminNelderMead::new(simplex)
        .with_sd_tolerance(0.0)
        .unwrap();
    let res = ArgminExecutor::new(ArgminProblem::new(rosenbrock, rosenbrock_gradient), nm)
        .configure(|s| s.max_iters(MAX_ITERS))
        .run()
        .unwrap();
    println!(
        "  argmin  {:>4} iters  cost={:.6e}",
        res.state().get_iter(),
        res.state().get_best_cost()
    );
}
