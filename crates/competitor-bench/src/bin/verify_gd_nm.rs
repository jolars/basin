//! One-shot GD and Nelder-Mead comparison (not a benchmark): basin vs
//! argmin on `Vec<f64>`, identical problem, start, configuration, and a
//! shared iteration cap. Each solver retains its internal stopping rules.
//! Prints completed iterations and costs; equal caps do not establish equal
//! work. See `verify_nm` for the published Basin/Gomez workload and callback
//! accounting.
//!
//! Also verifies the COBYLA migration cases against `cobyla` and NLopt with
//! matched objective budgets and final radii. Their trajectories differ;
//! reported constraint counts distinguish vector callbacks from scalar callbacks.
//!
//! Run: `cargo run -p competitor-bench --bin verify_gd_nm --release`.

use argmin::core::Executor as ArgminExecutor;
use argmin::core::State;
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::neldermead::NelderMead as ArgminNelderMead;
use basin::problems::{Rosenbrock, rosenbrock, rosenbrock_gradient};
use basin::{
    Executor, FirstOrderState, GradientDescent, IntoInitialSimplex,
    MoreThuente, NelderMead, SimplexProgress,
};
use competitor_bench::ArgminProblem;

const MAX_ITERS: u64 = 200;

fn main() {
    let start = vec![-1.2, 1.0];
    println!("Rosenbrock 2D from {start:?}, cap of {MAX_ITERS} iterations\n");

    // ---- Steepest descent + More-Thuente line search ----
    println!("== gradient descent (steepest + More-Thuente) ==");

    let r = Executor::new(
        Rosenbrock::<Vec<f64>>::default(),
        GradientDescent::with_line_search(MoreThuente::new()),
        FirstOrderState::new(start.clone()),
    )
    .max_iter(MAX_ITERS)
    .run()
    .unwrap();
    println!("  basin   {:>4} iters  cost={:.6e}", r.iter(), r.cost());

    let ls: MoreThuenteLineSearch<Vec<f64>, Vec<f64>, f64> =
        MoreThuenteLineSearch::new();
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
        SimplexProgress::new(start.clone()),
    )
    .max_iter(MAX_ITERS)
    .run()
    .unwrap();
    println!("  basin   {:>4} iters  cost={:.6e}", r.iter(), r.cost());

    // Hand argmin the *exact* simplex basin builds (relative step 0.05),
    // and an sd-tolerance of 0 to disable its standard-deviation stop.
    // Basin retains its default simplex convergence checks.
    let simplex = IntoInitialSimplex::into_initial_simplex(start.clone(), 0.05);
    let nm = ArgminNelderMead::new(simplex)
        .with_sd_tolerance(0.0)
        .unwrap();
    let res = ArgminExecutor::new(
        ArgminProblem::new(rosenbrock, rosenbrock_gradient),
        nm,
    )
    .configure(|s| s.max_iters(MAX_ITERS))
    .run()
    .unwrap();
    println!(
        "  argmin  {:>4} iters  cost={:.6e}",
        res.state().get_iter(),
        res.state().get_best_cost()
    );

    println!("\n== COBYLA (matched objective budgets and final radii) ==");
    for (case, name) in competitor_bench::cobyla::CASES {
        for (library, unit, result) in [
            ("basin", "vector", case.solve_basin()),
            ("cobyla", "scalar nonlinear", case.solve_reference()),
            ("nlopt", "scalar nonlinear", case.solve_nlopt()),
        ] {
            case.verify(&result);
            println!(
                "  {name} {library}: cost={:.12e} violation={:.3e} \
                 objective_calls={} constraint_calls={} ({unit}) \
                 iterations={:?} stop={:?} quality=pass",
                result.cost,
                case.violation(&result.point),
                result.objective_calls,
                result.constraint_calls,
                result.iterations,
                result.stop,
            );
        }
    }
}
