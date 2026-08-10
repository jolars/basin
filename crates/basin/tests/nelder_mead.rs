use basin::problems::{Rosenbrock, Sphere};
use basin::{
    BasicSimplexState, CostFunction, Executor, NelderMead, SimplexState,
    SimplexTolerance, TerminationReason,
};

#[test]
fn nelder_mead_standard_minimises_rosenbrock() {
    let problem = Rosenbrock::<Vec<f64>>::default();
    let initial_cost = problem.cost(&vec![-1.2, 1.0]).unwrap();

    let result = Executor::new(
        problem,
        NelderMead::new(),
        BasicSimplexState::new(vec![-1.2, 1.0]),
    )
    .max_iter(2_000)
    .run()
    .unwrap();

    assert!(
        result.cost() < 1e-6,
        "expected near-zero cost, got {} (initial {})",
        result.cost(),
        initial_cost
    );
    let best = result.param();
    assert!((best[0] - 1.0).abs() < 1e-3, "x[0] = {}", best[0]);
    assert!((best[1] - 1.0).abs() < 1e-3, "x[1] = {}", best[1]);
}

#[test]
fn nelder_mead_adaptive_minimises_rosenbrock() {
    let problem = Rosenbrock::<Vec<f64>>::default();

    let result = Executor::new(
        problem,
        NelderMead::adaptive(),
        BasicSimplexState::new(vec![-1.2, 1.0]),
    )
    .max_iter(2_000)
    .run()
    .unwrap();

    // For n = 2, adaptive collapses to standard, so the bar is the same.
    assert!(result.cost() < 1e-6, "cost = {}", result.cost());
}

#[test]
fn nelder_mead_hits_max_iter_when_too_few() {
    let problem = Rosenbrock::<Vec<f64>>::default();

    let result = Executor::new(
        problem,
        NelderMead::new(),
        BasicSimplexState::new(vec![-1.2, 1.0]),
    )
    .max_iter(5)
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::MaxIter);
    assert_eq!(result.iter(), 5);
}

#[test]
fn nelder_mead_keeps_best_first_after_each_iter() {
    // Quick invariant: after run completion, costs[] is ascending.
    let problem = Rosenbrock::<Vec<f64>>::default();
    let result = Executor::new(
        problem,
        NelderMead::new(),
        BasicSimplexState::new(vec![-1.2, 1.0]),
    )
    .max_iter(100)
    .run()
    .unwrap();

    for w in result.state.costs().windows(2) {
        assert!(
            w[0] <= w[1],
            "simplex not sorted: {:?}",
            result.state.costs()
        );
    }
}

#[test]
fn nelder_mead_adaptive_sphere_5d() {
    // Derivative-free path on a trivial convex problem.
    let problem = Sphere::<Vec<f64>>::default();

    let result = Executor::new(
        problem,
        NelderMead::adaptive(),
        BasicSimplexState::new(vec![1.0; 5]),
    )
    .max_iter(2_000)
    .run()
    .unwrap();

    assert!(result.cost() < 1e-8, "cost = {}", result.cost());
}

#[test]
fn simplex_tolerance_fires_when_simplex_collapses() {
    let problem = Rosenbrock::<Vec<f64>>::default();

    let result = Executor::new(
        problem,
        NelderMead::new(),
        BasicSimplexState::new(vec![-1.2, 1.0]),
    )
    .max_iter(2_000)
    .terminate_on(SimplexTolerance::new(1e-8, 1e-8))
    .run()
    .unwrap();

    assert_eq!(result.reason, TerminationReason::SimplexTolerance);
    assert!(result.iter() > 0 && result.iter() < 2_000);

    // Verify the (T1) invariant actually holds at termination.
    let vertices = result.state.vertices();
    let costs = result.state.costs();
    for x_i in &vertices[1..] {
        let max = x_i
            .iter()
            .zip(&vertices[0])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max <= 1e-8, "‖x_i − x_1‖_∞ = {} exceeds 1e-8", max);
    }
    for &f_i in &costs[1..] {
        assert!((f_i - costs[0]).abs() <= 1e-8);
    }
}

#[test]
fn nelder_mead_from_simplex_accepts_custom_geometry() {
    // Verifies the escape hatch still works for users who want a custom
    // initial simplex (regular simplex with edge length 1, not FMINSEARCH-style).
    let problem = Rosenbrock::<Vec<f64>>::default();
    let simplex = vec![vec![-1.2, 1.0], vec![-0.2, 1.0], vec![-1.2, 2.0]];

    let result = Executor::new(
        problem,
        NelderMead::new(),
        BasicSimplexState::from_simplex(simplex),
    )
    .max_iter(2_000)
    .run()
    .unwrap();

    assert!(result.cost() < 1e-6);
}
