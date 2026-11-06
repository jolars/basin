use basin::problems::Rosenbrock;
use basin::{
    BoxBounds, Constrained, ConstraintSet, Executor, FirstOrderState,
    GradientDescent, LineSearch, LineSearchOutcome, Problem, ScaledAdd, Solver,
    Tnc, Wolfe,
};
use std::hint::black_box;
use std::time::{Duration, Instant};

fn bench<S, R>(
    name: &str,
    iters: u32,
    mut setup: impl FnMut() -> S,
    mut run: impl FnMut(S) -> R,
) {
    for _ in 0..3 {
        let _ = black_box(run(setup()));
    }
    let mut total = Duration::ZERO;
    for _ in 0..iters {
        let s = setup();
        let t0 = Instant::now();
        let _ = black_box(run(s));
        total += t0.elapsed();
    }
    let per = total / iters;
    println!("{name:40}  {per:>10.3?} / iter  ({iters} iters)");
}

fn main() {
    for n in [2, 10, 50] {
        for fused in [false, true] {
            bench(
                &format!("tnc_rosenbrock_n{n}/fused={fused}"),
                100,
                || {
                    let problem = Constrained::new(
                        Rosenbrock::<Vec<f64>>::default(),
                        ConstraintSet::new().with_bounds(BoxBounds::new(
                            Some(vec![-2.0; n]),
                            Some(vec![2.0; n]),
                        )),
                    );
                    let start = (0..n)
                        .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
                        .collect();
                    (problem, FirstOrderState::new(start))
                },
                |(problem, state)| {
                    if fused {
                        Executor::new(
                            problem,
                            Tnc::with_line_search(
                                Wolfe::new()
                                    .max_iter(60)
                                    .with_fused_evaluations(),
                            ),
                            state,
                        )
                        .max_iter(5000)
                        .run()
                        .unwrap()
                    } else {
                        Executor::new(problem, Tnc::new(), state)
                            .max_iter(5000)
                            .run()
                            .unwrap()
                    }
                },
            );
        }
    }

    // Same Wolfe trial policy and accepted point; isolate the cost of throwing
    // away the search's evaluation and requesting it again in the caller.
    for retain in [false, true] {
        bench(
            if retain {
                "wolfe_rosenbrock_retain_evaluation"
            } else {
                "wolfe_rosenbrock_repeat_evaluation"
            },
            100_000,
            || {
                let mut problem =
                    Problem::new(Rosenbrock::<Vec<f64>>::default());
                let x = vec![-1.2, 1.0];
                let (cost, gradient) = problem.cost_and_gradient(&x).unwrap();
                let direction: Vec<_> = gradient.iter().map(|g| -g).collect();
                (problem, x, cost, gradient, direction)
            },
            |(mut problem, x, cost, gradient, direction)| {
                let result = Wolfe::new()
                    .next_with_evaluation(
                        &mut problem,
                        &x,
                        cost,
                        &gradient,
                        &direction,
                    )
                    .unwrap();
                let LineSearchOutcome::Step(alpha) = result.outcome else {
                    panic!("expected a Wolfe step")
                };
                if let Some(evaluation) = result.evaluation.filter(|_| retain) {
                    return (
                        evaluation.param,
                        evaluation.cost,
                        evaluation.gradient,
                    );
                }
                let mut next = x;
                next.scaled_add(alpha, &direction);
                let (cost, gradient) =
                    problem.cost_and_gradient(&next).unwrap();
                (next, cost, gradient)
            },
        );
    }

    bench(
        "gradient_descent_step",
        100_000,
        || {
            let mut solver = GradientDescent::new(0.001);
            let mut problem = Problem::new(Rosenbrock::<Vec<f64>>::default());
            // `Solver::init` populates cost + gradient at the initial param,
            // matching the contract `next_iter` expects (gradient cached
            // from the previous iter or from init).
            let state = solver
                .init(&mut problem, FirstOrderState::new(vec![-1.2, 1.0]))
                .unwrap();
            (solver, problem, state)
        },
        |(mut solver, mut problem, state)| {
            solver.next_iter(&mut problem, state).unwrap()
        },
    );

    bench(
        "gradient_descent_full_run_10k",
        50,
        || (),
        |_| {
            Executor::new(
                Rosenbrock::<Vec<f64>>::default(),
                GradientDescent::new(0.001),
                FirstOrderState::new(vec![-1.2, 1.0]),
            )
            .max_iter(10_000)
            .run()
        },
    );
}
