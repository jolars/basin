use basin::problems::Rosenbrock;
use basin::{BasicState, Executor, GradientDescent, Problem, Solver};
use std::hint::black_box;
use std::time::{Duration, Instant};

fn bench<S, R>(name: &str, iters: u32, mut setup: impl FnMut() -> S, mut run: impl FnMut(S) -> R) {
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
                .init(&mut problem, BasicState::new(vec![-1.2, 1.0]))
                .unwrap();
            (solver, problem, state)
        },
        |(mut solver, mut problem, state)| solver.next_iter(&mut problem, state).unwrap(),
    );

    bench(
        "gradient_descent_full_run_10k",
        50,
        || (),
        |_| {
            Executor::new(
                Rosenbrock::<Vec<f64>>::default(),
                GradientDescent::new(0.001),
                BasicState::new(vec![-1.2, 1.0]),
            )
            .max_iter(10_000)
            .run()
        },
    );
}
