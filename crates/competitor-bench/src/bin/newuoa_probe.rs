//! Unobserved NEWUOA solves, including initialization and teardown.
//!
//! Usage: `newuoa_probe [styblinski|rosenbrock|quadratic] [n] [samples]
//! [batch] [all|basin|nlopt] [absolute objective error target]`.
//! Omit the target to complete the final-radius stage. Verification calls run
//! outside the timer and are excluded from the reported objective counts.

use std::{convert::Infallible, hint::black_box, time::Instant};

use basin::{CostFunction, Executor, Newuoa, PointState};

#[derive(Clone, Copy)]
enum Case {
    Styblinski,
    Rosenbrock,
    Quadratic,
}

impl Case {
    fn cost(self, x: &[f64]) -> f64 {
        match self {
            Self::Styblinski => basin::problems::styblinski_tang(x),
            Self::Rosenbrock => basin::problems::rosenbrock(x),
            Self::Quadratic => x
                .iter()
                .enumerate()
                .map(|(i, v)| (i + 1) as f64 * (v - 1.0).powi(2))
                .sum(),
        }
    }

    fn start(self, n: usize) -> Vec<f64> {
        match self {
            Self::Styblinski | Self::Quadratic => vec![0.0; n],
            Self::Rosenbrock => (0..n)
                .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
                .collect(),
        }
    }

    fn optimum(self, n: usize) -> f64 {
        match self {
            Self::Styblinski => {
                let mut t = -2.903534_f64;
                for _ in 0..8 {
                    t -= (2.0 * t.powi(3) - 16.0 * t + 2.5)
                        / (6.0 * t * t - 16.0);
                }
                self.cost(&vec![t; n])
            }
            Self::Rosenbrock | Self::Quadratic => 0.0,
        }
    }

    fn gradient_norm(self, x: &[f64]) -> f64 {
        let mut gradient = vec![0.0; x.len()];
        match self {
            Self::Styblinski => {
                for (g, v) in gradient.iter_mut().zip(x) {
                    *g = 2.0 * v.powi(3) - 16.0 * v + 2.5;
                }
            }
            Self::Rosenbrock => {
                basin::problems::rosenbrock_gradient(x, &mut gradient);
            }
            Self::Quadratic => {
                for (i, (g, v)) in gradient.iter_mut().zip(x).enumerate() {
                    *g = 2.0 * (i + 1) as f64 * (v - 1.0);
                }
            }
        }
        gradient.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
}

impl CostFunction for Case {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok((*self).cost(x))
    }
}

struct Outcome {
    x: Vec<f64>,
    f: f64,
    nf: u64,
    iterations: Option<u64>,
    status: String,
}

fn solve(case: Case, n: usize, library: &str, target: Option<f64>) -> Outcome {
    if library == "basin" {
        let mut executor = Executor::new(
            case,
            Newuoa::new()
                .with_initial_radius(0.5)
                .with_final_radius(1e-6),
            PointState::new(case.start(n)),
        )
        .max_iter(10_000)
        .max_cost_evals(10_000);
        if let Some(target) = target {
            executor = executor.target_objective(target);
        }
        let result = executor.run().unwrap();
        Outcome {
            x: result.param().clone(),
            f: result.cost(),
            nf: result.cost_evals(),
            iterations: Some(result.iter()),
            status: format!("{:?}", result.report.code()),
        }
    } else {
        let mut opt = nlopt::Nlopt::new(
            nlopt::Algorithm::Newuoa,
            n,
            move |x: &[f64], _: Option<&mut [f64]>, nf: &mut u64| {
                *nf += 1;
                case.cost(x)
            },
            nlopt::Target::Minimize,
            0,
        );
        opt.set_initial_step1(0.5).unwrap();
        opt.set_xtol_abs1(1e-6).unwrap();
        opt.set_maxeval(10_000).unwrap();
        if let Some(target) = target {
            opt.set_stopval(target).unwrap();
        }
        let mut x = case.start(n);
        let result = opt.optimize(&mut x);
        let (status, f) = match result {
            Ok((status, f)) => (format!("{status:?}"), f),
            Err((status, f)) => (format!("{status:?}"), f),
        };
        Outcome {
            x,
            f,
            nf: opt.recover_user_data(),
            iterations: None,
            status,
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let case = match args.first().map(String::as_str).unwrap_or("styblinski") {
        "styblinski" => Case::Styblinski,
        "rosenbrock" => Case::Rosenbrock,
        "quadratic" => Case::Quadratic,
        other => panic!("unknown case: {other}"),
    };
    let number = |i: usize, default| {
        args.get(i)
            .map(|s| s.parse::<usize>().unwrap())
            .unwrap_or(default)
    };
    let n = number(1, 30);
    let samples = number(2, 21);
    let batch = number(3, 5);
    assert!(n >= 2 && samples > 0 && batch > 0);
    let libraries: &[&str] =
        match args.get(4).map(String::as_str).unwrap_or("all") {
            "all" => &["basin", "nlopt"],
            "basin" => &["basin"],
            "nlopt" => &["nlopt"],
            other => panic!("unknown library: {other}"),
        };
    let optimum = case.optimum(n);
    let target = args.get(5).map(|s| {
        let error = s.parse::<f64>().unwrap();
        assert!(error.is_finite() && error >= 0.0);
        optimum + error
    });
    for &library in libraries {
        let result = solve(case, n, library, target);
        let verified = case.cost(&result.x);
        assert!(result.x.iter().all(|v| v.is_finite()));
        assert!(verified.is_finite());
        assert!((verified - result.f).abs() <= 1e-12 * (1.0 + verified.abs()));
        eprintln!(
            "{library},n={n},f={:.17e},error={:.6e},gradient_norm={:.6e},nf={},iterations={:?},status={},target_met={:?},x={:?}",
            result.f,
            verified - optimum,
            case.gradient_norm(&result.x),
            result.nf,
            result.iterations,
            result.status,
            target.map(|target| verified <= target),
            result.x,
        );
    }
    println!("library,sample,ns");
    for sample in 0..samples {
        // Alternate contestant order to reduce bias from clock and thermal drift.
        for offset in 0..libraries.len() {
            let library = libraries[(sample + offset) % libraries.len()];
            let start = Instant::now();
            for _ in 0..batch {
                black_box(solve(black_box(case), n, library, target));
            }
            println!(
                "{library},{sample},{}",
                start.elapsed().as_nanos() / batch as u128
            );
        }
    }
}
