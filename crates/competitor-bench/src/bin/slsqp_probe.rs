//! Repeat the published SLSQP solve and production LSEI kernels on fixed inputs.
//! Run with `--profile` to collect stacks without mixing profiling and timing.

use std::{hint::black_box, time::Instant};

use basin::{RawEvaluationState, State};
pub use basin::{SlsqpFailure, core};
use competitor_bench::slsqp::{Library, run, workloads};

// Compile the production kernel without exposing private math as public API.
#[path = "../../../basin/src/solver/slsqp/least_squares.rs"]
#[allow(dead_code, unexpected_cfgs)]
mod least_squares;
use least_squares::{Matrix, lsei};

#[derive(Clone)]
struct Kernel {
    c: Matrix<f64>,
    d: Vec<f64>,
    e: Matrix<f64>,
    f: Vec<f64>,
    g: Matrix<f64>,
    h: Vec<f64>,
}

impl Kernel {
    fn new(n: usize, constrained: bool) -> Self {
        let mut e = Matrix::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                e.set(i, j, if i == j { 1.0 + i as f64 } else { 0.25 });
            }
        }
        let f = (0..n).map(|i| (0..n).map(|j| e.get(i, j)).sum()).collect();
        let mut c = Matrix::zeros(usize::from(constrained), n);
        let mut g = Matrix::zeros(usize::from(constrained), n);
        if constrained {
            for j in 0..n {
                c.set(0, j, 1.0);
            }
            g.set(0, 0, 1.0);
        }
        Self {
            c,
            d: if constrained { vec![n as f64] } else { vec![] },
            e,
            f,
            g,
            h: if constrained { vec![0.0] } else { vec![] },
        }
    }

    fn solve(&self) -> (Vec<f64>, Vec<f64>) {
        lsei(
            self.c.clone(),
            &self.d,
            self.e.clone(),
            &self.f,
            self.g.clone(),
            &self.h,
            None,
        )
        .unwrap()
    }
}

fn measure<T>(name: &str, batch: usize, mut solve: impl FnMut() -> T) {
    for _ in 0..batch {
        black_box(solve());
    }
    for sample in 0..21 {
        let start = Instant::now();
        for _ in 0..batch {
            black_box(solve());
        }
        println!(
            "{name},{sample},{}",
            start.elapsed().as_nanos() / batch as u128
        );
    }
}

fn main() {
    let profile = std::env::args().any(|a| a == "--profile");
    let libraries = [
        Library::Basin,
        Library::BasinManual,
        Library::Slsqp,
        Library::Nlopt,
    ];
    for library in libraries {
        let result = run(library);
        result.verify();
        eprintln!(
            "{}: f={:.17e}, x={:?}, nf={}, ng={}, iterations={:?}, status={}",
            library.name(),
            result.cost,
            result.x,
            result.cost_evals,
            result.gradient_evals,
            result.iterations,
            result.status,
        );
    }
    if profile {
        for _ in 0..300_000 {
            black_box(run(Library::Basin));
        }
        return;
    }
    println!("case,sample,ns");
    for library in libraries {
        measure(library.name(), 1000, || run(library));
    }
    let hs71 = workloads::hs71();
    workloads::verify_hs71(&hs71);
    eprintln!(
        "hs71: f={:.17e}, x={:?}, iterations={}, counts={:?}, status={:?}",
        hs71.state.cost(),
        hs71.state.param(),
        hs71.state.iter(),
        hs71.state.raw_counts(),
        hs71.reason,
    );
    measure("basin_hs71", 1000, workloads::hs71);
    for n in [2, 8, 32] {
        for constrained in [false, true] {
            let kernel = Kernel::new(n, constrained);
            let (x, _) = kernel.solve();
            assert!(x.iter().all(|v| (v - 1.0).abs() < 1e-10));
            let name = format!("lsei_n{n}_constrained{constrained}");
            measure(&name, 1000 / n, || black_box(&kernel).solve());
        }
    }
}
