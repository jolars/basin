use globalsearch::local_solver::builders::*;
use globalsearch::local_solver::runner::LocalSolver;
use globalsearch::problem::Problem;
use globalsearch::types::EvaluationError;
use ndarray::{Array1, Array2, array};
use std::cell::{Cell, RefCell};
use std::fs::{self, File};
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::time::Instant;

const STARTS: usize = 24;
const MAX_ITER: u64 = 1000;
const MAX_EVALS: u64 = 20_000;
const ROUNDS: usize = 11;
const TARGET_GAP: f64 = 1e-6;

#[derive(Clone, Copy, Debug)]
enum Function {
    Sphere,
    Ellipsoid,
    Rosenbrock,
    Camel,
}

#[derive(Default)]
struct Stats {
    f: Cell<u64>,
    g: Cell<u64>,
    h: Cell<u64>,
    fingerprint: Cell<u64>,
    best: RefCell<Option<(f64, Array1<f64>)>>,
    hit: Cell<bool>,
    budget: Cell<bool>,
}

impl Stats {
    fn record(&self, kind: u64, x: &Array1<f64>) {
        let mut hash = self.fingerprint.get().wrapping_add(kind);
        for value in x {
            hash = (hash ^ value.to_bits()).wrapping_mul(0x100000001b3);
        }
        self.fingerprint.set(hash);
    }
}

// An observable recurrence supplies CPU work without changing the derivatives
// or adding clock reads and scheduler sleeps to each callback.
#[inline(never)]
fn derivative_work(units: usize) {
    let mut value = black_box(0x37ba51_u64);
    for _ in 0..units {
        value = (value.rotate_left(7) ^ 0x9e3779b97f4a7c15)
            .wrapping_mul(6364136223846793005);
    }
    black_box(value);
}

#[derive(Clone, Copy)]
struct Work {
    mode: &'static str,
    units: usize,
}

struct BenchProblem<'a, const TRACK: bool> {
    case: Case,
    stats: &'a Stats,
    work: Work,
}

impl<const TRACK: bool> Problem for BenchProblem<'_, TRACK> {
    fn objective(&self, x: &Array1<f64>) -> Result<f64, EvaluationError> {
        let f = self.case.value(x);
        self.stats.f.set(self.stats.f.get() + 1);
        if TRACK {
            self.stats.record(1, x);
            if f.is_finite()
                && self
                    .stats
                    .best
                    .borrow()
                    .as_ref()
                    .is_none_or(|(best, _)| f < *best)
            {
                *self.stats.best.borrow_mut() = Some((f, x.clone()));
            }
        }
        if f.is_finite() && f <= self.case.minimum() + TARGET_GAP {
            self.stats.hit.set(true);
            return Err(EvaluationError::InvalidInput {
                reason: "benchmark target reached".into(),
            });
        }
        if self.stats.f.get() >= MAX_EVALS {
            self.stats.budget.set(true);
            return Err(EvaluationError::InvalidInput {
                reason: "benchmark evaluation budget exhausted".into(),
            });
        }
        Ok(f)
    }

    fn gradient(
        &self,
        x: &Array1<f64>,
    ) -> Result<Array1<f64>, EvaluationError> {
        if TRACK {
            self.stats.g.set(self.stats.g.get() + 1);
            self.stats.record(2, x);
        }
        if self.work.mode == "both" && self.work.units > 0 {
            derivative_work(self.work.units);
        }
        Ok(self.case.grad(x))
    }

    fn hessian(&self, x: &Array1<f64>) -> Result<Array2<f64>, EvaluationError> {
        if TRACK {
            self.stats.h.set(self.stats.h.get() + 1);
            self.stats.record(3, x);
        }
        if self.work.units > 0 {
            derivative_work(self.work.units);
        }
        Ok(self.case.hess(x))
    }

    fn variable_bounds(&self) -> Array2<f64> {
        Array2::from_shape_fn((self.case.n, 2), |(_, j)| {
            if j == 0 { -100.0 } else { 100.0 }
        })
    }
}

fn config(basin: bool) -> LocalSolverConfig {
    if basin {
        BasinTrustRegionBuilder::default()
            .max_iter(MAX_ITER)
            .method(TrustRegionRadiusMethod::Steihaug)
            .radius(1.0)
            .max_radius(100.0)
            .eta(0.125)
            .tolerance_grad(None)
            .build()
    } else {
        TrustRegionBuilder::default()
            .max_iter(MAX_ITER)
            .method(TrustRegionRadiusMethod::Steihaug)
            .radius(1.0)
            .max_radius(100.0)
            .eta(0.125)
            .build()
    }
}

#[derive(Debug, PartialEq)]
struct Signature {
    status: &'static str,
    f: u64,
    g: u64,
    h: u64,
    fingerprint: u64,
    best_bits: Option<u64>,
}

fn diagnostic(
    case: Case,
    basin: bool,
    work: Work,
    start: &Array1<f64>,
    id: usize,
    out: &mut impl Write,
) -> Signature {
    let stats = Stats::default();
    let problem = BenchProblem::<true> {
        case,
        stats: &stats,
        work,
    };
    let cfg = config(basin);
    let result =
        LocalSolver::new(problem, cfg.solver_type(), cfg).solve(start.clone());
    let mut returned = f64::NAN;
    let status = match result {
        Ok(solution) => {
            returned = case.value(&solution.point);
            assert!(returned.is_finite());
            assert!(
                (returned - solution.objective).abs()
                    <= 1e-10 * (1.0 + returned.abs())
            );
            "returned"
        }
        Err(_) if stats.hit.get() => "target",
        Err(_) if stats.budget.get() => "budget",
        Err(error) => {
            eprintln!("{} start {id}, basin={basin}: {error}", case.name);
            "error"
        }
    };
    let best = stats.best.borrow();
    let (best_cost, gradient_inf) = if let Some((cost, point)) = best.as_ref() {
        // Verification happens outside timing, including for a target trial
        // point that the solver never had a chance to accept.
        assert_eq!(case.value(point).to_bits(), cost.to_bits());
        let norm = case.grad(point).iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        assert!(norm.is_finite());
        if status == "target" {
            assert!(*cost <= case.minimum() + TARGET_GAP);
        }
        (*cost, norm)
    } else {
        (f64::NAN, f64::NAN)
    };
    writeln!(
        out,
        "{},{},{},{},{},{},{},{},{},{:.12e},{:.12e},{:.12e},{}",
        case.name,
        if basin { "basin" } else { "argmin" },
        work.mode,
        work.units,
        id,
        status,
        stats.f.get(),
        stats.g.get(),
        stats.h.get(),
        best_cost,
        gradient_inf,
        returned,
        stats.fingerprint.get()
    )
    .unwrap();
    Signature {
        status,
        f: stats.f.get(),
        g: stats.g.get(),
        h: stats.h.get(),
        fingerprint: stats.fingerprint.get(),
        best_bits: best.as_ref().map(|(f, _)| f.to_bits()),
    }
}

fn time_batch(
    case: Case,
    basin: bool,
    work: Work,
    starts: &[Array1<f64>],
    ids: &[usize],
    repeats: usize,
) -> f64 {
    let stats = Stats::default();
    let problem = BenchProblem::<false> {
        case,
        stats: &stats,
        work,
    };
    let cfg = config(basin);
    let runner = LocalSolver::new(problem, cfg.solver_type(), cfg);
    let timer = Instant::now();
    for _ in 0..repeats {
        for &id in ids {
            stats.f.set(0);
            stats.hit.set(false);
            stats.budget.set(false);
            let result = black_box(runner.solve(black_box(starts[id].clone())));
            assert!(result.is_err() && stats.hit.get());
        }
    }
    timer.elapsed().as_secs_f64()
}

fn calibrate_work(units: usize) -> f64 {
    let count = (2_000_000 / units.max(1)).max(100);
    let timer = Instant::now();
    for _ in 0..count {
        if units > 0 {
            derivative_work(black_box(units));
        } else {
            black_box(units);
        }
    }
    timer.elapsed().as_secs_f64() / count as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let output = args
        .get(1)
        .expect("usage: steihaug-derivatives OUTPUT [--verify-only]");
    let verify_only = args.iter().any(|arg| arg == "--verify-only");
    let refine = args.iter().any(|arg| arg == "--refine");
    fs::create_dir_all(output).unwrap();
    let mut diagnostics = BufWriter::new(
        File::create(format!("{output}/diagnostics.csv")).unwrap(),
    );
    let mut times =
        BufWriter::new(File::create(format!("{output}/timings.csv")).unwrap());
    let mut calibration = BufWriter::new(
        File::create(format!("{output}/calibration.csv")).unwrap(),
    );
    let mut saved_starts =
        BufWriter::new(File::create(format!("{output}/starts.csv")).unwrap());
    writeln!(diagnostics, "problem,backend,mode,units,start,status,cost_evals,gradient_evals,hessian_evals,best_evaluated,gradient_inf,returned_cost,fingerprint").unwrap();
    writeln!(times, "problem,backend,mode,units,starts,round,repeats,seconds_per_solve,sample_seconds").unwrap();
    writeln!(calibration, "problem,mode,units,round,seconds_per_call").unwrap();
    writeln!(saved_starts, "problem,start,coordinate,value").unwrap();
    let cases = [
        Case {
            name: "rosenbrock2",
            function: Function::Rosenbrock,
            n: 2,
        },
        Case {
            name: "rosenbrock20",
            function: Function::Rosenbrock,
            n: 20,
        },
        Case {
            name: "sphere20",
            function: Function::Sphere,
            n: 20,
        },
        Case {
            name: "ellipsoid20",
            function: Function::Ellipsoid,
            n: 20,
        },
        Case {
            name: "camel2",
            function: Function::Camel,
            n: 2,
        },
    ];
    for (case_id, case) in cases.into_iter().enumerate() {
        if refine && case_id >= 2 {
            continue;
        }
        let starts = case.starts();
        case.verify(&starts);
        for (id, start) in starts.iter().enumerate() {
            for (coordinate, value) in start.iter().enumerate() {
                writeln!(
                    saved_starts,
                    "{},{id},{coordinate},{value:.17e}",
                    case.name
                )
                .unwrap();
            }
        }
        let baseline: Vec<Vec<_>> = [false, true]
            .into_iter()
            .map(|basin| {
                starts
                    .iter()
                    .enumerate()
                    .map(|(id, start)| {
                        diagnostic(
                            case,
                            basin,
                            Work {
                                mode: "baseline",
                                units: 0,
                            },
                            start,
                            id,
                            &mut diagnostics,
                        )
                    })
                    .collect()
            })
            .collect();
        let common: Vec<_> = (0..STARTS)
            .filter(|&id| {
                baseline[0][id].status == "target"
                    && baseline[1][id].status == "target"
            })
            .collect();
        eprintln!(
            "{}: argmin {}/{STARTS}, basin {}/{STARTS}, shared {}",
            case.name,
            baseline[0].iter().filter(|s| s.status == "target").count(),
            baseline[1].iter().filter(|s| s.status == "target").count(),
            common.len()
        );
        // The two Rosenbrock sizes and sphere control receive the full sweep.
        // Keep the other original workloads' failures in the diagnostic corpus.
        if case_id >= 3 {
            continue;
        }
        for mode in ["hessian", "both"] {
            let levels: &[usize] = if refine {
                match (case_id, mode) {
                    (0, "hessian") => &[0, 480, 512, 544, 576],
                    (0, _) => &[0, 224, 256, 288, 320],
                    (1, "hessian") => &[0, 1536, 1664, 1792, 1920],
                    (1, _) => &[0, 768, 832, 896, 960],
                    _ => unreachable!(),
                }
            } else {
                &[0, 32, 128, 512, 2048, 8192, 32768]
            };
            for &units in levels {
                let work = Work { mode, units };
                for (backend, expected) in baseline.iter().enumerate() {
                    for (id, start) in starts.iter().enumerate() {
                        assert_eq!(
                            diagnostic(
                                case,
                                backend == 1,
                                work,
                                start,
                                id,
                                &mut diagnostics
                            ),
                            expected[id],
                            "added work changed a trajectory: {} {mode} {units} start {id}",
                            case.name
                        );
                    }
                }
                if verify_only || common.is_empty() {
                    continue;
                }
                let mut repeats = [1; 2];
                for (backend, count) in repeats.iter_mut().enumerate() {
                    let elapsed = time_batch(
                        case,
                        backend == 1,
                        work,
                        &starts,
                        &common,
                        1,
                    );
                    *count =
                        (0.035 / elapsed).ceil().clamp(1.0, 10000.0) as usize;
                    black_box(time_batch(
                        case,
                        backend == 1,
                        work,
                        &starts,
                        &common,
                        *count,
                    ));
                }
                for round in 0..ROUNDS {
                    writeln!(
                        calibration,
                        "{},{mode},{units},{round},{:.12e}",
                        case.name,
                        calibrate_work(units)
                    )
                    .unwrap();
                    for offset in 0..2 {
                        let backend = (round + offset + case_id) % 2;
                        let elapsed = time_batch(
                            case,
                            backend == 1,
                            work,
                            &starts,
                            &common,
                            repeats[backend],
                        );
                        writeln!(times, "{},{},{mode},{units},{},{round},{},{:.12e},{elapsed:.12e}",
                            case.name, if backend == 1 { "basin" } else { "argmin" },
                            common.len(), repeats[backend], elapsed / (common.len() * repeats[backend]) as f64).unwrap();
                    }
                }
                times.flush().unwrap();
                calibration.flush().unwrap();
                eprintln!("{} {mode} units={units} complete", case.name);
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    function: Function,
    n: usize,
}

impl Case {
    fn minimum(self) -> f64 {
        if matches!(self.function, Function::Camel) {
            -1.031628453489877
        } else {
            0.0
        }
    }

    fn value(self, x: &Array1<f64>) -> f64 {
        match self.function {
            Function::Sphere => x.iter().map(|v| v * v).sum(),
            Function::Ellipsoid => x
                .iter()
                .enumerate()
                .map(|(i, v)| self.weight(i) * v * v)
                .sum(),
            Function::Rosenbrock => (0..self.n - 1)
                .map(|i| {
                    100.0 * (x[i + 1] - x[i] * x[i]).powi(2)
                        + (1.0 - x[i]).powi(2)
                })
                .sum(),
            Function::Camel => {
                4.0 * x[0].powi(2) - 2.1 * x[0].powi(4)
                    + x[0].powi(6) / 3.0
                    + x[0] * x[1]
                    - 4.0 * x[1].powi(2)
                    + 4.0 * x[1].powi(4)
            }
        }
    }

    fn weight(self, i: usize) -> f64 {
        10.0_f64.powf(4.0 * i as f64 / (self.n - 1) as f64)
    }

    fn grad(self, x: &Array1<f64>) -> Array1<f64> {
        let mut g = Array1::zeros(self.n);
        match self.function {
            Function::Sphere => g.assign(&(x * 2.0)),
            Function::Ellipsoid => {
                for i in 0..self.n {
                    g[i] = 2.0 * self.weight(i) * x[i];
                }
            }
            Function::Rosenbrock => {
                for i in 0..self.n - 1 {
                    let t = x[i + 1] - x[i] * x[i];
                    g[i] += -400.0 * x[i] * t + 2.0 * (x[i] - 1.0);
                    g[i + 1] += 200.0 * t;
                }
            }
            Function::Camel => {
                g[0] =
                    8.0 * x[0] - 8.4 * x[0].powi(3) + 2.0 * x[0].powi(5) + x[1];
                g[1] = x[0] - 8.0 * x[1] + 16.0 * x[1].powi(3);
            }
        }
        g
    }

    fn hess(self, x: &Array1<f64>) -> Array2<f64> {
        let mut h = Array2::zeros((self.n, self.n));
        match self.function {
            Function::Sphere => {
                for i in 0..self.n {
                    h[[i, i]] = 2.0;
                }
            }
            Function::Ellipsoid => {
                for i in 0..self.n {
                    h[[i, i]] = 2.0 * self.weight(i);
                }
            }
            Function::Rosenbrock => {
                for i in 0..self.n - 1 {
                    h[[i, i]] += 1200.0 * x[i] * x[i] - 400.0 * x[i + 1] + 2.0;
                    h[[i, i + 1]] = -400.0 * x[i];
                    h[[i + 1, i]] = -400.0 * x[i];
                    h[[i + 1, i + 1]] += 200.0;
                }
            }
            Function::Camel => {
                h[[0, 0]] = 8.0 - 25.2 * x[0] * x[0] + 10.0 * x[0].powi(4);
                h[[0, 1]] = 1.0;
                h[[1, 0]] = 1.0;
                h[[1, 1]] = -8.0 + 48.0 * x[1] * x[1];
            }
        }
        h
    }

    fn starts(self) -> Vec<Array1<f64>> {
        // A fixed generator makes the corpus reproducible without a version-dependent RNG.
        let mut seed = 0x37ba51_u64 + self.n as u64;
        (0..STARTS)
            .map(|k| {
                Array1::from_iter((0..self.n).map(|i| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let u = (seed >> 11) as f64 / ((1u64 << 53) as f64);
                    match self.function {
                        Function::Rosenbrock if k == 0 => {
                            if i % 2 == 0 {
                                -1.2
                            } else {
                                1.0
                            }
                        }
                        Function::Rosenbrock => 4.0 * u - 2.0,
                        Function::Camel => {
                            (2.0 * u - 1.0) * if i == 0 { 2.0 } else { 1.0 }
                        }
                        _ => 6.0 * u - 3.0,
                    }
                }))
            })
            .collect()
    }

    fn verify(self, starts: &[Array1<f64>]) {
        for x in starts.iter().take(3) {
            let g = self.grad(x);
            let h = self.hess(x);
            for j in 0..self.n {
                let step = 1e-5 * x[j].abs().max(1.0);
                let mut xp = x.clone();
                xp[j] += step;
                let mut xm = x.clone();
                xm[j] -= step;
                let fd = (self.value(&xp) - self.value(&xm)) / (2.0 * step);
                assert!(
                    (fd - g[j]).abs() <= 1e-4 * (1.0 + g[j].abs()),
                    "{} gradient {j}",
                    self.name
                );
                let hd = (self.grad(&xp) - self.grad(&xm)) / (2.0 * step);
                for i in 0..self.n {
                    assert!(
                        (hd[i] - h[[i, j]]).abs()
                            <= 1e-5 * (1.0 + h[[i, j]].abs()),
                        "{} Hessian {i},{j}",
                        self.name
                    );
                }
            }
        }
        let optimum = match self.function {
            Function::Rosenbrock => Array1::ones(self.n),
            Function::Camel => array![0.08984201368301331, -0.7126564032704135],
            _ => Array1::zeros(self.n),
        };
        assert!((self.value(&optimum) - self.minimum()).abs() < 1e-12);
        assert!(self.grad(&optimum).iter().all(|v| v.abs() < 1e-7));
    }
}
