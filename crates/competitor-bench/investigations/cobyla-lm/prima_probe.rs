//! Built in an isolated workspace by reproduce_cobyla_prima.py.
#![allow(dead_code, clippy::needless_range_loop)]

pub use basin::core;
mod raw;

use basin::{
    Cobyla, CostFunction, Executor, MaxCostEvals,
    NonlinearInequalityConstraints, TerminationReason,
};
use serde::{Deserialize, Serialize};
use std::{
    cell::Cell, convert::Infallible, ffi::c_void, hint::black_box,
    time::Instant,
};

unsafe extern "C" {
    fn basin_prima_solve(
        n: i32,
        m: i32,
        x: *mut f64,
        callback: unsafe extern "C" fn(
            *const f64,
            *mut f64,
            *mut f64,
            *const c_void,
        ),
        data: *const c_void,
        budget: i32,
        f: *mut f64,
        cv: *mut f64,
        nf: *mut i32,
        info: *mut i32,
    );
    fn basin_prima_lp(
        n: i32,
        m: i32,
        a: *const f64,
        b: *const f64,
        delta: f64,
        g: *const f64,
        d: *mut f64,
    );
}

#[derive(Clone, Copy)]
enum Case {
    Camel,
    Sphere,
    SphereN(usize),
    Quadratic,
}

impl Case {
    fn parse(name: &str) -> Self {
        match name {
            "camel" => Self::Camel,
            "sphere" => Self::Sphere,
            "quadratic" => Self::Quadratic,
            s => Self::SphereN(
                s.strip_prefix("sphere_").unwrap().parse().unwrap(),
            ),
        }
    }
    fn start(self) -> Vec<f64> {
        match self {
            Self::Camel => vec![0.0, 0.0],
            Self::Quadratic => vec![0.5, 0.5],
            Self::Sphere => {
                vec![2.5, -2.0, 1.5, -1.0, 0.5, 2.25, -1.75, 1.25, -0.75, 0.25]
            }
            Self::SphereN(n) => {
                (0..n).map(|i| ((i * 7) % 17) as f64 / 4.0 - 2.0).collect()
            }
        }
    }
    fn n(self) -> usize {
        match self {
            Self::Camel | Self::Quadratic => 2,
            Self::Sphere => 10,
            Self::SphereN(n) => n,
        }
    }
    fn m(self) -> usize {
        2 * self.n() + usize::from(matches!(self, Self::Quadratic))
    }
    fn budget(self) -> usize {
        match self {
            Self::Camel => 50,
            Self::Quadratic => 100,
            _ => 20 * self.n(),
        }
    }
    fn bounds(self, i: usize) -> (f64, f64) {
        match self {
            Self::Camel => {
                if i == 0 {
                    (-3.0, 3.0)
                } else {
                    (-2.0, 2.0)
                }
            }
            Self::Quadratic => (0.0, 2.0),
            _ => (-5.0, 5.0),
        }
    }
    fn objective(self, x: &[f64]) -> f64 {
        match self {
            Self::Camel => {
                (4.0 - 2.1 * x[0].powi(2) + x[0].powi(4) / 3.0) * x[0].powi(2)
                    + x[0] * x[1]
                    + (-4.0 + 4.0 * x[1].powi(2)) * x[1].powi(2)
            }
            Self::Quadratic => (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2),
            _ => x.iter().map(|v| v * v).sum(),
        }
    }
    fn constraints(self, x: &[f64], c: &mut [f64]) {
        let offset = usize::from(matches!(self, Self::Quadratic));
        if offset == 1 {
            c[0] = -(1.5 - x[0] - x[1]);
        }
        for (i, &v) in x.iter().enumerate() {
            let (lower, upper) = self.bounds(i);
            c[offset + 2 * i] = lower - v;
            c[offset + 2 * i + 1] = v - upper;
        }
    }
    fn quality(self, f: f64, cv: f64) -> bool {
        let (target, tolerance) = match self {
            Self::Camel => (-1.031628453489877, 1e-5),
            Self::Quadratic => (0.125, 1e-6),
            Self::Sphere => (0.0, 1e-5),
            Self::SphereN(_) => (0.0, 1e-6),
        };
        (f - target).abs() < tolerance && cv <= f64::EPSILON.sqrt()
    }
}

struct Problem {
    case: Case,
    nf: Cell<usize>,
    nc: Cell<usize>,
}
impl Problem {
    fn new(case: Case) -> Self {
        Self {
            case,
            nf: Cell::new(0),
            nc: Cell::new(0),
        }
    }
    fn objective(&self, x: &[f64], guard: bool) -> f64 {
        if guard && self.nf.get() >= self.case.budget() {
            return f64::INFINITY;
        }
        self.nf.set(self.nf.get() + 1);
        self.case.objective(x)
    }
    fn constraints_into(&self, x: &[f64], c: &mut [f64]) {
        self.nc.set(self.nc.get() + 1);
        self.case.constraints(x, c);
    }
    fn constraints_vec(&self, x: &[f64]) -> Vec<f64> {
        let mut c = vec![0.0; self.case.m()];
        self.constraints_into(x, &mut c);
        c
    }
}
impl CostFunction for &Problem {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        Ok(self.objective(x, true))
    }
}
impl NonlinearInequalityConstraints for &Problem {
    fn num_constraints(&self) -> usize {
        self.case.m()
    }
    fn constraints(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        Ok(self.constraints_vec(x))
    }
}

unsafe extern "C" fn callback(
    x: *const f64,
    f: *mut f64,
    c: *mut f64,
    data: *const c_void,
) {
    // The synchronous PRIMA call retains this context and supplies n/m elements.
    unsafe {
        let p = &*data.cast::<Problem>();
        let x = std::slice::from_raw_parts(x, p.case.n());
        *f = p.objective(x, false);
        p.constraints_into(x, std::slice::from_raw_parts_mut(c, p.case.m()));
    }
}

#[derive(Serialize)]
struct Outcome {
    x: Vec<f64>,
    f: f64,
    objective_calls: usize,
    constraint_calls: usize,
    wrapper_calls: Option<u64>,
    iterations: Option<u64>,
    stop: &'static str,
    native_status: Option<i32>,
    current: Option<(Vec<f64>, f64)>,
}

fn solve(mode: &str, case: Case) -> Outcome {
    let p = Problem::new(case);
    let mut x = case.start();
    let mut result = match mode {
        "prima" => {
            let (mut f, mut cv, mut nf, mut info) = (0.0, 0.0, 0, 0);
            // Both bindings use 64-bit reals and C integers; x lives through the call.
            unsafe {
                basin_prima_solve(
                    case.n() as i32,
                    case.m() as i32,
                    x.as_mut_ptr(),
                    callback,
                    (&p as *const Problem).cast(),
                    case.budget() as i32,
                    &mut f,
                    &mut cv,
                    &mut nf,
                    &mut info,
                );
            }
            assert_eq!(nf as usize, p.nf.get());
            Outcome {
                x,
                f,
                objective_calls: 0,
                constraint_calls: 0,
                wrapper_calls: None,
                iterations: None,
                stop: match info {
                    0 => "rho_end",
                    3 => "budget",
                    _ => "other",
                },
                native_status: Some(info),
                current: None,
            }
        }
        "raw" => {
            let mut eval = |x: &[f64]| -> Result<_, Infallible> {
                Ok((p.objective(x, true), p.constraints_vec(x)))
            };
            let (mut work, _, _) = raw::driver::CobylaWork::try_init(
                x,
                case.m(),
                0.5,
                f64::EPSILON.sqrt() * 0.5,
                &mut eval,
            )
            .unwrap();
            let mut iterations = 0;
            let mut stop = "budget";
            while p.nc.get() < case.budget() && iterations < 10000 {
                let transition = work.step(&mut eval).unwrap();
                iterations += 1;
                match transition {
                    raw::driver::Transition::Converged => {
                        stop = "rho_end";
                        break;
                    }
                    raw::driver::Transition::Failed => {
                        stop = "failed";
                        break;
                    }
                    _ => {}
                }
            }
            if iterations == 10000 {
                stop = "max_iter";
            }
            let (x, f) = work.best();
            Outcome {
                x,
                f,
                objective_calls: 0,
                constraint_calls: 0,
                wrapper_calls: None,
                iterations: Some(iterations),
                stop,
                native_status: None,
                current: None,
            }
        }
        "executor" => {
            let r = Executor::from_start(
                &p,
                Cobyla::new()
                    .with_rho_beg(0.5)
                    .with_rho_end(f64::EPSILON.sqrt() * 0.5),
                x,
            )
            .max_iter(10000)
            .terminate_on(MaxCostEvals(case.budget() as u64))
            .run()
            .unwrap();
            Outcome {
                x: r.best_param().clone(),
                f: r.best_cost(),
                objective_calls: 0,
                constraint_calls: 0,
                wrapper_calls: Some(r.cost_evals()),
                iterations: Some(r.iter()),
                stop: match r.reason {
                    TerminationReason::SolverConverged => "rho_end",
                    TerminationReason::MaxCostEvals => "budget",
                    _ => "other",
                },
                native_status: None,
                current: Some((r.param().clone(), r.cost())),
            }
        }
        _ => panic!("unknown mode {mode}"),
    };
    result.objective_calls = p.nf.get();
    result.constraint_calls = p.nc.get();
    result
}

#[derive(Deserialize, Serialize)]
struct Lp {
    n: usize,
    m: usize,
    a: Vec<f64>,
    b: Vec<f64>,
    delta: f64,
    g: Vec<f64>,
}

#[cfg(feature = "capture")]
fn capture_lp<F: core::math::Scalar>(
    n: usize,
    a: &[F],
    b: &[F],
    delta: F,
    g: &[F],
) {
    use std::io::Write;
    let path = std::env::var("COBYLA_LP_TRACE")
        .expect("capture requires COBYLA_LP_TRACE");
    let lp = Lp {
        n,
        m: b.len(),
        a: a.iter().map(|x| x.to_f64().unwrap()).collect(),
        b: b.iter().map(|x| x.to_f64().unwrap()).collect(),
        delta: delta.to_f64().unwrap(),
        g: g.iter().map(|x| x.to_f64().unwrap()).collect(),
    };
    let mut output = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();
    writeln!(output, "{}", serde_json::to_string(&lp).unwrap()).unwrap();
}

fn reference_lp(lp: &Lp, d: &mut [f64]) {
    // Shapes were validated on load; A is already column-major.
    unsafe {
        basin_prima_lp(
            lp.n as i32,
            lp.m as i32,
            lp.a.as_ptr(),
            lp.b.as_ptr(),
            lp.delta,
            lp.g.as_ptr(),
            d.as_mut_ptr(),
        );
    }
}

fn lp_metrics(lp: &Lp, d: &[f64]) -> (f64, f64, f64) {
    let dot =
        |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let cv =
        lp.a.chunks_exact(lp.n)
            .zip(&lp.b)
            .map(|(a, b)| dot(a, d) - b)
            .fold(0.0, f64::max);
    (dot(d, d).sqrt(), cv, dot(&lp.g, d))
}

fn kernels(mode: &str, path: &str, repeats: usize) {
    let inputs: Vec<Lp> = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert!(!inputs.is_empty());
    let n = inputs[0].n;
    let m = inputs[0].m;
    for lp in &inputs {
        assert!(lp.n == n && lp.m == m && n > 0 && lp.delta > 0.0);
        assert!(lp.a.len() == n * m && lp.b.len() == m && lp.g.len() == n);
        assert!(lp.a.iter().chain(&lp.b).chain(&lp.g).all(|x| x.is_finite()));
    }
    let mut work = raw::trstlp::TrstlpWork::new(n, m);
    let mut d = vec![0.0; n];
    let mut max_step_error = 0.0_f64;
    let mut max_metric_error = 0.0_f64;
    let mut mismatches = 0;
    let mut basin_step_bits = Vec::with_capacity(inputs.len());
    let mut reference_step_bits = Vec::with_capacity(inputs.len());
    for lp in &inputs {
        let basin = work.solve(&lp.a, &lp.b, lp.delta, &lp.g);
        reference_lp(lp, &mut d);
        let step_error = basin
            .iter()
            .zip(&d)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
            / lp.delta;
        let (bn, bc, bf) = lp_metrics(lp, basin);
        let (pn, pc, pf) = lp_metrics(lp, &d);
        let cscale =
            lp.b.iter()
                .map(|x| x.abs())
                .fold(1.0, f64::max)
                .max(lp.delta * lp.a.iter().map(|x| x.abs()).sum::<f64>());
        let fscale = (lp.delta * lp.g.iter().map(|x| x.abs()).sum::<f64>())
            .max(f64::MIN_POSITIVE);
        let metric_error =
            ((bc - pc).abs() / cscale).max((bf - pf).abs() / fscale);
        assert!(basin.iter().chain(&d).all(|x| x.is_finite()));
        assert!(bn <= lp.delta * (1.0 + 1e-7) && pn <= lp.delta * (1.0 + 1e-7));
        if metric_error > 1e-7 || step_error > 1e-6 {
            mismatches += 1;
        }
        max_step_error = max_step_error.max(step_error);
        max_metric_error = max_metric_error.max(metric_error);
        basin_step_bits
            .push(basin.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        reference_step_bits
            .push(d.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    }
    assert_eq!(
        mismatches, 0,
        "LP reference verification failed before timing"
    );
    let mut run = || {
        for lp in &inputs {
            let lp = black_box(lp);
            match mode {
                "lp-basin" => {
                    black_box(work.solve(&lp.a, &lp.b, lp.delta, &lp.g));
                }
                "lp-prima" => {
                    reference_lp(lp, &mut d);
                    black_box(&d);
                }
                "lp-setup" => {
                    black_box(raw::trstlp::TrstlpWork::<f64>::new(n, m));
                }
                _ => panic!("unknown kernel mode"),
            }
        }
    };
    for _ in 0..16 {
        run();
    }
    let start = Instant::now();
    for _ in 0..repeats {
        run();
    }
    let ns =
        start.elapsed().as_nanos() as f64 / (repeats * inputs.len()) as f64;
    println!(
        "{}",
        serde_json::json!({"mode":mode,"ns":ns,"inputs":inputs.len(),
        "mismatches":mismatches,"max_relative_step_error":max_step_error,
        "max_scaled_metric_error":max_metric_error,
        "basin_step_bits":basin_step_bits,"reference_step_bits":reference_step_bits})
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        4,
        "usage: probe <prima|raw|executor|lp-basin|lp-prima|lp-setup> <case|trace> <repeats>"
    );
    let mode = &args[1];
    let repeats: usize = args[3].parse().unwrap();
    assert!(repeats > 0);
    if mode.starts_with("lp-") {
        kernels(mode, &args[2], repeats);
        return;
    }
    let case = Case::parse(&args[2]);
    #[cfg(not(feature = "capture"))]
    let ns = {
        for _ in 0..16 {
            black_box(solve(mode, case));
        }
        let start = Instant::now();
        for _ in 0..repeats {
            black_box(solve(mode, case));
        }
        start.elapsed().as_nanos() as f64 / repeats as f64
    };
    #[cfg(feature = "capture")]
    let ns = 0.0;
    let r = solve(mode, case);
    let f = case.objective(&r.x);
    let mut c = vec![0.0; case.m()];
    case.constraints(&r.x, &mut c);
    let cv = c.iter().copied().fold(0.0, f64::max);
    assert!(
        r.x.iter().all(|x| x.is_finite()) && f.is_finite() && r.f.is_finite()
    );
    assert!((r.f - f).abs() <= 1e-12 * f.abs().max(1.0));
    assert!(r.objective_calls > 0 && r.objective_calls <= case.budget());
    println!(
        "{}",
        serde_json::json!({"mode":mode,"case":args[2],"ns":ns,"result":r,
        "violation":cv,"verified_cost":f,"quality_pass":case.quality(f,cv),
        "verification_calls":{"objective":1,"constraints":1}})
    );
}
