//! Layer-by-layer reproduction of the COBYLA migration cases in TODO.md.
//! Build with reproduce.py; the historical adapters are generated dependencies.

use basin::{
    Cobyla, CobylaState, CostFunction, Executor, MaxCostEvals,
    NonlinearInequalityConstraints, Problem, Solver, State,
};
use ndarray::{Array1, Array2};
use std::{cell::Cell, convert::Infallible, hint::black_box, time::Instant};

#[cfg(feature = "allocations")]
mod allocation {
    use std::{
        alloc::{GlobalAlloc, Layout, System},
        sync::atomic::{AtomicU64, Ordering},
    };
    pub static COUNT: AtomicU64 = AtomicU64::new(0);
    pub static BYTES: AtomicU64 = AtomicU64::new(0);
    pub struct Counting;
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            COUNT.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            unsafe { System.alloc(layout) }
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            COUNT.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            unsafe { System.alloc_zeroed(layout) }
        }
        unsafe fn realloc(
            &self,
            ptr: *mut u8,
            layout: Layout,
            size: usize,
        ) -> *mut u8 {
            COUNT.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(size as u64, Ordering::Relaxed);
            unsafe { System.realloc(ptr, layout, size) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
    }
}
#[cfg(feature = "allocations")]
#[global_allocator]
static ALLOCATOR: allocation::Counting = allocation::Counting;

#[derive(Clone, Copy)]
enum Case {
    Camel,
    Sphere,
    Quadratic,
}
impl Case {
    fn start(self) -> Vec<f64> {
        match self {
            Self::Camel => vec![0.0, 0.0],
            Self::Sphere => {
                vec![2.5, -2.0, 1.5, -1.0, 0.5, 2.25, -1.75, 1.25, -0.75, 0.25]
            }
            Self::Quadratic => vec![0.5, 0.5],
        }
    }
    fn bounds(self) -> Vec<(f64, f64)> {
        match self {
            Self::Camel => vec![(-3.0, 3.0), (-2.0, 2.0)],
            Self::Sphere => vec![(-5.0, 5.0); 10],
            Self::Quadratic => vec![(0.0, 2.0); 2],
        }
    }
    fn budget(self) -> u64 {
        match self {
            Self::Camel => 50,
            Self::Sphere => 200,
            Self::Quadratic => 100,
        }
    }
    fn objective(self, x: &[f64]) -> f64 {
        match self {
            Self::Camel => {
                (4.0 - 2.1 * x[0].powi(2) + x[0].powi(4) / 3.0) * x[0].powi(2)
                    + x[0] * x[1]
                    + (-4.0 + 4.0 * x[1].powi(2)) * x[1].powi(2)
            }
            Self::Sphere => x.iter().map(|v| v * v).sum(),
            Self::Quadratic => (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2),
        }
    }
    fn nonlinear(self, x: &[f64]) -> Vec<f64> {
        if matches!(self, Self::Quadratic) {
            vec![-(1.5 - x[0] - x[1])]
        } else {
            vec![]
        }
    }
}
struct Outcome {
    x: Vec<f64>,
    f: f64,
    nf: u64,
    nc: u64,
    iterations: u64,
}
struct BasinProblem<'a> {
    case: Case,
    bounds: Vec<(f64, f64)>,
    nf: &'a Cell<u64>,
    nc: &'a Cell<u64>,
    projected: bool,
}
impl BasinProblem<'_> {
    fn project(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(&self.bounds)
            .map(|(&x, &(l, u))| x.clamp(l, u))
            .collect()
    }
}
impl CostFunction for BasinProblem<'_> {
    type Param = Vec<f64>;
    type Output = f64;
    type Error = Infallible;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
        if self.nf.get() >= self.case.budget() {
            return Ok(f64::INFINITY);
        }
        self.nf.set(self.nf.get() + 1);
        Ok(if self.projected {
            self.case.objective(&self.project(x))
        } else {
            self.case.objective(x)
        })
    }
}
impl NonlinearInequalityConstraints for BasinProblem<'_> {
    fn num_constraints(&self) -> usize {
        self.bounds.len() * 2
            + usize::from(matches!(self.case, Case::Quadratic))
    }
    fn constraints(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
        self.nc.set(self.nc.get() + 1);
        let mut c = Vec::with_capacity(self.num_constraints());
        c.extend(if self.projected {
            self.case.nonlinear(&self.project(x))
        } else {
            self.case.nonlinear(x)
        });
        for (&v, &(l, u)) in x.iter().zip(&self.bounds) {
            c.extend([l - v, v - u]);
        }
        Ok(c)
    }
}
fn basin_solve(case: Case, projected: bool, manual: bool) -> Outcome {
    let nf = Cell::new(0);
    let nc = Cell::new(0);
    let p = BasinProblem {
        case,
        bounds: case.bounds(),
        nf: &nf,
        nc: &nc,
        projected,
    };
    let mut solver = Cobyla::new()
        .with_initial_radius(0.5)
        .with_final_radius(f64::EPSILON.sqrt() * 0.5);
    if manual {
        let mut p = Problem::new(p);
        let mut state =
            solver.init(&mut p, CobylaState::new(case.start())).unwrap();
        let mut iterations = 0;
        while p.counts().cost_evals < case.budget() && iterations < 10000 {
            let (next, reason) = solver.next_iter(&mut p, state).unwrap();
            state = next;
            iterations += 1;
            if reason.is_some() {
                break;
            }
        }
        Outcome {
            x: state.param().clone(),
            f: state.cost(),
            nf: nf.get(),
            nc: nc.get(),
            iterations,
        }
    } else {
        let result = Executor::from_start(p, solver, case.start())
            .max_iter(10000)
            .max_cost_evals(case.budget())
            .run()
            .unwrap();
        Outcome {
            x: result.best_param().clone(),
            f: result.best_cost(),
            nf: nf.get(),
            nc: nc.get(),
            iterations: result.state.iter(),
        }
    }
}
fn raw_solve(case: Case) -> Outcome {
    let nf = Cell::new(0);
    let nc = Cell::new(0);
    let p = BasinProblem {
        case,
        bounds: case.bounds(),
        nf: &nf,
        nc: &nc,
        projected: false,
    };
    let (x, f, iterations) = basin_gap_investigation::raw_solve(
        |x| {
            let x = x.to_vec();
            (p.cost(&x).unwrap(), p.constraints(&x).unwrap())
        },
        case.start(),
        p.num_constraints(),
        case.budget() as usize,
    );
    Outcome {
        x,
        f,
        nf: nf.get(),
        nc: nc.get(),
        iterations: iterations as u64,
    }
}
fn old_solve(case: Case) -> Outcome {
    let nf = Cell::new(0);
    let nc = Cell::new(0);
    let c = |x: &[f64], _: &mut ()| {
        nc.set(nc.get() + 1);
        1.5 - x[0] - x[1]
    };
    let constraints: Vec<&dyn cobyla::Func<()>> =
        if matches!(case, Case::Quadratic) {
            vec![&c]
        } else {
            vec![]
        };
    let (_, x, f) = cobyla::minimize(
        |x: &[f64], _: &mut ()| {
            nf.set(nf.get() + 1);
            case.objective(x)
        },
        &case.start(),
        &case.bounds(),
        &constraints,
        (),
        case.budget() as usize,
        cobyla::RhoBeg::All(0.5),
        Some(cobyla::StopTols {
            ftol_rel: 0.0,
            ftol_abs: 0.0,
            xtol_rel: 0.0,
            xtol_abs: vec![],
        }),
    )
    .unwrap();
    Outcome {
        x,
        f,
        nf: nf.get(),
        nc: nc.get(),
        iterations: 0,
    }
}
macro_rules! adapter {
    ($module:ident, $dependency:ident) => {
        mod $module {
            use super::*;
            use $dependency::{
                local_solver::{
                    builders::LocalSolverConfig, runner::LocalSolver,
                },
                problem::Problem,
                types::{EvaluationError, LocalSolverType},
            };
            struct TrackedCase<'a> {
                case: Case,
                nc: &'a Cell<u64>,
            }
            impl Problem for TrackedCase<'_> {
                fn objective(
                    &self,
                    x: &Array1<f64>,
                ) -> Result<f64, EvaluationError> {
                    Ok(self.case.objective(x.as_slice().unwrap()))
                }
                fn variable_bounds(&self) -> Array2<f64> {
                    let b = self.case.bounds();
                    Array2::from_shape_fn((b.len(), 2), |(i, j)| {
                        if j == 0 { b[i].0 } else { b[i].1 }
                    })
                }
                fn constraints(
                    &self,
                    x: &Array1<f64>,
                ) -> Result<Array1<f64>, EvaluationError> {
                    self.nc.set(self.nc.get() + 1);
                    Ok(Array1::from_vec(
                        self.case
                            .nonlinear(x.as_slice().unwrap())
                            .into_iter()
                            .map(|c| -c)
                            .collect(),
                    ))
                }
            }
            pub fn solve(case: Case) -> Outcome {
                let config = LocalSolverConfig::COBYLA {
                    max_iter: case.budget(),
                    initial_step_size: 0.5,
                    ftol_rel: 0.0,
                    ftol_abs: 0.0,
                    xtol_rel: 0.0,
                    xtol_abs: vec![],
                };
                let nc = Cell::new(0);
                let solver = LocalSolver::new(
                    TrackedCase { case, nc: &nc },
                    LocalSolverType::COBYLA,
                    config,
                );
                let (result, nf) = solver
                    .solve_with_tracking(Array1::from_vec(case.start()), true)
                    .unwrap();
                Outcome {
                    x: result.point.to_vec(),
                    f: result.objective,
                    nf,
                    nc: nc.get(),
                    iterations: 0,
                }
            }
        }
    };
}
adapter!(old_adapter, globalsearch_old);
adapter!(new_adapter, globalsearch_new);
adapter!(current_adapter, globalsearch_current);
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("basin");
    let case = match args.get(2).map(String::as_str).unwrap_or("camel") {
        "camel" => Case::Camel,
        "sphere" => Case::Sphere,
        "quadratic" => Case::Quadratic,
        _ => panic!("unknown case"),
    };
    let repeats: usize =
        args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1000);
    assert!(repeats > 0, "at least one solve is required");
    let run = || match mode {
        "raw" => raw_solve(case),
        "old" => old_solve(case),
        "basin" => basin_solve(case, false, false),
        "manual" => basin_solve(case, false, true),
        "projected" => basin_solve(case, true, false),
        "old-adapter" => old_adapter::solve(case),
        "new-adapter" => new_adapter::solve(case),
        "current-adapter" => current_adapter::solve(case),
        _ => panic!("unknown mode"),
    };
    for _ in 0..16 {
        black_box(run());
    }
    #[cfg(feature = "allocations")]
    let before = (
        allocation::COUNT.load(std::sync::atomic::Ordering::Relaxed),
        allocation::BYTES.load(std::sync::atomic::Ordering::Relaxed),
    );
    let now = Instant::now();
    for _ in 0..repeats {
        black_box(run());
    }
    let ns = now.elapsed().as_nanos() as f64 / repeats as f64;
    #[cfg(feature = "allocations")]
    println!(
        "allocations={},bytes={}",
        (allocation::COUNT.load(std::sync::atomic::Ordering::Relaxed)
            - before.0)
            / repeats as u64,
        (allocation::BYTES.load(std::sync::atomic::Ordering::Relaxed)
            - before.1)
            / repeats as u64
    );
    let r = run();
    assert!(r.nf <= case.budget(), "objective budget exceeded");
    assert!(r.f.is_finite() && r.x.iter().all(|x| x.is_finite()));
    let (optimum, tolerance) = match case {
        Case::Camel => (-1.031_628_453_489_877, 1e-5),
        Case::Sphere => (0.0, 1e-5),
        Case::Quadratic => (0.125, 1e-6),
    };
    assert!(
        (r.f - optimum).abs() < tolerance,
        "objective quality regressed: {}",
        r.f
    );
    let violation = case
        .nonlinear(&r.x)
        .into_iter()
        .chain(
            r.x.iter()
                .zip(case.bounds())
                .flat_map(|(&v, (l, u))| [l - v, v - u]),
        )
        .fold(0.0_f64, f64::max);
    assert!(
        violation <= f64::EPSILON.sqrt(),
        "infeasible return: {violation}"
    );
    println!(
        "{mode},{},{ns:.1},{},{},{},{:.14e},{violation:.5e},{:?}",
        args.get(2).map(String::as_str).unwrap_or("camel"),
        r.nf,
        r.nc,
        r.iterations,
        r.f,
        r.x
    );
}
