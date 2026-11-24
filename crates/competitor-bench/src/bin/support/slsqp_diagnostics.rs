//! Compare constrained SLSQP with NLopt's public result and callbacks.

use std::cell::Cell;

use basin::{TerminationCode, ViolationMeasure};
use competitor_bench::slsqp::{ACCURACY, BUDGET, workloads::Case};
use nalgebra::{DMatrix, DVector};
use nlopt::{Algorithm, Nlopt, SuccessState, Target};

const FEASIBILITY: f64 = 1e-8;
const ACTIVE_TOLERANCE: f64 = 1e-6;

struct Evidence {
    feasibility: f64,
    gradient: Vec<f64>,
    equality: Option<f64>,
    inequality: f64,
    equality_gradient: Vec<f64>,
    inequality_gradient: Vec<f64>,
}

impl Evidence {
    fn at(case: Case, x: &[f64], lower: &[f64], upper: &[f64]) -> Self {
        let mut gradient = vec![0.0; x.len()];
        case.gradient(x, &mut gradient);
        let mut equality_gradient = vec![0.0; x.len()];
        let equality = case
            .has_equality()
            .then(|| Case::equality(x, Some(&mut equality_gradient)));
        let mut inequality_gradient = vec![0.0; x.len()];
        let inequality = case.inequality(x, Some(&mut inequality_gradient));
        let box_violation = x
            .iter()
            .zip(lower.iter().zip(upper))
            .map(|(&x, (&lo, &hi))| (lo - x).max(0.0) + (x - hi).max(0.0))
            .sum::<f64>();
        let feasibility =
            equality.unwrap_or(0.0).abs() + inequality.max(0.0) + box_violation;
        Self {
            feasibility,
            gradient,
            equality,
            inequality,
            equality_gradient,
            inequality_gradient,
        }
    }

    // Fit nonlinear multipliers independently of the solver using components
    // away from box faces. Stationarity and dual signs are then tested on the
    // whole point, including the box faces. Both fixtures have an active
    // nonlinear inequality and a full-rank fit on the interior coordinates.
    fn estimate_multipliers(
        &self,
        x: &[f64],
        lower: &[f64],
        upper: &[f64],
    ) -> (f64, f64) {
        assert!(self.inequality.abs() <= ACTIVE_TOLERANCE);
        let interior: Vec<_> = (0..x.len())
            .filter(|&i| {
                x[i] - lower[i] > ACTIVE_TOLERANCE
                    && upper[i] - x[i] > ACTIVE_TOLERANCE
            })
            .collect();
        let has_eq = self.equality.is_some();
        let columns = 1 + usize::from(has_eq);
        let a = DMatrix::from_fn(interior.len(), columns, |row, col| {
            let i = interior[row];
            if has_eq && col == 0 {
                self.equality_gradient[i]
            } else {
                self.inequality_gradient[i]
            }
        });
        let rhs = DVector::from_iterator(
            interior.len(),
            interior.iter().map(|&i| -self.gradient[i]),
        );
        let svd = a.svd(true, true);
        assert_eq!(svd.rank(1e-12), columns);
        let multipliers = svd.solve(&rhs, 1e-12).unwrap();
        (
            if has_eq { multipliers[0] } else { 0.0 },
            multipliers[columns - 1],
        )
    }

    fn kkt(
        &self,
        x: &[f64],
        lower: &[f64],
        upper: &[f64],
        lambda: f64,
        mu: f64,
    ) -> (f64, f64, f64) {
        let mut projected: f64 = 0.0;
        let mut box_complementarity: f64 = 0.0;
        for i in 0..x.len() {
            let lag = self.gradient[i]
                + lambda * self.equality_gradient[i]
                + mu * self.inequality_gradient[i];
            let to_lower = x[i] - lower[i];
            let to_upper = upper[i] - x[i];
            projected = projected.max(lag.clamp(-to_upper, to_lower).abs());
            if to_lower <= ACTIVE_TOLERANCE {
                box_complementarity =
                    box_complementarity.max(to_lower.abs() * lag.max(0.0));
            }
            if to_upper <= ACTIVE_TOLERANCE {
                box_complementarity =
                    box_complementarity.max(to_upper.abs() * (-lag).max(0.0));
            }
        }
        (projected, (self.inequality * mu).abs(), box_complementarity)
    }
}

fn verify_point(
    case: Case,
    x: &[f64],
    reported_cost: f64,
    lower: &[f64],
    upper: &[f64],
) -> bool {
    assert!(x.iter().all(|v| v.is_finite()));
    let f = case.objective(x);
    let expected = match case {
        Case::Hs71 => 17.014_017_289_156,
        Case::Quadratic => {
            assert!(x.iter().all(|x| (x - 0.75).abs() < 1e-5));
            x.len() as f64 / 16.0
        }
    };
    assert!((f - reported_cost).abs() < 1e-12);
    assert!((f - expected).abs() < 5e-8, "{}: {f}", case.name());

    let evidence = Evidence::at(case, x, lower, upper);
    let (lambda, mu) = evidence.estimate_multipliers(x, lower, upper);
    let (stationarity, complementarity, box_complementarity) =
        evidence.kkt(x, lower, upper, lambda, mu);
    assert!(evidence.feasibility <= FEASIBILITY);
    assert!(mu >= 0.0, "negative nonlinear inequality multiplier");
    let meets_kkt = stationarity <= 1e-5
        && complementarity <= 1e-8
        && box_complementarity <= 1e-8;
    println!(
        "  rebuilt: f={f:.12e} violation={:.3e} stationarity={stationarity:.3e} \
         nonlinear_complementarity={complementarity:.3e} box_complementarity={box_complementarity:.3e} \
         eq_multiplier={lambda:.9e} ineq_multiplier={mu:.9e} meets_kkt={meets_kkt}",
        evidence.feasibility,
    );
    meets_kkt
}

// NLopt fuses the value and optional derivative for each scalar callback.
// Count their work separately without adding them as if they were callbacks.
fn bump(counts: &Cell<(u64, u64)>, derivative: bool) {
    let (values, derivatives) = counts.get();
    counts.set((values + 1, derivatives + u64::from(derivative)));
}

pub fn run() {
    println!("# Constrained SLSQP");
    for case in [Case::Hs71, Case::Quadratic] {
        let start = case.start();
        let (lower, upper) = case.bounds(start.len());
        let run = case.solve(start.clone(), lower.clone(), upper.clone());
        let (x, f, g, violation) = run.state.current().unwrap();
        let (selected, _, selected_violation) = run.state.best().unwrap();
        assert_eq!(x, selected);
        assert_eq!(violation, selected_violation);
        assert_eq!(violation.measure(), &ViolationMeasure::L1);
        assert_eq!(run.report.code(), TerminationCode::SolverConverged);
        assert_eq!(run.solver.failure(), None);
        let evidence = Evidence::at(case, x, &lower, &upper);
        assert_eq!(*g, evidence.gradient);
        assert!((violation.value - evidence.feasibility).abs() < 1e-12);
        assert_eq!(run.solver.constraint_violation(), Some(violation.value));
        let eq = run.solver.equality_multipliers().unwrap();
        assert_eq!(eq.len(), usize::from(case.has_equality()));
        let &[mu] = run.solver.inequality_multipliers().unwrap() else {
            panic!("expected one inequality multiplier");
        };
        let (stationarity, complementarity, _) = evidence.kkt(
            x,
            &lower,
            &upper,
            eq.first().copied().unwrap_or(0.0),
            mu,
        );
        let native_stationarity = run.solver.stationarity().unwrap();
        let native_complementarity = run.solver.complementarity().unwrap();
        assert!((native_stationarity - stationarity).abs() < 1e-12);
        assert!((native_complementarity - complementarity).abs() < 1e-12);
        assert!(native_stationarity <= 1e-5 && native_complementarity <= 1e-8);
        assert!(mu >= 0.0);
        println!(
            "{} Basin {:?} iter={} f_calls={} grad_calls={} constraint_blocks={} constraint_jacobians={}",
            case.name(),
            run.report.code(),
            run.iter(),
            run.counts.cost_evals,
            run.counts.gradient_evals,
            run.counts.residual_evals,
            run.counts.jacobian_evals,
        );
        println!(
            "  native: violation={:.3e} stationarity={native_stationarity:.3e} \
             nonlinear_complementarity={native_complementarity:.3e} eq_multipliers={eq:?} \
             ineq_multiplier={mu:.9e} stop={:?}",
            violation.value, run.report.termination,
        );
        assert!(verify_point(case, x, f, &lower, &upper));

        let objective_calls = Cell::new((0, 0));
        let constraint_calls = Cell::new((0, 0));
        let objective = |x: &[f64], g: Option<&mut [f64]>, _: &mut ()| {
            bump(&objective_calls, g.is_some());
            if let Some(g) = g {
                case.gradient(x, g);
            }
            case.objective(x)
        };
        let mut opt = Nlopt::new(
            Algorithm::Slsqp,
            start.len(),
            objective,
            Target::Minimize,
            (),
        );
        opt.set_lower_bounds(&lower).unwrap();
        opt.set_upper_bounds(&upper).unwrap();
        if case.has_equality() {
            opt.add_equality_constraint(
                |x: &[f64], g: Option<&mut [f64]>, _: &mut ()| {
                    bump(&constraint_calls, g.is_some());
                    Case::equality(x, g)
                },
                (),
                FEASIBILITY,
            )
            .unwrap();
        }
        opt.add_inequality_constraint(
            |x: &[f64], g: Option<&mut [f64]>, _: &mut ()| {
                bump(&constraint_calls, g.is_some());
                case.inequality(x, g)
            },
            (),
            FEASIBILITY,
        )
        .unwrap();
        opt.set_maxeval(BUDGET as u32).unwrap();
        opt.set_ftol_abs(ACCURACY).unwrap();
        let mut x = start;
        let result = opt.optimize(&mut x);
        let (values, gradients) = objective_calls.get();
        let (constraints, constraint_gradients) = constraint_calls.get();
        println!(
            "{} NLopt {result:?} f_calls={values} grad_calls={gradients} \
             constraint_values={constraints} constraint_derivatives={constraint_gradients}",
            case.name(),
        );
        let (status, f) = result.unwrap();
        assert!(matches!(status, SuccessState::FtolReached));
        let meets_kkt = verify_point(case, &x, f, &lower, &upper);
        if matches!(case, Case::Hs71) {
            assert!(meets_kkt);
        }
    }
}

#[test]
fn constrained_slsqp_public_evidence() {
    run();
}
