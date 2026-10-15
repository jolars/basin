//! Log-barrier adapter for linear inequality constraints.
//!
//! [`LogBarrier`] wraps a [`Constraints`](crate::Constraints) problem
//! `min f(x) s.t. A x ≤ b` together with a fixed barrier parameter `μ > 0`
//! and exposes the unconstrained barrier objective
//!
//! ```text
//! φ_μ(x) = f(x) − μ · Σᵢ log(bᵢ − aᵢᵀ x)
//! ```
//!
//! as a plain [`CostFunction`] + [`Gradient`]. Minimizing `φ_μ` for a
//! decreasing sequence of `μ` traces the central path to the constrained
//! optimum: the [`BarrierMethod`](crate::solver::BarrierMethod) automates
//! that continuation, but `LogBarrier` is also usable on its own with any
//! unconstrained solver or [`Executor`](crate::core::executor::Executor),
//! mirroring R's `constrOptim` layering a barrier on `optim`.
//!
//! # Constraint transformation
//!
//! `LogBarrier` consumes a set containing only linear inequality constraints and
//! exposes [`NoConstraints`](crate::NoConstraints) to the inner solver. Other
//! blocks are rejected at compile time. Derivative adapters such as
//! [`FiniteDiff`](crate::FiniteDiff) instead preserve the full declaration.
//!
//! # Feasibility
//!
//! [`cost`](CostFunction::cost) returns `+∞` at any infeasible point (some
//! `bᵢ − aᵢᵀ x ≤ 0`), so a feasibility-respecting line search (backtracking,
//! Wolfe, Moré–Thuente) rejects steps that leave the feasible set. Given a
//! strictly feasible start the iterate path therefore stays interior. The
//! [`gradient`](Gradient::gradient) is only meaningful at feasible points;
//! it still returns a finite-shaped value at infeasible ones (no panic),
//! but callers should not rely on it there.
//! [`BarrierMethod`](crate::solver::BarrierMethod) supplies a Phase I solve
//! automatically when its starting point is not strictly feasible; standalone
//! `LogBarrier` users must still provide an interior start themselves.
//!
//! # Backends
//!
//! Requires checked matrix shape and finiteness, plus
//! [`MatVecInto`](crate::MatVecInto) (`A x`) and
//! [`MatTransposeVecInto`](crate::MatTransposeVecInto) (`Aᵀ v`): a
//! strict subset of the LA tier that never includes a linear solve. That
//! covers all four backends: `DenseMatrix`/`Vec`, nalgebra
//! `DMatrix`/`DVector`, faer `Mat`/`Col`, and ndarray `Array2`/`Array1`
//! (tenet 5).

use crate::ConstraintSolveError;
use crate::core::constraint::evaluation::{
    LinearInequalityGradient, LinearInequalityProblem, invalid,
};
use crate::core::math::{
    NegInPlace, Scalar, ScaledAdd, VectorIndex, VectorLen,
};
use crate::core::problem::{CostFunction, Gradient};

/// A [`Constraints`](crate::Constraints) problem rewritten as the unconstrained
/// log-barrier objective `f(x) − μ · Σ log(bᵢ − aᵢᵀ x)` at a fixed `μ`.
///
/// Borrows the underlying problem (`&'a P`) so the barrier parameter can be
/// swapped cheaply between solves: the
/// [`BarrierMethod`](crate::solver::BarrierMethod) builds a fresh
/// `LogBarrier` per outer iteration as it shrinks `μ`. See the
/// [module docs](self) for the formulation, the tenet-4 adapter asymmetry,
/// and the feasibility/backend notes.
pub struct LogBarrier<'a, P, F = f64> {
    problem: &'a P,
    mu: F,
    objective: BarrierObjective,
}

#[derive(Clone, Copy)]
enum BarrierObjective {
    PhaseOne,
    PhaseTwo,
}

impl<'a, P, F: Scalar> LogBarrier<'a, P, F> {
    /// Wrap `problem` with barrier parameter `mu` (`μ > 0`). Smaller `μ`
    /// hews closer to the true constrained objective but makes `φ_μ`
    /// stiffer near the feasible boundary.
    pub fn new(problem: &'a P, mu: F) -> Self {
        Self {
            problem,
            mu,
            objective: BarrierObjective::PhaseTwo,
        }
    }

    /// Build the reduced Phase I objective. Kept crate-private because Phase I
    /// is an implementation detail of `BarrierMethod`; the public
    /// `LogBarrier::new` contract remains the Phase II objective.
    pub(crate) fn phase_one(problem: &'a P, mu: F) -> Self {
        Self {
            problem,
            mu,
            objective: BarrierObjective::PhaseOne,
        }
    }

    /// The barrier parameter `μ` this adapter was built with.
    pub fn mu(&self) -> F {
        self.mu
    }

    /// Validate the constraint data at `x` and report strict feasibility.
    pub(crate) fn strict_feasibility<V>(&self, x: &V) -> Option<bool>
    where
        P: LinearInequalityProblem<V, F>,
        V: ScaledAdd<F> + VectorIndex<F> + VectorLen,
    {
        strict_feasibility(self.problem, x)
    }
}

/// Return `Some(true)` for `A x < b`, `Some(false)` for finite but non-strict
/// data, and `None` for a non-finite or inconsistent vector shape.
pub(crate) fn strict_feasibility<P, V, F>(problem: &P, x: &V) -> Option<bool>
where
    F: Scalar,
    P: LinearInequalityProblem<V, F>,
    V: ScaledAdd<F> + VectorIndex<F> + VectorLen,
{
    if !vector_is_finite(x) {
        return None;
    }
    let residual = problem.linear_residual(x).ok()?;
    if !vector_is_finite(&residual) {
        return None;
    }
    Some(residual.iter().all(|&v| v < F::zero()))
}

fn vector_is_finite<V, F>(x: &V) -> bool
where
    F: Scalar,
    V: VectorIndex<F> + VectorLen,
{
    (0..x.vec_len()).all(|i| x.get_scalar(i).is_finite())
}

/// Replace violations `r = A x - b` by Phase I slacks `s - r` and return
/// the minimizing auxiliary scalar `s` for the current `x` and `μ`.
///
/// The scalar is eliminated by solving
/// `μ Σᵢ 1/(s-rᵢ) = 1`. If `r_max = max rᵢ`, then
/// `δ = s-r_max` lies in `(0, mμ]`, which gives a finite bracket requiring
/// only scalar arithmetic. Slacks are formed as
/// `δ + (r_max-rᵢ)` so they remain positive even when `s` and `r_max`
/// round to the same floating-point number.
fn phase_one_slacks<V, F>(violations: &mut V, mu: F) -> Option<F>
where
    F: Scalar,
    V: VectorIndex<F> + VectorLen,
{
    let m = violations.vec_len();
    if m == 0 || !(mu.is_finite() && mu > F::zero()) {
        return None;
    }

    let mut r_max = violations.get_scalar(0);
    if !r_max.is_finite() {
        return None;
    }
    for i in 1..m {
        let ri = violations.get_scalar(i);
        if !ri.is_finite() {
            return None;
        }
        r_max = r_max.max(ri);
    }

    let upper = F::from_usize(m)? * mu;
    if !(upper.is_finite() && upper > F::zero()) {
        return None;
    }
    let two = F::one() + F::one();
    let mut lo = F::zero();
    let mut hi = upper;

    // 128 bisections exceed the mantissa width of both supported scalar
    // widths. The equality guards terminate once the bracket can no longer be
    // represented more finely.
    for _ in 0..128 {
        let mid = lo + (hi - lo) / two;
        if mid == lo || mid == hi {
            break;
        }
        let mut reciprocal_sum = F::zero();
        for i in 0..m {
            let slack = mid + (r_max - violations.get_scalar(i));
            reciprocal_sum = reciprocal_sum + F::one() / slack;
        }
        if mu * reciprocal_sum > F::one() {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let s = r_max + hi;
    if !s.is_finite() {
        return None;
    }
    for i in 0..m {
        let slack = hi + (r_max - violations.get_scalar(i));
        if !(slack.is_finite() && slack > F::zero()) {
            return None;
        }
        violations.set_scalar(i, slack);
    }
    Some(s)
}

impl<P, V, F> CostFunction for LogBarrier<'_, P, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + LinearInequalityProblem<V, F>,
    V: VectorLen + VectorIndex<F>,
{
    type Param = V;
    type Output = F;
    type Error = ConstraintSolveError<P::Error>;
    fn cost(&self, x: &V) -> Result<F, Self::Error> {
        let mut values = self.problem.linear_residual(x).map_err(invalid)?;
        if !vector_is_finite(x) || !vector_is_finite(&values) {
            return Ok(F::infinity());
        }
        let phase_one_s = match self.objective {
            BarrierObjective::PhaseOne => {
                match phase_one_slacks(&mut values, self.mu) {
                    Some(s) => Some(s),
                    None => return Ok(F::infinity()),
                }
            }
            BarrierObjective::PhaseTwo => {
                values.neg_in_place();
                None
            }
        };
        let mut log_sum = F::zero();
        for &slack in &values {
            if !(slack.is_finite() && slack > F::zero()) {
                return Ok(F::infinity());
            }
            log_sum = log_sum + slack.ln();
        }
        match phase_one_s {
            Some(s) => Ok(s - self.mu * log_sum),
            None => Ok(self.problem.cost(x)? - self.mu * log_sum),
        }
    }
}
impl<P, V, F> Gradient for LogBarrier<'_, P, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>
        + LinearInequalityGradient<V, F>
        + Gradient<Gradient = V>,
    V: Clone + VectorLen + VectorIndex<F> + ScaledAdd<F>,
{
    type Gradient = V;
    fn gradient(&self, x: &V) -> Result<V, <Self as CostFunction>::Error> {
        let mut values = self.problem.linear_residual(x).map_err(invalid)?;
        let valid = vector_is_finite(x)
            && vector_is_finite(&values)
            && match self.objective {
                BarrierObjective::PhaseOne => {
                    phase_one_slacks(&mut values, self.mu).is_some()
                }
                BarrierObjective::PhaseTwo => {
                    values.neg_in_place();
                    values.iter().all(|v| v.is_finite() && *v > F::zero())
                }
            };
        for value in &mut values {
            *value = if valid { self.mu / *value } else { F::zero() };
        }
        let mut barrier_grad = x.clone();
        self.problem
            .linear_transpose(&values, &mut barrier_grad)
            .map_err(invalid)?;
        match self.objective {
            BarrierObjective::PhaseOne => Ok(barrier_grad),
            BarrierObjective::PhaseTwo => {
                let mut g = self.problem.gradient(x)?;
                g.scaled_add(F::one(), &barrier_grad);
                Ok(g)
            }
        }
    }
}

#[cfg(all(test, feature = "nalgebra_all"))]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    /// `min ½‖x‖²` subject to a single row `x₀ + x₁ ≤ 2`.
    struct Probe {
        a: DMatrix<f64>,
        b: DVector<f64>,
    }

    impl Probe {
        fn new() -> Self {
            Self {
                a: DMatrix::from_row_slice(1, 2, &[1.0, 1.0]),
                b: DVector::from_vec(vec![2.0]),
            }
        }
    }

    impl crate::ProblemConstraints for Probe {
        type Constraints = crate::Constraints<
            crate::NoBounds,
            crate::LinearBlocks<
                crate::NoLinearBlock,
                crate::LinearBlock<DMatrix<f64>, DVector<f64>>,
            >,
        >;
        fn constraint_set(
            &self,
        ) -> <Self::Constraints as crate::ConstraintDescription>::Borrowed<'_>
        {
            crate::Constraints::new().with_linear_inequalities(
                crate::LinearBlock::new(&self.a, &self.b),
            )
        }
    }
    impl CostFunction for Probe {
        type Param = DVector<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(
            &self,
            x: &DVector<f64>,
        ) -> Result<f64, std::convert::Infallible> {
            Ok(0.5 * x.dot(x))
        }
    }

    impl Gradient for Probe {
        type Gradient = DVector<f64>;
        fn gradient(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, std::convert::Infallible> {
            Ok(x.clone())
        }
    }

    #[test]
    fn cost_matches_closed_form_at_feasible_point() {
        let p = Probe::new();
        let mu = 0.5;
        let lb = LogBarrier::new(&p, mu);
        let x = DVector::from_vec(vec![0.0, 0.0]);
        // f = 0; slack = 2 − 0 = 2; φ = 0 − μ·ln(2).
        let expected = -mu * 2.0_f64.ln();
        assert!((lb.cost(&x).unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn cost_is_infinite_outside_the_feasible_set() {
        let p = Probe::new();
        let lb = LogBarrier::new(&p, 1.0);
        // x₀ + x₁ = 3 > 2 ⇒ slack negative ⇒ +∞.
        let x = DVector::from_vec(vec![2.0, 1.0]);
        assert!(lb.cost(&x).unwrap().is_infinite());
    }

    #[test]
    fn gradient_agrees_with_finite_differences() {
        let p = Probe::new();
        let lb = LogBarrier::new(&p, 0.7);
        let x = DVector::from_vec(vec![0.3, -0.4]);
        let analytic = lb.gradient(&x).unwrap();

        let h = 1e-6;
        for j in 0..2 {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += h;
            xm[j] -= h;
            let fd =
                (lb.cost(&xp).unwrap() - lb.cost(&xm).unwrap()) / (2.0 * h);
            assert!(
                (analytic[j] - fd).abs() < 1e-5,
                "component {j}: analytic {} vs fd {}",
                analytic[j],
                fd
            );
        }
    }

    #[test]
    fn phase_one_auxiliary_scalar_satisfies_stationarity() {
        let mu = 0.7;
        let mut violations = DVector::from_vec(vec![2.0, -1.0, 0.5]);
        let original = violations.clone();
        let s = phase_one_slacks(&mut violations, mu).unwrap();

        let stationarity: f64 = violations.iter().map(|slack| mu / slack).sum();
        assert!((stationarity - 1.0).abs() < 1e-14);
        for i in 0..violations.len() {
            assert!((violations[i] - (s - original[i])).abs() < 1e-14);
            assert!(violations[i] > 0.0);
        }
    }

    #[test]
    fn phase_one_reduced_gradient_agrees_with_finite_differences() {
        let p = Probe::new();
        let phase_one = LogBarrier::phase_one(&p, 0.7);
        // Deliberately infeasible for x₀ + x₁ ≤ 2: the reduced
        // auxiliary objective is nevertheless finite everywhere.
        let x = DVector::from_vec(vec![2.0, 1.0]);
        assert!(phase_one.cost(&x).unwrap().is_finite());
        let analytic = phase_one.gradient(&x).unwrap();

        let h = 1e-6;
        for j in 0..2 {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += h;
            xm[j] -= h;
            let fd = (phase_one.cost(&xp).unwrap()
                - phase_one.cost(&xm).unwrap())
                / (2.0 * h);
            assert!(
                (analytic[j] - fd).abs() < 1e-6,
                "component {j}: analytic {} vs fd {}",
                analytic[j],
                fd
            );
        }
    }
}

impl<P, F> crate::ProblemConstraints for LogBarrier<'_, P, F> {
    type Constraints = crate::NoConstraints;
    fn constraint_set(&self) -> crate::NoConstraints {
        crate::NoConstraints::EMPTY
    }
}
