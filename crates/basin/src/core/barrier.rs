//! Log-barrier adapter for linear inequality constraints.
//!
//! [`LogBarrier`] wraps a [`LinearInequalityConstraints`] problem
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
//! # Adapter asymmetry (tenet 4, load-bearing)
//!
//! `LogBarrier` *consumes* [`LinearInequalityConstraints`] and exposes
//! [`CostFunction`] + [`Gradient`] **only**: it deliberately does **not**
//! implement [`LinearInequalityConstraints`] itself. That asymmetry is what
//! routes the wrapped problem to *unconstrained* solvers: if the barrier
//! re-exposed the constraint trait it would route straight back into
//! constrained solvers and the adapter model would collapse. (Contrast
//! [`FiniteDiff`](crate::core::numdiff::FiniteDiff), which *adds* a
//! capability and therefore *forwards* [`BoxConstraints`](crate::core::constraint::BoxConstraints).)
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
//! Requires the constraint matrix to implement
//! [`MatVec`] (`A x`) and
//! [`MatTransposeVec`] (`Aᵀ v`): a
//! strict subset of the LA tier that never includes a linear solve. That
//! covers all four backends: `DenseMatrix`/`Vec`, nalgebra
//! `DMatrix`/`DVector`, faer `Mat`/`Col`, and ndarray `Array2`/`Array1`
//! (tenet 5).

use crate::core::constraint::LinearInequalityConstraints;
use crate::core::math::{
    MatTransposeVec, MatVec, NegInPlace, Scalar, ScaledAdd, VectorIndex,
    VectorLen,
};
use crate::core::problem::{CostFunction, Gradient};

/// A [`LinearInequalityConstraints`] problem rewritten as the unconstrained
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
    pub(crate) fn strict_feasibility<V, M>(&self, x: &V) -> Option<bool>
    where
        P: LinearInequalityConstraints<Param = V, Matrix = M>,
        M: MatVec<V>,
        V: ScaledAdd<F> + VectorIndex<F> + VectorLen,
    {
        strict_feasibility(self.problem, x)
    }
}

/// Return `Some(true)` for `A x < b`, `Some(false)` for finite but non-strict
/// data, and `None` for a non-finite or inconsistent vector shape.
pub(crate) fn strict_feasibility<P, V, M, F>(problem: &P, x: &V) -> Option<bool>
where
    F: Scalar,
    P: LinearInequalityConstraints<Param = V, Matrix = M>,
    M: MatVec<V>,
    V: ScaledAdd<F> + VectorIndex<F> + VectorLen,
{
    if !vector_is_finite(x) || !vector_is_finite(problem.b()) {
        return None;
    }

    let mut residual = problem.a().matvec(x);
    if residual.vec_len() != problem.b().vec_len() {
        return None;
    }
    residual.scaled_add(-F::one(), problem.b());
    if !vector_is_finite(&residual) {
        return None;
    }

    Some((0..residual.vec_len()).all(|i| residual.get_scalar(i) < F::zero()))
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

impl<P, V, M, F> CostFunction for LogBarrier<'_, P, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>
        + LinearInequalityConstraints<Param = V, Matrix = M>,
    M: MatVec<V>,
    V: ScaledAdd<F> + NegInPlace + VectorIndex<F> + VectorLen,
{
    type Param = V;
    type Output = F;
    // Pass through the wrapped problem's hard-abort error: barrier-internal
    // issues (slack ≤ 0) still use the soft `+∞` reject path, so the only
    // `Err` that can come out of this `cost` is one the user's `cost`
    // returned.
    type Error = <P as CostFunction>::Error;

    fn cost(&self, x: &V) -> Result<F, Self::Error> {
        if !vector_is_finite(x) || !vector_is_finite(self.problem.b()) {
            return Ok(F::infinity());
        }

        let mut values = self.problem.a().matvec(x);
        if values.vec_len() != self.problem.b().vec_len() {
            return Ok(F::infinity());
        }

        let phase_one_s = match self.objective {
            BarrierObjective::PhaseOne => {
                // violations r = A x - b, replaced by slacks s - r after
                // analytically minimizing over the auxiliary scalar s.
                values.scaled_add(-F::one(), self.problem.b());
                match phase_one_slacks(&mut values, self.mu) {
                    Some(s) => Some(s),
                    None => return Ok(F::infinity()),
                }
            }
            BarrierObjective::PhaseTwo => {
                // slack = b - A x.
                values.neg_in_place();
                values.scaled_add(F::one(), self.problem.b());
                None
            }
        };

        let mut log_sum = F::zero();
        for i in 0..values.vec_len() {
            let si = values.get_scalar(i);
            if !(si.is_finite() && si > F::zero()) {
                // Infeasible: barrier is +∞, so the whole objective is +∞.
                return Ok(F::infinity());
            }
            log_sum = log_sum + si.ln();
        }
        match phase_one_s {
            Some(s) => Ok(s - self.mu * log_sum),
            None => Ok(self.problem.cost(x)? - self.mu * log_sum),
        }
    }
}

impl<P, V, M, F> Gradient for LogBarrier<'_, P, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>
        + Gradient<Gradient = V>
        + LinearInequalityConstraints<Param = V, Matrix = M>,
    M: MatVec<V> + MatTransposeVec<V>,
    V: ScaledAdd<F> + NegInPlace + VectorIndex<F> + VectorLen,
{
    type Gradient = V;

    fn gradient(&self, x: &V) -> Result<V, <Self as CostFunction>::Error> {
        let mut values = self.problem.a().matvec(x);
        let valid_shape = values.vec_len() == self.problem.b().vec_len();
        let valid = valid_shape
            && vector_is_finite(x)
            && vector_is_finite(self.problem.b())
            && match self.objective {
                BarrierObjective::PhaseOne => {
                    values.scaled_add(-F::one(), self.problem.b());
                    phase_one_slacks(&mut values, self.mu).is_some()
                }
                BarrierObjective::PhaseTwo => {
                    values.neg_in_place();
                    values.scaled_add(F::one(), self.problem.b());
                    (0..values.vec_len()).all(|i| {
                        let slack = values.get_scalar(i);
                        slack.is_finite() && slack > F::zero()
                    })
                }
            };

        for i in 0..values.vec_len() {
            let weight = if valid {
                self.mu / values.get_scalar(i)
            } else {
                F::zero()
            };
            values.set_scalar(i, weight);
        }

        let barrier_grad = self.problem.a().mat_transpose_vec(&values);
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

#[cfg(all(test, feature = "nalgebra"))]
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

    impl LinearInequalityConstraints for Probe {
        type Matrix = DMatrix<f64>;
        fn a(&self) -> &DMatrix<f64> {
            &self.a
        }
        fn b(&self) -> &DVector<f64> {
            &self.b
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
