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
}

impl<'a, P, F: Scalar> LogBarrier<'a, P, F> {
    /// Wrap `problem` with barrier parameter `mu` (`μ > 0`). Smaller `μ`
    /// hews closer to the true constrained objective but makes `φ_μ`
    /// stiffer near the feasible boundary.
    pub fn new(problem: &'a P, mu: F) -> Self {
        Self { problem, mu }
    }

    /// The barrier parameter `μ` this adapter was built with.
    pub fn mu(&self) -> F {
        self.mu
    }
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
        // slack s = b − A x.
        let mut s = self.problem.a().matvec(x);
        s.neg_in_place();
        s.scaled_add(F::one(), self.problem.b());

        let mut log_sum = F::zero();
        for i in 0..s.vec_len() {
            let si = s.get_scalar(i);
            if si <= F::zero() {
                // Infeasible: barrier is +∞, so the whole objective is +∞.
                return Ok(F::infinity());
            }
            log_sum = log_sum + si.ln();
        }
        Ok(self.problem.cost(x)? - self.mu * log_sum)
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
        // slack s = b − A x, reused in place as the barrier weights
        // wᵢ = μ / sᵢ. ∇[−μ log(bᵢ − aᵢᵀx)] = μ aᵢ / sᵢ, summed = Aᵀ w.
        let mut s = self.problem.a().matvec(x);
        s.neg_in_place();
        s.scaled_add(F::one(), self.problem.b());
        for i in 0..s.vec_len() {
            let si = s.get_scalar(i);
            s.set_scalar(i, self.mu / si);
        }

        let mut g = self.problem.gradient(x)?;
        let barrier_grad = self.problem.a().mat_transpose_vec(&s);
        g.scaled_add(F::one(), &barrier_grad);
        Ok(g)
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
}
