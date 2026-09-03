//! Augmented-Lagrangian adapter for linear equality constraints.
//!
//! [`AugmentedLagrangian`] wraps a [`LinearEqualityConstraints`] problem
//! `min f(x) s.t. A x = b` together with a multiplier estimate `λ ∈ ℝᵐ` and
//! a penalty parameter `ρ > 0`, and exposes the unconstrained augmented
//! Lagrangian
//!
//! ```text
//! L_ρ(x, λ) = f(x) + λᵀ c(x) + (ρ/2) ‖c(x)‖²,   c(x) = A x − b
//! ```
//!
//! as a plain [`CostFunction`] + [`Gradient`]. Minimizing `L_ρ` for a
//! sequence of multiplier updates `λ ← λ + ρ c(x)` (and, when feasibility
//! stalls, an increasing `ρ`) drives the iterate to the constrained
//! optimum: the [`AugmentedLagrangianMethod`](crate::solver::AugmentedLagrangianMethod)
//! automates that outer loop, but `AugmentedLagrangian` is also usable on its
//! own with any unconstrained solver or [`Executor`](crate::core::executor::Executor)
//! at a fixed `(λ, ρ)`.
//!
//! Unlike the [`LogBarrier`](crate::core::barrier::LogBarrier), `L_ρ` is
//! finite and smooth *everywhere* (there is no feasibility wall), so the
//! inner solver may start from an **infeasible** point and use any line
//! search (or none).
//!
//! # Adapter asymmetry (tenet 4, load-bearing)
//!
//! `AugmentedLagrangian` *consumes* [`LinearEqualityConstraints`] and exposes
//! [`CostFunction`] + [`Gradient`] **only**: it deliberately does **not**
//! implement [`LinearEqualityConstraints`] itself. That asymmetry is what
//! routes the wrapped problem to *unconstrained* solvers: if the adapter
//! re-exposed the constraint trait it would route straight back into
//! constrained solvers and the adapter model would collapse. (Contrast
//! [`FiniteDiff`](crate::core::numdiff::FiniteDiff), which *adds* a
//! capability and therefore *forwards* [`BoxConstraints`](crate::core::constraint::BoxConstraints).)
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

use crate::core::constraint::LinearEqualityConstraints;
use crate::core::math::{
    Dot, MatTransposeVec, MatVec, NormSquared, Scalar, ScaledAdd,
};
use crate::core::problem::{CostFunction, Gradient};

/// A [`LinearEqualityConstraints`] problem rewritten as the unconstrained
/// augmented Lagrangian `f(x) + λᵀ(A x − b) + (ρ/2)‖A x − b‖²` at a fixed
/// multiplier estimate `λ` and penalty `ρ`.
///
/// Borrows the underlying problem (`&'a P`) and the multiplier vector
/// (`&'a V`) so both can be swapped cheaply between solves: the
/// [`AugmentedLagrangianMethod`](crate::solver::AugmentedLagrangianMethod)
/// builds a fresh `AugmentedLagrangian` per outer iteration as it updates
/// `λ` and `ρ`. (The extra `V` type parameter, relative to
/// [`LogBarrier`](crate::core::barrier::LogBarrier), is because this adapter
/// carries a borrowed *vector* `λ`, not just a scalar.) See the
/// [module docs](self) for the formulation, the tenet-4 adapter asymmetry,
/// and the backend note.
pub struct AugmentedLagrangian<'a, P, V, F = f64> {
    problem: &'a P,
    lambda: &'a V,
    rho: F,
}

impl<'a, P, V, F: Scalar> AugmentedLagrangian<'a, P, V, F> {
    /// Wrap `problem` with multiplier estimate `lambda` (`λ ∈ ℝᵐ`, one entry
    /// per constraint row) and penalty parameter `rho` (`ρ > 0`). Larger `ρ`
    /// pushes the iterate harder onto the feasible affine subspace but makes
    /// `L_ρ` more ill-conditioned.
    pub fn new(problem: &'a P, lambda: &'a V, rho: F) -> Self {
        Self {
            problem,
            lambda,
            rho,
        }
    }

    /// The penalty parameter `ρ` this adapter was built with.
    pub fn rho(&self) -> F {
        self.rho
    }
}

impl<P, V, M, F> CostFunction for AugmentedLagrangian<'_, P, V, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>
        + LinearEqualityConstraints<Param = V, Matrix = M>,
    M: MatVec<V>,
    V: ScaledAdd<F> + Dot<F> + NormSquared<F>,
{
    type Param = V;
    type Output = F;
    // Pass through the wrapped problem's hard-abort error. `L_ρ` is finite
    // everywhere (no feasibility wall, no soft-reject path), so the only
    // `Err` that can come out of this `cost` is one the user's `cost`
    // returned.
    type Error = <P as CostFunction>::Error;

    fn cost(&self, x: &V) -> Result<F, Self::Error> {
        // Constraint residual c = A x − b.
        let mut c = self.problem.a().matvec(x);
        c.scaled_add(-F::one(), self.problem.b());
        // L_ρ = f(x) + λᵀc + (ρ/2)‖c‖².
        Ok(self.problem.cost(x)?
            + self.lambda.dot(&c)
            + F::from_f64(0.5).unwrap() * self.rho * c.norm_squared())
    }
}

impl<P, V, M, F> Gradient for AugmentedLagrangian<'_, P, V, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>
        + Gradient<Gradient = V>
        + LinearEqualityConstraints<Param = V, Matrix = M>,
    M: MatVec<V> + MatTransposeVec<V>,
    V: ScaledAdd<F> + Dot<F> + NormSquared<F> + Clone,
{
    type Gradient = V;

    fn gradient(&self, x: &V) -> Result<V, <Self as CostFunction>::Error> {
        // Constraint residual c = A x − b.
        let mut c = self.problem.a().matvec(x);
        c.scaled_add(-F::one(), self.problem.b());
        // Weight w = λ + ρ c, so the constraint contribution is Aᵀ w.
        let mut w = self.lambda.clone();
        w.scaled_add(self.rho, &c);
        // ∇L_ρ = ∇f + Aᵀ(λ + ρ c).
        let mut g = self.problem.gradient(x)?;
        let constraint_grad = self.problem.a().mat_transpose_vec(&w);
        g.scaled_add(F::one(), &constraint_grad);
        Ok(g)
    }
}

#[cfg(all(test, feature = "nalgebra_all"))]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    /// `min ½‖x‖²` subject to a single row `x₀ + x₁ = 2`.
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

    impl LinearEqualityConstraints for Probe {
        type Matrix = DMatrix<f64>;
        fn a(&self) -> &DMatrix<f64> {
            &self.a
        }
        fn b(&self) -> &DVector<f64> {
            &self.b
        }
    }

    #[test]
    fn cost_matches_closed_form() {
        let p = Probe::new();
        let lambda = DVector::from_vec(vec![0.5]);
        let rho = 2.0;
        let al = AugmentedLagrangian::new(&p, &lambda, rho);
        let x = DVector::from_vec(vec![0.3, -0.4]);
        // f = ½(0.09 + 0.16) = 0.125; c = (0.3 − 0.4) − 2 = −2.1;
        // L = 0.125 + 0.5·(−2.1) + 0.5·2·(−2.1)² = 0.125 − 1.05 + 4.41.
        let c = (0.3 - 0.4) - 2.0;
        let expected = 0.125 + 0.5 * c + 0.5 * rho * c * c;
        assert!((al.cost(&x).unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn cost_is_finite_at_infeasible_points() {
        // No feasibility wall: L_ρ is finite even far from the affine set.
        let p = Probe::new();
        let lambda = DVector::from_vec(vec![0.0]);
        let al = AugmentedLagrangian::new(&p, &lambda, 1.0);
        let x = DVector::from_vec(vec![100.0, 100.0]);
        assert!(al.cost(&x).unwrap().is_finite());
    }

    #[test]
    fn gradient_agrees_with_finite_differences() {
        let p = Probe::new();
        let lambda = DVector::from_vec(vec![0.7]);
        let al = AugmentedLagrangian::new(&p, &lambda, 1.3);
        let x = DVector::from_vec(vec![0.3, -0.4]);
        let analytic = al.gradient(&x).unwrap();

        let h = 1e-6;
        for j in 0..2 {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[j] += h;
            xm[j] -= h;
            let fd =
                (al.cost(&xp).unwrap() - al.cost(&xm).unwrap()) / (2.0 * h);
            assert!(
                (analytic[j] - fd).abs() < 1e-5,
                "component {j}: analytic {} vs fd {}",
                analytic[j],
                fd
            );
        }
    }
}
