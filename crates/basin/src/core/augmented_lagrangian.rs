//! Augmented-Lagrangian adapter for linear equality constraints.
//!
//! [`AugmentedLagrangian`] wraps a [`Constraints`](crate::Constraints) problem
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
//! # Constraint transformation
//!
//! `AugmentedLagrangian` consumes a set containing only linear equality constraints and
//! exposes [`NoConstraints`](crate::NoConstraints) to the inner solver. Other
//! blocks are rejected at compile time. Derivative adapters such as
//! [`FiniteDiff`](crate::FiniteDiff) instead preserve the full declaration.
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
    LinearEqualityGradient, LinearEqualityProblem, invalid,
};
use crate::core::math::{NormSquared, Scalar, ScaledAdd, VectorLen};
use crate::core::problem::{CostFunction, Gradient};

/// A [`Constraints`](crate::Constraints) problem rewritten as the unconstrained
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

impl<P, X, R, F> CostFunction for AugmentedLagrangian<'_, P, R, F>
where
    F: Scalar,
    P: CostFunction<Param = X, Output = F> + LinearEqualityProblem<X, F>,
    X: VectorLen,
    R: VectorLen + std::ops::Index<usize, Output = F>,
{
    type Param = X;
    type Output = F;
    type Error = ConstraintSolveError<P::Error>;
    fn cost(&self, x: &X) -> Result<F, Self::Error> {
        augmented_cost(
            self,
            x,
            || {
                self.problem
                    .cost(x)
                    .map_err(ConstraintSolveError::Evaluation)
            },
            invalid,
        )
    }
    fn cost_with_context(
        &self,
        x: &X,
        context: &crate::EvaluationContext,
    ) -> Result<F, crate::EvaluationError<<Self as CostFunction>::Error>> {
        augmented_cost(
            self,
            x,
            || {
                self.problem.cost_with_context(x, context).map_err(|error| {
                    error.map_evaluation(ConstraintSolveError::Evaluation)
                })
            },
            |error| crate::EvaluationError::Evaluation(invalid(error)),
        )
    }
    fn cost_batch_request(&self, points: u64) -> crate::EvalCounts {
        self.problem.cost_batch_request(points)
    }
    fn prepare_cost_batch(
        &self,
        points: u64,
    ) -> Result<crate::CostBatchReservation, Self::Error> {
        self.problem
            .prepare_cost_batch(points)
            .map_err(ConstraintSolveError::Evaluation)
    }
    fn cost_in_batch(
        &self,
        x: &X,
        reservation: &crate::CostBatchReservation,
    ) -> Result<F, Self::Error> {
        augmented_cost(
            self,
            x,
            || {
                self.problem
                    .cost_in_batch(x, reservation)
                    .map_err(ConstraintSolveError::Evaluation)
            },
            invalid,
        )
    }
}

fn augmented_cost<P, X, R, F, E>(
    this: &AugmentedLagrangian<'_, P, R, F>,
    x: &X,
    evaluate: impl FnOnce() -> Result<F, E>,
    constraint_error: impl Fn(crate::ConstraintError) -> E,
) -> Result<F, E>
where
    F: Scalar,
    P: CostFunction<Param = X, Output = F> + LinearEqualityProblem<X, F>,
    X: VectorLen,
    R: VectorLen + std::ops::Index<usize, Output = F>,
{
    let c = this.problem.linear_residual(x).map_err(&constraint_error)?;
    crate::core::constraint::model::dimension(
        crate::ConstraintBlock::LinearEqualities,
        crate::ConstraintAxis::Length,
        c.len(),
        this.lambda.vec_len(),
    )
    .map_err(&constraint_error)?;
    let multiplier = (0..c.len()).map(|i| this.lambda[i] * c[i]).sum::<F>();
    Ok(evaluate()?
        + multiplier
        + F::from_f64(0.5).unwrap() * this.rho * c.norm_squared())
}

impl<P, X, R, F> Gradient for AugmentedLagrangian<'_, P, R, F>
where
    F: Scalar,
    P: CostFunction<Param = X, Output = F>
        + LinearEqualityGradient<X, F>
        + Gradient<Gradient = X>,
    X: Clone + VectorLen + ScaledAdd<F>,
    R: VectorLen + std::ops::Index<usize, Output = F>,
{
    type Gradient = X;
    fn gradient(&self, x: &X) -> Result<X, <Self as CostFunction>::Error> {
        augmented_gradient(
            self,
            x,
            || {
                self.problem
                    .gradient(x)
                    .map_err(ConstraintSolveError::Evaluation)
            },
            invalid,
        )
    }
    fn gradient_with_context(
        &self,
        x: &X,
        context: &crate::EvaluationContext,
    ) -> Result<X, crate::EvaluationError<<Self as CostFunction>::Error>> {
        augmented_gradient(
            self,
            x,
            || {
                self.problem.gradient_with_context(x, context).map_err(
                    |error| {
                        error.map_evaluation(ConstraintSolveError::Evaluation)
                    },
                )
            },
            |error| crate::EvaluationError::Evaluation(invalid(error)),
        )
    }
    fn cost_and_gradient_with_context(
        &self,
        x: &X,
        context: &crate::EvaluationContext,
    ) -> Result<(F, X), crate::EvaluationError<<Self as CostFunction>::Error>>
    {
        context.reserve_kinds(&[
            crate::EvaluationKind::Cost,
            crate::EvaluationKind::Gradient,
        ])?;
        let local = context.prepaid(crate::EvalCounts {
            cost_evals: 1,
            gradient_evals: 1,
            ..crate::EvalCounts::default()
        });
        Ok((
            self.cost_with_context(x, &local)?,
            self.gradient_with_context(x, &local)?,
        ))
    }
}

fn augmented_gradient<P, X, R, F, E>(
    this: &AugmentedLagrangian<'_, P, R, F>,
    x: &X,
    evaluate: impl FnOnce() -> Result<X, E>,
    constraint_error: impl Fn(crate::ConstraintError) -> E,
) -> Result<X, E>
where
    F: Scalar,
    P: CostFunction<Param = X, Output = F>
        + LinearEqualityGradient<X, F>
        + Gradient<Gradient = X>,
    X: Clone + VectorLen + ScaledAdd<F>,
    R: VectorLen + std::ops::Index<usize, Output = F>,
{
    let c = this.problem.linear_residual(x).map_err(&constraint_error)?;
    crate::core::constraint::model::dimension(
        crate::ConstraintBlock::LinearEqualities,
        crate::ConstraintAxis::Length,
        c.len(),
        this.lambda.vec_len(),
    )
    .map_err(&constraint_error)?;
    let weights: Vec<F> = (0..c.len())
        .map(|i| this.lambda[i] + this.rho * c[i])
        .collect();
    let mut contribution = x.clone();
    this.problem
        .linear_transpose(&weights, &mut contribution)
        .map_err(&constraint_error)?;
    let mut g = evaluate()?;
    g.scaled_add(F::one(), &contribution);
    Ok(g)
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

    impl crate::ProblemConstraints for Probe {
        type Constraints = crate::Constraints<
            crate::NoBounds,
            crate::LinearBlocks<
                crate::LinearBlock<DMatrix<f64>, DVector<f64>>,
                crate::NoLinearBlock,
            >,
        >;
        fn constraint_set(
            &self,
        ) -> <Self::Constraints as crate::ConstraintDescription>::Borrowed<'_>
        {
            crate::Constraints::new().with_linear_equalities(
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

impl<P, V, F> crate::ProblemConstraints for AugmentedLagrangian<'_, P, V, F> {
    type Constraints = crate::NoConstraints;
    fn constraint_set(&self) -> crate::NoConstraints {
        crate::NoConstraints::EMPTY
    }
}
