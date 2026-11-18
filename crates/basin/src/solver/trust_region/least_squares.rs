use super::{
    GaussNewtonModel, Subproblem, TrialEvaluation, TrustRegion, tr_next_iter,
    zero_gradient,
};
use crate::core::least_squares::evaluation::Evaluation;
use crate::core::least_squares::{ReductionSum, squared_loss_reduction};
use crate::core::math::{
    GramMatrix, MatTransposeVec, MatrixFinite, NormSquared, Scalar,
    ScaleInPlace, ScaledAdd, VectorIndex, VectorLen,
};
use crate::{FirstOrderState, Problem, Solver, SolverStep, State, Termination};

/// The residual and either the Jacobian or its Gram matrix retained by
/// [`TrustRegion::least_squares_with`].
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussNewtonWorkspace<V, M> {
    residual: V,
    matrix: GaussNewtonMatrix<M>,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum GaussNewtonMatrix<M> {
    Unavailable,
    Gram(M),
    Jacobian(M),
}

impl<M: GramMatrix + MatrixFinite> GaussNewtonMatrix<M> {
    fn gram(&mut self) -> Option<&M> {
        if let Self::Jacobian(jacobian) = self {
            let gram = jacobian.gram();
            *self = if gram.matrix_is_finite() {
                Self::Gram(gram)
            } else {
                Self::Unavailable
            };
        }
        if let Self::Gram(gram) = self {
            Some(gram)
        } else {
            None
        }
    }
}

impl<Sub, F: Scalar, V, M>
    TrustRegion<Sub, F, GaussNewtonModel, Option<GaussNewtonWorkspace<V, M>>>
{
    /// Trust-region solver for unconstrained nonlinear least squares, with an
    /// explicit matrix strategy such as [`Dogleg`](super::Dogleg).
    ///
    /// Supply [`Residual`](crate::Residual) and [`Jacobian`](crate::Jacobian),
    /// optionally wrapped in [`RobustLeastSquares`](crate::RobustLeastSquares).
    /// The solver builds the Gram matrix on the first step from each point
    /// and reuses it across rejections. Radius defaults match
    /// [`TrustRegion::new`].
    /// If a step rounds back to the current point, the solver reports a
    /// numerical stall before requesting another residual.
    ///
    /// ```
    /// use basin::{DenseMatrix, Dogleg, Executor, Jacobian, Residual, TrustRegion};
    /// use std::convert::Infallible;
    ///
    /// struct Fit;
    /// impl Residual for Fit {
    ///     type Param = Vec<f64>;
    ///     type Output = Vec<f64>;
    ///     type Error = Infallible;
    ///
    ///     fn residual(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
    ///         Ok(vec![x[0] - 2.0, x[0] - 4.0])
    ///     }
    /// }
    /// impl Jacobian for Fit {
    ///     type Jacobian = DenseMatrix;
    ///
    ///     fn jacobian(&self, _: &Vec<f64>) -> Result<DenseMatrix, Infallible> {
    ///         Ok(DenseMatrix::from_row_slice(2, 1, &[1.0, 1.0]))
    ///     }
    /// }
    ///
    /// let solver = TrustRegion::least_squares_with(Dogleg)
    ///     .with_absolute_gradient_tolerance(1e-8);
    /// let result = Executor::from_start(Fit, solver, vec![0.0])
    ///     .max_iter(20)
    ///     .run()
    ///     .unwrap();
    /// assert!((result.param()[0] - 3.0).abs() < 1e-8);
    /// ```
    pub fn least_squares_with(subproblem: Sub) -> Self {
        Self::with_workspace(subproblem, None)
    }
}

fn quadratic_cost<F: Scalar, V: VectorLen + VectorIndex<F>>(r: &V) -> F {
    let half = F::from_f64(0.5).unwrap();
    (0..r.vec_len())
        .map(|i| {
            let value = r.get_scalar(i);
            (half * value) * value
        })
        .sum()
}

fn quadratic_cost_and_reduction<F: Scalar, V: VectorLen + VectorIndex<F>>(
    current: &V,
    trial: &V,
) -> (F, F) {
    let half = F::from_f64(0.5).unwrap();
    let mut cost = F::zero();
    let mut reduction = ReductionSum::new();
    for i in 0..trial.vec_len() {
        let before = current.get_scalar(i);
        let after = trial.get_scalar(i);
        cost = cost + (half * after) * after;
        let (term, error) = squared_loss_reduction(before, after);
        reduction.add(term);
        reduction.add(error);
    }
    (cost, reduction.total())
}

fn finite_gradient_model<V, M, F>(
    problem: &impl Evaluation<V, M, F>,
    residual: &V,
    jacobian: M,
) -> Option<(V, GaussNewtonMatrix<M>)>
where
    F: Scalar,
    V: VectorLen + VectorIndex<F>,
    M: MatTransposeVec<V>,
{
    let (gradient, jacobian) = problem.gradient_model(residual, jacobian)?;
    let dimension = gradient.vec_len();
    if !(0..dimension).all(|i| gradient.get_scalar(i).is_finite()) {
        return None;
    }
    let matrix = if dimension == 0 {
        GaussNewtonMatrix::Unavailable
    } else {
        GaussNewtonMatrix::Jacobian(jacobian)
    };
    Some((gradient, matrix))
}

struct LeastSquaresTrial<'a, V, M> {
    current_residual: &'a V,
    residual: V,
    matrix: GaussNewtonMatrix<M>,
}

impl<P, V, M, F, E> TrialEvaluation<P, V, F> for LeastSquaresTrial<'_, V, M>
where
    F: Scalar,
    V: VectorLen + VectorIndex<F>,
    M: MatTransposeVec<V>,
    Problem<P>: Evaluation<V, M, F, Error = E>,
{
    type Error = E;

    fn trial_is_unchanged(&self, current: &V, trial: &V) -> bool {
        let len = current.vec_len();
        len == trial.vec_len()
            && (0..len).all(|i| {
                let base = current.get_scalar(i);
                base.is_finite() && base == trial.get_scalar(i)
            })
    }

    fn evaluate_trial(
        &mut self,
        problem: &mut Problem<P>,
        trial: &V,
        cost_old: F,
    ) -> Result<(F, F), E> {
        Evaluation::residual_into(problem, trial, &mut self.residual)?;
        if self.current_residual.vec_len() != self.residual.vec_len() {
            let cost = problem.cost(&self.residual, quadratic_cost);
            return Ok((cost, cost_old - cost));
        }
        let (cost, reduction) = problem.cost_and_reduction(
            self.current_residual,
            &self.residual,
            quadratic_cost_and_reduction,
        );
        let reduction = if cost.is_finite() {
            reduction
        } else {
            cost_old - cost
        };
        Ok((cost, reduction))
    }

    fn refresh_gradient(
        &mut self,
        problem: &mut Problem<P>,
        trial: &V,
        gradient: &mut V,
    ) -> Result<bool, E> {
        let jacobian = Evaluation::jacobian(problem, trial)?;
        let Some((next_gradient, matrix)) =
            finite_gradient_model(problem, &self.residual, jacobian)
        else {
            return Ok(false);
        };
        *gradient = next_gradient;
        self.matrix = matrix;
        Ok(true)
    }
}

impl<P, Sub, V, M, F, E> Solver<P, FirstOrderState<V, F>>
    for TrustRegion<
        Sub,
        F,
        GaussNewtonModel,
        Option<GaussNewtonWorkspace<V, M>>,
    >
where
    F: Scalar,
    V: Clone
        + NormSquared<F>
        + ScaleInPlace<F>
        + ScaledAdd<F>
        + VectorLen
        + VectorIndex<F>,
    M: GramMatrix + MatTransposeVec<V> + MatrixFinite,
    Sub: Subproblem<V, M, F>,
    Problem<P>: Evaluation<V, M, F, Error = E>,
{
    type Error = E;

    fn should_check_iterate_change(&self) -> bool {
        !self.rejected_step
    }

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: FirstOrderState<V, F>,
    ) -> Result<FirstOrderState<V, F>, E> {
        self.reset_run();
        self.workspace = None;
        state.reset();

        let (residual, jacobian) =
            Evaluation::residual_and_jacobian(problem, state.param())?;
        let cost = problem.cost(&residual, quadratic_cost);
        let initial_model = if cost.is_finite() {
            finite_gradient_model(problem, &residual, jacobian)
        } else {
            None
        };
        let gradient = if let Some((gradient, matrix)) = initial_model {
            self.workspace = Some(GaussNewtonWorkspace { residual, matrix });
            gradient
        } else {
            // Publish the evaluated cost and signal the unavailable gradient
            // to observers and convergence checks.
            let mut gradient = state.param().clone();
            gradient.scale_in_place(F::nan());
            gradient
        };
        state
            .set_evaluation(cost, gradient)
            .expect("gradient dimension differs from parameter");
        Ok(state)
    }

    fn terminate(&self, _: &FirstOrderState<V, F>) -> Option<Termination<F>> {
        self.workspace.is_none().then(|| {
            Termination::numerical_failure(
                "The trust region could not initialize a finite cost and gradient.",
            )
        })
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: FirstOrderState<V, F>,
    ) -> Result<SolverStep<FirstOrderState<V, F>>, E> {
        let workspace = self
            .workspace
            .as_mut()
            .expect("the trust region requires an initialized evaluation");
        let Some(gram) = workspace.matrix.gram() else {
            let norm = state.current().unwrap().2.norm_l2();
            let reason = if norm == F::zero() {
                zero_gradient(norm)
            } else {
                Termination::numerical_failure(
                    "The trust region requires a finite Gram matrix to take another step.",
                )
            };
            return Ok(SolverStep::stopped(state, reason));
        };
        let subproblem = &self.subproblem;
        let mut trial = LeastSquaresTrial {
            current_residual: &workspace.residual,
            residual: state.param().clone(),
            matrix: GaussNewtonMatrix::Unavailable,
        };
        let (state, reason, accepted) = tr_next_iter(
            &mut self.radius,
            self.max_radius,
            self.eta,
            self.max_inner,
            &mut self.trials,
            problem,
            state,
            &mut trial,
            |_, _, gradient, radius| {
                Ok(subproblem.solve(gradient, gram, radius))
            },
        )?;
        if accepted {
            let LeastSquaresTrial {
                residual, matrix, ..
            } = trial;
            *workspace = GaussNewtonWorkspace { residual, matrix };
        }
        self.rejected_step = !accepted;
        Ok(SolverStep::from((state, reason)))
    }
}
