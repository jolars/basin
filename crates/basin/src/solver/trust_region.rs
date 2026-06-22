//! Trust-region (Newton) minimization.
//!
//! [`TrustRegion`] is a general unconstrained minimizer over the
//! second-order model `m(p) = f + gᵀp + ½ pᵀ B p`, where `g = ∇f(x)` and
//! `B = ∇²f(x)` are supplied by the problem's
//! [`Gradient`](crate::core::problem::Gradient) and [`Hessian`](crate::core::problem::Hessian) impls. Each iteration approximately
//! minimizes `m` over a ball `‖p‖ ≤ Δ`, compares the achieved reduction to
//! the model's prediction, and grows or shrinks the radius `Δ` accordingly
//! (Nocedal & Wright, *Numerical Optimization*, 2e, Algorithm 4.1).
//!
//! The trust-region *subproblem* — the constrained quadratic minimization —
//! is solved by a pluggable strategy implementing the crate-internal
//! `Subproblem` seam. Three ship today, mirroring the standard textbook
//! set:
//!
//! - [`Steihaug`] — truncated conjugate gradient (the default). Matrix-free
//!   (needs only Hessian-vector products), handles indefinite `B` by
//!   following a negative-curvature direction to the boundary. The
//!   large-scale-capable workhorse; runs on every backend.
//! - [`Dogleg`] — the Cauchy/Newton dogleg path (N&W eq. 4.16). Needs a
//!   Cholesky solve for the Newton step, so it requires a backend with
//!   [`LinearSolveSpd`](crate::core::math::LinearSolveSpd); it falls back to
//!   the Cauchy point when `B` is not positive definite.
//! - [`CauchyPoint`] — the steepest-descent-to-boundary closed form (N&W
//!   eq. 4.11–4.12). A bulletproof baseline with only linear convergence;
//!   mostly a reference strategy.

pub mod dogleg;
pub mod steihaug;

pub use dogleg::Dogleg;
pub use steihaug::Steihaug;

use crate::core::inner::InitialState;
use crate::core::math::{Dot, MatVec, NegInPlace, NormSquared, Scalar, ScaleInPlace, ScaledAdd};
use crate::core::problem::{CostFunction, Gradient, Hessian, Problem};
use crate::core::solver::Solver;
use crate::core::state::BasicState;
use crate::core::termination::TerminationReason;

/// The outcome of an (approximate) trust-region subproblem solve: the step
/// `d`, the predicted model decrease `m(0) − m(d) ≥ 0`, and whether the
/// step landed on the trust-region boundary (which gates radius growth).
pub(crate) struct Step<V, F> {
    /// The step `d` (relative to the current iterate).
    pub(crate) d: V,
    /// Predicted reduction `m(0) − m(d) = −(gᵀd) − ½ dᵀBd`, always `≥ 0`
    /// for the shipped strategies. Zero only when `d = 0` (gradient already
    /// negligible), which the driver reads as convergence.
    pub(crate) predicted_reduction: F,
    /// `true` when `‖d‖ ≈ Δ` — the constraint is active. Only then may the
    /// driver grow the radius on a very good step (N&W Algorithm 4.1).
    pub(crate) hit_boundary: bool,
}

/// A strategy that approximately minimizes the quadratic model
/// `m(p) = gᵀp + ½ pᵀ B p` over the trust region `‖p‖ ≤ radius`.
///
/// Crate-internal: the shipped strategies ([`Steihaug`], [`Dogleg`],
/// [`CauchyPoint`]) are the closed set for now. Each binds `M` (the Hessian
/// matrix type) on only the ops it needs — [`MatVec`] for the matrix-free
/// strategies, plus [`LinearSolveSpd`](crate::core::math::LinearSolveSpd)
/// for [`Dogleg`] — so a backend missing an op is a compile error for that
/// strategy alone (tenet 5). Promoting this trait to public is an additive,
/// non-breaking change if user-defined subproblem solvers are ever wanted.
pub(crate) trait Subproblem<V, M, F> {
    /// Approximately minimize `m(p) = gᵀp + ½ pᵀ B p` over `‖p‖ ≤ radius`.
    fn solve(&self, gradient: &V, hessian: &M, radius: F) -> Step<V, F>;
}

/// Model decrease `m(0) − m(d) = −(gᵀd) − ½ dᵀBd` for the local quadratic
/// model with gradient `g` and Hessian `B`. Shared by every subproblem
/// strategy so the predicted-reduction convention lives in one place.
pub(crate) fn model_decrease<V, M, F>(g: &V, b: &M, d: &V) -> F
where
    F: Scalar,
    V: Dot<F>,
    M: MatVec<V>,
{
    let bd = b.matvec(d);
    let half = F::from_f64(0.5).unwrap();
    -g.dot(d) - half * d.dot(&bd)
}

/// The largest `τ ≥ 0` with `‖z + τ d‖ = radius`: the positive root of the
/// quadratic `‖z + τ d‖² = radius²`. Used to walk a CG iterate (Steihaug)
/// or a negative-curvature direction out to the trust-region boundary.
/// Assumes `z` lies inside the ball (`‖z‖ ≤ radius`), so the discriminant
/// is non-negative; it is clamped to zero defensively against roundoff.
pub(crate) fn tau_to_boundary<V, F>(z: &V, d: &V, radius: F) -> F
where
    F: Scalar,
    V: Dot<F>,
{
    let dd = d.dot(d);
    let zd = z.dot(d);
    let zz = z.dot(z);
    let rr = radius * radius;
    let disc = zd * zd - dd * (zz - rr);
    let disc = if disc < F::zero() { F::zero() } else { disc };
    (-zd + disc.sqrt()) / dd
}

/// Cauchy-point subproblem strategy: minimize the model along the
/// steepest-descent direction `−g`, capped at the trust-region boundary
/// (Nocedal & Wright eq. 4.11–4.12).
///
/// The step is `p = −τ (Δ / ‖g‖) g`, with the scalar `τ` chosen to minimize
/// the model along `−g` within the region: `τ = 1` when the curvature
/// `gᵀBg ≤ 0` (the model decreases without bound, so go to the boundary),
/// otherwise `τ = min(‖g‖³ / (Δ gᵀBg), 1)`. Robust but only linearly
/// convergent — it ignores all curvature off the gradient direction. Useful
/// as a baseline, and as [`Dogleg`]'s fallback when the Hessian is not
/// positive definite.
#[derive(Debug, Clone, Copy, Default)]
pub struct CauchyPoint;

impl<V, M, F> Subproblem<V, M, F> for CauchyPoint
where
    F: Scalar,
    V: Clone + Dot<F> + NormSquared<F> + ScaleInPlace<F> + NegInPlace,
    M: MatVec<V>,
{
    fn solve(&self, g: &V, b: &M, radius: F) -> Step<V, F> {
        let g_norm = g.norm_squared().sqrt();
        if g_norm <= F::zero() {
            // Gradient already negligible: the zero step is optimal.
            let mut d = g.clone();
            d.scale_in_place(F::zero());
            return Step {
                d,
                predicted_reduction: F::zero(),
                hit_boundary: false,
            };
        }
        let bg = b.matvec(g);
        let gbg = g.dot(&bg);
        let tau = if gbg <= F::zero() {
            F::one()
        } else {
            let t = g_norm * g_norm * g_norm / (radius * gbg);
            if t < F::one() { t } else { F::one() }
        };
        // p = −τ (Δ / ‖g‖) g.
        let mut d = g.clone();
        d.scale_in_place(-(tau * radius / g_norm));
        let predicted_reduction = model_decrease(g, b, &d);
        Step {
            d,
            predicted_reduction,
            // τ = 1 ⟺ the step is the full Δ-length steepest-descent step,
            // i.e. it sits on the boundary.
            hit_boundary: tau >= F::one(),
        }
    }
}

/// Trust-region Newton minimizer (Nocedal & Wright, *Numerical
/// Optimization*, 2e, §4 / Algorithm 4.1).
///
/// At each iterate `x` the local quadratic model
/// `m(p) = f(x) + ∇f(x)ᵀp + ½ pᵀ ∇²f(x) p` is approximately minimized over a
/// ball `‖p‖ ≤ Δ` by the configured `Subproblem` strategy. The ratio of
/// achieved to predicted reduction,
/// `ρ = (f(x) − f(x + p)) / (m(0) − m(p))`, drives the radius: `Δ` shrinks
/// when `ρ` is poor (`< ¼`), grows when `ρ` is excellent (`> ¾`) and the
/// step is constrained by the boundary, and the step is accepted when
/// `ρ > η`. A rejected step shrinks `Δ` and re-solves the subproblem with
/// the *same* gradient and Hessian — no extra derivative evaluations — up to
/// [`with_max_inner_attempts`](Self::with_max_inner_attempts) times per
/// outer iteration (the same reuse pattern as
/// [`LevenbergMarquardt`](crate::solver::LevenbergMarquardt)).
///
/// The subproblem strategy defaults to [`Steihaug`] (truncated CG); choose
/// another with [`with_subproblem`](Self::with_subproblem). The radius `Δ`
/// is solver-internal working state — there are no framework termination
/// knobs for it; pair the solver with
/// [`GradientTolerance`](crate::core::termination::GradientTolerance) /
/// [`MaxIter`](crate::core::termination::MaxIter) like any first-order
/// solver.
///
/// # Backends
///
/// The solver itself needs only `Clone`, [`ScaledAdd`], and
/// [`NormSquared`] on the parameter vector, plus a [`Hessian`] impl. The
/// effective backend coverage is set by the chosen subproblem:
/// [`Steihaug`] and [`CauchyPoint`] need only [`MatVec`] on the Hessian, so
/// they run on every backend that has a dense matrix type (`Vec<f64>` via
/// [`DenseMatrix`](crate::core::math::DenseMatrix), nalgebra, faer, and
/// `ndarray`); [`Dogleg`] additionally needs
/// [`LinearSolveSpd`](crate::core::math::LinearSolveSpd), which `ndarray`
/// does not provide (a compile error there, per tenet 5). All shipped
/// strategies are wasm-clean (pure-Rust, no BLAS/LAPACK).
///
/// Note that a [`Hessian`] is required: an analytic one from the problem, or
/// a finite-difference one via
/// [`FiniteDiff`](crate::core::numdiff::FiniteDiff) over a backend with a
/// dense matrix type (nalgebra / faer).
///
/// # References
///
/// Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.),
/// Chapter 4 (trust-region methods). Springer.
/// [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5).
///
/// # Examples
///
/// Minimize the 2-D Rosenbrock function with an analytic Hessian over the
/// nalgebra backend (the dense-matrix backend the example's `DMatrix`
/// Hessian uses):
///
/// ```
/// # #[cfg(feature = "nalgebra")] {
/// use basin::{CostFunction, Executor, Gradient, GradientTolerance, Hessian, TrustRegion};
/// use nalgebra::{DMatrix, DVector};
///
/// struct Rosenbrock;
/// impl CostFunction for Rosenbrock {
///     type Param = DVector<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &DVector<f64>) -> Result<f64, Self::Error> {
///         Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2))
///     }
/// }
/// impl Gradient for Rosenbrock {
///     type Gradient = DVector<f64>;
///     fn gradient(&self, x: &DVector<f64>) -> Result<DVector<f64>, Self::Error> {
///         Ok(DVector::from_vec(vec![
///             -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2)),
///             200.0 * (x[1] - x[0].powi(2)),
///         ]))
///     }
/// }
/// impl Hessian for Rosenbrock {
///     type Hessian = DMatrix<f64>;
///     fn hessian(&self, x: &DVector<f64>) -> Result<DMatrix<f64>, Self::Error> {
///         let h11 = 2.0 - 400.0 * (x[1] - 3.0 * x[0].powi(2));
///         Ok(DMatrix::from_row_slice(2, 2, &[
///             h11, -400.0 * x[0],
///             -400.0 * x[0], 200.0,
///         ]))
///     }
/// }
///
/// let result = Executor::new(Rosenbrock, TrustRegion::new(), basin::BasicState::new(DVector::from_vec(vec![-1.2, 1.0])))
///     .max_iter(100)
///     .terminate_on(GradientTolerance(1e-8))
///     .run()
///     .unwrap();
/// assert!(result.cost() < 1e-10);
/// # }
/// ```
pub struct TrustRegion<Sub = Steihaug, F = f64> {
    subproblem: Sub,
    /// Current trust radius `Δ`, mutated across iterations. Reset to
    /// `initial_radius` by [`Solver::init`].
    radius: F,
    initial_radius: F,
    max_radius: F,
    eta: F,
    max_inner: u32,
}

impl Default for TrustRegion<Steihaug> {
    fn default() -> Self {
        Self::new()
    }
}

impl TrustRegion<Steihaug> {
    /// Trust-region solver with the default [`Steihaug`] (truncated CG)
    /// subproblem: initial radius `1.0`, maximum radius `100.0`, acceptance
    /// threshold `η = 0.125`, and up to `10` radius reductions per outer
    /// iteration.
    pub fn new() -> Self {
        Self::with_subproblem(Steihaug::new())
    }
}

impl<Sub, F: Scalar> TrustRegion<Sub, F> {
    /// Trust-region solver with an explicit subproblem strategy
    /// ([`Steihaug`], [`Dogleg`], or [`CauchyPoint`]).
    pub fn with_subproblem(subproblem: Sub) -> Self {
        Self {
            subproblem,
            radius: F::one(),
            initial_radius: F::one(),
            max_radius: F::from_f64(100.0).unwrap(),
            eta: F::from_f64(0.125).unwrap(),
            max_inner: 10,
        }
    }

    /// Initial trust radius `Δ₀` (default `1.0`). Must be positive. A good
    /// `Δ₀` is the order of magnitude of the expected step to the minimum.
    pub fn with_radius(mut self, radius: F) -> Self {
        assert!(radius > F::zero(), "initial radius must be > 0");
        self.initial_radius = radius;
        self.radius = radius;
        self
    }

    /// Upper bound on the trust radius `Δ_max` (default `100.0`). Must be
    /// positive. Caps radius growth on a run of excellent steps.
    pub fn with_max_radius(mut self, max_radius: F) -> Self {
        assert!(max_radius > F::zero(), "max radius must be > 0");
        self.max_radius = max_radius;
        self
    }

    /// Step-acceptance threshold `η` (default `0.125`): a step is accepted
    /// when the reduction ratio `ρ > η`. Nocedal & Wright require
    /// `η ∈ [0, ¼)`; this is asserted.
    pub fn with_eta(mut self, eta: F) -> Self {
        assert!(
            eta >= F::zero() && eta < F::from_f64(0.25).unwrap(),
            "eta must be in [0, 1/4)"
        );
        self.eta = eta;
        self
    }

    /// Maximum radius reductions per outer iteration (default `10`). Each
    /// rejected step shrinks `Δ` and re-solves the subproblem with the same
    /// gradient and Hessian; after this many rejections the iteration
    /// returns the iterate unmoved with the shrunken radius, and the next
    /// outer iteration retries. Must be `≥ 1`.
    pub fn with_max_inner_attempts(mut self, n: u32) -> Self {
        assert!(n >= 1, "max inner attempts must be ≥ 1");
        self.max_inner = n;
        self
    }
}

impl<Sub, V, F> InitialState<V> for TrustRegion<Sub, F>
where
    F: Scalar,
    V: Clone,
{
    type State = BasicState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        BasicState::new(x.clone())
    }
}

impl<P, Sub, V, M, F> Solver<P, BasicState<V, F>> for TrustRegion<Sub, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V> + Hessian<Hessian = M>,
    V: Clone + ScaledAdd<F> + NormSquared<F>,
    Sub: Subproblem<V, M, F>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: BasicState<V, F>,
    ) -> Result<BasicState<V, F>, Self::Error> {
        // A reused solver instance must restart from the configured radius.
        self.radius = self.initial_radius;
        // Seed cost and gradient so iter-0 termination checks (e.g.
        // GradientTolerance on a near-optimal start) see a complete state.
        // The Hessian is recomputed per iteration in `next_iter`, so it is
        // not seeded here.
        let (cost, grad) = problem.cost_and_gradient(&state.param)?;
        state.cost = Some(cost);
        state.gradient = Some(grad);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: BasicState<V, F>,
    ) -> Result<(BasicState<V, F>, Option<TerminationReason>), Self::Error> {
        let g = state
            .gradient
            .take()
            .expect("gradient not set: Solver::init must run before next_iter");
        let cost_old = state
            .cost
            .expect("cost not set: Solver::init must run before next_iter");

        // One Hessian per outer iteration, reused across all inner radius
        // reductions below (the gradient is likewise fixed while x is).
        let b = problem.hessian(&state.param)?;

        let quarter = F::from_f64(0.25).unwrap();
        let three_quarters = F::from_f64(0.75).unwrap();
        let two = F::from_f64(2.0).unwrap();

        for _ in 0..self.max_inner {
            let step = self.subproblem.solve(&g, &b, self.radius);

            // Predicted reduction ≤ 0 means the model cannot decrease — for
            // the shipped strategies this only happens at a stationary point
            // (g ≈ 0). Report a clean convergence stop.
            if step.predicted_reduction <= F::zero() {
                state.gradient = Some(g);
                return Ok((state, Some(TerminationReason::SolverConverged)));
            }

            let mut trial = state.param.clone();
            trial.scaled_add(F::one(), &step.d);
            let cost_trial = problem.cost(&trial)?;

            let rho = (cost_old - cost_trial) / step.predicted_reduction;
            let step_norm = step.d.norm_squared().sqrt();

            // Radius update (N&W Algorithm 4.1). A non-finite ρ (trial cost
            // Inf/NaN from a soft rejection) routes to the shrink branch, so
            // the radius always decreases on a bad step and the inner loop
            // can't stall.
            //
            // The shrink uses ¼‖p‖ rather than the literal ¼Δ of Algorithm
            // 4.1: the two agree for a boundary step (‖p‖ ≈ Δ) but ¼‖p‖
            // shrinks harder on a rejected *interior* step, anchoring the new
            // radius to the step the model actually mispredicted. This
            // follows argmin's `TrustRegion` (0.25 * pk_norm); deliberate, not
            // the textbook ¼Δ.
            if rho < quarter || !rho.is_finite() {
                self.radius = quarter * step_norm;
            } else if rho > three_quarters && step.hit_boundary {
                let grown = two * self.radius;
                self.radius = if grown < self.max_radius {
                    grown
                } else {
                    self.max_radius
                };
            }

            if rho > self.eta {
                // Accept: move to the trial point and refresh the gradient
                // there (the Hessian is recomputed by the next iteration).
                state.param = trial;
                state.cost = Some(cost_trial);
                let g_new = problem.gradient(&state.param)?;
                state.gradient = Some(g_new);
                return Ok((state, None));
            }
            // Reject: radius has shrunk; retry with the same g and b.
        }

        // Inner attempts exhausted without an acceptable step. Keep the
        // shrunken radius and the current iterate; restore the gradient so
        // the state stays consistent and let the next outer iteration retry
        // (or a termination criterion fire).
        state.gradient = Some(g);
        Ok((state, None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BasicState, Executor, GradientTolerance};

    /// Ill-conditioned quadratic `f(x) = ½ xᵀ A x` with `A = diag(1, 100)`,
    /// gradient `A x`, constant Hessian `A`. A single Newton step solves it
    /// exactly, so any curvature-aware trust-region run reaches the origin
    /// fast.
    struct Quadratic;

    impl CostFunction for Quadratic {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
            Ok(0.5 * (x[0] * x[0] + 100.0 * x[1] * x[1]))
        }
    }
    impl Gradient for Quadratic {
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            Ok(vec![x[0], 100.0 * x[1]])
        }
    }
    impl Hessian for Quadratic {
        type Hessian = crate::core::math::DenseMatrix<f64>;
        fn hessian(&self, _x: &Vec<f64>) -> Result<Self::Hessian, Self::Error> {
            Ok(crate::core::math::DenseMatrix::from_row_slice(
                2,
                2,
                &[1.0, 0.0, 0.0, 100.0],
            ))
        }
    }

    /// 2-D Rosenbrock with an analytic Hessian over the dependency-free
    /// `Vec<f64>` backend (`DenseMatrix`). The classic nonconvex test: the
    /// Hessian is indefinite far from the valley floor, so it exercises the
    /// curvature handling of every subproblem.
    struct Rosenbrock;

    impl CostFunction for Rosenbrock {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
            Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2))
        }
    }
    impl Gradient for Rosenbrock {
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            Ok(vec![
                -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2)),
                200.0 * (x[1] - x[0].powi(2)),
            ])
        }
    }
    impl Hessian for Rosenbrock {
        type Hessian = crate::core::math::DenseMatrix<f64>;
        fn hessian(&self, x: &Vec<f64>) -> Result<Self::Hessian, Self::Error> {
            let h11 = 2.0 + 1200.0 * x[0] * x[0] - 400.0 * x[1];
            let h12 = -400.0 * x[0];
            Ok(crate::core::math::DenseMatrix::from_row_slice(
                2,
                2,
                &[h11, h12, h12, 200.0],
            ))
        }
    }

    #[test]
    fn cauchy_point_minimizes_quadratic() {
        let result = Executor::new(
            Quadratic,
            TrustRegion::with_subproblem(CauchyPoint),
            BasicState::new(vec![5.0, 1.0]),
        )
        .max_iter(500)
        .terminate_on(GradientTolerance(1e-8))
        .run()
        .unwrap();
        // Cauchy point is only linearly convergent on this conditioning, so
        // a strong-but-not-machine-precision bound is the honest check.
        assert!(result.cost() < 1e-8, "cost = {}", result.cost());
    }

    #[test]
    fn steihaug_minimizes_quadratic() {
        // Truncated CG is curvature-aware: it reaches the minimizer of a
        // quadratic to machine precision in a handful of iterations.
        let result = Executor::new(
            Quadratic,
            TrustRegion::with_subproblem(Steihaug::new()),
            BasicState::new(vec![5.0, 1.0]),
        )
        .max_iter(100)
        .terminate_on(GradientTolerance(1e-10))
        .run()
        .unwrap();
        assert!(result.cost() < 1e-16, "cost = {}", result.cost());
    }

    #[test]
    fn dogleg_minimizes_quadratic() {
        let result = Executor::new(
            Quadratic,
            TrustRegion::with_subproblem(Dogleg),
            BasicState::new(vec![5.0, 1.0]),
        )
        .max_iter(100)
        .terminate_on(GradientTolerance(1e-10))
        .run()
        .unwrap();
        assert!(result.cost() < 1e-16, "cost = {}", result.cost());
    }

    #[test]
    fn steihaug_minimizes_rosenbrock() {
        // The default subproblem on the canonical nonconvex problem, from
        // the standard hard start. Indefinite Hessians along the way are
        // handled by the negative-curvature-to-boundary rule.
        let result = Executor::new(
            Rosenbrock,
            TrustRegion::new(),
            BasicState::new(vec![-1.2, 1.0]),
        )
        .max_iter(200)
        .terminate_on(GradientTolerance(1e-8))
        .run()
        .unwrap();
        assert!(result.cost() < 1e-10, "cost = {}", result.cost());
    }

    #[test]
    fn dogleg_minimizes_rosenbrock() {
        // Dogleg falls back to the Cauchy point wherever the Hessian is
        // indefinite, so it still drives Rosenbrock to the minimum.
        let result = Executor::new(
            Rosenbrock,
            TrustRegion::with_subproblem(Dogleg),
            BasicState::new(vec![-1.2, 1.0]),
        )
        .max_iter(500)
        .terminate_on(GradientTolerance(1e-8))
        .run()
        .unwrap();
        assert!(result.cost() < 1e-10, "cost = {}", result.cost());
    }
}
