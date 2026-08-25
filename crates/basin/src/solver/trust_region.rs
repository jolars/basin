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
//! The trust-region *subproblem*, the constrained quadratic minimization,
//! is solved by a pluggable strategy implementing the crate-internal
//! `Subproblem` seam. Four ship today:
//!
//! - [`Steihaug`]: truncated conjugate gradient (the default). Matrix-free
//!   (needs only Hessian-vector products), handles indefinite `B` by
//!   following a negative-curvature direction to the boundary. The
//!   large-scale-capable workhorse; runs on every backend.
//! - [`Dogleg`]: the Cauchy and Newton dogleg path (N&W eq. 4.16). Needs a
//!   Cholesky solve for the Newton step, so it requires a backend with
//!   [`LinearSolveSpd`](crate::core::math::LinearSolveSpd); it falls back to
//!   the Cauchy point when `B` is not positive definite.
//! - [`MoreSorensen`]: a near-exact global solve using the safeguarded secular
//!   equation and explicit hard-case treatment (Moré & Sorensen 1983). It
//!   needs a full eigendecomposition plus Cholesky solves; use it when
//!   subproblem robustness matters more than large-scale cost.
//! - [`CauchyPoint`]: the steepest-descent-to-boundary closed form (N&W
//!   eq. 4.11–4.12). A bulletproof baseline with only linear convergence;
//!   mostly a reference strategy.
//!
//! The solver runs in one of two modes, selected by the `Mode` type
//! parameter: [`ExactHessian`] (the default) forms the full Hessian via the
//! problem's [`Hessian`](crate::core::problem::Hessian) impl, while
//! [`MatrixFree`] (via [`TrustRegion::matrix_free`]) drives the subproblem
//! purely through the problem's
//! [`HessianProduct`](crate::core::problem::HessianProduct) impl and never
//! forms a matrix.

pub mod dogleg;
pub mod more_sorensen;
pub mod steihaug;

pub use dogleg::Dogleg;
pub use more_sorensen::MoreSorensen;
pub use steihaug::Steihaug;

use std::marker::PhantomData;

use crate::core::inner::InitialState;
use crate::core::math::{
    Dot, MatVec, NegInPlace, NormSquared, Scalar, ScaleInPlace, ScaledAdd,
};
use crate::core::problem::{
    CostFunction, Gradient, Hessian, HessianProduct, Problem,
};
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
    /// negligible), which the driver reads as convergence. A non-finite
    /// value means the strategy could not solve the subproblem because the
    /// model data was non-finite; the driver reports that as a solver
    /// failure so a bad derivative cannot masquerade as a stationary point.
    pub(crate) predicted_reduction: F,
    /// `true` when `‖d‖ ≈ Δ`: the constraint is active. Only then may the
    /// driver grow the radius on a very good step (N&W Algorithm 4.1).
    pub(crate) hit_boundary: bool,
}

/// A strategy that approximately minimizes the quadratic model
/// `m(p) = gᵀp + ½ pᵀ B p` over the trust region `‖p‖ ≤ radius`.
///
/// Crate-internal: the shipped strategies ([`Steihaug`], [`Dogleg`],
/// [`MoreSorensen`], [`CauchyPoint`]) are the closed set for now. Each binds
/// `M` (the Hessian matrix type) on only the operations it needs: [`MatVec`]
/// for the universal strategies, [`LinearSolveSpd`](crate::core::math::LinearSolveSpd)
/// for [`Dogleg`], and Cholesky plus
/// [`SymmetricEigen`](crate::core::math::SymmetricEigen) for
/// [`MoreSorensen`]. A missing backend operation is therefore a compile error
/// for that strategy alone (tenet 5). Promoting this trait to public is an
/// additive, non-breaking change if user-defined subproblem solvers are ever
/// wanted.
pub(crate) trait Subproblem<V, M, F> {
    /// Approximately minimize `m(p) = gᵀp + ½ pᵀ B p` over `‖p‖ ≤ radius`.
    fn solve(&self, gradient: &V, hessian: &M, radius: F) -> Step<V, F>;
}

/// The matrix-free sibling of [`Subproblem`]: the Hessian is reachable only
/// through the fallible product closure `bv: v ↦ B·v` (in practice the
/// counted [`HessianProduct`] call, so
/// errors are the problem's own and must propagate). Crate-internal, like
/// [`Subproblem`]. [`Steihaug`] and [`CauchyPoint`] implement it; [`Dogleg`]
/// and [`MoreSorensen`] need the actual matrix, so pairing either with
/// [`MatrixFree`] mode is a compile error (tenet 5).
pub(crate) trait SubproblemHvp<V, F> {
    /// Approximately minimize `m(p) = gᵀp + ½ pᵀ B p` over `‖p‖ ≤ radius`,
    /// touching `B` only through `bv`.
    fn solve_hvp<E>(
        &self,
        gradient: &V,
        radius: F,
        bv: impl FnMut(&V) -> Result<V, E>,
    ) -> Result<Step<V, F>, E>;
}

/// Model decrease `m(0) − m(d) = −(gᵀd) − ½ dᵀBd` from a precomputed
/// product `bd = B·d`. Shared by every subproblem strategy so the
/// predicted-reduction convention lives in one place.
pub(crate) fn model_decrease_from_bd<V, F>(g: &V, d: &V, bd: &V) -> F
where
    F: Scalar,
    V: Dot<F>,
{
    let half = F::from_f64(0.5).unwrap();
    -g.dot(d) - half * d.dot(bd)
}

/// [`model_decrease_from_bd`] with the product computed via [`MatVec`], for
/// the matrix-based strategies.
pub(crate) fn model_decrease<V, M, F>(g: &V, b: &M, d: &V) -> F
where
    F: Scalar,
    V: Dot<F>,
    M: MatVec<V>,
{
    let bd = b.matvec(d);
    model_decrease_from_bd(g, d, &bd)
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
/// convergent; it ignores all curvature off the gradient direction. Useful
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

impl<V, F> SubproblemHvp<V, F> for CauchyPoint
where
    F: Scalar,
    V: Clone + Dot<F> + NormSquared<F> + ScaleInPlace<F> + NegInPlace,
{
    fn solve_hvp<E>(
        &self,
        g: &V,
        radius: F,
        mut bv: impl FnMut(&V) -> Result<V, E>,
    ) -> Result<Step<V, F>, E> {
        let g_norm = g.norm_squared().sqrt();
        if g_norm <= F::zero() {
            // Gradient already negligible: the zero step is optimal.
            let mut d = g.clone();
            d.scale_in_place(F::zero());
            return Ok(Step {
                d,
                predicted_reduction: F::zero(),
                hit_boundary: false,
            });
        }
        let bg = bv(g)?;
        let gbg = g.dot(&bg);
        let tau = if gbg <= F::zero() {
            F::one()
        } else {
            let t = g_norm * g_norm * g_norm / (radius * gbg);
            if t < F::one() { t } else { F::one() }
        };
        // p = −τ (Δ / ‖g‖) g, so B·p is the already-computed B·g rescaled:
        // one product per solve instead of two.
        let c = -(tau * radius / g_norm);
        let mut d = g.clone();
        d.scale_in_place(c);
        let mut bd = bg;
        bd.scale_in_place(c);
        let predicted_reduction = model_decrease_from_bd(g, &d, &bd);
        Ok(Step {
            d,
            predicted_reduction,
            // τ = 1 ⟺ the step is the full Δ-length steepest-descent step,
            // i.e. it sits on the boundary.
            hit_boundary: tau >= F::one(),
        })
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
/// the *same* gradient and Hessian (no extra derivative evaluations) up to
/// [`with_max_inner_attempts`](Self::with_max_inner_attempts) times per
/// outer iteration (the same reuse pattern as
/// [`LevenbergMarquardt`](crate::solver::LevenbergMarquardt)).
///
/// The subproblem strategy defaults to [`Steihaug`] (truncated CG); choose
/// another with [`with_subproblem`](Self::with_subproblem). The radius `Δ`
/// is solver-internal working state; there are no framework termination
/// knobs for it; pair the solver with
/// [`GradientTolerance`](crate::core::termination::GradientTolerance) /
/// [`MaxIter`](crate::core::termination::MaxIter) like any first-order
/// solver.
///
/// # Matrix-free mode
///
/// [`matrix_free`](Self::matrix_free) /
/// [`matrix_free_with`](Self::matrix_free_with) switch the `Mode` parameter
/// to [`MatrixFree`]: the problem then implements
/// [`HessianProduct`] instead of
/// [`Hessian`], and the subproblem reaches `B` only through counted
/// `v ↦ ∇²f(x)·v` calls. No matrix is ever formed, so the mode scales to
/// large `n` and runs on any vector backend, including plain `Vec<F>`
/// (which has no Hessian matrix type at all). [`Steihaug`] and
/// [`CauchyPoint`] support it; [`Dogleg`] and [`MoreSorensen`] need the actual
/// matrix, so pairing either with matrix-free mode is a compile error.
///
/// One cost asymmetry to know about: exact mode evaluates one Hessian per
/// outer iteration and reuses it across inner radius reductions for free,
/// while matrix-free mode re-pays the CG products when a rejected step
/// re-solves at the same iterate (the shrunken radius truncates CG earlier,
/// so re-attempts get cheaper). This is inherent to truncated-CG trust
/// regions; [`EvalCounts::hessian_product_evals`](crate::EvalCounts) makes
/// the cost visible.
///
/// # Backends
///
/// The solver itself needs only `Clone`, [`ScaledAdd`], and
/// [`NormSquared`] on the parameter vector, plus a [`Hessian`] impl in the
/// default [`ExactHessian`] mode. There, the effective backend coverage is
/// set by the chosen subproblem:
/// [`Steihaug`] and [`CauchyPoint`] need only [`MatVec`] on the Hessian, so
/// they run on every backend that has a dense matrix type (`Vec<f64>` via
/// [`DenseMatrix`](crate::core::math::DenseMatrix), nalgebra, faer, and
/// `ndarray`); [`Dogleg`] additionally needs
/// [`LinearSolveSpd`](crate::core::math::LinearSolveSpd), while
/// [`MoreSorensen`] needs that Cholesky solve plus
/// [`SymmetricEigen`](crate::core::math::SymmetricEigen). Those operations are
/// available on all four dense backends. In [`MatrixFree`] mode no matrix type
/// is bound at all, so every vector backend works. All shipped strategies are
/// wasm-clean in their pure-Rust configurations; the optional nalgebra LAPACK
/// acceleration remains non-WASM as documented by that feature.
///
/// Second-order information can reach the solver by three routes: an
/// analytic [`Hessian`] (exact mode), an analytic
/// [`HessianProduct`] (matrix-free
/// mode; see also the gradient-difference helpers
/// [`forward_difference_hessian_product`](crate::core::numdiff::forward_difference_hessian_product)
/// /
/// [`central_difference_hessian_product`](crate::core::numdiff::central_difference_hessian_product)),
/// or [`FiniteDiff`](crate::core::numdiff::FiniteDiff), which synthesizes
/// both: a finite-difference [`Hessian`] over a backend with a dense matrix
/// type (nalgebra / faer), or a finite-difference Hessian product on any
/// backend.
///
/// # References
///
/// Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.),
/// Chapter 4 (trust-region methods). Springer.
/// [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5).
///
/// Moré, J. J., & Sorensen, D. C. (1983). Computing a trust region step.
/// *SIAM Journal on Scientific and Statistical Computing*, 4(3), 553–572.
/// [doi:10.1137/0904038](https://doi.org/10.1137/0904038).
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
///
/// Matrix-free on the dependency-free `Vec<f64>` backend: the problem
/// implements [`HessianProduct`]
/// (here `∇²f = diag(1, 100)`, so `∇²f·v` is a componentwise scale) and no
/// Hessian matrix ever exists:
///
/// ```
/// use basin::{
///     BasicState, CostFunction, Executor, Gradient, GradientTolerance, HessianProduct,
///     TrustRegion,
/// };
///
/// struct IllQuadratic;
/// impl CostFunction for IllQuadratic {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
///         Ok(0.5 * (x[0] * x[0] + 100.0 * x[1] * x[1]))
///     }
/// }
/// impl Gradient for IllQuadratic {
///     type Gradient = Vec<f64>;
///     fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
///         Ok(vec![x[0], 100.0 * x[1]])
///     }
/// }
/// impl HessianProduct for IllQuadratic {
///     fn hessian_product(&self, _x: &Vec<f64>, v: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
///         Ok(vec![v[0], 100.0 * v[1]])
///     }
/// }
///
/// let result = Executor::new(
///     IllQuadratic,
///     TrustRegion::matrix_free(),
///     BasicState::new(vec![5.0, 1.0]),
/// )
/// .max_iter(100)
/// .terminate_on(GradientTolerance(1e-10))
/// .run()
/// .unwrap();
/// assert!(result.cost() < 1e-16);
/// ```
pub struct TrustRegion<Sub = Steihaug, F = f64, Mode = ExactHessian> {
    subproblem: Sub,
    /// Current trust radius `Δ`, mutated across iterations. Reset to
    /// `initial_radius` by [`Solver::init`].
    radius: F,
    initial_radius: F,
    max_radius: F,
    eta: F,
    max_inner: u32,
    mode: PhantomData<Mode>,
}

/// Marker for [`TrustRegion`]'s default mode: the full Hessian matrix `B`
/// is formed once per outer iteration via the problem's
/// [`Hessian`] impl and reused across inner
/// radius reductions.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExactHessian;

/// Marker for [`TrustRegion`]'s matrix-free mode: `B` is never formed; the
/// subproblem touches it only through the problem's
/// [`HessianProduct`] impl. See
/// [`TrustRegion::matrix_free`].
#[derive(Debug, Clone, Copy, Default)]
pub struct MatrixFree;

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

impl TrustRegion<Steihaug, f64, MatrixFree> {
    /// Matrix-free trust-region solver with the default [`Steihaug`]
    /// subproblem and the same defaults as [`new`](TrustRegion::new). The
    /// problem must implement
    /// [`HessianProduct`] instead of
    /// [`Hessian`]; no Hessian matrix is ever
    /// formed. See the type-level "Matrix-free mode" section.
    pub fn matrix_free() -> Self {
        Self::matrix_free_with(Steihaug::new())
    }
}

impl<Sub, F: Scalar> TrustRegion<Sub, F, MatrixFree> {
    /// Matrix-free trust-region solver with an explicit subproblem strategy
    /// ([`Steihaug`] or [`CauchyPoint`]; [`Dogleg`] and [`MoreSorensen`] need
    /// the actual matrix and are compile errors here).
    pub fn matrix_free_with(subproblem: Sub) -> Self {
        Self {
            subproblem,
            radius: F::one(),
            initial_radius: F::one(),
            max_radius: F::from_f64(100.0).unwrap(),
            eta: F::from_f64(0.125).unwrap(),
            max_inner: 10,
            mode: PhantomData,
        }
    }
}

impl<Sub, F: Scalar> TrustRegion<Sub, F> {
    /// Trust-region solver with an explicit subproblem strategy
    /// ([`Steihaug`], [`Dogleg`], [`MoreSorensen`], or [`CauchyPoint`]).
    pub fn with_subproblem(subproblem: Sub) -> Self {
        Self {
            subproblem,
            radius: F::one(),
            initial_radius: F::one(),
            max_radius: F::from_f64(100.0).unwrap(),
            eta: F::from_f64(0.125).unwrap(),
            max_inner: 10,
            mode: PhantomData,
        }
    }
}

impl<Sub, F: Scalar, Mode> TrustRegion<Sub, F, Mode> {
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

impl<Sub, V, F, Mode> InitialState<V> for TrustRegion<Sub, F, Mode>
where
    F: Scalar,
    V: Clone,
{
    type State = BasicState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        BasicState::new(x.clone())
    }
}

/// Shared `Solver::init` body for both modes: reset the radius and seed
/// cost + gradient so iter-0 termination checks (e.g. `GradientTolerance`
/// on a near-optimal start) see a complete state. Second-order information
/// is (re)computed per iteration in `next_iter`, so none is seeded here.
fn tr_init<P, V, F>(
    radius: &mut F,
    initial_radius: F,
    problem: &mut Problem<P>,
    mut state: BasicState<V, F>,
) -> Result<BasicState<V, F>, P::Error>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V>,
{
    // A reused solver instance must restart from the configured radius.
    *radius = initial_radius;
    let (cost, grad) = problem.cost_and_gradient(&state.param)?;
    state.cost = Some(cost);
    state.gradient = Some(grad);
    Ok(state)
}

/// Shared `Solver::next_iter` body for both modes: the accept/shrink loop
/// of N&W Algorithm 4.1. The mode-specific part — how a subproblem attempt
/// obtains `B·v` — is injected as `attempt(problem, x, g, radius)`, called
/// once per inner radius reduction at the *same* iterate `x`.
#[allow(clippy::type_complexity)]
fn tr_next_iter<P, V, F>(
    radius: &mut F,
    max_radius: F,
    eta: F,
    max_inner: u32,
    problem: &mut Problem<P>,
    mut state: BasicState<V, F>,
    mut attempt: impl FnMut(
        &mut Problem<P>,
        &V,
        &V,
        F,
    ) -> Result<Step<V, F>, P::Error>,
) -> Result<(BasicState<V, F>, Option<TerminationReason>), P::Error>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V>,
    V: Clone + ScaledAdd<F> + NormSquared<F>,
{
    let g = state
        .gradient
        .take()
        .expect("gradient not set: Solver::init must run before next_iter");
    let cost_old = state
        .cost
        .expect("cost not set: Solver::init must run before next_iter");

    let quarter = F::from_f64(0.25).unwrap();
    let three_quarters = F::from_f64(0.75).unwrap();
    let two = F::from_f64(2.0).unwrap();

    for _ in 0..max_inner {
        let step = attempt(problem, &state.param, &g, *radius)?;

        // A non-finite predicted reduction is the strategies' signal that
        // the model data itself was non-finite, so no step is trustworthy.
        // That is a failure, not the stationary point the zero-reduction
        // branch below describes.
        if !step.predicted_reduction.is_finite() {
            state.gradient = Some(g);
            return Ok((state, Some(TerminationReason::SolverFailed)));
        }

        // Predicted reduction ≤ 0 means the model cannot decrease; for
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
            *radius = quarter * step_norm;
        } else if rho > three_quarters && step.hit_boundary {
            let grown = two * *radius;
            *radius = if grown < max_radius {
                grown
            } else {
                max_radius
            };
        }

        if rho > eta {
            // Accept: move to the trial point and refresh the gradient
            // there (second-order information is recomputed by the next
            // iteration).
            state.param = trial;
            state.cost = Some(cost_trial);
            let g_new = problem.gradient(&state.param)?;
            state.gradient = Some(g_new);
            return Ok((state, None));
        }
        // Reject: radius has shrunk; retry at the same iterate.
    }

    // Inner attempts exhausted without an acceptable step. Keep the
    // shrunken radius and the current iterate; restore the gradient so
    // the state stays consistent and let the next outer iteration retry
    // (or a termination criterion fire).
    state.gradient = Some(g);
    Ok((state, None))
}

impl<P, Sub, V, M, F> Solver<P, BasicState<V, F>>
    for TrustRegion<Sub, F, ExactHessian>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>
        + Gradient<Gradient = V>
        + Hessian<Hessian = M>,
    V: Clone + ScaledAdd<F> + NormSquared<F>,
    Sub: Subproblem<V, M, F>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        state: BasicState<V, F>,
    ) -> Result<BasicState<V, F>, Self::Error> {
        tr_init(&mut self.radius, self.initial_radius, problem, state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: BasicState<V, F>,
    ) -> Result<(BasicState<V, F>, Option<TerminationReason>), Self::Error>
    {
        // One Hessian per outer iteration, reused across all inner radius
        // reductions at zero extra derivative evaluations (the gradient is
        // likewise fixed while x is).
        let b = problem.hessian(&state.param)?;
        let subproblem = &self.subproblem;
        tr_next_iter(
            &mut self.radius,
            self.max_radius,
            self.eta,
            self.max_inner,
            problem,
            state,
            |_, _, g, radius| Ok(subproblem.solve(g, &b, radius)),
        )
    }
}

impl<P, Sub, V, F> Solver<P, BasicState<V, F>>
    for TrustRegion<Sub, F, MatrixFree>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>
        + Gradient<Gradient = V>
        + HessianProduct,
    V: Clone + ScaledAdd<F> + NormSquared<F>,
    Sub: SubproblemHvp<V, F>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        state: BasicState<V, F>,
    ) -> Result<BasicState<V, F>, Self::Error> {
        tr_init(&mut self.radius, self.initial_radius, problem, state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: BasicState<V, F>,
    ) -> Result<(BasicState<V, F>, Option<TerminationReason>), Self::Error>
    {
        // No Hessian is formed: every product goes through the problem's
        // counted `hessian_product`. Unlike exact mode, an inner radius
        // reduction re-pays its CG products at the same iterate (the
        // shrunken radius truncates CG earlier, so re-attempts get
        // cheaper); `hessian_product_evals` makes that cost visible.
        let subproblem = &self.subproblem;
        tr_next_iter(
            &mut self.radius,
            self.max_radius,
            self.eta,
            self.max_inner,
            problem,
            state,
            |problem, x, g, radius| {
                subproblem
                    .solve_hvp(g, radius, |v| problem.hessian_product(x, v))
            },
        )
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

    /// A quadratic whose gradient is poisoned with a NaN: the subproblem
    /// cannot produce a trustworthy step from it.
    struct NonFiniteGradient;

    impl CostFunction for NonFiniteGradient {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
            Ok(0.5 * (x[0] * x[0] + x[1] * x[1]))
        }
    }
    impl Gradient for NonFiniteGradient {
        type Gradient = Vec<f64>;
        fn gradient(&self, _x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            Ok(vec![f64::NAN, 1.0])
        }
    }
    impl Hessian for NonFiniteGradient {
        type Hessian = crate::core::math::DenseMatrix<f64>;
        fn hessian(&self, _x: &Vec<f64>) -> Result<Self::Hessian, Self::Error> {
            Ok(crate::core::math::DenseMatrix::from_row_slice(
                2,
                2,
                &[1.0, 0.0, 0.0, 1.0],
            ))
        }
    }

    #[test]
    fn nonfinite_gradient_is_reported_as_a_failure_not_convergence() {
        let result = Executor::new(
            NonFiniteGradient,
            TrustRegion::with_subproblem(MoreSorensen::new()),
            BasicState::new(vec![3.0, 3.0]),
        )
        .max_iter(10)
        .run()
        .unwrap();

        assert_eq!(result.reason, TerminationReason::SolverFailed);
        assert!(result.reason.is_failure());
    }

    #[test]
    fn more_sorensen_minimizes_quadratic() {
        let result = Executor::new(
            Quadratic,
            TrustRegion::with_subproblem(MoreSorensen::new()),
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

    #[test]
    fn more_sorensen_minimizes_rosenbrock() {
        let result = Executor::new(
            Rosenbrock,
            TrustRegion::with_subproblem(MoreSorensen::new()),
            BasicState::new(vec![-1.2, 1.0]),
        )
        .max_iter(200)
        .terminate_on(GradientTolerance(1e-8))
        .run()
        .unwrap();
        assert!(result.cost() < 1e-10, "cost = {}", result.cost());
    }

    /// The quadratic again, but exposing only a Hessian-vector product and
    /// deliberately *not* implementing `Hessian`: that these tests compile
    /// is the proof that matrix-free mode never forms (or names) a matrix.
    struct QuadraticHvOnly;

    impl CostFunction for QuadraticHvOnly {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
            Ok(0.5 * (x[0] * x[0] + 100.0 * x[1] * x[1]))
        }
    }
    impl Gradient for QuadraticHvOnly {
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            Ok(vec![x[0], 100.0 * x[1]])
        }
    }
    impl HessianProduct for QuadraticHvOnly {
        fn hessian_product(
            &self,
            _x: &Vec<f64>,
            v: &Vec<f64>,
        ) -> Result<Vec<f64>, Self::Error> {
            Ok(vec![v[0], 100.0 * v[1]])
        }
    }

    /// Rosenbrock with only an analytic Hessian-vector product (no
    /// `Hessian` impl).
    struct RosenbrockHvOnly;

    impl CostFunction for RosenbrockHvOnly {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
            Rosenbrock.cost(x)
        }
    }
    impl Gradient for RosenbrockHvOnly {
        type Gradient = Vec<f64>;
        fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
            Rosenbrock.gradient(x)
        }
    }
    impl HessianProduct for RosenbrockHvOnly {
        fn hessian_product(
            &self,
            x: &Vec<f64>,
            v: &Vec<f64>,
        ) -> Result<Vec<f64>, Self::Error> {
            let h11 = 2.0 + 1200.0 * x[0] * x[0] - 400.0 * x[1];
            let h12 = -400.0 * x[0];
            Ok(vec![h11 * v[0] + h12 * v[1], h12 * v[0] + 200.0 * v[1]])
        }
    }

    #[test]
    fn matrix_free_steihaug_minimizes_quadratic() {
        let result = Executor::new(
            QuadraticHvOnly,
            TrustRegion::matrix_free(),
            BasicState::new(vec![5.0, 1.0]),
        )
        .max_iter(100)
        .terminate_on(GradientTolerance(1e-10))
        .run()
        .unwrap();
        assert!(result.cost() < 1e-16, "cost = {}", result.cost());
    }

    #[test]
    fn matrix_free_steihaug_minimizes_rosenbrock() {
        let result = Executor::new(
            RosenbrockHvOnly,
            TrustRegion::matrix_free(),
            BasicState::new(vec![-1.2, 1.0]),
        )
        .max_iter(200)
        .terminate_on(GradientTolerance(1e-8))
        .run()
        .unwrap();
        assert!(result.cost() < 1e-10, "cost = {}", result.cost());
    }

    #[test]
    fn matrix_free_cauchy_point_minimizes_quadratic() {
        let result = Executor::new(
            QuadraticHvOnly,
            TrustRegion::matrix_free_with(CauchyPoint),
            BasicState::new(vec![5.0, 1.0]),
        )
        .max_iter(500)
        .terminate_on(GradientTolerance(1e-8))
        .run()
        .unwrap();
        assert!(result.cost() < 1e-8, "cost = {}", result.cost());
    }

    #[test]
    fn matrix_free_matches_exact_steihaug() {
        // The two modes run the identical CG arithmetic (one via `MatVec` on
        // the formed matrix, one via the analytic product), so on the same
        // problem they must land on the same iterate in the same number of
        // iterations, up to product-vs-matvec rounding.
        let exact = Executor::new(
            Quadratic,
            TrustRegion::new(),
            BasicState::new(vec![5.0, 1.0]),
        )
        .max_iter(100)
        .terminate_on(GradientTolerance(1e-10))
        .run()
        .unwrap();
        let free = Executor::new(
            QuadraticHvOnly,
            TrustRegion::matrix_free(),
            BasicState::new(vec![5.0, 1.0]),
        )
        .max_iter(100)
        .terminate_on(GradientTolerance(1e-10))
        .run()
        .unwrap();
        assert_eq!(exact.state.iter, free.state.iter);
        assert!((exact.cost() - free.cost()).abs() < 1e-15);
    }

    #[test]
    fn matrix_free_counts_products_not_hessians() {
        use crate::core::solver::Solver as _;

        let mut problem = Problem::new(RosenbrockHvOnly);
        let mut solver = TrustRegion::matrix_free();
        let mut state = solver
            .init(&mut problem, BasicState::new(vec![-1.2, 1.0]))
            .unwrap();
        for _ in 0..3 {
            let (next, _) = solver.next_iter(&mut problem, state).unwrap();
            state = next;
        }
        let counts = problem.counts();
        assert!(counts.hessian_product_evals > 0);
        assert_eq!(counts.hessian_evals, 0);

        // The state mirror folds products into the gradient slot.
        use crate::core::state::CountsMirror as _;
        state.mirror(counts);
        assert!(state.gradient_evals >= counts.hessian_product_evals);
    }
}
