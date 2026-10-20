use core::marker::PhantomData;

/// Opt-in native simplex trial records.
pub mod diagnostics;
use diagnostics::{Trace, Trial, TrialKind};

use crate::core::constraint::evaluation::BoxProblem;
use crate::core::convergence::{Check, SimplexChecks, optional_tolerance};
use crate::core::math::{
    ClampInPlace, Scalar, ScaleInPlace, ScaledAdd, VectorIndex, VectorLen,
};
use crate::core::problem::{CostFunction, Problem};
use crate::core::solver::Solver;
use crate::core::state::SimplexProgress;

/// Nelder-Mead simplex method (derivative-free).
///
/// Implements the algorithm as stated in Lagarias et al. (1998) with the
/// adaptive parameter option of Gao & Han (2012). The four parameters are:
/// `α` (reflection), `β` (expansion), `γ` (contraction), `δ` (shrink), with
/// the constraints `α > 0`, `β > 1`, `0 < γ < 1`, `0 < δ < 1`.
///
/// # Convergence
///
/// By default, both unbounded and projected variants stop when the whole
/// simplex is small in parameters AND costs. Let `b` be its current best vertex,
/// `D_x` the maximum `||x_i - b||_infinity`, and `D_f` the maximum `|f_i - f_b|`. The tests
/// are `D_x <= max(x_abs, x_rel * ||b||_infinity)` and
/// `D_f <= max(f_abs, f_rel * max(|f_b|, initial_cost_spread))`.
/// Both absolute tolerances default to `1e-8`; both relative tolerances default
/// to `sqrt(f64::EPSILON)` for `f64` (about `1.5e-8`) and
/// `32 * f32::EPSILON` for `f32` (about `3.8e-6`).
/// The initial cost spread is captured after projection and evaluation, or at
/// the first subsequent boundary with a finite spread. It keeps the relative
/// cost test useful near a zero minimum under large objective multipliers.
/// Fresh solves reset this reference; exact checkpoints preserve it.
///
/// The size and cost-spread tolerance setters replace individual settings.
/// `None` disables a setting; zero requests an exact-zero threshold. Disabled
/// terms are omitted from each maximum, and a dimension with both settings
/// disabled is omitted from the AND group. Disable all four settings to turn
/// off this group. Other configured tests combine with this group using OR.
/// Nonfinite measurements or overflowed bounds cannot establish
/// convergence. Checks require no additional evaluations or vector clones.
///
/// The optional [`with_absolute_simplex_cost_std_dev_tolerance`](Self::with_absolute_simplex_cost_std_dev_tolerance)
/// check stops when the sample standard deviation of the current `m` simplex
/// costs is at most its tolerance:
/// `s = sqrt(sum_i (f_i - mean(f))^2 / (m - 1))`, with `m >= 2`.
/// It is disabled by default and combines with the paired simplex checks and
/// other checks using OR. When both simplex predicates pass, the report retains
/// both. It uses only costs, independently of simplex size.
/// NaN or infinite costs, fewer than two costs, or an unrepresentable standard
/// deviation cannot pass this check. Finite costs are centered and scaled to
/// avoid intermediate overflow and squared-deviation underflow. Zero tolerance
/// requires identical finite costs. Like cost spread alone, a small standard
/// deviation does not establish parameter accuracy or stationarity.
///
/// An unchanged best vertex alone does not establish collapse. Flat objectives
/// and simplexes collapsed by clipping can pass these tests without establishing
/// stationarity or parameter accuracy. Fixed absolute floors limit accuracy for
/// small quantities, and one vector norm cannot account for different units
/// across coordinates. Scale parameters and tune the floors for the problem;
/// disable relative settings when configuring absolute-only tests. Keep an
/// iteration or evaluation budget on the executor as well.
///
/// # Bounds
///
/// `NelderMead` is generic over a type-state [`Mode`](Unbounded) marker
/// that switches between the unconstrained algorithm ([`Unbounded`], the
/// default) and the projection-style box-constrained variant
/// ([`Projected`]). Construct unbounded NM with [`new`](Self::new),
/// [`adaptive`](Self::adaptive), or [`with_params`](Self::with_params), then
/// transition with [`projected`](Self::projected) when the problem carries
/// box bounds. The projected `Solver` impl requires `P::Set = ConstraintSet<BoxBounds<V>>`
/// and `V: ClampInPlace`, so handing a non-bounded problem to a projected
/// `NelderMead` is a compile-time error per CONTRIBUTING.md tenet 4.
///
/// # Progress and lifecycle
///
/// [`SimplexProgress<V, F>`] owns the observable vertices and matching costs,
/// while this solver owns the three scratch vectors and resolved coefficients.
/// Progress preserves all six evaluation categories and an independent
/// historical incumbent. Observe the simplex through `vertices()`, `costs()`,
/// or the checked `evaluated_vertices()` reader without copying it.
///
/// Fresh initialization clears counters and incumbent metadata, rebuilds
/// scratch for the current dimension, resolves adaptive coefficients, and
/// reevaluates every supplied vertex. Use `Executor::from_start` to build the
/// default simplex around a point, or pass explicit `SimplexProgress` vertices
/// to `Executor::new`. Exact continuation uses a solver-aware checkpoint and
/// skips initialization. With `serde`, the solver and progress can be stored
/// together whenever the vector type supports serialization. Old
/// `BasicSimplexState` payloads, which included scratch, are incompatible.
///
/// # Backends
///
/// Supports `Vec<F>`, nalgebra `DVector<F>`, ndarray `Array1<F>`, and faer
/// `Col<F>`, with `F = f32` or `f64`. Requires [`Clone`], [`VectorLen`],
/// [`VectorIndex`], [`ScaleInPlace`], and [`ScaledAdd`]. The projected variant
/// also requires [`ClampInPlace`]. Default point seeding uses
/// [`IntoInitialSimplex`](crate::IntoInitialSimplex), implemented for vectors
/// with coordinate access. Explicit simplex seeding does not need this
/// conversion trait, but convergence still requires coordinate access.
///
/// # Examples
///
/// Derivative-free minimization of Rosenbrock: Nelder–Mead needs only
/// [`CostFunction`] and iterates a [`SimplexProgress`] seeded from a
/// single point (the initial simplex is built automatically):
///
/// ```
/// use basin::{SimplexProgress, CostFunction, Executor, NelderMead};
///
/// struct Rosenbrock;
/// impl CostFunction for Rosenbrock {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
///         Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2))
///     }
/// }
///
/// let result = Executor::new(
///     Rosenbrock,
///     NelderMead::new(),
///     SimplexProgress::new(vec![-1.2, 1.0]),
/// )
/// .max_iter(1_000)
/// .run()
/// .unwrap();
/// assert!(result.cost() < 1e-6);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NelderMead<V, F: Scalar = f64, Mode = Unbounded> {
    scratch: Vec<V>,
    #[cfg_attr(feature = "serde", serde(skip, default = "Trace::default"))]
    trace: Trace<V, F>,
    config: ParamConfig<F>,
    convergence: SimplexChecks<F>,
    /// Resolved parameters; populated by `init` once the dimension is known.
    params: Option<Params<F>>,
    /// Type-state marker; carries the mode at the type level only.
    _mode: PhantomData<fn() -> Mode>,
}

/// Type-state marker for unconstrained Nelder-Mead (the default).
/// Constructors live on `NelderMead<V, F, Unbounded>`; the `Solver` impl
/// makes no constraint requirements on the problem.
pub struct Unbounded;

/// Type-state marker for the projection-style box-constrained
/// Nelder-Mead variant. Obtain via
/// [`NelderMead::projected`](NelderMead::projected). The `Solver` impl
/// requires `P::Set = ConstraintSet<BoxBounds<V>>` and `V: ClampInPlace`.
///
/// # Algorithm
///
/// Standard Nelder-Mead with an element-wise clamp into `[lower, upper]`
/// applied to every trial vertex (reflection, expansion, both
/// contractions, and each shrunk vertex) before the cost evaluation.
/// This is the same approach scipy uses for
/// `scipy.optimize.minimize(method='Nelder-Mead', bounds=...)`.
///
/// At [`init`](Solver::init) every vertex of the initial simplex is
/// projected once, so an infeasible starting simplex is silently
/// corrected (and downstream termination criteria see a feasible
/// simplex at iter 0). Subsequent iterations preserve feasibility by
/// construction.
///
/// # Known limitation
///
/// The simple projection approach can stall when many vertices collapse
/// onto the same boundary face: the simplex becomes degenerate and the
/// reflection step loses descent direction. This is a known weakness of
/// the projection variant; scipy ships it anyway because it works well
/// enough in practice. This type intentionally remains the clamp-only local
/// method. Use [`Gbnm`](crate::Gbnm) for Luersen and Le Riche's full
/// Globalized Bounded Nelder-Mead algorithm, including probabilistic,
/// small-test, and large-test restarts.
pub struct Projected;

#[derive(Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Params<F> {
    alpha: F,
    beta: F,
    gamma: F,
    delta: F,
}

#[derive(Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum ParamConfig<F> {
    Standard,
    Adaptive,
    Fixed(Params<F>),
}

impl<V, F: Scalar> Default for NelderMead<V, F, Unbounded> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V, F: Scalar> NelderMead<V, F, Unbounded> {
    /// Nelder-Mead with the standard parameters (Nelder & Mead 1965):
    /// α=1, β=2, γ=0.5, δ=0.5. These coefficients *are* the default, so
    /// this is the canonical entry point; [`adaptive`](Self::adaptive)
    /// and [`with_params`](Self::with_params) are presets and overrides.
    /// Enables paired absolute and relative simplex tests; see
    /// [Convergence](Self#convergence).
    pub fn new() -> Self {
        Self {
            config: ParamConfig::Standard,
            convergence: SimplexChecks::nelder_mead_defaults(),
            params: None,
            scratch: Vec::new(),
            trace: Trace {
                enabled: false,
                sequence: 0,
                trials: Vec::new(),
            },
            _mode: PhantomData,
        }
    }

    /// Adaptive parameters from Gao & Han (2012), eq. (4.1):
    /// α=1, β=1+2/n, γ=0.75−1/(2n), δ=1−1/n, with `n` inferred from the
    /// initial simplex during `Solver::init`. Coincides with the standard
    /// parameters ([`new`](Self::new)) when `n == 2`.
    pub fn adaptive() -> Self {
        Self {
            config: ParamConfig::Adaptive,
            convergence: SimplexChecks::nelder_mead_defaults(),
            params: None,
            scratch: Vec::new(),
            trace: Trace {
                enabled: false,
                sequence: 0,
                trials: Vec::new(),
            },
            _mode: PhantomData,
        }
    }

    /// Nelder-Mead with explicit reflection/expansion/contraction/
    /// shrink coefficients (`α`, `β`, `γ`, `δ`). Panics if any coefficient
    /// is outside its admissible range.
    pub fn with_params(alpha: F, beta: F, gamma: F, delta: F) -> Self {
        assert!(alpha > F::zero(), "α must be > 0");
        assert!(beta > F::one(), "β must be > 1");
        assert!(gamma > F::zero() && gamma < F::one(), "γ must be in (0, 1)");
        assert!(delta > F::zero() && delta < F::one(), "δ must be in (0, 1)");
        Self {
            config: ParamConfig::Fixed(Params {
                alpha,
                beta,
                gamma,
                delta,
            }),
            convergence: SimplexChecks::nelder_mead_defaults(),
            params: None,
            scratch: Vec::new(),
            trace: Trace {
                enabled: false,
                sequence: 0,
                trials: Vec::new(),
            },
            _mode: PhantomData,
        }
    }

    /// Switch to the projection-style box-constrained variant
    /// ([`Projected`]). The algorithm parameters configured on this
    /// builder are preserved; the resulting solver requires the problem
    /// to declare a box-only set using [`BoxBounds`](crate::BoxBounds) and projects every trial vertex
    /// element-wise into `[lower, upper]`. See the type-level rustdoc on
    /// [`Projected`] for the algorithm contract and limitations.
    pub fn projected(self) -> NelderMead<V, F, Projected> {
        NelderMead {
            config: self.config,
            convergence: self.convergence,
            params: self.params,
            scratch: self.scratch,
            trace: self.trace,
            _mode: PhantomData,
        }
    }
}

impl<V, Mode, F: Scalar> NelderMead<V, F, Mode> {
    /// Set the absolute floor on maximum infinity-norm distance from the best vertex.
    ///
    /// Defaults to `1e-8`. Combines with the relative size tolerance using OR;
    /// enabled size and cost conditions combine with AND. `None` disables only
    /// this setting, and zero requests an exact-zero threshold. Values must be
    /// finite and nonnegative. See [Convergence](Self#convergence).
    pub fn with_absolute_simplex_size_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.convergence.size = optional_tolerance(value);
        self
    }

    /// Set the relative tolerance on maximum infinity-norm distance from the best vertex.
    ///
    /// Defaults to `sqrt(f64::EPSILON)` for `f64` and `32 * f32::EPSILON` for
    /// `f32`, multiplied by the current best vertex's infinity norm.
    /// Combines with the absolute size tolerance using OR;
    /// enabled size and cost conditions combine with AND. `None` disables only
    /// this setting, and zero requests an exact-zero threshold. Values must be
    /// finite and nonnegative. See [Convergence](Self#convergence).
    pub fn with_relative_simplex_size_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.convergence.relative_size = optional_tolerance(value);
        self
    }

    /// Set the absolute floor on maximum cost difference from the best vertex.
    ///
    /// Defaults to `1e-8`. Combines with the relative cost tolerance using OR;
    /// enabled size and cost conditions combine with AND. `None` disables only
    /// this setting, and zero requests an exact-zero threshold. Values must be
    /// finite and nonnegative. See [Convergence](Self#convergence).
    pub fn with_absolute_simplex_cost_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.convergence.cost = optional_tolerance(value);
        self
    }

    /// Set the relative tolerance on maximum cost difference from the best vertex.
    ///
    /// Defaults to `sqrt(f64::EPSILON)` for `f64` and `32 * f32::EPSILON` for
    /// `f32`, multiplied by the larger of the current best cost magnitude and
    /// the initial cost spread. The saved initial spread
    /// survives setter calls and exact continuation. Combines with the absolute
    /// cost tolerance using OR; enabled size and cost conditions combine with
    /// AND. `None` disables only this setting, and zero requests an exact-zero
    /// threshold. Values must be finite and nonnegative. See
    /// [Convergence](Self#convergence).
    pub fn with_relative_simplex_cost_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.convergence.relative_cost = optional_tolerance(value);
        self
    }

    /// Stop when the sample standard deviation of simplex costs is at most the tolerance.
    ///
    /// Disabled by default. For `m >= 2` costs, this tests
    /// `sqrt(sum_i (f_i - mean(f))^2 / (m - 1)) <= tolerance`.
    /// Combines with the paired simplex checks and other checks using OR,
    /// without changing their settings. To use only this check, disable all
    /// four size and cost-spread tolerances. The check runs at initialized
    /// iteration boundaries without evaluations or vector clones.
    /// It uses only the current costs and has no history to reset.
    ///
    /// `None` disables this setting; zero requires identical finite costs.
    /// NaN or infinite costs and an unrepresentable standard deviation cannot
    /// establish convergence. See [Convergence](Self#convergence).
    ///
    /// # Panics
    ///
    /// Panics if the tolerance is negative or nonfinite.
    pub fn with_absolute_simplex_cost_std_dev_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.convergence.cost_std_dev = optional_tolerance(value);
        self
    }

    /// Record the most recent iteration's native trial attempts. Disabled by default.
    /// Fresh initialization resets ordinals; exact continuation retains them.
    /// Execution controls can leave an earlier batch intact. Getters make no callbacks.
    /// Diagnostic settings and records are omitted from serde checkpoints.
    pub fn with_trial_diagnostics(mut self, enabled: bool) -> Self {
        self.trace.enabled = enabled;
        self.trace.trials.clear();
        self
    }

    /// Native attempts from the most recent iteration, including callback errors.
    pub fn trial_observations(&self) -> &[Trial<V, F>] {
        &self.trace.trials
    }

    fn resolve(config: ParamConfig<F>, n: usize) -> Params<F> {
        assert!(n >= 1, "NelderMead requires at least a 1-D problem");
        match config {
            ParamConfig::Standard => {
                let half = F::from_f64(0.5).unwrap();
                Params {
                    alpha: F::one(),
                    beta: F::from_f64(2.0).unwrap(),
                    gamma: half,
                    delta: half,
                }
            }
            ParamConfig::Adaptive => {
                let n = F::from_usize(n).unwrap();
                let two = F::from_f64(2.0).unwrap();
                Params {
                    alpha: F::one(),
                    beta: F::one() + two / n,
                    gamma: F::from_f64(0.75).unwrap() - F::one() / (two * n),
                    delta: F::one() - F::one() / n,
                }
            }
            ParamConfig::Fixed(p) => p,
        }
    }
}

/// Write `(1 − t)·a + t·b` into `out`, in place. Works for any
/// `t ∈ ℝ`; values outside `[0, 1]` extrapolate, which is what
/// reflection needs. `out`'s previous contents are overwritten; the
/// caller must guarantee it has the same shape as `a`/`b` (the
/// solver enforces this by pre-allocating scratch in `init`).
fn affine_into<V, F>(out: &mut V, a: &V, b: &V, t: F)
where
    V: Clone + ScaleInPlace<F> + ScaledAdd<F>,
    F: Scalar,
{
    out.scale_in_place(F::zero());
    out.scaled_add(F::one() - t, a);
    out.scaled_add(t, b);
}

/// Write the mean of `vertices` into `out`, in place. `out`'s previous
/// contents are overwritten.
fn centroid_into<V, F>(out: &mut V, vertices: &[V])
where
    V: Clone + ScaleInPlace<F> + ScaledAdd<F>,
    F: Scalar,
{
    let inv = F::from_usize(vertices.len()).unwrap().recip();
    out.scale_in_place(F::zero());
    for v in vertices {
        out.scaled_add(inv, v);
    }
}

/// Evaluate every vertex's cost and sort the simplex ascending. Shared
/// between the `Unbounded` and `Projected` `Solver::init` paths after
/// any projection of the initial vertices.
fn init_costs_and_sort<P, V, F>(
    problem: &mut Problem<P>,
    state: &mut SimplexProgress<V, F>,
) -> Result<(), P::Error>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>,
{
    for v in &state.vertices {
        state.costs.push(problem.cost(v)?);
    }
    state.sort();
    Ok(())
}

/// One Nelder-Mead iteration, parameterised by a projection closure.
///
/// The `Unbounded` `Solver` impl passes a no-op closure; the `Projected`
/// impl passes one that clamps into `[lower, upper]`. Vertices are
/// sorted (best at index 0) on entry; the invariant is restored before
/// returning. The simplex has `n + 1` vertices in `n`-D.
///
/// The three trial points (centroid, reflection, expansion-or-contraction)
/// are computed in solver-owned scratch slots `[0]`, `[1]`, `[2]` so the hot
/// path performs no heap allocation. `Solver::init` pre-sizes the
/// scratch.
#[allow(clippy::type_complexity)]
fn next_iter_inner<P, V, F, Proj>(
    problem: &mut Problem<P>,
    mut state: SimplexProgress<V, F>,
    p: Params<F>,
    scratch: &mut [V],
    trace: &mut Trace<V, F>,
    project: &Proj,
) -> Result<crate::SolverStep<SimplexProgress<V, F>>, P::Error>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>,
    V: Clone + ScaleInPlace<F> + ScaledAdd<F>,
    Proj: Fn(&P, &mut V),
{
    trace.trials.clear();
    let m = state.vertices.len();
    let n = m - 1;
    let worst = m - 1;

    let f1 = state.costs[0];
    let fn_ = state.costs[n - 1];
    let fnp1 = state.costs[worst];

    // Split scratch into three independent mutable slots: x_bar, x_r,
    // and x_alt (expansion or contraction, depending on branch). Then
    // borrow the worst vertex immutably for the affine builds.
    let (xbar_slice, rest) = scratch.split_at_mut(1);
    let (xr_slice, alt_slice) = rest.split_at_mut(1);
    let x_bar = &mut xbar_slice[0];
    let x_r = &mut xr_slice[0];
    let x_alt = &mut alt_slice[0];

    centroid_into(x_bar, &state.vertices[..n]);

    // Reflection: x_r = x_bar + α(x_bar − x_{n+1}) = (1+α)·x_bar − α·x_{n+1}
    affine_into(x_r, x_bar, &state.vertices[worst], -p.alpha);
    project(problem.inner(), x_r);
    let fr = trace.evaluate(problem, x_r, TrialKind::Reflection)?;

    if f1 <= fr && fr < fn_ {
        trace.accept(0, true);
        // Accept reflection.
        std::mem::swap(&mut state.vertices[worst], x_r);
        state.costs[worst] = fr;
    } else if fr < f1 {
        // Try expansion: x_e = x_bar + β(x_r − x_bar).
        affine_into(x_alt, x_bar, x_r, p.beta);
        project(problem.inner(), x_alt);
        let fe = trace.evaluate(problem, x_alt, TrialKind::Expansion)?;
        let expanded = fe < fr;
        trace.accept(0, !expanded);
        trace.accept(1, expanded);
        if fe < fr {
            std::mem::swap(&mut state.vertices[worst], x_alt);
            state.costs[worst] = fe;
        } else {
            std::mem::swap(&mut state.vertices[worst], x_r);
            state.costs[worst] = fr;
        }
    } else if fr < fnp1 {
        // fn ≤ fr < f_{n+1}: outside contraction.
        // x_oc = x_bar + γ(x_r − x_bar).
        affine_into(x_alt, x_bar, x_r, p.gamma);
        project(problem.inner(), x_alt);
        trace.accept(0, false);
        let foc =
            trace.evaluate(problem, x_alt, TrialKind::OutsideContraction)?;
        trace.accept(1, foc <= fr);
        if foc <= fr {
            std::mem::swap(&mut state.vertices[worst], x_alt);
            state.costs[worst] = foc;
        } else {
            shrink_inner(problem, &mut state, p.delta, project, trace)?;
        }
    } else {
        // fr ≥ f_{n+1}: inside contraction.
        // x_ic = x_bar − γ(x_bar − x_{n+1}) = (1−γ)·x_bar + γ·x_{n+1}.
        affine_into(x_alt, x_bar, &state.vertices[worst], p.gamma);
        project(problem.inner(), x_alt);
        trace.accept(0, false);
        let fic =
            trace.evaluate(problem, x_alt, TrialKind::InsideContraction)?;
        trace.accept(1, fic < fnp1);
        if fic < fnp1 {
            std::mem::swap(&mut state.vertices[worst], x_alt);
            state.costs[worst] = fic;
        } else {
            shrink_inner(problem, &mut state, p.delta, project, trace)?;
        }
    }

    state.sort();
    Ok(crate::SolverStep::from((state, None)))
}

fn shrink_inner<P, V, F, Proj>(
    problem: &mut Problem<P>,
    state: &mut SimplexProgress<V, F>,
    delta: F,
    project: &Proj,
    trace: &mut Trace<V, F>,
) -> Result<(), P::Error>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>,
    V: Clone + ScaleInPlace<F> + ScaledAdd<F>,
    Proj: Fn(&P, &mut V),
{
    // Best vertex is fixed at index 0; shrink every other vertex toward
    // it in place: v ← best + δ·(v − best) = (1 − δ)·best + δ·v.
    // Split-borrow lets us read x[0] while mutating x[i], and the
    // affine write goes directly into the vertex slot, with no scratch
    // alloc per shrunk vertex.
    let one = F::one();
    let (best_slice, rest) = state.vertices.split_at_mut(1);
    let best = &best_slice[0];
    for (i, (v, c)) in rest.iter_mut().zip(&mut state.costs[1..]).enumerate() {
        // Multiply v by δ in place, then add (1−δ)·best; finally project.
        v.scale_in_place(delta);
        v.scaled_add(one - delta, best);
        project(problem.inner(), v);
        *c = trace.evaluate(problem, v, TrialKind::Shrink(i + 1))?;
        trace.accept(trace.trials.len().saturating_sub(1), true);
    }
    Ok(())
}

impl<P, V, F> Solver<P, SimplexProgress<V, F>> for NelderMead<V, F, Unbounded>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>,
    V: Clone + VectorLen + VectorIndex<F> + ScaleInPlace<F> + ScaledAdd<F>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: SimplexProgress<V, F>,
    ) -> Result<SimplexProgress<V, F>, Self::Error> {
        state.reset();
        self.convergence.reset();
        self.trace.trials.clear();
        self.trace.sequence = 0;
        let n = state.vertices.len() - 1;
        self.params = Some(Self::resolve(self.config, n));
        self.scratch = vec![state.vertices[0].clone(); 3];
        init_costs_and_sort(problem, &mut state)?;
        self.convergence.capture_initial_cost_spread(state.costs());
        Ok(state)
    }

    fn reset_convergence(&mut self) {
        self.convergence.reset();
    }

    fn check_convergence(
        &mut self,
        problem: &Problem<P>,
        state: &SimplexProgress<V, F>,
    ) -> Option<crate::Termination<F>> {
        self.convergence.check(problem, state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: SimplexProgress<V, F>,
    ) -> Result<crate::SolverStep<SimplexProgress<V, F>>, Self::Error> {
        let p = self
            .params
            .expect("NelderMead::init must run before next_iter");
        next_iter_inner(
            problem,
            state,
            p,
            &mut self.scratch,
            &mut self.trace,
            &|_: &P, _: &mut V| {},
        )
    }
}

impl<P, V, F> Solver<P, SimplexProgress<V, F>> for NelderMead<V, F, Projected>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + BoxProblem<Param = V>,
    V: Clone
        + VectorLen
        + VectorIndex<F>
        + ScaleInPlace<F>
        + ScaledAdd<F>
        + ClampInPlace,
{
    type Error = crate::ConstraintSolveError<P::Error>;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: SimplexProgress<V, F>,
    ) -> Result<SimplexProgress<V, F>, Self::Error> {
        crate::core::constraint::evaluation::validate_coordinate_box_problem(
            problem.inner(),
            state.vertices[0].vec_len(),
            false,
        )
        .map_err(crate::core::constraint::evaluation::invalid)?;
        state.reset();
        self.convergence.reset();
        self.trace.trials.clear();
        self.trace.sequence = 0;
        let n = state.vertices.len() - 1;
        self.params = Some(Self::resolve(self.config, n));
        // Project every initial vertex once so iter-0 termination
        // checks see a feasible simplex (mirrors
        // ProjectedGradientDescent::init's project-an-infeasible-start
        // pattern).
        for v in state.vertices.iter_mut() {
            v.clamp_in_place(problem.inner().lower(), problem.inner().upper());
        }
        self.scratch = vec![state.vertices[0].clone(); 3];
        init_costs_and_sort(problem, &mut state)?;
        self.convergence.capture_initial_cost_spread(state.costs());
        Ok(state)
    }

    fn reset_convergence(&mut self) {
        self.convergence.reset();
    }

    fn check_convergence(
        &mut self,
        problem: &Problem<P>,
        state: &SimplexProgress<V, F>,
    ) -> Option<crate::Termination<F>> {
        self.convergence.check(problem, state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: SimplexProgress<V, F>,
    ) -> Result<crate::SolverStep<SimplexProgress<V, F>>, Self::Error> {
        let p = self
            .params
            .expect("NelderMead::init must run before next_iter");
        // Borrow the problem only for projection. The borrow ends before
        // the counted cost call, so no bound-vector copies are needed.
        next_iter_inner(
            problem,
            state,
            p,
            &mut self.scratch,
            &mut self.trace,
            &|problem: &P, v: &mut V| {
                v.clamp_in_place(problem.lower(), problem.upper())
            },
        )
        .map_err(Into::into)
    }
}
