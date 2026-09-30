// Index-based loops mirror Powell's exposition (BOBYQA, Powell 2009) and the
// dense index arithmetic of the H-factorization algebra; blanket-allowed here.
#![allow(clippy::needless_range_loop)]

//! BOBYQA (Powell 2009): bound-constrained model-based derivative-free
//! optimization.
//!
//! BOBYQA reuses the shared Powell quadratic-model core (the crate-internal
//! `powell` module, shared with NEWUOA) and swaps NEWUOA's TRSAPP for the
//! box-aware TRSBOX subproblem (§3),
//! ALTMOV geometry (§3), and the RESCUE restoration procedure (§5). It binds
//! [`CostFunction`](crate::core::problem::CostFunction) +
//! [`BoxConstraints`](crate::core::constraint::BoxConstraints).

pub(crate) mod driver;
pub(crate) mod geometry;
pub(crate) mod init;
pub(crate) mod rescue;
pub(crate) mod trsbox;

#[cfg(test)]
mod parity;

use std::marker::PhantomData;

use crate::core::constraint::BoxConstraints;
use crate::core::inner::InitialState;
use crate::core::math::{Scalar, VectorLen};
use crate::core::problem::{CostFunction, Problem};
use crate::core::solver::Solver;
use crate::core::state::{PointState, State};
use crate::core::termination::TerminationReason;

use driver::{BobyqaWork, Transition};

/// Type-state marker: box-constrained BOBYQA (the only mode). A marker keeps the
/// door open for an unconstrained alias later without an API break, matching the
/// `Lbfgs<V, F, Bounded>`/`NelderMead<V, F, Projected>` precedent.
pub struct Bounded;

/// BOBYQA (Powell 2009): bound-constrained model-based derivative-free
/// trust-region optimization.
///
/// Configure the trust-region radii and interpolation-set size, then drive it
/// with an [`Executor`](crate::Executor) over a [`PointState`] on a problem
/// that implements [`CostFunction`] + [`BoxConstraints`]:
///
/// ```
/// use basin::{Bobyqa, PointState, BoxConstraints, CostFunction, Executor};
///
/// struct Booth {
///     lower: Vec<f64>,
///     upper: Vec<f64>,
/// }
/// impl CostFunction for Booth {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
///         Ok((x[0] + 2.0 * x[1] - 7.0).powi(2)
///             + (2.0 * x[0] + x[1] - 5.0).powi(2))
///     }
/// }
/// impl BoxConstraints for Booth {
///     fn lower(&self) -> &Vec<f64> {
///         &self.lower
///     }
///     fn upper(&self) -> &Vec<f64> {
///         &self.upper
///     }
/// }
///
/// let problem = Booth {
///     lower: vec![-5.0, -5.0],
///     upper: vec![5.0, 5.0],
/// };
/// let solver = Bobyqa::new()
///     .with_initial_radius(0.5)
///     .with_final_radius(1e-8);
/// let state = PointState::new(vec![0.0, 0.0]);
/// let result = Executor::new(problem, solver, state)
///     .max_cost_evals(500)
///     .run()
///     .unwrap();
/// assert!(result.best_param()[0].is_finite());
/// ```
///
/// # Configuration
///
/// - [`with_initial_radius`](Self::with_initial_radius): initial trust-region radius `ρ_beg`
///   (default `1.0`). Also the initial `Δ`. The initial sampling needs room
///   `b_i ≥ a_i + 2ρ_beg` (Powell 2009, eq. 2.1); if `ρ_beg` is too large for a
///   narrow box it is reduced to `min(b_i − a_i)/4` (with `ρ_end` pulled down to
///   match), mirroring PRIMA rather than rejecting the solve.
/// - [`with_final_radius`](Self::with_final_radius): final radius `ρ_end` (default
///   `1e-6`); must satisfy `ρ_beg > ρ_end > 0`.
/// - [`with_npt`](Self::with_npt): interpolation-set size `npt`, in
///   `[2n+1, ½(n+1)(n+2)]` (default `2n+1`).
///
/// # Panics
///
/// [`Executor::run`](crate::Executor::run) panics (in `init`) if the start point
/// is empty, if `npt` is outside `[2n+1, ½(n+1)(n+2)]` (the `n+2 ≤ npt ≤ 2n`
/// partial-model regime is not yet implemented), or if after the narrow-box
/// revision `ρ_beg > ρ_end > 0` cannot hold (e.g. a degenerate `b_i = a_i`).
///
/// # RESCUE
///
/// The full method of RESCUE (Powell 2009, §5: restoring the interpolation set
/// when rounding damages the update denominator) is implemented and wired, but
/// it fires only on severely ill-conditioned geometry. As Powell and PRIMA both
/// note, it is essentially never invoked on well-behaved problems without heavy
/// noise on the objective; expect it to stay dormant in normal use.
///
/// # Progress and continuation
///
/// Use [`PointState::new`] or [`crate::Executor::from_start`] with an unevaluated
/// starting point. Fresh initialization clears progress and rebuilds the
/// interpolation model, even when reusing a populated state or solver. The
/// shared state retains the current evaluated point, its historical objective
/// incumbent, and all six raw evaluation categories. It supplies no gradient
/// or radius capability. The solver owns every model buffer and the radius
/// schedule; [`rho`](Self::rho) returns `None` before initialization. Inspect
/// it through [`crate::Executor::run_with_solver`] or [`crate::Stepper::solver`].
///
/// Preserve [`crate::ExactCheckpoint`] for continuation without initialization
/// or reevaluation. With `serde`, the solver and shared state support
/// serialization, including the complete numerical model. A progress-only
/// snapshot is a fresh point warm start. Legacy algorithm-specific state
/// payloads cannot be read as `PointState` or as a new exact checkpoint.
/// Radius-based convergence remains solver-owned; objective targets and stalls
/// use the shared state's historical incumbent.
///
/// # Backends
///
/// `Vec<F>`, nalgebra `DVector<F>`, ndarray `Array1<F>`, and faer `Col<F>`,
/// for every supported backend release and `F = f64` or `f32`. The parameter
/// type needs only [`Clone`], [`VectorLen`], and `Index`/`IndexMut` element access.
/// The model uses pure-Rust `Vec<F>` storage, with no BLAS/LAPACK dependency.
/// Choose radii appropriate to the scalar precision.
///
/// # References
///
/// M. J. D. Powell, *The BOBYQA algorithm for bound constrained optimization
/// without derivatives*, DAMTP report 2009/NA06, University of Cambridge.
/// Cross-validated against [PRIMA](https://github.com/libprima/prima) v0.7.2,
/// the authoritative source for the exact formulas. PRIMA is BSD 3-Clause
/// licensed; its required notice is retained in the crate's `COPYRIGHT` file.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "F: serde::Serialize",
        deserialize = "F: serde::Deserialize<'de>"
    ))
)]
pub struct Bobyqa<Mode = Bounded, F: Scalar = f64> {
    radius_tolerance: Option<F>,
    rho_beg: F,
    rho_end: F,
    npt: Option<usize>,
    /// Built in [`Solver::init`]; the resumable model + shifted bounds + ρ/Δ
    /// schedule.
    work: Option<BobyqaWork<F>>,
    _mode: PhantomData<fn() -> Mode>,
}

impl<F: Scalar> Bobyqa<Bounded, F> {
    /// Current trust-region resolution, or `None` before initialization.
    /// Read it through `run_with_solver()` or `Stepper::solver()`.
    pub fn rho(&self) -> Option<F> {
        self.work.as_ref().map(|work| work.rho())
    }

    /// Stop when the observed radius or step size is <= the tolerance.
    ///
    /// Disabled by default. `None` disables the test and zero requests an
    /// exact-zero threshold. The tolerance must be finite and nonnegative.
    /// Checked at initialized iteration boundaries. This observation does not
    /// change the algorithm's radius or step-size update schedule.
    pub fn with_absolute_radius_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.radius_tolerance =
            crate::core::convergence::optional_tolerance(value);
        self
    }

    /// A BOBYQA solver with the default schedule (`ρ_beg = 1`, `ρ_end = 1e-6`,
    /// `npt = 2n+1`). Tune with the `with_*` builders.
    pub fn new() -> Self {
        Self {
            rho_beg: F::from_f64(1.0).expect("1.0 representable"),
            radius_tolerance: None,
            rho_end: F::from_f64(1e-6).expect("1e-6 representable"),
            npt: None,
            work: None,
            _mode: PhantomData,
        }
    }

    /// Configure the initial radius.
    /// Retains the algorithm's existing formula, validation, and default.
    pub fn with_initial_radius(mut self, rho_beg: F) -> Self {
        self.rho_beg = rho_beg;
        self
    }

    /// Configure the final radius.
    /// Retains the algorithm's existing formula, validation, and default.
    pub fn with_final_radius(mut self, rho_end: F) -> Self {
        self.rho_end = rho_end;
        self
    }

    /// Set the interpolation-set size `npt`, in `[2n+1, ½(n+1)(n+2)]`. Defaults
    /// to `2n+1` when unset.
    pub fn with_npt(mut self, npt: usize) -> Self {
        self.npt = Some(npt);
        self
    }
}

impl<F: Scalar> Default for Bobyqa<Bounded, F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a `V` from a flat `&[F]`, reusing `template` for type and length.
fn fill_from<V, F>(template: &V, slice: &[F]) -> V
where
    V: Clone + std::ops::IndexMut<usize, Output = F>,
    F: Copy,
{
    let mut v = template.clone();
    for (i, &x) in slice.iter().enumerate() {
        v[i] = x;
    }
    v
}

/// Copy a backend vector `V` into a flat `Vec<F>` of length `n`.
fn to_vec<V, F>(v: &V, n: usize) -> Vec<F>
where
    V: std::ops::Index<usize, Output = F>,
    F: Copy,
{
    (0..n).map(|i| v[i]).collect()
}

impl<V, F> InitialState<V> for Bobyqa<Bounded, F>
where
    F: Scalar,
    V: Clone,
{
    type State = PointState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        PointState::new(x.clone())
    }
}

impl<P, V, F> Solver<P, PointState<V, F>> for Bobyqa<Bounded, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + BoxConstraints,
    V: Clone
        + VectorLen
        + std::ops::Index<usize, Output = F>
        + std::ops::IndexMut<usize, Output = F>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PointState<V, F>,
    ) -> Result<PointState<V, F>, Self::Error> {
        state.reset();
        self.work = None;
        let n = state.param().vec_len();
        assert!(n >= 1, "Bobyqa requires a non-empty start point");
        let npt = self.npt.unwrap_or(2 * n + 1);

        let x0 = to_vec(state.param(), n);
        let lower = to_vec(problem.inner().lower(), n);
        let upper = to_vec(problem.inner().upper(), n);
        let template = state.param().clone();

        let (work, best_x, best_f) = {
            let mut eval = |slice: &[F]| -> Result<F, P::Error> {
                problem.cost(&fill_from(&template, slice))
            };
            BobyqaWork::try_init(
                x0,
                &lower,
                &upper,
                self.rho_beg,
                self.rho_end,
                npt,
                &mut eval,
            )?
        };

        state.replace(fill_from(&template, &best_x), best_f);
        self.work = Some(work);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PointState<V, F>,
    ) -> Result<(PointState<V, F>, Option<TerminationReason>), Self::Error>
    {
        let template = state.param().clone();
        let work = self
            .work
            .as_mut()
            .expect("Bobyqa::init must run before next_iter");

        let out = {
            let mut eval = |slice: &[F]| -> Result<F, P::Error> {
                problem.cost(&fill_from(&template, slice))
            };
            work.step(&mut eval)?
        };

        // BOBYQA reports the least-F feasible point seen, so param/cost are
        // monotone and coincide with best_param/best_cost.
        let mut best_f = state.cost();
        for (xabs, f_new) in &out.evaluated {
            if *f_new < best_f {
                best_f = *f_new;
                state.replace(fill_from(&template, xabs), best_f);
            }
        }

        let reason = match out.transition {
            Transition::Converged => Some(TerminationReason::SolverConverged),
            Transition::Continue | Transition::RhoReduced => None,
        };
        Ok((state, reason))
    }

    fn terminate(
        &self,
        _state: &PointState<V, F>,
    ) -> Option<TerminationReason> {
        let tolerance = self.radius_tolerance?;
        let metric = self.rho()?;
        (metric.is_finite() && metric <= tolerance)
            .then_some(TerminationReason::RhoTolerance)
    }
}

#[cfg(test)]
mod tests {
    use crate::core::constraint::BoxConstraints;
    use crate::core::problem::CostFunction;
    use crate::{Bobyqa, Executor, PointState};

    struct Quad {
        lower: Vec<f64>,
        upper: Vec<f64>,
    }
    impl CostFunction for Quad {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
            Ok((x[0] - 1.0).powi(2) + 2.0 * (x[1] + 2.0).powi(2))
        }
    }
    impl BoxConstraints for Quad {
        fn lower(&self) -> &Vec<f64> {
            &self.lower
        }
        fn upper(&self) -> &Vec<f64> {
            &self.upper
        }
    }

    /// Converges to the interior minimizer (1, -2), well inside a slack box.
    #[test]
    fn converges_interior_minimum() {
        let problem = Quad {
            lower: vec![-5.0, -5.0],
            upper: vec![5.0, 5.0],
        };
        let result = Executor::new(
            problem,
            Bobyqa::new()
                .with_initial_radius(0.5)
                .with_final_radius(1e-8),
            PointState::new(vec![0.0, 0.0]),
        )
        .max_cost_evals(500)
        .run()
        .unwrap();
        let x = result.best_param();
        assert!((x[0] - 1.0).abs() < 1e-5, "x0 = {}", x[0]);
        assert!((x[1] + 2.0).abs() < 1e-5, "x1 = {}", x[1]);
        assert!(result.best_cost() < 1e-9, "cost = {}", result.best_cost());
    }

    /// A box narrower than `2·ρ_beg` triggers PRIMA's narrow-box revision: the
    /// default `ρ_beg = 1.0` is reduced to `min(b_i − a_i)/4` instead of
    /// panicking, and BOBYQA still converges to the interior optimum (1, −2).
    #[test]
    fn narrow_box_revises_rho_beg() {
        let problem = Quad {
            // widths 0.2 and 0.4 ⇒ min/2 = 0.1 ≪ default ρ_beg = 1.0.
            lower: vec![0.9, -2.2],
            upper: vec![1.1, -1.8],
        };
        let result = Executor::new(
            problem,
            Bobyqa::new(),
            PointState::new(vec![1.0, -2.0]),
        )
        .max_cost_evals(500)
        .run()
        .unwrap();
        let x = result.best_param();
        assert!((x[0] - 1.0).abs() < 1e-3, "x0 = {}", x[0]);
        assert!((x[1] + 2.0).abs() < 1e-3, "x1 = {}", x[1]);
    }

    /// An infeasible start is clipped into the box, and BOBYQA still converges.
    #[test]
    fn infeasible_start_converges() {
        let problem = Quad {
            lower: vec![-3.0, -3.0],
            upper: vec![3.0, 3.0],
        };
        let result = Executor::new(
            problem,
            Bobyqa::new()
                .with_initial_radius(0.5)
                .with_final_radius(1e-8),
            PointState::new(vec![100.0, -100.0]),
        )
        .max_cost_evals(500)
        .run()
        .unwrap();
        let x = result.best_param();
        assert!((x[0] - 1.0).abs() < 1e-5, "x0 = {}", x[0]);
        assert!((x[1] + 2.0).abs() < 1e-5, "x1 = {}", x[1]);
    }

    /// The minimizer lies outside the box; BOBYQA converges to the constrained
    /// optimum at the corner (2, 2).
    #[test]
    fn converges_to_active_bound() {
        struct Bowl {
            lower: Vec<f64>,
            upper: Vec<f64>,
        }
        impl CostFunction for Bowl {
            type Param = Vec<f64>;
            type Output = f64;
            type Error = std::convert::Infallible;
            fn cost(
                &self,
                x: &Vec<f64>,
            ) -> Result<f64, std::convert::Infallible> {
                Ok((x[0] - 5.0).powi(2) + (x[1] - 5.0).powi(2))
            }
        }
        impl BoxConstraints for Bowl {
            fn lower(&self) -> &Vec<f64> {
                &self.lower
            }
            fn upper(&self) -> &Vec<f64> {
                &self.upper
            }
        }
        let problem = Bowl {
            lower: vec![-2.0, -2.0],
            upper: vec![2.0, 2.0],
        };
        let result = Executor::new(
            problem,
            Bobyqa::new()
                .with_initial_radius(0.5)
                .with_final_radius(1e-8),
            PointState::new(vec![0.0, 0.0]),
        )
        .max_cost_evals(500)
        .run()
        .unwrap();
        let x = result.best_param();
        assert!((x[0] - 2.0).abs() < 1e-5, "x0 = {}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-5, "x1 = {}", x[1]);
    }

    /// Rosenbrock with slack bounds: BOBYQA reaches the valley minimum (1, 1).
    #[test]
    fn converges_rosenbrock_box() {
        struct Rosen {
            lower: Vec<f64>,
            upper: Vec<f64>,
        }
        impl CostFunction for Rosen {
            type Param = Vec<f64>;
            type Output = f64;
            type Error = std::convert::Infallible;
            fn cost(
                &self,
                x: &Vec<f64>,
            ) -> Result<f64, std::convert::Infallible> {
                Ok(100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2))
            }
        }
        impl BoxConstraints for Rosen {
            fn lower(&self) -> &Vec<f64> {
                &self.lower
            }
            fn upper(&self) -> &Vec<f64> {
                &self.upper
            }
        }
        let problem = Rosen {
            lower: vec![-5.0, -5.0],
            upper: vec![5.0, 5.0],
        };
        let result = Executor::new(
            problem,
            Bobyqa::new()
                .with_initial_radius(0.5)
                .with_final_radius(1e-6),
            PointState::new(vec![-1.2, 1.0]),
        )
        .max_cost_evals(2000)
        .run()
        .unwrap();
        let x = result.best_param();
        assert!((x[0] - 1.0).abs() < 1e-3, "x0 = {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-3, "x1 = {}", x[1]);
        assert!(result.best_cost() < 1e-6, "cost = {}", result.best_cost());
    }
}
