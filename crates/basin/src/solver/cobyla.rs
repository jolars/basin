// Index-based loops mirror Powell's exposition (COBYLA, Powell 1994) and the
// dense index arithmetic of the simplex / active-set algebra; blanket-allowed.
#![allow(clippy::needless_range_loop)]

//! COBYLA (Powell 1994): Constrained Optimization BY Linear Approximations.
//!
//! Derivative-free solver for nonlinearly-constrained optimization. Unlike the
//! quadratic-model Powell trio (NEWUOA/BOBYQA/LINCOA), COBYLA builds **linear**
//! models by interpolation at the `n+1` vertices of a simplex and steers by an
//! L-infinity exact-penalty merit function, so it is the only Powell-family
//! solver that handles general nonlinear inequality constraints `c(x) ≤ 0`.
//!
//! Ported from [PRIMA](https://github.com/libprima/prima)'s modern COBYLA
//! (`cobylb`/`trstlp`/`geostep`/`update`), anchored to Powell's 1994 paper.
//! PRIMA is BSD 3-Clause licensed; its required notice is retained in the
//! crate's `COPYRIGHT` file.

pub(crate) mod driver;
pub(crate) mod filter;
pub(crate) mod geometry;
pub(crate) mod init;
pub(crate) mod linalg;
pub(crate) mod model;
pub(crate) mod trstlp;
pub(crate) mod update;

#[cfg(test)]
mod parity;
#[cfg(test)]
mod tests;

#[cfg(test)]
mod regression;

use crate::core::constraint::{
    FoldedConstraints, NonlinearConstraints, NonlinearInequalityConstraints,
};
use crate::core::inner::InitialState;
use crate::core::math::{MatVec, Scalar, VectorLen};
use crate::core::problem::{CostFunction, Problem};
use crate::core::solver::Solver;
use crate::core::state::CobylaState;
use crate::core::termination::TerminationReason;

use driver::{CobylaWork, Transition};

/// COBYLA (Powell 1994): derivative-free optimization with general nonlinear
/// inequality constraints.
///
/// COBYLA minimizes a smooth-or-not objective `F(x)` subject to `c(x) ≤ 0` for
/// an arbitrary vector-valued constraint function `c`, using only objective and
/// constraint *values*, no derivatives. Each iteration interpolates linear
/// models of `F` and `c` at the vertices of a simplex, takes a trust-region step
/// under those linear models, and ranks points by the L-infinity merit
/// `Φ = F + μ·[maxᵢ cᵢ]₊`. The trust-region resolution `ρ` shrinks from `ρ_beg`
/// to `ρ_end`.
///
/// Drive it with an [`Executor`](crate::Executor) over a [`CobylaState`] on a
/// problem implementing [`CostFunction`] and [`NonlinearInequalityConstraints`]:
///
/// ```
/// use basin::{
///     Cobyla, CobylaState, CostFunction, Executor,
///     NonlinearInequalityConstraints,
/// };
///
/// // min x0·x1  s.t.  x0² + x1² ≤ 1   (optimum F* = −1/2 on the unit circle).
/// struct Disk;
/// impl CostFunction for Disk {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, std::convert::Infallible> {
///         Ok(x[0] * x[1])
///     }
/// }
/// impl NonlinearInequalityConstraints for Disk {
///     fn constraints(
///         &self,
///         x: &Vec<f64>,
///     ) -> Result<Vec<f64>, std::convert::Infallible> {
///         Ok(vec![x[0] * x[0] + x[1] * x[1] - 1.0])
///     }
///     fn num_constraints(&self) -> usize {
///         1
///     }
/// }
///
/// let solver = Cobyla::new()
///     .with_initial_radius(0.5)
///     .with_final_radius(1e-6);
/// let state = CobylaState::new(vec![1.0, 1.0]);
/// let result = Executor::new(Disk, solver, state)
///     .max_cost_evals(500)
///     .run()
///     .unwrap();
/// assert!((result.best_cost() - (-0.5)).abs() < 1e-3);
/// ```
///
/// # Configuration
///
/// - [`with_rho_beg`](Self::with_rho_beg): initial trust-region radius `ρ_beg`
///   (a reasonable coarse change to the variables; default `1.0`).
/// - [`with_rho_end`](Self::with_rho_end): final radius `ρ_end`, ~ the required
///   accuracy (default `1e-6`); must satisfy `ρ_beg > ρ_end > 0`.
///
/// # Constraints
///
/// For inequality-only input, implement [`NonlinearInequalityConstraints`]:
/// the problem returns `c(x)` (feasible iff every `cᵢ(x) ≤ 0`). For the full
/// form, implement [`NonlinearConstraints`] and pass
/// `FoldedConstraints::new(problem)` to the executor. The adapter folds
/// nonlinear inequalities, optional linear inequalities and equalities, and
/// optional box bounds into one constraint vector. Equalities become paired
/// residuals of opposite signs; rows are neither normalized nor relaxed.
///
/// The start point need not be feasible, and callbacks may be evaluated outside
/// every supplied constraint, including box bounds. Folding retains COBYLA's
/// interpolation of all constraint models; it does not select PRIMA's separate
/// exact-linear-model path.
///
/// # Termination
///
/// Natural convergence is `ρ` reaching `ρ_end`, signalled as
/// [`TerminationReason::SolverConverged`]; this does not certify feasibility.
/// Add
/// [`max_cost_evals`](crate::Executor::max_cost_evals) to cap the budget (each evaluated point
/// counts once) or [`RhoTolerance`](crate::RhoTolerance) to stop at a coarser
/// `ρ`.
///
/// # Backends
///
/// Backend-generic: the parameter vector needs only [`Clone`], [`VectorLen`],
/// and indexing; COBYLA's models are pure-Rust `Vec<F>` scratch (no backend
/// matrix, no linear solve), so `Vec<f64>`, nalgebra, ndarray, and faer all
/// work, and it is wasm-clean.
/// The [`FoldedConstraints`] path additionally requires [`MatVec`] on the
/// constraint matrix: `DenseMatrix`/`Vec`, `DMatrix`/`DVector`, `Array2`/`Array1`,
/// and `Mat`/`Col` all provide it in pure Rust. Both paths support `f32` and
/// `f64`.
///
/// # References
///
/// M. J. D. Powell, *A direct search optimization method that models the
/// objective and constraint functions by linear interpolation*, in *Advances in
/// Optimization and Numerical Analysis* (eds. Gomez & Hennart), Kluwer (1994),
/// pp. 51–67. Ported from [PRIMA](https://github.com/libprima/prima).
///
/// [`CostFunction`]: crate::core::problem::CostFunction
/// [`TerminationReason::SolverConverged`]: crate::TerminationReason::SolverConverged
pub struct Cobyla<F = f64> {
    radius_tolerance: Option<F>,
    rho_beg: F,
    rho_end: F,
    /// Built in [`Solver::init`]; the resumable simplex + schedule + filter.
    work: Option<CobylaWork<F>>,
}

impl<F: Scalar> Cobyla<F> {
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

    /// A COBYLA solver with the default schedule (`ρ_beg = 1`, `ρ_end = 1e-6`).
    pub fn new() -> Self {
        Self {
            rho_beg: F::from_f64(1.0).expect("1.0 representable"),
            radius_tolerance: None,
            rho_end: F::from_f64(1e-6).expect("1e-6 representable"),
            work: None,
        }
    }

    /// Set the initial trust-region radius `ρ_beg` (also the initial `Δ`).
    #[deprecated(
        note = "use `with_initial_radius`; removal scheduled for Basin 2.0"
    )]
    pub fn with_rho_beg(self, rho_beg: F) -> Self {
        self.with_initial_radius(rho_beg)
    }

    /// Configure the initial radius.
    /// Retains the algorithm's existing formula, validation, and default.
    pub fn with_initial_radius(mut self, rho_beg: F) -> Self {
        self.rho_beg = rho_beg;
        self
    }

    /// Set the final trust-region radius `ρ_end`. Must satisfy `ρ_beg > ρ_end > 0`.
    #[deprecated(
        note = "use `with_final_radius`; removal scheduled for Basin 2.0"
    )]
    pub fn with_rho_end(self, rho_end: F) -> Self {
        self.with_final_radius(rho_end)
    }

    /// Configure the final radius.
    /// Retains the algorithm's existing formula, validation, and default.
    pub fn with_final_radius(mut self, rho_end: F) -> Self {
        self.rho_end = rho_end;
        self
    }
}

impl<F: Scalar> Default for Cobyla<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Reuse the backend parameter buffer across callbacks and incumbent updates.
fn fill_into<V, F>(v: &mut V, slice: &[F])
where
    V: std::ops::IndexMut<usize, Output = F>,
    F: Copy,
{
    for (i, &x) in slice.iter().enumerate() {
        v[i] = x;
    }
}

impl<V, F> InitialState<V> for Cobyla<F>
where
    F: Scalar,
    V: Clone,
{
    type State = CobylaState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        CobylaState::new(x.clone())
    }
}

type CobylaStep<V, F, E> =
    Result<(CobylaState<V, F>, Option<TerminationReason>), E>;

impl<F: Scalar> Cobyla<F> {
    fn init_with<P, V>(
        &mut self,
        problem: &mut Problem<P>,
        mut state: CobylaState<V, F>,
        m: usize,
        mut constraints: impl FnMut(&P, &V) -> Result<Vec<F>, P::Error>,
    ) -> Result<CobylaState<V, F>, P::Error>
    where
        P: CostFunction<Param = V, Output = F>,
        V: Clone
            + VectorLen
            + std::ops::Index<usize, Output = F>
            + std::ops::IndexMut<usize, Output = F>,
    {
        let n = state.param.vec_len();
        assert!(n >= 1, "Cobyla requires a non-empty start point");
        let x0: Vec<F> = (0..n).map(|i| state.param[i]).collect();

        let (work, best_x, best_f) = {
            let mut eval = |slice: &[F]| -> Result<(F, Vec<F>), P::Error> {
                fill_into(&mut state.param, slice);
                let f = problem.cost(&state.param)?;
                let c = constraints(problem.inner(), &state.param)?;
                assert_eq!(
                    c.len(),
                    m,
                    "Cobyla constraint count must remain fixed during a solve",
                );
                Ok((f, c))
            };
            CobylaWork::try_init(x0, m, self.rho_beg, self.rho_end, &mut eval)?
        };

        fill_into(&mut state.param, &best_x);
        state.cost = Some(best_f);
        state.rho = work.rho();
        self.work = Some(work);
        Ok(state)
    }

    fn next_iter_with<P, V>(
        &mut self,
        problem: &mut Problem<P>,
        mut state: CobylaState<V, F>,
        mut constraints: impl FnMut(&P, &V) -> Result<Vec<F>, P::Error>,
    ) -> CobylaStep<V, F, P::Error>
    where
        P: CostFunction<Param = V, Output = F>,
        V: Clone
            + VectorLen
            + std::ops::Index<usize, Output = F>
            + std::ops::IndexMut<usize, Output = F>,
    {
        let work = self
            .work
            .as_mut()
            .expect("Cobyla::init must run before next_iter");
        let m = work.num_constraints();

        let transition = {
            let mut eval = |slice: &[F]| -> Result<(F, Vec<F>), P::Error> {
                fill_into(&mut state.param, slice);
                let f = problem.cost(&state.param)?;
                let c = constraints(problem.inner(), &state.param)?;
                assert_eq!(
                    c.len(),
                    m,
                    "Cobyla constraint count must remain fixed during a solve",
                );
                Ok((f, c))
            };
            work.step(&mut eval)?
        };
        state.rho = work.rho();

        let (best_x, best_f) = work.best_ref();
        fill_into(&mut state.param, best_x);
        state.cost = Some(best_f);

        let reason = match transition {
            Transition::Converged => Some(TerminationReason::SolverConverged),
            Transition::Failed => Some(TerminationReason::SolverFailed),
            Transition::Continue | Transition::RhoReduced => None,
        };
        Ok((state, reason))
    }

    fn radius_termination<V: Clone>(
        &self,
        state: &CobylaState<V, F>,
    ) -> Option<TerminationReason> {
        let tolerance = self.radius_tolerance?;
        let metric = crate::RhoState::rho(state);
        (metric.is_finite() && metric <= tolerance)
            .then_some(TerminationReason::RhoTolerance)
    }
}

fn inequality_values<P, V, F>(
    problem: &P,
    x: &V,
    m: usize,
) -> Result<Vec<F>, P::Error>
where
    P: NonlinearInequalityConstraints<Param = V, Output = F>,
    V: VectorLen + std::ops::Index<usize, Output = F>,
    F: Scalar,
{
    let cv = problem.constraints(x)?;
    debug_assert_eq!(
        cv.vec_len(),
        m,
        "constraints() returned {} values but num_constraints() = {m}",
        cv.vec_len(),
    );
    Ok((0..m).map(|i| cv[i]).collect())
}

impl<P, V, F> Solver<P, CobylaState<V, F>> for Cobyla<F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + NonlinearInequalityConstraints,
    V: Clone
        + VectorLen
        + std::ops::Index<usize, Output = F>
        + std::ops::IndexMut<usize, Output = F>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        state: CobylaState<V, F>,
    ) -> Result<CobylaState<V, F>, Self::Error> {
        let m = problem.inner().num_constraints();
        self.init_with(problem, state, m, |p, x| inequality_values(p, x, m))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: CobylaState<V, F>,
    ) -> Result<(CobylaState<V, F>, Option<TerminationReason>), Self::Error>
    {
        let m = problem.inner().num_constraints();
        self.next_iter_with(problem, state, |p, x| inequality_values(p, x, m))
    }

    fn terminate(
        &self,
        state: &CobylaState<V, F>,
    ) -> Option<TerminationReason> {
        self.radius_termination(state)
    }
}

impl<P, V, F> Solver<FoldedConstraints<P>, CobylaState<V, F>> for Cobyla<F>
where
    F: Scalar,
    P: NonlinearConstraints<Param = V, Output = F>,
    P::Matrix: MatVec<V>,
    V: Clone
        + VectorLen
        + std::ops::Index<usize, Output = F>
        + std::ops::IndexMut<usize, Output = F>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<FoldedConstraints<P>>,
        state: CobylaState<V, F>,
    ) -> Result<CobylaState<V, F>, Self::Error> {
        let m = problem.inner().constraint_count(state.param.vec_len());
        self.init_with(
            problem,
            state,
            m,
            FoldedConstraints::evaluate_constraints,
        )
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<FoldedConstraints<P>>,
        state: CobylaState<V, F>,
    ) -> Result<(CobylaState<V, F>, Option<TerminationReason>), Self::Error>
    {
        self.next_iter_with(
            problem,
            state,
            FoldedConstraints::evaluate_constraints,
        )
    }

    fn terminate(
        &self,
        state: &CobylaState<V, F>,
    ) -> Option<TerminationReason> {
        self.radius_termination(state)
    }
}
