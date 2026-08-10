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

use crate::core::constraint::NonlinearInequalityConstraints;
use crate::core::inner::InitialState;
use crate::core::math::{Scalar, VectorLen};
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
/// use basin::{CostFunction, Cobyla, CobylaState, Executor, MaxCostEvals, NonlinearInequalityConstraints};
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
///     fn constraints(&self, x: &Vec<f64>) -> Result<Vec<f64>, std::convert::Infallible> {
///         Ok(vec![x[0] * x[0] + x[1] * x[1] - 1.0])
///     }
///     fn num_constraints(&self) -> usize { 1 }
/// }
///
/// let solver = Cobyla::new().with_rho_beg(0.5).with_rho_end(1e-6);
/// let state = CobylaState::new(vec![1.0, 1.0]);
/// let result = Executor::new(Disk, solver, state)
///     .terminate_on(MaxCostEvals(500))
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
/// COBYLA binds [`NonlinearInequalityConstraints`]: the problem returns the
/// constraint vector `c(x)` (feasible iff every `cᵢ(x) ≤ 0`). Express box bounds
/// or linear constraints as nonlinear ones, and an equality `g(x) = 0` as the
/// pair `g ≤ 0`, `−g ≤ 0`. The start point need not be feasible.
///
/// # Termination
///
/// Natural convergence is `ρ` reaching `ρ_end`, signalled as
/// [`TerminationReason::SolverConverged`]. Add
/// [`MaxCostEvals`](crate::MaxCostEvals) to cap the budget (each evaluated point
/// counts once) or [`RhoTolerance`](crate::RhoTolerance) to stop at a coarser
/// `ρ`.
///
/// # Backends
///
/// Backend-generic: the parameter vector needs only [`Clone`], [`VectorLen`],
/// and indexing; COBYLA's models are pure-Rust `Vec<f64>` scratch (no backend
/// matrix, no linear solve), so `Vec<f64>`, nalgebra, ndarray, and faer all
/// work, and it is wasm-clean.
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
    rho_beg: F,
    rho_end: F,
    /// Built in [`Solver::init`]; the resumable simplex + schedule + filter.
    work: Option<CobylaWork<F>>,
}

impl<F: Scalar> Cobyla<F> {
    /// A COBYLA solver with the default schedule (`ρ_beg = 1`, `ρ_end = 1e-6`).
    pub fn new() -> Self {
        Self {
            rho_beg: F::from_f64(1.0).expect("1.0 representable"),
            rho_end: F::from_f64(1e-6).expect("1e-6 representable"),
            work: None,
        }
    }

    /// Set the initial trust-region radius `ρ_beg` (also the initial `Δ`).
    pub fn with_rho_beg(mut self, rho_beg: F) -> Self {
        self.rho_beg = rho_beg;
        self
    }

    /// Set the final trust-region radius `ρ_end`. Must satisfy `ρ_beg > ρ_end > 0`.
    pub fn with_rho_end(mut self, rho_end: F) -> Self {
        self.rho_end = rho_end;
        self
    }
}

impl<F: Scalar> Default for Cobyla<F> {
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
        mut state: CobylaState<V, F>,
    ) -> Result<CobylaState<V, F>, Self::Error> {
        let n = state.param.vec_len();
        assert!(n >= 1, "Cobyla requires a non-empty start point");
        let m = problem.inner().num_constraints();

        let x0: Vec<F> = (0..n).map(|i| state.param[i]).collect();
        let template = state.param.clone();

        let (work, best_x, best_f) = {
            let mut eval = |slice: &[F]| -> Result<(F, Vec<F>), P::Error> {
                let xv = fill_from(&template, slice);
                let f = problem.cost(&xv)?;
                let cv = problem.inner().constraints(&xv)?;
                debug_assert_eq!(
                    cv.vec_len(),
                    m,
                    "constraints() returned {} values but num_constraints() = {m}",
                    cv.vec_len(),
                );
                let c: Vec<F> = (0..m).map(|i| cv[i]).collect();
                Ok((f, c))
            };
            CobylaWork::try_init(x0, m, self.rho_beg, self.rho_end, &mut eval)?
        };

        state.param = fill_from(&template, &best_x);
        state.cost = Some(best_f);
        state.rho = work.rho();
        self.work = Some(work);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: CobylaState<V, F>,
    ) -> Result<(CobylaState<V, F>, Option<TerminationReason>), Self::Error>
    {
        let template = state.param.clone();
        let m = problem.inner().num_constraints();
        let work = self
            .work
            .as_mut()
            .expect("Cobyla::init must run before next_iter");

        let transition = {
            let mut eval = |slice: &[F]| -> Result<(F, Vec<F>), P::Error> {
                let xv = fill_from(&template, slice);
                let f = problem.cost(&xv)?;
                let cv = problem.inner().constraints(&xv)?;
                debug_assert_eq!(
                    cv.vec_len(),
                    m,
                    "constraints() returned {} values but num_constraints() = {m}",
                    cv.vec_len(),
                );
                let c: Vec<F> = (0..m).map(|i| cv[i]).collect();
                Ok((f, c))
            };
            work.step(&mut eval)?
        };
        state.rho = work.rho();

        let (best_x, best_f) = work.best();
        state.param = fill_from(&template, &best_x);
        state.cost = Some(best_f);

        let reason = match transition {
            Transition::Converged => Some(TerminationReason::SolverConverged),
            Transition::Failed => Some(TerminationReason::SolverFailed),
            Transition::Continue | Transition::RhoReduced => None,
        };
        Ok((state, reason))
    }
}
