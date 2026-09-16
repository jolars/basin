//! Dense sequential least-squares programming (Kraft's Han–Powell method).

#![allow(clippy::needless_range_loop)]

mod factor;
mod least_squares;

use crate::core::constraint::{ConstraintJacobian, NonlinearConstraints};
use crate::core::inner::InitialState;
use crate::core::math::{MatrixIndex, Scalar, VectorIndex, VectorLen};
use crate::core::problem::{CostFunction, Gradient, Problem};
use crate::core::solver::Solver;
use crate::core::state::SlsqpState;
use crate::core::termination::TerminationReason;
use factor::Factor;
use least_squares::{Lsei, Matrix, dot, norm, number};

/// Numerical failure reported alongside [`TerminationReason::SolverFailed`].
/// User callback errors propagate separately, unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum SlsqpFailure {
    /// More equality rows than free variables.
    TooManyEqualities,
    /// The equality Jacobian is numerically rank deficient.
    RankDeficientEqualities,
    /// The constrained least-squares factorization is singular.
    SingularSubproblem,
    /// The linearized constraints remain incompatible after slack recovery.
    IncompatibleConstraints,
    /// NNLS exhausted its active-set iteration limit.
    SubproblemIterationLimit,
    /// Hessian resets could not produce descent in the merit function.
    NonDescentDirection,
    /// No finite trial could be accepted by the line search.
    LineSearchFailed,
    /// An objective, derivative, constraint value, or model calculation is not finite.
    NonFiniteEvaluation,
}

/// Sequential least-squares programming for smooth constrained optimization.
///
/// Implements Kraft's Han–Powell SLSQP with a positive-definite, damped BFGS
/// approximation to the Lagrangian Hessian, constrained least-squares QP
/// subproblems, and an inexact L1 merit line search. Inconsistent linearizations
/// use Kraft's additional slack variable. Dense storage is quadratic in the
/// numbers of variables/constraints; this is intended for small and medium
/// dense problems. Exact line search and the alternative BVLS kernel are not
/// exposed.
///
/// # Problem and progress
///
/// Supply [`Gradient`], [`NonlinearConstraints`], and [`ConstraintJacobian`],
/// or synthesize derivatives with [`crate::FiniteDiff`] or
/// [`crate::BoundedFiniteDiff`]. Native equalities are `h(x) = 0` and public
/// inequalities are `c(x) ≤ 0`. Linear blocks are evaluated exactly. Constraint
/// counts, shapes, bounds, and linear coefficients must remain fixed throughout
/// the solve. Malformed shapes and invalid configurations panic.
///
/// Nonlinear feasibility is not required at the start. Initial and trial points
/// are clipped to the box, and fixed coordinates are eliminated internally.
/// As in [`NonlinearConstraints`], non-finite bounds mean an unbounded side.
/// Numerical differentiation respects the box only when explicitly configured
/// with the bounded adapter. Every accepted point has a matching cost and
/// objective gradient, even at termination. [`SlsqpState`] publishes the latest
/// accepted point; its `best_*` readers do not select by objective alone.
///
/// # Convergence and safeguards
///
/// [`with_absolute_accuracy_tolerance`](Self::with_absolute_accuracy_tolerance)
/// controls Kraft's composite tests (default `1e-6`): the sum of constraint
/// violations must be small, together with either the QP directional optimality
/// measure, absolute objective change, or Euclidean step norm. After five
/// Hessian resets the reference permits the objective/step test at ten times
/// this accuracy. These tests are not a strict KKT certificate; inspect the
/// state's feasibility, projected Lagrangian-gradient, and complementarity
/// diagnostics. `None` disables convergence tests; zero requests exact zero.
/// Executor controls own iteration/evaluation/time budgets and cancellation.
///
/// The original NNLS iteration limit is three times its number of columns;
/// [`with_max_subproblem_iterations`](Self::with_max_subproblem_iterations)
/// overrides it. The inexact line search permits eleven trials, as in the
/// reference, but never accepts non-finite values. Failed subproblems, exhausted
/// recovery, and invalid derivatives return `SolverFailed` with details on the
/// state. Typed callback errors abort immediately without promising rollback.
/// Constraint blocks count as residual evaluations and their Jacobians count
/// as Jacobian evaluations through [`Problem`].
///
/// # Backends
///
/// `Vec<F>`/[`crate::DenseMatrix<F>`], nalgebra `DVector<F>`/`DMatrix<F>`,
/// ndarray `Array1<F>`/`Array2<F>`, and faer `Col<F>`/`Mat<F>`, with `F = f64`
/// or `f32`. Choose an accuracy appropriate to the scalar precision and
/// constraint scales; `f32` generally needs a looser accuracy. Only vector indexing and dense matrix shape/entry access are
/// required from the backend. All factorizations are pure Rust and work on
/// WASM without BLAS/LAPACK. The reference fixtures use `f64`.
///
/// # References
///
/// Kraft, D. (1988). *A software package for sequential quadratic programming*.
/// Technical Report DFVLR-FB 88-28, Institut für Dynamik der Flugsysteme,
/// Oberpfaffenhofen. [Report](https://degenerateconic.com/uploads/2018/03/DFVLR_FB_88_28.pdf).
/// Implements §§2.2 and 3.2 and Lawson–Hanson constrained least squares.
/// The executable reference is Jacob Williams's SLSQP 1.6.1, commit
/// `97884f98042624007f736dc536fa5636906ff26a`, using the original NNLS and
/// inexact line search. Applicable notices are retained in `COPYRIGHT`.
///
/// # Examples
///
/// Minimize `x²` with the nonlinear equality `x - 1 = 0`:
/// ```
/// use basin::{CostFunction, DenseMatrix, Executor, FiniteDiff,
///     NonlinearConstraints, Slsqp, SlsqpState, State};
/// struct Example;
/// impl CostFunction for Example {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> { Ok(x[0]*x[0]) }
/// }
/// impl NonlinearConstraints for Example {
///     type Matrix = DenseMatrix;
///     fn num_nonlinear_constraints(&self) -> usize { 0 }
///     fn nonlinear_constraints(&self, _: &Vec<f64>) -> Result<Vec<f64>, Self::Error> { Ok(vec![]) }
///     fn num_nonlinear_equalities(&self) -> usize { 1 }
///     fn nonlinear_equalities(&self, x: &Vec<f64>) -> Result<Option<Vec<f64>>, Self::Error> {
///         Ok(Some(vec![x[0]-1.0]))
///     }
/// }
/// let result = Executor::new(FiniteDiff::new(Example), Slsqp::new(),
///     SlsqpState::new(vec![0.0])).max_iter(100).run().unwrap();
/// assert!((result.state.param()[0]-1.0).abs() < 1e-6);
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Slsqp<F: Scalar = f64> {
    accuracy: Option<F>,
    max_subproblem_iterations: Option<usize>,
    work: Option<Work<F>>,
}

impl<F: Scalar> Default for Slsqp<F> {
    fn default() -> Self {
        Self::new()
    }
}
impl<F: Scalar> Slsqp<F> {
    /// Construct SLSQP with absolute accuracy `1e-6` and automatic NNLS limits.
    pub fn new() -> Self {
        Self {
            accuracy: Some(number(1e-6)),
            max_subproblem_iterations: None,
            work: None,
        }
    }
    /// Set the native composite accuracy. `None` disables convergence and
    /// zero requests exact-zero tests. Negative or non-finite values panic.
    pub fn with_absolute_accuracy_tolerance(
        mut self,
        value: impl Into<Option<F>>,
    ) -> Self {
        self.accuracy = crate::core::convergence::optional_tolerance(value);
        self
    }
    /// Set a positive NNLS active-set iteration limit per QP subproblem.
    pub fn with_max_subproblem_iterations(mut self, limit: usize) -> Self {
        assert!(
            limit > 0,
            "SLSQP subproblem iteration limit must be positive"
        );
        self.max_subproblem_iterations = Some(limit);
        self
    }
    /// Latest QP equality multipliers, linear rows before nonlinear rows.
    /// Signs use `L = f + λᵀh + μᵀc` with public `c ≤ 0` inequalities.
    pub fn equality_multipliers(&self) -> Option<&[F]> {
        self.work
            .as_ref()
            .filter(|w| w.have_multipliers)
            .map(|w| &w.public_multipliers[..w.meq()])
    }
    /// Latest QP inequality multipliers, linear rows before nonlinear rows.
    /// Bound multipliers are excluded, as in the reference SLSQP interface.
    pub fn inequality_multipliers(&self) -> Option<&[F]> {
        self.work
            .as_ref()
            .filter(|w| w.have_multipliers)
            .map(|w| &w.public_multipliers[w.meq()..])
    }
}

impl<V: Clone, F: Scalar> InitialState<V> for Slsqp<F> {
    type State = SlsqpState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        SlsqpState::new(x.clone())
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Work<F> {
    free: Vec<usize>,
    lower: Vec<F>,
    upper: Vec<F>,
    linear_eq: Matrix<F>,
    rhs_eq: Vec<F>,
    linear_ineq: Matrix<F>,
    rhs_ineq: Vec<F>,
    nonlinear_eq: usize,
    nonlinear_ineq: usize,
    c: Vec<F>,
    a: Matrix<F>,
    g: Vec<F>,
    factor: Factor<F>,
    multipliers: Vec<F>,
    public_multipliers: Vec<F>,
    have_multipliers: bool,
    penalties: Vec<F>,
    direction: Vec<F>,
    derivative: F,
    inconsistent: bool,
    resets: usize,
    last_change: Option<F>,
    last_step: Option<F>,
    converged: bool,
    failure: Option<SlsqpFailure>,
    // Scratch is overwritten before use, so checkpoints need only the model.
    #[cfg_attr(feature = "serde", serde(skip))]
    scratch: Scratch<F>,
}

#[derive(Clone, Debug)]
struct Scratch<F> {
    y: Vec<F>,
    bs: Vec<F>,
    rank_one: Vec<F>,
    eq: Matrix<F>,
    eq_rhs: Vec<F>,
    ls: Matrix<F>,
    ls_rhs: Vec<F>,
    ineq: Matrix<F>,
    ineq_rhs: Vec<F>,
    least_squares: least_squares::Workspace<F>,
}

impl<F> Default for Scratch<F> {
    fn default() -> Self {
        Self {
            y: Vec::new(),
            bs: Vec::new(),
            rank_one: Vec::new(),
            eq: Matrix::default(),
            eq_rhs: Vec::new(),
            ls: Matrix::default(),
            ls_rhs: Vec::new(),
            ineq: Matrix::default(),
            ineq_rhs: Vec::new(),
            least_squares: least_squares::Workspace::default(),
        }
    }
}

fn below<F: Scalar>(value: F, tolerance: Option<F>) -> bool {
    tolerance.is_some_and(|t| {
        value.is_finite()
            && (value < t || (t == F::zero() && value == F::zero()))
    })
}
fn vector<V: VectorLen + VectorIndex<F>, F: Scalar>(v: &V) -> Vec<F> {
    (0..v.vec_len()).map(|i| v.get_scalar(i)).collect()
}
fn linear_block<V, M, F>(
    block: Option<(&M, &V)>,
    n: usize,
) -> (Matrix<F>, Vec<F>)
where
    V: VectorLen + VectorIndex<F>,
    M: MatrixIndex<F>,
    F: Scalar,
{
    let Some((a, b)) = block else {
        return (Matrix::zeros(0, n), vec![]);
    };
    assert_eq!(
        a.matrix_rows(),
        b.vec_len(),
        "constraint matrix row count mismatch"
    );
    assert_eq!(
        a.matrix_cols(),
        n,
        "constraint matrix column count mismatch"
    );
    let mut matrix = Matrix::zeros(a.matrix_rows(), n);
    for i in 0..matrix.rows {
        for j in 0..n {
            matrix.set(i, j, a.matrix_entry(i, j));
        }
    }
    (matrix, vector(b))
}

impl<F: Scalar> Work<F> {
    fn meq(&self) -> usize {
        self.linear_eq.rows + self.nonlinear_eq
    }
    fn violation(&self) -> F {
        self.c
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if i < self.meq() {
                    v.abs()
                } else {
                    (-v).max(F::zero())
                }
            })
            .sum()
    }
    fn weighted_violation(&self) -> F {
        self.c
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                self.penalties[i]
                    * if i < self.meq() {
                        v.abs()
                    } else {
                        (-v).max(F::zero())
                    }
            })
            .sum()
    }
    fn lagrangian_component(&self, g: F, a: &Matrix<F>, j: usize) -> F {
        g - (0..self.c.len())
            .map(|i| a.get(i, j) * self.multipliers[i])
            .sum::<F>()
    }
    fn qp<V: VectorIndex<F>>(
        &mut self,
        x: &V,
        slack: Option<F>,
        limit: Option<usize>,
    ) -> Result<(Vec<F>, Vec<F>), SlsqpFailure> {
        let meq = self.meq();
        let n = self.free.len() + usize::from(slack.is_some());
        let scratch = &mut self.scratch;
        let (eq, iq) = (&mut scratch.eq, &mut scratch.ineq);
        eq.resize_zeroed(meq, n);
        iq.resize_zeroed(self.c.len() - meq, n);
        for i in 0..self.c.len() {
            let matrix = if i < meq { &mut *eq } else { &mut *iq };
            let row = if i < meq { i } else { i - meq };
            for j in 0..self.free.len() {
                matrix.set(row, j, self.a.get(i, j));
            }
            if slack.is_some() {
                matrix.set(
                    row,
                    n - 1,
                    if i < meq {
                        -self.c[i]
                    } else {
                        (-self.c[i]).max(F::zero())
                    },
                );
            }
        }
        let (d, h) = (&mut scratch.eq_rhs, &mut scratch.ineq_rhs);
        d.clear();
        d.extend(self.c[..meq].iter().map(|v| -*v));
        h.clear();
        h.extend(self.c[meq..].iter().map(|v| -*v));
        for lower in [true, false] {
            for (j, &index) in self.free.iter().enumerate() {
                let bound = if lower {
                    self.lower[index]
                } else {
                    self.upper[index]
                };
                if bound.is_finite() {
                    let sign = if lower { F::one() } else { -F::one() };
                    iq.data.extend(
                        (0..n).map(|k| if k == j { sign } else { F::zero() }),
                    );
                    iq.rows += 1;
                    h.push(sign * (bound - x.get_scalar(index)));
                }
            }
        }
        if slack.is_some() {
            for (sign, bound) in [(F::one(), F::zero()), (-F::one(), -F::one())]
            {
                iq.data.extend(
                    (0..n).map(|k| if k == n - 1 { sign } else { F::zero() }),
                );
                iq.rows += 1;
                h.push(bound);
            }
        }
        self.factor.least_squares(
            &self.g,
            slack,
            &mut scratch.ls,
            &mut scratch.ls_rhs,
        );
        let (mut step, mut multipliers) = Lsei {
            c: eq,
            d,
            e: &mut scratch.ls,
            f: &scratch.ls_rhs,
            g: iq,
            h,
        }
        .solve(limit, &mut scratch.least_squares)?;
        multipliers.truncate(self.c.len());
        for (j, &index) in self.free.iter().enumerate() {
            step[j] = step[j]
                .max(self.lower[index] - x.get_scalar(index))
                .min(self.upper[index] - x.get_scalar(index));
        }
        Ok((step, multipliers))
    }
    fn prepare<V: VectorIndex<F>>(
        &mut self,
        x: &V,
        accuracy: Option<F>,
        limit: Option<usize>,
    ) {
        if self
            .c
            .iter()
            .chain(&self.g)
            .chain(&self.a.data)
            .any(|v| !v.is_finite())
        {
            self.failure = Some(SlsqpFailure::NonFiniteEvaluation);
            return;
        }
        if self.free.is_empty() {
            let feasible =
                below(self.violation(), Some(accuracy.unwrap_or_else(F::zero)));
            self.converged = accuracy.is_some() && feasible;
            if !feasible {
                self.failure = Some(SlsqpFailure::IncompatibleConstraints);
            }
            self.have_multipliers = true;
            return;
        }
        loop {
            self.inconsistent = false;
            let mut result = self.qp(x, None, limit);
            if matches!(result, Err(SlsqpFailure::RankDeficientEqualities))
                && self.meq() == self.free.len()
            {
                result = Err(SlsqpFailure::IncompatibleConstraints);
            }
            let mut slack_factor = F::one();
            if matches!(result, Err(SlsqpFailure::IncompatibleConstraints)) {
                self.inconsistent = true;
                let mut weight = number(100.0);
                for _ in 0..6 {
                    result = self.qp(x, Some(weight), limit);
                    if !matches!(
                        result,
                        Err(SlsqpFailure::IncompatibleConstraints)
                    ) {
                        break;
                    }
                    weight = weight * number(10.0);
                }
            }
            let (mut step, multipliers) = match result {
                Ok(v) => v,
                Err(e) => {
                    self.failure = Some(e);
                    return;
                }
            };
            if self.inconsistent {
                slack_factor = F::one() - step.pop().unwrap();
            }
            self.multipliers = multipliers;
            self.public_multipliers = self
                .multipliers
                .iter()
                .enumerate()
                .map(|(i, &v)| if i < self.meq() { -v } else { v })
                .collect();
            self.have_multipliers = true;
            let gs = dot(&self.g, &step);
            let optimality = gs.abs()
                + self
                    .multipliers
                    .iter()
                    .zip(&self.c)
                    .map(|(&r, &c)| (r * c).abs())
                    .sum::<F>();
            if !optimality.is_finite() || !self.violation().is_finite() {
                self.failure = Some(SlsqpFailure::NonFiniteEvaluation);
                return;
            }
            for i in 0..self.c.len() {
                let r = self.multipliers[i].abs();
                self.penalties[i] =
                    r.max((self.penalties[i] + r) / number(2.0));
            }
            self.direction = step;
            if !self.inconsistent
                && below(optimality, accuracy)
                && below(self.violation(), accuracy)
            {
                self.converged = true;
                return;
            }
            self.derivative = gs - self.weighted_violation() * slack_factor;
            if !self.derivative.is_finite() {
                self.failure = Some(SlsqpFailure::NonFiniteEvaluation);
                return;
            }
            if self.derivative < F::zero() {
                return;
            }
            if !self.reset_factor(accuracy) {
                return;
            }
        }
    }
    fn reset_factor(&mut self, accuracy: Option<F>) -> bool {
        self.resets += 1;
        if self.resets > 5 {
            let relaxed = accuracy.map(|a| a * number(10.0));
            self.converged = !self.inconsistent
                && below(self.violation(), relaxed)
                && (self.last_change.is_some_and(|v| below(v, relaxed))
                    || self.last_step.is_some_and(|v| below(v, relaxed)));
            if !self.converged {
                self.failure = Some(SlsqpFailure::NonDescentDirection);
            }
            false
        } else {
            self.factor = Factor::identity(self.free.len());
            true
        }
    }
    fn values<P, V>(
        &self,
        problem: &mut Problem<P>,
        x: &V,
    ) -> Result<Vec<F>, P::Error>
    where
        P: NonlinearConstraints<Param = V, Output = F>,
        V: VectorLen + VectorIndex<F>,
    {
        assert_eq!(
            problem.inner().num_nonlinear_equalities(),
            self.nonlinear_eq,
            "nonlinear equality count changed"
        );
        assert_eq!(
            problem.inner().num_nonlinear_constraints(),
            self.nonlinear_ineq,
            "nonlinear inequality count changed"
        );
        let mut c: Vec<_> = (0..self.linear_eq.rows)
            .map(|i| {
                (0..x.vec_len())
                    .map(|j| self.linear_eq.get(i, j) * x.get_scalar(j))
                    .sum::<F>()
                    - self.rhs_eq[i]
            })
            .collect();
        if self.nonlinear_eq > 0 {
            let v = problem
                .nonlinear_equalities(x)?
                .expect("declared nonlinear equalities must return a vector");
            assert_eq!(
                v.vec_len(),
                self.nonlinear_eq,
                "nonlinear equality output length mismatch"
            );
            c.extend(vector(&v));
        }
        c.extend((0..self.linear_ineq.rows).map(|i| {
            self.rhs_ineq[i]
                - (0..x.vec_len())
                    .map(|j| self.linear_ineq.get(i, j) * x.get_scalar(j))
                    .sum::<F>()
        }));
        if self.nonlinear_ineq > 0 {
            let v = problem.nonlinear_constraints(x)?;
            assert_eq!(
                v.vec_len(),
                self.nonlinear_ineq,
                "nonlinear inequality output length mismatch"
            );
            c.extend(vector(&v).into_iter().map(|v| -v));
        }
        Ok(c)
    }
    fn jacobian<P, V>(
        &mut self,
        problem: &mut Problem<P>,
        x: &V,
    ) -> Result<Matrix<F>, P::Error>
    where
        P: ConstraintJacobian<Param = V, Output = F>,
        P::Matrix: MatrixIndex<F>,
        V: VectorLen,
    {
        let mut a = Matrix::zeros(self.c.len(), self.free.len());
        for i in 0..self.linear_eq.rows {
            for (j, &k) in self.free.iter().enumerate() {
                a.set(i, j, self.linear_eq.get(i, k));
            }
        }
        for i in 0..self.linear_ineq.rows {
            for (j, &k) in self.free.iter().enumerate() {
                a.set(self.meq() + i, j, -self.linear_ineq.get(i, k));
            }
        }
        if self.nonlinear_eq + self.nonlinear_ineq > 0 {
            let jac = problem.constraint_jacobian(x)?;
            assert_eq!(
                jac.matrix_rows(),
                self.nonlinear_eq + self.nonlinear_ineq,
                "constraint Jacobian row mismatch"
            );
            assert_eq!(
                jac.matrix_cols(),
                x.vec_len(),
                "constraint Jacobian column mismatch"
            );
            if (0..jac.matrix_rows()).any(|i| {
                (0..jac.matrix_cols())
                    .any(|j| !jac.matrix_entry(i, j).is_finite())
            }) {
                self.failure = Some(SlsqpFailure::NonFiniteEvaluation);
            }
            for i in 0..self.nonlinear_eq + self.nonlinear_ineq {
                let (row, sign) = if i < self.nonlinear_eq {
                    (self.linear_eq.rows + i, F::one())
                } else {
                    (
                        self.meq() + self.linear_ineq.rows + i
                            - self.nonlinear_eq,
                        -F::one(),
                    )
                };
                for (j, &k) in self.free.iter().enumerate() {
                    a.set(row, j, sign * jac.matrix_entry(i, k));
                }
            }
        }
        Ok(a)
    }
    fn diagnostics<V: VectorIndex<F>>(&self, state: &mut SlsqpState<V, F>) {
        state.violation = Some(if self.c.iter().all(|v| v.is_finite()) {
            self.violation()
        } else {
            F::infinity()
        });
        state.failure = self.failure;
        if self.have_multipliers {
            let mut stationarity = F::zero();
            for (j, &i) in self.free.iter().enumerate() {
                let lag = self.lagrangian_component(self.g[j], &self.a, j);
                if !lag.is_finite() {
                    stationarity = F::infinity();
                    break;
                }
                let x = state.param.get_scalar(i);
                // Clip the displacement directly so x - (x - g) cannot
                // cancel a small gradient at a large-magnitude parameter.
                let projected =
                    lag.max(x - self.upper[i]).min(x - self.lower[i]);
                stationarity = stationarity.max(projected.abs());
            }
            state.stationarity = Some(stationarity);
            state.complementarity = Some(
                (self.meq()..self.c.len())
                    .map(|i| (self.c[i] * self.multipliers[i]).abs())
                    .fold(F::zero(), F::max),
            );
        }
    }
}

impl<P, V, F> Solver<P, SlsqpState<V, F>> for Slsqp<F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>
        + Gradient<Gradient = V>
        + ConstraintJacobian,
    V: Clone + VectorLen + VectorIndex<F>,
    P::Matrix: MatrixIndex<F>,
{
    type Error = P::Error;
    fn init(
        &mut self,
        problem: &mut Problem<P>,
        state: SlsqpState<V, F>,
    ) -> Result<SlsqpState<V, F>, P::Error> {
        self.work = None;
        let mut state = SlsqpState::new(state.param);
        let n = state.param.vec_len();
        assert!(n > 0, "SLSQP requires at least one parameter");
        let mut lower = vec![F::neg_infinity(); n];
        let mut upper = vec![F::infinity(); n];
        for (bounds, out) in [
            (problem.inner().lower(), &mut lower),
            (problem.inner().upper(), &mut upper),
        ] {
            if let Some(bounds) = bounds {
                assert_eq!(bounds.vec_len(), n, "SLSQP bound length mismatch");
                for i in 0..n {
                    let v = bounds.get_scalar(i);
                    if v.is_finite() {
                        out[i] = v;
                    }
                }
            }
        }
        for i in 0..n {
            assert!(
                lower[i] <= upper[i],
                "SLSQP lower bound exceeds upper bound"
            );
            assert!(
                state.param.get_scalar(i).is_finite(),
                "SLSQP initial parameters must be finite"
            );
            state.param.set_scalar(
                i,
                state.param.get_scalar(i).max(lower[i]).min(upper[i]),
            );
        }
        let free: Vec<_> = (0..n).filter(|&i| lower[i] != upper[i]).collect();
        let nf = free.len();
        let (linear_eq, rhs_eq) = linear_block(problem.inner().equalities(), n);
        let (linear_ineq, rhs_ineq) =
            linear_block(problem.inner().inequalities(), n);
        let nonlinear_eq = problem.inner().num_nonlinear_equalities();
        let nonlinear_ineq = problem.inner().num_nonlinear_constraints();
        let m =
            linear_eq.rows + linear_ineq.rows + nonlinear_eq + nonlinear_ineq;
        let mut work = Work {
            free,
            lower,
            upper,
            linear_eq,
            rhs_eq,
            linear_ineq,
            rhs_ineq,
            nonlinear_eq,
            nonlinear_ineq,
            c: vec![F::zero(); m],
            a: Matrix::zeros(m, nf),
            g: vec![],
            factor: Factor::identity(nf),
            multipliers: vec![F::zero(); m],
            public_multipliers: vec![F::zero(); m],
            have_multipliers: false,
            penalties: vec![F::zero(); m],
            direction: vec![],
            derivative: F::zero(),
            inconsistent: false,
            resets: 1,
            last_change: None,
            last_step: None,
            converged: false,
            failure: None,
            scratch: Scratch::default(),
        };
        let (cost, gradient) = problem.cost_and_gradient(&state.param)?;
        assert_eq!(
            gradient.vec_len(),
            n,
            "SLSQP objective gradient length mismatch"
        );
        work.g = work.free.iter().map(|&i| gradient.get_scalar(i)).collect();
        work.c = work.values(problem, &state.param)?;
        work.a = work.jacobian(problem, &state.param)?;
        if work.failure.is_some()
            || !cost.is_finite()
            || (0..gradient.vec_len())
                .any(|i| !gradient.get_scalar(i).is_finite())
        {
            work.failure = Some(SlsqpFailure::NonFiniteEvaluation);
        } else {
            work.prepare(
                &state.param,
                self.accuracy,
                self.max_subproblem_iterations,
            );
        }
        state.record = Some((cost, gradient));
        state.pending_publication = true;
        work.diagnostics(&mut state);
        self.work = Some(work);
        Ok(state)
    }
    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: SlsqpState<V, F>,
    ) -> Result<(SlsqpState<V, F>, Option<TerminationReason>), P::Error> {
        let work = self.work.as_mut().expect("SLSQP must be initialized");
        if work.failure.is_some() {
            work.diagnostics(&mut state);
            return Ok((state, Some(TerminationReason::SolverFailed)));
        }
        if work.free.is_empty() {
            // No step is possible, so disabled convergence leaves termination
            // to executor controls without attempting empty BFGS updates.
            return Ok((state, None));
        }
        let old_cost = state.record.as_ref().unwrap().0;
        let merit = old_cost + work.weighted_violation();
        if !merit.is_finite() {
            work.failure = Some(SlsqpFailure::NonFiniteEvaluation);
            work.diagnostics(&mut state);
            return Ok((state, Some(TerminationReason::SolverFailed)));
        }
        let mut alpha = F::one();
        let mut accepted = None;
        let mut param = state.param.clone();
        for trial in 0..11 {
            for (j, &i) in work.free.iter().enumerate() {
                param.set_scalar(
                    i,
                    (state.param.get_scalar(i) + alpha * work.direction[j])
                        .max(work.lower[i])
                        .min(work.upper[i]),
                );
            }
            let cost = problem.cost(&param)?;
            let c = work.values(problem, &param)?;
            if cost.is_finite() && c.iter().all(|v| v.is_finite()) {
                let penalty = c
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        work.penalties[i]
                            * if i < work.meq() {
                                v.abs()
                            } else {
                                (-v).max(F::zero())
                            }
                    })
                    .sum::<F>();
                let change = cost + penalty - merit;
                if !change.is_finite() {
                    alpha = alpha * number(0.5);
                    continue;
                }
                if change <= alpha * work.derivative / number(10.0)
                    || trial == 10
                {
                    accepted = Some((param, cost, c));
                    break;
                }
                let scaled = alpha * work.derivative;
                alpha = alpha
                    * (scaled / (number::<F>(2.0) * (scaled - change)))
                        .max(number(0.1))
                        .min(F::one());
            } else {
                alpha = alpha * number(0.5);
            }
        }
        let Some((param, cost, c)) = accepted else {
            work.failure = Some(SlsqpFailure::LineSearchFailed);
            work.diagnostics(&mut state);
            return Ok((state, Some(TerminationReason::SolverFailed)));
        };
        let gradient = problem.gradient(&param)?;
        assert_eq!(
            gradient.vec_len(),
            state.param.vec_len(),
            "SLSQP objective gradient length mismatch"
        );
        let a = work.jacobian(problem, &param)?;
        if work.failure.is_some()
            || (0..gradient.vec_len())
                .any(|i| !gradient.get_scalar(i).is_finite())
            || a.data.iter().any(|v| !v.is_finite())
        {
            work.failure = Some(SlsqpFailure::NonFiniteEvaluation);
            work.diagnostics(&mut state);
            return Ok((state, Some(TerminationReason::SolverFailed)));
        }
        work.scratch.y.resize(work.free.len(), F::zero());
        for (j, &i) in work.free.iter().enumerate() {
            let g = gradient.get_scalar(i);
            work.scratch.y[j] = work.lagrangian_component(g, &a, j)
                - work.lagrangian_component(work.g[j], &work.a, j);
            work.g[j] = g;
            // The accepted displacement replaces the exhausted search direction.
            work.direction[j] = param.get_scalar(i) - state.param.get_scalar(i);
        }
        work.c = c;
        work.a = a;
        work.last_change = Some((cost - old_cost).abs());
        work.last_step = Some(norm(&work.direction));
        work.converged = !work.inconsistent
            && below(work.violation(), self.accuracy)
            && (below((cost - old_cost).abs(), self.accuracy)
                || below(norm(&work.direction), self.accuracy));
        if !work.converged
            && (work.factor.update(
                &work.direction,
                &mut work.scratch.y,
                &mut work.scratch.bs,
                &mut work.scratch.rank_one,
            ) || work.reset_factor(self.accuracy))
        {
            work.prepare(&param, self.accuracy, self.max_subproblem_iterations);
        }
        state.param = param;
        state.record = Some((cost, gradient));
        state.pending_publication = true;
        work.diagnostics(&mut state);
        Ok((state, None))
    }
    fn terminate(
        &self,
        _state: &SlsqpState<V, F>,
    ) -> Option<TerminationReason> {
        self.work
            .as_ref()
            .filter(|w| w.converged)
            .map(|_| TerminationReason::SolverConverged)
    }
}
