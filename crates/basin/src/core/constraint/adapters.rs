//! Explicit problem and local-search transformations.

use super::evaluation::{BoxOnly, BoxProblem, invalid};
use super::*;
use crate::Scalar;
use crate::core::inner::{InitialState, WarmStart};
use crate::core::problem::{
    CostFunction, Gradient, Hessian, HessianProduct, Jacobian,
    MiniBatchGradient, Problem, Residual,
};
use crate::core::solver::{Solver, SolverStep};
use crate::core::state::State;
use crate::core::termination::Termination;

/// A sampling domain used by algorithms whose local search can leave the box.
///
/// Sampling bounds do not declare mathematical constraints. MA-LS accepts only
/// an unconstrained problem with this capability.
pub trait SamplingBounds: CostFunction<Constraints = NoConstraints> {
    /// Borrow both sampling endpoints, reporting missing data explicitly.
    fn sampling_bounds(
        &self,
    ) -> Result<(&Self::Param, &Self::Param), ConstraintError>;
    /// Borrow the lower endpoint after validating the sampling box.
    fn sampling_lower(&self) -> &Self::Param {
        self.sampling_bounds().expect("validated sampling bounds").0
    }
    /// Borrow the upper endpoint after validating the sampling box.
    fn sampling_upper(&self) -> &Self::Param {
        self.sampling_bounds().expect("validated sampling bounds").1
    }
}

/// Explicitly reinterpret a box-constrained objective's bounds as a sampling domain.
///
/// The objective and its derivatives are unchanged. Mathematical bounds are
/// removed, so local searches and accepted results may lie outside the box.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SamplingDomain<P> {
    problem: P,
}
impl<P> SamplingDomain<P> {
    /// Own the problem. Evaluation capabilities require a box-only declaration.
    pub fn new(problem: P) -> Self {
        Self { problem }
    }
    /// Borrow the original problem.
    pub fn inner(&self) -> &P {
        &self.problem
    }
    /// Recover the original problem.
    pub fn into_inner(self) -> P {
        self.problem
    }
}
impl<P> ProblemConstraints for SamplingDomain<P> {
    type Constraints = NoConstraints;
    fn constraint_set(&self) -> NoConstraints {
        NoConstraints::EMPTY
    }
}
impl<P, V> CostFunction for SamplingDomain<P>
where
    P: CostFunction<Param = V> + BoxProblem<Param = V>,
{
    type Param = V;
    type Output = P::Output;
    type Error = P::Error;
    fn cost(&self, x: &V) -> Result<Self::Output, Self::Error> {
        self.problem.cost(x)
    }
}
impl<P, V> SamplingBounds for SamplingDomain<P>
where
    P: CostFunction<Param = V> + BoxProblem<Param = V>,
{
    fn sampling_bounds(&self) -> Result<(&V, &V), ConstraintError> {
        self.problem.box_bounds()
    }
}
/// Explicitly run an unconstrained local solver on a box-constrained problem's objective.
///
/// Use this inside bounded CMA or DE injection when the outer algorithm repairs
/// the local result. This adapter permits exploratory and returned points outside
/// the box. It preserves callback errors and every evaluation category.
/// Robust least-squares solvers retain the loss function and residual scale.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RelaxedInner<I> {
    inner: I,
}
impl<I> RelaxedInner<I> {
    /// Select an unconstrained local relaxation explicitly.
    pub fn new(inner: I) -> Self {
        Self { inner }
    }
    /// Borrow the local solver.
    pub fn inner(&self) -> &I {
        &self.inner
    }
    /// Recover the local solver.
    pub fn into_inner(self) -> I {
        self.inner
    }
}

pub(crate) struct RelaxedProblem<'a, P> {
    pub(crate) problem: &'a P,
}
impl<P> ProblemConstraints for RelaxedProblem<'_, P> {
    type Constraints = NoConstraints;
    fn constraint_set(&self) -> NoConstraints {
        NoConstraints::EMPTY
    }
}
impl<P: CostFunction> CostFunction for RelaxedProblem<'_, P> {
    type Param = P::Param;
    type Output = P::Output;
    type Error = P::Error;
    fn cost(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
        self.problem.cost(x)
    }
}
macro_rules! forward_derivatives {
    ($wrapper:ty,$($bound:tt)*)=>{
        impl<P> MiniBatchGradient for $wrapper where P:MiniBatchGradient,$($bound)* {
            type Gradient=P::Gradient;
            fn n_samples(&self)->usize{self.problem.n_samples()}
            fn batch_gradient(&self,x:&Self::Param,batch:&[usize])->Result<Self::Gradient,Self::Error>{self.problem.batch_gradient(x,batch)}
        }
        impl<P> Gradient for $wrapper where P:Gradient,$($bound)* {
            type Gradient=P::Gradient;
            fn gradient(&self,x:&Self::Param)->Result<Self::Gradient,Self::Error>{self.problem.gradient(x)}
            fn cost_and_gradient(&self,x:&Self::Param)->Result<(Self::Output,Self::Gradient),Self::Error>{self.problem.cost_and_gradient(x)}
        }
        impl<P> Hessian for $wrapper where P:Hessian,$($bound)* {
            type Hessian=P::Hessian;
            fn hessian(&self,x:&Self::Param)->Result<Self::Hessian,Self::Error>{self.problem.hessian(x)}
            fn cost_and_gradient_and_hessian(&self,x:&Self::Param)->Result<(Self::Output,Self::Gradient,Self::Hessian),Self::Error>{self.problem.cost_and_gradient_and_hessian(x)}
        }
        impl<P> HessianProduct for $wrapper where P:HessianProduct,$($bound)* {
            fn hessian_product(&self,x:&Self::Param,v:&Self::Param)->Result<Self::Gradient,Self::Error>{self.problem.hessian_product(x,v)}
        }
    }
}
forward_derivatives!(RelaxedProblem<'_, P>,);
forward_derivatives!(SamplingDomain<P>,P:BoxProblem<Param=<P as CostFunction>::Param>,);
macro_rules! forward_residuals {
    ($wrapper:ty,$($bound:tt)*)=>{
        impl<P> Residual for $wrapper where P:Residual,$($bound)* {
            type Param=<P as Residual>::Param;type Output=P::Output;type Error=P::Error;
            fn residual(&self,x:&Self::Param)->Result<Self::Output,Self::Error>{self.problem.residual(x)}
        }
        impl<P> Jacobian for $wrapper where P:Jacobian,$($bound)* {
            type Jacobian=P::Jacobian;
            fn jacobian(&self,x:&Self::Param)->Result<Self::Jacobian,Self::Error>{self.problem.jacobian(x)}
            fn residual_and_jacobian(&self,x:&Self::Param)->Result<(Self::Output,Self::Jacobian),Self::Error>{self.problem.residual_and_jacobian(x)}
        }
    }
}
forward_residuals!(RelaxedProblem<'_, P>,);
forward_residuals!(SamplingDomain<P>, P:BoxProblem<Param=<P as Residual>::Param>,);

impl<I, V> InitialState<V> for RelaxedInner<I>
where
    I: InitialState<V>,
{
    type State = I::State;
    fn seed(&self, x: &V) -> Self::State {
        self.inner.seed(x)
    }
}
impl<I, V> WarmStart<V> for RelaxedInner<I> where I: WarmStart<V> {}
impl<I, V, F: Scalar> crate::solver::cma_inject::MemeticInner<V, F>
    for RelaxedInner<I>
where
    I: crate::solver::cma_inject::MemeticInner<V, F>,
{
    fn seed_scaled(&mut self, x: &V, sigma: F) -> Self::State {
        self.inner.seed_scaled(x, sigma)
    }
}

impl<P, S, I> Solver<P, S> for RelaxedInner<I>
where
    P: CostFunction,
    P::Constraints: BoxOnly<P::Param>,
    S: State,
    I: for<'a> Solver<RelaxedProblem<'a, P>, S, Error = P::Error>,
{
    type Error = ConstraintSolveError<P::Error>;
    fn init(&mut self, p: &mut Problem<P>, state: S) -> Result<S, Self::Error> {
        let mut view = Problem::new(RelaxedProblem { problem: p.inner() });
        *view.counts_mut() = *p.counts();
        let result = self.inner.init(&mut view, state);
        let counts = *view.counts();
        *p.counts_mut() = counts;
        result.map_err(ConstraintSolveError::Evaluation)
    }
    fn next_iter(
        &mut self,
        p: &mut Problem<P>,
        state: S,
    ) -> Result<SolverStep<S>, Self::Error> {
        let mut view = Problem::new(RelaxedProblem { problem: p.inner() });
        *view.counts_mut() = *p.counts();
        let result = self.inner.next_iter(&mut view, state);
        let counts = *view.counts();
        *p.counts_mut() = counts;
        result.map_err(ConstraintSolveError::Evaluation)
    }
    fn terminate(&self, state: &S) -> Option<Termination<S::Float>> {
        <I as Solver<RelaxedProblem<'_, P>, S>>::terminate(&self.inner, state)
    }
    fn reset_convergence(&mut self) {
        <I as Solver<RelaxedProblem<'_, P>, S>>::reset_convergence(
            &mut self.inner,
        )
    }
    fn should_check_iterate_change(&self) -> bool {
        <I as Solver<RelaxedProblem<'_, P>, S>>::should_check_iterate_change(
            &self.inner,
        )
    }
    fn check_convergence(
        &mut self,
        p: &Problem<P>,
        state: &S,
    ) -> Option<Termination<S::Float>> {
        let mut view = Problem::new(RelaxedProblem { problem: p.inner() });
        *view.counts_mut() = *p.counts();
        self.inner.check_convergence(&view, state)
    }
}

pub(crate) fn validate_sampling<P, V, F>(
    p: &P,
    n: Option<usize>,
) -> Result<(), ConstraintSolveError<P::Error>>
where
    P: SamplingBounds<Param = V, Output = F>,
    V: crate::VectorLen + std::ops::Index<usize, Output = F>,
    F: Scalar,
{
    let (lower, upper) = p.sampling_bounds().map_err(invalid)?;
    let n = n.unwrap_or_else(|| lower.vec_len());
    super::model::dimension(
        ConstraintBlock::Bounds,
        ConstraintAxis::Length,
        n,
        lower.vec_len(),
    )
    .map_err(invalid)?;
    super::model::dimension(
        ConstraintBlock::Bounds,
        ConstraintAxis::Length,
        n,
        upper.vec_len(),
    )
    .map_err(invalid)?;
    for i in 0..n {
        super::model::validate_bound(lower[i], BoundSide::Lower, i)
            .map_err(invalid)?;
        super::model::validate_bound(upper[i], BoundSide::Upper, i)
            .map_err(invalid)?;
        if lower[i] > upper[i] {
            return Err(invalid(ConstraintError::InvertedBounds {
                coordinate: i,
            }));
        }
        if !lower[i].is_finite() || !upper[i].is_finite() {
            return Err(invalid(ConstraintError::NonFiniteInterval {
                coordinate: i,
            }));
        }
    }
    Ok(())
}

/// Attach a canonical constraint set to an unconstrained objective or residual problem.
///
/// Evaluation capabilities are forwarded without copying the problem data. The
/// wrapped problem must declare [`NoConstraints`], so this operation cannot
/// silently discard an existing constraint block.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Constrained<P, C> {
    problem: P,
    constraints: C,
}
impl<P, C> Constrained<P, C>
where
    P: ProblemConstraints<Constraints = NoConstraints>,
    C: ConstraintDescription,
{
    /// Own the objective and its complete constraint declaration.
    pub fn new(problem: P, constraints: C) -> Self {
        Self {
            problem,
            constraints,
        }
    }
    /// Borrow the original problem.
    pub fn inner(&self) -> &P {
        &self.problem
    }
    /// Recover the problem and constraint set.
    pub fn into_parts(self) -> (P, C) {
        (self.problem, self.constraints)
    }
}
impl<P, C: ConstraintDescription> ProblemConstraints for Constrained<P, C> {
    type Constraints = C;
    fn constraint_set(&self) -> C::Borrowed<'_> {
        self.constraints.borrowed()
    }
}
impl<P, C> CostFunction for Constrained<P, C>
where
    P: CostFunction<Constraints = NoConstraints>,
    C: ConstraintDescription,
{
    type Param = P::Param;
    type Output = P::Output;
    type Error = P::Error;
    fn cost(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
        self.problem.cost(x)
    }
}
impl<P, C> Residual for Constrained<P, C>
where
    P: Residual<Constraints = NoConstraints>,
    C: ConstraintDescription,
{
    type Param = P::Param;
    type Output = P::Output;
    type Error = P::Error;
    fn residual(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
        self.problem.residual(x)
    }
}
impl<P, C> Jacobian for Constrained<P, C>
where
    P: Jacobian<Constraints = NoConstraints>,
    C: ConstraintDescription,
{
    type Jacobian = P::Jacobian;
    fn jacobian(&self, x: &Self::Param) -> Result<Self::Jacobian, Self::Error> {
        self.problem.jacobian(x)
    }
    fn residual_and_jacobian(
        &self,
        x: &Self::Param,
    ) -> Result<(Self::Output, Self::Jacobian), Self::Error> {
        self.problem.residual_and_jacobian(x)
    }
}
macro_rules! constrained_values {
    ($cap:ident,$method:ident,$value:ident) => {
        impl<P, C> $cap for Constrained<P, C>
        where
            P: $cap<Constraints = NoConstraints>,
            C: ConstraintDescription,
        {
            type Param = P::Param;
            type $value = P::$value;
            type Error = P::Error;
            fn $method(
                &self,
                x: &Self::Param,
            ) -> Result<Self::$value, ConstraintSolveError<Self::Error>> {
                self.problem.$method(x)
            }
        }
    };
}
constrained_values!(NonlinearEqualityValues, nonlinear_equalities, Values);
constrained_values!(NonlinearInequalityValues, nonlinear_inequalities, Values);
constrained_values!(ConstraintJacobian, constraint_jacobian, Jacobian);

impl<P, C> MiniBatchGradient for Constrained<P, C>
where
    P: MiniBatchGradient<Constraints = NoConstraints>,
    C: ConstraintDescription,
{
    type Gradient = P::Gradient;
    fn n_samples(&self) -> usize {
        self.problem.n_samples()
    }
    fn batch_gradient(
        &self,
        x: &Self::Param,
        batch: &[usize],
    ) -> Result<Self::Gradient, Self::Error> {
        self.problem.batch_gradient(x, batch)
    }
}

impl<P, C> Gradient for Constrained<P, C>
where
    P: Gradient<Constraints = NoConstraints>,
    C: ConstraintDescription,
{
    type Gradient = P::Gradient;
    fn gradient(&self, x: &Self::Param) -> Result<Self::Gradient, Self::Error> {
        self.problem.gradient(x)
    }
    fn cost_and_gradient(
        &self,
        x: &Self::Param,
    ) -> Result<(Self::Output, Self::Gradient), Self::Error> {
        self.problem.cost_and_gradient(x)
    }
}
impl<P, C> Hessian for Constrained<P, C>
where
    P: Hessian<Constraints = NoConstraints>,
    C: ConstraintDescription,
{
    type Hessian = P::Hessian;
    fn hessian(&self, x: &Self::Param) -> Result<Self::Hessian, Self::Error> {
        self.problem.hessian(x)
    }
    fn cost_and_gradient_and_hessian(
        &self,
        x: &Self::Param,
    ) -> Result<(Self::Output, Self::Gradient, Self::Hessian), Self::Error>
    {
        self.problem.cost_and_gradient_and_hessian(x)
    }
}
impl<P, C> HessianProduct for Constrained<P, C>
where
    P: HessianProduct<Constraints = NoConstraints>,
    C: ConstraintDescription,
{
    fn hessian_product(
        &self,
        x: &Self::Param,
        v: &Self::Param,
    ) -> Result<Self::Gradient, Self::Error> {
        self.problem.hessian_product(x, v)
    }
}
impl<I: crate::NativeConvergenceDiagnostics> crate::NativeConvergenceDiagnostics
    for RelaxedInner<I>
{
    fn native_convergence_tests(&self) -> &[crate::NativeConvergenceTest] {
        self.inner.native_convergence_tests()
    }
}
