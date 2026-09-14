//! Solver-owned models for the state API experiment.

use super::state::{FirstOrderState, PointState, SimplexState};
use basin::core::math::{
    Dot, MatVec, MatrixIdentity, NegInPlace, NormSquared, Scalar, ScaleInPlace,
    ScaledAdd, VectorLen,
};
use basin::core::rng::{ChaCha8Rng, RngExt, SeedableRng};
use basin::line_search::{LineSearch, LineSearchOutcome, MoreThuente, Wolfe};
use basin::solver::simulated_annealing::{Neighbor, TemperatureSchedule};
use basin::{
    CostFunction, Gradient, Problem, Solver, State, TerminationReason,
};
use std::marker::PhantomData;

// Basin's rank-update trait and DenseMatrix mutation are private. Keeping this
// adapter local lets the external prototype exercise the same operation without
// expanding the stable math API or rebuilding the matrix on every update.
pub trait RankUpdate<V, F> {
    fn general_rank_one_update(&mut self, alpha: F, u: &V, v: &V);
}

pub struct Dense<F: Scalar> {
    data: Vec<F>,
    n: usize,
}

impl<F: Scalar> Dense<F> {
    pub fn nrows(&self) -> usize {
        self.n
    }
    pub fn data(&self) -> &[F] {
        &self.data
    }
}

impl<F: Scalar> MatrixIdentity for Dense<F> {
    fn identity(n: usize) -> Self {
        let mut data = vec![F::zero(); n * n];
        for i in 0..n {
            data[i * n + i] = F::one();
        }
        Self { data, n }
    }
}
impl<F: Scalar> MatVec<Vec<F>> for Dense<F> {
    fn matvec(&self, x: &Vec<F>) -> Vec<F> {
        assert_eq!(x.len(), self.n);
        if self.n == 0 {
            return Vec::new();
        }
        self.data
            .chunks(self.n)
            .map(|row| row.iter().zip(x).map(|(&a, &b)| a * b).sum())
            .collect()
    }
}
impl<F: Scalar> ScaleInPlace<F> for Dense<F> {
    fn scale_in_place(&mut self, alpha: F) {
        for x in &mut self.data {
            *x = *x * alpha;
        }
    }
}
impl<F: Scalar> RankUpdate<Vec<F>, F> for Dense<F> {
    fn general_rank_one_update(&mut self, alpha: F, u: &Vec<F>, v: &Vec<F>) {
        assert_eq!(u.len(), self.n);
        assert_eq!(v.len(), self.n);
        if self.n == 0 {
            return;
        }
        for (row, &ui) in self.data.chunks_mut(self.n).zip(u) {
            for (a, &vj) in row.iter_mut().zip(v) {
                *a = *a + alpha * ui * vj;
            }
        }
    }
}

#[cfg(feature = "nalgebra_all")]
impl<F: Scalar + crate::backend_aliases::nalgebra::Scalar>
    RankUpdate<crate::backend_aliases::nalgebra::DVector<F>, F>
    for crate::backend_aliases::nalgebra::DMatrix<F>
{
    fn general_rank_one_update(
        &mut self,
        alpha: F,
        u: &crate::backend_aliases::nalgebra::DVector<F>,
        v: &crate::backend_aliases::nalgebra::DVector<F>,
    ) {
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                self[(i, j)] = self[(i, j)] + alpha * u[i] * v[j];
            }
        }
    }
}
#[cfg(feature = "ndarray_all")]
impl<F: Scalar> RankUpdate<crate::backend_aliases::ndarray::Array1<F>, F>
    for crate::backend_aliases::ndarray::Array2<F>
{
    fn general_rank_one_update(
        &mut self,
        alpha: F,
        u: &crate::backend_aliases::ndarray::Array1<F>,
        v: &crate::backend_aliases::ndarray::Array1<F>,
    ) {
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                self[(i, j)] = self[(i, j)] + alpha * u[i] * v[j];
            }
        }
    }
}
#[cfg(feature = "faer_all")]
impl<F: Scalar> RankUpdate<crate::backend_aliases::faer::Col<F>, F>
    for crate::backend_aliases::faer::Mat<F>
{
    fn general_rank_one_update(
        &mut self,
        alpha: F,
        u: &crate::backend_aliases::faer::Col<F>,
        v: &crate::backend_aliases::faer::Col<F>,
    ) {
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                self[(i, j)] = self[(i, j)] + alpha * u[i] * v[j];
            }
        }
    }
}

/// A default matrix association fixes inference without fixing the matrix type.
/// The explicit `M` parameter on BFGS still allows custom matrix storage.
pub trait Backend<F: Scalar>: VectorLen {
    type Matrix;
}
impl<F: Scalar> Backend<F> for Vec<F> {
    type Matrix = Dense<F>;
}

#[cfg(feature = "nalgebra_all")]
impl<F: Scalar + crate::backend_aliases::nalgebra::Scalar> Backend<F>
    for crate::backend_aliases::nalgebra::DVector<F>
{
    type Matrix = crate::backend_aliases::nalgebra::DMatrix<F>;
}
#[cfg(feature = "ndarray_all")]
impl<F: Scalar> Backend<F> for crate::backend_aliases::ndarray::Array1<F> {
    type Matrix = crate::backend_aliases::ndarray::Array2<F>;
}
#[cfg(feature = "faer_all")]
impl<F: Scalar> Backend<F> for crate::backend_aliases::faer::Col<F> {
    type Matrix = crate::backend_aliases::faer::Mat<F>;
}

/// BFGS's existing update, with the matrix and scaling flag in the solver.
///
/// # Backends
///
/// Exercised by this prototype: Vec, nalgebra, ndarray, and faer.
pub struct Bfgs<
    V: Backend<F>,
    F: Scalar = f64,
    M = <V as Backend<F>>::Matrix,
    L = Wolfe<F>,
> {
    matrix: Option<M>,
    scaled: bool,
    search: L,
    make_search: fn() -> L,
    epsilon: F,
    param: PhantomData<V>,
}

impl<V: Backend<F>, F: Scalar> Bfgs<V, F> {
    pub fn new() -> Self {
        Self::with_search(Wolfe::new)
    }
}

impl<V: Backend<F>, F: Scalar, M, L> Bfgs<V, F, M, L> {
    pub fn with_search(make_search: fn() -> L) -> Self {
        Self {
            matrix: None,
            scaled: false,
            search: make_search(),
            make_search,
            epsilon: F::from_f64(1e-10).unwrap(),
            param: PhantomData,
        }
    }

    pub fn inverse_hessian(&self) -> Option<&M> {
        self.matrix.as_ref()
    }
}

impl<V: Backend<F> + Clone, F: Scalar, M, L> basin::InitialState<V>
    for Bfgs<V, F, M, L>
{
    type State = FirstOrderState<V, F>;
    fn seed(&self, point: &V) -> Self::State {
        FirstOrderState::new(point.clone())
    }
}

impl<P, V, F, M, L> Solver<P, FirstOrderState<V, F>> for Bfgs<V, F, M, L>
where
    F: Scalar,
    V: Backend<F>
        + Clone
        + Dot<F>
        + NormSquared<F>
        + ScaledAdd<F>
        + ScaleInPlace<F>
        + NegInPlace,
    M: MatrixIdentity + MatVec<V> + ScaleInPlace<F> + RankUpdate<V, F>,
    P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V>,
    L: LineSearch<P, V, F, Error = P::Error>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: FirstOrderState<V, F>,
    ) -> Result<FirstOrderState<V, F>, P::Error> {
        self.matrix = Some(M::identity(state.param().vec_len()));
        self.scaled = false;
        self.search = (self.make_search)();
        let x = state.param().clone();
        let (cost, gradient) = problem.cost_and_gradient(&x)?;
        state.replace(x, cost, gradient).unwrap();
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: FirstOrderState<V, F>,
    ) -> Result<(FirstOrderState<V, F>, Option<TerminationReason>), P::Error>
    {
        let (mut x, cost, gradient) = state.take_current().unwrap();
        let matrix = self.matrix.as_mut().unwrap();
        let mut direction = matrix.matvec(&gradient);
        direction.neg_in_place();
        let alpha = match self
            .search
            .next_with_outcome(problem, &x, cost, &gradient, &direction)?
        {
            LineSearchOutcome::Step(alpha) => alpha,
            _ => {
                state.replace(x, cost, gradient).unwrap();
                return Ok((state, Some(TerminationReason::SolverFailed)));
            }
        };
        if !(alpha.is_finite() && alpha > F::zero()) {
            state.replace(x, cost, gradient).unwrap();
            return Ok((state, Some(TerminationReason::SolverConverged)));
        }
        let mut s = direction;
        s.scale_in_place(alpha);
        x.scaled_add(F::one(), &s);
        let (cost_new, gradient_new) = problem.cost_and_gradient(&x)?;
        let mut y = gradient_new.clone();
        y.scaled_add(-F::one(), &gradient);
        let sy = s.dot(&y);
        if sy > self.epsilon * s.norm_squared().sqrt() * y.norm_squared().sqrt()
        {
            if !self.scaled {
                let yy = y.dot(&y);
                if yy > F::zero() {
                    *matrix = M::identity(x.vec_len());
                    matrix.scale_in_place(sy / yy);
                }
                self.scaled = true;
            }
            let rho = F::one() / sy;
            let hy = matrix.matvec(&y);
            let coef = rho * (F::one() + rho * y.dot(&hy));
            matrix.general_rank_one_update(coef, &s, &s);
            matrix.general_rank_one_update(-rho, &s, &hy);
            matrix.general_rank_one_update(-rho, &hy, &s);
        }
        state.replace(x, cost_new, gradient_new).unwrap();
        Ok((state, None))
    }
}

struct Pair<V, F> {
    s: V,
    y: V,
    sy: F,
}

/// Unbounded L-BFGS, retaining only the pairs needed by two-loop recursion.
///
/// # Backends
///
/// Exercised by this prototype: Vec, nalgebra, ndarray, and faer.
pub struct Lbfgs<V, F: Scalar = f64, L = MoreThuente<F>> {
    pairs: Vec<Pair<V, F>>,
    capacity: usize,
    theta: F,
    search: L,
    make_search: fn() -> L,
}

impl<V, F: Scalar> Lbfgs<V, F> {
    pub fn new(capacity: usize) -> Self {
        Self::with_search(capacity, MoreThuente::new)
    }
}

impl<V: Clone + VectorLen, F: Scalar, L> basin::InitialState<V>
    for Lbfgs<V, F, L>
{
    type State = FirstOrderState<V, F>;
    fn seed(&self, point: &V) -> Self::State {
        FirstOrderState::new(point.clone())
    }
}

impl<V, F: Scalar, L> Lbfgs<V, F, L> {
    pub fn with_search(capacity: usize, make_search: fn() -> L) -> Self {
        assert!(capacity > 0);
        Self {
            pairs: Vec::with_capacity(capacity),
            capacity,
            theta: F::one(),
            search: make_search(),
            make_search,
        }
    }

    pub fn history_len(&self) -> usize {
        self.pairs.len()
    }
}

impl<P, V, F, L> Solver<P, FirstOrderState<V, F>> for Lbfgs<V, F, L>
where
    F: Scalar,
    V: Clone + VectorLen + Dot<F> + ScaledAdd<F> + ScaleInPlace<F> + NegInPlace,
    P: CostFunction<Param = V, Output = F> + Gradient<Gradient = V>,
    L: LineSearch<P, V, F, Error = P::Error>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: FirstOrderState<V, F>,
    ) -> Result<FirstOrderState<V, F>, P::Error> {
        self.pairs.clear();
        self.theta = F::one();
        self.search = (self.make_search)();
        let x = state.param().clone();
        let (cost, gradient) = problem.cost_and_gradient(&x)?;
        state.replace(x, cost, gradient).unwrap();
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: FirstOrderState<V, F>,
    ) -> Result<(FirstOrderState<V, F>, Option<TerminationReason>), P::Error>
    {
        let (mut x, cost, gradient) = state.take_current().unwrap();
        let mut direction = gradient.clone();
        let mut alpha = vec![F::zero(); self.pairs.len()];
        for (pair, a) in self.pairs.iter().zip(&mut alpha).rev() {
            *a = pair.s.dot(&direction) / pair.sy;
            direction.scaled_add(-*a, &pair.y);
        }
        direction.scale_in_place(F::one() / self.theta);
        for (pair, a) in self.pairs.iter().zip(alpha) {
            let beta = pair.y.dot(&direction) / pair.sy;
            direction.scaled_add(a - beta, &pair.s);
        }
        direction.neg_in_place();
        let gdold = gradient.dot(&direction);
        let result = self
            .search
            .next_with_evaluation(problem, &x, cost, &gradient, &direction)?;
        let step = match result.outcome {
            LineSearchOutcome::Step(step) => step,
            _ => F::zero(),
        };
        if !(step.is_finite() && step > F::zero()) {
            state.replace(x, cost, gradient).unwrap();
            return Ok((state, Some(TerminationReason::SolverFailed)));
        }
        let (cost_new, gradient_new) =
            if let Some(evaluation) = result.evaluation {
                x = evaluation.param;
                (evaluation.cost, evaluation.gradient)
            } else {
                x.scaled_add(step, &direction);
                problem.cost_and_gradient(&x)?
            };
        let mut s = direction;
        s.scale_in_place(step);
        let mut y = gradient_new.clone();
        y.scaled_add(-F::one(), &gradient);
        let sy = s.dot(&y);
        let yy = y.dot(&y);
        if sy > F::epsilon() * (-gdold * step).abs()
            && sy.is_finite()
            && yy.is_finite()
        {
            if self.pairs.len() == self.capacity {
                self.pairs.remove(0);
            }
            self.theta = yy / sy;
            self.pairs.push(Pair { s, y, sy });
        }
        state.replace(x, cost_new, gradient_new).unwrap();
        Ok((state, None))
    }
}

/// Classical Nelder-Mead with authoritative vertices in the shared state.
///
/// # Backends
///
/// Exercised by this prototype: Vec, nalgebra, ndarray, and faer.
pub struct NelderMead<V, F: Scalar = f64> {
    scratch: Vec<V>,
    float: PhantomData<F>,
}

impl<V, F: Scalar> NelderMead<V, F> {
    pub fn new() -> Self {
        Self {
            scratch: Vec::new(),
            float: PhantomData,
        }
    }
    pub fn scratch(&self) -> &[V] {
        &self.scratch
    }
}

impl<P, V, F> Solver<P, SimplexState<V, F>> for NelderMead<V, F>
where
    F: Scalar,
    V: Clone + VectorLen + ScaleInPlace<F> + ScaledAdd<F>,
    P: CostFunction<Param = V, Output = F>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: SimplexState<V, F>,
    ) -> Result<SimplexState<V, F>, P::Error> {
        let (points, _) = state.take_members();
        self.scratch = vec![points[0].clone(); 3];
        let costs = points
            .iter()
            .map(|x| problem.cost(x))
            .collect::<Result<Vec<_>, _>>()?;
        state.replace_members(points, costs).unwrap();
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: SimplexState<V, F>,
    ) -> Result<(SimplexState<V, F>, Option<TerminationReason>), P::Error> {
        let (mut points, mut costs) = state.take_members();
        let n = points.len() - 1;
        let one = F::one();
        let half = F::from_f64(0.5).unwrap();
        let (centroid, rest) = self.scratch.split_at_mut(1);
        let (reflected, alternate) = rest.split_at_mut(1);
        let center = &mut centroid[0];
        let reflected = &mut reflected[0];
        let alternate = &mut alternate[0];
        center.scale_in_place(F::zero());
        for point in &points[..n] {
            center.scaled_add(one / F::from_usize(n).unwrap(), point);
        }
        affine(reflected, center, &points[n], -one);
        let fr = problem.cost(reflected)?;
        let mut shrink = false;
        if costs[0] <= fr && fr < costs[n - 1] {
            std::mem::swap(&mut points[n], reflected);
            costs[n] = fr;
        } else if fr < costs[0] {
            affine(alternate, center, reflected, one + one);
            let fe = problem.cost(alternate)?;
            if fe < fr {
                std::mem::swap(&mut points[n], alternate);
                costs[n] = fe;
            } else {
                std::mem::swap(&mut points[n], reflected);
                costs[n] = fr;
            }
        } else {
            let outside = fr < costs[n];
            affine(
                alternate,
                center,
                if outside { reflected } else { &points[n] },
                half,
            );
            let fc = problem.cost(alternate)?;
            if (outside && fc <= fr) || (!outside && fc < costs[n]) {
                std::mem::swap(&mut points[n], alternate);
                costs[n] = fc;
            } else {
                shrink = true;
            }
        }
        if shrink {
            let (best, others) = points.split_at_mut(1);
            for (point, cost) in others.iter_mut().zip(&mut costs[1..]) {
                point.scale_in_place(half);
                point.scaled_add(half, &best[0]);
                *cost = problem.cost(point)?;
            }
        }
        state.replace_members(points, costs).unwrap();
        Ok((state, None))
    }
}

fn affine<V: ScaleInPlace<F> + ScaledAdd<F>, F: Scalar>(
    out: &mut V,
    a: &V,
    b: &V,
    t: F,
) {
    out.scale_in_place(F::zero());
    out.scaled_add(F::one() - t, a);
    out.scaled_add(t, b);
}

/// Metropolis annealing using Basin's neighbor and temperature interfaces.
///
/// # Backends
///
/// Exercised by this prototype: Vec, nalgebra, ndarray, and faer.
pub struct Annealing<N, F: Scalar = f64> {
    neighbor: N,
    make_neighbor: fn() -> N,
    rng: ChaCha8Rng,
    seed: u64,
    age: u64,
    initial_temperature: F,
    schedule: TemperatureSchedule<F>,
}

impl<N, F: Scalar> Annealing<N, F> {
    pub fn new(
        make_neighbor: fn() -> N,
        initial_temperature: F,
        schedule: TemperatureSchedule<F>,
        seed: u64,
    ) -> Self {
        assert!(
            initial_temperature.is_finite() && initial_temperature > F::zero()
        );
        Self {
            neighbor: make_neighbor(),
            make_neighbor,
            rng: ChaCha8Rng::seed_from_u64(seed),
            seed,
            age: 0,
            initial_temperature,
            schedule,
        }
    }

    pub fn seed_chain(&self, seed: u64) -> Self {
        Self::new(
            self.make_neighbor,
            self.initial_temperature,
            self.schedule,
            seed,
        )
    }

    pub fn temperature(&self) -> F {
        self.schedule
            .temperature(self.initial_temperature, self.age)
    }
    pub fn neighbor(&self) -> &N {
        &self.neighbor
    }
}

impl<P, V, N, F> Solver<P, PointState<V, F>> for Annealing<N, F>
where
    F: Scalar,
    V: Clone + VectorLen,
    P: CostFunction<Param = V, Output = F>,
    N: Neighbor<V, F, ChaCha8Rng, Error = P::Error>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PointState<V, F>,
    ) -> Result<PointState<V, F>, P::Error> {
        self.neighbor = (self.make_neighbor)();
        self.rng = ChaCha8Rng::seed_from_u64(self.seed);
        self.age = 0;
        let cost = problem.cost(state.param())?;
        state.replace(state.param().clone(), cost);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PointState<V, F>,
    ) -> Result<(PointState<V, F>, Option<TerminationReason>), P::Error> {
        let temperature = self.temperature();
        let candidate =
            self.neighbor
                .propose(state.param(), temperature, &mut self.rng)?;
        let cost = problem.cost(&candidate)?;
        if !cost.is_nan() && cost != F::infinity() {
            let accept = cost <= state.cost()
                || F::from_f64(self.rng.random::<f64>()).unwrap()
                    < (-(cost - state.cost()) / temperature).exp();
            if accept {
                state.replace(candidate, cost);
            }
        }
        self.age += 1;
        Ok((state, None))
    }
}
