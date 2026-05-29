use core::marker::PhantomData;

use crate::core::constraint::BoxConstraints;
use crate::core::math::{ClampInPlace, Scalar, ScaledAdd};
use crate::core::problem::{CostFunction, Problem};
use crate::core::solver::Solver;
use crate::core::state::BasicSimplexState;
use crate::core::termination::TerminationReason;

/// Nelder-Mead simplex method (derivative-free).
///
/// Implements the algorithm as stated in Lagarias et al. (1998) with the
/// adaptive parameter option of Gao & Han (2012). The four parameters are:
/// `α` (reflection), `β` (expansion), `γ` (contraction), `δ` (shrink), with
/// the constraints `α > 0`, `β > 1`, `0 < γ < 1`, `0 < δ < 1`.
///
/// # Bounds
///
/// `NelderMead` is generic over a type-state [`Mode`](Unbounded) marker
/// that switches between the unconstrained algorithm ([`Unbounded`], the
/// default) and the projection-style box-constrained variant
/// ([`Projected`]). Construct unbounded NM with [`standard`](Self::standard),
/// [`adaptive`](Self::adaptive), or [`with_params`](Self::with_params), then
/// transition with [`projected`](Self::projected) when the problem carries
/// box bounds. The projected `Solver` impl requires `P: BoxConstraints`
/// and `V: ClampInPlace`, so handing a non-bounded problem to a projected
/// `NelderMead` is a compile-time error per AGENTS.md tenet 4.
///
/// # Backends
///
/// Backend-generic — works with any `V` implementing
/// [`ScaledAdd<F>`](crate::core::math::ScaledAdd) + `Clone`, paired
/// with a [`BasicSimplexState<V, F>`]. With the default `F = f64` that
/// covers `Vec<f64>`, `nalgebra::DVector<f64>` (feature `nalgebra`),
/// `ndarray::Array1<f64>` (feature `ndarray`), and `faer::Col<f64>`
/// (feature `faer`). The projected variant additionally requires
/// [`ClampInPlace`] on `V`, which every shipped backend implements.
///
/// # Examples
///
/// Derivative-free minimization of Rosenbrock — Nelder–Mead needs only
/// [`CostFunction`] and iterates a [`BasicSimplexState`] seeded from a
/// single point (the initial simplex is built automatically):
///
/// ```
/// use basin::{BasicSimplexState, CostFunction, Executor, NelderMead, SimplexTolerance};
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
///     NelderMead::standard(),
///     BasicSimplexState::new(vec![-1.2, 1.0]),
/// )
/// .max_iter(1_000)
/// .terminate_on(SimplexTolerance::new(1e-10, 1e-10))
/// .run()
/// .unwrap();
/// assert!(result.cost() < 1e-6);
/// ```
pub struct NelderMead<Mode = Unbounded, F = f64> {
    config: ParamConfig<F>,
    /// Resolved parameters; populated by `init` once the dimension is known.
    params: Option<Params<F>>,
    /// Type-state marker; carries the mode at the type level only.
    _mode: PhantomData<fn() -> Mode>,
}

/// Type-state marker for unconstrained Nelder-Mead (the default).
/// Constructors live on `NelderMead<Unbounded>`; the `Solver` impl
/// makes no constraint requirements on the problem.
pub struct Unbounded;

/// Type-state marker for the projection-style box-constrained
/// Nelder-Mead variant. Obtain via
/// [`NelderMead::projected`](NelderMead::projected). The `Solver` impl
/// requires `P: BoxConstraints` and `V: ClampInPlace`.
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
/// onto the same boundary face — the simplex becomes degenerate and the
/// reflection step loses descent direction. This is a known weakness of
/// the projection variant; scipy ships it anyway because it works well
/// enough in practice. For tighter behavior near active bounds consider
/// a Globalized-and-Bounded Nelder-Mead variant (Luersen & Le Riche
/// 2004), which adds a restart heuristic on degeneracy.
pub struct Projected;

#[derive(Clone, Copy)]
struct Params<F> {
    alpha: F,
    beta: F,
    gamma: F,
    delta: F,
}

#[derive(Clone, Copy)]
enum ParamConfig<F> {
    Standard,
    Adaptive,
    Fixed(Params<F>),
}

impl<F: Scalar> NelderMead<Unbounded, F> {
    /// Standard parameters (Nelder & Mead 1965): α=1, β=2, γ=0.5, δ=0.5.
    pub fn standard() -> Self {
        Self {
            config: ParamConfig::Standard,
            params: None,
            _mode: PhantomData,
        }
    }

    /// Adaptive parameters from Gao & Han (2012), eq. (4.1):
    /// α=1, β=1+2/n, γ=0.75−1/(2n), δ=1−1/n, with `n` inferred from the
    /// initial simplex during `Solver::init`. Coincides with `standard()`
    /// when `n == 2`.
    pub fn adaptive() -> Self {
        Self {
            config: ParamConfig::Adaptive,
            params: None,
            _mode: PhantomData,
        }
    }

    /// Nelder-Mead with explicit reflection / expansion / contraction /
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
            params: None,
            _mode: PhantomData,
        }
    }

    /// Switch to the projection-style box-constrained variant
    /// ([`Projected`]). The algorithm parameters configured on this
    /// builder are preserved; the resulting solver requires the problem
    /// to implement [`BoxConstraints`] and projects every trial vertex
    /// element-wise into `[lower, upper]`. See the type-level rustdoc on
    /// [`Projected`] for the algorithm contract and limitations.
    pub fn projected(self) -> NelderMead<Projected, F> {
        NelderMead {
            config: self.config,
            params: self.params,
            _mode: PhantomData,
        }
    }
}

impl<Mode, F: Scalar> NelderMead<Mode, F> {
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

/// Build `(1 - t) * a + t * b` from two vectors and a scalar interpolant.
/// Works for any `t ∈ ℝ` — values outside `[0, 1]` extrapolate, which is
/// what reflection needs.
fn affine<V, F>(a: &V, b: &V, t: F) -> V
where
    V: Clone + ScaledAdd<F>,
    F: Scalar,
{
    let mut out = a.clone();
    out.scaled_add(-t, a);
    out.scaled_add(t, b);
    out
}

/// Centroid of `vertices` (mean of all entries).
fn centroid<V, F>(vertices: &[V]) -> V
where
    V: Clone + ScaledAdd<F>,
    F: Scalar,
{
    let inv = F::from_usize(vertices.len()).unwrap().recip();
    let mut c = vertices[0].clone();
    c.scaled_add(inv - F::one(), &vertices[0]);
    for v in &vertices[1..] {
        c.scaled_add(inv, v);
    }
    c
}

/// Sort `vertices` and `costs` jointly by ascending cost. NaN costs sort
/// last so a single bad evaluation can't drag itself to the front.
fn sort_simplex<V, F: PartialOrd>(vertices: &mut [V], costs: &mut [F]) {
    let n = vertices.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        costs[i]
            .partial_cmp(&costs[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    apply_permutation(vertices, &idx);
    apply_permutation(costs, &idx);
}

fn apply_permutation<T>(slice: &mut [T], idx: &[usize]) {
    let mut visited = vec![false; slice.len()];
    for start in 0..slice.len() {
        if visited[start] || idx[start] == start {
            visited[start] = true;
            continue;
        }
        let mut current = start;
        loop {
            let next = idx[current];
            visited[current] = true;
            if next == start {
                break;
            }
            slice.swap(current, next);
            current = next;
        }
    }
}

/// Evaluate every vertex's cost and sort the simplex ascending. Shared
/// between the `Unbounded` and `Projected` `Solver::init` paths after
/// any projection of the initial vertices.
fn init_costs_and_sort<P, V, F>(
    problem: &mut Problem<P>,
    state: &mut BasicSimplexState<V, F>,
) -> Result<(), P::Error>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>,
{
    for (v, c) in state.vertices.iter().zip(state.costs.iter_mut()) {
        *c = problem.cost(v)?;
    }
    sort_simplex(&mut state.vertices, &mut state.costs);
    Ok(())
}

/// One Nelder-Mead iteration, parameterised by a projection closure.
///
/// The `Unbounded` `Solver` impl passes a no-op closure; the `Projected`
/// impl passes one that clamps into `[lower, upper]`. Vertices are
/// sorted (best at index 0) on entry; the invariant is restored before
/// returning. The simplex has `n + 1` vertices in `n`-D.
#[allow(clippy::type_complexity)]
fn next_iter_inner<P, V, F, Proj>(
    problem: &mut Problem<P>,
    mut state: BasicSimplexState<V, F>,
    p: Params<F>,
    project: &Proj,
) -> Result<(BasicSimplexState<V, F>, Option<TerminationReason>), P::Error>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>,
    V: Clone + ScaledAdd<F>,
    Proj: Fn(&mut V),
{
    let m = state.vertices.len();
    let n = m - 1;
    let worst = m - 1;

    let x_bar = centroid(&state.vertices[..n]);

    let f1 = state.costs[0];
    let fn_ = state.costs[n - 1];
    let fnp1 = state.costs[worst];

    // Reflection: x_r = x_bar + α(x_bar − x_{n+1}) = (1+α)·x_bar − α·x_{n+1}
    let mut x_r = affine(&x_bar, &state.vertices[worst], -p.alpha);
    project(&mut x_r);
    let fr = problem.cost(&x_r)?;

    if f1 <= fr && fr < fn_ {
        // Accept reflection.
        state.vertices[worst] = x_r;
        state.costs[worst] = fr;
    } else if fr < f1 {
        // Try expansion: x_e = x_bar + β(x_r − x_bar).
        let mut x_e = affine(&x_bar, &x_r, p.beta);
        project(&mut x_e);
        let fe = problem.cost(&x_e)?;
        if fe < fr {
            state.vertices[worst] = x_e;
            state.costs[worst] = fe;
        } else {
            state.vertices[worst] = x_r;
            state.costs[worst] = fr;
        }
    } else if fr < fnp1 {
        // fn ≤ fr < f_{n+1}: outside contraction.
        // x_oc = x_bar + γ(x_r − x_bar).
        let mut x_oc = affine(&x_bar, &x_r, p.gamma);
        project(&mut x_oc);
        let foc = problem.cost(&x_oc)?;
        if foc <= fr {
            state.vertices[worst] = x_oc;
            state.costs[worst] = foc;
        } else {
            shrink_inner(problem, &mut state, p.delta, project)?;
        }
    } else {
        // fr ≥ f_{n+1}: inside contraction.
        // x_ic = x_bar − γ(x_bar − x_{n+1}) = (1−γ)·x_bar + γ·x_{n+1}.
        let mut x_ic = affine(&x_bar, &state.vertices[worst], p.gamma);
        project(&mut x_ic);
        let fic = problem.cost(&x_ic)?;
        if fic < fnp1 {
            state.vertices[worst] = x_ic;
            state.costs[worst] = fic;
        } else {
            shrink_inner(problem, &mut state, p.delta, project)?;
        }
    }

    sort_simplex(&mut state.vertices, &mut state.costs);
    Ok((state, None))
}

fn shrink_inner<P, V, F, Proj>(
    problem: &mut Problem<P>,
    state: &mut BasicSimplexState<V, F>,
    delta: F,
    project: &Proj,
) -> Result<(), P::Error>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>,
    V: Clone + ScaledAdd<F>,
    Proj: Fn(&mut V),
{
    // Best vertex is fixed at index 0; shrink every other vertex toward it.
    // Split-borrow lets us read x[0] while mutating x[i].
    let (best_slice, rest) = state.vertices.split_at_mut(1);
    let best = &best_slice[0];
    for (v, c) in rest.iter_mut().zip(&mut state.costs[1..]) {
        let mut new_v = affine(best, v, delta);
        project(&mut new_v);
        *v = new_v;
        *c = problem.cost(v)?;
    }
    Ok(())
}

impl<P, V, F> Solver<P, BasicSimplexState<V, F>> for NelderMead<Unbounded, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F>,
    V: Clone + ScaledAdd<F>,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: BasicSimplexState<V, F>,
    ) -> Result<BasicSimplexState<V, F>, Self::Error> {
        let n = state.vertices.len() - 1;
        self.params = Some(Self::resolve(self.config, n));
        init_costs_and_sort(problem, &mut state)?;
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: BasicSimplexState<V, F>,
    ) -> Result<(BasicSimplexState<V, F>, Option<TerminationReason>), Self::Error> {
        let p = self
            .params
            .expect("NelderMead::init must run before next_iter");
        next_iter_inner(problem, state, p, &|_: &mut V| {})
    }
}

impl<P, V, F> Solver<P, BasicSimplexState<V, F>> for NelderMead<Projected, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + BoxConstraints,
    V: Clone + ScaledAdd<F> + ClampInPlace,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: BasicSimplexState<V, F>,
    ) -> Result<BasicSimplexState<V, F>, Self::Error> {
        let n = state.vertices.len() - 1;
        self.params = Some(Self::resolve(self.config, n));
        // Project every initial vertex once so iter-0 termination
        // checks see a feasible simplex (mirrors
        // ProjectedGradientDescent::init's project-an-infeasible-start
        // pattern).
        let lo = problem.inner().lower().clone();
        let hi = problem.inner().upper().clone();
        for v in state.vertices.iter_mut() {
            v.clamp_in_place(&lo, &hi);
        }
        init_costs_and_sort(problem, &mut state)?;
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        state: BasicSimplexState<V, F>,
    ) -> Result<(BasicSimplexState<V, F>, Option<TerminationReason>), Self::Error> {
        let p = self
            .params
            .expect("NelderMead::init must run before next_iter");
        let lo = problem.inner().lower().clone();
        let hi = problem.inner().upper().clone();
        next_iter_inner(problem, state, p, &|v: &mut V| v.clamp_in_place(&lo, &hi))
    }
}
