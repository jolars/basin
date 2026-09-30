use rand::seq::SliceRandom;

use crate::core::inner::InitialState;
use crate::core::math::{Scalar, ScaleInPlace, ScaledAdd};
use crate::core::problem::{CostFunction, MiniBatchGradient, Problem};
use crate::core::rng::{ChaCha8Rng, SeedableRng};
use crate::core::solver::Solver;
use crate::core::state::{PointState, State};

/// Vanilla mini-batch stochastic gradient descent (SGD) with a constant
/// learning rate and optional heavy-ball momentum.
///
/// Each step draws a mini-batch of `batch_size` indices from a permutation
/// the solver maintains, calls [`MiniBatchGradient::batch_gradient`] to
/// get the averaged batch gradient, and takes a step `x ← x − α·g` (or,
/// with momentum, a heavy-ball step).
///
/// # Sampling
///
/// Epoch-shuffle without replacement (the standard PyTorch/JAX/Bottou
/// 2012 convention):
///
/// 1. At [`Solver::init`], the solver builds a permutation
///    `perm = [0, 1, …, n−1]` and Fisher-Yates–shuffles it with its
///    [`ChaCha8Rng`].
/// 2. Each [`Solver::next_iter`] consumes the next contiguous slice of
///    `batch_size` indices from `perm`.
/// 3. When fewer than `batch_size` indices remain in the current epoch,
///    the solver reshuffles `perm` and starts a new epoch: `drop_last`
///    behavior (the short tail is discarded). Keeps every step's batch
///    size *exactly* `batch_size`, so the learning-rate interpretation is
///    stable across the run.
///
/// Same `seed` in, same iterate trajectory out (the reproducibility
/// contract every stochastic solver in basin honors). If `batch_size`
/// exceeds `n_samples()`, it is clamped to `n_samples()` once at
/// [`Solver::init`].
///
/// # Momentum
///
/// [`with_momentum`](Self::with_momentum) adds a heavy-ball velocity term
/// (Polyak 1964), identical in form to the variant on
/// [`GradientDescent`](crate::solver::GradientDescent). With momentum
/// coefficient `β` and learning rate `α` the update becomes
///
/// ```text
/// vₖ₊₁ = β · vₖ − α · ĝₖ
/// xₖ₊₁ = xₖ + vₖ₊₁
/// ```
///
/// starting from `v₀ = 0`, where `ĝₖ` is the *mini-batch* gradient.
/// `β = 0` (the default) is plain SGD; `β ∈ (0, 1)` (commonly `0.9`)
/// cancels noisy oscillations across iterations and accelerates
/// convergence along consistent gradient directions. A too-large
/// effective step (roughly `α / (1 − β)`) diverges, so reduce `α` when
/// adding momentum, the same stability caveat as the full-batch case.
///
/// # Cost tracking
///
/// The shared [`PointState`] publishes the latest fully evaluated point and
/// its matching objective. The solver evaluates the seed, then refreshes
/// progress every `n_samples / batch_size` mini-batch steps by default. The
/// short tail of each epoch is discarded. Use
/// [`with_cost_eval_every`](Self::with_cost_eval_every) to change this period.
///
/// Between refreshes, the working iterate advances inside the solver while
/// the published point and cost stay together. Ordinary results and observers
/// report that evaluated point, which can precede the latest mini-batch step.
/// Retain the solver with [`Executor::run_with_solver`](crate::Executor::run_with_solver)
/// to inspect [`working_param`](Self::working_param), or use a period of `1`
/// to publish every step. Reading either view performs no evaluations.
///
/// Iteration and batch-gradient budgets still count mini-batch steps. Cost-
/// and step-change convergence checks observe only objective refreshes;
/// objective targets and stalls use the retained evaluated incumbent.
/// Choose stall patience with the refresh period in mind. Batch gradients
/// are counted in the raw gradient category, but are never advertised as
/// derivatives of the published point. The state has no gradient capability.
///
/// Gradient-tolerance setters are therefore unavailable:
///
/// ```compile_fail,E0599
/// use basin::Sgd;
/// let solver = Sgd::<Vec<f64>>::new(0.1, 1, 42)
///     .with_absolute_gradient_tolerance(0.0);
/// ```
///
/// ```compile_fail,E0599
/// use basin::Sgd;
/// let solver = Sgd::<Vec<f64>>::new(0.1, 1, 42)
///     .with_relative_gradient_tolerance(0.0);
/// ```
///
/// ```compile_fail,E0599
/// use basin::Sgd;
/// let solver = Sgd::<Vec<f64>>::new(0.1, 1, 42)
///     .with_absolute_cost_change_tolerance(0.0)
///     .with_absolute_gradient_tolerance(0.0);
/// ```
///
/// ```compile_fail,E0599
/// use basin::Sgd;
/// let solver = Sgd::<Vec<f64>>::new(0.1, 1, 42)
///     .with_absolute_cost_change_tolerance(0.0)
///     .with_relative_gradient_tolerance(0.0);
/// ```
///
/// # Backends
///
/// `Vec<F>`, `nalgebra::DVector<F>`, `ndarray::Array1<F>`, and `faer::Col<F>`
/// with `F = f64` (default) or `f32`. Custom vectors need [`ScaledAdd<F>`],
/// [`ScaleInPlace<F>`], and `Clone`.
///
/// # Initialization and continuation
///
/// Fresh initialization resets progress, momentum, the seeded RNG, batch
/// order, and refresh phase, then evaluates the supplied point. An exact
/// solver-aware checkpoint preserves all of these, including the working
/// iterate between refreshes. A state-only snapshot starts afresh from its
/// last evaluated point. With `serde`, solver-aware serialization is available
/// when the parameter and scalar types support it.
///
/// # References
///
/// Robbins, H. & Monro, S. (1951). "A stochastic approximation method."
/// *Annals of Mathematical Statistics*, 22(3), 400–407.
/// [doi:10.1214/aoms/1177729586](https://doi.org/10.1214/aoms/1177729586).
///
/// Polyak, B. T. (1964). "Some methods of speeding up the convergence of
/// iteration methods." *USSR Computational Mathematics and Mathematical
/// Physics*, 4(5), 1–17.
/// [doi:10.1016/0041-5553(64)90137-5](https://doi.org/10.1016/0041-5553(64)90137-5).
///
/// Bottou, L. (2012). "Stochastic gradient descent tricks." In *Neural
/// Networks: Tricks of the Trade* (2nd ed., pp. 421–436). Springer.
/// [doi:10.1007/978-3-642-35289-8_25](https://doi.org/10.1007/978-3-642-35289-8_25).
///
/// # Examples
///
/// Fit a linear model `y = a·x` by minimizing `(1/n) Σ (aᵢ x − yᵢ)²`,
/// with a fixed learning rate, mini-batch size 4, and momentum 0.9:
///
/// ```
/// use basin::{
///     PointState, CostFunction, Executor, MiniBatchGradient,
///     Sgd,
/// };
///
/// struct LinReg {
///     rows: Vec<Vec<f64>>,
///     y: Vec<f64>,
/// }
/// impl CostFunction for LinReg {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
///         let n = self.rows.len() as f64;
///         let mut s = 0.0;
///         for (a, &yi) in self.rows.iter().zip(self.y.iter()) {
///             let r = a.iter().zip(x).map(|(ai, xi)| ai * xi).sum::<f64>() - yi;
///             s += r * r;
///         }
///         Ok(s / n)
///     }
/// }
/// impl MiniBatchGradient for LinReg {
///     type Gradient = Vec<f64>;
///     fn n_samples(&self) -> usize {
///         self.rows.len()
///     }
///     fn batch_gradient(
///         &self,
///         x: &Vec<f64>,
///         batch: &[usize],
///     ) -> Result<Vec<f64>, Self::Error> {
///         let inv = 2.0 / batch.len() as f64;
///         let mut g = vec![0.0; x.len()];
///         for &i in batch {
///             let a = &self.rows[i];
///             let r = a.iter().zip(x).map(|(ai, xi)| ai * xi).sum::<f64>() - self.y[i];
///             for (gj, aj) in g.iter_mut().zip(a) {
///                 *gj += inv * r * aj;
///             }
///         }
///         Ok(g)
///     }
/// }
///
/// let problem = LinReg {
///     rows: vec![vec![1.0, 2.0], vec![2.0, 1.0], vec![3.0, 4.0], vec![4.0, 3.0]],
///     y:    vec![5.0, 4.0, 11.0, 10.0],
/// };
/// let sgd = Sgd::new(0.02, 2, 0xC0FFEE).with_momentum(0.9);
/// let result = Executor::new(problem, sgd, PointState::new(vec![0.0, 0.0]))
///     .max_iter(2_000)
///     .run()
///     .unwrap();
/// assert!(result.cost() < 1e-6);
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Sgd<V, F: Scalar = f64> {
    alpha: F,
    working_param: Option<V>,
    batch_size: usize,
    seed: u64,
    /// Momentum coefficient `β`; `0.0` disables momentum (plain SGD).
    beta: F,
    /// Heavy-ball velocity `vₖ`. `None` until the first momentum step
    /// (treated as zero) and reset by [`init`](Solver::init) so a reused
    /// solver restarts from rest. Stays `None` when `β = 0`.
    velocity: Option<V>,
    /// Seeded at [`init`](Solver::init), held across iters. `None`
    /// before the first init.
    rng: Option<ChaCha8Rng>,
    /// Current epoch's permutation of `0..n_samples`. Reshuffled at every
    /// epoch boundary.
    perm: Vec<usize>,
    /// Position of the next batch's first index within `perm`.
    cursor: usize,
    /// `batch_size.min(n_samples)`, resolved at [`init`](Solver::init).
    effective_batch: usize,
    /// User-set cost-refresh period in iters; `None` means "use the
    /// epoch-boundary default" (`batches_per_epoch`, resolved at init).
    cost_eval_every: Option<usize>,
    /// Effective period in iters, resolved at [`init`](Solver::init).
    cost_period: usize,
    /// Iters since the last full cost eval; refresh when this reaches
    /// `cost_period`.
    iters_since_cost: usize,
}

impl<V, F: Scalar> Sgd<V, F> {
    /// Mini-batch SGD with a fixed learning rate `alpha`, batch size
    /// `batch_size`, and PRNG seed `seed`. Same seed in → same iterate
    /// trajectory out.
    ///
    /// `batch_size` must be `> 0`; it is clamped down to
    /// [`n_samples`](crate::core::problem::MiniBatchGradient::n_samples)
    /// at [`Solver::init`] if it exceeds the dataset size.
    pub fn new(alpha: F, batch_size: usize, seed: u64) -> Self {
        assert!(batch_size > 0, "Sgd: batch_size must be > 0");
        Self {
            alpha,
            working_param: None,
            batch_size,
            seed,
            beta: F::zero(),
            velocity: None,
            rng: None,
            perm: Vec::new(),
            cursor: 0,
            effective_batch: 0,
            cost_eval_every: None,
            cost_period: 1,
            iters_since_cost: 0,
        }
    }

    /// Enable Polyak heavy-ball momentum with coefficient `beta`.
    /// `beta = 0.0` is plain SGD; `beta` in `(0, 1)` (commonly `0.9`)
    /// adds momentum. See the [type docs](Self#momentum) for the update
    /// rule and stability caveat.
    pub fn with_momentum(mut self, beta: F) -> Self {
        self.beta = beta;
        self
    }

    /// Evaluate and publish the working point every `period` mini-batch steps.
    ///
    /// Defaults to `n_samples / batch_size`, with batch size clamped to the
    /// sample count. A period of `1` publishes every step at the cost of a
    /// full objective evaluation each time. Larger periods amortize that work
    /// and leave the reported point and cost at the last refresh until the
    /// next one. Cost- and step-change checks observe only those refreshes.
    /// Changing the setting takes effect on the next fresh initialization.
    ///
    /// # Panics
    ///
    /// Panics if `period` is zero.
    pub fn with_cost_eval_every(mut self, period: usize) -> Self {
        assert!(period > 0, "Sgd: cost_eval_every period must be > 0");
        self.cost_eval_every = Some(period);
        self
    }

    /// Latest mini-batch iterate, or `None` before initialization.
    ///
    /// Between objective refreshes this point has no reported cost and can
    /// differ from `state.param()`. Inspection does not evaluate it.
    pub fn working_param(&self) -> Option<&V> {
        self.working_param.as_ref()
    }
}

impl<V, F> InitialState<V> for Sgd<V, F>
where
    F: Scalar,
    V: Clone,
{
    type State = PointState<V, F>;
    fn seed(&self, x: &V) -> Self::State {
        PointState::new(x.clone())
    }
}

impl<P, V, F> Solver<P, PointState<V, F>> for Sgd<V, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + MiniBatchGradient<Gradient = V>,
    V: ScaledAdd<F> + ScaleInPlace<F> + Clone,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PointState<V, F>,
    ) -> Result<PointState<V, F>, Self::Error> {
        state.reset();
        self.working_param = Some(state.param().clone());
        // Start momentum from rest, even if this solver instance is reused
        // across runs (composition): velocity must not leak between runs.
        self.velocity = None;

        let n = problem.inner().n_samples();
        assert!(
            n > 0,
            "Sgd: problem.n_samples() == 0; no batches to draw from",
        );
        self.effective_batch = self.batch_size.min(n);

        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        self.perm = (0..n).collect();
        self.perm.as_mut_slice().shuffle(&mut rng);
        self.cursor = 0;
        self.rng = Some(rng);

        // Resolve cost-refresh period: user override, or default to
        // batches-per-epoch so cost is refreshed once at every epoch
        // boundary. `max(1)` covers degenerate cases (effective_batch
        // equals n_samples) where batches_per_epoch is 1.
        let batches_per_epoch = (n / self.effective_batch).max(1);
        self.cost_period = self.cost_eval_every.unwrap_or(batches_per_epoch);
        self.iters_since_cost = 0;

        let cost = problem.cost(state.param())?;
        state.replace(state.param().clone(), cost);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: PointState<V, F>,
    ) -> Result<crate::SolverStep<PointState<V, F>>, Self::Error> {
        let bs = self.effective_batch;
        let n = self.perm.len();

        // Epoch boundary: not enough indices left for a full batch.
        // Reshuffle and start over. `drop_last` semantics: any short tail
        // is discarded, so every step sees exactly `bs` samples.
        if self.cursor + bs > n {
            let rng = self
                .rng
                .as_mut()
                .expect("rng not set: Solver::init must run before next_iter");
            self.perm.as_mut_slice().shuffle(rng);
            self.cursor = 0;
        }

        let batch = &self.perm[self.cursor..self.cursor + bs];
        let working = self.working_param.as_mut().expect(
            "working iterate not set: Solver::init must run before next_iter",
        );
        let grad = problem.batch_gradient(working, batch)?;
        self.cursor += bs;

        if self.beta == F::zero() {
            // No momentum: x ← x − α·g. One fused pass via
            // `scaled_add(-α, &g)`, instead of materializing `direction = −g`
            // and stepping `x ← x + α·direction`; the latter touched the
            // dim-sized buffer twice per step.
            working.scaled_add(-self.alpha, &grad);
        } else {
            // Heavy ball: v ← β·v − α·g, then x ← x + v.
            // With v₀ = 0 the first step is just −α·g; form it by consuming
            // `grad` to avoid materializing a zero vector.
            let velocity = match self.velocity.take() {
                Some(mut v) => {
                    v.scale_in_place(self.beta);
                    v.scaled_add(-self.alpha, &grad);
                    v
                }
                None => {
                    let mut v = grad;
                    v.scale_in_place(-self.alpha);
                    v
                }
            };
            working.scaled_add(F::one(), &velocity);
            self.velocity = Some(velocity);
        }

        self.iters_since_cost += 1;
        if self.iters_since_cost >= self.cost_period {
            let cost = problem.cost(working)?;
            state.replace(working.clone(), cost);
            self.iters_since_cost = 0;
        }

        Ok(crate::SolverStep::from((state, None)))
    }

    fn should_check_iterate_change(&self) -> bool {
        self.iters_since_cost == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::state::State;
    use crate::{Executor, PointState};

    /// Finite-sum quadratic `f(x) = (1/n) Σᵢ ‖x − cᵢ‖²` with per-sample
    /// gradient `2·(x − cᵢ)`. The (unique) minimizer is the centroid of
    /// the `centers`. Used across most tests because it lets us write
    /// down the optimum in closed form and check both convergence and
    /// per-batch correctness.
    struct FiniteSumQuadratic {
        centers: Vec<Vec<f64>>,
    }

    impl FiniteSumQuadratic {
        fn centroid(&self) -> Vec<f64> {
            let d = self.centers[0].len();
            let n = self.centers.len() as f64;
            let mut c = vec![0.0; d];
            for ci in &self.centers {
                for (cj, &v) in c.iter_mut().zip(ci) {
                    *cj += v;
                }
            }
            for cj in &mut c {
                *cj /= n;
            }
            c
        }
    }

    impl CostFunction for FiniteSumQuadratic {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
            let n = self.centers.len() as f64;
            let mut s = 0.0;
            for c in &self.centers {
                for (xi, ci) in x.iter().zip(c) {
                    let d = xi - ci;
                    s += d * d;
                }
            }
            Ok(s / n)
        }
    }

    impl MiniBatchGradient for FiniteSumQuadratic {
        type Gradient = Vec<f64>;
        fn n_samples(&self) -> usize {
            self.centers.len()
        }
        fn batch_gradient(
            &self,
            x: &Vec<f64>,
            batch: &[usize],
        ) -> Result<Vec<f64>, Self::Error> {
            let d = x.len();
            let inv = 2.0 / batch.len() as f64;
            let mut g = vec![0.0; d];
            for &i in batch {
                let c = &self.centers[i];
                for (gj, (xj, cj)) in g.iter_mut().zip(x.iter().zip(c)) {
                    *gj += inv * (xj - cj);
                }
            }
            Ok(g)
        }
    }

    fn problem_5_centers() -> FiniteSumQuadratic {
        FiniteSumQuadratic {
            centers: vec![
                vec![1.0, 0.0],
                vec![2.0, 1.0],
                vec![0.0, 2.0],
                vec![-1.0, 1.0],
                vec![3.0, -1.0],
            ],
        }
    }

    #[test]
    fn converges_to_centroid_without_momentum() {
        // SGD with a *constant* learning rate has an O(α) noise floor
        // around the optimum (the well-known constant-LR SGD result; see
        // Bottou 2012). It does not converge *to* the optimum without LR
        // decay, but it does enter and stay in a small neighborhood
        // proportional to α. Use a small batch_size to keep the test
        // honest about the stochastic regime, and a tolerance sized to
        // the noise floor rather than to a deterministic GD optimum.
        let problem = problem_5_centers();
        let centroid = problem.centroid();
        let sgd = Sgd::new(0.01, 2, 0xABCDEF);
        let result =
            Executor::new(problem, sgd, PointState::new(vec![0.0, 0.0]))
                .max_iter(3_000)
                .run()
                .unwrap();
        let x = result.param();
        for (xi, ci) in x.iter().zip(centroid.iter()) {
            assert!(
                (xi - ci).abs() < 5e-2,
                "x = {x:?}, centroid = {centroid:?}",
            );
        }
    }

    #[test]
    fn full_batch_recovers_deterministic_gradient_descent() {
        // batch_size = n_samples → batch gradient equals the true full
        // gradient, every step is deterministic regardless of seed, and
        // a fixed-LR step converges geometrically on a strongly-convex
        // quadratic. Tighter tolerance than the noisy-SGD test above.
        let problem = problem_5_centers();
        let centroid = problem.centroid();
        let sgd = Sgd::new(0.1, problem.n_samples(), 0);
        let result =
            Executor::new(problem, sgd, PointState::new(vec![0.0, 0.0]))
                .max_iter(500)
                .run()
                .unwrap();
        let x = result.param();
        for (xi, ci) in x.iter().zip(centroid.iter()) {
            assert!((xi - ci).abs() < 1e-6, "x={x:?}, centroid={centroid:?}");
        }
    }

    #[test]
    fn same_seed_same_trajectory() {
        let problem_a = problem_5_centers();
        let problem_b = problem_5_centers();
        let run = |p: FiniteSumQuadratic| {
            let sgd = Sgd::new(0.05, 2, 12345);
            Executor::new(p, sgd, PointState::new(vec![0.5, -0.5]))
                .max_iter(50)
                .run()
                .unwrap()
                .param()
                .clone()
        };
        let xa = run(problem_a);
        let xb = run(problem_b);
        for (a, b) in xa.iter().zip(xb.iter()) {
            assert!((a - b).abs() < 1e-15, "xa={xa:?}, xb={xb:?}");
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let run = |seed: u64| {
            let sgd = Sgd::new(0.05, 2, seed);
            Executor::new(
                problem_5_centers(),
                sgd,
                PointState::new(vec![0.5, -0.5]),
            )
            .max_iter(20)
            .run()
            .unwrap()
            .param()
            .clone()
        };
        let xa = run(1);
        let xb = run(2);
        // Different seeds produce different batch orderings → different
        // trajectories after a handful of steps.
        let diff: f64 =
            xa.iter().zip(xb.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-6, "seeds 1 and 2 produced identical trajectory");
    }

    #[test]
    fn momentum_resets_between_runs() {
        // Reusing the same solver across two `Executor::run` calls must
        // restart from rest: identical iterate trajectories.
        let start = vec![1.0, 1.0];
        let mut sgd = Sgd::new(0.03, 2, 7).with_momentum(0.85);

        let run_once = |solver: &mut Sgd<Vec<f64>>| {
            let mut p = Problem::new(problem_5_centers());
            let mut state =
                solver.init(&mut p, PointState::new(start.clone())).unwrap();
            for _ in 0..15 {
                let (next, _, _) =
                    solver.next_iter(&mut p, state).unwrap().into_parts();
                state = next;
            }
            state.param().clone()
        };

        let first = run_once(&mut sgd);
        let second = run_once(&mut sgd);
        for (a, b) in first.iter().zip(second.iter()) {
            assert!(
                (a - b).abs() < 1e-15,
                "first={first:?}, second={second:?}"
            );
        }
    }

    #[test]
    fn reshuffles_at_epoch_boundary() {
        // n_samples = 7, batch_size = 3 → 2 batches per epoch, last
        // sample dropped. After step 2 (cursor would jump to 6 + 3 = 9,
        // which exceeds 7), the solver must reshuffle and reset cursor.
        // We check this indirectly: with momentum off and a fixed seed,
        // the trajectory must be deterministic and reach a state where
        // the post-epoch reshuffle has occurred.
        let problem = FiniteSumQuadratic {
            centers: (0..7).map(|i| vec![i as f64, -(i as f64)]).collect(),
        };
        let mut sgd = Sgd::new(0.01, 3, 99);
        let mut p = Problem::new(problem);
        let mut state =
            sgd.init(&mut p, PointState::new(vec![0.0, 0.0])).unwrap();
        // 3 steps: enough to trigger the reshuffle at step 3 (cursor
        // would be 6, and 6 + 3 > 7).
        for _ in 0..3 {
            let (next, _, _) =
                sgd.next_iter(&mut p, state).unwrap().into_parts();
            state = next;
        }
        // Cursor after step 3 should be 3 (we reshuffled before step 3,
        // then advanced by batch_size).
        assert_eq!(sgd.cursor, 3);
    }

    #[test]
    fn batch_size_clamped_to_n_samples() {
        // batch_size = 10 but n_samples = 3 → effective batch = 3. Should
        // run without panic and the effective batch matches every batch
        // covering the whole dataset. Same seed → reshuffle each step
        // (since cursor jumps 3 → 6 > 3 immediately), but result remains
        // deterministic.
        let problem = FiniteSumQuadratic {
            centers: vec![vec![1.0], vec![2.0], vec![3.0]],
        };
        let centroid = problem.centroid();
        let sgd = Sgd::new(0.05, 10, 13);
        let result = Executor::new(problem, sgd, PointState::new(vec![0.0]))
            .max_iter(500)
            .run()
            .unwrap();
        assert!((result.param()[0] - centroid[0]).abs() < 1e-3);
    }

    #[test]
    fn published_point_matches_its_cost_between_refreshes() {
        let result = Executor::from_start(
            problem_5_centers(),
            Sgd::new(0.05, 2, 42),
            vec![10.0, 10.0],
        )
        .max_iter(1)
        .run()
        .unwrap();
        assert_eq!(
            result.cost(),
            problem_5_centers().cost(result.param()).unwrap()
        );
    }

    #[test]
    fn cost_refresh_default_is_epoch_boundary() {
        // n_samples = 5, batch_size = 2 → batches_per_epoch = 2.
        // After 1 iter (still mid-epoch), state.cost must equal the
        // *initial* cost; the default schedule does not refresh until
        // the second iter wraps an epoch.
        let problem = problem_5_centers();
        let initial_cost = problem.cost(&vec![10.0, 10.0]).unwrap();
        let mut sgd = Sgd::new(0.05, 2, 42);
        let mut p = Problem::new(problem);
        let state =
            sgd.init(&mut p, PointState::new(vec![10.0, 10.0])).unwrap();
        assert_eq!(state.cost(), initial_cost);
        let (state, _, _) = sgd.next_iter(&mut p, state).unwrap().into_parts();
        assert_eq!(
            state.cost(),
            initial_cost,
            "default schedule must retain the evaluated record within an epoch",
        );
        let (state, _, _) = sgd.next_iter(&mut p, state).unwrap().into_parts();
        assert_ne!(
            state.cost(),
            initial_cost,
            "default schedule must refresh at the epoch boundary (iter 2)",
        );
    }

    #[test]
    fn with_cost_eval_every_one_refreshes_per_iter() {
        // Per-iter refresh: state.cost must change after every step on
        // a non-stationary trajectory.
        let problem = problem_5_centers();
        let initial_cost = problem.cost(&vec![10.0, 10.0]).unwrap();
        let mut sgd = Sgd::new(0.05, 2, 42).with_cost_eval_every(1);
        let mut p = Problem::new(problem);
        let state =
            sgd.init(&mut p, PointState::new(vec![10.0, 10.0])).unwrap();
        let (state, _, _) = sgd.next_iter(&mut p, state).unwrap().into_parts();
        assert_ne!(
            state.cost(),
            initial_cost,
            "with_cost_eval_every(1) must refresh state.cost after every step",
        );
    }

    #[test]
    fn zero_momentum_matches_plain_sgd_branch() {
        // β = 0 must follow the no-momentum branch and produce a clean
        // x − α·g step at the first iter. With batch_size = n (full
        // batch), the batch gradient at x = (1, 1) on the 5-center
        // problem equals 2·(x − centroid) and the step is deterministic.
        let problem = problem_5_centers();
        let centroid = problem.centroid();
        let mut sgd = Sgd::new(0.1, 5, 0).with_momentum(0.0);
        let mut p = Problem::new(problem);
        let state = sgd.init(&mut p, PointState::new(vec![1.0, 1.0])).unwrap();
        let (state, _, reason) =
            sgd.next_iter(&mut p, state).unwrap().into_parts();
        assert!(reason.is_none());
        // Full batch gradient is 2·(x − centroid), so x₁ = x − α·2·(x − centroid).
        let alpha = 0.1;
        let x0 = [1.0, 1.0];
        let expected: Vec<f64> = x0
            .iter()
            .zip(centroid.iter())
            .map(|(x, c)| x - alpha * 2.0 * (x - c))
            .collect();
        for (xi, ei) in state.param().iter().zip(expected.iter()) {
            assert!(
                (xi - ei).abs() < 1e-12,
                "got {:?}, expected {:?}",
                state.param(),
                expected
            );
        }
    }
}
