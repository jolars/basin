use rand::seq::SliceRandom;

use crate::core::math::{NegInPlace, Scalar, ScaleInPlace, ScaledAdd};
use crate::core::problem::{CostFunction, MiniBatchGradient, Problem};
use crate::core::rng::{ChaCha8Rng, SeedableRng};
use crate::core::solver::Solver;
use crate::core::state::BasicState;
use crate::core::termination::TerminationReason;

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
/// Epoch-shuffle without replacement (the standard PyTorch / JAX / Bottou
/// 2012 convention):
///
/// 1. At [`Solver::init`], the solver builds a permutation
///    `perm = [0, 1, …, n−1]` and Fisher-Yates–shuffles it with its
///    [`ChaCha8Rng`].
/// 2. Each [`Solver::next_iter`] consumes the next contiguous slice of
///    `batch_size` indices from `perm`.
/// 3. When fewer than `batch_size` indices remain in the current epoch,
///    the solver reshuffles `perm` and starts a new epoch — `drop_last`
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
/// adding momentum — same stability caveat as the full-batch case.
///
/// # Cost tracking
///
/// `state.cost` is seeded at [`Solver::init`] with the full objective at
/// the starting iterate, then refreshed by a full evaluation every
/// **epoch boundary** by default (`batches_per_epoch` iters, where
/// `batches_per_epoch = n_samples / batch_size`). This matches the
/// standard ML rhythm — full loss reported once per epoch — and keeps
/// per-iter cost overhead well under 1 % for typical batch sizes.
///
/// Two consequences flow from this default:
///
/// - Within an epoch, `state.cost` is *stale*: it still reflects the
///   cost at the most recent boundary, not the current iterate.
///   Cost-based termination
///   ([`TargetCost`](crate::TargetCost),
///   [`NoImprovement`](crate::NoImprovement),
///   [`CostTolerance`](crate::CostTolerance)) therefore fires at
///   epoch granularity, with a worst-case overshoot of one epoch's
///   worth of work.
/// - The final `result.cost()` reads the most recent epoch-boundary
///   cost, which may be one epoch stale relative to the final iterate.
///   The user can recompute the exact final cost via `problem.cost(x)`
///   if they need it sharper.
///
/// Use [`with_cost_eval_every`](Self::with_cost_eval_every) to override
/// the refresh period — pass `1` for per-iter cost (debugging, plotting,
/// tight termination), or a larger value to amortize even more
/// aggressively on huge datasets.
///
/// The mini-batch gradient is not cached on the state
/// (`state.gradient` stays `None`), so gradient-based termination
/// criteria do not fire — by design, since the batch estimate is noisy.
///
/// # Backends
///
/// Backend-generic — works with any `V` implementing
/// [`ScaledAdd<F>`](crate::core::math::ScaledAdd) +
/// [`NegInPlace`] + [`ScaleInPlace<F>`] + `Clone`. With the default
/// `F = f64` that covers `Vec<f64>`, `nalgebra::DVector<f64>` (feature
/// `nalgebra`), `ndarray::Array1<f64>` (feature `ndarray`), and
/// `faer::Col<f64>` (feature `faer`).
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
///     BasicState, CostFunction, Executor, MaxIter, MiniBatchGradient,
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
/// let result = Executor::new(problem, sgd, BasicState::new(vec![0.0, 0.0]))
///     .terminate_on(MaxIter(2_000))
///     .run()
///     .unwrap();
/// assert!(result.cost() < 1e-6);
/// ```
pub struct Sgd<V, F = f64> {
    alpha: F,
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

    /// Refresh the cached full cost in `state.cost` every `period` iters
    /// rather than at every epoch boundary (the default).
    ///
    /// - `period = 1`: per-iter refresh. Most accurate per-step cost,
    ///   but adds one full-data pass per iteration on top of the
    ///   mini-batch gradient — closer to full-batch GD overhead.
    ///   Pick this for debugging, plotting per-iter convergence curves,
    ///   tight cost-based termination (`TargetCost` with small `eps`,
    ///   `NoImprovement` with short patience), or when the dataset is
    ///   small enough that the extra cost passes are negligible.
    /// - `period = batches_per_epoch`: the default if this builder is
    ///   not called. One full-cost pass per epoch — matches the
    ///   PyTorch / JAX "report training loss per epoch" rhythm and
    ///   keeps the cost overhead well under 1 % for typical batch
    ///   sizes.
    /// - Larger `period`: even less overhead, at the cost of staler
    ///   `state.cost` (and later firing of cost-based termination).
    ///
    /// `period = 0` panics.
    pub fn with_cost_eval_every(mut self, period: usize) -> Self {
        assert!(period > 0, "Sgd: cost_eval_every period must be > 0");
        self.cost_eval_every = Some(period);
        self
    }
}

impl<P, V, F> Solver<P, BasicState<V, F>> for Sgd<V, F>
where
    F: Scalar,
    P: CostFunction<Param = V, Output = F> + MiniBatchGradient<Gradient = V>,
    V: ScaledAdd<F> + NegInPlace + ScaleInPlace<F> + Clone,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: BasicState<V, F>,
    ) -> Result<BasicState<V, F>, Self::Error> {
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

        // Seed cost at the initial param so iter-0 termination checks
        // (e.g. `TargetCost` on a near-optimal start) and
        // `OptimizationResult::cost()` see a defined value. The mini-batch
        // gradient is *not* cached — `state.gradient` stays `None`.
        let cost = problem.cost(&state.param)?;
        state.cost = Some(cost);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: BasicState<V, F>,
    ) -> Result<(BasicState<V, F>, Option<TerminationReason>), Self::Error> {
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
        let grad = problem.batch_gradient(&state.param, batch)?;
        self.cursor += bs;

        let mut direction = grad;
        direction.neg_in_place();

        if self.beta == F::zero() {
            // No momentum: x ← x + α·(−g). Allocation-free.
            state.param.scaled_add(self.alpha, &direction);
        } else {
            // Heavy ball: v ← β·v + α·(−g) = β·v − α·g, then x ← x + v.
            // With v₀ = 0, the first step is α·(−g); form it by consuming
            // `direction` to avoid materializing a zero vector.
            let velocity = match self.velocity.take() {
                Some(mut v) => {
                    v.scale_in_place(self.beta);
                    v.scaled_add(self.alpha, &direction);
                    v
                }
                None => {
                    direction.scale_in_place(self.alpha);
                    direction
                }
            };
            state.param.scaled_add(F::one(), &velocity);
            self.velocity = Some(velocity);
        }

        // Refresh cached full cost every `cost_period` iters (epoch
        // boundary by default; see the type-level "Cost tracking"
        // section for the rationale and the `with_cost_eval_every`
        // builder).
        self.iters_since_cost += 1;
        if self.iters_since_cost >= self.cost_period {
            let cost = problem.cost(&state.param)?;
            state.cost = Some(cost);
            self.iters_since_cost = 0;
        }

        Ok((state, None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::state::State;
    use crate::{BasicState, Executor, MaxIter};

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
        fn batch_gradient(&self, x: &Vec<f64>, batch: &[usize]) -> Result<Vec<f64>, Self::Error> {
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
        // around the optimum (the well-known constant-LR SGD result — see
        // Bottou 2012). It does not converge *to* the optimum without LR
        // decay, but it does enter and stay in a small neighborhood
        // proportional to α. Use a small batch_size to keep the test
        // honest about the stochastic regime, and a tolerance sized to
        // the noise floor rather than to a deterministic GD optimum.
        let problem = problem_5_centers();
        let centroid = problem.centroid();
        let sgd = Sgd::new(0.01, 2, 0xABCDEF);
        let result = Executor::new(problem, sgd, BasicState::new(vec![0.0, 0.0]))
            .terminate_on(MaxIter(3_000))
            .run()
            .unwrap();
        let x = result.param();
        for (xi, ci) in x.iter().zip(centroid.iter()) {
            assert!((xi - ci).abs() < 5e-2, "x = {x:?}, centroid = {centroid:?}",);
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
        let result = Executor::new(problem, sgd, BasicState::new(vec![0.0, 0.0]))
            .terminate_on(MaxIter(500))
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
            Executor::new(p, sgd, BasicState::new(vec![0.5, -0.5]))
                .terminate_on(MaxIter(50))
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
            Executor::new(problem_5_centers(), sgd, BasicState::new(vec![0.5, -0.5]))
                .terminate_on(MaxIter(20))
                .run()
                .unwrap()
                .param()
                .clone()
        };
        let xa = run(1);
        let xb = run(2);
        // Different seeds produce different batch orderings → different
        // trajectories after a handful of steps.
        let diff: f64 = xa.iter().zip(xb.iter()).map(|(a, b)| (a - b).abs()).sum();
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
            let mut state = solver.init(&mut p, BasicState::new(start.clone())).unwrap();
            for _ in 0..15 {
                let (next, _) = solver.next_iter(&mut p, state).unwrap();
                state = next;
            }
            state.param().clone()
        };

        let first = run_once(&mut sgd);
        let second = run_once(&mut sgd);
        for (a, b) in first.iter().zip(second.iter()) {
            assert!((a - b).abs() < 1e-15, "first={first:?}, second={second:?}");
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
        let mut state = sgd.init(&mut p, BasicState::new(vec![0.0, 0.0])).unwrap();
        // 3 steps: enough to trigger the reshuffle at step 3 (cursor
        // would be 6, and 6 + 3 > 7).
        for _ in 0..3 {
            let (next, _) = sgd.next_iter(&mut p, state).unwrap();
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
        let result = Executor::new(problem, sgd, BasicState::new(vec![0.0]))
            .terminate_on(MaxIter(500))
            .run()
            .unwrap();
        assert!((result.param()[0] - centroid[0]).abs() < 1e-3);
    }

    #[test]
    fn cost_refresh_default_is_epoch_boundary() {
        // n_samples = 5, batch_size = 2 → batches_per_epoch = 2.
        // After 1 iter (still mid-epoch), state.cost must equal the
        // *initial* cost — the default schedule does not refresh until
        // the second iter wraps an epoch.
        let problem = problem_5_centers();
        let initial_cost = problem.cost(&vec![10.0, 10.0]).unwrap();
        let mut sgd = Sgd::new(0.05, 2, 42);
        let mut p = Problem::new(problem);
        let state = sgd.init(&mut p, BasicState::new(vec![10.0, 10.0])).unwrap();
        assert_eq!(state.cost(), initial_cost);
        let (state, _) = sgd.next_iter(&mut p, state).unwrap();
        assert_eq!(
            state.cost(),
            initial_cost,
            "default schedule must hold state.cost stale within an epoch",
        );
        let (state, _) = sgd.next_iter(&mut p, state).unwrap();
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
        let state = sgd.init(&mut p, BasicState::new(vec![10.0, 10.0])).unwrap();
        let (state, _) = sgd.next_iter(&mut p, state).unwrap();
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
        let state = sgd.init(&mut p, BasicState::new(vec![1.0, 1.0])).unwrap();
        let (state, reason) = sgd.next_iter(&mut p, state).unwrap();
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
