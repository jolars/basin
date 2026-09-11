use rand_distr::uniform::SampleUniform;

use crate::core::constraint::BoxConstraints;
use crate::core::inner::InitialState;
use crate::core::math::{
    ClampInPlace, SampleUniformBox, Scalar, ScaleInPlace, ScaledAdd,
    VectorIndex, VectorLen,
};
use crate::core::problem::{CostFunction, Problem};
use crate::core::rng::{ChaCha8Rng, RngExt, SeedableRng};
use crate::core::solver::Solver;
use crate::core::state::gbnm::GbnmRestart;
use crate::core::state::{BasicSimplexState, GbnmState};
use crate::core::termination::TerminationReason;
use crate::solver::{NelderMead, Projected, Unbounded};

mod geometry;
mod restart;

use geometry::{
    free_coordinates, log_parzen_density, normalized_l1_distance,
    point_on_bounds, regular_simplex, scaled_minimum_width,
    simplex_is_degenerate, simplex_is_small,
};
use restart::{OptimumAction, select_restart};

struct BoxGeometry<'a, V> {
    lower: &'a V,
    upper: &'a V,
    free: &'a [usize],
}

/// Globalized Bounded Nelder-Mead (GBNM).
///
/// This is Luersen and Le Riche's bounded local Nelder-Mead search embedded in
/// their probabilistic-restart state machine. Each local search projects trial
/// points into the problem's box. Small-, flat-, and degenerate-simplex tests
/// select probabilistic, small-test, or large-test restarts and retain a list
/// of possible local optima.
///
/// Probabilistic restarts draw candidate centers uniformly from the box and
/// choose the point with the lowest Parzen density relative to all previous
/// local-search starts and retained local convergence points. The diagonal
/// Gaussian variance is
/// `0.01 * (upper[i] - lower[i])^2` by default. The restarted regular simplex
/// has a uniformly sampled edge length between 2% and 10% of the smallest
/// free-coordinate box width.
///
/// The first local search is centered on the caller-provided
/// [`GbnmState::new`] point, after projection into the box. This makes a known
/// engineering design usable while retaining the paper's global restart
/// mechanism for subsequent searches.
///
/// # Configuration
///
/// The defaults use standard Nelder-Mead coefficients, 10 restart candidates,
/// Gaussian variance factor `0.01`, probabilistic simplex fractions
/// `[0.02, 0.10]`, normalized small-simplex tolerance `1e-6`, flat-simplex
/// tolerance `1e-20`, and both degeneracy tolerances at `1e-7`. The initial
/// and large-test simplex fractions are `0.05`; the small-test fraction is
/// `1e-5`. Builder methods validate all coefficients, fractions, and
/// tolerances when they are set.
///
/// The paper prescribes a small test simplex that is slightly larger than the
/// convergence tolerance and a larger test simplex, but it does not give
/// universal sizes for them. Basin's `1e-5`/`0.05` defaults are the documented
/// interpretation of that relationship.
///
/// # Termination
///
/// GBNM is a budget-driven global algorithm. Use
/// [`max_cost_evals`](crate::Executor::max_cost_evals), [`max_iter`](crate::Executor::max_iter),
/// [`max_time`](crate::Executor::max_time), or [`target_cost`](crate::Executor::target_cost). One
/// simplex operation is atomic. A local step may evaluate up to `n + 2` points
/// and a restart evaluates `n + 1`, where `n` is the number of free
/// coordinates, so an evaluation budget may be exceeded by at most `n + 1`
/// evaluations.
///
/// Do not apply [`SimplexTolerance`](crate::SimplexTolerance): local simplex
/// convergence triggers GBNM's restart logic and is not convergence of the
/// outer search. [`GbnmState`] deliberately does not implement
/// [`SimplexState`](crate::SimplexState), making that mismatch a compile-time
/// error.
///
/// # Pinned coordinates
///
/// Equal lower and upper bounds pin a coordinate. Pinned coordinates are
/// excluded from simplex geometry, density calculations, and convergence
/// tests. At least one coordinate must remain free.
///
/// # Numerical behavior
///
/// The Parzen mixture is compared in log space with a log-sum-exp reduction,
/// avoiding underflow when candidates lie far from the start history. The
/// normalized determinant in the degeneracy test uses partial-pivot Gaussian
/// elimination and needs no backend-specific matrix operation. `NaN` costs
/// sort behind all comparable values; `+infinity` remains a supported soft
/// rejection and sorts behind finite values.
///
/// # Backends
///
/// Backend-generic; works with `Vec<F>`, `nalgebra::DVector<F>` (feature
/// `nalgebra`), `ndarray::Array1<F>` (feature `ndarray`), and `faer::Col<F>`
/// (feature `faer`) for `F = f64` or `F = f32`.
///
/// # Example
///
/// ```
/// use basin::{BoxConstraints, CostFunction, Executor, Gbnm, GbnmState};
///
/// struct BoundedSphere {
///     lower: Vec<f64>,
///     upper: Vec<f64>,
/// }
/// impl CostFunction for BoundedSphere {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
///         Ok(x.iter().map(|xi| xi * xi).sum())
///     }
/// }
/// impl BoxConstraints for BoundedSphere {
///     fn lower(&self) -> &Vec<f64> {
///         &self.lower
///     }
///     fn upper(&self) -> &Vec<f64> {
///         &self.upper
///     }
/// }
///
/// let problem = BoundedSphere {
///     lower: vec![-5.0, -5.0],
///     upper: vec![5.0, 5.0],
/// };
/// let result =
///     Executor::new(problem, Gbnm::new(42), GbnmState::new(vec![4.0, 4.0]))
///         .max_cost_evals(2_000)
///         .run()
///         .unwrap();
/// assert!(result.best_cost() < 1e-4);
/// ```
///
/// # References
///
/// M. A. Luersen and R. Le Riche, “Globalized Nelder–Mead method for
/// engineering optimization,” *Computers & Structures*, 82 (2004), 2251–2260.
/// [doi:10.1016/j.compstruc.2004.03.072](https://doi.org/10.1016/j.compstruc.2004.03.072).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Gbnm<F = f64> {
    local: NelderMead<Projected, F>,
    rng: ChaCha8Rng,
    restart_candidates: usize,
    gaussian_variance_factor: F,
    initial_simplex_fraction: F,
    probabilistic_min_fraction: F,
    probabilistic_max_fraction: F,
    small_test_fraction: F,
    large_test_fraction: F,
    small_tolerance: F,
    flat_tolerance: F,
    degeneracy_edge_ratio: F,
    degeneracy_shape: F,
}

impl<F: Scalar> Gbnm<F> {
    /// Construct GBNM with the paper's standard stochastic settings and a
    /// reproducible PRNG seed.
    pub fn new(seed: u64) -> Self {
        Self {
            local: NelderMead::new().projected(),
            rng: ChaCha8Rng::seed_from_u64(seed),
            restart_candidates: 10,
            gaussian_variance_factor: F::from_f64(0.01).unwrap(),
            initial_simplex_fraction: F::from_f64(0.05).unwrap(),
            probabilistic_min_fraction: F::from_f64(0.02).unwrap(),
            probabilistic_max_fraction: F::from_f64(0.10).unwrap(),
            small_test_fraction: F::from_f64(1e-5).unwrap(),
            large_test_fraction: F::from_f64(0.05).unwrap(),
            small_tolerance: F::from_f64(1e-6).unwrap(),
            flat_tolerance: F::from_f64(1e-20).unwrap(),
            degeneracy_edge_ratio: F::from_f64(1e-7).unwrap(),
            degeneracy_shape: F::from_f64(1e-7).unwrap(),
        }
    }

    /// Set the local Nelder-Mead reflection, expansion, contraction, and
    /// shrink coefficients (`alpha`, `beta`, `gamma`, and `delta`).
    pub fn with_nm_params(
        mut self,
        alpha: F,
        beta: F,
        gamma: F,
        delta: F,
    ) -> Self {
        self.local =
            NelderMead::<Unbounded, F>::with_params(alpha, beta, gamma, delta)
                .projected();
        self
    }

    /// Set the number of uniformly sampled centers considered at each
    /// probabilistic restart. The paper uses 10.
    pub fn with_restart_candidates(mut self, count: usize) -> Self {
        assert!(count >= 1, "GBNM requires at least one restart candidate");
        self.restart_candidates = count;
        self
    }

    /// Set the dimensionless diagonal Gaussian variance factor used by the
    /// Parzen density. The paper uses `0.01`.
    pub fn with_gaussian_variance_factor(mut self, factor: F) -> Self {
        assert_positive_finite(factor, "Gaussian variance factor");
        self.gaussian_variance_factor = factor;
        self
    }

    /// Set the initial regular-simplex edge as a fraction of the smallest
    /// free-coordinate box width. The default is `0.05`.
    pub fn with_initial_simplex_fraction(mut self, fraction: F) -> Self {
        assert_fraction(fraction, "initial simplex fraction");
        self.initial_simplex_fraction = fraction;
        self
    }

    /// Set the inclusive range for probabilistic-restart simplex-edge
    /// fractions. The paper uses `[0.02, 0.10]`.
    pub fn with_probabilistic_simplex_fractions(
        mut self,
        min: F,
        max: F,
    ) -> Self {
        assert_fraction(min, "minimum probabilistic simplex fraction");
        assert_fraction(max, "maximum probabilistic simplex fraction");
        assert!(
            min <= max,
            "minimum probabilistic simplex fraction must not exceed maximum"
        );
        self.probabilistic_min_fraction = min;
        self.probabilistic_max_fraction = max;
        self
    }

    /// Set the small- and large-test regular-simplex edge fractions.
    ///
    /// Luersen and Le Riche specify the relationship but not universal
    /// numerical values. Basin uses `1e-5` and `0.05` by default: the small
    /// test is just above the default normalized simplex tolerance, and the
    /// large test matches the ordinary initial simplex scale.
    pub fn with_test_simplex_fractions(mut self, small: F, large: F) -> Self {
        assert_fraction(small, "small-test simplex fraction");
        assert_fraction(large, "large-test simplex fraction");
        assert!(
            small < large,
            "small-test simplex fraction must be smaller than large-test fraction"
        );
        self.small_test_fraction = small;
        self.large_test_fraction = large;
        self
    }

    /// Set the normalized simplex-size tolerance `epsilon_s1` from equation
    /// (7). It also defines when convergence points count as identical for the
    /// restart and local-optimum memory tests. The default is `1e-6`.
    #[deprecated(
        note = "use `with_normalized_simplex_size_tolerance`; removal scheduled for Basin 2.0"
    )]
    pub fn with_small_tolerance(self, tolerance: F) -> Self {
        self.with_normalized_simplex_size_tolerance(tolerance)
    }

    /// Configure the normalized simplex size tolerance.
    /// Retains the algorithm's existing formula, validation, and default.
    pub fn with_normalized_simplex_size_tolerance(
        mut self,
        tolerance: F,
    ) -> Self {
        assert_positive_finite(tolerance, "small-simplex tolerance");
        self.small_tolerance = tolerance;
        self
    }

    /// Set the absolute cost-spread tolerance `epsilon_s2` from equation (8).
    /// The default is `1e-20`.
    #[deprecated(
        note = "use `with_absolute_simplex_cost_tolerance`; removal scheduled for Basin 2.0"
    )]
    pub fn with_flat_tolerance(self, tolerance: F) -> Self {
        self.with_absolute_simplex_cost_tolerance(tolerance)
    }

    /// Configure the absolute simplex cost tolerance.
    /// Retains the algorithm's existing formula, validation, and default.
    pub fn with_absolute_simplex_cost_tolerance(
        mut self,
        tolerance: F,
    ) -> Self {
        assert_positive_finite(tolerance, "flat-simplex tolerance");
        self.flat_tolerance = tolerance;
        self
    }

    /// Set the edge-ratio and normalized-determinant degeneracy tolerances
    /// `epsilon_s3` and `epsilon_s4` from equation (9). Both default to
    /// `1e-7`.
    /// Set the dimensionless edge-ratio threshold in `(0, 1)`.
    pub fn with_edge_ratio_tolerance(mut self, value: F) -> Self {
        assert_unit_tolerance(value, "degeneracy edge-ratio tolerance");
        self.degeneracy_edge_ratio = value;
        self
    }

    /// Set the normalized simplex-determinant threshold in `(0, 1)`.
    pub fn with_normalized_determinant_tolerance(mut self, value: F) -> Self {
        assert_unit_tolerance(value, "degeneracy shape tolerance");
        self.degeneracy_shape = value;
        self
    }

    /// Set both legacy degeneracy thresholds.
    #[deprecated(
        note = "use `with_edge_ratio_tolerance` and `with_normalized_determinant_tolerance`; removal scheduled for Basin 2.0"
    )]
    pub fn with_degeneracy_tolerances(
        mut self,
        edge_ratio: F,
        shape: F,
    ) -> Self {
        assert_unit_tolerance(edge_ratio, "degeneracy edge-ratio tolerance");
        assert_unit_tolerance(shape, "degeneracy shape tolerance");
        self.degeneracy_edge_ratio = edge_ratio;
        self.degeneracy_shape = shape;
        self
    }
}

fn assert_positive_finite<F: Scalar>(value: F, name: &str) {
    assert!(
        value.is_finite() && value > F::zero(),
        "GBNM {name} must be finite and > 0, got {value:?}"
    );
}

fn assert_fraction<F: Scalar>(value: F, name: &str) {
    assert!(
        value.is_finite() && value > F::zero() && value <= F::one(),
        "GBNM {name} must be finite and in (0, 1], got {value:?}"
    );
}

fn assert_unit_tolerance<F: Scalar>(value: F, name: &str) {
    assert!(
        value.is_finite() && value > F::zero() && value < F::one(),
        "GBNM {name} must be finite and in (0, 1), got {value:?}"
    );
}

impl<F> Gbnm<F>
where
    F: Scalar + SampleUniform,
{
    fn sample_restart_center<V>(
        &mut self,
        lower: &V,
        upper: &V,
        free: &[usize],
    ) -> V
    where
        V: Clone + SampleUniformBox + VectorIndex<F>,
    {
        if free
            .iter()
            .all(|&i| (upper.get_scalar(i) - lower.get_scalar(i)).is_finite())
        {
            return V::sample_uniform_box(lower, upper, &mut self.rng);
        }

        // `rand` rejects finite ranges when their span overflows, while a
        // convex combination of opposite-sign endpoints stays representable.
        let mut sample = lower.clone();
        for &i in free {
            let lower = lower.get_scalar(i);
            let upper = upper.get_scalar(i);
            let value = if (upper - lower).is_finite() {
                self.rng.random_range(lower..=upper)
            } else {
                let weight = self.rng.random_range(F::zero()..=F::one());
                lower * (F::one() - weight) + upper * weight
            };
            sample.set_scalar(i, value);
        }
        sample
    }

    fn probabilistic_center<V>(
        &mut self,
        starts: &[V],
        local_optima: &[(V, F)],
        lower: &V,
        upper: &V,
        free: &[usize],
    ) -> V
    where
        V: Clone + SampleUniformBox + VectorIndex<F>,
    {
        let mut selected = self.sample_restart_center(lower, upper, free);
        let mut selected_density = log_parzen_density(
            &selected,
            starts,
            local_optima,
            lower,
            upper,
            free,
            self.gaussian_variance_factor,
        );
        for _ in 1..self.restart_candidates {
            let candidate = self.sample_restart_center(lower, upper, free);
            let density = log_parzen_density(
                &candidate,
                starts,
                local_optima,
                lower,
                upper,
                free,
                self.gaussian_variance_factor,
            );
            if density < selected_density {
                selected = candidate;
                selected_density = density;
            }
        }
        selected
    }

    fn install_restart<P, V>(
        &mut self,
        problem: &mut Problem<P>,
        mut state: GbnmState<V, F>,
        restart: GbnmRestart,
        center: V,
        edge_fraction: F,
        geometry: BoxGeometry<'_, V>,
    ) -> Result<GbnmState<V, F>, P::Error>
    where
        P: CostFunction<Param = V, Output = F> + BoxConstraints<Param = V>,
        V: Clone
            + ClampInPlace
            + ScaleInPlace<F>
            + ScaledAdd<F>
            + VectorIndex<F>,
    {
        let edge = scaled_minimum_width(
            geometry.lower,
            geometry.upper,
            geometry.free,
            edge_fraction,
        );
        let simplex = BasicSimplexState::from_simplex(regular_simplex(
            &center,
            edge,
            geometry.lower,
            geometry.upper,
            geometry.free,
        ));
        let simplex = <NelderMead<Projected, F> as Solver<
            P,
            BasicSimplexState<V, F>,
        >>::init(&mut self.local, problem, simplex)?;
        state.search_starts.push(center.clone());
        state.restart = restart;
        state.restart_center = Some(center);
        state.simplex = Some(simplex);
        Ok(state)
    }

    fn is_known_optimum<V>(
        &self,
        point: &V,
        state: &GbnmState<V, F>,
        lower: &V,
        upper: &V,
        free: &[usize],
    ) -> bool
    where
        V: VectorIndex<F>,
    {
        state.local_optima.iter().any(|(known, _)| {
            normalized_l1_distance(point, known, lower, upper, free)
                < self.small_tolerance
        })
    }

    fn save_optimum<V>(
        &self,
        state: &mut GbnmState<V, F>,
        point: V,
        cost: F,
        lower: &V,
        upper: &V,
        free: &[usize],
    ) where
        V: VectorIndex<F>,
    {
        if !cost.is_finite()
            || self.is_known_optimum(&point, state, lower, upper, free)
        {
            return;
        }
        state.local_optima.push((point, cost));
    }
}

impl<P, V, F> Solver<P, GbnmState<V, F>> for Gbnm<F>
where
    F: Scalar + SampleUniform,
    P: CostFunction<Param = V, Output = F> + BoxConstraints<Param = V>,
    V: Clone
        + ClampInPlace
        + SampleUniformBox
        + ScaleInPlace<F>
        + ScaledAdd<F>
        + VectorIndex<F>
        + VectorLen,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: GbnmState<V, F>,
    ) -> Result<GbnmState<V, F>, Self::Error> {
        let lower = problem.inner().lower().clone();
        let upper = problem.inner().upper().clone();
        assert_eq!(
            state.initial.vec_len(),
            lower.vec_len(),
            "GBNM initial point and bounds length mismatch"
        );
        let free = free_coordinates(&lower, &upper);
        let mut center = state.initial.clone();
        center.clamp_in_place(&lower, &upper);

        state.initial = center.clone();
        state.simplex = None;
        state.search_starts.clear();
        state.local_optima.clear();
        state.restart_count = 0;
        state.restart = GbnmRestart::Probabilistic;
        state.restart_center = None;
        self.install_restart(
            problem,
            state,
            GbnmRestart::Probabilistic,
            center,
            self.initial_simplex_fraction,
            BoxGeometry {
                lower: &lower,
                upper: &upper,
                free: &free,
            },
        )
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: GbnmState<V, F>,
    ) -> Result<(GbnmState<V, F>, Option<TerminationReason>), Self::Error> {
        let lower = problem.inner().lower().clone();
        let upper = problem.inner().upper().clone();
        let free = free_coordinates(&lower, &upper);
        let simplex = state
            .simplex
            .take()
            .expect("Gbnm::init must run before next_iter");

        let small = simplex_is_small(
            &simplex.vertices,
            &lower,
            &upper,
            &free,
            self.small_tolerance,
        );
        let spread = simplex.costs[simplex.costs.len() - 1] - simplex.costs[0];
        let flat = spread.is_finite() && spread.abs() < self.flat_tolerance;
        let degenerate = simplex_is_degenerate(
            &simplex.vertices,
            &lower,
            &upper,
            &free,
            small,
            self.degeneracy_edge_ratio,
            self.degeneracy_shape,
        );

        if !(small || flat || degenerate) {
            let (simplex, reason) = <NelderMead<Projected, F> as Solver<
                P,
                BasicSimplexState<V, F>,
            >>::next_iter(
                &mut self.local, problem, simplex
            )?;
            state.simplex = Some(simplex);
            return Ok((state, reason));
        }

        let point = simplex.vertices[0].clone();
        let cost = simplex.costs[0];
        let center = state
            .restart_center
            .as_ref()
            .expect("GBNM restart center missing after initialization");
        let returned_to_center =
            normalized_l1_distance(&point, center, &lower, &upper, &free)
                < self.small_tolerance;
        let on_bounds = point_on_bounds(&point, &lower, &upper, &free);

        let (next_restart, optimum_action) = select_restart(
            state.restart,
            self.is_known_optimum(&point, &state, &lower, &upper, &free),
            flat,
            small,
            degenerate,
            returned_to_center,
            on_bounds,
        );
        if optimum_action == OptimumAction::Save {
            self.save_optimum(
                &mut state,
                point.clone(),
                cost,
                &lower,
                &upper,
                &free,
            );
        }

        let (next_center, edge_fraction) = match next_restart {
            GbnmRestart::Probabilistic => {
                let center = self.probabilistic_center(
                    &state.search_starts,
                    &state.local_optima,
                    &lower,
                    &upper,
                    &free,
                );
                let fraction = self.rng.random_range(
                    self.probabilistic_min_fraction
                        ..=self.probabilistic_max_fraction,
                );
                (center, fraction)
            }
            GbnmRestart::SmallTest => (point, self.small_test_fraction),
            GbnmRestart::LargeTest => (point, self.large_test_fraction),
        };
        state.restart_count += 1;
        let state = self.install_restart(
            problem,
            state,
            next_restart,
            next_center,
            edge_fraction,
            BoxGeometry {
                lower: &lower,
                upper: &upper,
                free: &free,
            },
        )?;
        Ok((state, None))
    }
}

impl<V, F> InitialState<V> for Gbnm<F>
where
    F: Scalar,
    V: Clone,
{
    type State = GbnmState<V, F>;

    fn seed(&self, x: &V) -> Self::State {
        GbnmState::new(x.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::Gbnm;
    use crate::core::state::GbnmState;

    #[test]
    fn possible_local_optima_are_finite_and_deduplicated() {
        let solver = Gbnm::new(1);
        let mut state = GbnmState::new(vec![0.0, 0.0]);
        let lower = vec![-5.0, -5.0];
        let upper = vec![5.0, 5.0];
        let free = [0, 1];

        solver.save_optimum(
            &mut state,
            vec![1.0, 1.0],
            2.0,
            &lower,
            &upper,
            &free,
        );
        solver.save_optimum(
            &mut state,
            vec![1.0 + 1e-7, 1.0],
            1.9,
            &lower,
            &upper,
            &free,
        );
        solver.save_optimum(
            &mut state,
            vec![3.0, 3.0],
            f64::INFINITY,
            &lower,
            &upper,
            &free,
        );

        assert_eq!(state.local_optima().len(), 1);
    }
}
