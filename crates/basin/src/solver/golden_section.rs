use crate::core::constraint::BoxConstraints;
use crate::core::math::Scalar;
use crate::core::problem::{CostFunction, Problem};
use crate::core::solver::Solver;
use crate::core::state::ScalarState;
use crate::core::termination::TerminationReason;

/// Golden-section search for 1D minimization on a closed interval
/// `[lower, upper]` supplied via `BoxConstraints`. Maintains a shrinking
/// bracket with two interior points placed at the golden ratio and discards
/// the worse-side endpoint each step, reusing one of the two interior
/// evaluations so that every iteration costs exactly **one** new function
/// evaluation. Kiefer (1953); Numerical Recipes §10.1.
///
/// Golden section is the robust, derivative-free companion to
/// [`Brent`](crate::solver::Brent): it converges only linearly (the bracket
/// shrinks by the factor `r = (√5 − 1)/2 ≈ 0.618` each step) so it is slower
/// than Brent on smooth functions, but it never relies on parabolic
/// interpolation, which makes it dependable on unimodal functions where
/// Brent's parabola step can misbehave. Brent itself falls back to a
/// golden-section step when its parabola is rejected.
///
/// Convergence test (in `Solver::terminate`): `(b − a) ≤ 2·tol`, where
/// `tol = tol_rel·|x| + tol_abs` and `x` is the current best interior point.
/// NR-style defaults: `tol_rel = √ε`, `tol_abs = 1e-12`.
///
/// Unlike Brent, golden section ignores the starting point: the bracket
/// `[lower, upper]` fully determines the two interior points, so the value
/// passed to [`ScalarState::new`](crate::core::state::ScalarState::new) does
/// not affect the search.
///
/// `GoldenSection` runs on [`ScalarState`], the cost-only single-iterate
/// state. It does **not** impl
/// [`GradientState`](crate::core::state::GradientState) (a 1-D minimizer has
/// no gradient), so attaching a gradient criterion such as
/// [`GradientTolerance`](crate::core::termination::GradientTolerance) is a
/// **compile error** rather than one that silently never fires.
///
/// # Backends
///
/// Scalar by construction: the parameter is a single `F` (default
/// `f64`), so `GoldenSection` is backend-agnostic and needs no
/// linear-algebra backend. The problem's `BoxConstraints` carry
/// `F`-valued lower and upper bounds.
///
/// # Examples
///
/// `GoldenSection` minimizes a 1-D function over a bracket, so the problem
/// implements `CostFunction` *and* `BoxConstraints` with scalar (`F`)
/// bounds. See [`Brent`](crate::solver::Brent) for the faster, parabolic
/// sibling and [`NelderMead`](crate::NelderMead) for the general
/// derivative-free `Executor` pattern.
///
/// # References
///
/// Kiefer, J. (1953). "Sequential minimax search for a maximum."
/// *Proceedings of the American Mathematical Society* 4(3): 502–506.
/// Transcribed (golden-section search in one dimension) in Numerical
/// Recipes §10.1.
pub struct GoldenSection<F = f64> {
    tol_rel: F,
    tol_abs: F,
    inner: Option<Inner<F>>,
}

/// `(√5 − 1) / 2`, the golden-ratio bracket-reduction factor. The new
/// bracket keeps this fraction of the previous width each step.
fn golden_r<F: Scalar>() -> F {
    F::from_f64(0.618_033_988_749_894_9).unwrap()
}

#[derive(Clone, Copy)]
struct Inner<F> {
    a: F,
    b: F,
    c1: F,
    c2: F,
    f1: F,
    f2: F,
}

impl<F: Scalar> Default for GoldenSection<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Scalar> GoldenSection<F> {
    /// Golden-section solver with the standard tolerances: `tol_rel = √ε_F`,
    /// `tol_abs = 1e-12`.
    pub fn new() -> Self {
        Self {
            tol_rel: F::epsilon().sqrt(),
            tol_abs: F::from_f64(1e-12).unwrap(),
            inner: None,
        }
    }

    /// Golden-section solver with explicit relative and absolute tolerances.
    /// Both must be strictly positive.
    pub fn with_tol(tol_rel: F, tol_abs: F) -> Self {
        assert!(tol_rel > F::zero(), "tol_rel must be > 0");
        assert!(tol_abs > F::zero(), "tol_abs must be > 0");
        Self {
            tol_rel,
            tol_abs,
            inner: None,
        }
    }
}

impl<P, F> Solver<P, ScalarState<F>> for GoldenSection<F>
where
    F: Scalar,
    P: CostFunction<Param = F, Output = F> + BoxConstraints,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: ScalarState<F>,
    ) -> Result<ScalarState<F>, Self::Error> {
        let a = *problem.inner().lower();
        let b = *problem.inner().upper();
        assert!(
            a.is_finite() && b.is_finite() && a < b,
            "GoldenSection requires a finite, ordered bracket: lower < upper"
        );
        // Place the two interior points at the golden ratio: with
        // `r = (√5−1)/2`, `c1 = a + (1−r)(b−a)` and `c2 = a + r(b−a)`, so
        // `a < c1 < c2 < b`. The starting point in `state` is ignored; the
        // bracket alone determines the search.
        let r = golden_r::<F>();
        let span = b - a;
        let c1 = a + (F::one() - r) * span;
        let c2 = a + r * span;
        let f1 = problem.cost(&c1)?;
        let f2 = problem.cost(&c2)?;
        self.inner = Some(Inner {
            a,
            b,
            c1,
            c2,
            f1,
            f2,
        });
        // Post the better of the two interior points so iter-0 termination
        // criteria read a valid cost.
        if f1 <= f2 {
            state.param = c1;
            state.cost = Some(f1);
        } else {
            state.param = c2;
            state.cost = Some(f2);
        }
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: ScalarState<F>,
    ) -> Result<(ScalarState<F>, Option<TerminationReason>), Self::Error> {
        let s = self
            .inner
            .as_mut()
            .expect("GoldenSection::init must run first");
        let r = golden_r::<F>();

        // Reduce the bracket toward the lower-cost interior point. Exactly one
        // new cost evaluation per step: the surviving interior point is reused
        // as one of the next pair, and only the fresh point is probed.
        let (u, fu);
        if s.f1 <= s.f2 {
            // Minimum lies in [a, c2]; drop the upper end.
            s.b = s.c2;
            s.c2 = s.c1;
            s.f2 = s.f1;
            s.c1 = s.a + (F::one() - r) * (s.b - s.a);
            s.f1 = problem.cost(&s.c1)?;
            u = s.c1;
            fu = s.f1;
        } else {
            // Minimum lies in [c1, b]; drop the lower end.
            s.a = s.c1;
            s.c1 = s.c2;
            s.f1 = s.f2;
            s.c2 = s.a + r * (s.b - s.a);
            s.f2 = problem.cost(&s.c2)?;
            u = s.c2;
            fu = s.f2;
        }

        // Post the just-probed point (u, fu), not the retained best, so that
        // one-step change tests like `CostTolerance` see real Δf signals
        // rather than firing on an unchanged cost after a non-improving probe
        // (issue #36). The executor's best-so-far tracking
        // (`state.best_param`/`state.best_cost`) captures the true optimum
        // independently, and the bracket-collapse `terminate` only fires once
        // the bracket has shrunk below tolerance, so the two coincide at
        // convergence.
        state.param = u;
        state.cost = Some(fu);
        Ok((state, None))
    }

    fn terminate(&self, _state: &ScalarState<F>) -> Option<TerminationReason> {
        let s = self.inner.as_ref()?;
        let two = F::from_f64(2.0).unwrap();
        // Relative term anchored on the current best interior point.
        let x = if s.f1 <= s.f2 { s.c1 } else { s.c2 };
        let tol = self.tol_rel * x.abs() + self.tol_abs;
        if s.b - s.a <= two * tol {
            Some(TerminationReason::SolverConverged)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::executor::Executor;
    use crate::core::state::State;
    use crate::core::termination::TerminationReason;

    struct Quadratic {
        lo: f64,
        hi: f64,
    }
    impl CostFunction for Quadratic {
        type Param = f64;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, x: &f64) -> Result<f64, Self::Error> {
            Ok((x - 2.0).powi(2))
        }
    }
    impl BoxConstraints for Quadratic {
        fn lower(&self) -> &f64 {
            &self.lo
        }
        fn upper(&self) -> &f64 {
            &self.hi
        }
    }

    #[test]
    fn quadratic_finds_interior_min() {
        let r = Executor::new(
            Quadratic { lo: 0.0, hi: 5.0 },
            GoldenSection::new(),
            ScalarState::new(2.5),
        )
        .max_iter(100)
        .run()
        .unwrap();
        assert_eq!(r.reason, TerminationReason::SolverConverged);
        assert!((r.param() - 2.0).abs() < 1e-6, "x = {}", r.param());
        assert!(*r.param() >= 0.0 && *r.param() <= 5.0);
    }

    #[test]
    fn monotonic_function_converges_to_boundary() {
        // True min of (x-2)^2 is at x=2, but feasible region is [3, 5];
        // so the constrained min is the lower bound, x = 3.
        let r = Executor::new(
            Quadratic { lo: 3.0, hi: 5.0 },
            GoldenSection::new(),
            ScalarState::new(4.0),
        )
        .max_iter(200)
        .run()
        .unwrap();
        assert!((r.param() - 3.0).abs() < 1e-5, "x = {}", r.param());
    }

    struct Cubic {
        lo: f64,
        hi: f64,
    }
    impl CostFunction for Cubic {
        type Param = f64;
        type Output = f64;
        type Error = std::convert::Infallible;
        // x^3 − 3x; on [0, 2] the unique min is at x = 1, f(1) = −2.
        fn cost(&self, x: &f64) -> Result<f64, Self::Error> {
            Ok(x.powi(3) - 3.0 * x)
        }
    }
    impl BoxConstraints for Cubic {
        fn lower(&self) -> &f64 {
            &self.lo
        }
        fn upper(&self) -> &f64 {
            &self.hi
        }
    }

    #[test]
    fn cubic_unimodal_on_interval() {
        let r = Executor::new(
            Cubic { lo: 0.0, hi: 2.0 },
            GoldenSection::new(),
            ScalarState::new(0.5),
        )
        .max_iter(100)
        .run()
        .unwrap();
        assert_eq!(r.reason, TerminationReason::SolverConverged);
        assert!(
            (r.best_param() - 1.0).abs() < 1e-6,
            "x = {}",
            r.best_param()
        );
        assert!((r.best_cost() + 2.0).abs() < 1e-10, "f = {}", r.best_cost());
        // Golden section converges linearly (bracket shrinks by ≈0.618 per
        // step), so it needs ~40 evals to reach √ε on [0, 2], looser than
        // Brent's parabolic budget. A generous ceiling guards against a real
        // regression without pinning the exact count.
        assert!(
            r.state.cost_evals() < 80,
            "evals = {}",
            r.state.cost_evals()
        );
    }

    #[test]
    fn seed_is_ignored() {
        // Golden section ignores the starting point: two solves with very
        // different seeds on the same bracket must land on the same optimum.
        let run = |seed: f64| {
            Executor::new(
                Quadratic { lo: 0.0, hi: 5.0 },
                GoldenSection::new(),
                ScalarState::new(seed),
            )
            .max_iter(100)
            .run()
            .unwrap()
        };
        let a = run(0.01);
        let b = run(4.99);
        assert!(
            (a.param() - b.param()).abs() < 1e-9,
            "{} vs {}",
            a.param(),
            b.param()
        );
    }

    #[test]
    fn cost_tolerance_does_not_fire_on_non_improving_probe() {
        // Regression for issue #36 (mirrors the Brent test): the solver posts
        // the just-probed (u, fu) into `state.cost`, so `CostTolerance` sees
        // the real Δf signal and only stops at genuine convergence.
        use crate::core::termination::CostTolerance;
        let r = Executor::new(
            Cubic { lo: 0.0, hi: 2.0 },
            GoldenSection::new(),
            ScalarState::new(0.5),
        )
        .max_iter(200)
        .terminate_on(CostTolerance::new(1e-12))
        .run()
        .unwrap();
        // Looser than the Brent analog: golden section converges linearly, so
        // CostTolerance(1e-12) stops it while x is still ~1e-4 from the
        // optimum. These bounds still rule out the issue-#36 bug: a spurious
        // fire on the first probe would leave best_x near the ≈0.76 init point.
        assert!(
            (r.best_param() - 1.0).abs() < 1e-3,
            "best_x = {}, reason = {:?}",
            r.best_param(),
            r.reason
        );
        assert!(
            (r.best_cost() + 2.0).abs() < 1e-6,
            "best_cost = {}, reason = {:?}",
            r.best_cost(),
            r.reason
        );
        assert!(r.best_iter() > 0, "best_iter = {}", r.best_iter());
        assert!(
            r.best_cost_evals() > 0,
            "best_cost_evals = {}",
            r.best_cost_evals()
        );
    }
}
