use crate::core::constraint::BoxConstraints;
use crate::core::math::Scalar;
use crate::core::problem::{CostFunction, Problem};
use crate::core::solver::Solver;
use crate::core::state::ScalarState;
use crate::core::termination::TerminationReason;

/// Brent's method for 1D minimization on a closed interval `[lower, upper]`
/// supplied via `BoxConstraints`. Combines parabolic interpolation through
/// the three best points so far with a golden-section fallback when the
/// parabolic step is unacceptable. Brent (1973), as transcribed in
/// Numerical Recipes §10.2.
///
/// Convergence test (in `Solver::terminate`):
/// `|x − m| + 0.5·(b − a) ≤ 2·tol`, where `m = (a+b)/2`,
/// `tol = tol_rel·|x| + tol_abs`. NR-style defaults: `tol_rel = √ε`,
/// `tol_abs = 1e-12`.
///
/// Brent runs on [`ScalarState`], the
/// cost-only single-iterate state. It does **not** impl
/// [`GradientState`](crate::core::state::GradientState) (a 1-D minimizer has
/// no gradient), so attaching a gradient criterion such as
/// [`GradientTolerance`](crate::core::termination::GradientTolerance) is a
/// **compile error** rather than one that silently never fires.
///
/// # Backends
///
/// Scalar by construction: the parameter is a single `F` (default
/// `f64`), so Brent is backend-agnostic and needs no linear-algebra
/// backend. The problem's `BoxConstraints` carry `F`-valued lower and
/// upper bounds.
///
/// # Examples
///
/// `Brent` minimizes a 1-D function over a bracket, so the problem
/// implements `CostFunction` *and* `BoxConstraints` with scalar (`F`)
/// bounds. See [`NelderMead`](crate::NelderMead) for the general
/// derivative-free `Executor` pattern.
pub struct Brent<F = f64> {
    tol_rel: F,
    tol_abs: F,
    inner: Option<Inner<F>>,
}

/// `(3 − √5) / 2`, the golden-section reduction factor used for the fallback
/// step (split the larger sub-interval at its golden ratio).
fn golden_c<F: Scalar>() -> F {
    F::from_f64(0.381_966_011_250_105_2).unwrap()
}

#[derive(Clone, Copy)]
struct Inner<F> {
    a: F,
    b: F,
    x: F,
    fx: F,
    w: F,
    fw: F,
    v: F,
    fv: F,
    d: F,
    e: F,
}

impl<F: Scalar> Default for Brent<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Scalar> Brent<F> {
    /// Brent solver with the standard tolerances: `tol_rel = √ε_F`,
    /// `tol_abs = 1e-12`.
    pub fn new() -> Self {
        Self {
            tol_rel: F::epsilon().sqrt(),
            tol_abs: F::from_f64(1e-12).unwrap(),
            inner: None,
        }
    }

    /// Brent solver with explicit relative and absolute tolerances. Both
    /// must be strictly positive.
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

impl<P, F> Solver<P, ScalarState<F>> for Brent<F>
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
            "Brent requires a finite, ordered bracket: lower < upper"
        );
        // Clamp the user-supplied seed into the bracket. If it lands on a
        // bound, nudge to a golden-section interior point so the first
        // iteration has somewhere to step. `Float` has no `clamp`, so use
        // `.max(a).min(b)`, the same output as `f64::clamp` on a well-ordered
        // finite bracket (which the assert above guarantees).
        let mut x = state.param.max(a).min(b);
        if x == a || x == b {
            x = a + golden_c::<F>() * (b - a);
        }
        let fx = problem.cost(&x)?;
        self.inner = Some(Inner {
            a,
            b,
            x,
            fx,
            w: x,
            fw: fx,
            v: x,
            fv: fx,
            d: F::zero(),
            e: F::zero(),
        });
        state.param = x;
        state.cost = Some(fx);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: ScalarState<F>,
    ) -> Result<(ScalarState<F>, Option<TerminationReason>), Self::Error> {
        let s = self.inner.as_mut().expect("Brent::init must run first");
        let half = F::from_f64(0.5).unwrap();
        let two = F::from_f64(2.0).unwrap();
        let m = half * (s.a + s.b);
        let tol1 = self.tol_rel * s.x.abs() + self.tol_abs;
        let tol2 = two * tol1;

        let mut use_golden = true;
        if s.e.abs() > tol1 {
            // Parabola through (v, fv), (w, fw), (x, fx).
            let r = (s.x - s.w) * (s.fx - s.fv);
            let q0 = (s.x - s.v) * (s.fx - s.fw);
            let mut p = (s.x - s.v) * q0 - (s.x - s.w) * r;
            let mut q = two * (q0 - r);
            if q > F::zero() {
                p = -p;
            }
            q = q.abs();
            let e_prev = s.e;
            // Accept only if the step is < half of the step before last and
            // stays strictly inside (a, b). Otherwise fall through to golden.
            if p.abs() < (half * q * e_prev).abs() && p > q * (s.a - s.x) && p < q * (s.b - s.x) {
                s.e = s.d;
                s.d = p / q;
                let u = s.x + s.d;
                // Don't probe within `tol2` of either bound; round the step
                // off in the direction of the midpoint instead.
                if u - s.a < tol2 || s.b - u < tol2 {
                    s.d = if m - s.x >= F::zero() { tol1 } else { -tol1 };
                }
                use_golden = false;
            }
        }
        if use_golden {
            s.e = if s.x >= m { s.a - s.x } else { s.b - s.x };
            s.d = golden_c::<F>() * s.e;
        }

        // Floor the magnitude of the step at `tol1` so we never evaluate
        // the cost at a point indistinguishable from `x`.
        let step = if s.d.abs() >= tol1 {
            s.d
        } else if s.d >= F::zero() {
            tol1
        } else {
            -tol1
        };
        let u = s.x + step;
        let fu = problem.cost(&u)?;

        if fu <= s.fx {
            if u >= s.x {
                s.a = s.x;
            } else {
                s.b = s.x;
            }
            s.v = s.w;
            s.fv = s.fw;
            s.w = s.x;
            s.fw = s.fx;
            s.x = u;
            s.fx = fu;
        } else {
            if u < s.x {
                s.a = u;
            } else {
                s.b = u;
            }
            if fu <= s.fw || s.w == s.x {
                s.v = s.w;
                s.fv = s.fw;
                s.w = u;
                s.fw = fu;
            } else if fu <= s.fv || s.v == s.x || s.v == s.w {
                s.v = u;
                s.fv = fu;
            }
        }

        // Post the just-probed point (u, fu) into the state, not the
        // retained best (s.x, s.fx). This honors `ScalarState`'s
        // documented "current iterate" semantics, so one-step change
        // tests like `CostTolerance` see real Δf signals instead of
        // firing on the unchanged s.fx after a non-improving probe
        // (issue #36). The executor's best-so-far tracking
        // (`state.best_param`/`state.best_cost`) captures the
        // true optimum independently of what's reported here, and
        // Brent's bracket-collapse `terminate` only fires once
        // `|u - x| < 2·tol`, so the two coincide at convergence.
        state.param = u;
        state.cost = Some(fu);
        Ok((state, None))
    }

    fn terminate(&self, _state: &ScalarState<F>) -> Option<TerminationReason> {
        let s = self.inner.as_ref()?;
        let half = F::from_f64(0.5).unwrap();
        let two = F::from_f64(2.0).unwrap();
        let m = half * (s.a + s.b);
        let tol = self.tol_rel * s.x.abs() + self.tol_abs;
        if (s.x - m).abs() + half * (s.b - s.a) <= two * tol {
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
            Brent::new(),
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
    fn quadratic_seed_outside_bracket_is_clamped() {
        // Seed > upper; init should clamp into [0, 5] and still converge.
        let r = Executor::new(
            Quadratic { lo: 0.0, hi: 5.0 },
            Brent::new(),
            ScalarState::new(42.0),
        )
        .max_iter(100)
        .run()
        .unwrap();
        assert!((r.param() - 2.0).abs() < 1e-6, "x = {}", r.param());
    }

    #[test]
    fn monotonic_function_converges_to_boundary() {
        // True min of (x-2)^2 is at x=2, but feasible region is [3, 5];
        // so the constrained min is the lower bound, x = 3.
        let r = Executor::new(
            Quadratic { lo: 3.0, hi: 5.0 },
            Brent::new(),
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
            Brent::new(),
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
        // Brent should converge in well under 100 cost evals on a smooth
        // cubic; if this regresses we want to know.
        assert!(r.state.cost_evals() < 30);
    }

    #[test]
    fn cost_tolerance_does_not_fire_on_non_improving_probe() {
        // Regression for issue #36: before the fix, Brent posted its
        // retained best `s.fx` into `state.cost` every iter, so a
        // non-improving probe left `state.cost` unchanged and
        // `CostTolerance` fired immediately (|Δf| = 0 ≤ tol). Now
        // Brent posts the just-probed `(u, fu)`, so CostTolerance
        // sees the real Δf signal; the only legitimate trigger is
        // genuine convergence.
        use crate::core::termination::CostTolerance;
        let r = Executor::new(
            Cubic { lo: 0.0, hi: 2.0 },
            Brent::new(),
            ScalarState::new(0.5),
        )
        .max_iter(200)
        .terminate_on(CostTolerance::new(1e-12))
        .run()
        .unwrap();
        // Whatever stop fires (Brent's own bracket-collapse or
        // CostTolerance kicking in once Δf legitimately drops below
        // 1e-12 near the optimum), the best-so-far must still be the
        // analytical optimum.
        assert!(
            (r.best_param() - 1.0).abs() < 1e-5,
            "best_x = {}, reason = {:?}",
            r.best_param(),
            r.reason
        );
        assert!(
            (r.best_cost() + 2.0).abs() < 1e-9,
            "best_cost = {}, reason = {:?}",
            r.best_cost(),
            r.reason
        );
        // And the best-tracking metadata should be populated.
        assert!(r.best_iter() > 0, "best_iter = {}", r.best_iter());
        assert!(
            r.best_cost_evals() > 0,
            "best_cost_evals = {}",
            r.best_cost_evals()
        );
    }
}
