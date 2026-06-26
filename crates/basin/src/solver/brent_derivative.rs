use crate::core::constraint::BoxConstraints;
use crate::core::math::Scalar;
use crate::core::problem::{CostFunction, Gradient, Problem};
use crate::core::solver::Solver;
use crate::core::state::ScalarGradientState;
use crate::core::termination::TerminationReason;

/// Brent's method using first derivatives ("dbrent") for 1D minimization on a
/// closed interval `[lower, upper]` supplied via `BoxConstraints`. Within the
/// bracket it uses the **sign of `f'`** to choose which half holds the minimum
/// and **secant extrapolation on `f'`** (through the two best points) to pick
/// the step, falling back to derivative-informed bisection when the secant
/// step is unacceptable. Brent (1973), as transcribed in Numerical Recipes
/// §10.3.
///
/// This is the gradient-using sibling of [`Brent`](crate::solver::Brent):
/// same bracketing robustness, but a cheap first derivative lets it converge
/// faster and enables a natural stopping test. It runs on
/// [`ScalarGradientState`], which carries the scalar `f'(x)` and *does* impl
/// [`GradientState`](crate::core::state::GradientState), so
/// [`GradientTolerance`](crate::core::termination::GradientTolerance) ("stop
/// when `|f'(x)| ≤ tol`") works here, unlike on the derivative-free
/// [`Brent`](crate::solver::Brent)/[`GoldenSection`](crate::solver::GoldenSection).
///
/// Convergence test (in `Solver::terminate`):
/// `|x − m| + 0.5·(b − a) ≤ 2·tol`, where `m = (a+b)/2`,
/// `tol = tol_rel·|x| + tol_abs`. NR-style defaults: `tol_rel = √ε`,
/// `tol_abs = 1e-12`.
///
/// # Backends
///
/// Scalar by construction: the parameter and gradient are each a single `F`
/// (default `f64`), so `BrentDerivative` is backend-agnostic and needs no
/// linear-algebra backend. The problem's `BoxConstraints` carry `F`-valued
/// lower and upper bounds.
///
/// # Examples
///
/// `BrentDerivative` minimizes a 1-D function over a bracket using its
/// derivative, so the problem implements `CostFunction`, `Gradient` (with
/// `Gradient = F`), *and* `BoxConstraints` with scalar (`F`) bounds. See
/// [`Brent`](crate::solver::Brent) for the derivative-free sibling.
///
/// # References
///
/// Brent, R. P. (1973). *Algorithms for Minimization without Derivatives*,
/// chapter on the derivative form. Transcribed (the `dbrent` routine) in
/// Numerical Recipes §10.3.
pub struct BrentDerivative<F = f64> {
    tol_rel: F,
    tol_abs: F,
    inner: Option<Inner<F>>,
}

#[derive(Clone, Copy)]
struct Inner<F> {
    a: F,
    b: F,
    x: F,
    fx: F,
    dx: F,
    w: F,
    fw: F,
    dw: F,
    v: F,
    fv: F,
    dv: F,
    d: F,
    e: F,
}

/// `(3 − √5) / 2`, the golden-section interior point used only to nudge a seed
/// off a bracket boundary so the first iteration has somewhere to step.
fn golden_c<F: Scalar>() -> F {
    F::from_f64(0.381_966_011_250_105_2).unwrap()
}

impl<F: Scalar> Default for BrentDerivative<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Scalar> BrentDerivative<F> {
    /// Solver with the standard tolerances: `tol_rel = √ε_F`,
    /// `tol_abs = 1e-12`.
    pub fn new() -> Self {
        Self {
            tol_rel: F::epsilon().sqrt(),
            tol_abs: F::from_f64(1e-12).unwrap(),
            inner: None,
        }
    }

    /// Solver with explicit relative and absolute tolerances. Both must be
    /// strictly positive.
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

impl<P, F> Solver<P, ScalarGradientState<F>> for BrentDerivative<F>
where
    F: Scalar,
    P: CostFunction<Param = F, Output = F> + Gradient<Gradient = F> + BoxConstraints,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: ScalarGradientState<F>,
    ) -> Result<ScalarGradientState<F>, Self::Error> {
        let a = *problem.inner().lower();
        let b = *problem.inner().upper();
        assert!(
            a.is_finite() && b.is_finite() && a < b,
            "BrentDerivative requires a finite, ordered bracket: lower < upper"
        );
        // Clamp the user-supplied seed into the bracket. If it lands on a
        // bound, nudge to a golden-section interior point so the first
        // iteration has somewhere to step. (`Float` has no `clamp`; on a
        // well-ordered finite bracket `.max(a).min(b)` matches `f64::clamp`.)
        let mut x = state.param.max(a).min(b);
        if x == a || x == b {
            x = a + golden_c::<F>() * (b - a);
        }
        let (fx, dx) = problem.cost_and_gradient(&x)?;
        self.inner = Some(Inner {
            a,
            b,
            x,
            fx,
            dx,
            w: x,
            fw: fx,
            dw: dx,
            v: x,
            fv: fx,
            dv: dx,
            d: F::zero(),
            e: F::zero(),
        });
        state.param = x;
        state.cost = Some(fx);
        state.gradient = Some(dx);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: ScalarGradientState<F>,
    ) -> Result<(ScalarGradientState<F>, Option<TerminationReason>), Self::Error> {
        let s = self
            .inner
            .as_mut()
            .expect("BrentDerivative::init must run first");
        let half = F::from_f64(0.5).unwrap();
        let two = F::from_f64(2.0).unwrap();
        let m = half * (s.a + s.b);
        let tol1 = self.tol_rel * s.x.abs() + self.tol_abs;
        let tol2 = two * tol1;

        // Derivative-informed bisection step: head into the half the sign of
        // `f'(x)` points at. Used both as the fallback and when no usable
        // secant history exists.
        let bisect = |s: &Inner<F>| -> (F, F) {
            let e = if s.dx >= F::zero() {
                s.a - s.x
            } else {
                s.b - s.x
            };
            (e, half * e)
        };

        if s.e.abs() > tol1 {
            // Two secant estimates of where `f'` vanishes, through (x, dx) and
            // each of (w, dw) / (v, dv). Default large so they're rejected when
            // a partner derivative coincides with `dx`.
            let big = two * (s.b - s.a);
            let mut d1 = big;
            let mut d2 = big;
            if s.dw != s.dx {
                d1 = (s.w - s.x) * s.dx / (s.dx - s.dw);
            }
            if s.dv != s.dx {
                d2 = (s.v - s.x) * s.dx / (s.dx - s.dv);
            }
            // Accept a step only if it lands strictly inside (a, b) and points
            // downhill (opposite sign to `f'(x)`).
            let u1 = s.x + d1;
            let u2 = s.x + d2;
            let ok1 = (s.a - u1) * (u1 - s.b) > F::zero() && s.dx * d1 <= F::zero();
            let ok2 = (s.a - u2) * (u2 - s.b) > F::zero() && s.dx * d2 <= F::zero();
            let olde = s.e;
            s.e = s.d;
            let chosen = if ok1 && ok2 {
                if d1.abs() < d2.abs() {
                    Some(d1)
                } else {
                    Some(d2)
                }
            } else if ok1 {
                Some(d1)
            } else if ok2 {
                Some(d2)
            } else {
                None
            };
            match chosen {
                // Keep the secant step only if it moves less than half the step
                // before last; otherwise bisect.
                Some(d) if d.abs() <= (half * olde).abs() => {
                    s.d = d;
                    let u = s.x + d;
                    // Don't probe within `tol2` of a bound; round toward the
                    // midpoint instead.
                    if u - s.a < tol2 || s.b - u < tol2 {
                        s.d = if m - s.x >= F::zero() { tol1 } else { -tol1 };
                    }
                }
                _ => {
                    let (e, d) = bisect(s);
                    s.e = e;
                    s.d = d;
                }
            }
        } else {
            let (e, d) = bisect(s);
            s.e = e;
            s.d = d;
        }

        // Floor the magnitude of the step at `tol1` so we never probe a point
        // indistinguishable from `x`.
        let step = if s.d.abs() >= tol1 {
            s.d
        } else if s.d >= F::zero() {
            tol1
        } else {
            -tol1
        };
        let u = s.x + step;
        let (fu, du) = problem.cost_and_gradient(&u)?;

        if fu <= s.fx {
            if u >= s.x {
                s.a = s.x;
            } else {
                s.b = s.x;
            }
            s.v = s.w;
            s.fv = s.fw;
            s.dv = s.dw;
            s.w = s.x;
            s.fw = s.fx;
            s.dw = s.dx;
            s.x = u;
            s.fx = fu;
            s.dx = du;
        } else {
            if u < s.x {
                s.a = u;
            } else {
                s.b = u;
            }
            if fu <= s.fw || s.w == s.x {
                s.v = s.w;
                s.fv = s.fw;
                s.dv = s.dw;
                s.w = u;
                s.fw = fu;
                s.dw = du;
            } else if fu <= s.fv || s.v == s.x || s.v == s.w {
                s.v = u;
                s.fv = fu;
                s.dv = du;
            }
        }

        // Post the just-probed (u, fu, du) into the state, not the retained
        // best (s.x, ...). This honors `ScalarGradientState`'s "current
        // iterate" semantics, so one-step change tests like `CostTolerance`
        // see real Δf signals instead of firing on an unchanged cost after a
        // non-improving probe (issue #36), and `GradientTolerance` reads the
        // gradient at the point actually probed. The executor's best-so-far
        // tracking captures the true optimum independently, and the
        // bracket-collapse `terminate` only fires once the bracket has shrunk
        // below tolerance, so the two coincide at convergence.
        state.param = u;
        state.cost = Some(fu);
        state.gradient = Some(du);
        Ok((state, None))
    }

    fn terminate(&self, _state: &ScalarGradientState<F>) -> Option<TerminationReason> {
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
    use crate::core::termination::{GradientTolerance, TerminationReason};

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
    impl Gradient for Quadratic {
        type Gradient = f64;
        fn gradient(&self, x: &f64) -> Result<f64, Self::Error> {
            Ok(2.0 * (x - 2.0))
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
            BrentDerivative::new(),
            ScalarGradientState::new(2.5),
        )
        .max_iter(100)
        .run()
        .unwrap();
        assert_eq!(r.reason, TerminationReason::SolverConverged);
        assert!((r.param() - 2.0).abs() < 1e-7, "x = {}", r.param());
        assert!(*r.param() >= 0.0 && *r.param() <= 5.0);
    }

    #[test]
    fn monotonic_function_converges_to_boundary() {
        // True min of (x-2)^2 is at x=2, but feasible region is [3, 5];
        // so the constrained min is the lower bound, x = 3.
        let r = Executor::new(
            Quadratic { lo: 3.0, hi: 5.0 },
            BrentDerivative::new(),
            ScalarGradientState::new(4.0),
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
    impl Gradient for Cubic {
        type Gradient = f64;
        fn gradient(&self, x: &f64) -> Result<f64, Self::Error> {
            Ok(3.0 * x * x - 3.0)
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
            BrentDerivative::new(),
            ScalarGradientState::new(0.5),
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
        // Derivative information should make this converge in few evals on a
        // smooth cubic; guard against a real regression.
        assert!(
            r.state.cost_evals() < 25,
            "evals = {}",
            r.state.cost_evals()
        );
    }

    #[test]
    fn gradient_tolerance_stops() {
        // The payoff of the gradient-carrying state: a first-order termination
        // criterion is usable on a 1D solver. Stop when |f'(x)| ≤ 1e-4, looser
        // than the solver's own bracket-collapse tolerance (≈√ε), so the
        // gradient criterion is what fires.
        let r = Executor::new(
            Cubic { lo: 0.0, hi: 2.0 },
            BrentDerivative::new(),
            ScalarGradientState::new(0.5),
        )
        .max_iter(200)
        .terminate_on(GradientTolerance(1e-4))
        .run()
        .unwrap();
        assert_eq!(r.reason, TerminationReason::GradientTolerance);
        // |f'(x)| = 3|x²−1| ≈ 6|x−1| near the optimum, so |f'| ≤ 1e-4 puts the
        // best iterate within ~2e-5 of x = 1.
        assert!(
            (r.best_param() - 1.0).abs() < 1e-3,
            "best_x = {}",
            r.best_param()
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
            BrentDerivative::new(),
            ScalarGradientState::new(0.5),
        )
        .max_iter(200)
        .terminate_on(CostTolerance::new(1e-12))
        .run()
        .unwrap();
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
        assert!(r.best_iter() > 0, "best_iter = {}", r.best_iter());
        assert!(
            r.best_cost_evals() > 0,
            "best_cost_evals = {}",
            r.best_cost_evals()
        );
    }
}
