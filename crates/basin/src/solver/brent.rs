use crate::core::constraint::BoxConstraints;
use crate::core::problem::{CostFunction, Problem};
use crate::core::solver::Solver;
use crate::core::state::BasicState;
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
/// # Backends
///
/// Scalar by construction: the parameter is a single `f64`, so Brent is
/// backend-agnostic and needs no linear-algebra backend. The problem's
/// `BoxConstraints` carry `f64` lower / upper bounds.
///
/// # Examples
///
/// `Brent` minimizes a 1-D function over a bracket, so the problem
/// implements `CostFunction` *and* `BoxConstraints` with scalar (`f64`)
/// bounds. See [`NelderMead`](crate::NelderMead) for the general
/// derivative-free `Executor` pattern.
pub struct Brent {
    tol_rel: f64,
    tol_abs: f64,
    inner: Option<Inner>,
}

/// `(3 − √5) / 2` — golden-section reduction factor used for the fallback
/// step (split the larger sub-interval at its golden ratio).
const GOLDEN_C: f64 = 0.381_966_011_250_105_2;

#[derive(Clone, Copy)]
struct Inner {
    a: f64,
    b: f64,
    x: f64,
    fx: f64,
    w: f64,
    fw: f64,
    v: f64,
    fv: f64,
    d: f64,
    e: f64,
}

impl Default for Brent {
    fn default() -> Self {
        Self::new()
    }
}

impl Brent {
    /// Brent solver with the standard tolerances: `tol_rel = √ε_f64`,
    /// `tol_abs = 1e-12`.
    pub fn new() -> Self {
        Self {
            tol_rel: f64::EPSILON.sqrt(),
            tol_abs: 1e-12,
            inner: None,
        }
    }

    /// Brent solver with explicit relative and absolute tolerances. Both
    /// must be strictly positive.
    pub fn with_tol(tol_rel: f64, tol_abs: f64) -> Self {
        assert!(tol_rel > 0.0, "tol_rel must be > 0");
        assert!(tol_abs > 0.0, "tol_abs must be > 0");
        Self {
            tol_rel,
            tol_abs,
            inner: None,
        }
    }
}

impl<P> Solver<P, BasicState<f64>> for Brent
where
    P: CostFunction<Param = f64, Output = f64> + BoxConstraints,
{
    type Error = P::Error;

    fn init(
        &mut self,
        problem: &mut Problem<P>,
        mut state: BasicState<f64>,
    ) -> Result<BasicState<f64>, Self::Error> {
        let a = *problem.inner().lower();
        let b = *problem.inner().upper();
        assert!(
            a.is_finite() && b.is_finite() && a < b,
            "Brent requires a finite, ordered bracket: lower < upper"
        );
        // Clamp the user-supplied seed into the bracket. If it lands on a
        // bound, nudge to a golden-section interior point so the first
        // iteration has somewhere to step.
        let mut x = state.param.clamp(a, b);
        if x == a || x == b {
            x = a + GOLDEN_C * (b - a);
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
            d: 0.0,
            e: 0.0,
        });
        state.param = x;
        state.cost = Some(fx);
        Ok(state)
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<P>,
        mut state: BasicState<f64>,
    ) -> Result<(BasicState<f64>, Option<TerminationReason>), Self::Error> {
        let s = self.inner.as_mut().expect("Brent::init must run first");
        let m = 0.5 * (s.a + s.b);
        let tol1 = self.tol_rel * s.x.abs() + self.tol_abs;
        let tol2 = 2.0 * tol1;

        let mut use_golden = true;
        if s.e.abs() > tol1 {
            // Parabola through (v, fv), (w, fw), (x, fx).
            let r = (s.x - s.w) * (s.fx - s.fv);
            let q0 = (s.x - s.v) * (s.fx - s.fw);
            let mut p = (s.x - s.v) * q0 - (s.x - s.w) * r;
            let mut q = 2.0 * (q0 - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let e_prev = s.e;
            // Accept only if the step is < half of the step before last and
            // stays strictly inside (a, b). Otherwise fall through to golden.
            if p.abs() < (0.5 * q * e_prev).abs() && p > q * (s.a - s.x) && p < q * (s.b - s.x) {
                s.e = s.d;
                s.d = p / q;
                let u = s.x + s.d;
                // Don't probe within `tol2` of either bound; round the step
                // off in the direction of the midpoint instead.
                if u - s.a < tol2 || s.b - u < tol2 {
                    s.d = if m - s.x >= 0.0 { tol1 } else { -tol1 };
                }
                use_golden = false;
            }
        }
        if use_golden {
            s.e = if s.x >= m { s.a - s.x } else { s.b - s.x };
            s.d = GOLDEN_C * s.e;
        }

        // Floor the magnitude of the step at `tol1` so we never evaluate
        // the cost at a point indistinguishable from `x`.
        let step = if s.d.abs() >= tol1 {
            s.d
        } else if s.d >= 0.0 {
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

        state.param = s.x;
        state.cost = Some(s.fx);
        Ok((state, None))
    }

    fn terminate(&self, _state: &BasicState<f64>) -> Option<TerminationReason> {
        let s = self.inner.as_ref()?;
        let m = 0.5 * (s.a + s.b);
        let tol = self.tol_rel * s.x.abs() + self.tol_abs;
        if (s.x - m).abs() + 0.5 * (s.b - s.a) <= 2.0 * tol {
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
            BasicState::new(2.5),
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
            BasicState::new(42.0),
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
            BasicState::new(4.0),
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
            BasicState::new(0.5),
        )
        .max_iter(100)
        .run()
        .unwrap();
        assert_eq!(r.reason, TerminationReason::SolverConverged);
        assert!((r.param() - 1.0).abs() < 1e-6, "x = {}", r.param());
        assert!((r.cost() + 2.0).abs() < 1e-10, "f = {}", r.cost());
        // Brent should converge in well under 100 cost evals on a smooth
        // cubic; if this regresses we want to know.
        assert!(r.state.cost_evals() < 30);
    }
}
