use crate::core::math::Scalar;

use super::common::{
    Bracket, Counts, Evaluator, Point, Settings, ValueOnly, num, root_builders,
    same_sign, secant,
};
use super::{RootError, RootResult};

/// TOMS Algorithm 748, Alefeld–Potra–Shi Algorithm 4.2 (`k = 2`).
///
/// A startup secant step is followed by cycles with two inverse cubic or
/// Newton-quadratic interpolations and a double-length secant step. If the
/// interval has not halved during a cycle, bisection enforces that reduction.
/// Newton-quadratic interpolation uses a polynomial model, not derivatives of
/// the callback. No user derivatives are required.
///
/// The callback must satisfy [`super::SecantRoot`]'s finite-value, continuity,
/// and endpoint sign contract. Convergence requires an exact zero or interval
/// width at most `absolute + relative * abs(best_endpoint)`. The result keeps
/// the evaluated endpoint with the smaller absolute residual and its value.
///
/// The iteration count includes the startup step and each subsequent cycle
/// that evaluates at least one trial. A cycle uses up to four evaluations and
/// can stop early on convergence. Zero iterations evaluates only endpoints.
/// Callback failures and invalid input return [`RootError`]; exhausting the
/// limit returns a non-converged [`RootResult`].
///
/// Function values are scaled before interpolation. Repeated values,
/// non-finite arithmetic, or exterior proposals cause lower-order interpolation
/// or bisection. These floating-point safeguards preserve the enclosing interval.
///
/// # Backends
///
/// Scalar `f64` and `f32`; no linear-algebra backend or optional feature.
///
/// # Example
///
/// ```
/// use basin::Toms748Root;
/// use std::convert::Infallible;
/// let root = Toms748Root::new(0.0, 2.0)
///     .solve(|x| Ok::<_, Infallible>(x * x - 2.0)).unwrap();
/// assert!(root.converged());
/// assert!((root.root() - 2.0_f64.sqrt()).abs() < 1e-10);
/// ```
///
/// # References
///
/// Alefeld, G. E., Potra, F. A., and Shi, Y. (1995). "Algorithm 748: Enclosing
/// Zeros of Continuous Functions." *ACM Transactions on Mathematical
/// Software*, 21(3), 327–344. <https://doi.org/10.1145/210089.210111>.
/// Algorithm 4.2 with `μ = 0.5`, independently implemented from the paper.
/// Final-output reference checks use SciPy 1.16.3 with `k = 2`; floating-point
/// safeguards and stopping rules differ, so general trajectory parity is not
/// claimed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Toms748Root<F: Scalar = f64> {
    settings: Settings<F>,
}

impl<F: Scalar> Toms748Root<F> {
    /// Create a bracketed solver with absolute tolerance `1e-12`, relative
    /// tolerance `4ε_F`, and a limit of 100 iterations (including startup).
    pub fn new(lower: F, upper: F) -> Self {
        Self {
            settings: Settings::new(lower, upper),
        }
    }

    root_builders!();

    /// Solve with a fallible value callback, starting afresh on every call.
    pub fn solve<C, E>(
        &self,
        function: C,
    ) -> Result<RootResult<F>, RootError<E, F>>
    where
        C: FnMut(F) -> Result<F, E>,
    {
        let mut eval = ValueOnly(function);
        let mut counts = Counts::default();
        let mut bracket =
            Bracket::initialize(&self.settings, &mut eval, &mut counts)?;
        if self.settings.max_iter == 0 || bracket.converged(&self.settings) {
            return Ok(bracket.result(&self.settings, 0, counts));
        }
        let candidate = interior(&bracket, secant(bracket.a, bracket.b));
        let point = eval.evaluate(candidate, &mut counts)?;
        let mut d = bracket.update(point);
        let mut e = None;
        let mut iterations = 1;
        while iterations < self.settings.max_iter
            && !bracket.converged(&self.settings)
        {
            iterations += 1;
            let start_width = bracket.width();
            for steps in [2, 3] {
                let cubic =
                    e.map(|e| inverse_cubic(bracket.a, bracket.b, d, e));
                let candidate = match cubic {
                    Some(c) if bracket.contains(c) => c,
                    _ => newton_quadratic(&bracket, d, steps),
                };
                let point =
                    eval.evaluate(interior(&bracket, candidate), &mut counts)?;
                e = Some(d);
                d = bracket.update(point);
                if bracket.converged(&self.settings) {
                    return Ok(bracket.result(
                        &self.settings,
                        iterations,
                        counts,
                    ));
                }
            }
            let best = bracket.best();
            let step = secant(bracket.a, bracket.b) - best.x;
            let candidate = if step.abs() <= num::<F>(0.25) * bracket.width() {
                best.x + num::<F>(2.0) * step
            } else {
                bracket.midpoint()
            };
            let point =
                eval.evaluate(interior(&bracket, candidate), &mut counts)?;
            e = Some(d);
            d = bracket.update(point);
            if bracket.converged(&self.settings) {
                return Ok(bracket.result(&self.settings, iterations, counts));
            }
            if bracket.width() > num::<F>(0.5) * start_width {
                let point = eval.evaluate(bracket.midpoint(), &mut counts)?;
                e = Some(d);
                d = bracket.update(point);
            }
        }
        Ok(bracket.result(&self.settings, iterations, counts))
    }
}

fn interior<F: Scalar>(bracket: &Bracket<F>, candidate: F) -> F {
    if bracket.contains(candidate) {
        candidate
    } else {
        bracket.midpoint()
    }
}

fn inverse_cubic<F: Scalar>(
    a: Point<F>,
    b: Point<F>,
    c: Point<F>,
    d: Point<F>,
) -> F {
    // The paper's ipzero recurrence (pp. 333–334) uses differences of
    // function values. Scaling prevents those differences from overflowing.
    let scale = a
        .value
        .abs()
        .max(b.value.abs())
        .max(c.value.abs())
        .max(d.value.abs());
    let [fa, fb, fc, fd] = [
        a.value / scale,
        b.value / scale,
        c.value / scale,
        d.value / scale,
    ];
    let values = [fa, fb, fc, fd];
    for i in 0..4 {
        for j in 0..i {
            if values[i] == values[j] {
                return F::nan();
            }
        }
    }
    let q11 = (c.x - d.x) * (fc / (fd - fc));
    let q21 = (b.x - c.x) * (fb / (fc - fb));
    let q31 = (a.x - b.x) * (fa / (fb - fa));
    let d21 = (b.x - c.x) * (fc / (fc - fb));
    let d31 = (a.x - b.x) * (fb / (fb - fa));
    let q22 = (d21 - q11) * (fb / (fd - fb));
    let q32 = (d31 - q21) * (fa / (fc - fa));
    let d32 = (d31 - q21) * (fc / (fc - fa));
    let q33 = (d32 - q22) * (fa / (fd - fa));
    a.x + (q31 + q32 + q33)
}

fn newton_quadratic<F: Scalar>(
    bracket: &Bracket<F>,
    d: Point<F>,
    steps: usize,
) -> F {
    // Affine coordinates put the live endpoints at 0 and 1. Together with
    // value scaling, this avoids forming dimensional second differences
    // that can overflow even when the polynomial's root is well behaved.
    let width = bracket.width();
    let td = (d.x - bracket.a.x) / width;
    let scale = bracket
        .a
        .value
        .abs()
        .max(bracket.b.value.abs())
        .max(d.value.abs());
    let fa = bracket.a.value / scale;
    let fb = bracket.b.value / scale;
    let fd = d.value / scale;
    let b = fb - fa;
    let a = ((fd - fb) / (td - F::one()) - b) / td;
    if !a.is_finite() || a == F::zero() {
        return secant(bracket.a, bracket.b);
    }
    let mut r = if same_sign(a, fa) {
        F::zero()
    } else {
        F::one()
    };
    for _ in 0..steps {
        let polynomial = (a * (r - F::one()) + b) * r + fa;
        let derivative = b + a * (num::<F>(2.0) * r - F::one());
        r = r - polynomial / derivative;
    }
    interior(bracket, bracket.a.x + width * r)
}
