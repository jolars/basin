use std::fmt;

use crate::core::math::Scalar;

/// Why a root-finding run stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RootTerminationReason {
    /// The current point is an exact root or the bracket meets the requested
    /// position tolerance.
    Converged,
    /// The configured iteration limit was reached.
    MaxIter,
}

/// Result of a bracketed root-finding run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootResult<F = f64> {
    root: F,
    value: F,
    lower: F,
    upper: F,
    iterations: u64,
    function_evals: u64,
    reason: RootTerminationReason,
}

impl<F: Scalar> RootResult<F> {
    fn new(
        root: F,
        value: F,
        bracket: (F, F),
        work: (u64, u64),
        reason: RootTerminationReason,
    ) -> Self {
        let (a, b) = bracket;
        let (iterations, function_evals) = work;
        let (lower, upper) = if a <= b { (a, b) } else { (b, a) };
        Self {
            root,
            value,
            lower,
            upper,
            iterations,
            function_evals,
            reason,
        }
    }

    /// Best available root estimate.
    pub fn root(&self) -> F {
        self.root
    }

    /// Signed function value at [`root`](Self::root).
    pub fn value(&self) -> F {
        self.value
    }

    /// Final ordered bracket containing [`root`](Self::root), provided the
    /// callback satisfies [`BrentRoot`]'s continuity contract.
    pub fn bracket(&self) -> (F, F) {
        (self.lower, self.upper)
    }

    /// Number of interpolation or bisection steps performed.
    pub fn iterations(&self) -> u64 {
        self.iterations
    }

    /// Number of function evaluations performed, including the endpoints.
    pub fn function_evals(&self) -> u64 {
        self.function_evals
    }

    /// Why the run stopped.
    pub fn reason(&self) -> RootTerminationReason {
        self.reason
    }

    /// Whether the root criterion was met before the iteration limit.
    pub fn converged(&self) -> bool {
        self.reason == RootTerminationReason::Converged
    }
}

/// Failure to initialize or evaluate a Brent root-finding run.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum BrentRootError<E, F = f64> {
    /// The callback returned an application error.
    Evaluation(E),
    /// The interval is non-finite, empty, or reversed.
    InvalidInterval {
        /// Supplied lower endpoint.
        lower: F,
        /// Supplied upper endpoint.
        upper: F,
    },
    /// The finite endpoint values have the same nonzero sign.
    NotBracketed {
        /// Supplied lower endpoint.
        lower: F,
        /// Supplied upper endpoint.
        upper: F,
        /// Function value at `lower`.
        f_lower: F,
        /// Function value at `upper`.
        f_upper: F,
    },
    /// The callback returned `NaN` or infinity.
    NonFiniteValue {
        /// Point that was evaluated.
        x: F,
        /// Non-finite value returned at `x`.
        value: F,
    },
}

impl<E, F> fmt::Display for BrentRootError<E, F>
where
    E: fmt::Display,
    F: fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Evaluation(error) => {
                write!(formatter, "root function failed: {error}")
            }
            Self::InvalidInterval { lower, upper } => write!(
                formatter,
                "root interval must be finite and ordered, got [{lower:?}, {upper:?}]"
            ),
            Self::NotBracketed {
                lower,
                upper,
                f_lower,
                f_upper,
            } => write!(
                formatter,
                "root is not bracketed on [{lower:?}, {upper:?}]: endpoint values are {f_lower:?} and {f_upper:?}"
            ),
            Self::NonFiniteValue { x, value } => write!(
                formatter,
                "root function returned non-finite value {value:?} at {x:?}"
            ),
        }
    }
}

impl<E, F> std::error::Error for BrentRootError<E, F>
where
    E: std::error::Error + 'static,
    F: fmt::Debug + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Evaluation(error) => Some(error),
            _ => None,
        }
    }
}

/// Brent's bracketed method for a scalar equation `f(x) = 0`.
///
/// Each step chooses between bisection, the secant method, and inverse
/// quadratic interpolation. The sign-changing bracket makes convergence as
/// reliable as bisection while interpolation normally makes it substantially
/// faster.
///
/// This is a direct root-finding API, not an optimization
/// [`Solver`](crate::Solver). The interval belongs to the algorithm and is
/// therefore passed to [`new`](Self::new), rather than represented as
/// [`BoxConstraints`](crate::BoxConstraints). Clean iteration-limit exits
/// return a [`RootResult`]; invalid brackets, non-finite values, and callback
/// failures return [`BrentRootError`].
///
/// # Function contract
///
/// The callback must be a deterministic scalar function that is continuous on
/// the interval. Its values at the endpoints must be finite and have opposite
/// signs, unless an endpoint is exactly zero. Errors returned by the callback
/// are preserved in [`BrentRootError::Evaluation`].
///
/// # Backends
///
/// Scalar by construction. `F` may be `f64` or `f32`; no linear-algebra
/// backend or optional feature is required.
///
/// # Example
///
/// ```
/// use std::convert::Infallible;
///
/// use basin::BrentRoot;
///
/// let result = BrentRoot::new(0.0, 2.0)
///     .solve(|x| Ok::<_, Infallible>(x * x - 2.0))
///     .unwrap();
///
/// assert!(result.converged());
/// assert!((result.root() - 2.0_f64.sqrt()).abs() < 1e-10);
/// ```
///
/// # References
///
/// Brent, R. P. (1971). "An algorithm with guaranteed convergence for finding
/// a zero of a function." *The Computer Journal*, 14(4), 422–425.
/// <https://doi.org/10.1093/comjnl/14.4.422>
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrentRoot<F = f64> {
    lower: F,
    upper: F,
    tol_rel: F,
    tol_abs: F,
    max_iter: u64,
}

impl<F: Scalar> BrentRoot<F> {
    /// Construct a root finder on `[lower, upper]` with default tolerances.
    ///
    /// The defaults are `tol_rel = 4ε_F`, `tol_abs = 1e-12`, and a limit of
    /// 100 iterations. Interval validity and the endpoint signs are checked by
    /// [`solve`](Self::solve), where they can be reported as typed errors.
    pub fn new(lower: F, upper: F) -> Self {
        Self {
            lower,
            upper,
            tol_rel: F::from_f64(4.0).unwrap() * F::epsilon(),
            tol_abs: F::from_f64(1e-12).unwrap(),
            max_iter: 100,
        }
    }

    /// Set relative and absolute position tolerances.
    ///
    /// # Panics
    ///
    /// Panics unless both tolerances are finite, `tol_rel` is at least four
    /// times the machine epsilon of `F`, and `tol_abs` is strictly positive.
    #[deprecated(
        note = "use `with_relative_position_tolerance` and `with_absolute_position_tolerance`; removal scheduled for Basin 2.0"
    )]
    pub fn with_tol(mut self, tol_rel: F, tol_abs: F) -> Self {
        let min_tol_rel = F::from_f64(4.0).unwrap() * F::epsilon();
        assert!(
            tol_rel.is_finite() && tol_rel >= min_tol_rel,
            "BrentRoot relative tolerance must be finite and at least four times machine epsilon"
        );
        assert!(
            tol_abs.is_finite() && tol_abs > F::zero(),
            "BrentRoot absolute tolerance must be finite and positive"
        );
        self.tol_rel = tol_rel;
        self.tol_abs = tol_abs;
        self
    }

    /// Set the finite, strictly positive absolute position tolerance.
    pub fn with_absolute_position_tolerance(mut self, value: F) -> Self {
        assert!(
            value.is_finite() && value > F::zero(),
            "absolute position tolerance must be finite and positive"
        );
        self.tol_abs = value;
        self
    }
    /// Set the finite relative position tolerance, at least four times machine epsilon.
    pub fn with_relative_position_tolerance(mut self, value: F) -> Self {
        assert!(
            value.is_finite()
                && value >= F::from_f64(4.0).unwrap() * F::epsilon(),
            "relative position tolerance must be finite and at least four times machine epsilon"
        );
        self.tol_rel = value;
        self
    }

    /// Set the maximum number of interpolation or bisection steps.
    ///
    /// A zero limit still evaluates both endpoints so exact endpoint roots and
    /// invalid brackets remain distinguishable.
    pub fn with_max_iter(mut self, max_iter: u64) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Find a root using a fallible scalar callback.
    pub fn solve<Function, E>(
        &self,
        mut function: Function,
    ) -> Result<RootResult<F>, BrentRootError<E, F>>
    where
        Function: FnMut(F) -> Result<F, E>,
    {
        if !self.lower.is_finite()
            || !self.upper.is_finite()
            || self.lower >= self.upper
            || !(self.upper - self.lower).is_finite()
        {
            return Err(BrentRootError::InvalidInterval {
                lower: self.lower,
                upper: self.upper,
            });
        }

        let mut function_evals = 1;
        let mut a = self.lower;
        let mut fa = function(a).map_err(BrentRootError::Evaluation)?;
        if !fa.is_finite() {
            return Err(BrentRootError::NonFiniteValue { x: a, value: fa });
        }
        if fa == F::zero() {
            return Ok(RootResult::new(
                a,
                fa,
                (self.lower, self.upper),
                (0, function_evals),
                RootTerminationReason::Converged,
            ));
        }

        function_evals += 1;
        let mut b = self.upper;
        let mut fb = function(b).map_err(BrentRootError::Evaluation)?;
        if !fb.is_finite() {
            return Err(BrentRootError::NonFiniteValue { x: b, value: fb });
        }
        if fb == F::zero() {
            return Ok(RootResult::new(
                b,
                fb,
                (self.lower, self.upper),
                (0, function_evals),
                RootTerminationReason::Converged,
            ));
        }
        if same_nonzero_sign(fa, fb) {
            return Err(BrentRootError::NotBracketed {
                lower: self.lower,
                upper: self.upper,
                f_lower: fa,
                f_upper: fb,
            });
        }

        let mut c = b;
        let mut fc = fb;
        let mut d = b - a;
        let mut e = d;
        let half = F::from_f64(0.5).unwrap();
        let two = F::from_f64(2.0).unwrap();
        let three = F::from_f64(3.0).unwrap();

        for iteration in 0..self.max_iter {
            if same_nonzero_sign(fb, fc) {
                c = a;
                fc = fa;
                d = b - a;
                e = d;
            }
            if fc.abs() < fb.abs() {
                (a, b, c) = (b, c, b);
                (fa, fb, fc) = (fb, fc, fb);
            }

            let tolerance = half * (self.tol_abs + self.tol_rel * b.abs());
            let midpoint = half * (c - b);
            if midpoint.abs() <= tolerance || fb == F::zero() {
                return Ok(RootResult::new(
                    b,
                    fb,
                    (b, c),
                    (iteration, function_evals),
                    RootTerminationReason::Converged,
                ));
            }

            if e.abs() >= tolerance && fa.abs() > fb.abs() {
                let s = fb / fa;
                let (mut p, mut q) = if a == c {
                    (two * midpoint * s, F::one() - s)
                } else {
                    let q = fa / fc;
                    let r = fb / fc;
                    (
                        s * (two * midpoint * q * (q - r)
                            - (b - a) * (r - F::one())),
                        (q - F::one()) * (r - F::one()) * (s - F::one()),
                    )
                };
                if p > F::zero() {
                    q = -q;
                }
                p = p.abs();
                let interpolation_bound =
                    three * midpoint * q - (tolerance * q).abs();
                let history_bound = (e * q).abs();
                if two * p < interpolation_bound.min(history_bound) {
                    e = d;
                    d = p / q;
                } else {
                    d = midpoint;
                    e = d;
                }
            } else {
                d = midpoint;
                e = d;
            }

            a = b;
            fa = fb;
            b = if d.abs() > tolerance {
                b + d
            } else if midpoint > F::zero() {
                b + tolerance
            } else {
                b - tolerance
            };
            function_evals += 1;
            fb = function(b).map_err(BrentRootError::Evaluation)?;
            if !fb.is_finite() {
                return Err(BrentRootError::NonFiniteValue { x: b, value: fb });
            }
            if fb == F::zero() {
                return Ok(RootResult::new(
                    b,
                    fb,
                    (b, b),
                    (iteration + 1, function_evals),
                    RootTerminationReason::Converged,
                ));
            }
        }

        if same_nonzero_sign(fb, fc) {
            c = a;
            fc = fa;
        }
        if fc.abs() < fb.abs() {
            std::mem::swap(&mut b, &mut c);
            fb = fc;
        }
        Ok(RootResult::new(
            b,
            fb,
            (b, c),
            (self.max_iter, function_evals),
            RootTerminationReason::MaxIter,
        ))
    }
}

fn same_nonzero_sign<F: Scalar>(a: F, b: F) -> bool {
    (a > F::zero() && b > F::zero()) || (a < F::zero() && b < F::zero())
}
