use crate::core::math::Scalar;

use super::{
    BracketError, BracketTerminationReason, Sample, Search, evaluate,
    result_accessors, search_builders,
};

/// Result of an automatic minimum bracket search, including unsuccessful searches.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinimumBracketResult<F: Scalar = f64> {
    points: [Sample<F>; 3],
    iterations: u64,
    function_evals: u64,
    reason: BracketTerminationReason,
}

impl<F: Scalar> MinimumBracketResult<F> {
    /// Ordered sampled points `(left, middle, right)`. Check
    /// [`bracketed`](Self::bracketed) before treating these as a minimum bracket.
    pub fn bracket(&self) -> (F, F, F) {
        (self.points[0].x, self.points[1].x, self.points[2].x)
    }

    /// Function values at the three returned points.
    pub fn values(&self) -> (F, F, F) {
        (
            self.points[0].value,
            self.points[1].value,
            self.points[2].value,
        )
    }

    result_accessors!();
}

/// Geometric downhill search for an interior scalar minimum bracket.
///
/// A bracket consists of `left < middle < right`, with the middle value no
/// greater than either endpoint and at least one strict inequality. Starting
/// with three finite ordered samples, move toward the lower-cost endpoint
/// (right on a tie), shifting the last two points into the next triple.
///
/// Unbounded trial distances grow geometrically from the initial leading
/// endpoint. With a finite search limit, the remaining gap shrinks by the
/// growth factor until the limit is reached. No callback is invoked outside
/// the supplied limits or at infinity. A boundary minimum returns
/// [`BracketTerminationReason::BoundsReached`], not an interior bracket.
///
/// The callback must be deterministic and finite at the sampled points.
/// For a continuous strongly unimodal function with an interior minimum, the
/// search can find a bracket subject to arithmetic and iteration limits.
/// For other functions it may miss a minimum; a bracket is only a local
/// enclosure and never a global-optimality certificate.
///
/// # Backends
///
/// Scalar `f64` and `f32`; no linear-algebra backend or optional feature.
///
/// # Example
///
/// The discovered interval becomes problem-side bounds for a scalar solver.
/// Bracketing and optimization have separate evaluation counts.
///
/// ```
/// use basin::{BoxConstraints, Brent, CostFunction, Executor, MinimumBracketer,
///             ScalarState};
/// use std::convert::Infallible;
/// struct Quadratic { lower: f64, upper: f64 }
/// impl CostFunction for Quadratic {
///     type Param = f64;
///     type Output = f64;
///     type Error = Infallible;
///     fn cost(&self, x: &f64) -> Result<f64, Infallible> { Ok((x - 10.0).powi(2)) }
/// }
/// impl BoxConstraints for Quadratic {
///     fn lower(&self) -> &f64 { &self.lower }
///     fn upper(&self) -> &f64 { &self.upper }
/// }
/// let bracket = MinimumBracketer::new(-1.0, 0.0, 1.0)
///     .bracket(|x: f64| Ok::<_, Infallible>((x - 10.0).powi(2))).unwrap();
/// assert!(bracket.bracketed());
/// let (lower, middle, upper) = bracket.bracket();
/// let result = Executor::new(Quadratic { lower, upper }, Brent::new(),
///                            ScalarState::new(middle)).max_iter(100).run().unwrap();
/// assert!((result.best_param() - 10.0).abs() < 1e-6);
/// ```
///
/// # References
///
/// SciPy 1.16.3, `scipy.optimize.elementwise.bracket_minimum`, scalar geometric
/// search. The implementation's unbounded distances grow from the initial
/// leading endpoint; Basin follows that recurrence.
/// <https://docs.scipy.org/doc/scipy-1.16.3/reference/generated/scipy.optimize.elementwise.bracket_minimum.html>
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinimumBracketer<F: Scalar = f64> {
    points: [F; 3],
    search: Search<F>,
}

impl<F: Scalar> MinimumBracketer<F> {
    /// Start from three finite ordered points. Defaults to unrestricted search,
    /// growth factor `2`, and `1000` expansion rounds.
    pub fn new(left: F, middle: F, right: F) -> Self {
        Self {
            points: [left, middle, right],
            search: Search::default(),
        }
    }

    search_builders!();

    /// Search for a minimum bracket with a fallible callback.
    pub fn bracket<C, E>(
        &self,
        mut function: C,
    ) -> Result<MinimumBracketResult<F>, BracketError<E, F>>
    where
        C: FnMut(F) -> Result<F, E>,
    {
        self.search.validate(&self.points)?;
        let mut evaluations = 0;
        let [a, m, b] = self.points;
        let mut points = [
            evaluate(&mut function, a, &mut evaluations)?,
            evaluate(&mut function, m, &mut evaluations)?,
            evaluate(&mut function, b, &mut evaluations)?,
        ];
        let limit = if points[0].value < points[2].value {
            points.swap(0, 2);
            self.search.lower
        } else {
            self.search.upper
        };
        let anchor = points[2].x;
        let mut distance = points[2].x - points[1].x;
        let mut iterations = 0;
        let reason = loop {
            let [a, m, b] = points.map(|p| p.value);
            if m <= a && m <= b && (m < a || m < b) {
                break BracketTerminationReason::Bracketed;
            }
            if limit == Some(points[2].x) {
                break BracketTerminationReason::BoundsReached;
            }
            if iterations == self.search.max_iter {
                break BracketTerminationReason::MaxIter;
            }
            let Some(x) =
                self.search.next(points[2].x, anchor, &mut distance, limit)
            else {
                break BracketTerminationReason::NoProgress;
            };
            // The next triple must have a finite width so existing bounded
            // minimizers can consume a successful search without overflow.
            if !(x - points[1].x).is_finite() {
                break BracketTerminationReason::NoProgress;
            }
            let next = evaluate(&mut function, x, &mut evaluations)?;
            points = [points[1], points[2], next];
            iterations += 1;
        };
        if points[0].x > points[2].x {
            points.swap(0, 2);
        }
        Ok(MinimumBracketResult {
            points,
            iterations,
            function_evals: evaluations,
            reason,
        })
    }
}
