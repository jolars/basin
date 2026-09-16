use crate::core::math::Scalar;

use super::{
    BracketError, BracketTerminationReason, Sample, Search, evaluate,
    result_accessors, search_builders,
};

/// Result of an automatic root bracket search, including unsuccessful searches.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootBracketResult<F: Scalar = f64> {
    points: [Sample<F>; 2],
    iterations: u64,
    function_evals: u64,
    reason: BracketTerminationReason,
}

impl<F: Scalar> RootBracketResult<F> {
    /// Ordered, distinct endpoints. On failure these are the outermost samples,
    /// which need not bracket a root and may have an overflowing total width.
    pub fn bracket(&self) -> (F, F) {
        (self.points[0].x, self.points[1].x)
    }

    /// Function values at the returned endpoints.
    pub fn values(&self) -> (F, F) {
        (self.points[0].value, self.points[1].value)
    }

    result_accessors!();
}

/// Geometric search for a sign-changing scalar root bracket.
///
/// Starting from two ordered finite points, expand left and right independently.
/// Without a limit, multiply the distance from the opposite initial endpoint
/// by the growth factor each round. With a limit, divide the remaining gap by
/// that factor. On rounding stagnation at a finite limit, try the limit itself.
/// No callback is invoked outside the supplied search limits or at infinity.
///
/// The callback must be deterministic and continuous, returning finite values
/// at sampled points. For a monotonic function with a sign change reachable
/// inside the search domain, geometric expansion can find a bracket, subject
/// to iteration and floating-point limits. Nonmonotonic functions can have
/// roots skipped by the samples; failure does not prove absence of a root.
///
/// Both initial points are evaluated, even if one is a root. Every round
/// evaluates each direction that can advance. If both directions bracket a
/// root in the same round, return the narrower bracket, preferring the left
/// on a tie. Exact roots retain a distinct neighboring sampled endpoint, so
/// the result can be supplied directly to [`crate::BrentRoot`].
///
/// # Backends
///
/// Scalar `f64` and `f32`; no linear-algebra backend or optional feature.
///
/// # Example
///
/// ```
/// use basin::{BrentRoot, RootBracketer};
/// use std::convert::Infallible;
/// let f = |x| Ok::<_, Infallible>(x * x - 100.0);
/// let bracket = RootBracketer::new(0.0, 1.0).with_lower_bound(0.0)
///     .bracket(f).unwrap();
/// assert!(bracket.bracketed());
/// let (a, b) = bracket.bracket();
/// let root = BrentRoot::new(a, b).solve(f).unwrap();
/// assert!((root.root() - 10.0_f64).abs() < 1e-10);
/// ```
///
/// # References
///
/// SciPy 1.16.3, `scipy.optimize.elementwise.bracket_root`, scalar geometric
/// expansion. Basin retains distinct endpoints at exact roots and reports
/// non-finite function values as typed errors.
/// <https://docs.scipy.org/doc/scipy-1.16.3/reference/generated/scipy.optimize.elementwise.bracket_root.html>
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootBracketer<F: Scalar = f64> {
    lower: F,
    upper: F,
    search: Search<F>,
}

impl<F: Scalar> RootBracketer<F> {
    /// Start from two finite ordered points. Defaults to unrestricted search,
    /// growth factor `2`, and `1000` expansion rounds.
    pub fn new(lower: F, upper: F) -> Self {
        Self {
            lower,
            upper,
            search: Search::default(),
        }
    }

    search_builders!();

    /// Search for a bracket with a fallible callback.
    ///
    /// Invalid seeds fail before any evaluation. Exhausted limits and
    /// arithmetic stagnation return clean results with the last samples.
    pub fn bracket<C, E>(
        &self,
        mut function: C,
    ) -> Result<RootBracketResult<F>, BracketError<E, F>>
    where
        C: FnMut(F) -> Result<F, E>,
    {
        self.search.validate(&[self.lower, self.upper])?;
        let mut evaluations = 0;
        let mut points = [
            evaluate(&mut function, self.lower, &mut evaluations)?,
            evaluate(&mut function, self.upper, &mut evaluations)?,
        ];
        let mut iterations = 0;
        let mut distances = [self.lower - self.upper, self.upper - self.lower];
        let anchors = [self.upper, self.lower];
        let limits = [self.search.lower, self.search.upper];
        let mut stopped = [false; 2];
        let reason = if brackets(points) {
            BracketTerminationReason::Bracketed
        } else {
            loop {
                for i in 0..2 {
                    if limits[i] == Some(points[i].x) {
                        stopped[i] = true;
                    }
                }
                if stopped.iter().all(|&s| s) {
                    break if limits[0] == Some(points[0].x)
                        && limits[1] == Some(points[1].x)
                    {
                        BracketTerminationReason::BoundsReached
                    } else {
                        BracketTerminationReason::NoProgress
                    };
                }
                if iterations == self.search.max_iter {
                    break BracketTerminationReason::MaxIter;
                }
                let mut found: Option<[Sample<F>; 2]> = None;
                let before = evaluations;
                for i in 0..2 {
                    if stopped[i] {
                        continue;
                    }
                    let next = self.search.next(
                        points[i].x,
                        anchors[i],
                        &mut distances[i],
                        limits[i],
                    );
                    let Some(x) = next else {
                        stopped[i] = true;
                        continue;
                    };
                    let sample = evaluate(&mut function, x, &mut evaluations)?;
                    let pair = if i == 0 {
                        [sample, points[i]]
                    } else {
                        [points[i], sample]
                    };
                    if brackets(pair)
                        && found.is_none_or(|old| {
                            pair[1].x - pair[0].x < old[1].x - old[0].x
                        })
                    {
                        found = Some(pair);
                    }
                    points[i] = sample;
                }
                iterations += u64::from(evaluations != before);
                if let Some(pair) = found {
                    points = pair;
                    break BracketTerminationReason::Bracketed;
                }
            }
        };
        Ok(RootBracketResult {
            points,
            iterations,
            function_evals: evaluations,
            reason,
        })
    }
}

fn brackets<F: Scalar>(points: [Sample<F>; 2]) -> bool {
    let [a, b] = points;
    a.value == F::zero()
        || b.value == F::zero()
        || (a.value > F::zero()) != (b.value > F::zero())
}
