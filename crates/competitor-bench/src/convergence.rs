//! Callback traces shared by the published LM and COBYLA comparisons.

use std::time::Instant;

#[derive(Debug)]
pub struct Summary {
    pub point: Vec<f64>,
    pub cost: f64,
    pub violation: f64,
    pub objective_calls: usize,
    pub residual_calls: usize,
    pub jacobian_calls: usize,
    pub gradient_calls: usize,
    pub constraint_calls: usize,
    pub status: String,
    /// Independent solution accuracy, separate from the solver's termination.
    pub verified: bool,
}

pub struct Run {
    pub points: Vec<(u128, f64)>,
    pub violations: Vec<(u128, f64)>,
    pub summary: Summary,
}

impl Summary {
    fn same_result(&self, other: &Self) -> bool {
        // Repeated numerical failures must remain publishable even when a
        // returned value is NaN, which does not compare equal to itself.
        let same_float = |a: f64, b: f64| a == b || (a.is_nan() && b.is_nan());
        self.point.len() == other.point.len()
            && self
                .point
                .iter()
                .zip(&other.point)
                .all(|(&a, &b)| same_float(a, b))
            && same_float(self.cost, other.cost)
            && same_float(self.violation, other.violation)
            && self.objective_calls == other.objective_calls
            && self.residual_calls == other.residual_calls
            && self.jacobian_calls == other.jacobian_calls
            && self.gradient_calls == other.gradient_calls
            && self.constraint_calls == other.constraint_calls
            && self.status == other.status
            && self.verified == other.verified
    }
}

pub(crate) struct Log {
    start: Instant,
    samples: Vec<(u128, f64, Vec<f64>)>,
}

impl Log {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            samples: Vec::new(),
        }
    }

    pub fn elapsed(&self) -> u128 {
        self.start.elapsed().as_nanos()
    }

    pub fn restart(&mut self) {
        assert!(self.samples.is_empty());
        self.start = Instant::now();
    }

    pub fn record(&mut self, cost: f64, point: &[f64]) {
        self.samples.push((self.elapsed(), cost, point.to_vec()));
    }

    pub fn finish(
        mut self,
        elapsed: u128,
        summary: Summary,
        feasibility_tolerance: Option<f64>,
        violation: impl Fn(&[f64]) -> f64,
    ) -> Run {
        // The endpoint includes work after the final callback. Verification and
        // feasibility reconstruction happen after the caller stops the clock.
        self.samples
            .push((elapsed, summary.cost, summary.point.clone()));
        let mut points = Vec::new();
        let mut violations = Vec::new();
        let mut best = f64::INFINITY;
        for (time, cost, point) in self.samples {
            let v = violation(&point);
            if let Some(tolerance) = feasibility_tolerance {
                if v.is_finite() {
                    violations.push((time, v));
                }
                if !v.is_finite() || v > tolerance {
                    if best.is_finite() {
                        points.push((time, best));
                    }
                    continue;
                }
            }
            if cost.is_finite() && point.iter().all(|x| x.is_finite()) {
                best = best.min(cost);
            }
            if best.is_finite() {
                points.push((time, best));
            }
        }
        Run {
            points,
            violations,
            summary,
        }
    }
}

/// Warm up once, then median timestamps only when all numerical paths agree.
pub fn median_run(mut run: impl FnMut() -> Run, repetitions: usize) -> Run {
    assert!(repetitions > 0 && repetitions % 2 == 1);
    drop(run());
    let mut runs: Vec<_> = (0..repetitions).map(|_| run()).collect();
    for candidate in &runs[1..] {
        assert!(
            candidate.summary.same_result(&runs[0].summary),
            "returned solution changed between runs"
        );
        for (a, b) in [
            (&candidate.points, &runs[0].points),
            (&candidate.violations, &runs[0].violations),
        ] {
            assert_eq!(a.len(), b.len(), "trace length changed between runs");
            assert!(a.iter().zip(b).all(|(a, b)| a.1 == b.1));
        }
    }
    let points = median_points(runs.iter().map(|r| &r.points).collect());
    let violations =
        median_points(runs.iter().map(|r| &r.violations).collect());
    Run {
        points,
        violations,
        summary: runs.remove(0).summary,
    }
}

fn median_points(runs: Vec<&Vec<(u128, f64)>>) -> Vec<(u128, f64)> {
    (0..runs[0].len())
        .map(|i| {
            let mut times: Vec<_> = runs.iter().map(|r| r[i].0).collect();
            times.sort_unstable();
            (times[times.len() / 2], runs[0][i].1)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_nonfinite_failures_retain_their_status() {
        let result = median_run(
            || Run {
                points: Vec::new(),
                violations: Vec::new(),
                summary: Summary {
                    point: vec![f64::NAN],
                    cost: f64::NAN,
                    violation: f64::INFINITY,
                    objective_calls: 0,
                    residual_calls: 0,
                    jacobian_calls: 0,
                    gradient_calls: 0,
                    constraint_calls: 0,
                    status: "SolverFailed".into(),
                    verified: false,
                },
            },
            3,
        );
        assert!(!result.summary.verified);
        assert_eq!(result.summary.status, "SolverFailed");
        assert!(result.points.is_empty());
    }

    #[test]
    fn infeasible_lower_objective_does_not_improve_the_curve() {
        let mut log = Log::new();
        log.record(0.5, &[0.5, 0.5]);
        log.record(0.0, &[1.0, 1.0]);
        let elapsed = log.elapsed();
        let run = log.finish(
            elapsed,
            Summary {
                point: vec![0.75, 0.75],
                cost: 0.125,
                violation: 0.0,
                objective_calls: 2,
                residual_calls: 0,
                jacobian_calls: 0,
                gradient_calls: 0,
                constraint_calls: 2,
                status: "Radius".into(),
                verified: true,
            },
            Some(1e-8),
            |x| (x[0] + x[1] - 1.5).max(0.0),
        );
        assert_eq!(
            run.points.iter().map(|p| p.1).collect::<Vec<_>>(),
            [0.5, 0.5, 0.125]
        );
        assert_eq!(run.violations[1].1, 0.5);
    }
}
