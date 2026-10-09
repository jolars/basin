//! Independent CDP-1 quality tests in declared, fixed units.

#[derive(Clone, Copy, Debug)]
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
}

impl Interval {
    pub fn exact(value: f64) -> Self {
        Self {
            lower: value,
            upper: value,
        }
    }

    fn finite_ordered(self) -> bool {
        self.lower.is_finite()
            && self.upper.is_finite()
            && self.lower <= self.upper
    }
}

/// Eligibility must be established independently before candidate execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Eligibility {
    Eligible,
    ReferencePending,
    PrecisionIneligible,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetStatus {
    Passed,
    Missed,
    NonFinite,
    InvalidReference,
    ReferencePending,
    PrecisionIneligible,
}

#[derive(Clone, Debug)]
pub struct Quality {
    pub objective_error: f64,
    pub stationarity: f64,
    pub feasibility: f64,
    pub parameter_error: Option<f64>,
    invalid_reference: bool,
    require_parameter: bool,
}

impl Quality {
    pub fn target(&self, q: f64, eligibility: Eligibility) -> TargetStatus {
        assert!(q.is_finite() && q > 0.0);
        if self.invalid_reference {
            return TargetStatus::InvalidReference;
        }
        if ![self.objective_error, self.stationarity, self.feasibility]
            .iter()
            .all(|v| v.is_finite())
            || self.parameter_error.is_some_and(|e| !e.is_finite())
        {
            return TargetStatus::NonFinite;
        }
        match eligibility {
            Eligibility::ReferencePending => {
                return TargetStatus::ReferencePending;
            }
            Eligibility::PrecisionIneligible => {
                return TargetStatus::PrecisionIneligible;
            }
            Eligibility::Eligible => {}
        }
        if self.objective_error <= q
            && self.stationarity <= q.sqrt()
            && self.feasibility <= q
            && (!self.require_parameter
                || self.parameter_error.is_some_and(|e| e <= q.sqrt()))
        {
            TargetStatus::Passed
        } else {
            TargetStatus::Missed
        }
    }
}

/// Smooth unconstrained or box-constrained certificate. General constraints
/// require an independent multiplier verifier and are deliberately unsupported.
#[derive(Clone, Debug)]
pub struct SmoothCertificate {
    pub reference: Interval,
    pub objective_scale: f64,
    pub coordinate_scales: Vec<f64>,
    pub solution_set: Vec<Vec<f64>>,
    pub require_parameter: bool,
    pub bounds: Option<(Vec<f64>, Vec<f64>)>,
}

impl SmoothCertificate {
    pub fn validate(&self) -> Result<(), &'static str> {
        let n = self.coordinate_scales.len();
        if n == 0
            || !self.reference.finite_ordered()
            || !self.objective_scale.is_finite()
            || self.objective_scale <= 0.0
            || self
                .coordinate_scales
                .iter()
                .any(|s| !s.is_finite() || *s <= 0.0)
        {
            return Err("invalid reference or scales");
        }
        if self
            .solution_set
            .iter()
            .any(|x| x.len() != n || x.iter().any(|v| !v.is_finite()))
            || (self.require_parameter && self.solution_set.is_empty())
        {
            return Err("invalid solution set");
        }
        if let Some((lower, upper)) = &self.bounds {
            if lower.len() != n
                || upper.len() != n
                || lower.iter().zip(upper).any(|(l, u)| {
                    l.is_nan()
                        || u.is_nan()
                        || l > u
                        || *l == f64::INFINITY
                        || *u == f64::NEG_INFINITY
                })
            {
                return Err("invalid bounds");
            }
        }
        Ok(())
    }

    /// `objective` uses the unshifted formula; cached solver cost is excluded.
    pub fn assess(
        &self,
        x: &[f64],
        objective: Interval,
        gradient: &[f64],
    ) -> Quality {
        self.validate().expect("validated certificate");
        assert_eq!(x.len(), self.coordinate_scales.len());
        assert_eq!(gradient.len(), x.len());
        let nonfinite = x.iter().chain(gradient).any(|v| !v.is_finite())
            || !objective.finite_ordered();
        if nonfinite {
            return Quality {
                objective_error: f64::INFINITY,
                stationarity: f64::INFINITY,
                feasibility: f64::INFINITY,
                parameter_error: None,
                invalid_reference: false,
                require_parameter: self.require_parameter,
            };
        }
        let mut stationarity: f64 = 0.0;
        let mut feasibility: f64 = 0.0;
        for (i, (&xi, &gi)) in x.iter().zip(gradient).enumerate() {
            let scale = self.coordinate_scales[i];
            let g = scale * gi / self.objective_scale;
            let displacement = if let Some((lower, upper)) = &self.bounds {
                feasibility = feasibility.max(
                    ((lower[i] - xi) / scale)
                        .max((xi - upper[i]) / scale)
                        .max(0.0),
                );
                // Work in displacement units to avoid cancellation in z-project(z-g).
                g.clamp((xi - upper[i]) / scale, (xi - lower[i]) / scale)
            } else {
                g
            };
            stationarity = stationarity.max(displacement.abs());
        }
        let parameter_error = (!self.solution_set.is_empty()).then(|| {
            self.solution_set
                .iter()
                .map(|r| {
                    x.iter()
                        .zip(r)
                        .zip(&self.coordinate_scales)
                        .map(|((&v, &r), &s)| ((v - r) / s).abs())
                        .fold(0.0, f64::max)
                })
                .fold(f64::INFINITY, f64::min)
        });
        Quality {
            objective_error: (objective.upper - self.reference.lower).max(0.0)
                / self.objective_scale,
            stationarity,
            feasibility,
            parameter_error,
            invalid_reference: objective.finite_ordered()
                && objective.upper < self.reference.lower,
            require_parameter: self.require_parameter,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RootCertificate {
    pub roots: Vec<f64>,
    pub position_scale: f64,
    pub residual_scale: f64,
}

impl RootCertificate {
    /// Exact-root exits may retain a wide bracket, but still need independent
    /// position and residual checks. A rounded callback zero never suffices.
    pub fn target(
        &self,
        x: f64,
        residual: f64,
        bracket: Option<(f64, f64)>,
        exact_exit: bool,
        q: f64,
        eligibility: Eligibility,
    ) -> TargetStatus {
        assert!(q.is_finite() && q > 0.0);
        assert!(
            !self.roots.is_empty() && self.roots.iter().all(|r| r.is_finite())
        );
        assert!(self.position_scale.is_finite() && self.position_scale > 0.0);
        assert!(self.residual_scale.is_finite() && self.residual_scale > 0.0);
        if !x.is_finite() || !residual.is_finite() {
            return TargetStatus::NonFinite;
        }
        match eligibility {
            Eligibility::ReferencePending => {
                return TargetStatus::ReferencePending;
            }
            Eligibility::PrecisionIneligible => {
                return TargetStatus::PrecisionIneligible;
            }
            Eligibility::Eligible => {}
        }
        let position = self
            .roots
            .iter()
            .map(|r| (x - r).abs())
            .fold(f64::INFINITY, f64::min);
        let point_passes = position <= q * self.position_scale
            && residual.abs() <= q * self.residual_scale;
        let bracket_passes = bracket.is_none_or(|(l, u)| {
            l.is_finite()
                && u.is_finite()
                && l <= u
                && self.roots.iter().any(|r| l <= *r && *r <= u)
                && ((u - l) * 0.5 <= q * self.position_scale
                    || (exact_exit && point_passes))
        });
        if point_passes && bracket_passes {
            TargetStatus::Passed
        } else {
            TargetStatus::Missed
        }
    }
}
