use super::ModelSolveError;
use crate::core::math::{
    ComponentMulAssign, Dot, NormInfinity, Scalar, ScaleInPlace,
};

pub(super) fn scaled_norm<V, F>(x: &V, d: &V) -> F
where
    F: Scalar,
    V: Clone + ComponentMulAssign + Dot<F> + NormInfinity<F> + ScaleInPlace<F>,
{
    let mut dx = d.clone();
    dx.component_mul_assign(x);
    let square = x.dot(&dx);
    if square.is_finite() && square > F::zero() {
        return square.sqrt();
    }

    // A large inactive parameter can have a representable scaled norm even
    // when its square overflows. Normalize both operands only on this rare
    // path, preserving the usual arithmetic for ordinary problem scales.
    let xscale = x.norm_infinity();
    let dscale = d.norm_infinity();
    if !xscale.is_finite() || !dscale.is_finite() {
        return F::nan();
    }
    if xscale == F::zero() || dscale == F::zero() {
        return F::zero();
    }
    let mut normalized = x.clone();
    // Two square-root factors keep reciprocals finite for subnormal scales.
    let xfactor = F::one() / xscale.sqrt();
    normalized.scale_in_place(xfactor);
    normalized.scale_in_place(xfactor);
    let mut diagonal = d.clone();
    let dfactor = F::one() / dscale.sqrt();
    diagonal.scale_in_place(dfactor);
    diagonal.scale_in_place(dfactor);
    diagonal.component_mul_assign(&normalized);
    (normalized.dot(&diagonal).sqrt() * dscale.sqrt()) * xscale
}

/// Search the secular equation using only the existing fixed-RHS solves.
/// Keeping the search independent of a triangular-factor representation also
/// preserves the capabilities required by downstream Cholesky backends.
pub(super) fn trust_region_step<V, F: Scalar>(
    radius: F,
    mu: &mut F,
    max_attempts: u32,
    gradient_norm: F,
    mut solve: impl FnMut(F) -> Result<V, ModelSolveError>,
    norm: impl Fn(&V) -> F,
) -> Result<V, ModelSolveError> {
    if !radius.is_finite() || radius <= F::zero() || !gradient_norm.is_finite()
    {
        return Err(ModelSolveError::Failed);
    }
    let ten = F::from_f64(10.0).unwrap();
    let tolerance = F::from_f64(0.1).unwrap();
    let previous_mu = *mu;
    let mut low = F::zero();
    let mut low_ratio = None;
    let mut high = (gradient_norm / radius).max(F::min_positive_value());
    let mut high_ratio = F::zero();
    let mut best = None;
    let mut candidate = F::zero();

    for attempt in 0..max_attempts {
        if !candidate.is_finite() {
            break;
        }
        match solve(candidate) {
            Ok(step) => {
                let pnorm = norm(&step);
                if !pnorm.is_finite() {
                    return Err(ModelSolveError::Failed);
                }
                let ratio = pnorm / radius;
                if ratio <= F::one() + tolerance
                    && (candidate == F::zero() || ratio >= F::one() - tolerance)
                {
                    *mu = candidate;
                    return Ok(step);
                }
                if ratio < F::one() {
                    high = candidate;
                    high_ratio = ratio;
                    best = Some(step);
                } else {
                    low = candidate;
                    low_ratio = Some(ratio);
                }
            }
            Err(ModelSolveError::Failed) => {
                return Err(ModelSolveError::Failed);
            }
            Err(ModelSolveError::Retry) => {
                // Rank failure gives a computational lower bound, but no
                // secular-function value suitable for interpolation.
                low = candidate;
                low_ratio = None;
            }
        }

        if best.is_none() {
            // The analytic upper bound need not be numerically solvable.
            // Establish a feasible endpoint before interpolating the bracket.
            high = if attempt == 0 { high } else { high * ten };
            candidate = high;
            continue;
        }

        let next = if attempt == 1 && previous_mu > low && previous_mu < high {
            previous_mu
        } else if low > F::zero() && low < high / ten {
            // A geometric midpoint resolves damping across many decades.
            low.sqrt() * high.sqrt()
        } else if let Some(low_ratio) = low_ratio {
            // Interpolate the reciprocal norm, whose secular function is
            // much closer to linear than the norm itself near singularity.
            let fraction = ((low_ratio - F::one()) / low_ratio) * high_ratio
                / (F::one() - high_ratio / low_ratio);
            let interpolated = low + fraction * (high - low);
            if interpolated > low && interpolated < high {
                interpolated
            } else {
                low + (high - low) / ten
            }
        } else if low == F::zero() {
            high / ten
        } else {
            low + (high - low) / F::from_f64(2.0).unwrap()
        };
        if next <= low || next >= high {
            break;
        }
        candidate = next;
    }

    // A deficient model may have no boundary root: its least-squares step
    // can lie strictly inside the radius. Retain a feasible regularized step
    // at the rank safeguard or attempt limit instead of accepting a long one.
    best.inspect(|_| {
        *mu = high;
    })
    .ok_or(ModelSolveError::Failed)
}

pub(super) fn update_radius<F: Scalar>(
    radius: &mut F,
    mu: &mut F,
    pnorm: F,
    rho: F,
    actual_reduction: F,
    directional_derivative: F,
) {
    if pnorm == F::zero() {
        return;
    }
    let half = F::from_f64(0.5).unwrap();
    let tenth = F::from_f64(0.1).unwrap();
    if !rho.is_finite() || rho <= F::from_f64(0.25).unwrap() {
        // MINPACK's interpolation shrinks further for a badly failed model.
        // Referencing the attempted step prevents repeated oversized GN steps
        // when the initial radius is much larger than the step itself.
        let shrink = if actual_reduction >= F::zero() {
            half
        } else {
            (half * directional_derivative
                / (directional_derivative + actual_reduction))
                .max(tenth)
                .min(half)
        };
        let shrink = if shrink.is_finite() { shrink } else { tenth };
        *radius =
            (shrink * radius.min(pnorm / tenth)).max(F::min_positive_value());
        *mu = *mu / shrink;
    } else if *mu == F::zero() || rho >= F::from_f64(0.75).unwrap() {
        *radius = (pnorm / half).min(F::max_value());
        *mu = *mu * half;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn scaled_norm_avoids_spurious_square_overflow_and_underflow() {
        for magnitude in [1e-200_f64, 1e200] {
            let norm = scaled_norm(&vec![0., magnitude], &vec![1., 1.]);
            assert!((norm / magnitude - 1.).abs() < 1e-14);
        }
        let norm = scaled_norm(&vec![0_f32, 1e20], &vec![1_f32, 1.]);
        assert!((norm / 1e20 - 1.).abs() < 1e-6);
    }

    #[test]
    fn scalar_radius_search_matches_analytic_damping() {
        // For J=2, r=-3, D=4, h=1.5/(1+mu). A radius of 1
        // therefore gives h=0.5 and mu=2, without trial callbacks.
        let mut mu = 0.;
        let h = trust_region_step(
            1.,
            &mut mu,
            50,
            3.,
            |mu| Ok(1.5 / (1. + mu)),
            |h: &f64| 2. * h.abs(),
        )
        .ok()
        .unwrap();
        assert!((h - 0.5).abs() < 1e-12);
        assert!((mu - 2.).abs() < 1e-12);
    }

    #[test]
    fn search_resolves_widely_separated_curvatures() {
        for curvature in [1e-4, 1e-12, 1e-20] {
            let mut mu = 0.;
            let h = trust_region_step(
                0.5,
                &mut mu,
                50,
                2.0_f64.sqrt(),
                |mu| Ok([1e-3 / (1. + mu), curvature / (curvature + mu)]),
                |h| (h[0] * h[0] + h[1] * h[1]).sqrt(),
            )
            .ok()
            .unwrap();
            let pnorm = (h[0] * h[0] + h[1] * h[1]).sqrt();
            assert!((pnorm - 0.5).abs() <= 0.05, "{curvature}: {pnorm}");
            assert!((mu / curvature - 1.).abs() < 0.25);
        }
    }

    #[test]
    fn rank_guard_can_require_more_than_the_analytic_upper_bound() {
        let mut mu = 0.;
        let h = trust_region_step(
            1.,
            &mut mu,
            50,
            1.,
            |mu| {
                if mu < 4. {
                    Err(ModelSolveError::Retry)
                } else {
                    Ok(1. / (1. + mu))
                }
            },
            |h: &f64| h.abs(),
        )
        .ok()
        .unwrap();
        assert!(mu >= 4.);
        assert!(h <= 0.2);
    }

    #[test]
    fn attempt_limit_returns_only_a_feasible_step() {
        for budget in [1, 2, 3] {
            let calls = Cell::new(0);
            let mut mu = 0.;
            let h = trust_region_step(
                0.1,
                &mut mu,
                budget,
                1.,
                |mu| {
                    calls.set(calls.get() + 1);
                    Ok(1. / (1. + mu))
                },
                |h: &f64| h.abs(),
            );
            assert!(calls.get() <= budget);
            match h {
                Ok(h) => assert!(h <= 0.11),
                Err(_) => assert_eq!(budget, 1),
            }
        }
    }

    #[test]
    fn failed_trial_always_shrinks_radius() {
        for actual in [-0.1, -9., -10., -100., f64::NEG_INFINITY] {
            let mut radius = 100.;
            let mut mu = 1.;
            update_radius(&mut radius, &mut mu, 2., -1., actual, -10.);
            assert!((2. ..=10.).contains(&radius), "{actual}: {radius}");
            assert!(mu >= 2.);
            if actual == -10. {
                assert_eq!(radius, 5.);
                assert_eq!(mu, 4.);
            }
        }
    }

    #[test]
    fn invalid_radius_fails_without_solving() {
        for radius in [0., -1., f64::NAN, f64::INFINITY] {
            assert!(
                trust_region_step(
                    radius,
                    &mut 0.,
                    50,
                    1.,
                    |_| -> Result<f64, _> {
                        panic!("invalid radius reached solve")
                    },
                    |h| h.abs(),
                )
                .is_err()
            );
        }
    }
}
