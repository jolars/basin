use crate::core::math::{
    ComponentZip, NormInfinity, NormSquared, Scalar, ScaleInPlace,
};

pub(super) fn finite_unchanged_trial<V, F>(x: &V, trial: &V) -> bool
where
    F: Scalar,
    V: ComponentZip<F>,
{
    x.all_zip(trial, |base, proposed| base.is_finite() && base == proposed)
}

pub(super) fn all_finite<V, F>(v: &V) -> bool
where
    F: Scalar,
    V: ComponentZip<F>,
{
    // An infinity norm can hide NaNs on backends that reduce with scalar max.
    v.all_zip(v, |value, _| value.is_finite())
}

pub(super) fn orthogonality_converged<V, F>(
    g: &V,
    diagonal: &V,
    r: &V,
    tol: F,
) -> bool
where
    F: Scalar,
    V: Clone
        + NormInfinity<F>
        + NormSquared<F>
        + ScaleInPlace<F>
        + ComponentZip<F>,
{
    let gradient = g.norm_infinity();
    let Some((rscale, rnorm)) = norm_parts(r) else {
        return false;
    };
    if !gradient.is_finite() || !diagonal.norm_infinity().is_finite() {
        return false;
    }
    // Exact zero refers to the gradient, before any normalization can underflow.
    if gradient == F::zero() {
        return true;
    }
    if tol == F::zero() || rscale == F::zero() {
        return false;
    }
    // Compare each cosine in factored form. Even dividing by a column norm
    // first can erase a subnormal projection before residual normalization.
    g.all_zip(diagonal, |value, square| {
        if square < F::zero() || !square.is_finite() {
            return false;
        }
        if square == F::zero() {
            return value == F::zero();
        }
        product_le(&[value.abs()], &[tol, rscale, rnorm, square.sqrt()])
    })
}

pub(super) fn relative_step_converged<V, F>(h: &V, x: &V, tol: F) -> bool
where
    F: Scalar,
    V: Clone + NormInfinity<F> + NormSquared<F> + ScaleInPlace<F>,
{
    let (Some((hscale, hnorm)), Some((xscale, xnorm))) =
        (norm_parts(h), norm_parts(x))
    else {
        return false;
    };
    if hscale == F::zero() {
        return true;
    }
    if tol == F::zero() || xscale == F::zero() {
        return false;
    }
    product_le(&[hscale, hnorm], &[tol, xscale, xnorm])
}

pub(super) fn relative_trust_radius_converged<V, F>(
    radius: F,
    x: &V,
    diagonal: &V,
    tol: F,
) -> bool
where
    F: Scalar,
    V: ComponentZip<F>,
{
    if !radius.is_finite()
        || radius < F::zero()
        || !tol.is_finite()
        || tol < F::zero()
    {
        return false;
    }
    let two = F::from_f64(2.).unwrap();
    let mut exponent = 0;
    let mut sum = F::zero();
    // Factor each weighted coordinate before summing. Separate normalizations
    // of x and D can erase every term when their large entries do not coincide.
    let finite = x.all_zip(diagonal, |value, square| {
        if !value.is_finite() || !square.is_finite() || square < F::zero() {
            return false;
        }
        if value == F::zero() || square == F::zero() {
            return true;
        }
        let (power, fraction) = product_parts(&[value.abs(), square.sqrt()]);
        if sum == F::zero() {
            exponent = power;
            sum = fraction * fraction;
        } else if power > exponent {
            let factor = two.powi(exponent - power);
            sum = (sum * factor) * factor + fraction * fraction;
            exponent = power;
        } else {
            let term = fraction * two.powi(power - exponent);
            sum = sum + term * term;
        }
        true
    });
    if !finite {
        return false;
    }
    if radius == F::zero() {
        return true;
    }
    if tol == F::zero() || sum == F::zero() {
        return false;
    }
    // Keep the norm's exponent separate even when the norm itself cannot fit.
    let (power, fraction) = product_parts(&[tol, sum.sqrt()]);
    product_parts(&[radius]) <= (power + exponent, fraction)
}

// Keep the norm factored: even an unsquared norm can exceed the scalar range.
fn norm_parts<V, F>(v: &V) -> Option<(F, F)>
where
    F: Scalar,
    V: Clone + NormInfinity<F> + NormSquared<F> + ScaleInPlace<F>,
{
    let scale = v.norm_infinity();
    if !scale.is_finite() {
        return None;
    }
    if scale == F::zero() {
        return Some((F::zero(), F::zero()));
    }
    let mut normalized = v.clone();
    let reciprocal = F::one() / scale;
    if reciprocal.is_finite() {
        normalized.scale_in_place(reciprocal);
    } else {
        // The reciprocal of a subnormal scale may not fit, but these factors do.
        let factor = F::one() / scale.sqrt();
        normalized.scale_in_place(factor);
        normalized.scale_in_place(factor);
    }
    Some((scale, normalized.norm_squared().sqrt()))
}

// Compare nonnegative finite products without overflowing or rounding a tiny
// tolerance to zero. Binary exponents remain integers until after comparison.
fn product_le<F: Scalar>(left: &[F], right: &[F]) -> bool {
    if left.contains(&F::zero()) {
        return true;
    }
    if right.contains(&F::zero()) {
        return false;
    }
    product_parts(left) <= product_parts(right)
}

// Factors must be positive and finite; callers handle zero before decoding.
fn product_parts<F: Scalar>(factors: &[F]) -> (i32, F) {
    let mut fraction = F::one();
    let mut exponent = 0_i32;
    for factor in factors {
        let (mantissa, power, _) = factor.integer_decode();
        let shift = 63 - mantissa.leading_zeros();
        fraction = fraction
            * (F::from_u64(mantissa).unwrap()
                / F::from_u64(1_u64 << shift).unwrap());
        exponent += i32::from(power) + shift as i32;
        if fraction >= F::from_f64(2.).unwrap() {
            fraction = fraction * F::from_f64(0.5).unwrap();
            exponent += 1;
        }
    }
    (exponent, fraction)
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! extreme_weighted_norms {
        ($name:ident, $scalar:ty, $power:expr) => {
            #[test]
            fn $name() {
                let two: $scalar = 2.;
                let large = two.powi($power);
                let small = two.powi(-$power);
                assert!(relative_trust_radius_converged(
                    5.,
                    &vec![3., -4.],
                    &vec![4., 4.],
                    0.5
                ));
                assert!(!relative_trust_radius_converged(
                    6.,
                    &vec![3., -4.],
                    &vec![4., 4.],
                    0.5
                ));
                // The weighted norm itself overflows or underflows, but
                // multiplying by the tolerance gives a representable bound.
                for (x, d, tol, bound) in [
                    (large, large, small, two.powi($power / 2)),
                    (small, small, large, two.powi(-$power / 2)),
                ] {
                    assert!(relative_trust_radius_converged(
                        bound,
                        &vec![x],
                        &vec![d],
                        tol
                    ));
                    assert!(!relative_trust_radius_converged(
                        two * bound,
                        &vec![x],
                        &vec![d],
                        tol
                    ));
                }
                // Independently normalizing x and D would lose both terms.
                let x = vec![large, small];
                let d = vec![small, large];
                let bound = two.powi($power / 2);
                assert!(relative_trust_radius_converged(
                    bound / two,
                    &x,
                    &d,
                    1.
                ));
                assert!(!relative_trust_radius_converged(
                    two * bound,
                    &x,
                    &d,
                    1.
                ));
                let tiny = <$scalar>::from_bits(1);
                assert!(relative_trust_radius_converged(
                    tiny,
                    &vec![tiny],
                    &vec![1.],
                    1.
                ));
                assert!(!relative_trust_radius_converged(
                    tiny,
                    &vec![large],
                    &vec![large],
                    0.
                ));
                assert!(relative_trust_radius_converged(
                    0.,
                    &vec![0.],
                    &vec![1.],
                    0.
                ));
                assert!(!relative_trust_radius_converged(
                    tiny,
                    &vec![0.],
                    &vec![1.],
                    large
                ));
                assert!(!relative_trust_radius_converged(
                    tiny,
                    &vec![large],
                    &vec![0.],
                    large
                ));
                for bad in [
                    <$scalar>::NAN,
                    <$scalar>::INFINITY,
                    <$scalar>::NEG_INFINITY,
                ] {
                    assert!(!relative_trust_radius_converged(
                        bad,
                        &vec![1.],
                        &vec![1.],
                        1.
                    ));
                    assert!(!relative_trust_radius_converged(
                        0.,
                        &vec![1., bad],
                        &vec![1., 1.],
                        0.
                    ));
                    assert!(!relative_trust_radius_converged(
                        0.,
                        &vec![1., 1.],
                        &vec![1., bad],
                        0.
                    ));
                    assert!(!relative_trust_radius_converged(
                        0.,
                        &vec![1.],
                        &vec![1.],
                        bad
                    ));
                }
                assert!(!relative_trust_radius_converged(
                    -1.,
                    &vec![1.],
                    &vec![1.],
                    1.
                ));
                assert!(!relative_trust_radius_converged(
                    0.,
                    &vec![1.],
                    &vec![-1.],
                    1.
                ));
                assert!(!relative_trust_radius_converged(
                    0.,
                    &vec![1.],
                    &vec![1.],
                    -1.
                ));
            }
        };
    }
    extreme_weighted_norms!(f32_extreme_weighted_norms, f32, 100);
    extreme_weighted_norms!(f64_extreme_weighted_norms, f64, 800);

    macro_rules! extreme_norms {
        ($name:ident, $scalar:ty) => {
            #[test]
            fn $name() {
                let tiny = <$scalar>::from_bits(1);
                let huge = <$scalar>::MAX;
                for scale in [tiny, huge] {
                    let h = vec![scale, scale];
                    let x = vec![scale, 0.];
                    assert!(!relative_step_converged(&h, &x, 1.));
                    assert!(relative_step_converged(&h, &x, 2.));
                    assert!(!relative_step_converged(&h, &x, 0.));
                    assert!(relative_step_converged(&vec![0., 0.], &x, 0.));
                }
                // The ratio underflows, but a nonzero step still fails exact zero.
                assert!(!relative_step_converged(&vec![tiny], &vec![huge], 0.));
                assert!(relative_step_converged(
                    &vec![tiny],
                    &vec![huge],
                    tiny
                ));
                // The right-hand norm overflows, but the finite bound is below one.
                assert!(!relative_step_converged(
                    &vec![1., 0.],
                    &vec![huge, huge],
                    tiny
                ));
            }
        };
    }
    extreme_norms!(f64_extreme_norms, f64);
    extreme_norms!(f32_extreme_norms, f32);

    #[test]
    fn nonfinite_inputs_cannot_satisfy_stopping_checks() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let bad = vec![value, 0.];
            let zero = vec![0., 0.];
            let finite = vec![1., 1.];
            for tolerance in [0., 1., f64::MAX] {
                assert!(!relative_step_converged(&bad, &finite, tolerance));
                assert!(!relative_step_converged(&zero, &bad, tolerance));
                assert!(!orthogonality_converged(
                    &bad, &finite, &finite, tolerance
                ));
                assert!(!orthogonality_converged(
                    &zero, &bad, &finite, tolerance
                ));
                assert!(!orthogonality_converged(
                    &zero, &finite, &bad, tolerance
                ));
            }
        }
    }

    #[test]
    fn zero_columns_and_exact_orthogonality() {
        let diagonal = vec![1., 0.];
        assert!(!orthogonality_converged(
            &vec![1., 0.],
            &diagonal,
            &vec![1., 0.],
            0.5
        ));
        assert!(orthogonality_converged(
            &vec![1., 0.],
            &diagonal,
            &vec![1., 0.],
            1.
        ));
        assert!(orthogonality_converged(
            &vec![0., 0.],
            &diagonal,
            &vec![0., 1.],
            0.
        ));
        assert!(!orthogonality_converged(
            &vec![1e-200, 0.],
            &vec![1e200, 0.],
            &vec![1e100, 1.],
            0.
        ));
    }

    #[test]
    fn subnormal_projection_does_not_become_orthogonal() {
        // J = [2, -1] and r = [tiny, tiny] give g = tiny and cosine 1/sqrt(10).
        let tiny = f64::from_bits(1);
        assert!(!orthogonality_converged(
            &vec![tiny],
            &vec![5.],
            &vec![tiny, tiny],
            0.25
        ));
        assert!(orthogonality_converged(
            &vec![tiny],
            &vec![5.],
            &vec![tiny, tiny],
            0.5
        ));
    }
}
