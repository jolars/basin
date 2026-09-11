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
    let parts = |factors: &[F]| {
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
    };
    parts(left) <= parts(right)
}

#[cfg(test)]
mod tests {
    use super::*;

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
