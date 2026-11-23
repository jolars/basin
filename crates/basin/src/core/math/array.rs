use rand::{Rng, RngExt};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};

use super::Scalar;
use super::cl_scaling::{
    BoxAffineScaling, cl_scaling_pair, max_feasible_step_component,
    project_strictly_inside_component,
};
use super::sample::{
    SampleStandardNormal, SampleUniformBox, assert_finite_box,
};
use super::{
    ClampInPlace, ComponentDivAssign, ComponentMaxAssign, ComponentMulAssign,
    ComponentZip, Dot, FloorZerosInPlace, NegInPlace, NormInfinity,
    NormSquared, ScaleInPlace, ScaledAdd, VectorIndex, VectorLen,
};

impl<F: Scalar, const N: usize> ScaledAdd<F> for [F; N] {
    fn scaled_add(&mut self, scalar: F, other: &Self) {
        for (x, y) in self.iter_mut().zip(other.iter()) {
            *x = *x + scalar * *y;
        }
    }

    fn scaled_add_with_change(
        &mut self,
        scalar: F,
        other: &Self,
    ) -> Option<bool> {
        let mut changed = false;
        for (x, y) in self.iter_mut().zip(other.iter()) {
            let next = *x + scalar * *y;
            changed |= next != *x;
            *x = next;
        }
        Some(changed)
    }
}

impl<F: Scalar, const N: usize> NormSquared<F> for [F; N] {
    fn norm_squared(&self) -> F {
        self.iter().map(|x| *x * *x).sum()
    }

    fn norm_l2(&self) -> F {
        super::norm_l2(self.iter().copied())
    }
}

impl<F: Scalar, const N: usize> NormInfinity<F> for [F; N] {
    fn norm_infinity(&self) -> F {
        super::norm_infinity(self.iter().copied())
    }
}

impl<F: Scalar, const N: usize> Dot<F> for [F; N] {
    fn dot(&self, other: &Self) -> F {
        self.iter().zip(other.iter()).map(|(a, b)| *a * *b).sum()
    }
}

impl<F: Scalar, const N: usize> NegInPlace for [F; N] {
    fn neg_in_place(&mut self) {
        for x in self.iter_mut() {
            *x = -*x;
        }
    }
}

impl<F: Scalar, const N: usize> ScaleInPlace<F> for [F; N] {
    fn scale_in_place(&mut self, scalar: F) {
        for x in self.iter_mut() {
            *x = *x * scalar;
        }
    }
}

impl<F: Scalar, const N: usize> ComponentMulAssign for [F; N] {
    fn component_mul_assign(&mut self, other: &Self) {
        for (x, y) in self.iter_mut().zip(other.iter()) {
            *x = *x * *y;
        }
    }
}

impl<F: Scalar, const N: usize> ComponentMaxAssign for [F; N] {
    fn component_max_assign(&mut self, other: &Self) {
        for (x, y) in self.iter_mut().zip(other.iter()) {
            *x = x.max(*y);
        }
    }
}

impl<F: Scalar, const N: usize> FloorZerosInPlace<F> for [F; N] {
    fn floor_zeros_in_place(&mut self, value: F) {
        for x in self.iter_mut() {
            if *x <= F::zero() {
                *x = value;
            }
        }
    }
}

impl<F: Scalar, const N: usize> ComponentDivAssign for [F; N] {
    fn component_div_assign(&mut self, other: &Self) {
        for (x, y) in self.iter_mut().zip(other.iter()) {
            *x = *x / *y;
        }
    }
}

impl<F: Scalar, const N: usize> ComponentZip<F> for [F; N] {
    fn all_zip(
        &self,
        other: &Self,
        mut predicate: impl FnMut(F, F) -> bool,
    ) -> bool {
        self.iter()
            .zip(other.iter())
            .all(|(x, y)| predicate(*x, *y))
    }
}

impl<F: Scalar, const N: usize> VectorLen for [F; N] {
    fn vec_len(&self) -> usize {
        N
    }

    fn residual_buffer() -> Option<Self> {
        Some([F::zero(); N])
    }
}

impl<F: Scalar, const N: usize> VectorIndex<F> for [F; N] {
    fn get_scalar(&self, i: usize) -> F {
        self[i]
    }
    fn set_scalar(&mut self, i: usize, value: F) {
        self[i] = value;
    }
}

impl<F: Scalar, const N: usize> SampleStandardNormal for [F; N]
where
    StandardNormal: Distribution<F>,
{
    fn sample_standard_normal<R: Rng + ?Sized>(
        _template: &Self,
        rng: &mut R,
    ) -> Self {
        std::array::from_fn(|_| StandardNormal.sample(rng))
    }
}

impl<F: Scalar + SampleUniform, const N: usize> SampleUniformBox for [F; N] {
    fn sample_uniform_box<R: Rng + ?Sized>(
        lower: &Self,
        upper: &Self,
        rng: &mut R,
    ) -> Self {
        assert_finite_box(lower, upper);
        std::array::from_fn(|i| rng.random_range(lower[i]..=upper[i]))
    }
}

impl<F: Scalar, const N: usize> ClampInPlace for [F; N] {
    fn clamp_in_place(&mut self, lower: &Self, upper: &Self) {
        for ((x, &lo), &hi) in
            self.iter_mut().zip(lower.iter()).zip(upper.iter())
        {
            // `Float` has no `clamp`; `max(lo).min(hi)` matches the
            // `f64::clamp` result on finite, ordered bounds.
            *x = (*x).max(lo).min(hi);
        }
    }
}

impl<F: Scalar, const N: usize> BoxAffineScaling<F> for [F; N] {
    fn compute_cl_scaling(
        &self,
        gradient: &Self,
        lower: &Self,
        upper: &Self,
        d_sq: &mut Self,
        c_diag: &mut Self,
    ) {
        for i in 0..N {
            let (d_sq_i, c_i) =
                cl_scaling_pair::<F>(self[i], gradient[i], lower[i], upper[i]);
            d_sq[i] = d_sq_i;
            c_diag[i] = c_i;
        }
    }

    fn max_feasible_step(&self, step: &Self, lower: &Self, upper: &Self) -> F {
        let mut tau = F::infinity();
        for i in 0..N {
            let t = max_feasible_step_component::<F>(
                self[i], step[i], lower[i], upper[i],
            );
            if t < tau {
                tau = t;
            }
        }
        tau
    }

    fn cl_kkt_inf_norm(&self, d_sq: &Self) -> F {
        self.iter()
            .zip(d_sq.iter())
            .map(|(&v, &d)| v.abs() / d)
            .fold(F::zero(), |a, b| if b > a { b } else { a })
    }

    fn weighted_norm_squared(&self, weights: &Self) -> F {
        self.iter()
            .zip(weights.iter())
            .map(|(&v, &w)| v * v * w)
            .sum()
    }

    fn project_strictly_inside(
        &mut self,
        lower: &Self,
        upper: &Self,
        rstep: F,
    ) {
        for i in 0..N {
            self[i] = project_strictly_inside_component::<F>(
                self[i], lower[i], upper[i], rstep,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn arithmetic_and_norms() {
        let mut x = [1.0_f64; 40];
        x.scaled_add(2.0, &[3.0; 40]);
        x.neg_in_place();
        x.scale_in_place(2.0);
        assert_eq!(x, [-14.0; 40]);
        assert_eq!(x.dot(&x), 7840.0);
        assert_eq!(x.norm_squared(), 7840.0);
        assert_eq!(x.norm_infinity(), 14.0);
        assert_eq!(x.vec_len(), 40);
        assert_eq!(<[f64; 40]>::residual_buffer(), Some([0.0; 40]));
        let empty: [f64; 0] = [];
        assert_eq!(empty.dot(&empty), 0.0);
        assert_eq!(empty.norm_infinity(), 0.0);
        assert_eq!(empty.norm_l2(), 0.0);

        assert_eq!([f64::MIN_POSITIVE].norm_l2(), f64::MIN_POSITIVE);
        assert_eq!([f32::MAX].norm_l2(), f32::MAX);
        assert_eq!([f32::INFINITY].norm_l2(), f32::INFINITY);
        assert!([f32::INFINITY, f32::NAN].norm_l2().is_nan());
        assert!([f64::NAN].norm_infinity().is_nan());
    }

    fn check_sampling<F: Scalar + SampleUniform>()
    where
        StandardNormal: Distribution<F>,
    {
        let lower = [-F::one(), F::zero()];
        let upper = [F::one(), F::zero()];
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut reference = rng.clone();
        let sample = <[F; 2]>::sample_uniform_box(&lower, &upper, &mut rng);
        assert_eq!(
            sample.as_slice(),
            Vec::sample_uniform_box(
                &lower.to_vec(),
                &upper.to_vec(),
                &mut reference
            )
        );
        assert_eq!(sample[1], F::zero());
        assert_eq!(
            <[F; 2]>::sample_standard_normal(&sample, &mut rng).as_slice(),
            Vec::sample_standard_normal(&sample.to_vec(), &mut reference)
        );
    }

    #[test]
    fn seeded_samples_match_vec() {
        check_sampling::<f32>();
        check_sampling::<f64>();
    }

    #[test]
    #[should_panic(expected = "sample_uniform_box requires finite bounds")]
    fn rejects_non_finite_bounds() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        <[f64; 1]>::sample_uniform_box(&[0.0], &[f64::INFINITY], &mut rng);
    }
}
