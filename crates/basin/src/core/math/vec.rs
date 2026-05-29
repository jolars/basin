use rand::{Rng, RngExt};
use rand_distr::{Distribution, StandardNormal};

use super::cl_scaling::{
    cl_scaling_pair, max_feasible_step_component, project_strictly_inside_component,
    BoxAffineScaling,
};
use super::sample::{SampleStandardNormal, SampleUniformBox};
use super::{
    ClampInPlace, ComponentDivAssign, ComponentMaxAssign, ComponentMulAssign, Dot,
    FloorZerosInPlace, NegInPlace, NormInfinity, NormSquared, ScaleInPlace, ScaledAdd, VectorIndex,
    VectorLen,
};

impl ScaledAdd<f64> for Vec<f64> {
    fn scaled_add(&mut self, scalar: f64, other: &Self) {
        assert_eq!(self.len(), other.len(), "scaled_add: length mismatch");
        for (x, y) in self.iter_mut().zip(other.iter()) {
            *x += scalar * y;
        }
    }
}

impl NormSquared for Vec<f64> {
    fn norm_squared(&self) -> f64 {
        self.iter().map(|x| x * x).sum()
    }
}

impl NormInfinity for Vec<f64> {
    fn norm_infinity(&self) -> f64 {
        self.iter().map(|x| x.abs()).fold(0.0, f64::max)
    }
}

impl Dot for Vec<f64> {
    fn dot(&self, other: &Self) -> f64 {
        assert_eq!(self.len(), other.len(), "dot: length mismatch");
        self.iter().zip(other.iter()).map(|(a, b)| a * b).sum()
    }
}

impl NegInPlace for Vec<f64> {
    fn neg_in_place(&mut self) {
        for x in self.iter_mut() {
            *x = -*x;
        }
    }
}

impl ScaleInPlace for Vec<f64> {
    fn scale_in_place(&mut self, scalar: f64) {
        for x in self.iter_mut() {
            *x *= scalar;
        }
    }
}

impl ComponentMulAssign for Vec<f64> {
    fn component_mul_assign(&mut self, other: &Self) {
        assert_eq!(
            self.len(),
            other.len(),
            "component_mul_assign: length mismatch"
        );
        for (x, y) in self.iter_mut().zip(other.iter()) {
            *x *= *y;
        }
    }
}

impl ComponentMaxAssign for Vec<f64> {
    fn component_max_assign(&mut self, other: &Self) {
        assert_eq!(
            self.len(),
            other.len(),
            "component_max_assign: length mismatch"
        );
        for (x, y) in self.iter_mut().zip(other.iter()) {
            *x = x.max(*y);
        }
    }
}

impl FloorZerosInPlace for Vec<f64> {
    fn floor_zeros_in_place(&mut self, value: f64) {
        for x in self.iter_mut() {
            if *x <= 0.0 {
                *x = value;
            }
        }
    }
}

impl ComponentDivAssign for Vec<f64> {
    fn component_div_assign(&mut self, other: &Self) {
        assert_eq!(
            self.len(),
            other.len(),
            "component_div_assign: length mismatch"
        );
        for (x, y) in self.iter_mut().zip(other.iter()) {
            *x /= *y;
        }
    }
}

impl VectorLen for Vec<f64> {
    fn vec_len(&self) -> usize {
        self.len()
    }
}

impl VectorIndex for Vec<f64> {
    fn get_scalar(&self, i: usize) -> f64 {
        self[i]
    }
    fn set_scalar(&mut self, i: usize, value: f64) {
        self[i] = value;
    }
}

impl SampleStandardNormal for Vec<f64> {
    fn sample_standard_normal<R: Rng + ?Sized>(template: &Self, rng: &mut R) -> Self {
        let n = template.len();
        let mut out = Self::with_capacity(n);
        for _ in 0..n {
            out.push(StandardNormal.sample(rng));
        }
        out
    }
}

impl SampleUniformBox for Vec<f64> {
    fn sample_uniform_box<R: Rng + ?Sized>(lower: &Self, upper: &Self, rng: &mut R) -> Self {
        assert_eq!(
            lower.len(),
            upper.len(),
            "sample_uniform_box: bounds length mismatch"
        );
        let n = lower.len();
        let mut out = Self::with_capacity(n);
        for i in 0..n {
            out.push(rng.random_range(lower[i]..=upper[i]));
        }
        out
    }
}

impl ClampInPlace for Vec<f64> {
    fn clamp_in_place(&mut self, lower: &Self, upper: &Self) {
        assert_eq!(
            self.len(),
            lower.len(),
            "clamp_in_place: lower length mismatch"
        );
        assert_eq!(
            self.len(),
            upper.len(),
            "clamp_in_place: upper length mismatch"
        );
        for ((x, &lo), &hi) in self.iter_mut().zip(lower.iter()).zip(upper.iter()) {
            *x = x.clamp(lo, hi);
        }
    }
}

impl BoxAffineScaling for Vec<f64> {
    fn compute_cl_scaling(
        &self,
        gradient: &Self,
        lower: &Self,
        upper: &Self,
        d_sq: &mut Self,
        c_diag: &mut Self,
    ) {
        let n = self.len();
        assert_eq!(
            n,
            gradient.len(),
            "compute_cl_scaling: gradient length mismatch"
        );
        assert_eq!(n, lower.len(), "compute_cl_scaling: lower length mismatch");
        assert_eq!(n, upper.len(), "compute_cl_scaling: upper length mismatch");
        assert_eq!(n, d_sq.len(), "compute_cl_scaling: d_sq length mismatch");
        assert_eq!(
            n,
            c_diag.len(),
            "compute_cl_scaling: c_diag length mismatch"
        );
        for i in 0..n {
            let (d_sq_i, c_i) = cl_scaling_pair(self[i], gradient[i], lower[i], upper[i]);
            d_sq[i] = d_sq_i;
            c_diag[i] = c_i;
        }
    }

    fn max_feasible_step(&self, step: &Self, lower: &Self, upper: &Self) -> f64 {
        let n = self.len();
        assert_eq!(n, step.len(), "max_feasible_step: step length mismatch");
        assert_eq!(n, lower.len(), "max_feasible_step: lower length mismatch");
        assert_eq!(n, upper.len(), "max_feasible_step: upper length mismatch");
        let mut tau = f64::INFINITY;
        for i in 0..n {
            let t = max_feasible_step_component(self[i], step[i], lower[i], upper[i]);
            if t < tau {
                tau = t;
            }
        }
        tau
    }

    fn cl_kkt_inf_norm(&self, d_sq: &Self) -> f64 {
        assert_eq!(self.len(), d_sq.len(), "cl_kkt_inf_norm: length mismatch");
        self.iter()
            .zip(d_sq.iter())
            .map(|(&v, &d)| v.abs() / d)
            .fold(0.0, f64::max)
    }

    fn weighted_norm_squared(&self, weights: &Self) -> f64 {
        assert_eq!(
            self.len(),
            weights.len(),
            "weighted_norm_squared: length mismatch"
        );
        self.iter()
            .zip(weights.iter())
            .map(|(&v, &w)| v * v * w)
            .sum()
    }

    fn project_strictly_inside(&mut self, lower: &Self, upper: &Self, rstep: f64) {
        let n = self.len();
        assert_eq!(
            n,
            lower.len(),
            "project_strictly_inside: lower length mismatch"
        );
        assert_eq!(
            n,
            upper.len(),
            "project_strictly_inside: upper length mismatch"
        );
        for i in 0..n {
            self[i] = project_strictly_inside_component(self[i], lower[i], upper[i], rstep);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn sample_uniform_box_stays_within_bounds() {
        let lower = vec![-1.0, 0.5, -10.0];
        let upper = vec![1.0, 0.5, 10.0]; // Middle component pinned (lo == hi).
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        for _ in 0..200 {
            let x = Vec::<f64>::sample_uniform_box(&lower, &upper, &mut rng);
            assert_eq!(x.len(), 3);
            for i in 0..3 {
                assert!(
                    x[i] >= lower[i] && x[i] <= upper[i],
                    "x[{i}] = {} not in [{}, {}]",
                    x[i],
                    lower[i],
                    upper[i]
                );
            }
            // Pinned coordinate is exactly its single bound.
            assert_eq!(x[1], 0.5);
        }
    }

    #[test]
    fn sample_uniform_box_is_seed_reproducible() {
        let lower = vec![-1.0, -1.0];
        let upper = vec![1.0, 1.0];
        let mut rng_a = ChaCha8Rng::seed_from_u64(7);
        let mut rng_b = ChaCha8Rng::seed_from_u64(7);
        let a = Vec::<f64>::sample_uniform_box(&lower, &upper, &mut rng_a);
        let b = Vec::<f64>::sample_uniform_box(&lower, &upper, &mut rng_b);
        assert_eq!(a, b);
    }

    #[test]
    fn clamp_inside_box_is_identity() {
        let mut x = vec![0.5, -0.5, 1.5];
        x.clamp_in_place(&vec![-1.0, -1.0, 0.0], &vec![1.0, 1.0, 2.0]);
        assert_eq!(x, vec![0.5, -0.5, 1.5]);
    }

    #[test]
    fn clamp_partially_outside_pins_only_offending_components() {
        let mut x = vec![-2.0, 0.5, 3.0];
        x.clamp_in_place(&vec![-1.0, -1.0, -1.0], &vec![1.0, 1.0, 1.0]);
        assert_eq!(x, vec![-1.0, 0.5, 1.0]);
    }

    #[test]
    fn clamp_entirely_outside_pins_to_nearest_face() {
        let mut x = vec![-10.0, 10.0];
        x.clamp_in_place(&vec![-1.0, -1.0], &vec![1.0, 1.0]);
        assert_eq!(x, vec![-1.0, 1.0]);
    }

    #[test]
    fn clamp_with_equal_bounds_pins_to_value() {
        let mut x = vec![5.0, -5.0];
        x.clamp_in_place(&vec![1.0, 2.0], &vec![1.0, 2.0]);
        assert_eq!(x, vec![1.0, 2.0]);
    }

    #[test]
    fn cl_scaling_finite_bounds_negative_gradient_uses_upper() {
        // x = 0.5, g = -1, bounds [-2, 2].
        // Case (i): v = x - u = -1.5, |v| = 1.5, d_sq = 1/1.5, c = 1/1.5.
        let x = vec![0.5];
        let g = vec![-1.0];
        let lower = vec![-2.0];
        let upper = vec![2.0];
        let mut d_sq = vec![0.0];
        let mut c = vec![0.0];
        x.compute_cl_scaling(&g, &lower, &upper, &mut d_sq, &mut c);
        assert!((d_sq[0] - (1.0 / 1.5)).abs() < 1e-12);
        assert!((c[0] - (1.0 / 1.5)).abs() < 1e-12);
    }

    #[test]
    fn cl_scaling_finite_bounds_positive_gradient_uses_lower() {
        // x = 0.5, g = 2, bounds [-2, 2].
        // Case (ii): v = x - l = 2.5, |v| = 2.5, d_sq = 1/2.5, c = 2/2.5.
        let x = vec![0.5];
        let g = vec![2.0];
        let lower = vec![-2.0];
        let upper = vec![2.0];
        let mut d_sq = vec![0.0];
        let mut c = vec![0.0];
        x.compute_cl_scaling(&g, &lower, &upper, &mut d_sq, &mut c);
        assert!((d_sq[0] - (1.0 / 2.5)).abs() < 1e-12);
        assert!((c[0] - (2.0 / 2.5)).abs() < 1e-12);
    }

    #[test]
    fn cl_scaling_infinite_bounds_yields_unit_d_and_zero_c() {
        // Both bounds infinite: d_sq = 1, c = 0 (cases iii / iv).
        // Effectively reduces to LM (D = I, C = 0).
        let x = vec![0.0, 0.0];
        let g = vec![-1.0, 1.0];
        let lower = vec![f64::NEG_INFINITY, f64::NEG_INFINITY];
        let upper = vec![f64::INFINITY, f64::INFINITY];
        let mut d_sq = vec![0.0; 2];
        let mut c = vec![0.0; 2];
        x.compute_cl_scaling(&g, &lower, &upper, &mut d_sq, &mut c);
        assert_eq!(d_sq, vec![1.0, 1.0]);
        assert_eq!(c, vec![0.0, 0.0]);
    }

    #[test]
    fn max_feasible_step_finds_first_binding_component() {
        // x = (0, 0), step = (1, 2), bounds [-1, 1]^2.
        // Component 0: τ_0 = (1 - 0) / 1 = 1.
        // Component 1: τ_1 = (1 - 0) / 2 = 0.5.
        // τ_max = 0.5.
        let x = vec![0.0, 0.0];
        let step = vec![1.0, 2.0];
        let lower = vec![-1.0, -1.0];
        let upper = vec![1.0, 1.0];
        let tau = x.max_feasible_step(&step, &lower, &upper);
        assert!((tau - 0.5).abs() < 1e-12);
    }

    #[test]
    fn max_feasible_step_with_no_binding_bound_is_infinite() {
        // step is zero: no component reaches a bound.
        let x = vec![0.0, 0.0];
        let step = vec![0.0, 0.0];
        let lower = vec![-1.0, -1.0];
        let upper = vec![1.0, 1.0];
        assert_eq!(x.max_feasible_step(&step, &lower, &upper), f64::INFINITY);
    }

    #[test]
    fn cl_kkt_inf_norm_matches_max_abs_g_over_d_sq() {
        // g = (3, -8), d_sq = (1, 4) → |g| / d_sq = (3, 2) → ‖·‖_∞ = 3.
        let g = vec![3.0, -8.0];
        let d_sq = vec![1.0, 4.0];
        assert!((g.cl_kkt_inf_norm(&d_sq) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn cl_kkt_inf_norm_vanishes_at_face_active_point() {
        // Face-active emulation: g_i bounded but d_sq_i = 1/|v_i| huge
        // (because v_i ≈ 0 near the boundary). The KKT metric should
        // go to zero, *not* blow up.
        let g = vec![-16.0, -20.0]; // gradients at corner (1, 1)
        let d_sq = vec![1e10, 1e10]; // d_sq → ∞ as iterate → boundary
        assert!(g.cl_kkt_inf_norm(&d_sq) < 1e-8);
    }
}
