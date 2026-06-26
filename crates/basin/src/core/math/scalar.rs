use super::{Dot, NegInPlace, NormInfinity, NormSquared, ScaledAdd};

// These scalar-element impls treat `f64` itself as the parameter or vector
// type, used by 1-D solvers that work directly on a scalar `x: f64`.
// They are not blanketed over `F: Scalar` because that would conflict
// with the container impls (`Vec<F>`, `DVector<F>`, `Col<F>`, …): Rust
// can't rule out a future upstream `impl Scalar for Vec<...>` etc., so
// `impl<F: Scalar> NormSquared<F> for F` would collide with
// `impl<F: Scalar> NormSquared<F> for Vec<F>` under the coherence rules.
// Add `impl ... for f32` here too once a consumer needs it.

impl ScaledAdd for f64 {
    fn scaled_add(&mut self, scalar: f64, other: &Self) {
        *self += scalar * other;
    }
}

impl NormSquared for f64 {
    fn norm_squared(&self) -> f64 {
        self * self
    }
}

impl NormInfinity for f64 {
    fn norm_infinity(&self) -> f64 {
        self.abs()
    }
}

impl Dot for f64 {
    fn dot(&self, other: &Self) -> f64 {
        self * other
    }
}

impl NegInPlace for f64 {
    fn neg_in_place(&mut self) {
        *self = -*self;
    }
}
