use super::{Dot, NegInPlace, NormInfinity, NormSquared, ScaledAdd};

impl ScaledAdd<Self> for f64 {
    fn scaled_add(&mut self, scalar: Self, other: &Self) {
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
