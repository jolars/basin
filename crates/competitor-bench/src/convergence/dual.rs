//! Forward derivatives keep the NIST formula and its Jacobian in one place.

use basin::Scalar;
use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait Real:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    fn number(x: f64) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn atan(self) -> Self;
    fn pow(self, exponent: Self) -> Self {
        (exponent * self.ln()).exp()
    }
    fn square(self) -> Self {
        self * self
    }
}

macro_rules! real {
    ($f:ty) => {
        impl Real for $f {
            fn number(x: f64) -> Self {
                x as Self
            }
            fn exp(self) -> Self {
                self.exp()
            }
            fn ln(self) -> Self {
                self.ln()
            }
            fn sin(self) -> Self {
                self.sin()
            }
            fn cos(self) -> Self {
                self.cos()
            }
            fn atan(self) -> Self {
                self.atan()
            }
        }
    };
}
real!(f32);
real!(f64);

pub const MAX_PARAMETERS: usize = 16;

#[derive(Clone, Copy, Debug)]
pub struct Dual<F> {
    pub value: F,
    pub derivative: [F; MAX_PARAMETERS],
}

impl<F: Scalar> Dual<F> {
    pub fn variable(value: F, coordinate: usize) -> Self {
        let mut derivative = [F::zero(); MAX_PARAMETERS];
        derivative[coordinate] = F::one();
        Self { value, derivative }
    }
    fn chain(self, value: F, slope: F) -> Self {
        Self {
            value,
            derivative: self.derivative.map(|d| slope * d),
        }
    }
}

impl<F: Scalar> Real for Dual<F> {
    fn number(x: f64) -> Self {
        Self {
            value: F::from_f64(x).unwrap(),
            derivative: [F::zero(); MAX_PARAMETERS],
        }
    }
    fn exp(self) -> Self {
        let value = self.value.exp();
        self.chain(value, value)
    }
    fn ln(self) -> Self {
        self.chain(self.value.ln(), self.value.recip())
    }
    fn sin(self) -> Self {
        self.chain(self.value.sin(), self.value.cos())
    }
    fn cos(self) -> Self {
        self.chain(self.value.cos(), -self.value.sin())
    }
    fn atan(self) -> Self {
        self.chain(
            self.value.atan(),
            (F::one() + self.value * self.value).recip(),
        )
    }
}

impl<F: Scalar> Add for Dual<F> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
            derivative: std::array::from_fn(|i| {
                self.derivative[i] + rhs.derivative[i]
            }),
        }
    }
}
impl<F: Scalar> Sub for Dual<F> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}
impl<F: Scalar> Neg for Dual<F> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            derivative: self.derivative.map(|d| -d),
        }
    }
}
impl<F: Scalar> Mul for Dual<F> {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            value: self.value * rhs.value,
            derivative: std::array::from_fn(|i| {
                self.derivative[i] * rhs.value + self.value * rhs.derivative[i]
            }),
        }
    }
}
impl<F: Scalar> Div for Dual<F> {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        let value = self.value / rhs.value;
        Self {
            value,
            derivative: std::array::from_fn(|i| {
                (self.derivative[i] - value * rhs.derivative[i]) / rhs.value
            }),
        }
    }
}
