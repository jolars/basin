//! Shared inputs for timing and independently instrumented work measurements.

use std::{convert::Infallible, hint::black_box, marker::PhantomData};

use basin::{
    BoundedFiniteDiff, CostFunction, DenseMatrixFromFn, FiniteDiff, Gradient,
    Hessian, HessianProduct, HessianProductScratch, Jacobian, MatrixIndex,
    MatrixShape, Method, Residual, VectorIndex, VectorLen,
    central_difference_hessian_product, forward_difference_hessian_product,
};

pub trait BenchVector:
    Clone + VectorLen + VectorIndex + DenseMatrixFromFn + Send + Sync
{
    fn from_fn(n: usize, f: impl FnMut(usize) -> f64) -> Self;
    fn fill(n: usize, f: impl FnMut(usize) -> f64, out: &mut Self);
}

impl BenchVector for Vec<f64> {
    fn from_fn(n: usize, f: impl FnMut(usize) -> f64) -> Self {
        (0..n).map(f).collect()
    }
    fn fill(n: usize, mut f: impl FnMut(usize) -> f64, out: &mut Self) {
        out.resize(n, 0.0);
        for (i, value) in out.iter_mut().enumerate() {
            *value = f(i);
        }
    }
}

#[cfg(not(feature = "basin-latest"))]
use nalgebra as na;
#[cfg(feature = "basin-latest")]
use nalgebra_latest as na;

impl BenchVector for na::DVector<f64> {
    fn from_fn(n: usize, mut f: impl FnMut(usize) -> f64) -> Self {
        Self::from_fn(n, |i, _| f(i))
    }
    fn fill(n: usize, mut f: impl FnMut(usize) -> f64, out: &mut Self) {
        if out.len() != n {
            *out = <Self as BenchVector>::from_fn(n, f);
        } else {
            for (i, value) in out.iter_mut().enumerate() {
                *value = f(i);
            }
        }
    }
}

impl BenchVector for ndarray::Array1<f64> {
    fn from_fn(n: usize, f: impl FnMut(usize) -> f64) -> Self {
        Self::from_shape_fn(n, f)
    }
    fn fill(n: usize, mut f: impl FnMut(usize) -> f64, out: &mut Self) {
        if out.len() != n {
            *out = <Self as BenchVector>::from_fn(n, f);
        } else {
            for (i, value) in out.iter_mut().enumerate() {
                *value = f(i);
            }
        }
    }
}

impl BenchVector for faer::Col<f64> {
    fn from_fn(n: usize, f: impl FnMut(usize) -> f64) -> Self {
        Self::from_fn(n, f)
    }
    fn fill(n: usize, mut f: impl FnMut(usize) -> f64, out: &mut Self) {
        out.resize_with(n, |_| 0.0);
        for i in 0..n {
            out[i] = f(i);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Work {
    Cheap,
    Expensive,
}

impl Work {
    pub fn name(self) -> &'static str {
        match self {
            Self::Cheap => "cheap",
            Self::Expensive => "expensive",
        }
    }

    fn terms(self) -> usize {
        match self {
            Self::Cheap => 0,
            Self::Expensive => 16,
        }
    }

    fn value(self, x: f64) -> f64 {
        let mut value = 0.5 * x * x;
        for k in 1..=self.terms() {
            value += 1.0 - (x / k as f64).cos();
        }
        value
    }

    fn first(self, x: f64) -> f64 {
        let mut value = x;
        for k in 1..=self.terms() {
            value += (x / k as f64).sin() / k as f64;
        }
        value
    }

    fn second(self, x: f64) -> f64 {
        let mut value = 1.0;
        for k in 1..=self.terms() {
            value += (x / k as f64).cos() / (k * k) as f64;
        }
        value
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Kernel {
    Gradient,
    CostGradient,
    BoundedGradient,
    BoundedCostGradient,
    Jacobian,
    ResidualJacobian,
    BoundedJacobian,
    Hessian,
    CostGradientHessian,
    HessianProduct,
    AnalyticHessianProduct,
}

impl Kernel {
    pub fn name(self) -> &'static str {
        match self {
            Self::Gradient => "gradient",
            Self::CostGradient => "cost_gradient",
            Self::BoundedGradient => "bounded_gradient",
            Self::BoundedCostGradient => "bounded_cost_gradient",
            Self::Jacobian => "jacobian",
            Self::ResidualJacobian => "residual_jacobian",
            Self::BoundedJacobian => "bounded_jacobian",
            Self::Hessian => "hessian",
            Self::CostGradientHessian => "cost_gradient_hessian",
            Self::HessianProduct => "hessian_product",
            Self::AnalyticHessianProduct => "analytic_hessian_product",
        }
    }

    fn is_jacobian(self) -> bool {
        matches!(
            self,
            Self::Jacobian | Self::ResidualJacobian | Self::BoundedJacobian
        )
    }

    fn is_hessian(self) -> bool {
        matches!(self, Self::Hessian | Self::CostGradientHessian)
    }

    fn is_bounded(self) -> bool {
        matches!(
            self,
            Self::BoundedGradient
                | Self::BoundedCostGradient
                | Self::BoundedJacobian
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Call {
    Owned,
    IntoReused,
    ScratchReused,
}

impl Call {
    pub const ALL: [Self; 2] = [Self::Owned, Self::IntoReused];

    pub fn name(self) -> &'static str {
        match self {
            Self::Owned => "owned",
            Self::IntoReused => "into_reused",
            Self::ScratchReused => "scratch_reused",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub kernel: Kernel,
    pub work: Work,
    pub method: Method,
    pub n: usize,
    pub m: usize,
}

impl Case {
    pub fn calls(self) -> &'static [Call] {
        if matches!(self.kernel, Kernel::AnalyticHessianProduct) {
            &[Call::Owned, Call::ScratchReused]
        } else if matches!(self.kernel, Kernel::HessianProduct) {
            &[Call::Owned, Call::IntoReused, Call::ScratchReused]
        } else {
            &Call::ALL
        }
    }

    pub fn name(self) -> String {
        let method = match self.method {
            Method::Forward => "forward",
            Method::Central => "central",
            _ => unreachable!(
                "the harness selects only forward and central stencils"
            ),
        };
        format!(
            "{}/{}/{method}/n{}_m{}",
            self.kernel.name(),
            self.work.name(),
            self.n,
            self.m,
        )
    }

    fn coordinate(self, j: usize) -> f64 {
        // Include active lower/upper bounds, fixed coordinates, and interior probes.
        if self.kernel.is_bounded() {
            match j % 4 {
                0 | 2 => -1.0,
                1 => 1.0,
                _ => 0.25,
            }
        } else {
            0.25 + (j % 7) as f64 / 16.0
        }
    }

    fn fixed(self, j: usize) -> bool {
        self.kernel.is_bounded() && j % 4 == 2
    }
}

pub fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for kernel in [
        Kernel::Gradient,
        Kernel::CostGradient,
        Kernel::BoundedGradient,
        Kernel::BoundedCostGradient,
        Kernel::Jacobian,
        Kernel::ResidualJacobian,
        Kernel::BoundedJacobian,
        Kernel::Hessian,
        Kernel::CostGradientHessian,
        Kernel::HessianProduct,
        Kernel::AnalyticHessianProduct,
    ] {
        for work in [Work::Cheap, Work::Expensive] {
            for method in [Method::Forward, Method::Central] {
                let shapes: &[(usize, usize)] = if kernel.is_jacobian() {
                    &[(4, 16), (16, 4), (64, 256), (256, 64)]
                } else if kernel.is_hessian() {
                    &[(4, 0), (64, 0), (256, 0)]
                } else {
                    &[(4, 0), (256, 0)]
                };
                for &(n, m) in shapes {
                    cases.push(Case {
                        kernel,
                        work,
                        method,
                        n,
                        m,
                    });
                }
            }
        }
    }
    cases
}

#[derive(Clone)]
pub struct Function<V> {
    pub case: Case,
    vector: PhantomData<V>,
}

impl<V> Function<V> {
    pub fn new(case: Case) -> Self {
        Self {
            case,
            vector: PhantomData,
        }
    }
}

impl<V: BenchVector> CostFunction for Function<V> {
    type Param = V;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &V) -> Result<f64, Infallible> {
        Ok((0..x.vec_len())
            .map(|j| weight(j) * self.case.work.value(x.get_scalar(j)))
            .sum())
    }
}

impl<V: BenchVector> Gradient for Function<V> {
    type Gradient = V;

    fn gradient(&self, x: &V) -> Result<V, Infallible> {
        let mut out = V::from_fn(x.vec_len(), |_| 0.0);
        self.gradient_into(x, &mut out)?;
        Ok(out)
    }

    fn gradient_into(&self, x: &V, out: &mut V) -> Result<(), Infallible> {
        V::fill(
            x.vec_len(),
            |j| weight(j) * self.case.work.first(x.get_scalar(j)),
            out,
        );
        Ok(())
    }
}

impl<V: BenchVector> Residual for Function<V> {
    type Param = V;
    type Output = V;
    type Error = Infallible;

    fn residual(&self, x: &V) -> Result<V, Infallible> {
        let mut out = V::from_fn(self.case.m, |_| 0.0);
        self.residual_into(x, &mut out)?;
        Ok(out)
    }

    fn residual_into(&self, x: &V, out: &mut V) -> Result<(), Infallible> {
        let base: f64 = (0..x.vec_len())
            .map(|j| x.get_scalar(j) / (j + 1) as f64)
            .sum();
        V::fill(
            self.case.m,
            |i| {
                self.case.work.value(x.get_scalar(i % self.case.n))
                    + row_weight(i, self.case.m) * base
            },
            out,
        );
        Ok(())
    }
}

fn weight(j: usize) -> f64 {
    1.0 + (j % 5) as f64 / 8.0
}

fn row_weight(i: usize, m: usize) -> f64 {
    0.01 * (i + 1) as f64 / m as f64
}

pub struct Fixture<V: BenchVector, P> {
    case: Case,
    problem: P,
    fd: FiniteDiff<P>,
    bounded: BoundedFiniteDiff<P>,
    x: V,
    direction: V,
    gradient: V,
    product_scratch: HessianProductScratch<V>,
    residual: V,
    matrix: V::Matrix,
    cost: f64,
}

impl<V, P> Fixture<V, P>
where
    V: BenchVector,
    V::Matrix: MatrixIndex,
    P: Clone
        + Send
        + Sync
        + CostFunction<Param = V, Output = f64, Error = Infallible>
        + Gradient<Gradient = V>
        + Residual<Param = V, Output = V, Error = Infallible>,
{
    pub fn new(case: Case, problem: P) -> Self {
        let mut fd = FiniteDiff::new(problem.clone())
            .gradient_method(case.method)
            .jacobian_method(case.method)
            .hessian_method(case.method);
        // Differencing FD gradients needs a wider step to avoid stacked roundoff.
        if matches!(case.kernel, Kernel::HessianProduct) {
            fd = fd.with_step(1e-3);
        }
        let bounded = BoundedFiniteDiff::new(
            problem.clone(),
            vec![-1.0; case.n],
            (0..case.n)
                .map(|j| if case.fixed(j) { -1.0 } else { 1.0 })
                .collect::<Vec<_>>(),
        )
        .gradient_method(case.method)
        .jacobian_method(case.method);
        let (rows, cols) = if case.kernel.is_jacobian() {
            (case.m, case.n)
        } else if case.kernel.is_hessian() {
            (case.n, case.n)
        } else {
            (0, 0)
        };
        Self {
            case,
            problem,
            fd,
            bounded,
            x: V::from_fn(case.n, |j| case.coordinate(j)),
            direction: V::from_fn(case.n, |j| 0.25 + (j % 3) as f64 / 8.0),
            gradient: V::from_fn(case.n, |_| 0.0),
            product_scratch: HessianProductScratch::new(),
            residual: V::from_fn(case.m, |_| 0.0),
            matrix: V::dense_from_fn(rows, cols, |_, _| 0.0),
            cost: 0.0,
        }
    }

    pub fn run(&mut self, call: Call) {
        let x = black_box(&self.x);
        match (self.case.kernel, call) {
            (Kernel::Gradient, Call::Owned) => {
                self.gradient = self.fd.gradient(x).unwrap()
            }
            (Kernel::Gradient, Call::IntoReused) => {
                self.fd.gradient_into(x, &mut self.gradient).unwrap()
            }
            (Kernel::CostGradient, Call::Owned) => {
                (self.cost, self.gradient) =
                    self.fd.cost_and_gradient(x).unwrap()
            }
            (Kernel::CostGradient, Call::IntoReused) => {
                self.cost = self
                    .fd
                    .cost_and_gradient_into(x, &mut self.gradient)
                    .unwrap()
            }
            (Kernel::BoundedGradient, Call::Owned) => {
                self.gradient = self.bounded.gradient(x).unwrap()
            }
            (Kernel::BoundedGradient, Call::IntoReused) => {
                self.bounded.gradient_into(x, &mut self.gradient).unwrap()
            }
            (Kernel::BoundedCostGradient, Call::Owned) => {
                (self.cost, self.gradient) =
                    self.bounded.cost_and_gradient(x).unwrap()
            }
            (Kernel::BoundedCostGradient, Call::IntoReused) => {
                self.cost = self
                    .bounded
                    .cost_and_gradient_into(x, &mut self.gradient)
                    .unwrap()
            }
            (Kernel::Jacobian, Call::Owned) => {
                self.matrix = self.fd.jacobian(x).unwrap()
            }
            (Kernel::Jacobian, Call::IntoReused) => {
                self.fd.jacobian_into(x, &mut self.matrix).unwrap()
            }
            (Kernel::ResidualJacobian, Call::Owned) => {
                (self.residual, self.matrix) =
                    self.fd.residual_and_jacobian(x).unwrap()
            }
            (Kernel::ResidualJacobian, Call::IntoReused) => self
                .fd
                .residual_and_jacobian_into(
                    x,
                    &mut self.residual,
                    &mut self.matrix,
                )
                .unwrap(),
            (Kernel::BoundedJacobian, Call::Owned) => {
                self.matrix = self.bounded.jacobian(x).unwrap()
            }
            (Kernel::BoundedJacobian, Call::IntoReused) => {
                self.bounded.jacobian_into(x, &mut self.matrix).unwrap()
            }
            (Kernel::Hessian, Call::Owned) => {
                self.matrix = self.fd.hessian(x).unwrap()
            }
            (Kernel::Hessian, Call::IntoReused) => {
                self.fd.hessian_into(x, &mut self.matrix).unwrap()
            }
            (Kernel::CostGradientHessian, Call::Owned) => {
                (self.cost, self.gradient, self.matrix) =
                    self.fd.cost_and_gradient_and_hessian(x).unwrap()
            }
            (Kernel::CostGradientHessian, Call::IntoReused) => {
                self.cost = self
                    .fd
                    .cost_and_gradient_and_hessian_into(
                        x,
                        &mut self.gradient,
                        &mut self.matrix,
                    )
                    .unwrap()
            }
            (Kernel::HessianProduct, Call::Owned) => {
                self.gradient = self
                    .fd
                    .hessian_product(x, black_box(&self.direction))
                    .unwrap()
            }
            (Kernel::HessianProduct, Call::IntoReused) => self
                .fd
                .hessian_product_into(
                    x,
                    black_box(&self.direction),
                    &mut self.gradient,
                )
                .unwrap(),
            (Kernel::AnalyticHessianProduct, Call::Owned) => {
                self.gradient = match self.case.method {
                    Method::Forward => forward_difference_hessian_product(
                        &self.problem,
                        x,
                        black_box(&self.direction),
                        f64::EPSILON,
                        None,
                    ),
                    Method::Central => central_difference_hessian_product(
                        &self.problem,
                        x,
                        black_box(&self.direction),
                        f64::EPSILON,
                        None,
                    ),
                    _ => unreachable!(),
                }
                .unwrap();
            }
            (Kernel::AnalyticHessianProduct, Call::IntoReused) => {
                unreachable!()
            }
            (Kernel::HessianProduct, Call::ScratchReused) => self
                .fd
                .hessian_product_with_scratch_into(
                    x,
                    black_box(&self.direction),
                    &mut self.gradient,
                    &mut self.product_scratch,
                )
                .unwrap(),
            (Kernel::AnalyticHessianProduct, Call::ScratchReused) => {
                match self.case.method {
                    Method::Forward => self.product_scratch.forward_into(
                        &self.problem,
                        x,
                        black_box(&self.direction),
                        f64::EPSILON,
                        None,
                        &mut self.gradient,
                    ),
                    Method::Central => self.product_scratch.central_into(
                        &self.problem,
                        x,
                        black_box(&self.direction),
                        f64::EPSILON,
                        None,
                        &mut self.gradient,
                    ),
                    _ => unreachable!(),
                }
                .unwrap();
            }
            (_, Call::ScratchReused) => unreachable!(),
        }
        black_box((&self.gradient, &self.residual, &self.matrix, self.cost));
    }

    pub fn verify(&self) {
        let case = self.case;
        let close = |actual: f64, expected: f64, tolerance: f64| {
            assert!(
                actual.is_finite()
                    && (actual - expected).abs()
                        <= tolerance * (1.0 + expected.abs()),
                "{}: {actual} != {expected} (tolerance {tolerance})",
                case.name()
            );
        };
        if case.kernel.is_jacobian() {
            assert_eq!(
                (self.matrix.matrix_rows(), self.matrix.matrix_cols()),
                (case.m, case.n)
            );
            for i in 0..case.m {
                for j in 0..case.n {
                    let expected = if case.fixed(j) {
                        0.0
                    } else {
                        row_weight(i, case.m) / (j + 1) as f64
                            + if i % case.n == j {
                                case.work.first(case.coordinate(j))
                            } else {
                                0.0
                            }
                    };
                    close(self.matrix.matrix_entry(i, j), expected, 2e-5);
                }
            }
            if matches!(case.kernel, Kernel::ResidualJacobian) {
                let expected =
                    Function::<V>::new(case).residual(&self.x).unwrap();
                assert_eq!(self.residual.vec_len(), case.m);
                for i in 0..case.m {
                    close(
                        self.residual.get_scalar(i),
                        expected.get_scalar(i),
                        1e-14,
                    );
                }
            }
        } else if case.kernel.is_hessian() {
            assert_eq!(
                (self.matrix.matrix_rows(), self.matrix.matrix_cols()),
                (case.n, case.n)
            );
            for i in 0..case.n {
                for j in 0..case.n {
                    let actual = self.matrix.matrix_entry(i, j);
                    let expected = if i == j {
                        weight(i) * case.work.second(case.coordinate(i))
                    } else {
                        0.0
                    };
                    close(actual, expected, 5e-4);
                    assert_eq!(
                        actual.to_bits(),
                        self.matrix.matrix_entry(j, i).to_bits()
                    );
                }
            }
        }
        if !case.kernel.is_jacobian() && !matches!(case.kernel, Kernel::Hessian)
        {
            assert_eq!(self.gradient.vec_len(), case.n);
            for j in 0..case.n {
                let (expected, tolerance) = if case.fixed(j) {
                    (0.0, 0.0)
                } else if matches!(
                    case.kernel,
                    Kernel::HessianProduct | Kernel::AnalyticHessianProduct
                ) {
                    (
                        weight(j)
                            * case.work.second(case.coordinate(j))
                            * self.direction.get_scalar(j),
                        if matches!(case.kernel, Kernel::HessianProduct) {
                            1e-2
                        } else {
                            2e-5
                        },
                    )
                } else {
                    (weight(j) * case.work.first(case.coordinate(j)), 2e-5)
                };
                close(self.gradient.get_scalar(j), expected, tolerance);
            }
        }
        if matches!(
            case.kernel,
            Kernel::CostGradient
                | Kernel::BoundedCostGradient
                | Kernel::CostGradientHessian
        ) {
            close(
                self.cost,
                Function::<V>::new(case).cost(&self.x).unwrap(),
                1e-14,
            );
        }
    }
}

pub fn execution() -> String {
    #[cfg(feature = "parallel")]
    {
        format!("parallel_t{}", rayon::current_num_threads())
    }
    #[cfg(not(feature = "parallel"))]
    {
        "sequential".to_owned()
    }
}

pub type NalgebraVector = na::DVector<f64>;
