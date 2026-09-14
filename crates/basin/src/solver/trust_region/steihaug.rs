//! Steihaug truncated conjugate-gradient trust-region subproblem.

use super::{
    Step, Subproblem, SubproblemHvp, model_decrease_from_bd, tau_to_boundary,
};
use crate::core::math::{
    Dot, MatVec, NegInPlace, NormSquared, Scalar, ScaleInPlace, ScaledAdd,
    VectorLen,
};

/// Steihaug truncated conjugate-gradient subproblem solver (Nocedal &
/// Wright, *Numerical Optimization*, 2e, Algorithm 7.2; Steihaug 1983).
///
/// Runs conjugate gradient on the model `m(p) = gᵀp + ½ pᵀ B p`, truncating
/// as soon as it (a) generates a direction of non-positive curvature
/// `dᵀBd ≤ 0`, or (b) would step outside the trust region. In either case it
/// follows the current direction to the boundary and stops; otherwise it
/// converges to the unconstrained model minimizer inside the region, with a
/// residual tolerance `ε = min(½, √‖g‖) · ‖g‖` (the standard forcing
/// sequence). Matrix-free, it touches the Hessian only through
/// Hessian-vector products — [`MatVec`] in
/// [`TrustRegion`](super::TrustRegion)'s exact mode, the problem's
/// [`HessianProduct`](crate::core::problem::HessianProduct) in
/// [`MatrixFree`](super::MatrixFree) mode — so it runs on every backend and
/// scales to large, possibly indefinite problems.
///
/// Each returned step costs one extra product for the predicted model
/// reduction (identically in both modes); recovering the model value from
/// the CG recurrence instead is a possible future micro-optimization.
///
/// This is [`TrustRegion`](super::TrustRegion)'s default subproblem.
/// Use [`with_forcing_parameters`](Self::with_forcing_parameters)
/// to configure the adaptive residual tolerance, and
/// [`with_max_iter`](Self::with_max_iter) to allow more than `n` iterations
/// when roundoff destroys conjugacy on an ill-conditioned model.
///
/// # Backends
///
/// `Vec<F>`, nalgebra, ndarray, and faer, in both exact-Hessian and
/// matrix-free trust-region modes.
#[derive(Debug, Clone, Copy)]
pub struct Steihaug {
    /// CG iteration cap; `None` uses the problem dimension `n` (CG's natural
    /// bound for exact arithmetic).
    max_iter: Option<usize>,
}

impl Steihaug {
    /// Steihaug truncated CG with the default iteration cap (the problem
    /// dimension `n`).
    pub fn new() -> Self {
        Self { max_iter: None }
    }

    /// Cap the number of CG iterations per subproblem solve. By default the
    /// cap is the problem dimension `n`; lowering it yields a cheaper, looser
    /// step. Raising it can help ill-conditioned models because roundoff
    /// can require more than `n` CG iterations. Must be `≥ 1`.
    pub fn with_max_iter(mut self, n: usize) -> Self {
        assert!(n >= 1, "max_iter must be ≥ 1");
        self.max_iter = Some(n);
        self
    }

    /// Configure the inner CG forcing rule: stop when
    /// `‖r‖₂ < min(kappa, ‖g‖₂^theta) · ‖g‖₂`, where `r = g + Bp` is the
    /// recursively updated CG residual. An exact-zero residual is always
    /// accepted. Boundary and non-positive-curvature stops still apply.
    ///
    /// The unconfigured solver uses `kappa = 0.5` and `theta = 0.5`.
    /// `kappa` caps the relative residual threshold; positive `theta`
    /// tightens it as the gradient norm approaches zero. For example,
    /// `(0.1, 1.0)` gives a tighter adaptive rule. Setting `theta = 0`
    /// gives a fixed relative tolerance of `kappa`; setting `kappa = 0`
    /// requests an exact-zero residual.
    ///
    /// A tighter tolerance can reduce outer iterations and derivative
    /// evaluations, but spends more Hessian-vector products on each model.
    /// On ill-conditioned problems, also consider increasing
    /// [`with_max_iter`](Self::with_max_iter): the dimension-sized default
    /// need not suffice in floating-point arithmetic. This controls the
    /// inner model solve, not the outer solver's convergence tests.
    ///
    /// The returned configuration uses the scalar type of the parameters.
    /// The unconfigured `Steihaug` remains usable with any scalar type.
    ///
    /// # Panics
    ///
    /// Panics unless `kappa` is finite and in `[0, 1)`, and `theta` is
    /// finite and nonnegative.
    ///
    /// # Example
    ///
    /// ```
    /// use basin::{Steihaug, TrustRegion};
    /// let subproblem = Steihaug::new().with_forcing_parameters(0.1, 1.0);
    /// let solver: TrustRegion<_> = TrustRegion::with_subproblem(subproblem);
    /// ```
    pub fn with_forcing_parameters<F: Scalar>(
        self,
        kappa: F,
        theta: F,
    ) -> SteihaugWithForcing<F> {
        assert!(
            kappa.is_finite() && kappa >= F::zero() && kappa < F::one(),
            "kappa must be finite and in [0, 1)"
        );
        assert!(
            theta.is_finite() && theta >= F::zero(),
            "theta must be finite and nonnegative"
        );
        SteihaugWithForcing {
            inner: self,
            kappa,
            theta,
        }
    }

    /// The CG core, driven by a fallible Hessian-vector product `bv: v ↦ B·v`.
    /// Both subproblem impls delegate here: the exact path wraps
    /// [`MatVec`] in an infallible closure, the matrix-free path wraps the
    /// counted [`HessianProduct`](crate::core::problem::HessianProduct) call.
    pub(crate) fn solve_with<V, F, E>(
        &self,
        g: &V,
        radius: F,
        forcing_parameters: Option<(F, F)>,
        mut bv: impl FnMut(&V) -> Result<V, E>,
    ) -> Result<Step<V, F>, E>
    where
        F: Scalar,
        V: Clone
            + Dot<F>
            + NormSquared<F>
            + ScaledAdd<F>
            + ScaleInPlace<F>
            + NegInPlace
            + VectorLen,
    {
        let n = g.vec_len();
        let max_iter = self.max_iter.unwrap_or(n.max(1));

        // z₀ = 0, r₀ = g, d₀ = −r₀.
        let mut z = g.clone();
        z.scale_in_place(F::zero());
        let mut r = g.clone();
        let mut r_dot = r.dot(&r);

        let g_norm = r_dot.sqrt();
        // Avoid comparing the norm with the tolerance here: an infinite
        // gradient can make both infinite and falsely signal convergence.
        // A zero step needs no product to establish its model decrease.
        if g_norm == F::zero() {
            return Ok(Step {
                d: z,
                predicted_reduction: F::zero(),
                hit_boundary: false,
            });
        }

        let half = F::from_f64(0.5).unwrap();
        let (kappa, theta) = forcing_parameters.unwrap_or((half, half));
        // Avoid a general power for common rules, retaining the default's
        // square-root rounding as well as its computational cost.
        let power = if theta == F::zero() {
            F::one()
        } else if theta == half {
            g_norm.sqrt()
        } else if theta == F::one() {
            g_norm
        } else {
            g_norm.powf(theta)
        };
        let tol = (if power < kappa { power } else { kappa }) * g_norm;

        let mut d = r.clone();
        d.neg_in_place();

        for _ in 0..max_iter {
            let bd = bv(&d)?;
            let dbd = d.dot(&bd);

            // Non-positive curvature: the model is unbounded along ±d, so
            // walk to the trust-region boundary and stop.
            if dbd <= F::zero() {
                let tau = tau_to_boundary(&z, &d, radius);
                z.scaled_add(tau, &d);
                let bz = bv(&z)?;
                let predicted_reduction = model_decrease_from_bd(g, &z, &bz);
                return Ok(Step {
                    d: z,
                    predicted_reduction,
                    hit_boundary: true,
                });
            }

            let alpha = r_dot / dbd;

            // Tentative CG iterate z + α d. If it leaves the region, clip to
            // the boundary along d instead.
            let mut z_next = z.clone();
            z_next.scaled_add(alpha, &d);
            if z_next.norm_squared().sqrt() >= radius {
                let tau = tau_to_boundary(&z, &d, radius);
                z.scaled_add(tau, &d);
                let bz = bv(&z)?;
                let predicted_reduction = model_decrease_from_bd(g, &z, &bz);
                return Ok(Step {
                    d: z,
                    predicted_reduction,
                    hit_boundary: true,
                });
            }
            z = z_next;

            // r ← r + α B d; converged inside the region once it is small.
            r.scaled_add(alpha, &bd);
            let r_dot_next = r.dot(&r);
            let residual_norm = r_dot_next.sqrt();
            // An exact residual also satisfies a zero threshold, including
            // one caused by underflow in an aggressively tight forcing rule.
            let converged = residual_norm == F::zero() || residual_norm < tol;
            if converged {
                let bz = bv(&z)?;
                let predicted_reduction = model_decrease_from_bd(g, &z, &bz);
                return Ok(Step {
                    d: z,
                    predicted_reduction,
                    hit_boundary: false,
                });
            }

            // d ← −r + β d.
            let beta = r_dot_next / r_dot;
            let mut d_next = r.clone();
            d_next.neg_in_place();
            d_next.scaled_add(beta, &d);
            d = d_next;
            r_dot = r_dot_next;
        }

        // Iteration cap reached: return the best interior iterate so far.
        let bz = bv(&z)?;
        let predicted_reduction = model_decrease_from_bd(g, &z, &bz);
        Ok(Step {
            d: z,
            predicted_reduction,
            hit_boundary: false,
        })
    }
}

impl Default for Steihaug {
    fn default() -> Self {
        Self::new()
    }
}

impl<V, M, F> Subproblem<V, M, F> for Steihaug
where
    F: Scalar,
    V: Clone
        + Dot<F>
        + NormSquared<F>
        + ScaledAdd<F>
        + ScaleInPlace<F>
        + NegInPlace
        + VectorLen,
    M: MatVec<V>,
{
    fn solve(&self, g: &V, b: &M, radius: F) -> Step<V, F> {
        match self.solve_with(g, radius, None, |v| {
            Ok::<_, std::convert::Infallible>(b.matvec(v))
        }) {
            Ok(step) => step,
            Err(never) => match never {},
        }
    }
}

impl<V, F> SubproblemHvp<V, F> for Steihaug
where
    F: Scalar,
    V: Clone
        + Dot<F>
        + NormSquared<F>
        + ScaledAdd<F>
        + ScaleInPlace<F>
        + NegInPlace
        + VectorLen,
{
    fn solve_hvp<E>(
        &self,
        g: &V,
        radius: F,
        bv: impl FnMut(&V) -> Result<V, E>,
    ) -> Result<Step<V, F>, E> {
        self.solve_with(g, radius, None, bv)
    }
}

/// Steihaug CG configured with an adaptive residual forcing rule.
///
/// Construct this with
/// [`Steihaug::with_forcing_parameters`]. The iteration cap is
/// inherited from that `Steihaug` configuration and can be changed with
/// [`with_max_iter`](Self::with_max_iter).
///
/// # Backends
///
/// The same as [`Steihaug`]: `Vec<F>`, nalgebra, ndarray, and faer, in
/// both exact-Hessian and matrix-free trust-region modes.
#[derive(Debug, Clone, Copy)]
pub struct SteihaugWithForcing<F: Scalar = f64> {
    inner: Steihaug,
    kappa: F,
    theta: F,
}

impl<F: Scalar> SteihaugWithForcing<F> {
    /// Cap CG iterations per subproblem solve. Must be at least one.
    ///
    /// See [`Steihaug::with_max_iter`] for the dimension-sized default.
    pub fn with_max_iter(mut self, n: usize) -> Self {
        self.inner = self.inner.with_max_iter(n);
        self
    }
}

impl<V, M, F> Subproblem<V, M, F> for SteihaugWithForcing<F>
where
    F: Scalar,
    V: Clone
        + Dot<F>
        + NormSquared<F>
        + ScaledAdd<F>
        + ScaleInPlace<F>
        + NegInPlace
        + VectorLen,
    M: MatVec<V>,
{
    fn solve(&self, g: &V, b: &M, radius: F) -> Step<V, F> {
        match self.inner.solve_with(
            g,
            radius,
            Some((self.kappa, self.theta)),
            |v| Ok::<_, std::convert::Infallible>(b.matvec(v)),
        ) {
            Ok(step) => step,
            Err(never) => match never {},
        }
    }
}

impl<V, F> SubproblemHvp<V, F> for SteihaugWithForcing<F>
where
    F: Scalar,
    V: Clone
        + Dot<F>
        + NormSquared<F>
        + ScaledAdd<F>
        + ScaleInPlace<F>
        + NegInPlace
        + VectorLen,
{
    fn solve_hvp<E>(
        &self,
        g: &V,
        radius: F,
        bv: impl FnMut(&V) -> Result<V, E>,
    ) -> Result<Step<V, F>, E> {
        self.inner
            .solve_with(g, radius, Some((self.kappa, self.theta)), bv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::math::DenseMatrix;

    #[test]
    fn forcing_parameters_control_inner_accuracy() {
        let h = DenseMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 2.0]);
        // On this model, the first CG step reduces the residual norm by
        // exactly a factor of three. A tighter threshold needs both steps.
        for (norm, kappa, theta, iterations) in [
            (10.0_f64, 0.5, 0.5, 1),
            (10.0, 0.1, 1.0, 2),
            (0.16, 0.5, 0.0, 1),
            (0.16, 0.5, 0.5, 1),
            (0.16, 0.5, 1.0, 2),
            (0.5, 0.5, 1.5, 1),
            (0.5, 0.5, 2.0, 2),
            (1e-8, 0.5, 0.0, 1),
            (1e-8, 0.1, 1.0, 2),
        ] {
            let scale = norm / 2.0_f64.sqrt();
            let g = vec![scale, scale];
            let sub = Steihaug::new().with_forcing_parameters(kappa, theta);
            let mut products = 0;
            let free = sub
                .solve_hvp(&g, 100.0, |v| {
                    products += 1;
                    Ok::<_, ()>(h.matvec(v))
                })
                .unwrap();
            assert_eq!(products, iterations + 1);
            let expected = if iterations == 1 {
                [-2.0 * scale / 3.0; 2]
            } else {
                [-scale, -scale / 2.0]
            };
            for step in [free, sub.solve(&g, &h, 100.0)] {
                assert!(!step.hit_boundary);
                for (actual, expected) in step.d.iter().zip(expected) {
                    assert!((actual - expected).abs() < 1e-12 * norm);
                }
                let expected_reduction = if iterations == 1 {
                    2.0 * scale * scale / 3.0
                } else {
                    0.75 * scale * scale
                };
                assert!(
                    (step.predicted_reduction - expected_reduction).abs()
                        < 1e-12 * norm * norm
                );
            }
        }
    }

    #[test]
    fn explicit_default_forcing_preserves_steps_and_iteration_caps() {
        let h = DenseMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 3.0]);
        // Equal components place the first residual exactly on the default
        // relative threshold, guarding its strict comparison as well.
        for value in [0.0_f64, 0.005, 3.0] {
            let g = vec![value, value];
            for cap in [1, 2, 5] {
                let default = Steihaug::new().with_max_iter(cap);
                let mut default_products = 0;
                let expected = default
                    .solve_hvp(&g, 100.0, |v| {
                        default_products += 1;
                        Ok::<_, ()>(h.matvec(v))
                    })
                    .unwrap();
                for configured in [
                    default.with_forcing_parameters(0.5, 0.5),
                    Steihaug::new()
                        .with_forcing_parameters(0.5, 0.5)
                        .with_max_iter(cap),
                ] {
                    let mut products = 0;
                    let actual = configured
                        .solve_hvp(&g, 100.0, |v| {
                            products += 1;
                            Ok::<_, ()>(h.matvec(v))
                        })
                        .unwrap();
                    assert_eq!(actual.d, expected.d);
                    assert_eq!(
                        actual.predicted_reduction,
                        expected.predicted_reduction
                    );
                    assert_eq!(actual.hit_boundary, expected.hit_boundary);
                    assert_eq!(products, default_products);
                }
            }
        }
    }

    #[test]
    fn fixed_tolerance_needs_more_than_dimension_steps() {
        let n = 20;
        let diagonal: Vec<_> = (0..n)
            .map(|i| 2.0 * 10.0_f64.powf(4.0 * i as f64 / (n - 1) as f64))
            .collect();
        let g = diagonal.clone();
        let run = |cap| {
            let mut products = 0;
            let step = Steihaug::new()
                .with_forcing_parameters(1e-9, 0.0)
                .with_max_iter(cap)
                .solve_hvp(&g, 100.0, |v: &Vec<f64>| {
                    products += 1;
                    Ok::<_, ()>(
                        v.iter().zip(&diagonal).map(|(v, a)| v * a).collect(),
                    )
                })
                .unwrap();
            let residual = g
                .iter()
                .zip(&step.d)
                .zip(&diagonal)
                .map(|((g, p), a)| (g + a * p).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(!step.hit_boundary);
            (residual / g.norm_squared().sqrt(), products, step)
        };
        let (capped, _, _) = run(n);
        let (converged, products, step) = run(2 * n);
        assert!(capped > 1e-9, "{capped}");
        assert!(converged < 1e-9, "{converged}");
        assert!(products > n + 1 && products <= 2 * n + 1);
        assert!(step.d.iter().all(|p| (p + 1.0).abs() < 1e-6));
        assert!(
            (step.predicted_reduction - diagonal.iter().sum::<f64>() / 2.0)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn configured_forcing_preserves_boundary_and_negative_curvature() {
        for curvature in [2.0_f64, 0.0, -2.0] {
            let h = DenseMatrix::from_fn(2, 2, |i, j| {
                if i == j { curvature } else { 0.0 }
            });
            let g = vec![3.0, 4.0];
            for (kappa, theta) in [(1e-9, 0.0), (0.1, 1.0)] {
                let sub = Steihaug::new().with_forcing_parameters(kappa, theta);
                let free = sub
                    .solve_hvp(&g, 0.5, |v| Ok::<_, ()>(h.matvec(v)))
                    .unwrap();
                for step in [sub.solve(&g, &h, 0.5), free] {
                    assert!(step.hit_boundary);
                    assert!((step.d.norm_squared().sqrt() - 0.5).abs() < 1e-12);
                    assert!(
                        (step.predicted_reduction - (2.5 - 0.125 * curvature))
                            .abs()
                            < 1e-12
                    );
                }
            }
        }
    }

    #[test]
    fn zero_tolerance_accepts_an_exact_residual_and_zero_gradient() {
        for (kappa, theta) in [(0.0_f32, 0.0), (0.0, 1.0), (0.1, 1e6)] {
            let sub = Steihaug::new().with_forcing_parameters(kappa, theta);
            for g in [vec![0.0_f32, 0.0], vec![0.002, 0.004]] {
                let mut products = 0;
                let step = sub
                    .solve_hvp(&g, 10.0, |v: &Vec<f32>| {
                        products += 1;
                        Ok::<_, ()>(v.iter().map(|v| 2.0 * v).collect())
                    })
                    .unwrap();
                assert_eq!(
                    step.d,
                    g.iter().map(|g| -g / 2.0).collect::<Vec<_>>()
                );
                assert_eq!(products, if g[0] == 0.0 { 0 } else { 2 });
            }
        }
    }

    #[test]
    fn configured_forcing_propagates_product_errors() {
        for (kappa, theta) in [(1e-9, 0.0), (0.1, 1.0)] {
            let result = Steihaug::new()
                .with_forcing_parameters(kappa, theta)
                .solve_hvp(&vec![1.0], 10.0, |_| {
                    Err::<Vec<f64>, _>("product failed")
                });
            assert!(matches!(result, Err("product failed")));
        }
    }

    #[test]
    fn forcing_parameters_reject_invalid_settings() {
        for kappa in [-1.0, 1.0, f64::NEG_INFINITY, f64::INFINITY, f64::NAN] {
            assert!(
                std::panic::catch_unwind(
                    || Steihaug::new().with_forcing_parameters(kappa, 0.5)
                )
                .is_err()
            );
        }
        for theta in [-1.0, f64::NEG_INFINITY, f64::INFINITY, f64::NAN] {
            assert!(
                std::panic::catch_unwind(
                    || Steihaug::new().with_forcing_parameters(0.5, theta)
                )
                .is_err()
            );
        }
    }
    fn check_backend<V, M>(g: V, h: M)
    where
        V: Clone
            + Dot<f64>
            + NormSquared<f64>
            + ScaledAdd<f64>
            + ScaleInPlace<f64>
            + NegInPlace
            + VectorLen,
        M: MatVec<V>,
    {
        for (kappa, theta) in [(1e-10, 0.0), (0.1, 1.0)] {
            let sub = Steihaug::new().with_forcing_parameters(kappa, theta);
            let exact = sub.solve(&g, &h, 100.0);
            let free = sub
                .solve_hvp(&g, 100.0, |v| Ok::<_, ()>(h.matvec(v)))
                .unwrap();
            for step in [exact, free] {
                let mut residual = h.matvec(&step.d);
                residual.scaled_add(1.0, &g);
                assert!(residual.norm_squared().sqrt() < 1e-8);
                assert!((step.predicted_reduction - 7.0).abs() < 1e-8);
                assert!(!step.hit_boundary);
            }
        }
    }

    #[test]
    fn configured_vec_backend() {
        check_backend(
            vec![2.0, 6.0],
            DenseMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]),
        );
    }

    #[cfg(feature = "nalgebra_all")]
    #[test]
    fn configured_nalgebra_backend() {
        check_backend(
            nalgebra::DVector::from_vec(vec![2.0, 6.0]),
            nalgebra::DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]),
        );
    }

    #[cfg(feature = "ndarray_all")]
    #[test]
    fn configured_ndarray_backend() {
        check_backend(
            ndarray::Array1::from_vec(vec![2.0, 6.0]),
            ndarray::Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 3.0])
                .unwrap(),
        );
    }

    #[cfg(feature = "faer_all")]
    #[test]
    fn configured_faer_backend() {
        check_backend(
            faer::Col::from_fn(2, |i| [2.0, 6.0][i]),
            faer::Mat::from_fn(2, 2, |i, j| [[2.0, 0.0], [0.0, 3.0]][i][j]),
        );
    }

    #[test]
    fn nonfinite_gradient_does_not_signal_convergence() {
        for value in [f64::NAN, f64::INFINITY, f64::MAX] {
            let g = vec![value, 1.0];
            let h = DenseMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
            assert!(
                !Steihaug::new()
                    .solve(&g, &h, 1.0)
                    .predicted_reduction
                    .is_finite()
            );
            for (kappa, theta) in [(1e-9, 0.0), (0.1, 1.0)] {
                assert!(
                    !Steihaug::new()
                        .with_forcing_parameters(kappa, theta)
                        .solve(&g, &h, 1.0)
                        .predicted_reduction
                        .is_finite()
                );
            }
        }
    }
    #[test]
    fn tighter_model_solve_reduces_outer_derivative_work() {
        use crate::{
            BasicState, CostFunction, Executor, Gradient, Hessian, TrustRegion,
        };
        struct Quadratic;
        fn diagonal(i: usize) -> f64 {
            2.0 * 10.0_f64.powf(4.0 * i as f64 / 19.0)
        }
        impl CostFunction for Quadratic {
            type Param = Vec<f64>;
            type Output = f64;
            type Error = std::convert::Infallible;
            fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
                Ok(x.iter()
                    .enumerate()
                    .map(|(i, x)| 0.5 * diagonal(i) * x * x)
                    .sum())
            }
        }
        impl Gradient for Quadratic {
            type Gradient = Vec<f64>;
            fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
                Ok(x.iter().enumerate().map(|(i, x)| diagonal(i) * x).collect())
            }
        }
        impl Hessian for Quadratic {
            type Hessian = DenseMatrix;
            fn hessian(
                &self,
                _: &Vec<f64>,
            ) -> Result<DenseMatrix, Self::Error> {
                Ok(DenseMatrix::from_fn(20, 20, |i, j| {
                    if i == j { diagonal(i) } else { 0.0 }
                }))
            }
        }
        let result = Executor::new(
            Quadratic,
            TrustRegion::with_subproblem(
                Steihaug::new()
                    .with_forcing_parameters(1e-9, 0.0)
                    .with_max_iter(40),
            ),
            BasicState::new(vec![1.0; 20]),
        )
        .max_iter(100)
        .target_cost(1e-6)
        .run_with_solver()
        .unwrap();
        assert!(result.cost() <= 1e-6);
        assert!(result.counts.hessian_evals <= 4);
        let cost = Quadratic.cost(result.param()).unwrap();
        assert!((cost - result.cost()).abs() < 1e-15);
    }
}
