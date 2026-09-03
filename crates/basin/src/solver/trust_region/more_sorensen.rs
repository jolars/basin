//! Moré–Sorensen near-exact trust-region subproblem.
//!
//! This strategy minimizes `gᵀp + ½pᵀBp` over `‖p‖ ≤ Δ` by enforcing the
//! Moré–Sorensen optimality system
//! `(B + λI)p = −g`, `B + λI ⪰ 0`, `λ ≥ 0`, and
//! `λ(Δ − ‖p‖) = 0`. Regular boundary cases solve the secular equation
//! `1/Δ − 1/‖p(λ)‖ = 0` with the safeguarded Newton update from Algorithm
//! 3.2. The eigensystem gives the exact lower shift `−λ_min(B)` and the
//! leftmost direction used to complete the step in the hard case.

use super::{CauchyPoint, Step, Subproblem, model_decrease};
use crate::core::math::{
    AddDiagonalVectorInPlace, Dot, LinearSolveSpd, MatTransposeVec, MatVec,
    NegInPlace, NormSquared, Scalar, ScaleInPlace, ScaledAdd, SymmetricEigen,
    VectorIndex, VectorLen,
};

/// Moré–Sorensen near-exact trust-region subproblem solver.
///
/// The strategy first returns an interior Newton step when `B` is positive
/// definite and `‖B⁻¹g‖ ≤ Δ`. Otherwise it eigendecomposes `B`, identifies
/// the positive-semidefinite endpoint `λ = max(0, −λ_min(B))`, and either:
///
/// - completes the endpoint's minimum-norm solution along a leftmost
///   eigenvector in the hard case; or
/// - applies safeguarded Newton iterations to the secular equation, using a
///   Cholesky solve for each shifted system.
///
/// The default cap is 50 secular iterations, which drive the radius residual
/// towards a scalar-relative floating-point tolerance; near the hard case the
/// bracket can collapse first, in which case the best feasible shift is
/// returned. If an eigendecomposition or the safeguarded factorization
/// sequence fails, the strategy returns the [`CauchyPoint`], preserving a
/// finite descent step on a finite model. Non-finite model data instead
/// produces a zero no-progress step with a non-finite predicted reduction,
/// which the driver reports as a solver failure rather than convergence.
///
/// This is an exact-Hessian strategy only; it cannot be paired with
/// [`MatrixFree`](super::MatrixFree). Its `O(n³)` eigendecomposition and
/// factorizations make it appropriate for small or medium dense problems
/// where a globally near-exact subproblem step justifies the extra work. For
/// large problems, prefer [`Steihaug`](super::Steihaug).
///
/// # Backends
///
/// Supported parameter and Hessian pairs are `Vec<F>` with
/// [`DenseMatrix`](crate::core::math::DenseMatrix), nalgebra `DVector<F>` with
/// `DMatrix<F>`, ndarray `Array1<F>` with `Array2<F>`, and faer `Col<F>` with
/// `Mat<F>`. Each pair supplies
/// [`SymmetricEigen`], [`LinearSolveSpd`], and the required dense matrix-vector
/// operations. The pure-Rust configurations are WASM-clean.
///
/// # References
///
/// Moré, J. J., & Sorensen, D. C. (1983). Computing a trust region step.
/// *SIAM Journal on Scientific and Statistical Computing*, 4(3), 553–572.
/// [doi:10.1137/0904038](https://doi.org/10.1137/0904038).
#[derive(Debug, Clone, Copy)]
pub struct MoreSorensen {
    max_iter: usize,
}

impl MoreSorensen {
    /// Construct the strategy with at most 50 safeguarded secular iterations.
    pub const fn new() -> Self {
        Self { max_iter: 50 }
    }

    /// Cap the number of safeguarded secular iterations.
    ///
    /// # Panics
    ///
    /// Panics if `max_iter` is zero.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        assert!(max_iter >= 1, "max_iter must be at least 1");
        self.max_iter = max_iter;
        self
    }
}

impl Default for MoreSorensen {
    fn default() -> Self {
        Self::new()
    }
}

impl<V, M, F> Subproblem<V, M, F> for MoreSorensen
where
    F: Scalar,
    V: Clone
        + Dot<F>
        + NormSquared<F>
        + ScaledAdd<F>
        + ScaleInPlace<F>
        + NegInPlace
        + VectorIndex<F>
        + VectorLen,
    M: Clone
        + AddDiagonalVectorInPlace<V>
        + LinearSolveSpd<V>
        + MatTransposeVec<V>
        + MatVec<V>
        + SymmetricEigen<V>,
{
    fn solve(&self, g: &V, b: &M, radius: F) -> Step<V, F> {
        let n = g.vec_len();
        if n == 0 {
            return Step {
                d: g.clone(),
                predicted_reduction: F::zero(),
                hit_boundary: false,
            };
        }

        let no_progress_step = |predicted_reduction: F| {
            let mut d = g.clone();
            for i in 0..n {
                d.set_scalar(i, F::zero());
            }
            Step {
                d,
                predicted_reduction,
                hit_boundary: false,
            }
        };
        // Non-finite model data cannot define a trustworthy secular equation.
        // A zero step keeps NaNs out of the driver, and the non-finite
        // predicted reduction distinguishes the numerical failure from the
        // stationary point that a zero reduction would otherwise claim.
        let failed_step = || no_progress_step(F::nan());

        let g_norm = g.norm_squared().sqrt();
        if !g_norm.is_finite() || !radius.is_finite() {
            return failed_step();
        }
        if radius <= F::zero() {
            // A collapsed radius admits only the zero step, which the driver
            // reads as a clean stop.
            return no_progress_step(F::zero());
        }

        let mut minus_g = g.clone();
        minus_g.neg_in_place();

        // Lemma 2.3(i): an unconstrained Newton step inside the ball solves
        // the subproblem. Trying it first also avoids an eigendecomposition
        // for the common positive-definite interior case.
        if let Ok(p) = b.solve_spd(&minus_g) {
            let p_norm = p.norm_squared().sqrt();
            if p_norm <= radius {
                let predicted_reduction = model_decrease(g, b, &p);
                return Step {
                    d: p,
                    predicted_reduction,
                    hit_boundary: p_norm >= radius,
                };
            }
        }

        // The eigensystem supplies the exact lower bound -lambda_min and the
        // leftmost eigendirection required in the hard case. A failed backend
        // eigensolve is rare and leaves the Cauchy point as a safe descent
        // fallback, matching Dogleg's treatment of a failed factorization.
        let (eigenvectors, eigenvalues) = match b.try_eigh() {
            Ok(eig) => eig,
            Err(_) => return CauchyPoint.solve(g, b, radius),
        };
        let mut min_index = 0;
        let mut lambda_min = eigenvalues.get_scalar(0);
        // Both `<` and `Float::max` propagate the non-NaN operand, so a
        // running extremum can never observe a NaN eigenvalue. Each value
        // needs its own finiteness test.
        let mut eigenvalues_finite = lambda_min.is_finite();
        for i in 1..n {
            let value = eigenvalues.get_scalar(i);
            eigenvalues_finite = eigenvalues_finite && value.is_finite();
            if value < lambda_min {
                lambda_min = value;
                min_index = i;
            }
        }
        if !eigenvalues_finite {
            return failed_step();
        }

        let lambda_floor = (-lambda_min).max(F::zero());
        let gamma = eigenvectors.mat_transpose_vec(g);
        let sqrt_eps = F::epsilon().sqrt();
        let hundred = F::from_f64(100.0).unwrap();
        let numerical_tol = hundred * F::epsilon();
        let gamma_tol = numerical_tol * g_norm;

        // Evaluate the Moore–Penrose solution at the positive-semidefinite
        // endpoint B + lambda_floor I. If the gradient has no component in
        // its null space and the minimum-norm solution lies inside the ball,
        // this is either a singular-PSD interior solution (lambda = 0) or the
        // hard case (lambda > 0). Equations (3.3) and (3.6) then say to add a
        // leftmost eigenvector until the boundary is reached.
        let mut coefficients = gamma.clone();
        let mut singular_gradient = false;
        for i in 0..n {
            let eigenvalue = eigenvalues.get_scalar(i);
            let denominator = eigenvalue + lambda_floor;
            let denominator_scale = eigenvalue.abs().max(lambda_floor.abs());
            let denom_tol = numerical_tol * denominator_scale;
            let gamma_i = gamma.get_scalar(i);
            if denominator.abs() <= denom_tol {
                singular_gradient |= gamma_i.abs() > gamma_tol;
                coefficients.set_scalar(i, F::zero());
                continue;
            }
            coefficients.set_scalar(i, -gamma_i / denominator);
        }
        // The Moore-Penrose solution is Q c in the eigenbasis, so one matvec
        // replaces the per-direction column extraction.
        let p_floor = eigenvectors.matvec(&coefficients);
        let p_floor_norm = p_floor.norm_squared().sqrt();

        let complete_to_boundary = |p_base: &V| {
            let mut basis = g.clone();
            basis.scale_in_place(F::zero());
            basis.set_scalar(min_index, F::one());
            let z = eigenvectors.matvec(&basis);
            let p_base_norm = p_base.norm_squared().sqrt();
            let pz = p_base.dot(&z);
            let radial_gap =
                (radius * radius - p_base_norm * p_base_norm).max(F::zero());
            let root = (pz * pz + radial_gap).sqrt();

            let mut positive = p_base.clone();
            positive.scaled_add(-pz + root, &z);
            let positive_reduction = model_decrease(g, b, &positive);

            let mut negative = p_base.clone();
            negative.scaled_add(-pz - root, &z);
            let negative_reduction = model_decrease(g, b, &negative);

            if positive_reduction >= negative_reduction {
                Step {
                    d: positive,
                    predicted_reduction: positive_reduction,
                    hit_boundary: true,
                }
            } else {
                Step {
                    d: negative,
                    predicted_reduction: negative_reduction,
                    hit_boundary: true,
                }
            }
        };

        if !singular_gradient && p_floor_norm <= radius {
            if lambda_floor == F::zero() {
                let predicted_reduction = model_decrease(g, b, &p_floor);
                return Step {
                    d: p_floor,
                    predicted_reduction,
                    hit_boundary: p_floor_norm >= radius,
                };
            }

            return complete_to_boundary(&p_floor);
        }

        let shifted_solve = |lambda: F| {
            let mut shifted = b.clone();
            let mut diagonal = g.clone();
            diagonal.scale_in_place(F::zero());
            for i in 0..n {
                diagonal.set_scalar(i, lambda);
            }
            shifted.add_diagonal_vector_in_place(&diagonal);
            let p = shifted.solve_spd(&minus_g).ok()?;
            Some((shifted, p))
        };

        // From (2.4), ||p(lambda)|| <= ||g|| / (lambda + lambda_min).
        // Thus lambda_floor + ||g|| / radius is a valid feasible upper
        // bracket. The small padding keeps the trial strictly above the
        // semidefinite endpoint when roundoff collapses the expression.
        let pad = sqrt_eps * (F::one() + lambda_floor.abs());
        let mut lambda_lower = lambda_floor;
        let mut lambda_upper = lambda_floor + g_norm / radius;
        if lambda_upper <= lambda_lower {
            lambda_upper = lambda_lower + pad;
        }

        let two = F::from_f64(2.0).unwrap();
        let mut feasible = None;
        for _ in 0..self.max_iter {
            if let Some((shifted, p)) = shifted_solve(lambda_upper) {
                if p.norm_squared().sqrt() <= radius {
                    feasible = Some((shifted, p));
                    break;
                }
            }
            lambda_upper =
                lambda_lower + two * (lambda_upper - lambda_lower).max(pad);
        }
        let Some((upper_shifted, upper_p)) = feasible else {
            return CauchyPoint.solve(g, b, radius);
        };

        let radius_tol = numerical_tol * radius.max(F::min_positive_value());
        let tenth = F::from_f64(0.1).unwrap();
        let mut lambda = lambda_upper;
        let mut best_feasible = upper_p.clone();
        // The bracket search already factorized `B + lambda_upper I`; the
        // first secular iteration reuses that solve instead of repeating it.
        let mut bracket_solve = Some((upper_shifted, upper_p));

        for _ in 0..self.max_iter {
            let solved = match bracket_solve.take() {
                cached @ Some(_) => cached,
                None => shifted_solve(lambda),
            };
            let Some((shifted, p)) = solved else {
                lambda_lower = lambda;
                let midpoint = (lambda_lower + lambda_upper) / two;
                if midpoint <= lambda_lower || midpoint >= lambda_upper {
                    break;
                }
                lambda = midpoint;
                continue;
            };
            let p_norm = p.norm_squared().sqrt();
            if !p_norm.is_finite() {
                return failed_step();
            }
            if (p_norm - radius).abs() <= radius_tol {
                let predicted_reduction = model_decrease(g, b, &p);
                return Step {
                    d: p,
                    predicted_reduction,
                    hit_boundary: true,
                };
            }

            if p_norm > radius {
                lambda_lower = lambda;
            } else {
                lambda_upper = lambda;
                best_feasible = p.clone();
            }

            // Algorithm 3.2's Newton update. Solving A q = p gives
            // p^T q = ||R^{-T}p||^2 for A = B + lambda I = R^T R, so the
            // update can use the existing Cholesky solve abstraction without
            // exposing backend-specific triangular factors.
            let candidate = shifted.solve_spd(&p).ok().and_then(|q| {
                let p_q = p.dot(&q);
                if p_q <= F::zero() || !p_q.is_finite() {
                    return None;
                }
                let correction =
                    (p_norm * p_norm / p_q) * ((p_norm - radius) / radius);
                let trial = lambda + correction;
                trial.is_finite().then_some(trial)
            });

            let width = lambda_upper - lambda_lower;
            let lower_guard = lambda_lower + tenth * width;
            let upper_guard = lambda_upper - tenth * width;
            let next_lambda = match candidate {
                // Safeguard by clamping into the bracket's middle band rather
                // than discarding the trial. Newton converges monotonically
                // towards the root from whichever side it starts, so its
                // corrections shrink as it closes in; rejecting them for the
                // bracket midpoint degrades the iteration to bisection
                // exactly where Newton is most accurate.
                Some(trial) => trial.max(lower_guard).min(upper_guard),
                None => (lambda_lower + lambda_upper) / two,
            };
            if next_lambda <= lambda_lower || next_lambda >= lambda_upper {
                break;
            }
            lambda = next_lambda;
        }

        // Close to the semidefinite endpoint, the secular root can lie
        // between adjacent representable shifts. Completing the endpoint's
        // minimum-norm solution along a leftmost eigenvector avoids returning
        // a materially interior step when the bracket can no longer shrink.
        if p_floor_norm <= radius && lambda_upper - lambda_floor <= pad {
            return complete_to_boundary(&p_floor);
        }

        let predicted_reduction = model_decrease(g, b, &best_feasible);
        Step {
            d: best_feasible,
            predicted_reduction,
            // The secular loop runs only when the semidefinite endpoint is
            // infeasible or its gradient component is singular, so lambda* is
            // strictly positive and complementarity puts the solution on the
            // boundary. Re-deriving the flag from the radius residual would
            // report a boundary step as interior whenever the bracket
            // collapses before that tolerance is met, silently disabling the
            // driver's radius growth (N&W Algorithm 4.1).
            hit_boundary: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::math::DenseMatrix;

    fn solve(g: Vec<f64>, b: &[f64], radius: f64) -> Step<Vec<f64>, f64> {
        let n = g.len();
        MoreSorensen::new().solve(
            &g,
            &DenseMatrix::from_row_slice(n, n, b),
            radius,
        )
    }

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "actual={actual}, expected={expected}, tol={tol}"
        );
    }

    #[test]
    fn returns_the_interior_newton_step() {
        let step = solve(vec![-2.0, 1.0], &[2.0, 0.0, 0.0, 1.0], 10.0);

        assert_close(step.d[0], 1.0, 1e-12);
        assert_close(step.d[1], -1.0, 1e-12);
        assert!(!step.hit_boundary);
        assert_close(step.predicted_reduction, 1.5, 1e-12);
    }

    #[test]
    fn solves_the_positive_definite_boundary_case() {
        let g = vec![-2.0, -1.0];
        let step = solve(g.clone(), &[2.0, 0.0, 0.0, 1.0], 0.5);
        let norm = step.d.iter().map(|x| x * x).sum::<f64>().sqrt();
        let lambda0 = -g[0] / step.d[0] - 2.0;
        let lambda1 = -g[1] / step.d[1] - 1.0;

        assert_close(norm, 0.5, 1e-10);
        assert_close(lambda0, lambda1, 1e-9);
        assert!(lambda0 > 0.0);
        assert!(step.hit_boundary);
        assert!(step.predicted_reduction > 0.0);
    }

    #[test]
    fn does_not_mistake_small_positive_curvature_for_a_null_space() {
        let step = solve(vec![-1e-10, 0.0], &[1e-12, 0.0, 0.0, 1.0], 1.0);

        assert_close(step.d[0], 1.0, 1e-8);
        assert_close(step.d[1], 0.0, 1e-12);
        assert!(step.hit_boundary);
    }

    #[test]
    fn scales_null_space_tolerances_for_f32_models() {
        let b = DenseMatrix::from_row_slice(2, 2, &[1e-12_f32, 0.0, 0.0, 1.0]);
        let step = MoreSorensen::new().solve(&vec![-1e-6_f32, 0.0], &b, 1.0);

        assert!((step.d[0] - 1.0).abs() < 1e-5);
        assert!(step.d[1].abs() < 1e-6);
        assert!(step.hit_boundary);
        assert!(step.predicted_reduction > 0.0);
    }

    #[test]
    fn boundary_tolerance_scales_with_a_small_radius() {
        let step = solve(vec![-1.0], &[1.0], 1e-12);

        assert_close(step.d[0], 1e-12, 1e-24);
        assert!(step.hit_boundary);
    }

    #[test]
    fn solves_an_indefinite_regular_case() {
        let g = vec![1.0, 1.0];
        let step = solve(g.clone(), &[-1.0, 0.0, 0.0, 2.0], 1.0);
        let norm = step.d.iter().map(|x| x * x).sum::<f64>().sqrt();
        let lambda0 = -g[0] / step.d[0] + 1.0;
        let lambda1 = -g[1] / step.d[1] - 2.0;

        assert_close(norm, 1.0, 1e-10);
        assert_close(lambda0, lambda1, 1e-9);
        assert!(lambda0 > 1.0);
        assert!(step.hit_boundary);
    }

    #[test]
    fn handles_the_paper_hard_case() {
        // Moré–Sorensen, p. 556: B = diag(-1, 1), g = (0, 1). At Δ = 1,
        // p = (0, -1/2) is completed along the leftmost eigendirection.
        let step = solve(vec![0.0, 1.0], &[-1.0, 0.0, 0.0, 1.0], 1.0);

        assert_close(step.d[0].abs(), 3.0_f64.sqrt() / 2.0, 1e-10);
        assert_close(step.d[1], -0.5, 1e-10);
        assert_close(step.d.iter().map(|x| x * x).sum::<f64>(), 1.0, 1e-10);
        assert_close(step.predicted_reduction, 0.75, 1e-10);
        assert!(step.hit_boundary);
    }

    #[test]
    fn completes_a_stalled_near_hard_case_on_the_boundary() {
        let step = solve(vec![5e-14, 1.0], &[-1.0, 0.0, 0.0, 1.0], 100.0);
        let norm = step.d.iter().map(|x| x * x).sum::<f64>().sqrt();

        assert_close(norm, 100.0, 1e-10);
        assert!(step.d[0] < 0.0);
        assert_close(step.d[1], -0.5, 1e-10);
        assert!(step.predicted_reduction > 5000.24);
        assert!(step.hit_boundary);
    }

    #[test]
    fn follows_negative_curvature_when_the_gradient_is_zero() {
        let step = solve(vec![0.0, 0.0], &[-2.0, 0.0, 0.0, 1.0], 3.0);

        assert_close(step.d[0].abs(), 3.0, 1e-12);
        assert_close(step.d[1], 0.0, 1e-12);
        assert_close(step.predicted_reduction, 9.0, 1e-12);
        assert!(step.hit_boundary);
    }

    #[test]
    fn returns_the_minimum_norm_solution_for_a_singular_psd_model() {
        let step = solve(vec![0.0, -2.0], &[0.0, 0.0, 0.0, 2.0], 2.0);

        assert_close(step.d[0], 0.0, 1e-12);
        assert_close(step.d[1], 1.0, 1e-12);
        assert!(!step.hit_boundary);
        assert_close(step.predicted_reduction, 1.0, 1e-12);
    }

    #[test]
    fn flags_a_near_hard_case_boundary_step_as_boundary() {
        // A near-hard case: the secular root sits just above -lambda_min, so
        // the bracket collapses before the radius residual reaches the
        // scalar-relative tolerance and the step returns through the
        // best-feasible fallback. It is still a boundary step, and reporting
        // it as interior would block the driver's radius growth.
        let step = solve(vec![1e-5, 1.0], &[-1.0, 0.0, 0.0, 1.0], 1.0);
        let norm = step.d.iter().map(|x| x * x).sum::<f64>().sqrt();

        assert_close(norm, 1.0, 1e-9);
        assert!(norm <= 1.0);
        assert!(step.hit_boundary);
    }

    #[test]
    fn nonfinite_gradient_reports_a_numerical_failure() {
        let step = solve(vec![f64::NAN, 1.0], &[1.0, 0.0, 0.0, 1.0], 1.0);

        assert_eq!(step.d, vec![0.0, 0.0]);
        assert!(step.predicted_reduction.is_nan());
        assert!(!step.hit_boundary);
    }

    #[test]
    fn supports_f32() {
        let b = DenseMatrix::from_row_slice(2, 2, &[2.0_f32, 0.0, 0.0, 1.0]);
        let step = MoreSorensen::new().solve(&vec![-2.0_f32, 1.0], &b, 10.0);

        assert!((step.d[0] - 1.0).abs() < 1e-5);
        assert!((step.d[1] + 1.0).abs() < 1e-5);
        assert!(!step.hit_boundary);
    }

    #[cfg(feature = "nalgebra_all")]
    #[test]
    fn hard_case_runs_on_nalgebra() {
        use nalgebra::{DMatrix, DVector};

        let g = DVector::<f64>::from_vec(vec![0.0, 1.0]);
        let b = DMatrix::<f64>::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, 1.0]);
        let step = MoreSorensen::new().solve(&g, &b, 1.0);

        assert_close(step.d[0].abs(), 3.0_f64.sqrt() / 2.0, 1e-10);
        assert_close(step.d[1], -0.5, 1e-10);
    }

    #[cfg(feature = "ndarray_all")]
    #[test]
    fn hard_case_runs_on_ndarray() {
        use ndarray::{Array1, Array2};

        let g = Array1::<f64>::from_vec(vec![0.0, 1.0]);
        let b =
            Array2::<f64>::from_shape_vec((2, 2), vec![-1.0, 0.0, 0.0, 1.0])
                .unwrap();
        let step = MoreSorensen::new().solve(&g, &b, 1.0);

        assert_close(step.d[0].abs(), 3.0_f64.sqrt() / 2.0, 1e-10);
        assert_close(step.d[1], -0.5, 1e-10);
    }

    #[cfg(feature = "faer_all")]
    #[test]
    fn hard_case_runs_on_faer() {
        use faer::{Col, Mat};

        let g = Col::<f64>::from_fn(2, |i| [0.0, 1.0][i]);
        let b =
            Mat::<f64>::from_fn(2, 2, |i, j| [[-1.0, 0.0], [0.0, 1.0]][i][j]);
        let step = MoreSorensen::new().solve(&g, &b, 1.0);

        assert_close(step.d[0].abs(), 3.0_f64.sqrt() / 2.0, 1e-10);
        assert_close(step.d[1], -0.5, 1e-10);
    }
}
