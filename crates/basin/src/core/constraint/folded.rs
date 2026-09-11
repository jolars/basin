use std::ops::Index;

use num_traits::Float;

use super::NonlinearConstraints;
use crate::core::{
    math::{MatVec, Scalar, VectorLen},
    problem::CostFunction,
};

/// Adapt a [`NonlinearConstraints`] problem for [`Cobyla`](crate::Cobyla).
///
/// The adapter forwards objective evaluations and folds all constraint blocks
/// into a single `c(x) ≤ 0` vector, in this order:
///
/// 1. Finite lower bounds: `lower − x`.
/// 2. Finite upper bounds: `x − upper`.
/// 3. Negative equality residuals: `b_eq − A_eq x`.
/// 4. Positive equality residuals: `A_eq x − b_eq`.
/// 5. Linear inequalities: `A_ineq x − b_ineq`.
/// 6. Nonlinear inequalities.
///
/// Every block keeps its original scale; no normalization or relaxation is
/// applied. This preserves the full feasible set, while COBYLA retains its
/// existing interpolation algorithm. Bounds do not guarantee feasible callback
/// points, and radius convergence does not certify constraint feasibility.
/// Callback errors propagate unchanged, and non-finite evaluated residuals
/// retain COBYLA's existing handling.
///
/// The adapter implements [`CostFunction`], but does not implement the
/// single-kind [`NonlinearInequalityConstraints`](super::NonlinearInequalityConstraints)
/// trait. COBYLA consumes the full form directly, without constructing a
/// parameter-backend vector large enough for every folded constraint.
///
/// # Backends
///
/// COBYLA accepts `Vec<f64>` with [`DenseMatrix`](crate::DenseMatrix),
/// `nalgebra::DVector<f64>` with `DMatrix<f64>`, `ndarray::Array1<f64>` with
/// `Array2<f64>`, and `faer::Col<f64>` with `Mat<f64>`. The corresponding `f32`
/// types are also supported. Optional backends require their backend features.
///
/// # Examples
///
/// Minimize a quadratic on the interval `[1, 2]`. The nonlinear block is empty.
///
/// ```
/// use basin::{Cobyla, CostFunction, DenseMatrix, Executor, FoldedConstraints,
///             NonlinearConstraints, State};
///
/// struct BoundedQuadratic { lower: Vec<f64>, upper: Vec<f64> }
/// impl CostFunction for BoundedQuadratic {
///     type Param = Vec<f64>;
///     type Output = f64;
///     type Error = std::convert::Infallible;
///     fn cost(&self, x: &Vec<f64>) -> Result<f64, Self::Error> {
///         Ok(x[0] * x[0])
///     }
/// }
/// impl NonlinearConstraints for BoundedQuadratic {
///     type Matrix = DenseMatrix;
///     fn nonlinear_constraints(&self, _: &Vec<f64>) -> Result<Vec<f64>, Self::Error> {
///         Ok(vec![])
///     }
///     fn num_nonlinear_constraints(&self) -> usize { 0 }
///     fn lower(&self) -> Option<&Vec<f64>> { Some(&self.lower) }
///     fn upper(&self) -> Option<&Vec<f64>> { Some(&self.upper) }
/// }
///
/// let problem = BoundedQuadratic { lower: vec![1.0], upper: vec![2.0] };
/// let result = Executor::from_start(
///     FoldedConstraints::new(problem), Cobyla::new(), vec![1.5],
/// ).run().unwrap();
/// assert!((result.state.param()[0] - 1.0).abs() < 1e-4);
/// ```
#[derive(Clone, Debug)]
pub struct FoldedConstraints<P> {
    problem: P,
}

impl<P> FoldedConstraints<P> {
    /// Own the problem and expose all of its constraint blocks to COBYLA.
    pub fn new(problem: P) -> Self {
        Self { problem }
    }

    /// Borrow the underlying problem.
    pub fn get_ref(&self) -> &P {
        &self.problem
    }

    /// Recover the underlying problem.
    pub fn into_inner(self) -> P {
        self.problem
    }
}

impl<P: CostFunction> CostFunction for FoldedConstraints<P> {
    type Param = P::Param;
    type Output = P::Output;
    type Error = P::Error;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Self::Error> {
        self.problem.cost(param)
    }
}

impl<P> FoldedConstraints<P>
where
    P: NonlinearConstraints,
    P::Output: Scalar,
    P::Param: VectorLen + Index<usize, Output = P::Output>,
    P::Matrix: MatVec<P::Param>,
{
    pub(crate) fn constraint_count(&self, n: usize) -> usize {
        let mut count = self.problem.num_nonlinear_constraints();
        for (bounds, name) in [
            (self.problem.lower(), "lower"),
            (self.problem.upper(), "upper"),
        ] {
            if let Some(bounds) = bounds {
                assert_eq!(
                    bounds.vec_len(),
                    n,
                    "FoldedConstraints: {name} bounds must match parameter length"
                );
                count += (0..n).filter(|&i| bounds[i].is_finite()).count();
            }
        }
        if let Some((_, b)) = self.problem.equalities() {
            count += 2 * b.vec_len();
        }
        if let Some((_, b)) = self.problem.inequalities() {
            count += b.vec_len();
        }
        count
    }

    pub(crate) fn evaluate_constraints(
        &self,
        x: &P::Param,
    ) -> Result<Vec<P::Output>, P::Error> {
        let n = x.vec_len();
        let mut constraints = Vec::with_capacity(self.constraint_count(n));
        if let Some(lower) = self.problem.lower() {
            for i in 0..n {
                if lower[i].is_finite() {
                    constraints.push(lower[i] - x[i]);
                }
            }
        }
        if let Some(upper) = self.problem.upper() {
            for i in 0..n {
                if upper[i].is_finite() {
                    constraints.push(x[i] - upper[i]);
                }
            }
        }
        if let Some((a, b)) = self.problem.equalities() {
            let ax = a.matvec(x);
            assert_eq!(
                ax.vec_len(),
                b.vec_len(),
                "FoldedConstraints: equality product and right-hand side must match"
            );
            constraints.extend((0..b.vec_len()).map(|i| b[i] - ax[i]));
            constraints.extend((0..b.vec_len()).map(|i| ax[i] - b[i]));
        }
        if let Some((a, b)) = self.problem.inequalities() {
            let ax = a.matvec(x);
            assert_eq!(
                ax.vec_len(),
                b.vec_len(),
                "FoldedConstraints: inequality product and right-hand side must match"
            );
            constraints.extend((0..b.vec_len()).map(|i| ax[i] - b[i]));
        }
        let nonlinear = self.problem.nonlinear_constraints(x)?;
        assert_eq!(
            nonlinear.vec_len(),
            self.problem.num_nonlinear_constraints(),
            "FoldedConstraints: nonlinear output must match declared count"
        );
        constraints.extend((0..nonlinear.vec_len()).map(|i| nonlinear[i]));
        Ok(constraints)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::DenseMatrix;

    #[derive(Default)]
    struct Blocks {
        lower: Option<Vec<f64>>,
        upper: Option<Vec<f64>>,
        equalities: Option<(DenseMatrix, Vec<f64>)>,
        inequalities: Option<(DenseMatrix, Vec<f64>)>,
        nonlinear: Vec<f64>,
        nonlinear_count: usize,
        calls: Cell<usize>,
        error: Option<&'static str>,
    }

    impl CostFunction for Blocks {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = &'static str;

        fn cost(&self, x: &Self::Param) -> Result<f64, Self::Error> {
            Ok(x.iter().sum())
        }
    }

    impl NonlinearConstraints for Blocks {
        type Matrix = DenseMatrix;

        fn nonlinear_constraints(
            &self,
            _: &Self::Param,
        ) -> Result<Self::Param, Self::Error> {
            self.calls.set(self.calls.get() + 1);
            match self.error {
                Some(error) => Err(error),
                None => Ok(self.nonlinear.clone()),
            }
        }

        fn num_nonlinear_constraints(&self) -> usize {
            self.nonlinear_count
        }

        fn lower(&self) -> Option<&Self::Param> {
            self.lower.as_ref()
        }

        fn upper(&self) -> Option<&Self::Param> {
            self.upper.as_ref()
        }

        fn equalities(&self) -> Option<(&Self::Matrix, &Self::Param)> {
            self.equalities.as_ref().map(|(a, b)| (a, b))
        }

        fn inequalities(&self) -> Option<(&Self::Matrix, &Self::Param)> {
            self.inequalities.as_ref().map(|(a, b)| (a, b))
        }
    }

    #[test]
    fn full_form_preserves_block_order_signs_and_scale() {
        let folded = FoldedConstraints::new(Blocks {
            lower: Some(vec![0.0, f64::INFINITY, 4.0, f64::NAN]),
            upper: Some(vec![f64::INFINITY, 9.0, f64::NEG_INFINITY, f64::NAN]),
            equalities: Some((
                DenseMatrix::from_row_slice(
                    2,
                    4,
                    &[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, -1.0],
                ),
                vec![1.0, 10.0],
            )),
            inequalities: Some((
                DenseMatrix::from_row_slice(
                    3,
                    4,
                    &[
                        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                        2.0,
                    ],
                ),
                vec![-10.0, 0.0, 11.0],
            )),
            nonlinear: vec![-9.0],
            nonlinear_count: 1,
            ..Blocks::default()
        });
        assert_eq!(folded.constraint_count(4), 11);
        assert_eq!(
            folded.evaluate_constraints(&vec![2.0, 3.0, 5.0, 7.0]),
            Ok(vec![
                -2.0, -1.0, -6.0, -7.0, 2.0, 7.0, -2.0, 12.0, 8.0, 3.0, -9.0,
            ])
        );
        assert_eq!(folded.get_ref().calls.get(), 1);
    }

    #[test]
    fn absent_blocks_and_empty_nonlinear_still_evaluate_callback() {
        let folded = FoldedConstraints::new(Blocks::default());
        assert_eq!(folded.constraint_count(2), 0);
        assert_eq!(folded.evaluate_constraints(&vec![2.0, 3.0]), Ok(vec![]));
        assert_eq!(folded.get_ref().calls.get(), 1);
        assert_eq!(folded.cost(&vec![2.0, 3.0]), Ok(5.0));
        assert_eq!(folded.into_inner().calls.get(), 1);
    }

    #[test]
    fn empty_present_blocks_support_zero_rows_and_zero_dimensions() {
        for n in [0, 3] {
            let folded = FoldedConstraints::new(Blocks {
                lower: Some(vec![f64::NEG_INFINITY; n]),
                upper: Some(vec![f64::INFINITY; n]),
                equalities: Some((
                    DenseMatrix::from_row_slice(0, n, &[]),
                    vec![],
                )),
                inequalities: Some((
                    DenseMatrix::from_row_slice(0, n, &[]),
                    vec![],
                )),
                ..Blocks::default()
            });
            assert_eq!(folded.constraint_count(n), 0);
            assert_eq!(folded.evaluate_constraints(&vec![0.0; n]), Ok(vec![]));
        }
    }

    #[test]
    fn one_sided_bounds_and_all_nonfinite_bound_variants() {
        let bounds = vec![f64::NEG_INFINITY, f64::INFINITY, f64::NAN, 2.0];
        for lower in [true, false] {
            let mut blocks = Blocks::default();
            if lower {
                blocks.lower = Some(bounds.clone());
            } else {
                blocks.upper = Some(bounds.clone());
            }
            let folded = FoldedConstraints::new(blocks);
            assert_eq!(folded.constraint_count(4), 1);
            assert_eq!(
                folded.evaluate_constraints(&vec![0.0, 0.0, 0.0, 5.0]),
                Ok(vec![if lower { -3.0 } else { 3.0 }])
            );
        }
    }

    #[test]
    fn nonfinite_nonlinear_residuals_are_preserved() {
        let folded = FoldedConstraints::new(Blocks {
            nonlinear: vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
            nonlinear_count: 3,
            ..Blocks::default()
        });
        let values = folded.evaluate_constraints(&vec![0.0]).unwrap();
        assert!(values[0].is_nan());
        assert_eq!(&values[1..], &[f64::INFINITY, f64::NEG_INFINITY]);
    }

    #[test]
    fn nonfinite_linear_residuals_are_preserved() {
        let folded = FoldedConstraints::new(Blocks {
            equalities: Some((
                DenseMatrix::from_row_slice(1, 1, &[1.0]),
                vec![f64::INFINITY],
            )),
            inequalities: Some((
                DenseMatrix::from_row_slice(1, 1, &[f64::NAN]),
                vec![0.0],
            )),
            ..Blocks::default()
        });
        let values = folded.evaluate_constraints(&vec![0.0]).unwrap();
        assert_eq!(&values[..2], &[f64::INFINITY, f64::NEG_INFINITY]);
        assert!(values[2].is_nan());
    }

    #[test]
    fn empty_nonlinear_callback_errors_propagate() {
        let folded = FoldedConstraints::new(Blocks {
            error: Some("constraint failure"),
            ..Blocks::default()
        });
        assert_eq!(
            folded.evaluate_constraints(&vec![0.0]),
            Err("constraint failure")
        );
        assert_eq!(folded.get_ref().calls.get(), 1);
    }

    #[test]
    #[should_panic(expected = "lower bounds must match parameter length")]
    fn malformed_lower_bounds_panic() {
        FoldedConstraints::new(Blocks {
            lower: Some(vec![0.0, 1.0]),
            ..Blocks::default()
        })
        .evaluate_constraints(&vec![0.0])
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "upper bounds must match parameter length")]
    fn malformed_upper_bounds_panic() {
        FoldedConstraints::new(Blocks {
            upper: Some(vec![]),
            ..Blocks::default()
        })
        .evaluate_constraints(&vec![0.0])
        .unwrap();
    }

    #[test]
    #[should_panic(
        expected = "equality product and right-hand side must match"
    )]
    fn malformed_equality_rhs_panics() {
        FoldedConstraints::new(Blocks {
            equalities: Some((
                DenseMatrix::from_row_slice(1, 2, &[1.0, 2.0]),
                vec![1.0, 2.0],
            )),
            ..Blocks::default()
        })
        .evaluate_constraints(&vec![0.0, 0.0])
        .unwrap();
    }

    #[test]
    #[should_panic(
        expected = "inequality product and right-hand side must match"
    )]
    fn malformed_inequality_rhs_panics() {
        FoldedConstraints::new(Blocks {
            inequalities: Some((
                DenseMatrix::from_row_slice(1, 2, &[1.0, 2.0]),
                vec![],
            )),
            ..Blocks::default()
        })
        .evaluate_constraints(&vec![0.0, 0.0])
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "nonlinear output must match declared count")]
    fn malformed_nonlinear_output_panics() {
        FoldedConstraints::new(Blocks {
            nonlinear: vec![1.0],
            ..Blocks::default()
        })
        .evaluate_constraints(&vec![0.0])
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "nonlinear output must match declared count")]
    fn malformed_short_nonlinear_output_panics() {
        FoldedConstraints::new(Blocks {
            nonlinear: vec![1.0],
            nonlinear_count: 2,
            ..Blocks::default()
        })
        .evaluate_constraints(&vec![0.0])
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "matrix has 2 columns")]
    fn malformed_matrix_columns_follow_matvec_panic() {
        FoldedConstraints::new(Blocks {
            inequalities: Some((
                DenseMatrix::from_row_slice(1, 2, &[1.0, 2.0]),
                vec![1.0],
            )),
            ..Blocks::default()
        })
        .evaluate_constraints(&vec![0.0])
        .unwrap();
    }
}
