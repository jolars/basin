//! Shared checked block evaluation. Absent blocks require no math capabilities.
use std::ops::{Index, IndexMut};

use super::*;
use crate::core::math::{
    MatTransposeVecInto, MatVecInto, MatrixFinite, MatrixIndex, MatrixShape,
    Scalar, VectorLen,
};
use crate::core::problem::{CostFunction, Problem};

type NonlinearResult<F, E> = Result<(Vec<F>, Vec<F>), ConstraintSolveError<E>>;

pub(crate) fn invalid<E>(e: ConstraintError) -> ConstraintSolveError<E> {
    ConstraintSolveError::InvalidConstraints(e)
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct SetLayout {
    pub n: usize,
    pub lower: Vec<usize>,
    pub upper: Vec<usize>,
    pub linear_eq: usize,
    pub linear_ineq: usize,
    pub nonlinear_eq: usize,
    pub nonlinear_ineq: usize,
}
impl SetLayout {
    pub fn folded_count(&self) -> Result<usize, ConstraintError> {
        self.lower
            .len()
            .checked_add(self.upper.len())
            .and_then(|m| {
                self.linear_eq
                    .checked_add(self.nonlinear_eq)
                    .and_then(|e| e.checked_mul(2))
                    .and_then(|e| m.checked_add(e))
            })
            .and_then(|m| m.checked_add(self.linear_ineq))
            .and_then(|m| m.checked_add(self.nonlinear_ineq))
            .ok_or(ConstraintError::CountOverflow)
    }
}
pub(crate) struct BoundsData<F> {
    pub lower: Option<Vec<F>>,
    pub upper: Option<Vec<F>>,
}
pub(crate) trait BoundsBlock<V, F: Scalar>: BorrowConstraint {
    fn data(
        view: Self::Borrowed<'_>,
        n: usize,
    ) -> Result<BoundsData<F>, ConstraintError>;
}
impl<V, F: Scalar> BoundsBlock<V, F> for NoBounds {
    fn data(_: Self, _: usize) -> Result<BoundsData<F>, ConstraintError> {
        Ok(BoundsData {
            lower: None,
            upper: None,
        })
    }
}
impl<V, F> BoundsBlock<V, F> for BoxBounds<V>
where
    V: VectorLen + Index<usize, Output = F>,
    F: Scalar,
{
    fn data(
        view: Self::Borrowed<'_>,
        n: usize,
    ) -> Result<BoundsData<F>, ConstraintError> {
        let lo = view.lower().copied();
        let hi = view.upper().copied();
        for (values, side) in [(lo, BoundSide::Lower), (hi, BoundSide::Upper)] {
            if let Some(values) = values {
                dimension(
                    ConstraintBlock::Bounds,
                    ConstraintAxis::Length,
                    n,
                    values.vec_len(),
                )?;
                for i in 0..n {
                    validate_bound(values[i], side, i)?;
                }
            }
        }
        for i in 0..n {
            if lo.map_or(F::neg_infinity(), |v| v[i])
                > hi.map_or(F::infinity(), |v| v[i])
            {
                return Err(ConstraintError::InvertedBounds { coordinate: i });
            }
        }
        Ok(BoundsData {
            lower: lo.map(|v| (0..n).map(|i| v[i]).collect()),
            upper: hi.map(|v| (0..n).map(|i| v[i]).collect()),
        })
    }
}

pub(crate) trait LinearValues<V, F: Scalar>: BorrowConstraint {
    fn rows(
        view: Self::Borrowed<'_>,
        n: usize,
        block: ConstraintBlock,
    ) -> Result<usize, ConstraintError>;
    fn values(
        view: Self::Borrowed<'_>,
        x: &V,
        block: ConstraintBlock,
    ) -> Result<Vec<F>, ConstraintError>;
}
impl<V, F: Scalar> LinearValues<V, F> for NoLinearBlock {
    fn rows(
        _: Self,
        _: usize,
        _: ConstraintBlock,
    ) -> Result<usize, ConstraintError> {
        Ok(0)
    }
    fn values(
        _: Self,
        _: &V,
        _: ConstraintBlock,
    ) -> Result<Vec<F>, ConstraintError> {
        Ok(vec![])
    }
}
fn linear_shape<M, R, F>(
    matrix: &M,
    rhs: &R,
    n: usize,
    block: ConstraintBlock,
) -> Result<usize, ConstraintError>
where
    M: MatrixShape + MatrixFinite,
    R: VectorLen + Index<usize, Output = F>,
    F: Scalar,
{
    dimension(
        block,
        ConstraintAxis::Rows,
        rhs.vec_len(),
        matrix.matrix_rows(),
    )?;
    dimension(block, ConstraintAxis::Columns, n, matrix.matrix_cols())?;
    if !matrix.matrix_is_finite()
        || (0..rhs.vec_len()).any(|i| !rhs[i].is_finite())
    {
        return Err(ConstraintError::NonFiniteLinearData { block });
    }
    Ok(rhs.vec_len())
}
impl<V, M, R, F> LinearValues<V, F> for LinearBlock<M, R>
where
    M: MatrixShape + MatrixFinite + MatVecInto<V, F>,
    R: VectorLen + Index<usize, Output = F>,
    V: VectorLen,
    F: Scalar,
{
    fn rows(
        view: Self::Borrowed<'_>,
        n: usize,
        block: ConstraintBlock,
    ) -> Result<usize, ConstraintError> {
        match view.matrix.zip(view.rhs) {
            Some((a, b)) => linear_shape(a, b, n, block),
            None => Ok(0),
        }
    }
    fn values(
        view: Self::Borrowed<'_>,
        x: &V,
        block: ConstraintBlock,
    ) -> Result<Vec<F>, ConstraintError> {
        let Some((a, b)) = view.matrix.zip(view.rhs) else {
            return Ok(vec![]);
        };
        let m = linear_shape(a, b, x.vec_len(), block)?;
        let mut out = vec![F::zero(); m];
        a.matvec_into(x, &mut out);
        for i in 0..m {
            out[i] = out[i] - b[i];
        }
        Ok(out)
    }
}
pub(crate) trait LinearRows<V, F: Scalar>: BorrowConstraint {
    fn rows(
        view: Self::Borrowed<'_>,
        x: &V,
        block: ConstraintBlock,
    ) -> Result<Vec<(Vec<F>, F)>, ConstraintError>;
}
impl<V, F: Scalar> LinearRows<V, F> for NoLinearBlock {
    fn rows(
        _: Self,
        _: &V,
        _: ConstraintBlock,
    ) -> Result<Vec<(Vec<F>, F)>, ConstraintError> {
        Ok(vec![])
    }
}
impl<V, M, R, F> LinearRows<V, F> for LinearBlock<M, R>
where
    M: MatrixShape + MatrixFinite + MatTransposeVecInto<V, F>,
    R: VectorLen + Index<usize, Output = F>,
    V: Clone + VectorLen + IndexMut<usize, Output = F>,
    F: Scalar,
{
    fn rows(
        view: Self::Borrowed<'_>,
        x: &V,
        block: ConstraintBlock,
    ) -> Result<Vec<(Vec<F>, F)>, ConstraintError> {
        let Some((a, b)) = view.matrix.zip(view.rhs) else {
            return Ok(vec![]);
        };
        let m = linear_shape(a, b, x.vec_len(), block)?;
        let mut result = Vec::with_capacity(m);
        let mut unit = vec![F::zero(); m];
        let mut normal = x.clone();
        for j in 0..m {
            unit[j] = F::one();
            a.mat_transpose_vec_into(&unit, &mut normal);
            result.push(((0..x.vec_len()).map(|i| normal[i]).collect(), b[j]));
            unit[j] = F::zero();
        }
        Ok(result)
    }
}
pub(crate) trait IndexedRows<F: Scalar>: BorrowConstraint {
    fn rows(
        view: Self::Borrowed<'_>,
        n: usize,
        block: ConstraintBlock,
    ) -> Result<Vec<(Vec<F>, F)>, ConstraintError>;
}
impl<F: Scalar> IndexedRows<F> for NoLinearBlock {
    fn rows(
        _: Self,
        _: usize,
        _: ConstraintBlock,
    ) -> Result<Vec<(Vec<F>, F)>, ConstraintError> {
        Ok(vec![])
    }
}
impl<M, R, F> IndexedRows<F> for LinearBlock<M, R>
where
    M: MatrixIndex<F> + MatrixFinite,
    R: VectorLen + Index<usize, Output = F>,
    F: Scalar,
{
    fn rows(
        view: Self::Borrowed<'_>,
        n: usize,
        block: ConstraintBlock,
    ) -> Result<Vec<(Vec<F>, F)>, ConstraintError> {
        let Some((a, b)) = view.matrix.zip(view.rhs) else {
            return Ok(vec![]);
        };
        let m = linear_shape(a, b, n, block)?;
        Ok((0..m)
            .map(|i| ((0..n).map(|j| a.matrix_entry(i, j)).collect(), b[i]))
            .collect())
    }
}

pub(crate) trait EqualityValues<
    P: CostFunction<Param = V, Output = F>,
    V,
    F: Scalar,
>: BorrowConstraint
{
    fn write(
        view: Self::Borrowed<'_>,
        p: &P,
        x: &V,
        counts: &mut crate::EvalCounts,
        output: &mut [F],
    ) -> Result<(), ConstraintSolveError<P::Error>>;
    fn count(view: Self::Borrowed<'_>) -> usize;
    fn values(
        view: Self::Borrowed<'_>,
        p: &P,
        x: &V,
        counts: &mut crate::EvalCounts,
    ) -> Result<Vec<F>, ConstraintSolveError<P::Error>>;
}
pub(crate) trait InequalityValues<
    P: CostFunction<Param = V, Output = F>,
    V,
    F: Scalar,
>: BorrowConstraint
{
    fn write(
        view: Self::Borrowed<'_>,
        p: &P,
        x: &V,
        counts: &mut crate::EvalCounts,
        output: &mut [F],
    ) -> Result<(), ConstraintSolveError<P::Error>>;
    fn count(view: Self::Borrowed<'_>) -> usize;
    fn values(
        view: Self::Borrowed<'_>,
        p: &P,
        x: &V,
        counts: &mut crate::EvalCounts,
    ) -> Result<Vec<F>, ConstraintSolveError<P::Error>>;
}
macro_rules! nonlinear_values {
    ($helper:ident,$callback:ident,$method:ident,$block:ident) => {
        impl<P, V, F> $helper<P, V, F> for NoNonlinearBlock
        where
            P: CostFunction<Param = V, Output = F>,
            F: Scalar,
        {
            fn count(_: Self) -> usize {
                0
            }
            fn values(
                _: Self,
                _: &P,
                _: &V,
                _: &mut crate::EvalCounts,
            ) -> Result<Vec<F>, ConstraintSolveError<P::Error>> {
                Ok(vec![])
            }
            fn write(
                _: Self,
                _: &P,
                _: &V,
                _: &mut crate::EvalCounts,
                _: &mut [F],
            ) -> Result<(), ConstraintSolveError<P::Error>> {
                Ok(())
            }
        }
        impl<P, V, F> $helper<P, V, F> for NonlinearBlock
        where
            P: CostFunction<Param = V, Output = F>
                + $callback<Param = V, Error = <P as CostFunction>::Error>,
            <P as $callback>::Values:
                VectorLen + Index<usize, Output = F>,
            F: Scalar,
        {
            fn count(view: Self) -> usize {
                view.count
            }
            fn values(
                view: Self,
                p: &P,
                x: &V,
                counts: &mut crate::EvalCounts,
            ) -> Result<Vec<F>, ConstraintSolveError<<P as CostFunction>::Error>>
            {
                if view.count == 0 {
                    return Ok(vec![]);
                }
                counts.residual_evals += 1;
                let values = <P as $callback>::$method(p, x)?;
                dimension(
                    ConstraintBlock::$block,
                    ConstraintAxis::Length,
                    view.count,
                    values.vec_len(),
                )
                .map_err(invalid)?;
                Ok((0..values.vec_len()).map(|i| values[i]).collect())
            }
            fn write(
                view: Self,
                p: &P,
                x: &V,
                counts: &mut crate::EvalCounts,
                output: &mut [F],
            ) -> Result<(), ConstraintSolveError<<P as CostFunction>::Error>>
            {
                if view.count != output.len() {
                    return Err(invalid(ConstraintError::LayoutChanged));
                }
                if view.count == 0 {
                    return Ok(());
                }
                counts.residual_evals += 1;
                let values = <P as $callback>::$method(p, x)?;
                dimension(
                    ConstraintBlock::$block,
                    ConstraintAxis::Length,
                    view.count,
                    values.vec_len(),
                )
                .map_err(invalid)?;
                for (i, out) in output.iter_mut().enumerate() {
                    *out = values[i];
                }
                Ok(())
            }
        }
    }
}
nonlinear_values!(
    EqualityValues,
    NonlinearEqualityValues,
    nonlinear_equalities,
    NonlinearEqualities
);
nonlinear_values!(
    InequalityValues,
    NonlinearInequalityValues,
    nonlinear_inequalities,
    NonlinearInequalities
);

pub(crate) trait CobylaSet<P: CostFunction<Param = V, Output = F>, V, F: Scalar>:
    ConstraintDescription
{
    fn layout(
        view: Self::Borrowed<'_>,
        n: usize,
    ) -> Result<SetLayout, ConstraintError>;
    fn values(
        p: &mut Problem<P>,
        x: &V,
        layout: &SetLayout,
    ) -> Result<Vec<F>, ConstraintSolveError<P::Error>>
    where
        P: Constraints<Set = Self>;
}
impl<P, V, F, B, LE, LI, NE, NI> CobylaSet<P, V, F>
    for ConstraintSet<B, LinearBlocks<LE, LI>, NonlinearBlocks<NE, NI>>
where
    P: CostFunction<Param = V, Output = F>,
    V: VectorLen + Index<usize, Output = F>,
    F: Scalar,
    B: BoundsBlock<V, F>,
    LE: LinearValues<V, F>,
    LI: LinearValues<V, F>,
    NE: EqualityValues<P, V, F>,
    NI: InequalityValues<P, V, F>,
{
    fn layout(
        view: Self::Borrowed<'_>,
        n: usize,
    ) -> Result<SetLayout, ConstraintError> {
        let bounds = B::data(view.bounds, n)?;
        let lower = bounds.lower.map_or_else(Vec::new, |v| {
            (0..n).filter(|&i| v[i].is_finite()).collect()
        });
        let upper = bounds.upper.map_or_else(Vec::new, |v| {
            (0..n).filter(|&i| v[i].is_finite()).collect()
        });
        let result = SetLayout {
            n,
            lower,
            upper,
            linear_eq: LE::rows(
                view.linear.equalities,
                n,
                ConstraintBlock::LinearEqualities,
            )?,
            linear_ineq: LI::rows(
                view.linear.inequalities,
                n,
                ConstraintBlock::LinearInequalities,
            )?,
            nonlinear_eq: NE::count(view.nonlinear.equalities),
            nonlinear_ineq: NI::count(view.nonlinear.inequalities),
        };
        result.folded_count()?;
        Ok(result)
    }
    fn values(
        p: &mut Problem<P>,
        x: &V,
        layout: &SetLayout,
    ) -> Result<Vec<F>, ConstraintSolveError<P::Error>>
    where
        P: Constraints<Set = Self>,
    {
        let now = Self::layout(p.inner().constraints(), x.vec_len())
            .map_err(invalid)?;
        if &now != layout {
            return Err(invalid(ConstraintError::LayoutChanged));
        }
        if layout.folded_count().map_err(invalid)? == layout.nonlinear_ineq {
            // Reuse the inequality buffer when no other rows need folding.
            let mut counts = crate::EvalCounts::default();
            let result = NI::values(
                p.inner().constraints().nonlinear.inequalities,
                p.inner(),
                x,
                &mut counts,
            );
            p.counts_mut().add(&counts);
            return result;
        }
        let bounds = {
            let view = p.inner().constraints();
            B::data(view.bounds, x.vec_len()).map_err(invalid)?
        };
        let mut out =
            Vec::with_capacity(layout.folded_count().map_err(invalid)?);
        if let Some(lower) = bounds.lower {
            out.extend(layout.lower.iter().map(|&i| lower[i] - x[i]));
        }
        if let Some(upper) = bounds.upper {
            out.extend(layout.upper.iter().map(|&i| x[i] - upper[i]));
        }
        let eq = LE::values(
            p.inner().constraints().linear.equalities,
            x,
            ConstraintBlock::LinearEqualities,
        )
        .map_err(invalid)?;
        out.extend(eq.iter().map(|&v| -v));
        out.extend(eq);
        out.extend(
            LI::values(
                p.inner().constraints().linear.inequalities,
                x,
                ConstraintBlock::LinearInequalities,
            )
            .map_err(invalid)?,
        );
        let mut counts = crate::EvalCounts::default();
        let result: NonlinearResult<F, P::Error> = (|| {
            let view = p.inner().constraints();
            let ni = NI::values(
                view.nonlinear.inequalities,
                p.inner(),
                x,
                &mut counts,
            )?;
            let view = p.inner().constraints();
            let eq = NE::values(
                view.nonlinear.equalities,
                p.inner(),
                x,
                &mut counts,
            )?;
            Ok((ni, eq))
        })();
        p.counts_mut().add(&counts);
        let (ni, eq): (Vec<F>, Vec<F>) = result?;
        out.extend(ni);
        out.extend(eq.iter().copied());
        out.extend(eq.iter().map(|&v| -v));
        Ok(out)
    }
}
pub(crate) trait BoxOnly<V>: ConstraintDescription {
    fn bounds(view: Self::Borrowed<'_>) -> Result<(&V, &V), ConstraintError>;
}
impl<V> BoxOnly<V> for ConstraintSet<BoxBounds<V>> {
    fn bounds(view: Self::Borrowed<'_>) -> Result<(&V, &V), ConstraintError> {
        Ok((
            view.bounds.lower().copied().ok_or(
                ConstraintError::MissingBound {
                    side: BoundSide::Lower,
                },
            )?,
            view.bounds.upper().copied().ok_or(
                ConstraintError::MissingBound {
                    side: BoundSide::Upper,
                },
            )?,
        ))
    }
}
/// Read access used internally by algorithms that support exactly box constraints.
pub(crate) trait BoxProblem: Constraints {
    type Param;
    fn box_bounds(
        &self,
    ) -> Result<(&Self::Param, &Self::Param), ConstraintError>;
    fn lower(&self) -> &Self::Param {
        self.box_bounds().expect("validated box bounds").0
    }
    fn upper(&self) -> &Self::Param {
        self.box_bounds().expect("validated box bounds").1
    }
}
impl<P, V> BoxProblem for P
where
    P: Constraints<Set = ConstraintSet<BoxBounds<V>>>,
{
    type Param = V;
    fn box_bounds(&self) -> Result<(&V, &V), ConstraintError> {
        <ConstraintSet<BoxBounds<V>> as BoxOnly<V>>::bounds(self.constraints())
    }
}

pub(crate) trait LinearInequalityProblem<V, F: Scalar>:
    CostFunction<Param = V, Output = F>
{
    fn linear_count(&self) -> usize;
    fn validate_linear(&self, n: usize) -> Result<(), ConstraintError>;
    fn linear_residual(&self, x: &V) -> Result<Vec<F>, ConstraintError>;
}
pub(crate) trait LinearEqualityProblem<V, F: Scalar>:
    CostFunction<Param = V, Output = F>
{
    fn linear_count(&self) -> usize;
    fn validate_linear(&self, n: usize) -> Result<(), ConstraintError>;
    fn linear_residual(&self, x: &V) -> Result<Vec<F>, ConstraintError>;
}
pub(crate) trait LinearInequalityGradient<V, F: Scalar>:
    LinearInequalityProblem<V, F>
{
    fn linear_transpose(
        &self,
        weights: &[F],
        out: &mut V,
    ) -> Result<(), ConstraintError>;
}
pub(crate) trait LinearEqualityGradient<V, F: Scalar>:
    LinearEqualityProblem<V, F>
{
    fn linear_transpose(
        &self,
        weights: &[F],
        out: &mut V,
    ) -> Result<(), ConstraintError>;
}
macro_rules! single_linear {
    ($problem:ident,$gradient:ident,$eq:ty,$ineq:ty,$field:ident,$kind:ident) => {
        impl<P, V, M, R, F> $problem<V, F> for P
        where
            P: CostFunction<Param = V, Output = F>,
            P: Constraints<
                Set = ConstraintSet<NoBounds, LinearBlocks<$eq, $ineq>>,
            >,
            V: VectorLen,
            F: Scalar,
            M: MatrixShape + MatrixFinite + MatVecInto<V, F>,
            R: VectorLen + Index<usize, Output = F>,
        {
            fn linear_count(&self) -> usize {
                self.constraints()
                    .linear
                    .$field
                    .rhs
                    .map_or(0, VectorLen::vec_len)
            }
            fn validate_linear(&self, n: usize) -> Result<(), ConstraintError> {
                <LinearBlock<M, R> as LinearValues<V, F>>::rows(
                    self.constraints().linear.$field,
                    n,
                    ConstraintBlock::$kind,
                )
                .map(|_| ())
            }
            fn linear_residual(
                &self,
                x: &V,
            ) -> Result<Vec<F>, ConstraintError> {
                <LinearBlock<M, R> as LinearValues<V, F>>::values(
                    self.constraints().linear.$field,
                    x,
                    ConstraintBlock::$kind,
                )
            }
        }
        impl<P, V, M, R, F> $gradient<V, F> for P
        where
            P: CostFunction<Param = V, Output = F>,
            P: Constraints<
                Set = ConstraintSet<NoBounds, LinearBlocks<$eq, $ineq>>,
            >,
            V: VectorLen + IndexMut<usize, Output = F>,
            F: Scalar,
            M: MatrixShape
                + MatrixFinite
                + MatVecInto<V, F>
                + MatTransposeVecInto<V, F>,
            R: VectorLen + Index<usize, Output = F>,
        {
            fn linear_transpose(
                &self,
                weights: &[F],
                out: &mut V,
            ) -> Result<(), ConstraintError> {
                self.validate_linear(out.vec_len())?;
                let view = self.constraints();
                if let Some(a) = view.linear.$field.matrix {
                    dimension(
                        ConstraintBlock::$kind,
                        ConstraintAxis::Rows,
                        a.matrix_rows(),
                        weights.len(),
                    )?;
                    a.mat_transpose_vec_into(weights, out);
                } else {
                    dimension(
                        ConstraintBlock::$kind,
                        ConstraintAxis::Rows,
                        0,
                        weights.len(),
                    )?;
                    for i in 0..out.vec_len() {
                        out[i] = F::zero();
                    }
                }
                Ok(())
            }
        }
    };
}
single_linear!(LinearInequalityProblem,LinearInequalityGradient,NoLinearBlock,LinearBlock<M,R>,inequalities,LinearInequalities);
single_linear!(LinearEqualityProblem,LinearEqualityGradient,LinearBlock<M,R>,NoLinearBlock,equalities,LinearEqualities);

pub(crate) fn validate_box_problem<P, V, F>(
    p: &P,
    n: Option<usize>,
    finite: bool,
) -> Result<(), ConstraintError>
where
    P: BoxProblem<Param = V>,
    V: VectorLen + Index<usize, Output = F>,
    F: Scalar,
{
    let (lower, upper) = p.box_bounds()?;
    let n = n.unwrap_or_else(|| lower.vec_len());
    for (values, side) in [(lower, BoundSide::Lower), (upper, BoundSide::Upper)]
    {
        dimension(
            ConstraintBlock::Bounds,
            ConstraintAxis::Length,
            n,
            values.vec_len(),
        )?;
        for i in 0..n {
            validate_bound(values[i], side, i)?;
        }
    }
    for i in 0..n {
        if lower[i] > upper[i] {
            return Err(ConstraintError::InvertedBounds { coordinate: i });
        }
        if finite && (!lower[i].is_finite() || !upper[i].is_finite()) {
            return Err(ConstraintError::NonFiniteInterval { coordinate: i });
        }
    }
    Ok(())
}
pub(crate) fn validate_scalar_box_problem<P, F>(
    p: &P,
) -> Result<(), ConstraintError>
where
    P: BoxProblem<Param = F>,
    F: Scalar,
{
    let (lower, upper) = p.box_bounds()?;
    validate_bound(*lower, BoundSide::Lower, 0)?;
    validate_bound(*upper, BoundSide::Upper, 0)?;
    if lower > upper {
        return Err(ConstraintError::InvertedBounds { coordinate: 0 });
    }
    if !lower.is_finite() || !upper.is_finite() {
        return Err(ConstraintError::NonFiniteInterval { coordinate: 0 });
    }
    Ok(())
}

pub(crate) trait NonlinearSet<
    P: CostFunction<Param = V, Output = F>,
    V,
    F: Scalar,
>: ConstraintDescription
{
    fn counts(view: Self::Borrowed<'_>) -> (usize, usize);
    fn values(
        p: &P,
        x: &V,
        counts: &mut crate::EvalCounts,
    ) -> NonlinearResult<F, P::Error>
    where
        P: Constraints<Set = Self>;
    fn write_values(
        p: &P,
        x: &V,
        counts: &mut crate::EvalCounts,
        equalities: &mut [F],
        inequalities: &mut [F],
    ) -> Result<(), ConstraintSolveError<P::Error>>
    where
        P: Constraints<Set = Self>;
}
impl<P, V, F, B, L, NE, NI> NonlinearSet<P, V, F>
    for ConstraintSet<B, L, NonlinearBlocks<NE, NI>>
where
    P: CostFunction<Param = V, Output = F>,
    F: Scalar,
    B: BorrowConstraint,
    L: BorrowConstraint,
    NE: EqualityValues<P, V, F>,
    NI: InequalityValues<P, V, F>,
{
    fn counts(view: Self::Borrowed<'_>) -> (usize, usize) {
        (
            NE::count(view.nonlinear.equalities),
            NI::count(view.nonlinear.inequalities),
        )
    }
    fn values(
        p: &P,
        x: &V,
        counts: &mut crate::EvalCounts,
    ) -> NonlinearResult<F, P::Error>
    where
        P: Constraints<Set = Self>,
    {
        let eq =
            NE::values(p.constraints().nonlinear.equalities, p, x, counts)?;
        let ni =
            NI::values(p.constraints().nonlinear.inequalities, p, x, counts)?;
        Ok((eq, ni))
    }
    fn write_values(
        p: &P,
        x: &V,
        counts: &mut crate::EvalCounts,
        equalities: &mut [F],
        inequalities: &mut [F],
    ) -> Result<(), ConstraintSolveError<P::Error>>
    where
        P: Constraints<Set = Self>,
    {
        NE::write(
            p.constraints().nonlinear.equalities,
            p,
            x,
            counts,
            equalities,
        )?;
        NI::write(
            p.constraints().nonlinear.inequalities,
            p,
            x,
            counts,
            inequalities,
        )
    }
}

pub(crate) trait LincoaSet<V, F: Scalar>: ConstraintDescription {
    fn data(
        view: Self::Borrowed<'_>,
        x: &V,
    ) -> Result<LinearData<F>, ConstraintError>;
}
pub(crate) struct LinearData<F> {
    pub bounds: BoundsData<F>,
    pub equalities: Vec<(Vec<F>, F)>,
    pub inequalities: Vec<(Vec<F>, F)>,
}
impl<V, F, B, LE, LI> LincoaSet<V, F> for ConstraintSet<B, LinearBlocks<LE, LI>>
where
    V: VectorLen,
    F: Scalar,
    B: BoundsBlock<V, F>,
    LE: LinearRows<V, F>,
    LI: LinearRows<V, F>,
{
    fn data(
        view: Self::Borrowed<'_>,
        x: &V,
    ) -> Result<LinearData<F>, ConstraintError> {
        Ok(LinearData {
            bounds: B::data(view.bounds, x.vec_len())?,
            equalities: LE::rows(
                view.linear.equalities,
                x,
                ConstraintBlock::LinearEqualities,
            )?,
            inequalities: LI::rows(
                view.linear.inequalities,
                x,
                ConstraintBlock::LinearInequalities,
            )?,
        })
    }
}

pub(crate) trait NonlinearDerivatives<
    P: CostFunction<Param = V, Output = F>,
    V,
    F: Scalar,
>: BorrowConstraint
{
    fn jacobian(
        p: &P,
        x: &V,
        n: usize,
        m: usize,
        counts: &mut crate::EvalCounts,
    ) -> Result<Vec<Vec<F>>, ConstraintSolveError<P::Error>>;
}
impl<P, V, F> NonlinearDerivatives<P, V, F> for NoNonlinearConstraints
where
    P: CostFunction<Param = V, Output = F>,
    F: Scalar,
{
    fn jacobian(
        _: &P,
        _: &V,
        _: usize,
        _: usize,
        _: &mut crate::EvalCounts,
    ) -> Result<Vec<Vec<F>>, ConstraintSolveError<P::Error>> {
        Ok(vec![])
    }
}
fn checked_jacobian<P, V, F>(
    p: &P,
    x: &V,
    n: usize,
    m: usize,
    counts: &mut crate::EvalCounts,
) -> Result<Vec<Vec<F>>, ConstraintSolveError<<P as CostFunction>::Error>>
where
    P: CostFunction<Param = V, Output = F>
        + ConstraintJacobian<Param = V, Error = <P as CostFunction>::Error>,
    <P as ConstraintJacobian>::Jacobian: MatrixIndex<F>,
    F: Scalar,
{
    if m == 0 {
        return Ok(vec![]);
    }
    counts.jacobian_evals += 1;
    let jac = <P as ConstraintJacobian>::constraint_jacobian(p, x)?;
    dimension(
        ConstraintBlock::Jacobian,
        ConstraintAxis::Rows,
        m,
        jac.matrix_rows(),
    )
    .map_err(invalid)?;
    dimension(
        ConstraintBlock::Jacobian,
        ConstraintAxis::Columns,
        n,
        jac.matrix_cols(),
    )
    .map_err(invalid)?;
    Ok((0..m)
        .map(|i| (0..n).map(|j| jac.matrix_entry(i, j)).collect())
        .collect())
}
macro_rules! nonlinear_derivatives {
    ($e:ty,$i:ty)=>{
        impl<P, V, F> NonlinearDerivatives<P, V, F>
            for NonlinearBlocks<$e, $i>
        where
            P: CostFunction<Param = V, Output = F>
                + ConstraintJacobian<Param = V, Error = <P as CostFunction>::Error>,
            <P as ConstraintJacobian>::Jacobian: MatrixIndex<F>,
            F: Scalar,
        {
            fn jacobian(
                p: &P,
                x: &V,
                n: usize,
                m: usize,
                counts: &mut crate::EvalCounts,
            ) -> Result<Vec<Vec<F>>, ConstraintSolveError<<P as CostFunction>::Error>>
            {
                checked_jacobian(p, x, n, m, counts)
            }
        }
    }
}
nonlinear_derivatives!(NonlinearBlock, NoNonlinearBlock);
nonlinear_derivatives!(NoNonlinearBlock, NonlinearBlock);
nonlinear_derivatives!(NonlinearBlock, NonlinearBlock);

pub(crate) trait SlsqpSet<P: CostFunction<Param = V, Output = F>, V, F: Scalar>:
    NonlinearSet<P, V, F>
{
    fn data(
        view: Self::Borrowed<'_>,
        n: usize,
    ) -> Result<LinearData<F>, ConstraintError>;
    fn jacobian(
        p: &P,
        x: &V,
        n: usize,
        m: usize,
        counts: &mut crate::EvalCounts,
    ) -> Result<Vec<Vec<F>>, ConstraintSolveError<P::Error>>;
}
impl<P, V, F, B, LE, LI, NE, NI> SlsqpSet<P, V, F>
    for ConstraintSet<B, LinearBlocks<LE, LI>, NonlinearBlocks<NE, NI>>
where
    P: CostFunction<Param = V, Output = F>,
    F: Scalar,
    B: BoundsBlock<V, F>,
    LE: IndexedRows<F>,
    LI: IndexedRows<F>,
    NE: EqualityValues<P, V, F>,
    NI: InequalityValues<P, V, F>,
    NonlinearBlocks<NE, NI>: NonlinearDerivatives<P, V, F>,
{
    fn data(
        view: Self::Borrowed<'_>,
        n: usize,
    ) -> Result<LinearData<F>, ConstraintError> {
        Ok(LinearData {
            bounds: B::data(view.bounds, n)?,
            equalities: LE::rows(
                view.linear.equalities,
                n,
                ConstraintBlock::LinearEqualities,
            )?,
            inequalities: LI::rows(
                view.linear.inequalities,
                n,
                ConstraintBlock::LinearInequalities,
            )?,
        })
    }
    fn jacobian(
        p: &P,
        x: &V,
        n: usize,
        m: usize,
        counts: &mut crate::EvalCounts,
    ) -> Result<Vec<Vec<F>>, ConstraintSolveError<P::Error>> {
        <NonlinearBlocks<NE, NI> as NonlinearDerivatives<P, V, F>>::jacobian(
            p, x, n, m, counts,
        )
    }
}

pub(crate) fn validate_bound_slices<F: Scalar>(
    lower: &[F],
    upper: &[F],
    n: usize,
    finite: bool,
) -> Result<(), ConstraintError> {
    for (values, side) in [(lower, BoundSide::Lower), (upper, BoundSide::Upper)]
    {
        dimension(
            ConstraintBlock::Bounds,
            ConstraintAxis::Length,
            n,
            values.len(),
        )?;
        for (i, &value) in values.iter().enumerate() {
            validate_bound(value, side, i)?;
        }
    }
    for i in 0..n {
        if lower[i] > upper[i] {
            return Err(ConstraintError::InvertedBounds { coordinate: i });
        }
        if finite && (!lower[i].is_finite() || !upper[i].is_finite()) {
            return Err(ConstraintError::NonFiniteInterval { coordinate: i });
        }
    }
    Ok(())
}

pub(crate) fn validate_coordinate_box_problem<P, V, F>(
    p: &P,
    n: usize,
    finite: bool,
) -> Result<(), ConstraintError>
where
    P: BoxProblem<Param = V>,
    V: VectorLen + crate::VectorIndex<F>,
    F: Scalar,
{
    let (lower, upper) = p.box_bounds()?;
    for (values, side) in [(lower, BoundSide::Lower), (upper, BoundSide::Upper)]
    {
        dimension(
            ConstraintBlock::Bounds,
            ConstraintAxis::Length,
            n,
            values.vec_len(),
        )?;
        for i in 0..n {
            validate_bound(values.get_scalar(i), side, i)?;
        }
    }
    for i in 0..n {
        if lower.get_scalar(i) > upper.get_scalar(i) {
            return Err(ConstraintError::InvertedBounds { coordinate: i });
        }
        if finite
            && (!lower.get_scalar(i).is_finite()
                || !upper.get_scalar(i).is_finite())
        {
            return Err(ConstraintError::NonFiniteInterval { coordinate: i });
        }
    }
    Ok(())
}
