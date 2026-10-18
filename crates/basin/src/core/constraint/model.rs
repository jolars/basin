use crate::core::math::Scalar;

#[cfg(test)]
#[path = "model/integer_domain_api_prototype.rs"]
mod integer_domain_api_prototype;

/// The authoritative constraint declaration shared by objective and residual problems.
///
/// Implement this even for an unconstrained problem, using [`NoConstraints`]
/// and returning `NoConstraints::EMPTY`. Adding derivatives preserves this
/// declaration; an explicit problem transformation may replace it.
///
/// # Contract
///
/// The getter must not evaluate the objective or constraints. Bound endpoints,
/// linear data, and nonlinear counts must remain fixed during a solve. Fresh
/// initialization may use new dimensions or data. Solvers validate the shapes
/// they consume and report a changed nonlinear count as [`ConstraintError::LayoutChanged`].
pub trait ProblemConstraints {
    /// The complete constraint set. Its type determines solver compatibility.
    type Constraints: ConstraintDescription;
    /// Borrow the complete set without evaluating the problem.
    fn constraint_set(
        &self,
    ) -> <Self::Constraints as ConstraintDescription>::Borrowed<'_>;
}

mod sealed {
    pub trait Set {}
}

/// A canonical constraint-set description. Implementations are sealed.
pub trait ConstraintDescription: sealed::Set {
    /// A view that borrows the block data without cloning it.
    type Borrowed<'a>
    where
        Self: 'a;
    /// Borrow every block in the set.
    fn borrowed(&self) -> Self::Borrowed<'_>;
}
/// Borrow a canonical block while preserving its static presence.
pub trait BorrowConstraint {
    /// Borrowed representation of this block.
    type Borrowed<'a>
    where
        Self: 'a;
    /// Borrow this block's data.
    fn borrowed(&self) -> Self::Borrowed<'_>;
}

/// A complete set of problem-side constraints.
///
/// Equality and inequality blocks remain distinct. Builders change the set's
/// type, so an unconstrained or box-only solver cannot ignore additional blocks.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Constraints<
    B = NoBounds,
    L = NoLinearConstraints,
    N = NoNonlinearConstraints,
> {
    pub(crate) bounds: B,
    pub(crate) linear: L,
    pub(crate) nonlinear: N,
}
impl<B, L, N> sealed::Set for Constraints<B, L, N> {}
impl<B: BorrowConstraint, L: BorrowConstraint, N: BorrowConstraint>
    ConstraintDescription for Constraints<B, L, N>
{
    type Borrowed<'a>
        = Constraints<B::Borrowed<'a>, L::Borrowed<'a>, N::Borrowed<'a>>
    where
        Self: 'a;
    fn borrowed(&self) -> Self::Borrowed<'_> {
        Constraints {
            bounds: self.bounds.borrowed(),
            linear: self.linear.borrowed(),
            nonlinear: self.nonlinear.borrowed(),
        }
    }
}

/// A problem with no mathematical constraints.
pub type NoConstraints = Constraints;
/// No parameter bounds.
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NoBounds;
/// An absent linear block, requiring no matrix type or operations.
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NoLinearBlock;
/// An absent nonlinear block, requiring no callback or Jacobian.
#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NoNonlinearBlock;
/// No linear equalities or inequalities.
pub type NoLinearConstraints = LinearBlocks<NoLinearBlock, NoLinearBlock>;
/// No nonlinear equalities or inequalities.
pub type NoNonlinearConstraints =
    NonlinearBlocks<NoNonlinearBlock, NoNonlinearBlock>;

/// Optional lower and upper bounds, represented in the parameter's type.
///
/// Only negative infinity denotes an absent lower bound, and only positive
/// infinity denotes an absent upper bound. NaN is invalid on either side.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BoxBounds<V> {
    lower: Option<V>,
    upper: Option<V>,
}
impl<V> BoxBounds<V> {
    /// Store bounds for validation against the parameter at initialization.
    pub fn new(lower: Option<V>, upper: Option<V>) -> Self {
        Self { lower, upper }
    }
    /// Borrow the lower bounds, when supplied.
    pub fn lower(&self) -> Option<&V> {
        self.lower.as_ref()
    }
    /// Borrow the upper bounds, when supplied.
    pub fn upper(&self) -> Option<&V> {
        self.upper.as_ref()
    }
}

/// One linear block. Its location in [`LinearBlocks`] determines equality or inequality semantics.
///
/// The matrix and right-hand side must either both be present or both be absent.
/// Deserialization rejects blocks with only one of them present.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct LinearBlock<M, R> {
    pub(crate) matrix: Option<M>,
    pub(crate) rhs: Option<R>,
}
impl<M, R> LinearBlock<M, R> {
    /// Store a matrix and right-hand side for checked initialization.
    pub fn new(matrix: M, rhs: R) -> Self {
        Self {
            matrix: Some(matrix),
            rhs: Some(rhs),
        }
    }
    /// Store an optional block while retaining its kind in the set type.
    pub fn optional(data: Option<(M, R)>) -> Self {
        match data {
            Some((m, r)) => Self::new(m, r),
            None => Self {
                matrix: None,
                rhs: None,
            },
        }
    }
    /// Borrow the constraint matrix.
    pub fn matrix(&self) -> Option<&M> {
        self.matrix.as_ref()
    }
    /// Borrow the right-hand side, independently of the parameter representation.
    pub fn rhs(&self) -> Option<&R> {
        self.rhs.as_ref()
    }
}

#[cfg(feature = "serde")]
impl<'de, M, R> serde::Deserialize<'de> for LinearBlock<M, R>
where
    M: serde::Deserialize<'de>,
    R: serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        #[serde(rename = "LinearBlock")]
        struct StoredLinearBlock<M, R> {
            matrix: Option<M>,
            rhs: Option<R>,
        }

        let stored = StoredLinearBlock::<M, R>::deserialize(deserializer)?;
        if stored.matrix.is_some() != stored.rhs.is_some() {
            return Err(serde::de::Error::custom(
                "linear block matrix and rhs must both be present or both be absent",
            ));
        }
        Ok(Self {
            matrix: stored.matrix,
            rhs: stored.rhs,
        })
    }
}

/// Distinct linear equality and inequality blocks.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinearBlocks<E = NoLinearBlock, I = NoLinearBlock> {
    pub(crate) equalities: E,
    pub(crate) inequalities: I,
}
impl<E, I> LinearBlocks<E, I> {
    /// Borrow the equality block.
    pub fn equalities(&self) -> &E {
        &self.equalities
    }
    /// Borrow the inequality block.
    pub fn inequalities(&self) -> &I {
        &self.inequalities
    }
}

/// A declared nonlinear block with a fixed callback output length.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NonlinearBlock {
    pub(crate) count: usize,
}
impl NonlinearBlock {
    /// The declared number of scalar constraint values.
    pub fn count(&self) -> usize {
        self.count
    }
}
/// Distinct nonlinear equality and inequality declarations.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NonlinearBlocks<E = NoNonlinearBlock, I = NoNonlinearBlock> {
    pub(crate) equalities: E,
    pub(crate) inequalities: I,
}
impl<E, I> NonlinearBlocks<E, I> {
    /// Borrow the equality declaration.
    pub fn equalities(&self) -> &E {
        &self.equalities
    }
    /// Borrow the inequality declaration.
    pub fn inequalities(&self) -> &I {
        &self.inequalities
    }
}

impl NoConstraints {
    /// A shared empty declaration for problems without stored constraints.
    pub const EMPTY: Self = Self {
        bounds: NoBounds,
        linear: LinearBlocks {
            equalities: NoLinearBlock,
            inequalities: NoLinearBlock,
        },
        nonlinear: NonlinearBlocks {
            equalities: NoNonlinearBlock,
            inequalities: NoNonlinearBlock,
        },
    };
    /// Begin an empty set and add its supported blocks with the builders.
    pub const fn new() -> Self {
        Self::EMPTY
    }
}
impl Default for NoConstraints {
    fn default() -> Self {
        Self::new()
    }
}
impl<B, L, N> Constraints<B, L, N> {
    /// Borrow the bounds block.
    pub fn bounds(&self) -> &B {
        &self.bounds
    }
    /// Borrow both linear blocks.
    pub fn linear(&self) -> &L {
        &self.linear
    }
    /// Borrow both nonlinear declarations.
    pub fn nonlinear(&self) -> &N {
        &self.nonlinear
    }
    /// Set the bounds without changing other blocks.
    pub fn with_bounds<V>(
        self,
        bounds: BoxBounds<V>,
    ) -> Constraints<BoxBounds<V>, L, N> {
        Constraints {
            bounds,
            linear: self.linear,
            nonlinear: self.nonlinear,
        }
    }
}
impl<B, E, I, N> Constraints<B, LinearBlocks<E, I>, N> {
    /// Set the linear equalities `A x = b`.
    pub fn with_linear_equalities<M, R>(
        self,
        block: LinearBlock<M, R>,
    ) -> Constraints<B, LinearBlocks<LinearBlock<M, R>, I>, N> {
        Constraints {
            bounds: self.bounds,
            linear: LinearBlocks {
                equalities: block,
                inequalities: self.linear.inequalities,
            },
            nonlinear: self.nonlinear,
        }
    }
    /// Set the linear inequalities `A x <= b`.
    pub fn with_linear_inequalities<M, R>(
        self,
        block: LinearBlock<M, R>,
    ) -> Constraints<B, LinearBlocks<E, LinearBlock<M, R>>, N> {
        Constraints {
            bounds: self.bounds,
            linear: LinearBlocks {
                equalities: self.linear.equalities,
                inequalities: block,
            },
            nonlinear: self.nonlinear,
        }
    }
}
impl<B, L, E, I> Constraints<B, L, NonlinearBlocks<E, I>> {
    /// Declare a nonlinear equality callback `h(x) = 0` with `count` values.
    pub fn with_nonlinear_equalities(
        self,
        count: usize,
    ) -> Constraints<B, L, NonlinearBlocks<NonlinearBlock, I>> {
        Constraints {
            bounds: self.bounds,
            linear: self.linear,
            nonlinear: NonlinearBlocks {
                equalities: NonlinearBlock { count },
                inequalities: self.nonlinear.inequalities,
            },
        }
    }
    /// Declare a nonlinear inequality callback `c(x) <= 0` with `count` values.
    pub fn with_nonlinear_inequalities(
        self,
        count: usize,
    ) -> Constraints<B, L, NonlinearBlocks<E, NonlinearBlock>> {
        Constraints {
            bounds: self.bounds,
            linear: self.linear,
            nonlinear: NonlinearBlocks {
                equalities: self.nonlinear.equalities,
                inequalities: NonlinearBlock { count },
            },
        }
    }
}

/// The block responsible for an invalid declaration or evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConstraintBlock {
    /// Parameter bounds.
    Bounds,
    /// Linear equality rows.
    LinearEqualities,
    /// Linear inequality rows.
    LinearInequalities,
    /// Nonlinear equality residuals.
    NonlinearEqualities,
    /// Nonlinear inequality values.
    NonlinearInequalities,
    /// Nonlinear constraint Jacobian.
    Jacobian,
}
/// The side of a parameter bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundSide {
    /// Lower bound.
    Lower,
    /// Upper bound.
    Upper,
}
/// A dimension in a constraint shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintAxis {
    /// Vector length.
    Length,
    /// Matrix rows.
    Rows,
    /// Matrix columns.
    Columns,
}
/// Invalid constraint data or a violation of a callback's shape contract.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConstraintError {
    /// NaN is never an absent-bound sentinel.
    NaNBound {
        /// The affected bound side.
        side: BoundSide,
        /// The zero-based parameter coordinate.
        coordinate: usize,
    },
    /// An infinity has the wrong sign for this bound side.
    InvalidInfinity {
        /// The affected bound side.
        side: BoundSide,
        /// The zero-based parameter coordinate.
        coordinate: usize,
    },
    /// A lower bound exceeds its upper bound.
    InvertedBounds {
        /// The zero-based parameter coordinate.
        coordinate: usize,
    },
    /// A block has an unexpected dimension.
    DimensionMismatch {
        /// The affected constraint block.
        block: ConstraintBlock,
        /// The dimension being checked.
        axis: ConstraintAxis,
        /// The required dimension.
        expected: usize,
        /// The supplied dimension.
        actual: usize,
    },
    /// A solver requires both sides of the box.
    MissingBound {
        /// The affected bound side.
        side: BoundSide,
    },
    /// A solver requires a finite sampling or search interval.
    NonFiniteInterval {
        /// The zero-based parameter coordinate.
        coordinate: usize,
    },
    /// Linear coefficients or right-hand sides must be finite.
    NonFiniteLinearData {
        /// The affected constraint block.
        block: ConstraintBlock,
    },
    /// Constraint layout changed after initialization.
    LayoutChanged,
    /// The total number of folded constraints overflowed.
    CountOverflow,
}
impl std::fmt::Display for ConstraintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid constraints: {self:?}")
    }
}
impl std::error::Error for ConstraintError {}
/// A checked constrained solve preserves the original callback error in `Evaluation`.
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConstraintSolveError<E> {
    /// Invalid declarations or malformed callback outputs.
    InvalidConstraints(ConstraintError),
    /// The original user-chosen callback error.
    Evaluation(E),
}
impl<E> From<E> for ConstraintSolveError<E> {
    fn from(error: E) -> Self {
        Self::Evaluation(error)
    }
}
impl<E: std::fmt::Display> std::fmt::Display for ConstraintSolveError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConstraints(e) => e.fmt(f),
            Self::Evaluation(e) => write!(f, "problem evaluation failed: {e}"),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error
    for ConstraintSolveError<E>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidConstraints(e) => Some(e),
            Self::Evaluation(e) => Some(e),
        }
    }
}

/// Values of the nonlinear inequalities declared in the canonical set.
///
/// The parameter, value, and error carriers remain independent of objective and
/// residual traits. Solvers require the matching parameter and error types.
/// Statically absent and zero-count blocks skip this callback. Non-finite values
/// retain the consuming solver's numerical failure or moderation policy.
pub trait NonlinearInequalityValues: ProblemConstraints {
    /// Parameter type.
    type Param;
    /// Readable vector of constraint values, independent of the parameter type.
    type Values;
    /// Original callback error type.
    type Error;
    /// Evaluate the inequalities in the public `c(x) <= 0` convention.
    fn nonlinear_inequalities(
        &self,
        x: &Self::Param,
    ) -> Result<Self::Values, ConstraintSolveError<Self::Error>>;
}
/// Values of the nonlinear equalities declared in the canonical set.
pub trait NonlinearEqualityValues: ProblemConstraints {
    /// Parameter type.
    type Param;
    /// Readable vector of equality residuals.
    type Values;
    /// Original callback error type.
    type Error;
    /// Evaluate `h(x) = 0`.
    fn nonlinear_equalities(
        &self,
        x: &Self::Param,
    ) -> Result<Self::Values, ConstraintSolveError<Self::Error>>;
}

pub(crate) fn dimension(
    block: ConstraintBlock,
    axis: ConstraintAxis,
    expected: usize,
    actual: usize,
) -> Result<(), ConstraintError> {
    if expected == actual {
        Ok(())
    } else {
        Err(ConstraintError::DimensionMismatch {
            block,
            axis,
            expected,
            actual,
        })
    }
}
pub(crate) fn validate_bound<F: Scalar>(
    value: F,
    side: BoundSide,
    coordinate: usize,
) -> Result<(), ConstraintError> {
    if value.is_nan() {
        return Err(ConstraintError::NaNBound { side, coordinate });
    }
    if (side == BoundSide::Lower && value == F::infinity())
        || (side == BoundSide::Upper && value == F::neg_infinity())
    {
        return Err(ConstraintError::InvalidInfinity { side, coordinate });
    }
    Ok(())
}
macro_rules! borrow_copy {
    ($($ty:ty),* $(,)?) => {$ (
        impl BorrowConstraint for $ty {
            type Borrowed<'a> = Self where Self: 'a;
            fn borrowed(&self) -> Self { *self }
        }
    )*};
}
borrow_copy!(NoBounds, NoLinearBlock, NoNonlinearBlock, NonlinearBlock);
impl<V> BorrowConstraint for BoxBounds<V> {
    type Borrowed<'a>
        = BoxBounds<&'a V>
    where
        Self: 'a;
    fn borrowed(&self) -> Self::Borrowed<'_> {
        BoxBounds::new(self.lower(), self.upper())
    }
}
impl<M, R> BorrowConstraint for LinearBlock<M, R> {
    type Borrowed<'a>
        = LinearBlock<&'a M, &'a R>
    where
        Self: 'a;
    fn borrowed(&self) -> Self::Borrowed<'_> {
        LinearBlock::optional(self.matrix.as_ref().zip(self.rhs.as_ref()))
    }
}
impl<E: BorrowConstraint, I: BorrowConstraint> BorrowConstraint
    for LinearBlocks<E, I>
{
    type Borrowed<'a>
        = LinearBlocks<E::Borrowed<'a>, I::Borrowed<'a>>
    where
        Self: 'a;
    fn borrowed(&self) -> Self::Borrowed<'_> {
        LinearBlocks {
            equalities: self.equalities.borrowed(),
            inequalities: self.inequalities.borrowed(),
        }
    }
}
impl<E: BorrowConstraint, I: BorrowConstraint> BorrowConstraint
    for NonlinearBlocks<E, I>
{
    type Borrowed<'a>
        = NonlinearBlocks<E::Borrowed<'a>, I::Borrowed<'a>>
    where
        Self: 'a;
    fn borrowed(&self) -> Self::Borrowed<'_> {
        NonlinearBlocks {
            equalities: self.equalities.borrowed(),
            inequalities: self.inequalities.borrowed(),
        }
    }
}
