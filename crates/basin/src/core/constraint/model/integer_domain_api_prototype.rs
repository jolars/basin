//! Integer-domain API experiment. These types are never exported.
//!
//! This child module can extend the sealed canonical declaration in tests.
//! The wrapper models a future fourth `Constraints` parameter without changing
//! production types. Run with `cargo test -p basin integer_domain_api_prototype`.

use std::{
    cell::RefCell,
    ops::{Index, IndexMut},
    rc::Rc,
};

use super::*;
use crate::{
    Constrained, CostFunction, De, DeInject, EvalCounts, Executor, FiniteDiff,
    InitialState, InnerExecutor, NelderMead, PopulationProgress, Problem,
    RelaxedInner, RobustLeastSquares, SimplexProgress, SoftL1Loss, Solver,
    State, Termination, TerminationReport, VectorLen,
};

#[derive(Clone, Copy, Debug, Default)]
struct ContinuousDomain;
impl BorrowConstraint for ContinuousDomain {
    type Borrowed<'a> = Self;
    fn borrowed(&self) -> Self {
        *self
    }
}

#[derive(Clone, Debug)]
struct IntegerCoordinates(Vec<bool>);
impl BorrowConstraint for IntegerCoordinates {
    type Borrowed<'a> = &'a Self;
    fn borrowed(&self) -> &Self {
        self
    }
}

#[derive(Clone, Debug)]
struct DomainConstraints<C, D = ContinuousDomain> {
    constraints: C,
    domain: D,
}
impl<C> DomainConstraints<C> {
    fn new(constraints: C) -> Self {
        Self {
            constraints,
            domain: ContinuousDomain,
        }
    }
}
impl<C, D> DomainConstraints<C, D> {
    fn with_domains<T>(self, domain: T) -> DomainConstraints<C, T> {
        DomainConstraints {
            constraints: self.constraints,
            domain,
        }
    }
}
impl<B, L, N, D> DomainConstraints<Constraints<B, L, N>, D> {
    fn with_bounds<V>(
        self,
        bounds: BoxBounds<V>,
    ) -> DomainConstraints<Constraints<BoxBounds<V>, L, N>, D> {
        DomainConstraints {
            constraints: self.constraints.with_bounds(bounds),
            domain: self.domain,
        }
    }
}
impl<C, D> sealed::Set for DomainConstraints<C, D> {}
impl<C: ConstraintDescription, D: BorrowConstraint> ConstraintDescription
    for DomainConstraints<C, D>
{
    type Borrowed<'a>
        = DomainConstraints<C::Borrowed<'a>, D::Borrowed<'a>>
    where
        Self: 'a;
    fn borrowed(&self) -> Self::Borrowed<'_> {
        DomainConstraints {
            constraints: self.constraints.borrowed(),
            domain: self.domain.borrowed(),
        }
    }
}

type MixedBox<V> =
    DomainConstraints<Constraints<BoxBounds<V>>, IntegerCoordinates>;
type MixedBoxView<'a, V> =
    DomainConstraints<Constraints<BoxBounds<&'a V>>, &'a IntegerCoordinates>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DomainError {
    Shape,
    InvalidBounds(usize),
    EmptyIntegerInterval(usize),
    UnrepresentableIntegerInterval(usize),
    InvalidSeed(usize),
}

fn validate<V, F>(
    declaration: &MixedBoxView<'_, V>,
    seed: &V,
) -> Result<Vec<usize>, DomainError>
where
    V: VectorLen + Index<usize, Output = F>,
    F: Scalar,
{
    let bounds = declaration.constraints.bounds();
    let (Some(lower), Some(upper)) = (bounds.lower(), bounds.upper()) else {
        return Err(DomainError::Shape);
    };
    let n = seed.vec_len();
    let mask = &declaration.domain.0;
    if n == 0 || lower.vec_len() != n || upper.vec_len() != n || mask.len() != n
    {
        return Err(DomainError::Shape);
    }
    // Outside this range the scalar cannot distinguish consecutive integers.
    let exact_limit = (F::one() + F::one()) / F::epsilon();
    let mut free = Vec::new();
    for i in 0..n {
        if !lower[i].is_finite() || !upper[i].is_finite() || lower[i] > upper[i]
        {
            return Err(DomainError::InvalidBounds(i));
        }
        if mask[i] {
            if lower[i].ceil() > upper[i].floor() {
                return Err(DomainError::EmptyIntegerInterval(i));
            }
            if lower[i].ceil() < -exact_limit || upper[i].floor() > exact_limit
            {
                return Err(DomainError::UnrepresentableIntegerInterval(i));
            }
        } else if lower[i] != upper[i] {
            free.push(i);
        }
        if !seed[i].is_finite()
            || seed[i] < lower[i]
            || seed[i] > upper[i]
            || (mask[i] && seed[i] != seed[i].round())
        {
            return Err(DomainError::InvalidSeed(i));
        }
    }
    Ok(free)
}

struct FixedIntegers<'a, P, V, F> {
    problem: &'a P,
    anchor: V,
    free: Vec<usize>,
    bounds: Constraints<BoxBounds<Vec<F>>>,
}
impl<P, V, F> ProblemConstraints for FixedIntegers<'_, P, V, F> {
    type Constraints = Constraints<BoxBounds<Vec<F>>>;
    fn constraint_set(&self) -> Constraints<BoxBounds<&Vec<F>>> {
        self.bounds.borrowed()
    }
}
impl<P, V, F: Copy> FixedIntegers<'_, P, V, F>
where
    V: Clone + IndexMut<usize, Output = F>,
{
    fn expand(&self, reduced: &[F]) -> V {
        assert_eq!(reduced.len(), self.free.len());
        let mut full = self.anchor.clone();
        for (&i, &value) in self.free.iter().zip(reduced) {
            full[i] = value;
        }
        full
    }
}
impl<P, V, F> CostFunction for FixedIntegers<'_, P, V, F>
where
    P: CostFunction<Param = V, Output = F>,
    V: Clone + IndexMut<usize, Output = F>,
    F: Scalar,
{
    type Param = Vec<F>;
    type Output = F;
    type Error = P::Error;
    fn cost(&self, reduced: &Vec<F>) -> Result<F, Self::Error> {
        self.problem.cost(&self.expand(reduced))
    }
}

#[derive(Debug)]
struct Refinement<V, F: Scalar> {
    point: V,
    cost: F,
    report: TerminationReport<F>,
}
#[derive(Debug, PartialEq, Eq)]
enum RefineError<E> {
    Domain(DomainError),
    Inner(ConstraintSolveError<E>),
}

fn refine<P, V, F>(
    problem: &mut Problem<P>,
    seed: V,
    max_iter: u64,
) -> Result<Option<Refinement<V, F>>, RefineError<P::Error>>
where
    P: CostFunction<Param = V, Output = F, Constraints = MixedBox<V>>,
    V: VectorLen + Clone + IndexMut<usize, Output = F>,
    F: Scalar,
{
    let declaration = problem.inner().constraint_set();
    let free = validate(&declaration, &seed).map_err(RefineError::Domain)?;
    if free.is_empty() {
        return Ok(None);
    }
    let bounds = declaration.constraints.bounds();
    let lower = bounds.lower().unwrap();
    let upper = bounds.upper().unwrap();
    let reduced = free.iter().map(|&i| seed[i]).collect::<Vec<_>>();
    let reduced_bounds = Constraints::new().with_bounds(BoxBounds::new(
        Some(free.iter().map(|&i| lower[i]).collect()),
        Some(free.iter().map(|&i| upper[i]).collect()),
    ));
    let (outcome, counts) = {
        let adapter = FixedIntegers {
            problem: problem.inner(),
            anchor: seed,
            free,
            bounds: reduced_bounds,
        };
        let mut adapted = Problem::new(adapter);
        let mut inner = InnerExecutor::new(NelderMead::new().projected())
            .max_iter(max_iter);
        let state = inner.solver().seed(&reduced);
        let outcome = inner.run(&mut adapted, state).map(|result| Refinement {
            point: adapted.inner().expand(result.state.param()),
            cost: result.state.cost(),
            report: result.report,
        });
        (outcome, *adapted.counts())
    };
    // Fold adapter work once, including callback attempts preceding an error.
    problem.counts_mut().add(&counts);
    outcome.map(Some).map_err(RefineError::Inner)
}

type Trace<F> = Rc<RefCell<Vec<Vec<F>>>>;
struct Fixture<V, F: Scalar> {
    declaration: MixedBox<V>,
    trace: Trace<F>,
    fail_after: Option<usize>,
}
impl<V, F: Scalar> ProblemConstraints for Fixture<V, F> {
    type Constraints = MixedBox<V>;
    fn constraint_set(
        &self,
    ) -> <Self::Constraints as ConstraintDescription>::Borrowed<'_> {
        self.declaration.borrowed()
    }
}
impl<V, F> CostFunction for Fixture<V, F>
where
    V: VectorLen + Index<usize, Output = F>,
    F: Scalar,
{
    type Param = V;
    type Output = F;
    type Error = &'static str;
    fn cost(&self, point: &V) -> Result<F, Self::Error> {
        let mut trace = self.trace.borrow_mut();
        trace.push((0..point.vec_len()).map(|i| point[i]).collect());
        for (i, &integer) in self.declaration.domain.0.iter().enumerate() {
            if integer {
                assert_eq!(point[i], point[i].round());
            }
        }
        if self.fail_after.is_some_and(|n| trace.len() >= n) {
            return Err("fixture callback failed");
        }
        Ok((0..point.vec_len())
            .map(|i| (point[i] - num(0.25)).powi(2))
            .sum())
    }
}

fn num<F: Scalar>(value: f64) -> F {
    F::from_f64(value).unwrap()
}
fn declaration<V>(lower: V, upper: V, mask: Vec<bool>) -> MixedBox<V> {
    DomainConstraints::new(Constraints::new())
        .with_domains(IntegerCoordinates(mask))
        .with_bounds(BoxBounds::new(Some(lower), Some(upper)))
}
fn fixture<V, F: Scalar>(make: impl Fn(&[F]) -> V) -> Fixture<V, F> {
    Fixture {
        declaration: declaration(
            make(&[num(-2.5), num(-2.0), num(3.0)]),
            make(&[num(2.5), num(2.0), num(3.0)]),
            vec![true, false, false],
        ),
        trace: Rc::default(),
        fail_after: None,
    }
}

#[test]
fn ordinary_continuous_de_still_accepts_existing_declarations() {
    struct Sphere;
    impl ProblemConstraints for Sphere {
        type Constraints = NoConstraints;
        fn constraint_set(&self) -> NoConstraints {
            NoConstraints::EMPTY
        }
    }
    impl CostFunction for Sphere {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = std::convert::Infallible;
        fn cost(&self, point: &Vec<f64>) -> Result<f64, Self::Error> {
            Ok(point.iter().map(|x| x * x).sum())
        }
    }
    let problem = Constrained::new(
        Sphere,
        Constraints::new()
            .with_bounds(BoxBounds::new(Some(vec![-2.0]), Some(vec![2.0]))),
    );
    let result = Executor::new(
        problem,
        De::new(42).with_pop_size(4),
        PopulationProgress::empty(),
    )
    .max_iter(1)
    .run()
    .unwrap();
    assert_eq!(result.iter(), 1);
    assert_eq!(result.counts.cost_evals, 8);
}

// A newly available Solver implementation makes the inferred marker ambiguous,
// so these checks fail to compile if an unsupported solver starts accepting it.
macro_rules! assert_not_solver {
    ($solver:ty, $state:ty) => {{
        trait AmbiguousIfSolver<Marker> {
            fn check() {}
        }
        impl<T> AmbiguousIfSolver<()> for T {}
        impl<T: Solver<Fixture<Vec<f64>, f64>, $state>>
            AmbiguousIfSolver<u8> for T
        {
        }
        let _ = <$solver as AmbiguousIfSolver<_>>::check;
    }};
}

#[test]
fn continuous_solvers_and_box_relaxation_reject_integer_domains() {
    type Local = NelderMead<Vec<f64>>;
    assert_not_solver!(Local, SimplexProgress<Vec<f64>>);
    assert_not_solver!(
        NelderMead<Vec<f64>, f64, crate::solver::nelder_mead::Projected>,
        SimplexProgress<Vec<f64>>
    );
    assert_not_solver!(De, PopulationProgress<Vec<f64>>);
    assert_not_solver!(DeInject<Local, Vec<f64>>, PopulationProgress<Vec<f64>>);
    assert_not_solver!(RelaxedInner<Local>, SimplexProgress<Vec<f64>>);
    assert_not_solver!(
        DeInject<RelaxedInner<Local>, Vec<f64>>,
        PopulationProgress<Vec<f64>>
    );
}

#[test]
fn builders_and_derivative_and_loss_adapters_preserve_domains() {
    let continuous = DomainConstraints::new(Constraints::new());
    let _: DomainConstraints<NoConstraints, ContinuousDomain> =
        continuous.borrowed();
    let p = fixture::<_, f64>(|x| x.to_vec());
    let original_bounds = p.declaration.constraints.bounds().lower().unwrap();
    let borrowed = p.constraint_set();
    assert!(std::ptr::eq(
        borrowed.constraints.bounds().lower().copied().unwrap(),
        original_bounds
    ));
    assert!(std::ptr::eq(borrowed.domain, &p.declaration.domain));

    let wrapped =
        RobustLeastSquares::<_, _, f64>::new(FiniteDiff::new(p), SoftL1Loss);
    fn requires_mixed_declaration<
        P: ProblemConstraints<Constraints = MixedBox<Vec<f64>>>,
    >(
        _: &P,
    ) {
    }
    requires_mixed_declaration(&wrapped);
    assert_eq!(wrapped.constraint_set().domain.0, [true, false, false]);
    let changed = fixture::<_, f64>(|x| x.to_vec())
        .declaration
        .with_bounds(BoxBounds::new(Some(vec![-4.0; 3]), Some(vec![4.0; 3])));
    assert_eq!(changed.domain.0, [true, false, false]);
    let nonlinear = DomainConstraints::new(
        Constraints::new().with_nonlinear_inequalities(2),
    )
    .with_domains(IntegerCoordinates(vec![true]))
    .with_bounds(BoxBounds::new(Some(vec![-1.0]), Some(vec![1.0])));
    assert_eq!(nonlinear.domain.0, [true]);
    assert_eq!(nonlinear.constraints.nonlinear().inequalities().count(), 2);
}

fn domain_cases<F: Scalar>() {
    let check = |lo: &[f64], hi: &[f64], mask: Vec<bool>, seed: &[f64]| {
        validate(
            &declaration(
                lo.iter().copied().map(num::<F>).collect::<Vec<_>>(),
                hi.iter().copied().map(num::<F>).collect::<Vec<_>>(),
                mask,
            )
            .borrowed(),
            &seed.iter().copied().map(num::<F>).collect::<Vec<_>>(),
        )
    };
    assert_eq!(
        check(&[0.2], &[0.8], vec![true], &[0.5]),
        Err(DomainError::EmptyIntegerInterval(0))
    );
    assert_eq!(
        check(&[-0.8], &[-0.2], vec![true], &[-0.5]),
        Err(DomainError::EmptyIntegerInterval(0))
    );
    assert_eq!(check(&[0.2], &[1.8], vec![true], &[1.0]), Ok(vec![]));
    assert_eq!(check(&[-1.8], &[-0.2], vec![true], &[-1.0]), Ok(vec![]));
    assert_eq!(check(&[1.0], &[1.0], vec![true], &[1.0]), Ok(vec![]));
    assert_eq!(
        check(&[0.2], &[1.8], vec![true], &[0.5]),
        Err(DomainError::InvalidSeed(0))
    );
    assert_eq!(
        check(&[0.0], &[1.0], vec![false], &[f64::NAN]),
        Err(DomainError::InvalidSeed(0))
    );
    for seed in [f64::NEG_INFINITY, f64::INFINITY, -1.0, 2.0] {
        assert_eq!(
            check(&[0.0], &[1.0], vec![true], &[seed]),
            Err(DomainError::InvalidSeed(0))
        );
    }
    assert_eq!(check(&[0.0], &[1.0], vec![false], &[0.5]), Ok(vec![0]));
    assert_eq!(
        check(&[0.0], &[1.0], vec![], &[0.0]),
        Err(DomainError::Shape)
    );
    assert_eq!(check(&[], &[], vec![], &[]), Err(DomainError::Shape));
    assert_eq!(
        check(&[0.0], &[], vec![true], &[0.0]),
        Err(DomainError::Shape)
    );
    for (lo, hi) in [
        (f64::NAN, 1.0),
        (0.0, f64::NAN),
        (f64::NEG_INFINITY, 1.0),
        (0.0, f64::INFINITY),
        (1.0, 0.0),
    ] {
        assert_eq!(
            check(&[lo], &[hi], vec![true], &[0.0]),
            Err(DomainError::InvalidBounds(0))
        );
    }
    let limit = (F::one() + F::one()) / F::epsilon();
    let at_limit = declaration(vec![-limit], vec![limit], vec![true]);
    assert_eq!(validate(&at_limit.borrowed(), &vec![limit]), Ok(vec![]));
    let outside = declaration(vec![-limit], vec![limit + num(2.0)], vec![true]);
    assert_eq!(
        validate(&outside.borrowed(), &vec![F::zero()]),
        Err(DomainError::UnrepresentableIntegerInterval(0))
    );
    let outside = declaration(vec![-limit - num(2.0)], vec![limit], vec![true]);
    assert_eq!(
        validate(&outside.borrowed(), &vec![F::zero()]),
        Err(DomainError::UnrepresentableIntegerInterval(0))
    );
    for bounds in [
        BoxBounds::new(None, Some(vec![F::one()])),
        BoxBounds::new(Some(vec![F::zero()]), None),
    ] {
        let missing = DomainConstraints::new(Constraints::new())
            .with_domains(IntegerCoordinates(vec![true]))
            .with_bounds(bounds);
        assert_eq!(
            validate(&missing.borrowed(), &vec![F::zero()]),
            Err(DomainError::Shape)
        );
    }
}

#[test]
fn integer_intervals_and_seeds_are_checked_in_both_precisions() {
    domain_cases::<f32>();
    domain_cases::<f64>();
}

fn refinement_cases<V, F>(make: impl Fn(&[F]) -> V)
where
    V: VectorLen + Clone + IndexMut<usize, Output = F>,
    F: Scalar,
{
    let mut problem = Problem::new(fixture(&make));
    let seed = make(&[num(2.0), num(-1.0), num(3.0)]);
    let before = problem.cost(&seed).unwrap();
    let trace = problem.inner().trace.clone();
    let result = refine(&mut problem, seed.clone(), 100).unwrap().unwrap();
    assert_eq!(result.point[0], num(2.0));
    assert_eq!(result.point[2], num(3.0));
    assert!((result.point[1] - num(0.25)).abs() < num(0.01));
    assert!(result.cost < before);
    assert!(matches!(
        result.report.termination,
        Termination::Converged(_) | Termination::Limit(_)
    ));
    assert_eq!(problem.counts().cost_evals as usize, trace.borrow().len());
    for point in trace.borrow().iter() {
        assert_eq!(point[0], num(2.0));
        assert_eq!(point[2], num(3.0));
        assert!(point[1] >= num(-2.0) && point[1] <= num(2.0));
    }
    let cost = problem.cost(&result.point).unwrap();
    assert_eq!(cost, result.cost);

    let partial = refine(&mut problem, seed.clone(), 0).unwrap().unwrap();
    assert!(matches!(partial.report.termination, Termination::Limit(_)));
    assert_eq!(partial.point[0], num(2.0));
    assert_eq!(problem.counts().cost_evals as usize, trace.borrow().len());

    let mut failed = fixture(&make);
    failed.fail_after = Some(2);
    let mut problem = Problem::new(failed);
    assert!(matches!(
        refine(&mut problem, seed.clone(), 10),
        Err(RefineError::Inner(ConstraintSolveError::Evaluation(
            "fixture callback failed"
        )))
    ));
    assert_eq!(problem.counts().cost_evals, 2);
    assert_eq!(problem.inner().trace.borrow().len(), 2);

    let mut invalid = seed.clone();
    invalid[0] = num(0.5);
    let mut problem = Problem::new(fixture(&make));
    assert!(matches!(
        refine(&mut problem, invalid, 10),
        Err(RefineError::Domain(DomainError::InvalidSeed(0)))
    ));
    assert_eq!(*problem.counts(), EvalCounts::default());
    assert!(problem.inner().trace.borrow().is_empty());

    let mut fixed = fixture(&make);
    fixed.declaration.domain.0 = vec![true; 3];
    let mut problem = Problem::new(fixed);
    assert!(refine(&mut problem, seed, 10).unwrap().is_none());
    assert_eq!(*problem.counts(), EvalCounts::default());
    let mut fixed = fixture(&make);
    fixed.declaration = declaration(
        make(&[num(2.0), num(-1.0), num(3.0)]),
        make(&[num(2.0), num(-1.0), num(3.0)]),
        vec![false; 3],
    );
    let mut problem = Problem::new(fixed);
    assert!(
        refine(&mut problem, make(&[num(2.0), num(-1.0), num(3.0)]), 10)
            .unwrap()
            .is_none()
    );
    assert_eq!(*problem.counts(), EvalCounts::default());
}

#[test]
fn refinement_preserves_integer_callbacks_vec() {
    refinement_cases::<_, f32>(|x| x.to_vec());
    refinement_cases::<_, f64>(|x| x.to_vec());
}

#[cfg(any(
    feature = "nalgebra_all",
    feature = "ndarray_all",
    feature = "faer_all"
))]
macro_rules! backend_cases {
    ($test:ident, $make:expr) => {
        #[test]
        fn $test() {
            refinement_cases::<_, f32>($make);
            refinement_cases::<_, f64>($make);
        }
    };
}
#[cfg(feature = "nalgebra_v0_35")]
backend_cases!(refinement_nalgebra_035, |x| {
    nalgebra::DVector::from_column_slice(x)
});
#[cfg(feature = "nalgebra_v0_34")]
backend_cases!(refinement_nalgebra_034, |x| {
    nalgebra_0_34::DVector::from_column_slice(x)
});
#[cfg(feature = "nalgebra_v0_33")]
backend_cases!(refinement_nalgebra_033, |x| {
    nalgebra_0_33::DVector::from_column_slice(x)
});
#[cfg(feature = "nalgebra_v0_32")]
backend_cases!(refinement_nalgebra_032, |x| {
    nalgebra_0_32::DVector::from_column_slice(x)
});
#[cfg(feature = "ndarray_v0_17")]
backend_cases!(refinement_ndarray_017, |x| ndarray::Array1::from_vec(
    x.to_vec()
));
#[cfg(feature = "ndarray_v0_16")]
backend_cases!(refinement_ndarray_016, |x| ndarray_0_16::Array1::from_vec(
    x.to_vec()
));
#[cfg(feature = "ndarray_v0_15")]
backend_cases!(refinement_ndarray_015, |x| ndarray_0_15::Array1::from_vec(
    x.to_vec()
));
#[cfg(feature = "faer_v0_24")]
backend_cases!(refinement_faer_024, |x| {
    faer::Col::from_fn(x.len(), |i| x[i])
});
#[cfg(feature = "faer_v0_23")]
backend_cases!(refinement_faer_023, |x| {
    faer_0_23::Col::from_fn(x.len(), |i| x[i])
});
#[cfg(feature = "faer_v0_22")]
backend_cases!(refinement_faer_022, |x| {
    faer_0_22::Col::from_fn(x.len(), |i| x[i])
});
