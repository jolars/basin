//! Reports preserve native measures and the records that published them.

use basin::{
    ConstraintViolationState, CountsMirror, EvalCounts, EvaluatedState,
    IncumbentRef, IncumbentState, Observe, Scalar, SelectedFirstOrderState,
    SelectedState, State, VectorLen, ViolationMeasure, ViolationReport,
    ViolationScaling, ViolationTransformation,
};

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    convert::Infallible,
};

thread_local! {
    // Exclude concurrent tests and test-harness activity from allocation counts.
    static ALLOCATION_REQUESTS: Cell<usize> = const { Cell::new(0) };
}

struct Counting;

fn record_allocation() {
    let _ = ALLOCATION_REQUESTS.try_with(|count| count.set(count.get() + 1));
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(
        &self,
        ptr: *mut u8,
        layout: Layout,
        size: usize,
    ) -> *mut u8 {
        record_allocation();
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

struct Residuals<F: Scalar>([F; 2]);

impl<F: Scalar> basin::ProblemConstraints for Residuals<F> {
    type Constraints = basin::NoConstraints;
    fn constraint_set(&self) -> basin::NoConstraints {
        basin::NoConstraints::EMPTY
    }
}
impl<F: Scalar> basin::CostFunction for Residuals<F> {
    type Param = Vec<F>;
    type Output = F;
    type Error = Infallible;
    fn cost(&self, _: &Vec<F>) -> Result<F, Infallible> {
        Ok(F::zero())
    }
}
impl<F: Scalar> basin::Gradient for Residuals<F> {
    type Gradient = Vec<F>;
    fn gradient(&self, x: &Vec<F>) -> Result<Vec<F>, Infallible> {
        Ok(vec![F::zero(); x.len()])
    }
}
impl<F: Scalar> basin::NonlinearInequalityValues for Residuals<F> {
    type Param = Vec<F>;
    type Values = Vec<F>;
    type Error = Infallible;
    fn nonlinear_inequalities(
        &self,
        _: &Vec<F>,
    ) -> Result<Vec<F>, basin::ConstraintSolveError<Infallible>> {
        Ok(self.0.to_vec())
    }
}
impl<F: Scalar> basin::ConstraintJacobian for Residuals<F> {
    type Param = Vec<F>;
    type Jacobian = basin::DenseMatrix<F>;
    type Error = Infallible;
    fn constraint_jacobian(
        &self,
        x: &Vec<F>,
    ) -> Result<Self::Jacobian, basin::ConstraintSolveError<Infallible>> {
        Ok(basin::DenseMatrix::from_fn(2, x.len(), |_, _| F::zero()))
    }
}

fn assert_native<S: ConstraintViolationState>(
    state: &S,
    value: S::Float,
    measure: ViolationMeasure,
    transformation: ViolationTransformation,
) {
    let expected = ViolationReport::new(value, measure)
        .with_transformation(transformation);
    assert_eq!(state.current_violation(), Some(&expected));
    assert_eq!(state.incumbent_violation(), Some(&expected));
}

macro_rules! native_reports {
    ($test:ident, $scalar:ty) => {
        #[test]
        fn $test() {
            use basin::{
                AugmentedLagrangianMethod, Constrained, Constraints,
                DenseMatrix, Executor, GradientDescent, LinearBlock,
            };
            type F = $scalar;
            let problem = || {
                Constrained::new(
                    Residuals([3.0 as F, 4.0 as F]),
                    Constraints::new().with_nonlinear_inequalities(2),
                )
            };
            let mads = Executor::from_start(
                problem(),
                basin::Mads::new().constrained(),
                vec![0.0 as F; 2],
            )
            .max_iter(0)
            .run()
            .unwrap();
            assert_native(
                &mads.state,
                25.0,
                ViolationMeasure::SquaredL2,
                ViolationTransformation::Identity,
            );
            assert_eq!(mads.counts.cost_evals, 1);
            assert_eq!(mads.counts.residual_evals, 1);
            let slsqp = Executor::from_start(
                problem(),
                basin::Slsqp::new(),
                vec![0.0 as F; 2],
            )
            .max_iter(0)
            .run()
            .unwrap();
            assert_native(
                &slsqp.state,
                7.0,
                ViolationMeasure::L1,
                ViolationTransformation::NonFiniteToInfinity,
            );
            assert_eq!(slsqp.counts.cost_evals, 1);
            assert_eq!(slsqp.counts.gradient_evals, 1);
            assert_eq!(slsqp.counts.residual_evals, 1);
            assert_eq!(slsqp.counts.jacobian_evals, 1);
            let cobyla = Executor::from_start(
                problem(),
                basin::Cobyla::new(),
                vec![0.0 as F; 2],
            )
            .max_iter(0)
            .run()
            .unwrap();
            assert_native(
                &cobyla.state,
                4.0,
                ViolationMeasure::LInf,
                ViolationTransformation::PrimaModeration,
            );
            assert_eq!(cobyla.counts.cost_evals, 3);
            assert_eq!(cobyla.counts.residual_evals, 3);
            let linear =
                Constraints::new().with_linear_equalities(LinearBlock::new(
                    DenseMatrix::from_row_slice(
                        2,
                        2,
                        &[1.0 as F, 0.0, 0.0, 1.0],
                    ),
                    vec![0.0 as F; 2],
                ));
            let al = Executor::from_start(
                Constrained::new(Residuals([0.0 as F; 2]), linear),
                AugmentedLagrangianMethod::with_inner_solver(
                    GradientDescent::new(0.05 as F),
                ),
                vec![3.0 as F, 4.0],
            )
            .max_iter(0)
            .run()
            .unwrap();
            assert_native(
                &al.state,
                5.0,
                ViolationMeasure::L2,
                ViolationTransformation::Identity,
            );
            assert_eq!(al.counts.cost_evals, 1);
            assert_eq!(al.counts.total_work(), 1);
        }
    };
}
native_reports!(native_f64_reports_preserve_different_aggregates, f64);
native_reports!(native_f32_reports_preserve_different_aggregates, f32);

#[test]
fn native_reports_describe_nonfinite_handling() {
    use basin::{Constrained, Constraints, Executor};
    let problem = || {
        Constrained::new(
            Residuals([f64::NAN, f64::NEG_INFINITY]),
            Constraints::new().with_nonlinear_inequalities(2),
        )
    };
    let cobyla =
        Executor::from_start(problem(), basin::Cobyla::new(), vec![0.0; 2])
            .max_iter(0)
            .run()
            .unwrap();
    assert_native(
        &cobyla.state,
        2_f64.powi(100),
        ViolationMeasure::LInf,
        ViolationTransformation::PrimaModeration,
    );
    let slsqp =
        Executor::from_start(problem(), basin::Slsqp::new(), vec![0.0; 2])
            .max_iter(0)
            .run()
            .unwrap();
    assert_native(
        &slsqp.state,
        f64::INFINITY,
        ViolationMeasure::L1,
        ViolationTransformation::NonFiniteToInfinity,
    );
    let mads = Executor::from_start(
        problem(),
        basin::Mads::new().constrained(),
        vec![0.0; 2],
    )
    .max_iter(0)
    .run()
    .unwrap();
    // Reporting preserves MADS's positive-part comparisons, which omit NaN.
    assert_native(
        &mads.state,
        0.0,
        ViolationMeasure::SquaredL2,
        ViolationTransformation::Identity,
    );
}

fn described<F: Scalar>(value: F) -> ViolationReport<F> {
    ViolationReport::new(
        value,
        ViolationMeasure::Custom("sum of weighted residuals".into()),
    )
    .with_scaling(ViolationScaling::Custom("residual i / scale i".into()))
    .with_transformation(ViolationTransformation::Custom(
        "non-finite residuals become infinity before aggregation".into(),
    ))
}

#[test]
fn reports_keep_native_metadata_compact_and_thread_safe() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<ViolationReport<f32>>();
    send_sync::<ViolationReport<f64>>();
    let maximum = size_of::<f64>() + 2 * size_of::<usize>();
    assert!(size_of::<ViolationReport<f32>>() <= maximum);
    assert!(size_of::<ViolationReport<f64>>() <= maximum);
    #[cfg(target_pointer_width = "64")]
    assert!(size_of::<SelectedState<Vec<f64>>>() <= 6 * size_of::<usize>());
    for measure in [
        ViolationMeasure::L1,
        ViolationMeasure::L2,
        ViolationMeasure::SquaredL2,
        ViolationMeasure::LInf,
    ] {
        for transformation in [
            ViolationTransformation::Identity,
            ViolationTransformation::PrimaModeration,
            ViolationTransformation::NonFiniteToInfinity,
        ] {
            let report = ViolationReport::new(1.0, measure.clone())
                .with_transformation(transformation.clone());
            let another = ViolationReport::new(2.0, measure.clone())
                .with_transformation(transformation.clone());
            assert_eq!(report.measure(), &measure);
            assert_eq!(report.scaling(), &ViolationScaling::Unscaled);
            assert_eq!(report.transformation(), &transformation);
            assert!(std::ptr::eq(report.measure(), another.measure()));
            let changed = report
                .with_measure(ViolationMeasure::L2)
                .with_scaling(ViolationScaling::Unscaled);
            assert_eq!(changed.measure(), &ViolationMeasure::L2);
            assert_eq!(changed.transformation(), &transformation);
        }
    }
}

#[test]
fn native_publication_and_selection_reuse_state_storage() {
    use std::hint::black_box;
    let mut state = SelectedState::new(0.0);
    for (measure, transformation) in [
        (ViolationMeasure::L1, ViolationTransformation::Identity),
        (ViolationMeasure::L2, ViolationTransformation::Identity),
        (
            ViolationMeasure::SquaredL2,
            ViolationTransformation::Identity,
        ),
        (
            ViolationMeasure::LInf,
            ViolationTransformation::PrimaModeration,
        ),
        (
            ViolationMeasure::L1,
            ViolationTransformation::NonFiniteToInfinity,
        ),
    ] {
        ALLOCATION_REQUESTS.set(0);
        for index in 0..100 {
            let value = f64::from(index);
            let report = ViolationReport::new(value, measure.clone())
                .with_transformation(transformation.clone());
            state.replace(value, value, report);
            assert!(state.select_current());
            state.update_best();
            black_box(&state);
        }
        state.reset();
        let requests = ALLOCATION_REQUESTS.get();
        assert_eq!(requests, 0);
        assert!(state.current().is_none());
    }
    let custom = described(1.0);
    ALLOCATION_REQUESTS.set(0);
    let copied = black_box(custom.clone());
    let requests = ALLOCATION_REQUESTS.get();
    assert_eq!(requests, 0);
    assert_eq!(copied, custom);
}

#[test]
fn builders_can_add_custom_metadata_to_native_reports() {
    let native = ViolationReport::new(1.0, ViolationMeasure::L2);
    let measure = native
        .clone()
        .with_measure(ViolationMeasure::Custom("weighted norm".into()));
    let scaling = native
        .clone()
        .with_scaling(ViolationScaling::Custom("normalized".into()));
    let transformation = native
        .clone()
        .with_transformation(ViolationTransformation::Custom("clipped".into()));
    assert_eq!(native.measure(), &ViolationMeasure::L2);
    assert_eq!(
        measure.measure(),
        &ViolationMeasure::Custom("weighted norm".into())
    );
    assert_eq!(scaling.measure(), native.measure());
    assert_eq!(
        scaling.scaling(),
        &ViolationScaling::Custom("normalized".into())
    );
    assert_eq!(transformation.measure(), native.measure());
    assert_eq!(
        transformation.transformation(),
        &ViolationTransformation::Custom("clipped".into())
    );
    assert_eq!(
        format!("{native:?}"),
        "ViolationReport { value: 1.0, measure: L2, scaling: Unscaled, transformation: Identity }"
    );
}

#[test]
fn custom_metadata_is_shared_until_a_builder_changes_it() {
    let original = described(0.5);
    let copied = original.clone();
    assert!(std::ptr::eq(original.measure(), copied.measure()));
    let changed = copied
        .with_measure(ViolationMeasure::L2)
        .with_scaling(ViolationScaling::Unscaled)
        .with_transformation(ViolationTransformation::Identity);
    assert_eq!(original, described(0.5));
    assert_eq!(changed, ViolationReport::new(0.5, ViolationMeasure::L2));
    let native = ViolationReport::new(1.0, ViolationMeasure::L2);
    assert!(std::ptr::eq(changed.measure(), native.measure()));
}

#[cfg(feature = "serde")]
#[test]
fn compact_reports_keep_the_inline_serialization_format() {
    #[derive(serde::Serialize, serde::Deserialize)]
    struct InlineReport<F> {
        value: F,
        measure: ViolationMeasure,
        scaling: ViolationScaling,
        transformation: ViolationTransformation,
    }
    fn round_trip<
        F: Scalar + serde::Serialize + serde::de::DeserializeOwned,
    >() {
        for report in [
            ViolationReport::new(F::one(), ViolationMeasure::L1),
            ViolationReport::new(F::infinity(), ViolationMeasure::LInf)
                .with_transformation(ViolationTransformation::PrimaModeration),
            described(F::zero()),
        ] {
            let inline = InlineReport {
                value: report.value,
                measure: report.measure().clone(),
                scaling: report.scaling().clone(),
                transformation: report.transformation().clone(),
            };
            let bytes = postcard::to_allocvec(&inline).unwrap();
            assert_eq!(postcard::to_allocvec(&report).unwrap(), bytes);
            let restored: ViolationReport<F> =
                postcard::from_bytes(&bytes).unwrap();
            assert_eq!(restored, report);
        }
    }
    round_trip::<f32>();
    round_trip::<f64>();
}

fn records<
    V: Clone + VectorLen + PartialEq + std::fmt::Debug + basin::VectorIndex<F>,
    F: Scalar,
>(
    x: V,
    gradient: V,
) {
    let zero = F::zero();
    let one = F::one();
    let original = described(one);
    let later = ViolationReport::new(zero, ViolationMeasure::LInf);
    let mut state = SelectedState::<_, F>::new(x.clone());
    assert!(state.current_violation().is_none());
    assert!(state.incumbent_violation().is_none());
    assert!(!state.select_current());
    state.replace(x.clone(), zero, original.clone());
    assert_eq!(state.current_violation(), Some(&original));
    assert!(state.incumbent_violation().is_none());
    state.select_current();
    state.replace(x.clone(), one, later.clone());
    let mut counts = EvalCounts::default();
    counts.cost_evals = 3;
    state.mirror(&counts);
    state.update_best();
    assert_eq!(state.current_violation(), Some(&later));
    assert_eq!(state.incumbent_violation(), Some(&original));
    assert_eq!(state.current().unwrap().2, &later);
    assert_eq!(state.best().unwrap().2, &original);
    assert_eq!(state.incumbent_record().unwrap().cost, zero);
    state.increment_iter();
    state.update_best();
    assert_eq!(state.best_iter(), 0);
    assert_eq!(state.best_counts(), Some(&counts));
    state.select_current();
    state.reset_best();
    state.update_best();
    assert!(state.incumbent_violation().is_none());
    assert_eq!(state.current_violation(), Some(&later));
    state.reset();
    assert!(state.current_violation().is_none());
    assert!(state.incumbent_violation().is_none());

    let mut first = SelectedFirstOrderState::<_, F>::new(x.clone());
    assert!(first.current_violation().is_none());
    assert!(first.incumbent_violation().is_none());
    first
        .replace(x.clone(), zero, gradient.clone(), original.clone())
        .unwrap();
    first.select_current();
    first.replace(x, one, gradient, later.clone()).unwrap();
    first.update_best();
    assert_eq!(first.current_violation(), Some(&later));
    assert_eq!(first.incumbent_violation(), Some(&original));
    assert_eq!(first.current().unwrap().3, &later);
    assert_eq!(first.best().unwrap().2, &original);
    first.reset_best();
    assert!(first.incumbent_violation().is_none());
    assert_eq!(first.current_violation(), Some(&later));
    first.reset();
    assert!(first.current_violation().is_none());
    assert!(first.incumbent_violation().is_none());
}

#[test]
fn vec_records_keep_each_reports_metadata() {
    records(vec![1.0_f64, 2.0], vec![2.0_f64, 4.0]);
    records(vec![1.0_f32, 2.0], vec![2.0_f32, 4.0]);
}

#[cfg(feature = "nalgebra")]
#[test]
fn nalgebra_records_keep_each_reports_metadata() {
    records(
        nalgebra::dvector![1.0_f64, 2.0],
        nalgebra::dvector![2.0_f64, 4.0],
    );
    records(
        nalgebra::dvector![1.0_f32, 2.0],
        nalgebra::dvector![2.0_f32, 4.0],
    );
}

#[cfg(feature = "ndarray")]
#[test]
fn ndarray_records_keep_each_reports_metadata() {
    records(ndarray::array![1.0_f64, 2.0], ndarray::array![2.0_f64, 4.0]);
    records(ndarray::array![1.0_f32, 2.0], ndarray::array![2.0_f32, 4.0]);
}

#[cfg(feature = "faer")]
#[test]
fn faer_records_keep_each_reports_metadata() {
    records(faer::col![1.0_f64, 2.0], faer::col![2.0_f64, 4.0]);
    records(faer::col![1.0_f32, 2.0], faer::col![2.0_f32, 4.0]);
}

#[test]
fn failed_gradient_replacement_preserves_report_and_pending_selection() {
    let mut state = SelectedFirstOrderState::new(vec![0.0]);
    let report = described(1.0);
    state
        .replace(vec![0.0], 0.0, vec![0.0], report.clone())
        .unwrap();
    state.select_current();
    let before = state.clone();
    assert!(
        state
            .replace(
                vec![1.0],
                1.0,
                vec![],
                ViolationReport::new(0.0, ViolationMeasure::L1)
            )
            .is_err()
    );
    assert_eq!(state, before);
    state.update_best();
    assert_eq!(state.incumbent_violation(), Some(&report));
}

fn nonfinite<F: Scalar>() {
    for value in [F::zero(), F::infinity(), F::neg_infinity(), F::nan()] {
        let report = ViolationReport::new(value, ViolationMeasure::L1);
        let mut state = SelectedState::new(0);
        state.replace(0, F::zero(), report.clone());
        state.select_current();
        state.update_best();
        let mut first = SelectedFirstOrderState::new(vec![F::zero()]);
        first
            .replace(vec![F::zero()], F::zero(), vec![F::zero()], report)
            .unwrap();
        first.select_current();
        first.update_best();
        for report in [
            state.current_violation(),
            state.incumbent_violation(),
            first.current_violation(),
            first.incumbent_violation(),
        ] {
            let actual = report.unwrap().value;
            assert!(actual == value || (actual.is_nan() && value.is_nan()));
        }
    }
}

#[test]
fn evaluated_zero_and_nonfinite_values_remain_available() {
    nonfinite::<f32>();
    nonfinite::<f64>();
}

struct ExternalState {
    point: f64,
    report: ViolationReport,
    counts: EvalCounts,
}

impl State for ExternalState {
    type Param = f64;
    type Float = f64;
    fn iter(&self) -> u64 {
        0
    }
    fn increment_iter(&mut self) {}
    fn cost_evals(&self) -> u64 {
        self.counts.cost_evals
    }
    fn param(&self) -> &f64 {
        &self.point
    }
    fn cost(&self) -> f64 {
        1.0
    }
    fn best_param(&self) -> &f64 {
        &self.point
    }
    fn best_cost(&self) -> f64 {
        1.0
    }
    fn best_iter(&self) -> u64 {
        0
    }
    fn best_cost_evals(&self) -> u64 {
        self.counts.cost_evals
    }
    fn update_best(&mut self) {}
    fn reset_best(&mut self) {}
}
impl EvaluatedState for ExternalState {
    fn current_record(&self) -> Option<(&f64, f64)> {
        Some((&self.point, 1.0))
    }
}
impl IncumbentState for ExternalState {
    fn incumbent_record(&self) -> Option<IncumbentRef<'_, f64>> {
        Some(IncumbentRef::new(&self.point, 1.0, 0, &self.counts))
    }
}
impl ConstraintViolationState for ExternalState {
    fn current_violation(&self) -> Option<&ViolationReport> {
        Some(&self.report)
    }
    fn incumbent_violation(&self) -> Option<&ViolationReport> {
        Some(&self.report)
    }
}

#[derive(Default)]
struct Reports(Vec<ViolationReport>);
impl<S: ConstraintViolationState<Float = f64>> Observe<S> for Reports {
    fn observe_init(&mut self, state: &S) {
        self.0.push(state.current_violation().unwrap().clone());
        self.0.push(state.incumbent_violation().unwrap().clone());
    }
}

#[test]
fn generic_observer_reads_shared_and_external_states_without_changing_counts() {
    let report = described(0.5);
    let state = ExternalState {
        point: 1.0,
        report: report.clone(),
        counts: EvalCounts::default(),
    };
    let mut shared = SelectedState::new(1.0);
    shared.replace(1.0, 1.0, report.clone());
    shared.select_current();
    shared.update_best();
    let counts = *shared.counts();
    let mut observer = Reports::default();
    observer.observe_init(&state);
    observer.observe_init(&shared);
    assert_eq!(observer.0, vec![report; 4]);
    assert_eq!(*shared.counts(), counts);
    assert_eq!(state.counts, EvalCounts::default());
}

#[cfg(feature = "serde")]
#[test]
fn serialization_preserves_custom_and_pending_report_metadata() {
    fn round_trip<
        F: Scalar + serde::Serialize + serde::de::DeserializeOwned,
    >() {
        let original = described(F::one());
        let later = ViolationReport::new(F::zero(), ViolationMeasure::LInf)
            .with_transformation(ViolationTransformation::PrimaModeration);
        let mut state = SelectedState::new(vec![F::zero()]);
        state.replace(vec![F::zero()], F::zero(), original.clone());
        state.select_current();
        state.replace(vec![F::one()], F::one(), later.clone());
        let bytes = postcard::to_allocvec(&state).unwrap();
        let mut restored: SelectedState<Vec<F>, F> =
            postcard::from_bytes(&bytes).unwrap();
        assert_eq!(restored, state);
        restored.update_best();
        assert_eq!(restored.current_violation(), Some(&later));
        assert_eq!(restored.incumbent_violation(), Some(&original));

        let mut first = SelectedFirstOrderState::new(vec![F::zero()]);
        first
            .replace(
                vec![F::zero()],
                F::zero(),
                vec![F::zero()],
                original.clone(),
            )
            .unwrap();
        first.select_current();
        first
            .replace(vec![F::one()], F::one(), vec![F::one()], later.clone())
            .unwrap();
        let bytes = postcard::to_allocvec(&first).unwrap();
        let mut restored: SelectedFirstOrderState<Vec<F>, F> =
            postcard::from_bytes(&bytes).unwrap();
        assert_eq!(restored, first);
        restored.update_best();
        assert_eq!(restored.current_violation(), Some(&later));
        assert_eq!(restored.incumbent_violation(), Some(&original));
    }
    round_trip::<f32>();
    round_trip::<f64>();
}
