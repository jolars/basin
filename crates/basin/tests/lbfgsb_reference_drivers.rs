//! Reference work and numerical checks also used by the L-BFGS-B benchmark.

#[path = "support/backend_aliases.rs"]
mod backend_aliases;
#[path = "../benches/support/lbfgsb.rs"]
mod support;

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

thread_local! {
    // Per-thread accounting excludes concurrent tests and harness activity.
    static REQUESTS: Cell<usize> = const { Cell::new(0) };
}

struct Counting;

fn record() {
    let _ = REQUESTS.try_with(|count| count.set(count.get() + 1));
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(
        &self,
        ptr: *mut u8,
        layout: Layout,
        size: usize,
    ) -> *mut u8 {
        record();
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[test]
fn reference_drivers_preserve_accepted_work_and_feasibility() {
    for number in 1..=3 {
        let mut problem = support::Driver::new(number);
        problem.check_feasibility = true;
        problem.verify(&problem.solve());
    }
}

#[test]
fn reference_driver_allocations_stay_bounded() {
    // Allow a few extra workspace allocations, but reject per-iteration
    // history views and duplicate direction, gradient, and incumbent vectors.
    for (number, ceiling) in [(1, 140), (2, 235), (3, 250)] {
        let problem = support::Driver::new(number);
        REQUESTS.set(0);
        let result = problem.solve();
        let requests = REQUESTS.get();
        problem.verify(&result);
        assert!(
            requests <= ceiling,
            "driver{number}: {requests} allocation requests exceed {ceiling}"
        );
    }
}

#[cfg(any(
    feature = "faer_all",
    feature = "nalgebra_all",
    feature = "ndarray_all"
))]
fn check_backend<V: support::Vector>()
where
    basin::Lbfgsb: for<'a> basin::Solver<
            &'a support::Driver<V>,
            basin::LbfgsState<V>,
            Error = std::convert::Infallible,
        >,
{
    use basin::GradientState;
    for number in 1..=3 {
        let mut vec = support::Driver::new(number);
        let mut backend = support::Driver::<V>::with_backend(number);
        vec.check_feasibility = true;
        backend.check_feasibility = true;
        let vec_result = vec.solve();
        let backend_result = backend.solve();
        vec.verify(&vec_result);
        backend.verify(&backend_result);
        for (&a, &b) in vec_result
            .param()
            .iter()
            .zip(backend_result.param().as_slice())
        {
            assert!((a - b).abs() <= 1e-12 * (1.0 + a.abs()));
        }
        for (&a, &b) in vec_result
            .state
            .gradient()
            .unwrap()
            .iter()
            .zip(backend_result.state.gradient().unwrap().as_slice())
        {
            assert!((a - b).abs() <= 1e-12 * (1.0 + a.abs()));
        }
    }
}

#[cfg(feature = "faer_all")]
#[test]
fn faer_reference_drivers_match_vec_work_and_results() {
    check_backend::<backend_aliases::faer::Col<f64>>();
    check_short_backend::<backend_aliases::faer::Col<f64>>();
}

#[cfg(feature = "nalgebra_all")]
#[test]
fn nalgebra_reference_drivers_match_vec_work_and_results() {
    check_backend::<backend_aliases::nalgebra::DVector<f64>>();
    check_short_backend::<backend_aliases::nalgebra::DVector<f64>>();
}

#[cfg(feature = "ndarray_all")]
#[test]
fn ndarray_reference_drivers_match_vec_work_and_results() {
    check_backend::<backend_aliases::ndarray::Array1<f64>>();
    check_short_backend::<backend_aliases::ndarray::Array1<f64>>();
}

#[test]
fn short_cases_preserve_reference_work_and_adapter_results() {
    check_short_backend::<Vec<f64>>();
}

fn check_short_backend<V: support::Vector>()
where
    basin::Lbfgsb: for<'a> basin::Solver<
            &'a support::short::Short<V>,
            basin::LbfgsState<V>,
            Error = std::convert::Infallible,
        >,
{
    use support::short::{Adapter, Case, Short};
    for case in [Case::Mixed, Case::Rollover] {
        let mut problem = Short::<V>::new(case);
        problem.check_feasibility = true;
        for adapter in
            [Adapter::Reference, Adapter::Trimmed, Adapter::RequiredStops]
        {
            let result = problem.solve(adapter);
            problem.verify(&result);
            let initialized = problem.initialize(adapter);
            assert_eq!(initialized.counts().cost_evals, 1);
            assert_eq!(initialized.counts().gradient_evals, 1);
            let resumed = initialized.run_to_end().unwrap();
            problem.verify(&resumed);
            assert_eq!(
                problem.extract(&result).0.as_slice(),
                resumed.param().as_slice()
            );
        }
    }
}

#[test]
fn short_case_allocations_separate_setup_iterations_and_extraction() {
    use support::short::{Adapter, Case, Short};
    for (case, ceiling) in [(Case::Mixed, 36), (Case::Rollover, 64)] {
        let problem = Short::<Vec<f64>>::new(case);
        for adapter in
            [Adapter::Reference, Adapter::Trimmed, Adapter::RequiredStops]
        {
            REQUESTS.set(0);
            let initialized = problem.initialize(adapter);
            let setup = REQUESTS.get();
            let result = initialized.run_to_end().unwrap();
            let iterations = REQUESTS.get() - setup;
            let extracted = problem.extract(&result);
            let extraction = REQUESTS.get() - setup - iterations;
            assert_eq!(extraction, 1);
            let total = setup + iterations + extraction;
            let ceiling = ceiling
                - match (case, adapter) {
                    (_, Adapter::Reference) => 0,
                    (Case::Mixed, Adapter::RequiredStops) => 4,
                    _ => 2,
                };
            assert!(total <= ceiling, "{total} allocations exceed {ceiling}");
            assert_eq!(extracted.0, result.param().as_slice());
            problem.verify(&result);
            eprintln!(
                "{} {}: setup={setup}, iterations={iterations}, extraction={extraction}, total={total}",
                case.name(),
                adapter.name()
            );
        }
    }
}
