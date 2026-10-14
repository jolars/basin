//! Guard the temporary-allocation cost of the published SLSQP case.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

use competitor_bench::slsqp::workloads;

thread_local! {
    // Per-thread accounting excludes concurrent tests and harness activity.
    static REQUESTS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}

struct Counting;
fn record(bytes: usize) {
    let _ = REQUESTS.try_with(|counts| {
        let (requests, total) = counts.get();
        counts.set((requests + 1, total + bytes));
    });
}
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(
        &self,
        ptr: *mut u8,
        layout: Layout,
        size: usize,
    ) -> *mut u8 {
        record(size);
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[test]
fn rosenbrock_allocations_preserve_the_reference_work() {
    use basin::{
        Executor, SelectedFirstOrderState, Slsqp, problems::Rosenbrock,
    };
    use competitor_bench::slsqp::{Library, run};
    use competitor_bench::unconstrained::Unconstrained;

    // The comparison harness retains only the report's code. Obtain the actual
    // owned evidence from the same public problem and stopping configuration.
    let reference = Executor::new(
        Unconstrained(Rosenbrock::<Vec<f64>>::new()),
        Slsqp::new().with_absolute_accuracy_tolerance(
            competitor_bench::slsqp::ACCURACY,
        ),
        SelectedFirstOrderState::new(competitor_bench::slsqp::START.to_vec()),
    )
    .max_iter(competitor_bench::slsqp::BUDGET as u64)
    .run()
    .unwrap();
    assert_eq!(reference.cost_evals(), 49);
    assert_eq!(reference.counts.gradient_evals, 36);
    assert!(
        (reference.best_cost() - 1.543_632_681_658_564_5e-16).abs() < 1e-25
    );
    let before_copy = REQUESTS.get();
    let copied = std::hint::black_box(reference.report.clone());
    let after_copy = REQUESTS.get();
    let report_requests = after_copy.0 - before_copy.0;
    let report_bytes = after_copy.1 - before_copy.1;
    drop(copied);

    REQUESTS.set((0, 0));
    let result = run(Library::Basin);
    let (requests, bytes) = REQUESTS.get();
    eprintln!(
        "{requests} allocation requests, {bytes} requested bytes; one report copy: {report_requests} requests, {report_bytes} bytes"
    );
    result.verify();
    assert_eq!(&result.x, reference.best_param());
    assert_eq!(result.cost, reference.best_cost());
    assert_eq!(result.cost_evals, 49);
    assert_eq!(result.gradient_evals, 36);
    assert!((result.cost - 1.543_632_681_658_564_5e-16).abs() < 1e-25);
    // Include callbacks and trace storage, but reject renewed scratch copies
    // throughout the least-squares reduction and BFGS update. Final reporting
    // owns evidence in SLSQP, the returned report, and the stepper's stop record.
    assert!(
        requests <= 140 + 3 * report_requests,
        "{requests} allocation requests"
    );
}

#[test]
fn hs71_allocations_preserve_constraints_and_reference_work() {
    use basin::{RawEvaluationState, State};

    REQUESTS.set((0, 0));
    let result = workloads::hs71();
    let (requests, bytes) = REQUESTS.get();
    eprintln!("HS71: {requests} allocation requests, {bytes} requested bytes");
    workloads::verify_hs71(&result);
    assert_eq!(result.state.iter(), 5);
    let counts = result.state.raw_counts();
    assert_eq!(counts.cost_evals, 6);
    assert_eq!(counts.gradient_evals, 6);
    assert_eq!(counts.residual_evals, 12);
    assert_eq!(counts.jacobian_evals, 6);
    assert!(requests <= 250, "{requests} allocation requests");
}
