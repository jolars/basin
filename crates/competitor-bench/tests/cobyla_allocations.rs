//! Deterministic allocation ceilings complement the driver's timing benchmark.

pub use basin::core;

#[path = "../benches/support/cobyla.rs"]
mod cobyla;

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

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
fn driver_allocations_stay_bounded_without_changing_numerical_results() {
    // Leave room for modest workspace changes while rejecting per-iteration
    // scratch allocation and eager allocation of the 2,000-point filter.
    for ((case, name), (cost, nf, nc, iterations, max_requests, max_bytes)) in
        cobyla::CASES.into_iter().zip([
            (-1.031_628_452_721_29, 50, 50, 41, 140, 5000),
            (5.725_381_469_125_41e-9, 200, 201, 148, 340, 80000),
            (0.124_999_997_365_822, 61, 61, 50, 160, 6500),
        ])
    {
        REQUESTS.set((0, 0));
        let result = case.solve();
        let (requests, bytes) = REQUESTS.get();
        eprintln!("{name}: {requests} allocation requests, {bytes} bytes");
        assert!(
            requests <= max_requests,
            "{name}: {requests} > {max_requests}"
        );
        assert!(bytes <= max_bytes, "{name}: {bytes} > {max_bytes}");
        assert_eq!(result.evaluations, nf, "{name}");
        assert_eq!(result.constraint_calls, nc, "{name}");
        assert_eq!(result.iterations, iterations, "{name}");
        assert!((result.cost - cost).abs() < 1e-13, "{name}");
        assert!(result.point.iter().all(|x| x.is_finite()));
    }
}

#[test]
fn public_quadratic_iterations_reuse_trial_buffers() {
    use basin::{
        Cobyla, CobylaState, CostFunction, Executor,
        NonlinearInequalityConstraints, Observe, ObserverMode,
        TerminationReason,
    };
    use std::{convert::Infallible, rc::Rc};

    #[derive(Default)]
    struct Counts {
        callbacks: Cell<usize>,
        initial_callbacks: Cell<usize>,
        start: Cell<(usize, usize)>,
        end: Cell<(usize, usize)>,
    }
    struct Quadratic(Rc<Counts>);
    impl CostFunction for Quadratic {
        type Param = Vec<f64>;
        type Output = f64;
        type Error = Infallible;
        fn cost(&self, x: &Vec<f64>) -> Result<f64, Infallible> {
            Ok((x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2))
        }
    }
    impl NonlinearInequalityConstraints for Quadratic {
        fn num_constraints(&self) -> usize {
            5
        }
        fn constraints(&self, x: &Vec<f64>) -> Result<Vec<f64>, Infallible> {
            self.0.callbacks.set(self.0.callbacks.get() + 1);
            Ok(vec![
                -(1.5 - x[0] - x[1]),
                -x[0],
                x[0] - 2.0,
                -x[1],
                x[1] - 2.0,
            ])
        }
    }
    struct AllocationObserver(Rc<Counts>);
    impl Observe<CobylaState<Vec<f64>>> for AllocationObserver {
        fn observe_init(&mut self, _: &CobylaState<Vec<f64>>) {
            self.0.initial_callbacks.set(self.0.callbacks.get());
            self.0.start.set(REQUESTS.get());
        }
        fn observe_final(
            &mut self,
            _: &CobylaState<Vec<f64>>,
            _: &TerminationReason,
        ) {
            self.0.end.set(REQUESTS.get());
        }
    }

    let counts = Rc::new(Counts::default());
    let result = Executor::from_start(
        Quadratic(counts.clone()),
        Cobyla::new()
            .with_rho_beg(0.5)
            .with_rho_end(f64::EPSILON.sqrt() * 0.5),
        vec![0.5, 0.5],
    )
    .max_iter(1000)
    .observe_with(AllocationObserver(counts.clone()), ObserverMode::Never)
    .run()
    .unwrap();
    let (start_requests, start_bytes) = counts.start.get();
    let (end_requests, end_bytes) = counts.end.get();
    let calls = counts.callbacks.get() - counts.initial_callbacks.get();
    // Each constraint callback and its public adapter allocate one five-element
    // vector each. Eight resolution reductions allocate fcratio's two row-range
    // vectors. Reject renewed per-iteration trial or geometry-step allocation.
    assert!(end_requests - start_requests <= 2 * calls + 16);
    assert!(end_bytes - start_bytes <= (2 * calls + 16) * 5 * size_of::<f64>());
    assert_eq!(result.reason, TerminationReason::SolverConverged);
    assert_eq!(result.cost_evals(), 61);
    assert_eq!(counts.callbacks.get(), 61);
    assert!((result.best_cost() - 0.124_999_997_365_822).abs() < 1e-13);
    assert!(
        result.best_param().iter().sum::<f64>() - 1.5 <= f64::EPSILON.sqrt()
    );
}
