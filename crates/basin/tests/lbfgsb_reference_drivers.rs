//! Reference work and numerical checks also used by the L-BFGS-B benchmark.

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
