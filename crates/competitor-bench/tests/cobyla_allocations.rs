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
            (-1.031_628_452_721_29, 50, 50, 41, 300, 8000),
            (5.725_381_469_125_41e-9, 200, 201, 148, 900, 120000),
            (0.124_999_997_365_822, 61, 61, 50, 350, 10000),
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
