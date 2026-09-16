//! Guard the temporary-allocation cost of the published SLSQP case.

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
fn rosenbrock_allocations_preserve_the_reference_work() {
    use competitor_bench::slsqp::{Library, run};

    REQUESTS.set((0, 0));
    let result = run(Library::Basin);
    let (requests, bytes) = REQUESTS.get();
    eprintln!("{requests} allocation requests, {bytes} requested bytes");
    result.verify();
    assert_eq!(result.cost_evals, 49);
    assert_eq!(result.gradient_evals, 36);
    assert!((result.cost - 1.543_632_681_658_564_5e-16).abs() < 1e-25);
    // Include callbacks and trace storage, but reject renewed scratch copies
    // throughout the least-squares reduction and BFGS update.
    assert!(requests <= 800, "{requests} allocation requests");
}
