#[path = "support/solver_observer.rs"]
mod support;
use basin::{ObserverMode, StepOutcome};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    hint::black_box,
};
use support::*;
thread_local! { static REQUESTS: Cell<usize> = const { Cell::new(0) }; }
struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = REQUESTS.try_with(|n| n.set(n.get() + 1));
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let _ = REQUESTS.try_with(|n| n.set(n.get() + 1));
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(
        &self,
        ptr: *mut u8,
        layout: Layout,
        size: usize,
    ) -> *mut u8 {
        let _ = REQUESTS.try_with(|n| n.set(n.get() + 1));
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;
#[test]
fn observation_borrows_workspace_without_allocations_during_steps() {
    let mut stepper = executor(Scenario::Normal)
        .max_iter(100)
        .observe_solver(
            |state, solver, event| {
                black_box((state, solver.diagnostic, &solver.workspace, event));
            },
            ObserverMode::Always,
        )
        .into_stepper()
        .unwrap();
    REQUESTS.set(0);
    for _ in 0..100 {
        assert_eq!(stepper.step().unwrap(), StepOutcome::Continue);
    }
    let requests = REQUESTS.get();
    assert_eq!(requests, 0);
    // Publishing the final owned report may allocate; observation borrows it.
    assert!(matches!(stepper.step().unwrap(), StepOutcome::Stopped(_)));
    assert_eq!(stepper.counts().cost_evals, 101);
}
