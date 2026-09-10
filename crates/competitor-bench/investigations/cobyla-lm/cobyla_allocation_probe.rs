//! Allocation diagnostics, compiled separately from timing and trace capture.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

#[derive(Clone, Copy)]
pub enum Phase {
    Setup,
    Iteration,
    Callback,
    Output,
}

thread_local! {
    static PHASE: Cell<Phase> = const { Cell::new(Phase::Setup) };
    static COUNTS: Cell<[[usize; 2]; 4]> = const { Cell::new([[0; 2]; 4]) };
}

struct Counting;

fn record(bytes: usize) {
    let _ = PHASE.try_with(|phase| {
        let _ = COUNTS.try_with(|counts| {
            let mut values = counts.get();
            values[phase.get() as usize][0] += 1;
            values[phase.get() as usize][1] += bytes;
            counts.set(values);
        });
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

pub fn set_phase(phase: Phase) {
    PHASE.set(phase);
}

pub struct Callback(Phase);

pub fn callback() -> Callback {
    Callback(PHASE.replace(Phase::Callback))
}

impl Drop for Callback {
    fn drop(&mut self) {
        set_phase(self.0);
    }
}

pub fn start() {
    set_phase(Phase::Setup);
    COUNTS.set([[0; 2]; 4]);
}

pub fn finish() -> [[usize; 2]; 4] {
    COUNTS.get()
}

pub struct Observer;

impl<S> basin::Observe<S> for Observer {
    fn observe_init(&mut self, _state: &S) {
        set_phase(Phase::Iteration);
    }

    fn observe_final(
        &mut self,
        _state: &S,
        _reason: &basin::TerminationReason,
    ) {
        set_phase(Phase::Output);
    }
}
