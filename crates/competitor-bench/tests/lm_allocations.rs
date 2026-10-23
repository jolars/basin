//! Allocation guards for repeated QR solves on an unchanged linear model.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

use basin::{DenseMatrix, FactorizePivotedQr, RegularizedQrSolve};
use nalgebra::{DMatrix, DVector};

thread_local! {
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

fn check_undamped<M, V>(matrix: M, rhs: V, values: impl Fn(&V) -> &[f64])
where
    M: FactorizePivotedQr<V>,
{
    let factor = matrix.factorize_pivoted_qr(&rhs).unwrap();
    let diagonal = factor.column_norms_squared();
    let before = REQUESTS.get();
    for _ in 0..100 {
        let x = factor.solve_regularized(0., &diagonal, None).unwrap();
        for (&actual, expected) in values(&x).iter().zip([1., 2., 3.]) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
    let requests = REQUESTS.get() - before;
    // Each solve returns an owned vector; an undamped triangular factor needs
    // no additional damping workspace or copied matrix.
    assert!(requests <= 100, "{requests} allocations for 100 solves");
}

#[test]
fn vec_undamped_solves_allocate_only_the_returned_vector() {
    check_undamped(
        DenseMatrix::from_row_slice(
            3,
            3,
            &[0., 0., 100., 2., 0., 0., 0., 10., 0.],
        ),
        vec![300., 2., 20.],
        |x| x.as_slice(),
    );
}

#[test]
fn nalgebra_undamped_solves_allocate_only_the_returned_vector() {
    check_undamped(
        DMatrix::from_row_slice(3, 3, &[0., 0., 100., 2., 0., 0., 0., 10., 0.]),
        DVector::from_vec(vec![300., 2., 20.]),
        DVector::as_slice,
    );
}
