//! Numerical validation and work accounting, deliberately separate from timings.

#[path = "../../benches/support/numdiff.rs"]
mod support;

use std::{
    alloc::{GlobalAlloc, Layout, System},
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed},
    },
};

use basin::{CostFunction, Gradient, MatrixIndex, Residual};
use support::{BenchVector, Fixture, Function};

static RECORDING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static REALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

fn record(bytes: usize, counter: &AtomicUsize) {
    // Global atomics include allocations made by Rayon workers during a request.
    if RECORDING.load(Relaxed) {
        counter.fetch_add(1, Relaxed);
        BYTES.fetch_add(bytes, Relaxed);
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            record(layout.size(), &ALLOCATIONS);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            record(layout.size(), &ALLOCATIONS);
        }
        ptr
    }

    unsafe fn realloc(
        &self,
        ptr: *mut u8,
        layout: Layout,
        size: usize,
    ) -> *mut u8 {
        let result = unsafe { System.realloc(ptr, layout, size) };
        if !result.is_null() {
            record(size, &REALLOCATIONS);
        }
        result
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Default)]
struct Callbacks {
    cost: AtomicUsize,
    gradient: AtomicUsize,
    gradient_into: AtomicUsize,
    residual: AtomicUsize,
    residual_into: AtomicUsize,
}

impl Callbacks {
    fn reset(&self) {
        self.cost.store(0, Relaxed);
        self.gradient.store(0, Relaxed);
        self.gradient_into.store(0, Relaxed);
        self.residual.store(0, Relaxed);
        self.residual_into.store(0, Relaxed);
    }

    fn snapshot(&self) -> [usize; 5] {
        [
            self.cost.load(Relaxed),
            self.residual.load(Relaxed),
            self.residual_into.load(Relaxed),
            self.gradient.load(Relaxed),
            self.gradient_into.load(Relaxed),
        ]
    }
}

#[derive(Clone)]
struct Counted<P> {
    inner: P,
    callbacks: Arc<Callbacks>,
}

impl<V: BenchVector> CostFunction for Counted<Function<V>> {
    type Param = V;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &V) -> Result<f64, Infallible> {
        self.callbacks.cost.fetch_add(1, Relaxed);
        self.inner.cost(x)
    }
}

impl<V: BenchVector> Residual for Counted<Function<V>> {
    type Param = V;
    type Output = V;
    type Error = Infallible;

    fn residual(&self, x: &V) -> Result<V, Infallible> {
        self.callbacks.residual.fetch_add(1, Relaxed);
        self.inner.residual(x)
    }

    fn residual_into(&self, x: &V, out: &mut V) -> Result<(), Infallible> {
        self.callbacks.residual_into.fetch_add(1, Relaxed);
        self.inner.residual_into(x, out)
    }
}

impl<V: BenchVector> Gradient for Counted<Function<V>> {
    type Gradient = V;

    fn gradient(&self, x: &V) -> Result<V, Infallible> {
        self.callbacks.gradient.fetch_add(1, Relaxed);
        self.inner.gradient(x)
    }

    fn gradient_into(&self, x: &V, out: &mut V) -> Result<(), Infallible> {
        self.callbacks.gradient_into.fetch_add(1, Relaxed);
        self.inner.gradient_into(x, out)
    }
}

struct AllocationScope;

impl AllocationScope {
    fn start() -> Self {
        ALLOCATIONS.store(0, Relaxed);
        REALLOCATIONS.store(0, Relaxed);
        BYTES.store(0, Relaxed);
        assert!(!RECORDING.swap(true, Relaxed));
        Self
    }
}

impl Drop for AllocationScope {
    fn drop(&mut self) {
        RECORDING.store(false, Relaxed);
    }
}

struct Options {
    filter: String,
    repeats: usize,
}

impl Options {
    fn parse() -> Self {
        let mut options = Self {
            filter: String::new(),
            repeats: 5,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--filter" => {
                    options.filter =
                        args.next().expect("--filter needs a substring")
                }
                "--repeats" => {
                    options.repeats = args
                        .next()
                        .expect("--repeats needs a count")
                        .parse()
                        .expect("invalid repeat count");
                    assert!(
                        options.repeats > 0,
                        "repeat count must be positive"
                    );
                }
                "--help" | "-h" => {
                    println!(
                        "verify_numdiff [--filter SUBSTRING] [--repeats COUNT]\n\nValidates analytic derivatives, then prints separate callback and allocation\nmeasurements as CSV totals over COUNT repeated calls (default: 5).\nSee crates/competitor-bench/NUMDIFF.md for the measurement protocol."
                    );
                    std::process::exit(0);
                }
                _ => panic!("unknown option: {arg}"),
            }
        }
        options
    }
}

fn backend<V>(options: &Options, mode: &str, backend: &str) -> usize
where
    V: BenchVector,
    V::Matrix: MatrixIndex,
{
    let mut checked = 0;
    for case in support::cases() {
        for &call in case.calls() {
            let name =
                format!("numdiff/{backend}/{}/{}", case.name(), call.name());
            if !name.contains(&options.filter) {
                continue;
            }

            let callbacks = Arc::new(Callbacks::default());
            let counted = Counted {
                inner: Function::<V>::new(case),
                callbacks: callbacks.clone(),
            };
            let mut fixture = Fixture::<V, _>::new(case, counted);
            fixture.run(call);
            fixture.verify();
            callbacks.reset();
            fixture.run(call);
            let per_call = callbacks.snapshot();
            callbacks.reset();
            for _ in 0..options.repeats {
                fixture.run(call);
            }
            let counts = callbacks.snapshot();
            assert_eq!(
                counts,
                per_call.map(|count| count * options.repeats),
                "{name}: callback count changed on repetition"
            );
            fixture.verify();

            // Measure allocations using the same uncounted callbacks as the timer.
            let mut fixture = Fixture::<V, _>::new(case, Function::new(case));
            fixture.run(call);
            fixture.verify();
            let scope = AllocationScope::start();
            for _ in 0..options.repeats {
                fixture.run(call);
            }
            drop(scope);
            let allocations = ALLOCATIONS.load(Relaxed);
            let reallocations = REALLOCATIONS.load(Relaxed);
            let bytes = BYTES.load(Relaxed);
            fixture.verify();
            println!(
                "{mode},{name},{},{},{},{},{},{},{allocations},{reallocations},{bytes}",
                options.repeats,
                counts[0],
                counts[1],
                counts[2],
                counts[3],
                counts[4]
            );
            checked += 1;
        }
    }
    checked
}

fn main() {
    let options = Options::parse();
    let mode = support::execution();
    #[cfg(feature = "parallel")]
    rayon::broadcast(|_| {});
    eprintln!(
        "verify_numdiff: {mode} (f64), totals over {} calls",
        options.repeats
    );
    println!(
        "execution,benchmark,repeats,cost_calls,residual_calls,residual_into_calls,gradient_calls,gradient_into_calls,alloc_calls,realloc_calls,requested_bytes"
    );
    let checked = backend::<Vec<f64>>(&options, &mode, "vec")
        + backend::<support::NalgebraVector>(&options, &mode, "nalgebra")
        + backend::<ndarray::Array1<f64>>(&options, &mode, "ndarray")
        + backend::<faer::Col<f64>>(&options, &mode, "faer");
    assert!(checked > 0, "filter matched no cases");
    eprintln!("verified {checked} cases");
}
