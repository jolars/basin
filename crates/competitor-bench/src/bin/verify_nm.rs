//! Verify the published Nelder-Mead workload without timing instrumentation.
//!
//! Run: `cargo run --release -p competitor-bench --bin verify_nm`.
//! For repeated solves under a profiler, add
//! `-- --dimension 12 --library basin --repetitions 10000`.

use std::hint::black_box;

use competitor_bench::nm;

fn main() {
    let mut dimensions = vec![2, 12, 24];
    let mut selected_library = None;
    let mut repetitions = 0_usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args.next().expect("expected an option value");
        match arg.as_str() {
            "--dimension" => dimensions = vec![value.parse().unwrap()],
            "--library" => {
                assert!(matches!(value.as_str(), "basin" | "gomez"));
                selected_library = Some(value);
            }
            "--repetitions" => repetitions = value.parse().unwrap(),
            _ => panic!("unknown option: {arg}"),
        }
    }
    for n in dimensions {
        for library in ["basin", "gomez"] {
            if selected_library
                .as_deref()
                .is_some_and(|name| name != library)
            {
                continue;
            }
            let result = match library {
                "basin" => nm::solve_basin(n),
                _ => nm::solve_gomez::<true>(n),
            };
            let success = result.verify();
            let max_error = result
                .point
                .iter()
                .map(|x| (x - 1.0).abs())
                .fold(0.0_f64, f64::max);
            println!(
                "n={n} {library}: cost={:.12e} max_parameter_error={max_error:.3e} \
                 iterations={} objective_calls={} stop={} success={success}",
                result.cost,
                result.iterations,
                result.objective_calls,
                result.stop,
            );
            for _ in 0..repetitions {
                black_box(match library {
                    "basin" => nm::solve_basin(black_box(n)),
                    _ => nm::solve_gomez::<false>(black_box(n)),
                });
            }
        }
    }
}
