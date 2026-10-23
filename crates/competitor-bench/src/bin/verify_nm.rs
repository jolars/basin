//! Verify the published Nelder-Mead workload without timing instrumentation.
//!
//! Run: `cargo run --release -p competitor-bench --bin verify_nm`.

use competitor_bench::nm;

fn main() {
    for n in [2, 12, 24] {
        for (library, result) in [
            ("basin", nm::solve_basin(n)),
            ("gomez", nm::solve_gomez::<true>(n)),
        ] {
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
        }
    }
}
