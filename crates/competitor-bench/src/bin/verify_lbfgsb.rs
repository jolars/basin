//! Verify the published bounded comparison; use `--calibrate` to size it.

use competitor_bench::{
    convergence,
    lbfgsb::{self, Library},
};

fn main() {
    assert!(!cfg!(feature = "parallel") && !cfg!(feature = "basin-latest"));
    let calibrate = std::env::args().nth(1).as_deref() == Some("--calibrate");
    let mut n = if calibrate { 20_000 } else { lbfgsb::DIMENSION };
    loop {
        let mut shortest = u128::MAX;
        for library in [Library::Basin, Library::Reference] {
            let mut samples = Vec::new();
            let run = convergence::median_run(
                || {
                    let run = lbfgsb::run_logged(library, n, !calibrate);
                    samples.push(run.points.last().unwrap().0);
                    run
                },
                11,
            );
            samples.remove(0);
            samples.sort_unstable();
            let elapsed = run.points.last().unwrap().0;
            shortest = shortest.min(elapsed);
            println!(
                "n={n} {}: {:.3} ms, f={:.6e}, pg={:.3e}, violation={:.3e}, iterations={}, objective/gradient calls={}/{}, status={}, verified={}",
                library.name(),
                elapsed as f64 / 1e6,
                run.summary.cost,
                run.summary.projected_gradient.unwrap(),
                run.summary.violation,
                run.summary.iterations.unwrap(),
                run.summary.objective_calls,
                run.summary.gradient_calls,
                run.summary.status,
                run.summary.verified
            );
            println!(
                "  11 solve times: {} ms",
                samples
                    .iter()
                    .map(|&t| format!("{:.3}", t as f64 / 1e6))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            assert!(run.summary.verified);
            assert_eq!(run.summary.status, "ProjectedGradientTolerance");
        }
        if !calibrate || shortest >= 20_000_000 {
            break;
        }
        n = n.checked_mul(2).unwrap();
    }
}
