//! Verify all published CMA-ES seeds; `--untraced` disables intermediate samples.

use competitor_bench::{
    cmaes::{self, Case, Library},
    convergence,
};

fn main() {
    assert!(!cfg!(feature = "parallel") && !cfg!(feature = "basin-latest"));
    let tracing = std::env::args().nth(1).as_deref() != Some("--untraced");
    for case in [Case::RotatedEllipsoid, Case::Ackley] {
        for seed in cmaes::SEEDS {
            for library in [Library::Basin, Library::Reference] {
                let mut times = Vec::new();
                let run = convergence::median_run(
                    || {
                        let run =
                            cmaes::run_logged(library, case, seed, tracing);
                        times.push(run.points.last().unwrap().0);
                        run
                    },
                    11,
                );
                times.remove(0);
                times.sort_unstable();
                println!(
                    "{}/{} seed={seed}: {:.3} ms (range {:.3}–{:.3}), f={:.6e}, generations={}, calls={}, status={}, verified={}",
                    case.name(),
                    library.name(),
                    times[5] as f64 / 1e6,
                    times[0] as f64 / 1e6,
                    times[10] as f64 / 1e6,
                    run.summary.cost,
                    run.summary.iterations.unwrap(),
                    run.summary.objective_calls,
                    run.summary.status,
                    run.summary.verified,
                );
            }
        }
    }
}
