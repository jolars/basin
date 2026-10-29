//! Verify all published CMA-ES seeds; `--untraced` disables intermediate samples.
//! Investigation options: `--dimension N`, `--eigen-update-period N`,
//! `--generations N` (fixed work), and `--repetitions N` (positive and odd).

use competitor_bench::{
    cmaes::{self, Case, Configuration, Library},
    convergence,
};

fn main() {
    assert!(!cfg!(feature = "parallel") && !cfg!(feature = "basin-latest"));
    let mut tracing = true;
    let mut config = Configuration::default();
    let mut repetitions = 11;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--untraced" => tracing = false,
            "--dimension" => {
                config.dimension = args.next().unwrap().parse().unwrap()
            }
            "--eigen-update-period" => {
                config.eigen_update_period =
                    args.next().unwrap().parse().unwrap()
            }
            "--generations" => {
                config.generations = Some(args.next().unwrap().parse().unwrap())
            }
            "--repetitions" => {
                repetitions = args.next().unwrap().parse().unwrap()
            }
            _ => panic!("unknown argument: {arg}"),
        }
    }
    eprintln!("{config:?}, repetitions={repetitions}, tracing={tracing}");
    eprintln!(
        "reference eigen update period={}",
        cmaes::reference_eigen_update_period(config.dimension)
    );
    for case in [Case::RotatedEllipsoid, Case::Ackley] {
        for seed in cmaes::SEEDS {
            for library in [Library::Basin, Library::Reference] {
                let mut times = Vec::new();
                let run = convergence::median_run(
                    || {
                        let run = cmaes::run_configured(
                            library, case, seed, config, tracing,
                        );
                        times.push(run.points.last().unwrap().0);
                        run
                    },
                    repetitions,
                );
                times.remove(0);
                times.sort_unstable();
                eprintln!(
                    "{}/{} seed={seed} samples_ns={times:?}",
                    case.name(),
                    library.name()
                );
                println!(
                    "{}/{} seed={seed}: {:.3} ms (range {:.3}–{:.3}), f={:.6e}, generations={}, calls={}, status={}, verified={}",
                    case.name(),
                    library.name(),
                    times[times.len() / 2] as f64 / 1e6,
                    times[0] as f64 / 1e6,
                    *times.last().unwrap() as f64 / 1e6,
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
