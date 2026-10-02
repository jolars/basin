//! Expanded convergence fixtures. See `dev/convergence-defaults/coverage.md`.

use basin::{
    FirstOrderState, Lbfgsb, LevenbergMarquardt, NelderMead, PointState,
    SelectedFirstOrderState, Sgd, SimplexProgress, Slsqp,
    TrustRegionReflective,
};
use competitor_bench::convergence::{self, Fixture, constraints, trace};

struct Args {
    list: bool,
    final_only: bool,
    suite: String,
    precision: String,
    case: Option<String>,
    validation: bool,
    max_iter: u64,
    max_passes: u64,
    seeds: u64,
}
impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = Self {
            list: false,
            final_only: false,
            suite: "all".into(),
            precision: "both".into(),
            case: None,
            validation: false,
            max_iter: 200,
            max_passes: 2000,
            seeds: 3,
        };
        let mut words = std::env::args().skip(1);
        while let Some(word) = words.next() {
            match word.as_str() {
                "--list" => args.list = true,
                "--final-only" => args.final_only = true,
                "--include-validation" => args.validation = true,
                "--suite" => {
                    args.suite = words.next().ok_or("missing suite")?
                }
                "--precision" => {
                    args.precision = words.next().ok_or("missing precision")?
                }
                "--case" => {
                    args.case = Some(words.next().ok_or("missing case")?)
                }
                "--max-iter" => {
                    args.max_iter = words
                        .next()
                        .ok_or("missing max-iter")?
                        .parse()
                        .map_err(|_| "invalid max-iter")?
                }
                "--max-passes" => {
                    args.max_passes = words
                        .next()
                        .ok_or("missing max-passes")?
                        .parse()
                        .map_err(|_| "invalid max-passes")?
                }
                "--seeds" => {
                    args.seeds = words
                        .next()
                        .ok_or("missing seeds")?
                        .parse()
                        .map_err(|_| "invalid seeds")?
                }
                "--help" => {
                    println!(
                        "convergence_suite [--list] [--suite all|nist|analytic|constraints|stochastic] [--precision both|f32|f64] [--case NAME] [--include-validation] [--max-iter 200] [--max-passes 2000] [--seeds 3] [--final-only]"
                    );
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown option {word}")),
            }
        }
        if !matches!(
            args.suite.as_str(),
            "all" | "nist" | "analytic" | "constraints" | "stochastic"
        ) {
            return Err("invalid suite".into());
        }
        if !matches!(args.precision.as_str(), "both" | "f32" | "f64") {
            return Err("invalid precision".into());
        }
        if args.max_passes == 0 || args.seeds == 0 {
            return Err("max-passes and seeds must be positive".into());
        }
        Ok(args)
    }
    fn includes(&self, suite: &str, name: &str, validation: bool) -> bool {
        (self.suite == "all" || self.suite == suite)
            && (self.list || self.validation || !validation)
            && self.case.as_deref().is_none_or(|filter| filter == name)
    }
}

struct Identity<'a> {
    suite: &'a str,
    name: &'a str,
    family: &'a str,
    validation: bool,
    precision: &'a str,
    start: usize,
    solver: &'a str,
    seed: Option<u64>,
}
impl Identity<'_> {
    fn prefix(&self) -> Vec<String> {
        vec![
            self.suite.into(),
            self.name.into(),
            self.family.into(),
            partition(self.validation).into(),
            self.precision.into(),
            "Vec".into(),
            (self.start + 1).to_string(),
            self.solver.into(),
            self.seed.map(|s| s.to_string()).unwrap_or_default(),
        ]
    }
}

fn partition(validation: bool) -> &'static str {
    if validation {
        "validation"
    } else {
        "development"
    }
}

fn stochastic_case(case: &convergence::Case) -> bool {
    matches!(&case.model, convergence::Model::Quadratic(q) if q.noise > 0.0 && q.condition == 1.0)
}

macro_rules! define_runner {
    ($run:ident, $f:ty) => {
        fn $run(args: &Args) -> usize {
            let mut selected = 0;
            for case in convergence::cases() {
                let deterministic = args.includes(case.suite(), &case.name, case.validation);
                let stochastic = args.includes("stochastic", &case.name, case.validation) && stochastic_case(&case);
                if !deterministic && !stochastic { continue; }
                selected += 1;
                for (start_index, start) in case.starts.iter().enumerate() {
                    let start: Vec<$f> = start.iter().map(|&x| x as $f).collect();
                    macro_rules! solve {
                        ($suite:expr, $name:expr, $solver:expr, $state:expr, $seed:expr) => {{
                            let problem = Fixture::<$f>::new(case.clone());
                            let work = problem.work.clone();
                            let rows = trace::run(problem, $solver, $state, work, args.max_iter, args.max_passes,
                                |p| trace::vector_quality(&case, p));
                            let id = Identity { suite: $suite, name: &case.name, family: case.family,
                                validation: case.validation, precision: stringify!($f), start: start_index,
                                solver: $name, seed: $seed };
                            trace::print_rows(&id.prefix(), &rows, args.final_only);
                        }};
                    }
                    if deterministic {
                        solve!(case.suite(), "NelderMead", NelderMead::new(), SimplexProgress::new(start.clone()), None);
                        solve!(case.suite(), "Lbfgsb", Lbfgsb::new(), FirstOrderState::new(start.clone()), None);
                        solve!(case.suite(), "LevenbergMarquardt", LevenbergMarquardt::<_, _, $f>::default(), PointState::<_, $f>::new(start.clone()), None);
                        solve!(case.suite(), "TrustRegionReflective", TrustRegionReflective::new(), PointState::new(start.clone()), None);
                    }
                    if stochastic {
                        for seed in 0..args.seeds {
                            solve!("stochastic", "Sgd", Sgd::new(0.02 as $f, 1, seed), PointState::new(start.clone()), Some(seed));
                        }
                    }
                }
            }
            for case in constraints::cases() {
                if !args.includes("constraints", case.name, case.validation) { continue; }
                selected += 1;
                for (start_index, start) in case.starts.iter().enumerate() {
                    let problem = constraints::Fixture::<$f>::new(case.clone());
                    let work = problem.work.clone();
                    let start = start.iter().map(|&x| x as $f).collect::<Vec<_>>();
                    let rows = trace::run(problem, Slsqp::new(), SelectedFirstOrderState::new(start), work,
                        args.max_iter, args.max_passes, |p| {
                            let param: Vec<_> = p.iter().map(|&x| x as f64).collect();
                            let cost = case.cost(&param);
                            trace::Quality { violation: case.violation(&param),
                                parameter_error: trace::norm_inf(param.iter().zip(&case.reference).map(|(a, b)| a-b)),
                                param, cost, difference: cost-case.reference_cost, gradient_inf: f64::NAN }
                        });
                    let id = Identity { suite: "constraints", name: case.name, family: case.family,
                        validation: case.validation, precision: stringify!($f), start: start_index,
                        solver: "Slsqp", seed: None };
                    trace::print_rows(&id.prefix(), &rows, args.final_only);
                }
            }
            selected
        }
    };
}
define_runner!(run_f32, f32);
define_runner!(run_f64, f64);

fn list_cases(args: &Args) -> usize {
    println!(
        "suite,case,family,partition,dimension,starts,reference,reference_cost,identifiable"
    );
    let mut selected = 0;
    for case in convergence::cases() {
        let suite = if args.suite == "stochastic" && stochastic_case(&case) {
            "stochastic"
        } else {
            case.suite()
        };
        if !args.includes(suite, &case.name, case.validation) {
            continue;
        }
        selected += 1;
        println!(
            "{}",
            trace::csv([
                suite.into(),
                case.name,
                case.family.into(),
                partition(case.validation).into(),
                case.reference.len().to_string(),
                format!("{:?}", case.starts),
                format!("{:?}", case.reference),
                case.reference_cost.to_string(),
                case.identifiable.to_string()
            ])
        );
    }
    for case in constraints::cases() {
        if !args.includes("constraints", case.name, case.validation) {
            continue;
        }
        selected += 1;
        println!(
            "{}",
            trace::csv([
                "constraints".into(),
                case.name.into(),
                case.family.into(),
                partition(case.validation).into(),
                case.reference.len().to_string(),
                format!("{:?}", case.starts),
                format!("{:?}", case.reference),
                case.reference_cost.to_string(),
                "true".into()
            ])
        );
    }
    selected
}

fn main() {
    let args = Args::parse().unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    let selected = if args.list {
        list_cases(&args)
    } else {
        println!("{}", trace::HEADER);
        let mut selected = 0;
        if args.precision != "f64" {
            selected += run_f32(&args);
        }
        if args.precision != "f32" {
            selected += run_f64(&args);
        }
        selected
    };
    if selected == 0 {
        eprintln!(
            "no matching case; solving validation cases requires --include-validation"
        );
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn holdouts_require_explicit_execution_opt_in() {
        let mut args = Args {
            list: false,
            final_only: false,
            suite: "all".into(),
            precision: "both".into(),
            case: None,
            validation: false,
            max_iter: 200,
            max_passes: 2000,
            seeds: 3,
        };
        assert!(args.includes("nist", "Misra1a", false));
        assert!(!args.includes("nist", "Gauss1", true));
        args.list = true;
        assert!(args.includes("nist", "Gauss1", true));
        args.list = false;
        args.validation = true;
        assert!(args.includes("nist", "Gauss1", true));
        args.case = Some("Misra1a".into());
        assert!(!args.includes("nist", "Gauss1", true));
    }
}
