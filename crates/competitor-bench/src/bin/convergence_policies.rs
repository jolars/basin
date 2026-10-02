//! Development-only policy and timing probes. No validation families are run.

use basin::{
    CountsMirror, FirstOrderState, Lbfgsb, LevenbergMarquardt, NelderMead,
    PointState, Scalar, SimplexProgress, Solver, State, TrustRegionReflective,
};
use competitor_bench::convergence::{self, Case, Fixture, dual::Real, trace};
use std::{convert::Infallible, time::Duration};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Policy {
    Default,
    Probe,
    Strict,
}
impl Policy {
    fn name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Probe => "probe",
            Self::Strict => "strict",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Trace,
    Final,
    Plain,
}
impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Trace => "trace",
            Self::Final => "final",
            Self::Plain => "plain",
        }
    }
}
struct Args {
    cases: Vec<String>,
    precision: String,
    solver: String,
    policy: Policy,
    mode: Mode,
    limits: trace::Limits,
    repetitions: usize,
    warmup: usize,
    emit_trace: bool,
}
impl Args {
    fn parse(words: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut a = Self {
            cases: vec![],
            precision: "both".into(),
            solver: "all".into(),
            policy: Policy::Default,
            mode: Mode::Trace,
            limits: trace::Limits {
                iterations: 5000,
                passes: None,
                time: None,
            },
            repetitions: 1,
            warmup: 1,
            emit_trace: false,
        };
        let mut words = words.into_iter();
        while let Some(word) = words.next() {
            if word == "--emit-trace" {
                a.emit_trace = true;
                continue;
            }
            let value = words
                .next()
                .ok_or_else(|| format!("missing value for {word}"))?;
            match word.as_str() {
                "--case" => a.cases.push(value),
                "--precision" => a.precision = value,
                "--solver" => a.solver = value,
                "--policy" => {
                    a.policy = match value.as_str() {
                        "default" => Policy::Default,
                        "probe" => Policy::Probe,
                        "strict" => Policy::Strict,
                        _ => return Err("invalid policy".into()),
                    }
                }
                "--mode" => {
                    a.mode = match value.as_str() {
                        "trace" => Mode::Trace,
                        "final" => Mode::Final,
                        "plain" => Mode::Plain,
                        _ => return Err("invalid mode".into()),
                    }
                }
                "--seconds" => {
                    let seconds: f64 =
                        value.parse().map_err(|_| "invalid seconds")?;
                    let duration = Duration::try_from_secs_f64(seconds)
                        .map_err(|_| "seconds must be finite and positive")?;
                    if duration.is_zero() {
                        return Err(
                            "seconds must be positive and representable".into(),
                        );
                    }
                    a.limits.time = Some(duration);
                }
                "--max-iter" => {
                    a.limits.iterations =
                        value.parse().map_err(|_| "invalid max-iter")?
                }
                "--max-passes" => {
                    let passes =
                        value.parse().map_err(|_| "invalid max-passes")?;
                    if passes == 0 {
                        return Err("max-passes must be positive".into());
                    }
                    a.limits.passes = Some(passes);
                }
                "--repetitions" => {
                    a.repetitions =
                        value.parse().map_err(|_| "invalid repetitions")?
                }
                "--warmup" => {
                    a.warmup = value.parse().map_err(|_| "invalid warmup")?
                }
                _ => return Err(format!("unknown option {word}")),
            }
        }
        if !matches!(a.precision.as_str(), "both" | "f32" | "f64") {
            return Err("invalid precision".into());
        }
        if !matches!(
            a.solver.as_str(),
            "all"
                | "NelderMead"
                | "Lbfgsb"
                | "LevenbergMarquardt"
                | "TrustRegionReflective"
        ) {
            return Err("invalid solver".into());
        }
        if a.mode == Mode::Plain
            && (a.limits.time.is_some()
                || a.limits.passes.is_some()
                || a.emit_trace)
        {
            return Err("plain mode measures uncapped matching work; omit time/pass limits and --emit-trace".into());
        }
        if a.emit_trace && a.mode != Mode::Trace {
            return Err("--emit-trace requires trace mode".into());
        }
        if a.repetitions == 0 {
            return Err("repetitions must be positive".into());
        }
        if a.cases.is_empty() {
            a.cases = [
                "Misra1a",
                "Chwirut2",
                "DanWood",
                "quadratic_n5_k100000000_rot1_rank5_noise0",
            ]
            .map(String::from)
            .to_vec();
        }
        let registered = convergence::cases();
        for name in &a.cases {
            let case = registered
                .iter()
                .find(|c| &c.name == name)
                .ok_or_else(|| format!("unknown case {name}"))?;
            if case.validation {
                return Err(format!(
                    "{name} belongs to validation; this pilot only runs development families"
                ));
            }
        }
        Ok(a)
    }
}

const EXTRA_HEADER: &str = "policy,policy_settings,repetition,mode,time_limit_seconds,pass_limit,iteration_limit,max_segment_seconds,time_limit_exceeded,pass_limit_exceeded,work_recorded,initial_cost,reference_cost,reference_checked_cost,harness_seconds";

fn execute<F, S, So, B>(
    a: &Args,
    case: &Case,
    solver: &str,
    settings: &str,
    build: B,
) where
    F: Scalar + Real,
    S: State<Param = Vec<F>, Float = F> + CountsMirror,
    So: Solver<Fixture<F>, S, Error = Infallible>,
    B: Fn(Vec<F>) -> (So, S),
{
    if a.solver != "all" && a.solver != solver {
        return;
    }
    for (start_index, start) in case.starts.iter().enumerate() {
        let rounded_start: Vec<f64> = start
            .iter()
            .map(|&x| F::from_f64(x).unwrap().to_f64().unwrap())
            .collect();
        let initial_cost = case.cost(&rounded_start);
        for repetition in 0..a.warmup + a.repetitions {
            let (solver_impl, state) =
                build(start.iter().map(|&x| F::from_f64(x).unwrap()).collect());
            let p = Fixture::<F>::new(case.clone());
            let began = std::time::Instant::now();
            let rows = if a.mode == Mode::Plain {
                vec![trace::run_plain(
                    p.without_recording(),
                    solver_impl,
                    state,
                    a.limits.iterations,
                    |x| trace::vector_quality(case, x),
                )]
            } else {
                let work = p.work.clone();
                trace::run_with_settings(
                    p,
                    solver_impl,
                    state,
                    work,
                    trace::Settings {
                        limits: a.limits,
                        final_only: a.mode == Mode::Final,
                    },
                    |x| trace::vector_quality(case, x),
                )
            };
            let harness_seconds = began.elapsed().as_secs_f64();
            if repetition < a.warmup {
                continue;
            }
            let prefix = [
                case.suite().into(),
                case.name.clone(),
                case.family.into(),
                "development".into(),
                std::any::type_name::<F>().into(),
                "Vec".into(),
                (start_index + 1).to_string(),
                solver.into(),
                String::new(),
            ];
            for row in rows
                .iter()
                .filter(|r| a.emit_trace || !r.termination.is_empty())
            {
                let mut fields = trace::fields(&prefix, row);
                fields.extend([
                    a.policy.name().into(),
                    settings.into(),
                    (repetition - a.warmup + 1).to_string(),
                    a.mode.name().into(),
                    a.limits
                        .time
                        .map(|t| t.as_secs_f64().to_string())
                        .unwrap_or_default(),
                    a.limits.passes.map(|p| p.to_string()).unwrap_or_default(),
                    a.limits.iterations.to_string(),
                    row.max_segment_seconds.to_string(),
                    row.time_limit_exceeded.to_string(),
                    row.pass_limit_exceeded.to_string(),
                    row.work_recorded.to_string(),
                    initial_cost.to_string(),
                    case.reference_cost.to_string(),
                    case.cost(&case.reference).to_string(),
                    harness_seconds.to_string(),
                ]);
                println!("{}", trace::csv(fields));
            }
        }
    }
}

macro_rules! define_runner {
    ($run:ident, $f:ty) => {
        fn $run(a: &Args) {
            let single = std::mem::size_of::<$f>() == 4;
            for case in convergence::cases().into_iter().filter(|c| a.cases.contains(&c.name)) {
                // These named probes exercise existing tests. They are not
                // fitted defaults or a port of a competitor's whole policy.
                let simplex: Option<$f> = (a.policy == Policy::Probe).then_some(if single { 1e-3 } else { 1e-4 });
                execute::<$f, _, _, _>(a, &case, "NelderMead", &format!("simplex_size={simplex:?};simplex_cost={simplex:?};AND"), |start| {
                    (NelderMead::<Vec<$f>, $f>::new().with_absolute_simplex_size_tolerance(simplex).with_absolute_simplex_cost_tolerance(simplex), SimplexProgress::new(start))
                });
                let pg: Option<$f> = match a.policy {
                    Policy::Default => Some(1e-10), Policy::Probe => Some(if single { 1e-3 } else { 1e-5 }), Policy::Strict => None,
                };
                execute::<$f, _, _, _>(a, &case, "Lbfgsb", &format!("projected_gradient={pg:?}"), |start| {
                    (Lbfgsb::new().with_absolute_projected_gradient_tolerance(pg), FirstOrderState::new(start))
                });
                let gradient: Option<$f> = (a.policy == Policy::Default).then_some(1e-8);
                let relative: Option<$f> = (a.policy == Policy::Probe).then_some(if single { 1e-4 } else { 1e-8 });
                execute::<$f, _, _, _>(a, &case, "LevenbergMarquardt", &format!("gradient={gradient:?};orthogonality={relative:?};model_reduction={relative:?};OR;Nielsen"), |start| {
                    (LevenbergMarquardt::<_, _, $f>::default().with_absolute_gradient_tolerance(gradient)
                        .with_gradient_orthogonality_tolerance(relative).with_relative_model_reduction_tolerance(relative), PointState::new(start))
                });
                let scaled: Option<$f> = match a.policy {
                    Policy::Default => Some(1e-8), Policy::Probe => Some(if single { 1e-4 } else { 1e-6 }), Policy::Strict => None,
                };
                execute::<$f, _, _, _>(a, &case, "TrustRegionReflective", &format!("scaled_gradient={scaled:?}"), |start| {
                    (TrustRegionReflective::new().with_absolute_scaled_gradient_tolerance(scaled), PointState::new(start))
                });
            }
        }
    };
}
define_runner!(run_f32, f32);
define_runner!(run_f64, f64);

fn main() {
    let words: Vec<_> = std::env::args().skip(1).collect();
    if words.iter().any(|s| s == "--help") {
        println!(
            "convergence_policies [--case NAME (repeatable)] [--solver all|NelderMead|Lbfgsb|LevenbergMarquardt|TrustRegionReflective] [--precision both|f32|f64] [--policy default|probe|strict] [--mode trace|final|plain] [--seconds SECONDS] [--max-iter 5000] [--max-passes N] [--repetitions 1] [--warmup 1] [--emit-trace]"
        );
        return;
    }
    let a = Args::parse(words).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(2);
    });
    println!("{},{EXTRA_HEADER}", trace::HEADER);
    if a.precision != "f64" {
        run_f32(&a);
    }
    if a.precision != "f32" {
        run_f64(&a);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reject_invalid_budgets_and_holdouts_before_running() {
        for args in [
            vec!["--seconds", "NaN"],
            vec!["--seconds", "inf"],
            vec!["--seconds", "-1"],
            vec!["--seconds", "0"],
            vec!["--case", "Gauss1"],
            vec!["--mode", "plain", "--seconds", "1"],
            vec!["--mode", "plain", "--max-passes", "2"],
            vec!["--mode", "final", "--emit-trace"],
            vec!["--repetitions", "0"],
        ] {
            assert!(Args::parse(args.into_iter().map(String::from)).is_err());
        }
        assert!(
            Args::parse(
                ["--seconds", "0.001", "--policy", "strict"].map(String::from)
            )
            .is_ok()
        );
    }
}
