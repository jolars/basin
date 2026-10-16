//! Compare the four LM configurations on the retained NIST fixtures.
//!
//! `--timing` rotates the configuration order within each paired round and
//! writes raw batched samples. It never enables diagnostic tracing. Independent
//! accuracy checks are a separate pass, outside all timed regions.

use std::{
    hint::black_box,
    io::{self, Write},
    time::{Duration, Instant},
};

use basin::Scalar;
use competitor_bench::{
    lm_defaults::*,
    nist::{Dataset, Nist, datasets},
    vardim_start,
};

const HEADER: &str = "backend,scalar,dataset,family,m,n,start,configuration,termination,iterations,residual_evals,jacobian_evals,state_rss,native_rss,recomputed_rss,rss_relative_error,max_parameter_relative_error,gradient_inf,orthogonality,recovered_1e3,recovered_1e4,recovered_1e6,model_solves,retries,accepted,rejected,last_damping,last_radius,parameters,detail";

#[derive(Default)]
struct Options {
    backend: Option<String>,
    scalar: Option<String>,
    dataset: Option<String>,
    start: Option<usize>,
    trace: bool,
    timing: bool,
    scaling: bool,
    reference: bool,
    samples: usize,
    sample_time: Duration,
}

fn options() -> Result<Options, String> {
    let mut options = Options {
        samples: 32,
        sample_time: Duration::from_millis(2),
        ..Options::default()
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--backend" => {
                options.backend = Some(args.next().ok_or("missing backend")?)
            }
            "--scalar" => {
                options.scalar = Some(args.next().ok_or("missing scalar")?)
            }
            "--dataset" => {
                options.dataset = Some(args.next().ok_or("missing dataset")?)
            }
            "--start" => {
                options.start = Some(
                    args.next()
                        .ok_or("missing start")?
                        .parse()
                        .map_err(|_| "invalid start")?,
                )
            }
            "--trace" => options.trace = true,
            "--timing" => options.timing = true,
            "--scaling" => options.scaling = true,
            "--reference" => options.reference = true,
            "--samples" => {
                options.samples = args
                    .next()
                    .ok_or("missing samples")?
                    .parse()
                    .map_err(|_| "invalid samples")?
            }
            "--sample-ms" => {
                let milliseconds: f64 = args
                    .next()
                    .ok_or("missing sample time")?
                    .parse()
                    .map_err(|_| "invalid sample time")?;
                if !milliseconds.is_finite()
                    || milliseconds <= 0.0
                    || milliseconds > 1000.0
                {
                    return Err(
                        "sample time must be finite and in (0, 1000] ms".into(),
                    );
                }
                options.sample_time =
                    Duration::from_secs_f64(milliseconds / 1000.0);
            }
            "--help" => {
                println!(
                    "verify_lm_defaults [--backend vec|nalgebra|ndarray|faer] [--scalar f32|f64] [--dataset ID] [--start 1|2] [--trace | --timing [--samples 32] [--sample-ms 2]] [--scaling | --reference]\nThe reference uses nalgebra 0.34, f64, and its own stopping policy."
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }
    if options
        .backend
        .as_deref()
        .is_some_and(|v| !["vec", "nalgebra", "ndarray", "faer"].contains(&v))
        || options
            .scalar
            .as_deref()
            .is_some_and(|v| !["f32", "f64"].contains(&v))
        || options.start.is_some_and(|v| !(1..=2).contains(&v))
        || options.samples < 10
    {
        return Err(
            "invalid backend, scalar, start, or sample count (minimum 10)"
                .into(),
        );
    }
    if (options.timing && options.trace)
        || (options.reference
            && (options.timing || options.trace || options.scaling))
    {
        return Err(
            "timing, tracing, and reference modes must run separately".into()
        );
    }
    if (options.scaling || options.reference)
        && options.scalar.as_deref() == Some("f32")
    {
        return Err("scaling and reference modes use f64".into());
    }
    if options.reference
        && options.backend.as_deref().is_some_and(|v| v != "nalgebra")
    {
        return Err("the reference uses nalgebra".into());
    }
    Ok(options)
}

fn number<F: Scalar>(value: F) -> f64 {
    value.to_f64().unwrap()
}
fn csv(value: impl AsRef<str>) -> String {
    format!("\"{}\"", value.as_ref().replace('"', "\"\""))
}

struct Metadata<'a> {
    id: &'a str,
    family: &'a str,
    m: usize,
    n: usize,
    start: usize,
}

fn prefix<B: Backend>(
    metadata: &Metadata<'_>,
    configuration: Configuration,
) -> Vec<String> {
    vec![
        B::NAME.into(),
        B::PRECISION.into(),
        csv(metadata.id),
        csv(metadata.family),
        metadata.m.to_string(),
        metadata.n.to_string(),
        metadata.start.to_string(),
        configuration.name().into(),
    ]
}

fn write_result<B: Backend>(
    out: &mut impl Write,
    metadata: &Metadata<'_>,
    configuration: Configuration,
    result: Result<Outcome<B>, String>,
    quality_at: impl FnOnce(&[f64]) -> Result<Quality, String>,
    native_rss_at: impl FnOnce(&[B::Float]) -> Result<f64, String>,
) -> io::Result<()> {
    let mut row = prefix::<B>(metadata, configuration);
    match result {
        Ok(outcome) => {
            let result = &outcome.result;
            let native = B::values(result.param());
            let parameters: Vec<_> =
                native.iter().copied().map(number).collect();
            let state_rss = 2.0 * number(result.cost());
            let quality =
                quality_at(&parameters).unwrap_or_else(|_| Quality::invalid());
            let valid = state_rss.is_finite()
                && parameters.iter().all(|p| p.is_finite());
            row.extend([
                format!("{:?}", result.report.code()),
                result.iter().to_string(),
                result.counts.residual_evals.to_string(),
                result.counts.jacobian_evals.to_string(),
                format!("{state_rss:.16e}"),
                format!(
                    "{:.16e}",
                    native_rss_at(&native).unwrap_or(f64::INFINITY)
                ),
                format!("{:.16e}", quality.rss),
                format!("{:.16e}", quality.rss_relative_error),
                format!("{:.16e}", quality.parameter_relative_error),
                format!("{:.16e}", quality.gradient_inf),
                format!("{:.16e}", quality.orthogonality),
                (valid && quality.recovered(1e-3)).to_string(),
                (valid && quality.recovered(1e-4)).to_string(),
                (valid && quality.recovered(1e-6)).to_string(),
            ]);
            if let Some(trace) = outcome.trace {
                row.extend([
                    trace.model_solves.to_string(),
                    trace.retries.to_string(),
                    trace.accepted.to_string(),
                    trace.rejected.to_string(),
                    trace
                        .last_damping
                        .map_or(String::new(), |v| format!("{v:.16e}")),
                    trace
                        .last_radius
                        .map_or(String::new(), |v| format!("{v:.16e}")),
                ]);
            } else {
                row.extend(vec![String::new(); 6]);
            }
            row.extend([
                csv(format!("{parameters:?}")),
                csv(format!("{:?}", result.report.termination)),
            ]);
        }
        Err(error) => {
            row.extend([
                "Error".into(),
                String::new(),
                String::new(),
                String::new(),
            ]);
            row.extend(vec!["inf".into(); 7]);
            row.extend(vec!["false".into(); 3]);
            row.extend(vec![String::new(); 7]);
            row.push(csv(error));
        }
    }
    writeln!(out, "{}", row.join(","))
}

fn batch<B: Backend, P: Model<B::Float>>(
    model: &P,
    start: &[B::Float],
    configuration: Configuration,
    repetitions: usize,
) -> Duration {
    let before = Instant::now();
    for _ in 0..repetitions {
        drop(black_box(B::solve(
            black_box(model),
            black_box(start.to_vec()),
            configuration,
            false,
        )));
    }
    before.elapsed()
}

fn time<B: Backend, P: Model<B::Float>>(
    out: &mut impl Write,
    metadata: &Metadata<'_>,
    model: &P,
    start: &[B::Float],
    options: &Options,
) -> io::Result<()> {
    let mut repetitions = [1; 4];
    for (index, configuration) in Configuration::ALL.into_iter().enumerate() {
        while batch::<B, _>(model, start, configuration, repetitions[index])
            < options.sample_time
            && repetitions[index] < 4096
        {
            repetitions[index] *= 2;
        }
    }
    // Each configuration occupies every position and precedes every other
    // configuration once per four rounds, balancing drift and carryover.
    let orders = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]];
    for round in 0..options.samples {
        for position in 0..4 {
            let index = orders[round % 4][position];
            let configuration = Configuration::ALL[index];
            let elapsed =
                batch::<B, _>(model, start, configuration, repetitions[index]);
            let mut row = prefix::<B>(metadata, configuration);
            row.extend([
                round.to_string(),
                repetitions[index].to_string(),
                elapsed.as_nanos().to_string(),
                format!(
                    "{:.8}",
                    elapsed.as_secs_f64() * 1e9 / repetitions[index] as f64
                ),
            ]);
            writeln!(out, "{}", row.join(","))?;
        }
    }
    Ok(())
}

fn selected<B: Backend>(options: &Options) -> bool {
    options.backend.as_deref().is_none_or(|v| v == B::NAME)
        && options.scalar.as_deref().is_none_or(|v| v == B::PRECISION)
}

fn nist<B: Backend>(
    dataset: &Dataset,
    options: &Options,
    out: &mut impl Write,
) -> io::Result<()> {
    if !selected::<B>(options) {
        return Ok(());
    }
    let model = Nist::<B::Float>::new(dataset.clone());
    for start in 0..2 {
        if options.start.is_some_and(|v| v != start + 1) {
            continue;
        }
        let x = dataset.start::<B::Float>(start);
        let metadata = Metadata {
            id: dataset.id,
            family: dataset.family,
            m: dataset.observations.len(),
            n: x.len(),
            start: start + 1,
        };
        if options.timing {
            time::<B, _>(out, &metadata, &model, &x, options)?;
        } else {
            for configuration in Configuration::ALL {
                let result =
                    B::solve(&model, x.clone(), configuration, options.trace);
                write_result::<B>(
                    out,
                    &metadata,
                    configuration,
                    result,
                    |parameters| Quality::at(dataset, parameters),
                    |parameters| model.rss(parameters).map(number),
                )?;
            }
        }
    }
    Ok(())
}

fn scaling<B: Backend<Float = f64>>(
    options: &Options,
    out: &mut impl Write,
) -> io::Result<()> {
    if !selected::<B>(options) {
        return Ok(());
    }
    for n in [10, 30, 100, 300] {
        let id = format!("VarDim-{n}");
        if options.dataset.as_deref().is_some_and(|v| v != id) {
            continue;
        }
        let x = vardim_start(n);
        let metadata = Metadata {
            id: &id,
            family: "scaling",
            m: n + 2,
            n,
            start: 1,
        };
        if options.timing {
            time::<B, _>(out, &metadata, &VarDimModel, &x, options)?;
        } else {
            for configuration in Configuration::ALL {
                let result = B::solve(
                    &VarDimModel,
                    x.clone(),
                    configuration,
                    options.trace,
                );
                write_result::<B>(
                    out,
                    &metadata,
                    configuration,
                    result,
                    |parameters| {
                        let residuals = VarDimModel.residuals(parameters)?;
                        let rows = VarDimModel.jacobian_rows(parameters)?;
                        let rss = residuals.iter().map(|r| r * r).sum::<f64>();
                        let gradient_inf = (0..n)
                            .map(|j| {
                                rows.iter()
                                    .zip(&residuals)
                                    .map(|(row, r)| row[j] * r)
                                    .sum::<f64>()
                                    .abs()
                            })
                            .fold(0.0, f64::max);
                        Ok(Quality {
                            rss,
                            rss_relative_error: rss,
                            parameter_relative_error: parameters
                                .iter()
                                .map(|x| (x - 1.0).abs())
                                .fold(0.0, f64::max),
                            gradient_inf,
                            orthogonality: f64::NAN,
                        })
                    },
                    |parameters| {
                        Ok(VarDimModel
                            .residuals(parameters)?
                            .iter()
                            .map(|r| r * r)
                            .sum())
                    },
                )?;
            }
        }
    }
    Ok(())
}

struct Reference<'a> {
    model: &'a Nist,
    parameters: nalgebra::DVector<f64>,
    residuals: std::cell::Cell<u64>,
    jacobians: std::cell::Cell<u64>,
}

impl levenberg_marquardt::LeastSquaresProblem<f64, nalgebra::Dyn, nalgebra::Dyn>
    for Reference<'_>
{
    type ParameterStorage = nalgebra::Owned<f64, nalgebra::Dyn>;
    type ResidualStorage = nalgebra::Owned<f64, nalgebra::Dyn>;
    type JacobianStorage = nalgebra::Owned<f64, nalgebra::Dyn, nalgebra::Dyn>;
    fn params(&self) -> nalgebra::DVector<f64> {
        self.parameters.clone()
    }
    fn set_params(&mut self, p: &nalgebra::DVector<f64>) {
        self.parameters.copy_from(p);
    }
    fn residuals(&self) -> Option<nalgebra::DVector<f64>> {
        self.residuals.set(self.residuals.get() + 1);
        self.model
            .residuals(self.parameters.as_slice())
            .ok()
            .map(nalgebra::DVector::from_vec)
    }
    fn jacobian(&self) -> Option<nalgebra::DMatrix<f64>> {
        self.jacobians.set(self.jacobians.get() + 1);
        let rows = self.model.jacobian_rows(self.parameters.as_slice()).ok()?;
        Some(nalgebra::DMatrix::from_fn(
            rows.len(),
            self.parameters.len(),
            |i, j| rows[i][j],
        ))
    }
}

fn reference(
    dataset: &Dataset,
    options: &Options,
    out: &mut impl Write,
) -> io::Result<()> {
    let model = Nist::<f64>::new(dataset.clone());
    for start in 0..2 {
        if options.start.is_some_and(|v| v != start + 1) {
            continue;
        }
        let problem = Reference {
            model: &model,
            parameters: nalgebra::DVector::from_vec(dataset.start(start)),
            residuals: 0.into(),
            jacobians: 0.into(),
        };
        let (problem, report) = levenberg_marquardt::LevenbergMarquardt::new()
            .with_patience(
                MAX_EVALUATIONS as usize / (dataset.reference.len() + 1),
            )
            .minimize(problem);
        let quality = Quality::at(dataset, problem.parameters.as_slice())
            .unwrap_or_else(|_| Quality::invalid());
        let mut row = vec![
            "nalgebra-reference".into(),
            "f64".into(),
            csv(dataset.id),
            csv(dataset.family),
            dataset.observations.len().to_string(),
            dataset.reference.len().to_string(),
            (start + 1).to_string(),
            "minpack-reference".into(),
            csv(format!("{:?}", report.termination)),
            String::new(),
            problem.residuals.get().to_string(),
            problem.jacobians.get().to_string(),
            format!("{:.16e}", 2.0 * report.objective_function),
            format!("{:.16e}", quality.rss),
            format!("{:.16e}", quality.rss),
            format!("{:.16e}", quality.rss_relative_error),
            format!("{:.16e}", quality.parameter_relative_error),
            format!("{:.16e}", quality.gradient_inf),
            format!("{:.16e}", quality.orthogonality),
            quality.recovered(1e-3).to_string(),
            quality.recovered(1e-4).to_string(),
            quality.recovered(1e-6).to_string(),
        ];
        row.extend(vec![String::new(); 6]);
        row.extend([
            csv(format!("{:?}", problem.parameters.as_slice())),
            csv(format!("{:?}", report.termination)),
        ]);
        assert_eq!(
            report.number_of_evaluations as u64,
            problem.residuals.get()
        );
        writeln!(out, "{}", row.join(","))?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = options()?;
    let mut out = io::BufWriter::new(io::stdout().lock());
    if options.timing {
        writeln!(
            out,
            "backend,scalar,dataset,family,m,n,start,configuration,round,repetitions,elapsed_ns,ns_per_solve"
        )?;
    } else {
        writeln!(out, "{HEADER}")?;
    }
    if options.scaling {
        scaling::<VecF64>(&options, &mut out)?;
        scaling::<NalgebraF64>(&options, &mut out)?;
        scaling::<NdarrayF64>(&options, &mut out)?;
        scaling::<FaerF64>(&options, &mut out)?;
    } else {
        let mut found = false;
        for dataset in datasets()? {
            if options.dataset.as_deref().is_some_and(|v| v != dataset.id) {
                continue;
            }
            found = true;
            if options.reference {
                reference(&dataset, &options, &mut out)?;
            } else {
                nist::<VecF64>(&dataset, &options, &mut out)?;
                nist::<NalgebraF64>(&dataset, &options, &mut out)?;
                nist::<NdarrayF64>(&dataset, &options, &mut out)?;
                nist::<FaerF64>(&dataset, &options, &mut out)?;
                nist::<VecF32>(&dataset, &options, &mut out)?;
                nist::<NalgebraF32>(&dataset, &options, &mut out)?;
                nist::<NdarrayF32>(&dataset, &options, &mut out)?;
                nist::<FaerF32>(&dataset, &options, &mut out)?;
            }
        }
        if !found {
            return Err("no dataset matched".into());
        }
    }
    out.flush()?;
    Ok(())
}
