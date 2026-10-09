//! Emit model validation observations without running or tuning a solver.

use basin::Scalar;
use competitor_bench::convergence::nist::{Dataset, Nist, datasets};
use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
    path::Path,
};

fn emit<F: Scalar>(
    out: &mut impl Write,
    dataset: Dataset,
    precision: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let model = Nist::<F>::new(dataset);
    for (label, point) in [
        ("start1", model.dataset.start::<F>(0)),
        ("start2", model.dataset.start::<F>(1)),
        ("reference", model.dataset.reference_point::<F>()),
    ] {
        let residuals = model.residuals(&point)?;
        let rss = model.rss(&point)?;
        for (i, (observation, residual)) in
            model.dataset.observations.iter().zip(residuals).enumerate()
        {
            let predictors: Vec<F> = observation[1..]
                .iter()
                .map(|v| F::from_f64(v.midpoint).unwrap())
                .collect();
            let (response, derivatives) =
                model.response(&point, &predictors)?;
            write!(
                out,
                "1,{},{},{},{precision},{label},{i},{},{},{}",
                model.dataset.id,
                model.dataset.family,
                model.dataset.partition,
                response.to_f64().unwrap(),
                residual.to_f64().unwrap(),
                rss.to_f64().unwrap()
            )?;
            for j in 0..9 {
                if let Some(v) = derivatives.get(j) {
                    write!(out, ",{}", v.to_f64().unwrap())?;
                } else {
                    write!(out, ",")?;
                }
            }
            writeln!(out)?;
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 || args[1] != "--output" {
        return Err("usage: verify_nist --output <new-csv-file>".into());
    }
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(Path::new(&args[2]))?;
    let mut out = BufWriter::new(file);
    writeln!(
        out,
        "schema,dataset,family,partition,precision,point,row,response,residual,rss,j0,j1,j2,j3,j4,j5,j6,j7,j8"
    )?;
    for dataset in datasets()? {
        emit::<f64>(&mut out, dataset.clone(), "f64")?;
        emit::<f32>(&mut out, dataset, "f32")?;
    }
    out.flush()?;
    Ok(())
}
