//! Evaluate frozen development-reference witnesses in native arithmetic.

use basin::{CostFunction, Gradient, Scalar};
use competitor_bench::convergence::nist::{Dataset, Nist, datasets};
#[cfg(not(feature = "basin-latest"))]
use nalgebra::DVector;
#[cfg(feature = "basin-latest")]
use nalgebra_latest::DVector;
use std::{
    fs::{OpenOptions, read_to_string},
    io::{BufWriter, Write},
};

fn emit<F: Scalar>(
    out: &mut impl Write,
    dataset: Dataset,
    precision: &str,
    label: &str,
    values: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let model = Nist::<F>::new(dataset);
    let point = DVector::from_vec(
        values.iter().map(|v| F::from_f64(*v).unwrap()).collect(),
    );
    if point
        .iter()
        .zip(values)
        .any(|(a, b)| !a.is_finite() || a.to_f64().unwrap() != *b)
    {
        return Err("coordinates must already be native representable".into());
    }
    let cost = model.cost(&point)?;
    let gradient = model.gradient(&point)?;
    let residuals = model.residuals(point.as_slice())?;
    let rows = model.jacobian_rows(point.as_slice())?;
    let mut absolute_gradient = vec![F::zero(); point.len()];
    for (r, row) in residuals.iter().zip(rows) {
        for (sum, j) in absolute_gradient.iter_mut().zip(row) {
            *sum = *sum + (*r * j).abs();
        }
    }
    let absolute_cost =
        residuals.iter().fold(F::zero(), |sum, r| sum + *r * *r)
            / F::from_f64(2.0).unwrap();
    if !cost.is_finite()
        || !gradient.iter().all(|g| g.is_finite())
        || !absolute_gradient.iter().all(|g| g.is_finite())
    {
        return Err("non-finite witness evaluation".into());
    }
    write!(
        out,
        "1,{},{},{},{precision},{label},{},{}",
        model.dataset.id,
        model.dataset.family,
        model.dataset.partition,
        cost.to_f64().unwrap(),
        absolute_cost.to_f64().unwrap(),
    )?;
    for slice in [point.as_slice(), gradient.as_slice(), &absolute_gradient] {
        for j in 0..6 {
            if let Some(v) = slice.get(j) {
                write!(out, ",{}", v.to_f64().unwrap())?;
            } else {
                write!(out, ",")?;
            }
        }
    }
    writeln!(out)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 || args[1] != "--points" || args[3] != "--output" {
        return Err(
            "usage: verify_nist_witness --points <points.csv> --output <new.csv>"
                .into(),
        );
    }
    let input = read_to_string(&args[2])?;
    let mut lines = input.lines();
    if lines.next() != Some("dataset,precision,point,x0,x1,x2,x3,x4,x5") {
        return Err("unexpected point schema".into());
    }
    let all = datasets()?;
    let mut out = BufWriter::new(
        OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&args[4])?,
    );
    writeln!(
        out,
        "schema,dataset,family,partition,precision,point,cost,cost_abs_terms,x0,x1,x2,x3,x4,x5,g0,g1,g2,g3,g4,g5,ga0,ga1,ga2,ga3,ga4,ga5"
    )?;
    for line in lines {
        let fields: Vec<_> = line.split(',').collect();
        if fields.len() != 9 {
            return Err("malformed point row".into());
        }
        let dataset = all
            .iter()
            .find(|d| d.id == fields[0])
            .ok_or("unknown NIST dataset")?;
        if dataset.partition != "development" {
            return Err(
                "reference witness probe accepts development only".into()
            );
        }
        let n = dataset.reference.len();
        if !fields[3 + n..].iter().all(|s| s.is_empty()) {
            return Err("extra coordinates".into());
        }
        let point = fields[3..3 + n]
            .iter()
            .map(|s| s.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()?;
        match fields[1] {
            "f64" => emit::<f64>(
                &mut out,
                dataset.clone(),
                "f64",
                fields[2],
                &point,
            )?,
            "f32" => emit::<f32>(
                &mut out,
                dataset.clone(),
                "f32",
                fields[2],
                &point,
            )?,
            _ => return Err("unknown precision".into()),
        }
    }
    out.flush()?;
    Ok(())
}
