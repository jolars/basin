//! NIST StRD reproduction fixtures, not additional public corpus problems.
//!
//! Residuals are model minus response; Nelson uses the natural log of the
//! response. Costs use half RSS. Published decimal rounding intervals describe
//! the printed values only, not certified attainable solver targets.
//! Native f32/f64 arithmetic is supported with the benchmark-selected nalgebra
//! version. Other backend versions remain a separate pilot gate.

use basin::{CostFunction, Gradient, Jacobian, Residual, Scalar};
#[cfg(not(feature = "basin-latest"))]
use nalgebra::{DMatrix, DVector};
#[cfg(feature = "basin-latest")]
use nalgebra_latest::{DMatrix, DVector};

mod inputs;
mod models;

struct Source {
    id: &'static str,
    family: &'static str,
    partition: &'static str,
    parameters: usize,
    observations: usize,
    predictors: usize,
    text: &'static str,
}

/// Decimal value and half a unit in its last printed place.
#[derive(Clone, Debug)]
pub struct PrintedValue {
    pub decimal: String,
    pub midpoint: f64,
    pub half_width: f64,
}

impl PrintedValue {
    pub fn parse(decimal: &str) -> Result<Self, String> {
        let midpoint: f64 = decimal
            .parse()
            .map_err(|_| format!("invalid decimal: {decimal}"))?;
        if !midpoint.is_finite() {
            return Err(format!("non-finite decimal: {decimal}"));
        }
        let (mantissa, exponent) =
            decimal.split_once(['e', 'E']).unwrap_or((decimal, "0"));
        let exponent: i32 = exponent.parse().map_err(|_| "invalid exponent")?;
        let places = mantissa
            .split_once('.')
            .map_or(0, |(_, digits)| digits.len() as i32);
        Ok(Self {
            decimal: decimal.to_owned(),
            midpoint,
            half_width: 0.5 * 10_f64.powi(exponent - places),
        })
    }
}

/// Frozen source metadata, both primary starts, and all observations.
#[derive(Clone, Debug)]
pub struct Dataset {
    pub id: &'static str,
    pub family: &'static str,
    pub partition: &'static str,
    pub starts: [Vec<PrintedValue>; 2],
    pub reference: Vec<PrintedValue>,
    pub rss: PrintedValue,
    /// Original response followed by one or two predictors.
    pub observations: Vec<Vec<PrintedValue>>,
}

impl Dataset {
    fn parse(source: &Source) -> Result<Self, String> {
        let mut starts = [Vec::new(), Vec::new()];
        let mut reference = Vec::new();
        let mut rss = None;
        let mut observations = Vec::new();
        let mut data = false;
        for line in source.text.lines() {
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields.first().is_some_and(|s| {
                s.starts_with('b') && s[1..].parse::<usize>().is_ok()
            }) && fields.get(1) == Some(&"=")
            {
                if fields.len() != 6 {
                    return Err(format!(
                        "{}: malformed parameter row",
                        source.id
                    ));
                }
                starts[0].push(PrintedValue::parse(fields[2])?);
                starts[1].push(PrintedValue::parse(fields[3])?);
                reference.push(PrintedValue::parse(fields[4])?);
            }
            if let Some(value) = line.strip_prefix("Residual Sum of Squares:") {
                rss = Some(PrintedValue::parse(value.trim())?);
            }
            if line.starts_with("Data:") && fields.get(1) == Some(&"y") {
                data = true;
                continue;
            }
            if data && !fields.is_empty() {
                if fields.len() != source.predictors + 1 {
                    return Err(format!(
                        "{}: malformed observation",
                        source.id
                    ));
                }
                observations.push(
                    fields
                        .iter()
                        .map(|s| PrintedValue::parse(s))
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
        }
        if reference.len() != source.parameters
            || observations.len() != source.observations
        {
            return Err(format!("{}: dimension mismatch", source.id));
        }
        if source.id == "Nelson"
            && observations.iter().any(|row| row[0].midpoint <= 0.0)
        {
            return Err("Nelson requires positive responses".into());
        }
        Ok(Self {
            id: source.id,
            family: source.family,
            partition: source.partition,
            starts,
            reference,
            rss: rss.ok_or("missing RSS")?,
            observations,
        })
    }

    pub fn start<F: Scalar>(&self, index: usize) -> Vec<F> {
        native(&self.starts[index])
    }
    pub fn reference_point<F: Scalar>(&self) -> Vec<F> {
        native(&self.reference)
    }
}

fn native<F: Scalar>(values: &[PrintedValue]) -> Vec<F> {
    values
        .iter()
        .map(|v| F::from_f64(v.midpoint).unwrap())
        .collect()
}

/// Parse all 27 embedded snapshots in manifest order, preserving partitions.
pub fn datasets() -> Result<Vec<Dataset>, String> {
    inputs::SOURCES.iter().map(Dataset::parse).collect()
}

/// Analytic NIST adapter for cost, gradient, residual, and Jacobian solvers.
#[derive(Clone, Debug)]
pub struct Nist<F: Scalar = f64> {
    pub dataset: Dataset,
    observations: Vec<Vec<F>>,
}

impl<F: Scalar> Nist<F> {
    pub fn new(dataset: Dataset) -> Self {
        let observations = dataset
            .observations
            .iter()
            .map(|row| {
                let mut row = native::<F>(row);
                if dataset.id == "Nelson" {
                    row[0] = row[0].ln();
                }
                row
            })
            .collect();
        Self {
            dataset,
            observations,
        }
    }

    /// One published response and its analytic derivatives, before subtraction.
    pub fn response(
        &self,
        parameters: &[F],
        predictors: &[F],
    ) -> Result<(F, Vec<F>), String> {
        if parameters.len() != self.dataset.reference.len()
            || predictors.len() + 1 != self.observations[0].len()
        {
            return Err("NIST input dimension mismatch".into());
        }
        // Non-finite model values remain visible to the solver's safeguards.
        Ok(models::response(self.dataset.id, parameters, predictors))
    }

    pub fn residuals(&self, parameters: &[F]) -> Result<Vec<F>, String> {
        self.observations
            .iter()
            .map(|row| Ok(self.response(parameters, &row[1..])?.0 - row[0]))
            .collect()
    }

    pub fn jacobian_rows(
        &self,
        parameters: &[F],
    ) -> Result<Vec<Vec<F>>, String> {
        self.observations
            .iter()
            .map(|row| Ok(self.response(parameters, &row[1..])?.1))
            .collect()
    }

    pub fn rss(&self, parameters: &[F]) -> Result<F, String> {
        Ok(self.residuals(parameters)?.iter().map(|r| *r * *r).sum())
    }
}

impl<F: Scalar> CostFunction for Nist<F> {
    type Param = DVector<F>;
    type Output = F;
    type Error = String;
    fn cost(&self, x: &DVector<F>) -> Result<F, String> {
        Ok(self.rss(x.as_slice())? / F::from_f64(2.0).unwrap())
    }
}

impl<F: Scalar> Gradient for Nist<F> {
    type Gradient = DVector<F>;
    fn gradient(&self, x: &DVector<F>) -> Result<DVector<F>, String> {
        let r = self.residuals(x.as_slice())?;
        let j = self.jacobian_rows(x.as_slice())?;
        let mut g = vec![F::zero(); x.len()];
        for (r, row) in r.iter().zip(j) {
            for (g, derivative) in g.iter_mut().zip(row) {
                *g = *g + *r * derivative;
            }
        }
        Ok(DVector::from_vec(g))
    }
}

impl<F: Scalar> Residual for Nist<F> {
    type Param = DVector<F>;
    type Output = DVector<F>;
    type Error = String;
    fn residual(&self, x: &DVector<F>) -> Result<DVector<F>, String> {
        Ok(DVector::from_vec(self.residuals(x.as_slice())?))
    }
}

impl<F: Scalar> Jacobian for Nist<F> {
    type Jacobian = DMatrix<F>;
    fn jacobian(&self, x: &DVector<F>) -> Result<DMatrix<F>, String> {
        let rows = self.jacobian_rows(x.as_slice())?;
        Ok(DMatrix::from_fn(rows.len(), x.len(), |i, j| rows[i][j]))
    }
}
