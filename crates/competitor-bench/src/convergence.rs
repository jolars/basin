//! Experimental convergence fixtures, separate from Basin's public corpus.
//!
//! Native `f32` and `f64` evaluation share formulas with forward derivatives.
//! These fixtures currently implement the `Vec` backend only. No case here
//! establishes a selected default or an attainable target by itself.

pub mod analytic;
pub mod constraints;
pub mod dual;
pub mod nist;
pub mod scalar;
pub mod trace;

use basin::{
    BoxConstraints, CostFunction, DenseMatrix, Gradient, Jacobian,
    MiniBatchGradient, Residual, Scalar,
};
use dual::{Dual, Real};
use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
};

#[derive(Clone, Debug)]
pub enum Model {
    Nist(nist::Nist),
    Quadratic(analytic::Quadratic),
}

#[derive(Clone, Debug)]
pub struct Case {
    pub name: String,
    pub family: &'static str,
    pub validation: bool,
    pub starts: [Vec<f64>; 2],
    pub reference: Vec<f64>,
    pub reference_cost: f64,
    pub identifiable: bool,
    pub model: Model,
}

impl Case {
    pub fn residuals<T: Real>(&self, p: &[T]) -> Vec<T> {
        match &self.model {
            Model::Nist(m) => m.residuals(p),
            Model::Quadratic(m) => m.residuals(p),
        }
    }

    pub fn cost<F: Scalar + Real>(&self, p: &[F]) -> F {
        self.residuals(p).into_iter().map(|r| r * r).sum::<F>()
            / F::from_f64(2.0).unwrap()
    }

    pub fn derivatives<F: Scalar>(&self, p: &[F]) -> (Vec<F>, Vec<F>) {
        assert!(p.len() <= dual::MAX_PARAMETERS);
        let variables: Vec<_> = p
            .iter()
            .enumerate()
            .map(|(i, &v)| Dual::variable(v, i))
            .collect();
        let result = self.residuals(&variables);
        let residuals = result.iter().map(|r| r.value).collect();
        let jacobian = result
            .iter()
            .flat_map(|r| r.derivative[..p.len()].iter().copied())
            .collect();
        (residuals, jacobian)
    }

    pub fn gradient<F: Scalar>(&self, p: &[F]) -> Vec<F> {
        let (residuals, jacobian) = self.derivatives(p);
        jt_r(&residuals, &jacobian, p.len())
    }

    pub fn suite(&self) -> &'static str {
        match self.model {
            Model::Nist(_) => "nist",
            Model::Quadratic(_) => "analytic",
        }
    }

    pub fn n_residuals(&self) -> usize {
        match &self.model {
            Model::Nist(m) => m.data.len(),
            Model::Quadratic(m) => 2 * m.dimension,
        }
    }
}

fn jt_r<F: Scalar>(r: &[F], j: &[F], n: usize) -> Vec<F> {
    let mut g = vec![F::zero(); n];
    for (&r, row) in r.iter().zip(j.chunks_exact(n)) {
        for (g, &j) in g.iter_mut().zip(row) {
            *g = *g + r * j;
        }
    }
    g
}

pub fn cases() -> Vec<Case> {
    let mut cases: Vec<_> = nist::cases()
        .into_iter()
        .map(|m| Case {
            name: m.name.into(),
            family: m.family,
            validation: m.validation,
            starts: m.starts.clone(),
            reference: m.certified.clone(),
            reference_cost: m.rss / 2.0,
            identifiable: !matches!(
                m.family,
                "exponential_sum" | "gaussian" | "harmonic"
            ),
            model: Model::Nist(m),
        })
        .collect();
    for n in [2, 5, 10] {
        for (family, condition, rotated, rank_deficient, noise, validation) in [
            ("quadratic", 1.0, false, false, 0.0, false),
            ("quadratic", 1e4, false, false, 0.0, false),
            ("quadratic", 1e8, true, false, 0.0, false),
            ("quadratic", 1e4, true, true, 0.0, false),
            ("quadratic", 1.0, true, false, 0.1, false),
        ] {
            let model = analytic::Quadratic {
                dimension: n,
                condition,
                rotated,
                rank_deficient,
                noise,
            };
            let reference = model.reference();
            let starts = [
                reference.iter().map(|&x| x + 1.0).collect(),
                reference
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| x + if i % 2 == 0 { -2.0 } else { 0.5 })
                    .collect(),
            ];
            cases.push(Case {
                name: format!(
                    "quadratic_n{n}_k{condition:.0}_rot{}_rank{}_noise{noise}",
                    u8::from(rotated),
                    n - usize::from(rank_deficient)
                ),
                family,
                validation,
                starts,
                reference,
                reference_cost: n as f64 * noise * noise,
                identifiable: !rank_deficient,
                model: Model::Quadratic(model),
            });
        }
    }
    cases
}

/// Physical callback work. A derivative pass propagates all n directions and
/// is reported separately from a value pass; their runtimes need not be equal.
#[derive(Clone, Debug)]
pub struct Work {
    pub value_passes: u64,
    pub derivative_passes: u64,
    pub sample_derivatives: u64,
    pub best_cost: f64,
    pub best_param: Vec<f64>,
}
impl Default for Work {
    fn default() -> Self {
        Self {
            value_passes: 0,
            derivative_passes: 0,
            sample_derivatives: 0,
            best_cost: f64::INFINITY,
            best_param: vec![],
        }
    }
}
impl Work {
    pub fn passes(&self) -> u64 {
        self.value_passes + self.derivative_passes
    }
    pub fn observe<F: Scalar>(&mut self, p: &[F], cost: F) {
        let cost = cost.to_f64().unwrap();
        if cost < self.best_cost {
            self.best_cost = cost;
            self.best_param = p.iter().map(|x| x.to_f64().unwrap()).collect();
        }
    }
}

#[derive(Clone)]
pub struct Fixture<F> {
    pub case: Case,
    pub work: Arc<Mutex<Work>>,
    lower: Vec<F>,
    upper: Vec<F>,
    record_work: bool,
}
impl<F: Scalar + Real> Fixture<F> {
    pub fn new(case: Case) -> Self {
        let n = case.reference.len();
        Self {
            case,
            work: Arc::default(),
            lower: vec![F::neg_infinity(); n],
            upper: vec![F::infinity(); n],
            record_work: true,
        }
    }
    /// Disable fixture bookkeeping for the whole-executor timing control.
    /// Mathematical callbacks and Basin's authoritative counts are unchanged.
    pub fn without_recording(mut self) -> Self {
        self.record_work = false;
        self
    }
    fn value(&self, p: &[F]) -> Vec<F> {
        let r = self.case.residuals(p);
        if !self.record_work {
            return r;
        }
        let mut work = self.work.lock().unwrap();
        work.value_passes += 1;
        work.observe(
            p,
            r.iter().map(|&r| r * r).sum::<F>() / F::from_f64(2.0).unwrap(),
        );
        r
    }
    fn derivative(&self, p: &[F]) -> (Vec<F>, Vec<F>) {
        let (r, j) = self.case.derivatives(p);
        if !self.record_work {
            return (r, j);
        }
        let mut work = self.work.lock().unwrap();
        work.derivative_passes += 1;
        work.observe(
            p,
            r.iter().map(|&r| r * r).sum::<F>() / F::from_f64(2.0).unwrap(),
        );
        (r, j)
    }
}
impl<F: Scalar + Real> CostFunction for Fixture<F> {
    type Param = Vec<F>;
    type Output = F;
    type Error = Infallible;
    fn cost(&self, p: &Vec<F>) -> Result<F, Infallible> {
        Ok(self.value(p).into_iter().map(|r| r * r).sum::<F>()
            / F::from_f64(2.0).unwrap())
    }
}
impl<F: Scalar + Real> Gradient for Fixture<F> {
    type Gradient = Vec<F>;
    fn gradient(&self, p: &Vec<F>) -> Result<Vec<F>, Infallible> {
        let (r, j) = self.derivative(p);
        Ok(jt_r(&r, &j, p.len()))
    }
    fn cost_and_gradient(&self, p: &Vec<F>) -> Result<(F, Vec<F>), Infallible> {
        let (r, j) = self.derivative(p);
        Ok((
            r.iter().map(|&r| r * r).sum::<F>() / F::from_f64(2.0).unwrap(),
            jt_r(&r, &j, p.len()),
        ))
    }
}
impl<F: Scalar + Real> Residual for Fixture<F> {
    type Param = Vec<F>;
    type Output = Vec<F>;
    type Error = Infallible;
    fn residual(&self, p: &Vec<F>) -> Result<Vec<F>, Infallible> {
        Ok(self.value(p))
    }
}
impl<F: Scalar + Real> Jacobian for Fixture<F> {
    type Jacobian = DenseMatrix<F>;
    fn jacobian(&self, p: &Vec<F>) -> Result<DenseMatrix<F>, Infallible> {
        let (r, j) = self.derivative(p);
        Ok(DenseMatrix::from_row_slice(r.len(), p.len(), &j))
    }
    fn residual_and_jacobian(
        &self,
        p: &Vec<F>,
    ) -> Result<(Vec<F>, DenseMatrix<F>), Infallible> {
        let (r, j) = self.derivative(p);
        let matrix = DenseMatrix::from_row_slice(r.len(), p.len(), &j);
        Ok((r, matrix))
    }
}
impl<F: Scalar + Real> BoxConstraints for Fixture<F> {
    fn lower(&self) -> &Vec<F> {
        &self.lower
    }
    fn upper(&self) -> &Vec<F> {
        &self.upper
    }
}

impl<F: Scalar + Real> MiniBatchGradient for Fixture<F> {
    type Gradient = Vec<F>;
    fn n_samples(&self) -> usize {
        self.case.n_residuals()
    }
    fn batch_gradient(
        &self,
        p: &Vec<F>,
        indices: &[usize],
    ) -> Result<Vec<F>, Infallible> {
        // This deliberately evaluates the full model. Charge that physical
        // work, even though only the selected residual gradients are used.
        let (r, j) = self.derivative(p);
        if self.record_work {
            self.work.lock().unwrap().sample_derivatives +=
                indices.len() as u64;
        }
        let scale = F::from_usize(r.len()).unwrap()
            / F::from_usize(indices.len()).unwrap();
        let mut g = vec![F::zero(); p.len()];
        for &i in indices {
            for (k, gk) in g.iter_mut().enumerate() {
                *gk = *gk + scale * r[i] * j[i * p.len() + k];
            }
        }
        Ok(g)
    }
}

#[cfg(test)]
mod tests;
