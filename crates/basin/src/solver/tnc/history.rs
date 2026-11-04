//! Limited-memory inverse BFGS preconditioning, with no dense Hessian.
// Adapted from sisyfuzz 0.1.0, Copyright 2026 Frans Henskens (Apache-2.0).
// Contributed to Basin under MIT OR Apache-2.0; see COPYRIGHT for provenance.
use super::{Scalar, dot};

#[derive(Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(super) struct History<F> {
    pairs: Vec<(Vec<F>, Vec<F>, F)>,
    scale: Option<F>,
}
impl<F: Scalar> History<F> {
    pub(super) fn apply(&self, r: &[F]) -> Vec<F> {
        let mut out = r.to_vec();
        let mut alphas = Vec::with_capacity(self.pairs.len());
        for (s, y, rho) in self.pairs.iter().rev() {
            let a = *rho * dot(s, &out);
            alphas.push(a);
            for (q, &v) in out.iter_mut().zip(y) {
                *q = *q - a * v;
            }
        }
        for q in &mut out {
            *q = *q * self.scale.unwrap_or(F::one());
        }
        for ((s, y, rho), a) in self.pairs.iter().zip(alphas.into_iter().rev())
        {
            let beta = *rho * dot(y, &out);
            for (q, &v) in out.iter_mut().zip(s) {
                *q = *q + (a - beta) * v;
            }
        }
        out
    }
    pub(super) fn update(&mut self, s: Vec<F>, y: Vec<F>, memory: usize) {
        if memory == 0 {
            return;
        }
        let sy = dot(&s, &y);
        let ss = dot(&s, &s);
        let yy = dot(&y, &y);
        let scale = sy / yy;
        let rho = F::one() / sy;
        if !sy.is_finite()
            || !ss.is_finite()
            || !yy.is_finite()
            || sy <= F::from_f64(1e-12).unwrap() * ss.sqrt() * yy.sqrt()
            || !scale.is_finite()
            || scale <= F::zero()
            || !rho.is_finite()
        {
            *self = Self::default();
            return;
        }
        if self.pairs.len() == memory {
            self.pairs.remove(0);
        }
        self.pairs.push((s, y, rho));
        self.scale = Some(scale);
    }
}
