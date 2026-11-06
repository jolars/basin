//! Limited-memory inverse BFGS preconditioning, with no dense Hessian.
// Adapted from sisyfuzz 0.1.0, Copyright 2026 Frans Henskens (Apache-2.0).
// Contributed to Basin under MIT OR Apache-2.0; see COPYRIGHT for provenance.
use super::{Scalar, VectorIndex, VectorLen, dot};

#[derive(Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(super) struct History<F> {
    pairs: Vec<(Vec<F>, Vec<F>, F)>,
    scale: Option<F>,
    #[cfg_attr(feature = "serde", serde(skip))]
    alphas: Vec<F>,
}

impl<F: Scalar> History<F> {
    pub(super) fn apply(&mut self, r: &[F], out: &mut Vec<F>) {
        out.clear();
        out.extend_from_slice(r);
        self.alphas.clear();
        for (s, y, rho) in self.pairs.iter().rev() {
            let a = *rho * dot(s, out);
            self.alphas.push(a);
            for (q, &v) in out.iter_mut().zip(y) {
                *q = *q - a * v;
            }
        }
        for q in out.iter_mut() {
            *q = *q * self.scale.unwrap_or(F::one());
        }
        for ((s, y, rho), &a) in self.pairs.iter().zip(self.alphas.iter().rev())
        {
            let beta = *rho * dot(y, out);
            for (q, &v) in out.iter_mut().zip(s) {
                *q = *q + (a - beta) * v;
            }
        }
    }
    pub(super) fn update<V: VectorLen + VectorIndex<F>>(
        &mut self,
        x: &V,
        next: &V,
        g: &V,
        next_g: &V,
        memory: usize,
    ) {
        if memory == 0 {
            return;
        }
        // Recycle the oldest pair after it leaves the model, as in sisyfuzz's
        // fixed-capacity history. Keep chronological storage for checkpoint
        // compatibility with earlier Basin TNC versions.
        let (mut s, mut y, _) = if self.pairs.len() == memory {
            self.pairs.remove(0)
        } else {
            (Vec::new(), Vec::new(), F::zero())
        };
        s.resize(x.vec_len(), F::zero());
        y.resize(x.vec_len(), F::zero());
        for i in 0..x.vec_len() {
            s[i] = next.get_scalar(i) - x.get_scalar(i);
            y[i] = next_g.get_scalar(i) - g.get_scalar(i);
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
        self.pairs.push((s, y, rho));
        self.scale = Some(scale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_reuse<F: Scalar>() {
        let zero = vec![F::zero(); 3];
        let mut history = History::<F>::default();
        let mut out = Vec::new();
        let mut old_storage = None;
        for update in 0..12 {
            let s: Vec<_> = (0..3)
                .map(|i| {
                    F::from_f64(
                        ((update * 7 + i * 11) % 17) as f64 / 17.0 + 0.1,
                    )
                    .unwrap()
                })
                .collect();
            let two = F::from_f64(2.0).unwrap();
            let three = F::from_f64(3.0).unwrap();
            let half = F::from_f64(0.5).unwrap();
            let y = vec![
                three * s[0] + s[1],
                s[0] + two * s[1] + half * s[2],
                half * s[1] + F::from_f64(1.5).unwrap() * s[2],
            ];
            let reused = (history.pairs.len() == 2).then(|| {
                (history.pairs[0].0.as_ptr(), history.pairs[0].1.as_ptr())
            });
            history.update(&zero, &s, &zero, &y, 2);
            if let Some((s_ptr, y_ptr)) = reused {
                assert_eq!(history.pairs[1].0.as_ptr(), s_ptr);
                assert_eq!(history.pairs[1].1.as_ptr(), y_ptr);
            }
            let mut h = vec![vec![F::zero(); 3]; 3];
            for (i, row) in h.iter_mut().enumerate() {
                row[i] = history.scale.unwrap();
            }
            for (s, y, rho) in &history.pairs {
                let hy: Vec<_> = h.iter().map(|row| dot(row, y)).collect();
                let factor = *rho * (F::one() + *rho * dot(y, &hy));
                for i in 0..3 {
                    for j in 0..3 {
                        h[i][j] = h[i][j]
                            - *rho * (s[i] * hy[j] + hy[i] * s[j])
                            + factor * s[i] * s[j];
                    }
                }
            }
            let r = vec![F::one(), -two, half];
            history.apply(&r, &mut out);
            for (actual, row) in out.iter().zip(&h) {
                let expected = dot(row, &r);
                assert!(
                    (*actual - expected).abs()
                        <= F::epsilon()
                            * F::from_f64(100.0).unwrap()
                            * (F::one() + expected.abs())
                );
            }
            let storage = (out.as_ptr(), history.alphas.as_ptr());
            if update >= 2 {
                assert_eq!(Some(storage), old_storage);
            }
            old_storage = Some(storage);
        }
        // A globally unsafe pair still clears the model and restores identity.
        let bad = vec![F::nan(); 3];
        history.update(&zero, &zero, &zero, &bad, 2);
        assert!(history.pairs.is_empty());
        history.apply(&zero, &mut out);
        assert_eq!(out, zero);
    }

    #[test]
    fn reused_inverse_and_evicted_pairs_match_dense_bfgs_f64() {
        check_reuse::<f64>();
    }
    #[test]
    fn reused_inverse_and_evicted_pairs_match_dense_bfgs_f32() {
        check_reuse::<f32>();
    }
}
