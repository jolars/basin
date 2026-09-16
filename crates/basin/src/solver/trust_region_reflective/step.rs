//! Feasible candidate construction in Coleman–Li scaled coordinates.

// Candidate-selection bounds and the radius policy follow SciPy 1.16.2
// (scipy/optimize/_lsq/{trf,common}.py), distributed under these terms.
// Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use crate::core::math::{Scalar, dense_svd::norm};

pub(super) fn number<F: Scalar>(x: f64) -> F {
    F::from_f64(x).unwrap()
}

pub(super) fn dot<F: Scalar>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Keep the boundary repair separate from legacy Trf's projection, whose
/// behavior is part of its Basin 1.x contract.
pub(super) fn interior<F: Scalar>(
    mut x: F,
    lo: F,
    hi: F,
    initial: bool,
) -> Option<F> {
    if lo == hi {
        return Some(lo);
    }
    if !x.is_finite() {
        return None;
    }
    let tiny = F::min_positive_value() * F::epsilon();
    let offset = |b: F| {
        if initial {
            number::<F>(1e-10).max(F::epsilon()) * b.abs().max(F::one())
        } else {
            (F::epsilon() * b.abs()).max(tiny)
        }
    };
    if lo.is_finite() && (x <= lo || (initial && x - lo <= offset(lo))) {
        x = lo + offset(lo);
    }
    if hi.is_finite() && (x >= hi || (initial && hi - x <= offset(hi))) {
        x = hi - offset(hi);
    }
    if !(x > lo && x < hi) && lo.is_finite() && hi.is_finite() {
        x = lo * number(0.5) + hi * number(0.5);
        // Halving subnormal endpoints can lose their only interior value.
        if !(x > lo && x < hi) {
            x = lo + (hi - lo) * number(0.5);
        }
    }
    (x.is_finite() && x > lo && x < hi).then_some(x)
}

fn stride<F: Scalar>(x: &[F], p: &[F], lo: &[F], hi: &[F]) -> F {
    (0..x.len())
        .map(|i| {
            if p[i] > F::zero() {
                (hi[i] - x[i]) / p[i]
            } else if p[i] < F::zero() {
                (lo[i] - x[i]) / p[i]
            } else {
                F::infinity()
            }
        })
        .fold(F::infinity(), F::min)
}

fn sphere_stride<F: Scalar>(base: &[F], direction: &[F], radius: F) -> F {
    let length = norm(direction);
    if length == F::zero() {
        return F::zero();
    }
    let b: F = base
        .iter()
        .zip(direction)
        .map(|(&x, &p)| (x / radius) * (p / length))
        .sum();
    let c = ((norm(base) / radius).powi(2) - F::one()).min(F::zero());
    let root = (b * b - c).sqrt();
    let t = if b > F::zero() {
        -c / (b + root)
    } else {
        -b + root
    };
    t * (radius / length)
}

pub(super) struct Model<F> {
    pub j: Vec<F>,
    pub g: Vec<F>,
    pub c: Vec<F>,
    pub d: Vec<F>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Kind {
    TrustRegion,
    Reflected,
    Gradient,
}

impl<F: Scalar> Model<F> {
    fn product(&self, p: &[F]) -> Vec<F> {
        self.j.chunks(p.len()).map(|row| dot(row, p)).collect()
    }

    pub fn value(&self, p: &[F]) -> F {
        let jp = self.product(p);
        dot(&self.g, p)
            + number::<F>(0.5)
                * (dot(&jp, &jp)
                    + self.c.iter().zip(p).map(|(&c, &p)| c * p * p).sum::<F>())
    }

    fn minimize_line(
        &self,
        base: &[F],
        direction: &[F],
        lower: F,
        upper: F,
    ) -> Option<Vec<F>> {
        if !upper.is_finite() || lower > upper || lower < F::zero() {
            return None;
        }
        let jd = self.product(direction);
        let jb = self.product(base);
        let a = dot(&jd, &jd)
            + self
                .c
                .iter()
                .zip(direction)
                .map(|(&c, &d)| c * d * d)
                .sum::<F>();
        let b = dot(&self.g, direction)
            + dot(&jb, &jd)
            + (0..base.len())
                .map(|i| self.c[i] * base[i] * direction[i])
                .sum::<F>();
        if !a.is_finite() || !b.is_finite() {
            return None;
        }
        let t = if a > F::zero() {
            (-b / a).max(lower).min(upper)
        } else if b < F::zero() {
            upper
        } else {
            lower
        };
        Some(
            base.iter()
                .zip(direction)
                .map(|(&x, &p)| x + t * p)
                .collect(),
        )
    }

    pub fn select(
        &self,
        x: &[F],
        lo: &[F],
        hi: &[F],
        p: &[F],
        radius: F,
        theta: F,
    ) -> (Vec<F>, Kind) {
        let original: Vec<F> =
            p.iter().zip(&self.d).map(|(&p, &d)| p * d).collect();
        let to_bound = stride(x, &original, lo, hi);
        if to_bound >= F::one() {
            return (p.to_vec(), Kind::TrustRegion);
        }
        let on_sphere: Vec<F> = p.iter().map(|&p| p * to_bound).collect();
        let mut on_bound: Vec<F> = (0..x.len())
            .map(|i| x[i] + original[i] * to_bound)
            .collect();
        let mut reflected = p.to_vec();
        for i in 0..x.len() {
            let hit = if original[i] > F::zero() {
                (hi[i] - x[i]) / original[i] == to_bound
            } else if original[i] < F::zero() {
                (lo[i] - x[i]) / original[i] == to_bound
            } else {
                false
            };
            if hit {
                reflected[i] = -reflected[i];
                on_bound[i] = if original[i] > F::zero() {
                    hi[i]
                } else {
                    lo[i]
                };
            } else {
                on_bound[i] = on_bound[i].max(lo[i]).min(hi[i]);
            }
        }
        let reflected_original: Vec<F> = reflected
            .iter()
            .zip(&self.d)
            .map(|(&r, &d)| r * d)
            .collect();
        let r_bound = stride(&on_bound, &reflected_original, lo, hi);
        let r_sphere = sphere_stride(&on_sphere, &reflected, radius);
        let r_limit = r_bound.min(r_sphere);
        let mut best: Vec<F> = on_sphere.iter().map(|&p| theta * p).collect();
        let mut kind = Kind::TrustRegion;
        let mut best_value = self.value(&best);
        if r_limit > F::zero() {
            let lower = (F::one() - theta) * to_bound / r_limit;
            let upper = if r_bound <= r_sphere {
                theta * r_bound
            } else {
                r_sphere
            };
            if let Some(candidate) =
                self.minimize_line(&on_sphere, &reflected, lower, upper)
            {
                let value = self.value(&candidate);
                if value < best_value {
                    best = candidate;
                    best_value = value;
                    kind = Kind::Reflected;
                }
            }
        }
        let gradient: Vec<F> = self.g.iter().map(|&g| -g).collect();
        let original: Vec<F> =
            gradient.iter().zip(&self.d).map(|(&g, &d)| g * d).collect();
        let g_bound = stride(x, &original, lo, hi);
        let g_sphere = radius / norm(&gradient);
        let limit = if g_bound < g_sphere {
            theta * g_bound
        } else {
            g_sphere
        };
        if let Some(candidate) = self.minimize_line(
            &vec![F::zero(); x.len()],
            &gradient,
            F::zero(),
            limit,
        ) {
            if self.value(&candidate) < best_value {
                best = candidate;
                kind = Kind::Gradient;
            }
        }
        (best, kind)
    }
}

pub(super) fn update_radius<F: Scalar>(radius: F, step_norm: F, ratio: F) -> F {
    if ratio < number(0.25) {
        number::<F>(0.25) * step_norm
    } else if ratio > number(0.75) && step_norm > number::<F>(0.95) * radius {
        number::<F>(2.0) * radius
    } else {
        radius
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_three_candidates_match_scipy() {
        for line in include_str!(
            "../../../tests/fixtures/trust_region_reflective_steps.tsv"
        )
        .lines()
        .filter(|s| !s.starts_with('#'))
        {
            let fields: Vec<_> = line.split('|').collect();
            let values = |i: usize| -> Vec<f64> {
                fields[i].split(',').map(|s| s.parse().unwrap()).collect()
            };
            let model = Model {
                j: values(1),
                g: values(2),
                c: values(3),
                d: vec![1.0; 2],
            };
            let (step, kind) = model.select(
                &[0.0; 2],
                &[-0.25; 2],
                &[0.25; 2],
                &values(4),
                2.0,
                0.995,
            );
            assert_eq!(format!("{kind:?}"), fields[0]);
            for (got, want) in step.iter().zip(values(5)) {
                assert!((got - want).abs() < 1e-12);
            }
            assert!(
                (-model.value(&step) - fields[6].parse::<f64>().unwrap()).abs()
                    < 1e-12
            );
            assert!(step.iter().all(|x| x.abs() < 0.25));
        }
    }

    #[test]
    fn simultaneous_bound_hits_stay_feasible() {
        let model = Model {
            j: vec![1.0, 0.0, 0.0, 1.0],
            g: vec![-10.0, -10.0],
            c: vec![1.0; 2],
            d: vec![1.0; 2],
        };
        let (p, _) = model
            .select(&[0.0; 2], &[-1.0; 2], &[0.1; 2], &[1.0; 2], 2.0, 0.995);
        assert!(p.iter().all(|&p| p > -1.0 && p < 0.1));
        assert!(model.value(&p) < 0.0);
    }
    #[test]
    fn radius_and_interior_repair() {
        assert_eq!(update_radius(1.0, 0.8, 0.1), 0.2);
        assert_eq!(update_radius(1.0, 0.98, 0.9), 2.0);
        assert_eq!(update_radius(1.0, 0.5, 0.9), 1.0);
        assert!(interior(1.0_f32, 1.0, 2.0, true).unwrap() > 1.0);
        assert!(
            interior(1.0, 1.0, f64::from_bits(1.0_f64.to_bits() + 1), false)
                .is_none()
        );
        assert!(
            interior(1.0, 1.0, f64::from_bits(1.0_f64.to_bits() + 2), false)
                .is_some()
        );
    }
}
