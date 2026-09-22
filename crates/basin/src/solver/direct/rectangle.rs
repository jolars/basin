use std::collections::BTreeMap;

use crate::core::math::{Scalar, VectorIndex};

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub(super) struct Rectangle<F> {
    pub center: Vec<F>,
    pub depths: Vec<usize>,
    pub cost: F,
}

impl<F: Scalar> Rectangle<F> {
    pub fn new(center: Vec<F>, cost: F) -> Self {
        Self {
            depths: vec![0; center.len()],
            center,
            cost,
        }
    }

    pub fn level(&self) -> u64 {
        self.depths.iter().map(|&depth| depth as u64).sum()
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub(super) struct Work<F> {
    bounds: Vec<(F, F)>,
    pub active: Vec<usize>,
    pub rectangles: Vec<Rectangle<F>>,
    thirds: Vec<F>,
    pub best: Option<usize>,
}

pub(super) struct Probes<V, F> {
    pub axes: Vec<usize>,
    pub centers: Vec<Vec<F>>,
    pub params: Vec<V>,
}

impl<F: Scalar> Work<F> {
    pub fn new(
        bounds: Vec<(F, F)>,
        active: Vec<usize>,
        root: Rectangle<F>,
    ) -> Self {
        Self {
            bounds,
            active,
            rectangles: vec![root],
            thirds: vec![F::one()],
            best: None,
        }
    }

    pub fn consider(&mut self, id: usize) {
        let cost = self.rectangles[id].cost;
        if cost < F::infinity()
            && self
                .best
                .is_none_or(|best| cost < self.rectangles[best].cost)
        {
            self.best = Some(id);
        }
    }

    pub fn point<V: Clone + VectorIndex<F>>(
        &self,
        template: &V,
        center: &[F],
    ) -> V {
        let mut point = template.clone();
        for (i, &(lo, _)) in self.bounds.iter().enumerate() {
            point.set_scalar(i, lo);
        }
        for (&axis, &z) in self.active.iter().zip(center) {
            let (lo, hi) = self.bounds[axis];
            point.set_scalar(axis, interpolate(lo, hi, z));
        }
        point
    }

    pub fn radius(&self, rectangle: &Rectangle<F>) -> F {
        let Some(&depth) = rectangle.depths.iter().min() else {
            return F::zero();
        };
        // Longest-axis trisection keeps side depths at most one apart.
        // A common scale avoids underflow in squared side lengths.
        let long = rectangle.depths.iter().filter(|&&d| d == depth).count();
        let short = rectangle.depths.len() - long;
        self.thirds[depth]
            * F::from_f64(0.5).unwrap()
            * (F::from_usize(long).unwrap()
                + F::from_usize(short).unwrap() / F::from_f64(9.0).unwrap())
            .sqrt()
    }

    pub fn select(&self, epsilon: F) -> Vec<usize> {
        let mut groups: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        let mut rejected: Option<(u64, usize)> = None;
        for (id, rectangle) in self.rectangles.iter().enumerate() {
            let level = rectangle.level();
            if !rectangle.cost.is_finite() {
                if rejected.is_none_or(|previous| (level, id) < previous) {
                    rejected = Some((level, id));
                }
                continue;
            }
            let group = groups.entry(level).or_default();
            if let Some(&first) = group.first() {
                let minimum = self.rectangles[first].cost;
                if rectangle.cost > minimum {
                    continue;
                }
                if rectangle.cost < minimum {
                    group.clear();
                }
            }
            group.push(id);
        }
        // Depth sums order sizes exactly, including non-cubic rectangles.
        let groups: Vec<_> = groups.into_values().rev().collect();
        let points: Vec<_> = groups
            .iter()
            .map(|ids| {
                let rectangle = &self.rectangles[ids[0]];
                (self.radius(rectangle), rectangle.cost)
            })
            .collect();
        let mut selected = Vec::new();
        for index in potentially_optimal(&points, epsilon) {
            selected.extend_from_slice(&groups[index]);
        }
        if let Some((_, id)) = rejected {
            selected.push(id);
        }
        selected.sort_unstable_by_key(|&id| (self.rectangles[id].level(), id));
        selected
    }

    pub fn probes<V: Clone + VectorIndex<F>>(
        &mut self,
        template: &V,
        id: usize,
    ) -> Option<Probes<V, F>> {
        let rectangle = &self.rectangles[id];
        let depth = *rectangle.depths.iter().min()?;
        while self.thirds.len() <= depth + 1 {
            self.thirds
                .push(*self.thirds.last().unwrap() / F::from_f64(3.0).unwrap());
        }
        let delta = self.thirds[depth + 1];
        let axes: Vec<_> = rectangle
            .depths
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| (d == depth).then_some(i))
            .collect();
        let mut centers = Vec::with_capacity(2 * axes.len());
        let mut params = Vec::with_capacity(2 * axes.len());
        for &i in &axes {
            let c = rectangle.center[i];
            let minus = c - delta;
            let plus = c + delta;
            let (lo, hi) = self.bounds[self.active[i]];
            let x = interpolate(lo, hi, c);
            if !(minus >= F::zero()
                && minus < c
                && c < plus
                && plus <= F::one()
                && interpolate(lo, hi, minus) < x
                && x < interpolate(lo, hi, plus))
            {
                return None;
            }
            for coordinate in [minus, plus] {
                let mut center = rectangle.center.clone();
                center[i] = coordinate;
                params.push(self.point(template, &center));
                centers.push(center);
            }
        }
        Some(Probes {
            axes,
            centers,
            params,
        })
    }

    pub fn divide(
        &mut self,
        id: usize,
        axes: Vec<usize>,
        centers: Vec<Vec<F>>,
        costs: Vec<F>,
    ) {
        let mut order: Vec<_> = (0..axes.len()).collect();
        let rejected_as_infinity =
            |f: F| if f.is_nan() { F::infinity() } else { f };
        let axis_cost = |i: usize| {
            rejected_as_infinity(costs[2 * i])
                .min(rejected_as_infinity(costs[2 * i + 1]))
        };
        order.sort_by(|&a, &b| {
            axis_cost(a)
                .partial_cmp(&axis_cost(b))
                .unwrap()
                .then_with(|| axes[a].cmp(&axes[b]))
        });
        let mut depths = self.rectangles[id].depths.clone();
        let mut children: Vec<_> = centers
            .into_iter()
            .zip(costs)
            .map(|(center, cost)| Rectangle {
                center,
                cost,
                depths: Vec::new(),
            })
            .collect();
        for i in order {
            depths[axes[i]] += 1;
            children[2 * i].depths = depths.clone();
            children[2 * i + 1].depths = depths.clone();
        }
        self.rectangles[id].depths = depths;
        for rectangle in children {
            let id = self.rectangles.len();
            self.rectangles.push(rectangle);
            self.consider(id);
        }
    }
}

fn interpolate<F: Scalar>(lo: F, hi: F, z: F) -> F {
    // A convex combination avoids an overflowing width across zero; same-sign
    // endpoints use the nearer endpoint to retain precision in narrow boxes.
    let x = if lo <= F::zero() && hi >= F::zero() {
        (F::one() - z) * lo + z * hi
    } else if z <= F::from_f64(0.5).unwrap() {
        lo + z * (hi - lo)
    } else {
        hi - (F::one() - z) * (hi - lo)
    };
    x.max(lo).min(hi)
}

/// Indices satisfying Definition 4.1; points have strictly increasing radii.
fn potentially_optimal<F: Scalar>(points: &[(F, F)], epsilon: F) -> Vec<usize> {
    let Some(&(max_radius, _)) = points.last() else {
        return Vec::new();
    };
    if max_radius <= F::zero() {
        return Vec::new();
    }
    let best = points
        .iter()
        .fold(F::infinity(), |best, &(_, f)| best.min(f));
    let margin = Scaled::new(epsilon).mul(Scaled::new(best.abs()));
    let mut hull: Vec<usize> = Vec::new();
    for (i, &(dc, fc)) in points.iter().enumerate() {
        while hull.len() >= 2 {
            let (da, fa) = points[hull[hull.len() - 2]];
            let (db, fb) = points[hull[hull.len() - 1]];
            let left = Scaled::difference(fb, fa).mul(Scaled::new(dc - db));
            let right = Scaled::difference(fc, fb).mul(Scaled::new(db - da));
            if left.le(right) {
                break;
            }
            hull.pop();
        }
        hull.push(i);
    }
    hull.iter()
        .enumerate()
        .filter_map(|(position, &i)| {
            let (d, f) = points[i];
            let Some(&next) = hull.get(position + 1) else {
                return Some(i);
            };
            let (dn, fn_) = points[next];
            let upper = Scaled::difference(fn_, f).mul(Scaled::new(d));
            let required = Scaled::difference(f, best)
                .add(margin)
                .mul(Scaled::new(dn - d));
            (fn_ > f && required.le(upper)).then_some(i)
        })
        .collect()
}

// Separate binary exponents keep differences and products in range without
// erasing small costs beside large penalties. Fractions retain F's precision,
// and powers of two preserve exact collinearity even for subnormal inputs.
#[derive(Clone, Copy)]
struct Scaled<F> {
    fraction: F,
    exponent: i32,
}

impl<F: Scalar> Scaled<F> {
    fn new(value: F) -> Self {
        if value == F::zero() {
            return Self {
                fraction: F::zero(),
                exponent: 0,
            };
        }
        let (mantissa, exponent, sign) = value.integer_decode();
        let shift = u64::BITS - mantissa.leading_zeros() - 1;
        let fraction = F::from_u64(mantissa).unwrap()
            / F::from_u64(1_u64 << shift).unwrap();
        Self {
            fraction: if sign < 0 { -fraction } else { fraction },
            exponent: i32::from(exponent) + shift as i32,
        }
    }

    fn difference(a: F, b: F) -> Self {
        let difference = a - b;
        if difference.is_finite() {
            return Self::new(difference);
        }
        // Opposite extreme costs may overflow on subtraction, but halving
        // both operands preserves their difference with one extra exponent bit.
        let half = F::from_f64(0.5).unwrap();
        let mut difference = Self::new(a * half - b * half);
        difference.exponent += 1;
        difference
    }

    fn mul(self, other: Self) -> Self {
        let mut product = Self::new(self.fraction * other.fraction);
        product.exponent += self.exponent + other.exponent;
        product
    }

    fn add(self, other: Self) -> Self {
        if self.fraction == F::zero() {
            return other;
        }
        if other.fraction == F::zero() {
            return self;
        }
        if self.exponent < other.exponent {
            return other.add(self);
        }
        let factor = F::from_f64(2.0)
            .unwrap()
            .powi(other.exponent - self.exponent);
        let mut sum = Self::new(self.fraction + other.fraction * factor);
        sum.exponent += self.exponent;
        sum
    }

    fn le(self, other: Self) -> bool {
        if self.exponent == other.exponent
            || self.fraction * other.fraction <= F::zero()
        {
            self.fraction <= other.fraction
        } else if self.fraction > F::zero() {
            self.exponent < other.exponent
        } else {
            self.exponent > other.exponent
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
