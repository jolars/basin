//! Alternative BFGS model adapted from sisyfuzz 0.1.2 (Apache-2.0).
//! Copyright 2026 FransHenskens. See the crate COPYRIGHT notice.
//! Signed columns for n <= m; a scaled m-dimensional Cholesky otherwise.
use super::backend::AsFloatSlice;
use crate::Scalar;

pub(super) struct Model<F: Scalar> {
    theta: F,
    s: Vec<Vec<F>>,
    y: Vec<Vec<F>>,
    order: Vec<usize>,
    scales: Vec<F>,
    matrix: Vec<F>,
    factor: Vec<F>,
    scratch: Vec<F>,
    len: usize,
    small: bool,
    signed: Vec<Vec<F>>,
    alpha: Vec<F>,
    rho: Vec<F>,
    generation: Option<(u32, usize, F)>,
    valid: bool,
}
pub(super) struct Workspace<F: Scalar> {
    free: Vec<usize>,
    displacement: Vec<F>,
    product: Vec<F>,
    direction: Vec<F>,
    previous: Vec<F>,
    matrix: Vec<F>,
    rhs: Vec<F>,
    breakpoints: Vec<(F, usize)>,
    compact_displacement: Vec<F>,
    compact_direction: Vec<F>,
}
impl<F: Scalar> Workspace<F> {
    pub(super) fn new(n: usize, m: usize) -> Self {
        Self {
            free: Vec::with_capacity(n),
            displacement: vec![F::zero(); n],
            product: vec![F::zero(); n],
            direction: vec![F::zero(); n],
            previous: vec![F::zero(); n],
            matrix: vec![F::zero(); 4 * m * m],
            rhs: vec![F::zero(); (2 * m).max(n)],
            breakpoints: Vec::with_capacity(n),
            compact_displacement: vec![F::zero(); 2 * m],
            compact_direction: vec![F::zero(); 2 * m],
        }
    }
}
impl<F: Scalar> Model<F> {
    pub(super) fn new(n: usize, m: usize) -> Self {
        Self {
            theta: F::one(),
            s: vec![vec![F::zero(); n]; m],
            y: vec![vec![F::zero(); n]; m],
            order: Vec::with_capacity(m),
            scales: vec![F::zero(); 2 * m],
            matrix: vec![F::zero(); 4 * m * m],
            factor: vec![F::zero(); m * m],
            scratch: vec![F::zero(); 2 * m],
            len: 0,
            small: n <= m,
            signed: if n <= m {
                vec![vec![F::zero(); n]; 2 * m]
            } else {
                Vec::new()
            },
            alpha: vec![F::zero(); m],
            rho: vec![F::zero(); m],
            generation: None,
            valid: false,
        }
    }
    // Inverse policies need raw pairs and scaling, but no signed columns or
    // compact factor. Invalidate derived data so a later bounded fallback
    // rebuilds from the current history, including delayed evictions/resets.
    pub(super) fn prepare_inverse<V: AsFloatSlice<F>>(
        &mut self,
        ws: &[V],
        wy: &[V],
        theta: F,
    ) -> bool {
        self.valid = false;
        self.generation = None;
        self.theta = theta;
        self.len = 2 * ws.len();
        self.order.clear();
        self.order.extend(0..ws.len());
        if !theta.is_finite() || theta <= F::zero() {
            return false;
        }
        for j in 0..ws.len() {
            self.s[j].copy_from_slice(ws[j].as_float_slice());
            self.y[j].copy_from_slice(wy[j].as_float_slice());
            if !self.s[j].iter().chain(&self.y[j]).all(|v| v.is_finite()) {
                return false;
            }
        }
        true
    }
    pub(super) fn rebuild<V: AsFloatSlice<F>>(
        &mut self,
        ws: &[V],
        wy: &[V],
        sy: &[F],
        ss: &[F],
        theta: F,
        update: u32,
    ) -> bool {
        let h = ws.len();
        let m = self.s.len();
        let generation = (update, h, theta);
        if self.valid
            && update != u32::MAX
            && self.generation == Some(generation)
        {
            return true;
        }
        if !self.prepare_inverse(ws, wy, theta) {
            return false;
        }
        self.generation = Some(generation);
        if self.small {
            for a in 0..h {
                let s = &self.s[a];
                let y = &self.y[a];
                let (live, unused) = self.signed.split_at_mut(2 * a);
                for (out, v) in unused[0].iter_mut().zip(s) {
                    *out = theta * *v;
                }
                for (j, column) in live.iter().enumerate() {
                    let coefficient = dot(column, s)
                        * if j % 2 == 0 { -F::one() } else { F::one() };
                    for (out, v) in unused[0].iter_mut().zip(column) {
                        *out = *out + coefficient * *v;
                    }
                }
                let sbs = dot(s, &unused[0]);
                let sy = sy[a * m + a];
                if !sbs.is_finite()
                    || sbs <= F::zero()
                    || !sy.is_finite()
                    || sy <= F::zero()
                {
                    return false;
                }
                for out in &mut unused[0] {
                    *out = *out / sbs.sqrt();
                }
                for (out, v) in unused[1].iter_mut().zip(y) {
                    *out = *v / sy.sqrt();
                }
            }
            let k = self.len;
            self.matrix[..k * k].fill(F::zero());
            for j in 0..k {
                self.matrix[j * k + j] =
                    if j % 2 == 0 { F::one() } else { -F::one() };
            }
            self.valid =
                self.signed[..k].iter().flatten().all(|v| v.is_finite());
            return self.valid;
        }
        for a in 0..h {
            if !ss[a * m + a].is_finite()
                || ss[a * m + a] <= F::zero()
                || !sy[a * m + a].is_finite()
                || sy[a * m + a] <= F::zero()
            {
                return false;
            }
            self.scales[a] = theta.sqrt() / ss[a * m + a].sqrt();
            self.scales[h + a] = F::one() / sy[a * m + a].sqrt();
        }
        let k = self.len;
        self.matrix[..k * k].fill(F::zero());
        for a in 0..h {
            for b in 0..h {
                self.matrix[a * k + b] =
                    ss[a * m + b] * (self.scales[a] / theta) * self.scales[b];
                if a > b {
                    let v = sy[a * m + b]
                        * (self.scales[a] / theta)
                        * self.scales[h + b];
                    self.matrix[a * k + h + b] = v;
                    self.matrix[(h + b) * k + a] = v;
                }
            }
            self.matrix[(h + a) * k + h + a] = -F::one();
        }
        self.valid = self.matrix[..k * k].iter().all(|v| v.is_finite())
            && factorize(&self.matrix[..k * k], &mut self.factor[..h * h], h);
        self.valid
    }
    pub(super) fn inverse(
        &mut self,
        g: &[F],
        free: Option<&[bool]>,
        out: &mut [F],
    ) {
        let h = self.len / 2;
        let selected = |i: usize| free.is_none_or(|f| f[i]);
        let mut scale = if free.is_some() {
            F::one()
        } else {
            F::one() / self.theta
        };
        for a in 0..h {
            let mut sy = F::zero();
            let mut ss = F::zero();
            let mut yy = F::zero();
            for i in 0..g.len() {
                if selected(i) {
                    let s = self.s[a][i];
                    let y = self.y[a][i];
                    sy = sy + s * y;
                    ss = ss + s * s;
                    yy = yy + y * y;
                }
            }
            let threshold = if free.is_some() {
                F::from_f64(1e-10).unwrap()
            } else {
                F::zero()
            };
            self.rho[a] = if sy.is_finite()
                && yy.is_finite()
                && ss.is_finite()
                && sy > threshold * ss.sqrt() * yy.sqrt()
                && (F::one() / sy).is_finite()
                && (sy / yy).is_finite()
                && sy / yy > F::zero()
            {
                F::one() / sy
            } else {
                F::zero()
            };
            if self.rho[a] > F::zero() && free.is_some() {
                scale = sy / yy;
            }
        }
        for i in 0..g.len() {
            out[i] = if selected(i) { g[i] } else { F::zero() };
        }
        for a in (0..h).rev() {
            if self.rho[a] == F::zero() {
                continue;
            }
            self.alpha[a] = self.rho[a] * dot(&self.s[a], out);
            for i in 0..g.len() {
                if selected(i) {
                    out[i] = out[i] - self.alpha[a] * self.y[a][i];
                }
            }
        }
        for v in out.iter_mut() {
            *v = *v * scale;
        }
        for a in 0..h {
            if self.rho[a] == F::zero() {
                continue;
            }
            let beta = self.rho[a] * dot(&self.y[a], out);
            for i in 0..g.len() {
                if selected(i) {
                    out[i] = out[i] + (self.alpha[a] - beta) * self.s[a][i];
                }
            }
        }
    }
    pub(super) fn value(&self, column: usize, coordinate: usize) -> F {
        if self.small {
            return self.signed[column][coordinate];
        }
        let h = self.order.len();
        if column < h {
            self.s[self.order[column]][coordinate] * self.scales[column]
        } else {
            self.y[self.order[column - h]][coordinate] * self.scales[column]
        }
    }
    pub(super) fn apply(&mut self, vector: &[F], out: &mut [F]) {
        let k = self.len;
        let h = self.order.len();
        if self.small {
            for (o, v) in out.iter_mut().zip(vector) {
                *o = self.theta * *v;
            }
            for (j, column) in self.signed[..k].iter().enumerate() {
                let coefficient = dot(column, vector)
                    * if j % 2 == 0 { -F::one() } else { F::one() };
                for (o, v) in out.iter_mut().zip(column) {
                    *o = *o + (coefficient * *v);
                }
            }
            return;
        }
        for j in 0..k {
            let column = if j < h {
                &self.s[self.order[j]]
            } else {
                &self.y[self.order[j - h]]
            };
            self.scratch[j] = dot(column, vector) * self.scales[j];
        }
        self.solve_scratch();
        for (o, v) in out.iter_mut().zip(vector) {
            *o = self.theta * *v;
        }
        for j in 0..k {
            let column = if j < h {
                &self.s[self.order[j]]
            } else {
                &self.y[self.order[j - h]]
            };
            let coefficient = self.scales[j] * self.scratch[j];
            for (o, v) in out.iter_mut().zip(column) {
                *o = *o - (coefficient * *v);
            }
        }
    }
    fn solve_scratch(&mut self) {
        let k = self.len;
        if self.small {
            for j in 0..k {
                if j % 2 == 1 {
                    self.scratch[j] = -self.scratch[j];
                }
            }
        } else {
            solve(
                &self.matrix[..k * k],
                &self.factor[..k * k / 4],
                &mut self.scratch[..k],
            );
        }
    }
    pub(super) fn cauchy(
        &mut self,
        x: &[F],
        g: &[F],
        work: &mut Workspace<F>,
        z: &mut [F],
        lower: &[F],
        upper: &[F],
    ) {
        let k = self.len;
        work.breakpoints.clear();
        for i in 0..x.len() {
            work.direction[i] = if blocked(x[i], -g[i], lower[i], upper[i]) {
                F::zero()
            } else {
                -g[i]
            };
            let d = work.direction[i];
            if d != F::zero() {
                let distance = if d > F::zero() {
                    upper[i] - x[i]
                } else {
                    lower[i] - x[i]
                };
                let hit = distance / d;
                if hit.is_finite() {
                    work.breakpoints.push((hit, i));
                }
            }
        }
        work.breakpoints.sort_unstable_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1))
        });
        work.previous.copy_from_slice(&work.direction);
        work.compact_displacement[..k].fill(F::zero());
        for j in 0..k {
            if self.small {
                work.compact_direction[j] =
                    dot(&self.signed[j], &work.direction);
                continue;
            }
            let h = self.order.len();
            let column = if j < h {
                &self.s[self.order[j]]
            } else {
                &self.y[self.order[j - h]]
            };
            work.compact_direction[j] =
                dot(column, &work.direction) * self.scales[j];
        }
        let mut gd = dot(g, &work.direction);
        let mut dd = dot(&work.direction, &work.direction);
        let mut displacement_d = F::zero();
        let mut time = F::zero();
        let mut next = 0;
        loop {
            self.scratch[..k].copy_from_slice(&work.compact_direction[..k]);
            self.solve_scratch();
            let derivative = gd + self.theta * displacement_d
                - dot(&work.compact_displacement[..k], &self.scratch[..k]);
            let curvature = self.theta * dd
                - dot(&work.compact_direction[..k], &self.scratch[..k]);
            if !derivative.is_finite()
                || derivative >= F::zero()
                || !curvature.is_finite()
                || curvature <= F::zero()
            {
                break;
            }
            let boundary_time =
                work.breakpoints.get(next).map_or(F::infinity(), |v| v.0);
            let limit = (boundary_time - time).max(F::zero());
            let stationary = -derivative / curvature;
            if stationary < limit {
                time = time + (stationary);
                break;
            }
            for j in 0..k {
                work.compact_displacement[j] = work.compact_displacement[j]
                    + (limit * work.compact_direction[j]);
            }
            displacement_d = displacement_d + (limit * dd);
            time = boundary_time;
            // Process simultaneous endpoints together. Remove just the affected
            // coordinates from the compact products instead of reapplying B.
            while next < work.breakpoints.len()
                && work.breakpoints[next].0 == boundary_time
            {
                let i = work.breakpoints[next].1;
                let d = work.direction[i];
                gd = gd - (g[i] * d);
                dd = (dd - d * d).max(F::zero());
                let distance = if d > F::zero() {
                    upper[i] - x[i]
                } else {
                    lower[i] - x[i]
                };
                displacement_d = displacement_d - (distance * d);
                for j in 0..k {
                    work.compact_direction[j] =
                        work.compact_direction[j] - (self.value(j, i) * d);
                }
                next += 1;
            }
        }
        box_step(x, &work.previous, time, lower, upper, z);
    }
    pub(super) fn subspace(
        &mut self,
        x: &[F],
        g: &[F],
        z: &mut [F],
        work: &mut Workspace<F>,
        lower: &[F],
        upper: &[F],
    ) {
        work.free.clear();
        work.free.extend(
            (0..z.len()).filter(|&i| z[i] > lower[i] && z[i] < upper[i]),
        );
        if work.free.is_empty() || self.len == 0 {
            return;
        }
        if work.free.len() == x.len() {
            // -H(g+B(z-x)) = -Hg-(z-x). This preserves the correction
            // ray from z, including truncation at bounds, without forming B(z-x).
            self.inverse(g, None, &mut work.direction);
            for i in 0..x.len() {
                work.direction[i] = x[i] - z[i] - work.direction[i];
            }
            if work.direction.iter().any(|v| !v.is_finite()) {
                return;
            }
            for (j, &i) in work.free.iter().enumerate() {
                work.rhs[j] = work.direction[i];
            }
            super::subsm::project(
                z,
                &mut work.rhs[..work.free.len()],
                &mut work.previous,
                x,
                g,
                &work.free,
                lower,
                upper,
            );
            return;
        }
        for i in 0..x.len() {
            work.displacement[i] = z[i] - x[i];
        }
        self.apply(&work.displacement, &mut work.product);
        for (r, g) in work.product.iter_mut().zip(g) {
            *r = *r + *g;
        }
        work.direction.fill(F::zero());
        if work.free.len() < self.len {
            let nfree = work.free.len();
            let k = self.len;
            for (j, &coordinate) in work.free.iter().enumerate() {
                work.rhs[j] = -work.product[coordinate];
                for a in 0..k {
                    self.scratch[a] = self.value(a, coordinate);
                }
                self.solve_scratch();
                for (i, &other) in work.free.iter().enumerate().take(j + 1) {
                    let value = if i == j { self.theta } else { F::zero() }
                        - (0..k)
                            .map(|a| self.value(a, other) * self.scratch[a])
                            .sum::<F>();
                    work.matrix[i * nfree + j] = value;
                    work.matrix[j * nfree + i] = value;
                }
            }
            if !solve_pivoted(
                &mut work.matrix[..nfree * nfree],
                &mut work.rhs[..nfree],
            ) {
                return;
            }
            for (j, &i) in work.free.iter().enumerate() {
                work.direction[i] = work.rhs[j];
            }
        } else {
            let k = self.len;
            for j in 0..k {
                work.rhs[j] = work
                    .free
                    .iter()
                    .map(|&i| self.value(j, i) * work.product[i])
                    .sum::<F>()
                    / self.theta;
                for l in 0..=j {
                    let value = self.matrix[j * k + l]
                        - work
                            .free
                            .iter()
                            .map(|&i| self.value(j, i) * self.value(l, i))
                            .sum::<F>()
                            / self.theta;
                    work.matrix[j * k + l] = value;
                    work.matrix[l * k + j] = value;
                }
            }
            if !solve_pivoted(&mut work.matrix[..k * k], &mut work.rhs[..k]) {
                return;
            }
            for &i in &work.free {
                work.direction[i] = (-work.product[i]
                    - (0..k)
                        .map(|j| self.value(j, i) * work.rhs[j])
                        .sum::<F>())
                    / self.theta;
            }
        }
        if work.direction.iter().any(|d| !d.is_finite())
            || dot(&work.direction, &work.product) >= F::zero()
        {
            return;
        }
        for (j, &i) in work.free.iter().enumerate() {
            work.rhs[j] = work.direction[i];
        }
        super::subsm::project(
            z,
            &mut work.rhs[..work.free.len()],
            &mut work.previous,
            x,
            g,
            &work.free,
            lower,
            upper,
        );
    }
}

fn factorize<F: Scalar>(matrix: &[F], factor: &mut [F], n: usize) -> bool {
    let k = 2 * n;
    for i in 0..n {
        for j in 0..=i {
            let mut value = matrix[i * k + j];
            for l in 0..j {
                value = value + (matrix[i * k + n + l] * matrix[j * k + n + l]);
            }
            for l in 0..j {
                value = value - (factor[i * n + l] * factor[j * n + l]);
            }
            if i == j {
                if !value.is_finite() || value <= F::zero() {
                    return false;
                }
                factor[i * n + j] = value.sqrt();
            } else {
                factor[i * n + j] = value / factor[j * n + j];
            }
        }
    }
    factor.iter().all(|v| v.is_finite())
}
fn solve<F: Scalar>(matrix: &[F], factor: &[F], rhs: &mut [F]) {
    let n = rhs.len() / 2;
    let k = 2 * n;
    for i in 0..n {
        for j in 0..i {
            rhs[i] = rhs[i] + (matrix[i * k + n + j] * rhs[n + j]);
        }
    }
    for i in 0..n {
        for j in 0..i {
            rhs[i] = rhs[i] - (factor[i * n + j] * rhs[j]);
        }
        rhs[i] = rhs[i] / (factor[i * n + i]);
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            rhs[i] = rhs[i] - (factor[j * n + i] * rhs[j]);
        }
        rhs[i] = rhs[i] / (factor[i * n + i]);
    }
    for i in 0..n {
        let mut value = -rhs[n + i];
        for j in i + 1..n {
            value = value + (matrix[j * k + n + i] * rhs[j]);
        }
        rhs[n + i] = value;
    }
}
fn solve_pivoted<F: Scalar>(matrix: &mut [F], rhs: &mut [F]) -> bool {
    let n = rhs.len();
    let norm = matrix.iter().map(|v| v.abs()).fold(F::zero(), F::max);
    for j in 0..n {
        let pivot = (j..n)
            .max_by(|&a, &b| {
                matrix[a * n + j]
                    .abs()
                    .partial_cmp(&matrix[b * n + j].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        if !matrix[pivot * n + j].is_finite()
            || matrix[pivot * n + j].abs() <= F::epsilon() * norm
        {
            return false;
        }
        for k in 0..n {
            matrix.swap(j * n + k, pivot * n + k);
        }
        rhs.swap(j, pivot);
        for i in j + 1..n {
            let factor = matrix[i * n + j] / matrix[j * n + j];
            for k in j + 1..n {
                matrix[i * n + k] =
                    matrix[i * n + k] - (factor * matrix[j * n + k]);
            }
            rhs[i] = rhs[i] - (factor * rhs[j]);
        }
    }
    for i in (0..n).rev() {
        rhs[i] = (rhs[i]
            - (i + 1..n).map(|j| matrix[i * n + j] * rhs[j]).sum::<F>())
            / matrix[i * n + i];
    }
    rhs.iter().all(|v| v.is_finite())
}

fn dot<F: Scalar>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b).fold(F::zero(), |v, (a, b)| v + *a * *b)
}
fn blocked<F: Scalar>(x: F, d: F, l: F, u: F) -> bool {
    (x <= l && d < F::zero()) || (x >= u && d > F::zero()) || l == u
}
fn box_step<F: Scalar>(
    x: &[F],
    d: &[F],
    t: F,
    l: &[F],
    u: &[F],
    out: &mut [F],
) {
    for i in 0..x.len() {
        out[i] = (x[i] + t * d[i]).max(l[i]).min(u[i]);
    }
}
