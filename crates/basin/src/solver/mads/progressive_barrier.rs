// Index-based loops mirror the OrthoMADS / PB exposition; the lint is allowed.
#![allow(clippy::needless_range_loop)]

//! Progressive-barrier driver for constrained MADS (Audet & Dennis 2009;
//! Audet & Hare, *Derivative-Free and Blackbox Optimization*, 2nd ed., §10.5,
//! Algorithm 10.2).
//!
//! Where the [extreme barrier](super::driver) rejects every infeasible point
//! (`f = +∞`) and so needs a feasible start, the **progressive barrier** (PB)
//! handles general nonlinear inequality constraints `c(x) ≤ 0` by tracking the
//! aggregate constraint violation
//!
//! ```text
//! h(x) = Σⱼ max(cⱼ(x), 0)²     (h = 0 ⇔ feasible)
//! ```
//!
//! and exploring around **two** incumbents at once: the best **feasible** point
//! `x_feas` (least `f`), and the best **infeasible** point `x_inf` (the
//! nondominated infeasible point with `h ≤ h_max` and least `f`). The barrier
//! threshold `h_max` starts at `+∞` (so an infeasible start is allowed) and is
//! driven monotonically down to zero, abandoning ever-lower-violation infeasible
//! incumbents until the iterates reach feasibility.
//!
//! Each [`step`](PbWork::step) polls the `2n`-direction OrthoMADS set
//! `D_k = [H −H]` around both incumbents (reusing [`poll_directions`] and
//! [`mesh_and_poll_size`] from [`geometry`](super::geometry)), then classifies
//! the iteration and updates the mesh index `ℓ` and `h_max` per Figure 10.6:
//!
//! - **Dominating**: a trial dominates `x_feas` (`≺f`) or `x_inf` (`≺h`):
//!   coarsen the mesh (`ℓ−1`), set `h_max ← h(x_inf)`. Opportunistic poll may
//!   stop here.
//! - **Improving**: non-dominating, but some visited infeasible point has
//!   `0 < h < h(x_inf)`: keep the mesh, set
//!   `h_max ← max{h(v) : h(v) < h(x_inf)}` (abandon the current `x_inf`).
//! - **Unsuccessful**: neither: refine the mesh (`ℓ+1`), set
//!   `h_max ← h(x_inf)`.
//!
//! The dominance / filter logic here is a fresh PB implementation, **not** a
//! reuse of [`cobyla::filter`](crate::solver::cobyla): COBYLA's
//! `selectx`/`savefilt` encode a merit-`φ` selection over `cstrv = [maxⱼ cⱼ]₊`,
//! whereas PB selects two incumbents over the `(h, f)` Pareto front with the
//! squared-sum `h`. They are cross-referenced but share no code.
//!
//! [`poll_directions`]: super::geometry::poll_directions
//! [`mesh_and_poll_size`]: super::geometry::mesh_and_poll_size

use crate::core::math::Scalar;

use super::driver::Transition;
use super::geometry::{mesh_and_poll_size, poll_directions};
use super::halton::halton_seed;

/// The `(f, h)` oracle a [`PbWork`] step evaluates: it maps a flat point to its
/// objective value `f` and aggregate constraint violation `h`.
type PbEval<'a, F, E> = dyn FnMut(&[F]) -> Result<(F, F), E> + 'a;

/// The outcome of one [`PbWork::step`]: the transition plus the reported
/// incumbent `(x*, f*, h*)`: `x_feas` if a feasible point has been found, else
/// the current `x_inf`.
pub(crate) struct PbOutcome<F> {
    pub(crate) transition: Transition,
    pub(crate) incumbent: (Vec<F>, F, F),
}

/// Resumable progressive-barrier state carried on the
/// [`Mads`](crate::solver::Mads) solver in the constrained mode.
pub(crate) struct PbWork<F> {
    n: usize,
    /// Mesh index `ℓ` (OrthoMADS eq. (1)).
    ell: i32,
    /// Halton seed `t₀ = p_n`.
    t0: usize,
    /// Largest `ℓ` reached so far: detects "poll size is the smallest so far".
    ell_max: i32,
    /// Largest Halton index used so far.
    t_max: usize,
    /// Initial poll size `Δ₀` (uniform mesh scale); `Δᵖ = Δ₀·2^{-ℓ}`.
    scale: F,
    /// Convergence floor on the poll size.
    poll_size_min: F,
    /// Barrier threshold `h_max` (init `+∞`, nonincreasing toward `0`).
    h_max: F,
    /// Best feasible point found so far `(x, f)`, if any.
    feasible: Option<(Vec<F>, F)>,
    /// Every visited infeasible point `(x, f, h)` (`h > 0`). Needed in full: the
    /// "improving" rule takes a max over all visited violations below `h(x_inf)`,
    /// which can include dominated points.
    infeasible: Vec<(Vec<F>, F, F)>,
}

impl<F: Scalar> PbWork<F> {
    /// Evaluate the starting point and build the initial state. `eval` returns
    /// `(f, h)`: the objective and the aggregate violation. Returns the work
    /// plus the seeded incumbent `(x*, f*, h*)`.
    pub(crate) fn try_init<E>(
        x0: Vec<F>,
        poll_size_init: F,
        poll_size_min: F,
        eval: &mut PbEval<F, E>,
    ) -> Result<(Self, Vec<F>, F, F), E> {
        let n = x0.len();
        assert!(n >= 1, "Mads requires a non-empty start point");
        let (f0, h0) = eval(&x0)?;
        let mut work = Self {
            n,
            ell: 0,
            t0: halton_seed(n),
            ell_max: 0,
            t_max: 0,
            scale: poll_size_init,
            poll_size_min,
            h_max: F::infinity(),
            feasible: None,
            infeasible: Vec::new(),
        };
        work.record(x0, f0, h0);
        let (xs, fs, hs) = work.reported_incumbent();
        Ok((work, xs, fs, hs))
    }

    /// The current poll size `Δᵖ = Δ₀·2^{-ℓ}`.
    pub(crate) fn poll_size(&self) -> F {
        let (_, poll) = mesh_and_poll_size(self.ell);
        self.scale * F::from_f64(poll).expect("poll size representable")
    }

    /// The current mesh index `ℓ`.
    pub(crate) fn mesh_index(&self) -> i32 {
        self.ell
    }

    /// File a visited point into the feasible incumbent or the infeasible list.
    fn record(&mut self, x: Vec<F>, f: F, h: F) {
        if h <= F::zero() {
            // Feasible: keep only the least-f point (feasible dominance is f-only).
            match &self.feasible {
                Some((_, best)) if *best <= f => {}
                _ => self.feasible = Some((x, f)),
            }
        } else {
            self.infeasible.push((x, f, h));
        }
    }

    /// Index into `infeasible` of `x_inf`: the point with `h ≤ h_max` and least
    /// `f` (ties broken by least `h`), or `None` if no infeasible point lies
    /// within the threshold.
    fn select_x_inf(&self) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (i, (_, f, h)) in self.infeasible.iter().enumerate() {
            if *h > self.h_max {
                continue;
            }
            match best {
                Some(j) => {
                    let (_, fj, hj) = &self.infeasible[j];
                    if f < fj || (f == fj && h < hj) {
                        best = Some(i);
                    }
                }
                None => best = Some(i),
            }
        }
        best
    }

    /// The point reported to the solver: `x_feas` if one exists, else `x_inf`.
    /// At least one always exists (the start point was recorded).
    fn reported_incumbent(&self) -> (Vec<F>, F, F) {
        if let Some((x, f)) = &self.feasible {
            (x.clone(), *f, F::zero())
        } else {
            let i = self
                .select_x_inf()
                .expect("a recorded point always yields an incumbent");
            let (x, f, h) = &self.infeasible[i];
            (x.clone(), *f, *h)
        }
    }

    /// Run one PB iteration: poll the `2n` directions around both incumbents,
    /// classify, then update `ℓ` and `h_max`.
    pub(crate) fn step<E>(
        &mut self,
        eval: &mut PbEval<F, E>,
    ) -> Result<PbOutcome<F>, E> {
        // 1. Incumbents at the start of the iteration.
        let feas_pre = self.feasible.clone();
        let inf_pre = self.select_x_inf().map(|i| self.infeasible[i].clone());
        let h_inf_pre = inf_pre.as_ref().map(|t| t.2);

        // 2. Halton index for this iteration (OrthoMADS Fig. 2).
        let t = if self.ell >= self.ell_max {
            self.ell_max = self.ell;
            (self.ell + self.t0 as i32) as usize
        } else {
            self.t_max + 1
        };
        self.t_max = self.t_max.max(t);

        let (mesh_f64, _) = mesh_and_poll_size(self.ell);
        let mesh = self.scale
            * F::from_f64(mesh_f64).expect("mesh size representable");
        let dirs = poll_directions(t, self.ell, self.n);

        // Poll around both incumbents (those that exist). Opportunistic: stop on
        // the first dominating trial.
        let centers: Vec<Vec<F>> = [
            feas_pre.as_ref().map(|(x, _)| x.clone()),
            inf_pre.as_ref().map(|(x, _, _)| x.clone()),
        ]
        .into_iter()
        .flatten()
        .collect();

        let mut dominating = false;
        'poll: for center in &centers {
            for d in &dirs {
                let mut trial = center.clone();
                for i in 0..self.n {
                    let di = F::from_i64(d[i])
                        .expect("integer direction representable");
                    trial[i] = trial[i] + mesh * di;
                }
                let (f, h) = eval(&trial)?;
                let feasible = h <= F::zero();
                self.record(trial, f, h);

                // Dominating? (vs the pre-iteration incumbents.)
                if feasible {
                    // y ≺f x_feas.
                    if feas_pre.as_ref().is_none_or(|(_, ff)| f < *ff) {
                        dominating = true;
                        break 'poll;
                    }
                } else if let Some((_, fi, hi)) = &inf_pre {
                    // y ≺h x_inf: f ≤ f_inf and h ≤ h_inf, at least one strict.
                    if f <= *fi && h <= *hi && (f < *fi || h < *hi) {
                        dominating = true;
                        break 'poll;
                    }
                }
            }
        }

        // 3. Classify and update ℓ and h_max (Algorithm 10.2 step 3).
        if dominating {
            self.ell -= 1;
            if let Some(h) = h_inf_pre {
                self.h_max = h;
            }
        } else if let Some(hp) = h_inf_pre {
            // Improving iff some visited infeasible point has 0 < h < h(x_inf).
            let below: Vec<F> = self
                .infeasible
                .iter()
                .map(|(_, _, h)| *h)
                .filter(|h| *h < hp)
                .collect();
            if below.is_empty() {
                // Unsuccessful.
                self.ell += 1;
                self.h_max = hp;
            } else {
                // Improving: drop h_max just below the abandoned x_inf.
                let new_h = below.into_iter().fold(F::neg_infinity(), F::max);
                self.h_max = new_h;
            }
        } else {
            // No infeasible incumbent (only feasible points so far) and no
            // dominating feasible trial: an unsuccessful iteration.
            self.ell += 1;
        }

        let incumbent = self.reported_incumbent();
        let transition = if self.poll_size() <= self.poll_size_min {
            Transition::Converged
        } else {
            Transition::Continue
        };
        Ok(PbOutcome {
            transition,
            incumbent,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    /// `(f, h)` oracle over a flat point, for driving `PbWork` directly.
    fn run_steps<O>(x0: Vec<f64>, mut oracle: O, steps: usize) -> PbWork<f64>
    where
        O: FnMut(&[f64]) -> (f64, f64),
    {
        let mut eval =
            |x: &[f64]| -> Result<(f64, f64), Infallible> { Ok(oracle(x)) };
        let (mut work, _, _, _) = PbWork::try_init(x0, 1.0, 1e-9, &mut eval)
            .expect("init infallible");
        for _ in 0..steps {
            work.step(&mut eval).expect("step infallible");
        }
        work
    }

    /// h(x) = Σ max(cⱼ,0)² is exactly 0 on the feasible set and positive off it.
    #[test]
    fn record_routes_by_feasibility() {
        let mut w: PbWork<f64> = PbWork {
            n: 1,
            ell: 0,
            t0: 2,
            ell_max: 0,
            t_max: 0,
            scale: 1.0,
            poll_size_min: 1e-9,
            h_max: f64::INFINITY,
            feasible: None,
            infeasible: Vec::new(),
        };
        w.record(vec![0.0], 1.0, 0.0); // feasible
        w.record(vec![1.0], 0.5, 0.25); // infeasible
        assert_eq!(w.feasible, Some((vec![0.0], 1.0)));
        assert_eq!(w.infeasible.len(), 1);
    }

    /// x_inf is the least-f infeasible point within the h_max threshold.
    #[test]
    fn select_x_inf_picks_least_f_within_threshold() {
        let mut w: PbWork<f64> = PbWork {
            n: 1,
            ell: 0,
            t0: 2,
            ell_max: 0,
            t_max: 0,
            scale: 1.0,
            poll_size_min: 1e-9,
            h_max: 2.0,
            feasible: None,
            infeasible: vec![
                (vec![0.0], 5.0, 0.5),
                (vec![1.0], 1.0, 1.5),
                (vec![2.0], 0.0, 3.0), // h above threshold → excluded
            ],
        };
        assert_eq!(w.select_x_inf(), Some(1));
        w.h_max = 0.6; // now only the first qualifies
        assert_eq!(w.select_x_inf(), Some(0));
        w.h_max = 0.1; // none qualifies
        assert_eq!(w.select_x_inf(), None);
    }

    /// A feasible objective: min f = x on x ≥ 0 (c = −x ≤ 0). The PB threshold
    /// must walk down to 0 and the iterate reach feasibility near x = 0.
    #[test]
    fn drives_threshold_to_feasibility() {
        // Start infeasible at x = −1 (h = 1). f = x.
        let oracle = |x: &[f64]| {
            let xi = x[0];
            let c = -xi; // c ≤ 0 ⇔ x ≥ 0
            let viol = c.max(0.0);
            (xi, viol * viol)
        };
        let work = run_steps(vec![-1.0], oracle, 60);
        // h_max has been driven down to (near) zero and a feasible incumbent
        // with f ≈ 0 has been found.
        assert!(work.h_max <= 1e-6, "h_max = {}", work.h_max);
        let (_, f, h) = work.reported_incumbent();
        assert!(h <= 1e-9, "reported violation {h}");
        assert!(f.abs() < 1e-2, "reported f {f}");
    }
}
