//! OrthoMADS driver: the resumable poll loop and mesh/Halton bookkeeping.
//!
//! [`MadsWork`] holds the incumbent and the integer mesh index `ℓ`, and runs one
//! OrthoMADS iteration per [`step`](MadsWork::step): generate the poll set
//! `D_k = [H −H]` (from the Halton index `t_k` and `ℓ`), poll opportunistically,
//! and update `ℓ` (success → `ℓ−1`, failure → `ℓ+1`). The incumbent is monotone
//! non-increasing (only improving mesh points are accepted). All sizes scale by
//! the configured initial poll size `Δ₀`, which keeps the trial points on the
//! (scaled) integer mesh. See Abramson et al. 2009, Figure 2, and
//! `references/orthomads-2009/NOTES.md`.

use crate::core::math::Scalar;

use super::geometry::{mesh_and_poll_size, poll_directions};
use super::halton::halton_seed;

/// What happened over one [`MadsWork::step`].
pub(crate) enum Transition {
    /// Keep going (a successful or unsuccessful iteration that has not yet
    /// shrunk the poll size to the floor).
    Continue,
    /// The poll size reached the configured floor — MADS's natural convergence.
    Converged,
}

/// The outcome of one step: the transition plus any improving evaluation (the
/// new incumbent), for the solver to fold into best-tracking.
pub(crate) struct StepOutcome<F> {
    pub(crate) transition: Transition,
    /// The accepted improving point (`(x, f(x))`) on a successful iteration,
    /// empty on an unsuccessful one. Every polled point is still counted by the
    /// `Problem` wrapper regardless.
    pub(crate) evaluated: Vec<(Vec<F>, F)>,
}

/// Resumable OrthoMADS state carried on the [`Mads`](crate::solver::Mads) solver.
pub(crate) struct MadsWork<F> {
    n: usize,
    /// Current incumbent (flat).
    x: Vec<F>,
    /// `f(x)`.
    fx: F,
    /// Mesh index `ℓ` (OrthoMADS eq. (1)).
    ell: i32,
    /// Halton seed `t₀ = p_n`.
    t0: usize,
    /// Largest `ℓ` reached so far — detects "poll size is the smallest so far".
    ell_max: i32,
    /// Largest Halton index used so far — feeds the non-smallest-poll branch.
    t_max: usize,
    /// Initial poll size `Δ₀` (uniform mesh scale); `Δᵖ = Δ₀·2^{-ℓ}`.
    scale: F,
    /// Convergence floor on the poll size.
    poll_size_min: F,
}

impl<F: Scalar> MadsWork<F> {
    /// Evaluate the starting point and build the initial state. Returns the work
    /// plus the seeded incumbent `(x₀, f(x₀))`.
    pub(crate) fn try_init<E>(
        x0: Vec<F>,
        poll_size_init: F,
        poll_size_min: F,
        eval: &mut dyn FnMut(&[F]) -> Result<F, E>,
    ) -> Result<(Self, Vec<F>, F), E> {
        let n = x0.len();
        assert!(n >= 1, "Mads requires a non-empty start point");
        let fx = eval(&x0)?;
        let work = Self {
            n,
            x: x0.clone(),
            fx,
            ell: 0,
            t0: halton_seed(n),
            ell_max: 0,
            t_max: 0,
            scale: poll_size_init,
            poll_size_min,
        };
        Ok((work, x0, fx))
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

    /// Run one OrthoMADS iteration: choose the Halton index, poll the
    /// `2n`-direction set opportunistically, then update `ℓ`.
    pub(crate) fn step<E>(
        &mut self,
        eval: &mut dyn FnMut(&[F]) -> Result<F, E>,
    ) -> Result<StepOutcome<F>, E> {
        // Halton index for this iteration (OrthoMADS Fig. 2): if the poll size is
        // the smallest seen so far (ℓ at a new/tied max), tie it to ℓ; else take
        // a fresh index past every one used before.
        let t = if self.ell >= self.ell_max {
            self.ell_max = self.ell;
            (self.ell + self.t0 as i32) as usize // ℓ ≥ 0 in this branch
        } else {
            self.t_max + 1
        };
        self.t_max = self.t_max.max(t);

        let (mesh_f64, _) = mesh_and_poll_size(self.ell);
        let mesh = self.scale * F::from_f64(mesh_f64).expect("mesh size representable");
        let dirs = poll_directions(t, self.ell, self.n);

        // Opportunistic poll: move to the first improving mesh point.
        let mut evaluated = Vec::new();
        let mut improved = false;
        for d in &dirs {
            let mut trial = self.x.clone();
            for i in 0..self.n {
                let di = F::from_i64(d[i]).expect("integer direction representable");
                trial[i] = trial[i] + mesh * di;
            }
            let f = eval(&trial)?;
            if f < self.fx {
                self.x = trial.clone();
                self.fx = f;
                evaluated.push((trial, f));
                improved = true;
                break;
            }
        }

        // Mesh update: success coarsens (ℓ−1), failure refines (ℓ+1).
        if improved {
            self.ell -= 1;
        } else {
            self.ell += 1;
        }

        let transition = if self.poll_size() <= self.poll_size_min {
            Transition::Converged
        } else {
            Transition::Continue
        };
        Ok(StepOutcome {
            transition,
            evaluated,
        })
    }
}
