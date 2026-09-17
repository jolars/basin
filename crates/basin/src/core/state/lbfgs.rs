//! Limited-memory BFGS/L-BFGS-B state.
//!
//! Carries the current iterate plus the limited-memory history
//! `(s_k, y_k)` capped at `m_capacity` pairs, and the compact-form
//! Gram matrices `SᵀY` and `SᵀS`. Mirrors the Fortran v3.0 storage
//! (`ws`, `wy`, `sy`, `ss`, `theta` in `references/lbfgsb-v3.0/`)
//! but keeps the history chronologically (oldest at index 0) rather
//! than in a ring buffer with a `head` pointer. The numerical result
//! is identical because cauchy/subsm only require "column j of W
//! in oldest-to-newest order" and never depend on the ring-buffer
//! index modulo `m_capacity`.

use crate::core::math::{Dot, Scalar};
use crate::core::problem::EvalCounts;
use crate::core::state::{CountsMirror, GradientState, State};
use crate::solver::lbfgs::{backend::AsFloatSlice, history::newest_products};

/// Solver state for L-BFGS-B and the unbounded L-BFGS solver.
///
/// `theta` initializes to `1`; after the first accepted update
/// it becomes `(y · y) / (s · y)`, matching the Fortran convention
/// at `mainlb`'s `matupd` call site.
///
/// The scalar `F` defaults to `f64` so existing `LbfgsState<V>` call
/// sites resolve unchanged. Both the bounded and unbounded paths now
/// run F-generic; the L-BFGS-B-specific `work` buffer
/// (`LbfgsbWork<F>`, private) carries the same scalar.
pub struct LbfgsState<V, F = f64> {
    pub(crate) param: V,
    pub(crate) cost: Option<F>,
    pub(crate) gradient: Option<V>,

    /// History capacity. Fortran's `m`; recommended `[3, 20]`.
    pub(crate) m_capacity: usize,
    /// `s_k = x_{k+1} − x_k`, chronological, oldest first.
    pub(crate) ws: Vec<V>,
    /// `y_k = g_{k+1} − g_k`, same length and order as `ws`.
    pub(crate) wy: Vec<V>,
    /// Lower triangle of `SᵀY` as row-major `m_capacity²` storage;
    /// only the leading `col × col` block (`col = ws.len()`) is live.
    pub(crate) sy: Vec<F>,
    /// `SᵀS`, same row-major layout as `sy`.
    pub(crate) ss: Vec<F>,
    /// Compact-form scaling. `1` until first accepted update,
    /// thereafter `(y · y) / (s · y)`.
    pub(crate) theta: F,

    pub(crate) iter: u64,
    pub(crate) cost_evals: u64,
    pub(crate) gradient_evals: u64,

    pub(crate) best_param: Option<V>,
    pub(crate) best_cost: F,
    pub(crate) best_iter: u64,
    pub(crate) best_cost_evals: u64,
    pub(crate) best_gradient_evals: u64,

    /// Working buffers and persistent solver-side scalars for the
    /// L-BFGS-B iteration (Fortran's `mainlb` scratch arrays plus the
    /// pieces of `isave`/`dsave` that survive across iterations).
    /// Initialized lazily by `LBFGSB::init`; absent when [`LbfgsState`]
    /// is used by other solvers (e.g. the unbounded L-BFGS path).
    // Indirection keeps workspace headers out of the state moved through
    // the executor on every iteration.
    pub(crate) work: Option<Box<LbfgsbWork<F>>>,
}

/// Mutable working storage threaded through the L-BFGS-B iteration.
///
/// Allocates once in [`crate::solver::LBFGSB::init`] and is reused
/// across every [`crate::core::solver::Solver::next_iter`] call.
/// Mirrors the layout Fortran `mainlb` carves out of the user-
/// supplied scratch arrays (`ws`, `wy`, `sy`, `ss`, `wt`, `wn`, `snd`,
/// `z`, `r`, `d`, `t`, `xp`, `wa`, `index`, `iwhere`, `indx2`) plus
/// the iteration-persistent scalars that live in `isave`/`dsave`
/// between coroutine returns.
///
/// Stored on [`LbfgsState`] rather than the solver struct so that
/// [`crate::core::solver::Solver`] implementations stay
/// configuration-only (mirroring [`crate::solver::BFGS`]).
pub(crate) struct LbfgsbWork<F = f64> {
    // ---- Compact-form matrices ----
    /// `2m × 2m` row-major; stores the `L·E·Lᵀ` factor of the
    /// indefinite middle matrix `K`. Output of `formk`, consumed by
    /// `subsm`.
    pub(crate) wn: Vec<F>,
    /// `2m × 2m` row-major; the lower-triangular `N` Gram cache that
    /// `formk` maintains incrementally across outer iterations.
    pub(crate) wn1: Vec<F>,
    /// `m × m` row-major; Cholesky factor of `T = θ SᵀS + LD⁻¹Lᵀ`,
    /// produced by `formt`, consumed by `bmv` inside `cauchy`.
    pub(crate) wt: Vec<F>,

    // ---- n-sized working vectors ----
    /// Cauchy point/subspace Newton point (Fortran `z`).
    pub(crate) z: Vec<F>,
    /// Reduced gradient at the Cauchy point (Fortran `r`).
    pub(crate) r: Vec<F>,
    /// Search direction `d = z − x` (Fortran `d`).
    pub(crate) d: Vec<F>,
    /// Cauchy breakpoint buffer/line-search previous iterate
    /// (Fortran `t`).
    pub(crate) t_buf: Vec<F>,
    /// Subspace projected-Newton safeguard slot (Fortran `xp`).
    pub(crate) xp: Vec<F>,

    // ---- 2m-sized cauchy/subsm scratch ----
    /// `Wᵀ d` accumulator inside `cauchy` (Fortran `wa(1..2m)`).
    pub(crate) wa_p: Vec<F>,
    /// `Wᵀ (xcp − x)` accumulator (Fortran `wa(2m+1..4m)`); fed to
    /// `subsm` via `cmprlb`.
    pub(crate) wa_c: Vec<F>,
    /// Breakpoint `W` row inside `cauchy` (Fortran `wa(4m+1..6m)`).
    pub(crate) wa_wbp: Vec<F>,
    /// Middle-matrix solve scratch (Fortran `wa(6m+1..8m)`); reused
    /// by `subsm` as `wv`.
    pub(crate) wa_v: Vec<F>,

    // ---- Integer working arrays ----
    /// Cauchy-point variable classification (`FREE_NOT_MOVED`, etc.).
    pub(crate) iwhere: Vec<i8>,
    /// Free + active partition at the GCP, free first.
    pub(crate) index: Vec<usize>,
    /// Entering + leaving variables since the previous GCP. Doubles
    /// as `iorder` inside `cauchy` (the breakpoint heap).
    pub(crate) indx2: Vec<usize>,

    // ---- Iteration-persistent scalars/flags ----
    /// True iff at least one variable has a finite bound.
    pub(crate) cnstnd: bool,
    /// True iff every variable is two-sided (both bounds finite).
    pub(crate) boxed: bool,
    /// True iff the limited-memory history was updated in the
    /// previous outer iteration.
    pub(crate) updatd: bool,
    /// Total number of accepted BFGS updates (Fortran `iupdat`).
    pub(crate) iupdat: u32,
    /// `‖d‖` from the last line search; used on subsequent calls to
    /// set the initial step.
    pub(crate) dnorm: F,
    /// `−gᵀd` from the previous line search (Fortran `gdold`); needed
    /// for the curvature-skip threshold.
    pub(crate) gdold: F,
    /// Number of free variables at the GCP (`nfree`). Carried across
    /// iterations because `freev` uses the *previous* `index` to
    /// detect leaving variables.
    pub(crate) nfree: usize,
}

impl<F: Scalar> LbfgsbWork<F> {
    /// Pre-allocate every buffer to its required size given the
    /// problem dimension `n` and history capacity `m`.
    pub(crate) fn new(n: usize, m: usize) -> Self {
        let two_m = 2 * m;
        Self {
            wn: vec![F::zero(); two_m * two_m],
            wn1: vec![F::zero(); two_m * two_m],
            wt: vec![F::zero(); m * m],
            z: vec![F::zero(); n],
            r: vec![F::zero(); n],
            d: vec![F::zero(); n],
            t_buf: vec![F::zero(); n],
            xp: vec![F::zero(); n],
            wa_p: vec![F::zero(); two_m],
            wa_c: vec![F::zero(); two_m],
            wa_wbp: vec![F::zero(); two_m],
            wa_v: vec![F::zero(); two_m],
            iwhere: vec![0; n],
            index: (0..n).collect(),
            indx2: vec![0; n],
            cnstnd: false,
            boxed: true,
            updatd: false,
            iupdat: 0,
            dnorm: F::zero(),
            gdold: F::zero(),
            nfree: n,
        }
    }

    /// Reset the limited-memory state for an iteration restart
    /// (matches Fortran `col = 0; head = 1; theta = 1; iupdat = 0;
    /// updatd = false`).
    pub(crate) fn reset_history(&mut self) {
        self.iupdat = 0;
        self.updatd = false;
    }
}

impl<V, F: Scalar> LbfgsState<V, F> {
    /// Build state at the given starting point with capacity for
    /// `m_capacity` history pairs. Use `m_capacity = 10` as a
    /// reasonable default; Fortran recommends `[3, 20]`.
    ///
    /// # Panics
    ///
    /// Panics if `m_capacity == 0`.
    pub fn new(param: V, m_capacity: usize) -> Self {
        assert!(m_capacity >= 1, "m_capacity must be ≥ 1");
        let mm = m_capacity * m_capacity;
        Self {
            param,
            cost: None,
            gradient: None,
            m_capacity,
            ws: Vec::with_capacity(m_capacity),
            wy: Vec::with_capacity(m_capacity),
            sy: vec![F::zero(); mm],
            ss: vec![F::zero(); mm],
            theta: F::one(),
            iter: 0,
            cost_evals: 0,
            gradient_evals: 0,
            best_param: None,
            best_cost: F::infinity(),
            best_iter: 0,
            best_cost_evals: 0,
            best_gradient_evals: 0,
            work: None,
        }
    }

    /// Current history length (`col` in Fortran). In `[0, m_capacity]`.
    /// Used by tests; the solver inlines `state.ws.len()` directly.
    #[allow(dead_code)]
    pub(crate) fn col(&self) -> usize {
        self.ws.len()
    }

    /// Append a `(s, y)` pair to the history and update `sy`, `ss`,
    /// `theta`. When the history is at capacity, the oldest pair is
    /// dropped (left shift on `ws`, `wy`, and the leading block of
    /// `sy`, `ss`).
    ///
    /// Returns `false` if `s·y ≤ 0` or any product is non-finite:
    /// the curvature condition is the caller's responsibility, this
    /// is just a final safeguard. The state is left unchanged in
    /// that case.
    pub(crate) fn append_pair(&mut self, s: V, y: V) -> bool
    where
        V: Dot<F> + AsFloatSlice<F>,
    {
        let sy_dot = s.dot(&y);
        let yy_dot = y.dot(&y);
        if !(sy_dot > F::zero() && sy_dot.is_finite() && yy_dot.is_finite()) {
            return false;
        }

        let m = self.m_capacity;

        // Drop oldest when at capacity.
        if self.ws.len() == m {
            self.ws.remove(0);
            self.wy.remove(0);
            // Shift the leading `(m-1) × (m-1)` block of sy and ss
            // up-and-left by one row+column. Use a forward sweep:
            // each (i, j) only reads from (i+1, j+1) which we haven't
            // written yet.
            for i in 0..m - 1 {
                for j in 0..m - 1 {
                    self.sy[i * m + j] = self.sy[(i + 1) * m + (j + 1)];
                    self.ss[i * m + j] = self.ss[(i + 1) * m + (j + 1)];
                }
            }
            // Zero the now-vacated last row and last column.
            for i in 0..m {
                self.sy[i * m + (m - 1)] = F::zero();
                self.sy[(m - 1) * m + i] = F::zero();
                self.ss[i * m + (m - 1)] = F::zero();
                self.ss[(m - 1) * m + i] = F::zero();
            }
        }

        // The compact form uses only the lower triangle of SᵀY, and
        // the two-loop recursion uses only its diagonal. Computing the
        // upper triangle would add an unused dot product per history pair.
        let new_idx = self.ws.len();
        newest_products(
            s.as_float_slice(),
            &self.ws,
            &self.wy,
            &mut self.sy[new_idx * m..new_idx * m + new_idx],
            &mut self.ss[new_idx * m..new_idx * m + new_idx],
        );
        for i in 0..new_idx {
            self.ss[i * m + new_idx] = self.ss[new_idx * m + i];
        }
        self.sy[new_idx * m + new_idx] = sy_dot;
        self.ss[new_idx * m + new_idx] = s.dot(&s);

        self.theta = yy_dot / sy_dot;

        // Push last so the dot products above saw the pre-extension
        // history.
        self.ws.push(s);
        self.wy.push(y);
        true
    }
}

impl<V: Clone, F: Scalar> State for LbfgsState<V, F> {
    type Param = V;
    type Float = F;

    fn iter(&self) -> u64 {
        self.iter
    }
    fn increment_iter(&mut self) {
        self.iter += 1;
    }
    fn cost_evals(&self) -> u64 {
        self.cost_evals
    }
    fn param(&self) -> &V {
        &self.param
    }
    /// # Panics
    ///
    /// Panics if read before [`Solver::init`](crate::core::solver::Solver::init)
    /// has populated the cached cost; see [`BasicState::cost`] for
    /// the full safety argument; same contract.
    ///
    /// [`BasicState::cost`]: crate::core::state::BasicState::cost
    fn cost(&self) -> F {
        self.cost
            .expect("LbfgsState::cost read before Solver::init populated it")
    }

    fn best_param(&self) -> &V {
        self.best_param.as_ref().expect(
            "LbfgsState::best_param read before Solver::init populated it",
        )
    }

    fn best_cost(&self) -> F {
        self.best_cost
    }

    fn best_iter(&self) -> u64 {
        self.best_iter
    }

    fn best_cost_evals(&self) -> u64 {
        self.best_cost_evals
    }

    fn update_best(&mut self) {
        if let Some(curr) = self.cost {
            if self.best_param.is_none() || curr < self.best_cost {
                if let Some(best) = self.best_param.as_mut() {
                    best.clone_from(&self.param);
                } else {
                    self.best_param = Some(self.param.clone());
                }
                self.best_cost = curr;
                self.best_iter = self.iter;
                self.best_cost_evals = self.cost_evals;
                self.best_gradient_evals = self.gradient_evals;
            }
        }
    }

    fn reset_best(&mut self) {
        self.best_param = None;
        self.best_cost = F::infinity();
        self.best_iter = 0;
        self.best_cost_evals = 0;
        self.best_gradient_evals = 0;
    }
}

impl<V: Clone, F: Scalar> GradientState for LbfgsState<V, F> {
    fn gradient(&self) -> Option<&V> {
        self.gradient.as_ref()
    }
    fn gradient_evals(&self) -> u64 {
        self.gradient_evals
    }
    fn best_gradient_evals(&self) -> u64 {
        self.best_gradient_evals
    }
}

impl<V: Clone, F: Scalar> CountsMirror for LbfgsState<V, F> {
    fn mirror(&mut self, delta: &EvalCounts) {
        self.cost_evals = delta.cost_evals + delta.residual_evals;
        self.gradient_evals = delta.gradient_evals
            + delta.jacobian_evals
            + delta.hessian_evals
            + delta.hessian_product_evals;
    }
}

#[cfg(test)]
// Explicit `i * m + j` indexing (including `0 * m + 0`) mirrors the
// Fortran source's 2-D layout for `sy`/`ss`: load-bearing for
// readability when cross-checking against `lbfgsb.f`.
#[allow(clippy::identity_op, clippy::erasing_op)]
mod tests {
    use super::*;

    fn check_history_products<F: Scalar>(scales: &[F]) {
        for &scale in scales {
            for n in [1, 3, 25, 180, 1000] {
                for m in [1, 2, 3, 5, 10] {
                    let mut state = LbfgsState::new(vec![F::zero(); n], m);
                    for step in 0..2 * m + 3 {
                        let s: Vec<_> = (0..n)
                            .map(|i| {
                                scale
                                    * F::from_f64(
                                        ((i * 13 + step * 7) % 37) as f64
                                            - 18.5,
                                    )
                                    .unwrap()
                            })
                            .collect();
                        let y: Vec<_> = s
                            .iter()
                            .enumerate()
                            .map(|(i, &s)| {
                                s * F::from_usize(1 + i % 7).unwrap()
                            })
                            .collect();
                        assert!(state.append_pair(s, y));
                        let col = state.col();
                        assert_eq!(
                            state.theta,
                            state.wy[col - 1].dot(&state.wy[col - 1])
                                / state.ws[col - 1].dot(&state.wy[col - 1])
                        );
                        for i in 0..col {
                            for j in 0..col {
                                assert_eq!(
                                    state.ss[i * m + j],
                                    state.ws[i].dot(&state.ws[j])
                                );
                                if j <= i {
                                    assert_eq!(
                                        state.sy[i * m + j],
                                        state.ws[i].dot(&state.wy[j])
                                    );
                                } else {
                                    assert_eq!(state.sy[i * m + j], F::zero());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn history_products_match_ordered_dots_f64() {
        check_history_products(&[1e-140_f64, 1.0, 1e140]);
    }

    #[test]
    fn history_products_match_ordered_dots_f32() {
        check_history_products(&[1e-15_f32, 1.0, 1e15]);
    }

    #[test]
    fn invalid_pair_preserves_existing_history() {
        let mut state = LbfgsState::new(vec![0.0; 2], 2);
        assert!(state.append_pair(vec![1.0, 2.0], vec![3.0, 4.0]));
        assert!(state.append_pair(vec![2.0, -1.0], vec![4.0, -3.0]));
        let before = (
            state.ws.clone(),
            state.wy.clone(),
            state.sy.clone(),
            state.ss.clone(),
            state.theta,
        );
        for (s, y) in [
            (vec![0.0, -0.0], vec![1.0, 1.0]),
            (vec![1.0, 2.0], vec![-1.0, -2.0]),
            (vec![f64::NAN, 1.0], vec![1.0, 1.0]),
            (vec![1.0, 1.0], vec![f64::INFINITY, 1.0]),
            (vec![1e-200, 0.0], vec![1e200, 0.0]),
        ] {
            assert!(!state.append_pair(s, y));
            assert_eq!(state.ws, before.0);
            assert_eq!(state.wy, before.1);
            assert_eq!(state.sy, before.2);
            assert_eq!(state.ss, before.3);
            assert_eq!(state.theta, before.4);
        }
    }

    #[test]
    fn history_product_work_stays_bounded_through_rollover() {
        use std::{cell::Cell, rc::Rc};

        struct Counted {
            values: Vec<f64>,
            calls: Rc<Cell<usize>>,
        }
        impl Dot for Counted {
            fn dot(&self, other: &Self) -> f64 {
                self.calls.set(self.calls.get() + 1);
                self.values.dot(&other.values)
            }
        }
        impl AsFloatSlice for Counted {
            fn as_float_slice(&self) -> &[f64] {
                &self.values
            }
        }

        for m in [1, 3, 10] {
            let calls = Rc::new(Cell::new(0));
            let vector = |values| Counted {
                values,
                calls: calls.clone(),
            };
            let mut state = LbfgsState::new(vector(vec![0.0; 4]), m);
            for step in 1..=2 * m + 1 {
                let a = step as f64;
                let s = vector(vec![a, -1.0, 0.5, 2.0]);
                let y = vector(vec![2.0 * a, -3.0, 2.0, 1.0]);
                calls.set(0);
                assert!(state.append_pair(s, y));
                let col = state.col();
                assert!(
                    calls.get() <= 3,
                    "unneeded history products: {} for {col} columns",
                    calls.get()
                );
                for i in 0..col {
                    for j in 0..col {
                        let expected_ss =
                            state.ws[i].values.dot(&state.ws[j].values);
                        assert_eq!(state.ss[i * m + j], expected_ss);
                        if j <= i {
                            let expected_sy =
                                state.ws[i].values.dot(&state.wy[j].values);
                            assert_eq!(state.sy[i * m + j], expected_sy);
                        }
                    }
                }
                assert_eq!(
                    state.theta,
                    (4.0 * a * a + 14.0) / (2.0 * a * a + 6.0)
                );
            }
        }
    }

    #[test]
    fn new_state_is_empty() {
        let s = LbfgsState::<Vec<f64>>::new(vec![0.0; 4], 5);
        assert_eq!(s.col(), 0);
        assert_eq!(s.m_capacity, 5);
        assert_eq!(s.theta, 1.0);
        assert!(s.cost.is_none());
        assert!(s.gradient.is_none());
        assert_eq!(s.ws.len(), 0);
        assert_eq!(s.wy.len(), 0);
        assert_eq!(s.sy.len(), 25);
        assert_eq!(s.ss.len(), 25);
    }

    #[test]
    fn first_append_sets_theta_and_diagonal() {
        let mut state = LbfgsState::<Vec<f64>>::new(vec![0.0, 0.0], 3);
        let s = vec![1.0, 2.0]; // ‖s‖² = 5
        let y = vec![3.0, 4.0]; // ‖y‖² = 25, s·y = 1·3 + 2·4 = 11
        let ok = state.append_pair(s, y);
        assert!(ok);
        assert_eq!(state.col(), 1);
        assert_eq!(state.theta, 25.0 / 11.0);
        assert_eq!(state.sy[0], 11.0); // sy[0,0] = s·y
        assert_eq!(state.ss[0], 5.0); // ss[0,0] = s·s
    }

    #[test]
    fn second_append_fills_off_diagonal_gram_blocks() {
        let mut state = LbfgsState::<Vec<f64>>::new(vec![0.0; 2], 3);
        let s1 = vec![1.0, 0.0];
        let y1 = vec![2.0, 0.0];
        let s2 = vec![0.0, 3.0];
        let y2 = vec![0.0, 4.0];
        state.append_pair(s1, y1);
        state.append_pair(s2, y2);

        // m = 3, so indexing is i * 3 + j.
        // The lower triangle stores s_i · y_j for i >= j.
        assert_eq!(state.sy[0 * 3 + 0], 2.0);
        assert_eq!(state.sy[1 * 3 + 0], 0.0);
        assert_eq!(state.sy[1 * 3 + 1], 12.0);

        // ss is symmetric: ss[0,0]=1, ss[0,1]=ss[1,0]=0, ss[1,1]=9.
        assert_eq!(state.ss[0 * 3 + 0], 1.0);
        assert_eq!(state.ss[0 * 3 + 1], 0.0);
        assert_eq!(state.ss[1 * 3 + 0], 0.0);
        assert_eq!(state.ss[1 * 3 + 1], 9.0);

        // theta from the most recent update: ‖y2‖²/(s2·y2) = 16/12.
        assert_eq!(state.theta, 16.0 / 12.0);
    }

    #[test]
    fn appending_beyond_capacity_drops_oldest() {
        let mut state = LbfgsState::<Vec<f64>>::new(vec![0.0], 2);
        // Three appends with distinct identifiable pairs; m_capacity=2
        // means the third should evict the first.
        let s1 = vec![1.0];
        let y1 = vec![2.0];
        let s2 = vec![3.0];
        let y2 = vec![4.0];
        let s3 = vec![5.0];
        let y3 = vec![6.0];
        state.append_pair(s1.clone(), y1.clone());
        state.append_pair(s2.clone(), y2.clone());
        state.append_pair(s3.clone(), y3.clone());

        assert_eq!(state.col(), 2);
        // After eviction, history is [s2, s3]/[y2, y3].
        assert_eq!(state.ws[0], s2);
        assert_eq!(state.ws[1], s3);
        assert_eq!(state.wy[0], y2);
        assert_eq!(state.wy[1], y3);

        // Gram blocks should reflect the post-eviction history.
        // sy[0,0] = s2·y2 = 12,
        // sy[1,0] = s3·y2 = 20, sy[1,1] = s3·y3 = 30.
        let m = 2;
        assert_eq!(state.sy[0 * m + 0], 12.0);
        assert_eq!(state.sy[1 * m + 0], 20.0);
        assert_eq!(state.sy[1 * m + 1], 30.0);
    }

    #[test]
    fn curvature_failure_leaves_state_untouched() {
        let mut state = LbfgsState::<Vec<f64>>::new(vec![0.0, 0.0], 3);
        // s · y = -1 (negative curvature): must be rejected.
        let s = vec![1.0, 0.0];
        let y = vec![-1.0, 0.0];
        let ok = state.append_pair(s, y);
        assert!(!ok);
        assert_eq!(state.col(), 0);
        assert_eq!(state.theta, 1.0);
    }

    #[test]
    fn state_implements_state_and_gradient_state_traits() {
        // Sanity check that the trait impls are reachable through the
        // generic State/GradientState bounds.
        let s: LbfgsState<Vec<f64>> = LbfgsState::new(vec![1.0, 2.0], 5);
        // Param round-trip via the State trait.
        let p: &Vec<f64> = State::param(&s);
        assert_eq!(p, &vec![1.0, 2.0]);
        // GradientState exposes the None gradient pre-init.
        assert!(GradientState::gradient(&s).is_none());
        assert_eq!(GradientState::gradient_evals(&s), 0);
        assert_eq!(State::iter(&s), 0);
        assert_eq!(State::cost_evals(&s), 0);
    }
}
