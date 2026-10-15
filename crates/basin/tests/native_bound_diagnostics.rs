//! Adversarial native identities and callback interruptions in both precisions.
use basin::line_search::more_thuente::diagnostics::SearchExit;
use basin::solver::nelder_mead::diagnostics::TrialKind;
use basin::{
    CostFunction, Gradient, LineSearch, LineSearchBounds, MoreThuente,
    NelderMead, Problem, Scalar, SimplexProgress, Solver,
};
use std::cell::Cell;

struct Script<F> {
    costs: Vec<F>,
    call: Cell<usize>,
    fail: Option<usize>,
}

impl<F: Scalar> basin::ProblemConstraints for Script<F> {
    type Constraints = basin::NoConstraints;
    fn constraint_set(
        &self,
    ) -> <Self::Constraints as basin::ConstraintDescription>::Borrowed<'_> {
        basin::Constraints::new()
    }
}
impl<F: Scalar> CostFunction for Script<F> {
    type Param = Vec<F>;
    type Output = F;
    type Error = &'static str;
    fn cost(&self, _: &Vec<F>) -> Result<F, Self::Error> {
        let i = self.call.get();
        self.call.set(i + 1);
        if self.fail == Some(i) {
            Err("interrupted")
        } else {
            Ok(self.costs[i])
        }
    }
}
fn simplex<F: Scalar>() {
    let f = |v| F::from_f64(v).unwrap();
    let cases = vec![
        (vec![0.5], vec![TrialKind::Reflection], vec![true]),
        (
            vec![-1., -2.],
            vec![TrialKind::Reflection, TrialKind::Expansion],
            vec![false, true],
        ),
        (
            vec![-1., 0.],
            vec![TrialKind::Reflection, TrialKind::Expansion],
            vec![true, false],
        ),
        (
            vec![1.5, 1.],
            vec![TrialKind::Reflection, TrialKind::OutsideContraction],
            vec![false, true],
        ),
        (
            vec![3., 1.],
            vec![TrialKind::Reflection, TrialKind::InsideContraction],
            vec![false, true],
        ),
        (
            vec![1.5, 2., 0.25, 0.5],
            vec![
                TrialKind::Reflection,
                TrialKind::OutsideContraction,
                TrialKind::Shrink(1),
                TrialKind::Shrink(2),
            ],
            vec![false, false, true, true],
        ),
        (
            vec![3., 3., 0.25, 0.5],
            vec![
                TrialKind::Reflection,
                TrialKind::InsideContraction,
                TrialKind::Shrink(1),
                TrialKind::Shrink(2),
            ],
            vec![false, false, true, true],
        ),
    ];
    let mut cases = cases;
    for nonfinite in [f64::NAN, f64::INFINITY] {
        cases.push((
            vec![nonfinite, nonfinite, 0.25, 0.5],
            vec![
                TrialKind::Reflection,
                TrialKind::InsideContraction,
                TrialKind::Shrink(1),
                TrialKind::Shrink(2),
            ],
            vec![false, false, true, true],
        ));
    }
    for (costs, kinds, accepted) in cases {
        for fail in (3..3 + costs.len()).map(Some).chain([None]) {
            let mut p = Problem::new(Script {
                costs: [
                    vec![f(0.), f(1.), f(2.)],
                    costs.iter().map(|v| f(*v)).collect(),
                ]
                .concat(),
                call: Cell::new(0),
                fail,
            });
            let mut solver = NelderMead::new().with_trial_diagnostics(true);
            let seed = || {
                SimplexProgress::from_simplex(vec![
                    vec![f(0.), f(0.)],
                    vec![f(1.), f(0.)],
                    vec![f(0.), f(1.)],
                ])
            };
            let mut state = seed();
            solver.init(&mut p, &mut state).unwrap();
            let result = solver.next_iter(&mut p, &mut state);
            assert_eq!(result.is_err(), fail.is_some());
            let trials = solver.trial_observations();
            let length = fail.map_or(costs.len(), |i| i - 2);
            assert_eq!(trials.len(), length);
            for (i, t) in trials.iter().enumerate() {
                assert_eq!(t.sequence, i as u64 + 1);
                assert_eq!(t.kind, kinds[i]);
                assert_eq!(t.counts_before.cost_evals, i as u64 + 3);
                assert_eq!(t.counts_after.cost_evals, i as u64 + 4);
                if Some(i + 3) == fail {
                    assert_eq!(t.cost, None);
                    assert_eq!(t.accepted, None);
                } else if costs[i].is_nan() {
                    assert!(t.cost.unwrap().is_nan());
                } else {
                    assert_eq!(t.cost, Some(f(costs[i])));
                }
                let point = match t.kind {
                    TrialKind::Reflection => [1., -1.],
                    TrialKind::Expansion => [1.5, -2.],
                    TrialKind::OutsideContraction => [0.75, -0.5],
                    TrialKind::InsideContraction => [0.25, 0.5],
                    TrialKind::Shrink(1) => [0.5, 0.],
                    TrialKind::Shrink(2) => [0., 0.5],
                    _ => unreachable!(),
                };
                assert_eq!(t.point, point.map(f));
                if fail.is_none() {
                    assert_eq!(t.accepted, Some(accepted[i]));
                }
            }
            let prior = trials.last().unwrap().sequence;
            // Fresh initialization resets records and ordinals without changing settings.
            p.inner().call.set(0);
            let mut state = seed();
            solver.init(&mut p, &mut state).unwrap();
            assert!(solver.trial_observations().is_empty());
            let _ = solver.next_iter(&mut p, &mut state);
            assert_eq!(solver.trial_observations()[0].sequence, 1);
            assert!(prior >= 1);
        }
    }
}
struct Quadratic<F> {
    flat: bool,
    fail: bool,
    scalar: std::marker::PhantomData<F>,
}

impl<F: Scalar> basin::ProblemConstraints for Quadratic<F> {
    type Constraints = basin::NoConstraints;
    fn constraint_set(
        &self,
    ) -> <Self::Constraints as basin::ConstraintDescription>::Borrowed<'_> {
        basin::Constraints::new()
    }
}
impl<F: Scalar> CostFunction for Quadratic<F> {
    type Param = Vec<F>;
    type Output = F;
    type Error = &'static str;
    fn cost(&self, x: &Vec<F>) -> Result<F, Self::Error> {
        if self.fail {
            Err("interrupted")
        } else {
            Ok(if self.flat {
                F::zero()
            } else {
                x[0] * x[0] / F::from_f64(2.).unwrap()
            })
        }
    }
}
impl<F: Scalar> Gradient for Quadratic<F> {
    type Gradient = Vec<F>;
    fn gradient(&self, x: &Vec<F>) -> Result<Vec<F>, Self::Error> {
        Ok(vec![if self.flat { F::one() } else { x[0] }])
    }
}
fn search<F: Scalar>() {
    let f = |v| F::from_f64(v).unwrap();
    for (flat, fail, max, alpha, exit) in [
        (false, false, 20, 1., SearchExit::Wolfe),
        (true, false, 20, 1., SearchExit::Warning),
        (false, false, 0, 1., SearchExit::Exhausted),
        (false, false, 1, 10., SearchExit::Exhausted),
        (false, true, 20, 1., SearchExit::CallbackError),
    ] {
        let mut p = Problem::new(Quadratic {
            flat,
            fail,
            scalar: std::marker::PhantomData,
        });
        let mut s = MoreThuente::new().maxfev(max).with_trial_diagnostics(true);
        let cost = if flat { f(0.) } else { f(0.5) };
        let r = s.next_with_bounds(
            &mut p,
            &vec![f(1.)],
            cost,
            &vec![f(1.)],
            &vec![f(-1.)],
            LineSearchBounds::new(f(alpha), f(10.)),
        );
        assert_eq!(s.search_exit(), Some(exit));
        assert_eq!(r.is_err(), fail);
        assert_eq!(p.counts().cost_evals, s.trial_observations().len() as u64);
        assert_eq!(
            p.counts().gradient_evals,
            s.trial_observations().len() as u64
        );
        if exit == SearchExit::Wolfe {
            assert_eq!(
                s.trial_observations().last().unwrap().wolfe,
                Some([true, true])
            );
        }
        if exit == SearchExit::Warning {
            assert_ne!(
                s.trial_observations().last().unwrap().wolfe,
                Some([true, true])
            );
        }
        if alpha == 10. {
            assert!(s.trial_observations()[0].update.is_some());
        }
        if fail {
            assert_eq!(s.trial_observations()[0].value, None);
            assert_eq!(s.selected_step(), None);
        }
    }
}
#[test]
fn simplex_f32() {
    simplex::<f32>();
}
#[test]
fn simplex_f64() {
    simplex::<f64>();
}
#[test]
fn search_f32() {
    search::<f32>();
}
#[test]
fn search_f64() {
    search::<f64>();
}

#[test]
fn exact_steps_keep_ordinals_and_fresh_init_resets() {
    let mut p = Problem::new(Quadratic::<f64> {
        flat: false,
        fail: false,
        scalar: std::marker::PhantomData,
    });
    let seed = || SimplexProgress::from_simplex(vec![vec![1.], vec![2.]]);
    let mut solver = NelderMead::new().with_trial_diagnostics(true);
    let mut state = seed();
    solver.init(&mut p, &mut state).unwrap();
    solver.next_iter(&mut p, &mut state).unwrap();
    let last = solver.trial_observations().last().unwrap().sequence;
    let _ = solver.next_iter(&mut p, &mut state).unwrap();
    assert_eq!(solver.trial_observations()[0].sequence, last + 1);
    solver.init(&mut p, &mut seed()).unwrap();
    assert!(solver.trial_observations().is_empty());
}

#[cfg(feature = "serde")]
#[test]
fn serde_diagnostics_are_explicitly_ephemeral() {
    let mut p = Problem::new(Quadratic::<f32> {
        flat: false,
        fail: false,
        scalar: std::marker::PhantomData,
    });
    let mut nm = NelderMead::new().with_trial_diagnostics(true);
    let mut state = SimplexProgress::from_simplex(vec![vec![1.], vec![2.]]);
    nm.init(&mut p, &mut state).unwrap();
    nm.next_iter(&mut p, &mut state).unwrap();
    assert!(!nm.trial_observations().is_empty());
    let encoded = postcard::to_allocvec(&(nm, state)).unwrap();
    let (mut nm, mut state): (
        NelderMead<Vec<f32>, f32>,
        SimplexProgress<Vec<f32>, f32>,
    ) = postcard::from_bytes(&encoded).unwrap();
    assert!(nm.trial_observations().is_empty());
    let _ = nm.next_iter(&mut p, &mut state).unwrap();
    assert!(nm.trial_observations().is_empty());
    let search = MoreThuente::<f32>::new().with_trial_diagnostics(true);
    let encoded = postcard::to_allocvec(&search).unwrap();
    let mut search: basin::line_search::more_thuente::diagnostics::TracedMoreThuente<f32> = postcard::from_bytes(&encoded).unwrap();
    let _ = search
        .next_with_evaluation(&mut p, &vec![1.], 0.5, &vec![1.], &vec![-1.])
        .unwrap();
    assert!(search.trial_observations().is_empty());
    assert_eq!(search.search_exit(), None);
}

#[test]
fn original_more_thuente_struct_literals_remain_supported() {
    let search = MoreThuente {
        ftol: 1e-3_f32,
        gtol: 0.9,
        xtol: 0.1,
        alpha_init: 1.,
        stpmin: 0.,
        stpmax: 1e10,
        maxfev: 20,
    };
    assert_eq!(search.maxfev, 20);
    let updated = MoreThuente {
        maxfev: 3,
        ..search
    };
    assert_eq!(updated.maxfev, 3);
}

#[cfg(feature = "serde")]
#[test]
fn original_more_thuente_serde_keeps_seven_fields() {
    let search = MoreThuente::<f32>::new();
    let fields = (
        search.ftol,
        search.gtol,
        search.xtol,
        search.alpha_init,
        search.stpmin,
        search.stpmax,
        search.maxfev,
    );
    assert_eq!(
        postcard::to_allocvec(&search).unwrap(),
        postcard::to_allocvec(&fields).unwrap()
    );
}
