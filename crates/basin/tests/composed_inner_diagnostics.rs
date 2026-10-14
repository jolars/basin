//! Inner-report tracing must leave composed numerical trajectories unchanged.

macro_rules! precision {
    ($module:ident, $f:ty) => {
        mod $module {
            use basin::{
                BasinHopping, BoundedCmaEs, BoundedCmaInject, BoxConstraints,
                CmaEs, CmaInject, CostFunction, De, DeInject, DenseMatrix,
                Executor, MaLsCh, MaLsChCma, MaLsChSw, NelderMead, PointState,
                PopulationProgress, Problem, Solver, State, Termination,
            };
            use std::sync::{Arc, Mutex};

            struct Objective {
                calls: Arc<Mutex<Vec<Vec<$f>>>>,
                lower: Vec<$f>,
                upper: Vec<$f>,
                cap: usize,
            }
            impl CostFunction for Objective {
                type Param = Vec<$f>;
                type Output = $f;
                type Error = &'static str;
                fn cost(&self, x: &Vec<$f>) -> Result<$f, Self::Error> {
                    let mut calls = self.calls.lock().unwrap();
                    if calls.len() >= self.cap {
                        return Err("physical cap");
                    }
                    calls.push(x.clone());
                    Ok(x.iter().map(|v| (v - 1.0) * (v - 1.0)).sum())
                }
            }
            impl BoxConstraints for Objective {
                fn lower(&self) -> &Vec<$f> { &self.lower }
                fn upper(&self) -> &Vec<$f> { &self.upper }
            }
            fn objective(cap: usize) -> Objective {
                Objective { calls: Arc::default(), lower: vec![-4.0; 2], upper: vec![4.0; 2], cap }
            }

            fn callback_points(calls: &Arc<Mutex<Vec<Vec<$f>>>>) -> Vec<Vec<$f>> {
                let mut points = calls.lock().unwrap().clone();
                // Batch callbacks may arrive in either order under parallel execution.
                points.sort_by(|left, right| left.iter().zip(right).map(|(a,b)| a.total_cmp(b)).find(|order| !order.is_eq()).unwrap_or(std::cmp::Ordering::Equal));
                points
            }

            #[test]
            fn tracing_preserves_all_shipped_compositions() {
                macro_rules! compare {
                    ($solver:expr, $state:expr) => {{
                        let mut outputs = Vec::new();
                        for enabled in [false, true] {
                            let p = objective(10000);
                            let calls = p.calls.clone();
                            let solver = ($solver).with_inner_diagnostics(enabled);
                            let result = Executor::new(p, solver, $state).max_iter(3).run_with_solver().unwrap();
                            let records = result.solver.inner_diagnostics();
                            assert_eq!(records.is_empty(), !enabled);
                            if enabled {
                                for (i, record) in records.iter().enumerate() {
                                    assert_eq!(record.sequence, i as u64);
                                    assert!(record.counts_after.unwrap().cost_evals >= record.counts_before.cost_evals);
                                    assert_eq!(record.status, basin::core::inner::diagnostics::Status::Returned);
                                    assert!(record.report.is_some());
                                }
                            }
                            outputs.push((result.state.param().clone(), result.state.cost(), result.report,
                                          result.counts, callback_points(&calls)));
                        }
                        assert_eq!(outputs[0], outputs[1]);
                    }};
                }
                compare!(BasinHopping::new(NelderMead::<Vec<$f>, $f>::new(), 17).with_inner_max_iter(2), PointState::new(vec![3.0, -2.0]));
                compare!(CmaInject::with_inner_solver(CmaEs::<Vec<$f>, DenseMatrix<$f>, $f>::new(17, 1.0), NelderMead::<Vec<$f>, $f>::new()).with_inner_max_iter(2), PopulationProgress::from_point(vec![3.0, -2.0]));
                compare!(BoundedCmaInject::with_inner_solver(BoundedCmaEs::<Vec<$f>, DenseMatrix<$f>, $f>::new(17, 1.0), NelderMead::<Vec<$f>, $f>::new()).with_inner_max_iter(2), PopulationProgress::from_point(vec![3.0, -2.0]));
                compare!(DeInject::with_inner_solver(De::<$f>::new(17).with_pop_size(8), NelderMead::<Vec<$f>, $f>::new()).with_inner_max_iter(2), PopulationProgress::empty());
                compare!(MaLsCh::with_inner(17, basin::SolisWets::<Vec<$f>, $f>::new(0)).with_pop_size(8).with_ls_intensity(5), PopulationProgress::empty());
                compare!(MaLsChCma::<Vec<$f>, DenseMatrix<$f>, $f>::new(17).with_pop_size(8).with_ls_intensity(5), PopulationProgress::empty());
                compare!(MaLsChSw::<Vec<$f>, $f>::new(17).with_pop_size(8).with_ls_intensity(5), PopulationProgress::empty());
            }

            #[test]
            fn hard_initial_error_retains_counts_without_a_report() {
                let p = objective(1);
                let calls = p.calls.clone();
                let mut problem = Problem::new(p);
                let mut solver = BasinHopping::new(NelderMead::<Vec<$f>, $f>::new(), 17).with_inner_diagnostics(true);
                assert!(solver.init(&mut problem, PointState::new(vec![3.0, -2.0])).is_err());
                let records = solver.inner_diagnostics();
                assert_eq!(records.len(), 1);
                assert_eq!(records[0].status, basin::core::inner::diagnostics::Status::CallbackError);
                assert!(records[0].report.is_none());
                assert_eq!(records[0].counts_before.cost_evals, 0);
                assert_eq!(records[0].counts_after.unwrap(), *problem.counts());
                assert_eq!(calls.lock().unwrap().len(), 1);
            }

            #[test]
            fn continuation_keeps_reports_and_fresh_initialization_clears_them() {
                let solver = BasinHopping::new(NelderMead::<Vec<$f>, $f>::new(), 17)
                    .with_inner_max_iter(0).with_inner_diagnostics(true);
                let result = Executor::new(objective(100), solver, PointState::new(vec![3.0, -2.0]))
                    .max_iter(1).run_with_solver().unwrap();
                assert_eq!(result.solver.inner_diagnostics().len(), 2);
                assert!(result.solver.inner_diagnostics().iter().all(|r| matches!(r.report.as_ref().unwrap().termination, Termination::Limit(_))));
                let continued = Executor::resume_from_checkpoint(objective(100), result.into_checkpoint())
                    .max_iter(2).run_with_solver().unwrap();
                assert_eq!(continued.solver.inner_diagnostics().len(), 3);
                let start = continued.state.param().clone();
                let fresh = Executor::new(objective(100), continued.solver, PointState::new(start))
                    .max_iter(0).run_with_solver().unwrap();
                assert_eq!(fresh.solver.inner_diagnostics().len(), 1);
            }


        }
    };
}
precision!(single, f32);
precision!(double, f64);
