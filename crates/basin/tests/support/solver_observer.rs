#![allow(dead_code)]
use basin::{
    CostFunction, Executor, PointState, Problem, Solver, State,
    TerminationReason,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scenario {
    Normal,
    InitError,
    StepError,
    MidStop,
    CheckStop,
    Flat,
}

pub struct Linear;
impl CostFunction for Linear {
    type Param = f64;
    type Output = f64;
    type Error = &'static str;
    fn cost(&self, x: &f64) -> Result<f64, Self::Error> {
        Ok(*x)
    }
}

// Keeping the solver non-Clone checks that registration only needs a borrow.
pub struct Probe {
    pub diagnostic: f64,
    pub workspace: Vec<f64>,
    pub scenario: Scenario,
}
impl Probe {
    pub fn new(scenario: Scenario) -> Self {
        Self {
            diagnostic: -1.0,
            workspace: vec![0.0; 16],
            scenario,
        }
    }
}
impl Solver<Linear, PointState<f64>> for Probe {
    type Error = &'static str;
    fn init(
        &mut self,
        p: &mut Problem<Linear>,
        mut s: PointState<f64>,
    ) -> Result<PointState<f64>, Self::Error> {
        self.diagnostic = 0.0;
        s.reset();
        s.replace(*s.param(), p.cost(s.param())?);
        if self.scenario == Scenario::InitError {
            return Err("init");
        }
        Ok(s)
    }
    fn next_iter(
        &mut self,
        p: &mut Problem<Linear>,
        mut s: PointState<f64>,
    ) -> Result<(PointState<f64>, Option<TerminationReason>), Self::Error> {
        let x = *s.param()
            - if self.scenario == Scenario::Flat {
                0.0
            } else {
                1.0
            };
        let cost = p.cost(&x)?;
        self.diagnostic += 1.0;
        self.workspace[0] = self.diagnostic;
        if self.scenario == Scenario::StepError {
            return Err("step");
        }
        s.replace(x, cost);
        Ok((
            s,
            (self.scenario == Scenario::MidStop)
                .then_some(TerminationReason::SolverFailed),
        ))
    }
    fn check_convergence(
        &mut self,
        _: &Problem<Linear>,
        s: &PointState<f64>,
    ) -> Option<TerminationReason> {
        if self.scenario == Scenario::CheckStop && s.iter() == 2 {
            self.diagnostic = 99.0;
            Some(TerminationReason::SolverConverged)
        } else {
            None
        }
    }
}

pub type Run = Executor<Linear, PointState<f64>, Probe>;
pub fn executor(scenario: Scenario) -> Run {
    Executor::new(Linear, Probe::new(scenario), PointState::new(10.0))
        .max_iter(4)
}
