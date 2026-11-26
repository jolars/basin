use crate::SerialEvaluation;

/// Schedule indexed probes while keeping scratch and matrix assembly local.
/// Passing borrowed inputs as data allows the serial implementation to accept
/// thread-local problems without imposing thread-safety bounds on the kernels.
pub trait ProbeEvaluation<C, T, E> {
    fn has_multiple_workers() -> bool;

    fn try_map<S>(
        n: usize,
        inputs: &C,
        init: impl Fn(&C) -> S + Sync,
        evaluate: impl Fn(&C, &mut S, usize) -> Result<T, E> + Sync,
    ) -> Result<Vec<T>, E>;

    fn try_fill<S>(
        values: &mut [T],
        inputs: &C,
        init: impl Fn(&C) -> S + Sync,
        evaluate: impl Fn(&C, &mut S, usize) -> Result<T, E> + Sync,
    ) -> Result<(), E>;
}

impl<C, T, E> ProbeEvaluation<C, T, E> for SerialEvaluation {
    fn has_multiple_workers() -> bool {
        false
    }

    fn try_map<S>(
        n: usize,
        inputs: &C,
        init: impl Fn(&C) -> S,
        evaluate: impl Fn(&C, &mut S, usize) -> Result<T, E>,
    ) -> Result<Vec<T>, E> {
        let mut scratch = init(inputs);
        (0..n).map(|j| evaluate(inputs, &mut scratch, j)).collect()
    }

    fn try_fill<S>(
        values: &mut [T],
        inputs: &C,
        init: impl Fn(&C) -> S,
        evaluate: impl Fn(&C, &mut S, usize) -> Result<T, E>,
    ) -> Result<(), E> {
        let mut scratch = init(inputs);
        for (j, value) in values.iter_mut().enumerate() {
            *value = evaluate(inputs, &mut scratch, j)?;
        }
        Ok(())
    }
}

#[cfg(feature = "parallel")]
impl<C: Sync, T: Send, E: Send> ProbeEvaluation<C, T, E>
    for crate::ParallelEvaluation
{
    fn has_multiple_workers() -> bool {
        rayon::current_num_threads() > 1
    }

    fn try_map<S>(
        n: usize,
        inputs: &C,
        init: impl Fn(&C) -> S + Sync,
        evaluate: impl Fn(&C, &mut S, usize) -> Result<T, E> + Sync,
    ) -> Result<Vec<T>, E> {
        crate::core::parallel::try_map_range_with(
            n,
            || init(inputs),
            |scratch, j| evaluate(inputs, scratch, j),
        )
    }

    fn try_fill<S>(
        values: &mut [T],
        inputs: &C,
        init: impl Fn(&C) -> S + Sync,
        evaluate: impl Fn(&C, &mut S, usize) -> Result<T, E> + Sync,
    ) -> Result<(), E> {
        use rayon::prelude::*;
        values.par_iter_mut().enumerate().try_for_each_init(
            || init(inputs),
            |scratch, (j, value)| {
                *value = evaluate(inputs, scratch, j)?;
                Ok(())
            },
        )
    }
}
