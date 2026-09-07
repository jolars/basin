use crate::core::state::gbnm::GbnmRestart;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OptimumAction {
    None,
    Save,
}

/// Apply tests T3–T8 in their published priority order.
pub(super) fn select_restart(
    current: GbnmRestart,
    known_optimum: bool,
    flat: bool,
    small: bool,
    degenerate: bool,
    returned_to_center: bool,
    on_bounds: bool,
) -> (GbnmRestart, OptimumAction) {
    if known_optimum || flat {
        (GbnmRestart::Probabilistic, OptimumAction::None)
    } else if ((current == GbnmRestart::SmallTest
        || current == GbnmRestart::LargeTest)
        && returned_to_center)
        || (current != GbnmRestart::SmallTest && !on_bounds && small)
    {
        (GbnmRestart::Probabilistic, OptimumAction::Save)
    } else if (current == GbnmRestart::LargeTest
        || current == GbnmRestart::Probabilistic)
        && !returned_to_center
        && on_bounds
    {
        (GbnmRestart::SmallTest, OptimumAction::None)
    } else if current == GbnmRestart::SmallTest
        && !returned_to_center
        && !on_bounds
    {
        (GbnmRestart::LargeTest, OptimumAction::Save)
    } else if current != GbnmRestart::SmallTest
        && !on_bounds
        && !small
        && degenerate
    {
        (GbnmRestart::LargeTest, OptimumAction::None)
    } else {
        (GbnmRestart::SmallTest, OptimumAction::None)
    }
}

#[cfg(test)]
mod tests {
    use super::{OptimumAction, select_restart};
    use crate::core::state::gbnm::GbnmRestart;

    #[test]
    fn selector_covers_the_paper_flowchart() {
        use GbnmRestart::{LargeTest, Probabilistic, SmallTest};
        use OptimumAction::{None, Save};

        assert_eq!(
            select_restart(
                Probabilistic,
                true,
                false,
                true,
                false,
                false,
                false,
            ),
            (Probabilistic, None)
        );
        assert_eq!(
            select_restart(
                Probabilistic,
                false,
                true,
                false,
                false,
                false,
                false,
            ),
            (Probabilistic, None)
        );
        assert_eq!(
            select_restart(SmallTest, false, false, true, false, true, true,),
            (Probabilistic, Save)
        );
        assert_eq!(
            select_restart(
                Probabilistic,
                false,
                false,
                true,
                false,
                false,
                false,
            ),
            (Probabilistic, Save)
        );
        assert_eq!(
            select_restart(LargeTest, false, false, false, false, false, true,),
            (SmallTest, None)
        );
        assert_eq!(
            select_restart(SmallTest, false, false, true, false, false, false,),
            (LargeTest, Save)
        );
        assert_eq!(
            select_restart(
                Probabilistic,
                false,
                false,
                false,
                true,
                false,
                false,
            ),
            (LargeTest, None)
        );
        assert_eq!(
            select_restart(
                Probabilistic,
                false,
                false,
                false,
                false,
                false,
                false,
            ),
            (SmallTest, None)
        );
    }
}
