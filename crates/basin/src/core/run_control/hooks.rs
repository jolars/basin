use crate::{ApplicationStop, CheckpointSink, ObserveSolver, State};

/// Hook storage that permits ownership transfer when the rest of the run is `Send`.
pub struct SendHooks;

/// Hook storage that accepts thread-local captures such as `Rc`.
pub struct LocalHooks;

/// Selects the callback storage used by an executor and its run controls.
///
/// Use this bound on the hook mode parameter of wrappers that support
/// both [`SendHooks`] and [`LocalHooks`]. To require thread transfer, bound
/// the complete wrapped type by [`Send`].
pub trait HookMode: sealed::Sealed {
    #[doc(hidden)]
    type Stop<S: State>: StopHook<S> + ?Sized;
    #[doc(hidden)]
    type Observer<S: State, So>: ObserveSolver<S, So> + ?Sized;
    #[doc(hidden)]
    type Sink<S: State, So>: CheckpointSink<So, S> + ?Sized;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::SendHooks {}
    impl Sealed for super::LocalHooks {}
}

#[doc(hidden)]
pub trait StoreStop<S: State, T>: HookMode {
    fn store_stop(hook: T) -> Box<Self::Stop<S>>;
}

#[doc(hidden)]
pub trait StoreObserver<S: State, So, T>: HookMode {
    fn store_observer(hook: T) -> Box<Self::Observer<S, So>>;
}

#[doc(hidden)]
pub trait StoreSink<S: State, So, T>: HookMode {
    fn store_sink(hook: T) -> Box<Self::Sink<S, So>>;
}

macro_rules! storage {
    ($mode:ty $(, $send:ident)?) => {
        impl HookMode for $mode {
            type Stop<S: State> = dyn StopHook<S> $(+ $send)?;
            type Observer<S: State, So> = dyn ObserveSolver<S, So> $(+ $send)?;
            type Sink<S: State, So> = dyn CheckpointSink<So, S> $(+ $send)?;
        }

        impl<S: State, T: StopHook<S> + 'static $(+ $send)?> StoreStop<S, T> for $mode {
            fn store_stop(hook: T) -> Box<Self::Stop<S>> {
                Box::new(hook)
            }
        }

        impl<S: State, So, T: ObserveSolver<S, So> + 'static $(+ $send)?>
            StoreObserver<S, So, T> for $mode
        {
            fn store_observer(hook: T) -> Box<Self::Observer<S, So>> {
                Box::new(hook)
            }
        }

        impl<S: State, So, T: CheckpointSink<So, S> + 'static $(+ $send)?>
            StoreSink<S, So, T> for $mode
        {
            fn store_sink(hook: T) -> Box<Self::Sink<S, So>> {
                Box::new(hook)
            }
        }
    };
}

storage!(SendHooks, Send);
storage!(LocalHooks);

#[doc(hidden)]
pub trait StopHook<S> {
    fn reset(&mut self) {}
    fn check(&mut self, state: &S) -> Option<ApplicationStop>;
}

#[doc(hidden)]
pub struct StopCallback<C>(pub(crate) C);

impl<S, C: FnMut(&S) -> Option<ApplicationStop>> StopHook<S>
    for StopCallback<C>
{
    fn check(&mut self, state: &S) -> Option<ApplicationStop> {
        (self.0)(state)
    }
}

#[doc(hidden)]
pub struct StopFactory<M, C> {
    make: M,
    current: Option<C>,
}

impl<M, C> StopFactory<M, C> {
    pub(crate) fn new(make: M) -> Self {
        Self {
            make,
            current: None,
        }
    }
}

impl<S, M, C> StopHook<S> for StopFactory<M, C>
where
    M: FnMut() -> C,
    C: FnMut(&S) -> Option<ApplicationStop>,
{
    fn reset(&mut self) {
        self.current = Some((self.make)());
    }

    fn check(&mut self, state: &S) -> Option<ApplicationStop> {
        self.current.get_or_insert_with(&mut self.make)(state)
    }
}
