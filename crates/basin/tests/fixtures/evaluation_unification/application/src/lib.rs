//! A configured factory can cross threads before it creates the executor.
//!
//! ```compile_fail,E0277
//! use basin::{Executor, FirstOrderState};
//! fn assert_send<T: Send>() {}
//! assert_send::<Executor<(), FirstOrderState<Vec<f64>>, ()>>();
//! ```
