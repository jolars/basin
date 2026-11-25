#![cfg(feature = "serde-codecs")]

use std::{convert::Infallible, error::Error, io::Cursor};

use basin::{
    Bfgs, CmaEs, Constant, CostFunction, ExactCheckpoint, Executor,
    FirstOrderState, Gradient, PopulationProgress, Stepper,
};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

#[derive(Clone, Copy, Debug)]
pub enum SerdeCodec {
    Postcard,
    MessagePackTuple,
    MessagePackNamed,
    BincodeNext,
}

impl SerdeCodec {
    pub const ALL: [Self; 4] = [
        Self::Postcard,
        Self::MessagePackTuple,
        Self::MessagePackNamed,
        Self::BincodeNext,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Postcard => "postcard",
            Self::MessagePackTuple => "msgpack-tuple",
            Self::MessagePackNamed => "msgpack-named",
            Self::BincodeNext => "bincode-next-serde",
        }
    }

    pub fn encode<T: Serialize>(
        self,
        value: &T,
    ) -> Result<Vec<u8>, Box<dyn Error>> {
        Ok(match self {
            Self::Postcard => postcard::to_allocvec(value)?,
            Self::MessagePackTuple => rmp_serde::to_vec(value)?,
            Self::MessagePackNamed => rmp_serde::to_vec_named(value)?,
            Self::BincodeNext => bincode_next::serde::encode_to_vec(
                value,
                bincode_next::config::standard(),
            )?,
        })
    }

    pub fn decode<T: DeserializeOwned>(
        self,
        bytes: &[u8],
    ) -> Result<T, Box<dyn Error>> {
        let (value, consumed) = match self {
            Self::Postcard => {
                let (value, remaining) = postcard::take_from_bytes(bytes)?;
                (value, bytes.len() - remaining.len())
            }
            Self::MessagePackTuple | Self::MessagePackNamed => {
                let mut decoder =
                    rmp_serde::Deserializer::new(Cursor::new(bytes));
                let value = Deserialize::deserialize(&mut decoder)?;
                (value, decoder.position() as usize)
            }
            Self::BincodeNext => bincode_next::serde::decode_from_slice(
                bytes,
                bincode_next::config::standard(),
            )?,
        };
        if consumed != bytes.len() {
            return Err("trailing checkpoint bytes".into());
        }
        Ok(value)
    }
}

pub struct Quadratic;

fn weight(index: usize) -> f64 {
    1.0 + (index % 7) as f64
}

impl CostFunction for Quadratic {
    type Param = Array1<f64>;
    type Output = f64;
    type Error = Infallible;

    fn cost(&self, x: &Self::Param) -> Result<f64, Infallible> {
        Ok(x.iter()
            .enumerate()
            .map(|(i, &x)| 0.5 * weight(i) * x * x)
            .sum())
    }
}

impl Gradient for Quadratic {
    type Gradient = Array1<f64>;

    fn gradient(&self, x: &Self::Param) -> Result<Self::Gradient, Infallible> {
        Ok(Array1::from_iter(
            x.iter().enumerate().map(|(i, &x)| weight(i) * x),
        ))
    }
}

type BfgsSolver = Bfgs<Array1<f64>, f64, Array2<f64>, Constant>;
type BfgsState = FirstOrderState<Array1<f64>>;
pub type BfgsCheckpoint = ExactCheckpoint<BfgsSolver, BfgsState>;

type CmaSolver = CmaEs<Array1<f64>, Array2<f64>>;
type CmaState = PopulationProgress<Array1<f64>>;

pub const SPLIT_ITER: u64 = 8;
pub const MAX_ITER: u64 = 20;

fn seed(dimension: usize) -> Array1<f64> {
    Array1::from_shape_fn(dimension, |i| 1.0 + (i % 5) as f64 * 0.25)
}

pub fn bfgs(dimension: usize) -> Stepper<Quadratic, BfgsState, BfgsSolver> {
    Executor::from_start(
        Quadratic,
        Bfgs::with_line_search(Constant(0.1)),
        seed(dimension),
    )
    .max_iter(MAX_ITER)
    .into_stepper()
    .unwrap()
}

pub fn cma_es(dimension: usize) -> Stepper<Quadratic, CmaState, CmaSolver> {
    let solver = CmaEs::new(0xc0_ffee, 0.8).with_eigen_update_period(3);
    Executor::new(
        Quadratic,
        solver,
        PopulationProgress::from_point(seed(dimension)),
    )
    .max_iter(MAX_ITER)
    .into_stepper()
    .unwrap()
}
