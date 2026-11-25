use bincode_next::{Decode, Encode, Fingerprint};
use rkyv::{Archive, Deserialize, Serialize, rancor::Error};
use wincode::{SchemaRead, SchemaWrite};

#[cfg_attr(
    feature = "serde-codecs",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(
    Archive,
    Deserialize,
    Serialize,
    Decode,
    Encode,
    Fingerprint,
    SchemaRead,
    SchemaWrite,
)]
struct Floats {
    f32s: [f32; 7],
    f64s: [f64; 7],
}

impl Floats {
    const F32_BITS: [u32; 7] = [
        0,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0001,
        0xffc0_5678,
        0x7f80_1234,
    ];
    const F64_BITS: [u64; 7] = [
        0,
        0x8000_0000_0000_0000,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_0001,
        0xfff8_1234_5678_9abc,
        0x7ff0_0000_0000_1234,
    ];

    fn new() -> Self {
        Self {
            f32s: Self::F32_BITS.map(f32::from_bits),
            f64s: Self::F64_BITS.map(f64::from_bits),
        }
    }

    fn assert_bits(self) {
        assert_eq!(self.f32s.map(f32::to_bits), Self::F32_BITS);
        assert_eq!(self.f64s.map(f64::to_bits), Self::F64_BITS);
    }
}

#[cfg(feature = "serde-codecs")]
#[test]
fn serde_codecs_preserve_float_bits() {
    for codec in checkpoint_codecs::SerdeCodec::ALL {
        let bytes = codec.encode(&Floats::new()).unwrap();
        codec.decode::<Floats>(&bytes).unwrap().assert_bits();
    }
}

#[test]
fn native_codecs_preserve_float_bits() {
    let sample = Floats::new();
    let bytes = rkyv::to_bytes::<Error>(&sample).unwrap();
    rkyv::from_bytes::<Floats, Error>(&bytes)
        .unwrap()
        .assert_bits();

    let bytes = wincode::serialize(&sample).unwrap();
    wincode::deserialize_exact::<Floats>(&bytes)
        .unwrap()
        .assert_bits();

    let config = bincode_next::config::standard().with_fingerprint();
    let bytes = bincode_next::encode_to_vec(sample, config).unwrap();
    let (decoded, consumed) =
        bincode_next::decode_from_slice::<Floats, _>(&bytes, config).unwrap();
    assert_eq!(consumed, bytes.len());
    decoded.assert_bits();
}
