//! Reed-Solomon error correction over GF(2^8).
//!
//! A Reed-Solomon code works on 255-byte blocks. It reserves `num_roots` bytes
//! for parity and carries `255 - num_roots` bytes of message. It repairs up to
//! `num_roots / 2` corrupted bytes per block, or more if some positions are
//! already flagged as erasures. Build an [`Encoder`] and a [`Decoder`] with
//! matching parameters. The decoder handles both plain errors and erasures.
//!
//! The [`PRIMITIVE_POLYNOMIAL_*`](PRIMITIVE_POLYNOMIAL_8_4_3_2_0) constants
//! supply the field polynomials for building custom codes. For the standard
//! CCSDS (255,223) code used in spacecraft telemetry, use
//! [`Encoder::new_ccsds`] and [`Decoder::new_ccsds`]. Telemetry places symbols
//! on the wire in a dual basis, so the [`ccsds`] module and the
//! `*_ccsds_dual` methods convert between that representation and the
//! conventional one.
//!
//! The codec is a Rust translation of
//! [libcorrect](https://github.com/quiet/libcorrect).

mod decoder;
mod encoder;
pub(crate) mod field;
pub(crate) mod polynomial;
#[cfg(test)]
mod tests;

pub mod ccsds;

#[doc(inline)]
pub use decoder::{DecodeError, Decoder};
#[doc(inline)]
pub use encoder::{EncodeError, Encoder};

use field::FieldOperation;

/// Primitive polynomial `x^8 + x^4 + x^3 + x^2 + 1` for GF(2^8).
pub const PRIMITIVE_POLYNOMIAL_8_4_3_2_0: FieldOperation = 0x11d;
/// Primitive polynomial `x^8 + x^5 + x^3 + x + 1` for GF(2^8).
pub const PRIMITIVE_POLYNOMIAL_8_5_3_1_0: FieldOperation = 0x12b;
/// Primitive polynomial `x^8 + x^5 + x^3 + x^2 + 1` for GF(2^8).
pub const PRIMITIVE_POLYNOMIAL_8_5_3_2_0: FieldOperation = 0x12d;
/// Primitive polynomial `x^8 + x^6 + x^3 + x^2 + 1` for GF(2^8).
pub const PRIMITIVE_POLYNOMIAL_8_6_3_2_0: FieldOperation = 0x14d;
/// Primitive polynomial `x^8 + x^6 + x^4 + x^3 + x^2 + x + 1` for GF(2^8).
pub const PRIMITIVE_POLYNOMIAL_8_6_4_3_2_1_0: FieldOperation = 0x15f;
/// Primitive polynomial `x^8 + x^6 + x^5 + x + 1` for GF(2^8).
pub const PRIMITIVE_POLYNOMIAL_8_6_5_1_0: FieldOperation = 0x163;
/// Primitive polynomial `x^8 + x^6 + x^5 + x^2 + 1` for GF(2^8).
pub const PRIMITIVE_POLYNOMIAL_8_6_5_2_0: FieldOperation = 0x165;
/// Primitive polynomial `x^8 + x^6 + x^5 + x^3 + 1` for GF(2^8).
pub const PRIMITIVE_POLYNOMIAL_8_6_5_3_0: FieldOperation = 0x169;
/// Primitive polynomial `x^8 + x^6 + x^5 + x^4 + 1` for GF(2^8).
pub const PRIMITIVE_POLYNOMIAL_8_6_5_4_0: FieldOperation = 0x171;
/// Primitive polynomial `x^8 + x^7 + x^2 + x + 1` for GF(2^8).
pub const PRIMITIVE_POLYNOMIAL_8_7_2_1_0: FieldOperation = 0x187;
/// Primitive polynomial `x^8 + x^7 + x^3 + x^2 + 1` for GF(2^8).
pub const PRIMITIVE_POLYNOMIAL_8_7_3_2_0: FieldOperation = 0x18d;
/// Primitive polynomial `x^8 + x^7 + x^5 + x^3 + 1` for GF(2^8).
pub const PRIMITIVE_POLYNOMIAL_8_7_5_3_0: FieldOperation = 0x1a9;
/// Primitive polynomial `x^8 + x^7 + x^6 + x + 1` for GF(2^8).
pub const PRIMITIVE_POLYNOMIAL_8_7_6_1_0: FieldOperation = 0x1c3;
/// Primitive polynomial `x^8 + x^7 + x^6 + x^3 + x^2 + x + 1` for GF(2^8).
pub const PRIMITIVE_POLYNOMIAL_8_7_6_3_2_1_0: FieldOperation = 0x1cf;
/// Primitive polynomial `x^8 + x^7 + x^6 + x^5 + x^2 + x + 1` for GF(2^8).
pub const PRIMITIVE_POLYNOMIAL_8_7_6_5_2_1_0: FieldOperation = 0x1e7;
/// Primitive polynomial `x^8 + x^7 + x^6 + x^5 + x^4 + x^2 + 1` for GF(2^8).
pub const PRIMITIVE_POLYNOMIAL_8_7_6_5_4_2_0: FieldOperation = 0x1f5;

/// Primitive polynomial `x^8 + x^7 + x^2 + x + 1` used by the CCSDS standard
/// code. This is the same value as [`PRIMITIVE_POLYNOMIAL_8_7_2_1_0`].
pub const PRIMITIVE_POLYNOMIAL_CCSDS: FieldOperation = 0x187;
