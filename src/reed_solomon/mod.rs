// Reed-Solomon error correction over GF(2^8), translated from 
// [libcorrect](https://github.com/quiet/libcorrect).

pub(crate) mod field;
pub(crate) mod polynomial;
pub mod encoder;
pub mod decoder;
mod reed_solomon;

pub mod ccsds;

pub use encoder::{EncodeError, RsEncoder};
pub use decoder::{DecodeError, RsDecoder};

use field::FieldOperation;

pub const PRIMITIVE_POLYNOMIAL_8_4_3_2_0: FieldOperation = 0x11d; // x^8 + x^4 + x^3 + x^2 + 1
pub const PRIMITIVE_POLYNOMIAL_8_5_3_1_0: FieldOperation = 0x12b; // x^8 + x^5 + x^3 + x + 1
pub const PRIMITIVE_POLYNOMIAL_8_5_3_2_0: FieldOperation = 0x12d; // x^8 + x^5 + x^3 + x^2 + 1
pub const PRIMITIVE_POLYNOMIAL_8_6_3_2_0: FieldOperation = 0x14d; // x^8 + x^6 + x^3 + x^2 + 1
pub const PRIMITIVE_POLYNOMIAL_8_6_4_3_2_1_0: FieldOperation = 0x15f; // x^8 + x^6 + x^4 + x^3 + x^2 + x + 1
pub const PRIMITIVE_POLYNOMIAL_8_6_5_1_0: FieldOperation = 0x163; // x^8 + x^6 + x^5 + x + 1
pub const PRIMITIVE_POLYNOMIAL_8_6_5_2_0: FieldOperation = 0x165; // x^8 + x^6 + x^5 + x^2 + 1
pub const PRIMITIVE_POLYNOMIAL_8_6_5_3_0: FieldOperation = 0x169; // x^8 + x^6 + x^5 + x^3 + 1
pub const PRIMITIVE_POLYNOMIAL_8_6_5_4_0: FieldOperation = 0x171; // x^8 + x^6 + x^5 + x^4 + 1
pub const PRIMITIVE_POLYNOMIAL_8_7_2_1_0: FieldOperation = 0x187; // x^8 + x^7 + x^2 + x + 1
pub const PRIMITIVE_POLYNOMIAL_8_7_3_2_0: FieldOperation = 0x18d; // x^8 + x^7 + x^3 + x^2 + 1
pub const PRIMITIVE_POLYNOMIAL_8_7_5_3_0: FieldOperation = 0x1a9; // x^8 + x^7 + x^5 + x^3 + 1
pub const PRIMITIVE_POLYNOMIAL_8_7_6_1_0: FieldOperation = 0x1c3; // x^8 + x^7 + x^6 + x + 1
pub const PRIMITIVE_POLYNOMIAL_8_7_6_3_2_1_0: FieldOperation = 0x1cf; // x^8 + x^7 + x^6 + x^3 + x^2 + x + 1
pub const PRIMITIVE_POLYNOMIAL_8_7_6_5_2_1_0: FieldOperation = 0x1e7; // x^8 + x^7 + x^6 + x^5 + x^2 + x + 1
pub const PRIMITIVE_POLYNOMIAL_8_7_6_5_4_2_0: FieldOperation = 0x1f5; // x^8 + x^7 + x^6 + x^5 + x^4 + x^2 + 1

// CCSDS dual-basis standard code. Same value as 8_7_2_1_0.
pub const PRIMITIVE_POLYNOMIAL_CCSDS: FieldOperation = 0x187; // x^8 + x^7 + x^2 + x + 1
