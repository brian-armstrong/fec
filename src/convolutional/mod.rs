//! Convolutional codes with a Viterbi decoder.
//!
//! A convolutional code is defined by its inverse rate, its order (the
//! constraint length), and a set of generator polynomials. Build an
//! [`Encoder`] and a [`Decoder`] with the same parameters. The decoder
//! recovers the message up to the error-correcting power of the code,
//! and cannot tell you when that limit is exceeded.
//!
//! The decoder does both hard-decision and soft-decision decoding. Soft
//! decisions take an 8-bit confidence per encoded bit and correct more errors.
#![cfg_attr(
    feature = "simd",
    doc = "With the `simd` feature there is also [`SimdDecoder`], which decodes"
)]
#![cfg_attr(feature = "simd", doc = "identically to [`Decoder`] but faster on x86.")]
//!
//! Standard parameters for common codes, such as the rate-1/2 order-7 NASA
//! code, follow the octal polynomial convention used across the SDR and space
//! literature. The [`sim`] module holds a channel and BER harness used by the
//! tests and tuning tools.

pub(crate) mod bit;
mod decoder;
mod encoder;
mod error;
pub mod sim;
#[cfg(feature = "simd")]
#[cfg_attr(docsrs, doc(cfg(feature = "simd")))]
pub mod simd;
pub(crate) mod util;

#[doc(inline)]
pub use self::decoder::Decoder;
#[doc(inline)]
pub use self::encoder::Encoder;
#[doc(inline)]
pub use self::error::{DecodeError, EncodeError};

#[cfg(feature = "simd")]
#[cfg_attr(docsrs, doc(cfg(feature = "simd")))]
#[doc(inline)]
pub use self::simd::SimdDecoder;

#[cfg(feature = "simd")]
pub use self::simd::ForcedPath;

#[cfg(feature = "simd")]
pub use self::simd::DecoderArch;
