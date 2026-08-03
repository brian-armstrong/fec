#![cfg_attr(feature = "simd", feature(portable_simd))]
#![cfg_attr(docsrs, feature(doc_cfg))]
// This crate does substantial index arithmetic: polynomial coefficients, trellis
// state addressing, and basis transforms index arrays by computed offsets and by
// more than one array at once. `needless_range_loop` assumes an iterator rewrite
// always applies, but here it frequently doesn't (its suggestions are often
// wrong or don't compile), so the plain indexed loop is the clearer form.
#![allow(clippy::needless_range_loop)]
// The public API is fully documented; keep it that way.
#![deny(missing_docs)]

//! Forward error correction for SDR, space, and satellite links.
//!
//! The two codecs each live in their own module. [`convolutional`] holds the
//! Viterbi encoder and decoder. [`reed_solomon`] holds the Reed-Solomon codec,
//! including the standard CCSDS (255,223) code.
//!
//! Each module names its types plainly as `Encoder` and `Decoder`. The crate
//! root also re-exports them under `Conv`- and `Rs`-prefixed aliases, so both
//! codecs can be used in one scope without a name clash. Reach for
//! [`convolutional::Encoder`] when you work with one codec, or [`ConvEncoder`]
//! and [`RsEncoder`] when you want both.
//!
//! The Reed-Solomon codec and the scalar convolutional codec build on stable
//! Rust. The SIMD convolutional decoder is behind the `simd` feature, which
//! needs a nightly compiler because it uses `portable_simd`.

pub mod convolutional;
pub mod reed_solomon;

mod util;

#[cfg(feature = "simd")]
#[cfg_attr(docsrs, doc(cfg(feature = "simd")))]
#[doc(inline)]
pub use convolutional::SimdDecoder as ConvSimdDecoder;
#[doc(inline)]
pub use convolutional::{
    DecodeError as ConvDecodeError, Decoder as ConvDecoder, EncodeError as ConvEncodeError, Encoder as ConvEncoder,
};
#[doc(inline)]
pub use reed_solomon::{
    DecodeError as RsDecodeError, Decoder as RsDecoder, EncodeError as RsEncodeError, Encoder as RsEncoder,
};
