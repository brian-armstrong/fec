//! SIMD Viterbi decoding for convolutional codes.
//!
//! This module holds [`SimdDecoder`], a faster decoder for x86 hosts. It
//! decodes identically to the scalar [`Decoder`](crate::convolutional::Decoder),
//! and picks the widest instruction set the host supports at run time, from
//! AVX-512 down to a portable 128-bit fallback.
//!
//! The module is only compiled with the `simd` feature, which needs a nightly
//! compiler because it uses `portable_simd`.

mod decoder;
mod lane;
mod oct_lookup;

pub use decoder::SimdDecoder;

pub use decoder::ForcedPath;

pub use decoder::DecoderArch;