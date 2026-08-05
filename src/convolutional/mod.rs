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
mod puncture;
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
pub use self::error::{DecodeError, EncodeError, PunctureError};
#[doc(inline)]
pub use self::puncture::Puncturer;

#[cfg(feature = "simd")]
#[cfg_attr(docsrs, doc(cfg(feature = "simd")))]
#[doc(inline)]
pub use self::simd::SimdDecoder;

#[cfg(feature = "simd")]
pub use self::simd::ForcedPath;

#[cfg(feature = "simd")]
pub use self::simd::DecoderArch;

/// Returns the encoded length, in *bits*, of a message of `len` *bytes* under a
/// code with this `rate` and `order`.
///
/// The count includes the `order + 1` flush bits the encoder appends.
pub fn encoded_len_bits(rate: u32, order: u32, len: usize) -> usize {
    rate as usize * (len * 8 + order as usize + 1)
}

/// Returns the payload length, in *bytes*, that decoding `num_encoded_bits` bits
/// produces under a code with this `rate` and `order`.
///
/// This is the inverse of [`encoded_len_bits`].
pub fn payload_len_bytes(rate: u32, order: u32, num_encoded_bits: usize) -> usize {
    let decoded_bits = num_encoded_bits / rate as usize;
    decoded_bits.saturating_sub(order as usize + 1) / 8
}

#[cfg(test)]
mod tests {
    use super::{encoded_len_bits, payload_len_bytes, Encoder};

    #[test]
    fn length_helpers_round_trip() {
        for rate in 2..=8u32 {
            for order in 4..=16u32 {
                let polys = vec![0o7u16; rate as usize];
                let enc = Encoder::new(rate, order, &polys);
                for len in [0usize, 1, 7, 8, 9, 100, 255, 4096] {
                    let bits = encoded_len_bits(rate, order, len);
                    assert_eq!(
                        bits,
                        enc.encode_len(len),
                        "helper disagrees with encoder at {rate}/{order} len={len}"
                    );
                    assert_eq!(
                        payload_len_bytes(rate, order, bits),
                        len,
                        "round trip failed at {rate}/{order} len={len}"
                    );
                }
            }
        }
    }
}
