use std::fmt;

/// Error returned by the convolutional decoders.
///
/// The scalar [`Decoder`](super::Decoder) returns this.
#[cfg_attr(
    feature = "simd",
    doc = "With the `simd` feature, so does [`SimdDecoder`](super::SimdDecoder)."
)]
///
/// A convolutional decoder always produces some output for a correctly-shaped
/// input. It cannot report "too many errors to correct". See
/// [`Decoder`](super::Decoder) for that caveat. These variants cover only
/// structural problems with the call. Either the encoded length is not a whole
/// number of code symbols, or the output buffer is too small for the decoded
/// payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecodeError {
    /// The encoded length in bits is not a multiple of the code's `rate`. It
    /// cannot be a sequence of whole code symbols.
    InvalidLength {
        /// The encoded length that was passed, in bits.
        num_encoded_bits: usize,
        /// The code's inverse rate (output bits per input bit).
        rate: u32,
    },
    /// The output buffer is too small to hold the decoded payload.
    OutputTooSmall {
        /// Payload length the decode would produce, in bytes.
        needed: usize,
        /// Length of the buffer that was supplied, in bytes.
        actual: usize,
    },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::InvalidLength { num_encoded_bits, rate } => write!(
                f,
                "encoded length {num_encoded_bits} bits is not a multiple of rate {rate}"
            ),
            DecodeError::OutputTooSmall { needed, actual } => write!(
                f,
                "output buffer holds {actual} bytes but the decoded payload needs {needed}"
            ),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Error returned by [`Encoder::encode`](super::Encoder::encode).
///
/// Encoding itself cannot fail. The only error is a caller-supplied output
/// buffer too small to hold the encoded block. A polynomial or parameter shape
/// that does not match the `(rate, order)` is a programmer error and panics in
/// [`Encoder::new`](super::Encoder::new) instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodeError {
    /// The output buffer is too small to hold the encoded block.
    OutputTooSmall {
        /// Encoded length the call would produce, in bytes.
        needed: usize,
        /// Length of the buffer that was supplied, in bytes.
        actual: usize,
    },
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncodeError::OutputTooSmall { needed, actual } => write!(
                f,
                "output buffer holds {actual} bytes but the encoded block needs {needed}"
            ),
        }
    }
}

impl std::error::Error for EncodeError {}

/// Validates a decode's encoded length against the code parameters and returns
/// the decoded bit count. This is shared by the scalar and SIMD decoders so
/// their preconditions cannot drift apart.
///
/// The length must be a whole number of code symbols, and it must be long
/// enough to run the head and tail phases. Both failures return
/// [`InvalidLength`](DecodeError::InvalidLength).
pub(crate) fn validate_encoded_len(num_encoded_bits: usize, rate: u32, order: u32) -> Result<u32, DecodeError> {
    let invalid = DecodeError::InvalidLength { num_encoded_bits, rate };

    if !num_encoded_bits.is_multiple_of(rate as usize) {
        return Err(invalid);
    }

    let num_decoded_bits = num_encoded_bits as u32 / rate;
    // we need at least 2 * order - 1 decoded bits to run head and tail
    if num_decoded_bits < 2 * order - 1 {
        return Err(invalid);
    }

    Ok(num_decoded_bits)
}

/// Length in bytes of the payload a decode of `num_encoded_bits` will produce.
/// It is the decoded bit count `num_encoded_bits / rate`, less the `order + 1`
/// flush tail, as whole bytes.
pub(crate) fn payload_len_bytes(num_encoded_bits: usize, rate: u32, order: u32) -> usize {
    let decoded_bits = num_encoded_bits / rate as usize;
    decoded_bits.saturating_sub(order as usize + 1) / 8
}
