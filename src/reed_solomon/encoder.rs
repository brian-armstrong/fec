// Reed-Solomon encoder, derived from libcorrect's encode.c.

use std::fmt;

use super::field::{Field, FieldElement, FieldLogarithm, FieldOperation};
use super::polynomial::{polynomial_mod, reed_solomon_build_generator, Polynomial};

/// Error returned by [`Encoder::encode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EncodeError {
    /// The message was longer than the code's message capacity.
    MessageTooLong,
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncodeError::MessageTooLong => {
                write!(f, "message is longer than the code's message capacity")
            }
        }
    }
}

impl std::error::Error for EncodeError {}

/// Reed-Solomon encoder over GF(2^8).
pub struct Encoder {
    block_length: usize,
    message_length: usize,
    min_distance: usize,

    field: Field,

    generator: Vec<FieldElement>,
    generator_order: usize,

    encoded_polynomial: Vec<FieldElement>,
    encoded_polynomial_order: usize,
    encoded_remainder: Vec<FieldElement>,
    encoded_remainder_order: usize,
}

impl Encoder {
    /// Build an encoder for a (255, 255 - `num_roots`) Reed-Solomon code over
    /// GF(2^8). The block size is always 255 bytes with 8-bit symbols. The
    /// resulting code can repair up to `num_roots / 2` corrupted bytes per
    /// block. A larger `num_roots` adds parity overhead and substantially slows
    /// decoding.
    ///
    /// `primitive_polynomial` should be one of the `PRIMITIVE_POLYNOMIAL_*`
    /// constants. Sane values for `first_consecutive_root` and
    /// `generator_root_gap` are 1 and 1. Not all combinations of values produce
    /// valid codes.
    pub fn new(
        primitive_polynomial: FieldOperation,
        first_consecutive_root: FieldLogarithm,
        generator_root_gap: FieldLogarithm,
        num_roots: usize,
    ) -> Encoder {
        let field = Field::new(primitive_polynomial);

        let block_length = 255usize;
        let min_distance = num_roots;
        let message_length = block_length - min_distance;

        let mut generator_roots = vec![0 as FieldElement; min_distance];
        // generator has order min_distance (min_distance+1 coefficients)
        let mut generator = vec![0 as FieldElement; min_distance + 1];
        reed_solomon_build_generator(
            &field,
            min_distance,
            first_consecutive_root,
            generator_root_gap as usize,
            &mut generator,
            &mut generator_roots,
        );

        // encoded_polynomial and encoded_remainder both have order block_length-1
        let encoded_polynomial = vec![0 as FieldElement; block_length];
        let encoded_remainder = vec![0 as FieldElement; block_length];

        Encoder {
            block_length,
            message_length,
            min_distance,
            field,
            generator,
            generator_order: min_distance,
            encoded_polynomial,
            encoded_polynomial_order: block_length - 1,
            encoded_remainder,
            encoded_remainder_order: block_length - 1,
        }
    }

    /// Build an encoder for the standard CCSDS (255,223) Reed-Solomon code
    /// (conventional-basis representation). Equivalent to `Encoder::new` with
    /// the CCSDS parameters. For the dual-basis representation used on the wire,
    /// see [`Encoder::encode_ccsds_dual`].
    pub fn new_ccsds() -> Encoder {
        use super::ccsds;
        Encoder::new(
            ccsds::CCSDS_PRIMITIVE_POLYNOMIAL,
            ccsds::CCSDS_FIRST_CONSECUTIVE_ROOT,
            ccsds::CCSDS_GENERATOR_ROOT_GAP,
            ccsds::CCSDS_NUM_ROOTS,
        )
    }

    /// The block length in bytes, always 255 for this GF(2^8) code.
    pub fn block_length(&self) -> usize {
        self.block_length
    }

    /// The message capacity in bytes, `block_length - num_roots`.
    pub fn message_length(&self) -> usize {
        self.message_length
    }

    /// The number of parity symbols, `num_roots`. The code corrects up to
    /// `min_distance / 2` byte errors per block.
    pub fn min_distance(&self) -> usize {
        self.min_distance
    }

    /// Encode `msg` into `encoded` (message bytes followed by parity), returning
    /// the encoded block length (always 255) on success, or
    /// [`EncodeError::MessageTooLong`] if `msg` exceeds the message capacity.
    ///
    /// `msg` may be shorter than the full payload, for example fewer than 223
    /// bytes for a (255, 223) code. Short messages are encoded with virtual
    /// padding that is not emitted. `encoded` must be at least
    /// `msg.len() + num_roots` bytes.
    pub fn encode(&mut self, msg: &[u8], encoded: &mut [u8]) -> Result<usize, EncodeError> {
        let msg_length = msg.len();
        if msg_length > self.message_length {
            return Err(EncodeError::MessageTooLong);
        }

        let order = self.encoded_polynomial_order;
        let pad_length = self.message_length - msg_length;

        for i in 0..msg_length {
            // message goes from high order to low order but polynomials go low to high
            // so we reverse on the way in and on the way out
            // we'd have to do a copy anyway so this reversal should be free
            self.encoded_polynomial[order - (i + pad_length)] = msg[i];
        }

        // 0-fill the rest of the coefficients -- this length will always be > 0
        // because the order of this poly is block_length and the msg_length <= message_length
        // e.g. 255 and 223
        for c in self.encoded_polynomial[(order + 1 - pad_length)..(order + 1)].iter_mut() {
            *c = 0;
        }
        for c in self.encoded_polynomial[..(order + 1 - self.message_length)].iter_mut() {
            *c = 0;
        }

        // remainder = encoded_polynomial mod generator
        let dividend = Polynomial::new(&self.encoded_polynomial, order);
        let divisor = Polynomial::new(&self.generator, self.generator_order);
        polynomial_mod(
            &self.field,
            &dividend,
            &divisor,
            &mut self.encoded_remainder,
            self.encoded_remainder_order,
        );

        // now return byte order to highest order to lowest order
        for i in 0..msg_length {
            encoded[i] = self.encoded_polynomial[order - (i + pad_length)];
        }

        for i in 0..self.min_distance {
            encoded[msg_length + i] = self.encoded_remainder[self.min_distance - (i + 1)];
        }

        Ok(self.block_length)
    }

    /// Encode a CCSDS dual-basis message. `msg` holds dual-basis symbols, as
    /// they appear on the wire. The 32 parity bytes are written to `parity`,
    /// also in the dual basis. The message is transformed to the conventional
    /// basis, encoded with this (CCSDS) code, and the parity transformed back.
    ///
    /// This encoder must have been built with the CCSDS parameters (see
    /// [`Encoder::new_ccsds`]). `parity` must be at least 32 bytes.
    pub fn encode_ccsds_dual(&mut self, msg: &[u8], parity: &mut [u8]) -> Result<(), EncodeError> {
        use super::ccsds;
        if msg.len() > self.message_length {
            return Err(EncodeError::MessageTooLong);
        }
        // transform the dual-basis message to conventional, encode, transform the
        // conventional parity back to dual
        let conv_msg: Vec<u8> = msg.iter().map(|&b| ccsds::dual_to_conv(b)).collect();
        let mut block = vec![0u8; self.block_length];
        self.encode(&conv_msg, &mut block)?;
        let m = msg.len();
        for (out, &c) in parity.iter_mut().zip(block[m..m + self.min_distance].iter()) {
            *out = ccsds::conv_to_dual(c);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ccsds primitive poly, first_consecutive_root=1, gap=1
    const CCSDS: FieldOperation = 0x187;

    #[test]
    fn encode_full_message() {
        let mut enc = Encoder::new(CCSDS, 1, 1, 32);
        let msg: Vec<u8> = (0..223u16).map(|i| i as u8).collect();
        let mut encoded = vec![0u8; 255];
        assert_eq!(enc.encode(&msg, &mut encoded), Ok(255));

        // message bytes pass through unchanged
        assert_eq!(&encoded[..223], &msg[..]);
        // the 32 parity bytes
        let parity: [u8; 32] = [
            250, 21, 66, 72, 244, 243, 22, 41, 243, 8, 201, 34, 14, 179, 56, 133, 151, 84, 252, 148, 217, 13, 168, 24,
            78, 91, 75, 252, 226, 117, 76, 40,
        ];
        assert_eq!(&encoded[223..], &parity);
    }

    #[test]
    fn encode_short_message() {
        let mut enc = Encoder::new(CCSDS, 1, 1, 32);
        let msg = [1u8, 2, 3, 4, 5];
        let mut encoded = vec![0u8; 255];
        assert_eq!(enc.encode(&msg, &mut encoded), Ok(255));

        let expected_head: [u8; 37] = [
            1, 2, 3, 4, 5, 205, 175, 54, 99, 247, 95, 68, 232, 240, 77, 62, 244, 127, 118, 152, 110, 225, 154, 248,
            117, 90, 78, 233, 19, 151, 103, 160, 78, 181, 80, 154, 240,
        ];
        // first 5 are the message, next 32 are parity; the rest are zero
        assert_eq!(&encoded[..37], &expected_head);
        assert!(encoded[37..].iter().all(|&b| b == 0));
    }

    #[test]
    fn encode_single_byte() {
        let mut enc = Encoder::new(CCSDS, 1, 1, 32);
        let msg = [0xABu8];
        let mut encoded = vec![0u8; 255];
        assert_eq!(enc.encode(&msg, &mut encoded), Ok(255));

        let expected_head: [u8; 33] = [
            171, 47, 46, 31, 254, 6, 84, 239, 205, 64, 128, 170, 100, 165, 105, 196, 228, 187, 196, 104, 6, 182, 9,
            245, 98, 231, 116, 72, 40, 189, 106, 250, 11,
        ];
        assert_eq!(&encoded[..33], &expected_head);
        assert!(encoded[33..].iter().all(|&b| b == 0));
    }

    #[test]
    fn encode_zeros_gives_zero_parity() {
        // an all-zero message has an all-zero codeword (0 mod g = 0)
        let mut enc = Encoder::new(CCSDS, 1, 1, 32);
        let msg = [0u8; 10];
        let mut encoded = vec![0u8; 255];
        assert_eq!(enc.encode(&msg, &mut encoded), Ok(255));
        assert!(encoded.iter().all(|&b| b == 0));
    }

    #[test]
    fn encode_rejects_oversized_message() {
        let mut enc = Encoder::new(CCSDS, 1, 1, 32);
        let msg = vec![0u8; 224]; // > message_length (223)
        let mut encoded = vec![0u8; 255];
        assert_eq!(enc.encode(&msg, &mut encoded), Err(EncodeError::MessageTooLong));
    }

    #[test]
    fn message_length_is_block_minus_distance() {
        let enc = Encoder::new(CCSDS, 1, 1, 32);
        assert_eq!(enc.block_length(), 255);
        assert_eq!(enc.min_distance(), 32);
        assert_eq!(enc.message_length(), 223);
    }
}
