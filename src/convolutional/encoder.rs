use super::bit::{BitReader, BitWriter};
use super::error::EncodeError;
use super::util;

/// A convolutional encoder for a given `(rate, order, polynomials)`.
///
/// The `rate` is the inverse code rate. A rate-1/2 code has `rate == 2` and
/// emits 2 output bits per input bit. The `order` is the constraint length, the
/// number of bits in the shift register. The `polys` are the generator
/// polynomials, one per output bit, so `polys.len()` must equal `rate`.
///
/// The polynomials are octal, matching the convention used throughout the SDR
/// and space literature. For example, the canonical rate-1/2, order-7 NASA code
/// uses `[0o161, 0o127]`.
///
/// An `Encoder` decodes nothing. Pair it with a [`Decoder`](super::Decoder), or
/// the SIMD decoder, to recover the message.
#[derive(Debug)]
pub struct Encoder {
    rate: u32,
    order: u32,
    poly_table: Vec<u16>,
}

impl Encoder {
    /// Creates an encoder for the convolutional code with the given parameters.
    ///
    /// `polys` must contain exactly `rate` generator polynomials. For example,
    /// to build a rate-1/2, order-7 encoder with polynomials `0o161` and
    /// `0o127`, call `Encoder::new(2, 7, &[0o161, 0o127])`.
    ///
    /// # Panics
    ///
    /// Panics if `polys.len()` is not equal to `rate`. The polynomial count is
    /// fixed configuration, so a mismatch is a programmer error rather than a
    /// runtime condition.
    pub fn new(rate: u32, order: u32, polys: &[u16]) -> Encoder {
        Encoder {
            rate,
            order,
            poly_table: util::conv_poly_table(rate, order, polys),
        }
    }

    /// Returns the encoded length, in *bits*, of a message of `len` *bytes*.
    ///
    /// The count includes the `order + 1` flush bits appended to drive the
    /// shift register back to zero. It is therefore slightly larger than
    /// `rate * len * 8`. To size a byte buffer for [`encode`](Self::encode),
    /// round up with `encode_len(len).div_ceil(8)`.
    pub fn encode_len(&self, len: usize) -> usize {
        let bits = len * 8;
        self.rate as usize * (bits + self.order as usize + 1)
    }

    /// Encodes `msg` into `dst`, returning the number of *bits* written.
    ///
    /// `dst` must be large enough to hold the encoded output. Size it with
    /// [`encode_len`](Self::encode_len) converted to bytes. If the bit count
    /// is not a multiple of 8, the final byte is padded. The returned bit
    /// count tells you how many of its bits are significant.
    ///
    /// Returns [`OutputTooSmall`](EncodeError::OutputTooSmall) if `dst` cannot
    /// hold the encoded block.
    pub fn encode(&mut self, msg: &[u8], dst: &mut [u8]) -> Result<usize, EncodeError> {
        let encode_len = self.encode_len(msg.len());
        let needed = encode_len.div_ceil(8);
        if dst.len() < needed {
            return Err(EncodeError::OutputTooSmall {
                needed,
                actual: dst.len(),
            });
        }

        let mut bit_reader = BitReader::new(msg);
        let mut bit_writer = BitWriter::new(dst);

        // a convolutional code convolves the filter coefficients, given by the
        // polynomials, with some history from the message. the history is stored
        // as the most recent `order` bits in the shift register: oldest bits on
        // the left, newest on the right.
        let mut shift_register: u32 = 0;
        // the shift mask removes bits that extend beyond `order`. for e.g. order 7 it
        // drops the 8th bit and above.
        let shift_mask: u32 = (1 << self.order) - 1;

        for _i in 0..8 * msg.len() {
            // shift the newest message bit in on the right, then trim to order.
            shift_register <<= 1;
            shift_register |= bit_reader.read(1) as u32;
            shift_register &= shift_mask;

            // direct lookup of the convolutional output. all `rate` output bits
            // for this register state are packed into one row of the table.
            bit_writer.write(self.poly_table[shift_register as usize] as u8, self.rate as usize);
        }

        // flush the shift register. run the same loop with no new inputs, e.g.
        // shifting in all 0s to drive the register back to the zero state.
        for _i in 0..self.order + 1 {
            shift_register <<= 1;
            shift_register &= shift_mask;
            bit_writer.write(self.poly_table[shift_register as usize] as u8, self.rate as usize);
        }

        // 0-fill any remaining bits in the final byte.
        bit_writer.flush();
        Ok(encode_len)
    }
}

#[cfg(test)]
mod tests {
    use super::{EncodeError, Encoder};

    #[test]
    fn encode_rejects_small_output() {
        let mut enc = Encoder::new(2, 7, &[0o161, 0o127]);
        let msg = [1u8, 2, 3, 4];
        // one byte short of the encoded block
        let needed = enc.encode_len(msg.len()).div_ceil(8);
        let mut too_small = vec![0u8; needed - 1];
        assert!(matches!(
            enc.encode(&msg, &mut too_small),
            Err(EncodeError::OutputTooSmall { .. })
        ));
    }

    #[test]
    #[should_panic(expected = "generator polynomials")]
    fn new_panics_on_wrong_poly_count() {
        // rate 2 needs 2 polynomials, only 1 given
        Encoder::new(2, 7, &[0o161]);
    }
}
