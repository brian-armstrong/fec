use super::bit::{BitReader, BitWriter};
use super::error::PunctureError;

/// A puncturing pattern for a convolutional code.
///
/// Puncturing raises the rate of a code by deleting some of the encoder's output
/// bits before transmission. The decoder reinserts the missing positions as
/// erasures, so they contribute nothing to the path metric, and runs the
/// original code's trellis. A rate-1/2 code punctured to send 3 of every 4
/// output bits becomes a rate-3/4 code.
///
/// The pattern is a repeating keep-mask over the encoded bit stream. Encoded bit
/// `k` is transmitted when `keep[k % period]` is true and deleted when it is
/// false. The pattern cycles continuously across the whole encoded block,
/// including the flush tail the encoder appends, so the caller never has to
/// reason about where the message ends.
///
/// Use [`Puncturer::new`] for a keep-mask in that flat form. Use
/// [`Puncturer::from_matrix`] for the `rate` by `period` puncturing matrix used
/// in the coding literature and in standards.
#[derive(Debug, Clone)]
pub struct Puncturer {
    keep: Vec<bool>,
    kept_per_period: usize,
}

impl Puncturer {
    /// Creates a puncturer from a flat keep-mask.
    ///
    /// `keep[i]` says whether encoded bit `i` of each period is transmitted. The
    /// period is `keep.len()`.
    ///
    /// Returns [`EmptyPattern`](PunctureError::EmptyPattern) if `keep` is empty,
    /// and [`NoKeptBits`](PunctureError::NoKeptBits) if every entry is false,
    /// which would delete the entire stream.
    pub fn new(keep: &[bool]) -> Result<Puncturer, PunctureError> {
        if keep.is_empty() {
            return Err(PunctureError::EmptyPattern);
        }

        let kept_per_period = keep.iter().filter(|&&k| k).count();
        if kept_per_period == 0 {
            return Err(PunctureError::NoKeptBits);
        }

        Ok(Puncturer {
            keep: keep.to_vec(),
            kept_per_period,
        })
    }

    /// Creates a puncturer from a `rate` by `period` puncturing matrix.
    ///
    /// This is the form used in the coding literature and in standards. Row `r`
    /// is the pattern applied to the output of generator polynomial `r`, and
    /// column `c` is one step of the period. A rate-1/2 code punctured to rate
    /// 3/4 uses `[[true, true], [true, false]]`.
    ///
    /// The encoder emits all `rate` output bits of one input step together, so
    /// column `c` of the matrix maps to encoded bits `c * rate .. (c + 1) * rate`
    /// and row `r` picks the bit within that group. The flattened keep-mask is
    /// therefore `keep[c * rate + r] = matrix[r][c]`.
    ///
    /// Returns [`EmptyPattern`](PunctureError::EmptyPattern) if the matrix has no
    /// rows or no columns, [`RaggedMatrix`](PunctureError::RaggedMatrix) if the
    /// rows are not all the same length, and
    /// [`NoKeptBits`](PunctureError::NoKeptBits) if every entry is false.
    pub fn from_matrix(matrix: &[&[bool]]) -> Result<Puncturer, PunctureError> {
        let rate = matrix.len();

        if rate == 0 {
            return Err(PunctureError::EmptyPattern);
        }

        let period = matrix[0].len();
        if period == 0 {
            return Err(PunctureError::EmptyPattern);
        }

        if matrix.iter().any(|row| row.len() != period) {
            return Err(PunctureError::RaggedMatrix);
        }

        let mut keep = vec![false; rate * period];
        for (r, row) in matrix.iter().enumerate() {
            for (c, &k) in row.iter().enumerate() {
                keep[c * rate + r] = k;
            }
        }

        Puncturer::new(&keep)
    }

    /// The length of the repeating pattern, in encoded bits.
    pub fn period(&self) -> usize {
        self.keep.len()
    }

    #[inline]
    fn keeps(&self, index: usize) -> bool {
        self.keep[index % self.keep.len()]
    }

    /// Returns the punctured length, in bits, of an encoded block of
    /// `encoded_bits` bits.
    ///
    /// `encoded_bits` is the length the encoder produced, as returned by
    /// [`Encoder::encode`](super::Encoder::encode).
    pub fn punctured_len(&self, encoded_bits: usize) -> usize {
        let periods = encoded_bits / self.period();
        let remainder = encoded_bits % self.period();
        let tail = self.keep[..remainder].iter().filter(|&&k| k).count();
        periods * self.kept_per_period + tail
    }

    /// Deletes the punctured bits from an encoded block, returning the number of
    /// bits written to `dst`.
    ///
    /// `src` holds the encoder's output and `encoded_bits` is its length in bits.
    /// Size `dst` with [`punctured_len`](Self::punctured_len), rounded up to
    /// bytes. Both buffers are packed bit streams, most significant bit first.
    ///
    /// Returns [`OutputTooSmall`](PunctureError::OutputTooSmall) if `dst` cannot
    /// hold the punctured block, and
    /// [`InputTooSmall`](PunctureError::InputTooSmall) if `src` is shorter than
    /// `encoded_bits` describes.
    pub fn puncture(&self, src: &[u8], encoded_bits: usize, dst: &mut [u8]) -> Result<usize, PunctureError> {
        if src.len() < encoded_bits.div_ceil(8) {
            return Err(PunctureError::InputTooSmall {
                needed: encoded_bits.div_ceil(8),
                actual: src.len(),
            });
        }

        let out_bits = self.punctured_len(encoded_bits);
        let needed = out_bits.div_ceil(8);
        if dst.len() < needed {
            return Err(PunctureError::OutputTooSmall {
                needed,
                actual: dst.len(),
            });
        }

        let mut reader = BitReader::new(src);
        let mut writer = BitWriter::new(dst);
        for i in 0..encoded_bits {
            let bit = reader.read(1);
            if self.keeps(i) {
                writer.write(bit, 1);
            }
        }
        writer.flush();

        Ok(out_bits)
    }

    /// Reinserts the punctured positions of a hard-decision block, building the
    /// erasure mask that marks them.
    ///
    /// `src` is the received punctured bit stream and `encoded_bits` is the
    /// length of the original unpunctured block. A receiver gets that from the
    /// payload length with
    /// [`Decoder::encoded_len_bits`](super::Decoder::encoded_len_bits) on the
    /// decoder it is about to use. `dst` receives the full-width bit stream and
    /// `erasure` receives one bit per encoded bit, set where the bit was
    /// punctured. Size both with `encoded_bits.div_ceil(8)`.
    ///
    /// Pass `dst` and `erasure` to
    /// [`decode_hard_with_erasure`](super::Decoder::decode_hard_with_erasure).
    ///
    /// Returns [`InputTooSmall`](PunctureError::InputTooSmall) if `src` is
    /// shorter than the punctured length, and
    /// [`OutputTooSmall`](PunctureError::OutputTooSmall) if `dst` or `erasure`
    /// is too small.
    pub fn depuncture_hard(
        &self,
        src: &[u8],
        encoded_bits: usize,
        dst: &mut [u8],
        erasure: &mut [u8],
    ) -> Result<(), PunctureError> {
        let in_bits = self.punctured_len(encoded_bits);
        if src.len() < in_bits.div_ceil(8) {
            return Err(PunctureError::InputTooSmall {
                needed: in_bits.div_ceil(8),
                actual: src.len(),
            });
        }

        let needed = encoded_bits.div_ceil(8);
        if dst.len() < needed {
            return Err(PunctureError::OutputTooSmall {
                needed,
                actual: dst.len(),
            });
        }

        if erasure.len() < needed {
            return Err(PunctureError::OutputTooSmall {
                needed,
                actual: erasure.len(),
            });
        }

        let mut reader = BitReader::new(src);
        let mut writer = BitWriter::new(dst);
        let mut erasure_writer = BitWriter::new(erasure);
        for i in 0..encoded_bits {
            if self.keeps(i) {
                writer.write(reader.read(1), 1);
                erasure_writer.write(0, 1);
            } else {
                // the value is arbitrary. the erasure flag is what the decoder reads.
                writer.write(0, 1);
                erasure_writer.write(1, 1);
            }
        }
        writer.flush();
        erasure_writer.flush();

        Ok(())
    }

    /// Reinserts the punctured positions of a soft-decision block, building the
    /// erasure mask that marks them.
    ///
    /// `src` holds one 8-bit soft symbol per received bit. `dst` receives one
    /// soft symbol per encoded bit, so its length is the length of the original
    /// unpunctured block. A receiver gets that from the payload length with
    /// [`Decoder::encoded_len_bits`](super::Decoder::encoded_len_bits) on the
    /// decoder it is about to use. `erasure` receives one bit per encoded bit
    /// and must hold `dst.len().div_ceil(8)` bytes.
    ///
    /// Pass `dst` and `erasure` to
    /// [`decode_soft_with_erasure`](super::Decoder::decode_soft_with_erasure).
    ///
    /// Returns [`InputTooSmall`](PunctureError::InputTooSmall) if `src` is
    /// shorter than the punctured length, and
    /// [`OutputTooSmall`](PunctureError::OutputTooSmall) if `erasure` is too
    /// small.
    pub fn depuncture_soft(&self, src: &[u8], dst: &mut [u8], erasure: &mut [u8]) -> Result<(), PunctureError> {
        let encoded_bits = dst.len();
        let in_bits = self.punctured_len(encoded_bits);

        if src.len() < in_bits {
            return Err(PunctureError::InputTooSmall {
                needed: in_bits,
                actual: src.len(),
            });
        }

        let mask_bytes = encoded_bits.div_ceil(8);
        if erasure.len() < mask_bytes {
            return Err(PunctureError::OutputTooSmall {
                needed: mask_bytes,
                actual: erasure.len(),
            });
        }

        let mut read = 0usize;
        let mut erasure_writer = BitWriter::new(erasure);
        for i in 0..encoded_bits {
            if self.keeps(i) {
                dst[i] = src[read];
                read += 1;
                erasure_writer.write(0, 1);
            } else {
                // this value also does not matter. only the erasure bit will be used by the decoder
                dst[i] = 128;
                erasure_writer.write(1, 1);
            }
        }
        erasure_writer.flush();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Puncturer;
    use crate::convolutional::{Decoder, Encoder, PunctureError};
    use crate::util::Rng;

    const RATE_3_4: [[bool; 2]; 2] = [[true, true], [true, false]];

    fn new(m: &[[bool; 2]; 2]) -> Puncturer {
        let rows: Vec<&[bool]> = m.iter().map(|r| r.as_slice()).collect();
        Puncturer::from_matrix(&rows).unwrap()
    }

    #[test]
    fn rejects_bad_patterns() {
        assert!(matches!(Puncturer::new(&[]), Err(PunctureError::EmptyPattern)));
        assert!(matches!(
            Puncturer::new(&[false, false]),
            Err(PunctureError::NoKeptBits)
        ));
        assert!(matches!(Puncturer::from_matrix(&[]), Err(PunctureError::EmptyPattern)));
        let ragged: [&[bool]; 2] = [&[true, true], &[true]];
        assert!(matches!(
            Puncturer::from_matrix(&ragged),
            Err(PunctureError::RaggedMatrix)
        ));
    }

    #[test]
    fn matrix_flattens_in_stream_order() {
        let p = new(&RATE_3_4);
        assert_eq!(p.period(), 4);
        assert_eq!(p.keep, vec![true, true, true, false]);
    }

    #[test]
    fn punctured_len_counts_kept_bits() {
        let p = new(&RATE_3_4);
        // 3 of every 4 bits survive
        assert_eq!(p.punctured_len(4), 3);
        assert_eq!(p.punctured_len(8), 6);
        assert_eq!(p.punctured_len(400), 300);
        // partial periods count only the kept positions of the prefix
        assert_eq!(p.punctured_len(1), 1);
        assert_eq!(p.punctured_len(2), 2);
        assert_eq!(p.punctured_len(3), 3);
        assert_eq!(p.punctured_len(5), 4);
    }

    #[test]
    fn keep_all_is_identity() {
        let p = Puncturer::new(&[true, true]).unwrap();
        let src = [0x77, 0xff];
        let bits = 16;
        let mut punctured = vec![0u8; 2];
        assert_eq!(p.puncture(&src, bits, &mut punctured).unwrap(), 16);
        assert_eq!(punctured, src);

        let mut expanded = vec![0u8; 2];
        let mut erasure = vec![0xFFu8; 2];
        p.depuncture_hard(&punctured, bits, &mut expanded, &mut erasure)
            .unwrap();
        assert_eq!(expanded, src);
        assert_eq!(erasure, vec![0u8; 2], "nothing should be flagged erased");
    }

    #[test]
    fn hard_round_trip_restores_kept_bits() {
        let p = new(&RATE_3_4);
        let mut rng = Rng::new(99);
        let bits = 800;
        let mut src = vec![0u8; bits / 8];

        for b in &mut src {
            *b = rng.next_u8();
        }

        let mut punctured = vec![0u8; p.punctured_len(bits).div_ceil(8)];
        p.puncture(&src, bits, &mut punctured).unwrap();

        let mut expanded = vec![0u8; bits / 8];
        let mut erasure = vec![0u8; bits / 8];
        p.depuncture_hard(&punctured, bits, &mut expanded, &mut erasure)
            .unwrap();

        for i in 0..bits {
            let erased = erasure[i / 8] & (0x80 >> (i % 8)) != 0;
            assert_eq!(erased, !p.keeps(i), "erasure flag wrong at bit {i}");
            if !erased {
                let want = src[i / 8] & (0x80 >> (i % 8));
                let got = expanded[i / 8] & (0x80 >> (i % 8));
                assert_eq!(got, want, "kept bit {i} did not survive the round trip");
            }
        }
    }

    #[test]
    fn soft_round_trip_restores_kept_symbols() {
        let p = new(&RATE_3_4);
        let bits = 400;
        let soft_full: Vec<u8> = (0..bits).map(|i| (i % 256) as u8).collect();

        let punctured: Vec<u8> = (0..bits).filter(|&i| p.keeps(i)).map(|i| soft_full[i]).collect();
        assert_eq!(punctured.len(), p.punctured_len(bits));

        let mut expanded = vec![0u8; bits];
        let mut erasure = vec![0u8; bits.div_ceil(8)];
        p.depuncture_soft(&punctured, &mut expanded, &mut erasure).unwrap();

        for i in 0..bits {
            let erased = erasure[i / 8] & (0x80 >> (i % 8)) != 0;
            assert_eq!(erased, !p.keeps(i), "erasure flag wrong at symbol {i}");
            if erased {
                assert_eq!(expanded[i], 128, "punctured slot should hold the neutral value");
            } else {
                assert_eq!(expanded[i], soft_full[i], "kept symbol {i} did not survive");
            }
        }
    }

    #[test]
    fn rejects_bad_buffers() {
        let p = new(&RATE_3_4);
        let src = vec![0u8; 100];
        let bits = src.len() * 8;

        // output too small for the punctured block
        let mut tiny = vec![0u8; 1];
        assert!(matches!(
            p.puncture(&src, bits, &mut tiny),
            Err(PunctureError::OutputTooSmall { .. })
        ));

        // input shorter than encoded_bits describes
        let mut out = vec![0u8; 100];
        assert!(matches!(
            p.puncture(&[0u8; 4], bits, &mut out),
            Err(PunctureError::InputTooSmall { .. })
        ));

        // depuncture with a short erasure buffer
        let punctured = vec![0u8; p.punctured_len(bits).div_ceil(8)];
        let mut expanded = vec![0u8; 100];

        assert!(matches!(
            p.depuncture_hard(&punctured, bits, &mut expanded, &mut [0u8; 1]),
            Err(PunctureError::OutputTooSmall { .. })
        ));
        assert!(matches!(
            p.depuncture_soft(&[0u8; 8], &mut vec![0u8; bits], &mut [0u8; 1]),
            Err(PunctureError::InputTooSmall { .. })
        ));
    }

    #[test]
    fn handles_non_byte_aligned_block() {
        let p = new(&RATE_3_4);
        let mut unaligned_enc_seen = 0;
        let mut unaligned_punc_seen = 0;

        let mut rng = Rng::new(0xABCD_1234);

        for enc_bits in 8usize..=512 {
            if !enc_bits.is_multiple_of(8) {
                unaligned_enc_seen += 1;
            }

            let mut encoded = vec![0u8; enc_bits.div_ceil(8)];
            for b in &mut encoded {
                *b = rng.next_u8();
            }

            let punctured_len = p.punctured_len(enc_bits);
            if !punctured_len.is_multiple_of(8) {
                unaligned_punc_seen += 1;
            }

            let mut punctured = vec![0u8; punctured_len.div_ceil(8)];
            p.puncture(&encoded, enc_bits, &mut punctured).unwrap();

            // hard: explicit count, so the trailing partial byte is handled exactly
            let mut expanded = vec![0u8; enc_bits.div_ceil(8)];
            let mut erasure = vec![0u8; enc_bits.div_ceil(8)];
            p.depuncture_hard(&punctured, enc_bits, &mut expanded, &mut erasure)
                .unwrap();
            for i in 0..enc_bits {
                let erased = erasure[i / 8] & (0x80 >> (i % 8)) != 0;
                assert_eq!(erased, !p.keeps(i), "hard erasure flag wrong at bit {i}");
                if !erased {
                    assert_eq!(
                        expanded[i / 8] & (0x80 >> (i % 8)),
                        encoded[i / 8] & (0x80 >> (i % 8)),
                        "hard kept bit {i} did not survive"
                    );
                }
            }

            // soft: dst.len() carries the exact bit count, no argument needed
            let soft_punctured: Vec<u8> = (0..enc_bits)
                .filter(|&i| p.keeps(i))
                .map(|i| {
                    if encoded[i / 8] & (0x80 >> (i % 8)) != 0 {
                        255
                    } else {
                        0
                    }
                })
                .collect();
            let mut soft_expanded = vec![0u8; enc_bits];
            let mut soft_erasure = vec![0u8; enc_bits.div_ceil(8)];
            p.depuncture_soft(&soft_punctured, &mut soft_expanded, &mut soft_erasure)
                .unwrap();
            assert_eq!(
                soft_erasure, erasure,
                "hard and soft must produce the same erasure mask for the same block"
            );
            for i in 0..enc_bits {
                if p.keeps(i) {
                    let want = if encoded[i / 8] & (0x80 >> (i % 8)) != 0 {
                        255
                    } else {
                        0
                    };
                    assert_eq!(
                        soft_expanded[i], want,
                        "kept soft symbol {i} of {enc_bits} did not survive"
                    );
                } else {
                    assert_eq!(
                        soft_expanded[i], 128,
                        "punctured soft slot {i} of {enc_bits} should hold the neutral value"
                    );
                }
            }
        }

        assert!(
            unaligned_enc_seen > 0,
            "sweep never produced a non-byte-aligned encoded length"
        );
        assert!(
            unaligned_punc_seen > 0,
            "sweep never produced a non-byte-aligned punctured length"
        );
    }

    fn assert_punctured_decode_recovers(keep: &[bool], hard: bool) {
        let rate = 2;
        let order = 7;
        let polys = [0o155u16, 0o117];
        let p = Puncturer::new(keep).unwrap();
        let msg_len = 400;

        let mut rng = Rng::new(0xABCD);
        let mut msg = vec![0u8; msg_len];
        for b in &mut msg {
            *b = rng.next_u8();
        }

        let mut enc = Encoder::new(rate, order, &polys);
        let enc_bits = enc.encode_len(msg_len);
        let mut encoded = vec![0u8; enc_bits.div_ceil(8)];
        enc.encode(&msg, &mut encoded).unwrap();

        let mut dec = Decoder::new(rate, order, &polys);
        let mut out = vec![0u8; msg_len];
        let mut erasure = vec![0u8; enc_bits.div_ceil(8)];
        let punctured_len = p.punctured_len(enc_bits);

        let mut punctured = vec![0u8; punctured_len.div_ceil(8)];
        p.puncture(&encoded, enc_bits, &mut punctured).unwrap();

        if hard {
            let mut expanded = vec![0u8; enc_bits.div_ceil(8)];
            p.depuncture_hard(&punctured, enc_bits, &mut expanded, &mut erasure)
                .unwrap();
            dec.decode_hard_with_erasure(&expanded, enc_bits, &erasure, &mut out)
                .unwrap();
        } else {
            let mut soft = vec![0u8; punctured_len];
            for (i, s) in soft.iter_mut().enumerate() {
                *s = if punctured[i / 8] & (0x80 >> (i % 8)) != 0 {
                    255
                } else {
                    0
                };
            }

            let mut expanded = vec![0u8; enc_bits];
            p.depuncture_soft(&soft, &mut expanded, &mut erasure).unwrap();
            dec.decode_soft_with_erasure(&expanded, &erasure, &mut out).unwrap();
        }

        let mode = if hard { "hard" } else { "soft" };
        assert_eq!(out, msg, "punctured decode failed: keep={keep:?} mode={mode}");
    }

    #[test]
    fn punctured_decode_recovers_rate_3_4() {
        assert_punctured_decode_recovers(&[true, true, true, false], true);
        assert_punctured_decode_recovers(&[true, true, true, false], false);
    }

    #[test]
    fn punctured_decode_recovers_rate_2_3() {
        assert_punctured_decode_recovers(&[true, true, true, false, false, true], true);
        assert_punctured_decode_recovers(&[true, true, true, false, false, true], false);
    }

    #[test]
    fn punctured_decode_recovers_unpunctured() {
        assert_punctured_decode_recovers(&[true, true], true);
        assert_punctured_decode_recovers(&[true, true], false);
    }
}
