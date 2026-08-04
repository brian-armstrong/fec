use super::bit::{BitReader, BitWriter};
use super::error::{self, DecodeError};
use super::util;

use std::collections::HashMap;
use std::mem;

pub(crate) enum ConvolutionalError<'a> {
    Hard(BitReader<'a>),
    Soft(&'a [u8]),
}

impl<'a> ConvolutionalError<'a> {
    #[inline(always)]
    pub(crate) fn fill_next_distances(&mut self, distances: &mut [u16], rate: u32) {
        match self {
            ConvolutionalError::Hard(encoded) => {
                // peel off `rate` bits to recover the same `out` the encoder
                // produced, now with the channel's noise applied. the distance to
                // each possible output `i` is the Hamming distance to `out`.
                let outputs = encoded.read(rate as usize);

                for (i, distance) in distances.iter_mut().enumerate() {
                    *distance = util::metric_distance(i as u32, outputs.into()) as u16;
                }
            }
            ConvolutionalError::Soft(encoded) => {
                let outputs = encoded.iter().take(rate as usize);

                // linear soft distance
                // for each possible hard output `i`, sum the absolute difference
                // between each expected soft value (a 1 bit expects 255,
                // a 0 bit expects 0) and the received soft symbol.
                for (i, distance) in distances.iter_mut().enumerate() {
                    let mut dist: u16 = 0;
                    let mut polys = i;
                    for &output in outputs.clone() {
                        let expected = if polys & 1 != 0 { 255i16 } else { 0i16 };
                        polys >>= 1;
                        dist += (output as i16 - expected).unsigned_abs();
                    }
                    *distance = dist;
                }
                *encoded = &encoded[(rate as usize)..];
            }
        }
    }
}

/// A Viterbi decoder for a convolutional code, with hard- and soft-decision
/// decoding.
///
/// Construct it with the same `(rate, order, polynomials)` used to
/// [`Encoder::new`](super::Encoder::new). The decoder tolerates some corrupted
/// bits and recovers the original message, up to the error-correcting power of
/// the code.
///
/// It cannot tell you when *too many* bits were corrupted. Past the code's
/// correction limit it still writes a message, just not the right one. If you
/// need to know whether decoding succeeded, wrap the payload in a checksum or
/// CRC and verify it yourself.
///
/// This is the portable scalar decoder. For higher throughput on x86, enable
/// the `simd` feature and use the SIMD decoder. It decodes identically.
#[cfg_attr(feature = "simd", doc = "See [`SimdDecoder`](super::SimdDecoder).")]
#[derive(Debug)]
pub struct Decoder {
    rate: u32,
    order: u32,
    highbit: u16,
    poly_table: Vec<u16>,
    pair_table: ConvolutionalPairTable,
    history_table: ConvolutionalHistoryTable,
    error_table: ConvolutionalErrorTable,
    distances: Vec<u16>,
}

impl Decoder {
    /// Creates a decoder for the convolutional code with the given parameters.
    ///
    /// These must match the encoder. `polys` contains exactly `rate` generator
    /// polynomials. See [`Encoder::new`](super::Encoder::new) for the parameter
    /// convention.
    ///
    /// # Panics
    ///
    /// Panics if `polys.len()` is not equal to `rate`.
    pub fn new(rate: u32, order: u32, polys: &[u16]) -> Decoder {
        let poly_table = util::conv_poly_table(rate, order, polys);
        let max_error = rate * u8::MAX as u32;
        let renorm = (i16::MAX as u32 / max_error).max(1);
        let highbit = 1 << (order - 1);
        let traceback_group_length = util::traceback_group_length(order) as u32;
        Decoder {
            rate,
            order,
            highbit,
            pair_table: ConvolutionalPairTable::new(rate, order, &poly_table),
            history_table: ConvolutionalHistoryTable::new(
                5 * order,
                traceback_group_length,
                renorm,
                util::num_states_for_order(order) / 2,
                highbit,
            ),
            error_table: ConvolutionalErrorTable::new((util::num_states_for_order(order) / 2) as usize),
            poly_table,
            distances: vec![0; 1 << rate],
        }
    }

    /// first phase: load the shift register up from 0 (the register fills from
    /// 1 bit up to `order` bits) building the error metrics for the first bits.
    /// no output bits are produced here.
    fn decode_head(&mut self, distance_fill: &mut ConvolutionalError) {
        for i in 0..(self.order - 1) {
            distance_fill.fill_next_distances(&mut self.distances, self.rate);

            // walk only the states reachable so far
            let num_states = 1 << (i + 1);
            for (j, error) in self.error_table.errors[..num_states].iter_mut().enumerate() {
                let previous_state = j >> 1;
                let distance = self.distances[self.poly_table[j] as usize];
                *error = distance + self.error_table.previous_errors[previous_state];
            }
            self.error_table.swap();
        }
    }

    /// main phase: decode every bit except the first (head) and last (tail).
    ///
    /// this walks all `2^(order-1)` predecessor states, considering two paths per
    /// state (the high-order bit set or clear), and for each successor keeps the
    /// path with the least aggregated bit errors, recording the winner in the
    /// history buffer for later traceback.
    ///
    /// the inner loop computes 4 successor states per iteration, grouped as two
    /// pairs. the first pair differ only in the *lowest* bit: they share a *predecessor*,
    /// because their high `order - 1` bits match. the second pair differs only in
    /// the *highest* bit: they share a *successor*, because that oldest high bit shifts out.
    /// this pairing is what lets the pair table serve a concatenated distance for both
    /// at once.
    fn decode_body(
        &mut self,
        distance_fill: &mut ConvolutionalError,
        num_encoded_bits: usize,
        decoded: &mut BitWriter,
    ) {
        let num_decoded_bits: u32 = num_encoded_bits as u32 / self.rate;
        for _ in (self.order - 1)..(num_decoded_bits - self.order + 1) {
            distance_fill.fill_next_distances(&mut self.distances, self.rate);
            self.pair_table.distances(&self.distances);

            let highbit = self.highbit as usize;
            let high_prev_offset = highbit >> 1;

            unsafe {
                let pair_keys = self.pair_table.keys.as_ptr();
                let pair_distances = self.pair_table.distances.as_ptr();

                let previous_errors = self.error_table.previous_errors.as_ptr();
                let errors = self.error_table.errors.as_mut_ptr();

                let history = self.history_table.get_slice().as_mut_ptr();

                let state_iter = (0..highbit).step_by(8);
                let prev_state_iter = (0..high_prev_offset).step_by(4);

                for (state, prev_state) in state_iter.zip(prev_state_iter) {
                    for (state_offset, prev_offset) in (0..8).step_by(2).zip(0..4) {
                        // the two candidate predecessors are the low and high
                        // shift-register states. each carries its aggregate error
                        // from the previous time slice and a concatenated distance
                        // for both of its successors (packed low in bits 0..16,
                        // high in bits 16..32).
                        let low_key = *pair_keys.add(prev_state + prev_offset);
                        let high_key = *pair_keys.add(prev_state + prev_offset + high_prev_offset);

                        let low_concat_distance = *pair_distances.add(low_key as usize);
                        let high_concat_distance = *pair_distances.add(high_key as usize);

                        let low_prev_error = *previous_errors.add(prev_state + prev_offset);
                        let high_prev_error = *previous_errors.add(prev_state + prev_offset + high_prev_offset);

                        // even successor
                        let low_error0 = (low_concat_distance & 0xffff) as u16 + low_prev_error;
                        let high_error0 = (high_concat_distance & 0xffff) as u16 + high_prev_error;
                        let (error0, successor0) = if low_error0 <= high_error0 {
                            (low_error0, 0)
                        } else {
                            (high_error0, 1)
                        };
                        *errors.add(state + state_offset) = error0;
                        *history.add(state + state_offset) = successor0;

                        // odd successor
                        let low_error1 = (low_concat_distance >> 16) as u16 + low_prev_error;
                        let high_error1 = (high_concat_distance >> 16) as u16 + high_prev_error;
                        let (error1, successor1) = if low_error1 <= high_error1 {
                            (low_error1, 0)
                        } else {
                            (high_error1, 1)
                        };
                        *errors.add(state + 1 + state_offset) = error1;
                        *history.add(state + 1 + state_offset) = successor1;
                    }
                }
            }

            self.history_table.process(&mut self.error_table.errors, decoded);
            self.error_table.swap();
        }
    }

    /// tail phase: decode the last bits, flushing the state registers.
    ///
    /// The encoder drove the shift register back to 0, so here only 0s shift in
    /// and the 1-successors can be skipped. `step` doubles each iteration as more
    /// of the register is known to be zero, so fewer states remain live.
    fn decode_tail(
        &mut self,
        distance_fill: &mut ConvolutionalError,
        num_encoded_bits: usize,
        decoded: &mut BitWriter,
    ) {
        let num_decoded_bits: u32 = num_encoded_bits as u32 / self.rate;
        let highbit = self.highbit as usize;
        let high_prev_offset = highbit >> 1;
        for i in (num_decoded_bits - self.order + 1)..num_decoded_bits {
            distance_fill.fill_next_distances(&mut self.distances, self.rate);

            let step = 1 << (self.order - (num_decoded_bits - i));
            let previous_errors = &self.error_table.previous_errors;
            let errors = &mut self.error_table.errors;
            let history = self.history_table.get_slice();

            let state_iter = (0..highbit).step_by(step);
            let prev_state_iter = (0..high_prev_offset).step_by(step / 2);
            for (state, prev_state) in state_iter.zip(prev_state_iter) {
                let low_output = self.poly_table[state];
                let high_output = self.poly_table[state + highbit];

                let low_prev_error = previous_errors[prev_state];
                let high_prev_error = previous_errors[prev_state + high_prev_offset];

                let low_error = self.distances[low_output as usize] + low_prev_error;
                let high_error = self.distances[high_output as usize] + high_prev_error;
                let (error, successor) = if low_error <= high_error {
                    (low_error, 0)
                } else {
                    (high_error, 1)
                };
                errors[state] = error;
                history[state] = successor;
            }

            self.history_table
                .process_step(step as u32, &mut self.error_table.errors, decoded);
            self.error_table.swap();
        }
    }

    fn _decode(
        &mut self,
        distance_fill: &mut ConvolutionalError,
        num_encoded_bits: usize,
        decoded: &mut BitWriter,
    ) -> usize {
        self.error_table.reset();
        self.history_table.reset();

        // three phases: warm up the register from zero (no output), decode the
        // steady-state body, then flush the register back to zero.
        self.decode_head(distance_fill);
        self.decode_body(distance_fill, num_encoded_bits, decoded);
        self.decode_tail(distance_fill, num_encoded_bits, decoded);

        self.history_table.flush(decoded);

        decoded.len()
    }

    /// Decodes a hard-decision block produced by [`Encoder::encode`](super::Encoder::encode).
    ///
    /// `num_encoded_bits` is the length of `encoded` in *bits*, as returned by
    /// [`Encoder::encode`](super::Encoder::encode). It need not be a multiple of
    /// 8, but it must be a multiple of `rate`. `msg` must be large enough to
    /// hold the decoded payload.
    ///
    /// On success this returns the number of bytes written to `msg`. It returns
    /// [`InvalidLength`](DecodeError::InvalidLength) if the encoded length is not
    /// a multiple of `rate`, if the block is too short to decode, or if
    /// `encoded` is shorter than `num_encoded_bits` describes. It returns
    /// [`OutputTooSmall`](DecodeError::OutputTooSmall) if `msg` is too small to
    /// hold the payload.
    pub fn decode_hard(
        &mut self,
        encoded: &[u8],
        num_encoded_bits: usize,
        msg: &mut [u8],
    ) -> Result<usize, DecodeError> {
        error::validate_encoded_len(num_encoded_bits, self.rate, self.order)?;

        if num_encoded_bits.div_ceil(8) > encoded.len() {
            return Err(DecodeError::InvalidLength {
                num_encoded_bits,
                rate: self.rate,
            });
        }
        let needed = error::payload_len_bytes(num_encoded_bits, self.rate, self.order);
        if msg.len() < needed {
            return Err(DecodeError::OutputTooSmall {
                needed,
                actual: msg.len(),
            });
        }

        let bit_reader = BitReader::new(encoded);
        let mut bit_writer = BitWriter::new(msg);
        let mut distance_fill = ConvolutionalError::Hard(bit_reader);
        Ok(self._decode(&mut distance_fill, num_encoded_bits, &mut bit_writer))
    }

    /// Decodes a soft-decision block into `msg`.
    ///
    /// Each element of `encoded` is one 8-bit soft symbol, one symbol per
    /// encoded bit. `255` is a confident `1`, `0` is a confident `0`, and `128`
    /// is a fully-erased symbol carrying no information. Values in between
    /// express confidence, so a demodulator that reports its certainty lets the
    /// decoder correct more errors than hard decisions alone.
    ///
    /// The encoded length in bits is `encoded.len()`, which must be a multiple
    /// of `rate`. `msg` must be large enough to hold the decoded payload.
    ///
    /// On success this returns the number of bytes written to `msg`. It returns
    /// [`InvalidLength`](DecodeError::InvalidLength) if `encoded.len()` is not a
    /// multiple of `rate`, or if the block is too short to decode. It returns
    /// [`OutputTooSmall`](DecodeError::OutputTooSmall) if `msg` is too small to
    /// hold the payload.
    pub fn decode_soft(&mut self, encoded: &[u8], msg: &mut [u8]) -> Result<usize, DecodeError> {
        let num_encoded_bits = encoded.len();

        error::validate_encoded_len(num_encoded_bits, self.rate, self.order)?;

        let needed = error::payload_len_bytes(num_encoded_bits, self.rate, self.order);
        if msg.len() < needed {
            return Err(DecodeError::OutputTooSmall {
                needed,
                actual: msg.len(),
            });
        }

        let mut bit_writer = BitWriter::new(msg);
        let mut distance_fill = ConvolutionalError::Soft(encoded);
        Ok(self._decode(&mut distance_fill, num_encoded_bits, &mut bit_writer))
    }
}

#[derive(Debug)]
pub(super) struct ConvolutionalErrorTable {
    pub(super) errors: Vec<u16>,
    pub(super) previous_errors: Vec<u16>,
}

impl ConvolutionalErrorTable {
    pub fn new(num_states: usize) -> ConvolutionalErrorTable {
        ConvolutionalErrorTable {
            errors: vec![0; num_states],
            previous_errors: vec![0; num_states],
        }
    }

    pub fn swap(&mut self) {
        mem::swap(&mut self.errors, &mut self.previous_errors);
    }

    pub fn reset(&mut self) {
        self.errors.fill(0);
        self.previous_errors.fill(0);
    }
}

#[derive(Debug)]
pub(super) struct ConvolutionalHistoryTable {
    min_traceback_length: u32,
    num_states: u32,
    highbit: u16,
    history: Vec<u8>,
    decode_buf: Vec<u8>,
    history_index: usize,
    history_len: usize,
    history_cap: usize,
    renormalize_interval: u32,
    renormalize_counter: u32,
}

impl ConvolutionalHistoryTable {
    pub fn new(
        min_traceback_length: u32,
        traceback_group_length: u32,
        renormalize_interval: u32,
        num_states: u32,
        highbit: u16,
    ) -> ConvolutionalHistoryTable {
        let cap = min_traceback_length + traceback_group_length;
        ConvolutionalHistoryTable {
            min_traceback_length,
            num_states,
            highbit,
            history: vec![0; num_states as usize * cap as usize],
            decode_buf: vec![0; cap as usize],
            history_index: 0,
            history_len: 0,
            history_cap: cap as usize,
            renormalize_interval,
            renormalize_counter: 0,
        }
    }

    pub fn get_slice(&mut self) -> &mut [u8] {
        &mut self.history
            [(self.history_index * self.num_states as usize)..((self.history_index + 1) * self.num_states as usize)]
    }

    /// find the shift-register state with the least accumulated error.
    fn least_error_path(&self, distances: &[u16], search_every: u32) -> u16 {
        let step = search_every as usize;
        let mut best_i = 0u16;
        let mut best = distances[0];
        let mut i = step;
        while i < distances.len() {
            let d = distances[i];
            if d < best {
                best = d;
                best_i = i as u16;
            }
            i += step;
        }
        best_i
    }

    /// subtract the minimum error from every state so the metrics can't overflow
    /// their 16-bit width as they accumulate across the message.
    fn renormalize(&self, distances: &mut [u16], least_register: u16) {
        let min_distance = distances[least_register as usize];
        for distance in distances.iter_mut() {
            *distance = (*distance as i16 - min_distance as i16) as u16;
        }
    }

    pub fn traceback(&mut self, init_best_path: u16, min_traceback_length: u32, bit_writer: &mut BitWriter) {
        let mut index = self.history_index;
        let mut best_path = init_best_path;

        // loop 1 - rewind the history table without collecting any bits
        // these most-recent bits are still converging
        //
        // walking backwards through the recorded winners, each step tells us the
        // high-order bit of the predecessor state. shift in that bit from the top and
        // shift right to recover the state one time slice earlier.
        for _ in 0..min_traceback_length {
            index = if index == 0 { self.history_cap - 1 } else { index - 1 };

            let bit = self.history[index * self.num_states as usize + best_path as usize];
            let reg_bit = if bit == 0 { 0 } else { self.highbit };
            best_path = (best_path | reg_bit) >> 1;
        }

        // loop 2 - keep rewinding and collect the decoded bits
        let num_decodes = self.history_len - min_traceback_length as usize;
        for decoded in self.decode_buf.iter_mut().take(num_decodes) {
            index = if index == 0 { self.history_cap - 1 } else { index - 1 };

            let bit = self.history[index * self.num_states as usize + best_path as usize];
            let (reg_bit, decoded_bit) = if bit == 0 { (0, 0) } else { (self.highbit, 1) };
            *decoded = decoded_bit;
            best_path = (best_path | reg_bit) >> 1;
        }

        bit_writer.write_iter(self.decode_buf[..num_decodes].iter().rev());
        self.history_len -= num_decodes;
    }

    pub fn process_step(&mut self, step: u32, distances: &mut [u16], bit_writer: &mut BitWriter) {
        self.history_index += 1;
        if self.history_index == self.history_cap {
            self.history_index = 0;
        }

        self.renormalize_counter += 1;
        self.history_len += 1;

        // four ways this resolves: (a) neither renormalize nor traceback,
        // (b) renormalize only, (c) both, (d) traceback only. in case (c) the
        // search for the best path is expensive, so we reuse the one found while
        // renormalizing rather than searching twice.
        if self.renormalize_counter == self.renormalize_interval {
            self.renormalize_counter = 0;
            let best_path = self.least_error_path(distances, step);
            self.renormalize(distances, best_path);
            if self.history_len == self.history_cap {
                // reuse the best path found for renormalizing
                let min_traceback_length = self.min_traceback_length;
                self.traceback(best_path, min_traceback_length, bit_writer);
            }
        } else if self.history_len == self.history_cap {
            // not renormalizing, so find the best path here
            let best_path = self.least_error_path(distances, step);
            let min_traceback_length = self.min_traceback_length;
            self.traceback(best_path, min_traceback_length, bit_writer);
        }
    }

    pub fn process(&mut self, distances: &mut [u16], bit_writer: &mut BitWriter) {
        self.process_step(1, distances, bit_writer)
    }

    pub fn flush(&mut self, bit_writer: &mut BitWriter) {
        self.traceback(0, 0, bit_writer)
    }

    pub fn reset(&mut self) {
        self.history_len = 0;
        self.history_index = 0;
        self.renormalize_counter = 0;
    }
}

#[derive(Debug)]
/// Represent convolutional distance metrics for a pair of shift register states
struct ConvolutionalPairTable {
    keys: Vec<u32>,
    outputs: Vec<u32>,
    distances: Vec<u32>,
    output_mask: u32,
    output_width: u32,
}

impl ConvolutionalPairTable {
    pub fn new(rate: u32, order: u32, poly_table: &[u16]) -> ConvolutionalPairTable {
        let num_pairs = util::num_states_for_order(order) / 2;
        let mut keys = vec![0u32; num_pairs as usize];

        let mut outputs: Vec<u32> = Vec::new();
        let mut outputs_lookup: HashMap<u32, u32> = HashMap::new();

        // for each even-numbered shift-register state, form the concatenated
        // output of that state and the subsequent state (low bit set). Many
        // states share the same concatenated output, so intern each distinct one
        // under a compact key and store that key for the state. the inner loop
        // then indexes distances by key instead of by full output.
        for (pairs, key) in poly_table.chunks(2).zip(&mut keys) {
            let output: u32 = ((pairs[1] as u32) << rate) | pairs[0] as u32;
            let next_idx = outputs.len() as u32;
            *key = *outputs_lookup.entry(output).or_insert_with(|| {
                outputs.push(output);
                next_idx
            });
        }

        ConvolutionalPairTable {
            keys,
            distances: vec![0u32; outputs.len()],
            outputs,
            output_mask: (1 << rate) - 1,
            output_width: rate,
        }
    }

    pub fn distances(&mut self, distances: &[u16]) -> &[u32] {
        // pack the two per-output distances of each interned pair into one u32.
        // the first output's distance lives in the low 16 bits and the second's
        // in the high 16 bits. the inner loop reads both successors in one load.
        for (distance, pair) in self.distances.iter_mut().zip(&self.outputs) {
            let first: u32 = pair & self.output_mask;
            let second: u32 = pair >> self.output_width;

            *distance = ((distances[second as usize]) as u32) << 16 | distances[first as usize] as u32;
        }
        &self.distances
    }
}

#[cfg(test)]
mod tests {
    use super::Decoder;
    use crate::convolutional::sim::{bpsk_params, flip_with_interval, Testbench};
    use crate::convolutional::{DecodeError, Encoder};
    use crate::util::Rng;

    #[test]
    fn rejects_malformed_inputs() {
        let mut d = Decoder::new(2, 7, &[0o155, 0o117]);
        let mut out = vec![0u8; 64];
        // not a multiple of rate
        assert!(matches!(
            d.decode_hard(&[0u8; 16], 7, &mut out),
            Err(DecodeError::InvalidLength { .. })
        ));
        // encoded shorter than num_encoded_bits describes
        assert!(matches!(
            d.decode_hard(&[0u8; 2], 64, &mut out),
            Err(DecodeError::InvalidLength { .. })
        ));
        // too short to decode (would underflow the body loop bound)
        assert!(matches!(
            d.decode_hard(&[0u8; 4], 8, &mut out),
            Err(DecodeError::InvalidLength { .. })
        ));
        assert!(matches!(
            d.decode_soft(&[0u8; 4], &mut out),
            Err(DecodeError::InvalidLength { .. })
        ));
        // output buffer too small
        let encoded = vec![0u8; 200];
        assert!(matches!(
            d.decode_soft(&encoded, &mut [0u8; 1]),
            Err(DecodeError::OutputTooSmall { .. })
        ));
    }

    fn decode_matches_msg<const RATE: u32, const ORDER: u32>(
        polys: &[u16],
        msg_len: usize,
        seed: u64,
        hard: bool,
        clean: bool,
    ) {
        let mut rng = Rng::new(seed);
        let mut msg = vec![0u8; msg_len];
        for b in &mut msg {
            *b = rng.next_u8();
        }

        let mut enc = Encoder::new(RATE, ORDER, polys);
        let enc_bits = enc.encode_len(msg_len);
        let mut encoded = vec![0u8; enc_bits.div_ceil(8)];
        enc.encode(&msg, &mut encoded).unwrap();

        if !clean {
            flip_with_interval(&mut encoded, enc_bits, RATE, ORDER, &mut rng);
        }

        let mut out = vec![0u8; msg_len];
        let mut dec = Decoder::new(RATE, ORDER, polys);
        if hard {
            dec.decode_hard(&encoded, enc_bits, &mut out).unwrap();
        } else {
            let mut soft = vec![0u8; enc_bits];
            for (i, s) in soft.iter_mut().enumerate() {
                *s = if encoded[i / 8] & (0x80 >> (i % 8)) != 0 {
                    255
                } else {
                    0
                };
            }
            dec.decode_soft(&soft, &mut out).unwrap();
        }

        let mode = if hard { "hard" } else { "soft" };
        assert_eq!(
            &out, &msg,
            "scalar decode wrong: rate={RATE} order={ORDER} len={msg_len} mode={mode} \
             clean={clean} seed={seed}"
        );
    }

    fn test_rate_order<const RATE: u32, const ORDER: u32>(polys: &[u16]) {
        for clean in [false, true] {
            for hard in [false, true] {
                for (msg_len, seeds) in [(256usize, 16u64), (1500usize, 8u64)] {
                    for seed in 1..=seeds {
                        decode_matches_msg::<RATE, ORDER>(polys, msg_len, seed, hard, clean);
                    }
                }
            }
        }
    }

    #[test]
    fn decoder_matches_msg_2_4() {
        test_rate_order::<2, 4>(&[0o017, 0o013]);
    }
    #[test]
    fn decoder_matches_msg_2_5() {
        test_rate_order::<2, 5>(&[0o027, 0o023]);
    }
    #[test]
    fn decoder_matches_msg_2_6() {
        test_rate_order::<2, 6>(&[0o065, 0o057]);
    }
    #[test]
    fn decoder_matches_msg_2_7() {
        test_rate_order::<2, 7>(&[0o155, 0o117]);
    }
    #[test]
    fn decoder_matches_msg_2_8() {
        test_rate_order::<2, 8>(&[0o367, 0o225]);
    }
    #[test]
    fn decoder_matches_msg_2_9() {
        test_rate_order::<2, 9>(&[0o657, 0o435]);
    }
    #[test]
    fn decoder_matches_msg_2_10() {
        test_rate_order::<2, 10>(&[0o1627, 0o1063]);
    }
    #[test]
    fn decoder_matches_msg_3_9() {
        test_rate_order::<3, 9>(&[0o755, 0o633, 0o447]);
    }
    #[test]
    fn decoder_matches_msg_4_7() {
        test_rate_order::<4, 7>(&[0o133, 0o175, 0o107, 0o101]);
    }
    #[test]
    fn decoder_matches_msg_6_15() {
        test_rate_order::<6, 15>(&[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537]);
    }
    #[test]
    fn decoder_matches_msg_7_7() {
        test_rate_order::<7, 7>(&[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145]);
    }
    #[test]
    fn decoder_matches_msg_8_7() {
        test_rate_order::<8, 7>(&[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145, 0o171]);
    }

    fn assert_ber_threshold(
        rate: u32,
        order: u32,
        polys: &[u16],
        eb_n0_db: f64,
        min_bytes: usize,
        max_ber: Option<f64>,
        hard: bool,
    ) {
        const BLOCK: usize = 4096;
        const MIN_ERRS: usize = 1000;
        let (volt, bit_energy) = bpsk_params(rate);

        let mut msg = vec![0u8; BLOCK];
        let mut bench = Testbench::new(rate, order, polys, BLOCK);
        let mut rng = Rng::new(0x1234_5678_9ABC_DEF0);
        let mut scalar = Decoder::new(rate, order, polys);
        let mut errs = 0usize;
        let mut uncoded_flips = 0usize;
        let mut channel_bits = 0usize;
        let mut sent = 0usize;

        while sent < min_bytes {
            for b in msg.iter_mut() {
                *b = rng.next_u8();
            }
            bench.build_noise(eb_n0_db, bit_energy, &mut rng);
            let (err, uncoded_err) = if hard {
                bench.test_decoder_with_noise_hard(&mut scalar, &msg, volt)
            } else {
                bench.test_decoder_with_noise(&mut scalar, &msg, volt)
            };
            errs += err;
            uncoded_flips += uncoded_err;
            channel_bits += bench.enc_bits();
            sent += BLOCK;
        }

        assert!(
            errs >= MIN_ERRS,
            "{rate}/{order}: only {errs} errors at {eb_n0_db}dB over {sent}B. Minimum of \
             {MIN_ERRS} error events required. Lower Eb/N0 or raise min_bytes."
        );

        let coded_bits = sent * 8;
        let ber = errs as f64 / coded_bits as f64;

        // do an absolute ber comparison, either to target or a 4x-uncoded floor
        match max_ber {
            Some(target) => {
                assert!(
                    ber <= target,
                    "{rate}/{order}: BER {ber:.2e} exceeds reference {target:.2e} \
                     at {eb_n0_db}dB ({errs} errors over {sent}B)."
                );
            }
            None => {
                const MIN_GAIN: f64 = 4.0;
                let uncoded_ber = uncoded_flips as f64 / channel_bits as f64;
                let gain = uncoded_ber / ber;
                assert!(
                    gain >= MIN_GAIN,
                    "{rate}/{order}: coding gain {gain:.1}x < {MIN_GAIN}x at {eb_n0_db}dB \
                     (uncoded BER {uncoded_ber:.2e}, coded BER {ber:.2e})."
                );
            }
        }
    }

    const R2K7_2_5_REF_BER: f64 = 2.5e-3;
    const R2K7_2_0_REF_BER: f64 = 9.0e-3;
    const R2K7_1_5_REF_BER: f64 = 2.5e-2;
    const R2K7_4_5_HARD_REF_BER: f64 = 2.0e-3;
    const R2K7_4_0_HARD_REF_BER: f64 = 6.0e-3;
    const R2K7_3_5_HARD_REF_BER: f64 = 2.0e-2;
    const R2K9_2_0_REF_BER: f64 = 4.5e-3;
    const R2K9_1_5_REF_BER: f64 = 2.0e-2;
    const R2K9_4_0_HARD_REF_BER: f64 = 2.5e-3;
    const R2K9_3_5_HARD_REF_BER: f64 = 9.0e-3;
    const R3K9_2_0_REF_BER: f64 = 3.0e-3;
    const R3K9_1_5_REF_BER: f64 = 9.0e-3;
    const R3K9_1_0_REF_BER: f64 = 3.0e-2;
    const R3K9_3_5_HARD_REF_BER: f64 = 3.0e-3;
    const R3K9_3_0_HARD_REF_BER: f64 = 1.0e-2;
    const R3K9_2_5_HARD_REF_BER: f64 = 3.0e-2;
    const R6K15_1_5_REF_BER: f64 = 1.5e-3;
    const R6K15_1_0_REF_BER: f64 = 8.0e-3;
    const R6K15_2_5_HARD_REF_BER: f64 = 3.0e-3;
    const R6K15_2_0_HARD_REF_BER: f64 = 1.5e-2;

    #[test]
    fn ber_threshold_2_5() {
        assert_ber_threshold(2, 5, &[0o027, 0o023], 3.0, 300_000, None, false);
        assert_ber_threshold(2, 5, &[0o027, 0o023], 2.5, 300_000, None, false);
        assert_ber_threshold(2, 5, &[0o027, 0o023], 2.0, 300_000, None, false);
    }

    #[test]
    fn ber_threshold_2_6() {
        assert_ber_threshold(2, 6, &[0o065, 0o057], 3.0, 300_000, None, false);
        assert_ber_threshold(2, 6, &[0o065, 0o057], 2.5, 300_000, None, false);
        assert_ber_threshold(2, 6, &[0o065, 0o057], 2.0, 300_000, None, false);
    }

    #[test]
    fn ber_threshold_2_7() {
        assert_ber_threshold(2, 7, &[0o155, 0o117], 2.5, 300_000, Some(R2K7_2_5_REF_BER), false);
        assert_ber_threshold(2, 7, &[0o155, 0o117], 2.0, 300_000, Some(R2K7_2_0_REF_BER), false);
        assert_ber_threshold(2, 7, &[0o155, 0o117], 1.5, 300_000, Some(R2K7_1_5_REF_BER), false);
    }

    #[test]
    fn ber_threshold_hard_2_7() {
        assert_ber_threshold(2, 7, &[0o155, 0o117], 4.5, 300_000, Some(R2K7_4_5_HARD_REF_BER), true);
        assert_ber_threshold(2, 7, &[0o155, 0o117], 4.0, 300_000, Some(R2K7_4_0_HARD_REF_BER), true);
        assert_ber_threshold(2, 7, &[0o155, 0o117], 3.5, 300_000, Some(R2K7_3_5_HARD_REF_BER), true);
    }

    #[test]
    fn ber_threshold_2_8() {
        assert_ber_threshold(2, 8, &[0o367, 0o225], 2.5, 200_000, None, false);
        assert_ber_threshold(2, 8, &[0o367, 0o225], 2.0, 200_000, None, false);
        assert_ber_threshold(2, 8, &[0o367, 0o225], 1.5, 200_000, None, false);
    }

    #[test]
    fn ber_threshold_2_9() {
        assert_ber_threshold(2, 9, &[0o657, 0o435], 2.0, 100_000, Some(R2K9_2_0_REF_BER), false);
        assert_ber_threshold(2, 9, &[0o657, 0o435], 1.5, 100_000, Some(R2K9_1_5_REF_BER), false);
    }

    #[test]
    fn ber_threshold_hard_2_9() {
        assert_ber_threshold(2, 9, &[0o657, 0o435], 4.0, 100_000, Some(R2K9_4_0_HARD_REF_BER), true);
        assert_ber_threshold(2, 9, &[0o657, 0o435], 3.5, 100_000, Some(R2K9_3_5_HARD_REF_BER), true);
    }

    #[test]
    fn ber_threshold_2_15() {
        assert_ber_threshold(2, 15, &[0o56711, 0o75063], 1.2, 10_000, None, false);
    }

    #[test]
    fn ber_threshold_3_7() {
        assert_ber_threshold(3, 7, &[0o175, 0o145, 0o133], 2.5, 300_000, None, false);
        assert_ber_threshold(3, 7, &[0o175, 0o145, 0o133], 2.0, 300_000, None, false);
        assert_ber_threshold(3, 7, &[0o175, 0o145, 0o133], 1.5, 300_000, None, false);
    }

    #[test]
    fn ber_threshold_3_9() {
        assert_ber_threshold(
            3,
            9,
            &[0o755, 0o633, 0o447],
            2.0,
            100_000,
            Some(R3K9_2_0_REF_BER),
            false,
        );
        assert_ber_threshold(
            3,
            9,
            &[0o755, 0o633, 0o447],
            1.5,
            100_000,
            Some(R3K9_1_5_REF_BER),
            false,
        );
        assert_ber_threshold(
            3,
            9,
            &[0o755, 0o633, 0o447],
            1.0,
            100_000,
            Some(R3K9_1_0_REF_BER),
            false,
        );
    }

    #[test]
    fn ber_threshold_hard_3_9() {
        assert_ber_threshold(
            3,
            9,
            &[0o755, 0o633, 0o447],
            3.5,
            100_000,
            Some(R3K9_3_5_HARD_REF_BER),
            true,
        );
        assert_ber_threshold(
            3,
            9,
            &[0o755, 0o633, 0o447],
            3.0,
            100_000,
            Some(R3K9_3_0_HARD_REF_BER),
            true,
        );
        assert_ber_threshold(
            3,
            9,
            &[0o755, 0o633, 0o447],
            2.5,
            100_000,
            Some(R3K9_2_5_HARD_REF_BER),
            true,
        );
    }

    #[test]
    fn ber_threshold_4_7() {
        assert_ber_threshold(4, 7, &[0o133, 0o175, 0o107, 0o101], 2.0, 200_000, None, false);
        assert_ber_threshold(4, 7, &[0o133, 0o175, 0o107, 0o101], 1.5, 200_000, None, false);
        assert_ber_threshold(4, 7, &[0o133, 0o175, 0o107, 0o101], 1.0, 200_000, None, false);
    }

    #[test]
    fn ber_threshold_5_7() {
        assert_ber_threshold(5, 7, &[0o175, 0o145, 0o133, 0o117, 0o127], 2.0, 200_000, None, false);
        assert_ber_threshold(5, 7, &[0o175, 0o145, 0o133, 0o117, 0o127], 1.5, 200_000, None, false);
        assert_ber_threshold(5, 7, &[0o175, 0o145, 0o133, 0o117, 0o127], 1.0, 200_000, None, false);
    }

    #[test]
    fn ber_threshold_6_7() {
        // roku nana! seis siete!
        assert_ber_threshold(
            6,
            7,
            &[0o111, 0o127, 0o133, 0o167, 0o173, 0o175],
            2.0,
            100_000,
            None,
            false,
        );
        assert_ber_threshold(
            6,
            7,
            &[0o111, 0o127, 0o133, 0o167, 0o173, 0o175],
            1.5,
            100_000,
            None,
            false,
        );
        assert_ber_threshold(
            6,
            7,
            &[0o111, 0o127, 0o133, 0o167, 0o173, 0o175],
            1.0,
            100_000,
            None,
            false,
        );
    }

    #[test]
    fn ber_threshold_6_15() {
        assert_ber_threshold(
            6,
            15,
            &[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537],
            0.7,
            10_000,
            None,
            false,
        );
        assert_ber_threshold(
            6,
            15,
            &[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537],
            0.5,
            10_000,
            None,
            false,
        );
    }

    #[test]
    #[ignore = "slow: rate 1/6 k=15 threshold"]
    fn ber_threshold_ref_6_15() {
        assert_ber_threshold(
            6,
            15,
            &[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537],
            1.5,
            200_000,
            Some(R6K15_1_5_REF_BER),
            false,
        );
        assert_ber_threshold(
            6,
            15,
            &[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537],
            1.0,
            50_000,
            Some(R6K15_1_0_REF_BER),
            false,
        );
    }

    #[test]
    fn ber_threshold_hard_6_15() {
        assert_ber_threshold(
            6,
            15,
            &[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537],
            1.7,
            10_000,
            None,
            true,
        );
        assert_ber_threshold(
            6,
            15,
            &[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537],
            1.5,
            10_000,
            None,
            true,
        );
    }

    #[test]
    #[ignore = "slow: rate 1/6 k=15 hard threshold"]
    fn ber_threshold_hard_ref_6_15() {
        assert_ber_threshold(
            6,
            15,
            &[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537],
            2.5,
            200_000,
            Some(R6K15_2_5_HARD_REF_BER),
            true,
        );
        assert_ber_threshold(
            6,
            15,
            &[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537],
            2.0,
            50_000,
            Some(R6K15_2_0_HARD_REF_BER),
            true,
        );
    }

    #[test]
    fn ber_threshold_7_7() {
        assert_ber_threshold(
            7,
            7,
            &[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145],
            2.0,
            100_000,
            None,
            false,
        );
        assert_ber_threshold(
            7,
            7,
            &[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145],
            1.5,
            100_000,
            None,
            false,
        );
        assert_ber_threshold(
            7,
            7,
            &[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145],
            1.0,
            100_000,
            None,
            false,
        );
    }

    #[test]
    fn ber_threshold_8_7() {
        assert_ber_threshold(
            8,
            7,
            &[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145, 0o171],
            2.0,
            100_000,
            None,
            false,
        );
        assert_ber_threshold(
            8,
            7,
            &[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145, 0o171],
            1.5,
            100_000,
            None,
            false,
        );
        assert_ber_threshold(
            8,
            7,
            &[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145, 0o171],
            1.0,
            100_000,
            None,
            false,
        );
    }
}
