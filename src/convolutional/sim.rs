//! A channel and bit-error-rate harness for the convolutional decoders.
//!
//! This module simulates a link. It encodes a message, maps the bits to BPSK
//! symbols, adds Gaussian noise for a target Eb/N0, and decodes the result. It
//! exists to measure and test the decoders, so most callers are the test suite
//! and the tuning binaries rather than users of the codec.
//!
//! [`measure_ber`] runs a full sweep and returns the bit-error count for a code
//! at a given Eb/N0. [`Testbench`] drives one block at a time and is the piece
//! the higher-level helpers build on. [`sigma_for_eb_n0`] and [`bpsk_params`]
//! convert between Eb/N0 in decibels and the noise and signal levels the
//! channel uses. [`Rng`] is the deterministic RNG the noise is drawn from, so
//! every run with a given seed is reproducible.

#[cfg(feature = "simd")]
use super::SimdDecoder;
use super::{Decoder, Encoder};

/// A deterministic SplitMix64 RNG. Re-exported here because the public `sim`
/// functions ([`gaussian`], [`flip_with_interval`]) take it by reference.
pub use crate::util::Rng;

/// Flips one random bit in each `5 * order * rate`-bit window of `encoded`.
///
/// The window is the minimum traceback length in encoded bits, so at most one
/// bit is corrupted before the trellis reconverges. This produces isolated,
/// correctable errors for the bit-exactness tests. Returns the number of bits
/// flipped.
pub fn flip_with_interval(encoded: &mut [u8], enc_bits: usize, rate: u32, order: u32, rng: &mut Rng) -> usize {
    // use 5 * order (same as the minimum traceback length) so that we never flip
    //    more than one bit before the trellis converges
    // since that measures decoded bits, multiply by rate to get encoded bits
    let interval_len = 5 * order as usize * rate as usize;
    let mut flips = 0;
    let mut base = 0;
    while base + interval_len <= enc_bits {
        // flip one random bit somewhere in the next `interval_len` bits
        let bit = base + (rng.next_u64() as usize) % interval_len;
        encoded[bit / 8] ^= 0x80 >> (bit % 8);
        flips += 1;
        base += interval_len;
    }
    flips
}

/// Fills `out` with Gaussian (AWGN) samples of standard deviation `sigma`.
///
/// The samples come from the Box-Muller transform, drawn from `rng` so a given
/// seed always produces the same noise.
pub fn gaussian(out: &mut [f64], sigma: f64, rng: &mut Rng) {
    // standard gaussian/AWGN generator using the Box-Muller transform
    let mut i = 0;
    while i < out.len() {
        let (u, v, s) = loop {
            let u = 2.0 * rng.next_f64() - 1.0;
            let v = 2.0 * rng.next_f64() - 1.0;
            let s = u * u + v * v;
            if s > f64::EPSILON && s < 1.0 {
                break (u, v, s);
            }
        };
        let base = ((-2.0 * s.ln()) / s).sqrt();
        out[i] = u * base * sigma;
        if i + 1 < out.len() {
            out[i + 1] = v * base * sigma;
        }
        i += 2;
    }
}

fn log2amp(l: f64) -> f64 {
    10f64.powf(l / 10.0)
}

#[allow(dead_code)]
fn amp2log(a: f64) -> f64 {
    10.0 * a.log10()
}

/// Converts a target Eb/N0 in decibels to the noise standard deviation the
/// channel should use, given the BPSK bit energy from [`bpsk_params`].
pub fn sigma_for_eb_n0(eb_n0_db: f64, bpsk_bit_energy: f64) -> f64 {
    let eb_n0_amp = log2amp(eb_n0_db);
    (bpsk_bit_energy / (2.0 * eb_n0_amp)).sqrt()
}

fn encode_bpsk(bytes: &[u8], voltages: &mut [f64], n_syms: usize, bpsk_voltage: f64) {
    for i in 0..n_syms {
        let bit = bytes[i / 8] & (0x80 >> (i % 8)) != 0;
        voltages[i] = if bit { bpsk_voltage } else { -bpsk_voltage };
    }
}

fn add_white_noise(signal: &mut [f64], noise: &[f64]) {
    let inv_sqrt2 = 1.0 / 2f64.sqrt();
    for (signal_i, noise_i) in signal.iter_mut().zip(noise) {
        *signal_i += noise_i * inv_sqrt2;
    }
}

fn decode_bpsk_soft(voltages: &[f64], soft: &mut [u8], bpsk_voltage: f64) {
    for (v, s) in voltages.iter().zip(soft.iter_mut()) {
        let rel = v / bpsk_voltage;
        *s = if rel > 1.0 {
            255
        } else if rel < -1.0 {
            0
        } else {
            (127.5 + 127.5 * rel) as u8
        };
    }
}

/// Counts the bits that differ between `a` and `b`, their Hamming distance.
pub fn bit_distance(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones() as usize).sum()
}

/// Returns the BPSK signal parameters for a code of the given `rate`.
///
/// The first value is the per-symbol voltage. The second is the energy per
/// information bit, which [`sigma_for_eb_n0`] needs to turn a target Eb/N0 into
/// a noise level.
pub fn bpsk_params(rate: u32) -> (f64, f64) {
    let bpsk_voltage = 1.0 / 2f64.sqrt();
    let bpsk_sym_energy = 2.0 * bpsk_voltage * bpsk_voltage; // == 1.0
    let bpsk_bit_energy = bpsk_sym_energy * rate as f64;
    (bpsk_voltage, bpsk_bit_energy)
}

/// Decodes `total_bytes` of random messages through the channel at `eb_n0_db`
/// and returns the total number of bit errors that survive decoding.
///
/// Messages are sent `block` bytes at a time. Divide the result by
/// `total_bytes * 8` to get the bit error rate.
pub fn measure_ber(
    rate: u32,
    order: u32,
    polys: &[u16],
    eb_n0_db: f64,
    total_bytes: usize,
    block: usize,
    seed: u64,
) -> usize {
    let (bpsk_voltage, bpsk_bit_energy) = bpsk_params(rate);
    let mut decoder = Decoder::new(rate, order, polys);
    let mut bench = Testbench::new(rate, order, polys, block);
    let mut rng = Rng::new(seed);

    let mut errors = 0usize;
    let mut sent = 0usize;
    let mut msg = vec![0u8; block];
    while sent < total_bytes {
        for b in msg.iter_mut() {
            *b = rng.next_u8();
        }
        bench.build_noise(eb_n0_db, bpsk_bit_energy, &mut rng);
        errors += bench.test_decoder_with_noise(&mut decoder, &msg, bpsk_voltage).0;
        sent += block;
    }
    errors
}

/// One precomputed noisy block, ready to feed a decoder repeatedly.
///
/// Built by [`Testbench::create_test_cases`]. Holding the message and its
/// soft and hard channel outputs together lets a benchmark decode the same
/// block many times without regenerating noise.
pub struct TestCase {
    /// The received soft symbols, one 8-bit sample per encoded bit.
    pub soft: Vec<u8>,
    /// The received bits after a hard decision, bit-packed.
    pub hard: Vec<u8>,
    /// The original message that was encoded.
    pub msg: Vec<u8>,
    /// The number of encoded bits in the block.
    pub enc_bits: usize,
}

/// A reusable channel for one code and message length.
///
/// It owns the scratch buffers for encoding, adding noise, and demodulating, so
/// repeated runs at the same shape avoid reallocating. Build it with
/// [`new`](Self::new), then drive it with [`build_noise`](Self::build_noise) and
/// the `test_decoder_with_noise` methods.
pub struct Testbench {
    encoder: Encoder,
    voltages: Vec<f64>,
    encoded: Vec<u8>,
    noise: Vec<f64>,
    channel: Vec<f64>,
    msg_len: usize,
    enclen_bits: usize,
}

impl Testbench {
    /// Builds a testbench for the given code and message length in bytes.
    pub fn new(rate: u32, order: u32, polys: &[u16], msg_len: usize) -> Testbench {
        let encoder = Encoder::new(rate, order, polys);
        let enclen_bits = encoder.encode_len(msg_len);
        let enclen_bytes = enclen_bits / 8 + 1;
        Testbench {
            encoder,
            voltages: vec![0.0; enclen_bits],
            encoded: vec![0u8; enclen_bytes],
            noise: vec![0.0; enclen_bits],
            channel: vec![0.0; enclen_bits],
            msg_len,
            enclen_bits,
        }
    }

    /// Returns the encoded length in bits for this code and message length.
    pub fn enc_bits(&self) -> usize {
        self.enclen_bits
    }

    /// Draws a fresh block of channel noise for the target `eb_n0_db`.
    ///
    /// The noise is held internally and applied by the next
    /// `test_decoder_with_noise` or [`bpsk_with_noise_soft`](Self::bpsk_with_noise_soft)
    /// call.
    pub fn build_noise(&mut self, eb_n0_db: f64, bpsk_bit_energy: f64, rng: &mut Rng) {
        let sigma = sigma_for_eb_n0(eb_n0_db, bpsk_bit_energy);
        gaussian(&mut self.noise, sigma, rng);
    }

    /// Encodes `msg`, sends it through the noisy channel, and writes the soft
    /// symbols to `out`.
    ///
    /// The current noise from [`build_noise`](Self::build_noise) is applied.
    /// Returns the number of channel bits that flipped, the uncoded error count
    /// before any decoding.
    pub fn bpsk_with_noise_soft(&mut self, msg: &[u8], bpsk_voltage: f64, out: &mut [u8]) -> usize {
        self.encoder.encode(msg, &mut self.encoded).unwrap();
        encode_bpsk(&self.encoded, &mut self.voltages, self.enclen_bits, bpsk_voltage);
        self.channel.copy_from_slice(&self.voltages);
        add_white_noise(&mut self.channel, &self.noise);
        decode_bpsk_soft(&self.channel, out, bpsk_voltage);

        let mut flips = 0;
        for i in 0..self.enclen_bits {
            let tx = self.encoded[i / 8] & (0x80 >> (i % 8)) != 0;
            let rx = out[i] >= 128;
            flips += (tx != rx) as usize;
        }
        flips
    }

    /// Soft-decodes `msg` through the channel with the scalar decoder.
    ///
    /// Returns a pair. The first value is the number of message bits still wrong
    /// after decoding. The second is the uncoded channel flip count from
    /// [`bpsk_with_noise_soft`](Self::bpsk_with_noise_soft). Comparing the two
    /// shows the coding gain.
    pub fn test_decoder_with_noise(&mut self, decoder: &mut Decoder, msg: &[u8], bpsk_voltage: f64) -> (usize, usize) {
        let n_bytes = msg.len();
        let mut soft = vec![0u8; self.enclen_bits];
        let mut msg_out = vec![0u8; n_bytes];
        let uncoded_flips = self.bpsk_with_noise_soft(msg, bpsk_voltage, &mut soft);

        let decoded_len = decoder.decode_soft(&soft, &mut msg_out).expect("soft decode failed");
        assert_eq!(
            decoded_len, n_bytes,
            "expected to decode {n_bytes} bytes, got {decoded_len}"
        );

        (bit_distance(msg, &msg_out), uncoded_flips)
    }

    /// Like [`test_decoder_with_noise`](Self::test_decoder_with_noise), but
    /// hard-slices the soft symbols and runs the scalar hard decoder. Returns
    /// the same `(coded errors, uncoded flips)` pair.
    pub fn test_decoder_with_noise_hard(
        &mut self,
        decoder: &mut Decoder,
        msg: &[u8],
        bpsk_voltage: f64,
    ) -> (usize, usize) {
        let n_bytes = msg.len();
        let mut soft = vec![0u8; self.enclen_bits];
        let uncoded_flips = self.bpsk_with_noise_soft(msg, bpsk_voltage, &mut soft);

        let mut encoded = vec![0u8; self.enclen_bits.div_ceil(8)];
        for (i, &s) in soft.iter().enumerate() {
            if s >= 128 {
                encoded[i / 8] |= 0x80 >> (i % 8);
            }
        }

        let mut msg_out = vec![0u8; n_bytes];
        let decoded_len = decoder
            .decode_hard(&encoded, self.enclen_bits, &mut msg_out)
            .expect("hard decode failed");
        assert_eq!(
            decoded_len, n_bytes,
            "expected to decode {n_bytes} bytes, got {decoded_len}"
        );

        (bit_distance(msg, &msg_out), uncoded_flips)
    }

    /// Like [`test_decoder_with_noise`](Self::test_decoder_with_noise), but
    /// runs the SIMD decoder. Returns the same `(coded errors, uncoded flips)`
    /// pair.
    #[cfg(feature = "simd")]
    #[cfg_attr(docsrs, doc(cfg(feature = "simd")))]
    pub fn test_decoder_with_noise_simd<const RATE: u32, const ORDER: u32>(
        &mut self,
        decoder: &mut SimdDecoder<RATE, ORDER>,
        msg: &[u8],
        bpsk_voltage: f64,
    ) -> (usize, usize) {
        let n_bytes = msg.len();
        let mut soft = vec![0u8; self.enclen_bits];
        let mut msg_out = vec![0u8; n_bytes];
        let uncoded_flips = self.bpsk_with_noise_soft(msg, bpsk_voltage, &mut soft);

        let decoded_len = decoder
            .decode_soft(&soft, &mut msg_out)
            .expect("simd soft decode failed");
        assert_eq!(
            decoded_len, n_bytes,
            "expected to decode {n_bytes} bytes, got {decoded_len}"
        );

        (bit_distance(msg, &msg_out), uncoded_flips)
    }

    /// Builds `count` independent [`TestCase`]s at the target `eb_n0_db`.
    ///
    /// Each case uses its own message and noise draw, so a benchmark can cycle
    /// through them and decode varied input rather than the same block over and
    /// over.
    pub fn create_test_cases(
        &mut self,
        count: usize,
        eb_n0_db: f64,
        bpsk_voltage: f64,
        bpsk_bit_energy: f64,
    ) -> Vec<TestCase> {
        (0..count)
            .map(|k| {
                let mut rng = Rng::new(1234_5678 + k as u64 * 6767_6767);
                let mut msg = vec![0u8; self.msg_len];
                for b in msg.iter_mut() {
                    *b = rng.next_u8();
                }
                let mut soft = vec![0u8; self.enclen_bits];
                self.build_noise(eb_n0_db, bpsk_bit_energy, &mut rng);
                self.bpsk_with_noise_soft(&msg, bpsk_voltage, &mut soft);
                let mut hard = vec![0u8; self.enclen_bits.div_ceil(8)];
                for (i, &s) in soft.iter().enumerate() {
                    if s >= 128 {
                        hard[i / 8] |= 0x80 >> (i % 8);
                    }
                }
                TestCase {
                    soft,
                    hard,
                    msg,
                    enc_bits: self.enclen_bits,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measure_ber(
        rate: u32,
        order: u32,
        polys: &[u16],
        eb_n0_db: f64,
        total_bytes: usize,
        block: usize,
        seed: u64,
    ) -> f64 {
        let errors = super::measure_ber(rate, order, polys, eb_n0_db, total_bytes, block, seed);
        let sent = total_bytes.div_ceil(block) * block;
        errors as f64 / (sent as f64 * 8.0)
    }

    fn uncoded_ber(eb_n0_db: f64, n_bits: usize, seed: u64) -> f64 {
        let mut rng = Rng::new(seed);
        let bpsk_voltage = 1.0 / 2f64.sqrt();
        let bpsk_bit_energy = 2.0 * bpsk_voltage * bpsk_voltage;
        let sigma = sigma_for_eb_n0(eb_n0_db, bpsk_bit_energy);
        let mut noise = vec![0.0f64; n_bits];
        gaussian(&mut noise, sigma, &mut rng);
        let mut signal = vec![bpsk_voltage; n_bits];
        add_white_noise(&mut signal, &noise);
        let errors = signal.iter().filter(|&s| *s < 0.0).count() as usize;
        errors as f64 / n_bits as f64
    }

    fn assert_coding_gain(rate: u32, order: u32, polys: &[u16], high_db: f64, mid_db: f64, bytes: usize) {
        let high = measure_ber(rate, order, polys, high_db, bytes, 4096, 1234_5678);
        let mid = measure_ber(rate, order, polys, mid_db, bytes, 4096, 1234_5678);

        let high_uncoded = uncoded_ber(high_db, bytes * 8, 1234_5678);
        let mid_uncoded = uncoded_ber(mid_db, bytes * 8, 1234_5678);

        assert!(high < 1e-4, "{rate}/{order}: {high_db}dB coded BER too high: {high}");
        assert!(
            high < high_uncoded / 100.0,
            "{rate}/{order}: no coding gain at {high_db}dB — coded {high} vs uncoded {high_uncoded}"
        );
        assert!(mid >= high, "{rate}/{order}: BER not monotonic ({mid} < {high})");
        assert!(
            mid < mid_uncoded / 25.0,
            "{rate}/{order}: no coding gain at {mid_db}dB — coded {mid} vs uncoded {mid_uncoded}"
        );
    }

    #[test]
    fn scalar_coding_gain_k7() {
        assert_coding_gain(2, 7, &[0o161, 0o127], 5.0, 3.0, 100_000);
    }

    #[test]
    fn scalar_coding_gain_k9() {
        assert_coding_gain(2, 9, &[0o657, 0o435], 4.5, 2.5, 100_000);
    }

    #[test]
    #[ignore = "slow: rate 1/6 k=15 coding gain"]
    fn scalar_coding_gain_k15() {
        assert_coding_gain(
            6,
            15,
            &[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537],
            4.0,
            2.5,
            20_000,
        );
    }

    #[test]
    fn deterministic_testing() {
        let polys = [0o161u16, 0o127];
        let a = measure_ber(2, 7, &polys, 3.0, 40_000, 4096, 1234_5678);
        let b = measure_ber(2, 7, &polys, 3.0, 40_000, 4096, 1234_5678);
        assert_eq!(a, b);
    }
}
