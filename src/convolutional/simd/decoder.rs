use std::simd::prelude::*;

use super::lane::*;
use super::oct_lookup::{DistanceShuffle, OctLookup};
use crate::convolutional::bit::{BitReader, BitWriter};
use crate::convolutional::decoder::ConvolutionalError;
use crate::convolutional::error::{self, DecodeError};
use crate::convolutional::util;

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForcedPath {
    RegisterAvx512,  // u16x32, fully register, AVX-512
    RegisterAvx2,    // u16x16, register-resident, AVX2
    RegisterSse41,   // u16x8, register-resident, SSE
    Register128,     // u16x8, register-resident, generic 128-bit
    ShuffleAvx512,   // distance-in-register, states-in-memory, AVX-512
    ShuffleAvx2,     // distance-in-register, states-in-memory, AVX2
    ShuffleSse41,    // distance-in-register, states-in-memory, SSE4.1
    Shuffle128,      // distance-in-register, states-in-memory, generic 128-bit
    PermuteAvx512,   // wide distance-in-register, states-in-memory, AVX-512
    OctLookupAvx512, // SIMD butterfly, in-memory, AVX-512
    OctLookupAvx2,   // SIMD butterfly, in-memory, AVX2
    OctLookupSse41,  // SIMD butterfly, in-memory, SSE4.1
    OctLookup128,    // SIMD butterfly, in-memory, generic 128-bit
}

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderArch {
    Sse41,
    Avx2,
    Avx512,
}

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnableArch {
    pub sse41: bool,
    pub avx2: bool,
    pub avx512: bool,
}

/// A SIMD Viterbi decoder for a convolutional code.
///
/// This decodes identically to the scalar
/// [`Decoder`](crate::convolutional::Decoder), just faster on x86. The `RATE`
/// and `ORDER` are const generic, so each concrete `(rate, order)` pair is its
/// own monomorphized type. `RATE` must be between 2 and 8, and `ORDER` between 4
/// and 16.
///
/// The decoder detects the host's SIMD features at construction and picks the
/// widest available path, from AVX-512 down to AVX2, SSE4.1, and a portable
/// 128-bit fallback. There is nothing to configure. Build it with
/// [`new`](Self::new) and call [`decode_hard`](Self::decode_hard) or
/// [`decode_soft`](Self::decode_soft).
///
/// This type needs the `simd` feature and a nightly compiler.
#[cfg_attr(docsrs, doc(cfg(feature = "simd")))]
#[derive(Debug)]
pub struct SimdDecoder<const RATE: u32, const ORDER: u32> {
    poly_table: Vec<u8>,
    oct_lookup: OctLookup,
    distance_shuffle: DistanceShuffle,
    errors: Vec<u16>,
    previous_errors: Vec<u16>,
    history: Vec<u8>,
    history_index: usize,
    history_len: usize,
    renormalize_counter: u32,
    decode_buf: Vec<u8>,
    distances: Vec<u16>,
    force_path: Option<ForcedPath>,
    enable_arch: EnableArch,
}

impl<const RATE: u32, const ORDER: u32> SimdDecoder<RATE, ORDER> {
    const fn num_survivors() -> usize {
        1 << (ORDER - 1)
    }
    const fn high_prev_offset() -> usize {
        Self::num_survivors() / 2
    }
    const fn num_distances() -> usize {
        let n = 1usize << RATE;
        // we need capacity for at least 64 to satisfy the permute 512's distance load
        if n < 64 {
            64
        } else {
            n
        }
    }
    const fn hist_stride() -> usize {
        // history is bit-packed (1 bit per state)
        Self::num_survivors() / 8
    }
    const fn min_traceback_length() -> u32 {
        5 * ORDER
    }
    const fn history_cap() -> usize {
        Self::min_traceback_length() as usize + util::traceback_group_length(ORDER)
    }
    const fn renormalize_interval() -> u32 {
        // we need to renormalize just often enough to not overflow
        // the aggregate error values are in i16 (unsigned, but with signed comparison)
        // each step can add up to RATE * 255 (max soft sample value)
        i16::MAX as u32 / (RATE * u8::MAX as u32)
    }

    /// Creates a decoder for the convolutional code with the given generator
    /// polynomials.
    ///
    /// `polys` must contain exactly `RATE` polynomials, in the same octal
    /// convention as [`Encoder::new`](crate::convolutional::Encoder::new). The
    /// host's SIMD features are detected here, so build the decoder once and
    /// reuse it.
    ///
    /// # Panics
    ///
    /// Panics if `polys.len()` is not equal to `RATE`. The `RATE` and `ORDER`
    /// bounds are checked at compile time, so an out-of-range shape fails to
    /// build rather than panicking.
    pub fn new(polys: &[u16]) -> Self {
        const { assert!(ORDER >= 4, "SimdDecoder requires order >= 4") };
        const { assert!(ORDER <= 16, "SimdDecoder requires order <= 16") };
        const { assert!(RATE >= 2, "SimdDecoder requires rate >= 2") };
        const { assert!(RATE <= 8, "SimdDecoder requires rate <= 8") };

        let rate = RATE;
        let order = ORDER;
        let poly_table: Vec<u8> = util::conv_poly_table(rate, order, polys)
            .iter()
            .map(|&p| p as u8)
            .collect();

        let cap = Self::history_cap();

        let num_history = Self::hist_stride() * cap;

        let num_distances = Self::num_distances();

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let has_sse41 = std::is_x86_feature_detected!("sse4.1");
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let has_sse41 = false;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let has_avx2 = std::is_x86_feature_detected!("avx2");
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let has_avx2 = false;

        #[cfg(target_arch = "x86_64")]
        let has_avx512 = std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl");
        #[cfg(not(target_arch = "x86_64"))]
        let has_avx512 = false;

        let enable_arch = EnableArch {
            sse41: has_sse41,
            avx2: has_avx2,
            avx512: has_avx512,
        };

        // allocate some extra space past the end of the history table so that we can always
        //    read a u64 off it in traceback (narrower ORDERs only)
        let history = vec![0; num_history + 8];

        SimdDecoder {
            oct_lookup: OctLookup::new(rate, order, &poly_table),
            distance_shuffle: DistanceShuffle::new(rate, order, &poly_table),
            poly_table,
            errors: vec![0; Self::num_survivors()],
            previous_errors: vec![0; Self::num_survivors()],
            history,
            history_index: 0,
            history_len: 0,
            renormalize_counter: 0,
            decode_buf: vec![0; cap],
            distances: vec![0; num_distances],
            enable_arch,
            force_path: None,
        }
    }

    // helper field for testing/performance tuning (force a specific decode path)
    #[doc(hidden)]
    pub fn with_path(mut self, path: Option<ForcedPath>) -> Self {
        self.force_path = path;
        self
    }

    // helper field for testing/performance tuning (force a specific SIMD arch)
    #[doc(hidden)]
    pub fn with_max_arch(mut self, arch: DecoderArch) -> Self {
        match arch {
            DecoderArch::Sse41 => {
                self.enable_arch.sse41 = true;
                self.enable_arch.avx2 = false;
                self.enable_arch.avx512 = false;
            }
            DecoderArch::Avx2 => {
                self.enable_arch.sse41 = true;
                self.enable_arch.avx2 = true;
                self.enable_arch.avx512 = false;
            }
            DecoderArch::Avx512 => {
                self.enable_arch.sse41 = true;
                self.enable_arch.avx2 = true;
                self.enable_arch.avx512 = true;
            }
        }
        self
    }

    fn reset(&mut self) {
        self.errors.fill(0);
        self.previous_errors.fill(0);
        self.history_len = 0;
        self.history_index = 0;
        self.renormalize_counter = 0;
    }

    /// Decodes a hard-decision block. This behaves identically to the scalar
    /// [`Decoder::decode_hard`](crate::convolutional::Decoder::decode_hard). See
    /// it for the parameter and return conventions.
    pub fn decode_hard(
        &mut self,
        encoded: &[u8],
        num_encoded_bits: usize,
        msg: &mut [u8],
    ) -> Result<usize, DecodeError> {
        error::validate_encoded_len(num_encoded_bits, RATE, ORDER)?;

        if num_encoded_bits.div_ceil(8) > encoded.len() {
            return Err(DecodeError::InvalidLength {
                num_encoded_bits,
                rate: RATE,
            });
        }
        let needed = error::payload_len_bytes(num_encoded_bits, RATE, ORDER);
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

    /// Decodes a soft-decision block. This behaves identically to the scalar
    /// [`Decoder::decode_soft`](crate::convolutional::Decoder::decode_soft). See
    /// it for the soft-symbol convention and return conventions.
    pub fn decode_soft(&mut self, encoded: &[u8], msg: &mut [u8]) -> Result<usize, DecodeError> {
        // encoded is just one byte per bit (soft samples)
        let num_encoded_bits = encoded.len();

        error::validate_encoded_len(num_encoded_bits, RATE, ORDER)?;

        let needed = error::payload_len_bytes(num_encoded_bits, RATE, ORDER);
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

    fn _decode(
        &mut self,
        distance_fill: &mut ConvolutionalError,
        num_encoded_bits: usize,
        decoded: &mut BitWriter,
    ) -> usize {
        self.reset();

        self.decode_head(distance_fill);
        self.decode_body(distance_fill, num_encoded_bits, decoded);
        self.decode_tail(distance_fill, num_encoded_bits, decoded);

        self.flush(decoded);

        decoded.len()
    }

    fn decode_head(&mut self, distance_fill: &mut ConvolutionalError) {
        for i in 0..(ORDER - 1) {
            distance_fill.fill_next_distances(&mut self.distances, RATE);

            let num_states = 1 << (i + 1);
            for j in 0..num_states {
                let previous_state = j >> 1;
                let distance = self.distances[self.poly_table[j] as usize];
                self.errors[j] = distance + self.previous_errors[previous_state];
            }
            self.swap_errors();
        }
    }

    fn decode_body(
        &mut self,
        distance_fill: &mut ConvolutionalError,
        num_encoded_bits: usize,
        decoded: &mut BitWriter,
    ) {
        if let Some(path) = self.force_path {
            self.dispatch_forced(path, distance_fill, num_encoded_bits, decoded);
            return;
        }

        let perf_register_512 = (7..=9).contains(&ORDER) && RATE <= 3;
        let perf_register_256 = (6..=8).contains(&ORDER) && RATE <= 3;
        let perf_register_128 = (5..=7).contains(&ORDER) && RATE <= 3;

        let register_512_ok = perf_register_512 && !self.distance_shuffle.shuffle32.is_empty();
        let register_256_ok = perf_register_256 && !self.distance_shuffle.shuffle16.is_empty();
        let register_128_ok = perf_register_128 && !self.distance_shuffle.shuffle8.is_empty();

        let perf_shuffle_512 = ORDER >= 6 && RATE <= 3;
        let perf_shuffle_256 = ORDER >= 5 && RATE <= 3;
        let perf_shuffle_128 = ORDER >= 4 && RATE <= 3;

        let shuffle_512_ok = perf_shuffle_512 && !self.distance_shuffle.shuffle32.is_empty();
        let shuffle_256_ok = perf_shuffle_256 && !self.distance_shuffle.shuffle16.is_empty();
        let shuffle_128_ok = perf_shuffle_128 && !self.distance_shuffle.shuffle8.is_empty();

        // at RATE <= 3, permute needs to beat the shuffle decoder, which only happens at large order
        // at RATE >= 3, the permute decoder easily beats the only alternative (octlookup decoder)
        let perf_permute_512 = (ORDER >= 14 && RATE <= 3) || RATE >= 3;

        let permute_512_ok = perf_permute_512 && RATE <= 6;

        let oct_lookup_512_ok = Self::num_survivors() >= Lane512::LANES;
        let oct_lookup_256_ok = Self::num_survivors() >= Lane256::LANES;
        let oct_lookup_128_ok = Self::num_survivors() >= Lane128::LANES;

        // pick a decoder path based on the available SIMD features and the code geometry

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if const { RATE <= 3 } {
                if register_512_ok && self.enable_arch.avx512 {
                    unsafe { self.decode_body_register_avx512(distance_fill, num_encoded_bits, decoded) };
                } else if register_256_ok && self.enable_arch.avx2 {
                    unsafe { self.decode_body_register_avx2(distance_fill, num_encoded_bits, decoded) };
                } else if register_128_ok && self.enable_arch.sse41 {
                    unsafe { self.decode_body_register_sse41(distance_fill, num_encoded_bits, decoded) };
                } else if register_128_ok {
                    self.decode_body_register::<Lane128>(distance_fill, num_encoded_bits, decoded);
                } else if permute_512_ok && self.enable_arch.avx512 {
                    unsafe { self.decode_body_permute_avx512(distance_fill, num_encoded_bits, decoded) };
                } else if shuffle_512_ok && self.enable_arch.avx512 {
                    unsafe { self.decode_body_shuffle_avx512(distance_fill, num_encoded_bits, decoded) };
                } else if shuffle_256_ok && self.enable_arch.avx2 {
                    unsafe { self.decode_body_shuffle_avx2(distance_fill, num_encoded_bits, decoded) };
                } else if shuffle_128_ok && self.enable_arch.sse41 {
                    unsafe { self.decode_body_shuffle_sse41(distance_fill, num_encoded_bits, decoded) };
                } else {
                    self.decode_body_shuffle::<Lane128>(distance_fill, num_encoded_bits, decoded);
                }
            } else if permute_512_ok && self.enable_arch.avx512 {
                unsafe { self.decode_body_permute_avx512(distance_fill, num_encoded_bits, decoded) };
            } else if oct_lookup_512_ok && self.enable_arch.avx512 {
                unsafe { self.decode_body_oct_lookup_avx512(distance_fill, num_encoded_bits, decoded) };
            } else if oct_lookup_256_ok && self.enable_arch.avx2 {
                unsafe { self.decode_body_oct_lookup_avx2(distance_fill, num_encoded_bits, decoded) };
            } else if oct_lookup_128_ok && self.enable_arch.sse41 {
                unsafe { self.decode_body_oct_lookup_sse41(distance_fill, num_encoded_bits, decoded) };
            } else {
                self.decode_body_oct_lookup::<Lane128>(distance_fill, num_encoded_bits, decoded);
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        if const { RATE <= 3 } {
            if register_128_ok {
                self.decode_body_register::<Lane128>(distance_fill, num_encoded_bits, decoded);
            } else if shuffle_128_ok {
                self.decode_body_shuffle::<Lane128>(distance_fill, num_encoded_bits, decoded);
            } else {
                self.decode_body_oct_lookup::<Lane128>(distance_fill, num_encoded_bits, decoded);
            }
        } else {
            self.decode_body_oct_lookup::<Lane128>(distance_fill, num_encoded_bits, decoded);
        }
    }

    fn decode_tail(
        &mut self,
        distance_fill: &mut ConvolutionalError,
        num_encoded_bits: usize,
        decoded: &mut BitWriter,
    ) {
        let num_decoded_bits = num_encoded_bits as u32 / RATE;
        let num_survivors = Self::num_survivors();
        let high_prev_offset = Self::high_prev_offset();

        for i in (num_decoded_bits - ORDER + 1)..num_decoded_bits {
            distance_fill.fill_next_distances(&mut self.distances, RATE);

            let step = 1usize << (ORDER - (num_decoded_bits - i));
            let hist_offset = self.history_offset();

            // since we are filling in strided survivor bits here, we should clear the unused ones
            for b in &mut self.history[hist_offset..hist_offset + Self::hist_stride()] {
                *b = 0;
            }

            for state in (0..num_survivors).step_by(step) {
                let prev_state = state / 2;

                let low_output = self.poly_table[state];
                let high_output = self.poly_table[state + num_survivors];

                let low_prev_error = self.previous_errors[prev_state];
                let high_prev_error = self.previous_errors[prev_state + high_prev_offset];

                let low_error = self.distances[low_output as usize] + low_prev_error;
                let high_error = self.distances[high_output as usize] + high_prev_error;

                let (error, successor) = if low_error <= high_error {
                    (low_error, 0)
                } else {
                    (high_error, 1)
                };

                self.errors[state] = error;
                self.history[hist_offset + state / 8] |= successor << (state % 8);
            }

            self.process_history::<Lane128>(step as u32, decoded);
            self.swap_errors();
        }
    }

    #[inline(always)]
    fn decode_body_oct_lookup<L: OctLookupLane>(
        &mut self,
        distance_fill: &mut ConvolutionalError,
        num_encoded_bits: usize,
        decoded: &mut BitWriter,
    ) {
        debug_assert!(
            Self::num_survivors() >= L::LANES,
            "oct lookup decoder requires wider order"
        );

        // the octlookup decoder is closest to the portable/scalar decoder, but we look up 8
        //   state distances at a time (16 on wider archs). everything else is roughly the same but
        //   with explicit vectorization

        let num_decoded_bits = num_encoded_bits as u32 / RATE;
        let num_survivors = Self::num_survivors();
        let high_prev_offset = Self::high_prev_offset();
        let high_lookup_offset = Self::num_survivors() / L::LOOKUP_WIDTH;

        for _ in (ORDER - 1)..(num_decoded_bits - ORDER + 1) {
            Self::fill_next_distances(distance_fill, &mut self.distances);

            L::fill_distances(&mut self.distances, &mut self.oct_lookup);

            let keys = L::keys(&self.oct_lookup);
            let octdist = L::octdist(&self.oct_lookup);

            let hist_offset = self.history_offset();
            let prev = self.previous_errors.as_ptr();
            let errors = self.errors.as_mut_ptr();
            let history = self.history.as_mut_ptr();

            let mut survivor = 0;
            unsafe {
                while survivor + L::LANES <= num_survivors {
                    let predecessor = survivor / 2;
                    let lookup_offset = survivor / L::LOOKUP_WIDTH;
                    let low_dist = L::load_dist(keys, octdist, lookup_offset, 0);
                    let high_dist = L::load_dist(keys, octdist, lookup_offset, high_lookup_offset);
                    let low_prev = L::load_pred_dup(prev, predecessor, 0);
                    let high_prev = L::load_pred_dup(prev, predecessor, high_prev_offset);
                    let low_error = L::add(low_dist, low_prev);
                    let high_error = L::add(high_dist, high_prev);
                    L::store_err(errors, survivor, L::min(low_error, high_error));
                    L::write_history(history, hist_offset + survivor / 8, low_error, high_error);
                    survivor += L::LANES;
                }
            }

            self.process_history::<L>(1, decoded);
            self.swap_errors();
        }
    }

    #[inline(always)]
    fn decode_body_shuffle<L: ShuffleLane>(
        &mut self,
        distance_fill: &mut ConvolutionalError,
        num_encoded_bits: usize,
        decoded: &mut BitWriter,
    ) {
        debug_assert!(
            Self::num_survivors() >= L::LANES,
            "shuffle decoder requires wider order"
        );
        debug_assert!(RATE <= 3, "shuffle decoder requires rate<=3");

        // the shuffle decoder loads the state distances into a register and then masks/shuffles
        //    them to the vectorized state registers. this skips the oct lookup entirely and
        //    bypasses an entire set of memory accesses

        let num_decoded_bits = num_encoded_bits as u32 / RATE;
        let num_survivors = Self::num_survivors();
        let high_prev_offset = Self::high_prev_offset();
        let high_shuffle_offset = num_survivors / L::SHUFFLE_WIDTH;

        for _ in (ORDER - 1)..(num_decoded_bits - ORDER + 1) {
            let dist_32 = Self::fill_next_distances_register(distance_fill);
            let dist_16: u8x16 = simd_swizzle!(dist_32, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
            let mask_ptr = L::shuffle_mask_ptr(&self.distance_shuffle);

            let hist_offset = self.history_offset();
            let prev = self.previous_errors.as_ptr();
            let errors = self.errors.as_mut_ptr();
            let history = self.history.as_mut_ptr();

            let mut survivor = 0;
            unsafe {
                while survivor + L::LANES <= num_survivors {
                    let predecessor = survivor / 2;
                    let shuffle_offset = survivor / L::SHUFFLE_WIDTH;
                    let low_dist = L::load_dist_shuffle(mask_ptr, shuffle_offset, 0, dist_16);
                    let high_dist = L::load_dist_shuffle(mask_ptr, shuffle_offset, high_shuffle_offset, dist_16);
                    let low_prev = L::load_pred_dup(prev, predecessor, 0);
                    let high_prev = L::load_pred_dup(prev, predecessor, high_prev_offset);
                    let low_error = L::add(low_dist, low_prev);
                    let high_error = L::add(high_dist, high_prev);
                    L::store_err(errors, survivor, L::min(low_error, high_error));
                    L::write_history(history, hist_offset + survivor / 8, low_error, high_error);
                    survivor += L::LANES;
                }
            }

            self.process_history::<L>(1, decoded);
            self.swap_errors();
        }
    }

    #[inline(always)]
    fn decode_body_permute<L: PermuteLane>(
        &mut self,
        distance_fill: &mut ConvolutionalError,
        num_encoded_bits: usize,
        decoded: &mut BitWriter,
    ) {
        debug_assert!(
            Self::num_survivors() >= L::LANES,
            "permute decoder requires wider order"
        );
        debug_assert!(RATE <= 6, "permute decoder requires rate<=6");

        // the permute decoder takes advantage of special vectorized permute instructions to
        //    load state distances into a register using the polys as a key into the distance
        //    table

        let num_decoded_bits = num_encoded_bits as u32 / RATE;
        let num_survivors = Self::num_survivors();
        let high_prev_offset = Self::high_prev_offset();

        for _ in (ORDER - 1)..(num_decoded_bits - ORDER + 1) {
            Self::fill_next_distances(distance_fill, &mut self.distances);

            let table = unsafe { L::dist_table(self.distances.as_ptr()) };

            let hist_offset = self.history_offset();
            let prev = self.previous_errors.as_ptr();
            let errors = self.errors.as_mut_ptr();
            let history = self.history.as_mut_ptr();
            let poly = self.poly_table.as_ptr();

            let mut survivor = 0;
            unsafe {
                while survivor + L::LANES <= num_survivors {
                    let predecessor = survivor / 2;
                    let low_dist = L::permute_dist(table, poly, survivor);
                    let high_dist = L::permute_dist(table, poly, survivor + num_survivors);
                    let low_prev = L::load_pred_dup(prev, predecessor, 0);
                    let high_prev = L::load_pred_dup(prev, predecessor, high_prev_offset);
                    let low_error = L::add(low_dist, low_prev);
                    let high_error = L::add(high_dist, high_prev);
                    L::store_err(errors, survivor, L::min(low_error, high_error));
                    L::write_history(history, hist_offset + survivor / 8, low_error, high_error);
                    survivor += L::LANES;
                }
            }

            self.process_history::<L>(1, decoded);
            self.swap_errors();
        }
    }

    #[inline(always)]
    fn decode_body_register<L: RegisterLane>(
        &mut self,
        distance_fill: &mut ConvolutionalError,
        num_encoded_bits: usize,
        decoded: &mut BitWriter,
    ) {
        debug_assert!(RATE <= 3, "register decoder requires rate<=3");

        // fully registerized decoder. the only memory accesses are to load the
        //    input bits in and to write the survivor bits out for traceback

        const MAX_REGISTERS: usize = 16;
        let num_decoded_bits = num_encoded_bits as u32 / RATE;
        let num_registers: usize = Self::num_survivors() / L::LANES;

        debug_assert!(MAX_REGISTERS >= num_registers, "MAX_REGISTERS must be >= num_registers");
        debug_assert!(num_registers >= 2, "num_registers must be >= 2");

        // these will not truly be stack values. we are expecting the compiler to put these
        // all into vector registers, and monomorphization will actually use fewer than
        // MAX_REGISTERS as needed (it should only make use of `num_registers`)
        let mut prev: [L::Vec; MAX_REGISTERS] = [L::zeros(); MAX_REGISTERS];
        let mut cur: [L::Vec; MAX_REGISTERS] = [L::zeros(); MAX_REGISTERS];

        // there's no load inside the loop itself. we just do one here to get the registers setup
        for register in 0..num_registers {
            prev[register] = unsafe { L::load(self.previous_errors.as_ptr().add(register * L::LANES)) };
        }

        let mask_ptr = L::shuffle_mask_ptr(&self.distance_shuffle);

        for _ in (ORDER - 1)..(num_decoded_bits - ORDER + 1) {
            let dist_32: u8x32 = Self::fill_next_distances_register(distance_fill);
            let dist_16: u8x16 = simd_swizzle!(dist_32, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);

            let hist_offset = self.history_offset();
            let history = self.history.as_mut_ptr();

            // note that in the following loop, even and odd-numbered registers share
            // low_prev_reg/high_prev_reg, but they differ on which half is read from (pred_half)
            unsafe {
                for register in 0..num_registers {
                    let low_dist = L::shuffle_dist(mask_ptr, register, 0, dist_16);
                    let high_dist = L::shuffle_dist(mask_ptr, register, num_registers, dist_16);
                    let low_prev_reg = L::pred_reg(prev.as_ptr(), register / 2, 0);
                    let high_prev_reg = L::pred_reg(prev.as_ptr(), register / 2, num_registers / 2);
                    let low_prev = L::pred_half(low_prev_reg, register % 2 == 1);
                    let high_prev = L::pred_half(high_prev_reg, register % 2 == 1);
                    let low_error = L::add(low_dist, low_prev);
                    let high_error = L::add(high_dist, high_prev);
                    L::store_err(&mut cur, register, L::min(low_error, high_error));
                    L::write_history(history, hist_offset + register * (L::LANES / 8), low_error, high_error);
                }
            }

            std::mem::swap(&mut prev, &mut cur);

            self.process_history_register::<L>(&mut prev, num_registers, decoded);
        }

        // now unload from register back to self.previous_errors
        for register in 0..num_registers {
            unsafe {
                L::store(
                    self.previous_errors.as_mut_ptr().add(register * L::LANES),
                    prev[register],
                )
            };
        }
    }

    #[inline(always)]
    #[rustfmt::skip]
    fn fill_next_distances_register(distance_fill: &mut ConvolutionalError) -> u8x32 {
        debug_assert!(
            RATE >= 2 && RATE <= 3,
            "fill_next_distances_register is rate<=3 only (got rate {RATE})"
        );

        let d = match distance_fill {
            ConvolutionalError::Soft(encoded) => {
                let d = Self::dist3_soft_from(&encoded[..RATE as usize], 0);
                *encoded = &encoded[RATE as usize..];
                d
            }
            ConvolutionalError::Hard(encoded) => {
                let outputs = encoded.read(RATE as usize);
                Self::dist3_hard_from(outputs, 0)
            }
        };
        let lo: u8x16 = unsafe { core::mem::transmute(d) };
        simd_swizzle!(
            lo,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        )
    }

    #[inline(always)]
    fn fill_next_distances(distance_fill: &mut ConvolutionalError, out: &mut [u16]) {
        // note that this method depends on the templated variable RATE, and for
        // RATE < 8, some of the work can be automatically eliminated as unused
        let r = RATE as usize;

        // create 3, 8-lane (2^3) registers for a total of 24 lanes for up to 8 input samples
        // each register will contain all permutations of distances for 3 input samples
        // we will then cross the registers to get the full set of (2^RATE) distances for all `RATE` samples
        let (d0, d1, d2) = match distance_fill {
            ConvolutionalError::Soft(encoded) => {
                let d0 = Self::dist3_soft_from(encoded, 0);
                let d1 = Self::dist3_soft_from(encoded, 3);
                let d2 = Self::dist3_soft_from(encoded, 6);
                *encoded = &encoded[r..];
                (d0, d1, d2)
            }
            ConvolutionalError::Hard(reader) => {
                let bits = reader.read(r);
                let d0 = Self::dist3_hard_from(bits, 0);
                let d1 = Self::dist3_hard_from(bits, 3);
                let d2 = Self::dist3_hard_from(bits, 6);
                (d0, d1, d2)
            }
        };

        // d0 now contains the 8 distances for input samples 0-2, d1 the 8 distances for 3-5, and d2 the 4 distances for 6-7

        let d1_lanes = 1usize << (r.min(6).saturating_sub(3));
        let d2_lanes = 1usize << r.saturating_sub(6);

        // some examples
        // RATE=2 or RATE=3, d1_lanes=1, d2_lanes=1
        // RATE=4, d1_lanes=2, d2_lanes=1
        // RATE=6, d1_lanes=8, d2_lanes=1
        // RATE=8, d1_lanes=8, d2_lanes=4

        // do a bounds check once (static bounds after this)
        let out = &mut out[..Self::num_distances()];

        for c in 0..d2_lanes {
            let mid = d0 + u16x8::splat(d2[c]);
            for b in 0..d1_lanes {
                let chunk = c * d1_lanes + b;
                let chunk_d = mid + u16x8::splat(d1[b]);
                chunk_d.copy_to_slice(&mut out[chunk * 8..chunk * 8 + 8]);
            }
        }
    }

    #[inline(always)]
    fn dist3_soft_from(encoded: &[u8], from: usize) -> u16x8 {
        let pair = |i: usize| -> (u16, u16) {
            if from + i < RATE as usize {
                let s = encoded[from + i] as u16;
                (s, 255 - s)
            } else {
                (0, 0)
            }
        };
        let (d0, d0_inv) = pair(0);
        let (d1, d1_inv) = pair(1);
        let (d2, d2_inv) = pair(2);
        Self::dist3_from(d0, d0_inv, d1, d1_inv, d2, d2_inv)
    }

    #[inline(always)]
    fn dist3_hard_from(bits: u8, from: usize) -> u16x8 {
        let pair = |i: usize| -> (u16, u16) {
            if from + i < RATE as usize {
                if (bits >> (from + i)) & 1 != 0 {
                    (1, 0)
                } else {
                    (0, 1)
                }
            } else {
                (0, 0)
            }
        };
        let (d0, d0_inv) = pair(0);
        let (d1, d1_inv) = pair(1);
        let (d2, d2_inv) = pair(2);
        Self::dist3_from(d0, d0_inv, d1, d1_inv, d2, d2_inv)
    }

    #[inline(always)]
    fn dist3_from(d0: u16, d0_inv: u16, d1: u16, d1_inv: u16, d2: u16, d2_inv: u16) -> u16x8 {
        // create the 8 combined distance measurements over the incoming 3 sets of values
        let base = u16x8::from_array([d0, d0_inv, d1, d1_inv, d2, d2_inv, 0, 0]);
        let a: u16x8 = simd_swizzle!(base, [0, 1, 0, 1, 0, 1, 0, 1]);
        let b: u16x8 = simd_swizzle!(base, [2, 2, 3, 3, 2, 2, 3, 3]);
        let c: u16x8 = simd_swizzle!(base, [4, 4, 4, 4, 5, 5, 5, 5]);
        a + b + c
    }

    fn process_history<L: MemoryLane>(&mut self, step: u32, bit_writer: &mut BitWriter) {
        if self.advance_history() {
            self.process_history_advance::<L>(step, bit_writer);
        }
    }

    #[inline]
    fn process_history_register<L: RegisterLane>(
        &mut self,
        prev: &mut [L::Vec],
        num_registers: usize,
        bit_writer: &mut BitWriter,
    ) {
        if self.advance_history() {
            let renorm_due = self.renormalize_counter == Self::renormalize_interval();
            let traceback_due = self.history_len == Self::history_cap();

            if renorm_due {
                self.renormalize_counter = 0;
                L::renorm_sub_min(prev, num_registers);
            }

            if traceback_due {
                for register in 0..num_registers {
                    unsafe { L::store(self.errors.as_mut_ptr().add(register * L::LANES), prev[register]) };
                }
                let best_path = self.least_error_path_scalar();
                let min_traceback_length = Self::min_traceback_length();
                self.traceback(best_path, min_traceback_length, bit_writer);
                for register in 0..num_registers {
                    prev[register] = unsafe { L::load(self.errors.as_ptr().add(register * L::LANES)) };
                }
            }
        }
    }

    #[inline]
    fn advance_history(&mut self) -> bool {
        self.history_index += 1;
        if self.history_index == Self::history_cap() {
            self.history_index = 0;
        }

        self.renormalize_counter += 1;
        self.history_len += 1;

        self.renormalize_counter == Self::renormalize_interval() || self.history_len == Self::history_cap()
    }

    fn process_history_advance<L: MemoryLane>(&mut self, step: u32, bit_writer: &mut BitWriter) {
        let renormalize_due = self.renormalize_counter == Self::renormalize_interval();
        let traceback_due = self.history_len == Self::history_cap();

        if traceback_due {
            let best_path = self.least_error_path::<L>(step);
            if renormalize_due {
                self.renormalize_counter = 0;
                let m = self.errors[best_path as usize];
                L::renorm(&mut self.errors, m);
            }
            self.traceback(best_path, Self::min_traceback_length(), bit_writer);
        } else if renormalize_due {
            self.renormalize_counter = 0;
            let m = self.least_error_value::<L>(step);
            L::renorm(&mut self.errors, m);
        }
    }

    fn least_error_value<L: MemoryLane>(&self, step: u32) -> u16 {
        if step == 1 {
            L::min_value(&self.errors)
        } else {
            *self.errors.iter().step_by(step as usize).min().unwrap_or(&u16::MAX)
        }
    }

    fn least_error_path<L: MemoryLane>(&self, step: u32) -> u16 {
        if step == 1 {
            // wide min-value scan, then a vectorized search for its position (common case)
            let m = L::min_value(&self.errors);
            let target = u16x8::splat(m);
            for (chunk, c) in self.errors.as_chunks::<8>().0.iter().enumerate() {
                let hits = u16x8::from_slice(c).simd_eq(target).to_bitmask();
                if hits != 0 {
                    return (chunk * 8) as u16 + hits.trailing_zeros() as u16;
                }
            }
            return 0;
        }

        // tail case (step != 1)
        let m = self.errors.iter().step_by(step as usize).min().unwrap_or(&u16::MAX);
        let pos = self
            .errors
            .iter()
            .step_by(step as usize)
            .position(|&d| d == *m)
            .unwrap_or(0);
        (pos * step as usize) as u16
    }

    fn least_error_path_scalar(&self) -> u16 {
        let m = self.errors.iter().min().unwrap_or(&u16::MAX);
        self.errors.iter().position(|&d| d == *m).unwrap_or(0) as u16
    }

    fn traceback(&mut self, init_best_path: u16, min_traceback_length: u32, bit_writer: &mut BitWriter) {
        let stride = Self::hist_stride();
        let mut index = self.history_index;
        let mut best_path = init_best_path;

        let cap = Self::history_cap();
        let num_survivors = Self::num_survivors() as u16;
        let hist = self.history.as_ptr();

        let survivor_bit = |index: usize, path: u16| -> u16 {
            if stride <= 8 {
                // for narrower tables (ORDER <= 7), we can remove the dependency from survivor to address.
                // the path dependency then only shows up on the bit shift
                // we load the entire stride into a single u64 containing all survivor bits
                let row = unsafe { (hist.add(index * stride) as *const u64).read_unaligned() };
                ((row >> path) & 1) as u16
            } else {
                // for wider tables, we'd have to add extra loads, so it's better to keep the dependency
                let p = path as usize;
                ((unsafe { *hist.add(index * stride + p / 8) } >> (p % 8)) & 1) as u16
            }
        };

        // for the first `min_traceback_length` bits, we won't actually decode anything
        // these bits haven't converged yet
        for _ in 0..min_traceback_length {
            index = if index == 0 { cap - 1 } else { index - 1 };

            let bit = survivor_bit(index, best_path);
            let reg_bit = bit.wrapping_neg() & num_survivors;
            best_path = (best_path | reg_bit) >> 1;
        }

        // for the remaining bits, the path has converged, and we can safely decode
        let num_decodes = self.history_len - min_traceback_length as usize;
        for decoded in self.decode_buf.iter_mut().take(num_decodes) {
            index = if index == 0 { cap - 1 } else { index - 1 };

            let bit = survivor_bit(index, best_path);
            let reg_bit = bit.wrapping_neg() & num_survivors;
            *decoded = bit as u8;
            best_path = (best_path | reg_bit) >> 1;
        }

        bit_writer.write_iter(self.decode_buf[..num_decodes].iter().rev());
        self.history_len -= num_decodes;
    }

    fn flush(&mut self, bit_writer: &mut BitWriter) {
        self.traceback(0, 0, bit_writer);
    }

    fn history_offset(&self) -> usize {
        self.history_index * Self::hist_stride()
    }

    fn swap_errors(&mut self) {
        std::mem::swap(&mut self.errors, &mut self.previous_errors);
    }

    fn dispatch_forced(
        &mut self,
        path: ForcedPath,
        distance_fill: &mut ConvolutionalError,
        num_encoded_bits: usize,
        decoded: &mut BitWriter,
    ) {
        let register_128_ok = ORDER >= 5 && ORDER <= 8 && RATE <= 3;
        let register_256_ok = ORDER >= 6 && ORDER <= 9 && RATE <= 3;
        let register_512_ok = ORDER >= 7 && ORDER <= 10 && RATE <= 3;

        match path {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            ForcedPath::RegisterAvx512 => {
                if const { RATE <= 3 } {
                    assert!(
                        register_512_ok,
                        "ForcedPath::RegisterAvx512 not instantiable at rate={RATE} order={ORDER} (needs order 7..=10, rate<=3)"
                    );
                    assert!(self.enable_arch.avx512, "ForcedPath::RegisterAvx512 needs AVX-512");
                    assert!(
                        !self.distance_shuffle.shuffle32.is_empty(),
                        "ForcedPath::RegisterAvx512 needs shuffle32 geometry (rate<=3) at order={ORDER}"
                    );
                    unsafe { self.decode_body_register_avx512(distance_fill, num_encoded_bits, decoded) };
                } else {
                    panic!("ForcedPath::RegisterAvx512 not instantiable at rate={RATE} order={ORDER} (needs order 7..=10, rate<=3)");
                }
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            ForcedPath::RegisterAvx2 => {
                if const { RATE <= 3 } {
                    assert!(
                        register_256_ok,
                        "ForcedPath::RegisterAvx2 not instantiable at rate={RATE} order={ORDER} (needs order 6..=9, rate<=3)"
                    );
                    assert!(self.enable_arch.avx2, "ForcedPath::RegisterAvx2 needs AVX2");
                    assert!(
                        !self.distance_shuffle.shuffle16.is_empty(),
                        "ForcedPath::RegisterAvx2 needs shuffle16 geometry (rate<=3) at order={ORDER}"
                    );
                    unsafe { self.decode_body_register_avx2(distance_fill, num_encoded_bits, decoded) };
                } else {
                    panic!("ForcedPath::RegisterAvx2 not instantiable at rate={RATE} order={ORDER} (needs order 6..=9, rate<=3)");
                }
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            ForcedPath::RegisterSse41 => {
                if const { RATE <= 3 } {
                    assert!(
                        register_128_ok,
                        "ForcedPath::RegisterSse41 not instantiable at rate={RATE} order={ORDER} (needs order 5..=8, rate<=3)"
                    );
                    assert!(self.enable_arch.sse41, "ForcedPath::RegisterSse41 needs SSE4.1");
                    unsafe { self.decode_body_register_sse41(distance_fill, num_encoded_bits, decoded) };
                } else {
                    panic!("ForcedPath::RegisterSse41 not instantiable at rate={RATE} order={ORDER} (needs order 5..=8, rate<=3)");
                }
            }
            ForcedPath::Register128 => {
                if const { RATE <= 3 } {
                    assert!(
                        register_128_ok,
                        "ForcedPath::Register128 not instantiable at rate={RATE} order={ORDER} (needs order 5..=8, rate<=3)"
                    );
                    self.decode_body_register::<Lane128>(distance_fill, num_encoded_bits, decoded);
                } else {
                    panic!("ForcedPath::Register128 not instantiable at rate={RATE} order={ORDER} (needs order 5..=8, rate<=3)");
                }
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            ForcedPath::ShuffleAvx512 => {
                if const { RATE <= 3 } {
                    assert!(self.enable_arch.avx512, "ForcedPath::ShuffleAvx512 needs AVX-512");
                    assert!(
                        !self.distance_shuffle.shuffle32.is_empty(),
                        "ForcedPath::ShuffleAvx512 geometry unavailable at rate={RATE} (needs rate<=3)"
                    );
                    assert!(
                        Self::num_survivors() >= Lane512::LANES,
                        "ForcedPath::ShuffleAvx512 requires wider order"
                    );
                    unsafe { self.decode_body_shuffle_avx512(distance_fill, num_encoded_bits, decoded) };
                } else {
                    panic!("ForcedPath::ShuffleAvx512 not instantiable at rate={RATE} order={ORDER} (needs rate<=3)");
                }
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            ForcedPath::ShuffleAvx2 => {
                if const { RATE <= 3 } {
                    assert!(self.enable_arch.avx2, "ForcedPath::ShuffleAvx2 needs AVX2");
                    assert!(
                        !self.distance_shuffle.shuffle16.is_empty(),
                        "ForcedPath::ShuffleAvx2 geometry unavailable at rate={RATE} (needs rate<=3)"
                    );
                    assert!(
                        Self::num_survivors() >= Lane256::LANES,
                        "ForcedPath::ShuffleAvx2 requires wider order"
                    );
                    unsafe { self.decode_body_shuffle_avx2(distance_fill, num_encoded_bits, decoded) };
                } else {
                    panic!("ForcedPath::ShuffleAvx2 not instantiable at rate={RATE} order={ORDER} (needs rate<=3)");
                }
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            ForcedPath::ShuffleSse41 => {
                if const { RATE <= 3 } {
                    assert!(self.enable_arch.sse41, "ForcedPath::ShuffleSse41 needs SSE4.1");
                    assert!(
                        RATE <= 3 && !self.distance_shuffle.shuffle8.is_empty(),
                        "ForcedPath::ShuffleSse41 geometry unavailable at rate={RATE} (needs rate<=3)"
                    );
                    assert!(
                        Self::num_survivors() >= Lane128::LANES,
                        "ForcedPath::ShuffleSse41 requires wider order"
                    );
                    unsafe { self.decode_body_shuffle_sse41(distance_fill, num_encoded_bits, decoded) };
                } else {
                    panic!("ForcedPath::ShuffleSse41 not instantiable at rate={RATE} order={ORDER} (needs rate<=3)");
                }
            }
            ForcedPath::Shuffle128 => {
                if const { RATE <= 3 } {
                    assert!(
                        RATE <= 3 && !self.distance_shuffle.shuffle8.is_empty(),
                        "ForcedPath::Shuffle128 geometry unavailable at rate={RATE} (needs rate<=3)"
                    );
                    assert!(
                        Self::num_survivors() >= Lane128::LANES,
                        "ForcedPath::Shuffle128 requires wider order"
                    );
                    self.decode_body_shuffle::<Lane128>(distance_fill, num_encoded_bits, decoded);
                } else {
                    panic!("ForcedPath::Shuffle128 not instantiable at rate={RATE} order={ORDER} (needs rate<=3)");
                }
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            ForcedPath::PermuteAvx512 => {
                if const { RATE <= 6 } {
                    assert!(self.enable_arch.avx512, "ForcedPath::PermuteAvx512 needs AVX-512");
                    assert!(
                        Self::num_survivors() >= Lane512::LANES,
                        "ForcedPath::PermuteAvx512 requires wider order"
                    );
                    unsafe { self.decode_body_permute_avx512(distance_fill, num_encoded_bits, decoded) };
                } else {
                    panic!("ForcedPath::PermuteAvx512 not instantiable at rate={RATE} order={ORDER} (needs rate<=6)");
                }
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            ForcedPath::OctLookupAvx512 => {
                assert!(self.enable_arch.avx512, "ForcedPath::OctLookupAvx512 needs AVX-512");
                assert!(
                    Self::num_survivors() >= Lane512::LANES,
                    "ForcedPath::OctLookupAvx512 requires wider order"
                );
                unsafe { self.decode_body_oct_lookup_avx512(distance_fill, num_encoded_bits, decoded) };
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            ForcedPath::OctLookupAvx2 => {
                assert!(self.enable_arch.avx2, "ForcedPath::OctLookupAvx2 needs AVX2");
                assert!(
                    Self::num_survivors() >= Lane256::LANES,
                    "ForcedPath::OctLookupAvx2 requires wider order"
                );
                unsafe { self.decode_body_oct_lookup_avx2(distance_fill, num_encoded_bits, decoded) };
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            ForcedPath::OctLookupSse41 => {
                assert!(self.enable_arch.sse41, "ForcedPath::OctLookupSse41 needs SSE4.1");
                assert!(
                    Self::num_survivors() >= Lane128::LANES,
                    "ForcedPath::OctLookupSse41 requires wider order"
                );
                unsafe { self.decode_body_oct_lookup_sse41(distance_fill, num_encoded_bits, decoded) };
            }
            ForcedPath::OctLookup128 => {
                self.decode_body_oct_lookup::<Lane128>(distance_fill, num_encoded_bits, decoded);
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            other => panic!("ForcedPath::{other:?} is x86-only; not available on this target"),
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn decode_body_oct_lookup_avx512(
        &mut self,
        distance_fill: &mut ConvolutionalError,
        num_encoded_bits: usize,
        decoded: &mut BitWriter,
    ) {
        self.decode_body_oct_lookup::<Lane512>(distance_fill, num_encoded_bits, decoded);
    }

    #[cfg(not(target_arch = "x86_64"))]
    unsafe fn decode_body_oct_lookup_avx512(
        &mut self,
        _distance_fill: &mut ConvolutionalError,
        _num_encoded_bits: usize,
        _decoded: &mut BitWriter,
    ) {
        unreachable!("AVX-512 is unavailable off x86_64")
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn decode_body_oct_lookup_avx2(
        &mut self,
        distance_fill: &mut ConvolutionalError,
        num_encoded_bits: usize,
        decoded: &mut BitWriter,
    ) {
        self.decode_body_oct_lookup::<Lane256>(distance_fill, num_encoded_bits, decoded);
    }

    // take advantage of SSE4.1's pminuw/pmaxuw (still 128-bit)
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse4.1")]
    unsafe fn decode_body_oct_lookup_sse41(
        &mut self,
        distance_fill: &mut ConvolutionalError,
        num_encoded_bits: usize,
        decoded: &mut BitWriter,
    ) {
        self.decode_body_oct_lookup::<Lane128>(distance_fill, num_encoded_bits, decoded);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn decode_body_shuffle_avx512(
        &mut self,
        distance_fill: &mut ConvolutionalError,
        num_encoded_bits: usize,
        decoded: &mut BitWriter,
    ) {
        self.decode_body_shuffle::<Lane512>(distance_fill, num_encoded_bits, decoded);
    }

    #[cfg(not(target_arch = "x86_64"))]
    unsafe fn decode_body_shuffle_avx512(
        &mut self,
        _distance_fill: &mut ConvolutionalError,
        _num_encoded_bits: usize,
        _decoded: &mut BitWriter,
    ) {
        unreachable!("AVX-512 is unavailable off x86_64")
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn decode_body_shuffle_avx2(
        &mut self,
        distance_fill: &mut ConvolutionalError,
        num_encoded_bits: usize,
        decoded: &mut BitWriter,
    ) {
        self.decode_body_shuffle::<Lane256>(distance_fill, num_encoded_bits, decoded);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse4.1")]
    unsafe fn decode_body_shuffle_sse41(
        &mut self,
        distance_fill: &mut ConvolutionalError,
        num_encoded_bits: usize,
        decoded: &mut BitWriter,
    ) {
        self.decode_body_shuffle::<Lane128>(distance_fill, num_encoded_bits, decoded);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn decode_body_permute_avx512(
        &mut self,
        distance_fill: &mut ConvolutionalError,
        num_encoded_bits: usize,
        decoded: &mut BitWriter,
    ) {
        self.decode_body_permute::<Lane512>(distance_fill, num_encoded_bits, decoded);
    }

    #[cfg(not(target_arch = "x86_64"))]
    unsafe fn decode_body_permute_avx512(
        &mut self,
        _distance_fill: &mut ConvolutionalError,
        _num_encoded_bits: usize,
        _decoded: &mut BitWriter,
    ) {
        unreachable!("AVX-512 is unavailable off x86_64")
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn decode_body_register_avx512(
        &mut self,
        distance_fill: &mut ConvolutionalError,
        num_encoded_bits: usize,
        decoded: &mut BitWriter,
    ) {
        self.decode_body_register::<Lane512>(distance_fill, num_encoded_bits, decoded);
    }

    #[cfg(not(target_arch = "x86_64"))]
    unsafe fn decode_body_register_avx512(
        &mut self,
        _distance_fill: &mut ConvolutionalError,
        _num_encoded_bits: usize,
        _decoded: &mut BitWriter,
    ) {
        unreachable!("AVX-512 is unavailable off x86_64")
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn decode_body_register_avx2(
        &mut self,
        distance_fill: &mut ConvolutionalError,
        num_encoded_bits: usize,
        decoded: &mut BitWriter,
    ) {
        self.decode_body_register::<Lane256>(distance_fill, num_encoded_bits, decoded);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse4.1")]
    unsafe fn decode_body_register_sse41(
        &mut self,
        distance_fill: &mut ConvolutionalError,
        num_encoded_bits: usize,
        decoded: &mut BitWriter,
    ) {
        self.decode_body_register::<Lane128>(distance_fill, num_encoded_bits, decoded);
    }
}

#[cfg(test)]
mod tests {
    use super::{DecoderArch, ForcedPath, SimdDecoder};
    use crate::convolutional::sim::{bpsk_params, flip_with_interval, Testbench};
    use crate::convolutional::{DecodeError, Decoder, Encoder};
    use crate::util::Rng;

    #[derive(Clone, Copy, Debug)]
    enum Arch {
        Sse,
        Avx2,
        Avx512,
    }

    impl Arch {
        fn caps(self) -> DecoderArch {
            match self {
                Arch::Sse => DecoderArch::Sse41,
                Arch::Avx2 => DecoderArch::Avx2,
                Arch::Avx512 => DecoderArch::Avx512,
            }
        }
    }

    fn host_archs() -> Vec<Arch> {
        let mut v = vec![Arch::Sse];
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                v.push(Arch::Avx2);
            }
            if std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512vl")
            {
                v.push(Arch::Avx512);
            }
        }
        v
    }

    fn host_supports(path: ForcedPath) -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let (has_sse41, has_avx2, has_avx512) = (
            std::is_x86_feature_detected!("sse4.1"),
            std::is_x86_feature_detected!("avx2"),
            std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512vl"),
        );
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let (has_sse41, has_avx2, has_avx512) = (false, false, false);

        match path {
            ForcedPath::Register128 | ForcedPath::Shuffle128 | ForcedPath::OctLookup128 => true,
            ForcedPath::RegisterSse41 | ForcedPath::ShuffleSse41 | ForcedPath::OctLookupSse41 => has_sse41,
            ForcedPath::RegisterAvx2 | ForcedPath::ShuffleAvx2 | ForcedPath::OctLookupAvx2 => has_avx2,
            ForcedPath::RegisterAvx512
            | ForcedPath::ShuffleAvx512
            | ForcedPath::PermuteAvx512
            | ForcedPath::OctLookupAvx512 => has_avx512,
        }
    }

    #[test]
    #[should_panic(expected = "generator polynomials")]
    fn new_panics_on_wrong_poly_count() {
        // rate 2 needs 2 polynomials, only 1 given
        SimdDecoder::<2, 7>::new(&[0o155]);
    }

    #[test]
    fn rejects_malformed_inputs() {
        let mut d = SimdDecoder::<2, 7>::new(&[0o155, 0o117]);
        let mut out = vec![0u8; 64];
        // must be a multiple of rate.
        assert!(matches!(
            d.decode_hard(&[0u8; 16], 7, &mut out),
            Err(DecodeError::InvalidLength { .. })
        ));
        // longer than input buffer holds
        assert!(matches!(
            d.decode_hard(&[0u8; 2], 64, &mut out),
            Err(DecodeError::InvalidLength { .. })
        ));
        // too short to decode
        assert!(matches!(
            d.decode_hard(&[0u8; 4], 8, &mut out),
            Err(DecodeError::InvalidLength { .. })
        ));
        // too short to decode
        assert!(matches!(
            d.decode_soft(&[0u8; 4], &mut out),
            Err(DecodeError::InvalidLength { .. })
        ));
    }

    fn assert_decode_matches<const RATE: u32, const ORDER: u32>(
        polys: &[u16],
        msg_len: usize,
        seed: u64,
        arch: Arch,
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

        let mut simd_out = vec![0u8; msg_len];
        let mut simd = SimdDecoder::<RATE, ORDER>::new(polys).with_max_arch(arch.caps());

        if hard {
            simd.decode_hard(&encoded, enc_bits, &mut simd_out).unwrap();
        } else {
            let mut soft = vec![0u8; enc_bits];
            for (i, s) in soft.iter_mut().enumerate() {
                *s = if encoded[i / 8] & (0x80 >> (i % 8)) != 0 {
                    255
                } else {
                    0
                };
            }
            simd.decode_soft(&soft, &mut simd_out).unwrap();
        }

        let mode = if hard { "hard" } else { "soft" };
        assert_eq!(
            &simd_out, &msg,
            "SIMD decode wrong: rate={RATE} order={ORDER} len={msg_len} mode={mode} \
             clean={clean} seed={seed} arch={arch:?}"
        );
    }

    fn assert_decode_matches_all_archs<const RATE: u32, const ORDER: u32>(polys: &[u16]) {
        for arch in host_archs() {
            for clean in [false, true] {
                for hard in [false, true] {
                    for (msg_len, seeds) in [(256usize, 32u64), (1500usize, 8u64)] {
                        for seed in 1..=seeds {
                            assert_decode_matches::<RATE, ORDER>(polys, msg_len, seed, arch, hard, clean);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn decode_matches_all_archs_2_4() {
        assert_decode_matches_all_archs::<2, 4>(&[0o017, 0o013]);
    }
    #[test]
    fn decode_matches_all_archs_2_5() {
        assert_decode_matches_all_archs::<2, 5>(&[0o027, 0o023]);
    }
    #[test]
    fn decode_matches_all_archs_2_6() {
        assert_decode_matches_all_archs::<2, 6>(&[0o065, 0o057]);
    }
    #[test]
    fn decode_matches_all_archs_2_7() {
        assert_decode_matches_all_archs::<2, 7>(&[0o155, 0o117]);
    }
    #[test]
    fn decode_matches_all_archs_2_8() {
        assert_decode_matches_all_archs::<2, 8>(&[0o367, 0o225]);
    }
    #[test]
    fn decode_matches_all_archs_2_9() {
        assert_decode_matches_all_archs::<2, 9>(&[0o657, 0o435]);
    }
    #[test]
    fn decode_matches_all_archs_2_10() {
        assert_decode_matches_all_archs::<2, 10>(&[0o1627, 0o1063]);
    }
    #[test]
    fn decode_matches_all_archs_3_9() {
        assert_decode_matches_all_archs::<3, 9>(&[0o755, 0o633, 0o447]);
    }
    #[test]
    fn decode_matches_all_archs_4_7() {
        assert_decode_matches_all_archs::<4, 7>(&[0o133, 0o175, 0o107, 0o101]);
    }
    #[test]
    fn decode_matches_all_archs_6_15() {
        assert_decode_matches_all_archs::<6, 15>(&[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537]);
    }
    #[test]
    fn decode_matches_all_archs_7_7() {
        assert_decode_matches_all_archs::<7, 7>(&[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145]);
    }
    #[test]
    fn decode_matches_all_archs_8_7() {
        assert_decode_matches_all_archs::<8, 7>(&[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145, 0o171]);
    }

    fn assert_decode_path_matches<const RATE: u32, const ORDER: u32>(
        polys: &[u16],
        msg_len: usize,
        seed: u64,
        path: ForcedPath,
        hard: bool,
        clean: bool,
    ) {
        if !host_supports(path) {
            eprintln!("SKIP {RATE}/{ORDER} {path:?}: host lacks the required arch");
            return;
        }

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

        let mut simd_out = vec![0u8; msg_len];
        let mut simd = SimdDecoder::<RATE, ORDER>::new(polys).with_path(Some(path));

        if hard {
            simd.decode_hard(&encoded, enc_bits, &mut simd_out).unwrap();
        } else {
            let mut soft = vec![0u8; enc_bits];
            for (i, s) in soft.iter_mut().enumerate() {
                *s = if encoded[i / 8] & (0x80 >> (i % 8)) != 0 {
                    255
                } else {
                    0
                };
            }
            simd.decode_soft(&soft, &mut simd_out).unwrap();
        }

        let mode = if hard { "hard" } else { "soft" };
        assert_eq!(
            &simd_out, &msg,
            "SIMD decode wrong: rate={RATE} order={ORDER} len={msg_len} mode={mode} \
             clean={clean} seed={seed} path={path:?}"
        );
    }

    fn assert_decode_path_matches_all<const RATE: u32, const ORDER: u32>(polys: &[u16], path: ForcedPath) {
        for clean in [false, true] {
            for hard in [false, true] {
                for (msg_len, seeds) in [(256usize, 32u64), (1500usize, 8u64)] {
                    for seed in 1..=seeds {
                        assert_decode_path_matches::<RATE, ORDER>(polys, msg_len, seed, path, clean, hard);
                    }
                }
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[rustfmt::skip]
    #[test]
    fn path_oct_lookup_avx512() {
        assert_decode_path_matches_all::<2, 6>(&[0o65, 0o57], ForcedPath::OctLookupAvx512);
        assert_decode_path_matches_all::<2, 7>(&[0o155, 0o117], ForcedPath::OctLookupAvx512);
        assert_decode_path_matches_all::<2, 9>(&[0o657, 0o435], ForcedPath::OctLookupAvx512);
        assert_decode_path_matches_all::<3, 9>(&[0o755, 0o633, 0o447], ForcedPath::OctLookupAvx512);
        assert_decode_path_matches_all::<4, 7>(&[0o133, 0o175, 0o107, 0o101], ForcedPath::OctLookupAvx512);
        assert_decode_path_matches_all::<6, 15>(&[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537], ForcedPath::OctLookupAvx512);
        assert_decode_path_matches_all::<7, 7>(&[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145], ForcedPath::OctLookupAvx512);
        assert_decode_path_matches_all::<8, 7>(&[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145, 0o171], ForcedPath::OctLookupAvx512);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[rustfmt::skip]
    #[test]
    fn path_oct_lookup_avx2() {
        assert_decode_path_matches_all::<2, 5>(&[0o27, 0o23], ForcedPath::OctLookupAvx2);
        assert_decode_path_matches_all::<2, 6>(&[0o65, 0o57], ForcedPath::OctLookupAvx2);
        assert_decode_path_matches_all::<2, 7>(&[0o155, 0o117], ForcedPath::OctLookupAvx2);
        assert_decode_path_matches_all::<2, 9>(&[0o657, 0o435], ForcedPath::OctLookupAvx2);
        assert_decode_path_matches_all::<3, 9>(&[0o755, 0o633, 0o447], ForcedPath::OctLookupAvx2);
        assert_decode_path_matches_all::<4, 7>(&[0o133, 0o175, 0o107, 0o101], ForcedPath::OctLookupAvx2);
        assert_decode_path_matches_all::<6, 15>(&[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537], ForcedPath::OctLookupAvx2);
        assert_decode_path_matches_all::<7, 7>(&[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145], ForcedPath::OctLookupAvx2);
        assert_decode_path_matches_all::<8, 7>(&[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145, 0o171], ForcedPath::OctLookupAvx2);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[rustfmt::skip]
    #[test]
    fn path_oct_lookup_sse() {
        assert_decode_path_matches_all::<2, 4>(&[0o17, 0o13], ForcedPath::OctLookupSse41);
        assert_decode_path_matches_all::<2, 5>(&[0o27, 0o23], ForcedPath::OctLookupSse41);
        assert_decode_path_matches_all::<2, 6>(&[0o65, 0o57], ForcedPath::OctLookupSse41);
        assert_decode_path_matches_all::<2, 7>(&[0o155, 0o117], ForcedPath::OctLookupSse41);
        assert_decode_path_matches_all::<2, 9>(&[0o657, 0o435], ForcedPath::OctLookupSse41);
        assert_decode_path_matches_all::<3, 9>(&[0o755, 0o633, 0o447], ForcedPath::OctLookupSse41);
        assert_decode_path_matches_all::<4, 7>(&[0o133, 0o175, 0o107, 0o101], ForcedPath::OctLookupSse41);
        assert_decode_path_matches_all::<6, 15>(&[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537], ForcedPath::OctLookupSse41);
        assert_decode_path_matches_all::<7, 7>(&[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145], ForcedPath::OctLookupSse41);
        assert_decode_path_matches_all::<8, 7>(&[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145, 0o171], ForcedPath::OctLookupSse41);
    }

    #[rustfmt::skip]
    #[test]
    fn path_oct_lookup_128() {
        assert_decode_path_matches_all::<2, 4>(&[0o17, 0o13], ForcedPath::OctLookup128);
        assert_decode_path_matches_all::<2, 5>(&[0o27, 0o23], ForcedPath::OctLookup128);
        assert_decode_path_matches_all::<2, 6>(&[0o65, 0o57], ForcedPath::OctLookup128);
        assert_decode_path_matches_all::<2, 7>(&[0o155, 0o117], ForcedPath::OctLookup128);
        assert_decode_path_matches_all::<2, 9>(&[0o657, 0o435], ForcedPath::OctLookup128);
        assert_decode_path_matches_all::<3, 9>(&[0o755, 0o633, 0o447], ForcedPath::OctLookup128);
        assert_decode_path_matches_all::<4, 7>(&[0o133, 0o175, 0o107, 0o101], ForcedPath::OctLookup128);
        assert_decode_path_matches_all::<6, 15>(&[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537], ForcedPath::OctLookup128);
        assert_decode_path_matches_all::<7, 7>(&[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145], ForcedPath::OctLookup128);
        assert_decode_path_matches_all::<8, 7>(&[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145, 0o171], ForcedPath::OctLookup128);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[rustfmt::skip]
    #[test]
    fn path_shuffle_avx512() {
        assert_decode_path_matches_all::<2, 6>(&[0o65, 0o57], ForcedPath::ShuffleAvx512);
        assert_decode_path_matches_all::<2, 7>(&[0o155, 0o117], ForcedPath::ShuffleAvx512);
        assert_decode_path_matches_all::<2, 9>(&[0o657, 0o435], ForcedPath::ShuffleAvx512);
        assert_decode_path_matches_all::<2, 11>(&[0o4335, 0o5723], ForcedPath::ShuffleAvx512);
        assert_decode_path_matches_all::<2, 13>(&[0o21645, 0o35661], ForcedPath::ShuffleAvx512);
        assert_decode_path_matches_all::<3, 9>(&[0o755, 0o633, 0o447], ForcedPath::ShuffleAvx512);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[rustfmt::skip]
    #[test]
    fn path_shuffle_avx2() {
        assert_decode_path_matches_all::<2, 5>(&[0o27, 0o23], ForcedPath::ShuffleAvx2);
        assert_decode_path_matches_all::<2, 6>(&[0o65, 0o57], ForcedPath::ShuffleAvx2);
        assert_decode_path_matches_all::<2, 7>(&[0o155, 0o117], ForcedPath::ShuffleAvx2);
        assert_decode_path_matches_all::<2, 9>(&[0o657, 0o435], ForcedPath::ShuffleAvx2);
        assert_decode_path_matches_all::<2, 11>(&[0o4335, 0o5723], ForcedPath::ShuffleAvx2);
        assert_decode_path_matches_all::<2, 13>(&[0o21645, 0o35661], ForcedPath::ShuffleAvx2);
        assert_decode_path_matches_all::<3, 9>(&[0o755, 0o633, 0o447], ForcedPath::ShuffleAvx2);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[rustfmt::skip]
    #[test]
    fn path_shuffle_sse() {
        assert_decode_path_matches_all::<2, 4>(&[0o17, 0o13], ForcedPath::ShuffleSse41);
        assert_decode_path_matches_all::<2, 5>(&[0o27, 0o23], ForcedPath::ShuffleSse41);
        assert_decode_path_matches_all::<2, 6>(&[0o65, 0o57], ForcedPath::ShuffleSse41);
        assert_decode_path_matches_all::<2, 7>(&[0o155, 0o117], ForcedPath::ShuffleSse41);
        assert_decode_path_matches_all::<2, 9>(&[0o657, 0o435], ForcedPath::ShuffleSse41);
        assert_decode_path_matches_all::<2, 11>(&[0o4335, 0o5723], ForcedPath::ShuffleSse41);
        assert_decode_path_matches_all::<2, 13>(&[0o21645, 0o35661], ForcedPath::ShuffleSse41);
        assert_decode_path_matches_all::<3, 9>(&[0o755, 0o633, 0o447], ForcedPath::ShuffleSse41);
    }

    #[rustfmt::skip]
    #[test]
    fn path_shuffle_128() {
        assert_decode_path_matches_all::<2, 4>(&[0o17, 0o13], ForcedPath::Shuffle128);
        assert_decode_path_matches_all::<2, 5>(&[0o27, 0o23], ForcedPath::Shuffle128);
        assert_decode_path_matches_all::<2, 6>(&[0o65, 0o57], ForcedPath::Shuffle128);
        assert_decode_path_matches_all::<2, 7>(&[0o155, 0o117], ForcedPath::Shuffle128);
        assert_decode_path_matches_all::<2, 9>(&[0o657, 0o435], ForcedPath::Shuffle128);
        assert_decode_path_matches_all::<2, 11>(&[0o4335, 0o5723], ForcedPath::Shuffle128);
        assert_decode_path_matches_all::<2, 13>(&[0o21645, 0o35661], ForcedPath::Shuffle128);
        assert_decode_path_matches_all::<3, 9>(&[0o755, 0o633, 0o447], ForcedPath::Shuffle128);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[rustfmt::skip]
    #[test]
    fn path_permute_avx512() {
        assert_decode_path_matches_all::<2, 7>(&[0o155, 0o117], ForcedPath::PermuteAvx512);
        assert_decode_path_matches_all::<2, 9>(&[0o657, 0o435], ForcedPath::PermuteAvx512);
        assert_decode_path_matches_all::<2, 15>(&[0o56711, 0o75063], ForcedPath::PermuteAvx512);
        assert_decode_path_matches_all::<3, 9>(&[0o755, 0o633, 0o447], ForcedPath::PermuteAvx512);
        assert_decode_path_matches_all::<4, 7>(&[0o133, 0o175, 0o107, 0o101], ForcedPath::PermuteAvx512);
        assert_decode_path_matches_all::<6, 15>(&[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537], ForcedPath::PermuteAvx512);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[rustfmt::skip]
    #[test]
    fn path_register_avx512() {
        assert_decode_path_matches_all::<2, 7>(&[0o155, 0o117], ForcedPath::RegisterAvx512);
        assert_decode_path_matches_all::<2, 8>(&[0o367, 0o225], ForcedPath::RegisterAvx512);
        assert_decode_path_matches_all::<2, 9>(&[0o657, 0o435], ForcedPath::RegisterAvx512);
        assert_decode_path_matches_all::<2, 10>(&[0o1627, 0o1063], ForcedPath::RegisterAvx512);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[rustfmt::skip]
    #[test]
    fn path_register_avx2() {
        assert_decode_path_matches_all::<2, 6>(&[0o65, 0o57], ForcedPath::RegisterAvx2);
        assert_decode_path_matches_all::<2, 7>(&[0o155, 0o117], ForcedPath::RegisterAvx2);
        assert_decode_path_matches_all::<2, 8>(&[0o367, 0o225], ForcedPath::RegisterAvx2);
        assert_decode_path_matches_all::<2, 9>(&[0o657, 0o435], ForcedPath::RegisterAvx2);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[rustfmt::skip]
    #[test]
    fn path_register_sse() {
        assert_decode_path_matches_all::<2, 5>(&[0o27, 0o23], ForcedPath::RegisterSse41);
        assert_decode_path_matches_all::<2, 6>(&[0o65, 0o57], ForcedPath::RegisterSse41);
        assert_decode_path_matches_all::<2, 7>(&[0o155, 0o117], ForcedPath::RegisterSse41);
        assert_decode_path_matches_all::<2, 8>(&[0o367, 0o225], ForcedPath::RegisterSse41);
    }

    #[rustfmt::skip]
    #[test]
    fn path_register_128() {
        assert_decode_path_matches_all::<2, 5>(&[0o27, 0o23], ForcedPath::Register128);
        assert_decode_path_matches_all::<2, 6>(&[0o65, 0o57], ForcedPath::Register128);
        assert_decode_path_matches_all::<2, 7>(&[0o155, 0o117], ForcedPath::Register128);
        assert_decode_path_matches_all::<2, 8>(&[0o367, 0o225], ForcedPath::Register128);
    }

    fn try_decode_path<const RATE: u32, const ORDER: u32>(polys: &[u16], path: ForcedPath) {
        let msg_len = 64;
        let enc_bits = RATE as usize * (msg_len * 8 + ORDER as usize + 1);
        let mut rng = Rng::new(1);
        let soft: Vec<u8> = (0..enc_bits).map(|_| rng.next_u8()).collect();
        let mut out = vec![0u8; msg_len];
        let _ = SimdDecoder::<RATE, ORDER>::new(polys)
            .with_path(Some(path))
            .decode_soft(&soft, &mut out);
    }

    #[test]
    #[should_panic(expected = "not instantiable")]
    fn forcing_register_128_panics() {
        try_decode_path::<2, 4>(&[0o17, 0o13], ForcedPath::Register128);
    }

    #[test]
    #[should_panic(expected = "not instantiable")]
    fn forcing_shuffle_128_panics() {
        try_decode_path::<4, 7>(&[0o133, 0o175, 0o107, 0o101], ForcedPath::Shuffle128);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    #[should_panic(expected = "not instantiable")]
    fn forcing_register_sse41_panics() {
        if !host_supports(ForcedPath::RegisterSse41) {
            eprintln!("SKIP forcing_register_sse41_panics: host lacks SSE4.1");
            panic!("not instantiable");
        }
        try_decode_path::<2, 4>(&[0o17, 0o13], ForcedPath::RegisterSse41);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    #[should_panic(expected = "not instantiable")]
    fn forcing_shuffle_sse41_panics() {
        if !host_supports(ForcedPath::ShuffleSse41) {
            eprintln!("SKIP forcing_shuffle_sse41_panics: host lacks SSE4.1");
            panic!("not instantiable");
        }
        try_decode_path::<4, 7>(&[0o133, 0o175, 0o107, 0o101], ForcedPath::ShuffleSse41);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    #[should_panic(expected = "not instantiable")]
    fn forcing_register_avx2_panics() {
        if !host_supports(ForcedPath::RegisterAvx2) {
            eprintln!("SKIP forcing_register_avx2_panics: host lacks AVX2");
            panic!("not instantiable");
        }
        try_decode_path::<2, 4>(&[0o17, 0o13], ForcedPath::RegisterAvx2);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    #[should_panic(expected = "not instantiable")]
    fn forcing_shuffle_avx2_panics() {
        if !host_supports(ForcedPath::ShuffleAvx2) {
            eprintln!("SKIP forcing_shuffle_avx2_panics: host lacks AVX2");
            panic!("not instantiable");
        }
        try_decode_path::<4, 7>(&[0o133, 0o175, 0o107, 0o101], ForcedPath::ShuffleAvx2);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    #[should_panic(expected = "requires wider order")]
    fn forcing_oct_lookup_avx2_panics() {
        if !host_supports(ForcedPath::OctLookupAvx2) {
            eprintln!("SKIP forcing_oct_lookup_avx2_panics: host lacks AVX2");
            panic!("requires wider order");
        }
        try_decode_path::<2, 4>(&[0o17, 0o13], ForcedPath::OctLookupAvx2);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[should_panic(expected = "not instantiable")]
    fn forcing_register_avx512_panics() {
        if !host_supports(ForcedPath::RegisterAvx512) {
            eprintln!("SKIP forcing_register_avx512_panics: host lacks AVX-512");
            panic!("not instantiable");
        }
        try_decode_path::<2, 4>(&[0o17, 0o13], ForcedPath::RegisterAvx512);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[should_panic(expected = "not instantiable")]
    fn forcing_shuffle_avx512_panics() {
        if !host_supports(ForcedPath::ShuffleAvx512) {
            eprintln!("SKIP forcing_shuffle_avx512_panics: host lacks AVX-512");
            panic!("not instantiable");
        }
        try_decode_path::<4, 7>(&[0o133, 0o175, 0o107, 0o101], ForcedPath::ShuffleAvx512);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[should_panic(expected = "not instantiable")]
    fn forcing_permute_avx512_panics() {
        if !host_supports(ForcedPath::PermuteAvx512) {
            eprintln!("SKIP forcing_permute_avx512_panics: host lacks AVX-512");
            panic!("not instantiable");
        }
        try_decode_path::<7, 7>(
            &[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145],
            ForcedPath::PermuteAvx512,
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[should_panic(expected = "requires wider order")]
    fn forcing_oct_lookup_avx512_panics() {
        if !host_supports(ForcedPath::OctLookupAvx512) {
            eprintln!("SKIP forcing_oct_lookup_avx512_panics: host lacks AVX-512");
            panic!("requires wider order");
        }
        try_decode_path::<2, 4>(&[0o17, 0o13], ForcedPath::OctLookupAvx512);
    }

    fn assert_simd_matches_scalar_noise<const RATE: u32, const ORDER: u32>(
        polys: &[u16],
        path: Option<ForcedPath>,
        eb_n0_db: f64,
        bytes: usize,
        hard: bool,
    ) {
        if let Some(p) = path {
            if !host_supports(p) {
                eprintln!("SKIP noise {RATE}/{ORDER} {p:?}: host lacks the required arch");
                return;
            }
        }

        const MIN_ERRS: usize = 8000;
        let (volt, bit_energy) = bpsk_params(RATE);

        let mut msg = vec![0u8; bytes];
        let mut bench = Testbench::new(RATE, ORDER, polys, bytes);
        let mut channel = vec![0u8; bench.enc_bits()];
        let mut scalar_out = vec![0u8; bytes];
        let mut simd_out = vec![0u8; bytes];
        let mut rng = Rng::new(0x1234_5678_9ABC_DEF0);
        let mut scalar = Decoder::new(RATE, ORDER, polys);
        let mut simd = SimdDecoder::<RATE, ORDER>::new(polys).with_path(path);

        for b in msg.iter_mut() {
            *b = rng.next_u8();
        }

        bench.build_noise(eb_n0_db, bit_energy, &mut rng);
        let uncoded_flips = bench.bpsk_with_noise_soft(&msg, volt, &mut channel);

        if hard {
            let enc_bits = bench.enc_bits();
            let mut encoded = vec![0u8; enc_bits.div_ceil(8)];
            for (i, &s) in channel.iter().enumerate() {
                if s >= 128 {
                    encoded[i / 8] |= 0x80 >> (i % 8);
                }
            }
            scalar.decode_hard(&encoded, enc_bits, &mut scalar_out).unwrap();
            simd.decode_hard(&encoded, enc_bits, &mut simd_out).unwrap();
        } else {
            scalar.decode_soft(&channel, &mut scalar_out).unwrap();
            simd.decode_soft(&channel, &mut simd_out).unwrap();
        }

        assert!(
            uncoded_flips >= MIN_ERRS,
            "{RATE}/{ORDER} {path:?}: only {uncoded_flips} uncoded errors at {eb_n0_db}dB over {bytes}B. Minimum of \
             {MIN_ERRS} error events required. Lower Eb/N0 or raise bytes."
        );

        let mode = if hard { "hard" } else { "soft" };
        assert_eq!(
            &scalar_out, &simd_out,
            "{RATE}/{ORDER} {path:?} {mode}: SIMD output differs from scalar at {eb_n0_db}dB over {bytes}B."
        );
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_dispatch_2_5() {
        assert_simd_matches_scalar_noise::<2, 5>(&[0o027, 0o023], None, 3.0, 300_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_oct_lookup_2_5() {
        assert_simd_matches_scalar_noise::<2, 5>(&[0o027, 0o023], Some(ForcedPath::OctLookupAvx2), 3.0, 300_000, false);
        assert_simd_matches_scalar_noise::<2, 5>(&[0o027, 0o023], Some(ForcedPath::OctLookupSse41), 3.0, 300_000, false);
        assert_simd_matches_scalar_noise::<2, 5>(&[0o027, 0o023], Some(ForcedPath::OctLookup128), 3.0, 300_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_shuffle_2_5() {
        assert_simd_matches_scalar_noise::<2, 5>(&[0o027, 0o023], Some(ForcedPath::ShuffleAvx2), 3.0, 300_000, false);
        assert_simd_matches_scalar_noise::<2, 5>(&[0o027, 0o023], Some(ForcedPath::ShuffleSse41), 3.0, 300_000, false);
        assert_simd_matches_scalar_noise::<2, 5>(&[0o027, 0o023], Some(ForcedPath::Shuffle128), 3.0, 300_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_register_2_5() {
        assert_simd_matches_scalar_noise::<2, 5>(&[0o027, 0o023], Some(ForcedPath::RegisterSse41), 3.0, 300_000, false);
        assert_simd_matches_scalar_noise::<2, 5>(&[0o027, 0o023], Some(ForcedPath::Register128), 3.0, 300_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_dispatch_2_6() {
        assert_simd_matches_scalar_noise::<2, 6>(&[0o065, 0o057], None, 3.0, 300_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_oct_lookup_2_6() {
        assert_simd_matches_scalar_noise::<2, 6>(&[0o065, 0o057], Some(ForcedPath::OctLookupAvx512), 3.0, 300_000, false);
        assert_simd_matches_scalar_noise::<2, 6>(&[0o065, 0o057], Some(ForcedPath::OctLookupAvx2), 3.0, 300_000, false);
        assert_simd_matches_scalar_noise::<2, 6>(&[0o065, 0o057], Some(ForcedPath::OctLookupSse41), 3.0, 300_000, false);
        assert_simd_matches_scalar_noise::<2, 6>(&[0o065, 0o057], Some(ForcedPath::OctLookup128), 3.0, 300_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_shuffle_2_6() {
        assert_simd_matches_scalar_noise::<2, 6>(&[0o065, 0o057], Some(ForcedPath::ShuffleAvx512), 3.0, 300_000, false);
        assert_simd_matches_scalar_noise::<2, 6>(&[0o065, 0o057], Some(ForcedPath::ShuffleAvx2), 3.0, 300_000, false);
        assert_simd_matches_scalar_noise::<2, 6>(&[0o065, 0o057], Some(ForcedPath::ShuffleSse41), 3.0, 300_000, false);
        assert_simd_matches_scalar_noise::<2, 6>(&[0o065, 0o057], Some(ForcedPath::Shuffle128), 3.0, 300_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_permute_2_6() {
        assert_simd_matches_scalar_noise::<2, 6>(&[0o065, 0o057], Some(ForcedPath::PermuteAvx512), 3.0, 300_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_register_2_6() {
        assert_simd_matches_scalar_noise::<2, 6>(&[0o065, 0o057], Some(ForcedPath::RegisterAvx2), 3.0, 300_000, false);
        assert_simd_matches_scalar_noise::<2, 6>(&[0o065, 0o057], Some(ForcedPath::RegisterSse41), 3.0, 300_000, false);
        assert_simd_matches_scalar_noise::<2, 6>(&[0o065, 0o057], Some(ForcedPath::Register128), 3.0, 300_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_dispatch_2_7() {
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], None, 2.5, 300_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_oct_lookup_2_7() {
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::OctLookupAvx512), 2.5, 300_000, false);
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::OctLookupAvx2), 2.5, 300_000, false);
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::OctLookupSse41), 2.5, 300_000, false);
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::OctLookup128), 2.5, 300_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_shuffle_2_7() {
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::ShuffleAvx512), 2.5, 300_000, false);
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::ShuffleAvx2), 2.5, 300_000, false);
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::ShuffleSse41), 2.5, 300_000, false);
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::Shuffle128), 2.5, 300_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_permute_2_7() {
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::PermuteAvx512), 2.5, 300_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_register_2_7() {
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::RegisterAvx512), 2.5, 300_000, false);
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::RegisterAvx2), 2.5, 300_000, false);
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::RegisterSse41), 2.5, 300_000, false);
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::Register128), 2.5, 300_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_hard_dispatch_2_7() {
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], None, 4.0, 300_000, true);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_hard_oct_lookup_2_7() {
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::OctLookupAvx512), 4.0, 300_000, true);
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::OctLookupAvx2), 4.0, 300_000, true);
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::OctLookupSse41), 4.0, 300_000, true);
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::OctLookup128), 4.0, 300_000, true);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_hard_shuffle_2_7() {
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::ShuffleAvx512), 4.0, 300_000, true);
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::ShuffleAvx2), 4.0, 300_000, true);
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::ShuffleSse41), 4.0, 300_000, true);
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::Shuffle128), 4.0, 300_000, true);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_hard_permute_2_7() {
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::PermuteAvx512), 4.0, 300_000, true);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_hard_register_2_7() {
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::RegisterAvx512), 4.0, 300_000, true);
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::RegisterAvx2), 4.0, 300_000, true);
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::RegisterSse41), 4.0, 300_000, true);
        assert_simd_matches_scalar_noise::<2, 7>(&[0o155, 0o117], Some(ForcedPath::Register128), 4.0, 300_000, true);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_dispatch_2_8() {
        assert_simd_matches_scalar_noise::<2, 8>(&[0o367, 0o225], None, 2.5, 200_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_oct_lookup_2_8() {
        assert_simd_matches_scalar_noise::<2, 8>(&[0o367, 0o225], Some(ForcedPath::OctLookupAvx512), 2.5, 200_000, false);
        assert_simd_matches_scalar_noise::<2, 8>(&[0o367, 0o225], Some(ForcedPath::OctLookupAvx2), 2.5, 200_000, false);
        assert_simd_matches_scalar_noise::<2, 8>(&[0o367, 0o225], Some(ForcedPath::OctLookupSse41), 2.5, 200_000, false);
        assert_simd_matches_scalar_noise::<2, 8>(&[0o367, 0o225], Some(ForcedPath::OctLookup128), 2.5, 200_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_shuffle_2_8() {
        assert_simd_matches_scalar_noise::<2, 8>(&[0o367, 0o225], Some(ForcedPath::ShuffleAvx512), 2.5, 200_000, false);
        assert_simd_matches_scalar_noise::<2, 8>(&[0o367, 0o225], Some(ForcedPath::ShuffleAvx2), 2.5, 200_000, false);
        assert_simd_matches_scalar_noise::<2, 8>(&[0o367, 0o225], Some(ForcedPath::ShuffleSse41), 2.5, 200_000, false);
        assert_simd_matches_scalar_noise::<2, 8>(&[0o367, 0o225], Some(ForcedPath::Shuffle128), 2.5, 200_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_permute_2_8() {
        assert_simd_matches_scalar_noise::<2, 8>(&[0o367, 0o225], Some(ForcedPath::PermuteAvx512), 2.5, 200_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_register_2_8() {
        assert_simd_matches_scalar_noise::<2, 8>(&[0o367, 0o225], Some(ForcedPath::RegisterAvx512), 2.5, 200_000, false);
        assert_simd_matches_scalar_noise::<2, 8>(&[0o367, 0o225], Some(ForcedPath::RegisterAvx2), 2.5, 200_000, false);
        assert_simd_matches_scalar_noise::<2, 8>(&[0o367, 0o225], Some(ForcedPath::RegisterSse41), 2.5, 200_000, false);
        assert_simd_matches_scalar_noise::<2, 8>(&[0o367, 0o225], Some(ForcedPath::Register128), 2.5, 200_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_dispatch_2_9() {
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], None, 2.0, 100_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_oct_lookup_2_9() {
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], Some(ForcedPath::OctLookupAvx512), 2.0, 100_000, false);
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], Some(ForcedPath::OctLookupAvx2), 2.0, 100_000, false);
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], Some(ForcedPath::OctLookupSse41), 2.0, 100_000, false);
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], Some(ForcedPath::OctLookup128), 2.0, 100_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_shuffle_2_9() {
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], Some(ForcedPath::ShuffleAvx512), 2.0, 100_000, false);
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], Some(ForcedPath::ShuffleAvx2), 2.0, 100_000, false);
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], Some(ForcedPath::ShuffleSse41), 2.0, 100_000, false);
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], Some(ForcedPath::Shuffle128), 2.0, 100_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_permute_2_9() {
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], Some(ForcedPath::PermuteAvx512), 2.0, 100_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_register_2_9() {
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], Some(ForcedPath::RegisterAvx512), 2.0, 100_000, false);
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], Some(ForcedPath::RegisterAvx2), 2.0, 100_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_hard_dispatch_2_9() {
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], None, 3.5, 100_000, true);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_hard_oct_lookup_2_9() {
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], Some(ForcedPath::OctLookupAvx512), 3.5, 100_000, true);
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], Some(ForcedPath::OctLookupAvx2), 3.5, 100_000, true);
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], Some(ForcedPath::OctLookupSse41), 3.5, 100_000, true);
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], Some(ForcedPath::OctLookup128), 3.5, 100_000, true);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_hard_shuffle_2_9() {
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], Some(ForcedPath::ShuffleAvx512), 3.5, 100_000, true);
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], Some(ForcedPath::ShuffleAvx2), 3.5, 100_000, true);
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], Some(ForcedPath::ShuffleSse41), 3.5, 100_000, true);
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], Some(ForcedPath::Shuffle128), 3.5, 100_000, true);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_hard_permute_2_9() {
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], Some(ForcedPath::PermuteAvx512), 3.5, 100_000, true);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_hard_register_2_9() {
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], Some(ForcedPath::RegisterAvx512), 3.5, 100_000, true);
        assert_simd_matches_scalar_noise::<2, 9>(&[0o657, 0o435], Some(ForcedPath::RegisterAvx2), 3.5, 100_000, true);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_dispatch_2_15() {
        assert_simd_matches_scalar_noise::<2, 15>(&[0o56711, 0o75063], None, 1.2, 5_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_oct_lookup_2_15() {
        assert_simd_matches_scalar_noise::<2, 15>(&[0o56711, 0o75063], Some(ForcedPath::OctLookupAvx512), 1.2, 5_000, false);
        assert_simd_matches_scalar_noise::<2, 15>(&[0o56711, 0o75063], Some(ForcedPath::OctLookupAvx2), 1.2, 5_000, false);
        assert_simd_matches_scalar_noise::<2, 15>(&[0o56711, 0o75063], Some(ForcedPath::OctLookupSse41), 1.2, 5_000, false);
        assert_simd_matches_scalar_noise::<2, 15>(&[0o56711, 0o75063], Some(ForcedPath::OctLookup128), 1.2, 5_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_permute_2_15() {
        assert_simd_matches_scalar_noise::<2, 15>(&[0o56711, 0o75063], Some(ForcedPath::PermuteAvx512), 1.2, 5_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_dispatch_3_7() {
        assert_simd_matches_scalar_noise::<3, 7>(&[0o175, 0o145, 0o133], None, 2.5, 300_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_oct_lookup_3_7() {
        assert_simd_matches_scalar_noise::<3, 7>(&[0o175, 0o145, 0o133], Some(ForcedPath::OctLookupAvx512), 2.5, 300_000, false);
        assert_simd_matches_scalar_noise::<3, 7>(&[0o175, 0o145, 0o133], Some(ForcedPath::OctLookupAvx2), 2.5, 300_000, false);
        assert_simd_matches_scalar_noise::<3, 7>(&[0o175, 0o145, 0o133], Some(ForcedPath::OctLookupSse41), 2.5, 300_000, false);
        assert_simd_matches_scalar_noise::<3, 7>(&[0o175, 0o145, 0o133], Some(ForcedPath::OctLookup128), 2.5, 300_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_shuffle_3_7() {
        assert_simd_matches_scalar_noise::<3, 7>(&[0o175, 0o145, 0o133], Some(ForcedPath::ShuffleAvx512), 2.5, 300_000, false);
        assert_simd_matches_scalar_noise::<3, 7>(&[0o175, 0o145, 0o133], Some(ForcedPath::ShuffleAvx2), 2.5, 300_000, false);
        assert_simd_matches_scalar_noise::<3, 7>(&[0o175, 0o145, 0o133], Some(ForcedPath::ShuffleSse41), 2.5, 300_000, false);
        assert_simd_matches_scalar_noise::<3, 7>(&[0o175, 0o145, 0o133], Some(ForcedPath::Shuffle128), 2.5, 300_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_permute_3_7() {
        assert_simd_matches_scalar_noise::<3, 7>(&[0o175, 0o145, 0o133], Some(ForcedPath::PermuteAvx512), 2.5, 300_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_register_3_7() {
        assert_simd_matches_scalar_noise::<3, 7>(&[0o175, 0o145, 0o133], Some(ForcedPath::RegisterAvx512), 2.5, 300_000, false);
        assert_simd_matches_scalar_noise::<3, 7>(&[0o175, 0o145, 0o133], Some(ForcedPath::RegisterAvx2), 2.5, 300_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_dispatch_3_9() {
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], None, 2.0, 100_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_oct_lookup_3_9() {
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], Some(ForcedPath::OctLookupAvx512), 2.0, 100_000, false);
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], Some(ForcedPath::OctLookupAvx2), 2.0, 100_000, false);
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], Some(ForcedPath::OctLookupSse41), 2.0, 100_000, false);
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], Some(ForcedPath::OctLookup128), 2.0, 100_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_shuffle_3_9() {
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], Some(ForcedPath::ShuffleAvx512), 2.0, 100_000, false);
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], Some(ForcedPath::ShuffleAvx2), 2.0, 100_000, false);
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], Some(ForcedPath::ShuffleSse41), 2.0, 100_000, false);
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], Some(ForcedPath::Shuffle128), 2.0, 100_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_permute_3_9() {
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], Some(ForcedPath::PermuteAvx512), 2.0, 100_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_register_3_9() {
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], Some(ForcedPath::RegisterAvx512), 2.0, 100_000, false);
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], Some(ForcedPath::RegisterAvx2), 2.0, 100_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_hard_dispatch_3_9() {
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], None, 3.5, 100_000, true);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_hard_oct_lookup_3_9() {
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], Some(ForcedPath::OctLookupAvx512), 3.5, 100_000, true);
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], Some(ForcedPath::OctLookupAvx2), 3.5, 100_000, true);
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], Some(ForcedPath::OctLookupSse41), 3.5, 100_000, true);
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], Some(ForcedPath::OctLookup128), 3.5, 100_000, true);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_hard_shuffle_3_9() {
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], Some(ForcedPath::ShuffleAvx512), 3.5, 100_000, true);
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], Some(ForcedPath::ShuffleAvx2), 3.5, 100_000, true);
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], Some(ForcedPath::ShuffleSse41), 3.5, 100_000, true);
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], Some(ForcedPath::Shuffle128), 3.5, 100_000, true);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_hard_permute_3_9() {
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], Some(ForcedPath::PermuteAvx512), 3.5, 100_000, true);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_hard_register_3_9() {
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], Some(ForcedPath::RegisterAvx512), 3.5, 100_000, true);
        assert_simd_matches_scalar_noise::<3, 9>(&[0o755, 0o633, 0o447], Some(ForcedPath::RegisterAvx2), 3.5, 100_000, true);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_dispatch_4_7() {
        assert_simd_matches_scalar_noise::<4, 7>(&[0o133, 0o175, 0o107, 0o101], None, 2.0, 150_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_oct_lookup_4_7() {
        assert_simd_matches_scalar_noise::<4, 7>(&[0o133, 0o175, 0o107, 0o101], Some(ForcedPath::OctLookupAvx512), 2.0, 150_000, false);
        assert_simd_matches_scalar_noise::<4, 7>(&[0o133, 0o175, 0o107, 0o101], Some(ForcedPath::OctLookupAvx2), 2.0, 150_000, false);
        assert_simd_matches_scalar_noise::<4, 7>(&[0o133, 0o175, 0o107, 0o101], Some(ForcedPath::OctLookupSse41), 2.0, 150_000, false);
        assert_simd_matches_scalar_noise::<4, 7>(&[0o133, 0o175, 0o107, 0o101], Some(ForcedPath::OctLookup128), 2.0, 150_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_permute_4_7() {
        assert_simd_matches_scalar_noise::<4, 7>(&[0o133, 0o175, 0o107, 0o101], Some(ForcedPath::PermuteAvx512), 2.0, 150_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_dispatch_5_7() {
        assert_simd_matches_scalar_noise::<5, 7>(&[0o175, 0o145, 0o133, 0o117, 0o127], None, 2.0, 150_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_oct_lookup_5_7() {
        assert_simd_matches_scalar_noise::<5, 7>(&[0o175, 0o145, 0o133, 0o117, 0o127], Some(ForcedPath::OctLookupAvx512), 2.0, 150_000, false);
        assert_simd_matches_scalar_noise::<5, 7>(&[0o175, 0o145, 0o133, 0o117, 0o127], Some(ForcedPath::OctLookupAvx2), 2.0, 150_000, false);
        assert_simd_matches_scalar_noise::<5, 7>(&[0o175, 0o145, 0o133, 0o117, 0o127], Some(ForcedPath::OctLookupSse41), 2.0, 150_000, false);
        assert_simd_matches_scalar_noise::<5, 7>(&[0o175, 0o145, 0o133, 0o117, 0o127], Some(ForcedPath::OctLookup128), 2.0, 150_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_permute_5_7() {
        assert_simd_matches_scalar_noise::<5, 7>(&[0o175, 0o145, 0o133, 0o117, 0o127], Some(ForcedPath::PermuteAvx512), 2.0, 150_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_dispatch_6_7() {
        assert_simd_matches_scalar_noise::<6, 7>(&[0o111, 0o127, 0o133, 0o167, 0o173, 0o175], None, 2.0, 100_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_oct_lookup_6_7() {
        assert_simd_matches_scalar_noise::<6, 7>(&[0o111, 0o127, 0o133, 0o167, 0o173, 0o175], Some(ForcedPath::OctLookupAvx512), 2.0, 100_000, false);
        assert_simd_matches_scalar_noise::<6, 7>(&[0o111, 0o127, 0o133, 0o167, 0o173, 0o175], Some(ForcedPath::OctLookupAvx2), 2.0, 100_000, false);
        assert_simd_matches_scalar_noise::<6, 7>(&[0o111, 0o127, 0o133, 0o167, 0o173, 0o175], Some(ForcedPath::OctLookupSse41), 2.0, 100_000, false);
        assert_simd_matches_scalar_noise::<6, 7>(&[0o111, 0o127, 0o133, 0o167, 0o173, 0o175], Some(ForcedPath::OctLookup128), 2.0, 100_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_permute_6_7() {
        assert_simd_matches_scalar_noise::<6, 7>(&[0o111, 0o127, 0o133, 0o167, 0o173, 0o175], Some(ForcedPath::PermuteAvx512), 2.0, 100_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_dispatch_6_15() {
        assert_simd_matches_scalar_noise::<6, 15>(&[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537], None, 0.5, 5_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_oct_lookup_6_15() {
        assert_simd_matches_scalar_noise::<6, 15>(&[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537], Some(ForcedPath::OctLookupAvx512), 0.5, 5_000, false);
        assert_simd_matches_scalar_noise::<6, 15>(&[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537], Some(ForcedPath::OctLookupAvx2), 0.5, 5_000, false);
        assert_simd_matches_scalar_noise::<6, 15>(&[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537], Some(ForcedPath::OctLookupSse41), 0.5, 5_000, false);
        assert_simd_matches_scalar_noise::<6, 15>(&[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537], Some(ForcedPath::OctLookup128), 0.5, 5_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_permute_6_15() {
        assert_simd_matches_scalar_noise::<6, 15>(&[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537], Some(ForcedPath::PermuteAvx512), 0.5, 5_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_hard_dispatch_6_15() {
        assert_simd_matches_scalar_noise::<6, 15>(&[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537], None, 1.5, 5_000, true);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_hard_oct_lookup_6_15() {
        assert_simd_matches_scalar_noise::<6, 15>(&[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537], Some(ForcedPath::OctLookupAvx512), 1.5, 5_000, true);
        assert_simd_matches_scalar_noise::<6, 15>(&[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537], Some(ForcedPath::OctLookupAvx2), 1.5, 5_000, true);
        assert_simd_matches_scalar_noise::<6, 15>(&[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537], Some(ForcedPath::OctLookupSse41), 1.5, 5_000, true);
        assert_simd_matches_scalar_noise::<6, 15>(&[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537], Some(ForcedPath::OctLookup128), 1.5, 5_000, true);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_hard_permute_6_15() {
        assert_simd_matches_scalar_noise::<6, 15>(&[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537], Some(ForcedPath::PermuteAvx512), 1.5, 5_000, true);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_dispatch_7_7() {
        assert_simd_matches_scalar_noise::<7, 7>(&[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145], None, 2.0, 50_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_oct_lookup_7_7() {
        assert_simd_matches_scalar_noise::<7, 7>(&[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145], Some(ForcedPath::OctLookupAvx512), 2.0, 50_000, false);
        assert_simd_matches_scalar_noise::<7, 7>(&[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145], Some(ForcedPath::OctLookupAvx2), 2.0, 50_000, false);
        assert_simd_matches_scalar_noise::<7, 7>(&[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145], Some(ForcedPath::OctLookupSse41), 2.0, 50_000, false);
        assert_simd_matches_scalar_noise::<7, 7>(&[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145], Some(ForcedPath::OctLookup128), 2.0, 50_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_dispatch_8_7() {
        assert_simd_matches_scalar_noise::<8, 7>(&[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145, 0o171], None, 2.0, 50_000, false);
    }

    #[rustfmt::skip]
    #[test]
    fn simd_matches_scalar_noise_oct_lookup_8_7() {
        assert_simd_matches_scalar_noise::<8, 7>(&[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145, 0o171], Some(ForcedPath::OctLookupAvx512), 2.0, 50_000, false);
        assert_simd_matches_scalar_noise::<8, 7>(&[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145, 0o171], Some(ForcedPath::OctLookupAvx2), 2.0, 50_000, false);
        assert_simd_matches_scalar_noise::<8, 7>(&[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145, 0o171], Some(ForcedPath::OctLookupSse41), 2.0, 50_000, false);
        assert_simd_matches_scalar_noise::<8, 7>(&[0o155, 0o117, 0o123, 0o161, 0o127, 0o133, 0o145, 0o171], Some(ForcedPath::OctLookup128), 2.0, 50_000, false);
    }
}
