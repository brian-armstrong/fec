/// A small deterministic SplitMix64 RNG. It seeds the noise in the `sim`
/// harness, so a given seed always reproduces the same run.
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Creates an RNG seeded with `seed`. Public because the `sim` functions
    /// that take an [`Rng`] need the caller to construct one. The generator
    /// methods are crate-internal.
    pub fn new(seed: u64) -> Rng {
        Rng { state: seed }
    }

    #[inline]
    pub(crate) fn next_u64(&mut self) -> u64 {
        // SplitMix64
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    // only the Reed-Solomon tests need bounded draws
    #[cfg(test)]
    pub(crate) fn next_u64_below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    #[inline]
    pub(crate) fn next_f64(&mut self) -> f64 {
        // convert a random u64 to a random f64 in [0, 1)
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    #[inline]
    pub(crate) fn next_u8(&mut self) -> u8 {
        (self.next_u64() >> (64 - 8)) as u8
    }
}
