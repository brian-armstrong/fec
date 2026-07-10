use super::field::{Field, FieldElement, FieldLogarithm, FieldOperation};
use super::polynomial::{
    polynomial_build_exp_lut, polynomial_eval_log_lut, polynomial_eval_lut,
    polynomial_formal_derivative, polynomial_init_from_roots, polynomial_mul,
    Polynomial, reed_solomon_build_generator,
};

/// Error returned by [`RsDecoder::decode`] / [`RsDecoder::decode_with_erasures`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// The encoded block was longer than the code's block length.
    BlockTooLong,
    /// More erasures were supplied than the code has parity symbols.
    TooManyErasures,
    /// The block had more corruption than the code can correct.
    TooManyErrors,
}

/// Reed-Solomon decoder over GF(2^8).
pub struct RsDecoder {
    pub block_length: usize,
    pub message_length: usize,
    pub min_distance: usize,

    first_consecutive_root: FieldLogarithm,
    generator_root_gap: FieldLogarithm,

    field: Field,

    #[allow(dead_code)]
    generator_roots: Vec<FieldElement>,

    // syndromes[i] = received(generator_root[i]); length min_distance
    syndromes: Vec<FieldElement>,
    // received block, flipped + zero-padded; order block_length-1
    received_polynomial: Vec<FieldElement>,
    // Berlekamp-Massey error locator and its log-domain copy; order min_distance
    error_locator: Vec<FieldElement>,
    error_locator_order: usize,
    error_locator_log: Vec<FieldElement>,
    error_locator_log_order: usize,
    // roots of the error locator (Chien search) and the byte positions they map to
    error_roots: Vec<FieldElement>,
    error_vals: Vec<FieldElement>,
    error_locations: Vec<FieldLogarithm>,
    // Berlekamp-Massey scratch
    last_error_locator: Vec<FieldElement>,
    last_error_locator_order: usize,
    // Forney scratch: error evaluator omega(x) and locator derivative lambda'(x)
    error_evaluator: Vec<FieldElement>,
    error_evaluator_order: usize,
    error_locator_derivative: Vec<FieldElement>,
    error_locator_derivative_order: usize,

    // generator_root_exp[i] = successive powers of generator_roots[i], block_length long
    generator_root_exp: Vec<Vec<FieldLogarithm>>,
    // element_exp[i] = successive powers of field element i, min_distance long
    element_exp: Vec<Vec<FieldLogarithm>>,

    // modified syndromes S(x)*erasure_locator(x); length 2*min_distance
    modified_syndromes: Vec<FieldElement>,
    // erasure locator built from the known erasure positions; order min_distance
    erasure_locator: Vec<FieldElement>,
    erasure_locator_order: usize,
    // two scratch buffers for polynomial_init_from_roots; each min_distance+1 long
    init_from_roots_scratch0: Vec<FieldElement>,
    init_from_roots_scratch1: Vec<FieldElement>,
    // saves the syndromes while the low half is overwritten with modified
    // syndromes, so the Forney step can be run on the originals; min_distance long
    syndrome_copy: Vec<FieldElement>,
    // scratch that the erasure path swaps in for error_locator to hold the
    // combined erasure*error locator; order up to min_distance
    combined_locator: Vec<FieldElement>,
}

impl RsDecoder {
    /// Build a decoder for a (255, 255 - `num_roots`) Reed-Solomon code over
    /// GF(2^8). The parameters must match those used to encode: see
    /// [`RsEncoder::new`] for their meaning. Building the decoder precomputes
    /// ~16 KB of lookup tables, so prefer to construct it once and reuse it.
    pub fn new(
        primitive_polynomial: FieldOperation,
        first_consecutive_root: FieldLogarithm,
        generator_root_gap: FieldLogarithm,
        num_roots: usize,
    ) -> RsDecoder {
        let field = Field::new(primitive_polynomial);

        let block_length = 255usize;
        let min_distance = num_roots;
        let message_length = block_length - min_distance;

        let mut generator_roots = vec![0 as FieldElement; min_distance];
        // we need the generator roots to build the syndrome-power LUT; the
        // generator polynomial itself isn't needed on the decode side, so we
        // build the roots directly and discard the polynomial.
        let mut generator = vec![0 as FieldElement; min_distance + 1];
        reed_solomon_build_generator(
            &field,
            min_distance,
            first_consecutive_root,
            generator_root_gap as usize,
            &mut generator,
            &mut generator_roots,
        );

        let syndromes = vec![0 as FieldElement; min_distance];
        let received_polynomial = vec![0 as FieldElement; block_length];
        let error_locator = vec![0 as FieldElement; min_distance + 1];
        let error_locator_log = vec![0 as FieldElement; min_distance + 1];
        let error_roots = vec![0 as FieldElement; 2 * min_distance];
        let error_vals = vec![0 as FieldElement; min_distance];
        let error_locations = vec![0 as FieldLogarithm; min_distance];
        let last_error_locator = vec![0 as FieldElement; min_distance + 1];
        let error_evaluator = vec![0 as FieldElement; min_distance];
        let error_locator_derivative = vec![0 as FieldElement; min_distance];

        // calculate and store the first block_length powers of every generator root
        // we would have to do this work in order to calculate the syndromes
        // if we save it, we can prevent the need to recalculate it on subsequent calls
        // total memory usage is min_distance * block_length bytes e.g. 32 * 255 ~= 8k
        let mut generator_root_exp: Vec<Vec<FieldLogarithm>> = Vec::with_capacity(min_distance);
        for i in 0..min_distance {
            let mut lut = vec![0 as FieldLogarithm; block_length];
            polynomial_build_exp_lut(&field, generator_roots[i], block_length - 1, &mut lut);
            generator_root_exp.push(lut);
        }

        // calculate and store the first min_distance powers of every element in the field
        // we would have to do this for chien search anyway, and its size is only 256 * min_distance bytes
        // for min_distance = 32 this is 8k of memory, a pittance for the speedup we receive in exchange
        // we also get to reuse this work during error value calculation
        let mut element_exp: Vec<Vec<FieldLogarithm>> = Vec::with_capacity(256);
        for i in 0..256u16 {
            let mut lut = vec![0 as FieldLogarithm; min_distance];
            polynomial_build_exp_lut(&field, i as FieldElement, min_distance - 1, &mut lut);
            element_exp.push(lut);
        }

        RsDecoder {
            block_length,
            message_length,
            min_distance,
            first_consecutive_root,
            generator_root_gap,
            field,
            generator_roots,
            syndromes,
            received_polynomial,
            error_locator,
            error_locator_order: 0,
            error_locator_log,
            error_locator_log_order: 0,
            error_roots,
            error_vals,
            error_locations,
            last_error_locator,
            last_error_locator_order: 0,
            error_evaluator,
            error_evaluator_order: min_distance - 1,
            error_locator_derivative,
            error_locator_derivative_order: min_distance - 1,
            generator_root_exp,
            element_exp,
            modified_syndromes: vec![0 as FieldElement; 2 * min_distance],
            erasure_locator: vec![0 as FieldElement; min_distance + 1],
            erasure_locator_order: 0,
            init_from_roots_scratch0: vec![0 as FieldElement; min_distance + 1],
            init_from_roots_scratch1: vec![0 as FieldElement; min_distance + 1],
            syndrome_copy: vec![0 as FieldElement; min_distance],
            combined_locator: vec![0 as FieldElement; min_distance + 1],
        }
    }

    /// Build a decoder for the standard CCSDS (255,223) Reed-Solomon code
    /// (conventional-basis representation). For the dual-basis representation used
    /// on the wire, see [`RsDecoder::decode_ccsds_dual`].
    pub fn new_ccsds() -> RsDecoder {
        use super::ccsds;
        RsDecoder::new(
            ccsds::CCSDS_PRIMITIVE_POLYNOMIAL,
            ccsds::CCSDS_FIRST_CONSECUTIVE_ROOT,
            ccsds::CCSDS_GENERATOR_ROOT_GAP,
            ccsds::CCSDS_NUM_ROOTS,
        )
    }

    // calculate all syndromes of the received polynomial at the roots of the generator
    // because we're evaluating at the roots of the generator, and because the transmitted
    //   polynomial was made to be a product of the generator, we know that the transmitted
    //   polynomial is 0 at these roots
    // any nonzero syndromes we find here are the values of the error polynomial evaluated
    //   at these roots, so these values give us a window into the error polynomial. if
    //   these syndromes are all zero, then we can conclude the error polynomial is also
    //   zero. if they're nonzero, then we know our message received an error in transit.
    // returns true if syndromes are all zero
    fn find_syndromes(&mut self) -> bool {
        let mut all_zero = true;
        for s in self.syndromes.iter_mut() {
            *s = 0;
        }
        let msgpoly = Polynomial::new(&self.received_polynomial, self.block_length - 1);
        for i in 0..self.min_distance {
            // profiling reveals that this function takes about 50% of the cpu time of
            // decoding. so, in order to speed it up a little, we precompute and save
            // the successive powers of the roots of the generator, which are
            // located in generator_root_exp
            let eval = polynomial_eval_lut(&self.field, &msgpoly, &self.generator_root_exp[i]);
            if eval != 0 {
                all_zero = false;
            }
            self.syndromes[i] = eval;
        }
        all_zero
    }

    // Berlekamp-Massey algorithm to find LFSR that describes syndromes
    // returns number of errors and writes the error locator polynomial to error_locator
    fn find_error_locator(&mut self, num_erasures: usize) -> usize {
        let mut numerrors = 0usize;

        for c in self.error_locator[..self.min_distance + 1].iter_mut() {
            *c = 0;
        }

        // initialize to f(x) = 1
        self.error_locator[0] = 1;
        self.error_locator_order = 0;

        self.last_error_locator[..self.min_distance + 1]
            .copy_from_slice(&self.error_locator[..self.min_distance + 1]);
        self.last_error_locator_order = self.error_locator_order;

        let mut last_discrepancy: FieldElement = 1;
        let mut delay_length = 1usize;

        let field = &self.field;
        for i in self.error_locator_order..(self.min_distance - num_erasures) {
            let mut discrepancy = self.syndromes[i];
            for j in 1..=numerrors {
                discrepancy = field.add(
                    discrepancy,
                    field.mul(self.error_locator[j], self.syndromes[i - j]),
                );
            }

            if discrepancy == 0 {
                // our existing LFSR describes the new syndrome as well
                // leave it as-is but update the number of delay elements
                //   so that if a discrepancy occurs later we can eliminate it
                delay_length += 1;
                continue;
            }

            if 2 * numerrors <= i {
                // there's a discrepancy, but we still have room for more taps
                // lengthen LFSR by one tap and set weight to eliminate discrepancy

                // shift the last locator by the delay length, multiply by discrepancy,
                //   and divide by the last discrepancy
                // we move down because we're shifting up, and this prevents overwriting
                for j in (0..=self.last_error_locator_order).rev() {
                    // the bounds here will be ok since we have a headroom of numerrors
                    self.last_error_locator[j + delay_length] = field.div(
                        field.mul(self.last_error_locator[j], discrepancy),
                        last_discrepancy,
                    );
                }
                for j in (0..delay_length).rev() {
                    self.last_error_locator[j] = 0;
                }

                // locator = locator - last_locator
                // we will also update last_locator to be locator before this loop takes place
                for j in 0..=(self.last_error_locator_order + delay_length) {
                    let temp = self.error_locator[j];
                    self.error_locator[j] =
                        field.add(self.error_locator[j], self.last_error_locator[j]);
                    self.last_error_locator[j] = temp;
                }
                let temp_order = self.error_locator_order;
                self.error_locator_order = self.last_error_locator_order + delay_length;
                self.last_error_locator_order = temp_order;

                // now last_locator is locator before we started,
                //   and locator is (locator - (discrepancy/last_discrepancy) * x^(delay_length) * last_locator)

                numerrors = i + 1 - numerrors;
                last_discrepancy = discrepancy;
                delay_length = 1;
                continue;
            }

            // no more taps
            // unlike the previous case, we are preserving last locator,
            //    but we'll update locator as before
            // we're basically flattening the two loops from the previous case because
            //    we no longer need to update last_locator
            for j in (0..=self.last_error_locator_order).rev() {
                self.error_locator[j + delay_length] = field.add(
                    self.error_locator[j + delay_length],
                    field.div(
                        field.mul(self.last_error_locator[j], discrepancy),
                        last_discrepancy,
                    ),
                );
            }
            self.error_locator_order =
                if self.last_error_locator_order + delay_length > self.error_locator_order {
                    self.last_error_locator_order + delay_length
                } else {
                    self.error_locator_order
                };
            delay_length += 1;
        }
        self.error_locator_order
    }

    // use error locator and syndromes to find the error evaluator polynomial
    fn find_error_evaluator(&mut self) {
        // the error evaluator, omega(x), is S(x)*Lamba(x) mod x^(2t)
        // where S(x) is a polynomial constructed from the syndromes
        //   S(1) + S(2)*x + ... + S(2t)*x(2t - 1)
        // and Lambda(x) is the error locator
        // the modulo is implicit here -- we have limited the max length of error_evaluator,
        //   which polynomial_mul will interpret to mean that it should not compute
        //   powers larger than that, which is the same as performing mod x^(2t)
        let locator = Polynomial::new(&self.error_locator, self.error_locator_order);
        let syndromes = Polynomial::new(&self.syndromes, self.min_distance - 1);
        polynomial_mul(
            &self.field,
            &locator,
            &syndromes,
            &mut self.error_evaluator,
            self.error_evaluator_order,
        );
    }

    // use error locator, error roots and syndromes to find the error values
    // that is, the elements in the finite field which can be added to the received
    //   polynomial at the locations of the error roots in order to produce the
    //   transmitted polynomial
    // forney algorithm
    fn find_error_values(&mut self) {
        // error value e(j) = -(X(j)^(1-c) * omega(X(j)^-1))/(lambda'(X(j)^-1))
        // where X(j)^-1 is a root of the error locator, omega(X) is the error evaluator,
        //   lambda'(X) is the first formal derivative of the error locator,
        //   and c is the first consecutive root of the generator used in encoding

        // first find omega(X), the error evaluator
        // we generate S(x), the polynomial constructed from the roots of the syndromes
        // this is *not* the polynomial constructed by expanding the products of roots
        // S(x) = S(1) + S(2)*x + ... + S(2t)*x(2t - 1)
        for c in self.error_evaluator[..self.error_evaluator_order + 1].iter_mut() {
            *c = 0;
        }
        self.find_error_evaluator();

        // now find lambda'(X)
        self.error_locator_derivative_order = self.error_locator_order - 1;
        let locator = Polynomial::new(&self.error_locator, self.error_locator_order);
        polynomial_formal_derivative(
            &self.field,
            &locator,
            &mut self.error_locator_derivative,
            self.error_locator_derivative_order,
        );

        // calculate each e(j)
        let field = &self.field;
        for i in 0..self.error_locator_order {
            if self.error_roots[i] == 0 {
                continue;
            }
            let eval_evaluator = polynomial_eval_lut(
                field,
                &Polynomial::new(&self.error_evaluator, self.error_evaluator_order),
                &self.element_exp[self.error_roots[i] as usize],
            );
            let eval_derivative = polynomial_eval_lut(
                field,
                &Polynomial::new(
                    &self.error_locator_derivative,
                    self.error_locator_derivative_order,
                ),
                &self.element_exp[self.error_roots[i] as usize],
            );
            self.error_vals[i] = field.mul(
                field.pow(self.error_roots[i], self.first_consecutive_root as i32 - 1),
                field.div(eval_evaluator, eval_derivative),
            );
        }
    }

    // find the roots of the error locator polynomial
    // Chien search
    // returns false if the locator does not have enough roots (too many errors)
    fn factorize_error_locator(&mut self, num_skip: usize) -> bool {
        // normally it'd be tricky to find all the roots
        // but, the finite field is awfully finite...
        // just brute force search across every field element
        let mut root = num_skip;
        let locator_order = self.error_locator_log_order;
        for r in self.error_roots[num_skip..(num_skip + locator_order)].iter_mut() {
            *r = 0;
        }
        let locator_log = Polynomial::new(&self.error_locator_log, locator_order);
        for i in 0..256u16 {
            // we make two optimizations here to help this search go faster
            // a) we have precomputed the first successive powers of every single element
            //   in the field. we need at most n powers, where n is the largest possible
            //   degree of the error locator
            // b) we have precomputed the error locator polynomial in log form, which
            //   helps reduce some lookups that would be done here
            if polynomial_eval_log_lut(&self.field, &locator_log, &self.element_exp[i as usize])
                == 0
            {
                self.error_roots[root] = i as FieldElement;
                root += 1;
            }
        }
        // this is where we find out if we are have too many errors to recover from
        // berlekamp-massey may have built an error locator that has 0 discrepancy
        // on the syndromes but doesn't have enough roots
        root == locator_order + num_skip
    }

    // map each error root to a byte position in the received block
    fn find_error_locations(&mut self, num_errors: usize, _num_skip: usize) {
        let field = &self.field;
        for i in 0..num_errors {
            // the error roots are the reciprocals of the error locations, so div 1 by them

            // we do mod 255 here because the log table aliases at index 1
            // the log of 1 is both 0 and 255 (alpha^255 = alpha^0 = 1)
            // for most uses it makes sense to have log(1) = 255, but in this case
            // we're interested in a byte index, and the 255th index is not even valid
            // just wrap it back to 0

            if self.error_roots[i] == 0 {
                continue;
            }

            let loc = field.div(1, self.error_roots[i]);
            if self.generator_root_gap == 1 {
                // fast path for the common gap=1 case. The brute-force loop below
                // finds the smallest j with pow(j,1)==loc and stores log[j].
                // pow(j,1)==j for all j EXCEPT j==0, where pow(0,1)==exp[0]==1.
                // So for loc==1 the loop matches j==0 first and stores log[0]==0
                // (this is the "wrap 255 -> 0" the comment describes); for any
                // other loc it matches j==loc and stores log[loc].
                self.error_locations[i] = if loc == 1 { 0 } else { field.log[loc as usize] };
                continue;
            }
            for j in 0..256u16 {
                if field.pow(j as FieldElement, self.generator_root_gap as i32) == loc {
                    self.error_locations[i] = field.log[j as usize];
                    break;
                }
            }
        }
    }

    /// Decode a received block, correcting up to `num_roots / 2` byte errors, and
    /// write the recovered message to `msg` (which must be long enough to hold the
    /// decoded payload). Returns the number of symbols corrected (0 if the block
    /// was already clean), or a [`DecodeError`].
    ///
    /// The returned count is "corrections the decoder believes it made": under
    /// more errors than the code can correct, RS can *miscorrect* -- decode to a
    /// wrong codeword and return success with a payload that still contains
    /// errors. This is possible but unlikely, and the count cannot detect it.
    ///
    /// Derived from correct_reed_solomon_decode.
    pub fn decode(&mut self, encoded: &[u8], msg: &mut [u8]) -> Result<usize, DecodeError> {
        let encoded_length = encoded.len();
        if encoded_length > self.block_length {
            return Err(DecodeError::BlockTooLong);
        }

        // the message is the non-remainder part
        let msg_length = encoded_length - self.min_distance;
        // if they handed us a nonfull block, we'll write in 0s
        let pad_length = self.block_length - encoded_length;

        // we need to copy to our local buffer
        // the buffer we're given has the coordinates in the wrong direction
        // e.g. byte 0 corresponds to the 254th order coefficient
        // so we're going to flip and then write padding
        // the final copied buffer will look like
        // | rem (min_distance) | msg (msg_length) | pad (pad_length) |
        for i in 0..encoded_length {
            self.received_polynomial[i] = encoded[encoded_length - (i + 1)];
        }

        // fill the pad_length with 0s
        for i in 0..pad_length {
            self.received_polynomial[i + encoded_length] = 0;
        }

        let all_zero = self.find_syndromes();

        if all_zero {
            // syndromes were all zero, so there was no error in the message
            // copy to msg and we are done -- zero corrections
            for i in 0..msg_length {
                msg[i] = self.received_polynomial[encoded_length - (i + 1)];
            }
            return Ok(0);
        }

        let order = self.find_error_locator(0);
        // XXX fix this vvvv
        self.error_locator_order = order;

        for i in 0..=self.error_locator_order {
            // this is a little strange since the coeffs are logs, not elements
            // also, we'll be storing log(0) = 0 for any 0 coeffs in the error locator
            // that would seem bad but we'll just be using this in chien search, and we'll skip all 0 coeffs
            // (you might point out that log(1) also = 0, which would seem to alias. however, that's ok,
            //   because log(1) = 255 as well, and in fact that's how it's represented in our log table)
            self.error_locator_log[i] = self.field.log[self.error_locator[i] as usize];
        }
        self.error_locator_log_order = self.error_locator_order;

        if !self.factorize_error_locator(0) {
            // roots couldn't be found, so there were too many errors to deal with
            // RS has failed for this message
            return Err(DecodeError::TooManyErrors);
        }

        self.find_error_locations(self.error_locator_order, 0);

        self.find_error_values();

        // the locator order is the number of error positions we're correcting
        let num_corrected = self.error_locator_order;
        for i in 0..self.error_locator_order {
            let loc = self.error_locations[i] as usize;
            self.received_polynomial[loc] =
                self.field.sub(self.received_polynomial[loc], self.error_vals[i]);
        }

        for i in 0..msg_length {
            msg[i] = self.received_polynomial[encoded_length - (i + 1)];
        }

        Ok(num_corrected)
    }

    // erasure method -- take given locations and convert to roots
    // this is the inverse of find_error_locations
    fn find_error_roots_from_locations(&mut self, num_errors: usize) {
        let field = &self.field;
        for i in 0..num_errors {
            let loc = field.pow(
                field.exp[self.error_locations[i] as usize],
                self.generator_root_gap as i32,
            );
            // field_element_t loc = field.exp[error_locations[i]];
            self.error_roots[i] = field.div(1, loc);
            // error_roots[i] = loc;
        }
    }

    // erasure method -- given the roots of the error locator, create the polynomial
    fn find_error_locator_from_roots(&mut self, num_errors: usize) {
        // multiply out roots to build the error locator polynomial
        polynomial_init_from_roots(
            &self.field,
            num_errors,
            &self.error_roots,
            &mut self.erasure_locator,
            &mut self.init_from_roots_scratch0,
            &mut self.init_from_roots_scratch1,
        );
        self.erasure_locator_order = num_errors;
    }

    // erasure method
    fn find_modified_syndromes(&mut self) {
        let syndrome_poly = Polynomial::new(&self.syndromes, self.min_distance - 1);
        let error_locator = Polynomial::new(&self.erasure_locator, self.erasure_locator_order);
        polynomial_mul(
            &self.field,
            &error_locator,
            &syndrome_poly,
            &mut self.modified_syndromes,
            self.min_distance - 1,
        );
    }

    /// Decode a received block where some byte positions are already suspected to
    /// be corrupted (the erasures), typically flagged by a demodulating or
    /// receiving device. `erasure_locations` indexes into the emitted block, low
    /// order first, and should not exceed `num_roots` entries.
    ///
    /// Each erasure costs only one parity symbol instead of two, so erasure
    /// information lets the decoder recover more total corruption. Decoding
    /// succeeds as long as `num_erasures + 2 * num_errors < num_roots`.
    ///
    /// Returns the number of symbols corrected (errors + erasures resolved), or a
    /// [`DecodeError`]. The same miscorrection caveat as [`RsDecoder::decode`]
    /// applies. Derived from correct_reed_solomon_decode_with_erasures.
    pub fn decode_with_erasures(
        &mut self,
        encoded: &[u8],
        erasure_locations: &[u8],
        msg: &mut [u8],
    ) -> Result<usize, DecodeError> {
        let erasure_length = erasure_locations.len();
        if erasure_length == 0 {
            return self.decode(encoded, msg);
        }

        let encoded_length = encoded.len();
        if encoded_length > self.block_length {
            return Err(DecodeError::BlockTooLong);
        }

        if erasure_length > self.min_distance {
            return Err(DecodeError::TooManyErasures);
        }

        // the message is the non-remainder part
        let msg_length = encoded_length - self.min_distance;
        // if they handed us a nonfull block, we'll write in 0s
        let pad_length = self.block_length - encoded_length;

        // we need to copy to our local buffer
        // the buffer we're given has the coordinates in the wrong direction
        // e.g. byte 0 corresponds to the 254th order coefficient
        // so we're going to flip and then write padding
        // the final copied buffer will look like
        // | rem (min_distance) | msg (msg_length) | pad (pad_length) |
        for i in 0..encoded_length {
            self.received_polynomial[i] = encoded[encoded_length - (i + 1)];
        }

        // fill the pad_length with 0s
        for i in 0..pad_length {
            self.received_polynomial[i + encoded_length] = 0;
        }

        for i in 0..erasure_length {
            // remap the coordinates of the erasures
            self.error_locations[i] =
                (self.block_length - (erasure_locations[i] as usize + pad_length + 1)) as FieldLogarithm;
        }

        self.find_error_roots_from_locations(erasure_length);

        self.find_error_locator_from_roots(erasure_length);

        let all_zero = self.find_syndromes();

        if all_zero {
            // syndromes were all zero, so there was no error in the message
            // copy to msg and we are done -- zero corrections
            for i in 0..msg_length {
                msg[i] = self.received_polynomial[encoded_length - (i + 1)];
            }
            return Ok(0);
        }

        self.find_modified_syndromes();

        self.syndrome_copy
            .copy_from_slice(&self.syndromes[..self.min_distance]);

        for i in erasure_length..self.min_distance {
            self.syndromes[i - erasure_length] = self.modified_syndromes[i];
        }

        let order = self.find_error_locator(erasure_length);
        // XXX fix this vvvv
        self.error_locator_order = order;

        for i in 0..=self.error_locator_order {
            // this is a little strange since the coeffs are logs, not elements
            // also, we'll be storing log(0) = 0 for any 0 coeffs in the error locator
            // that would seem bad but we'll just be using this in chien search, and we'll skip all 0 coeffs
            // (you might point out that log(1) also = 0, which would seem to alias. however, that's ok,
            //   because log(1) = 255 as well, and in fact that's how it's represented in our log table)
            self.error_locator_log[i] = self.field.log[self.error_locator[i] as usize];
        }
        self.error_locator_log_order = self.error_locator_order;

        if !self.factorize_error_locator(erasure_length) {
            // roots couldn't be found, so there were too many errors to deal with
            // RS has failed for this message
            return Err(DecodeError::TooManyErrors);
        }

        let temp_order = self.error_locator_order + erasure_length;
        // swap the two scratch buffers rather than allocating: after the swap,
        // combined_locator holds the original error locator (read below as the
        // `error` factor) and error_locator is the scratch we build the product
        // into. we swap back at the end to restore.
        core::mem::swap(&mut self.error_locator, &mut self.combined_locator);
        let placeholder_order = self.error_locator_order;
        let erasure = Polynomial::new(&self.erasure_locator, self.erasure_locator_order);
        let error = Polynomial::new(&self.combined_locator, placeholder_order);
        polynomial_mul(&self.field, &erasure, &error, &mut self.error_locator, temp_order);
        self.error_locator_order = temp_order;

        self.find_error_locations(self.error_locator_order, erasure_length);

        // restore the syndromes for the Forney step
        self.syndromes[..self.min_distance].copy_from_slice(&self.syndrome_copy);

        self.find_error_values();

        // the combined locator order is the total symbols corrected (errors +
        // erasures); capture it before we restore the original locator order
        let num_corrected = self.error_locator_order;
        for i in 0..self.error_locator_order {
            let loc = self.error_locations[i] as usize;
            self.received_polynomial[loc] =
                self.field.sub(self.received_polynomial[loc], self.error_vals[i]);
        }

        // restore the original error locator by swapping the buffers back
        core::mem::swap(&mut self.error_locator, &mut self.combined_locator);
        self.error_locator_order = placeholder_order;

        for i in 0..msg_length {
            msg[i] = self.received_polynomial[encoded_length - (i + 1)];
        }

        Ok(num_corrected)
    }

    /// Decode a CCSDS dual-basis received block. `encoded` holds dual-basis
    /// symbols (message followed by parity, as on the wire); the recovered
    /// dual-basis message is written to `msg`. The block is transformed to the
    /// conventional basis, decoded with this (CCSDS) code, and the recovered
    /// message transformed back to dual. Returns the number of symbols corrected,
    /// or a [`DecodeError`].
    ///
    /// This decoder must have been built with the CCSDS parameters (see
    /// [`RsDecoder::new_ccsds`]).
    pub fn decode_ccsds_dual(
        &mut self,
        encoded: &[u8],
        msg: &mut [u8],
    ) -> Result<usize, DecodeError> {
        use super::ccsds;
        // transform the whole dual-basis block to the conventional basis
        let conv_block: Vec<u8> = encoded.iter().map(|&b| ccsds::dual_to_conv(b)).collect();
        let corrected = self.decode(&conv_block, msg)?;
        // transform the recovered conventional message back to the dual basis
        let msg_length = encoded.len() - self.min_distance;
        for b in msg[..msg_length].iter_mut() {
            *b = ccsds::conv_to_dual(*b);
        }
        Ok(corrected)
    }
}
