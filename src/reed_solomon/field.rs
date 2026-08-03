// Translation of libcorrect's reed-solomon field.h / field.c
//
// Arithmetic in GF(2^8). The field is represented by a pair of lookup tables:
//   - exp: alpha^i for successive powers i of the primitive element alpha
//   - log: the inverse mapping, the logarithm (power of alpha) of a field element

// An element in GF(2^8).
pub type FieldElement = u8;

// A power of the primitive element alpha.
pub type FieldLogarithm = u8;

// Give us some bits of headroom to do arithmetic.
// Variables of this type aren't really in any proper space.
pub type FieldOperation = u16;

pub struct Field {
    pub exp: [FieldElement; 512],
    pub log: [FieldLogarithm; 256],
}

impl Field {
    pub fn new(primitive_poly: FieldOperation) -> Field {
        // in GF(2^8)
        // log and exp
        // bits are in GF(2), compute alpha^val in GF(2^8)
        // exp should be of size 512 so that it can hold a "wraparound" which prevents some modulo ops
        // log should be of size 256. no wraparound here, the indices into this table are field elements
        let mut exp = [0 as FieldElement; 512];
        let mut log = [0 as FieldLogarithm; 256];

        // assume alpha is a primitive element, p(x) (primitive_poly) irreducible in GF(2^8)
        // addition is xor
        // subtraction is addition (also xor)
        // e.g. x^5 + x^4 + x^4 + x^2 + 1 = x^5 + x^2 + 1
        // each row of exp contains the field element found by exponentiating
        //   alpha by the row index
        // each row of log contains the coefficients of
        //   alpha^7 + alpha^6 + alpha^5 + alpha^4 + alpha^3 + alpha^2 + alpha + 1
        // as 8 bits packed into one byte

        let mut element: FieldOperation = 1;
        exp[0] = element as FieldElement;
        log[0] = 0 as FieldLogarithm; // really, it's undefined. we shouldn't ever access this
        for i in 1..512 as FieldOperation {
            element *= 2;
            element = if element > 255 {
                element ^ primitive_poly
            } else {
                element
            };
            exp[i as usize] = element as FieldElement;
            if i < 256 {
                log[element as usize] = i as FieldLogarithm;
            }
        }

        Field { exp, log }
    }

    pub fn add(&self, l: FieldElement, r: FieldElement) -> FieldElement {
        l ^ r
    }

    pub fn sub(&self, l: FieldElement, r: FieldElement) -> FieldElement {
        l ^ r
    }

    pub fn sum(&self, elem: FieldElement, n: u32) -> FieldElement {
        // we'll do a closed-form expression of the sum, although we could also
        //   choose to call field_add n times

        // since the sum is actually the bytewise XOR operator, this suggests two
        // kinds of values: n odd, and n even

        // if you sum once, you have coeff
        // if you sum twice, you have coeff XOR coeff = 0
        // if you sum thrice, you are back at coeff
        // an even number of XORs puts you at 0
        // an odd number of XORs puts you back at your value

        // so, just throw away all the even n
        if !n.is_multiple_of(2) {
            elem
        } else {
            0
        }
    }

    pub fn mul(&self, l: FieldElement, r: FieldElement) -> FieldElement {
        if l == 0 || r == 0 {
            return 0;
        }
        // multiply two field elements by adding their logarithms.
        // yep, get your slide rules out
        let res: FieldOperation =
            self.log[l as usize] as FieldOperation + self.log[r as usize] as FieldOperation;

        // if coeff exceeds 255, we would normally have to wrap it back around
        // alpha^255 = 1; alpha^256 = alpha^255 * alpha^1 = alpha^1
        // however, we've constructed exponentiation table so that
        //   we can just directly lookup this result
        // the result must be clamped to [0, 511]
        // the greatest we can see at this step is alpha^255 * alpha^255
        //   = alpha^510
        self.exp[res as usize]
    }

    pub fn div(&self, l: FieldElement, r: FieldElement) -> FieldElement {
        if l == 0 {
            return 0;
        }

        if r == 0 {
            // XXX ???
            return 0;
        }

        // division as subtraction of logarithms

        // if rcoeff is larger, then log[l] - log[r] wraps under
        // so, instead, always add 255. in some cases, we'll wrap over, but
        // that's ok because the exp table runs up to 511.
        let res: FieldOperation = 255 as FieldOperation + self.log[l as usize] as FieldOperation
            - self.log[r as usize] as FieldOperation;
        self.exp[res as usize]
    }

    pub fn mul_log(&self, l: FieldLogarithm, r: FieldLogarithm) -> FieldLogarithm {
        // this function performs the equivalent of field_mul on two logarithms
        // we save a little time by skipping the lookup step at the beginning
        let res: FieldOperation = l as FieldOperation + r as FieldOperation;

        // because we arent using the table, the value we return must be a valid logarithm
        // which we have decided must live in [0, 255] (they are 8-bit values)
        // ensuring this makes it so that multiple muls will not reach past the end of the
        // exp table whenever we finally convert back to an element
        if res > 255 {
            return (res - 255) as FieldLogarithm;
        }
        res as FieldLogarithm
    }

    pub fn div_log(&self, l: FieldLogarithm, r: FieldLogarithm) -> FieldLogarithm {
        // like field_mul_log, this performs field_div without going through a field_element_t
        let res: FieldOperation =
            255 as FieldOperation + l as FieldOperation - r as FieldOperation;
        if res > 255 {
            return (res - 255) as FieldLogarithm;
        }
        res as FieldLogarithm
    }

    pub fn mul_log_element(&self, l: FieldLogarithm, r: FieldLogarithm) -> FieldElement {
        // like field_mul_log, but returns a field_element_t
        // because we are doing lookup here, we can safely skip the wrapover check
        let res: FieldOperation = l as FieldOperation + r as FieldOperation;
        self.exp[res as usize]
    }

    pub fn pow(&self, elem: FieldElement, pow: i32) -> FieldElement {
        // take the logarithm, multiply, and then "exponentiate"
        // n.b. the exp table only considers powers of alpha, the primitive element
        // but here we have an arbitrary coeff
        let log = self.log[elem as usize];
        let res_log = log as i32 * pow;
        let mut mod_ = res_log % 255;
        if mod_ < 0 {
            mod_ += 255;
        }
        self.exp[mod_ as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRIM: FieldOperation = 0x11d;

    #[test]
    fn exp_log_are_inverses() {
        let field = Field::new(PRIM);
        // every nonzero element has a logarithm that exponentiates back to it
        for e in 1..=255u16 {
            let e = e as FieldElement;
            assert_eq!(field.exp[field.log[e as usize] as usize], e);
        }
        // alpha^0 == 1 and the exp table wraps with period 255
        assert_eq!(field.exp[0], 1);
        for i in 0..255usize {
            assert_eq!(field.exp[i], field.exp[i + 255]);
        }
    }

    #[test]
    fn mul_div_roundtrip() {
        let field = Field::new(PRIM);
        for a in 1..=255u16 {
            for b in 1..=255u16 {
                let (a, b) = (a as FieldElement, b as FieldElement);
                let prod = field.mul(a, b);
                // dividing the product by one factor recovers the other
                assert_eq!(field.div(prod, b), a);
            }
        }
    }

    #[test]
    fn mul_zero() {
        let field = Field::new(PRIM);
        for a in 0..=255u16 {
            assert_eq!(field.mul(a as FieldElement, 0), 0);
            assert_eq!(field.mul(0, a as FieldElement), 0);
        }
    }

    #[test]
    fn add_is_xor_and_self_inverse() {
        let field = Field::new(PRIM);
        for a in 0..=255u16 {
            for b in 0..=255u16 {
                let (a, b) = (a as FieldElement, b as FieldElement);
                assert_eq!(field.add(a, b), a ^ b);
            }
            // adding an element to itself yields zero
            assert_eq!(field.add(a as FieldElement, a as FieldElement), 0);
        }
    }

    #[test]
    fn sum_parity() {
        let field = Field::new(PRIM);
        // sum is elem for odd n, 0 for even n
        assert_eq!(field.sum(0xab, 1), 0xab);
        assert_eq!(field.sum(0xab, 2), 0);
        assert_eq!(field.sum(0xab, 3), 0xab);
        assert_eq!(field.sum(0xab, 4), 0);
    }

    #[test]
    fn log_arithmetic_matches_element_arithmetic() {
        let field = Field::new(PRIM);
        for a in 1..=255u16 {
            for b in 1..=255u16 {
                let (a, b) = (a as FieldElement, b as FieldElement);
                let la = field.log[a as usize];
                let lb = field.log[b as usize];
                // mul_log_element on logs matches mul on elements
                assert_eq!(field.mul_log_element(la, lb), field.mul(a, b));
                // mul_log produces the log of the product
                assert_eq!(field.exp[field.mul_log(la, lb) as usize], field.mul(a, b));
                // div_log produces the log of the quotient
                assert_eq!(field.exp[field.div_log(la, lb) as usize], field.div(a, b));
            }
        }
    }

    #[test]
    fn pow_matches_repeated_mul() {
        let field = Field::new(PRIM);
        for a in 1..=255u16 {
            let a = a as FieldElement;
            let mut acc: FieldElement = 1;
            for p in 0..=8 {
                assert_eq!(field.pow(a, p), acc);
                acc = field.mul(acc, a);
            }
        }
    }
}
