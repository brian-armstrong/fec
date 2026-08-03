// Translation of libcorrect's reed-solomon polynomial.c

use super::field::{Field, FieldElement, FieldLogarithm};

// A borrowed, read-only view of a polynomial: order+1 coefficients with an
// explicit degree
#[derive(Clone, Copy)]
pub struct Polynomial<'a> {
    pub coeff: &'a [FieldElement],
    pub order: usize,
}

impl<'a> Polynomial<'a> {
    pub fn new(coeff: &'a [FieldElement], order: usize) -> Polynomial<'a> {
        Polynomial { coeff, order }
    }
}

// if you want a full multiplication, then make res.order = l.order + r.order
// but if you just care about a lower order, e.g. mul mod x^i, then you can select
//    fewer coefficients
//
// `res`/`res_order` is the destination (degree res_order, so res has res_order+1
// coefficients).
pub fn polynomial_mul(field: &Field, l: &Polynomial, r: &Polynomial, res: &mut [FieldElement], res_order: usize) {
    // perform an element-wise multiplication of two polynomials
    for c in res[..res_order + 1].iter_mut() {
        *c = 0;
    }
    for i in 0..=l.order {
        if i > res_order {
            continue;
        }
        let j_limit = if r.order > res_order - i {
            res_order - i
        } else {
            r.order
        };
        for j in 0..=j_limit {
            // e.g. alpha^5*x * alpha^37*x^2 --> alpha^42*x^3
            res[i + j] = field.add(res[i + j], field.mul(l.coeff[i], r.coeff[j]));
        }
    }
}

// find the polynomial remainder of dividend mod divisor
// do long division and return just the remainder (written to mod_)
//
// `mod_`/`mod_order` is the destination; mod_order must be >= dividend.order
// because the long division uses it as scratch space.
pub fn polynomial_mod(
    field: &Field,
    dividend: &Polynomial,
    divisor: &Polynomial,
    mod_: &mut [FieldElement],
    mod_order: usize,
) {
    if mod_order < dividend.order {
        // mod.order must be >= dividend.order (scratch space needed)
        // this is an error -- catch it in debug?
        return;
    }
    // initialize remainder as dividend
    mod_[..dividend.order + 1].copy_from_slice(&dividend.coeff[..dividend.order + 1]);

    // XXX make sure divisor[divisor_order] is nonzero
    let divisor_leading: FieldLogarithm = field.log[divisor.coeff[divisor.order] as usize];
    // long division steps along one order at a time, starting at the highest order
    let mut i = dividend.order;
    while i > 0 {
        // look at the leading coefficient of dividend and divisor
        // if leading coefficient of dividend / leading coefficient of divisor is q
        //   then the next row of subtraction will be q * divisor
        // if order of q < 0 then what we have is the remainder and we are done
        if i < divisor.order {
            break;
        }
        if mod_[i] == 0 {
            i -= 1;
            continue;
        }
        let q_order = i - divisor.order;
        let q_coeff: FieldLogarithm = field.div_log(field.log[mod_[i] as usize], divisor_leading);

        // now that we've chosen q, multiply the divisor by q and subtract from
        //   our remainder. subtracting in GF(2^8) is XOR, just like addition
        let dcoeff = &divisor.coeff[..=divisor.order];
        let window = &mut mod_[q_order..=q_order + divisor.order];
        for (m, &d) in window.iter_mut().zip(dcoeff.iter()) {
            if d == 0 {
                continue;
            }
            // all of the multiplication is shifted up by q_order places
            *m = field.add(*m, field.mul_log_element(field.log[d as usize], q_coeff));
        }

        i -= 1;
    }
}

// if f(x) = a(n)*x^n + ... + a(1)*x + a(0)
// then f'(x) = n*a(n)*x^(n-1) + ... + 2*a(2)*x + a(1)
// where n*a(n) = sum(k=1, n, a(n)) e.g. the nth sum of a(n) in GF(2^8)
//
// assumes der.order = poly.order - 1
pub fn polynomial_formal_derivative(field: &Field, poly: &Polynomial, der: &mut [FieldElement], der_order: usize) {
    for c in der[..der_order + 1].iter_mut() {
        *c = 0;
    }
    for i in 0..=der_order {
        // we're filling in the ith power of der, so we look ahead one power in poly
        // f(x) = a(i + 1)*x^(i + 1) -> f'(x) = (i + 1)*a(i + 1)*x^i
        // where (i + 1)*a(i + 1) is the sum of a(i + 1) (i + 1) times, not the product
        der[i] = field.sum(poly.coeff[i + 1], (i + 1) as u32);
    }
}

// evaluate the polynomial poly at a particular element val
#[allow(dead_code)]
pub fn polynomial_eval(field: &Field, poly: &Polynomial, val: FieldElement) -> FieldElement {
    if val == 0 {
        return poly.coeff[0];
    }

    let mut res: FieldElement = 0;

    // we're going to start at 0th order and multiply by val each time
    let mut val_exponentiated: FieldLogarithm = field.log[1];
    let val_log: FieldLogarithm = field.log[val as usize];

    for i in 0..=poly.order {
        if poly.coeff[i] != 0 {
            // multiply-accumulate by the next coeff times the next power of val
            res = field.add(
                res,
                field.mul_log_element(field.log[poly.coeff[i] as usize], val_exponentiated),
            );
        }
        // now advance to the next power
        val_exponentiated = field.mul_log(val_exponentiated, val_log);
    }
    res
}

// evaluate the polynomial poly at a particular element val
// in this case, all of the logarithms of the successive powers of val have been precalculated
// this removes the extra work we'd have to do to calculate val_exponentiated each time
//   if this function is to be called on the same val multiple times
pub fn polynomial_eval_lut(field: &Field, poly: &Polynomial, val_exp: &[FieldLogarithm]) -> FieldElement {
    if val_exp[0] == 0 {
        return poly.coeff[0];
    }

    let mut res: FieldElement = 0;

    let coeff = &poly.coeff[..=poly.order];
    let exps = &val_exp[..=poly.order];
    for (&c, &e) in coeff.iter().zip(exps.iter()) {
        if c != 0 {
            // multiply-accumulate by the next coeff times the next power of val
            res = field.add(res, field.mul_log_element(field.log[c as usize], e));
        }
    }
    res
}

// evaluate the log_polynomial poly at a particular element val
// like polynomial_eval_lut, the logarithms of the successive powers of val have been
//   precomputed
pub fn polynomial_eval_log_lut(field: &Field, poly_log: &Polynomial, val_exp: &[FieldLogarithm]) -> FieldElement {
    if val_exp[0] == 0 {
        if poly_log.coeff[0] == 0 {
            // special case for the non-existant log case
            return 0;
        }
        return field.exp[poly_log.coeff[0] as usize];
    }

    let mut res: FieldElement = 0;

    let coeff = &poly_log.coeff[..=poly_log.order];
    let exps = &val_exp[..=poly_log.order];
    for (&c, &e) in coeff.iter().zip(exps.iter()) {
        // using 0 as a sentinel value in log -- log(0) is really -inf
        if c != 0 {
            // multiply-accumulate by the next coeff times the next power of val
            res = field.add(res, field.mul_log_element(c, e));
        }
    }
    res
}

// create the lookup table of successive powers of val used by polynomial_eval_lut
pub fn polynomial_build_exp_lut(field: &Field, val: FieldElement, order: usize, val_exp: &mut [FieldLogarithm]) {
    let mut val_exponentiated: FieldLogarithm = field.log[1];
    let val_log: FieldLogarithm = field.log[val as usize];
    for i in 0..=order {
        if val == 0 {
            val_exp[i] = 0;
        } else {
            val_exp[i] = val_exponentiated;
            val_exponentiated = field.mul_log(val_exponentiated, val_log);
        }
    }
}

pub fn polynomial_init_from_roots(
    field: &Field,
    nroots: usize,
    roots: &[FieldElement],
    poly: &mut [FieldElement],
    scratch0: &mut [FieldElement],
    scratch1: &mut [FieldElement],
) {
    let order = nroots;

    // l is the linear factor (x + roots[i]); coeff[1] is always 1 (the x term),
    // coeff[0] is filled in each iteration.
    let mut l_coeff: [FieldElement; 2] = [0, 0];

    // we'll keep two temporary stores of rightside polynomial
    // each time through the loop, we take the previous result and use it as new rightside
    // swap back and forth (prevents the need for a copy)
    let (mut cur, mut prev): (&mut [FieldElement], &mut [FieldElement]) = (scratch0, scratch1);
    let mut cur_order;

    // initialize the result with x + roots[0]
    cur[1] = 1;
    cur[0] = roots[0];
    cur_order = 1usize;

    // initialize lcoeff[1] with x
    // we'll fill in the 0th order term in each loop iter
    l_coeff[1] = 1;

    // loop through, using previous run's result as the new right hand side
    // this allows us to multiply one group at a time
    for i in 1..nroots {
        l_coeff[0] = roots[i];
        core::mem::swap(&mut cur, &mut prev);
        let prev_order = cur_order;
        cur_order = i + 1;

        let l = Polynomial::new(&l_coeff, 1);
        let r_poly = Polynomial::new(&prev[..prev_order + 1], prev_order);
        polynomial_mul(field, &l, &r_poly, cur, cur_order);
    }

    // copy the final result into poly
    poly[..order + 1].copy_from_slice(&cur[..order + 1]);
}

// coeff must be of size nroots + 1
// e.g. 2 roots (x + alpha)(x + alpha^2) yields a poly with 3 terms x^2 + g0*x + g1
pub fn reed_solomon_build_generator(
    field: &Field,
    nroots: usize,
    first_consecutive_root: FieldElement,
    root_gap: usize,
    generator: &mut [FieldElement],
    roots: &mut [FieldElement],
) {
    // generator has order 2*t
    // of form (x + alpha^1)(x + alpha^2)...(x - alpha^2*t)
    for i in 0..nroots {
        roots[i] = field.exp[(root_gap * (i + first_consecutive_root as usize)) % 255];
    }
    let mut scratch0 = vec![0 as FieldElement; nroots + 1];
    let mut scratch1 = vec![0 as FieldElement; nroots + 1];
    polynomial_init_from_roots(field, nroots, roots, generator, &mut scratch0, &mut scratch1);
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRIM: super::super::field::FieldOperation = 0x11d;

    #[test]
    fn from_roots_vanishes_at_roots() {
        let field = Field::new(PRIM);
        let roots: [FieldElement; 4] = [1, 2, 3, 4];
        let nroots = roots.len();

        let mut poly = vec![0u8; nroots + 1];
        let mut s0 = vec![0u8; nroots + 1];
        let mut s1 = vec![0u8; nroots + 1];
        polynomial_init_from_roots(&field, nroots, &roots, &mut poly, &mut s0, &mut s1);

        // leading coefficient is 1 (monic), since each factor contributes x
        assert_eq!(poly[nroots], 1);

        let p = Polynomial::new(&poly, nroots);
        // the polynomial must evaluate to zero at each of its roots
        for &r in roots.iter() {
            assert_eq!(polynomial_eval(&field, &p, r), 0, "root {r} did not vanish");
        }
        // and (almost surely) nonzero at a non-root
        assert_ne!(polynomial_eval(&field, &p, 99), 0);
    }

    #[test]
    fn eval_lut_matches_direct() {
        let field = Field::new(PRIM);
        let coeff: [FieldElement; 5] = [0x1f, 0x00, 0xa3, 0x07, 0x01];
        let p = Polynomial::new(&coeff, 4);

        for val in 0..=255u16 {
            let val = val as FieldElement;
            let direct = polynomial_eval(&field, &p, val);

            let mut lut = vec![0u8; p.order + 1];
            polynomial_build_exp_lut(&field, val, p.order, &mut lut);
            let via_lut = polynomial_eval_lut(&field, &p, &lut);

            assert_eq!(direct, via_lut, "mismatch at val={val:#x}");
        }
    }

    #[test]
    fn eval_log_lut_matches_direct() {
        let field = Field::new(PRIM);
        // a polynomial whose coefficients are all nonzero, so we can take a
        // log-domain copy without hitting the log(0) sentinel
        let coeff: [FieldElement; 4] = [0x1f, 0x42, 0xa3, 0x07];
        let p = Polynomial::new(&coeff, 3);

        // log-domain version: each coeff replaced by its logarithm
        let coeff_log: Vec<u8> = coeff.iter().map(|&c| field.log[c as usize]).collect();
        let p_log = Polynomial::new(&coeff_log, 3);

        for val in 1..=255u16 {
            let val = val as FieldElement;
            let direct = polynomial_eval(&field, &p, val);

            let mut lut = vec![0u8; p.order + 1];
            polynomial_build_exp_lut(&field, val, p.order, &mut lut);
            let via_log_lut = polynomial_eval_log_lut(&field, &p_log, &lut);

            assert_eq!(direct, via_log_lut, "mismatch at val={val:#x}");
        }
    }

    #[test]
    fn mul_then_mod_is_zero() {
        let field = Field::new(PRIM);
        // (x + 1)(x + 2) as a product of two linear factors
        let a = [1u8, 1u8]; // 1 + x  (order 1)
        let b = [2u8, 1u8]; // 2 + x  (order 1)
        let pa = Polynomial::new(&a, 1);
        let pb = Polynomial::new(&b, 1);

        let mut prod = [0u8; 3];
        polynomial_mul(&field, &pa, &pb, &mut prod, 2);

        // dividing the product by (x + 1) must leave zero remainder
        let prod_poly = Polynomial::new(&prod, 2);
        let mut rem = [0u8; 3];
        polynomial_mod(&field, &prod_poly, &pa, &mut rem, 2);
        // remainder has degree < divisor.order (=1), so only rem[0] matters
        assert_eq!(rem[0], 0, "expected exact division, got remainder {:?}", rem);
    }

    #[test]
    fn truncated_mul_keeps_low_orders() {
        let field = Field::new(PRIM);
        let a = [3u8, 5u8, 7u8]; // order 2
        let b = [2u8, 4u8, 6u8]; // order 2
        let pa = Polynomial::new(&a, 2);
        let pb = Polynomial::new(&b, 2);

        // full product has order 4
        let mut full = [0u8; 5];
        polynomial_mul(&field, &pa, &pb, &mut full, 4);

        // mul mod x^3 (truncated to order 2) must equal the low coefficients of full
        let mut trunc = [0u8; 3];
        polynomial_mul(&field, &pa, &pb, &mut trunc, 2);

        assert_eq!(&trunc[..3], &full[..3]);
    }

    #[test]
    fn formal_derivative_drops_even_terms() {
        let field = Field::new(PRIM);
        // p = a0 + a1 x + a2 x^2 + a3 x^3
        let coeff = [0x10u8, 0x20, 0x30, 0x40];
        let p = Polynomial::new(&coeff, 3);

        let mut der = [0u8; 3];
        polynomial_formal_derivative(&field, &p, &mut der, 2);

        // der[i] = sum(coeff[i+1], i+1): odd multiplier keeps the term, even kills it
        // i=0 -> n=1 (odd)  -> coeff[1]
        // i=1 -> n=2 (even) -> 0
        // i=2 -> n=3 (odd)  -> coeff[3]
        assert_eq!(der[0], coeff[1]);
        assert_eq!(der[1], 0);
        assert_eq!(der[2], coeff[3]);
    }
}
