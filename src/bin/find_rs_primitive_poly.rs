// Translation of libcorrect's tools/find_rs_primitive_poly.c
//
// Searches for primitive polynomials of GF(2^8). A polynomial is primitive if
// the primitive element alpha (=2), exponentiated through all 255 nonzero
// powers, visits every nonzero field element exactly once. We test each
// candidate by building its log table and checking for collisions, then
// pretty-print the valid ones as x^8 + x^7 + ... form.
//
// These are the polynomials behind the PRIMITIVE_POLYNOMIAL_* constants in
// the reed_solomon module.
//
// Run with: cargo run --bin find_rs_primitive_poly

type FieldOperation = u16;
type FieldLogarithm = u8;

const BLOCK_SIZE: FieldOperation = 255;
const POWER_MAX: i32 = 8;

// visit all of the elements from the poly
fn trypoly(poly: FieldOperation, log: &mut [FieldLogarithm]) -> bool {
    for l in log.iter_mut() {
        *l = 0;
    }
    let mut element: FieldOperation = 1;
    log[0] = 0;
    for i in 1..(BLOCK_SIZE + 1) {
        element = element * 2;
        element = if element > BLOCK_SIZE {
            element ^ poly
        } else {
            element
        };
        if log[element as usize] != 0 {
            return false;
        }
        log[element as usize] = i as FieldLogarithm;
    }
    true
}

fn printpoly(poly: FieldOperation) {
    let mut poly = poly;
    let mut power = POWER_MAX;
    while poly != 0 {
        if poly & (BLOCK_SIZE + 1) != 0 {
            if power > 1 {
                print!("x^{}", power);
            } else if power != 0 {
                print!("x");
            } else {
                print!("1");
            }
            if poly & BLOCK_SIZE != 0 {
                print!(" + ");
            }
        }
        power -= 1;
        poly <<= 1;
        poly &= (BLOCK_SIZE << 1) + 1;
    }
}

fn main() {
    let mut log = vec![0 as FieldLogarithm; (BLOCK_SIZE + 1) as usize];
    for i in (BLOCK_SIZE + 1)..((BLOCK_SIZE + 1) << 1) {
        if trypoly(i, &mut log) {
            print!("0x{:x} valid: ", i);
            printpoly(i);
            println!();
        }
    }
}
