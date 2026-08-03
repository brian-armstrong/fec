//! Parameters and symbol transforms for the standard CCSDS (255,223)
//! Reed-Solomon code.
//!
//! Spacecraft telemetry represents RS symbols in Berlekamp's dual basis, but a
//! conventional codec works in the power-of-alpha basis. The two bases are
//! related by a fixed linear map over GF(2). [`conv_to_dual`] and
//! [`dual_to_conv`] convert a byte between them, so you can encode and decode
//! with the conventional codec while the wire stays in the dual basis. The
//! `CCSDS_*` constants are the code's field and generator parameters, ready to
//! pass to [`Encoder::new`](super::Encoder::new).
//!
//! The transform is derived from CCSDS 131.0-B-1, "TM Synchronization and
//! Channel Coding", Annex D.

// SOURCE / DERIVATION
// -------------------
// The conversion is the fixed GF(2) linear map given explicitly in CCSDS
// 131.0-B-1, Annex D ("Transformation between Berlekamp and Conventional
// Representations"), matrix Tα (conventional -> dual) and its inverse Tα^-1.

use super::field::{FieldLogarithm, FieldOperation};

/// Field generator polynomial of the CCSDS code, x^8 + x^7 + x^2 + x + 1.
pub const CCSDS_PRIMITIVE_POLYNOMIAL: FieldOperation = 0x187;
/// First consecutive root of the CCSDS code generator (index form).
pub const CCSDS_FIRST_CONSECUTIVE_ROOT: FieldLogarithm = 112;
/// Primitive element power used to generate the CCSDS code roots.
pub const CCSDS_GENERATOR_ROOT_GAP: FieldLogarithm = 11;
/// Number of generator roots (parity symbols) in the CCSDS (255,223) code.
pub const CCSDS_NUM_ROOTS: usize = 32;

// CCSDS 131.0-B-1 Annex D, Table D-1: Transformation Matrix Tα (conventional -> dual).
// Row r (r = 0..7) is the dual-basis representation of alpha^(7-r).
// Each row lists [z0 z1 ... z7].
const T_ALPHA: [[u8; 8]; 8] = [
    [1, 0, 0, 0, 1, 1, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 1, 1, 0],
    [1, 1, 1, 1, 1, 0, 1, 0],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 0, 1, 1],
];

// Convert one conventional-basis byte to its dual-basis byte, computing
// [z0,...,z7] = [u7,...,u0] Tα  (CCSDS 131.0-B-1 Annex D, section D2).
//
// The two byte-packing conventions below come straight from Annex D:
//   - conventional input: "[u7, u6, ... , u0] ... coefficients of α^j" -- so
//     the coefficient of α^(7-r) is bit (7-r) of x, which selects Tα row r
//     ("Row 1 ... in Tα are representations ... of α7 (10...0)").
//   - dual output: "[z0, z1, ... , z7] ... coefficients of l_i" listed MSB-first
//     (the "l01234567" column of Table D-1) -- so z_c goes to output bit (7-c).
//
// Verified against Table D-1 by the anchor test below (e.g. α^0 = 0x01 -> 0x7b).
const fn conv_to_dual_byte(x: u8) -> u8 {
    let mut z = [0u8; 8];
    let mut r = 0;
    while r < 8 {
        if (x >> (7 - r)) & 1 == 1 {
            let mut c = 0;
            while c < 8 {
                z[c] ^= T_ALPHA[r][c];
                c += 1;
            }
        }
        r += 1;
    }
    let mut out = 0u8;
    let mut c = 0;
    while c < 8 {
        if z[c] == 1 {
            out |= 1 << (7 - c);
        }
        c += 1;
    }
    out
}

const fn build_conv_to_dual() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = conv_to_dual_byte(i as u8);
        i += 1;
    }
    t
}

const fn build_dual_to_conv(fwd: &[u8; 256]) -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        t[fwd[i] as usize] = i as u8;
        i += 1;
    }
    t
}

/// Lookup table mapping a conventional-basis byte to its CCSDS dual-basis byte.
pub const CONV_TO_DUAL: [u8; 256] = build_conv_to_dual();
/// Lookup table mapping a CCSDS dual-basis byte to its conventional-basis byte.
pub const DUAL_TO_CONV: [u8; 256] = build_dual_to_conv(&CONV_TO_DUAL);

/// Converts one conventional-basis byte to its CCSDS dual-basis byte. This is
/// the direction applied to parity on its way onto the wire.
#[inline]
pub fn conv_to_dual(byte: u8) -> u8 {
    CONV_TO_DUAL[byte as usize]
}

/// Converts one CCSDS dual-basis byte back to its conventional-basis byte. This
/// is the direction applied to symbols received from the wire.
#[inline]
pub fn dual_to_conv(byte: u8) -> u8 {
    DUAL_TO_CONV[byte as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_are_inverses() {
        for i in 0..256usize {
            assert_eq!(DUAL_TO_CONV[CONV_TO_DUAL[i] as usize], i as u8);
            assert_eq!(CONV_TO_DUAL[DUAL_TO_CONV[i] as usize], i as u8);
        }
    }

    #[test]
    fn matches_ccsds_table_d1_anchors() {
        // CCSDS 131.0-B-1 Annex D, Table D-1: alpha-poly byte -> dual byte
        assert_eq!(CONV_TO_DUAL[0x00], 0x00);
        assert_eq!(CONV_TO_DUAL[0x01], 0x7b); // alpha^0
        assert_eq!(CONV_TO_DUAL[0x02], 0xaf); // alpha^1
        assert_eq!(CONV_TO_DUAL[0x04], 0x99); // alpha^2
        assert_eq!(CONV_TO_DUAL[0x80], 0x8d); // alpha^7
        assert_eq!(CONV_TO_DUAL[0xc3], 0xb6); // alpha^254
    }
}
