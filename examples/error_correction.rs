//! Shows the convolutional decoder correcting real bit errors.
//!
//! The clean round-trip in `main.rs` shows encode and decode with no noise.
//! This example corrupts the encoded data first, then recovers the original
//! message anyway, which is the whole point of an error-correcting code.
//!
//! Run with `cargo run --example error_correction`.

use fec::convolutional::{Decoder, Encoder};

fn main() {
    // Rate-1/2, order-7 code (the classic NASA/Voyager k=7).
    let polys: [u16; 2] = [0o161, 0o127];
    let message: &[u8] = b"the quick brown fox jumps over the lazy dog";

    let mut encoder = Encoder::new(2, 7, &polys);
    let enc_len = encoder.encode_len(message.len());
    let mut encoded = vec![0u8; enc_len.div_ceil(8)];
    encoder.encode(message, &mut encoded).expect("encode");

    // Corrupt the channel. We flip one bit per 100-bit window, far enough apart
    // that the trellis reconverges between them so they stay correctable.
    let mut errors = 0;
    let mut bit = 50;
    while bit < enc_len {
        encoded[bit / 8] ^= 0x80 >> (bit % 8);
        errors += 1;
        bit += 100;
    }
    println!("introduced {errors} bit errors into the channel");

    // Decode the corrupted data. The decoder recovers the original message.
    let mut decoded = vec![0u8; message.len()];
    let mut decoder = Decoder::new(2, 7, &polys);
    decoder
        .decode_hard(&encoded, enc_len, &mut decoded)
        .expect("decode");

    println!("recovered: {:?}", String::from_utf8_lossy(&decoded));
    assert_eq!(decoded, message, "decoder failed to correct the errors");
    println!("recovered the original message despite the errors");
}
