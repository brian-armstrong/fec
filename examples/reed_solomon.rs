//! Shows the Reed-Solomon decoder correcting byte errors.
//!
//! This uses the standard CCSDS (255,223) code. It encodes a message into a
//! 255-byte block, corrupts several bytes, then recovers the original. Unlike
//! the convolutional decoder, Reed-Solomon reports how many symbols it
//! corrected.
//!
//! Run with `cargo run --example reed_solomon`.

use fec::reed_solomon::{Decoder, Encoder};

fn main() {
    // Standard CCSDS (255,223): a 223-byte message, 32 parity bytes, 255-byte
    // block. It corrects up to 16 corrupted bytes per block.
    let mut encoder = Encoder::new_ccsds();
    let mut decoder = Decoder::new_ccsds();

    let message: Vec<u8> = (0..223).collect();
    let mut block = vec![0u8; 255];
    encoder.encode(&message, &mut block).expect("encode");

    // Corrupt 10 bytes, within the 16-byte correction limit.
    let corrupt_positions = [3, 40, 77, 100, 128, 150, 190, 200, 233, 254];
    for &pos in &corrupt_positions {
        block[pos] ^= 0xff;
    }
    println!("corrupted {} bytes of the 255-byte block", corrupt_positions.len());

    // Decode. The return value is the number of byte errors corrected.
    let mut recovered = vec![0u8; 223];
    let corrected = decoder.decode(&block, &mut recovered).expect("decode");

    println!("decoder corrected {corrected} byte errors");
    assert_eq!(recovered, message, "decoder failed to correct the errors");
    println!("recovered the original 223-byte message");
}
