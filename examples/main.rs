use fec::convolutional::{Decoder, Encoder};

fn main() {
    let bytes: [u8; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let polys: [u16; 2] = [0o161, 0o127];
    let mut encoder = Encoder::new(2, 7, &polys);

    let enc_len = encoder.encode_len(bytes.len());
    let enc_len_bytes = enc_len / 8 + 1;
    let mut encoded = vec![0; enc_len_bytes];

    encoder.encode(&bytes, &mut encoded).expect("encode failed");

    println!("Encoded {} bits:", enc_len);
    for b in &encoded {
        print!("{:02x} ", b);
    }
    println!();

    let mut decoder = Decoder::new(2, 7, &polys);
    let mut decoded = vec![0; bytes.len()];
    let decoded_len = decoder
        .decode_hard(&encoded, enc_len, &mut decoded)
        .expect("decode failed");

    println!("Decoded {} bytes:", decoded_len);
    println!("{:02x?}", &decoded[..decoded_len]);

    assert_eq!(&bytes[..], &decoded[..], "Round-trip failed");
    println!("Round-trip OK");
}
