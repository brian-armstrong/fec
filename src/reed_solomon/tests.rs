// Tests for the Reed-Solomon encoder and decoder.

use crate::reed_solomon::{DecodeError, EncodeError, FieldOperation, Decoder, Encoder};
use crate::util::Rng;

// ccsds primitive poly, first_consecutive_root=1, gap=1
const CCSDS: FieldOperation = 0x187;

struct Codec {
    enc: Encoder,
    dec: Decoder,
}
impl Codec {
    fn new(num_roots: usize) -> Codec {
        Codec {
            enc: Encoder::new(CCSDS, 1, 1, num_roots),
            dec: Decoder::new(CCSDS, 1, 1, num_roots),
        }
    }
    fn encode(&mut self, msg: &[u8], encoded: &mut [u8]) -> Result<usize, EncodeError> {
        self.enc.encode(msg, encoded)
    }
    fn decode(&mut self, encoded: &[u8], msg: &mut [u8]) -> Result<usize, DecodeError> {
        self.dec.decode(encoded, msg)
    }
    fn decode_with_erasures(
        &mut self,
        encoded: &[u8],
        erasures: &[u8],
        msg: &mut [u8],
    ) -> Result<usize, DecodeError> {
        self.dec.decode_with_erasures(encoded, erasures, msg)
    }
    // mirror the read-only accessors a few tests check
    fn min_distance(&self) -> usize {
        self.dec.min_distance()
    }
    fn message_length(&self) -> usize {
        self.dec.message_length()
    }
}

fn rs32() -> Codec {
    Codec::new(32)
}

#[test]
fn message_length_is_block_minus_distance() {
    let dec = Decoder::new(CCSDS, 1, 1, 32);
    assert_eq!(dec.block_length(), 255);
    assert_eq!(dec.min_distance(), 32);
    assert_eq!(dec.message_length(), 223);
}

#[test]
fn decode_clean_block_roundtrips() {
    let mut rs = rs32();
    let msg: Vec<u8> = (0..223u16).map(|i| i as u8).collect();
    let mut encoded = vec![0u8; 255];
    rs.encode(&msg, &mut encoded).unwrap();

    let mut decoded = vec![0u8; 223];
    // a clean block needs zero corrections
    assert_eq!(rs.decode(&encoded, &mut decoded), Ok(0));
    assert_eq!(&decoded[..], &msg[..]);
}

#[test]
fn decode_corrects_up_to_t_errors() {
    // t = min_distance / 2 = 16; corrupt exactly 16 bytes
    let mut rs = rs32();
    let msg: Vec<u8> = (0..223u16).map(|i| i as u8).collect();
    let mut encoded = vec![0u8; 255];
    rs.encode(&msg, &mut encoded).unwrap();

    let positions = [
        0usize, 5, 17, 33, 64, 99, 120, 150, 177, 200, 222, 223, 230, 240, 250, 254,
    ];
    let mut corrupted = encoded.clone();
    for &p in positions.iter() {
        corrupted[p] ^= 0x5A;
    }

    let mut decoded = vec![0u8; 223];
    // 16 errors injected -> 16 corrections reported
    assert_eq!(rs.decode(&corrupted, &mut decoded), Ok(16));
    assert_eq!(&decoded[..], &msg[..]);
}

#[test]
fn decode_rejects_too_many_errors() {
    // 20 errors > t=16; decode must report failure
    let mut rs = rs32();
    let msg: Vec<u8> = (0..223u16).map(|i| i as u8).collect();
    let mut encoded = vec![0u8; 255];
    rs.encode(&msg, &mut encoded).unwrap();

    let mut over = encoded.clone();
    for i in 0..20usize {
        over[(i * 11) % 255] ^= 0x77;
    }

    let mut decoded = vec![0u8; 223];
    assert_eq!(rs.decode(&over, &mut decoded), Err(DecodeError::TooManyErrors));
}

#[test]
fn decode_single_error_each_position() {
    // exhaustive single-error correction: flip one byte at every position and
    // confirm recovery. catches off-by-one bugs in location mapping.
    let msg: Vec<u8> = (0..223u16).map(|i| (i as u8).wrapping_mul(7)).collect();
    let mut rs = rs32();
    let mut encoded = vec![0u8; 255];
    rs.encode(&msg, &mut encoded).unwrap();

    for pos in 0..255usize {
        let mut corrupted = encoded.clone();
        corrupted[pos] ^= 0xFF;
        let mut decoded = vec![0u8; 223];
        assert_eq!(
            rs.decode(&corrupted, &mut decoded),
            Ok(1),
            "decode failed for error at position {pos}"
        );
        assert_eq!(&decoded[..], &msg[..], "wrong decode for error at position {pos}");
    }
}

#[test]
fn decode_short_message_roundtrips() {
    // short messages use virtual padding: encode emits msg + 32 parity, and
    // we decode that trimmed block (length msg_length + min_distance).
    let mut rs = rs32();
    let msg = [10u8, 20, 30, 40, 50];
    let mut encoded = vec![0u8; 255];
    rs.encode(&msg, &mut encoded).unwrap();

    // the meaningful emitted block is msg (5) + parity (32) = 37 bytes
    let block_len = msg.len() + rs.min_distance();
    let mut corrupted = encoded[..block_len].to_vec();
    // inject 2 errors (<= t)
    corrupted[0] ^= 0x11;
    corrupted[36] ^= 0x33;

    let mut decoded = vec![0u8; msg.len()];
    // 2 errors injected -> 2 corrections
    assert_eq!(rs.decode(&corrupted, &mut decoded), Ok(2));
    assert_eq!(&decoded[..], &msg[..]);
}

#[test]
fn decode_erasures_only() {
    // 28 corrupted bytes, all at KNOWN positions (erasures). With 32 parity
    // and erasures costing 1 each, this is recoverable.
    let mut rs = rs32();
    let msg: Vec<u8> = (0..223u16).map(|i| i as u8).collect();
    let mut encoded = vec![0u8; 255];
    rs.encode(&msg, &mut encoded).unwrap();

    let mut corrupted = encoded.clone();
    let mut erasures = Vec::new();
    for i in 0..28usize {
        let pos = (i * 9 + 1) % 255;
        erasures.push(pos as u8);
        corrupted[pos] ^= 0xC3;
    }

    let mut decoded = vec![0u8; 223];
    // 28 erasures -> 28 corrections
    assert_eq!(
        rs.decode_with_erasures(&corrupted, &erasures, &mut decoded),
        Ok(28)
    );
    assert_eq!(&decoded[..], &msg[..]);
}

#[test]
fn decode_mixed_erasures_and_errors() {
    // 10 erasures (known) + 10 errors (unknown): budget 2*10 + 10 = 30 <= 32.
    let mut rs = rs32();
    let msg: Vec<u8> = (0..223u16).map(|i| i as u8).collect();
    let mut encoded = vec![0u8; 255];
    rs.encode(&msg, &mut encoded).unwrap();

    let mut corrupted = encoded.clone();
    let mut erasures = Vec::new();
    for i in 0..10usize {
        let pos = (i * 7 + 2) % 255;
        erasures.push(pos as u8);
        corrupted[pos] ^= 0x5A;
    }
    for &pos in &[3usize, 40, 77, 110, 143, 176, 199, 220, 245, 253] {
        corrupted[pos] ^= 0x91;
    }

    let mut decoded = vec![0u8; 223];
    // 10 erasures + 10 errors -> 20 total corrections
    assert_eq!(
        rs.decode_with_erasures(&corrupted, &erasures, &mut decoded),
        Ok(20)
    );
    assert_eq!(&decoded[..], &msg[..]);
}

#[test]
fn decode_with_erasures_delegates_when_empty() {
    // erasure_length == 0 falls back to plain decode.
    let mut rs = rs32();
    let msg: Vec<u8> = (0..223u16).map(|i| i as u8).collect();
    let mut encoded = vec![0u8; 255];
    rs.encode(&msg, &mut encoded).unwrap();

    let mut corrupted = encoded.clone();
    corrupted[5] ^= 0x01;
    corrupted[100] ^= 0x02;

    let mut decoded = vec![0u8; 223];
    // delegates to plain decode; 2 errors -> 2 corrections
    assert_eq!(rs.decode_with_erasures(&corrupted, &[], &mut decoded), Ok(2));
    assert_eq!(&decoded[..], &msg[..]);
}

#[test]
fn decode_clean_block_with_declared_erasures() {
    // an uncorrupted block, but with erasures declared anyway (positions that
    // aren't actually wrong). The all-zero-syndromes fast path applies.
    let mut rs = rs32();
    let msg: Vec<u8> = (0..223u16).map(|i| i as u8).collect();
    let mut encoded = vec![0u8; 255];
    rs.encode(&msg, &mut encoded).unwrap();

    let erasures = [1u8, 2, 3, 4];
    let mut decoded = vec![0u8; 223];
    // block is actually clean -> all-zero syndromes -> zero corrections
    assert_eq!(
        rs.decode_with_erasures(&encoded, &erasures, &mut decoded),
        Ok(0)
    );
    assert_eq!(&decoded[..], &msg[..]);
}

#[test]
fn decode_with_erasures_rejects_too_many_erasures() {
    // erasure_length > min_distance is rejected up front
    let mut rs = rs32();
    let msg: Vec<u8> = (0..223u16).map(|i| i as u8).collect();
    let mut encoded = vec![0u8; 255];
    rs.encode(&msg, &mut encoded).unwrap();

    let erasures: Vec<u8> = (0..33u8).collect(); // 33 > 32
    let mut decoded = vec![0u8; 223];
    assert_eq!(
        rs.decode_with_erasures(&encoded, &erasures, &mut decoded),
        Err(DecodeError::TooManyErasures)
    );
}

#[test]
fn decode_then_encode_again_reuses_state() {
    // run an erasure decode, then a plain encode, then a plain decode on the
    // same object, to confirm the shared/erasure scratch is properly reset
    // between calls and one operation doesn't corrupt the next.
    let mut rs = rs32();
    let msg: Vec<u8> = (0..223u16).map(|i| (i as u8) ^ 0x3C).collect();
    let mut encoded = vec![0u8; 255];
    rs.encode(&msg, &mut encoded).unwrap();

    // erasure decode
    let mut corrupted = encoded.clone();
    let erasures = [10u8, 50, 90, 130];
    for &p in &erasures {
        corrupted[p as usize] ^= 0x77;
    }
    let mut decoded = vec![0u8; 223];
    // 4 erasures -> 4 corrections
    assert_eq!(
        rs.decode_with_erasures(&corrupted, &erasures, &mut decoded),
        Ok(4)
    );
    assert_eq!(&decoded[..], &msg[..]);

    // now a plain error decode on the same object: 1 error -> 1 correction
    let mut corrupted2 = encoded.clone();
    corrupted2[7] ^= 0x11;
    let mut decoded2 = vec![0u8; 223];
    assert_eq!(rs.decode(&corrupted2, &mut decoded2), Ok(1));
    assert_eq!(&decoded2[..], &msg[..]);
}

#[test]
fn decode_different_param_code() {
    // a different (smaller) code: min_distance=8 -> corrects up to 4 errors
    let mut rs = Codec::new(8);
    assert_eq!(rs.message_length(), 247);
    let msg: Vec<u8> = (0..247u16).map(|i| i as u8).collect();
    let mut encoded = vec![0u8; 255];
    rs.encode(&msg, &mut encoded).unwrap();

    let mut corrupted = encoded.clone();
    for &p in &[3usize, 88, 199, 254] {
        corrupted[p] ^= 0x42;
    }

    let mut decoded = vec![0u8; 247];
    // 4 errors injected -> 4 corrections
    assert_eq!(rs.decode(&corrupted, &mut decoded), Ok(4));
    assert_eq!(&decoded[..], &msg[..]);
}

fn stress_shuffle(a: &mut [usize], rng: &mut Rng) {
    let len = a.len();
    for i in 0..len - 2 {
        let j = rng.next_u64_below(len - i) + i;
        a.swap(i, j);
    }
}

fn stress_one(
    rs: &mut Codec,
    msg_length: usize,
    num_errors: usize,
    num_erasures: usize,
    rng: &mut Rng,
) {
    let min_distance = rs.min_distance();
    let block_length = msg_length + min_distance;

    let mut msg = vec![0u8; msg_length];
    for b in msg.iter_mut() {
        *b = rng.next_u64_below(256) as u8;
    }

    let mut encoded = vec![0u8; 255];
    rs.encode(&msg, &mut encoded).unwrap();

    let mut corrupted = encoded.clone();
    let mut indices: Vec<usize> = (0..block_length).collect();
    stress_shuffle(&mut indices, rng);

    let mut erasures = vec![0u8; num_erasures];
    for i in 0..num_erasures {
        let index = indices[i];
        let mask = (rng.next_u64_below(255) + 1) as u8;
        corrupted[index] ^= mask;
        erasures[i] = index as u8;
    }
    for i in 0..num_errors {
        let index = indices[i + num_erasures];
        let mask = (rng.next_u64_below(255) + 1) as u8;
        corrupted[index] ^= mask;
    }

    let mut recvmsg = vec![0u8; msg_length];
    let res =
        rs.decode_with_erasures(&corrupted[..block_length], &erasures, &mut recvmsg);
    // every corrupted position is distinct with a nonzero mask, so the
    // decoder should report exactly errors+erasures corrections
    assert_eq!(
        res,
        Ok(num_errors + num_erasures),
        "decode failed: msg_len={msg_length} errors={num_errors} erasures={num_erasures}"
    );
    assert_eq!(
        &recvmsg[..],
        &msg[..],
        "decode mismatch: msg_len={msg_length} errors={num_errors} erasures={num_erasures}"
    );
}

// run the full 4-code x 8-shape matrix for `iters` iterations each
fn stress_matrix(iters: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    for &min_distance in &[32usize, 16, 8, 4] {
        let mut rs = Codec::new(min_distance);
        let message_length = rs.message_length();
        let half = message_length / 2;
        let t = min_distance / 2; // correction capacity

        // (msg_length, num_errors, num_erasures)
        let shapes: [(usize, usize, usize); 8] = [
            (half, 0, 0),
            (message_length, 0, 0),
            (half, t, 0),
            (message_length, t, 0),
            (half, 0, min_distance),
            (message_length, 0, min_distance),
            (half, min_distance / 4, t),
            (message_length, min_distance / 4, t),
        ];

        for &(msg_length, num_errors, num_erasures) in shapes.iter() {
            for _ in 0..iters {
                stress_one(&mut rs, msg_length, num_errors, num_erasures, &mut rng);
            }
        }
    }
}

#[test]
fn stress_roundtrip_fast() {
    stress_matrix(50, 0xC0FF_EE12_3456_789A);
}

#[test]
#[ignore = "640k randomized stress test`"]
fn stress_roundtrip_heavy() {
    // 20k iters x 8 shapes x 4 codes = 640k randomized round-trips.
    // Opt-in for deep checks.
    stress_matrix(20_000, 0x1234_5678_9ABC_DEF0);
}

#[test]
fn ccsds_constructor_matches_explicit_params() {
    // new_ccsds() must equal new() with the CCSDS parameters: same parity.
    let mut a = Encoder::new_ccsds();
    let mut b = Encoder::new(0x187, 112, 11, 32);
    let msg: Vec<u8> = (0..223u16).map(|i| (i as u8).wrapping_mul(3)).collect();
    let (mut ea, mut eb) = (vec![0u8; 255], vec![0u8; 255]);
    a.encode(&msg, &mut ea).unwrap();
    b.encode(&msg, &mut eb).unwrap();
    assert_eq!(ea, eb);
    assert_eq!(
        &ea[223..231],
        &[7, 241, 220, 182, 219, 39, 138, 175]
    );
}

#[test]
fn ccsds_dual_basis() {
    let mut enc = Encoder::new_ccsds();
    let msg: Vec<u8> = (0..223u16).map(|i| ((i * 37 + 11) & 0xff) as u8).collect();
    let mut parity = vec![0u8; 32];
    enc.encode_ccsds_dual(&msg, &mut parity).unwrap();
    let expected: [u8; 32] = [
        0x5e, 0x90, 0x7c, 0x02, 0xde, 0xac, 0x84, 0x37, 0x2f, 0xb4, 0x52, 0x39, 0x29, 0x72,
        0x77, 0x61, 0xbc, 0x4d, 0xf1, 0x0b, 0x7a, 0xc5, 0xc5, 0x04, 0x2b, 0x25, 0x8d, 0xb0,
        0x17, 0xb1, 0x35, 0xed,
    ];
    assert_eq!(&parity[..], &expected);
}

#[test]
fn ccsds_dual_basis_roundtrip_corrects() {
    // full dual-basis round-trip with errors, via the native API
    let mut enc = Encoder::new_ccsds();
    let mut dec = Decoder::new_ccsds();
    let msg: Vec<u8> = (0..223u16).map(|i| (i as u8) ^ 0xA5).collect();
    let mut parity = vec![0u8; 32];
    enc.encode_ccsds_dual(&msg, &mut parity).unwrap();

    // assemble the dual-basis block: msg | parity
    let mut block = msg.clone();
    block.extend_from_slice(&parity);
    // corrupt 16 (= t) bytes
    for &p in &[1usize, 9, 30, 55, 80, 111, 140, 160, 177, 200, 222, 230, 240, 250, 252, 254] {
        block[p] ^= 0x5A;
    }

    let mut recovered = vec![0u8; 223];
    assert_eq!(dec.decode_ccsds_dual(&block, &mut recovered), Ok(16));
    assert_eq!(&recovered[..], &msg[..]);
}
