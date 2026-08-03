# fec

Forward error correction for SDR, space, and satellite applications.

`fec` implements two error-correcting codes that show up throughout
software-defined radio and spacecraft links:

- **Convolutional codes** with a Viterbi decoder (hard and soft decision),
  including the common rate-1/2 k=7, rate-1/2 k=9, rate-1/3 k=9, and
  rate-1/6 k=15 codes. Supports any rate from 1/2 to 1/8 and any order from
  k=4 to k=16. On nightly Rust, the `simd` feature enables a Viterbi decoder
  with acceleration on SSE/AVX2/AVX512.
- **Reed–Solomon codes** over GF(2⁸) with error and erasure decoding, including
  the standard **CCSDS (255,223)** code in both the conventional and the
  on-the-wire **dual-basis** (Berlekamp) representations.

`fec` started as and draws heavy inspiration from the author's own
[libcorrect](https://github.com/github/libcorrect), a C library
for forward error correction. This crate also credits Phil Karn's libfec
C library for offering an original implementation of these codes, although
this crate does not borrow any source or have any relationship with that
library, and the name is purely coincidental.

Standard parameters (primitive polynomials, the CCSDS dual-basis transform) are
derived from the published CCSDS standard ([CCSDS 131.0-B](https://public.ccsds.org/),
Annex D for the dual basis).

## Quick start

### Convolutional (Viterbi)

```rust
use fec::{ConvEncoder, ConvDecoder};

// Rate-1/2, order-7 NASA code.
let polys = [0o161, 0o127];
let mut enc = ConvEncoder::new(2, 7, &polys);
let mut dec = ConvDecoder::new(2, 7, &polys);

let msg = b"hello, error correction";
let mut encoded = vec![0u8; enc.encode_len(msg.len())];
let num_bits = enc.encode(msg, &mut encoded).unwrap();

// ... encoded is corrupted in transit ...

let mut recovered = vec![0u8; msg.len()];
dec.decode_hard(&encoded, num_bits, &mut recovered).unwrap();
```

`decode_soft` takes 8-bit soft symbols instead, which corrects more errors when
the demodulator can report its confidence.

### Reed–Solomon

```rust
use fec::{RsEncoder, RsDecoder};

// Standard CCSDS (255,223) code.
let mut enc = RsEncoder::new_ccsds();
let mut dec = RsDecoder::new_ccsds();

let msg: Vec<u8> = (0..223).collect();
let mut block = vec![0u8; 255];
enc.encode(&msg, &mut block).unwrap();

// ... block is corrupted in transit ...

let mut recovered = vec![0u8; 223];
let corrected = dec.decode(&block, &mut recovered).unwrap();
println!("corrected {corrected} symbol error(s)");
```

For real spacecraft telemetry (dual-basis symbols on the wire), use
`encode_ccsds_dual` / `decode_ccsds_dual`.

## Compatibility

The codes are **bit-compatible with [libfec](https://github.com/ka9q/libfec)**
(Phil Karn, KA9Q), so `fec` can decode data Karn's library produced and
vice versa. A companion shim crate exposes `fec` under libfec's C ABI
(`init_rs_char`, `create_viterbi27`, `encode_rs_ccsds`, etc) as a drop-in for
existing C codebases. With the `simd` feature enabled (requires nightly), this
crate is **more performant** than either libcorrect or libfec on x86.

## Roadmap

- More widths for the Reed-Solomon encoder/decoder (narrower than
  GF(2⁸) and as wide as GF(2¹⁶))
- Hard-decision erasures in the convolutional (Viterbi) decoder
- Punctured codes for the convolutional encoder and decoder

## License

BSD-3-Clause.
