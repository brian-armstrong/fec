# fec

[![Crates.io](https://img.shields.io/crates/v/fec.svg)](https://crates.io/crates/fec)
[![Docs.rs](https://docs.rs/fec/badge.svg)](https://docs.rs/fec)
[![CI](https://github.com/brian-armstrong/fec/actions/workflows/ci.yml/badge.svg)](https://github.com/brian-armstrong/fec/actions/workflows/ci.yml)

Forward error correction for SDR, space, and satellite applications.

`fec` implements two error-correcting codes that show up throughout
software-defined radio and spacecraft links:

- **Convolutional codes** with a Viterbi decoder (hard and soft decision),
  including the common rate-1/2 k=7, rate-1/2 k=9, rate-1/3 k=9, and
  rate-1/6 k=15 codes. Supports any rate from 1/2 to 1/8 and any order from
  k=4 to k=16. Erasures and punctured codes are supported on both the hard
  and soft decoders. On nightly Rust, the `simd` feature enables a Viterbi
  decoder with acceleration on SSE/AVX2/AVX512.
- **Reed–Solomon codes** over GF(2⁸) with error and erasure decoding, including
  the standard **CCSDS (255,223)** code in both the conventional and the
  on-the-wire **dual-basis** (Berlekamp) representations.

`fec` started as and draws heavy inspiration from the author's own
[libcorrect](https://github.com/quiet/libcorrect), a C library
for forward error correction. This crate also credits Phil Karn's libfec
C library for offering an original implementation of these codes, although
this crate does not borrow any source or have any relationship with that
library, and the name is purely coincidental.

Standard parameters (primitive polynomials, the CCSDS dual-basis transform) are
derived from the published CCSDS standard ([CCSDS 131.0-B](https://public.ccsds.org/),
Annex D for the dual basis).

## Performance

With the `simd` feature, `fec` decodes faster than libfec on every code.
Measured through libfec's own test programs with only the codec library
swapped, on a Zen4 laptop (Ryzen 7840HS). Higher is better.

| code                         | fec (64-bit) | libfec (32-bit) | libfec (64-bit) |
|------------------------------|-------------:|----------------:|----------------:|
| conv, rate 1/2, k=7          |   158 Mbps   |    148 Mbps[^a] |      17 Mbps    |
| conv, rate 1/2, k=9          |    66 Mbps   |     65 Mbps[^a] |       3 Mbps    |
| conv, rate 1/3, k=9          |    61 Mbps   |     23 Mbps[^a] |       2 Mbps    |
| conv, rate 1/6, k=15         |  1187 Kbps   |    415 Kbps[^a] |      40 Kbps    |
| RS (255,223), general        |   568 Mbps   |    123 Mbps     |     157 Mbps    |
| RS (255,223), CCSDS          |   568 Mbps   |    198 Mbps     |     223 Mbps    |
| RS (255,223), general, 2 err |   445 Mbps   |    108 Mbps     |     150 Mbps    |
| RS (255,223), CCSDS, 2 err   |   443 Mbps   |    165 Mbps     |     207 Mbps    |

Convolutional throughput is decoded payload bits per second. The Reed-Solomon
rows decode a (255,223) block, first with no errors (syndromes only) and then
with two symbol errors. Reed-Solomon uses no SIMD in either library, so its
32-bit and 64-bit rows differ only by pointer width.

[^a]: libfec's SIMD kernels are gated behind `#ifdef __i386__`, so its fastest
    convolutional build is the 32-bit one, running Karn's SSE2 assembly. A
    64-bit libfec build has no SIMD path at all. `fec`'s SIMD works on both.

See [shim/BENCH.md](https://github.com/brian-armstrong/fec/blob/main/shim/BENCH.md)
for the full tables, the bit error rate comparison, the 32-bit numbers, and how
to reproduce them.


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
vice versa. A companion shim crate, [`fec-shim`](https://crates.io/crates/fec-shim),
exposes `fec` under libfec's C ABI (`init_rs_char`, `create_viterbi27`,
`encode_rs_ccsds`, etc) as a drop-in for existing C codebases.

## Roadmap

- More widths for the Reed-Solomon encoder/decoder (narrower than
  GF(2⁸) and as wide as GF(2¹⁶))

## License

BSD-3-Clause.
