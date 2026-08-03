# Benchmarks

These benchmarks compare `fec` against [libfec](https://github.com/ka9q/libfec)
(Phil Karn, KA9Q) using libfec's own test programs. `fec` is exercised through
the C-ABI shim, so only the codec library differs between the two builds. The
comparison runs through libfec's own harness, which makes it apples-to-apples.
All numbers are from a Zen4 laptop (Ryzen 7840HS).

## Running

The harness lives in `shim/bench`. It builds libfec from source (via libfec's
own `configure`, at a known `-O2`) and links libfec's `vtest` programs against
both libfec and the `fec` shim.

```
cd shim/bench
make run      # build and run both the 64-bit and 32-bit tables
make run64    # 64-bit only
make run32    # 32-bit only (needs gcc multilib)
make clean
```

The only external input is a libfec checkout at `../../../libfec`, a sibling of
the `fec` repository. A fresh clone works. The 32-bit table also needs
`gcc-multilib` and a 32-bit assembler (`as --32`).

`make run` uses 2000 trials. For the smaller codes that is enough frames to be
noise-dominated by the 0.01 second timer, so the tables below use higher trial
counts. Pass `TRIALS=` to change it.

## Convolutional codes

The convolutional tables use four variants:

- **fec-64** is this crate on a 64-bit target, with the `simd` feature. It uses
  AVX-512 where the CPU has it, then AVX2, then SSE.
- **libfec-64** is libfec on a 64-bit target. libfec has no 64-bit SIMD path, so
  this is its portable C decoder.
- **fec-32** is this crate on a 32-bit (i686) target. It has no AVX-512, so it
  runs the AVX2 or SSE paths.
- **libfec-32** is libfec on a 32-bit target, which is its best decoder. This is
  Phil Karn's SSE2 code: hand-written assembly for k=7 and k=9, and SSE2
  intrinsics for the other codes.

The sections below are grouped by code. Each one gives a throughput table and a
bit error rate table, over frames of 2048 bits.

**Throughput** is decoded payload bits per second, higher is better. Trial counts
are 200,000 frames for the k=7 and k=9 codes and 2,000 frames for k=15.

**Bit error rate** is a property of the decoder algorithm and the channel, not of
the instruction set, lower is better. `fec` decodes identically on 64-bit and
32-bit targets, so one `fec` row covers both. For k=7 and k=9, libfec is split
into its two decoders, because they do not agree: libfec's portable C gives a
slightly lower error rate than its SSE2 implementation. For the other codes the
two libfec decoders agree, so a single `libfec` row covers both. Each point
decodes with soft decisions at the listed Eb/N0 over 200,000 frames, except k=15
at 30,000.

### Rate 1/2, k=7

Throughput:

| decoder    | throughput |
|------------|-----------:|
| fec-64     | 158.1 Mbps |
| libfec-64  |  16.6 Mbps |
| fec-32     | 126.0 Mbps |
| libfec-32  | 148.4 Mbps |

Bit error rate:

| decoder           | 2.5 dB  | 2.0 dB  | 1.5 dB  |
|-------------------|--------:|--------:|--------:|
| fec               | 1.37e-3 | 4.83e-3 | 1.49e-2 |
| libfec (portable) | 1.39e-3 | 4.90e-3 | 1.51e-2 |
| libfec (SSE2)     | 1.57e-3 | 5.50e-3 | 1.67e-2 |

### Rate 1/2, k=9

Throughput:

| decoder    | throughput |
|------------|-----------:|
| fec-64     |  65.9 Mbps |
| libfec-64  |   2.6 Mbps |
| fec-32     |  44.9 Mbps |
| libfec-32  |  64.5 Mbps |

Bit error rate:

| decoder           | 2.5 dB  | 2.0 dB  | 1.5 dB  |
|-------------------|--------:|--------:|--------:|
| fec               | 3.86e-4 | 2.03e-3 | 8.80e-3 |
| libfec (portable) | 4.14e-4 | 2.10e-3 | 8.96e-3 |
| libfec (SSE2)     | 4.68e-4 | 2.38e-3 | 1.02e-2 |

### Rate 1/3, k=9

Throughput:

| decoder    | throughput |
|------------|-----------:|
| fec-64     |  60.7 Mbps |
| libfec-64  |   2.4 Mbps |
| fec-32     |  43.2 Mbps |
| libfec-32  |  22.5 Mbps |

Bit error rate:

| decoder                  | 2.0 dB  | 1.5 dB  | 1.0 dB  |
|--------------------------|--------:|--------:|--------:|
| fec                      | 6.67e-4 | 2.80e-3 | 1.02e-2 |
| libfec (portable + SSE2) | 6.89e-4 | 2.88e-3 | 1.04e-2 |

### Rate 1/6, k=15

Throughput:

| decoder    | throughput  |
|------------|------------:|
| fec-64     | 1187.2 Kbps |
| libfec-64  |   40.3 Kbps |
| fec-32     |  587.7 Kbps |
| libfec-32  |  415.0 Kbps |

Bit error rate:

| decoder                  | 1.5 dB  | 1.0 dB  | 0.5 dB  |
|--------------------------|--------:|--------:|--------:|
| fec                      | 4.7e-5  | 3.7e-4  | 2.5e-3  |
| libfec (portable + SSE2) | 4.3e-5  | 3.5e-4  | 2.5e-3  |

## Reed-Solomon

The Reed-Solomon comparison uses libfec's `rs_speedtest`, decoding a (255,223)
block over GF(2^8). It runs two decoders: the general decoder built from
`init_rs_char`, and the specialized CCSDS decoder. Reed-Solomon uses no SIMD in
either library, so the 64-bit and 32-bit rows differ only by pointer width.
Throughput is decoded payload bits per second, higher is better.

Two workloads are shown. A clean block has no errors, so the decoder computes
syndromes, finds them zero, and returns. A block with two symbol errors runs the
full correction path: syndromes, Berlekamp-Massey, Chien search, and Forney. The
clean case is libfec's shipped default. The error case flips on the error
injection its test carries.

Clean block:

| decoder    | general  | CCSDS    |
|------------|---------:|---------:|
| fec-64     | 568 Mbps | 568 Mbps |
| libfec-64  | 157 Mbps | 223 Mbps |
| fec-32     | 443 Mbps | 439 Mbps |
| libfec-32  | 123 Mbps | 198 Mbps |

Two symbol errors:

| decoder    | general  | CCSDS    |
|------------|---------:|---------:|
| fec-64     | 445 Mbps | 443 Mbps |
| libfec-64  | 150 Mbps | 207 Mbps |
| fec-32     | 258 Mbps | 257 Mbps |
| libfec-32  | 108 Mbps | 165 Mbps |

`fec` leads on every row. libfec's CCSDS decoder is faster than its general one,
so `fec` leads by more on the general decoder (about 3x) than on the CCSDS
decoder (about 2x), and `fec` runs both at the same speed.
