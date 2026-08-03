# fec-shim

A C-ABI shim over the [`fec`](https://crates.io/crates/fec) crate. It builds a
`libfec.so` / `libfec.a` that a C program can link against as a **partial
drop-in** for Phil Karn's [libfec](https://github.com/ka9q/libfec).

This is not a full libfec replacement. It exposes the common codes and calls,
not every symbol libfec provides. If your code only uses the functions below, it
can link against this library instead of the C one and get fec's Rust
implementation underneath.

## Supported symbols

**Convolutional (Viterbi):** the standard libfec lifecycle
(`create` / `init` / `update_..._blk` / `chainback` / `delete`) for:

- `viterbi27` — rate 1/2, k=7
- `viterbi29` — rate 1/2, k=9
- `viterbi39` — rate 1/3, k=9
- `viterbi615` — rate 1/6, k=15

**Reed-Solomon:**

- `init_rs_char` / `encode_rs_char` / `decode_rs_char` / `free_rs_char`
- `encode_rs_8` / `decode_rs_8`
- `encode_rs_ccsds` / `decode_rs_ccsds`

## Building

Build the C library from this directory:

```sh
cargo build --release                            # scalar decoders, stable Rust
cargo +nightly build --release --features simd   # SIMD decoders, needs nightly
```

The artifacts land in `target/release/` as `libfec.so` (shared) and
`libfec.a` (static). Point your linker at them in place of the C libfec.

To declare the functions from C, include the bundled [`fec_shim.h`](fec_shim.h),
or use libfec's own `fec.h`. The bundled header declares exactly the symbols this
library exports, and no more.

## Performance

The reason to relink against this shim is speed. Measured through libfec's own
test programs, with only the codec library swapped, `fec` decodes faster than
libfec on every code. The table shows decoded throughput for `fec` on a 64-bit
target against libfec's two decoders: its 32-bit SSE2 assembly, which is its
fastest, and its 64-bit portable C.[^simd]

| code            | fec (64-bit) | libfec (32-bit, SSE2) | libfec (64-bit, C) |
|-----------------|-------------:|----------------------:|-------------------:|
| rate 1/2, k=7   |   158 Mbps   |        148 Mbps       |       17 Mbps      |
| rate 1/2, k=9   |    66 Mbps   |         65 Mbps       |        3 Mbps      |
| rate 1/3, k=9   |    61 Mbps   |         23 Mbps       |        2 Mbps      |
| rate 1/6, k=15  |  1187 Kbps   |        415 Kbps       |       40 Kbps      |

`fec` also beats Karn's SSE2 assembly with a modest improvement in BER for the
rate-1/2 codes. Reed-Solomon decoding is about 3x faster than libfec.

See [BENCH.md](https://github.com/brian-armstrong/fec/blob/main/shim/BENCH.md)
for the full throughput and bit error rate tables, the
32-bit comparison, and how to reproduce the numbers.

[^simd]: libfec's SIMD kernels are gated to 32-bit x86 (`#ifdef __i386__`), so a
    64-bit build has no SIMD path and runs portable C. `fec`'s SIMD works on both.

## Relationship to fec

Rust users should use the [`fec`](https://crates.io/crates/fec) crate
directly. This shim exists only to serve existing C codebases that already
speak the libfec API.
