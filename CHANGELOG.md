# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the crate is pre-1.0, minor version bumps may contain breaking changes.

## [Unreleased]

## [0.2.0] - 2026-07-31

### Added

- Reed-Solomon codec over GF(2^8) with error and erasure decoding, including the
  standard CCSDS (255,223) code in both the conventional and dual-basis
  (Berlekamp) representations.
- Soft-decision decoding for the convolutional decoder (`Decoder::decode_soft`),
  alongside the existing hard-decision path.
- SIMD convolutional decoder behind the `simd` feature (nightly Rust). It
  accelerates on SSE, AVX2, and AVX-512, selecting the widest available at run
  time. Exposed as `ConvSimdDecoder`.
- Typed errors: `DecodeError` and `EncodeError` for the convolutional codec and
  for Reed-Solomon. All implement `Display` and `std::error::Error`.
- Documentation across the public API, plus `README` examples for both codecs.
- `fec-shim`, a crate exposing `fec` under a libfec-compatible C ABI as
  a partial drop-in for existing C codebases.

### Fixed

- The convolutional decoder now validates that a block is long enough to decode,
  returning an error instead of risking an underflow on undersized input.

## 0.1.0

Initial release. A convolutional encoder and decoder
(`fec::convolutional::Encoder` and `Decoder`), hard-decision only.

[Unreleased]: https://github.com/brian-armstrong/fec/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/brian-armstrong/fec/tree/v0.2.0
