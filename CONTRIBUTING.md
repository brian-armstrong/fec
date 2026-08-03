# Contributing to fec

Thank you for your interest in contributing to fec. This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository and clone your fork
2. Install Rust. The Reed-Solomon codec and the scalar convolutional codec build on stable. The `simd` feature needs a nightly toolchain because it uses `portable_simd`.
3. Run `cargo build` to verify your setup
4. Run `cargo test --release` to ensure tests pass. The noise and BER tests need optimized code to run in reasonable time.
5. For the SIMD decoder, run `cargo +nightly test --release --features simd`

## How to Contribute

### Reporting Bugs

- Check existing issues to avoid duplicates
- Include a minimal reproducible example when possible
- Describe expected vs actual behavior
- Include your Rust version, platform, and CPU features

### Suggesting Features

- Open an issue describing the use case
- For coding-theory features, reference the relevant standard or literature when applicable

### Submitting Pull Requests

1. Create a feature branch from `main`
2. Make your changes with clear, focused commits
3. Add tests for new functionality
4. Ensure `cargo test --release` passes or `cargo +nightly test --release --features simd` for changes affecting the SIMD path
5. Ensure `cargo fmt` passes
6. Open a PR with a clear description of the changes

### Expectations

fec is maintained on a best-effort basis. Please keep in mind:

- Response times may vary. This is a side project, not a full-time job
- Feature requests are welcome but may not be implemented
- PRs are appreciated, but acceptance is not guaranteed
- The maintainer has final discretion on what gets merged

This isn't meant to discourage contributions, quite the opposite. Clear expectations help everyone have a better experience. If you're unsure whether a contribution would be welcome, open an issue to discuss before investing significant time.

## Code Standards

- Follow standard Rust conventions and idioms
- Use `rustfmt` for formatting
- Add documentation comments for public APIs
- Include tests for new functionality

## Testing

fec's correctness rests on a tiered test suite. When adding or changing functionality, match the tier that proves what your change needs:

- **Clean recovery.** The scalar decoders are tested against the original message under a guard-interval error model, so a decode either recovers the exact input or fails loudly.
- **Bit-exactness.** Every SIMD decode path is checked bit-for-bit against the scalar decoder, under both clean and noisy input. A new path should join this differential rather than weaken it.
- **Coding gain and BER.** The scalar decoder is measured against Eb/N0 waterfall points to confirm the code has correctly shaped error-correcting behavior.
- **Malformed input.** Bad lengths and undersized buffers return typed errors or panic as documented.

If you add a decode path or a code parameter, extend the differential matrix so the new cell is exercised. If you change the unsafe SIMD internals, run the suite under AddressSanitizer as well.

## Provenance and Licensing

fec is a Rust translation of [libcorrect](https://github.com/quiet/libcorrect). Standard parameters, such as the CCSDS dual-basis transform, are derived from the published CCSDS standard.

When contributing code derived from another source, ensure proper attribution and license compatibility with the project's BSD-3-Clause license.

## Development Process

This project was developed with assistance from large language models. Contributions from all sources, human or AI-assisted, are welcome. What matters is code quality, correctness, and clear communication. Communication on pull requests and issues should be made without AI assistance, please.

If you use AI tools in your contributions, please ensure you understand and can explain any code you submit.

## Questions?

Open an issue for questions about contributing. We're happy to help newcomers get started.
