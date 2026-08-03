# Benchmarking

The decoder has an integrated [Criterion](https://github.com/bheisler/criterion.rs)
throughput suite (`benches/decode_throughput.rs`) for catching performance regressions.
Criterion handles warmup, statistical sampling, and outlier rejection, and can save and
compare named baselines across changes.

## Requirements

- **Nightly toolchain** — the SIMD decoder uses `portable_simd`.
- **The `bench-internals` feature** — the benchmarks drive the decoder through the
  test/bench-only `with_max_arch` / `with_path` overrides, which are gated on this feature
  (it enables `simd`). These overrides are deliberately NOT public API. `with_path` pins a
  specific decode PATH (`ForcedPath::{SmallSse, WideAvx2, UltrawideAvx512, ShuffleAvx2,
  GenericAvx2, GenericSse, Scalar}`) through the real `decode_soft` and PANICS if that path
  isn't runnable at the `(rate, order)` / arch — so a pinned bench cell can never be silently
  attributed to the wrong path. Arch ceiling is the orthogonal `with_max_arch` axis.

## Running

```sh
# Run the whole suite (prints time + throughput per code per arch path).
cargo +nightly bench --features bench-internals --bench decode_throughput

# Run a subset by name filter (regex over the benchmark IDs).
cargo +nightly bench --features bench-internals --bench decode_throughput -- k9
cargo +nightly bench --features bench-internals --bench decode_throughput -- 'k7_r12/avx2'
```

> Use `--bench decode_throughput` to run *only* the criterion suite. A bare
> `cargo bench` also tries to run the lib unittests as benchmarks, which don't
> understand criterion's CLI flags (e.g. `--measurement-time`).

## Catching regressions (save / compare a baseline)

```sh
# 1. Record a baseline on the current code.
cargo +nightly bench --features bench-internals --bench decode_throughput -- --save-baseline main

# 2. Make your change, then compare against it.
cargo +nightly bench --features bench-internals --bench decode_throughput -- --baseline main
```

The comparison run prints a `change:` line per benchmark with a percentage and a
verdict (`Performance has improved` / `regressed` / `No change`), with criterion's
significance test so noise isn't flagged as a regression.

## What's covered

Each code is timed on **every arch path the host supports** — SSE, AVX2, AVX-512 — forced
via `SimdDecoder::with_max_arch(allow_avx2, allow_avx512)` (downgrade-only; it can't enable
a feature the CPU lacks). No env vars, no separate processes.

| Group     | Code            | Role |
|-----------|-----------------|------|
| `k7_r12`  | k=7, rate 1/2   | deployment code (regression guard) |
| `k9_r12`  | k=9, rate 1/2   | deployment code |
| `k9_r13`  | k=9, rate 1/3   | deployment code |
| `k15_r16` | k=15, rate 1/6  | deployment code |
| `k5_r12`  | k=5, rate 1/2   | shape-transition (dispatch-envelope) |
| `k6_r12`  | k=6, rate 1/2   | shape-transition |
| `k8_r12`  | k=8, rate 1/2   | shape-transition |

The benchmark IDs are `<group>/<arch>`, e.g. `k9_r12/avx512`.

### Reference numbers (AMD Ryzen 7 7840HS, message Mbit/s)

| code     |   sse |  avx2 | avx512 |
|----------|------:|------:|-------:|
| k7_r12   |  35.1 |  89.2 |   89.1 |
| k9_r12   |  19.8 |  39.0 |   42.4 |
| k9_r13   |  15.4 |  39.9 |   39.7 |
| k5_r12   |  44.2 |  91.6 |   98.6 |
| k6_r12   |  35.5 |  87.4 |   87.7 |
| k8_r12   |  26.6 |  61.9 |   61.9 |
| k15_r16  |   0.5 |   0.6 |    0.6 |

These are a snapshot, not a contract — use them to sanity-check, not as pass/fail
thresholds (that's what `--baseline` is for).

## Units (important)

Throughput is reported **per decoded message byte** (`Throughput::Bytes(msg_len)`) — the
SAME denominator libcorrect uses (message bytes, not coded). But the absolute number is
**not** comparable to libcorrect's "Mbps": this suite uses `msg_len = 8192` and times the
full standalone `decode_soft` (reset + warmup + inner + tail + flush), while libcorrect uses
`msg_len = 256` and times the streaming `init + update + chainback`. Both effects are real
(longer messages are ~25% slower per byte here — see the project memory on that), so the
same decoder reads ~89 here vs ~150 there. Neither is "wrong"; they're different operating
points. Rule of thumb:

- **This suite** → regression detection across `fec` changes (the source of truth here).
- **The libcorrect benchmark** → cross-library comparison (fec vs libcorrect vs libfec);
  see the harness notes in the project memory.

## Quick A/B probe (the everyday tool)

`src/bin/throughput.rs` is the fast same-process probe for iterating — sub-second per cell,
one line of output, a stable MEDIAN over a few batches (no criterion warmup/stats/chatter).
**This is what you reach for while changing the decoder**; the criterion suite above is the
deliberate regression *guard* you run before/after a change you want to certify.

It can pin a specific decode path via `--path`, or sweep every runnable path with `--all`,
built on the same `with_path` override the benches use (so it needs `bench-internals`). A
pinned path that isn't runnable at this `(rate, order)` on the host PANICS — the probe can't
silently measure a different path than you asked for. (`--all` evaluates the same gates up
front and simply SKIPS the invalid paths instead of panicking.)

```sh
cargo +nightly build --features bench-internals --release --bin throughput

# Production dispatch (let the decoder pick the path):
./target/release/throughput 2 7 0155 0117               # rate order poly1 poly2 ... (octal)

# Sweep every path runnable at this code (plus the dispatch baseline) — the one-shot
# "where are we?" view:
./target/release/throughput 2 7 0155 0117 --all

# Pin a path to A/B two strategies at the same code:
./target/release/throughput 2 7 0155 0117 --path register-avx2
./target/release/throughput 2 7 0155 0117 --path shuffle-avx2

# --path names: register-sse register-avx2 register-avx512
#               shuffle-avx2 vectorized-avx2 vectorized-sse scalar
# (--all and --path are mutually exclusive.)

# --msg-len sets the message length in bytes (default 8192). Drop to 256 to match
# libcorrect's operating point — per-byte throughput is ~25% higher there, and some
# paths take the length tax unequally, so always compare at a FIXED length:
./target/release/throughput 2 9 0657 0435 --all --msg-len 256
```

`--all` prints the production dispatch first as a baseline, then each runnable path widest-
register-first; e.g. at k=7 r1/2 it confirms dispatch (≈86) lands on the best path
(RegisterAvx2 ≈88) while shuffle (≈75) and the in-memory loops (≈31) sit below — a live
self-check that the routing picks the top path, not just a measurement.

When NO `--path` is given, the production dispatch still honors the construction-time env
toggles for quick A/B of the default routing:

| Env var (set before constructing the decoder) | Forces off |
|-----------------------------------------------|------------|
| `FEC_NO_AVX2`     | AVX2 + AVX-512 dispatch (cap at SSE) |
| `FEC_NO_AVX512`   | AVX-512 dispatch (cap at AVX2) |
| `FEC_NO_SMALL`    | the register-resident `decode_small` family (force the in-memory path) |
| `FEC_NO_K7_AVX2`  | the k=7 AVX2 register-resident path (use SSE register instead) |
| `FEC_NO_HEX_SHUF` | the broadcast-shuffle distance path |

For a certified regression verdict prefer the criterion suite — the probe reports a median
+ range but no confidence interval. The range column is your noise check: if it's wide,
the median's last digit isn't real.
