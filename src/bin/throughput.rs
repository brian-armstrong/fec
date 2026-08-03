//! Internal throughput probe for the convolutional decoders.
//!
//! This is a development tool, not part of the library API. It decodes many
//! random blocks through a chosen decode path and reports the median and range
//! of the measured throughput. It panics on any decode mismatch, so it doubles
//! as a fast correctness check while tuning.
//!
//! Flags. `--path <name>` pins one decode path, such as `register-avx512` or
//! `scalar`. `--all` sweeps every path instead. `--hard` uses hard-decision
//! decoding. `--eb-n0 <dB>` sets the channel. `--msg-len <bytes>` sets the
//! block size.

use fec::convolutional::sim::{bit_distance, bpsk_params, TestCase, Testbench};
use fec::convolutional::{Decoder, ForcedPath, SimdDecoder};
use std::env;
use std::time::Instant;

#[derive(Clone, Copy)]
enum Probe {
    Simd(ForcedPath),
    Scalar,
}

fn parse_path(name: &str) -> Probe {
    match name {
        "register-avx512" => Probe::Simd(ForcedPath::RegisterAvx512),
        "register-avx2" => Probe::Simd(ForcedPath::RegisterAvx2),
        "register-sse41" => Probe::Simd(ForcedPath::RegisterSse41),
        "register-128" => Probe::Simd(ForcedPath::Register128),
        "shuffle-avx512" => Probe::Simd(ForcedPath::ShuffleAvx512),
        "shuffle-avx2" => Probe::Simd(ForcedPath::ShuffleAvx2),
        "shuffle-sse41" => Probe::Simd(ForcedPath::ShuffleSse41),
        "shuffle-128" => Probe::Simd(ForcedPath::Shuffle128),
        "permute-avx512" => Probe::Simd(ForcedPath::PermuteAvx512),
        "oct-lookup-avx512" => Probe::Simd(ForcedPath::OctLookupAvx512),
        "oct-lookup-avx2" => Probe::Simd(ForcedPath::OctLookupAvx2),
        "oct-lookup-sse41" => Probe::Simd(ForcedPath::OctLookupSse41),
        "oct-lookup-128" => Probe::Simd(ForcedPath::OctLookup128),
        "scalar" => Probe::Scalar,
        other => {
            eprintln!(
                "unknown --path '{other}'; expected one of: register-avx512 register-avx2 \
                 register-sse41 register-128 shuffle-avx512 shuffle-avx2 shuffle-sse41 \
                 shuffle-128 permute-avx512 oct-lookup-avx512 oct-lookup-avx2 oct-lookup-sse41 \
                 oct-lookup-128 scalar"
            );
            std::process::exit(1);
        }
    }
}

fn default_polys(rate: u32, order: u32) -> Option<Vec<u16>> {
    let p: &[u16] = match (rate, order) {
        (2, 4) => &[0o17, 0o13],
        (2, 5) => &[0o27, 0o23],
        (2, 6) => &[0o65, 0o57],
        (2, 7) => &[0o155, 0o117],
        (2, 8) => &[0o367, 0o225],
        (2, 9) => &[0o657, 0o435],
        (2, 10) => &[0o1627, 0o1063],
        (2, 11) => &[0o2365, 0o3073],
        (2, 12) => &[0o4325, 0o6747],
        (2, 13) => &[0o10627, 0o16765],
        (2, 14) => &[0o21663, 0o32275],
        (2, 15) => &[0o44735, 0o63057],
        (3, 9) => &[0o755, 0o633, 0o447],
        (3, 10) => &[0o1157, 0o1227, 0o1755],
        (3, 11) => &[0o2335, 0o2531, 0o3477],
        (3, 12) => &[0o4713, 0o5175, 0o6557],
        (4, 7) => &[0o133, 0o175, 0o107, 0o101],
        (6, 15) => &[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537],
        _ => return None,
    };
    Some(p.to_vec())
}

fn time_decode(msg_len: usize, iters: usize, mut decode: impl FnMut(usize) -> usize) -> (f64, f64, f64, usize) {
    let mut errors = 0usize;
    // do a warmup pass
    for j in 0..iters {
        errors += decode(j);
    }
    let batches = 9;
    let bits = (msg_len * 8 * iters) as f64;
    let mut mbps: Vec<f64> = Vec::with_capacity(batches);
    for _ in 0..batches {
        let t = Instant::now();
        for j in 0..iters {
            std::hint::black_box(decode(j));
        }
        let elapsed = t.elapsed().as_secs_f64();
        mbps.push(bits / elapsed / 1e6);
    }
    mbps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // median, min, max
    (mbps[batches / 2], mbps[0], mbps[batches - 1], errors)
}

fn time_one<const RATE: u32, const ORDER: u32>(
    polys: &[u16],
    cases: &[TestCase],
    msg_len: usize,
    eb_n0_db: f64,
    hard: bool,
    path: Option<Probe>,
) {
    let iters = cases.len();
    let mut out = vec![0u8; msg_len];

    let (median, lo, hi, errors) = if let Some(Probe::Scalar) = path {
        let mut scalar = Decoder::new(RATE, ORDER, polys);
        time_decode(msg_len, iters, |j| {
            if hard {
                scalar.decode_hard(&cases[j].hard, cases[j].enc_bits, &mut out).unwrap();
            } else {
                scalar.decode_soft(&cases[j].soft, &mut out).unwrap();
            }
            bit_distance(&cases[j].msg, &out)
        })
    } else {
        let mut sim = SimdDecoder::<RATE, ORDER>::new(polys);
        if let Some(Probe::Simd(p)) = path {
            sim = sim.with_path(Some(p));
        }
        time_decode(msg_len, iters, |j| {
            if hard {
                sim.decode_hard(&cases[j].hard, cases[j].enc_bits, &mut out).unwrap();
            } else {
                sim.decode_soft(&cases[j].soft, &mut out).unwrap();
            }
            bit_distance(&cases[j].msg, &out)
        })
    };

    let tag = match path {
        Some(Probe::Simd(p)) => format!("{p:?}"),
        Some(Probe::Scalar) => "Scalar".into(),
        None => "default".into(),
    };
    let mode = if hard { "hard" } else { "soft" };
    let ber = errors as f64 / (msg_len * 8 * iters) as f64;
    println!(
        "r=1/{RATE} k={ORDER} len={msg_len} [{tag:>16}]: {median:7.2} Mbps  \
         (range {lo:6.2}–{hi:6.2})  BER {ber:.2e} @{eb_n0_db}dB {mode}"
    );
}

fn run<const RATE: u32, const ORDER: u32>(
    polys: &[u16],
    path: Option<Probe>,
    all: bool,
    msg_len: usize,
    eb_n0_db: f64,
    hard: bool,
) {
    let iters = 400;
    let mut bench = Testbench::new(RATE, ORDER, polys, msg_len);
    let (bpsk_voltage, bpsk_bit_energy) = bpsk_params(RATE);
    let cases = bench.create_test_cases(iters, eb_n0_db, bpsk_voltage, bpsk_bit_energy);
    if all {
        time_one::<RATE, ORDER>(polys, &cases, msg_len, eb_n0_db, hard, None);
        for p in valid_paths(RATE, ORDER) {
            time_one::<RATE, ORDER>(polys, &cases, msg_len, eb_n0_db, hard, Some(p));
        }
    } else {
        time_one::<RATE, ORDER>(polys, &cases, msg_len, eb_n0_db, hard, path);
    }
}

fn valid_paths(rate: u32, order: u32) -> Vec<Probe> {
    use ForcedPath::*;
    let has_sse4_1;
    let has_avx2;
    let has_avx512;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        has_sse4_1 = std::is_x86_feature_detected!("sse4.1");
        has_avx2 = std::is_x86_feature_detected!("avx2");
        has_avx512 = cfg!(target_arch = "x86_64")
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl");
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        has_sse4_1 = false;
        has_avx2 = false;
        has_avx512 = false;
    }

    let mut simd: Vec<ForcedPath> = Vec::new();
    if rate <= 3 {
        if has_avx512 && (7..=10).contains(&order) {
            simd.push(RegisterAvx512);
        }
        if has_avx2 && (6..=9).contains(&order) {
            simd.push(RegisterAvx2);
        }
        if has_sse4_1 && (5..=8).contains(&order) {
            simd.push(RegisterSse41);
        }
        if (5..=8).contains(&order) {
            simd.push(Register128);
        }
        if has_avx512 && order >= 6 {
            simd.push(ShuffleAvx512);
        }
        if has_avx2 && order >= 5 {
            simd.push(ShuffleAvx2);
        }
        if has_sse4_1 && order >= 4 {
            simd.push(ShuffleSse41);
        }
        if order >= 4 {
            simd.push(Shuffle128);
        }
    }
    if has_avx512 && order >= 6 && rate <= 6 {
        simd.push(PermuteAvx512);
    }
    if has_avx512 && order >= 6 {
        simd.push(OctLookupAvx512);
    }
    if has_avx2 && order >= 5 {
        simd.push(OctLookupAvx2);
    }
    if has_sse4_1 && order >= 4 {
        simd.push(OctLookupSse41);
    }
    if order >= 4 {
        simd.push(OctLookup128);
    }

    let mut v: Vec<Probe> = simd.into_iter().map(Probe::Simd).collect();
    v.push(Probe::Scalar);
    v
}

fn default_eb_n0(rate: u32, order: u32) -> f64 {
    match (rate, order) {
        (2, 4) | (2, 5) | (2, 6) => 3.5,
        (2, 7) | (2, 8) => 3.0,
        (2, 9) | (2, 10) | (3, 9) => 2.5,
        (2, 11) | (2, 13) | (2, 15) => 2.0,
        (4, 7) => 2.0,
        (6, 15) => 2.0,
        _ => 2.5,
    }
}

fn main() {
    let mut positional: Vec<String> = Vec::new();
    let mut path: Option<Probe> = None;
    let mut all = false;
    let mut msg_len = 1024;
    let mut eb_n0_db: Option<f64> = None;
    let mut hard = false;
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--hard" => hard = true,
            "--eb-n0" => {
                let v = args.next().unwrap_or_else(|| {
                    eprintln!("--eb-n0 needs a value (dB)");
                    std::process::exit(1);
                });
                eb_n0_db = Some(v.parse().expect("--eb-n0 dB"));
            }
            "--path" => {
                let name = args.next().unwrap_or_else(|| {
                    eprintln!("--path needs a value");
                    std::process::exit(1);
                });
                path = Some(parse_path(&name));
            }
            "--all" => all = true,
            "--msg-len" => {
                let v = args.next().unwrap_or_else(|| {
                    eprintln!("--msg-len needs a value (bytes)");
                    std::process::exit(1);
                });
                msg_len = v.parse().expect("--msg-len bytes");
            }
            _ => positional.push(a),
        }
    }
    if all && path.is_some() {
        eprintln!("--all and --path are mutually exclusive (specify one or the other)");
        std::process::exit(1);
    }
    if positional.len() < 2 {
        eprintln!("usage: throughput <rate> <order> [poly1_oct poly2_oct ...] [--path <name> | --all]");
        std::process::exit(1);
    }
    let rate: u32 = positional[0].parse().expect("rate");
    let order: u32 = positional[1].parse().expect("order");
    let polys: Vec<u16> = if positional.len() > 2 {
        positional[2..]
            .iter()
            .map(|s| u16::from_str_radix(s, 8).expect("octal poly"))
            .collect()
    } else {
        default_polys(rate, order).unwrap_or_else(|| {
            eprintln!("no default polys for (rate, order) = ({rate}, {order}); supply them explicitly");
            std::process::exit(1);
        })
    };

    let eb = eb_n0_db.unwrap_or_else(|| default_eb_n0(rate, order));

    match (rate, order) {
        (2, 4) => run::<2, 4>(&polys, path, all, msg_len, eb, hard),
        (2, 5) => run::<2, 5>(&polys, path, all, msg_len, eb, hard),
        (2, 6) => run::<2, 6>(&polys, path, all, msg_len, eb, hard),
        (2, 7) => run::<2, 7>(&polys, path, all, msg_len, eb, hard),
        (2, 8) => run::<2, 8>(&polys, path, all, msg_len, eb, hard),
        (2, 9) => run::<2, 9>(&polys, path, all, msg_len, eb, hard),
        (2, 10) => run::<2, 10>(&polys, path, all, msg_len, eb, hard),
        (2, 11) => run::<2, 11>(&polys, path, all, msg_len, eb, hard),
        (2, 13) => run::<2, 13>(&polys, path, all, msg_len, eb, hard),
        (2, 14) => run::<2, 14>(&polys, path, all, msg_len, eb, hard),
        (2, 15) => run::<2, 15>(&polys, path, all, msg_len, eb, hard),
        (3, 9) => run::<3, 9>(&polys, path, all, msg_len, eb, hard),
        (3, 10) => run::<3, 10>(&polys, path, all, msg_len, eb, hard),
        (3, 11) => run::<3, 11>(&polys, path, all, msg_len, eb, hard),
        (3, 12) => run::<3, 12>(&polys, path, all, msg_len, eb, hard),
        (4, 7) => run::<4, 7>(&polys, path, all, msg_len, eb, hard),
        (6, 15) => run::<6, 15>(&polys, path, all, msg_len, eb, hard),
        _ => {
            eprintln!("unsupported (rate, order) = ({rate}, {order}); add an arm");
            std::process::exit(1);
        }
    }
}
