use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion, Throughput};
use fec::convolutional::sim::{bpsk_params, TestCase, Testbench};
use fec::convolutional::{DecoderArch, ForcedPath, SimdDecoder};
use std::hint::black_box;

const MSG_LEN: usize = 8192;
const CASES: usize = 64;

fn eb_n0(rate: u32, order: u32) -> f64 {
    match (rate, order) {
        (6, 15) => 2.0,
        (_, order) => {
            if order >= 9 {
                2.5
            } else {
                3.0
            }
        }
    }
}

fn host_arch_caps() -> Vec<(&'static str, DecoderArch)> {
    let mut v = vec![("128", DecoderArch::Sse41)];
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            v.push(("avx2", DecoderArch::Avx2));
        }
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl")
        {
            v.push(("avx512", DecoderArch::Avx512));
        }
    }
    v
}

#[inline(always)]
fn decode_cycle<const RATE: u32, const ORDER: u32>(
    decoder: &mut SimdDecoder<RATE, ORDER>,
    cases: &[TestCase],
    idx: &mut usize,
    out: &mut [u8],
) {
    let case = &cases[*idx % cases.len()];
    *idx = idx.wrapping_add(1);
    decoder.decode_soft(black_box(&case.soft), black_box(out)).unwrap();
}

fn bench_code<const RATE: u32, const ORDER: u32>(c: &mut Criterion, tag: &str, polys: &[u16]) {
    let mut bench = Testbench::new(RATE, ORDER, polys, MSG_LEN);
    let (bpsk_voltage, bpsk_bit_energy) = bpsk_params(RATE);
    let cases = bench.create_test_cases(CASES, eb_n0(RATE, ORDER), bpsk_voltage, bpsk_bit_energy);
    let mut out = vec![0u8; MSG_LEN];

    let mut group = c.benchmark_group(tag);
    group.throughput(Throughput::Bytes(MSG_LEN as u64));
    for (label, arch) in host_arch_caps() {
        group.bench_function(label, |b| {
            let mut decoder = SimdDecoder::<RATE, ORDER>::new(polys).with_max_arch(arch);
            let mut idx = 0usize;
            b.iter(|| decode_cycle(&mut decoder, &cases, &mut idx, &mut out));
        });
    }
    group.finish();
}

#[allow(dead_code)]
fn bench_path<const RATE: u32, const ORDER: u32>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: &str,
    path: ForcedPath,
    polys: &[u16],
) {
    let mut bench = Testbench::new(RATE, ORDER, polys, MSG_LEN);
    let (bpsk_voltage, bpsk_bit_energy) = bpsk_params(RATE);
    let cases = bench.create_test_cases(CASES, eb_n0(RATE, ORDER), bpsk_voltage, bpsk_bit_energy);
    let mut out = vec![0u8; MSG_LEN];
    group.bench_function(label, |b| {
        let mut decoder = SimdDecoder::<RATE, ORDER>::new(polys).with_path(Some(path));
        let mut idx = 0usize;
        b.iter(|| decode_cycle(&mut decoder, &cases, &mut idx, &mut out));
    });
}

fn benches(c: &mut Criterion) {
    bench_code::<2, 5>(c, "k5_r12", &[0o27, 0o23]);
    bench_code::<2, 6>(c, "k6_r12", &[0o65, 0o57]);
    bench_code::<2, 7>(c, "k7_r12", &[0o155, 0o117]);
    bench_code::<2, 8>(c, "k8_r12", &[0o367, 0o225]);
    bench_code::<2, 9>(c, "k9_r12", &[0o657, 0o435]);
    bench_code::<3, 9>(c, "k9_r13", &[0o755, 0o633, 0o447]);
    bench_code::<6, 15>(c, "k15_r16", &[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537]);
}

criterion_group!(decode, benches);
criterion_main!(decode);
