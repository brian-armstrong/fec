#![allow(clippy::needless_range_loop)]

//! Exhaustive search for good convolutional-code generator polynomials at a given
//! (rate, order). Candidates are ranked by exact free distance (d_free, descending), with the
//! first distance-spectrum term (a_dfree, ascending) as the tiebreaker
//!
//! Run with `cargo run --release --bin find_conv_optim_poly -- <rate> <order>`. For
//! example, `-- 2 7` searches rate-1/2, order-7 codes. Optional flags: `--num-polys <n>`
//! to keep the top n, `--min-d-free <d>` to filter, `--threads <n>` to parallelize, and
//! `--ber <bytes>` to add an opt-in BER confirmation pass (needs 1M+ bytes to converge).
//!
//! References:
//! - Z. Abreu, J. Rosenthal, M. Schaller, "Algorithms for Computing the Free Distance of
//!   Convolutional Codes," arXiv:2402.02982, 2024. Gives definition for our `reciprocal`
//!   implementation and inspiration for `shortest_output_weight_to_zero`.
//! - M. Cedervall, R. Johannesson, "A Fast Algorithm for Computing Distance Spectrum of
//!   Convolutional Codes," IEEE Trans. Inf. Theory 35(6), 1989. The original FAST
//!   algorithm. a_dfree is the first term of the distance spectrum defined there.
//! - R. B. Dial, "Algorithm 360: Shortest-Path Forest with Topological Ordering," Comm.
//!   ACM 12(11), 1969. The bucket-queue shortest path used by `shortest_output_weight_to_zero`.
//! - J. L. Massey, M. K. Sain, "Inverses of Linear Sequential Circuits," IEEE Trans.
//!   Computers C-17(4), 1968. The gcd criterion behind `is_catastrophic`.
//! - K. J. Larsen, "Short Convolutional Codes with Maximal Free Distance for Rates 1/2, 1/3,
//!   and 1/4," IEEE Trans. Inform. Theory, IT-19(3), pp. 371–372, 1973. IEEE Xplore 1055014.
//!   (d_free test cases, application of heller bound and griesmer bound and their minimum)
//! - B. Friedrichs, "Error-Control Coding" (author's English edition of Kanalcodierung, Springer 1996),
//!   Ch. 9 Table 9.3b, bernd-friedrichs.de/downloads_ecc/ecc2010_ch09.pdf (k=15 d_free test case.
//!   the table attributes the code to Odenwalder 1970)
//! - James H. Griesmer, A Bound for Error-Correcting Codes, IBM Journal of Research and Development,
//!   vol. 4, no. 5, pp. 532-542, November 1960. (griesmer bound)
//! - J. A. Heller (1968). Short constraint length convolutional coding. PhD thesis,
//!   Massachusetts Institute of Technology (MIT), Department of Electrical Engineering. (heller bound)
use fec::convolutional::sim::measure_ber;
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const NUM_BER_POINTS: usize = 4;
const DEFAULT_NUM_KEPT: usize = 20;
const THREAD_BATCH: usize = 65536;
const START_OFFSET: f32 = 0.4;

fn heller(rate: u32, order: u32) -> u32 {
    // Heller bound for convolutional code
    //                         (     (  rate * 2^k * (order + k - 1) ))
    // d_free <=  min_{k >= 1} (floor(-------------------------------))
    //                         (     (        2 * (2^k - 1)          ))
    //
    // the _minimum_ bound will always be found with k <= order though

    let mut min = u32::MAX;
    for k in 1..=order {
        let numerator = rate * (1 << k) * (order + k - 1);
        let denominator = 2 * ((1 << k) - 1);
        // use integer division/floor
        min = min.min(numerator / denominator);
    }
    min
}

fn griesmer(rate: u32, order: u32) -> u32 {
    // Griesmer bound for Linear code [L, i, d]
    //   L >= sum_{j=0}^{i-1}(ceil(d / 2^j))) (for GF(2) codes)
    //   (this inequality must hold for all i >= 1)
    // for a convolutional code with message length 'i', L = rate * (order + i - 1)

    let mut d = 1;
    // start testing distances from 1 and increment until we find a distance that fails
    loop {
        let mut sum = 0;
        let mut d_i = d;
        let mut i = 0;
        loop {
            // test a message of length 'i' bits
            i += 1;
            sum += d_i; // implicit sum of j=0 -> i-1
            if rate * (order + i - 1) < sum {
                // this distance failed the check, so the last distance that passed
                //     is our bound
                return d - 1;
            }
            if d_i == 1 {
                // once the distance term is 1, every subsequent term will also be 1
                // and from here, the sum will grow more slowly than L
                // so every term after this succeeds - move to the next distance
                break;
            }
            d_i = d.div_ceil(1 << i);
        }
        d += 1;
    }
}

fn degree(p: u32) -> u32 {
    p.ilog2()
}

#[inline]
fn gf2_gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        // a %= b
        while a != 0 && degree(a) >= degree(b) {
            a ^= b << (degree(a) - degree(b));
        }
        (a, b) = (b, a);
    }
    a
}

fn is_catastrophic(polys: &[u16]) -> bool {
    let mut gcd = polys[0] as u32;
    for &poly in &polys[1..] {
        gcd = gf2_gcd(gcd, poly as u32);
    }
    gcd != 1
}

#[inline]
fn output_weight_for_state(state: u32, rate: u32, polys: &[u16]) -> u32 {
    polys
        .iter()
        .take(rate as usize)
        .map(|&poly| (state as u16 & poly).count_ones() & 1)
        .sum()
}

fn shortest_output_weight_to_zero(order: u32, rate: u32, polys: &[u16], cap: u32, target: Option<u32>) -> Vec<u16> {
    // scan through states in order of increasing weight until we find target, if specified
    // we are going in *reverse* trellis order, starting at state 0 and then working backwards to its predecessors
    // because the path weights are always non-negative, any predecessor has to be at least as heavily weighted as its successors
    let top_bit = 1u32 << (order - 1);
    let mut weight_to_zero = vec![u16::MAX; 1usize << order];
    weight_to_zero[0] = 0;
    let mut weight_queue: Vec<Vec<u32>> = vec![vec![]; cap as usize + 1];
    weight_queue[0].push(0);
    let mut weight = 0usize;
    while weight < weight_queue.len() {
        let mut idx = 0;
        while idx < weight_queue[weight].len() {
            let state = weight_queue[weight][idx];
            idx += 1;
            if weight_to_zero[state as usize] != weight as u16 {
                // it's possible that we already found a shorter path to this state
                // the same state can be queued by multiple paths before it resolves
                continue;
            }
            // compute the output weight of the current state
            let pred_weight = weight as u32 + output_weight_for_state(state, rate, polys);
            let predecessors = [state >> 1, (state >> 1) | top_bit];
            for pred in predecessors {
                if pred_weight < weight_to_zero[pred as usize] as u32 {
                    // we have found a shorter path to `pred`. record it.
                    weight_to_zero[pred as usize] = pred_weight as u16;
                    if pred_weight < weight_queue.len() as u32 {
                        // enqueue `pred` so we can scan its predecessors
                        weight_queue[pred_weight as usize].push(pred);
                    }
                }
            }
        }
        weight += 1;
        if target.is_some_and(|t| weight_to_zero[t as usize] as usize <= weight) {
            break;
        }
    }
    weight_to_zero
}

#[inline]
fn gf2_mul(u: u32, g: u32) -> u32 {
    // fast implementation of a carryless multiply
    let mut prod = 0;
    let mut rest = u;
    while rest != 0 {
        prod ^= g << rest.trailing_zeros();
        // Kernighan's algorithm
        rest &= rest - 1;
    }
    prod
}

#[inline]
fn output_weight_for_input(u: u32, polys: &[u16]) -> u32 {
    polys.iter().map(|&g| gf2_mul(u, g as u32).count_ones()).sum::<u32>()
}

fn free_distance_bound(polys: &[u16], order: u32, min_bound: Option<u32>) -> u32 {
    // simulate some inputs and find their output weights
    // we're trying to set a conservative upper bound on the free distance

    // this is just a tuned heuristic. more inputs means a lower bound but more compute
    let max_input = if min_bound.is_some() {
        // if we're trying to meet a goal, try really hard to find it
        1u32 << (order.max(3) - 2)
    } else {
        // no specific goal, just try a few
        64
    };
    let mut min_weight = u32::MAX;
    // only check odd inputs. even inputs are just odd inputs but delayed
    for u in (1..max_input).step_by(2) {
        let weight = output_weight_for_input(u, polys);
        min_weight = min_weight.min(weight);
        if min_bound.is_some_and(|bound| weight < bound) {
            break;
        }
    }
    min_weight
}

fn evaluate_free_distance(rate: u32, order: u32, polys: &[u16]) -> Option<u32> {
    // the free distance == the minimum output weight to get from state 0 to state 1 and back to state 0

    // catastrophic codes generate an unbounded number of errors, so quickly discard them
    if is_catastrophic(polys) {
        return None;
    }

    let state_1_weight = output_weight_for_state(1, rate, polys);
    // free_distance_bound gives us the full weight of a 0->1->0 path (conservative upper bound)
    // shortest_output_weight_to_zero looks for just 1->0, so subtract 0->1 from the bound
    let cap = free_distance_bound(polys, order, None) - state_1_weight;
    let to_zero = shortest_output_weight_to_zero(order, rate, polys, cap, Some(1));
    if to_zero[1] == u16::MAX {
        // we hit `cap` before finding a path to state 1
        return None;
    }
    Some(state_1_weight + to_zero[1] as u32)
}

fn evaluate_a_dfree(rate: u32, order: u32, polys: &[u16], d_free: u32) -> u64 {
    // a_dfree == the number of paths that start at state 0 and merge back to state 0
    //    with output weight d_free
    // this is the first term of the "distance spectrum"
    // more paths is worse (more ambiguity for the decoder)

    // `target` must be None here so that we get a full table
    let to_zero = shortest_output_weight_to_zero(order, rate, polys, d_free, None);
    let mask = (1u32 << order) - 1;

    let mut a_dfree = 0u64;
    let mut stack = vec![(0, 0u32)];
    while let Some((state, weight)) = stack.pop() {
        if weight.saturating_add(to_zero[state as usize] as u32) > d_free {
            continue;
        }
        let successor_prefix = (state << 1) & mask;
        let successors = [successor_prefix, successor_prefix | 1];
        for next_state in successors {
            let next_weight = weight + output_weight_for_state(next_state, rate, polys);
            if next_state == 0 {
                if next_weight == d_free {
                    a_dfree += 1;
                }
            } else if next_weight.saturating_add(to_zero[next_state as usize] as u32) <= d_free {
                stack.push((next_state, next_weight));
            }
        }
    }
    a_dfree
}

fn reciprocal(polys: &[u16], order: u32, out: &mut [u16]) {
    for (i, p) in polys.iter().enumerate() {
        out[i] = p.reverse_bits() >> (u16::BITS - order);
    }
    // polys need to remain sorted in ascending order
    out.sort_unstable();
}

fn combinations_with_repetition(values: u128, slots: u128) -> u128 {
    // order *does not* matter, we generate in increasing order, so we do not have permutations
    // c'(p, r) = c(p + r - 1, r)
    //          =                                                     (p + r - 1)! / (r! * (p - 1)!)
    //          = (p + r - 1) * (p + r - 2) * ... * (p + r - r) * (p + r - r - 1)! / (r! * (p - 1)!)
    //          =         (p + r - 1) * (p + r - 2) * ... * (p + r - r) * (p - 1)! / (r! * (p - 1)!)
    //          =                    (p + r - 1) * (p + r - 2) * ... * (p + r - r) / r!
    //          =                        p * (p + 1) * (p + 2) * ... * (p + r - 1) / (1 * 2 * 3 * ... * r)
    let mut count = 1u128;
    for i in 0..slots {
        // the iteration/division order matters here for integer division
        // the divisor must ascend/increment for integer division to work
        // and an ascending numerator makes overflow less likely
        count = count * (values + i) / (i + 1);
    }
    count
}

fn candidate_count(rate: u32, order: u32) -> u128 {
    // how many distinct values for one poly? top and bottom bits are fixed
    let p = 1u128 << (order.max(2) - 2);
    combinations_with_repetition(p, rate as u128)
}

struct PolyIterator {
    poly: Vec<u16>,
    rate: usize,
    maxcoeff: u16,
    done: bool,
}

impl PolyIterator {
    fn from_index(rate: u32, order: u32, mut index: u128) -> Self {
        // this is a standard unranking of combinations with repetition

        // we're filling polys from left to right, and each slot is at least as
        //   large as the previous slot (non-decreasing)

        // we'll start with the leftmost poly and try fixing it to the smallest valid poly.
        // then we'll count how many valid poly combinations fill the remaining slots.
        // if this count is *less* than our index, then we need to try the next poly in the
        //   first slot. once the count of valid polys remaining is greater than our index,
        //   we accept the current poly in the current slot and move on to the next one,
        //   repeating the same process

        // this is roughly analogous to a standard base conversion, but here the number
        //   of ways to fill the remaining digits decreases every time we increase the
        //   largest digit because of the non-decreasing requirement

        let p = 1u128 << (order.max(2) - 2); // distinct values per slot
        let rate = rate as usize;

        // n.b. in order to make this counting easier, we're mapping the polys to a sort of
        //   poly index. in order to retrieve the actual poly, we'll shift this value left by
        //   1 and then OR in the high and low order bits
        let mut values = vec![0u16; rate];
        let mut v = 0u128;
        for slot in 0..rate {
            let remaining_slots = (rate - slot - 1) as u128;
            loop {
                // start at the smallest valid poly and count up until we reach index
                let block = combinations_with_repetition(p - v, remaining_slots);
                if index < block {
                    // we have found the poly for this slot
                    break;
                }
                // we need to go larger. count up.
                index -= block;
                v += 1;
            }
            values[slot] = v as u16;
        }

        // all valid codes need the high and low order bits set or else they are degenerate to a lower order
        // (that also means consecutive candidate values for one poly differ by 2)
        let highbit = 1u16 << (order - 1);
        let lowbit = 1;
        PolyIterator {
            poly: values.iter().map(|&value| (value << 1) | highbit | lowbit).collect(),
            rate,
            maxcoeff: (1u16 << order) - 1,
            done: false,
        }
    }
}

impl PolyIterator {
    fn next(&mut self, into: &mut [u16]) -> bool {
        if self.done {
            return false;
        }

        into.copy_from_slice(&self.poly);
        let mut i = self.rate;

        if self.poly[i - 1] < self.maxcoeff {
            // common case: we can increment the last coefficient
            self.poly[i - 1] += 2;
            return true;
        }

        // find the largest poly index that *isn't* maxed out yet
        // (we're incrementing the polys right-to-left)
        while i > 0 && self.poly[i - 1] >= self.maxcoeff {
            i -= 1;
        }

        if i == 0 {
            // all indices are maxed out, so we're done
            self.done = true;
        } else {
            // we always want the lowest bit set, so count up by 2 every time
            self.poly[i - 1] += 2;
            for j in i..self.rate {
                // this is a kind-of carry, but rather than restart at startcoeff,
                //    we jumpstart the lower values to the value we just set
                // this helps us skip a bunch of duplicates (permutations of the same codes)
                // the polys will always be monotonically increasing from left to right
                self.poly[j] = self.poly[i - 1];
            }
        }
        true
    }
}

struct SharedState {
    evaluated: u128,
    step: u128,
    next_print: u128,
    total: u128,
}

impl SharedState {
    fn next_batch(&mut self) -> (u128, usize) {
        let start = self.evaluated;
        // do a sort of jumpstart in order to help bootstrap the d_free bound
        let start = (start + (self.total as f32 * START_OFFSET) as u128) % self.total;
        let length = (self.total - start)
            .min(self.total - self.evaluated)
            .min(THREAD_BATCH as u128) as usize;
        self.evaluated += length as u128;
        while self.next_print <= self.evaluated {
            println!(
                "evaluated {} / {} candidates ({}%)",
                self.next_print,
                self.total,
                self.next_print * 100 / self.total
            );
            self.next_print += self.step;
        }
        (start, length)
    }
}

struct Candidate {
    poly: Vec<u16>,
    d_free: u32,
    a_dfree: u64,
    rank: u128,
}

struct BestPolys {
    rate: u32,
    order: u32,
    reciprocal: Vec<u16>,
    kept: Vec<Candidate>,
    num_kept: usize,
}

impl BestPolys {
    fn new(rate: u32, order: u32, num_kept: usize) -> Self {
        BestPolys {
            rate,
            order,
            reciprocal: vec![0; rate as usize],
            kept: Vec::with_capacity(num_kept + 1),
            num_kept,
        }
    }

    fn offer(&mut self, poly: &[u16], d_free: u32, rank: u128) {
        if self.kept.len() == self.num_kept {
            let worst = self.kept.last().unwrap();
            if d_free < worst.d_free {
                return;
            }
            if d_free == worst.d_free {
                // use the a_dfree as a tiebreaker
                let a_dfree = evaluate_a_dfree(self.rate, self.order, poly, d_free);
                if a_dfree > worst.a_dfree || (a_dfree == worst.a_dfree && poly > &worst.poly) {
                    // as a final tiebreaker, use lexicographic order
                    // this means later polys lose
                    return;
                }
                self.admit(poly.to_vec(), d_free, a_dfree, rank);
                return;
            }
            // let d_free > worst.d_free fallthrough to below (same as kept nonfull)
        }
        let a_dfree = evaluate_a_dfree(self.rate, self.order, poly, d_free);
        self.admit(poly.to_vec(), d_free, a_dfree, rank);
    }

    fn admit(&mut self, poly: Vec<u16>, d_free: u32, a_dfree: u64, rank: u128) {
        // we can get the reciprocal for free (same distances, easy to compute)
        reciprocal(&poly, self.order, &mut self.reciprocal);
        if self.reciprocal != poly {
            self.insert(self.reciprocal.clone(), d_free, a_dfree, rank);
        }
        self.insert(poly, d_free, a_dfree, rank);
    }

    fn insert(&mut self, poly: Vec<u16>, d_free: u32, a_dfree: u64, rank: u128) {
        // preserve the lex sorting for the twin/reciprocal
        let pos = self
            .kept
            .iter()
            .position(|c| {
                c.d_free < d_free
                    || (c.d_free == d_free && c.a_dfree > a_dfree)
                    || (c.d_free == d_free && c.a_dfree == a_dfree && c.poly > poly)
            })
            .unwrap_or(self.kept.len());
        self.kept.insert(
            pos,
            Candidate {
                poly,
                d_free,
                a_dfree,
                rank,
            },
        );
        self.kept.truncate(self.num_kept);
    }

    fn threshold(&self) -> Option<u32> {
        // get the d_free of the worst kept candidate so that we can more cheaply reject candidates
        (self.kept.len() == self.num_kept).then(|| self.kept.last().unwrap().d_free)
    }
}

fn snr_points(order: u32) -> [f64; NUM_BER_POINTS] {
    // which points should we measure in the BER test?
    // try to capture the "knee". we need lower SNR for higher order codes
    match order {
        4 | 5 => [6.0, 5.5, 5.0, 4.5],
        6 => [5.5, 5.0, 4.5, 4.0],
        7 => [5.0, 4.5, 4.0, 3.5],
        8 | 9 => [4.5, 4.0, 3.5, 3.0],
        _ => [4.0, 3.5, 3.0, 2.5],
    }
}

fn main() {
    let mut positional: Vec<String> = Vec::new();

    let mut ber_len: Option<usize> = None;
    let mut base_seed = 0x1234u64;
    let mut threads: usize = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut min_d_free: Option<u32> = None;
    let mut num_kept: usize = DEFAULT_NUM_KEPT;

    let mut args = env::args();
    let name = args.next().unwrap();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--ber" => {
                let v = args.next().unwrap_or_else(|| {
                    eprintln!("--ber needs a length (bytes)");
                    std::process::exit(1);
                });
                ber_len = Some(v.parse().expect("--ber bytes"));
            }
            "--seed" => {
                let v = args.next().unwrap_or_else(|| {
                    eprintln!("--seed needs a value");
                    std::process::exit(1);
                });
                base_seed = v.parse().expect("--seed");
            }
            "--threads" => {
                let v = args.next().unwrap_or_else(|| {
                    eprintln!("--threads needs a count");
                    std::process::exit(1);
                });
                threads = v.parse().expect("--threads");
            }
            "--min-d-free" => {
                let v = args.next().unwrap_or_else(|| {
                    eprintln!("--min-d-free needs a value");
                    std::process::exit(1);
                });
                min_d_free = Some(v.parse().expect("--min-d-free"));
            }
            "--num-polys" => {
                let v = args.next().unwrap_or_else(|| {
                    eprintln!("--num-polys needs a value");
                    std::process::exit(1);
                });
                num_kept = v.parse().expect("--num-polys");
            }
            _ => positional.push(a),
        }
    }

    if positional.len() < 2 {
        eprintln!("usage: {} <rate> <order> [--ber <bytes>] [--seed <seed>] [--threads <n>] [--min-d-free <d_free>] [--num-polys <n>]", name);
        std::process::exit(1);
    }

    let rate: u32 = positional[0].parse().expect("rate");
    let order: u32 = positional[1].parse().expect("order");
    let threads = threads.max(1);

    let eb_n0 = snr_points(order);

    let total = candidate_count(rate, order);
    let heller_bound = heller(rate, order);
    let griesmer_bound = griesmer(rate, order);
    let min_d_free = min_d_free.unwrap_or(0);

    println!("Searching {total} ({total:.2e}) convolutional codes with rate 1/{rate} and order {order}, d_free upper bound: {griesmer_bound}, minimum bound: {min_d_free}");
    let step = (total / 20).max(1); // report roughly every 5%
    let shared = Mutex::new(SharedState {
        evaluated: 0,
        step,
        next_print: step,
        total,
    });

    // do a threaded search for polys
    let thread_best_polys: Vec<BestPolys> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut keep = BestPolys::new(rate, order, num_kept);
                    let mut recip = vec![0; rate as usize];
                    let mut poly = vec![0; rate as usize];
                    loop {
                        let (start, length) = shared.lock().unwrap().next_batch();
                        if length == 0 {
                            break;
                        }
                        let mut polys = PolyIterator::from_index(rate, order, start);
                        for offset in 0..length {
                            let rank = start + offset as u128;
                            assert!(polys.next(&mut poly), "unranking disagrees with iterator");
                            reciprocal(&poly, order, &mut recip);
                            if recip < poly {
                                // we've already seen the reciprocal, move on
                                continue;
                            }
                            let min_bound = keep.threshold().unwrap_or(min_d_free).max(min_d_free);
                            if free_distance_bound(&poly, order, Some(min_bound)) >= min_bound {
                                if let Some(d_free) = evaluate_free_distance(rate, order, &poly) {
                                    // this will insert the poly and its reciprocal
                                    keep.offer(&poly, d_free, rank);
                                }
                            }
                        }
                        if start + length as u128 == total {
                            assert!(!polys.next(&mut poly), "poly iterator disagrees with candidate_count");
                        }
                    }
                    keep
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let evaluated = shared.into_inner().unwrap().evaluated;
    assert_eq!(evaluated, total, "candidate_count disagrees with the generator");
    println!("evaluated {evaluated} candidates total");

    // merge the per-thread keep lists: same sort keys as BestPolys::insert
    let mut candidates: Vec<Candidate> = thread_best_polys.into_iter().flat_map(|k| k.kept).collect();
    candidates.sort_by(|a, b| {
        b.d_free
            .cmp(&a.d_free)
            .then(a.a_dfree.cmp(&b.a_dfree))
            .then(a.poly.cmp(&b.poly))
    });
    candidates.truncate(num_kept);

    // do a BER measurement if requested (opt-in with --ber <bytes>)
    let ber: Option<Vec<[usize; NUM_BER_POINTS]>> = ber_len.map(|len| {
        let block = len.min(16384);
        let num_candidates = candidates.len();
        let num_jobs = NUM_BER_POINTS * num_candidates;
        let results: Vec<AtomicUsize> = (0..num_jobs).map(|_| AtomicUsize::new(0)).collect();
        let next_job = AtomicUsize::new(0);
        let cand_ref = &candidates;
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| loop {
                    let j = next_job.fetch_add(1, Ordering::Relaxed);
                    if j >= num_jobs {
                        break;
                    }
                    let (ber_index, candidate_index) = (j / num_candidates, j % num_candidates);
                    let seed = base_seed ^ (ber_index as u64);
                    let errs = measure_ber(
                        rate,
                        order,
                        &cand_ref[candidate_index].poly,
                        eb_n0[ber_index],
                        len,
                        block,
                        seed,
                    );
                    results[j].store(errs, Ordering::Relaxed);
                });
            }
        });
        (0..num_candidates)
            .map(|candidate_index| {
                std::array::from_fn(|i| results[i * num_candidates + candidate_index].load(Ordering::Relaxed))
            })
            .collect()
    });

    assert!(candidates.iter().all(|c| c.d_free <= heller_bound));
    assert!(candidates.iter().all(|c| c.d_free <= griesmer_bound));

    println!(
        "\ntop {} codes (exact free distance{}):",
        candidates.len(),
        if ber.is_some() { ", + BER per Eb/N0" } else { "" }
    );
    let bits = ber_len.unwrap_or(0) as f64 * 8.0;
    for (ci, cand) in candidates.iter().enumerate() {
        for p in &cand.poly {
            print!(" {:06o}", p);
        }
        print!(
            "  [d_free={}, a_dfree={}, rank={} ({:.1}%)]",
            cand.d_free,
            cand.a_dfree,
            cand.rank,
            cand.rank as f64 * 100.0 / total as f64
        );
        if let Some(ber) = &ber {
            print!(" :");
            for s in 0..NUM_BER_POINTS {
                print!(" {:.2e}@{:.1}dB", ber[ci][s] as f64 / bits, eb_n0[s]);
            }
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dfree_matches_published_tables() {
        // compare against published/well-known tables
        // a_dfree generated locally (regression tested)
        let cases: &[(u32, u32, &[u16], u32, u64)] = &[
            // Larsen 1973 (d_free)
            (2, 3, &[0o5, 0o7], 5, 1),
            (2, 4, &[0o15, 0o17], 6, 1),
            (2, 5, &[0o23, 0o35], 7, 2),
            (2, 6, &[0o53, 0o75], 8, 1),
            (2, 7, &[0o133, 0o171], 10, 11),
            (2, 8, &[0o247, 0o371], 10, 1),
            (2, 9, &[0o561, 0o753], 12, 11),
            (2, 10, &[0o1167, 0o1545], 12, 2),
            (2, 11, &[0o2335, 0o3661], 14, 21),
            (2, 12, &[0o4335, 0o5723], 15, 16),
            (2, 13, &[0o10533, 0o17661], 16, 33),
            (2, 14, &[0o21675, 0o27123], 16, 4),
            (3, 3, &[0o5, 0o7, 0o7], 8, 2),
            (3, 4, &[0o13, 0o15, 0o17], 10, 3),
            (3, 5, &[0o25, 0o33, 0o37], 12, 5),
            (3, 6, &[0o47, 0o53, 0o75], 13, 1),
            (3, 7, &[0o133, 0o145, 0o175], 15, 3),
            (3, 8, &[0o225, 0o331, 0o367], 16, 1),
            (3, 9, &[0o557, 0o663, 0o711], 18, 5),
            (3, 10, &[0o1117, 0o7365, 0o1633], 20, 8),
            (3, 11, &[0o2353, 0o2671, 0o3175], 22, 14),
            (3, 12, &[0o4767, 0o5723, 0o6265], 24, 21),
            (3, 13, &[0o10533, 0o10675, 0o17661], 24, 10),
            (3, 14, &[0o21645, 0o35661, 0o37133], 26, 12),
            (4, 3, &[0o5, 0o7, 0o7, 0o7], 10, 1),
            (4, 4, &[0o13, 0o15, 0o15, 0o17], 13, 2),
            (4, 5, &[0o25, 0o27, 0o33, 0o37], 16, 4),
            (4, 6, &[0o53, 0o67, 0o71, 0o75], 18, 3),
            (4, 7, &[0o135, 0o135, 0o147, 0o163], 20, 10),
            (4, 8, &[0o235, 0o275, 0o313, 0o357], 22, 1),
            (4, 9, &[0o463, 0o535, 0o733, 0o745], 24, 2),
            (4, 10, &[0o1117, 0o1365, 0o1633, 0o1653], 27, 4),
            (4, 11, &[0o2327, 0o2353, 0o2671, 0o3175], 29, 5),
            (4, 12, &[0o4767, 0o5723, 0o6265, 0o7455], 32, 14),
            (4, 13, &[0o11145, 0o12477, 0o15573, 0o16727], 33, 5),
            (4, 14, &[0o21113, 0o23175, 0o35527, 0o35537], 36, 19),
            // Friedrichs 1996 (d_free)
            (2, 15, &[0o56721, 0o61713], 18, 33),
        ];
        for &(rate, order, polys, d_exp, a_exp) in cases {
            let d_free = evaluate_free_distance(rate, order, polys);
            let n_dfree = evaluate_a_dfree(rate, order, polys, d_free.unwrap());
            assert_eq!(d_free, Some(d_exp));
            assert_eq!(n_dfree, a_exp);
        }
    }

    #[test]
    fn reciprocal_code_shares_spectrum() {
        const ORIG: [u16; 2] = [0o133, 0o171];
        const RECIP: [u16; 2] = [0o117, 0o155];
        let mut recip = vec![0; 2];
        reciprocal(&ORIG, 7, &mut recip);
        assert_eq!(recip, RECIP);
        let d1 = evaluate_free_distance(2, 7, &ORIG).unwrap();
        let d2 = evaluate_free_distance(2, 7, &RECIP).unwrap();
        assert_eq!(d1, d2);
        assert_eq!(evaluate_a_dfree(2, 7, &ORIG, d1), evaluate_a_dfree(2, 7, &RECIP, d2));
    }

    #[test]
    fn catastrophic_codes_flagged() {
        assert!(is_catastrophic(&[0o4, 0o6]));
        assert!(evaluate_free_distance(2, 3, &[0o4, 0o6]).is_none());
        assert!(!is_catastrophic(&[0o5, 0o7]));
    }

    #[test]
    fn unranking_matches_iteration() {
        for (rate, order) in [(2u32, 5u32), (2, 8), (3, 4), (4, 3), (6, 3)] {
            let total = candidate_count(rate, order);
            let mut seq = PolyIterator::from_index(rate, order, 0);
            let mut expected = vec![0u16; rate as usize];
            let mut jumped = vec![0u16; rate as usize];
            for index in 0..total {
                // run through every poly and make sure we can restart from any index
                assert!(seq.next(&mut expected));
                let mut it = PolyIterator::from_index(rate, order, index);
                assert!(it.next(&mut jumped));
                assert_eq!(jumped, expected, "index {index} at rate {rate} order {order}");
            }
            assert!(!seq.next(&mut expected));
        }
    }

    #[test]
    fn test_heller_griesmer_bounds() {
        // Larsen 1973 gives us the min of {heller, griesmer}
        // we compute the other here locally (regression test)
        assert_eq!(heller(2, 3), 5);
        assert_eq!(griesmer(2, 3), 5);
        assert_eq!(heller(2, 4), 6);
        assert_eq!(griesmer(2, 4), 6);
        assert_eq!(heller(2, 5), 8);
        assert_eq!(griesmer(2, 5), 8);
        assert_eq!(heller(2, 6), 9);
        assert_eq!(griesmer(2, 6), 8);
        assert_eq!(heller(2, 7), 10);
        assert_eq!(griesmer(2, 7), 10);
        assert_eq!(heller(2, 8), 11);
        assert_eq!(griesmer(2, 8), 11);
        assert_eq!(heller(2, 9), 12);
        assert_eq!(griesmer(2, 9), 12);
        assert_eq!(heller(2, 10), 13);
        assert_eq!(griesmer(2, 10), 13);
        assert_eq!(heller(2, 11), 14);
        assert_eq!(griesmer(2, 11), 14);
        assert_eq!(heller(2, 12), 16);
        assert_eq!(griesmer(2, 12), 16);
        assert_eq!(heller(2, 13), 17);
        assert_eq!(griesmer(2, 13), 16);
        assert_eq!(heller(2, 14), 18);
        assert_eq!(griesmer(2, 14), 17);
        assert_eq!(heller(2, 15), 19);
        assert_eq!(griesmer(2, 15), 18);
        assert_eq!(heller(3, 3), 8);
        assert_eq!(griesmer(3, 3), 8);
        assert_eq!(heller(3, 4), 10);
        assert_eq!(griesmer(3, 4), 10);
        assert_eq!(heller(3, 5), 12);
        assert_eq!(griesmer(3, 5), 12);
        assert_eq!(heller(3, 6), 13);
        assert_eq!(griesmer(3, 6), 13);
        assert_eq!(heller(3, 7), 15);
        assert_eq!(griesmer(3, 7), 15);
        assert_eq!(heller(3, 8), 17);
        assert_eq!(griesmer(3, 8), 16);
        assert_eq!(heller(3, 9), 18);
        assert_eq!(griesmer(3, 9), 18);
        assert_eq!(heller(3, 10), 20);
        assert_eq!(griesmer(3, 10), 20);
        assert_eq!(heller(3, 11), 22);
        assert_eq!(griesmer(3, 11), 22);
        assert_eq!(heller(3, 12), 24);
        assert_eq!(griesmer(3, 12), 24);
        assert_eq!(heller(3, 13), 25);
        assert_eq!(griesmer(3, 13), 24);
        assert_eq!(heller(3, 14), 27);
        assert_eq!(griesmer(3, 14), 26);
        assert_eq!(heller(3, 15), 28);
        assert_eq!(griesmer(3, 15), 28);

        for rate in 2..=6 {
            for order in 3..=15 {
                assert!(heller(rate, order) >= griesmer(rate, order));
            }
        }
    }
}
