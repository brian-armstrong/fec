use std::simd::prelude::*;

use super::oct_lookup::{DistanceShuffle, OctLookup};

// explicit override for _mm_shuffle_epi8/pshufb on x86
// Rust can't quite seem to figure out to do this from swizzle_dyn consistently
// XXX limit this to SSSE3 and up only
#[inline(always)]
fn pshufb(src: u8x16, mask: u8x16) -> u8x16 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm_shuffle_epi8;
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm_shuffle_epi8;
        unsafe { core::mem::transmute(_mm_shuffle_epi8(src.into(), mask.into())) }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        src.swizzle_dyn(mask)
    }
}

#[inline(always)]
fn vpshufb256(src: u8x32, mask: u8x32) -> u8x32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm256_shuffle_epi8;
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm256_shuffle_epi8;
        unsafe { core::mem::transmute(_mm256_shuffle_epi8(src.into(), mask.into())) }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        src.swizzle_dyn(mask)
    }
}

#[inline(always)]
fn vpshufb512(src: u8x64, mask: u8x64) -> u8x64 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_mm512_shuffle_epi8;
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_mm512_shuffle_epi8;
        unsafe { core::mem::transmute(_mm512_shuffle_epi8(src.into(), mask.into())) }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        src.swizzle_dyn(mask)
    }
}

#[inline(always)]
fn vpermi2w512(idx: u16x32, a: u16x32, b: u16x32) -> u16x32 {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::_mm512_permutex2var_epi16;
        unsafe { core::mem::transmute(_mm512_permutex2var_epi16(a.into(), idx.into(), b.into())) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut out = [0u16; 32];
        for (o, &k) in out.iter_mut().zip(idx.as_array()) {
            let k = (k & 63) as usize;
            *o = if k < 32 { a[k] } else { b[k - 32] };
        }
        u16x32::from_array(out)
    }
}

pub (crate) trait Lane: Copy {
    type Vec: Copy;
    const LANES: usize;

    fn add(a: Self::Vec, b: Self::Vec) -> Self::Vec;
    fn min(a: Self::Vec, b: Self::Vec) -> Self::Vec;
    unsafe fn write_history(hist: *mut u8, byte_off: usize, lo: Self::Vec, hi: Self::Vec);
}

#[derive(Clone, Copy)]
pub(crate) struct Lane128;
impl Lane for Lane128 {
    type Vec = u16x8;
    const LANES: usize = 8;

    fn add(a: u16x8, b: u16x8) -> u16x8 {
        a + b
    }
    fn min(a: u16x8, b: u16x8) -> u16x8 {
        a.simd_min(b)
    }
    #[inline(always)]
    unsafe fn write_history(hist: *mut u8, byte_off: usize, lo: u16x8, hi: u16x8) {
        let lo_i: i16x8 = core::mem::transmute(lo);
        let hi_i: i16x8 = core::mem::transmute(hi);
        let mask = lo_i.simd_gt(hi_i).to_bitmask() as u8;
        *hist.add(byte_off) = mask;
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Lane256;
impl Lane for Lane256 {
    type Vec = u16x16;
    const LANES: usize = 16;

    fn add(a: u16x16, b: u16x16) -> u16x16 {
        a + b
    }
    fn min(a: u16x16, b: u16x16) -> u16x16 {
        a.simd_min(b)
    }
    #[inline(always)]
    unsafe fn write_history(hist: *mut u8, byte_off: usize, lo: u16x16, hi: u16x16) {
        let lo_i: i16x16 = core::mem::transmute(lo);
        let hi_i: i16x16 = core::mem::transmute(hi);
        let mask = lo_i.simd_gt(hi_i).to_bitmask() as u16;
        *hist.add(byte_off)     = (mask & 0xff) as u8;
        *hist.add(byte_off + 1) = (mask >> 8)  as u8;
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Lane512;
impl Lane for Lane512 {
    type Vec = u16x32;
    const LANES: usize = 32;

    fn add(a: u16x32, b: u16x32) -> u16x32 {
        a + b
    }
    fn min(a: u16x32, b: u16x32) -> u16x32 {
        a.simd_min(b)
    }
    #[inline(always)]
    unsafe fn write_history(hist: *mut u8, byte_off: usize, lo: u16x32, hi: u16x32) {
        let lo_i: i16x32 = core::mem::transmute(lo);
        let hi_i: i16x32 = core::mem::transmute(hi);
        let mask = lo_i.simd_gt(hi_i).to_bitmask() as u32;
        *hist.add(byte_off)     = (mask & 0xff) as u8;
        *hist.add(byte_off + 1) = ((mask >> 8)  & 0xff) as u8;
        *hist.add(byte_off + 2) = ((mask >> 16) & 0xff) as u8;
        *hist.add(byte_off + 3) = (mask >> 24) as u8;
    }
}

pub(crate) trait MemoryLane: Lane {
    unsafe fn load_pred_dup(prev: *const u16, pred: usize, off: usize) -> Self::Vec;
    unsafe fn store_err(errors: *mut u16, succ: usize, v: Self::Vec);
    fn min_value(errors: &[u16]) -> u16;
    fn renorm(errors: &mut [u16], min_distance: u16);
}

impl MemoryLane for Lane128 {
    #[inline(always)]
    unsafe fn load_pred_dup(prev: *const u16, pred: usize, off: usize) -> u16x8 {
        let p4 = u16x4::from_array(*(prev.add(pred + off) as *const [u16; 4]));
        simd_swizzle!(p4, [0, 0, 1, 1, 2, 2, 3, 3])
    }
    #[inline(always)]
    unsafe fn store_err(errors: *mut u16, succ: usize, v: u16x8) {
        *(errors.add(succ) as *mut [u16; 8]) = v.to_array();
    }
    #[inline(always)]
    fn min_value(errors: &[u16]) -> u16 {
        let mut acc = u16x8::splat(u16::MAX);
        for c in errors.as_chunks::<8>().0 {
            acc = acc.simd_min(u16x8::from_slice(c));
        }
        acc.reduce_min()
    }
    #[inline(always)]
    fn renorm(errors: &mut [u16], min_distance: u16) {
        let m = u16x8::splat(min_distance);
        for c in errors.as_chunks_mut::<8>().0 {
            let v = u16x8::from_slice(c);
            c.copy_from_slice(&v.saturating_sub(m).to_array());
        }
    }
}

impl MemoryLane for Lane256 {
    #[inline(always)]
    unsafe fn load_pred_dup(prev: *const u16, pred: usize, off: usize) -> u16x16 {
        let p8 = u16x8::from_array(*(prev.add(pred + off) as *const [u16; 8]));
        simd_swizzle!(p8, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
    }
    #[inline(always)]
    unsafe fn store_err(errors: *mut u16, succ: usize, v: u16x16) {
        *(errors.add(succ) as *mut [u16; 16]) = v.to_array();
    }
    #[inline(always)]
    fn min_value(errors: &[u16]) -> u16 {
        let mut acc = u16x16::splat(u16::MAX);
        for c in errors.as_chunks::<16>().0 {
            acc = acc.simd_min(u16x16::from_slice(c));
        }
        acc.reduce_min()
    }
    #[inline(always)]
    fn renorm(errors: &mut [u16], min_distance: u16) {
        let m = u16x16::splat(min_distance);
        for c in errors.as_chunks_mut::<16>().0 {
            let v = u16x16::from_slice(c);
            c.copy_from_slice(&v.saturating_sub(m).to_array());
        }
    }
}

impl MemoryLane for Lane512 {
    #[inline(always)]
    unsafe fn load_pred_dup(prev: *const u16, pred: usize, off: usize) -> u16x32 {
        let p16 = u16x16::from_array(*(prev.add(pred + off) as *const [u16; 16]));
        simd_swizzle!(p16, [
            0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
            8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15])
    }
    #[inline(always)]
    unsafe fn store_err(errors: *mut u16, succ: usize, v: u16x32) {
        *(errors.add(succ) as *mut [u16; 32]) = v.to_array();
    }
    #[inline(always)]
    fn min_value(errors: &[u16]) -> u16 {
        let mut acc = u16x32::splat(u16::MAX);
        for c in errors.as_chunks::<32>().0 {
            acc = acc.simd_min(u16x32::from_slice(c));
        }
        acc.reduce_min()
    }
    #[inline(always)]
    fn renorm(errors: &mut [u16], min_distance: u16) {
        let m = u16x32::splat(min_distance);
        for c in errors.as_chunks_mut::<32>().0 {
            let v = u16x32::from_slice(c);
            c.copy_from_slice(&v.saturating_sub(m).to_array());
        }
    }
}

pub(crate) trait OctLookupLane: MemoryLane {
    const LOOKUP_WIDTH: usize;

    fn fill_distances(distances: &mut [u16], oct_lookup: &mut OctLookup);
    fn keys(oct_lookup: &OctLookup) -> *const u16;
    fn octdist(oct_lookup: &OctLookup) -> *const u16;
    unsafe fn load_dist(keys: *const u16, octdist: *const u16, oct: usize, key_off: usize) -> Self::Vec;
}

impl OctLookupLane for Lane128 {
    const LOOKUP_WIDTH: usize = 8;

    #[inline(always)]
    fn fill_distances(distances: &mut [u16], oct_lookup: &mut OctLookup) {
        oct_lookup.fill_distances(distances);
    }
    #[inline(always)]
    fn keys(oct_lookup: &OctLookup) -> *const u16 {
        oct_lookup.keys.as_ptr()
    }
    #[inline(always)]
    fn octdist(oct_lookup: &OctLookup) -> *const u16 {
        oct_lookup.distances.as_ptr()
    }
    #[inline(always)]
    unsafe fn load_dist(keys: *const u16, octdist: *const u16, oct: usize, key_off: usize) -> u16x8 {
        let key = *keys.add(key_off + oct) as usize;
        u16x8::from_array(*(octdist.add(key) as *const [u16; 8]))
    }
}

impl OctLookupLane for Lane256 {
    const LOOKUP_WIDTH: usize = 16;

    #[inline(always)]
    fn fill_distances(distances: &mut [u16], oct_lookup: &mut OctLookup) {
        oct_lookup.fill_distances16(distances);
    }
    #[inline(always)]
    fn keys(oct_lookup: &OctLookup) -> *const u16 {
        oct_lookup.keys16.as_ptr()
    }
    #[inline(always)]
    fn octdist(oct_lookup: &OctLookup) -> *const u16 {
        oct_lookup.distances16.as_ptr()
    }
    #[inline(always)]
    unsafe fn load_dist(keys: *const u16, octdist: *const u16, oct: usize, key_off: usize) -> u16x16 {
        let key = *keys.add(key_off + oct) as usize;
        u16x16::from_array(*(octdist.add(key) as *const [u16; 16]))
    }
}

impl OctLookupLane for Lane512 {
    const LOOKUP_WIDTH: usize = 16;

    // note that we use the 16 versions here intentionally rather than going wider
    // there's a tension between wider with more cache use and narrower with more lookups
    #[inline(always)]
    fn fill_distances(distances: &mut [u16], oct_lookup: &mut OctLookup) {
        oct_lookup.fill_distances16(distances);
    }
    #[inline(always)]
    fn keys(oct_lookup: &OctLookup) -> *const u16 {
        oct_lookup.keys16.as_ptr()
    }
    #[inline(always)]
    fn octdist(oct_lookup: &OctLookup) -> *const u16 {
        oct_lookup.distances16.as_ptr()
    }
    #[inline(always)]
    unsafe fn load_dist(keys: *const u16, octdist: *const u16, oct: usize, key_off: usize) -> u16x32 {
        // because we stay at the 16-wide version, we need two lookups here
        let k0 = *keys.add(key_off + oct) as usize;
        let k1 = *keys.add(key_off + oct + 1) as usize;
        let d0 = u16x16::from_array(*(octdist.add(k0) as *const [u16; 16]));
        let d1 = u16x16::from_array(*(octdist.add(k1) as *const [u16; 16]));
        simd_swizzle!(d0, d1, [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
    }
}

pub(crate) trait ShuffleLane: MemoryLane {
    const SHUFFLE_WIDTH: usize;

    fn shuffle_mask_ptr(distance_shuffle: &DistanceShuffle) -> *const u8;
    unsafe fn load_dist_shuffle(ctrl: *const u8, g: usize, ctrl_off: usize, dsrc16: u8x16) -> Self::Vec;
}

impl ShuffleLane for Lane128 {
    const SHUFFLE_WIDTH: usize = 8;

    #[inline(always)]
    fn shuffle_mask_ptr(distance_shuffle: &DistanceShuffle) -> *const u8 {
        distance_shuffle.shuffle8.as_ptr() as *const u8
    }
    #[inline(always)]
    unsafe fn load_dist_shuffle(ctrl: *const u8, g: usize, ctrl_off: usize, dsrc16: u8x16) -> u16x8 {
        let mask = *(ctrl as *const u8x16).add(ctrl_off + g);
        core::mem::transmute(pshufb(dsrc16, mask))
    }
}

impl ShuffleLane for Lane256 {
    const SHUFFLE_WIDTH: usize = 16;

    #[inline(always)]
    fn shuffle_mask_ptr(distance_shuffle: &DistanceShuffle) -> *const u8 {
        distance_shuffle.shuffle16.as_ptr() as *const u8
    }
    #[inline(always)]
    unsafe fn load_dist_shuffle(ctrl: *const u8, g: usize, ctrl_off: usize, dsrc16: u8x16) -> u16x16 {
        // going wider here does not allow us to accomodate more distances
        // instead, we just broadcast the same number of distances to each 128-bit lane
        let dsrc32: u8x32 = simd_swizzle!(
            dsrc16,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        );
        let mask = *(ctrl as *const u8x32).add(ctrl_off + g);
        core::mem::transmute(vpshufb256(dsrc32, mask))
    }
}

impl ShuffleLane for Lane512 {
    const SHUFFLE_WIDTH: usize = 32;

    #[inline(always)]
    fn shuffle_mask_ptr(distance_shuffle: &DistanceShuffle) -> *const u8 {
        distance_shuffle.shuffle32.as_ptr() as *const u8
    }
    #[inline(always)]
    unsafe fn load_dist_shuffle(ctrl: *const u8, g: usize, ctrl_off: usize, dsrc16: u8x16) -> u16x32 {
        // like the 256 version, broadcast the same number of distances to all 128-bit lanes
        let dsrc64: u8x64 = simd_swizzle!(
            dsrc16,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        );
        let mask = *(ctrl as *const u8x64).add(ctrl_off + g);
        core::mem::transmute(vpshufb512(dsrc64, mask))
    }
}

pub(crate) trait PermuteLane: MemoryLane {
    type DistTable: Copy;

    unsafe fn dist_table(distances: *const u16) -> Self::DistTable;
    unsafe fn permute_dist(table: Self::DistTable, poly: *const u8, at: usize) -> Self::Vec;
}

impl PermuteLane for Lane512 {
    type DistTable = (u16x32, u16x32);

    #[inline(always)]
    unsafe fn dist_table(distances: *const u16) -> (u16x32, u16x32) {
        let lo = u16x32::from_array(*(distances as *const [u16; 32]));
        let hi = u16x32::from_array(*(distances.add(32) as *const [u16; 32]));
        (lo, hi)
    }

    #[inline(always)]
    unsafe fn permute_dist(table: (u16x32, u16x32), poly: *const u8, at: usize) -> u16x32 {
        let outputs: u16x32 = u8x32::from_array(*(poly.add(at) as *const [u8; 32])).cast();
        vpermi2w512(outputs, table.0, table.1)
    }
}

pub(crate) trait RegisterLane: Lane {
    type Ctrl: Copy;

    // the register version has to reimplement basically everything (can't reuse MemoryLane)

    fn zeros() -> Self::Vec;

    // initial load and store into/out of registers from memory
    unsafe fn load(p: *const u16) -> Self::Vec;
    unsafe fn store(p: *mut u16, v: Self::Vec);

    fn shuffle_mask_ptr(distance_shuffle: &DistanceShuffle) -> *const Self::Ctrl;
    unsafe fn shuffle_dist(mask_ptr: *const Self::Ctrl, register: usize, offset: usize, dist16: u8x16) -> Self::Vec;

    fn pred_reg(prev: *const Self::Vec, pred: usize, offset: usize) -> Self::Vec;
    fn pred_half(prev: Self::Vec, odd: bool) -> Self::Vec;

    unsafe fn store_err<const N: usize>(errors: &mut [Self::Vec; N], succ: usize, v: Self::Vec);

    fn renorm_sub_min(prev: &mut [Self::Vec], nvec: usize);
}

impl RegisterLane for Lane128 {
    type Ctrl = u8x16;

    #[inline(always)]
    fn zeros() -> u16x8 {
        u16x8::splat(0)
    }
    #[inline(always)]
    unsafe fn load(p: *const u16) -> u16x8 {
        u16x8::from_array(*(p as *const [u16; 8]))
    }
    #[inline(always)]
    unsafe fn store(p: *mut u16, v: u16x8) {
        *(p as *mut [u16; 8]) = v.to_array();
    }
    #[inline(always)]
    fn shuffle_mask_ptr(distance_shuffle: &DistanceShuffle) -> *const u8x16 {
        distance_shuffle.shuffle8.as_ptr()
    }
    #[inline(always)]
    unsafe fn shuffle_dist(mask_ptr: *const u8x16, register: usize, offset: usize, dist16: u8x16) -> u16x8 {
        core::mem::transmute(pshufb(dist16, *mask_ptr.add(register + offset)))
    }
    #[inline(always)]
    fn pred_reg(prev: *const u16x8, pred: usize, offset: usize) -> u16x8 {
        unsafe { u16x8::from_array(*(prev.add(pred + offset) as *const [u16; 8])) }
    }
    #[inline(always)]
    fn pred_half(prev: u16x8, odd: bool) -> u16x8 {
        if odd {
            simd_swizzle!(prev, [4, 4, 5, 5, 6, 6, 7, 7])
        } else {
            simd_swizzle!(prev, [0, 0, 1, 1, 2, 2, 3, 3])
        }
    }
    #[inline(always)]
    unsafe fn store_err<const N: usize>(errors: &mut [u16x8; N], succ: usize, v: u16x8) {
        errors[succ] = v;
    }
    #[inline(always)]
    fn renorm_sub_min(prev: &mut [u16x8], nvec: usize) {
        let mut acc = prev[0];
        for v in 1..nvec {
            acc = acc.simd_min(prev[v]);
        }
        let m = u16x8::splat(acc.reduce_min());
        for v in 0..nvec {
            prev[v] = prev[v].saturating_sub(m);
        }
    }
}

impl RegisterLane for Lane256 {
    type Ctrl = u8x32;

    #[inline(always)]
    fn zeros() -> u16x16 {
        u16x16::splat(0)
    }
    #[inline(always)]
    unsafe fn load(p: *const u16) -> u16x16 {
        u16x16::from_array(*(p as *const [u16; 16]))
    }
    #[inline(always)]
    unsafe fn store(p: *mut u16, v: u16x16) {
        *(p as *mut [u16; 16]) = v.to_array();
    }
    #[inline(always)]
    fn shuffle_mask_ptr(distance_shuffle: &DistanceShuffle) -> *const u8x32 {
        distance_shuffle.shuffle16.as_ptr()
    }
    #[inline(always)]
    unsafe fn shuffle_dist(mask_ptr: *const u8x32, register: usize, offset: usize, dist16: u8x16) -> u16x16 {
        let dist32: u8x32 = simd_swizzle!(
            dist16,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        );
        core::mem::transmute(vpshufb256(dist32, *mask_ptr.add(register + offset)))
    }
    #[inline(always)]
    fn pred_reg(prev: *const u16x16, pred: usize, offset: usize) -> u16x16 {
        unsafe { u16x16::from_array(*(prev.add(pred + offset) as *const [u16; 16])) }
    }
    #[inline(always)]
    fn pred_half(prev: u16x16, odd: bool) -> u16x16 {
        if odd {
            simd_swizzle!(prev, [8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15])
        } else {
            simd_swizzle!(prev, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
        }
    }
    #[inline(always)]
    unsafe fn store_err<const N: usize>(errors: &mut [u16x16; N], succ: usize, v: u16x16) {
        errors[succ] = v;
    }
    #[inline(always)]
    fn renorm_sub_min(prev: &mut [u16x16], nvec: usize) {
        let mut acc = prev[0];
        for v in 1..nvec {
            acc = acc.simd_min(prev[v]);
        }
        let m = u16x16::splat(acc.reduce_min());
        for v in 0..nvec {
            prev[v] = prev[v].saturating_sub(m);
        }
    }
}

impl RegisterLane for Lane512 {
    type Ctrl = u8x64;

    #[inline(always)]
    fn zeros() -> u16x32 {
        u16x32::splat(0)
    }
    #[inline(always)]
    unsafe fn load(p: *const u16) -> u16x32 {
        u16x32::from_array(*(p as *const [u16; 32]))
    }
    #[inline(always)]
    unsafe fn store(p: *mut u16, v: u16x32) {
        *(p as *mut [u16; 32]) = v.to_array();
    }
    #[inline(always)]
    fn shuffle_mask_ptr(distance_shuffle: &DistanceShuffle) -> *const u8x64 {
        distance_shuffle.shuffle32.as_ptr()
    }
    #[inline(always)]
    unsafe fn shuffle_dist(mask_ptr: *const u8x64, register: usize, offset: usize, dist16: u8x16) -> u16x32 {
        let dist64: u8x64 = simd_swizzle!(
            dist16,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        );
        core::mem::transmute(vpshufb512(dist64, *mask_ptr.add(register + offset)))
    }
    #[inline(always)]
    fn pred_reg(prev: *const u16x32, pred: usize, offset: usize) -> u16x32 {
        unsafe { u16x32::from_array(*(prev.add(pred + offset) as *const [u16; 32])) }
    }
    #[inline(always)]
    fn pred_half(prev: u16x32, odd: bool) -> u16x32 {
        if odd {
            simd_swizzle!(prev, [
                16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23,
                24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31
            ])
        } else {
            simd_swizzle!(prev, [
                0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
                8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15
            ])
        }
    }
    #[inline(always)]
    unsafe fn store_err<const N: usize>(errors: &mut [u16x32; N], succ: usize, v: u16x32) {
        errors[succ] = v;
    }
    #[inline(always)]
    fn renorm_sub_min(prev: &mut [u16x32], nvec: usize) {
        let mut acc = prev[0];
        for v in 1..nvec {
            acc = acc.simd_min(prev[v]);
        }
        let m = u16x32::splat(acc.reduce_min());
        for v in 0..nvec {
            prev[v] = prev[v].saturating_sub(m);
        }
    }
}
