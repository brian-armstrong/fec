//! A C-ABI shim over the [`fec`] crate, matching a subset of the libfec
//! interfaces.
//!
//! This crate builds a `libfec.so` and `libfec.a` that a C program can link
//! against as a partial drop-in for Phil Karn's libfec. It exposes the Viterbi
//! lifecycle functions (`create/init/update/chainback/delete`) for the k=7, 9,
//! and rate-1/3 k=9 and rate-1/6 k=15 codes, plus the Reed-Solomon
//! `init/encode/decode_rs_char`, `_rs_8`, and `_rs_ccsds` entry points.
//!
//! It is a subset, not a complete libfec replacement. It covers the common
//! codes and calls, and does not export every symbol libfec provides. See the
//! crate README for the supported-symbol list and build instructions.

use std::slice;
use std::sync::{Mutex, OnceLock};
use libc::{c_int, c_uint};

#[cfg(not(feature = "simd"))]
use fec::ConvDecoder as Decoder;
#[cfg(feature = "simd")]
use fec::ConvSimdDecoder as Decoder;

use fec::{RsDecoder, RsEncoder};
use fec::reed_solomon::ccsds::{conv_to_dual, dual_to_conv};

#[cfg(feature = "simd")]
fn make_decoder<const R: u32, const O: u32>(polys: &[u16]) -> Decoder<R, O> {
    Decoder::<R, O>::new(polys)
}
#[cfg(not(feature = "simd"))]
fn make_decoder<const R: u32, const O: u32>(polys: &[u16]) -> Decoder {
    Decoder::new(R, O, polys)
}

#[repr(C)]
pub struct Shim<const R: u32, const O: u32> {
    #[cfg(feature = "simd")]
    decoder: Decoder<R, O>,
    #[cfg(not(feature = "simd"))]
    decoder: Decoder,
    rate: u32,
    order: u32,
    decode_buffer: Vec<u8>,
    read_index: usize,
    write_index: usize,
}

impl<const R: u32, const O: u32> Shim<R, O> {
    fn new(num_decoded_bits: usize, rate: u32, order: u32, polys: &[u16]) -> Shim<R, O> {
        let num_decoded: usize;
        if num_decoded_bits % 8 == 0 {
            num_decoded = num_decoded_bits / 8;
        } else {
            num_decoded = num_decoded_bits / 8 + 1;
        }

        let decode_buffer = vec![0; num_decoded + 1];

        Shim {
            decoder: make_decoder::<R, O>(polys),
            rate,
            order,
            decode_buffer,
            read_index: 0,
            write_index: 0,
        }
    }

    fn init(&mut self) {
        self.read_index = 0;
        self.write_index = 0;
    }

    fn decode(&mut self, encoded: &[u8]) {
        let remaining_buffer = self.decode_buffer.len() - self.write_index;
        let remaining_bits = 8 * remaining_buffer;

        let mut decoded_len = (encoded.len() / self.rate as usize) - (self.order as usize - 1);
        if decoded_len > remaining_bits {
            decoded_len = remaining_bits;
        }

        let _ = self.decoder.decode_soft(
            encoded,
            &mut self.decode_buffer[self.write_index..],
        );
        self.write_index += decoded_len / 8;
    }

    fn receive(&mut self, decoded: &mut [u8]) {
        let remaining_buffer = self.write_index - self.read_index;
        let remaining_bits = remaining_buffer * 8;

        let mut receive_bits = decoded.len() * 8;
        if receive_bits > remaining_bits {
            receive_bits = remaining_bits;
        }

        let receive_len: usize;
        if receive_bits % 8 == 0 {
            receive_len = receive_bits / 8;
        } else {
            receive_len = receive_bits / 8 + 1;
        }

        decoded[..receive_len].clone_from_slice(
            &self.decode_buffer[self.read_index..(self.read_index + receive_len)],
        );
        self.read_index += receive_len;
    }
}

#[no_mangle]
pub extern "C" fn create_viterbi27(num_decoded_bits: c_int) -> *mut Shim<2, 7> {
    let shim = Box::new(Shim::<2, 7>::new(num_decoded_bits as usize, 2, 7, &[0o155, 0o117]));
    Box::into_raw(shim)
}

#[no_mangle]
pub extern "C" fn delete_viterbi27(shim_ptr: *mut Shim<2, 7>) {
    unsafe {
        drop(Box::from_raw(shim_ptr));
    }
}

#[no_mangle]
pub extern "C" fn init_viterbi27(shim_ptr: *mut Shim<2, 7>, _: c_int) -> c_int {
    let shim: &mut Shim<2, 7>;
    unsafe {
        shim = &mut *shim_ptr;
    }
    shim.init();
    0
}

#[no_mangle]
pub extern "C" fn update_viterbi27_blk(
    shim_ptr: *mut Shim<2, 7>,
    encoded_ptr: *const u8,
    num_groups: c_int,
) -> c_int {
    let shim: &mut Shim<2, 7>;
    let encoded: &[u8];
    unsafe {
        shim = &mut *shim_ptr;
        encoded = slice::from_raw_parts(
            encoded_ptr,
            num_groups as usize * shim.rate as usize + shim.order as usize - 1,
        );
    }
    shim.decode(encoded);
    0
}

#[no_mangle]
pub extern "C" fn chainback_viterbi27(
    shim_ptr: *mut Shim<2, 7>,
    decoded_ptr: *mut u8,
    num_bits: c_uint,
    _: c_int,
) -> c_int {
    let shim: &mut Shim<2, 7>;
    let decoded: &mut [u8];
    unsafe {
        shim = &mut *shim_ptr;
        decoded = slice::from_raw_parts_mut(decoded_ptr, num_bits as usize);
    }
    shim.receive(decoded);
    0
}

#[no_mangle]
pub extern "C" fn create_viterbi29(num_decoded_bits: c_int) -> *mut Shim<2, 9> {
    let shim = Box::new(Shim::<2, 9>::new(num_decoded_bits as usize, 2, 9, &[0o657, 0o435]));
    Box::into_raw(shim)
}

#[no_mangle]
pub extern "C" fn delete_viterbi29(shim_ptr: *mut Shim<2, 9>) {
    unsafe {
        drop(Box::from_raw(shim_ptr));
    }
}

#[no_mangle]
pub extern "C" fn init_viterbi29(shim_ptr: *mut Shim<2, 9>, _: c_int) -> c_int {
    let shim: &mut Shim<2, 9>;
    unsafe {
        shim = &mut *shim_ptr;
    }
    shim.init();
    0
}

#[no_mangle]
pub extern "C" fn update_viterbi29_blk(
    shim_ptr: *mut Shim<2, 9>,
    encoded_ptr: *const u8,
    num_groups: c_int,
) -> c_int {
    let shim: &mut Shim<2, 9>;
    let encoded: &[u8];
    unsafe {
        shim = &mut *shim_ptr;
        encoded = slice::from_raw_parts(encoded_ptr, num_groups as usize * shim.rate as usize);
    }
    shim.decode(encoded);
    0
}

#[no_mangle]
pub extern "C" fn chainback_viterbi29(
    shim_ptr: *mut Shim<2, 9>,
    decoded_ptr: *mut u8,
    num_bits: c_uint,
    _: c_int,
) -> c_int {
    let shim: &mut Shim<2, 9>;
    let decoded: &mut [u8];
    unsafe {
        shim = &mut *shim_ptr;
        decoded = slice::from_raw_parts_mut(decoded_ptr, num_bits as usize);
    }
    shim.receive(decoded);
    0
}

#[no_mangle]
pub extern "C" fn create_viterbi39(num_decoded_bits: c_int) -> *mut Shim<3, 9> {
    let shim = Box::new(Shim::<3, 9>::new(
        num_decoded_bits as usize,
        3,
        9,
        &[0o755, 0o633, 0o447],
    ));
    Box::into_raw(shim)
}

#[no_mangle]
pub extern "C" fn delete_viterbi39(shim_ptr: *mut Shim<3, 9>) {
    unsafe {
        drop(Box::from_raw(shim_ptr));
    }
}

#[no_mangle]
pub extern "C" fn init_viterbi39(shim_ptr: *mut Shim<3, 9>, _: c_int) -> c_int {
    let shim: &mut Shim<3, 9>;
    unsafe {
        shim = &mut *shim_ptr;
    }
    shim.init();
    0
}

#[no_mangle]
pub extern "C" fn update_viterbi39_blk(
    shim_ptr: *mut Shim<3, 9>,
    encoded_ptr: *const u8,
    num_groups: c_int,
) -> c_int {
    let shim: &mut Shim<3, 9>;
    let encoded: &[u8];
    unsafe {
        shim = &mut *shim_ptr;
        encoded = slice::from_raw_parts(encoded_ptr, num_groups as usize * shim.rate as usize);
    }
    shim.decode(encoded);
    0
}

#[no_mangle]
pub extern "C" fn chainback_viterbi39(
    shim_ptr: *mut Shim<3, 9>,
    decoded_ptr: *mut u8,
    num_bits: c_uint,
    _: c_int,
) -> c_int {
    let shim: &mut Shim<3, 9>;
    let decoded: &mut [u8];
    unsafe {
        shim = &mut *shim_ptr;
        decoded = slice::from_raw_parts_mut(decoded_ptr, num_bits as usize);
    }
    shim.receive(decoded);
    0
}

#[no_mangle]
pub extern "C" fn create_viterbi615(num_decoded_bits: c_int) -> *mut Shim<6, 15> {
    let shim = Box::new(Shim::<6, 15>::new(
        num_decoded_bits as usize,
        6,
        15,
        &[0o42631, 0o47245, 0o56507, 0o73363, 0o77267, 0o64537],
    ));
    Box::into_raw(shim)
}

#[no_mangle]
pub extern "C" fn delete_viterbi615(shim_ptr: *mut Shim<6, 15>) {
    unsafe {
        drop(Box::from_raw(shim_ptr));
    }
}

#[no_mangle]
pub extern "C" fn init_viterbi615(shim_ptr: *mut Shim<6, 15>, _: c_int) -> c_int {
    let shim: &mut Shim<6, 15>;
    unsafe {
        shim = &mut *shim_ptr;
    }
    shim.init();
    0
}

#[no_mangle]
pub extern "C" fn update_viterbi615_blk(
    shim_ptr: *mut Shim<6, 15>,
    encoded_ptr: *const u8,
    num_groups: c_int,
) -> c_int {
    let shim: &mut Shim<6, 15>;
    let encoded: &[u8];
    unsafe {
        shim = &mut *shim_ptr;
        encoded = slice::from_raw_parts(encoded_ptr, num_groups as usize * shim.rate as usize);
    }
    shim.decode(encoded);
    0
}

#[no_mangle]
pub extern "C" fn chainback_viterbi615(
    shim_ptr: *mut Shim<6, 15>,
    decoded_ptr: *mut u8,
    num_bits: c_uint,
    _: c_int,
) -> c_int {
    let shim: &mut Shim<6, 15>;
    let decoded: &mut [u8];
    unsafe {
        shim = &mut *shim_ptr;
        decoded = slice::from_raw_parts_mut(decoded_ptr, num_bits as usize);
    }
    shim.receive(decoded);
    0
}

#[repr(C)]
pub struct RsShim {
    enc: RsEncoder,
    dec: RsDecoder,
    msg_length: usize,
    block_length: usize,
    num_roots: usize,
    pad: usize,
    msg_out: Vec<u8>,
    erasures: Vec<u8>,
}

#[no_mangle]
pub extern "C" fn init_rs_char(
    symbol_size: c_int,
    primitive_polynomial: c_int,
    first_consecutive_root: c_int,
    root_gap: c_int,
    number_roots: c_int,
    pad: c_uint,
) -> *mut RsShim {
    if symbol_size != 8 {
        return std::ptr::null_mut();
    }

    let pad = pad as usize;
    let block_length = 255 - pad;
    let num_roots = number_roots as usize;
    let msg_length = block_length - num_roots;
    let enc = RsEncoder::new(
        primitive_polynomial as u16,
        first_consecutive_root as u8,
        root_gap as u8,
        num_roots,
    );
    let dec = RsDecoder::new(
        primitive_polynomial as u16,
        first_consecutive_root as u8,
        root_gap as u8,
        num_roots,
    );

    let shim = Box::new(RsShim {
        enc,
        dec,
        msg_length,
        block_length,
        num_roots,
        pad,
        msg_out: vec![0u8; block_length],
        erasures: vec![0u8; num_roots],
    });
    Box::into_raw(shim)
}

#[no_mangle]
pub extern "C" fn free_rs_char(rs_ptr: *mut RsShim) {
    unsafe {
        drop(Box::from_raw(rs_ptr));
    }
}

#[no_mangle]
pub extern "C" fn encode_rs_char(
    rs_ptr: *mut RsShim,
    msg_ptr: *const u8,
    parity_ptr: *mut u8,
) {
    let shim: &mut RsShim;
    let msg: &[u8];
    let parity: &mut [u8];
    unsafe {
        shim = &mut *rs_ptr;
        msg = slice::from_raw_parts(msg_ptr, shim.msg_length);
        parity = slice::from_raw_parts_mut(parity_ptr, shim.num_roots);
    }

    let _ = shim.enc.encode(msg, &mut shim.msg_out);
    parity.copy_from_slice(&shim.msg_out[shim.msg_length..(shim.msg_length + shim.num_roots)]);
}

#[no_mangle]
pub extern "C" fn decode_rs_char(
    rs_ptr: *mut RsShim,
    block_ptr: *mut u8,
    erasure_locations_ptr: *const c_int,
    num_erasures: c_int,
) -> c_int {
    let shim: &mut RsShim;
    unsafe {
        shim = &mut *rs_ptr;
    }
    let num_erasures = num_erasures as usize;

    unsafe {
        let erasure_locations = slice::from_raw_parts(erasure_locations_ptr, num_erasures);
        for i in 0..num_erasures {
            shim.erasures[i] = (erasure_locations[i] as u8).wrapping_sub(shim.pad as u8);
        }
    }

    let result = unsafe {
        let input = slice::from_raw_parts(block_ptr, shim.block_length);
        let output = slice::from_raw_parts_mut(block_ptr, shim.block_length);
        let erasures = &shim.erasures[..num_erasures];
        shim.dec.decode_with_erasures(input, erasures, output)
    };

    match result {
        Ok(corrected) => corrected as c_int,
        Err(_) => -1,
    }
}

fn ccsds_rs8_enc() -> &'static Mutex<RsEncoder> {
    static ENC: OnceLock<Mutex<RsEncoder>> = OnceLock::new();
    ENC.get_or_init(|| Mutex::new(RsEncoder::new_ccsds()))
}

fn ccsds_rs8_dec() -> &'static Mutex<RsDecoder> {
    static DEC: OnceLock<Mutex<RsDecoder>> = OnceLock::new();
    DEC.get_or_init(|| Mutex::new(RsDecoder::new_ccsds()))
}

#[no_mangle]
pub extern "C" fn encode_rs_8(data_ptr: *const u8, parity_ptr: *mut u8, pad: c_int) {
    let pad = pad as usize;
    // effective message length is shortened by pad (full is 223)
    let msg_length = 223 - pad;
    let mut rs = ccsds_rs8_enc().lock().unwrap();

    let data: &[u8];
    let parity: &mut [u8];
    unsafe {
        data = slice::from_raw_parts(data_ptr, msg_length);
        parity = slice::from_raw_parts_mut(parity_ptr, 32);
    }

    // encode into a local block, then hand back the 32-byte parity tail
    let mut block = vec![0u8; 255];
    let _ = rs.encode(data, &mut block);
    parity.copy_from_slice(&block[msg_length..(msg_length + 32)]);
}

#[no_mangle]
pub extern "C" fn decode_rs_8(
    data_ptr: *mut u8,
    erasure_locations_ptr: *const c_int,
    num_erasures: c_int,
    pad: c_int,
) -> c_int {
    if pad < 0 || pad > 222 {
        return -1;
    }
    let pad = pad as usize;
    let num_erasures = num_erasures as usize;
    let block_length = 255 - pad;
    let mut rs = ccsds_rs8_dec().lock().unwrap();

    let mut erasures = vec![0u8; num_erasures];
    if num_erasures > 0 {
        unsafe {
            let locs = slice::from_raw_parts(erasure_locations_ptr, num_erasures);
            for i in 0..num_erasures {
                erasures[i] = (locs[i] as u8).wrapping_sub(pad as u8);
            }
        }
    }

    let result = unsafe {
        let input = slice::from_raw_parts(data_ptr, block_length);
        let output = slice::from_raw_parts_mut(data_ptr, block_length);
        rs.decode_with_erasures(input, &erasures, output)
    };

    match result {
        Ok(corrected) => corrected as c_int,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn encode_rs_ccsds(data_ptr: *const u8, parity_ptr: *mut u8, pad: c_int) {
    let pad_u = pad as usize;
    let msg_length = 223 - pad_u;

    let data: &[u8];
    let parity: &mut [u8];
    unsafe {
        data = slice::from_raw_parts(data_ptr, msg_length);
        parity = slice::from_raw_parts_mut(parity_ptr, 32);
    }

    // transform the dual-basis message to conventional, encode, then transform
    // the conventional parity back to dual
    let conv_msg: Vec<u8> = data.iter().map(|&b| dual_to_conv(b)).collect();
    let mut conv_parity = vec![0u8; 32];
    encode_rs_8(conv_msg.as_ptr(), conv_parity.as_mut_ptr(), pad);
    for (out, &c) in parity.iter_mut().zip(conv_parity.iter()) {
        *out = conv_to_dual(c);
    }
}

#[no_mangle]
pub extern "C" fn decode_rs_ccsds(
    data_ptr: *mut u8,
    erasure_locations_ptr: *const c_int,
    num_erasures: c_int,
    pad: c_int,
) -> c_int {
    if pad < 0 || pad > 222 {
        return -1;
    }
    let pad_u = pad as usize;
    let block_length = 255 - pad_u;

    let block: &mut [u8] = unsafe { slice::from_raw_parts_mut(data_ptr, block_length) };

    // transform the whole dual-basis block to conventional in place
    for b in block.iter_mut() {
        *b = dual_to_conv(*b);
    }

    let rc = decode_rs_8(data_ptr, erasure_locations_ptr, num_erasures, pad);

    // transform back to dual basis
    for b in block.iter_mut() {
        *b = conv_to_dual(*b);
    }

    rc
}
