use std::slice;
use std::sync::{Mutex, OnceLock};
use libc::{c_int, c_uint};

use fec::reed_solomon::{RsDecoder, RsEncoder};
use fec::reed_solomon::ccsds::{conv_to_dual, dual_to_conv};
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
