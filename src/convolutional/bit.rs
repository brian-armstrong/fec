use std::cmp;

#[derive(Debug)]
pub(crate) struct BitWriter<'a> {
    buf: &'a mut [u8],
    byte_index: usize,
    current_byte_len: usize,
    current_byte: u8,
}

impl<'a> BitWriter<'a> {
    pub fn new(buf: &'a mut [u8]) -> BitWriter<'a> {
        BitWriter {
            buf,
            byte_index: 0,
            current_byte_len: 0,
            current_byte: 0,
        }
    }

    pub fn write(&mut self, mut val: u8, nbits: usize) {
        for _i in 0..nbits {
            self.current_byte |= val & 1;
            self.current_byte_len += 1;

            if self.current_byte_len == 8 {
                self.buf[self.byte_index] = self.current_byte;
                self.byte_index += 1;
                self.current_byte_len = 0;
                self.current_byte = 0;
            } else {
                self.current_byte <<= 1;
            }

            val >>= 1;
        }
    }

    pub fn write_iter<'b, T>(&mut self, bits: T)
    where
        T: IntoIterator<Item = &'b u8>,
        T::IntoIter: ExactSizeIterator,
    {
        let mut bits = bits.into_iter();
        let mut b: u16 = self.current_byte as u16;
        let mut len = bits.len();

        let first_byte_len = cmp::min(8 - self.current_byte_len, len);

        for bit in bits.by_ref().take(first_byte_len) {
            b |= *bit as u16;
            b <<= 1;
        }

        len -= first_byte_len;

        let buf = &mut self.buf;
        let mut byte_index = self.byte_index;

        if self.current_byte_len + first_byte_len == 8 {
            b >>= 1;
            buf[byte_index] = b as u8;
            byte_index += 1;
        } else {
            self.current_byte = b as u8;
            self.current_byte_len += first_byte_len;
            return;
        }

        let num_full_bytes = len / 8;

        for _i in 0..num_full_bytes {
            let mut byte = bits.next().unwrap() << 7;
            byte |= bits.next().unwrap() << 6;
            byte |= bits.next().unwrap() << 5;
            byte |= bits.next().unwrap() << 4;
            byte |= bits.next().unwrap() << 3;
            byte |= bits.next().unwrap() << 2;
            byte |= bits.next().unwrap() << 1;
            byte |= bits.next().unwrap();
            buf[byte_index] = byte;
            byte_index += 1;
        }

        len -= 8 * num_full_bytes;

        b = 0;
        for bit in bits {
            b |= *bit as u16;
            b <<= 1;
        }

        self.current_byte = b as u8;
        self.byte_index = byte_index;
        self.current_byte_len = len;
    }

    pub fn flush(&mut self) {
        if self.current_byte_len != 0 {
            self.current_byte <<= 8 - self.current_byte_len - 1;
            self.buf[self.byte_index] = self.current_byte;
            self.byte_index += 1;
            self.current_byte_len = 0;
        }
    }

    pub fn len(&self) -> usize {
        self.byte_index
    }
}

const fn reverse_byte(b: u8) -> u8 {
    (b & 0x80) >> 7
        | (b & 0x40) >> 5
        | (b & 0x20) >> 3
        | (b & 0x10) >> 1
        | (b & 0x08) << 1
        | (b & 0x04) << 3
        | (b & 0x02) << 5
        | (b & 0x01) << 7
}

const REVERSE_TABLE: [u8; 256] = {
    let mut table: [u8; 256] = [0; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = reverse_byte(i as u8);
        i += 1;
    }
    table
};

#[derive(Debug)]
pub(crate) struct BitReader<'a> {
    buf: &'a [u8],
    byte_index: usize,
    current_byte_len: usize,
    current_byte: u8,
}

impl<'a> BitReader<'a> {
    pub fn new(buf: &'a [u8]) -> BitReader<'a> {
        BitReader {
            buf,
            byte_index: 0,
            current_byte_len: 8,
            current_byte: buf[0],
        }
    }

    pub fn read(&mut self, mut nbits: usize) -> u8 {
        let mut byte: u8 = 0;
        let shift: usize = 8 - nbits;

        if self.current_byte_len < nbits {
            byte = self.current_byte & ((1 << self.current_byte_len) - 1);

            self.byte_index += 1;
            self.current_byte = self.buf[self.byte_index];

            nbits -= self.current_byte_len;
            self.current_byte_len = 8;

            byte <<= nbits;
        }

        let copy_mask_shift = self.current_byte_len - nbits;
        // this next specific needs to start as u16 so that at rate 8 (nbits == 8), we can fit it before the subtraction
        let copy_mask: u8 = (((1u16 << nbits) - 1) << copy_mask_shift) as u8;

        byte |= (self.current_byte & copy_mask) >> (self.current_byte_len - nbits);

        self.current_byte_len -= nbits;

        REVERSE_TABLE[byte as usize] >> shift
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.buf.len()
    }
}

#[cfg(test)]
mod tests {
    use super::{BitReader, BitWriter};

    #[test]
    fn round_trips_partial_trailing_byte() {
        for nbits in 1..=64usize {
            let pattern: Vec<u8> = (0..nbits).map(|i| ((i * 5 + 1) % 2) as u8).collect();

            let mut buf = vec![0u8; nbits.div_ceil(8)];
            {
                let mut w = BitWriter::new(&mut buf);
                for &b in &pattern {
                    w.write(b, 1);
                }
                w.flush();
            }

            let mut r = BitReader::new(&buf);
            let got: Vec<u8> = (0..nbits).map(|_| r.read(1)).collect();
            assert_eq!(got, pattern, "bit round trip failed at nbits={nbits}");
        }
    }

    #[test]
    fn partial_byte_is_left_aligned() {
        let mut buf = [0u8; 2];
        {
            let mut w = BitWriter::new(&mut buf);
            w.write(1, 1);
            w.write(0, 1);
            w.write(1, 1);
            w.flush();
        }
        // three bits 1,0,1 most significant first
        assert_eq!(buf[0], 0b1010_0000, "got {:08b}", buf[0]);
    }
}
