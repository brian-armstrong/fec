use std::collections::HashMap;
use std::simd::{u8x16, u8x32, u8x64};

#[derive(Clone, Debug)]
pub struct DistanceShuffle {
    pub shuffle8: Vec<u8x16>,
    pub shuffle16: Vec<u8x32>,
    pub shuffle32: Vec<u8x64>,
}

impl DistanceShuffle {
    pub fn new(rate: u32, order: u32, poly_table: &[u8]) -> Self {
        // for each state, look up the concatenated output from the poly_table.
        // the total distance of that output is in i16, so we have to shuffle
        // two bytes into it. but our shuffle is at the 8-bit level (epi8), 
        // so the shuffle mask is 2 entries long
        let shuffle8: Vec<u8x16> = if rate <= 3 {
            let num_groups = 1usize << (order - 3);
            (0..num_groups)
                .map(|g| {
                    let mut bytes = [0u8; 16];
                    for k in 0..8 {
                        let sel = poly_table[g * 8 + k] as usize;
                        bytes[2 * k] = (2 * sel) as u8;
                        bytes[2 * k + 1] = (2 * sel + 1) as u8;
                    }
                    u8x16::from_array(bytes)
                })
                .collect()
        } else {
            Vec::new()
        };

        // the 16-wide version is nearly the same as the 8-wide version, but
        // there are lane shuffle restrictions on AVX2, so we will target
        // the nearest lane for each shuffle. we will duplicate the distances
        // so that this will work
        let shuffle16: Vec<u8x32> = if order >= 4 && rate <= 3 {
            let num_groups = 1usize << (order - 4);
            (0..num_groups)
                .map(|g| {
                    let mut bytes = [0u8; 32];
                    for k in 0..16 {
                        let sel = poly_table[g * 16 + k] as usize;
                        let half = (k / 8) * 16;
                        bytes[2 * k] = (2 * sel + half) as u8;
                        bytes[2 * k + 1] = (2 * sel + 1 + half) as u8;
                    }
                    u8x32::from_array(bytes)
                })
                .collect()
        } else {
            Vec::new()
        };

        let shuffle32: Vec<u8x64> = if order >= 5 && rate <= 3 {
            let num_groups = 1usize << (order - 5);
            (0..num_groups)
                .map(|g| {
                    let mut bytes = [0u8; 64];
                    for k in 0..32 {
                        let sel = poly_table[g * 32 + k] as usize;
                        let quarter = (k / 8) * 16;
                        bytes[2 * k] = (2 * sel + quarter) as u8;
                        bytes[2 * k + 1] = (2 * sel + 1 + quarter) as u8;
                    }
                    u8x64::from_array(bytes)
                })
                .collect()
        } else {
            Vec::new()
        };

        DistanceShuffle {
            shuffle8,
            shuffle16,
            shuffle32,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OctLookup {
    pub keys: Vec<u16>,
    outputs: Vec<u8>,
    pub distances: Vec<u16>,
    pub keys16: Vec<u16>,
    outputs16: Vec<u8>,
    pub distances16: Vec<u16>,
}

impl OctLookup {
    pub fn new(rate: u32, order: u32, poly_table: &[u8]) -> Self {
        let (keys, outputs, distances) = OctLookup::fill_keys(rate, order, 8, poly_table);
        let (keys16, outputs16, distances16) = OctLookup::fill_keys(rate, order, 16, poly_table);

        OctLookup {
            keys,
            outputs,
            distances,
            keys16,
            outputs16,
            distances16,
        }
    }

    fn fill_keys(rate: u32, order: u32, group_width: usize, poly_table: &[u8]) -> (Vec<u16>, Vec<u8>, Vec<u16>) {
        let num_groups = (1usize << order) / group_width;
        let mut keys = vec![0u16; num_groups];

        let mut outputs: Vec<u8> = vec![];
        let mut dedup: HashMap<u128, u16> = HashMap::new();

        for (g, key) in keys.iter_mut().enumerate() {
            let base = g * group_width;

            let mut concat: u128 = 0;
            for i in (0..group_width).rev() {
                concat <<= rate;
                concat |= poly_table[base + i] as u128;
            }

            *key = match dedup.get(&concat) {
                Some(&existing) => existing,
                None => {
                    let offset = outputs.len() as u16;
                    for i in 0..group_width {
                        outputs.push(poly_table[base + i]);
                    }
                    dedup.insert(concat, offset);
                    offset
                }
            };
        }

        let distances = vec![0u16; outputs.len()];

        (keys, outputs, distances)
    }

    pub fn fill_distances(&mut self, distances: &[u16]) {
        for (d, &output) in self.distances.iter_mut().zip(self.outputs.iter()) {
            *d = distances[output as usize];
        }
    }

    pub fn fill_distances16(&mut self, distances: &[u16]) {
        for (d, &output) in self.distances16.iter_mut().zip(self.outputs16.iter()) {
            *d = distances[output as usize];
        }
    }
}
