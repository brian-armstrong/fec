pub fn num_states_for_order(order: u32) -> u32 {
    1 << order
}

#[inline]
pub fn metric_distance(x: u32, y: u32) -> u32 {
    (x ^ y).count_ones()
}

pub fn conv_poly_table(rate: u32, order: u32, polys: &[u16]) -> Vec<u16> {
    assert_eq!(
        polys.len(),
        rate as usize,
        "expected {rate} generator polynomials for rate {rate}, got {}",
        polys.len()
    );
    let num_states = 1usize << order;
    (0..num_states)
        .map(|state| {
            let mut concat: u16 = 0;
            for (j, &poly) in polys.iter().take(rate as usize).enumerate() {
                let state_and_poly = state as u16 & poly;
                if state_and_poly.count_ones() % 2 == 1 {
                    concat |= 1 << j;
                }
            }
            concat
        })
        .collect()
}

pub(crate) const fn traceback_group_length(order: u32) -> usize {
    // running traceback more frequently is less efficient, but more compact.
    // for larger orders, it's better to use less cache with more frequent traceback
    let group_mult: u32 = match order {
        0..=8 => 100,
        9..=10 => 50,
        11..=12 => 15,
        _ => 10,
    };
    (group_mult * order) as usize
}
