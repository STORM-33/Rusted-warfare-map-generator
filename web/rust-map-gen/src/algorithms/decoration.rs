use rand::Rng;



use crate::state::Matrix;



pub(crate) fn add_decoration_tiles(
    id_matrix: &mut Matrix,
    map_matrix: &Matrix,
    rng: &mut impl Rng,
    freq: f64,
    decoration_tiles: &[&[i32]],
) {
    for r in 0..map_matrix.rows {
        for c in 0..map_matrix.cols {
            let level = map_matrix.get(r, c);
            if level <= 0 {
                continue;
            }
            if get_all_neighbors(map_matrix, r, c) != [0, 0, 0, 0, 0, 0, 0, 0, 0] {
                continue;
            }
            if rng.gen::<f64>() >= freq {
                continue;
            }
            let idx = (level as usize).saturating_sub(1);
            if let Some(tiles) = decoration_tiles.get(idx) {
                if !tiles.is_empty() {
                    let pick = rng.gen_range(0..tiles.len());
                    id_matrix.set(r, c, tiles[pick]);
                }
            }
        }
    }
}



fn get_all_neighbors(matrix: &Matrix, r: usize, c: usize) -> [i32; 9] {
    let neighbors = [
        (r as isize - 1, c as isize - 1),
        (r as isize - 1, c as isize),
        (r as isize - 1, c as isize + 1),
        (r as isize, c as isize - 1),
        (r as isize, c as isize),
        (r as isize, c as isize + 1),
        (r as isize + 1, c as isize - 1),
        (r as isize + 1, c as isize),
        (r as isize + 1, c as isize + 1),
    ];
    let mut out = [0_i32; 9];
    for (i, (nr, nc)) in neighbors.into_iter().enumerate() {
        if nr < 0 || nc < 0 || nr >= matrix.rows as isize || nc >= matrix.cols as isize {
            out[i] = 0;
        } else {
            out[i] = i32::from(matrix.get(nr as usize, nc as usize) < matrix.get(r, c));
        }
    }
    out
}
