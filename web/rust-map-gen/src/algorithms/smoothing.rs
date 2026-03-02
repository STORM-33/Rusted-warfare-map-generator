use crate::state::Matrix;

use crate::tiles::{LARGE_ROCK_TILE_SET, TILE_ID_OFFSET};



pub(crate) fn smooth_wall_tiles(wall_matrix: &Matrix) -> Matrix {
    let h = wall_matrix.rows;
    let w = wall_matrix.cols;
    let mut cleaned = wall_matrix.clone();
    let mut result = Matrix::zeros(h, w);
    let rock_first_gid = TILE_ID_OFFSET + 135;

    for r in 0..h {
        for c in 0..w {
            if cleaned.get(r, c) != 1 {
                continue;
            }
            let pattern = wall_cardinal_passable_pattern(&cleaned, r, c, true);
            if is_wall_isolated_pattern(pattern) {
                cleaned.set(r, c, 0);
            }
        }
    }

    for r in 0..h {
        for c in 0..w {
            if cleaned.get(r, c) != 1 {
                continue;
            }
            let pattern = wall_cardinal_passable_pattern(&cleaned, r, c, false);
            let local_id = if let Some(idx) = wall_cardinal_map(pattern) {
                if idx == 0 {
                    let tl = wall_passable(&cleaned, r as isize - 1, c as isize - 1, false);
                    let tr = wall_passable(&cleaned, r as isize - 1, c as isize + 1, false);
                    let bl = wall_passable(&cleaned, r as isize + 1, c as isize - 1, false);
                    let br = wall_passable(&cleaned, r as isize + 1, c as isize + 1, false);
                    if tl == 1 && tr == 0 && bl == 0 && br == 0 {
                        LARGE_ROCK_TILE_SET[10]
                    } else if tr == 1 && tl == 0 && bl == 0 && br == 0 {
                        LARGE_ROCK_TILE_SET[11]
                    } else if bl == 1 && tl == 0 && tr == 0 && br == 0 {
                        LARGE_ROCK_TILE_SET[12]
                    } else if br == 1 && tl == 0 && tr == 0 && bl == 0 {
                        LARGE_ROCK_TILE_SET[13]
                    } else {
                        continue;
                    }
                } else {
                    LARGE_ROCK_TILE_SET[idx as usize + 1]
                }
            } else {
                LARGE_ROCK_TILE_SET[1]
            };
            if local_id >= 0 {
                result.set(r, c, rock_first_gid + local_id);
            }
        }
    }
    result
}



pub(crate) fn smooth_terrain_tiles(
    map_matrix: &mut Matrix,
    id_matrix: &mut Matrix,
    height_level: i32,
    tile_set: &[i32; 14],
) {
    remove_isolated_tiles(map_matrix, height_level, 3);
    let rows = map_matrix.rows;
    let cols = map_matrix.cols;
    for r in 0..rows {
        for c in 0..cols {
            if map_matrix.get(r, c) != height_level {
                continue;
            }
            let neighbors = get_all_neighbors(map_matrix, r, c);
            if is_diagonal_isolated(neighbors) {
                map_matrix.set(r, c, height_level - 1);
            }
        }
    }
    remove_isolated_tiles(map_matrix, height_level, 5);

    for r in 0..rows {
        for c in 0..cols {
            if map_matrix.get(r, c) != height_level {
                continue;
            }
            let neighbors = get_neighbors(map_matrix, r, c);
            let idx = cardinal_tile_map(neighbors);
            id_matrix.set(r, c, tile_set[(idx + 1) as usize]);
        }
    }

    for r in 0..rows {
        for c in 0..cols {
            if map_matrix.get(r, c) != height_level {
                continue;
            }
            let neighbors = get_all_neighbors(map_matrix, r, c);
            if neighbors == [1, 0, 0, 0, 0, 0, 0, 0, 0] {
                id_matrix.set(r, c, tile_set[13]);
            } else if neighbors == [0, 0, 1, 0, 0, 0, 0, 0, 0] {
                id_matrix.set(r, c, tile_set[12]);
            } else if neighbors == [0, 0, 0, 0, 0, 0, 1, 0, 0] {
                id_matrix.set(r, c, tile_set[11]);
            } else if neighbors == [0, 0, 0, 0, 0, 0, 0, 0, 1] {
                id_matrix.set(r, c, tile_set[10]);
            } else if neighbors == [0, 0, 1, 0, 0, 1, 1, 0, 0] {
                id_matrix.set(r, c, tile_set[9]);
            } else if neighbors == [0, 0, 1, 1, 0, 0, 1, 0, 0] {
                id_matrix.set(r, c, tile_set[2]);
            } else if neighbors == [1, 0, 0, 0, 0, 1, 0, 0, 1] {
                id_matrix.set(r, c, tile_set[4]);
            } else if neighbors == [1, 0, 0, 1, 0, 0, 0, 0, 1] {
                id_matrix.set(r, c, tile_set[7]);
            } else if neighbors == [1, 0, 0, 0, 0, 0, 0, 1, 1] {
                id_matrix.set(r, c, tile_set[7]);
            } else if neighbors == [1, 1, 0, 0, 0, 0, 0, 0, 1] {
                id_matrix.set(r, c, tile_set[4]);
            } else if neighbors == [0, 0, 1, 0, 0, 0, 1, 1, 0] {
                id_matrix.set(r, c, tile_set[9]);
            } else if neighbors == [0, 1, 1, 0, 0, 0, 1, 0, 0] {
                id_matrix.set(r, c, tile_set[2]);
            }
        }
    }
}



fn remove_isolated_tiles(map_matrix: &mut Matrix, height_level: i32, passes: usize) {
    for _ in 0..passes {
        for r in 0..map_matrix.rows {
            for c in 0..map_matrix.cols {
                if map_matrix.get(r, c) != height_level {
                    continue;
                }
                let pattern = get_neighbors(map_matrix, r, c);
                if is_isolated_pattern(pattern) {
                    map_matrix.set(r, c, height_level - 1);
                }
            }
        }
    }
}



fn get_neighbors(matrix: &Matrix, r: usize, c: usize) -> [i32; 4] {
    let neighbors = [
        (r as isize - 1, c as isize),
        (r as isize, c as isize + 1),
        (r as isize + 1, c as isize),
        (r as isize, c as isize - 1),
    ];
    let mut out = [0_i32; 4];
    for (i, (nr, nc)) in neighbors.into_iter().enumerate() {
        if nr < 0 || nc < 0 || nr >= matrix.rows as isize || nc >= matrix.cols as isize {
            out[i] = 0;
        } else {
            out[i] = i32::from(matrix.get(nr as usize, nc as usize) < matrix.get(r, c));
        }
    }
    out
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



fn wall_passable(matrix: &Matrix, r: isize, c: isize, default: bool) -> i32 {
    if r < 0 || c < 0 || r >= matrix.rows as isize || c >= matrix.cols as isize {
        return i32::from(default);
    }
    i32::from(matrix.get(r as usize, c as usize) == 0)
}



fn wall_cardinal_passable_pattern(
    matrix: &Matrix,
    r: usize,
    c: usize,
    oob_default: bool,
) -> [i32; 4] {
    [
        wall_passable(matrix, r as isize - 1, c as isize, oob_default),
        wall_passable(matrix, r as isize, c as isize + 1, oob_default),
        wall_passable(matrix, r as isize + 1, c as isize, oob_default),
        wall_passable(matrix, r as isize, c as isize - 1, oob_default),
    ]
}



fn wall_cardinal_map(pattern: [i32; 4]) -> Option<i32> {
    match pattern {
        [0, 0, 0, 0] => Some(0),
        [1, 0, 0, 0] => Some(2),
        [0, 1, 0, 0] => Some(5),
        [1, 1, 0, 0] => Some(3),
        [0, 0, 1, 0] => Some(7),
        [0, 1, 1, 0] => Some(8),
        [0, 0, 0, 1] => Some(4),
        [1, 0, 0, 1] => Some(1),
        [0, 0, 1, 1] => Some(6),
        _ => None,
    }
}



fn is_wall_isolated_pattern(pattern: [i32; 4]) -> bool {
    matches!(
        pattern,
        [1, 0, 1, 0]
            | [1, 1, 1, 0]
            | [0, 1, 0, 1]
            | [1, 1, 0, 1]
            | [1, 0, 1, 1]
            | [0, 1, 1, 1]
            | [1, 1, 1, 1]
    )
}



fn is_diagonal_isolated(pattern: [i32; 9]) -> bool {
    matches!(
        pattern,
        [1, 0, 0, 0, 0, 0, 0, 0, 1]
            | [0, 0, 1, 0, 0, 0, 1, 0, 0]
            | [0, 1, 0, 1, 0, 0, 0, 0, 0]
            | [0, 1, 0, 0, 0, 1, 0, 0, 0]
            | [0, 0, 0, 1, 0, 0, 0, 1, 0]
            | [0, 0, 0, 0, 0, 1, 0, 1, 0]
    )
}



fn is_isolated_pattern(pattern: [i32; 4]) -> bool {
    matches!(
        pattern,
        [1, 0, 1, 0]
            | [1, 1, 1, 0]
            | [0, 1, 0, 1]
            | [1, 1, 0, 1]
            | [1, 0, 1, 1]
            | [0, 1, 1, 1]
            | [1, 1, 1, 1]
    )
}



fn cardinal_tile_map(pattern: [i32; 4]) -> i32 {
    match pattern {
        [0, 0, 0, 0] => 0,
        [1, 0, 0, 0] => 2,
        [0, 1, 0, 0] => 5,
        [1, 1, 0, 0] => 3,
        [0, 0, 1, 0] => 7,
        [1, 0, 1, 0] => -1,
        [0, 1, 1, 0] => 8,
        [1, 1, 1, 0] => -1,
        [0, 0, 0, 1] => 4,
        [1, 0, 0, 1] => 1,
        [0, 1, 0, 1] => -1,
        [1, 1, 0, 1] => -1,
        [0, 0, 1, 1] => 6,
        [1, 0, 1, 1] => -1,
        [0, 1, 1, 1] => -1,
        [1, 1, 1, 1] => -1,
        _ => 0,
    }
}
