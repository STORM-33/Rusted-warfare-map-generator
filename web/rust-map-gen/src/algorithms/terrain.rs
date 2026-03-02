use std::collections::VecDeque;



use crate::state::Matrix;



const WALL_INFLUENCE_RADIUS: f64 = 12.0;



pub(crate) fn generate_level(
    map_matrix: &Matrix,
    perlin_noise: &[f64],
    level_type: &str,
    level: i32,
    min_perlin_value: f64,
    min_distance_to_prev_level: i32,
    min_distance_to_next_level: i32,
) -> Matrix {
    let mut new_map = map_matrix.clone();
    let rows = map_matrix.rows;
    let cols = map_matrix.cols;

    let mut forbidden_mask = vec![false; rows * cols];
    let mut has_forbidden = false;

    for r in 0..rows {
        for c in 0..cols {
            let current = map_matrix.get(r, c);
            let forbidden = if level_type == "height" {
                current == (level - 2)
            } else {
                current == (level + 2)
            };
            forbidden_mask[idx(r, c, cols)] = forbidden;
            has_forbidden |= forbidden;
        }
    }

    let dist = if has_forbidden {
        distance_to_true(&forbidden_mask, rows, cols)
    } else {
        vec![i32::MAX / 4; rows * cols]
    };
    let min_distance = if level_type == "height" {
        min_distance_to_prev_level
    } else {
        min_distance_to_next_level
    };

    for r in 0..rows {
        for c in 0..cols {
            let current = map_matrix.get(r, c);
            let candidate = if level_type == "height" {
                current == (level - 1)
            } else {
                current == (level + 1)
            };
            if !candidate {
                continue;
            }
            let p = perlin_noise[idx(r, c, cols)];
            if p < min_perlin_value {
                continue;
            }
            if dist[idx(r, c, cols)] <= min_distance {
                continue;
            }
            new_map.set(r, c, level);
        }
    }
    new_map
}



pub(crate) fn bias_terrain_near_walls(
    height_map: &Matrix,
    wall_matrix: &Matrix,
    num_height_levels: i32,
) -> Matrix {
    let rows = height_map.rows;
    let cols = height_map.cols;
    let wall_mask: Vec<bool> = wall_matrix.data.iter().map(|v| *v == 1).collect();
    let dist = distance_to_true(&wall_mask, rows, cols);
    let max_target = num_height_levels.min(5).max(1);
    let mut result = height_map.clone();
    for r in 0..rows {
        for c in 0..cols {
            let current = height_map.get(r, c);
            if current < 1 {
                continue;
            }
            if current >= max_target {
                result.set(r, c, current);
                continue;
            }
            let influence =
                (1.0 - dist[idx(r, c, cols)] as f64 / WALL_INFLUENCE_RADIUS).clamp(0.0, 1.0);
            let boosted =
                (current as f64 + influence * (max_target as f64 - current as f64)).round() as i32;
            result.set(r, c, boosted.clamp(current, max_target));
        }
    }
    result
}



fn distance_to_true(mask: &[bool], rows: usize, cols: usize) -> Vec<i32> {
    let mut dist = vec![i32::MAX / 4; rows * cols];
    let mut queue = VecDeque::new();
    for r in 0..rows {
        for c in 0..cols {
            if mask[idx(r, c, cols)] {
                dist[idx(r, c, cols)] = 0;
                queue.push_back((r, c));
            }
        }
    }
    if queue.is_empty() {
        return dist;
    }
    while let Some((r, c)) = queue.pop_front() {
        let current_dist = dist[idx(r, c, cols)];
        for dr in -1..=1 {
            for dc in -1..=1 {
                if dr == 0 && dc == 0 {
                    continue;
                }
                let nr = r as isize + dr;
                let nc = c as isize + dc;
                if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                    continue;
                }
                let nr = nr as usize;
                let nc = nc as usize;
                let ni = idx(nr, nc, cols);
                if current_dist + 1 < dist[ni] {
                    dist[ni] = current_dist + 1;
                    queue.push_back((nr, nc));
                }
            }
        }
    }
    dist
}



fn idx(r: usize, c: usize, cols: usize) -> usize {
    r * cols + c
}
