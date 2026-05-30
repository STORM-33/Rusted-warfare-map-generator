use std::collections::VecDeque;

use crate::state::Matrix;

const MAX_TERRAIN_LEVEL: i32 = 7;
// Magnetism pull radius scales with map size: max(MIN, min(rows,cols) / DIVISOR).
const MAGNETISM_RADIUS_DIVISOR: f64 = 16.0;
const MAGNETISM_MIN_RADIUS: f64 = 6.0;

const CARDINAL_DIRS: [(isize, isize); 4] = [(-1, 0), (0, 1), (1, 0), (0, -1)];
const EIGHT_DIRS: [(isize, isize); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];
const DISTANCE_TWO_DIRS: [(isize, isize); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

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

/// Pull terrain height-level transitions toward wall lines.
///
/// Rather than flattening regions, this blends each land cell's Perlin level toward
/// the target level of its depth zone, weighted by proximity to the nearest wall
/// (depth boundary). Near a wall the pull is strong, so the step between two zones'
/// targets lands right on the wall; far from walls the weight fades to zero and the
/// original Perlin texture is preserved. `strength` (0..1) scales the whole effect;
/// 0 is a no-op.
pub(crate) fn magnetize_terrain(
    height_map: &Matrix,
    depth_matrix: &Matrix,
    strength: f64,
    num_height_levels: i32,
) -> Matrix {
    let rows = height_map.rows;
    let cols = height_map.cols;
    let mut result = height_map.clone();
    if strength <= 0.0 || rows == 0 || cols == 0 {
        return result;
    }

    let max_depth = depth_matrix.data.iter().copied().max().unwrap_or(0).min(9);
    if max_depth <= 0 {
        return result;
    }

    let max_target = num_height_levels.clamp(2, MAX_TERRAIN_LEVEL);

    // depth -> target land level. depth 0 (flat land) targets level 1.
    let base_level = 2;
    let available_range = max_target - base_level;
    let mut depth_to_level = [1_i32; 10];
    for d in 1..=max_depth {
        let ratio = (d - 1) as f64 / (max_depth as f64).max(1.0);
        let level = base_level + (ratio * available_range as f64).round() as i32;
        depth_to_level[d as usize] = level.clamp(base_level, max_target);
    }
    // Ensure nested rings step strictly upward.
    for d in 2..=max_depth as usize {
        if depth_to_level[d] <= depth_to_level[d - 1] {
            depth_to_level[d] = (depth_to_level[d - 1] + 1).min(max_target);
        }
    }

    // Boundary = a cell whose depth differs from any 8-neighbor (the wall lines).
    let mut boundary = vec![false; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let d = depth_matrix.get(r, c);
            let mut is_boundary = false;
            for (dr, dc) in EIGHT_DIRS {
                let nr = r as isize + dr;
                let nc = c as isize + dc;
                if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                    continue;
                }
                if depth_matrix.get(nr as usize, nc as usize) != d {
                    is_boundary = true;
                    break;
                }
            }
            boundary[idx(r, c, cols)] = is_boundary;
        }
    }
    if !boundary.iter().any(|b| *b) {
        return result;
    }

    let dist = distance_to_true(&boundary, rows, cols);
    let radius = (rows.min(cols) as f64 / MAGNETISM_RADIUS_DIVISOR).max(MAGNETISM_MIN_RADIUS);

    for r in 0..rows {
        for c in 0..cols {
            let i = idx(r, c, cols);
            let base = height_map.get(r, c);
            if base <= 0 {
                continue; // leave water and ocean untouched
            }
            let falloff = (1.0 - dist[i] as f64 / radius).clamp(0.0, 1.0);
            let weight = strength * falloff;
            if weight <= 0.0 {
                continue;
            }
            let depth = depth_matrix.get(r, c).clamp(0, 9) as usize;
            let target = depth_to_level[depth];
            let blended = base as f64 * (1.0 - weight) + target as f64 * weight;
            result.set(r, c, (blended.round() as i32).clamp(1, max_target));
        }
    }

    result
}

/// Build a pseudo depth field for brush walls so `magnetize_terrain` can treat them
/// like a single-ring polygon: 1 for wall cells and the area they enclose, 0 outside.
pub(crate) fn brush_interior_depth(wall_matrix: &Matrix) -> Matrix {
    let rows = wall_matrix.rows;
    let cols = wall_matrix.cols;
    let wall_mask: Vec<bool> = wall_matrix.data.iter().map(|v| *v == 1).collect();
    let outside = outside_non_wall_mask(&wall_mask, rows, cols);
    let data = outside.iter().map(|o| if *o { 0 } else { 1 }).collect();
    Matrix::new(rows, cols, data).unwrap_or_else(|_| Matrix::zeros(rows, cols))
}

pub(crate) fn enforce_transition_safety(height_map: &mut Matrix, max_height_level: i32) {
    let rows = height_map.rows;
    let cols = height_map.cols;
    let max_target = max_height_level.clamp(1, MAX_TERRAIN_LEVEL);
    let mut changed = true;

    while changed {
        changed = false;
        let snapshot = height_map.clone();

        for r in 0..rows {
            for c in 0..cols {
                let current = snapshot.get(r, c);
                if current <= 0 {
                    continue;
                }
                for (dr, dc) in DISTANCE_TWO_DIRS {
                    let end_r = r as isize + dr * 2;
                    let end_c = c as isize + dc * 2;
                    if end_r < 0 || end_c < 0 || end_r >= rows as isize || end_c >= cols as isize
                    {
                        continue;
                    }

                    let end_r = end_r as usize;
                    let end_c = end_c as usize;
                    let endpoint = snapshot.get(end_r, end_c);
                    if endpoint <= 0 || (current - endpoint).abs() <= 1 {
                        continue;
                    }

                    let mid_r = (r as isize + dr) as usize;
                    let mid_c = (c as isize + dc) as usize;
                    let middle = height_map.get(mid_r, mid_c);
                    if middle <= 0 {
                        continue;
                    }

                    let required_middle = (current.max(endpoint) - 1).clamp(1, max_target);
                    if middle < required_middle {
                        height_map.set(mid_r, mid_c, required_middle);
                        changed = true;
                    }
                }
            }
        }
    }
}

fn outside_non_wall_mask(wall_mask: &[bool], rows: usize, cols: usize) -> Vec<bool> {
    let mut outside = vec![false; rows * cols];
    if rows == 0 || cols == 0 {
        return outside;
    }

    let mut queue = VecDeque::new();
    for r in 0..rows {
        for c in 0..cols {
            if r != 0 && c != 0 && r + 1 != rows && c + 1 != cols {
                continue;
            }
            let i = idx(r, c, cols);
            if wall_mask[i] || outside[i] {
                continue;
            }
            outside[i] = true;
            queue.push_back((r, c));
        }
    }

    while let Some((r, c)) = queue.pop_front() {
        for (dr, dc) in CARDINAL_DIRS {
            let nr = r as isize + dr;
            let nc = c as isize + dc;
            if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                continue;
            }
            let nr = nr as usize;
            let nc = nc as usize;
            let ni = idx(nr, nc, cols);
            if wall_mask[ni] || outside[ni] {
                continue;
            }
            outside[ni] = true;
            queue.push_back((nr, nc));
        }
    }
    outside
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
        for (dr, dc) in EIGHT_DIRS {
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
    dist
}

pub(crate) fn wall_protection_mask(wall_matrix: &Matrix, radius: i32) -> Vec<bool> {
    let rows = wall_matrix.rows;
    let cols = wall_matrix.cols;
    let wall_mask: Vec<bool> = wall_matrix.data.iter().map(|v| *v == 1).collect();
    let dist = distance_to_true(&wall_mask, rows, cols);
    dist.iter().map(|d| *d <= radius).collect()
}

fn idx(r: usize, c: usize, cols: usize) -> usize {
    r * cols + c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magnetism_zero_strength_is_noop() {
        let height_map = Matrix::from_rows(vec![
            vec![1, 2, 3, 2, 1],
            vec![1, 2, 3, 2, 1],
            vec![1, 2, 3, 2, 1],
        ])
        .expect("valid height map");
        let depth_matrix = Matrix::from_rows(vec![
            vec![0, 0, 1, 0, 0],
            vec![0, 1, 1, 1, 0],
            vec![0, 0, 1, 0, 0],
        ])
        .expect("valid depth matrix");

        let result = magnetize_terrain(&height_map, &depth_matrix, 0.0, 7);
        assert_eq!(result, height_map);
    }

    #[test]
    fn magnetism_pulls_wall_cells_toward_zone_target_and_keeps_water() {
        // A single hill ring (depth 1) surrounded by flat land, with one water cell.
        let height_map = Matrix::from_rows(vec![
            vec![1, 1, 1, 1, 1],
            vec![0, 1, 1, 1, 1],
            vec![1, 1, 1, 1, 1],
            vec![1, 1, 1, 1, 1],
            vec![1, 1, 1, 1, 1],
        ])
        .expect("valid height map");
        let depth_matrix = Matrix::from_rows(vec![
            vec![0, 0, 0, 0, 0],
            vec![0, 1, 1, 1, 0],
            vec![0, 1, 1, 1, 0],
            vec![0, 1, 1, 1, 0],
            vec![0, 0, 0, 0, 0],
        ])
        .expect("valid depth matrix");

        let result = magnetize_terrain(&height_map, &depth_matrix, 1.0, 7);

        // Depth-1 cells get pulled up above flat land.
        assert!(result.get(2, 2) > 1);
        // Water cell stays water.
        assert_eq!(result.get(1, 0), 0);
    }

    #[test]
    fn transition_safety_raises_intermediate_tile() {
        let mut height_map =
            Matrix::from_rows(vec![vec![1, 1, 5]]).expect("valid transition matrix");

        enforce_transition_safety(&mut height_map, 7);

        assert_eq!(height_map.get(0, 1), 4);
    }
}
