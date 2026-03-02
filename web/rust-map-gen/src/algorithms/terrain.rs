use std::collections::VecDeque;

use crate::state::Matrix;

const WALL_INFLUENCE_RADIUS: f64 = 12.0;
const WALL_BORDER_BOOST: i32 = 2;
const MAX_TERRAIN_LEVEL: i32 = 7;

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

pub(crate) fn bias_terrain_near_walls(
    height_map: &Matrix,
    wall_matrix: &Matrix,
    num_height_levels: i32,
) -> Matrix {
    let rows = height_map.rows;
    let cols = height_map.cols;
    let mut result = height_map.clone();
    let max_target = num_height_levels.clamp(1, MAX_TERRAIN_LEVEL);
    let wall_mask: Vec<bool> = wall_matrix.data.iter().map(|v| *v == 1).collect();
    let outside_mask = outside_non_wall_mask(&wall_mask, rows, cols);
    let interior_mask: Vec<bool> = wall_mask
        .iter()
        .zip(outside_mask.iter())
        .map(|(is_wall, is_outside)| !is_wall && !is_outside)
        .collect();

    let mut wall_targets = vec![0_i32; rows * cols];
    let mut visited = vec![false; rows * cols];

    for r in 0..rows {
        for c in 0..cols {
            let i = idx(r, c, cols);
            if !interior_mask[i] || visited[i] {
                continue;
            }

            let mut queue = VecDeque::new();
            let mut component = Vec::new();
            visited[i] = true;
            queue.push_back((r, c));

            while let Some((cr, cc)) = queue.pop_front() {
                component.push((cr, cc));
                for (dr, dc) in CARDINAL_DIRS {
                    let nr = cr as isize + dr;
                    let nc = cc as isize + dc;
                    if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                        continue;
                    }
                    let nr = nr as usize;
                    let nc = nc as usize;
                    let ni = idx(nr, nc, cols);
                    if interior_mask[ni] && !visited[ni] {
                        visited[ni] = true;
                        queue.push_back((nr, nc));
                    }
                }
            }

            let mut sum = 0_i64;
            let mut count = 0_i64;
            for (cr, cc) in &component {
                let current = height_map.get(*cr, *cc);
                if current > 0 {
                    sum += current as i64;
                    count += 1;
                }
            }
            if count == 0 {
                continue;
            }

            let border_height =
                ((sum as f64 / count as f64) + WALL_BORDER_BOOST as f64).round() as i32;
            let border_height = border_height.clamp(1, max_target);

            for (cr, cc) in &component {
                let current = height_map.get(*cr, *cc);
                if current > 0 && current < border_height {
                    result.set(*cr, *cc, border_height);
                }
            }

            for (cr, cc) in &component {
                for (dr, dc) in EIGHT_DIRS {
                    let nr = *cr as isize + dr;
                    let nc = *cc as isize + dc;
                    if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                        continue;
                    }
                    let ni = idx(nr as usize, nc as usize, cols);
                    if wall_mask[ni] {
                        wall_targets[ni] = wall_targets[ni].max(border_height);
                    }
                }
            }
        }
    }

    if !wall_targets.iter().any(|v| *v > 0) {
        return result;
    }

    let (dist, nearest_target) = distance_and_target_from_sources(&wall_targets, rows, cols);
    for r in 0..rows {
        for c in 0..cols {
            let i = idx(r, c, cols);
            if interior_mask[i] {
                continue;
            }
            let base_height = height_map.get(r, c);
            if base_height <= 0 {
                continue;
            }
            let target = nearest_target[i];
            if target <= 0 {
                continue;
            }

            let influence = (1.0 - dist[i] as f64 / WALL_INFLUENCE_RADIUS).clamp(0.0, 1.0);
            if influence <= 0.0 {
                continue;
            }
            let boosted =
                (base_height as f64 + influence * (target as f64 - base_height as f64)).round()
                    as i32;
            result.set(r, c, boosted.clamp(base_height, max_target));
        }
    }

    result
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

fn distance_and_target_from_sources(
    source_target: &[i32],
    rows: usize,
    cols: usize,
) -> (Vec<i32>, Vec<i32>) {
    let mut dist = vec![i32::MAX / 4; rows * cols];
    let mut target = vec![0_i32; rows * cols];
    let mut queue = VecDeque::new();

    for r in 0..rows {
        for c in 0..cols {
            let i = idx(r, c, cols);
            if source_target[i] > 0 {
                dist[i] = 0;
                target[i] = source_target[i];
                queue.push_back((r, c));
            }
        }
    }
    if queue.is_empty() {
        return (dist, target);
    }
    while let Some((r, c)) = queue.pop_front() {
        let i = idx(r, c, cols);
        let current_dist = dist[i];
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
                target[ni] = target[i];
                queue.push_back((nr, nc));
            }
        }
    }
    (dist, target)
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
    fn enclosed_walls_raise_interior_floor_and_keep_water() {
        let height_map = Matrix::from_rows(vec![
            vec![1, 1, 1, 1, 1],
            vec![0, 1, 1, 1, 1],
            vec![1, 1, 1, 1, 1],
            vec![1, 1, 1, 1, 1],
            vec![1, 1, 1, 1, 1],
        ])
        .expect("valid height map");
        let wall_matrix = Matrix::from_rows(vec![
            vec![0, 0, 0, 0, 0],
            vec![0, 1, 1, 1, 0],
            vec![0, 1, 0, 1, 0],
            vec![0, 1, 1, 1, 0],
            vec![0, 0, 0, 0, 0],
        ])
        .expect("valid wall matrix");

        let result = bias_terrain_near_walls(&height_map, &wall_matrix, 7);

        assert_eq!(result.get(2, 2), 3);
        assert_eq!(result.get(1, 1), 3);
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
