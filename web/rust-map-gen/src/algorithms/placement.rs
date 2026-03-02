use std::collections::HashSet;



use rand::seq::SliceRandom;

use rand::Rng;



use crate::state::{Matrix, WizardState};



const BORDER_MARGIN_RATIO: f64 = 0.08;

const CENTER_FORBIDDEN_RATIO: f64 = 0.06;

const CC_MARGIN_RATIO: f64 = 0.07;

const CC_MIN_DISTANCE_RATIO: f64 = 0.1;



pub(crate) fn add_command_centers(
    randomized_matrix: &Matrix,
    mut num_centers: i32,
    mirroring: &str,
    height_map_shape: (usize, usize),
    items_matrix: Option<&Matrix>,
    wall_matrix: Option<&Matrix>,
) -> Result<Matrix, String> {
    if mirroring != "horizontal"
        && mirroring != "vertical"
        && mirroring != "diagonal1"
        && mirroring != "diagonal2"
        && mirroring != "both"
        && mirroring != "none"
    {
        return Ok(Matrix::zeros(height_map_shape.0, height_map_shape.1));
    }
    if mirroring == "both" {
        num_centers /= 4;
    } else if mirroring != "none" {
        num_centers /= 2;
    }
    let num_centers = num_centers.max(0) as usize;
    if num_centers == 0 {
        return Ok(Matrix::zeros(height_map_shape.0, height_map_shape.1));
    }

    let height = randomized_matrix.rows;
    let margin = (CC_MARGIN_RATIO * height as f64) as usize;
    let (mut valid_positions, preferred_area) =
        find_valid_cc_positions(randomized_matrix, mirroring, margin);

    let scale_y = height_map_shape.0 as f64 / randomized_matrix.rows as f64;
    let scale_x = height_map_shape.1 as f64 / randomized_matrix.cols as f64;

    if let Some(items) = items_matrix {
        let cc_clearance = 2usize;
        valid_positions.retain(|(y, x)| {
            let sy = (*y as f64 * scale_y) as usize;
            let sx = (*x as f64 * scale_x) as usize;
            !has_non_zero_in_radius(items, sy, sx, cc_clearance)
        });
    }
    if let Some(walls) = wall_matrix {
        valid_positions.retain(|(y, x)| {
            let sy = (*y as f64 * scale_y) as usize;
            let sx = (*x as f64 * scale_x) as usize;
            !has_equals_one_in_radius(walls, sy, sx, 1)
        });
    }

    let mut preferred_positions: Vec<(usize, usize)> = valid_positions
        .iter()
        .copied()
        .filter(|(y, x)| preferred_area[idx(*y, *x, randomized_matrix.cols)])
        .collect();

    if valid_positions.len() < num_centers {
        return Err("Not enough valid positions for command centers".to_string());
    }

    let mut rng = rand::thread_rng();
    let mut selected_positions: Vec<(usize, usize)> = Vec::new();
    while selected_positions.len() < num_centers {
        let pos = if !preferred_positions.is_empty() && rng.gen::<f64>() < 0.7 {
            let i = rng.gen_range(0..preferred_positions.len());
            preferred_positions.remove(i)
        } else {
            let i = rng.gen_range(0..valid_positions.len());
            valid_positions.remove(i)
        };

        selected_positions.push(pos);
        let min_distance = randomized_matrix.rows as f64 * CC_MIN_DISTANCE_RATIO;
        valid_positions.retain(|p| euclidean_distance(pos, *p) > min_distance);
        preferred_positions.retain(|p| euclidean_distance(pos, *p) > min_distance);
    }

    Ok(mirror_command_centers(
        &selected_positions,
        mirroring,
        randomized_matrix,
        height_map_shape,
    ))
}



fn mirror_command_centers(
    selected_positions: &[(usize, usize)],
    mirroring: &str,
    randomized_matrix: &Matrix,
    height_map_shape: (usize, usize),
) -> Matrix {
    let height = randomized_matrix.rows;
    let width = randomized_matrix.cols;
    let scale_y = height_map_shape.0 as f64 / height as f64;
    let scale_x = height_map_shape.1 as f64 / width as f64;
    let mut units = Matrix::zeros(height_map_shape.0, height_map_shape.1);

    for (i, (y, x)) in selected_positions.iter().enumerate() {
        let sy = (*y as f64 * scale_y) as usize;
        let sx = (*x as f64 * scale_x) as usize;
        units.set(
            sy.min(units.rows - 1),
            sx.min(units.cols - 1),
            101 + i as i32,
        );
    }

    for (y, x) in selected_positions {
        let team1_val = units.get(
            ((*y as f64 * scale_y) as usize).min(units.rows - 1),
            ((*x as f64 * scale_x) as usize).min(units.cols - 1),
        );
        let mirrors = get_mirrors_for_mode(*y, *x, height, width, mirroring);
        for (my, mx) in mirrors {
            let sy = (my as f64 * scale_y) as usize;
            let sx = (mx as f64 * scale_x) as usize;
            units.set(
                sy.min(units.rows - 1),
                sx.min(units.cols - 1),
                team1_val + 5,
            );
        }
    }
    units
}



fn find_valid_cc_positions(
    randomized_matrix: &Matrix,
    mirroring: &str,
    margin: usize,
) -> (Vec<(usize, usize)>, Vec<bool>) {
    let height = randomized_matrix.rows;
    let width = randomized_matrix.cols;
    let mut valid_area = vec![false; height * width];
    let mut preferred_area = vec![false; height * width];

    match mirroring {
        "horizontal" => {
            fill_rect(
                &mut valid_area,
                width,
                margin,
                margin,
                (height / 2).saturating_sub(margin),
                width.saturating_sub(margin),
                true,
            );
            preferred_area.clone_from(&valid_area);
            fill_rect(
                &mut preferred_area,
                width,
                margin,
                0,
                margin.saturating_mul(2),
                width,
                true,
            );
        }
        "vertical" => {
            fill_rect(
                &mut valid_area,
                width,
                margin,
                margin,
                height.saturating_sub(margin),
                (width / 2).saturating_sub(margin),
                true,
            );
            preferred_area.clone_from(&valid_area);
            fill_rect(
                &mut preferred_area,
                width,
                0,
                margin,
                height,
                margin.saturating_mul(2),
                true,
            );
        }
        "diagonal1" => {
            for r in 0..height {
                for c in 0..width {
                    if c as isize >= r as isize + margin as isize {
                        valid_area[idx(r, c, width)] = true;
                    }
                }
            }
            for r in height.saturating_sub(margin)..height {
                for c in 0..width {
                    valid_area[idx(r, c, width)] = false;
                }
            }
            for r in 0..height {
                for c in width.saturating_sub(margin)..width {
                    valid_area[idx(r, c, width)] = false;
                }
            }
            fill_rect(
                &mut preferred_area,
                width,
                margin,
                width.saturating_sub(width / 4),
                height / 4,
                width.saturating_sub(margin),
                true,
            );
        }
        "diagonal2" => {
            for r in 0..height {
                for c in 0..width {
                    let mirror_c = width - 1 - c;
                    if mirror_c as isize >= r as isize + margin as isize {
                        valid_area[idx(r, c, width)] = true;
                    }
                }
            }
            for r in height.saturating_sub(margin)..height {
                for c in 0..width {
                    valid_area[idx(r, c, width)] = false;
                }
            }
            for r in 0..height {
                for c in 0..margin {
                    valid_area[idx(r, c, width)] = false;
                }
            }
            fill_rect(
                &mut preferred_area,
                width,
                margin,
                margin,
                height / 4,
                width / 4,
                true,
            );
        }
        "both" => {
            fill_rect(
                &mut valid_area,
                width,
                margin,
                margin,
                (height / 2).saturating_sub(margin),
                (width / 2).saturating_sub(margin),
                true,
            );
            preferred_area.clone_from(&valid_area);
            fill_rect(
                &mut preferred_area,
                width,
                margin,
                margin,
                margin.saturating_mul(2),
                margin.saturating_mul(2),
                true,
            );
        }
        _ => {
            fill_rect(
                &mut valid_area,
                width,
                margin,
                margin,
                height.saturating_sub(margin),
                width.saturating_sub(margin),
                true,
            );
            preferred_area.clone_from(&valid_area);
        }
    }

    let mut valid_positions = Vec::new();
    for r in 0..height {
        for c in 0..width {
            if !valid_area[idx(r, c, width)] || randomized_matrix.get(r, c) != 1 {
                continue;
            }
            if is_all_land_5x5(randomized_matrix, r, c) {
                valid_positions.push((r, c));
            }
        }
    }

    (valid_positions, preferred_area)
}



pub(crate) fn add_resource_pulls(
    randomized_matrix: &Matrix,
    num_resource_pulls: usize,
    mirroring: &str,
    height_map: &Matrix,
    items_matrix: &mut Matrix,
    wall_matrix: Option<&Matrix>,
    units_matrix: Option<&Matrix>,
) -> Vec<(usize, usize)> {
    let rows = randomized_matrix.rows;
    let cols = randomized_matrix.cols;
    let forbidden_zones = get_forbidden_zones(rows, cols, mirroring);
    let mut available_tiles = find_valid_resource_positions(randomized_matrix, &forbidden_zones);
    if available_tiles.is_empty() {
        return Vec::new();
    }
    let scale_factor_x = height_map.cols as f64 / cols as f64;
    let scale_factor_y = height_map.rows as f64 / rows as f64;

    let mut rng = rand::thread_rng();
    available_tiles.shuffle(&mut rng);
    let mut placed_positions: Vec<(usize, usize)> = Vec::new();

    for (ci, cj) in available_tiles.iter().copied() {
        if placed_positions.len() >= num_resource_pulls {
            break;
        }
        let remaining = num_resource_pulls - placed_positions.len();
        let mirrored = get_mirrored_positions(ci, cj, rows, cols, mirroring);
        let scaled_positions: Vec<(usize, usize)> = mirrored
            .iter()
            .map(|(mi, mj)| {
                (
                    ((*mi as f64) * scale_factor_y) as usize,
                    ((*mj as f64) * scale_factor_x) as usize,
                )
            })
            .collect();
        if scaled_positions.len() > remaining {
            continue;
        }
        let all_valid = scaled_positions.iter().all(|(si, sj)| {
            is_valid_pool_position(
                *si,
                *sj,
                height_map,
                &placed_positions,
                4,
                wall_matrix,
                units_matrix,
                4,
            )
        });
        if !all_valid {
            continue;
        }
        for (si, sj) in scaled_positions {
            placed_positions.push((si, sj));
            place_resource_pull(items_matrix, si, sj);
        }
    }

    if placed_positions.len() < num_resource_pulls && mirroring != "none" {
        let mut axis_tiles = find_mirror_axis_positions(randomized_matrix, mirroring);
        axis_tiles.shuffle(&mut rng);
        for (ci, cj) in axis_tiles {
            if placed_positions.len() >= num_resource_pulls {
                break;
            }
            let si = ((ci as f64) * scale_factor_y) as usize;
            let sj = ((cj as f64) * scale_factor_x) as usize;
            if is_valid_pool_position(
                si,
                sj,
                height_map,
                &placed_positions,
                4,
                wall_matrix,
                units_matrix,
                4,
            ) {
                placed_positions.push((si, sj));
                place_resource_pull(items_matrix, si, sj);
            }
        }
    }
    placed_positions
}



fn find_mirror_axis_positions(randomized_matrix: &Matrix, mirroring: &str) -> Vec<(usize, usize)> {
    let rows = randomized_matrix.rows;
    let cols = randomized_matrix.cols;
    let mut axis_tiles = Vec::new();
    match mirroring {
        "horizontal" => {
            let i = rows / 2;
            for j in 2..cols.saturating_sub(2) {
                axis_tiles.push((i, j));
            }
        }
        "vertical" => {
            let j = cols / 2;
            for i in 2..rows.saturating_sub(2) {
                axis_tiles.push((i, j));
            }
        }
        "diagonal1" => {
            for i in 2..rows.min(cols).saturating_sub(2) {
                axis_tiles.push((i, i));
            }
        }
        "diagonal2" => {
            for i in 2..rows.min(cols).saturating_sub(2) {
                axis_tiles.push((i, cols - 1 - i));
            }
        }
        "both" => axis_tiles.push((rows / 2, cols / 2)),
        _ => {}
    }
    axis_tiles
        .into_iter()
        .filter(|(i, j)| {
            *i >= 2
                && *j >= 2
                && *i < rows.saturating_sub(2)
                && *j < cols.saturating_sub(2)
                && randomized_matrix.get(*i, *j) == 1
                && is_all_land_5x5(randomized_matrix, *i, *j)
        })
        .collect()
}



fn is_valid_pool_position(
    i: usize,
    j: usize,
    height_map: &Matrix,
    placed_positions: &[(usize, usize)],
    min_pool_distance: usize,
    wall_matrix: Option<&Matrix>,
    units_matrix: Option<&Matrix>,
    cc_clearance: usize,
) -> bool {
    if i < 1 || j < 1 || i + 1 >= height_map.rows || j + 1 >= height_map.cols {
        return false;
    }
    for dr in -1..=1 {
        for dc in -1..=1 {
            let nr = (i as isize + dr) as usize;
            let nc = (j as isize + dc) as usize;
            if height_map.get(nr, nc) <= 0 {
                return false;
            }
            if let Some(walls) = wall_matrix {
                if walls.get(nr, nc) == 1 {
                    return false;
                }
            }
        }
    }
    if let Some(units) = units_matrix {
        if has_positive_in_radius(units, i, j, cc_clearance) {
            return false;
        }
    }
    for (pi, pj) in placed_positions {
        if i.abs_diff(*pi) < min_pool_distance && j.abs_diff(*pj) < min_pool_distance {
            return false;
        }
    }
    true
}



fn find_valid_resource_positions(
    randomized_matrix: &Matrix,
    forbidden_zones: &HashSet<(usize, usize)>,
) -> Vec<(usize, usize)> {
    let rows = randomized_matrix.rows;
    let cols = randomized_matrix.cols;
    let mut available = Vec::new();
    for i in 2..rows.saturating_sub(2) {
        for j in 2..cols.saturating_sub(2) {
            if randomized_matrix.get(i, j) != 1 || forbidden_zones.contains(&(i, j)) {
                continue;
            }
            if is_all_land_5x5(randomized_matrix, i, j) {
                available.push((i, j));
            }
        }
    }
    available
}



fn get_forbidden_zones(rows: usize, cols: usize, mirroring: &str) -> HashSet<(usize, usize)> {
    let mut zones = HashSet::new();
    let border_size = ((rows.min(cols) as f64) * BORDER_MARGIN_RATIO) as usize;
    let center_size = ((rows.min(cols) as f64) * CENTER_FORBIDDEN_RATIO) as usize;

    for i in 0..rows {
        for j in 0..cols {
            if i < border_size
                || i >= rows.saturating_sub(border_size)
                || j < border_size
                || j >= cols.saturating_sub(border_size)
            {
                zones.insert((i, j));
            }
        }
    }

    match mirroring {
        "horizontal" => {
            let center = rows / 2;
            let start = center.saturating_sub(center_size / 2);
            let end = (center + center_size / 2).min(rows);
            for i in start..end {
                for j in 0..cols {
                    zones.insert((i, j));
                }
            }
        }
        "vertical" => {
            let center = cols / 2;
            let start = center.saturating_sub(center_size / 2);
            let end = (center + center_size / 2).min(cols);
            for j in start..end {
                for i in 0..rows {
                    zones.insert((i, j));
                }
            }
        }
        "diagonal1" => {
            for i in 0..rows {
                let start = i.saturating_sub(center_size / 2);
                let end = (i + center_size / 2).min(cols);
                for j in start..end {
                    zones.insert((i, j));
                }
            }
        }
        "diagonal2" => {
            for i in 0..rows {
                let center = cols.saturating_sub(1).saturating_sub(i);
                let start = center.saturating_sub(center_size / 2);
                let end = (center + center_size / 2).min(cols);
                for j in start..end {
                    zones.insert((i, j));
                }
            }
        }
        "both" => {
            let center_row = rows / 2;
            let center_col = cols / 2;
            let row_start = center_row.saturating_sub(center_size / 2);
            let row_end = (center_row + center_size / 2).min(rows);
            for i in row_start..row_end {
                for j in 0..cols {
                    zones.insert((i, j));
                }
            }
            let col_start = center_col.saturating_sub(center_size / 2);
            let col_end = (center_col + center_size / 2).min(cols);
            for j in col_start..col_end {
                for i in 0..rows {
                    zones.insert((i, j));
                }
            }
        }
        _ => {}
    }
    zones
}



pub(crate) fn place_resource_pull(items_matrix: &mut Matrix, i: usize, j: usize) {
    const RESOURCE_PULL_TILES: [((isize, isize), i32); 9] = [
        ((-1, -1), 1),
        ((-1, 0), 2),
        ((-1, 1), 3),
        ((0, -1), 11),
        ((0, 0), 12),
        ((0, 1), 13),
        ((1, -1), 21),
        ((1, 0), 22),
        ((1, 1), 23),
    ];
    for ((dr, dc), value) in RESOURCE_PULL_TILES {
        let nr = i as isize + dr;
        let nc = j as isize + dc;
        if nr < 0 || nc < 0 || nr >= items_matrix.rows as isize || nc >= items_matrix.cols as isize
        {
            continue;
        }
        items_matrix.set(nr as usize, nc as usize, value);
    }
}



pub(crate) fn clear_pool(items_matrix: &mut Matrix, i: usize, j: usize) {
    for dr in -1..=1 {
        for dc in -1..=1 {
            let nr = i as isize + dr;
            let nc = j as isize + dc;
            if nr < 0
                || nc < 0
                || nr >= items_matrix.rows as isize
                || nc >= items_matrix.cols as isize
            {
                continue;
            }
            items_matrix.set(nr as usize, nc as usize, 0);
        }
    }
}



pub(crate) fn rebuild_cc_matrix(state: &mut WizardState) {
    if state.units_matrix.is_none() {
        let (h, w) = if let Some(height_map) = &state.height_map {
            (height_map.rows, height_map.cols)
        } else {
            (state.height, state.width)
        };
        state.units_matrix = Some(Matrix::zeros(h, w));
    }
    let Some(units) = &mut state.units_matrix else {
        return;
    };
    units.data.fill(0);
    for group in &state.cc_groups {
        for (idx, (r, c)) in group.positions.iter().copied().enumerate() {
            if idx == 0 {
                units.set(r, c, group.id);
            } else {
                units.set(r, c, group.id + 5);
            }
        }
    }
}



pub(crate) fn get_mirrored_positions(
    i: usize,
    j: usize,
    rows: usize,
    cols: usize,
    mirroring: &str,
) -> Vec<(usize, usize)> {
    let mut positions = vec![(i, j)];
    match mirroring {
        "horizontal" => positions.push((rows - 1 - i, j)),
        "vertical" => positions.push((i, cols - 1 - j)),
        "diagonal1" => positions.push((j, i)),
        "diagonal2" => positions.push((rows - 1 - j, cols - 1 - i)),
        "both" => {
            positions.push((rows - 1 - i, j));
            positions.push((i, cols - 1 - j));
            positions.push((rows - 1 - i, cols - 1 - j));
        }
        _ => {}
    }
    let mut dedup = Vec::new();
    let mut seen = HashSet::new();
    for p in positions {
        if seen.insert(p) {
            dedup.push(p);
        }
    }
    dedup
}



fn get_mirrors_for_mode(
    y: usize,
    x: usize,
    height: usize,
    width: usize,
    mirroring: &str,
) -> Vec<(usize, usize)> {
    match mirroring {
        "horizontal" => vec![(height - 1 - y, x)],
        "vertical" => vec![(y, width - 1 - x)],
        "diagonal1" => vec![(x, y)],
        "diagonal2" => vec![(width - 1 - x, height - 1 - y)],
        "both" => vec![
            (height - 1 - y, x),
            (y, width - 1 - x),
            (height - 1 - y, width - 1 - x),
        ],
        _ => Vec::new(),
    }
}



pub(crate) fn expand_brush(
    row: usize,
    col: usize,
    brush_size: usize,
    h: usize,
    w: usize,
) -> Vec<(usize, usize)> {
    let radius = brush_size / 2;
    let mut out = Vec::new();
    for dr in -(radius as isize)..=(radius as isize) {
        for dc in -(radius as isize)..=(radius as isize) {
            let nr = row as isize + dr;
            let nc = col as isize + dc;
            if nr >= 0 && nc >= 0 && nr < h as isize && nc < w as isize {
                out.push((nr as usize, nc as usize));
            }
        }
    }
    out
}



fn euclidean_distance(a: (usize, usize), b: (usize, usize)) -> f64 {
    let dr = a.0 as f64 - b.0 as f64;
    let dc = a.1 as f64 - b.1 as f64;
    (dr * dr + dc * dc).sqrt()
}



pub(crate) fn has_non_zero_in_radius(matrix: &Matrix, row: usize, col: usize, radius: usize) -> bool {
    let y_min = row.saturating_sub(radius);
    let y_max = (row + radius + 1).min(matrix.rows);
    let x_min = col.saturating_sub(radius);
    let x_max = (col + radius + 1).min(matrix.cols);
    for r in y_min..y_max {
        for c in x_min..x_max {
            if matrix.get(r, c) != 0 {
                return true;
            }
        }
    }
    false
}



pub(crate) fn has_positive_in_radius(matrix: &Matrix, row: usize, col: usize, radius: usize) -> bool {
    let y_min = row.saturating_sub(radius);
    let y_max = (row + radius + 1).min(matrix.rows);
    let x_min = col.saturating_sub(radius);
    let x_max = (col + radius + 1).min(matrix.cols);
    for r in y_min..y_max {
        for c in x_min..x_max {
            if matrix.get(r, c) > 0 {
                return true;
            }
        }
    }
    false
}



pub(crate) fn has_equals_one_in_radius(matrix: &Matrix, row: usize, col: usize, radius: usize) -> bool {
    let y_min = row.saturating_sub(radius);
    let y_max = (row + radius + 1).min(matrix.rows);
    let x_min = col.saturating_sub(radius);
    let x_max = (col + radius + 1).min(matrix.cols);
    for r in y_min..y_max {
        for c in x_min..x_max {
            if matrix.get(r, c) == 1 {
                return true;
            }
        }
    }
    false
}



fn is_all_land_5x5(matrix: &Matrix, r: usize, c: usize) -> bool {
    if r < 2 || c < 2 || r + 2 >= matrix.rows || c + 2 >= matrix.cols {
        return false;
    }
    for dr in -2..=2 {
        for dc in -2..=2 {
            let nr = (r as isize + dr) as usize;
            let nc = (c as isize + dc) as usize;
            if matrix.get(nr, nc) != 1 {
                return false;
            }
        }
    }
    true
}



fn fill_rect(
    mask: &mut [bool],
    cols: usize,
    start_r: usize,
    start_c: usize,
    end_r: usize,
    end_c: usize,
    value: bool,
) {
    for r in start_r..end_r {
        for c in start_c..end_c {
            if r * cols + c < mask.len() {
                mask[idx(r, c, cols)] = value;
            }
        }
    }
}



fn idx(r: usize, c: usize, cols: usize) -> usize {
    r * cols + c
}
