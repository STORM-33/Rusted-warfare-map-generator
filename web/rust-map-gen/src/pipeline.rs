use std::collections::{HashMap, HashSet};

use rand::Rng;
use serde::Serialize;

use crate::algorithms::coastline::{mirror, randomize, scale_matrix, subdivide};
use crate::algorithms::decoration::add_decoration_tiles;
use crate::algorithms::perlin::generate_perlin_map;
use crate::algorithms::placement::{
    add_command_centers, add_resource_pulls, clear_pool, expand_brush, get_mirrored_positions,
    has_non_zero_in_radius, has_positive_in_radius, place_resource_pull, rebuild_cc_matrix,
};
use crate::algorithms::smoothing::{smooth_terrain_tiles, smooth_wall_tiles};
use crate::algorithms::terrain::{
    bias_terrain_near_walls, enforce_transition_safety, generate_level, wall_protection_mask,
};
use crate::algorithms::tmx::write_tmx as write_tmx_impl;
use crate::state::{
    CcGroup, HillDrawingMode, Matrix, MatrixPayload, PolygonData, WizardState, WizardStep,
};
use crate::tiles::{
    DECORATION_FREQUENCY, DECORATION_TILES_LEVEL_1, DECORATION_TILES_LEVEL_2,
    DECORATION_TILES_LEVEL_3, DECORATION_TILES_LEVEL_4, DECORATION_TILES_LEVEL_5,
    DECORATION_TILES_LEVEL_6, DECORATION_TILES_LEVEL_7, TERRAIN_LEVELS,
};

const UPSCALES: [usize; 9] = [5, 10, 20, 40, 80, 160, 320, 640, 1280];

#[derive(Clone, Debug, Serialize)]
pub struct CoastlineFrame {
    pub label: String,
    pub shape: [i32; 2],
    pub data: Vec<i32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct QuickGenerateFrame {
    pub label: String,
    pub height_map: MatrixPayload,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_matrix: Option<MatrixPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items_matrix: Option<MatrixPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub units_matrix: Option<MatrixPayload>,
}

pub fn run_coastline(state: &mut WizardState) -> Vec<CoastlineFrame> {
    let mut frames = Vec::new();
    let min_dimension = state.height.min(state.width);
    let mut num_upscales = 1;
    for (idx, upscale) in UPSCALES.iter().enumerate() {
        if min_dimension >= *upscale {
            num_upscales = idx + 1;
        }
    }

    let mut randomized_matrix = state
        .initial_matrix
        .clone()
        .unwrap_or_else(default_seed_matrix);
    frames.push(matrix_frame("initial_matrix", &randomized_matrix));

    for i in 0..num_upscales {
        let subdivided = subdivide(&randomized_matrix);
        let randomized = randomize(&subdivided, 0.0);
        randomized_matrix = mirror(&randomized, &state.mirroring);
        frames.push(matrix_frame(
            format!("upscale_{}/{}", i + 1, num_upscales),
            &randomized_matrix,
        ));
    }

    let coastline_height_map = scale_matrix(&randomized_matrix, state.height, state.width);
    state.randomized_matrix = Some(randomized_matrix);
    state.coastline_height_map = Some(coastline_height_map.clone());
    state.wall_matrix = Some(Matrix::zeros(state.height, state.width));
    state.completed_step = state.completed_step.max(WizardStep::Coastline as i32);

    frames.push(matrix_frame("coastline_complete", &coastline_height_map));
    frames
}

pub fn draw_walls(
    state: &mut WizardState,
    points: &[[i32; 2]],
    value: i32,
    brush_size: usize,
) -> Result<(), String> {
    let h = state.height;
    let w = state.width;

    let coast = state.coastline_height_map.clone();
    let mirroring = state.mirroring.clone();
    let wall_matrix = state.ensure_wall_matrix();

    for [row_i32, col_i32] in points {
        if *row_i32 < 0 || *col_i32 < 0 {
            continue;
        }
        let row = *row_i32 as usize;
        let col = *col_i32 as usize;

        for (br, bc) in expand_brush(row, col, brush_size, h, w) {
            let mirrored = get_mirrored_positions(br, bc, h, w, &mirroring);
            for (mr, mc) in mirrored {
                if mr >= h || mc >= w {
                    continue;
                }
                if let Some(coastline) = &coast {
                    if coastline.get(mr, mc) <= 0 {
                        continue;
                    }
                }
                if value == 2 && wall_matrix.get(mr, mc) == 0 {
                    continue;
                }
                wall_matrix.set(mr, mc, value);
            }
        }
    }
    Ok(())
}

pub fn clear_walls(state: &mut WizardState) {
    state.wall_matrix = Some(Matrix::zeros(state.height, state.width));
}

pub fn set_wall_cells(
    state: &mut WizardState,
    cells: &[[i32; 2]],
    value: i32,
) -> Result<(), String> {
    let h = state.height;
    let w = state.width;
    let coast = state.coastline_height_map.clone();
    let wall_matrix = state.ensure_wall_matrix();
    for [row_i32, col_i32] in cells {
        if *row_i32 < 0 || *col_i32 < 0 {
            continue;
        }
        let row = *row_i32 as usize;
        let col = *col_i32 as usize;
        if row >= h || col >= w {
            continue;
        }
        if let Some(coastline) = &coast {
            if coastline.get(row, col) <= 0 {
                continue;
            }
        }
        wall_matrix.set(row, col, value);
    }
    Ok(())
}

pub fn set_polygon_walls(
    state: &mut WizardState,
    polygons: Vec<PolygonData>,
) -> Result<(), String> {
    let h = state.height;
    let w = state.width;
    let coast = state.coastline_height_map.clone();

    // Store polygon data for per-polygon rendering in run_finalize
    state.polygon_walls = polygons;

    // Build combined wall_matrix (union of all polygon fills) for elevation and snapshot display
    let wall_matrix = state.ensure_wall_matrix();
    wall_matrix.data.fill(0);

    let mirroring = state.mirroring.clone();

    // Collect all effective polygons (originals + mirrored)
    let mirrored_polys: Vec<PolygonData> = state
        .polygon_walls
        .iter()
        .flat_map(|poly| mirror_polygon_vertices(poly, h, w, &mirroring))
        .collect();
    let all_fills: Vec<Vec<(usize, usize)>> = state
        .polygon_walls
        .iter()
        .chain(mirrored_polys.iter())
        .map(|poly| fill_polygon_scanline(&poly.vertices, h, w))
        .collect();

    let wall_matrix = state.wall_matrix.as_mut().unwrap();
    for cells in &all_fills {
        for &(r, c) in cells {
            if let Some(coastline) = &coast {
                if coastline.get(r, c) <= 0 {
                    continue;
                }
            }
            wall_matrix.set(r, c, 1);
        }
    }

    Ok(())
}

/// Scanline polygon fill: returns all cells inside the polygon (inclusive of boundary).
fn fill_polygon_scanline(vertices: &[[i32; 2]], rows: usize, cols: usize) -> Vec<(usize, usize)> {
    if vertices.len() < 3 {
        return Vec::new();
    }

    let mut min_row = i32::MAX;
    let mut max_row = i32::MIN;
    let mut min_col = i32::MAX;
    let mut max_col = i32::MIN;
    for &[r, c] in vertices {
        min_row = min_row.min(r);
        max_row = max_row.max(r);
        min_col = min_col.min(c);
        max_col = max_col.max(c);
    }

    let mut result = Vec::new();
    let n = vertices.len();

    for row in min_row..=max_row {
        let ry = row as f64 + 0.5;
        let mut intersections = Vec::new();
        for i in 0..n {
            let [r0, c0] = vertices[i];
            let [r1, c1] = vertices[(i + 1) % n];
            let (r0f, c0f) = (r0 as f64, c0 as f64);
            let (r1f, c1f) = (r1 as f64, c1 as f64);
            if (r0f <= ry && ry < r1f) || (r1f <= ry && ry < r0f) {
                let col = c0f + (ry - r0f) * (c1f - c0f) / (r1f - r0f);
                intersections.push(col);
            }
        }
        intersections.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut i = 0;
        while i + 1 < intersections.len() {
            let start_col = (intersections[i] - 0.5).ceil().max(min_col as f64) as i32;
            let end_col = (intersections[i + 1] + 0.5).floor().min(max_col as f64) as i32;
            for col in start_col..=end_col {
                if row >= 0 && col >= 0 && (row as usize) < rows && (col as usize) < cols {
                    result.push((row as usize, col as usize));
                }
            }
            i += 2;
        }
    }

    // Include edge cells via Bresenham rasterization for boundary coverage
    let mut edge_set: HashSet<(usize, usize)> = result.iter().copied().collect();
    for i in 0..n {
        let [r0, c0] = vertices[i];
        let [r1, c1] = vertices[(i + 1) % n];
        for (r, c) in rasterize_line_cells(r0, c0, r1, c1) {
            if r < rows && c < cols && edge_set.insert((r, c)) {
                result.push((r, c));
            }
        }
    }

    result
}

/// Bresenham line rasterization. Returns cells as (row, col).
fn rasterize_line_cells(r0: i32, c0: i32, r1: i32, c1: i32) -> Vec<(usize, usize)> {
    let mut cells = Vec::new();
    let dr = (r1 - r0).abs();
    let dc = (c1 - c0).abs();
    let sr: i32 = if r0 < r1 { 1 } else { -1 };
    let sc: i32 = if c0 < c1 { 1 } else { -1 };
    let mut err = dr - dc;
    let mut r = r0;
    let mut c = c0;
    loop {
        if r >= 0 && c >= 0 {
            cells.push((r as usize, c as usize));
        }
        if r == r1 && c == c1 {
            break;
        }
        let e2 = 2 * err;
        if e2 > -dc {
            err -= dc;
            r += sr;
        }
        if e2 < dr {
            err += dr;
            c += sc;
        }
    }
    cells
}

/// Mirror a polygon's vertices using the same logic as get_mirrored_positions.
fn mirror_polygon_vertices(
    poly: &PolygonData,
    rows: usize,
    cols: usize,
    mirroring: &str,
) -> Vec<PolygonData> {
    let rows_i = rows as i32;
    let cols_i = cols as i32;
    let mirror_fns: Vec<Box<dyn Fn(i32, i32) -> [i32; 2]>> = match mirroring {
        "horizontal" => vec![Box::new(move |r, c| [rows_i - 1 - r, c])],
        "vertical" => vec![Box::new(move |r, c| [r, cols_i - 1 - c])],
        "diagonal1" => vec![Box::new(move |r, c| [c, r])],
        "diagonal2" => vec![Box::new(move |r, c| [rows_i - 1 - c, cols_i - 1 - r])],
        "both" => vec![
            Box::new(move |r, c| [rows_i - 1 - r, c]),
            Box::new(move |r, c| [r, cols_i - 1 - c]),
            Box::new(move |r, c| [rows_i - 1 - r, cols_i - 1 - c]),
        ],
        _ => return Vec::new(),
    };

    mirror_fns
        .iter()
        .map(|f| {
            let new_verts: Vec<[i32; 2]> = poly.vertices.iter().map(|&[r, c]| f(r, c)).collect();
            PolygonData {
                id: poly.id,
                vertices: new_verts,
                edge_gaps: poly.edge_gaps.clone(),
            }
        })
        .collect()
}

pub fn run_height_ocean(state: &mut WizardState, seed: Option<i32>) -> Result<(), String> {
    let mut rng = rand::thread_rng();
    let seed = seed.unwrap_or_else(|| rng.gen_range(0..100_000));
    let mut height_map = state
        .coastline_height_map
        .clone()
        .ok_or_else(|| "Missing coastline map".to_string())?;

    let height = state.height;
    let width = state.width;
    let perlin_map = generate_perlin_map(height, width, 9, seed);
    state.perlin_seed = Some(seed);
    state.perlin_map = Some(perlin_map.clone());

    let num_height_levels = state.num_height_levels.max(1) as usize;
    let num_ocean_levels = state.num_ocean_levels.max(1) as usize;

    let mut perlin_value = -0.5_f64;
    let perlin_change_height = 1.0_f64 / num_height_levels as f64;
    for level in 2..=(num_height_levels as i32) {
        perlin_value += perlin_change_height;
        height_map = generate_level(
            &height_map,
            &perlin_map,
            "height",
            level,
            perlin_value,
            3,
            4,
        );
    }

    perlin_value = -0.5_f64;
    let perlin_change_ocean = 1.0_f64 / num_ocean_levels as f64;
    for level in (1..=num_ocean_levels as i32).map(|v| -v) {
        perlin_value += perlin_change_ocean;
        height_map = generate_level(&height_map, &perlin_map, "ocean", level, perlin_value, 3, 4);
    }

    // Apply elevation based on active mode
    match state.hill_drawing_mode {
        HillDrawingMode::Brush => {
            if let Some(wall_matrix) = &state.brush_wall_matrix {
                if wall_matrix.data.iter().any(|v| *v == 1) {
                    height_map = bias_terrain_near_walls(&height_map, wall_matrix, state.num_height_levels);
                }
            }
        }
        HillDrawingMode::Polygon => {
            if let Some(depth_matrix) = &state.polygon_depth_matrix {
                if depth_matrix.data.iter().any(|v| *v > 0) {
                    height_map = apply_polygon_depth_elevation(&height_map, depth_matrix, state.num_height_levels);
                }
            }
        }
    }
    
    enforce_transition_safety(&mut height_map, state.num_height_levels);

    state.height_map = Some(height_map);
    state.completed_step = state.completed_step.max(WizardStep::HeightOcean as i32);
    Ok(())
}

/// New elevation function for polygon depth-based terrain
fn apply_polygon_depth_elevation(
    height_map: &Matrix,
    depth_matrix: &Matrix,
    num_height_levels: i32,
) -> Matrix {
    let mut result = height_map.clone();
    let max_depth = depth_matrix.data.iter().copied().max().unwrap_or(0).min(9);
    let max_target = num_height_levels.clamp(2, 7);
    
    // Pre-compute target level for each possible depth (1-9)
    let base_level = 2;
    let available_range = max_target - base_level;
    
    let mut depth_to_level = [0i32; 10];
    for d in 1..=max_depth {
        let ratio = (d - 1) as f32 / (max_depth as f32).max(1.0);
        let level = base_level + (ratio * available_range as f32).round() as i32;
        depth_to_level[d as usize] = level.min(max_target).max(base_level);
    }
    
    // Ensure consecutive depths have different levels
    for d in 2..=max_depth {
        if depth_to_level[d as usize] <= depth_to_level[(d - 1) as usize] {
            depth_to_level[d as usize] = depth_to_level[(d - 1) as usize] + 1;
            if depth_to_level[d as usize] > max_target {
                depth_to_level[d as usize] = max_target;
            }
        }
    }
    
    // Apply elevation: each cell gets the level corresponding to its depth
    for r in 0..height_map.rows {
        for c in 0..height_map.cols {
            let depth = depth_matrix.get(r, c);
            if depth > 0 {
                let target_level = depth_to_level[depth as usize];
                let current = result.get(r, c);
                if current > 0 && current < target_level {
                    result.set(r, c, target_level);
                }
            }
        }
    }

    result
}

pub fn run_place_cc_random(state: &mut WizardState) -> Result<(), String> {
    let randomized_matrix = state
        .randomized_matrix
        .clone()
        .unwrap_or_else(|| Matrix::fill(5, 5, 1));
    let height_map = state
        .height_map
        .clone()
        .unwrap_or_else(|| Matrix::fill(state.height, state.width, 1));
    let items_matrix = state
        .items_matrix
        .clone()
        .unwrap_or_else(|| Matrix::zeros(height_map.rows, height_map.cols));
    let units_matrix = add_command_centers(
        &randomized_matrix,
        state.num_command_centers,
        &state.mirroring,
        (height_map.rows, height_map.cols),
        Some(&items_matrix),
        state.wall_matrix.as_ref(),
    )?;

    let mut cc_positions = Vec::new();
    for r in 0..units_matrix.rows {
        for c in 0..units_matrix.cols {
            if units_matrix.get(r, c) > 0 {
                cc_positions.push((r, c));
            }
        }
    }
    state.units_matrix = Some(units_matrix);
    state.cc_positions = cc_positions.clone();
    state.cc_groups = vec![CcGroup {
        id: 101,
        positions: cc_positions,
    }];
    state.completed_step = state.completed_step.max(WizardStep::CommandCenters as i32);
    Ok(())
}

pub fn run_place_cc_manual(
    state: &mut WizardState,
    row: i32,
    col: i32,
    mirrored: bool,
) -> Vec<(usize, usize)> {
    if row < 0 || col < 0 {
        return Vec::new();
    }
    let row = row as usize;
    let col = col as usize;

    let Some(height_map) = state.height_map.clone() else {
        return Vec::new();
    };
    let Some(randomized_matrix) = state.randomized_matrix.clone() else {
        return Vec::new();
    };
    let h = height_map.rows;
    let w = height_map.cols;
    if row >= h || col >= w {
        return Vec::new();
    }

    let rm_h = randomized_matrix.rows;
    let rm_w = randomized_matrix.cols;
    let rm_row = row * rm_h / h;
    let rm_col = col * rm_w / w;
    if randomized_matrix.get(rm_row, rm_col) != 1 {
        return Vec::new();
    }
    if height_map.get(row, col) <= 0 {
        return Vec::new();
    }
    
    let wall_check = state.wall_matrix.as_ref().map(|m| m.get(row, col) == 1).unwrap_or(false)
        || state.brush_wall_matrix.as_ref().map(|m| m.get(row, col) == 1).unwrap_or(false)
        || state.polygon_depth_matrix.as_ref().map(|m| m.get(row, col) > 0).unwrap_or(false);
    if wall_check {
        return Vec::new();
    }
    
    if let Some(items_matrix) = &state.items_matrix {
        let cc_clearance = 2;
        if has_non_zero_in_radius(items_matrix, row, col, cc_clearance) {
            return Vec::new();
        }
    }

    let mirroring = if mirrored {
        state.mirroring.as_str()
    } else {
        "none"
    };
    let mirrored_rm = get_mirrored_positions(rm_row, rm_col, rm_h, rm_w, mirroring);
    let mut placed: Vec<(usize, usize)> = Vec::new();
    for (mr, mc) in mirrored_rm {
        let sr = ((mr * h) / rm_h).min(h - 1);
        let sc = ((mc * w) / rm_w).min(w - 1);
        placed.push((sr, sc));
    }

    if state.units_matrix.is_none() {
        state.units_matrix = Some(Matrix::zeros(h, w));
    }

    let max_ccs = if mirroring == "both" { 8usize } else { 10usize };
    while state.cc_positions.len() + placed.len() > max_ccs && !state.cc_groups.is_empty() {
        let oldest = state.cc_groups.remove(0);
        for (r, c) in oldest.positions {
            if let Some(units) = &mut state.units_matrix {
                units.set(r, c, 0);
            }
            if let Some(idx) = state.cc_positions.iter().position(|p| *p == (r, c)) {
                state.cc_positions.remove(idx);
            }
        }
    }

    state.cc_positions.extend(placed.clone());
    state.cc_groups.push(CcGroup {
        id: 0,
        positions: placed.clone(),
    });
    rebuild_cc_matrix(state);
    placed
}

pub fn run_remove_cc_manual(state: &mut WizardState, row: i32, col: i32) -> bool {
    if row < 0 || col < 0 || state.units_matrix.is_none() || state.cc_groups.is_empty() {
        return false;
    }
    let row = row as usize;
    let col = col as usize;

    let mut closest_group_idx: Option<usize> = None;
    let mut min_dist = usize::MAX;

    for (idx, group) in state.cc_groups.iter().enumerate() {
        for (gr, gc) in &group.positions {
            let dist = gr.abs_diff(row).max(gc.abs_diff(col));
            if dist <= 3 && dist < min_dist {
                min_dist = dist;
                closest_group_idx = Some(idx);
            }
        }
    }

    let Some(group_idx) = closest_group_idx else {
        return false;
    };
    let removed_group = state.cc_groups.remove(group_idx);
    let removed_positions = removed_group.positions.clone();

    for (r, c) in removed_positions.iter().copied() {
        if let Some(units) = &mut state.units_matrix {
            units.set(r, c, 0);
        }
        if let Some(pos_idx) = state.cc_positions.iter().position(|p| *p == (r, c)) {
            state.cc_positions.remove(pos_idx);
        }
    }

    if removed_positions.len() > 1 {
        let used_by_single: HashSet<i32> = state
            .cc_groups
            .iter()
            .filter(|g| g.positions.len() == 1)
            .map(|g| g.id)
            .collect();
        let mut candidate = 101_i32;
        for group in &mut state.cc_groups {
            if group.positions.len() > 1 {
                while used_by_single.contains(&candidate)
                    || used_by_single.contains(&(candidate + 5))
                {
                    candidate += 1;
                }
                group.id = candidate;
                candidate += 1;
            }
        }
    }

    rebuild_cc_matrix(state);
    true
}

pub fn undo_last_cc(state: &mut WizardState) {
    if state.cc_groups.is_empty() {
        return;
    }
    if let Some(last_group) = state.cc_groups.pop() {
        for (r, c) in last_group.positions {
            if let Some(idx) = state.cc_positions.iter().position(|p| *p == (r, c)) {
                state.cc_positions.remove(idx);
            }
        }
    }
    if state.cc_groups.is_empty() {
        if let Some(units) = &mut state.units_matrix {
            units.data.fill(0);
        }
        return;
    }
    rebuild_cc_matrix(state);
}

pub fn clear_all_cc(state: &mut WizardState) {
    let (h, w) = if let Some(height_map) = &state.height_map {
        (height_map.rows, height_map.cols)
    } else {
        (state.height, state.width)
    };
    state.units_matrix = Some(Matrix::zeros(h, w));
    state.cc_positions.clear();
    state.cc_groups.clear();
}

pub fn run_place_resources_random(state: &mut WizardState) {
    let randomized_matrix = state
        .randomized_matrix
        .clone()
        .unwrap_or_else(|| Matrix::fill(5, 5, 1));
    let height_map = state
        .height_map
        .clone()
        .unwrap_or_else(|| Matrix::fill(state.height, state.width, 1));
    let mut items_matrix = Matrix::zeros(height_map.rows, height_map.cols);
    let resource_positions = add_resource_pulls(
        &randomized_matrix,
        state.num_resource_pulls.max(0) as usize,
        &state.mirroring,
        &height_map,
        &mut items_matrix,
        state.wall_matrix.as_ref(),
        state.units_matrix.as_ref(),
    );
    state.items_matrix = Some(items_matrix);
    state.resource_positions = resource_positions.clone();
    state.resource_groups = vec![resource_positions.clone()];
    state.completed_step = state.completed_step.max(WizardStep::Resources as i32);
}

pub fn run_place_resource_manual(
    state: &mut WizardState,
    row: i32,
    col: i32,
    mirrored: bool,
) -> Vec<(usize, usize)> {
    if row < 0 || col < 0 {
        return Vec::new();
    }
    let row = row as usize;
    let col = col as usize;

    let Some(height_map) = state.height_map.clone() else {
        return Vec::new();
    };
    let Some(randomized_matrix) = state.randomized_matrix.clone() else {
        return Vec::new();
    };
    let h = height_map.rows;
    let w = height_map.cols;
    if row >= h || col >= w {
        return Vec::new();
    }

    let rm_h = randomized_matrix.rows;
    let rm_w = randomized_matrix.cols;
    let rm_row = row * rm_h / h;
    let rm_col = col * rm_w / w;
    if randomized_matrix.get(rm_row, rm_col) != 1 {
        return Vec::new();
    }
    if height_map.get(row, col) <= 0 {
        return Vec::new();
    }
    
    let wall_check = state.wall_matrix.as_ref().map(|m| m.get(row, col) == 1).unwrap_or(false)
        || state.brush_wall_matrix.as_ref().map(|m| m.get(row, col) == 1).unwrap_or(false)
        || state.polygon_depth_matrix.as_ref().map(|m| m.get(row, col) > 0).unwrap_or(false);
    if wall_check {
        return Vec::new();
    }
    
    if let Some(units_matrix) = &state.units_matrix {
        if has_positive_in_radius(units_matrix, row, col, 4) {
            return Vec::new();
        }
    }

    let mirroring = if mirrored {
        state.mirroring.as_str()
    } else {
        "none"
    };
    let mirrored_rm = get_mirrored_positions(rm_row, rm_col, rm_h, rm_w, mirroring);
    let mut placed: Vec<(usize, usize)> = Vec::new();
    for (mr, mc) in mirrored_rm {
        let sr = ((mr * h) / rm_h).min(h - 1);
        let sc = ((mc * w) / rm_w).min(w - 1);
        let too_close = placed
            .iter()
            .any(|(pr, pc)| pr.abs_diff(sr) <= 3 && pc.abs_diff(sc) <= 3);
        if !too_close {
            placed.push((sr, sc));
        }
    }

    if state.items_matrix.is_none() {
        state.items_matrix = Some(Matrix::zeros(h, w));
    }
    if let Some(items_matrix) = &mut state.items_matrix {
        for (sr, sc) in placed.iter().copied() {
            if items_matrix.get(sr, sc) == 0 {
                place_resource_pull(items_matrix, sr, sc);
            }
        }
    }

    state.resource_positions.extend(placed.clone());
    state.resource_groups.push(placed.clone());
    placed
}

pub fn run_remove_resource_manual(state: &mut WizardState, row: i32, col: i32) -> bool {
    if row < 0 || col < 0 || state.items_matrix.is_none() || state.resource_groups.is_empty() {
        return false;
    }
    let row = row as usize;
    let col = col as usize;

    let mut closest_group_idx: Option<usize> = None;
    let mut min_dist = usize::MAX;
    for (i, group) in state.resource_groups.iter().enumerate() {
        for (gr, gc) in group {
            let dist = gr.abs_diff(row).max(gc.abs_diff(col));
            if dist <= 3 && dist < min_dist {
                min_dist = dist;
                closest_group_idx = Some(i);
            }
        }
    }
    let Some(group_idx) = closest_group_idx else {
        return false;
    };
    let removed_group = state.resource_groups.remove(group_idx);
    if let Some(items_matrix) = &mut state.items_matrix {
        for (r, c) in removed_group.iter().copied() {
            clear_pool(items_matrix, r, c);
            if let Some(idx) = state.resource_positions.iter().position(|p| *p == (r, c)) {
                state.resource_positions.remove(idx);
            }
        }
    }
    true
}

pub fn undo_last_resource(state: &mut WizardState) {
    if state.resource_groups.is_empty() {
        return;
    }
    let Some(last_group) = state.resource_groups.pop() else {
        return;
    };
    let Some(items_matrix) = &mut state.items_matrix else {
        return;
    };
    for (r, c) in last_group {
        clear_pool(items_matrix, r, c);
        if let Some(idx) = state.resource_positions.iter().position(|p| *p == (r, c)) {
            state.resource_positions.remove(idx);
        }
    }
}

pub fn clear_all_resources(state: &mut WizardState) {
    let (h, w) = if let Some(height_map) = &state.height_map {
        (height_map.rows, height_map.cols)
    } else {
        (state.height, state.width)
    };
    state.items_matrix = Some(Matrix::zeros(h, w));
    state.resource_positions.clear();
    state.resource_groups.clear();
}

pub fn run_finalize(
    state: &mut WizardState,
    collect_frames: bool,
) -> Result<Vec<QuickGenerateFrame>, String> {
    let mut frames = Vec::new();
    let mut rng = rand::thread_rng();
    let mut height_map = state
        .height_map
        .clone()
        .ok_or_else(|| "Missing height map".to_string())?;
    let h = height_map.rows;
    let w = height_map.cols;

    let mut id_matrix = Matrix::fill(h, w, 83);
    let mut items_matrix = state
        .items_matrix
        .clone()
        .unwrap_or_else(|| Matrix::zeros(h, w));
    let units_matrix = state
        .units_matrix
        .clone()
        .unwrap_or_else(|| Matrix::zeros(h, w));

    let active_wall_matrix = match state.hill_drawing_mode {
        HillDrawingMode::Brush => state.brush_wall_matrix.clone(),
        HillDrawingMode::Polygon => state.polygon_depth_matrix.clone(),
    };
    
    let protection_mask = active_wall_matrix
        .as_ref()
        .filter(|wm| wm.data.iter().any(|v| *v != 0))
        .map(|wm| wall_protection_mask(wm, 3));
    let protected = protection_mask.as_deref();

    for (level, tile_set) in TERRAIN_LEVELS.iter().rev() {
        smooth_terrain_tiles(&mut height_map, &mut id_matrix, *level, tile_set, protected);
        if collect_frames {
            frames.push(make_quick_frame(
                format!("terrain_smooth_{level}"),
                &height_map,
                Some(&id_matrix),
                Some(&items_matrix),
                Some(&units_matrix),
            ));
        }
    }

    add_decoration_tiles(
        &mut id_matrix,
        &height_map,
        &mut rng,
        DECORATION_FREQUENCY,
        &[
            &DECORATION_TILES_LEVEL_1[..],
            &DECORATION_TILES_LEVEL_2[..],
            &DECORATION_TILES_LEVEL_3[..],
            &DECORATION_TILES_LEVEL_4[..],
            &DECORATION_TILES_LEVEL_5[..],
            &DECORATION_TILES_LEVEL_6[..],
            &DECORATION_TILES_LEVEL_7[..],
        ],
    );

    match state.hill_drawing_mode {
        HillDrawingMode::Brush => {
            if let Some(wall_matrix) = &state.brush_wall_matrix {
                if wall_matrix.data.iter().any(|v| *v != 0) {
                    let wall_id_matrix = smooth_wall_tiles(wall_matrix);
                    for r in 0..h {
                        for c in 0..w {
                            let val = wall_id_matrix.get(r, c);
                            if val > 0 {
                                items_matrix.set(r, c, val);
                            }
                        }
                    }
                    if collect_frames {
                        frames.push(make_quick_frame(
                            "wall_tiles",
                            &height_map,
                            Some(&id_matrix),
                            Some(&items_matrix),
                            Some(&units_matrix),
                        ));
                    }
                }
            }
        }
        HillDrawingMode::Polygon => {
            if !state.polygons.is_empty() {
                render_polygon_walls(&state.polygons, &state.mirrored_polygons,
                    state.polygon_depth_matrix.as_ref().unwrap_or(&Matrix::zeros(h, w)),
                    &mut items_matrix, &mut frames, collect_frames, &height_map, &id_matrix, &units_matrix);
            }
        }
    }

    state.id_matrix = Some(id_matrix.clone());
    state.items_matrix = Some(items_matrix.clone());
    state.completed_step = state.completed_step.max(WizardStep::Finalize as i32);

    if collect_frames {
        frames.push(make_quick_frame(
            "terrain_complete",
            &height_map,
            Some(&id_matrix),
            Some(&items_matrix),
            Some(&units_matrix),
        ));
    }
    Ok(frames)
}

/// Render polygon walls based on depth matrix per layer.
fn render_polygon_walls(
    polygons: &[PolygonData],
    mirrored_polygons: &[PolygonData],
    depth_matrix: &Matrix,
    items_matrix: &mut Matrix,
    frames: &mut Vec<QuickGenerateFrame>,
    collect_frames: bool,
    height_map: &Matrix,
    id_matrix: &Matrix,
    units_matrix: &Matrix,
) {
    let h = items_matrix.rows;
    let w = items_matrix.cols;

    // Gather all gap cells up front.
    let mut all_gap_cells = HashSet::new();
    for poly in polygons.iter().chain(mirrored_polygons.iter()) {
        let n = poly.vertices.len();
        for i in 0..n {
            if i < poly.edge_gaps.len() && poly.edge_gaps[i] {
                let [r0, c0] = poly.vertices[i];
                let [r1, c1] = poly.vertices[(i + 1) % n];
                for (r, c) in rasterize_line_cells(r0, c0, r1, c1) {
                    all_gap_cells.insert((r, c));
                }
            }
        }
    }

    let max_depth = depth_matrix.data.iter().copied().max().unwrap_or(0);

    for depth in 1..=max_depth {
        let mut isolated = Matrix::zeros(h, w);
        let mut has_cells = false;

        // Use cumulative footprint so each pass produces one contour band.
        for r in 0..h {
            for c in 0..w {
                if depth_matrix.get(r, c) >= depth {
                    isolated.set(r, c, 1);
                    has_cells = true;
                }
            }
        }

        if !has_cells {
            continue;
        }

        // Punch gap cells as ghost walls (2) to avoid inward wall wrapping.
        for &(r, c) in &all_gap_cells {
            if r < h && c < w {
                isolated.set(r, c, 2);
            }
        }

        let wall_id_matrix = smooth_wall_tiles(&isolated);

        for r in 0..h {
            for c in 0..w {
                let val = wall_id_matrix.get(r, c);
                if val > 0 && is_depth_boundary(depth_matrix, r, c, depth) {
                    items_matrix.set(r, c, val);
                }
            }
        }

        if collect_frames {
            frames.push(make_quick_frame(
                format!("wall_tiles_depth_{depth}"),
                height_map,
                Some(id_matrix),
                Some(items_matrix),
                Some(units_matrix),
            ));
        }
    }
}

fn is_depth_boundary(depth_matrix: &Matrix, r: usize, c: usize, depth: i32) -> bool {
    if depth_matrix.get(r, c) < depth {
        return false;
    }

    // Include diagonals so corner tiles are kept when only diagonal exposure exists.
    let neighbors = [
        (-1isize, 0isize),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ];
    for (dr, dc) in neighbors {
        let nr = r as isize + dr;
        let nc = c as isize + dc;
        if nr < 0 || nc < 0 || nr >= depth_matrix.rows as isize || nc >= depth_matrix.cols as isize {
            return true;
        }
        if depth_matrix.get(nr as usize, nc as usize) < depth {
            return true;
        }
    }
    false
}

pub fn write_tmx(state: &WizardState, blueprint_xml: &str) -> Result<Vec<u8>, String> {
    write_tmx_impl(state, blueprint_xml)
}

pub fn quick_generate(
    state: &mut WizardState,
    blueprint_xml: &str,
) -> Result<(Vec<u8>, Vec<QuickGenerateFrame>), String> {
    let mut quick_frames = Vec::new();

    let coastline_frames = run_coastline(state);
    for frame in coastline_frames {
        let height_map = Matrix::new(
            frame.shape[0] as usize,
            frame.shape[1] as usize,
            frame.data.clone(),
        )
        .map_err(|err| format!("Invalid coastline frame matrix: {err}"))?;
        quick_frames.push(make_quick_frame(frame.label, &height_map, None, None, None));
    }

    run_height_ocean(state, None)?;
    if let Some(height_map) = &state.height_map {
        quick_frames.push(make_quick_frame(
            "height_ocean",
            height_map,
            None,
            None,
            None,
        ));
    }

    run_place_cc_random(state)?;
    if let Some(height_map) = &state.height_map {
        quick_frames.push(make_quick_frame(
            "command_centers",
            height_map,
            None,
            None,
            state.units_matrix.as_ref(),
        ));
    }

    run_place_resources_random(state);
    if let Some(height_map) = &state.height_map {
        quick_frames.push(make_quick_frame(
            "resources",
            height_map,
            None,
            state.items_matrix.as_ref(),
            state.units_matrix.as_ref(),
        ));
    }

    let finalize_frames = run_finalize(state, true)?;
    quick_frames.extend(finalize_frames);
    let tmx = write_tmx(state, blueprint_xml)?;
    Ok((tmx, quick_frames))
}

fn matrix_frame(label: impl Into<String>, matrix: &Matrix) -> CoastlineFrame {
    CoastlineFrame {
        label: label.into(),
        shape: [matrix.rows as i32, matrix.cols as i32],
        data: matrix.data.clone(),
    }
}

fn make_quick_frame(
    label: impl Into<String>,
    height_map: &Matrix,
    id_matrix: Option<&Matrix>,
    items_matrix: Option<&Matrix>,
    units_matrix: Option<&Matrix>,
) -> QuickGenerateFrame {
    QuickGenerateFrame {
        label: label.into(),
        height_map: MatrixPayload::from(height_map),
        id_matrix: id_matrix.map(MatrixPayload::from),
        items_matrix: items_matrix.map(MatrixPayload::from),
        units_matrix: units_matrix.map(MatrixPayload::from),
    }
}

fn default_seed_matrix() -> Matrix {
    Matrix::from_rows(vec![
        vec![0, 0, 1, 0, 0],
        vec![0, 1, 1, 1, 0],
        vec![1, 1, 1, 1, 1],
        vec![0, 1, 1, 1, 0],
        vec![0, 0, 1, 0, 0],
    ])
    .expect("default seed matrix must be valid")
}

// ========== NEW: BRUSH MODE FUNCTIONS ==========

pub fn draw_brush_walls(
    state: &mut WizardState,
    points: &[[i32; 2]],
    value: i32,
    brush_size: usize,
) -> Result<(), String> {
    let h = state.height;
    let w = state.width;
    let coast = state.coastline_height_map.clone();
    let mirroring = state.mirroring.clone();
    
    state.brush_strokes_redo.clear();
    
    state.brush_strokes.push(crate::state::BrushStroke {
        points: points.to_vec(),
        value,
        brush_size,
    });
    
    let matrix = state.ensure_brush_wall_matrix();
    
    for [row_i32, col_i32] in points {
        if *row_i32 < 0 || *col_i32 < 0 {
            continue;
        }
        let row = *row_i32 as usize;
        let col = *col_i32 as usize;

        for (br, bc) in expand_brush(row, col, brush_size, h, w) {
            let mirrored = get_mirrored_positions(br, bc, h, w, &mirroring);
            for (mr, mc) in mirrored {
                if mr >= h || mc >= w {
                    continue;
                }
                if let Some(coastline) = &coast {
                    if coastline.get(mr, mc) <= 0 {
                        continue;
                    }
                }
                if value == 2 && matrix.get(mr, mc) == 0 {
                    continue;
                }
                matrix.set(mr, mc, value);
            }
        }
    }
    
    Ok(())
}

pub fn undo_brush_stroke(state: &mut WizardState) -> Result<(), String> {
    if state.brush_strokes.is_empty() {
        return Ok(());
    }
    
    let stroke = state.brush_strokes.pop().unwrap();
    state.brush_strokes_redo.push(stroke.clone());
    
    rebuild_brush_matrix_from_strokes(state)?;
    Ok(())
}

pub fn redo_brush_stroke(state: &mut WizardState) -> Result<(), String> {
    if state.brush_strokes_redo.is_empty() {
        return Ok(());
    }
    
    let stroke = state.brush_strokes_redo.pop().unwrap();
    
    let h = state.height;
    let w = state.width;
    let coast = state.coastline_height_map.clone();
    let mirroring = state.mirroring.clone();
    let matrix = state.ensure_brush_wall_matrix();
    
    for [row_i32, col_i32] in &stroke.points {
        if *row_i32 < 0 || *col_i32 < 0 {
            continue;
        }
        let row = *row_i32 as usize;
        let col = *col_i32 as usize;

        for (br, bc) in expand_brush(row, col, stroke.brush_size, h, w) {
            let mirrored = get_mirrored_positions(br, bc, h, w, &mirroring);
            for (mr, mc) in mirrored {
                if mr >= h || mc >= w {
                    continue;
                }
                if let Some(coastline) = &coast {
                    if coastline.get(mr, mc) <= 0 {
                        continue;
                    }
                }
                if stroke.value == 2 && matrix.get(mr, mc) == 0 {
                    continue;
                }
                matrix.set(mr, mc, stroke.value);
            }
        }
    }
    
    state.brush_strokes.push(stroke);
    
    Ok(())
}

fn rebuild_brush_matrix_from_strokes(state: &mut WizardState) -> Result<(), String> {
    let h = state.height;
    let w = state.width;
    let coast = state.coastline_height_map.clone();
    let mirroring = state.mirroring.clone();
    
    state.brush_wall_matrix = Some(Matrix::zeros(h, w));
    let matrix = state.brush_wall_matrix.as_mut().unwrap();
    
    for stroke in &state.brush_strokes {
        for [row_i32, col_i32] in &stroke.points {
            if *row_i32 < 0 || *col_i32 < 0 {
                continue;
            }
            let row = *row_i32 as usize;
            let col = *col_i32 as usize;

            for (br, bc) in expand_brush(row, col, stroke.brush_size, h, w) {
                let mirrored = get_mirrored_positions(br, bc, h, w, &mirroring);
                for (mr, mc) in mirrored {
                    if mr >= h || mc >= w {
                        continue;
                    }
                    if let Some(coastline) = &coast {
                        if coastline.get(mr, mc) <= 0 {
                            continue;
                        }
                    }
                    if stroke.value == 2 && matrix.get(mr, mc) == 0 {
                        continue;
                    }
                    matrix.set(mr, mc, stroke.value);
                }
            }
        }
    }
    
    Ok(())
}

pub fn clear_brush_walls(state: &mut WizardState) {
    state.brush_strokes.clear();
    state.brush_strokes_redo.clear();
    state.brush_wall_matrix = Some(Matrix::zeros(state.height, state.width));
}

// ========== NEW: POLYGON MODE FUNCTIONS ==========

pub fn update_polygons(
    state: &mut WizardState,
    polygons: Vec<PolygonData>,
) -> Result<(), String> {
    let h = state.height;
    let w = state.width;
    
    state.polygons_undo.push(state.polygons.clone());
    if state.polygons_undo.len() > 20 {
        state.polygons_undo.remove(0);
    }
    state.polygons_redo.clear();
    
    state.polygons = polygons;
    
    state.mirrored_polygons = state.polygons
        .iter()
        .flat_map(|poly| mirror_polygon_vertices(poly, h, w, &state.mirroring))
        .collect();
    
    let depth_matrix = compute_polygon_depth_matrix(
        &state.polygons,
        &state.mirrored_polygons,
        h,
        w,
        state.coastline_height_map.as_ref(),
    );

    state.polygon_depth_matrix = Some(depth_matrix);
    Ok(())
}

pub fn undo_polygons(state: &mut WizardState) -> Result<(), String> {
    if state.polygons_undo.is_empty() {
        return Ok(());
    }
    
    let previous = state.polygons_undo.pop().unwrap();
    state.polygons_redo.push(state.polygons.clone());
    
    state.polygons = previous;
    
    update_polygons_internal(state)
}

pub fn redo_polygons(state: &mut WizardState) -> Result<(), String> {
    if state.polygons_redo.is_empty() {
        return Ok(());
    }
    
    let next = state.polygons_redo.pop().unwrap();
    state.polygons_undo.push(state.polygons.clone());
    
    state.polygons = next;
    
    update_polygons_internal(state)
}

pub fn clear_all_polygons(state: &mut WizardState) {
    state.polygons_undo.push(state.polygons.clone());
    if state.polygons_undo.len() > 20 {
        state.polygons_undo.remove(0);
    }
    state.polygons.clear();
    state.polygons_redo.clear();
    state.polygon_depth_matrix = Some(Matrix::zeros(state.height, state.width));
    state.mirrored_polygons.clear();
}

pub fn toggle_polygon_edge_gap(
    state: &mut WizardState,
    polygon_id: u32,
    edge_index: usize,
) -> Result<(), String> {
    state.polygons_undo.push(state.polygons.clone());
    if state.polygons_undo.len() > 20 {
        state.polygons_undo.remove(0);
    }
    state.polygons_redo.clear();
    
    if let Some(poly) = state.polygons.iter_mut().find(|p| p.id == polygon_id) {
        if edge_index < poly.edge_gaps.len() {
            poly.edge_gaps[edge_index] = !poly.edge_gaps[edge_index];
        }
    }
    
    update_polygons_internal(state)
}

fn update_polygons_internal(state: &mut WizardState) -> Result<(), String> {
    let h = state.height;
    let w = state.width;
    
    state.mirrored_polygons = state.polygons
        .iter()
        .flat_map(|poly| mirror_polygon_vertices(poly, h, w, &state.mirroring))
        .collect();
    
    let depth_matrix = compute_polygon_depth_matrix(
        &state.polygons,
        &state.mirrored_polygons,
        h,
        w,
        state.coastline_height_map.as_ref(),
    );

    state.polygon_depth_matrix = Some(depth_matrix);
    Ok(())
}

struct Layer {
    cells: Vec<(usize, usize)>,
    #[allow(dead_code)]
    gap_cells: Vec<(usize, usize)>,
    depth: i32,
}

fn compute_polygon_depth_matrix(
    user_polygons: &[PolygonData],
    mirrored_polygons: &[PolygonData],
    rows: usize,
    cols: usize,
    coastline: Option<&Matrix>,
) -> Matrix {
    if user_polygons.is_empty() && mirrored_polygons.is_empty() {
        return Matrix::zeros(rows, cols);
    }
    
    // Combine ALL polygons so mirrored copies that touch user copies merge properly
    let mut all_polygons = Vec::new();
    all_polygons.extend_from_slice(user_polygons);
    all_polygons.extend_from_slice(mirrored_polygons);

    // 1. Group by intersection (union-find) for ALL polygons together
    let groups = group_intersecting_polygons(&all_polygons);

    // 2. Build layers
    let mut all_layers = build_layers_from_groups(&groups, &all_polygons, rows, cols);

    if all_layers.is_empty() {
        return Matrix::zeros(rows, cols);
    }

    // Sort by area (smaller first for containment check)
    all_layers.sort_by_key(|l| l.cells.len());

    // Assign depths based on containment - process LARGEST to smallest
    for i in (0..all_layers.len()).rev() {
        let mut max_parent_depth = 0;
        for j in (i + 1)..all_layers.len() {
            if is_layer_contained(&all_layers[i], &all_layers[j]) {
                max_parent_depth = max_parent_depth.max(all_layers[j].depth);
            }
        }
        all_layers[i].depth = (max_parent_depth + 1).min(9);
    }

    // 3. Fill matrix (skip ocean cells)
    let mut matrix = Matrix::zeros(rows, cols);
    for layer in all_layers {
        for (r, c) in layer.cells {
            if r < rows && c < cols {
                if let Some(coast) = coastline {
                    if coast.get(r, c) <= 0 {
                        continue;
                    }
                }
                let current = matrix.get(r, c);
                matrix.set(r, c, current.max(layer.depth));
            }
        }
    }

    matrix
}

fn group_intersecting_polygons(polygons: &[PolygonData]) -> Vec<Vec<usize>> {
    if polygons.is_empty() {
        return Vec::new();
    }
    
    let n = polygons.len();
    let mut parent: Vec<usize> = (0..n).collect();
    
    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }
    
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let pa = find(parent, a);
        let pb = find(parent, b);
        if pa != pb {
            parent[pa] = pb;
        }
    }
    
    for i in 0..n {
        for j in (i + 1)..n {
            if polygons_intersect(&polygons[i], &polygons[j]) {
                union(&mut parent, i, j);
            }
        }
    }
    
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(i);
    }
    
    groups.values().cloned().collect()
}

fn polygons_intersect(a: &PolygonData, b: &PolygonData) -> bool {
    // Quick bbox check first
    let (min_a_r, max_a_r, min_a_c, max_a_c) = bbox(&a.vertices);
    let (min_b_r, max_b_r, min_b_c, max_b_c) = bbox(&b.vertices);
    
    if max_a_r < min_b_r || max_b_r < min_a_r || max_a_c < min_b_c || max_b_c < min_a_c {
        return false;
    }
    
    // Check edge intersections ONLY.
    // If one is fully inside another, edges don't intersect -> separate groups -> stack as layers.
    let edges_a = get_edges(&a.vertices);
    let edges_b = get_edges(&b.vertices);
    
    for ea in &edges_a {
        for eb in &edges_b {
            if edges_intersect(ea, eb) {
                return true;
            }
        }
    }
    
    false
}

fn bbox(vertices: &[[i32; 2]]) -> (i32, i32, i32, i32) {
    let mut min_r = i32::MAX;
    let mut max_r = i32::MIN;
    let mut min_c = i32::MAX;
    let mut max_c = i32::MIN;
    
    for &[r, c] in vertices {
        min_r = min_r.min(r);
        max_r = max_r.max(r);
        min_c = min_c.min(c);
        max_c = max_c.max(c);
    }
    
    (min_r, max_r, min_c, max_c)
}

fn get_edges(vertices: &[[i32; 2]]) -> Vec<[[i32; 2]; 2]> {
    let mut edges = Vec::new();
    let n = vertices.len();
    for i in 0..n {
        edges.push([vertices[i], vertices[(i + 1) % n]]);
    }
    edges
}

fn edges_intersect(e1: &[[i32; 2]; 2], e2: &[[i32; 2]; 2]) -> bool {
    let [a1, a2] = e1;
    let [b1, b2] = e2;
    
    let d1 = direction(b1, b2, a1);
    let d2 = direction(b1, b2, a2);
    let d3 = direction(a1, a2, b1);
    let d4 = direction(a1, a2, b2);
    
    if ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0)) {
        return true;
    }
    
    if d1 == 0 && on_segment(b1, b2, a1) { return true; }
    if d2 == 0 && on_segment(b1, b2, a2) { return true; }
    if d3 == 0 && on_segment(a1, a2, b1) { return true; }
    if d4 == 0 && on_segment(a1, a2, b2) { return true; }
    
    false
}

fn direction(a: &[i32; 2], b: &[i32; 2], c: &[i32; 2]) -> i32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
}

fn on_segment(s1: &[i32; 2], s2: &[i32; 2], p: &[i32; 2]) -> bool {
    let min_r = s1[0].min(s2[0]);
    let max_r = s1[0].max(s2[0]);
    let min_c = s1[1].min(s2[1]);
    let max_c = s1[1].max(s2[1]);
    
    p[0] >= min_r && p[0] <= max_r && p[1] >= min_c && p[1] <= max_c
}

fn build_layers_from_groups(
    groups: &[Vec<usize>],
    polygons: &[PolygonData],
    rows: usize,
    cols: usize,
) -> Vec<Layer> {
    groups.iter().map(|group| {
        let mut all_cells = HashSet::new();
        let mut all_gap_cells = HashSet::new();
        
        for &idx in group {
            let poly = &polygons[idx];
            
            let cells = fill_polygon_scanline(&poly.vertices, rows, cols);
            all_cells.extend(cells);
            
            for (i, is_gap) in poly.edge_gaps.iter().enumerate() {
                if *is_gap {
                    let r0 = poly.vertices[i][0];
                    let c0 = poly.vertices[i][1];
                    let r1 = poly.vertices[(i + 1) % poly.vertices.len()][0];
                    let c1 = poly.vertices[(i + 1) % poly.vertices.len()][1];
                    
                    let gap_cells = rasterize_line_cells(r0, c0, r1, c1);
                    all_gap_cells.extend(gap_cells);
                }
            }
        }
        
        let final_cells: Vec<_> = all_cells
            .difference(&all_gap_cells)
            .copied()
            .collect();
        
        Layer {
            cells: final_cells,
            gap_cells: all_gap_cells.into_iter().collect(),
            depth: 1,
        }
    }).collect()
}

fn is_layer_contained(inner: &Layer, outer: &Layer) -> bool {
    if inner.cells.is_empty() || outer.cells.is_empty() {
        return false;
    }
    
    let outer_set: HashSet<_> = outer.cells.iter().copied().collect();
    
    let samples = if inner.cells.len() <= 5 {
        &inner.cells[..]
    } else {
        let step = inner.cells.len() / 5;
        &[
            inner.cells[0],
            inner.cells[step],
            inner.cells[step * 2],
            inner.cells[step * 3],
            inner.cells[step * 4],
        ]
    };
    
    let in_outer: usize = samples.iter()
        .filter(|(r, c)| outer_set.contains(&(*r, *c)))
        .count();
    
    in_outer > samples.len() / 2
}
