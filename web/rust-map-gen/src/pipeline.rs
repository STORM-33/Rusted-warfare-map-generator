use std::collections::HashSet;

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
use crate::algorithms::terrain::{bias_terrain_near_walls, generate_level};
use crate::algorithms::tmx::write_tmx as write_tmx_impl;
use crate::state::{CcGroup, Matrix, MatrixPayload, WizardState, WizardStep};
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

    if let Some(wall_matrix) = &state.wall_matrix {
        if wall_matrix.data.iter().any(|v| *v == 1) {
            height_map = bias_terrain_near_walls(&height_map, wall_matrix, state.num_height_levels);
        }
    }

    state.height_map = Some(height_map);
    state.completed_step = state.completed_step.max(WizardStep::HeightOcean as i32);
    Ok(())
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
    if let Some(wall_matrix) = &state.wall_matrix {
        if wall_matrix.get(row, col) == 1 {
            return Vec::new();
        }
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

    let max_ccs = 10usize;
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

    let mut used_ids = HashSet::new();
    for g in &state.cc_groups {
        used_ids.insert(g.id);
        if g.positions.len() > 1 {
            used_ids.insert(g.id + 5);
        }
    }

    let team_a_id = if placed.len() > 1 {
        let mut selected = 101;
        for candidate in 101..=105 {
            if !used_ids.contains(&candidate) && !used_ids.contains(&(candidate + 5)) {
                selected = candidate;
                break;
            }
        }
        selected
    } else {
        let id_to_team = [
            (101, 1),
            (106, 2),
            (102, 3),
            (107, 4),
            (103, 5),
            (108, 6),
            (104, 7),
            (109, 8),
            (105, 9),
            (110, 10),
        ];
        let team_to_id = [
            (1, 101),
            (2, 106),
            (3, 102),
            (4, 107),
            (5, 103),
            (6, 108),
            (7, 104),
            (8, 109),
            (9, 105),
            (10, 110),
        ];
        let mut used_teams = HashSet::new();
        for id in used_ids {
            if let Some((_, team)) = id_to_team.iter().find(|(cc_id, _)| *cc_id == id) {
                used_teams.insert(*team);
            }
        }
        let mut selected = 101;
        for team in 1..=10 {
            if !used_teams.contains(&team) {
                if let Some((_, id)) = team_to_id.iter().find(|(t, _)| *t == team) {
                    selected = *id;
                }
                break;
            }
        }
        selected
    };

    state.cc_positions.extend(placed.clone());
    state.cc_groups.push(CcGroup {
        id: team_a_id,
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
    state.resource_groups = vec![resource_positions];
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
    if let Some(wall_matrix) = &state.wall_matrix {
        if wall_matrix.get(row, col) == 1 {
            return Vec::new();
        }
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

    for (level, tile_set) in TERRAIN_LEVELS.iter().rev() {
        smooth_terrain_tiles(&mut height_map, &mut id_matrix, *level, tile_set);
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

    if let Some(wall_matrix) = &state.wall_matrix {
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
