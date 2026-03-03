mod algorithms;
mod pipeline;
mod state;
mod tiles;

use std::cell::RefCell;

use pipeline::{CoastlineFrame, QuickGenerateFrame};
use serde::Serialize;
use serde_json::Value;
use state::{HillDrawingMode, Matrix, PolygonData, WizardSnapshot, WizardState, WizardStep};
use wasm_bindgen::prelude::*;

thread_local! {
    static STATE: RefCell<WizardState> = RefCell::new(WizardState::default());
}

#[derive(Serialize)]
struct RpcResponse {
    snapshot: WizardSnapshot,
    #[serde(skip_serializing_if = "Option::is_none")]
    tmx_bytes: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frames: Option<Vec<CoastlineFrame>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quick_frames: Option<Vec<QuickGenerateFrame>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    placed: Option<Vec<[i32; 2]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    removed: Option<bool>,
}

#[wasm_bindgen]
pub fn rpc_call(method: &str, params_json: &str) -> Result<JsValue, JsValue> {
    let params = parse_params(params_json)?;
    let result = STATE.with(|cell| -> Result<RpcResponse, String> {
        let mut state = cell.borrow_mut();
        match method {
            "run_coastline" => {
                apply_common_generation_params(&mut state, &params);
                state.invalidate_from(WizardStep::Coastline);
                let frames = pipeline::run_coastline(&mut state);
                Ok(RpcResponse {
                    snapshot: state.snapshot(),
                    tmx_bytes: None,
                    frames: Some(frames),
                    quick_frames: None,
                    placed: None,
                    removed: None,
                })
            }
            "draw_walls" => {
                let points = parse_points(&params, "points");
                let value = get_i32(&params, &["value"]).unwrap_or(1);
                let brush_size = get_usize(&params, &["brush_size"]).unwrap_or(1).max(1);
                pipeline::draw_walls(&mut state, &points, value, brush_size)?;
                Ok(snapshot_only(&state))
            }
            "clear_walls" => {
                pipeline::clear_walls(&mut state);
                Ok(snapshot_only(&state))
            }
            "set_wall_cells" => {
                let cells = parse_points(&params, "cells");
                let value = get_i32(&params, &["value"]).unwrap_or(1);
                pipeline::set_wall_cells(&mut state, &cells, value)?;
                Ok(snapshot_only(&state))
            }
            "set_polygon_walls" => {
                let polygons = parse_polygon_data(&params);
                pipeline::set_polygon_walls(&mut state, polygons)?;
                Ok(snapshot_only(&state))
            }
            // ========== NEW: BRUSH MODE ACTIONS ==========
            "draw_brush_walls" => {
                let points = parse_points(&params, "points");
                let value = get_i32(&params, &["value"]).unwrap_or(1);
                let brush_size = get_usize(&params, &["brush_size"]).unwrap_or(1).max(1);
                pipeline::draw_brush_walls(&mut state, &points, value, brush_size)?;
                Ok(snapshot_only(&state))
            }
            "undo_brush" => {
                pipeline::undo_brush_stroke(&mut state)?;
                Ok(snapshot_only(&state))
            }
            "redo_brush" => {
                pipeline::redo_brush_stroke(&mut state)?;
                Ok(snapshot_only(&state))
            }
            "clear_brush_walls" => {
                pipeline::clear_brush_walls(&mut state);
                Ok(snapshot_only(&state))
            }
            // ========== NEW: POLYGON MODE ACTIONS ==========
            "update_polygons" => {
                let polygons = parse_polygon_data_with_id(&params);
                pipeline::update_polygons(&mut state, polygons)?;
                Ok(snapshot_only(&state))
            }
            "undo_polygons" => {
                pipeline::undo_polygons(&mut state)?;
                Ok(snapshot_only(&state))
            }
            "redo_polygons" => {
                pipeline::redo_polygons(&mut state)?;
                Ok(snapshot_only(&state))
            }
            "clear_all_polygons" => {
                pipeline::clear_all_polygons(&mut state);
                Ok(snapshot_only(&state))
            }
            "toggle_edge_gap" => {
                let polygon_id = get_u32(&params, &["polygon_id"]).unwrap_or(0);
                let edge_index = get_usize(&params, &["edge_index"]).unwrap_or(0);
                pipeline::toggle_polygon_edge_gap(&mut state, polygon_id, edge_index)?;
                Ok(snapshot_only(&state))
            }
            "set_hill_drawing_mode" => {
                let mode = params.get("mode").and_then(Value::as_str).unwrap_or("brush");
                state.hill_drawing_mode = match mode {
                    "polygon" => HillDrawingMode::Polygon,
                    _ => HillDrawingMode::Brush,
                };
                Ok(snapshot_only(&state))
            }
            "run_height_ocean" => {
                if let Some(levels) = get_i32(&params, &["heightLevels", "num_height_levels"]) {
                    state.num_height_levels = levels;
                }
                if let Some(levels) = get_i32(&params, &["oceanLevels", "num_ocean_levels"]) {
                    state.num_ocean_levels = levels;
                }
                let seed = get_i32(&params, &["seed"]);
                state.invalidate_from(WizardStep::HeightOcean);
                pipeline::run_height_ocean(&mut state, seed)?;
                Ok(snapshot_only(&state))
            }
            "place_cc_manual" => {
                let row = get_i32(&params, &["row"]).unwrap_or(-1);
                let col = get_i32(&params, &["col"]).unwrap_or(-1);
                let mirrored = get_bool(&params, &["mirrored"]).unwrap_or(true);
                let placed = pipeline::run_place_cc_manual(&mut state, row, col, mirrored)
                    .into_iter()
                    .map(|(r, c)| [r as i32, c as i32])
                    .collect::<Vec<_>>();
                Ok(RpcResponse {
                    snapshot: state.snapshot(),
                    tmx_bytes: None,
                    frames: None,
                    quick_frames: None,
                    placed: Some(placed),
                    removed: None,
                })
            }
            "remove_cc_manual" => {
                let row = get_i32(&params, &["row"]).unwrap_or(-1);
                let col = get_i32(&params, &["col"]).unwrap_or(-1);
                let removed = pipeline::run_remove_cc_manual(&mut state, row, col);
                Ok(RpcResponse {
                    snapshot: state.snapshot(),
                    tmx_bytes: None,
                    frames: None,
                    quick_frames: None,
                    placed: None,
                    removed: Some(removed),
                })
            }
            "place_cc_random" => {
                if let Some(players) = get_i32(&params, &["numPlayers", "num_command_centers"]) {
                    state.num_command_centers = players;
                }
                pipeline::clear_all_cc(&mut state);
                pipeline::run_place_cc_random(&mut state)?;
                Ok(snapshot_only(&state))
            }
            "undo_cc" => {
                pipeline::undo_last_cc(&mut state);
                Ok(snapshot_only(&state))
            }
            "clear_cc" => {
                pipeline::clear_all_cc(&mut state);
                Ok(snapshot_only(&state))
            }
            "place_resource_manual" => {
                let row = get_i32(&params, &["row"]).unwrap_or(-1);
                let col = get_i32(&params, &["col"]).unwrap_or(-1);
                let mirrored = get_bool(&params, &["mirrored"]).unwrap_or(true);
                let placed = pipeline::run_place_resource_manual(&mut state, row, col, mirrored)
                    .into_iter()
                    .map(|(r, c)| [r as i32, c as i32])
                    .collect::<Vec<_>>();
                Ok(RpcResponse {
                    snapshot: state.snapshot(),
                    tmx_bytes: None,
                    frames: None,
                    quick_frames: None,
                    placed: Some(placed),
                    removed: None,
                })
            }
            "remove_resource_manual" => {
                let row = get_i32(&params, &["row"]).unwrap_or(-1);
                let col = get_i32(&params, &["col"]).unwrap_or(-1);
                let removed = pipeline::run_remove_resource_manual(&mut state, row, col);
                Ok(RpcResponse {
                    snapshot: state.snapshot(),
                    tmx_bytes: None,
                    frames: None,
                    quick_frames: None,
                    placed: None,
                    removed: Some(removed),
                })
            }
            "place_resource_random" => {
                if let Some(resources) = get_i32(&params, &["numResources", "num_resource_pulls"]) {
                    state.num_resource_pulls = resources;
                }
                pipeline::clear_all_resources(&mut state);
                pipeline::run_place_resources_random(&mut state);
                Ok(snapshot_only(&state))
            }
            "undo_resource" => {
                pipeline::undo_last_resource(&mut state);
                Ok(snapshot_only(&state))
            }
            "clear_resource" => {
                pipeline::clear_all_resources(&mut state);
                Ok(snapshot_only(&state))
            }
            "get_state_snapshot" => Ok(snapshot_only(&state)),
            "run_finalize" => {
                let blueprint_xml = get_blueprint_xml(&params)
                    .ok_or_else(|| "Missing blueprintXml for finalize".to_string())?;
                pipeline::run_finalize(&mut state, false)?;
                let tmx = pipeline::write_tmx(&state, &blueprint_xml)?;
                Ok(RpcResponse {
                    snapshot: state.snapshot(),
                    tmx_bytes: Some(tmx),
                    frames: None,
                    quick_frames: None,
                    placed: None,
                    removed: None,
                })
            }
            "quick_generate" => {
                let blueprint_xml = get_blueprint_xml(&params)
                    .ok_or_else(|| "Missing blueprintXml for quick_generate".to_string())?;
                *state = WizardState::default();
                apply_common_generation_params(&mut state, &params);
                let (tmx, quick_frames) = pipeline::quick_generate(&mut state, &blueprint_xml)?;
                Ok(RpcResponse {
                    snapshot: state.snapshot(),
                    tmx_bytes: Some(tmx),
                    frames: None,
                    quick_frames: Some(quick_frames),
                    placed: None,
                    removed: None,
                })
            }
            "reset_state" => {
                *state = WizardState::default();
                Ok(snapshot_only(&state))
            }
            _ => Err(format!("Unknown RPC method: {method}")),
        }
    });

    match result {
        Ok(response) => to_js_value(&response),
        Err(err) => Err(JsValue::from_str(&err)),
    }
}

fn snapshot_only(state: &WizardState) -> RpcResponse {
    RpcResponse {
        snapshot: state.snapshot(),
        tmx_bytes: None,
        frames: None,
        quick_frames: None,
        placed: None,
        removed: None,
    }
}

fn parse_params(params_json: &str) -> Result<Value, JsValue> {
    if params_json.trim().is_empty() {
        return Ok(Value::Object(serde_json::Map::new()));
    }
    serde_json::from_str::<Value>(params_json)
        .map_err(|err| JsValue::from_str(&format!("Invalid params JSON: {err}")))
}

fn apply_common_generation_params(state: &mut WizardState, params: &Value) {
    if let Some(grid) = params.get("grid").and_then(matrix_from_grid_value) {
        state.initial_matrix = Some(grid);
    }
    if let Some(height) = get_usize(params, &["height"]) {
        state.height = height.max(1);
    }
    if let Some(width) = get_usize(params, &["width"]) {
        state.width = width.max(1);
    }
    if let Some(mirroring) = params.get("mirroring").and_then(Value::as_str) {
        state.mirroring = mirroring.to_string();
    }
    if let Some(pattern) = get_i32(params, &["tileset", "pattern"]) {
        state.pattern = pattern;
    }
    if let Some(levels) = get_i32(params, &["heightLevels", "num_height_levels"]) {
        state.num_height_levels = levels;
    }
    if let Some(levels) = get_i32(params, &["oceanLevels", "num_ocean_levels"]) {
        state.num_ocean_levels = levels;
    }
    if let Some(players) = get_i32(params, &["numPlayers", "num_command_centers"]) {
        state.num_command_centers = players;
    }
    if let Some(resources) = get_i32(params, &["numResources", "num_resource_pulls"]) {
        state.num_resource_pulls = resources;
    }
}

fn parse_points(params: &Value, key: &str) -> Vec<[i32; 2]> {
    let Some(points) = params.get(key).and_then(Value::as_array) else {
        return Vec::new();
    };
    points
        .iter()
        .filter_map(|pt| {
            let pair = pt.as_array()?;
            if pair.len() != 2 {
                return None;
            }
            let r = pair[0].as_i64()? as i32;
            let c = pair[1].as_i64()? as i32;
            Some([r, c])
        })
        .collect()
}

fn parse_polygon_data(params: &Value) -> Vec<PolygonData> {
    let Some(polys) = params.get("polygons").and_then(Value::as_array) else {
        return Vec::new();
    };
    polys
        .iter()
        .filter_map(|poly| {
            let vertices_arr = poly.get("vertices")?.as_array()?;
            let vertices: Vec<[i32; 2]> = vertices_arr
                .iter()
                .filter_map(|pt| {
                    let pair = pt.as_array()?;
                    if pair.len() != 2 {
                        return None;
                    }
                    Some([pair[0].as_i64()? as i32, pair[1].as_i64()? as i32])
                })
                .collect();
            if vertices.len() < 3 {
                return None;
            }
            let edge_gaps = poly
                .get("edgeGaps")
                .and_then(Value::as_array)
                .map(|arr| arr.iter().map(|v| v.as_bool().unwrap_or(false)).collect())
                .unwrap_or_else(|| vec![false; vertices.len()]);
            Some(PolygonData {
                id: 0, // Default ID
                vertices,
                edge_gaps,
            })
        })
        .collect()
}

fn parse_polygon_data_with_id(params: &Value) -> Vec<PolygonData> {
    let Some(polys) = params.get("polygons").and_then(Value::as_array) else {
        return Vec::new();
    };
    polys
        .iter()
        .filter_map(|poly| {
            let id = poly
                .get("id")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            let vertices_arr = poly.get("vertices")?.as_array()?;
            let vertices: Vec<[i32; 2]> = vertices_arr
                .iter()
                .filter_map(|pt| {
                    let pair = pt.as_array()?;
                    if pair.len() != 2 {
                        return None;
                    }
                    Some([pair[0].as_i64()? as i32, pair[1].as_i64()? as i32])
                })
                .collect();
            if vertices.len() < 3 {
                return None;
            }
            let edge_gaps = poly
                .get("edgeGaps")
                .and_then(Value::as_array)
                .map(|arr| arr.iter().map(|v| v.as_bool().unwrap_or(false)).collect())
                .unwrap_or_else(|| vec![false; vertices.len()]);
            Some(PolygonData {
                id,
                vertices,
                edge_gaps,
            })
        })
        .collect()
}

fn get_u32(params: &Value, keys: &[&str]) -> Option<u32> {
    get_i64(params, keys).and_then(|value| u32::try_from(value).ok())
}

fn matrix_from_grid_value(value: &Value) -> Option<Matrix> {
    let rows = value.as_array()?;
    if rows.is_empty() {
        return None;
    }
    let mut parsed: Vec<Vec<i32>> = Vec::with_capacity(rows.len());
    for row in rows {
        let row_array = row.as_array()?;
        let mut parsed_row = Vec::with_capacity(row_array.len());
        for cell in row_array {
            let value = cell.as_i64()?;
            parsed_row.push(value as i32);
        }
        parsed.push(parsed_row);
    }
    Matrix::from_rows(parsed)
}

fn get_blueprint_xml(params: &Value) -> Option<String> {
    params
        .get("blueprintXml")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .or_else(|| {
            params
                .get("blueprint_xml")
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
}

fn get_bool(params: &Value, keys: &[&str]) -> Option<bool> {
    for key in keys {
        if let Some(value) = params.get(*key) {
            if let Some(boolean) = value.as_bool() {
                return Some(boolean);
            }
            if let Some(number) = value.as_i64() {
                return Some(number != 0);
            }
            if let Some(text) = value.as_str() {
                if text.eq_ignore_ascii_case("true") {
                    return Some(true);
                }
                if text.eq_ignore_ascii_case("false") {
                    return Some(false);
                }
            }
        }
    }
    None
}

fn get_usize(params: &Value, keys: &[&str]) -> Option<usize> {
    get_i64(params, keys).and_then(|value| usize::try_from(value).ok())
}

fn get_i32(params: &Value, keys: &[&str]) -> Option<i32> {
    get_i64(params, keys).and_then(|value| i32::try_from(value).ok())
}

fn get_i64(params: &Value, keys: &[&str]) -> Option<i64> {
    for key in keys {
        if let Some(value) = params.get(*key) {
            if let Some(number) = value.as_i64() {
                return Some(number);
            }
            if let Some(number) = value.as_f64() {
                return Some(number as i64);
            }
            if let Some(text) = value.as_str() {
                if let Ok(parsed) = text.parse::<i64>() {
                    return Some(parsed);
                }
            }
        }
    }
    None
}

fn to_js_value<T: Serialize>(value: &T) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(value)
        .map_err(|err| JsValue::from_str(&format!("Failed to serialize RPC response: {err}")))
}
