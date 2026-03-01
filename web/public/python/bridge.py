import json
from typing import Dict, Any

import numpy as np

import map_pipeline
from wizard_state import WizardState, WizardStep
from procedural_map_generator_functions import _get_mirrored_positions


state = WizardState()


def _matrix_payload(matrix):
    if matrix is None:
        return None
    arr = np.asarray(matrix)
    return {
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "data": arr.astype(int, copy=False).ravel().tolist(),
    }


def _snapshot() -> Dict[str, Any]:
    return {
        "meta": {
            "height": int(state.height),
            "width": int(state.width),
            "mirroring": state.mirroring,
            "pattern": int(state.pattern),
            "num_height_levels": int(state.num_height_levels),
            "num_ocean_levels": int(state.num_ocean_levels),
            "num_command_centers": int(state.num_command_centers),
            "num_resource_pulls": int(state.num_resource_pulls),
            "completed_step": int(state.completed_step),
            "current_step": int(state.current_step),
        },
        "cc_positions": [[int(r), int(c)] for r, c in state.cc_positions],
        "resource_positions": [[int(r), int(c)] for r, c in state.resource_positions],
        "matrices": {
            "coastline_height_map": _matrix_payload(state.coastline_height_map),
            "wall_matrix": _matrix_payload(state.wall_matrix),
            "height_map": _matrix_payload(state.height_map),
            "id_matrix": _matrix_payload(state.id_matrix),
            "items_matrix": _matrix_payload(state.items_matrix),
            "units_matrix": _matrix_payload(state.units_matrix),
        },
    }


def _params(params_json):
    if params_json is None:
        return {}
    if isinstance(params_json, dict):
        return params_json
    return json.loads(params_json or "{}")


def run_coastline(params_json="{}"):
    params = _params(params_json)
    state.initial_matrix = params.get("grid", state.initial_matrix)
    state.height = int(params.get("height", state.height))
    state.width = int(params.get("width", state.width))
    state.mirroring = str(params.get("mirroring", state.mirroring))
    state.pattern = int(params.get("tileset", params.get("pattern", state.pattern)))
    state.num_height_levels = int(params.get("heightLevels", params.get("num_height_levels", state.num_height_levels)))
    state.num_ocean_levels = int(params.get("oceanLevels", params.get("num_ocean_levels", state.num_ocean_levels)))
    state.num_command_centers = int(params.get("numPlayers", params.get("num_command_centers", state.num_command_centers)))
    state.num_resource_pulls = int(params.get("numResources", params.get("num_resource_pulls", state.num_resource_pulls)))
    state.invalidate_from(WizardStep.COASTLINE)

    frames = []

    def collect_frame(label, matrix):
        frames.append({
            "label": label,
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "data": matrix.astype(int, copy=False).ravel().tolist(),
        })

    map_pipeline.run_coastline(state, preview_cb=collect_frame)
    return {"snapshot": _snapshot(), "frames": frames}


def _expand_brush(row, col, brush_size, h, w):
    """Expand a single point into a square brush area, like the EXE version."""
    r = brush_size // 2
    pts = []
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            nr, nc = row + dr, col + dc
            if 0 <= nr < h and 0 <= nc < w:
                pts.append((nr, nc))
    return pts


def draw_walls(params_json="{}"):
    params = _params(params_json)
    points = params.get("points", [])
    value = int(params.get("value", 1))
    brush_size = int(params.get("brush_size", 1))
    h, w = int(state.height), int(state.width)
    if state.wall_matrix is None or tuple(state.wall_matrix.shape) != (h, w):
        state.wall_matrix = np.zeros((h, w), dtype=int)

    coast = state.coastline_height_map

    for pt in points:
        row, col = int(pt[0]), int(pt[1])
        brush_pts = _expand_brush(row, col, brush_size, h, w)
        for br, bc in brush_pts:
            mirrored = _get_mirrored_positions(br, bc, h, w, state.mirroring)
            for mr, mc in mirrored:
                if 0 <= mr < h and 0 <= mc < w:
                    if coast is not None and coast[mr, mc] <= 0:
                        continue
                    if value == 2 and state.wall_matrix[mr, mc] == 0:
                        continue
                    state.wall_matrix[mr, mc] = value

    return {"snapshot": _snapshot()}


def clear_walls(params_json="{}"):
    h, w = int(state.height), int(state.width)
    state.wall_matrix = np.zeros((h, w), dtype=int)
    return {"snapshot": _snapshot()}


def run_height_ocean(params_json="{}"):
    params = _params(params_json)
    state.num_height_levels = int(params.get("heightLevels", params.get("num_height_levels", state.num_height_levels)))
    state.num_ocean_levels = int(params.get("oceanLevels", params.get("num_ocean_levels", state.num_ocean_levels)))
    seed = params.get("seed")
    state.invalidate_from(WizardStep.HEIGHT_OCEAN)
    map_pipeline.run_height_ocean(state, seed=seed)
    return {"snapshot": _snapshot()}


def place_cc_manual(params_json="{}"):
    params = _params(params_json)
    row = int(params.get("row", -1))
    col = int(params.get("col", -1))
    mirrored = bool(params.get("mirrored", True))
    placed = map_pipeline.run_place_cc_manual(state, row, col, mirrored)
    return {"placed": [[int(r), int(c)] for r, c in placed], "snapshot": _snapshot()}

def remove_cc_manual(params_json="{}"):
    params = _params(params_json)
    row = int(params.get("row", -1))
    col = int(params.get("col", -1))
    removed = map_pipeline.run_remove_cc_manual(state, row, col)
    return {"removed": removed, "snapshot": _snapshot()}


def place_cc_random(params_json="{}"):
    params = _params(params_json)
    state.num_command_centers = int(params.get("numPlayers", params.get("num_command_centers", state.num_command_centers)))
    map_pipeline.clear_all_cc(state)
    map_pipeline.run_place_cc_random(state)
    return {"snapshot": _snapshot()}


def undo_cc(params_json="{}"):
    map_pipeline.undo_last_cc(state)
    return {"snapshot": _snapshot()}


def clear_cc(params_json="{}"):
    map_pipeline.clear_all_cc(state)
    return {"snapshot": _snapshot()}


def place_resource_manual(params_json="{}"):
    params = _params(params_json)
    row = int(params.get("row", -1))
    col = int(params.get("col", -1))
    mirrored = bool(params.get("mirrored", True))
    placed = map_pipeline.run_place_resource_manual(state, row, col, mirrored)
    return {"placed": [[int(r), int(c)] for r, c in placed], "snapshot": _snapshot()}

def remove_resource_manual(params_json="{}"):
    params = _params(params_json)
    row = int(params.get("row", -1))
    col = int(params.get("col", -1))
    removed = map_pipeline.run_remove_resource_manual(state, row, col)
    return {"removed": removed, "snapshot": _snapshot()}


def place_resource_random(params_json="{}"):
    params = _params(params_json)
    state.num_resource_pulls = int(params.get("numResources", params.get("num_resource_pulls", state.num_resource_pulls)))
    map_pipeline.clear_all_resources(state)
    map_pipeline.run_place_resources_random(state)
    return {"snapshot": _snapshot()}


def undo_resource(params_json="{}"):
    map_pipeline.undo_last_resource(state)
    return {"snapshot": _snapshot()}


def clear_resource(params_json="{}"):
    map_pipeline.clear_all_resources(state)
    return {"snapshot": _snapshot()}


def get_state_snapshot(params_json="{}"):
    return {"snapshot": _snapshot()}


def run_finalize(params_json="{}"):
    params = _params(params_json)
    blueprint_xml = params.get("blueprintXml") or params.get("blueprint_xml")
    if not blueprint_xml:
        raise ValueError("Missing blueprintXml for finalize")
    map_pipeline.run_finalize(state)
    tmx_bytes = map_pipeline.write_tmx(state, blueprint_xml)
    return {"tmx_bytes": tmx_bytes, "snapshot": _snapshot()}


def quick_generate(params_json="{}"):
    """Run the entire map pipeline in one call with preview frames — matching the exe Quick Generate."""
    global state
    params = _params(params_json)

    state = WizardState()
    state.initial_matrix = params.get("grid", state.initial_matrix)
    state.height = int(params.get("height", state.height))
    state.width = int(params.get("width", state.width))
    state.mirroring = str(params.get("mirroring", state.mirroring))
    state.pattern = int(params.get("tileset", params.get("pattern", state.pattern)))
    state.num_height_levels = int(params.get("heightLevels", params.get("num_height_levels", state.num_height_levels)))
    state.num_ocean_levels = int(params.get("oceanLevels", params.get("num_ocean_levels", state.num_ocean_levels)))
    state.num_command_centers = int(params.get("numPlayers", params.get("num_command_centers", state.num_command_centers)))
    state.num_resource_pulls = int(params.get("numResources", params.get("num_resource_pulls", state.num_resource_pulls)))

    blueprint_xml = params.get("blueprintXml") or params.get("blueprint_xml")
    if not blueprint_xml:
        raise ValueError("Missing blueprintXml for quick_generate")

    quick_frames = []

    def _frame(label, height_map, id_matrix=None, items_matrix=None, units_matrix=None):
        f = {"label": label, "height_map": _matrix_payload(height_map)}
        if id_matrix is not None:
            f["id_matrix"] = _matrix_payload(id_matrix)
        if items_matrix is not None:
            f["items_matrix"] = _matrix_payload(items_matrix)
        if units_matrix is not None:
            f["units_matrix"] = _matrix_payload(units_matrix)
        quick_frames.append(f)

    # 1. Coastline  (preview_cb sends (label, matrix) — height_map only)
    def coastline_cb(label, matrix):
        _frame(label, matrix)
    map_pipeline.run_coastline(state, preview_cb=coastline_cb)

    # 2. Height / Ocean
    map_pipeline.run_height_ocean(state)
    _frame("height_ocean", state.height_map)

    # 3. Random command centers
    map_pipeline.run_place_cc_random(state)
    _frame("command_centers", state.height_map, units_matrix=state.units_matrix)

    # 4. Random resources
    map_pipeline.run_place_resources_random(state)
    _frame("resources", state.height_map, items_matrix=state.items_matrix, units_matrix=state.units_matrix)

    # 5. Finalize terrain smoothing (preview_cb sends all 4 matrices)
    def finalize_cb(label, hm, id_m, items_m, units_m):
        _frame(label, hm, id_m, items_m, units_m)
    map_pipeline.run_finalize(state, preview_cb=finalize_cb)

    # 6. Export TMX
    tmx_bytes = map_pipeline.write_tmx(state, blueprint_xml)

    return {"tmx_bytes": tmx_bytes, "snapshot": _snapshot(), "quick_frames": quick_frames}


def reset_state(params_json="{}"):
    global state
    state = WizardState()
    return {"snapshot": _snapshot()}


RPC_METHODS = {
    "run_coastline": run_coastline,
    "draw_walls": draw_walls,
    "clear_walls": clear_walls,
    "run_height_ocean": run_height_ocean,
    "place_cc_manual": place_cc_manual,
    "remove_cc_manual": remove_cc_manual,
    "place_cc_random": place_cc_random,
    "undo_cc": undo_cc,
    "clear_cc": clear_cc,
    "place_resource_manual": place_resource_manual,
    "remove_resource_manual": remove_resource_manual,
    "place_resource_random": place_resource_random,
    "undo_resource": undo_resource,
    "clear_resource": clear_resource,
    "get_state_snapshot": get_state_snapshot,
    "run_finalize": run_finalize,
    "quick_generate": quick_generate,
    "reset_state": reset_state,
}


def rpc_call(method, params_json="{}"):
    if method not in RPC_METHODS:
        raise ValueError(f"Unknown RPC method: {method}")
    return RPC_METHODS[method](params_json)
